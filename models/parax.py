from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


def _unwrap_pretrained_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Checkpoint must deserialize to a dict, got {type(checkpoint)!r}')

    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        checkpoint = checkpoint['state_dict']

    if checkpoint and all(isinstance(key, str) and key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}

    return checkpoint


def load_pretrained_weights(module_or_pretrained, pretrained=None, strict=False):
    if pretrained is None:
        pretrained = module_or_pretrained
        if not isinstance(pretrained, str):
            raise TypeError('pretrained must be a local checkpoint path string')
        checkpoint = torch.load(pretrained, map_location='cpu')
        return _unwrap_pretrained_state_dict(checkpoint)

    if not isinstance(pretrained, str):
        raise TypeError('pretrained must be a local checkpoint path string')
    state_dict = load_pretrained_weights(pretrained)
    return module_or_pretrained.load_state_dict(state_dict, strict=strict)


def normalize_parax_kernel_sizes(parax_kernel_sizes=None):
    if parax_kernel_sizes is None:
        return ()
    if not isinstance(parax_kernel_sizes, (list, tuple)):
        raise TypeError(
            'parax_kernel_sizes must be None, a list, or a tuple of positive odd integers; '
            'compact scalar forms like 357 are not supported')

    kernel_sizes = tuple(int(k) for k in parax_kernel_sizes)
    for kernel_size in kernel_sizes:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f'parax_kernel_sizes must contain positive odd integers, got {kernel_size}')
    return kernel_sizes


def parax_linear_param(in_features, out_features, num_layers):
    params = []
    for _ in range(num_layers):
        param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(param, std=0.02)
        params.append(param)
    return torch.stack(params, dim=0)


def parax_conv_param(dim, num_layers, kernel_size=3):
    params = []
    for _ in range(num_layers):
        param = nn.Parameter(torch.empty(dim, 1, kernel_size, kernel_size))
        nn.init.trunc_normal_(param, std=0.02)
        params.append(param)
    return torch.stack(params, dim=0)


def parax_parameter_dict(dim, rank, num_layers, parax_kernel_sizes=None, enable_conv=False):
    parax_kernel_sizes = normalize_parax_kernel_sizes(parax_kernel_sizes) if enable_conv else ()
    params = {
        'projA_params': parax_linear_param(dim, rank, num_layers=num_layers),
        'projB_params': parax_linear_param(rank, dim, num_layers=num_layers),
    }
    for kernel_size in parax_kernel_sizes:
        params[f'conv{kernel_size}x{kernel_size}_params'] = parax_conv_param(
            rank, num_layers=num_layers, kernel_size=kernel_size)
    return params


class ParaXAdapter(nn.Module):
    """ParaX routing adapter shared by the clean Swin, ConvNeXt, and ViT backbones.

    Args:
        dim (int): Input and output token dimension processed by the adapter.
        rank (int): Bottleneck width used by the routed projection and ParaX kernels.
        num_params (int): Number of expert slots mixed inside each routing group.
        num_groups (int | None): Expected number of routing groups. When provided,
            it must match ``2 + len(parax_kernel_sizes)``.
        ls_init_value (float | None): Initial value of the adapter output scaling
            applied before the residual merge.
        rs_init_value (float | None): Initial value of the shortcut scaling applied
            to the residual branch.
        use_temperature (bool): Whether to learn a per-group routing temperature.
        parax_kernel_sizes (Sequence[int] | None): Enabled odd kernel sizes for the
            ParaX convolution experts. ``None`` disables the convolution branch.
        enable_conv (bool): Whether to enable the depthwise and routed convolution
            paths. When disabled, the adapter stays in token space and ``H`` / ``W``
            are optional at call time.
        router_hidden (int): Hidden width of the lightweight routing MLP.
        force_fp16 (bool): Whether to force the adapter to run in fp16 precision for better efficiency.
            If you encouter training issues, please try setting it to False.
    """

    def __init__(self,
                 dim=32,
                 rank=64,
                 num_params=2,
                 num_groups=None,
                 ls_init_value=1,
                 rs_init_value=1,
                 use_temperature=False,
                 parax_kernel_sizes=None,
                 enable_conv=False,
                 router_hidden=16,
                 force_fp16=False):
        super().__init__()
        self.enable_conv = enable_conv
        self.parax_kernel_sizes = normalize_parax_kernel_sizes(parax_kernel_sizes) if enable_conv else ()
        expected_groups = 2 + len(self.parax_kernel_sizes)
        if num_groups is not None and num_groups != expected_groups:
            raise ValueError(
                f'num_groups={num_groups} is inconsistent with parax_kernel_sizes={self.parax_kernel_sizes}; '
                f'expected {expected_groups}')
        if router_hidden <= 0:
            raise ValueError(f'router_hidden must be a positive integer, got {router_hidden}')
        self.num_groups = expected_groups
        self.num_params = num_params
        self.force_fp16 = force_fp16

        self.norm = nn.LayerNorm(dim)
        if self.enable_conv:
            self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        else:
            self.dwconv1 = None
            self.dwconv2 = None

        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_params * self.num_groups),
        )

        if self.parax_kernel_sizes:
            self.attn = nn.Sequential(
                nn.Linear(rank, len(self.parax_kernel_sizes)),
                nn.Softmax(dim=-1),
            )
            self.conv_res_scale = nn.Parameter(torch.ones(max(0, 2 * (len(self.parax_kernel_sizes) - 1))))
        else:
            self.attn = None
            self.register_parameter('conv_res_scale', None)

        self.proj = nn.Linear(rank, rank)
        self.norm_gamma = nn.Parameter(torch.ones(dim) * 1e-5)

        if use_temperature:
            self.logit_scale = nn.Parameter(torch.zeros(self.num_groups))
        else:
            self.register_buffer('logit_scale', torch.zeros(self.num_groups))

        if ls_init_value is not None:
            self.layer_scale = nn.Parameter(torch.ones(dim) * ls_init_value)
            self.layer_bias = nn.Parameter(torch.zeros(dim))
        if rs_init_value is not None:
            self.res_scale = nn.Parameter(torch.ones(dim) * rs_init_value)
            self.res_bias = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _force_fp16_enabled(self, x):
        return self.force_fp16 and x.device.type == 'cuda'

    def _runtime_context(self, x):
        if self._force_fp16_enabled(x):
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _cast_runtime_tensor(tensor, dtype, device):
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return tensor
        if tensor.dtype == dtype and tensor.device == device:
            return tensor
        return tensor.to(device=device, dtype=dtype)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, a=pow(5, 0.5))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward_gate(self, x, params):
        gates = self.router(x.mean(dim=1, keepdim=True))
        gates = torch.chunk(gates, chunks=self.num_groups, dim=-1)
        logit_scale = self._cast_runtime_tensor(self.logit_scale, gates[0].dtype, gates[0].device)

        adapter_params = {}
        attn = (gates[0] * logit_scale[0].exp()).softmax(dim=-1)
        adapter_params['projA_params'] = einsum(
            attn,
            self._cast_runtime_tensor(params['projA_params'], attn.dtype, attn.device),
            'b l g, g c2 c1 -> b c2 c1')

        attn = (gates[1] * logit_scale[1].exp()).softmax(dim=-1)
        adapter_params['projB_params'] = einsum(
            attn,
            self._cast_runtime_tensor(params['projB_params'], attn.dtype, attn.device),
            'b l g, g c2 c1 -> b c2 c1')

        for gate_index, kernel_size in enumerate(self.parax_kernel_sizes, start=2):
            param_name = f'conv{kernel_size}x{kernel_size}_params'
            attn = (gates[gate_index] * logit_scale[gate_index].exp()).softmax(dim=-1)
            adapter_params[param_name] = einsum(
                attn,
                self._cast_runtime_tensor(params[param_name], attn.dtype, attn.device),
                'b l g, g c2 c1 k1 k2 -> b c2 c1 k1 k2')

        return adapter_params

    def _apply_parax_convs(self, x, adapter_params, height, width):
        if not self.enable_conv or not self.parax_kernel_sizes:
            return x

        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width).contiguous()
        batch_size, channels, height, width = x.shape
        x0 = x.reshape(1, -1, height, width)
        conv_outputs = []
        previous = None
        conv_res_scale = None
        if self.conv_res_scale is not None:
            conv_res_scale = self._cast_runtime_tensor(self.conv_res_scale, x0.dtype, x0.device)
        for conv_index, kernel_size in enumerate(self.parax_kernel_sizes):
            conv_input = x0
            if conv_index > 0:
                scale_index = 2 * (conv_index - 1)
                conv_input = x0 * conv_res_scale[scale_index] + previous * conv_res_scale[scale_index + 1]
            weight = adapter_params[f'conv{kernel_size}x{kernel_size}_params'].reshape(
                batch_size * channels, -1, kernel_size, kernel_size)
            previous = F.conv2d(
                conv_input,
                weight=weight,
                padding=kernel_size // 2,
                groups=batch_size * channels)
            conv_outputs.append(rearrange(previous, '1 (b c) h w -> b (h w) c', b=batch_size, c=channels))

        if len(conv_outputs) == 1:
            return conv_outputs[0]
        attn = self.attn(sum(conv_outputs)).unsqueeze(2)
        return sum(conv_output * attn[..., conv_index] for conv_index, conv_output in enumerate(conv_outputs))

    @staticmethod
    def _require_spatial_shape(H, W):
        if H is None or W is None:
            raise ValueError('H and W are required when enable_conv=True')

    def forward(self, x, params, H=None, W=None):
        input_dtype = x.dtype
        if self._force_fp16_enabled(x):
            x = self._cast_runtime_tensor(x, torch.float16, x.device)

        with self._runtime_context(x):
            shortcut_o = x
            runtime_dtype = x.dtype
            runtime_params = params

            norm_gamma = self._cast_runtime_tensor(self.norm_gamma, runtime_dtype, x.device)
            x = self.norm(x) * norm_gamma + x
            if self.enable_conv:
                self._require_spatial_shape(H, W)
                x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
                x = x + self.dwconv1(x)
                x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

            if self._force_fp16_enabled(x):
                x = self._cast_runtime_tensor(x, torch.float16, x.device)
                runtime_params = {
                    name: self._cast_runtime_tensor(param, x.dtype, x.device)
                    for name, param in params.items()
                }

            adapter_params = self.forward_gate(x, runtime_params)
            projA_params = adapter_params['projA_params']
            projB_params = adapter_params['projB_params']

            x = einsum(x, projA_params, 'b l c1, b c2 c1 -> b l c2')
            shortcut_c = x
            x = self._apply_parax_convs(x, adapter_params, H, W)

            x = x + shortcut_c
            x = x + self.proj(x)
            x = F.gelu(x)
            x = einsum(x, projB_params, 'b l c1, b c2 c1 -> b l c2')

            if self.enable_conv:
                x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
                x = x + self.dwconv2(x)
                x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

            if hasattr(self, 'layer_scale'):
                layer_scale = self._cast_runtime_tensor(self.layer_scale, x.dtype, x.device)
                layer_bias = self._cast_runtime_tensor(self.layer_bias, x.dtype, x.device)
                x = layer_scale * x + layer_bias

            if hasattr(self, 'res_scale'):
                res_scale = self._cast_runtime_tensor(self.res_scale, shortcut_o.dtype, shortcut_o.device)
                res_bias = self._cast_runtime_tensor(self.res_bias, shortcut_o.dtype, shortcut_o.device)
                shortcut_o = res_scale * shortcut_o + res_bias

            x = x + shortcut_o

        if x.dtype != input_dtype:
            x = x.to(input_dtype)
        return x