# ParaX Plug-and-Play Guide

This document explains how to integrate ParaX into your own vision model as a plug-and-play module. In this repository, we provide three reference model families: Swin, ConvNeXt, and ViT.

## 1. Overview

ParaX is an adapter-style PEFT module built around a shared expert center and dynamic parameter routing. In practice, integrating ParaX into a new backbone usually requires only three components:

1. A `ParaXAdapter` inserted into each target block.
2. A stage-wise or layer-group-wise `parax_parameter_dict(...)` that stores the routed expert parameters.
3. A forward path that passes the routed parameter dictionary to each adapter, and the spatial resolution only when the convolution path is enabled.

## 2. Minimal Integration Steps

### Step 1. Import the ParaX utilities

```python
from models.parax import ParaXAdapter, parax_parameter_dict
```

### Step 2. Create the shared expert parameters

In our implementations, each stage or layer group owns one routed parameter dictionary. The dictionary is created by `parax_parameter_dict(...)` and then registered as an `nn.ParameterDict`.

```python
dim = 256        # token / channel width handled by this stage or block
rank = 128       # ParaX bottleneck width
num_layers = 6   # number of blocks sharing this routed parameter dictionary
parax_params = parax_parameter_dict(
    dim=dim,
    rank=rank,
    num_layers=num_layers,
    parax_kernel_sizes=None,
    enable_conv=False,
)
parax_params = nn.ParameterDict(parax_params)
```

Wrapping the dictionary with `nn.ParameterDict(...)` registers these routed tensors on the module, so they are saved in the state dict and updated by the optimizer.

This example shows token-only integration. If you later enable the convolution path, set `enable_conv=True` and use the same `parax_kernel_sizes` setting in both `parax_parameter_dict(...)` and `ParaXAdapter`.

We recommend sharing one expert center across multiple blocks inside the same stage, which is also the default design used in our released models.

### Step 3. Insert `ParaXAdapter` into the target block

The adapter always takes token features of shape `(B, L, C)` together with the routed parameter dictionary. The spatial size `(H, W)` is required only when `enable_conv=True`.

```python
self.adapter = ParaXAdapter(
    dim=dim,
    rank=rank,
    num_params=num_layers,
    ls_init_value=1,
    rs_init_value=1,
    enable_conv=False,
    parax_kernel_sizes=None,
    router_hidden=16,
)
```

In this stage-sharing setup, `num_params` should match the `num_layers` used when creating the shared routed parameter dictionary.

For Transformer-style token backbones, you may try starting from `enable_conv=False`. This keeps the full adapter in token space and avoids any `BLC <-> BCHW` reshape inside `ParaXAdapter`.

### Step 4. Call the adapter in the forward pass

Once you add ParaX, the surrounding forward signatures usually need one extra argument such as `parax_params`. That argument does not appear automatically: the parent stage or model has to create the shared parameter dictionary, store it, and pass it into each block explicitly.

For token-based backbones, a typical block-level pattern is:

```python
def forward(self, x, parax_params):
    x = self.main_block(x)
    x = self.adapter(x, parax_params)
    return x
```

And the caller has to thread the same shared parameter dictionary through the block calls, often as a module member such as `self.parax_params`:

```python
for blk in self.blocks:
    x = blk(x, self.parax_params)
```

If you want ParaX to use its convolution path, enable it explicitly and pass `(H, W)`:

```python
self.adapter = ParaXAdapter(..., enable_conv=True, parax_kernel_sizes=(3, 5, 7))

def forward(self, x, parax_params, H, W):
    x = self.main_block(x)
    x = self.adapter(x, parax_params, H, W)
    return x
```

For ConvNet blocks, first flatten the spatial feature map into tokens, call the adapter, and then reshape the output back:

```python
x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
x = self.adapter(x, parax_params, H, W)
x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
```

For classification models with a class token, keep the class token outside the adapter path and apply ParaX only to the spatial tokens:

```python
cls_token = x[:, :1]
spatial_tokens = x[:, 1:]
spatial_tokens = self.adapter(spatial_tokens, parax_params)
x = torch.cat([cls_token, spatial_tokens], dim=1)
```

## 3. Reference Integration Patterns

### Swin-style hierarchical transformers

Our [Swin implementation](models/swin_parax.py) places one ParaX adapter after the attention branch and another after the FFN branch in each block. Each stage owns one shared expert center, and every block in that stage routes into the same parameter dictionary.

### ConvNeXt-style ConvNets

Our [ConvNeXt implementation](models/convnext_parax.py) applies the normal ConvNeXt block first, then reshapes the feature map into tokens and applies one ParaX adapter. Each stage again owns one shared expert center.

### ViT-style classification backbones

Our [ViT implementation](models/vit_parax.py) applies ParaX after each transformer block and routes only the spatial tokens through the adapter, while preserving the class token separately. When `enable_conv=False`, this path stays entirely in token space.

## 4. Training Recommendations

When using ParaX in a PEFT setting, we recommend the following procedure:

1. Freeze the original backbone parameters unless you intentionally want hybrid fine-tuning.
2. Keep the ParaX adapter parameters and the routed expert center trainable.
3. Start from the same `rank`, `router_hidden`, and `parax_kernel_sizes` settings used by the closest released backbone.
4. Share one expert center per stage before trying finer-grained sharing strategies.

## 5. Precision and Mixed Precision

`ParaXAdapter` supports two integration switches that matter in practice:

1. `enable_conv`: whether to activate the convolution path.
2. `force_fp16`: whether to compute the adapter path in fp16 on CUDA.

```python
self.adapter = ParaXAdapter(..., enable_conv=False)
```

With `enable_conv=False`, the adapter stays in token space and `H` / `W` can be omitted. With `enable_conv=True`, the adapter activates its convolution path and requires `H` / `W`.

`force_fp16` remains optional:

```python
self.adapter = ParaXAdapter(..., force_fp16=True)
```

When enabled, this path activates only on CUDA. The adapter internally casts the routed activations and routed expert tensors to `float16`, performs the adapter computation in fp16, and then casts the final adapter output back to the caller's original dtype. On CPU, the flag safely degrades to a no-op.

If training becomes unstable after enabling this flag, for example if the loss quickly becomes `NaN`, disable `force_fp16` first and retry.

For released dense prediction models in this repository, the corresponding backbone configs expose this behavior through `parax_force_fp16=True`.

## 6. Practical Checklist

Before training a new model with ParaX, make sure the following points are satisfied:

1. The adapter input is shaped as `(B, L, C)`.
2. If `enable_conv=True`, the spatial resolution `(H, W)` matches the token layout passed to the adapter.
3. The shared expert parameter dictionary uses the same `dim`, `rank`, `enable_conv`, and `parax_kernel_sizes` setting as the adapter.
4. The class token, if present, is excluded from the spatial adapter path.
5. Only the intended ParaX parameters remain trainable in PEFT mode.

## 7. Where to Look in This Repository

If you want concrete reference implementations, start from the following files:

1. `models/swin_parax.py`
2. `models/convnext_parax.py`
3. `models/vit_parax.py`
4. `models/parax.py`

These files cover the three integration patterns released in this repository and can usually be adapted to new backbones with only local block-level changes.