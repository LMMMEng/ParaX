_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# Replace with your local checkpoint path before training.
# Download: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
pretrained = 'swin_large_patch4_window7_224_22k.pth'

norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='SwinParaX',
        parax_rank=128,
        parax_kernel_sizes=(3, 5, 7),
        parax_enable_conv=True,
        parax_router_hidden=16,
        parax_force_fp16=True,
        pretrained=pretrained,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=150,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=150,
        norm_cfg=norm_cfg
    ))


# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006 * 5.0, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'backbone': dict(lr_mult=1.0),
                     'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


data = dict(samples_per_gpu=4) # as gpus = 4
checkpoint_config = dict(interval=8000, max_keep_ckpts=1)
evaluation = dict(interval=8000, save_best='mIoU')

# # AMP (faster but may meet nan loss) ->
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
fp16 = dict()