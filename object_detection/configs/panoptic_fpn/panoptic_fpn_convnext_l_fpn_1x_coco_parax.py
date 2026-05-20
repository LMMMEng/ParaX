_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# Replace with your local checkpoint path before training.
# Download: https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
pretrained = 'convnext_large_22k_224.pth'

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='convnext_large_parax',
        parax_rank=128,
        parax_kernel_sizes=(3, 5, 7),
        parax_enable_conv=True,
        parax_router_hidden=16,
        parax_force_fp16=True,
        pretrained=pretrained,
        drop_path_rate=0.25,
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
    type='PanopticFPN',
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)),
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=80,
        num_stuff_classes=53),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096)))

custom_hooks = []


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))



data = dict(samples_per_gpu=2)
# evaluation = dict(save_best='auto')
checkpoint_config = dict(interval=3, max_keep_ckpts=1, save_last=True)
auto_scale_lr = dict(enable=True, base_batch_size=16)

device = 'cuda'

# # AMP (faster but may meet nan loss) ->
fp16 = dict(loss_scale='dynamic')