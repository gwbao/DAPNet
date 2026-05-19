_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/upernet_swin.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

crop_size = (512, 512)
num_classes = 5
seed = 1337

img_norm_cfg = dict(
    mean=[98.99, 92.61, 85.97],
    std=[36.07, 35.24, 36.64],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='ISPRSDataset',
        data_root='datasets/ISPRS-preprocessed/potsdam_tiles/',
        img_dir='train/images/',
        ann_dir='train/labels/',
        split='train.txt',
        pipeline=train_pipeline
    ),
    val=dict(
        type='ISPRSDataset',
        data_root='datasets/ISPRS-preprocessed/vaihingen_tiles/',
        img_dir='test/images/',
        ann_dir='test/labels/',
        split='test.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        type='ISPRSDataset',
        data_root='datasets/ISPRS-preprocessed/vaihingen_tiles/',
        img_dir='test/images/',
        ann_dir='test/labels/',
        split='test.txt',
        pipeline=test_pipeline
    )
)


model = dict(
    type='EncoderDecoder',
    pretrained=None,
    use_domain_enhancer=True,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=16,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth'
        )
    ),

    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            #class_weight=[1.0, 2.5, 1.0, 1.0, 10.0]
        )
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=0.4,
            class_weight=[1.0, 2.5, 1.0, 1.0, 10.0]
        )
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'decode_head': dict(lr_mult=1.0),
            'auxiliary_head': dict(lr_mult=1.0),
        }
    )
)


optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=20000)

checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=5
)

evaluation = dict(
    interval=2000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU'
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ]
)

workflow = [('train', 1)]