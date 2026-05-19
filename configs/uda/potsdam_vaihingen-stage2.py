
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/potsdam_vaihingen_uda.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

crop_size = (512, 512)
num_classes = 5
seed = 1337


model = dict(
    type='EncoderDecoder',
    pretrained=None,

    use_domain_enhancer=True,
    domain_enhancer_cfg=dict(
        num_channels=1536,
        num_classes=2,
        grl=True,
        reverse=True,
        lambd=0.25,
        enhance_percent=0.5,
        enhance_factor=2.0,
        suppress_percent=0.2,
        suppress_factor=0.5,
        temperature=1.0,
        adaptive=True
    ),

    # Swin Transformer
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
            checkpoint='best_mIoU_iter_10000.pth'
        )
    ),

    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,  # 5类
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
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
            loss_weight=0.4 #1
        )
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# UDA
uda = dict(
    type='DACS',
    source_only=False,
    alpha=0.999,
    pseudo_threshold=0.9,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    use_memory=True,
    crop_size=crop_size,
    consistency_regularizer = 'confidence_based',  
    cmap='tab10',
)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        source=dict(
            type='ISPRSDataset',
            data_root='datasets/ISPRS-preprocessed/potsdam_tiles/',
            img_dir='train/images/',
            ann_dir='train/labels/',
            split='train.txt'
        ),
        target=dict(
            type='ISPRSDataset',
            data_root='datasets/ISPRS-preprocessed/vaihingen_tiles/',
            img_dir='test/images/',
            ann_dir='test/labels/',
            split='test.txt'
        )
    )
)

#
optimizer_config = None
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
            'backbone': dict(lr_mult=0.1),  # backbonex0.1
            'decode_head': dict(lr_mult=1.0),
            'auxiliary_head': dict(lr_mult=1.0),
        }
    )
)


lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)


n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=171600)

checkpoint_config = dict(
    by_epoch=False,
    interval=5000,
    max_keep_ckpts=20
)

evaluation = dict(
    interval=5000,  #
    metric=['mIoU', 'mF1'],  # IoU和F1
    pre_eval=True,
    save_best='mIoU'
)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        #dict(type='TensorboardLoggerHook')
    ]
)

workflow = [('train', 1)]