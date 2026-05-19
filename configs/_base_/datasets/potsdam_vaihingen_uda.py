
dataset_type = 'ISPRSDataset'
data_root_s = 'ISPRS-preprocessed/potsdam_tiles/'
data_root_t = 'ISPRS-preprocessed/vaihingen_tiles/'
img_norm_cfg = dict(

    mean=[98.99, 92.61, 85.97],
    std=[36.07, 35.24, 36.64],
    to_rgb=True  # 转RGB
    
)

crop_size = (512, 512)


source_train_pipeline = [
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

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
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
        type='OEM_UDADataset',
        source=dict(
            type=dataset_type,
            data_root=data_root_s,
            img_dir='train/images/',
            ann_dir='train/labels/',
            split='train.txt',
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=data_root_t,
            img_dir='test/images/',
            ann_dir='test/labels/',
            split='test.txt',
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root_t,
        img_dir='test/images/',
        ann_dir='test/labels/',
        split='test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root_t,
        img_dir='test/images/',
        ann_dir='test/labels/',
        split='test.txt',
        pipeline=test_pipeline)
)