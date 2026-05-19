
dataset_type = 'InriaDataset'
data_root_s = 'WHU_building/' 
data_root_t = 'Inria_building/AerialImageDataset/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='OEM_UDADataset',
        source=dict(
            type='WHUDataset',
            data_root=data_root_s,
            img_dir='train/A/',
            ann_dir='train/OUT_fixed/',
            split='train.txt',
            pipeline=source_train_pipeline),
        target=dict(
            type='InriaDataset',
            data_root=data_root_t,
            img_dir='test/images/',
            ann_dir='test/gt_fixed/',
            split='splits/test.txt',
            pipeline=target_train_pipeline)),
    val=dict(
        type='InriaDataset',
        data_root=data_root_t,
        img_dir='val/images/',
        ann_dir='val/gt_fixed/',
        split='splits/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type='InriaDataset',
        data_root=data_root_t,
        img_dir='train/images/',
        ann_dir='train/gt_fixed/',
        split='splits/train.txt',
        pipeline=test_pipeline)
)