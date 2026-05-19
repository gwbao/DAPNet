_base_ = './potsdam_vaihingen-stage2.py'

_delete_ = ['uda']

model = dict(
    type='EncoderDecoder',
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 512),
        stride=(341, 341)
    )
)

data = dict(
    test=dict(
        type='ISPRSDataset',
        data_root='datasets/ISPRS-preprocessed/vaihingen_tiles/',
        img_dir='test/images/',
        ann_dir='test/labels/',
        split='test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize',
                         mean=[98.99, 92.61, 85.97],
                         std=[36.07, 35.24, 36.64],
                         to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ]
            )
        ]
    )
)