# The new config inherits a base config to highlight the necessary modification for MineCraft 
# https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets

_base_ = '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'

#  MineCraft classes
minecraft_classes = [
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
    'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
                    ]
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    bbox_head=dict(
        type='FCOSHead',   # ← ОБЯЗАТЕЛЬНО
        center_sampling=False,     # Улучшение для мелких объектов
        norm_on_bbox=False,        # Более стабильный bbox-regression
        num_classes=len(minecraft_classes),
        loss_cls=dict(
            type='FocalLoss', 
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25
        )
    ),
    test_cfg=dict(
        nms_pre=1000,                 # сколько кандидатов оставить до NMS
        score_thr=0.25,               # подними, 0.001 даёт много дублей
        nms=dict(type='nms', iou_threshold=0.5),  # сделай жёстче: 0.5 -> 0.4/0.3
        max_per_img=100               # максимум боксов на картинку
    )
)


# уменьшенная картинка для экономии VRAM
img_scale = (514, 514)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
#    dict(type='Normalize',   mean=[103.530, 116.280, 123.675],
#                             std=[1.0, 1.0, 1.0],
#                             to_rgb=False),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=20,
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        hue_delta=10
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')  # заменяет DefaultFormatBundle + Collect
]


valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True), 
    dict(type='Resize', scale=img_scale , keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

test_pipeline = valid_pipeline


# Modify dataset related settings

data_root = 'dataset/minecraft/'


metainfo = {
    'classes': minecraft_classes,
    'palette': [(106, 255, 0)] * len(minecraft_classes)
    }


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/_train_annotations.coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/_valid_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False)))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/_test_annotations.coco.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False)))


# Modify metric related settings
val_evaluator  = dict(type='CocoMetric', ann_file = data_root + 'annotations/_valid_annotations.coco.json', metric=['bbox'])
test_evaluator = dict(type='CocoMetric', ann_file = data_root + 'annotations/_test_annotations.coco.json',  metric=['bbox'])

# -----------------------------------------------------------
# 6. Настройка обучения и скорости
# -----------------------------------------------------------

param_scheduler = [
#   dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[24, 33], gamma=0.1)
]


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=12)
# optimizer
optim_wrapper= dict(
   optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)


# -----------------------------------------------------------
# 7. Сохранение 
# -----------------------------------------------------------
checkpoint_config = dict(
                        interval=-1,
                        save_last=True,)
work_dir = './artifacts/fcos'

# -----------------------------------------------------------
# 8. FP16 ускорение
# -----------------------------------------------------------
fp16 = dict(loss_scale='dynamic')
# -----------------------------------------------------------















