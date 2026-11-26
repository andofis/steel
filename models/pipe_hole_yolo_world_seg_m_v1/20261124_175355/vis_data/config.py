_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
_train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
]
affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 0.0002
batch_shapes_cfg = None
close_mosaic_epochs = 10
coco_train_dataset = dict(
    _delete_=True,
    class_text_path='data/texts/lvis_v1_base_class_texts.json',
    dataset=dict(
        ann_file='lvis/lvis_v1_train_base.json',
        data_prefix=dict(img=''),
        data_root='data/coco',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        type='YOLOv5LVISV1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(
            mask2bbox=True,
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    mask2bbox=True,
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
            ],
            type='MultiModalMosaic'),
        dict(prob=0.1, type='YOLOv5CopyPaste'),
        dict(
            border=(
                -320,
                -320,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100.0,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            min_area_ratio=0.01,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine',
            use_mask_refine=True),
        dict(
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    mask2bbox=True,
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(
                    img_scale=(
                        640,
                        640,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(
                            mask2bbox=True,
                            type='LoadAnnotations',
                            with_bbox=True,
                            with_mask=True),
                    ],
                    type='MultiModalMosaic'),
                dict(prob=0.1, type='YOLOv5CopyPaste'),
                dict(
                    border=(
                        -320,
                        -320,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_aspect_ratio=100.0,
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    min_area_ratio=0.01,
                    scaling_ratio_range=(
                        0.09999999999999998,
                        1.9,
                    ),
                    type='YOLOv5RandomAffine',
                    use_mask_refine=True),
            ],
            prob=0.1,
            type='YOLOv5MultiModalMixUp'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
        dict(
            max_num_samples=80,
            num_neg_samples=(
                1203,
                1203,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
coco_val_dataset = dict(
    _delete_=True,
    class_text_path='data/captions/lvis_v1_class_captions.json',
    dataset=dict(
        ann_file='lvis/lvis_v1_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='data/coco/',
        test_mode=True,
        type='YOLOv5LVISV1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            640,
            640,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                640,
                640,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
copypaste_prob = 0.1
custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/segm_mAP',
        patience=5,
        rule='greater',
        type='EarlyStoppingHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'yolo_world',
        'pipe_world_fixed_text',
    ])
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.67
default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=35, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
downsample_ratio = 4
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=2))
find_unused_parameters = True
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 768
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
]
launcher = 'none'
load_from = 'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\yolo_world_seg_m_hole_prompt_init.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
mask_overlap = False
max_aspect_ratio = 100
max_epochs = 80
max_keep_ckpts = 2
min_area_ratio = 0.01
mixup_prob = 0.1
model = dict(
    backbone=dict(
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=0.67,
            last_stage_out_channels=768,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='YOLOv8CSPDarknet',
            widen_factor=0.75),
        text_model=dict(channels=512, type='FixedHoleText'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            embed_dims=512,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                768,
            ],
            mask_channels=32,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            proto_channels=256,
            reg_max=16,
            type='YOLOWorldSegHeadModule',
            widen_factor=0.75),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        loss_mask=dict(
            reduction='none', type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_mask_weight=1.0,
        mask_overlap=False,
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOWorldSegHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=True,
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        deepen_factor=0.67,
        embed_channels=[
            128,
            256,
            384,
        ],
        guide_channels=512,
        in_channels=[
            256,
            512,
            768,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        num_heads=[
            4,
            8,
            12,
        ],
        out_channels=[
            256,
            512,
            768,
        ],
        text_enhancder=dict(
            embed_channels=256,
            num_heads=8,
            type='ImagePoolingAttentionModule'),
        type='YOLOWorldDualPAFPN',
        widen_factor=0.75),
    num_test_classes=1,
    num_train_classes=1,
    test_cfg=dict(
        fast_test=True,
        mask_thr_binary=0.5,
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=1,
            topk=10,
            type='YOLOWorldSegAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
        ],
        type='MultiModalMosaic'),
    dict(prob=0.1, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
]
neck_embed_channels = [
    128,
    256,
    384,
]
neck_num_heads = [
    4,
    8,
    12,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1203
num_det_layers = 3
num_training_classes = 80
optim_wrapper = dict(
    accumulative_counts=4,
    clip_grad=dict(max_norm=10.0),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        T_max=24,
        begin=0,
        by_epoch=True,
        end=24,
        eta_min=2e-06,
        type='CosineAnnealingLR'),
]
persistent_workers = False
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
]
randomness = dict(deterministic=False, seed=42)
resume = False
save_epoch_intervals = 5
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        class_text_path=
        'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\class_texts.json',
        dataset=dict(
            ann_file=
            'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\instances_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='images/val'),
            data_root='D:\\钢管展开图\\04_孔洞分割迁移学习数据集\\yolo_seg',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=('hole', )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\instances_val.json',
    metric=[
        'bbox',
        'segm',
    ],
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 512
text_transform = [
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 8
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=2)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        class_text_path=
        'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\class_texts.json',
        dataset=dict(
            ann_file=
            'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\instances_train.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='images/train'),
            data_root='D:\\钢管展开图\\04_孔洞分割迁移学习数据集\\yolo_seg',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=('hole', )),
            test_mode=False,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                border=(
                    0,
                    0,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_rotate_degree=3.0,
                max_shear_degree=0.0,
                max_translate_ratio=0.08,
                min_area_ratio=0.05,
                scaling_ratio_range=(
                    0.8,
                    1.2,
                ),
                type='YOLOv5RandomAffine',
                use_mask_refine=True),
            dict(
                hue_delta=0.0,
                saturation_delta=0.0,
                type='YOLOv5HSVRandomAug',
                value_delta=0.22),
            dict(direction='horizontal', prob=0.5, type='mmdet.RandomFlip'),
            dict(direction='vertical', prob=0.5, type='mmdet.RandomFlip'),
            dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=False,
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
        ],
        type='MultiModalMosaic'),
    dict(prob=0.1, type='YOLOv5CopyPaste'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        mask2bbox=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                ],
                type='MultiModalMosaic'),
            dict(prob=0.1, type='YOLOv5CopyPaste'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                min_area_ratio=0.01,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine',
                use_mask_refine=True),
        ],
        prob=0.1,
        type='YOLOv5MultiModalMixUp'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        mask2bbox=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        min_area_ratio=0.01,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine',
        use_mask_refine=True),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(downsample_ratio=4, mask_overlap=False, type='Polygon2Mask'),
    dict(
        max_num_samples=80,
        num_neg_samples=(
            1203,
            1203,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
use_mask2refine = True
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='pseudo_collate'),
    dataset=dict(
        class_text_path=
        'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\class_texts.json',
        dataset=dict(
            ann_file=
            'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\instances_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='images/val'),
            data_root='D:\\钢管展开图\\04_孔洞分割迁移学习数据集\\yolo_seg',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(classes=('hole', )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                mask2bbox=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    'D:\\钢管拼接\\steel\\datasets\\_training_work\\yolo_world_seg\\instances_val.json',
    metric=[
        'bbox',
        'segm',
    ],
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
widen_factor = 0.75
work_dir = 'D:\\钢管拼接\\steel\\models\\pipe_hole_yolo_world_seg_m_v1'
