from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.obj_types = ['Car', 'Pedestrian']

## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 80, # for validation epoch 50 is enough
    disp_iter = 100,
    save_iter = 5,
    test_iter = 10,
    training_func = "train_stereo_detection",
    test_func = "test_stereo_detection",
    evaluate_func = "evaluate_kitti_obj",
)

cfg.trainer = trainer

## path
path = edict()
path.data_path = "/data/kitti_obj/training" # used in visualDet3D/data/.../dataset
path.test_path = "/data/kitti_obj/testing" # used in visualDet3D/data/.../dataset
path.visualDet3D_path = "/path/to/visualDet3D/visualDet3D" # The path should point to the inner subfolder
path.project_path = "/path/to/visualDet3D/workdirs" # or other path for pickle files, checkpoints, tensorboard logging and output files.
if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path, 'Stereo3D')
if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)

path.log_path = os.path.join(path.project_path, "log")
if not os.path.isdir(path.log_path):
    os.mkdir(path.log_path)

path.checkpoint_path = os.path.join(path.project_path, "checkpoint")
if not os.path.isdir(path.checkpoint_path):
    os.mkdir(path.checkpoint_path)

path.preprocessed_path = os.path.join(path.project_path, "output")
if not os.path.isdir(path.preprocessed_path):
    os.mkdir(path.preprocessed_path)

path.train_imdb_path = os.path.join(path.preprocessed_path, "training")
if not os.path.isdir(path.train_imdb_path):
    os.mkdir(path.train_imdb_path)

path.val_imdb_path = os.path.join(path.preprocessed_path, "validation")
if not os.path.isdir(path.val_imdb_path):
    os.mkdir(path.val_imdb_path)

cfg.path = path

## optimizer
optimizer = edict(
    type_name = 'adam',
    keywords = edict(
        lr        = 1e-4,
        weight_decay = 0,
    ),
    clipped_gradient_norm = 0.1
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(
        T_max     = cfg.trainer.max_epochs,
        eta_min   = 5e-6,
    )
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 4,
    num_workers = 4,
    rgb_shape = (288, 1280, 3),
    train_dataset = "KittiStereoDataset",
    val_dataset   = "KittiStereoDataset",
    test_dataset  = "KittiStereoTestDataset",
    train_split_file = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'test_split', 'train.txt'),
    val_split_file   = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'test_split', 'val.txt'),
)

data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    crop_top = 100,
)
data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='PhotometricDistort', keywords=edict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.obj_types = cfg.obj_types
detector.name = 'Stereo3D'
detector.backbone = edict(
    depth=34,
    pretrained=True,
    frozen_stages=-1,
    num_stages=3,
    out_indices=(0, 1, 2),
    norm_eval=True,
    dilations=(1, 1, 1),
)
head_loss = edict(
    fg_iou_threshold = 0.5,
    bg_iou_threshold = 0.4,
    L1_regression_alpha = 5 ** 2,
    focal_loss_gamma = 2.0,
    balance_weight   = [20.0, 40],
    regression_weight = [1, 1, 1, 1, 1, 1, 12, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
)
head_test = edict(
    score_thr=0.75,
    cls_agnostic = False,
    nms_iou_thr=0.4,
    post_optimization=False
)

anchors = edict(
        {
            'obj_types': cfg.obj_types,
            'pyramid_levels':[4],
            'strides': [2 ** 4],
            'sizes' : [24],
            'ratios': np.array([0.5, 1, 2.0]),
            'scales': np.array([2 ** (i / 4.0) for i in range(16)]),
        }
    )

head_layer = edict(
    num_features_in=1408,
    num_cls_output=len(cfg.obj_types)+1,
    num_reg_output=12,
    cls_feature_size=256,
    reg_feature_size=1408,
)
detector.head = edict(
    num_regression_loss_terms=13,
    preprocessed_path=path.preprocessed_path,
    num_classes     = len(cfg.obj_types),
    anchors_cfg     = anchors,
    layer_cfg       = head_layer,
    loss_cfg        = head_loss,
    test_cfg        = head_test
)
detector.anchors = anchors
detector.loss = head_loss
cfg.detector = detector
