from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()

## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 8,
    disp_iter = 10,
    save_iter = 8,
    test_iter = 2,
    training_func = "train_mono_depth",
    evaluate_func = "evaluate_kitti_depth",
)

cfg.trainer = trainer

## path
path = edict()
path.raw_path = "/home/kitti_raw"
path.depth_path = "/home/data_depth_annotated/train"
path.validation_path = "/home/data_depth_annotated/val_selection_cropped"
path.test_path = "/home/data_depth_annotated/test_depth_prediction_anonymous"

path.visualDet3D_path = "/path/to/visualDet3D/visualDet3D" # The path should point to the inner subfolder
path.project_path = "/path/to/visualDet3D/workdirs" # or other path for pickle files, checkpoints, tensorboard logging and output files.

if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path, 'MonoDepth')
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
    clipped_gradient_norm = 1.0
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(
        T_max     = cfg.trainer.max_epochs,
        eta_min   = 1e-5,
    ),
    is_iter_based = False
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 8,
    num_workers = 8,
    rgb_shape = (352, 1216, 3),
    train_dataset = "KittiDepthMonoDataset",
    val_dataset   = "KittiDepthMonoValTestDataset",
    test_dataset  = "KittiDepthMonoValTestDataset",
)

data.augmentation = edict(
    mirrorProb = 0.5,
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
)
data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(output_height=data.rgb_shape[0])),
    edict(type_name='RandomCropToWidth', keywords=dict(width=data.rgb_shape[1])),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(output_height=data.rgb_shape[0])),
    edict(type_name='CropRight', keywords=edict(output_width=data.rgb_shape[1])),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.name = 'MonoDepth'
detector.backbone = edict(
    depth=34,
    pretrained=True,
    frozen_stages=-1,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    norm_eval=False,
    dilations=(1, 1, 1, 1),
    strides=(1, 2, 2, 2),
)
detector.preprocessed_path = path.preprocessed_path
detector.max_depth=100
detector.output_channel=1
detector.SI_loss_lambda=0.3
detector.smooth_loss_weight = 0.0
detector.minor_weight=1.0
cfg.detector = detector
