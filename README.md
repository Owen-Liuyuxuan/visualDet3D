# Visual 3D Detection Package:

This repo aims to provide flexible and reproducible visual 3D detection on KITTI dataset. We expect scripts starting from the current directory, and treat ./visualDet3D as a package that we could modify and test directly instead of a library. Several useful scripts are provided in the main directory for easy usage.

We believe that visual tasks are interconnected, so we make this library extensible to more experiments. 
The package uses registry to register datasets, models, processing functions and more allowing easy inserting of new tasks/models while not interfere with the existing ones.

## Related Paper:

This repo contains the official implementation of 2021 *RAL* paper [**Ground-aware Monocular 3D Object Detection for Autonomous Driving**](https://ieeexplore.ieee.org/document/9327478). Pretrained model can be found at release pages.
```
@ARTICLE{9327478,
  author={Y. {Liu} and Y. {Yuan} and M. {Liu}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ground-aware Monocular 3D Object Detection for Autonomous Driving}, 
  year={2021},
  doi={10.1109/LRA.2021.3052442}}
```
## Key Features

- **SOTA Performance** State of the art result on visual 3D detection.
- **Modular Design** Modular design for dataset, network and running pipelines.
- **Support Various Task** Compatible with the training and testing of mono/stereo 3D detection and depth prediction.
- **Distributed & Single GPU** Support training with multiple GPUs.
- **Installation-Free Setup** The setup process only build operations and does not require installation to keep the environment clean.
- **Global Path-based IMDB** Do not need data placed inside the folder, convienient for managing data and code separately.


We provide start-up solutions for [Mono3D](docs/mono3d.md), [Depth Predictions](docs/monoDepth.md) and more (until further publication).

Reference: this repo borrows codes and ideas from [retinanet](https://github.com/yhenon/pytorch-retinanet),
[mmdetection](https://github.com/open-mmlab/mmdetection),
[M3D-RPN](https://github.com/garrickbrazil/M3D-RPN),
[DORN](https://github.com/dontLoveBugs/SupervisedDepthPrediction),
[EdgeNets](https://github.com/sacmehta/EdgeNets),
[det3](https://git.ram-lab.com/yun/det3)

## Setup
### Environment setup. 

```bash
pip3 install -r requirement.txt
```
or manually check dependencies.

```bash
# build ops (deform convs), We will not install operations into the system environment
./make.sh
```

## Start Training

Please check the corresponding task: [Mono3D](docs/mono3d.md), [Depth Predictions](docs/monoDepth.md). More demo will be available through contributions and further paper submission.

### Config and Path setup. 

Please modify the path and other parameters in **config/\*.py**. **config/\*_example** files are templates.

**Notice**:
*_examples are **NOT** utilized by the code and \*.py under /config is **ignored** by .gitignore.

The content of the selected config file will be recorded in tensorboard at the beginning of training.

**important paths to modify in config** :
1. cfg.path.data_path: Path to KITTI training data. We expect calib, image_2, image_3, label_2 being the subfolder (directly unzipping the downloaded zips will be fine)
2. cfg.path.test_path: Path to KITTI testing data.  We expect calib, image_2 being the subfolder.
3. cfg.path.visualDet3D_path: Path to the "visualDet3D" directorty of the current repo
4. cfg.path.project_path: Path to the workdirs of the projects (will have temp_outputs, log, checkpoints)

Please check the template's comments and other comments in codes to fully exploit the repo.

## Further Info

0. Read the [wiki](https://github.com/Owen-Liuyuxuan/visualDet3D/wiki)
1. Open issues on the repo if you meet troubles or find a bug or have some suggestions.
2. 