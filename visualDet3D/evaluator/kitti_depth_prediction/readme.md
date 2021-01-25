# KITTI Depth Prediction Toolkit - Python version

This package reformulates the official KITTI devkit of evaluating depth prediction c++ codes into python3 modules.

The script uses numba for speeding up while preserving the code structure of the original C++ code.

## Standable usage

The script can be used alone.
```sh
# label_path & result_path: absolute path to the folder containing all gt/pred depth images (uint16) 
python3 evaluate_depth.py $label_path $result_path
```

## Embedded Usage

The script is used as part of a larger python program. In visualDet3D:
```python
from torch.utils.tensorboard import SummaryWriter
from visualDet3D.evaluator.kitti_depth_prediction.evaluate_depth import evaluate_depth

writer = SummaryWriter()
epoch_num = 0

result_texts = evaluate_depth(
        label_path = os.path.join(cfg.path.validation_path, 'groundtruth_depth'),
        result_path = result_path
)
for index, result_text in enumerate(result_texts):
    writer.add_text("validation result {}".format(index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
    print(result_text, end='')
```
