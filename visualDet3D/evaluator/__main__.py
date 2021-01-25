from fire import Fire
from kitti.evaluate import evaluate
from kitti_depth_prediction import evaluate_depth

def main(evaluator='kitti_obj', **kwargs):
    if evaluator.lower() == 'kitti_obj':
        texts = evaluate(**kwargs)
        for text in texts:
            print(text)
        return 
    if evaluator.lower() == 'kitti_depth':
        print(evaluate_depth(**kwargs))
        return
    raise NotImplementedError

Fire(main)