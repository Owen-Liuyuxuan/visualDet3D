from typing import Callable, Optional
from easydict import EasyDict
from torch.utils.data import DataLoader
from visualDet3D.networks.utils.registry import SAMPLER_DICT

def build_dataloader(dataset, 
                    num_workers: int,
                    batch_size: int,
                    collate_fn: Callable,
                    local_rank: int = -1,
                    world_size: int = 1,
                    sampler_cfg: Optional[EasyDict] = dict(),
                    **kwargs):
    sampler_name = sampler_cfg.pop('name') if 'name' in sampler_cfg else 'TrainingSampler'
    sampler = SAMPLER_DICT[sampler_name](size=len(dataset), rank=local_rank, world_size=world_size, **sampler_cfg)
    
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, 
    sampler=sampler, **kwargs, drop_last=True)
    return dataloader
