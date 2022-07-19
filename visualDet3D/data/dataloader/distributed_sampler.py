# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import torch
from torch.utils.data.sampler import Sampler
from visualDet3D.networks.utils.registry import SAMPLER_DICT

@SAMPLER_DICT.register_module
class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)

    Note that this sampler does not shard based on pytorch DataLoader worker id.
    A sampler passed to pytorch DataLoader is used only with map-style dataset
    and will not be executed inside workers.
    But if this sampler is used in a way that it gets execute inside a dataloader
    worker, then extra work needs to be done to shard its outputs based on worker id.
    This is required so that workers don't produce identical data.
    :class:`ToIterableDataset` implements this logic.
    This note is true for all samplers in detectron2.
    """

    def __init__(self, size: int, rank: int = -1, world_size: int = 1, shuffle: bool = True):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._shuffle = shuffle

        self._rank = rank
        self._world_size = world_size
        self.generator = torch.Generator()

    def __len__(self):
        return self._size

    def __iter__(self):
        start = max(self._rank, 0)
        yield from itertools.islice(self._indices(), start, None, self._world_size)

    def _indices(self):
        if self._shuffle:
            yield from torch.randperm(self._size, generator=self.generator).tolist()
        else:
            yield from torch.arange(self._size).tolist()
