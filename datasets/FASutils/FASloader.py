from datasets.FASutils.FASbase import FASCLDataset
from datasets.FASutils.FASTransform import create_data_transforms_alb
from torch.utils.data import DataLoader
import torch
import random
from datasets.FASutils.FASsampler import MultilabelBalancedRandomSampler, banlanceDatasetClassSampler1
from torch.utils.data import SequentialSampler


def create_dataloader(args, task, split, batchsize):
    transform = create_data_transforms_alb(args, split=split)
    dataset = FASCLDataset(args.facedataset, split, task, args.margin, args.mode, args.image_size, transform)

    sampler = banlanceDatasetClassSampler1(labels=dataset.get_label(), batchsize=batchsize) if split == "train" else SequentialSampler(dataset)
    if args.distributed == "ddp":
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    drop_last = split == "train"

    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            sampler=sampler,
                            #shuffle=drop_last,
                            num_workers=4,
                            drop_last=drop_last,
                            pin_memory=False)

    return dataloader
