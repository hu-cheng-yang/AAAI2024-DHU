import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.continual_dataset import ContinualDataset
from backbone.ResNet18 import resnet18

import cv2
from albumentations.pytorch.transforms import ToTensorV2
from datasets.FASutils.FASloader import create_dataloader
# from datasets.FASutils.FASOCIMloader import create_dataloader
from datasets.FASutils.FASTransform import create_data_transforms_alb



def get_current_dataloader(args, task, batchsize):
    train_loader = create_dataloader(args, task, split="train", batchsize=batchsize)

    if "+" not in task:
        test_loader = [create_dataloader(args, task, split="test", batchsize=batchsize)]
    else:
        test_loader = []
        for t in task.split("+"):
            test_loader.append(create_dataloader(args, t, split="test", batchsize=batchsize))

    return train_loader, test_loader


class FASCLtask(ContinualDataset):
    NAME = 'seq-fas-task'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2
    RESUME = 0

    def __init__(self, args):
        super().__init__(args)
        self.seq = args.seq.split(',')
        self.N_TASKS = args.ntask
        self.batch_size = args.batch_size
        self.minibatch_size = args.batch_size
        assert self.N_TASKS == len(self.seq), "The length of input sequence must be same as the task"
        print("Dataset Info | FASCL contains {} tasks including {}".format(self.N_TASKS, self.seq))
        try:
            if args.resume == 1 and args.resume_iter < 0:
                self.RESUME = 1
                for k in range(0, args.resume_task):
                    self.test_loaders.append(create_dataloader(args, self.seq[k], split="test", batchsize=self.batch_size))
        except:
            pass

    def get_data_loaders(self):
        now_task = self.seq[self.i]
        train, test = get_current_dataloader(self.args, now_task, batchsize=self.batch_size)
        if self.RESUME == 1:
            self.RESUME = 0
            self.train_loader = train
            return train, None
        self.test_loaders.extend(test)
        self.train_loader = train
        return train, test

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_backbone():
        return resnet18(2)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 180

    @staticmethod
    def get_batch_size():
        return 16

    @staticmethod
    def get_minibatch_size():
        return FASCLtask.get_batch_size()
