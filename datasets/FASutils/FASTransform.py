import albumentations as alb
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2


def create_data_transforms_alb(args, split='train'):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if 'train' in split:
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            # alb.RandomSizedCrop([args.image_size // 4, args.image_size], args.image_size, args.image_size, p=0.8),
            # alb.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            # alb.ToGray(p=0.2),
            # alb.Flip(),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif 'val' in split:
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif 'test' in split:
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
