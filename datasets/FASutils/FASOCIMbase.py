import os
import random
import cv2
import json
import numpy as np
import lmdb
from torchvision.transforms import transforms
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

class LivenessDataset(Dataset):
    def __init__(self, split="train", task="O", margin=0.7, mode='rgb', img_size=256, transform=None):
        self.split = split
        self.margin = margin
        self.task = task
        self.img_mode = mode
        self.transform = transform
        self.img_size = img_size

        LMDB_root = "PATH"
        self.env = lmdb.open(LMDB_root, readonly=True, max_readers=512)
        self.data = self.env.begin(write=False)


        setting = self.task
        if setting == "C":
            datasetname = "CASIA_database"
        elif setting == "I":
            datasetname = "replayattack"
        elif setting == "O":
            datasetname = "Oulu_NPU"
        elif setting == "M":
            datasetname = "MSU-MFSD"
        else:
            raise Exception("Dataset name is not right!")
        self.datasetname = datasetname
        train_pos_list_path = "PATH/{}/lists/train_real_5points.list".format(datasetname)
        train_neg_list_path = "PATH/{}/lists/train_fake_5points.list".format(datasetname)
        test_pos_list_path = "PATH/{}/lists/test_real_5points.list".format(datasetname)
        test_neg_list_path = "PATH/{}/lists/test_fake_5points.list".format(datasetname)
        test_list_path = "PATH/{}/lists/test_5points.list".format(datasetname)
        if setting == "I" or setting == "O":
            test_list_path = "PATH/{}/lists/test_pic_5points.list".format(datasetname)
        if self.split == 'train':
            self.items = open(train_pos_list_path).read().splitlines() + open(train_neg_list_path).read().splitlines()
        elif self.split == 'test':
            if setting == "I" or setting == "O":
                self.items = open(test_pos_list_path).read().splitlines() + open(test_neg_list_path).read().splitlines()
            else:
                self.items = open(test_list_path).read().splitlines()
        else:
            self.items = []

        self._display_infos()

    def _display_infos(self):
        print(f'=> Dataset {self.datasetname} loaded')
        print(f'=> Split {self.split}')
        print(f'=> Total number of items: {len(self.items)}')
        print(f'=> Image mode: {self.img_mode}')
        print(f'===========================================')

    def _add_face_margin(self, x, y, w, h, margin=0.5):
        x_marign = int(w * margin / 2)
        y_marign = int(h * margin / 2)

        x1 = x - x_marign
        x2 = x + w + x_marign
        y1 = y - y_marign
        y2 = y + h + y_marign

        return x1, x2, y1, y2

    def _get_item_index(self,index=0):
        item = self.items[index]
        res = item.split(' ')
        img_path = res[0]
        print(img_path)
        label = int(res[1])

        if self.use_LMDB:
            img_bin = self.data.get(img_path.encode())
            try:
                img_buf = np.frombuffer(img_bin, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            except:
                print('load img_buf error')
                print(img_path)
                img_path, label, img, res = self._get_item_index(index+1)
        else:
            img = cv2.imread(img_path)

        return img_path, label, img, res

    def _reset_lmdb(self):
        self.env.close()
        self.env.close()
        self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
        self.data = self.env.begin(write=False)
        print(self.data.id())

    def __getitem__(self, index):
        item = self.items[index]
        res = item.split(' ')
        img_path = res[0]
        label = int(res[1])


        img_bin = self.data.get(img_path.encode())
        try:
            img_buf = np.frombuffer(img_bin, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        except:
            raise Exception("dataloader Error!")

        x_list = [int(float(res[6])), int(float(res[8])), int(float(res[10])), int(float(res[12])), int(float(res[14]))]
        y_list = [int(float(res[7])), int(float(res[9])), int(float(res[11])), int(float(res[13])), int(float(res[15]))]
        x, y = min(x_list), min(y_list)
        w, h = max(x_list) - x, max(y_list) - y
        side = w if w > h else h
        x1, x2, y1, y2 = self._add_face_margin(x, y, side, side, margin=self.margin)
        max_h, max_w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_w, x2)
        y2 = min(max_h, y2)
        if x1>=x2 or y1>=y2:
            return self.__getitem__(0)
        img = img[y1:y2, x1:x2]


        trans_base = alb.Compose([
            alb.Resize(self.img_size, self.img_size),
            alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])


        ori_img = img.copy()
        ori_img = trans_base(image=ori_img)["image"]

        if self.img_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.img_mode == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.img_mode == 'ycrcb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif self.img_mode == 'rgb_hsv':
            img_ori = img
            img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)

        assert self.transform is not None
        img = self.transform(image=img)["image"]

        if "hsv" in self.img_mode:
            img_hsv = trans_base(image=img_hsv)["image"]
            img = torch.cat([img, img_hsv], dim=1)
        else:
            img_hsv = None

        if self.split == "test":
            return img, label, index, res[0]
        else:
            return img, label, index


    def __len__(self):
        return len(self.items)

    def get_label(self):
        labels = []
        for item in self.items:
            res = item.split(' ')
            label = int(res[1])
            labels.append(label)
        return labels

def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
