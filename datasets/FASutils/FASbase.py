import random
import cv2
import numpy as np
import lmdb
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset
import random



class FASCLDataset(Dataset):
    def __init__(self, dataset="oulu", split="train", task="A", margin=0.7, mode='rgb', img_size=256,
                 transform=None):

        self.dataset = dataset
        self.split = split
        self.margin = margin
        self.task = task
        self.img_mode = mode
        self.trandform = transform
        self.img_size = img_size

        LMDB_root = "PATH"
        self.env = lmdb.open(LMDB_root, readonly=True, max_readers=10240)
        self.data = self.env.begin(write=False)

        if "+" not in self.task:
            if split == "train":
                self.items = open("PATH/{}_{}_{}.list".format(dataset, split, task)).read().splitlines()
            else:
                self.items = open("PATH/{}_{}_{}.list".format(dataset, split, task)).read().splitlines()
            if split == "train":
                self.items = self.items
        else:
            items = []
            for t in self.task.split("+"):
                if split == "train":
                    t_items = open("PATH/{}_{}_{}.list".format(dataset, split, t)).read().splitlines()
                else:
                    t_items = open("PATH/{}_{}_{}.list".format(dataset, split, t)).read().splitlines()
                if split == "train":
                    items.extend(t_items)
            self.items = items
            
        self._display_infos()

    def _display_infos(self):
        print(f'=> Dataset {self.dataset} loaded')
        print(f'=> Split {self.split}')
        print(f'=> Task {self.task}')
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

    def _reset_lmdb(self):
        self.env.close()
        self.env.close()
        self.env = lmdb.open(self.LMDB_root, readonly=True, max_readers=1024)
        self.data = self.env.begin(write=False)
        print(self.data.id())

    def get_label(self):
        data_map = {
            "oulu": 0,
            "siw": 1,
            "siwmv2": 2
        }
        labels = []
        for item in self.items:
            database = item.split(" ")[0].split("/")[0]
            real = 0 if int(item.split(" ")[0].split("/")[2]) == 0 else 1
            labels.append(data_map[database] * 2 + real)
        return labels

    def __getitem__(self, index):
        item = self.items[index]
        res = item.split(' ')
        img_path = res[0]
        label = 0 if int(res[0].split('/')[2]) == 0 else 1

        img_bin = self.data.get(img_path.encode())
        try:
            img_buf = np.frombuffer(img_bin, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        except:
            return self.__getitem__(0)

        x_list = [int(float(res[5])), int(float(res[7])), int(float(res[9])), int(float(res[11])),
                  int(float(res[13]))]
        y_list = [int(float(res[6])), int(float(res[8])), int(float(res[10])), int(float(res[12])),
                  int(float(res[14]))]

        x, y = min(x_list), min(y_list)
        w, h = max(x_list) - x, max(y_list) - y

        side = w if w > h else h

        x1, x2, y1, y2 = self._add_face_margin(x, y, side, side, margin=self.margin)
        max_h, max_w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_w, x2)
        y2 = min(max_h, y2)
        if x1 >= x2 or y1 >= y2:
            return self.__getitem__(0)
        img = img[y1:y2, x1:x2]


        trans_base = alb.Compose([
            alb.Resize(self.img_size, self.img_size),
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

        assert self.trandform is not None
        img = self.trandform(image=img)["image"]

        if "hsv" in self.img_mode:
            img_hsv = trans_base(image=img_hsv)["image"]
            img = torch.cat([img, img_hsv], dim=1)
        else:
            img_hsv = None

        if self.split == "test":
            return img, label, index, res[0]
        else:
            return img, label, index, res[0]

    def __len__(self):
        return len(self.items)
