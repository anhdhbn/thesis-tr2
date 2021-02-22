from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from got10k.datasets import GOT10k

from tr2.core.config import cfg
from tr2.datasets import transforms as T
import torch
import random
import os
from tr2.utils import box_ops

logger = logging.getLogger("global")

def make_got10k_transforms(image_set):
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    template_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=1333),
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            )
        ]), template_transforms, normalize

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333)
        ]), normalize

    raise ValueError(f'unknown {image_set}')

class TrkDataset(Dataset):
    def __init__(self, dataset, subset="train") -> None:
        super().__init__()
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', "training_dataset/got10k")
        self.dataset = GOT10k(self.root , subset=subset, return_meta=True)
        self.transforms_search, self.transforms_template, self.transform_norm = make_got10k_transforms(subset)
        self.indices = np.random.permutation(len(self.dataset))
    
    def shuffle(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_one(index)
        else:
            out = [ self.get_one(i) for i in index]
            template = [o[0] for o in out]
            search = [o[1] for o in out]
            targets = [o[2] for o in out]
            
            return template, search, targets

    def get_one(self, index):
        index = self.indices[index % len(self.dataset)]
        ignore = [1204,4224,4418,7787,7964,9171,9176]
        while index in ignore:
            index = np.random.choice(len(self.dataset))
            index = self.indices[index % len(self.dataset)]
        img_files, anno, meta = self.dataset[index]

        # search
        idx = random.randrange(1, len(img_files))
        search, bbox = Image.open(img_files[idx]), self.cvt_x0y0wh_xyxy(anno[idx, :])
        search, target = self.transforms_search(search, {"boxes": bbox})
        search, target = self.transform_norm(search, target)

        # template
        src, bbox_src = Image.open(img_files[0]), self.cvt_x0y0wh_xyxy(anno[0, :])
        src, target_src = self.transforms_template(src, {"boxes": bbox_src})
        template = src.crop(self.cvt_int(target_src["boxes"]))
        template, _ = self.transform_norm(template, None)
        
        label_cls = torch.tensor([1.0])
        label_bbox = target["boxes"].float()

        if meta['absence'][idx] == 1 or len(label_bbox) == 0:
            label_cls = torch.tensor([0.0])
            if len(label_bbox) == 0:
                label_bbox = torch.zeros(1, 4)
        return template, search, label_cls, label_bbox.squeeze(0)
    
    def __len__(self):
        return 10 * len(self.dataset)

    def cvt_x0y0wh_xyxy(self, box):
        x0, y0 ,w, h = box
        return torch.tensor([x0, y0, x0+w, y0+h]).unsqueeze(0)

    def cvt_int(self, box):
        x1, y1, x2, y2 = box.squeeze(0)
        return int(x1), int(y1), int(x2), int(y2)
class Got10kVal:
    def __init__(self, subset="val") -> None:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', "training_dataset/got10k")
        self.dataset = GOT10k(self.root , subset=subset, return_meta=False)
        self.transforms_img, self.transform_norm = make_got10k_transforms(subset)
        self.video_idx = 0
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def transforms(self, img_path, bbox, is_template=False):
        bbox = self.cvt_x0y0wh_xyxy(bbox)
        image_src = Image.open(img_path)
        w, h = image_src.size
        image, target = self.transforms_img(image_src, {"boxes": bbox, "orig_size": torch.tensor([h, w], dtype=bbox.dtype)})
        if is_template:
            image = image.crop(self.cvt_int(target["boxes"]))
        image_norm, target_norm = self.transform_norm(image, target)
        return image_src, image_norm, target_norm

    def cvt_int(self, box):
        x1, y1, x2, y2 = box.squeeze(0)
        return int(x1), int(y1), int(x2), int(y2)

    def cvt_x0y0wh_xyxy(self, box):
        x0, y0 ,w, h = box
        return torch.tensor([x0, y0, x0+w, y0+h]).unsqueeze(0)