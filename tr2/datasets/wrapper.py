import torch
from torch.utils.data import Dataset
import logging, os, random
import numpy as np
from got10k.datasets import GOT10k
from PIL import Image

from tr2.datasets import transforms as T

logger = logging.getLogger("global")

def cvt_x0y0wh_xyxy(box):
    x0, y0 ,w, h = box
    return torch.tensor([x0, y0, x0+w, y0+h]).unsqueeze(0)

def cvt_int(box):
    x1, y1, x2, y2 = box.squeeze(0)
    return int(x1), int(y1), int(x2), int(y2)

def norm_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def train_transforms():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    template_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=1333),
    ])

    search_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        )
    ])
    return template_transforms, search_transforms

def val_transforms():
    val_transforms = T.Compose([
        T.RandomResize([800], max_size=1333)
    ])
    return val_transforms, val_transforms

def make_transforms(subset):
    assert subset in ['train', 'val', "test"], 'Unknown subset.'
    if subset == "train": return train_transforms(), norm_transforms()
    else: return val_transforms(), norm_transforms()

class GOT10kWrapper(Dataset):
    def __init__(self, name, root, subset, frame_per_video, start_idx) -> None:
        super().__init__()
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root) if not os.path.isabs(root) else root
        assert subset in ['train', 'val'], 'Unknown subset.'
        self.subset = subset
        self.dataset = GOT10k(self.root , subset=subset, return_meta=True)
        self.length = len(self.dataset) * frame_per_video
        self.start_idx = start_idx
        self.indices = np.random.permutation(len(self.dataset))
        (self.template_transforms, self.search_transforms), self.norm_transform = make_transforms(subset)
        self.ignore = sorted([1204,4224,4418,7787,7964,9171,9176]) if subset == "train" else []

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        index = self.indices[index % len(self.dataset)]
        while index in self.ignore:	
            index = np.random.choice(len(self.dataset))	
            index = self.indices[index % len(self.dataset)]
        img_files, anno, meta = self.dataset[index]
        # search
        idx = random.randrange(1, len(img_files))
        search, bbox = Image.open(img_files[idx]), cvt_x0y0wh_xyxy(anno[idx, :])
        search, target = self.search_transforms(search, {"boxes": bbox})
        search, target = self.norm_transform(search, target)

        # template
        src, bbox_src = Image.open(img_files[0]), cvt_x0y0wh_xyxy(anno[0, :])
        src, target_src = self.template_transforms(src, {"boxes": bbox_src})

        template = src.crop(cvt_int(target_src["boxes"]))
        template, _ = self.norm_transform(template, None)
        
        label_cls = torch.tensor([1.0])
        label_bbox = target["boxes"].float()

        if meta['absence'][idx] == 1 or len(label_bbox) == 0:
            label_cls = torch.tensor([0.0])
            if len(label_bbox) == 0:
                label_bbox = torch.zeros(1, 4)
        return template, search, label_cls, label_bbox.squeeze(0)

    def __len__(self):
        return self.length

    def log(self):
        logger.info(f"Loading {self.name}, subset: {self.subset}, len: {self.length}, start-index: {self.start_idx}")


class VisualizeGot10k:
    def __init__(self, root, subset="val") -> None:
        assert subset in ['val', "test"], 'Unknown subset.'
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.join(cur_path, '../../', root) if not os.path.isabs(root) else root
        self.is_testing = (subset != "val")
        self.dataset = GOT10k(self.root , subset=subset, return_meta=self.is_testing)
        (self.transforms_img, _), self.transform_norm = make_transforms(subset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def transforms(self, img_path, bbox, is_template=False):
        bbox = cvt_x0y0wh_xyxy(bbox)
        src_box = bbox.clone()
        image_src = Image.open(img_path)
        w, h = image_src.size
        image, target = self.transforms_img(image_src, {"boxes": bbox, "orig_size": torch.tensor([h, w], dtype=bbox.dtype)})
        if is_template:
            image = image.crop(cvt_int(target["boxes"]))
        image_norm, target_norm = self.transform_norm(image, target)
        return image_src, image_norm, target_norm, src_box.squeeze(0)