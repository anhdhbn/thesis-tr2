from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import random
import os
from re import search, template
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tr2.core.config import cfg
from tr2.utils.distributed import get_world_size
from tr2.datasets import wrapper
from tr2.utils.log_helper import init_log, add_file_handler
from tr2.models.tr2 import build_tr2
from tr2.utils.misc import collate_fn
from tr2.utils import box_ops
from typing import Iterable
import math
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from glob import glob
from tr2.utils.misc import nested_tensor_from_tensor_list

try:
    from apex import amp
except ModuleNotFoundError:
    amp = None


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='tracking transformer')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
parser.add_argument('--subset', type=str, default="val")
args = parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True

def restore_from(model, optimizer, ckpt_path, apex):
    ckpt = torch.load(ckpt_path,map_location='cpu')
    epoch = ckpt['epoch']

    check_keys(model, ckpt['state_dict'])
    model.load_state_dict(ckpt['state_dict'], strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])

    if apex is not None:
        model, optimizer = apex.initialize(
            model, optimizer, opt_level="O1", 
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )
        if ckpt['amp'] is not None:
            check_keys(apex, ckpt['amp'])
            apex.load_state_dict(ckpt['amp'])
    return model, optimizer, epoch, apex

def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def load_model():
    logger.info("init done")
    cfg.merge_from_file(args.cfg)
    device = torch.device("cuda" if cfg.CUDA else "cpu")

    global amp
    if not cfg.APEX:
        amp = None
    
    if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)
    if cfg.TRAIN.LOG_DIR:
        add_file_handler('global',
                            os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                            logging.INFO)

    # logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    model, criterion = build_tr2(
        hidden_dims=cfg.TRANSFORMER.KWARGS['hidden_dims'],
        transformer_kwargs=cfg.TRANSFORMER.KWARGS,
        loss_weight=cfg.TRAIN.WEIGHT
    )

    criterion.to(device)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.BACKBONE.LR,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH, apex = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME, amp)
    
    else:
        # load pretrain
        if cfg.TRAIN.PRETRAINED:
            load_pretrain(model, cfg.TRAIN.PRETRAINED)
        # amp
        if amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O1", 
                keep_batchnorm_fp32=None, loss_scale="dynamic"
            )
    model.eval()
    return model, device


from got10k.trackers import Tracker

class IdentityTracker(Tracker):
    """Example on how to define a tracker.

        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self, model, evaluator: wrapper.EvaluateGot10K, device):
        super(IdentityTracker, self).__init__(
            name='Tr2', # name of the tracker
            is_deterministic=True   # deterministic (True) or stochastic (False)
        )
        self.model = model
        self.evaluator = evaluator
        self.device = device
        self.tmp_dir = "tmp"
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        template, template_cvt, template_norm, _, _ = self.evaluator.transforms(image, box, True)
        self.model.init(nested_tensor_from_tensor_list([template_norm.to(device)]))
        x, y, w, h = box
        self.center = [x + w/2, y + h/2, w, h]

    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        cx, cy, w, h = self.center
        box = [cx - w/2, cy - h/2, w, h]
        search, search_cvt, search_norm, target, src_box = self.evaluator.transforms(image, box)
        h_orig, w_orig  = target['orig_size']

        cls, boxes = model.track(nested_tensor_from_tensor_list([search_norm.to(device)]))
        if 'anchor' not in target.keys():
            self.center = [w_orig/2, h_orig/2, w_orig, h_orig]
            return [0, 0, h_orig, w_orig]
        # print(cls.sigmoid(), self.center, box, target['anchor'])
        (edgex, edgey), _, (w_img, h_img) = target['anchor']
        w_img_cvt, h_img_cvt = search_cvt.size
        w_img_cvt = torch.tensor(w_img_cvt, dtype=boxes.dtype)
        h_img_cvt = torch.tensor(h_img_cvt, dtype=boxes.dtype)
        
        boxesxyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxesxyxy = boxesxyxy * torch.stack([w_img_cvt, h_img_cvt, w_img_cvt, h_img_cvt]).unsqueeze(0).to(device)
        boxesxyxy = boxesxyxy + torch.stack([edgex, edgey, edgex, edgey]).unsqueeze(0).to(device)
        boxesxyxy = boxesxyxy / torch.stack([w_img, h_img, w_img, h_img]).unsqueeze(0).to(device)
        boxesxyxy = boxesxyxy * torch.stack([w_orig, h_orig, w_orig, h_orig]).unsqueeze(0).to(device)

        cx_new, cy_new, w_new, h_new = wrapper.cvt_int(box_ops.box_xyxy_to_cxcywh(boxesxyxy))
        cx_new = max(0, cx_new)
        cy_new = max(0, cy_new)
        self.center = [cx_new, cy_new, w_new, h_new]
        
        # Dasiam
        if cx_new / w_orig > 0.85 or cy_new / h_orig > 0.85:
            self.center = [cx_new, cy_new, w_orig / 2, h_orig / 2]
        if w_new / h_new > 7.5 or w_new / h_new < 1 / 7.5:
            self.center = [cx_new, cy_new, w_orig / 2, h_orig / 2]
        

        x = cx_new - w_new/2
        y = cy_new - h_new/2
        x = max(0, x)
        y = max(0, y)

        return [y, x, h_new, w_new]

if __name__ == '__main__':
    seed_torch(args.seed)
    model, device = load_model()
    evaluator = wrapper.EvaluateGot10K(cfg.DATASET.GOT10K.ROOT, args.subset)
    tracker = IdentityTracker(model, evaluator, device)

    evaluator.experiment.run(tracker, visualize=False)
    evaluator.experiment.report([tracker.name])
    evaluator.experiment.show([tracker.name])