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
from tr2.datasets.dataset import Got10kVal
from tr2.utils.log_helper import init_log, add_file_handler
from tr2.models.tr2 import build_tr2
from tr2.utils.misc import collate_fn
from tr2.utils import box_ops
from typing import Iterable
import math
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
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



def main():
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

    model, criterion = build_tr2()
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

    validator = Got10kVal()
    
    folder = "result"
    for video_idx in tqdm(range(len(validator))):
        img_paths, anno = validator[video_idx]
        
        template, template_norm, _ = validator.transforms(img_paths[0], anno[0, :], is_template=True)
        model.init(nested_tensor_from_tensor_list([template_norm.to(device)]))

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        video = cv2.VideoWriter(f"{folder}/{validator.dataset.seq_names[video_idx]}.avi", fourcc, 15, template.size)

        for idx in tqdm(range(1, len(img_paths))):
            search, search_norm, target = validator.transforms(img_paths[idx], anno[idx, :])
            cls, boxes = model.track(nested_tensor_from_tensor_list([search_norm.to(device)]))
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            img_h, img_w = target["orig_size"]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h]).unsqueeze(0)
            boxes = boxes * scale_fct.to(device)
            x1, y1, x2, y2 = validator.cvt_int(boxes)
            draw = ImageDraw.Draw(search)
            draw.rectangle((x1, y1, x2, y2), fill=None, outline=(255, 0, 0), width=2)
            del draw
            search_np = search.copy()
            video.write(cv2.cvtColor(np.array(search_np), cv2.COLOR_RGB2BGR))
        
        video.release()
        del video


if __name__ == '__main__':
    seed_torch(args.seed)
    main()