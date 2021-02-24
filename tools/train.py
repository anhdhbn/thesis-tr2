from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import random
import os
from re import template
import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tr2.core.config import cfg
from tr2.utils.distributed import get_world_size
from tr2.datasets.dataset import TrkDataset
from tr2.utils.log_helper import init_log, add_file_handler
from tr2.models.tr2 import build_tr2
from tr2.utils.misc import collate_fn
from tr2.utils.model_loader import load_pretrain, restore_from
from typing import Iterable
import math
from tqdm import tqdm

try:
    from apex import amp
except ModuleNotFoundError:
    amp = None


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='tracking transformer')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=1699,
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

def build_data_loader(subset="train"):
    logger.info("build train dataset")
    data_dataset = TrkDataset(subset)
    logger.info("build dataset done")

    data_sampler = None
    # if get_world_size() > 1:
    #     data_sampler = DistributedSampler(data_dataset)
    data_loader = DataLoader(data_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=data_sampler,
                              collate_fn=collate_fn,
                              drop_last=True)
    return data_loader

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm=0.1):
    model.train()
    criterion.train()
    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)
    for idx, (template, search, label_cls, label_bbox) in enumerate(data_loader):
        out = model(template.to(device), search.to(device))
        outputs = criterion(out,(label_cls.to(device), label_bbox.to(device)))
        if outputs is not None:
            loss = outputs['total_loss']
            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                if amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                info = "train epoch: [{}] {}/{}\n ".format(
                                epoch,
                                idx, len(data_loader),)
                for cc, (k, v) in enumerate(outputs.items()):
                    if cc % 2 == 0:
                        info += ("\t{name}: {val:.6f}\t").format(name=k,
                                val=outputs[k])
                    else:
                        info += ("{name}: {val:.6f}\n").format(name=k,
                                val=outputs[k])
                logger.info(info)

def evaluate(model: nn.Module, criterion: nn.Module, data_loader: Iterable, device: torch.device, epoch: int):
    model.eval()
    criterion.eval()
    all_outputs = {}
    pbar = tqdm(enumerate(data_loader))
    for idx, (template, search, label_cls, label_bbox) in pbar:
        out = model(template.to(device), search.to(device))
        outputs = criterion(out,(label_cls.to(device), label_bbox.to(device)))
        if outputs is not None:
            for k,v in outputs.items():
                if k not in all_outputs.keys(): all_outputs[k] = outputs[k].cpu().detach()
                else: all_outputs[k] += outputs[k].cpu().detach()
            current_loss = ("total loss: {:.6f}, giou: {:.6f}, iou: {:.6f}").format(
                all_outputs["total_loss"] / (idx + 1), 
                all_outputs["giou_loss"] / (idx + 1),
                all_outputs["iou_loss"] / (idx + 1)
            )
            pbar.set_description(current_loss)
    
    info = f"Val epoch: [{epoch}]\n"

    for cc, (k, v) in enumerate(all_outputs.items()):
        if cc % 2 == 0:
            info += ("\t{name}: {val:.6f}\t").format(name=k,
                    val=all_outputs[k]/len(data_loader))
        else:
            info += ("{name}: {val:.6f}\n").format(name=k,
                    val=all_outputs[k]/len(data_loader))
    logger.info(info)

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

    model, criterion = build_tr2(
        hidden_dims=cfg.TRANSFORMER.KWARGS['hidden_dims'],
        transformer_kwargs=cfg.TRANSFORMER.KWARGS,
        loss_weight=cfg.TRAIN.WEIGHT
    )
    criterion.to(device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.BACKBONE.LR,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40)

    dataset_train = build_data_loader()
    dataset_val = build_data_loader("val") if cfg.TRAIN.VAL_LOSS else None

    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH, amp = \
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

    logger.info(f"apex: {amp is not None}")
    logger.info(lr_scheduler)
    logger.info("model prepare done")
    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCH):
        train_one_epoch(model, criterion, dataset_train, optimizer, device, epoch, max_norm=0.1)
        lr_scheduler.step()
        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict() if amp else None
            },
            cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))
        dataset_train.dataset.shuffle()
        if dataset_val:
            evaluate(model, criterion, dataset_val, device, epoch)

if __name__ == '__main__':
    seed_torch(args.seed)
    main()