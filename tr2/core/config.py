from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "tr2_r50_l4"

__C.CUDA = True

__C.APEX = True
# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.WEIGHT = CN()

__C.TRAIN.WEIGHT.cls_weight = 1

__C.TRAIN.WEIGHT.loc_weight = 1.2

__C.TRAIN.WEIGHT.giou_weight = 1.2

__C.TRAIN.LR = 1e-4

__C.TRAIN.WEIGHT_DECAY = 1e-4

__C.TRAIN.APEX = True

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #

__C.DATASET = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

__C.BACKBONE.TYPE = 'resnet50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

__C.BACKBONE.LR = 1e-5


# ------------------------------------------------------------------------ #
# Transformer options
# ------------------------------------------------------------------------ #

__C.TRANSFORMER = CN()

__C.TRANSFORMER.KWARGS = CN(new_allowed=True)