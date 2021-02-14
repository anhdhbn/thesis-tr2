from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import socket
import logging

import torch
import torch.nn as nn
import torch.distributed as dist


logger = logging.getLogger('global')

inited = False

def get_world_size():
    if not inited:
        raise(Exception('dist not inited'))
    return world_size