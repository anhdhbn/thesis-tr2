from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from torch.utils.data import Dataset

from tr2.core.config import cfg
from tr2.datasets.wrapper import GOT10kWrapper

logger = logging.getLogger("global")

datasets = {
    'GOT10K': GOT10kWrapper,
}

class TrkDataset(Dataset):
    def __init__(self, subset="train") -> None:
        super().__init__()
        assert subset in ['train', "val"], 'Unknown subset.'

        self.all_dataset = []
        start = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            frame_per_video=subdata_cfg.FRAMES_PER_VIDEO
            if subset == "val" and subdata_cfg.FRAMES_PER_VIDEO_VAL is not None:
                frame_per_video=subdata_cfg.FRAMES_PER_VIDEO_VAL
            sub_dataset = datasets[name](
                name,
                subdata_cfg.ROOT,
                subset=subset,
                frame_per_video=frame_per_video,
                start_idx=start
            )
            self.all_dataset.append(sub_dataset)
            sub_dataset.log()
            start += len(sub_dataset)

        self.length = 0
        for dataset in self.all_dataset:
            self.length += len(dataset)

    def shuffle(self):
        for dataset in self.all_dataset:
            dataset.shuffle()

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.length > index:
                return dataset, index - dataset.start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dataset, index = self._find_dataset(index)

        # todos: negative
        
        return dataset[index]