import random

import numpy as np
import rasterio
import torch
from torch.utils.data import IterableDataset

from dataloader import DeepZData
from utils.misc import calRunTime
from .BaseDataset import BaseDataset


@DeepZData.register_module()
class Chesapeake_L2H(IterableDataset, BaseDataset):
    def __init__(self, image_list, label_lr_list, label_hr_list, chip_size=224, num_chips_per_tile=50, stride=None):
        super().__init__()
        self.fns = list(zip(image_list, label_lr_list, label_hr_list))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.stride = stride

    def __len__(self):
        return len(self.fns) * self.num_chips_per_tile

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        random.shuffle(self.fns)

        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)

        for image_path, label_lr_path, label_hr_path in self.fns[lower_idx: upper_idx]:
            # file object
            image_fp = rasterio.open(image_path, "r")
            label_lr_fp = rasterio.open(label_lr_path, "r")
            label_hr_fp = rasterio.open(label_hr_path, "r")

            # data
            label_hr = label_hr_fp.read()
            # image = image_fp.read()
            # label_lr = label_lr_fp.read()
            # 裁剪对齐读取
            image = self.clip_and_align(image_fp, label_hr_fp)
            label_lr = self.clip_and_align(label_lr_fp, label_hr_fp)

            # 转Tensor
            # TODO: 检查用哪个trans
            image = self.image_trans1(image)
            label_lr = self.label_trans(label_lr, 'lr')
            label_hr = self.label_trans(label_hr, 'hr')

            height, width = image.shape[-2:]  # 6389

            if not self.stride:
                x_list = np.random.randint(0, width - self.chip_size, size=(self.num_chips_per_tile))
                y_list = np.random.randint(0, height - self.chip_size, size=(self.num_chips_per_tile))

                for i in range(self.num_chips_per_tile):
                    x = x_list[i]
                    y = y_list[i]

                    image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                    label_lr_patch = label_lr[..., y:y + self.chip_size, x:x + self.chip_size]
                    label_hr_patch = label_hr[..., y:y + self.chip_size, x:x + self.chip_size]

                    # 舍去channel维度
                    label_lr_patch = label_lr_patch.squeeze()
                    label_hr_patch = label_hr_patch.squeeze()

                    yield image_patch, label_lr_patch, label_hr_patch
            else:
                x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                for x in x_list:
                    for y in y_list:
                        image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                        label_lr_patch = label_lr[..., y:y + self.chip_size, x:x + self.chip_size]
                        label_hr_patch = label_hr[..., y:y + self.chip_size, x:x + self.chip_size]

                        # 舍去channel维度
                        label_lr_patch = label_lr_patch.squeeze()
                        label_hr_patch = label_hr_patch.squeeze()

                        yield image_patch, label_lr_patch, label_hr_patch, np.array((y, x))

        pass

    @classmethod
    def build(cls):
        import glob
        image_re_path = r'F:\zpz\datasets\Remote Sensing\cvpr_chesapeake_landcover\ny_1m_2013_extended-debuffered-train_tiles\*_naip-new.tif'
        train_image_list = glob.glob(image_re_path)
        train_label_lr_list = [filename.replace('_naip-new.tif', '_nlcd.tif') for filename in train_image_list]
        train_label_hr_list = [filename.replace('_naip-new.tif', '_lc.tif') for filename in train_image_list]

        return cls(train_image_list, train_label_lr_list, train_label_hr_list)
