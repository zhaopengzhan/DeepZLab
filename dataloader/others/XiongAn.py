import random

import numpy as np
import rasterio
import torch
from einops import rearrange
from rasterio.enums import Resampling
from torch.utils.data import IterableDataset
from utils.label_maps import get_xx_label_map


class XiongAn(IterableDataset):
    # def __init__(self, image_list, chip_size=224, num_chips_per_tile=50):
    def __init__(self, image_list, label_lr_list, label_type, label_hr_list=None, chip_size=224, num_chips_per_tile=50,
                 ):
        # 路径集合
        self.fns = list(zip(image_list, label_lr_list, label_hr_list))

        self.label_type = label_type

        self.chip_size = chip_size

        self.num_chips_per_tile = num_chips_per_tile

    def __len__(self):
        return len(self.fns) * self.num_chips_per_tile

    def __iter__(self):
        return iter(self.create_stream())

    def create_stream(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # In this case we are not loading through a DataLoader with multiple workers
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

        for image_path, label_lr_path in self.fns[lower_idx: upper_idx]:
            # file object
            image_fp = rasterio.open(image_path, "r")
            label_lr_fp = rasterio.open(label_lr_path, "r")

            # data
            image = image_fp.read()
            label_lr = label_lr_fp.read()

            # 转Tensor
            image = self.image_trans(image)
            # label_lr = self.label_trans(label_lr, in_type=self.label_type, out_type=f'{self.label_type}_train')
            label_lr = self.label_trans(label_lr, in_type=self.label_type, out_type=f'Target_4_cls')

            height, width = image.shape[-2:]

            x_list = np.random.randint(0, width - self.chip_size, size=(self.num_chips_per_tile))
            y_list = np.random.randint(0, height - self.chip_size, size=(self.num_chips_per_tile))

            for i in range(self.num_chips_per_tile):
                x = x_list[i]
                y = y_list[i]

                image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                label_lr_patch = label_lr[..., y:y + self.chip_size, x:x + self.chip_size]

                # 舍去channel维度
                label_lr_patch = label_lr_patch.squeeze()

                yield image_patch, label_lr_patch
        pass

    def clip_and_align(self, src_ds, target_ds):
        # 裁剪对齐
        extend = target_ds.bounds
        # 根据地理范围计算裁剪窗口
        window = src_ds.window(*extend)
        # 读取裁剪后的数据
        result = src_ds.read(window=window,
                             out_shape=(src_ds.count, int(window.height), int(window.width)),
                             resampling=Resampling.nearest)

        return result

    def image_trans(self, img):
        img = rearrange(img, 'c h w -> h w c')
        IMAGE_MEANS = np.array([117.67, 130.39, 121.52])  # The setting here is for Chesapeake dataset
        IMAGE_STDS = np.array([39.25, 37.82, 24.24])
        img = (img - IMAGE_MEANS) / IMAGE_STDS
        img = rearrange(img, 'h w c -> c h w')

        # c h w
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        return img

    def label_trans(self, target, in_type='', out_type=''):
        # target = get_esa_to_train_map()[target]
        target = get_xx_label_map(in_type=in_type, out_type=out_type)[target]

        target = target.astype(np.int32)
        target = torch.from_numpy(target)
        return target
