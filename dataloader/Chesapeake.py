import random

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from einops import rearrange
from rasterio.enums import Resampling
from torch.utils.data import IterableDataset,Dataset,DataLoader
from torch.utils.data import Dataset
from utils.label_maps import get_xx_label_map


class BaseDataset():
    def __init__(self):
        pass

    def clip_and_align(self, src_ds, tgt_ds):
        # 输入的是rasterio对象，没法直接根据numpy数组裁切
        # 裁剪对齐
        extend = tgt_ds.bounds
        # 根据地理范围计算裁剪窗口
        window = src_ds.window(*extend)
        # 读取裁剪后的数据
        result = src_ds.read(window=window,
                             out_shape=(src_ds.count, int(window.height), int(window.width)),
                             resampling=Resampling.nearest)

        return result

    def image_trans1(self, img, in_channels=4):
        img = rearrange(img, 'c h w -> h w c')

        if in_channels == 3:
            IMAGE_MEANS = np.array([117.67, 130.39, 121.52])  # The setting here is for Chesapeake dataset
            IMAGE_STDS = np.array([39.25, 37.82, 24.24])

        if in_channels == 4:
            IMAGE_MEANS = np.array([117.67, 130.39, 121.52, 162.92])  # The setting here is for Chesapeake dataset
            IMAGE_STDS = np.array([39.25, 37.82, 24.24, 60.03])

        img = (img - IMAGE_MEANS) / IMAGE_STDS
        img = rearrange(img, 'h w c -> c h w')

        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        return img

    def image_trans2(self, img):
        image_patch = rearrange(img, 'c h w -> h w c')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image_ts = transform(image_patch[:, :, :3])
        return image_ts

    def image_trans3(self, img):
        img = rearrange(img, 'c h w -> h w c')
        IMAGE_MEANS = np.array([117.67, 130.39, 121.52, 162.92])
        IMAGE_STDS = np.array([39.25, 37.82, 24.24, 60.03])
        img = (img - IMAGE_MEANS) / IMAGE_STDS
        img = rearrange(img, 'h w c -> c h w')

        # c h w
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        # generate indexs
        indexs = []
        for i in range(img.size(0)):
            for j in range(i + 1, img.size(0)):
                a = img[i]
                b = img[j]
                index_feat = (a - b) / (a + b)
                index_feat_std = (index_feat - torch.mean(index_feat)) / (torch.std(index_feat) + 1e-5)
                # print('a, b', i, j)

                indexs.append(index_feat_std)
        indexs = torch.stack(indexs)

        # img = torch.cat([img, indexs], dim=0)
        img = torch.cat([img, indexs[0].unsqueeze(0), indexs[5].unsqueeze(0)], dim=0)

        return img

    def label_trans(self, target, type=''):
        # 转化成4类训练
        if type in ['lr', 'LR']:
            # 将大于等于96的数字标记为0
            if target.max() > 96:
                target[target > 96] = 0
            target = get_xx_label_map('nlcd_label', 'nlcd_label_train')[target]

        if type in ['hr', 'HR']:
            target = get_xx_label_map('lc_label', 'Target_4_cls')[target]

        target = target.astype(np.int64)
        target = torch.from_numpy(target)
        return target

    def nodata_check(self, img, labels):
        return np.any(labels.numpy() == 0) or np.any(np.sum(img.numpy() == 0, axis=0) == 4)


class Chesapeake(IterableDataset, BaseDataset):
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
