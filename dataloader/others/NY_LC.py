import glob
import random

import albumentations as A
import cv2
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import IterableDataset, DataLoader
from utils.label_maps.lm_NY import train_mappings
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

    def image_trans1(self, img):
        img = rearrange(img, 'c h w -> h w c')

        IMAGE_MEANS = np.array([99.4735539, 106.00333557, 97.74878, 83.67548654])
        IMAGE_STDS = np.array([53.60597144, 51.5370033, 49.19684983, 61.40449262])

        img = (img - IMAGE_MEANS) / IMAGE_STDS
        img = rearrange(img, 'h w c -> c h w')

        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        return img

    def label_trans(self, target, type='', return_tensor=True):
        fcs10_list = [v['FCS10'] for v in train_mappings.values()]
        # 如果你想拍平成一个一维大list（比如 [51, 52, 61,...] 这样）
        from itertools import chain
        fcs10_flat_list = list(chain.from_iterable(fcs10_list))
        if type in ['lr', 'LR']:
            target = get_xx_label_map('FCS10_label', 'FCS10_label_train')[target]
            # 把 target 不在 fcs10_flat_list 的全设为0
            target = np.where(np.isin(target, fcs10_flat_list), target, 0)

        if type in ['hr', 'HR']:
            target = get_xx_label_map('UW_label', 'UW_4cls_label')[target]



        target = target.astype(np.int64)
        if return_tensor:
            target = torch.from_numpy(target)
        return target

    def seg_aug(self, image, mask_hr, mask_lr):
        IMAGE_MEANS = np.array([99.4735539, 106.00333557, 97.74878, 83.67548654])
        IMAGE_STDS = np.array([53.60597144, 51.5370033, 49.19684983, 61.40449262])

        # + 3.1 增广管线：已无 RandomCrop
        seg_aug = A.Compose(
            [
                # A.RandomCrop(256, 256, p=1.0),
                A.HorizontalFlip(p=0.5),  # 几何
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),  # 等价于 scale_limit=0.10
                    translate_percent=(-0.05, 0.05),  # shift_limit=0.05
                    rotate=(-15, 15),  # rotate_limit
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),
                # A.RandomBrightnessContrast(p=0.3),  # 光谱
                # A.GaussNoise( p=0.2),
                A.Lambda(image=lambda x, **kwargs: (x.astype('float32') - IMAGE_MEANS) / IMAGE_STDS),
                # A.Normalize(mean=IMAGE_MEANS.tolist(),
                #             std=IMAGE_STDS.tolist()),
                ToTensorV2(transpose_mask=True)
            ],
            additional_targets={
                "mask_hr": "mask",
                "mask_lr": "mask"
            }
        )

        return seg_aug(image=image, mask_hr=mask_hr, mask_lr=mask_lr)

    def nodata_check(self, img, labels):
        return np.any(labels.numpy() == 0) or np.any(np.sum(img.numpy() == 0, axis=0) == 4)


class NY_LC(IterableDataset, BaseDataset):
    def __init__(self, image_list, label_lr_list, label_hr_list,
                 chip_size=224, num_chips_per_tile=50, stride=None):
        super().__init__()
        self.fns = list(zip(image_list, label_hr_list, label_lr_list))

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

        for image_path, label_hr_path, label_lr_path in self.fns[lower_idx: upper_idx]:
            with rasterio.open(image_path) as src_img, \
                    rasterio.open(label_hr_path) as src_lab_hr, \
                    rasterio.open(label_lr_path) as src_lab_lr:

                height, width = src_img.height, src_img.width

                if not self.stride:
                    for i in range(self.num_chips_per_tile):
                        # + 3.2 随机窗口坐标
                        x = np.random.randint(0, width - self.chip_size)
                        y = np.random.randint(0, height - self.chip_size)

                        win = Window(x, y, self.chip_size, self.chip_size)
                        # data
                        # + 3.3 直接窗口读取，避免整幅进内存
                        image = src_img.read(window=win)  # shape: (C,256,256)
                        label_hr = src_lab_hr.read(1, window=win)
                        label_lr = src_lab_lr.read(1, window=win)

                        label_hr = self.label_trans(label_hr, type='HR', return_tensor=False)
                        label_lr = self.label_trans(label_lr, type='LR', return_tensor=False)

                        # (C,H,W) → (H,W,C) 供 Albumentations
                        image = np.transpose(image, (1, 2, 0))

                        # + 3.4 增广 & 归一化
                        aug = self.seg_aug(image=image, mask_hr=label_hr, mask_lr=label_lr)
                        image_patch = aug["image"]
                        label_hr_patch, label_lr_patch = aug["mask_hr"].long(), aug["mask_lr"].long()
                        if (label_lr_patch == 0).float().mean() > 0.99:
                            continue

                        yield image_patch.float(), label_hr_patch, label_lr_patch
                else:
                    x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                    y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                    for x in x_list:
                        for y in y_list:
                            win = Window(x, y, self.chip_size, self.chip_size)

                            # data
                            image = src_img.read(window=win)
                            label_hr = src_lab_hr.read(1, window=win)
                            label_lr = src_lab_lr.read(1, window=win)

                            image_patch = self.image_trans1(image)
                            label_hr_patch = self.label_trans(label_hr, type='HR')
                            label_lr_patch = self.label_trans(label_lr, type='LR')

                            yield image_patch, label_hr_patch, label_lr_patch

        pass


if __name__ == '__main__':

    image_re_path = r'F:\datasets2\NY_lc\Image\*.tif'

    batch_size = 32
    num_workers = 3
    num_chips_per_tile = 32
    image_size = 224
    stride = image_size // 2

    train_image_list = glob.glob(image_re_path)
    train_label_lr_list = [filename.replace('Image', 'LR_label') for filename in train_image_list]
    train_label_hr_list = [filename.replace('Image', 'HR_label') for filename in train_image_list]

    train_ds = NY_LC(train_image_list, train_label_lr_list, train_label_hr_list,
                     chip_size=image_size, num_chips_per_tile=num_chips_per_tile)

    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                          pin_memory=True)

    for image, label_lr, label_hr in train_dl:
        print(np.unique(label_lr))
        print(np.unique(label_hr))
        print(image.shape)
        print(torch.mean(image))
        print(torch.std(image))
        break
    pass
