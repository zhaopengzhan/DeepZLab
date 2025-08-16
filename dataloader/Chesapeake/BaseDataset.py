import numpy as np
import torch
from einops import rearrange
from rasterio.enums import Resampling

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

    def label_trans(self, target, type=''):
        # 转化成4类训练
        if type in ['lr', 'LR']:
            # 将大于等于96的数字标记为0
            if target.max() > 96:
                target[target > 96] = 0
            target = get_xx_label_map('nlcd_label', 'nlcd_label_train')[target]

        if type in ['hr', 'HR']:
            # target = get_xx_label_map('lc_label', 'lc_label_train')[target]
            pass

        target = target.astype(np.int64)
        target = torch.from_numpy(target)
        return target

    def nodata_check(self, img, labels):
        return np.any(labels.numpy() == 0) or np.any(np.sum(img.numpy() == 0, axis=0) == 4)
