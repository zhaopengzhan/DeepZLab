import glob
import random

import albumentations as A
import cv2
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from rasterio.windows import Window
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm



class BaseDataset():
    def __init__(self):
        pass

    def image_trans1(self, img):
        IMAGE_MEANS = np.array([414.02, 318.62, 245.42, 291.41])
        IMAGE_STDS = np.array([57.91, 58.17, 67.88, 85.84])

        img = (img - IMAGE_MEANS[:, None, None]) / IMAGE_STDS[:, None, None]

        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        return img

    def image_trans2(self, img):
        IMAGE_MEANS = np.array([86.37, 118.56, 115.96])
        IMAGE_STDS = np.array([57.44, 66.14, 64.98])

        img = (img - IMAGE_MEANS[:, None, None]) / IMAGE_STDS[:, None, None]

        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        return img

    def label_trans(self, target):
        target = target.astype(np.int64)
        target = torch.from_numpy(target)
        return target

    def seg_aug(self, image, mask):
        IMAGE_MEANS = np.array([414.02, 318.62, 245.42, 291.41])
        IMAGE_STDS = np.array([57.91, 58.17, 67.88, 85.84])

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
            additional_targets={"mask": "mask"}
        )

        return seg_aug(image=image, mask=mask)


class FBP(IterableDataset, BaseDataset):
    def __init__(self, image_list, label_list, chip_size=224, num_chips_per_tile=50, stride=None):
        super().__init__()
        self.fns = list(zip(image_list, label_list))

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

        for image_path, label_path in self.fns[lower_idx: upper_idx]:
            with rasterio.open(image_path) as src_img, \
                    rasterio.open(label_path) as src_lab:

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
                        label = src_lab.read(1, window=win)

                        # (C,H,W) → (H,W,C) 供 Albumentations
                        image = np.transpose(image, (1, 2, 0))

                        # + 3.4 增广 & 归一化
                        aug = self.seg_aug(image=image, mask=label)
                        image_patch, label_patch = aug["image"], aug["mask"].long()

                        if (label_patch == 0).float().mean() > 0.9:
                            continue

                        yield image_patch.float(), label_patch
                else:
                    x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                    y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                    for x in x_list:
                        for y in y_list:
                            win = Window(x, y, self.chip_size, self.chip_size)

                            # data
                            image = src_img.read(window=win)
                            label = src_lab.read(1, window=win)

                            image_patch = self.image_trans1(image)
                            label_patch = self.label_trans(label)

                            yield image_patch, label_patch

        pass


def test_25cls():
    image_list = glob.glob(r'F:\datasets\Remote Sensing\FBP\Image_16bit_BGRNir\*.tiff')[::5]
    label_list = [fn.replace('.tiff', '_24label.png')
                  .replace('Image_16bit_BGRNir', 'Annotation_25cls')
                  for fn in image_list]
    ds = GID(image_list, label_list, num_chips_per_tile=96)
    dl = DataLoader(ds, batch_size=32, num_workers=3, pin_memory=True)
    for img, label in tqdm(dl):
        # print(img.shape, label.shape)
        # print(np.unique(label.numpy()))
        # break
        pass


def test_5cls():
    image_list = glob.glob(r'F:\datasets\Remote Sensing\FBP\Image_16bit_BGRNir\*.tiff')[::5]
    label_list = [fn.replace('.tiff', '_5label.png')
                  .replace('Image_16bit_BGRNir', 'Annotation_5cls')
                  for fn in image_list]
    ds = GID(image_list, label_list, num_chips_per_tile=96)
    dl = DataLoader(ds, batch_size=32, num_workers=3, pin_memory=True)
    for img, label in tqdm(dl):
        # print(img.shape, label.shape)
        # print(np.unique(label.numpy()))
        # break
        pass


if __name__ == '__main__':
    test_5cls()
    pass
    # image_list = glob.glob(r'F:\datasets2\Five-Billion-Pixels\Image_16bit_BGRNir\*.tiff')
    # label_list = [fn.replace('.tiff', '_24label.png')
    #               .replace('Image_16bit_BGRNir', 'Annotation__index')
    #               for fn in image_list]
    # ds = GID(image_list, label_list,num_chips_per_tile=24)
    # dl = DataLoader(ds, batch_size=32, num_workers=3, pin_memory=True)
    # for img, label in tqdm(dl):
    #     # print(img.shape, label.shape)
    #     # print(np.unique(label.numpy()))
    #     # break
    #     pass
