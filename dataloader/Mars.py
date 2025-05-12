import glob
import random

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm


class BaseDataset():
    def __init__(self):
        pass

    def image_trans1(self, img):
        img = img.astype('float32')
        image_patch = rearrange(img, 'c h w -> h w c')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),

            ]
        )
        image_ts = transform(image_patch)
        return image_ts

    def label_trans(self, target):
        target = target.astype(np.int8)
        target = torch.from_numpy(target)
        return target

    def nodata_check(self, img, labels):
        return np.any(labels.numpy() == 0) or np.any(np.sum(img.numpy() == 0, axis=0) == 4)


class Mars(IterableDataset, BaseDataset):
    def __init__(self, image_list, label_list, chip_size=14, num_chips_per_tile=50, stride=None):
        super().__init__()
        self.fns = list(zip(image_list, label_list))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.stride = stride

    def __len__(self):
        if self.stride is None:
            return len(self.fns) * self.num_chips_per_tile
        else:
            image = rasterio.open(self.fns[0][0], "r").read()
            height, width = image.shape[-2:]
            x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
            y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]
            return len(self.fns) * len(x_list) * len(y_list)

    def __iter__(self):
        # init num_workers
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

        for image_path, label_lr_path in self.fns[lower_idx: upper_idx]:
            # file object
            image_fp = rasterio.open(image_path, "r")
            label_lr_fp = rasterio.open(label_lr_path, "r")

            # data
            image = image_fp.read()
            label = label_lr_fp.read()

            # to Tensor
            image = self.image_trans1(image)
            label = self.label_trans(label)

            height, width = image.shape[-2:]

            if not self.stride:
                x_list = np.random.randint(0, width - self.chip_size, size=(self.num_chips_per_tile))
                y_list = np.random.randint(0, height - self.chip_size, size=(self.num_chips_per_tile))

                for i in range(self.num_chips_per_tile):
                    x = x_list[i]
                    y = y_list[i]

                    image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                    label_lr_patch = label[..., y:y + self.chip_size, x:x + self.chip_size]

                    label_lr_patch = label_lr_patch.squeeze()

                    yield image_patch, label_lr_patch
            else:
                x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                for x in x_list:
                    for y in y_list:
                        image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                        label_lr_patch = label[..., y:y + self.chip_size, x:x + self.chip_size]

                        label_lr_patch = label_lr_patch.squeeze()

                        yield image_patch, label_lr_patch, np.array((y, x))

        pass


if __name__ == '__main__':
    image_list = glob.glob(r'F:\datasets\competition\Mars\train\C*_train_img_mask.tif')
    label_list = [fn.replace('img', 'gt') for fn in image_list]
    ds = Mars(image_list=image_list,
              label_list=label_list,
              stride=7)
    dl = DataLoader(ds, batch_size=2, num_workers=1, pin_memory=True)
    for i, (img, label,coord) in enumerate(tqdm(dl)):
        print(img.shape)
        print(label.shape)
        print(coord)
        pass
