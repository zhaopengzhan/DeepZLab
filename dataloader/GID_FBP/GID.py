import glob
import random

import numpy as np
import rasterio
import torch
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


class GID(IterableDataset, BaseDataset):
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
            # data
            image = rasterio.open(image_path, "r").read()
            # image = rasterio.open(image_path, "r").read([1,2,3])
            label = rasterio.open(label_path, "r").read()

            # To Tensor
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
                    label_patch = label[..., y:y + self.chip_size, x:x + self.chip_size]

                    yield image_patch, label_patch.squeeze()
            else:
                x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                for x in x_list:
                    for y in y_list:
                        image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                        label_patch = label[..., y:y + self.chip_size, x:x + self.chip_size]

                        yield image_patch, label_patch.squeeze(), np.array((y, x))

        pass


if __name__ == '__main__':
    image_list = glob.glob(r'F:\datasets2\Five-Billion-Pixels\Image_16bit_BGRNir\*.tiff')
    label_list = [fn.replace('.tiff', '_24label.png')
                  .replace('Image_16bit_BGRNir', 'Annotation__index')
                  for fn in image_list]
    ds = GID(image_list, label_list)
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    for img, label in tqdm(dl):
        # print(img.shape, label.shape)
        print(np.unique(label.numpy()))
        # break
        pass