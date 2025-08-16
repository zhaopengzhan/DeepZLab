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
        IMAGE_MEANS = np.array([143.26, 142.72, 140.33])
        IMAGE_STDS = np.array([58.71, 55.44, 56.06])

        img = (img - IMAGE_MEANS[:, None, None]) / IMAGE_STDS[:, None, None]

        img = torch.from_numpy(img)
        img = img.to(torch.float32)
        return img

    def label_trans(self, target):
        target = torch.from_numpy(target)
        target = target.long()
        return target


class SHLU(IterableDataset, BaseDataset):
    def __init__(self, image_list,landuse_list, chip_size=224, num_chips_per_tile=50, stride=None):
        super().__init__()
        self.fns = list(zip(image_list, landuse_list))

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

        for image_path, landuse_path in self.fns[lower_idx: upper_idx]:
            # data
            image = rasterio.open(image_path, "r").read()
            label_LU = rasterio.open(landuse_path, "r").read()

            # To Tensor
            image = self.image_trans1(image)
            label_LU = self.label_trans(label_LU)

            height, width = image.shape[-2:]

            if not self.stride:
                x_list = np.random.randint(0, width - self.chip_size, size=(self.num_chips_per_tile))
                y_list = np.random.randint(0, height - self.chip_size, size=(self.num_chips_per_tile))

                for i in range(self.num_chips_per_tile):
                    x = x_list[i]
                    y = y_list[i]

                    image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                    label_LU_patch = label_LU[..., y:y + self.chip_size, x:x + self.chip_size]

                    # if (label_LU_patch == 0).float().mean() > (1 - 1e-7):
                    #     continue

                    yield image_patch, label_LU_patch.squeeze()
            else:
                x_list = list(range(0, width - self.chip_size, self.stride)) + [width - self.chip_size]
                y_list = list(range(0, height - self.chip_size, self.stride)) + [height - self.chip_size]

                for x in x_list:
                    for y in y_list:
                        image_patch = image[..., y:y + self.chip_size, x:x + self.chip_size]
                        label_LU_patch = label_LU[..., y:y + self.chip_size, x:x + self.chip_size]

                        yield image_patch, label_LU_patch.squeeze(), np.array((y, x))

        pass


if __name__ == '__main__':
    image_list = glob.glob(r'F:\zpz\datasets\SHLU\HR_Optical_image\*.tif')

    landuse_list = [fn.replace('HR_Optical_image', 'Weak_Label') for fn in image_list]
    ds = SHLU(image_list, landuse_list)
    dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
    for img, LU in tqdm(dl):
        print(img.shape, LU.shape)
        print(np.unique(LU.numpy()))
        break
        pass
