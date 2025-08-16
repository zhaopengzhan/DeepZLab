

import numpy as np
import rasterio
from torch.utils.data import Dataset

from dataloader import DeepZData
from .BaseDataset import BaseDataset


# @DeepZData.register_module()
class ChesapeakeTestDataset(Dataset, BaseDataset):
    def __init__(self, image_path, image_size=224, stride=112):
        super().__init__()
        with rasterio.open(image_path) as image_ds:
            height, width = image_ds.height, image_ds.width
            image_np = image_ds.read()

        self.image_np = image_np
        self.image_size = image_size
        self.stride = stride

        chip_coordinates = []
        # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.image_size, self.stride)) + [height - self.image_size]:
            for x in list(range(0, width - self.image_size, self.stride)) + [width - self.image_size]:
                chip_coordinates.append((y, x))

        self.chip_coordinates = chip_coordinates

    def __len__(self):
        return len(self.chip_coordinates)

    def __getitem__(self, item):

        y, x = self.chip_coordinates[item]

        image_patch = self.image_np[:, y:y + self.image_size, x:x + self.image_size]

        image_ts = self.image_trans1(image_patch)

        return image_ts, np.array((y, x))
