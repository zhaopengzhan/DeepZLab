from torch import nn
import torch
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

class HierarchyInvertClassifier(nn.Module):
    def __init__(self,
                 mapping: Dict[str, List[int]],
        reduce: str = "conv",
        bias: bool = True,):
        super().__init__()
        self.classifier21 = nn.Conv2d(10, 1,
                                      kernel_size=1)  # [51, 52, 61, 62, 71, 72, 81, 82, 91, 92] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.classifier22 = nn.Conv2d(5, 1, kernel_size=1)  # [121, 122, 130, 150, 140] [14, 15, 16, 17, 18]
        self.classifier23 = nn.Conv2d(3, 1, kernel_size=1)  # [11, 12, 20] [1, 2, 3]
        self.classifier24 = nn.Conv2d(1, 1, kernel_size=1)  # [210]	[29]

    def forward(self, pred_lr):
        x_nodata = pred_lr[:, [0], ...]

        # forest
        x_forest = self.classifier21(pred_lr[:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ...])

        # shrub
        x_shrub = self.classifier22(pred_lr[:, [14, 15, 16, 17, 18], ...])

        # crop
        x_crop = self.classifier23(pred_lr[:, [1, 2, 3], ...])

        # water
        x_water = self.classifier24(pred_lr[:, [29], ...])

        pred_hr = torch.cat([x_nodata, x_forest, x_shrub, x_crop, x_water], dim=1)
        return pred_hr

