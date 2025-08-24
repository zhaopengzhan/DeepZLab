import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DeepZMODELS


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=(1, 0.5, 0.25), bn_momentum=0.1):
        super(RPBlock, self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


@DeepZMODELS.register_module('L2HNet')
class L2HNet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 image_size=None,
                 width=64,
                 length=5,
                 ratios=(1, 0.5, 0.25),
                 bn_momentum=0.1):
        super(L2HNet, self).__init__()

        self.startconv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()

        for _ in range(length):
            rpblock = RPBlock(width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)

        self.classifier = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.startconv(x)

        for rpblk in self.rpblocks:
            x = rpblk(x)

        return self.classifier(x)

