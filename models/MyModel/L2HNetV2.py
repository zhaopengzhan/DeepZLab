import torch
import torch.nn.functional as F
from torch import nn


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class RPBlock(nn.Module):
    def __init__(self, in_channels, ratios=(1, 0.25, 0.125), bn_momentum=0.1):
        super(RPBlock, self).__init__()

        in_channelA = in_channels * ratios[0]
        self.branchA = nn.Sequential(
            nn.Conv2d(in_channels, in_channelA, 1),
            nn.BatchNorm2d(in_channelA, momentum=bn_momentum),
            nn.GELU()
        )

        in_channelB = int(in_channels * ratios[1])
        self.branchB = nn.Sequential(
            nn.Conv2d(in_channels, in_channelB, 1),
            nn.Conv2d(in_channelB, in_channelB, (1, 3), padding=(0, 3 // 2)),
            nn.Conv2d(in_channelB, in_channelB, (3, 1), padding=(3 // 2, 0)),
            nn.BatchNorm2d(in_channelB, momentum=bn_momentum),
            nn.GELU()
        )

        in_channelC = int(in_channels * ratios[2])
        self.branchC = nn.Sequential(
            nn.Conv2d(in_channels, in_channelC, 1),
            nn.Conv2d(in_channelC, in_channelC, (1, 3), padding=(0, 3 // 2)),
            nn.Conv2d(in_channelC, in_channelC, (3, 1), padding=(3 // 2, 0)),
            nn.Conv2d(in_channelC, in_channelC, (1, 3), padding=(0, 3 // 2)),
            nn.Conv2d(in_channelC, in_channelC, (3, 1), padding=(3 // 2, 0)),
            nn.BatchNorm2d(in_channelC, momentum=bn_momentum),
            nn.GELU()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(int(in_channels * sum(ratios)), in_channels, 1),
            nn.BatchNorm2d(in_channels, momentum=bn_momentum),
            nn.GELU()
        )

    def forward(self, x):
        x_a = self.branchA(x)
        x_b = self.branchB(x)
        x_c = self.branchC(x)
        branches = torch.cat([x_a, x_b, x_c], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class L2HNet(nn.Module):
    def __init__(self,
                 width,  # width=64 for light mode; width=128 for normal mode
                 image_band=4,
                 # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 # output_chs=128,
                 length=5,
                 # ratios=[1, 0.25, 0.125],
                 # bn_momentum=0.1
                 ):
        super(L2HNet, self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width)
            self.rpblocks.append(rpblock)

    def forward(self, x):
        x = self.startconv(x)

        for rpblk in self.rpblocks:
            x = rpblk(x)

        return x


class L2HNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, width=64):
        super().__init__()
        self.cnn = L2HNet(width=width, image_band=in_channels)

        self.fcn = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        last_hidden_cnn = self.cnn(x)
        return self.fcn(last_hidden_cnn)


if __name__ == '__main__':
    module = RPBlock(in_channels=64, bn_momentum=0.1)
    print(module(torch.randn(1, 64, 224, 224)).shape)
