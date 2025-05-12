import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock, self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU(),
                eca_layer(int(input_chs * ratio), k_size=5),
                # SE_Block(int(input_chs * ratio)),
                # CBAMLayer(int(input_chs * ratio)),
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU(),
            eca_layer(input_chs, k_size=5),
            # SE_Block(input_chs),
            # CBAMLayer(input_chs),
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class SE_Block(nn.Module):  # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


class L2HNet(nn.Module):
    def __init__(self,
                 width,  # width=64 for light mode; width=128 for normal mode
                 image_band=4,
                 # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 output_chs=128,
                 length=5,
                 ratios=[1, 0.5, 0.25],
                 bn_momentum=0.1):
        super(L2HNet, self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)

        self.out_conv1 = nn.Sequential(
            StdConv2d(self.width * length, output_chs * length, kernel_size=3, stride=2, bias=False, padding=1),

            nn.GroupNorm(32, output_chs * 5, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv2 = nn.Sequential(
            StdConv2d(output_chs * length, 1024, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv3 = nn.Sequential(
            StdConv2d(1024, 1024, kernel_size=5, stride=4, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.startconv(x)
        output_d1 = []
        for rpblk in self.rpblocks:
            x = rpblk(x)
            output_d1.append(x)
        output_d1 = self.out_conv1(torch.cat(output_d1, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        features = [output_d1, output_d2, output_d3, x]
        return output_d3, features[::-1], features[::-1]


class L2HNet_Light(nn.Module):
    def __init__(self,
                 width=64,
                 image_band=4,
                 num_class=17,
                 length=5,
                 ratios=[1, 0.5, 0.25],
                 bn_momentum=0.1):
        super(L2HNet_Light, self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)

        self.out_conv1 = nn.Sequential(
            StdConv2d(64, num_class * 4, kernel_size=3, stride=1, bias=False, padding=1),
            # nn.GroupNorm(32, 1024, eps=1e-6),
            nn.BatchNorm2d(num_class * 4),
            nn.ReLU()
        )

        self.out_conv2 = nn.Sequential(
            StdConv2d(num_class * 4, num_class, kernel_size=3, stride=1, bias=False, padding=1),
            # nn.GroupNorm(32, 1024, eps=1e-6),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.startconv(x)

        for rpblk in self.rpblocks:
            x = rpblk(x)

        x = self.out_conv1(x)
        x = self.out_conv2(x)

        return x, x
