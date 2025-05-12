import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GroupVIT.group_vit import PatchEmbed


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
                # eca_layer(int(input_chs * ratio))
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU(),
            # eca_layer(input_chs)
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class RPBlockPro(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        # branch1 1×1
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_chs, input_chs, kernel_size=(1), stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU(),

        )
        # branch2 1×1 3×3
        self.branch2 = nn.Sequential(
            # nn.Conv2d(input_chs, input_chs // 2, kernel_size=1),
            # nn.BatchNorm2d(input_chs // 2, momentum=bn_momentum),
            # nn.ReLU(),
            nn.Conv2d(input_chs, input_chs // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_chs // 2, momentum=bn_momentum),
            nn.ReLU()

        )
        # branch3 1×1 3×3 3×3
        self.branch3 = nn.Sequential(
            # nn.Conv2d(input_chs, input_chs // 4, kernel_size=1),
            # nn.BatchNorm2d(input_chs // 4, momentum=bn_momentum),
            # nn.ReLU(),
            nn.Conv2d(input_chs, input_chs // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(input_chs // 4, momentum=bn_momentum),
            nn.ReLU(),
            # nn.Conv2d(input_chs // 4, input_chs // 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(input_chs // 4, momentum=bn_momentum),
            # nn.ReLU(),

        )

        # branch4 pool3×3 conv1×1
        # self.branch4 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(input_chs, input_chs // 4, kernel_size=1),
        #     nn.BatchNorm2d(input_chs // 4),
        #     nn.ReLU()
        # )

        self.fuse_conv = nn.Sequential(
            # eca_layer(int(input_chs * sum(ratios))),
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU(),
            # eca_layer(input_chs),

        )

        # for name, param in self.named_parameters():
        #     param.requires_grad = False

        # self.eca = eca_layer(int(input_chs * sum(ratios)))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # branch4 = self.branch4(x)

        branches = torch.cat([branch1, branch2, branch3], dim=1)
        output = self.fuse_conv(branches) + x
        return output, branches


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
                 width=128,
                 image_band=4,
                 output_chs=128,
                 length=5,
                 ratios=[1, 0.5, 0.25, 0.25],
                 bn_momentum=0.1,
                 pro=True, embed_dim=384):
        super(L2HNet, self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)

        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            if pro:
                rpblock = RPBlockPro(self.width, )
            else:
                rpblock = RPBlock(self.width, )
            self.rpblocks.append(rpblock)

        self.pc1 = PatchEmbed(img_size=224,
                              kernel_size=16,
                              stride=16,
                              padding=0,
                              in_chans=int(width * 1.75),
                              embed_dim=embed_dim,
                              norm_layer=nn.LayerNorm, )

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
        # self.startconv_out = x

        x_token_list = []
        x_list = []
        for idx, rpblk in enumerate(self.rpblocks):
            x, br = rpblk(x)
            x_list.append(x)

            x_token = self.pc1(br)

            x_token_list.append(x_token)

        output_d1 = self.out_conv1(torch.cat(x_list, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        downscale_feat = [output_d3, output_d2, output_d1, x]

        return x, x_token_list, downscale_feat
