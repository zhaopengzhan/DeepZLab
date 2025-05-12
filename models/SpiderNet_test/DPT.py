import numpy as np
import timm

import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, in_chan=768, head_channels=512):
        super().__init__()

        self.conv_more = Conv2dReLU(
            in_chan,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv1 = Conv2dReLU(
            1024 + 512,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv2 = Conv2dReLU(
            1024 + 512,
            640,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv3 = Conv2dReLU(
            1280,
            1,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv4 = Conv2dReLU(
            1280,
            256,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        # X is the output of ViT branch
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv1(x)
        x = self.up4(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv2(x)
        x = self.up2(x)
        x = torch.cat([x, features[2]], dim=1)

        x1 = self.conv3(x)
        x1 = self.up2(x1)

        x2 = self.conv4(x)
        x2 = self.up2(x2)

        x1 = torch.cat([x1, features[3]], dim=1)
        return x1, x2


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, use_transpose=True):
        super().__init__()

        if use_transpose:
            upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        self.up = upsample
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        pass

    def forward(self, x, skip):
        x = self.up(x)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
        pass


class Decoder(nn.Module):
    def __init__(self, width=64, num_classes=17, in_chan=768):
        super().__init__()
        self.width = width
        self.out_fusion = DecoderCup(in_chan=in_chan)
        self.segmentation_head1 = SegmentationHead(  # Classifier for CNN branch
            in_channels=self.width + 1,
            out_channels=num_classes,
            kernel_size=3,
        )
        self.segmentation_head2 = SegmentationHead(  # Classifier for ViT branch
            in_channels=256,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, last_hidden_trans, downscale_feat, attn_dict_list):
        # attn_maps = get_attn_maps(attn_dict_list)
        x1, x2 = self.out_fusion(last_hidden_trans, downscale_feat)

        pred_cnn = self.segmentation_head1(x1)
        pred_fusion = self.segmentation_head2(x2)
        # pred_trans = rearrange(attn_maps[-1],'b h w n -> b n h w')
        pred_trans = None
        return pred_cnn, pred_fusion, pred_trans


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_transpose=True):
        super().__init__()
        self.layers = self.make_layers(encoder_channels, decoder_channels, use_transpose)

    def make_layers(self, encoder_channels, decoder_channels, use_transpose):
        num_layers = len(decoder_channels)
        layers = []

        for i in range(num_layers):
            in_ch = encoder_channels[-(i + 1)]
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            out_ch = decoder_channels[i]
            if i != 0:
                in_ch = decoder_channels[i - 1]

            layers.append(UpLayer(in_ch, out_ch, skip_ch, use_transpose))

        return nn.ModuleList(layers)

    def forward(self, features):

        x = features[-1]  # 解码器初始输入
        for i, layer in enumerate(self.layers):
            skip = features[-(i + 2)] if i < len(features) - 1 else None
            x = layer(x, skip)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.downscale = timm.create_model('resnet50', features_only=True, out_indices=(1, 2, 3), pretrained=False)
        self.downscale.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.decoder = UNetDecoder([256, 512, 1024 + embed_dim], [512, 256, 128])
        self.fcn = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, last_cnn_feat, last_vit_feat, return_last_hidden_states=False):
        down_feats = self.downscale(last_cnn_feat)
        group_token, vit_feat = torch.split(last_vit_feat, [last_vit_feat.size(1) - 784, 784], dim=1)
        vit_feat = vit_feat.permute(0, 2, 1).reshape(-1, self.embed_dim, 28, 28)
        down_feats[-1] = torch.cat((down_feats[-1], vit_feat.reshape(-1, self.embed_dim, 28, 28)), dim=1)
        x = self.decoder(down_feats)

        if return_last_hidden_states:
            return self.fcn(x), x
        else:
            return self.fcn(x)
