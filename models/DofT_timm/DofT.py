import os

import timm
import torch
from torch import nn


class DofTV1(nn.Module):
    def __init__(self, img_size=56, in_chans=3, num_classes=17):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans

        name = 'vit_huge_patch14_224.mae'

        weights_path = fr'F:/cache_hf/{name}cls.pth'
        if os.path.exists(weights_path):
            ViT = timm.create_model(name, pretrained=False)
            checkpoint = torch.load(weights_path, weights_only=True)
            ViT.load_state_dict(checkpoint['state_dict'])
        else:
            ViT = timm.create_model(name, pretrained=True)
            torch.save({
                'state_dict': ViT.state_dict(),
            }, weights_path)
        ViT.patch_embed.proj = nn.Conv2d(in_chans, 1280, kernel_size=1, stride=1, padding=0)
        ViT.patch_embed.img_size = (img_size, img_size)
        self.vit = ViT
        self.fcn = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.vit.forward_features(x)
        x = x[:, 1:, :].view(-1, 16, 16, 1280).permute(0, 3, 1, 2)
        x = self.fcn(x)
        return x

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.vit.named_parameters():
                if 'patch_embed' not in name:
                    param.requires_grad = False


if __name__ == '__main__':
    model = DofTV1(in_chans=300, num_classes=17, img_size=16)
    print(model(torch.randn(1, 300, 16, 16)).shape)
