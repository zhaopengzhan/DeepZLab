from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .DPT import UNet
from .L2HNetV2 import L2HNet
from .group_vit import PatchEmbed
from .vision_transformer import VisionTransformer


def get_attn_maps(attn_dicts, size=(224, 224), return_onehot=False, rescale=False):
    """
    Args:
        img: [B, C, H, W]

    Returns:
        attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
    """
    # results = self.model.img_encoder(img, return_attn=True, as_dict=True)

    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for idx, attn_dict in enumerate(attn_dicts):
            if attn_dict is None:
                assert idx == len(attn_dicts) - 1, 'only last layer can be None'
                continue
            # [B, G, HxW]
            # B: batch size (1), nH: number of heads, G: number of group token
            attn_masks = attn_dict['soft']
            # [B, nH, G, HxW] -> [B, nH, HxW, G]
            attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
            # if prev_attn_masks is None: # 以前是64 - 8依次变换，现在是17-17 所以不需要了
            if True:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            # [B, nH, HxW, G] -> [B, nH, H, W, G]
            attn_maps.append(resize_attn_map(prev_attn_masks, *size))

    for i in range(len(attn_maps)):
        attn_map = attn_maps[i]
        # [B, nh, H, W, G]
        assert attn_map.shape[1] == 1
        # [B, H, W, G]
        attn_map = attn_map.squeeze(1)

        if rescale:
            attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            attn_map = F.interpolate(
                attn_map, size=size, mode='bilinear', align_corners=False)
            attn_map = rearrange(attn_map, 'b g h w -> b h w g')

        if return_onehot:
            # [B, H, W, G]
            attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

        attn_maps[i] = attn_map

    return attn_maps


class SpiderNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_cls_token=(64, 32, 17), width=64, embed_dim=384):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = L2HNet(in_channels=in_channels, width=width)

        self.vit = VisionTransformer(patch_size=8, embed_dim=384, num_cls_token=num_cls_token, depth=12, num_heads=6,
                                     mlp_ratio=4, img_size=[224],
                                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.vit = VisionTransformer(patch_size=8, embed_dim=768, num_classes=64, depth=12, num_heads=12, mlp_ratio=4,
        #                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=[224])
        self.unet = UNet(in_channels=width, num_classes=num_classes, embed_dim=embed_dim)

        self.fcn = nn.Conv2d(width, num_classes, kernel_size=1)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(width, width // 4, kernel_size=1),
            PatchEmbed(img_size=224, kernel_size=8, stride=8, padding=0, in_chans=width // 4, embed_dim=embed_dim,
                       norm_layer=nn.LayerNorm, )
        )

        self._init_vit(num_cls_token=num_cls_token[0])

    def _init_vit(self, num_cls_token):
        # weight_path = r'F:\Projects3\models_factory\weights\dino_vitbase8_pretrain.pth'
        weight_path = r'F:\Projects3\models_factory\weights\dino_deitsmall8_pretrain.pth'
        checkpoint = torch.load(weight_path, weights_only=True)

        checkpoint['cls_token'] = checkpoint['cls_token'].repeat(1, num_cls_token, 1)
        del checkpoint['pos_embed']
        self.vit.load_state_dict(checkpoint, strict=False)

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.vit.named_parameters():
                if 'cls_token' not in name or 'pos_embed' not in name or 'group' not in name or 'patch_embed' not in name:
                    param.requires_grad = False

    def forward(self, x, return_grouping_baranch=False):
        last_cnn_feat, cnn_feats = self.cnn(x, return_features=True)
        cnn_tokens = [
            self.patch_embed(cnn_feats[1])[0],
            self.patch_embed(cnn_feats[3])[0],
            self.patch_embed(cnn_feats[5])[0],
        ]
        last_vit_feat, attn_dict_list, attn_list, vit_token_list, cross_attn_list = self.vit(x[:, :3], cnn_tokens)

        if return_grouping_baranch:
            pred_fusion, last_unet_feat = self.unet(last_cnn_feat, last_vit_feat, return_last_hidden_states=True)
        else:
            pred_fusion = self.unet(last_cnn_feat, last_vit_feat)

        pred_cnn = self.fcn(last_cnn_feat)

        # group_token, vit_feat = torch.split(last_vit_feat, [self.num_classes, 784], dim=1)
        # pre_trans = torch.einsum("bnd,bmd->bnm", group_token, vit_feat)
        # pre_trans = pre_trans.reshape(-1, self.num_classes, 28, 28)
        # pre_trans = F.interpolate(pre_trans, size=(224, 224), mode='bilinear', align_corners=True)

        return pred_cnn, pred_fusion
        # return pred_cnn, pred_fusion, attn_dict_list, attn_list, cnn_tokens, cnn_feats, vit_token_list, cross_attn_list
        # return (pred_cnn + pred_fusion)

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SpiderNet_out3(nn.Module):
    def __init__(self, in_channels, num_classes, num_cls_token=(64, 32, 17), width=64, embed_dim=384):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = L2HNet(in_channels=in_channels, width=width)

        self.vit = VisionTransformer(patch_size=8, embed_dim=384, num_cls_token=num_cls_token, depth=12, num_heads=6,
                                     mlp_ratio=4, img_size=[224],
                                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.vit = VisionTransformer(patch_size=8, embed_dim=768, num_classes=64, depth=12, num_heads=12, mlp_ratio=4,
        #                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=[224])
        self.unet = UNet(in_channels=width, num_classes=num_classes, embed_dim=embed_dim)

        self.fcn = nn.Conv2d(width, num_classes, kernel_size=1)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(width, width // 4, kernel_size=1),
            PatchEmbed(img_size=224, kernel_size=8, stride=8, padding=0, in_chans=width // 4, embed_dim=embed_dim,
                       norm_layer=nn.LayerNorm, )
        )

        self.mlp = MLP(input_dim=embed_dim, hidden_dim=embed_dim//2, output_dim=128, num_layers=2)

        self._init_vit(num_cls_token=num_cls_token[0])

    def _init_vit(self, num_cls_token):
        # weight_path = r'F:\Projects3\models_factory\weights\dino_vitbase8_pretrain.pth'
        weight_path = r'F:\Projects3\models_factory\weights\dino_deitsmall8_pretrain.pth'
        checkpoint = torch.load(weight_path, weights_only=True)

        checkpoint['cls_token'] = checkpoint['cls_token'].repeat(1, num_cls_token, 1)
        del checkpoint['pos_embed']
        self.vit.load_state_dict(checkpoint, strict=False)

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.vit.named_parameters():
                if 'cls_token' not in name or 'pos_embed' not in name or 'group' not in name or 'patch_embed' not in name:
                    param.requires_grad = False

    def forward(self, x, return_grouping_feat=False):
        last_cnn_feat, cnn_feats = self.cnn(x, return_features=True)
        cnn_tokens = [
            self.patch_embed(cnn_feats[1])[0],
            self.patch_embed(cnn_feats[3])[0],
            self.patch_embed(cnn_feats[5])[0],
        ]
        last_vit_feat, attn_dict_list, attn_list, vit_token_list, cross_attn_list = self.vit(x[:, :3], cnn_tokens)

        pred_fusion, last_unet_feat = self.unet(last_cnn_feat, last_vit_feat, return_last_hidden_states=True)
        group_token, vit_feat = torch.split(last_vit_feat, [self.num_classes, 784], dim=1)
        cls_map = self.mlp(group_token)
        pre_trans = torch.einsum("bnc,bchw->bnhw", cls_map, last_unet_feat)

        pred_cnn = self.fcn(last_cnn_feat)

        if return_grouping_feat:
            return pred_cnn, pred_fusion,pre_trans, attn_dict_list, attn_list, cnn_tokens, cnn_feats, vit_token_list, cross_attn_list
        return pred_cnn, pred_fusion,pre_trans
