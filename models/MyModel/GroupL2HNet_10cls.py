from functools import partial

import torch
from einops import rearrange
from torch import nn

# from GroupVIT.group_vit import get_attn_maps
from models.BEIT.modeling_finetune import beit_base_patch16_224, VisionTransformer
from models.Paraformer.vit_seg_modeling import DecoderCup, SegmentationHead
from models.Paraformer.vit_seg_modeling_L2HNet import L2HNet, StdConv2d


class MyModel(nn.Module):
    def __init__(self, in_chan, num_class, width=64, embed_dim=768):
        super().__init__()

        self.cnn = L2HNet(width=width, image_band=in_chan, pro=True, embed_dim=embed_dim)

        self.trans = VisionTransformer(patch_size=16, num_classes=num_class, embed_dim=embed_dim, depth=12,
                                       num_heads=12, mlp_ratio=4, qkv_bias=True, in_chans=3,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), )

        self.out = Decoder(width=width, in_chan=embed_dim, num_classes=num_class)

        self._init_weight(num_classes=num_class)

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.trans.named_parameters():
                if 'cls_token' not in name or 'pos_embed' not in name or 'group' not in name or 'patch_embed' not in name:
                    param.requires_grad = False

    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False):
        # x [bs, 4+6, 224, 224]
        last_hidden_cnn, cnn_feats, downscale_feat = self.cnn(x)
        last_hidden_trans, trans_feats, attn_dict_list = self.trans(x[:, :3], cnn_feats)
        # for f in trans_feats:
        #     print(f.shape)
        pred_cnn, pred_fusion, pred_trans = self.out(last_hidden_trans, downscale_feat, attn_dict_list)

        return pred_cnn, pred_fusion, pred_trans

    def _init_weight(self, num_classes=8):
        ckt_beit = torch.load(r'F:\Projects\MyGroupVIT6\weights\beit_base_patch16_224_pt22k_ft22k.pth')['model']
        # ckt_beit['head.weight'] = ckt_beit['head.weight'][:1000]
        # ckt_beit['head.bias'] = ckt_beit['head.bias'][:1000]
        ckt_beit['cls_token'] = torch.repeat_interleave(ckt_beit['cls_token'], num_classes, dim=1)
        self.trans.load_state_dict(ckt_beit, strict=False)


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


if __name__ == '__main__':
    model = MyModel()
    pred_cnn, pred_fusion, pred_trans = model(torch.rand((7, 4, 224, 224)))
    print(pred_cnn.shape, pred_fusion.shape)
