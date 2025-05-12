from functools import partial

import torch
from einops import rearrange
from torch import nn

from GroupVIT.group_vit import get_attn_maps
from models.BEIT.modeling_finetune import beit_base_patch16_224, VisionTransformer
from models.Paraformer.vit_seg_modeling import DecoderCup, SegmentationHead
from models.Paraformer.vit_seg_modeling_L2HNet import L2HNet, StdConv2d


class MyModel(nn.Module):
    def __init__(self, in_chan=4, width=64):
        super().__init__()
        # self.cnn = L2HNet(width=64,
        #                   image_band=in_chan, )

        self.cnn = L2HNet(width=width,
                          image_band=in_chan,
                          pro=True)

        self.trans = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=in_chan)

        self.out = Decoder(width=width)

    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False):
        # x [bs, 4+6, 224, 224]
        last_hidden_cnn, cnn_feats, downscale_feat = self.cnn(x)
        last_hidden_trans, trans_feats, attn_dict_list = self.trans(x, cnn_feats)
        # for f in trans_feats:
        #     print(f.shape)
        pred_cnn, pred_fusion, pred_trans = self.out(last_hidden_trans, downscale_feat, attn_dict_list)

        return pred_cnn, pred_fusion, pred_trans

    def _init_weight(self):
        ckt_beit = torch.load(r'F:\Projects\unilm-master\beit_base_patch16_224_pt22k_ft22k.pth')['model']
        ckt_beit['head.weight'] = ckt_beit['head.weight'][:1000]
        ckt_beit['head.bias'] = ckt_beit['head.bias'][:1000]
        # ckt_beit['cls_token'] = torch.repeat_interleave(ckt_beit['cls_token'], 17, dim=1)
        self.trans.load_state_dict(ckt_beit, strict=True)


class Decoder(nn.Module):
    def __init__(self, width=64, num_classes=17):
        super().__init__()
        self.width = width
        self.out_fusion = DecoderCup()
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
