import numpy as np
import torch

from .DCSwin.DCSwin import dcswin_tiny, dcswin_base
from .DofT_timm.DofT import DofTV1
from .HRNet import HighResulutionNet
from .MyModel.L2HNet import MyL2HNet
from .MyModel.L2HNetV2 import L2HNetV2
# from .MyModel.LSeg import MyLSeg
from .MyModel.MobileUNet import MobileUNet
from .MyModel.SimpleCNN import SimpleCNN
from .SpiderNet.SpiderNet import SpiderNet, SpiderNet_out3
from .SpiderNet_test.SpiderNet import SpiderNet_out3 as SpiderNet_out3_test
from .lzh_model.CoAtNet import CoAtNet_Seg
from .lzh_model.DCSwin import DCSwin
from .lzh_model.EfficientViT import EfficientViT, EfficientViT_Seg
from .lzh_model.UNetFormer import UNetFormer
from .lzh_model.convit import Convit_seg
from .lzh_model.models import Skip_FCN, L2HNet
from .old_SpiderNet.MyModel.GroupL2HNet_10cls import MyModel as MyModel2
from .Doft.swin_transformer_v2 import SwinTransformerV3, SwinTransformerV4, SwinTransformerV5


def build_models(model_name, in_channels, num_classes, img_size=None):
    if model_name == 'Doftv4':
        return DofTV1(in_chans=in_channels, num_classes=num_classes, img_size=img_size)

    if model_name == 'Doftv3':
        return SwinTransformerV5(in_chans=in_channels, num_classes=num_classes, img_size=img_size)

    if model_name == 'Doftv2':
        return SwinTransformerV4(in_chans=in_channels, num_classes=num_classes)

    if model_name == 'Doft':
        return SwinTransformerV3(in_chans=in_channels, num_classes=num_classes, img_size=img_size)

    if model_name == 'L2HNetV2':
        return L2HNetV2(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'SimpleCNN':
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes, hidden_dim=64)

    if model_name == 'SpiderNet':
        return SpiderNet(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'SpiderNet_out3':
        return SpiderNet_out3(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'SpiderNet_out3_test':
        return SpiderNet_out3_test(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'old_SpiderNet':
        return MyModel2(in_chan=in_channels, num_class=num_classes)

    if model_name == 'Paraformer':
        return build_paraformer(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'L2HNet':
        return MyL2HNet(in_chan=in_channels, num_classes=num_classes)

    if model_name == 'MobileUNet':
        return MobileUNet(in_chans=in_channels, num_classes=num_classes)

    if model_name == 'TransUNet':
        return build_transunet(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'LSeg':
        return build_LSeg()

    if model_name == 'HRNet':
        config = r'F:\Projects2\SpiderNet\models\HRNet\config.yml'
        model = HighResulutionNet(in_channel=in_channels, num_classes=num_classes, cfg=config)
        return model

    if model_name == 'DCSwin':
        return dcswin_base(pretrained=False, in_channels=in_channels, num_classes=num_classes)

    # =======================
    if model_name == 'SkipFCN':
        return Skip_FCN(num_input_channels=in_channels, num_output_classes=num_classes, num_filters=64)

    if model_name == 'L2HNet_lzh':
        return L2HNet(insize=in_channels, input_chs=128, num_output_classes=num_classes)

    if model_name == 'CoAtNet':
        return CoAtNet_Seg(img_size=(224, 224), in_channel=in_channels, num_classes=num_classes)

    if model_name == 'ConViT':
        return Convit_seg(in_chans=in_channels, num_classes=num_classes)

    if model_name == 'EfficientViT':
        return EfficientViT_Seg(in_channel=in_channels, num_classes=num_classes)

    if model_name == 'DCSwin_lzh':
        return DCSwin(in_channels=in_channels, num_classes=num_classes)

    if model_name == 'UNetFormer':
        return UNetFormer(in_channels=in_channels,num_classes=num_classes)
    pass


# def build_SpiderNet(in_chan=6, num_class=17):
#     model = SpiderNet(in_chan=in_chan, num_class=num_class)
#     return model


def build_transunet(vit_patches_size=16, img_size=224, num_classes=17,in_channels=3):
    from .TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    from .TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg

    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 17
    config_vit.n_skip = 3

    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    # pretrain_path = r'F:\Projects2\SpiderNet\models\paraformer_raw\pre-train_model\imagenet21k\ViT-B_16.npz'
    net = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes, in_channels=in_channels)
    net.load_from(weights=np.load(config_vit.pretrained_path))

    return net

def build_paraformer(vit_patches_size=16, img_size=224, width=64, num_classes=17,in_channels=3):
    from .paraformer_raw.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    from .paraformer_raw.vit_seg_modeling import VisionTransformer as ViT_seg
    from .paraformer_raw.vit_seg_modeling_L2HNet import L2HNet

    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    # width也可以是128
    net = ViT_seg(config_vit,
                  backbone=L2HNet(width=width,image_band=in_channels),
                  img_size=img_size,
                  num_classes=num_classes)

    pretrain_path = r'F:\Projects2\SpiderNet\models\paraformer_raw\pre-train_model\imagenet21k\ViT-B_16.npz'
    net.load_from(weights=np.load(pretrain_path))

    return net


def build_LSeg():
    checkpoint = torch.load(r'F:\Projects\lang-seg-main\checkpoints\demo_e200.ckpt', weights_only=False)
    model = MyLSeg(crop_size=224)
    model.load_state_dict(state_dict=checkpoint['state_dict'], strict=True)
    return model
