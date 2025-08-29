import os

import torch
from ptflops import get_model_complexity_info
from torch import nn
from transformers import MobileViTForSemanticSegmentation, AutoImageProcessor, MobileViTV2ForSemanticSegmentation, \
    UperNetForSemanticSegmentation
from models import DeepZMODELS
from models import root_dir

# @DeepZMODELS.register_module('UperNet')
class UperNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "openmmlab/upernet-convnext-tiny"):
        super(UperNet, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes


        local_dir = os.path.join(root_dir, model_id)
        _, self.model = self.load_model(model_id, local_dir)

        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.backbone.embeddings.patch_embeddings
        self.model.backbone.embeddings.patch_embeddings = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode
        )

        # 2. out
        old_cls = self.model.decode_head.classifier
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels=old_cls.in_channels,
            out_channels=self.num_classes,
            kernel_size=old_cls.kernel_size,
            stride=old_cls.stride,
            padding=old_cls.padding,
            dilation=old_cls.dilation,
            groups=old_cls.groups,
            bias=(old_cls.bias is not None),
            padding_mode=old_cls.padding_mode
        )

        # 3. auxiliary
        old_cls = self.model.auxiliary_head.classifier
        self.model.auxiliary_head.classifier = nn.Conv2d(
            in_channels=old_cls.in_channels,
            out_channels=self.num_classes,
            kernel_size=old_cls.kernel_size,
            stride=old_cls.stride,
            padding=old_cls.padding,
            dilation=old_cls.dilation,
            groups=old_cls.groups,
            bias=(old_cls.bias is not None),
            padding_mode=old_cls.padding_mode
        )

        pass

    def forward(self, img):
        outputs = self.model(pixel_values=img)

        return outputs.logits

    def load_model(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = AutoImageProcessor.from_pretrained(local_dir,use_fast=False)
            model = UperNetForSemanticSegmentation.from_pretrained(local_dir)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = AutoImageProcessor.from_pretrained(model_id,use_fast=False)
            model = UperNetForSemanticSegmentation.from_pretrained(model_id)
            os.makedirs(local_dir, exist_ok=True)
            processor.save_pretrained(local_dir)
            model.save_pretrained(local_dir)

        return processor, model

    @classmethod
    def get_model_ids(self) -> list[str]:
        '''
            from huggingface_hub import list_models
            models = list_models(search="nvidia/segformer")
            for m in models:
                print(m.modelId)
        '''
        return [
            "openmmlab/upernet-convnext-tiny"
        ]


def test_build_model():
    model = UperNet(model_id=UperNet.get_model_ids()[0],
                      in_channels=4,
                      num_classes=32)
    print(model)
    outputs = model(torch.randn(2, 4, 256, 256))
    print(outputs.shape)

    macs, params = get_model_complexity_info(model, (4, 256, 256),
                                             as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    pass


if __name__ == '__main__':
    test_build_model()

    pass
