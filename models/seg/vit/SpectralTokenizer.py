import os

import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from transformers import SegformerForSemanticSegmentation, AutoProcessor

from models import DeepZMODELS
from utils.initializer import init_conv


@DeepZMODELS.register_module('SpectralTokenizer')
class SegFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                 image_size=None):
        super(SegFormer, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes
        # config = SegformerConfig(num_channels=in_chans, num_labels=num_classes)
        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        _, self.model = self.load_segformer(model_id, local_dir)

        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
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
        init_conv(self.model.segformer.encoder.patch_embeddings[0].proj,
                  kind='gabor_init')
        self.model.segformer.encoder.patch_embeddings[0].proj.requires_grad = False

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

        pass

    def forward(self, img):
        outputs = self.model(pixel_values=img)
        logits = self.up(outputs.logits)
        return logits

    def load_segformer(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = AutoProcessor.from_pretrained(local_dir)
            model = SegformerForSemanticSegmentation.from_pretrained(local_dir)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = AutoProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
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
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-ade-512-512",
            "nvidia/segformer-b5-finetuned-ade-640-640",
        ]


def build_model():
    model = SegFormer(model_id=SegFormer.get_model_ids()[0],
                      in_channels=4,
                      num_classes=32)

    print(model.model.config)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    conv = model.model.segformer.encoder.patch_embeddings[0].proj

    print(conv.weight.shape)  # [64, 3, 11, 11]

    # 2. Visualize 25 randomly selected kernels (first input channel only)
    num_show = 25
    idx = np.random.choice(conv.weight.shape[0], num_show, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, i in zip(axes.flat, idx):
        kernel = conv.weight[i, 0].detach().cpu().numpy()
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
        ax.set_title(f"#{i}", fontsize=8)

    fig.suptitle(f"Conv kernels initialized with", fontsize=14)

    plt.tight_layout()
    plt.show()

    pass


if __name__ == '__main__':
    build_model()
