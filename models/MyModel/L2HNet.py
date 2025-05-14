from torch import nn

from models.Paraformer.vit_seg_modeling_L2HNet import L2HNet


class MyL2HNet(nn.Module):
    def __init__(self, in_chan, num_classes, width=64, embed_dim=768):
        super().__init__()
        self.cnn = L2HNet(width=width, image_band=in_chan, pro=True, embed_dim=embed_dim)

        self.fcn = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        last_hidden_cnn, cnn_feats, downscale_feat = self.cnn(x)
        return self.fcn(last_hidden_cnn)
