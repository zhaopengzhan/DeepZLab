# + 通用的“按配置聚合通道”的分层分类头  # +
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class HierarchyInvertClassifier(nn.Module):
    """
    按配置将输入通道分组聚合为高层类别输出。
    - mapping: 形如 {"nodata":[0], "forest":[4,5,...], "shrub":[14,...], ...}
      键名的插入顺序即输出通道顺序。
    - reduce:  聚合方式:
        * "conv": 每组一个 1x1 可学习卷积(默认)
        * "mean": 逐通道求均值
        * "sum" : 逐通道求和
        * "max" : 逐通道求最大值
    - bias:    reduce="conv" 时，是否使用偏置
    """
    def __init__(
        self,
        mapping: Dict[str, List[int]],
        reduce: str = "conv",
        bias: bool = True,
    ):
        super().__init__()
        assert len(mapping) > 0, "mapping 不能为空"
        assert reduce in {"conv", "mean", "sum", "max"}, f"不支持的 reduce={reduce}"

        # + 保持键的插入顺序，输出通道顺序可控  # +
        self.mapping = OrderedDict(mapping)
        self.reduce = reduce
        self.class_names = list(self.mapping.keys())  # 与输出通道一一对应

        if self.reduce == "conv":
            # + 按每个分组的 in_channels 构建 1x1 卷积头  # +
            heads = {}
            for name, idxs in self.mapping.items():
                in_ch = len(idxs)
                # 允许单通道分组，等价于一个可学习的缩放与偏置
                heads[name] = nn.Conv2d(in_ch, 1, kernel_size=1, bias=bias)
            self.heads = nn.ModuleDict(heads)
        else:
            # + 其他归约方式不需要可学习参数  # +
            self.heads = None

        # + 对 "mean" 语义友好地初始化 conv 头为均值（可选）  # +
        if self.reduce == "conv":
            self._maybe_init_as_mean()

    def _maybe_init_as_mean(self):
        """在 reduce='conv' 时，将每个 1x1 的权重初始化成“均值器”以便稳定起步。"""
        for name, idxs in self.mapping.items():
            conv = self.heads[name]
            nn.init.zeros_(conv.bias)
            # 均值权重：每个输入通道权重 = 1/n
            w = torch.full_like(conv.weight.data, 1.0 / max(1, len(idxs)))
            conv.weight.data.copy_(w)

    @torch.no_grad()
    def _validate_indices(self, C: int):
        """运行时校验索引合法性，避免静默越界。"""
        for name, idxs in self.mapping.items():
            if len(idxs) == 0:
                raise ValueError(f"分组 '{name}' 为空，请至少提供一个通道索引。")
            for i in idxs:
                if i < 0 or i >= C:
                    raise IndexError(
                        f"分组 '{name}' 含非法通道索引 {i}，应在 [0, {C-1}] 范围内。"
                    )

    def forward(self, pred_lr: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        参数:
            pred_lr: [B, C, H, W] 低层类别/子类的通道堆叠预测
        返回:
            pred_hr: [B, K, H, W] 按映射聚合后的高层类别预测
            class_names: List[str] 与通道对齐的类别名
        """
        assert pred_lr.dim() == 4, f"期望 [B,C,H,W]，但得到 {tuple(pred_lr.shape)}"
        B, C, H, W = pred_lr.shape
        self._validate_indices(C)

        outs = []
        if self.reduce == "conv":
            # + 使用每组一个可学习的 1x1 卷积进行聚合  # +
            for name, idxs in self.mapping.items():
                x = pred_lr[:, idxs, :, :]
                y = self.heads[name](x)  # [B,1,H,W]
                outs.append(y)
        else:
            # + 使用简单归约  # +
            for _, idxs in self.mapping.items():
                x = pred_lr[:, idxs, :, :]  # [B, n, H, W]
                if self.reduce == "mean":
                    y = x.mean(dim=1, keepdim=True)
                elif self.reduce == "sum":
                    y = x.sum(dim=1, keepdim=True)
                elif self.reduce == "max":
                    y, _ = x.max(dim=1, keepdim=True)
                else:
                    raise RuntimeError("不可能到达：未知 reduce")
                outs.append(y)

        pred_hr = torch.cat(outs, dim=1)  # [B, K, H, W]
        return pred_hr, self.class_names


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 你的原始映射（键顺序 = 输出通道顺序）
    mapping = OrderedDict({
        "nodata": [0],
        "forest": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "shrub":  [14, 15, 16, 17, 18],
        "crop":   [1, 2, 3],
        "water":  [29],
    })

    # 构建聚合头：可选 reduce="conv" | "mean" | "sum" | "max"
    head = HierarchyInvertClassifier(mapping=mapping, reduce="conv")

    # 伪造输入：C 至少覆盖到最大索引 29
    x = torch.randn(2, 30, 128, 128)  # [B=2,C=30,H=128,W=128]

    y, names = head(x)
    print(y.shape, names)  # torch.Size([2, 5, 128, 128]) ['nodata', 'forest', 'shrub', 'crop', 'water']
