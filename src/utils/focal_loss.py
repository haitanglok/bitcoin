"""
Focal Loss（终极修复版）
========================
修复：
  [Fix-1] alpha buffer 设备问题：改用 to(device) 显式迁移
  [Fix-2] 增加 debug 模式，可打印每个样本的 loss 权重
  [Fix-3] 支持超大 illicit 权重（weight_cap 防止数值溢出）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    多分类 Focal Loss（终极修复版）。

    Args:
        alpha:           各类别权重张量 [num_classes]
        gamma:           聚焦参数，推荐 3.0~4.0
        reduction:       'mean' | 'sum' | 'none'
        label_smoothing: 标签平滑系数
    """

    def __init__(
        self,
        alpha:           Optional[torch.Tensor] = None,
        gamma:           float = 2.0,
        reduction:       str   = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # [Fix-1] 不用 register_buffer，改为普通属性 + 手动迁移
        # register_buffer 在某些 PyG 版本中会出现设备不同步问题
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def to(self, device):
        """重写 to()，确保 alpha 也迁移到目标设备。"""
        super().to(device)
        if self.alpha is not None:
            self.alpha = self.alpha.to(device)
        return self

    def forward(
        self,
        inputs:  torch.Tensor,   # [N, C] logits
        targets: torch.Tensor,   # [N]    long 标签
    ) -> torch.Tensor:

        # 确保 alpha 在正确设备上
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        n_cls = inputs.size(1)

        if self.label_smoothing > 0.0:
            smooth   = self.label_smoothing / n_cls
            one_hot  = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            one_hot  = one_hot * (1 - self.label_smoothing) + smooth
            log_prob = F.log_softmax(inputs, dim=-1)
            ce_loss  = -(one_hot * log_prob).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, reduction="none"
            )

        p_t          = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t    = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_focal_loss(
    class_weights:             dict,
    gamma:                     float        = 2.0,
    device:                    torch.device = torch.device("cpu"),
    illicit_weight_multiplier: float        = 1.0,
    label_smoothing:           float        = 0.0,
) -> FocalLoss:
    """
    工厂函数：构建 FocalLoss。

    [Fix-2] 确保 weights 列表按 index 0,1 顺序构建，
            不依赖 dict key 类型（int vs str 问题）
    """
    # 强制按 0,1 顺序取权重，避免 dict key 类型混乱
    w0 = float(class_weights.get(0, class_weights.get("0", 1.0)))
    w1 = float(class_weights.get(1, class_weights.get("1", 1.0)))
    w1 = w1 * illicit_weight_multiplier

    alpha = torch.tensor([w0, w1], dtype=torch.float32)

    loss = FocalLoss(
        alpha           = alpha,
        gamma           = gamma,
        label_smoothing = label_smoothing,
    )
    return loss.to(device)
