"""
GAT 图注意力网络模型（修复 + 优化版）
=======================================
修复：num_layers=1 时 convs/bns 不对齐 Bug
优化：增加残差连接 + 暴露 num_classes 属性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple


class GATClassifier(nn.Module):
    """
    用于区块链洗钱检测的 GAT 节点分类器（修复版）。

    Args:
        in_channels:  输入特征维度
        hidden_dim:   每个注意力头的隐藏维度
        num_classes:  分类类别数（默认 2）
        num_layers:   图注意力层数（≥1）
        num_heads:    中间层多头注意力头数
        dropout:      Dropout 比率
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int   = 128,
        num_classes: int   = 2,
        num_layers:  int   = 4,
        num_heads:   int   = 8,
        dropout:     float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers 至少为 1"

        self.dropout     = dropout
        self.num_layers  = num_layers
        self.num_classes = num_classes
        self.out_dim     = hidden_dim

        self.convs     = nn.ModuleList()
        self.bns       = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(num_layers):
            is_first = (i == 0)
            is_last  = (i == num_layers - 1)

            in_dim = in_channels if is_first else hidden_dim * num_heads

            if is_last:
                self.convs.append(
                    GATConv(in_dim, hidden_dim, heads=1,
                            concat=False, dropout=dropout)
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                out_dim = hidden_dim
            else:
                self.convs.append(
                    GATConv(in_dim, hidden_dim, heads=num_heads,
                            concat=True, dropout=dropout)
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
                out_dim = hidden_dim * num_heads

            self.res_projs.append(
                nn.Linear(in_dim, out_dim, bias=False)
                if in_dim != out_dim else nn.Identity()
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for conv, bn, res_proj in zip(self.convs, self.bns, self.res_projs):
            residual = res_proj(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)
            h = h + residual
        return h

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x, edge_index)
        logits     = self.classifier(embeddings)
        return logits, embeddings

    @torch.no_grad()
    def predict_proba(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits, _ = self.forward(x, edge_index)
        proba = torch.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return proba
