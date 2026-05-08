"""
GAT 图注意力网络模型
=====================
实现多层多头 GAT 用于节点二分类（正常 / 洗钱）。

架构：
  输入层 → [GATConv(多头) + ELU + Dropout] × n_layers → 线性分类头
  最后一层注意力头做平均聚合（concat=False）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple


class GATClassifier(nn.Module):
    """
    用于区块链交易洗钱检测的 GAT 节点分类器。

    Args:
        in_channels:  输入特征维度
        hidden_dim:   每个注意力头的隐藏维度
        num_classes:  分类类别数（2）
        num_layers:   图注意力层数
        num_heads:    多头注意力的头数
        dropout:      Dropout 比率
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int   = 64,
        num_classes: int   = 2,
        num_layers:  int   = 2,
        num_heads:   int   = 8,
        dropout:     float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout    = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # ── 第一层：多头，输出 concat → hidden_dim * num_heads ────
        self.convs.append(
            GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # ── 中间层（如有）─────────────────────────────────────────
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim,
                        heads=num_heads, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # ── 最后一层：平均聚合，输出 hidden_dim ───────────────────
        last_in = hidden_dim * num_heads if num_layers > 1 else in_channels
        if num_layers > 1:
            self.convs.append(
                GATConv(last_in, hidden_dim, heads=1,
                        concat=False, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.out_dim    = hidden_dim
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # ──────────────────────────────────────────────────────────────
    def encode(
        self,x: torch.Tensor,edge_index: torch.Tensor,) -> torch.Tensor:
        """
        通过 GAT 层生成节点嵌入向量。
        Returns:
            节点嵌入 [N, hidden_dim]
        """
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Returns:
            (logits [N, C], embeddings [N, D])
        """
        embeddings = self.encode(x, edge_index)
        logits     = self.classifier(embeddings)
        return logits, embeddings

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """返回 softmax 概率 [N, num_classes]。"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index)
        return torch.softmax(logits, dim=-1)