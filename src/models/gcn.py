"""
GCN 图卷积网络模型（冲刺版）
==============================
含：残差连接 + Jumping Knowledge + DropEdge + 3层MLP分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from typing import Tuple


class GCNClassifier(nn.Module):
    """
    带 Jumping Knowledge + DropEdge 的 GCN 节点分类器。

    Args:
        in_channels:    输入特征维度
        hidden_dim:     隐藏层特征维度
        num_classes:    分类类别数（默认 2）
        num_layers:     图卷积层数（默认 4）
        dropout:        Dropout 比率（默认 0.2）
        use_jumping:    是否启用 Jumping Knowledge（默认 True）
        use_dropedge:   是否启用 DropEdge（默认 True）
        dropedge_rate:  DropEdge 丢弃比率（默认 0.1）
    """

    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int   = 256,
        num_classes:   int   = 2,
        num_layers:    int   = 4,
        dropout:       float = 0.2,
        use_jumping:   bool  = True,
        use_dropedge:  bool  = True,
        dropedge_rate: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers 至少为 1"

        self.num_layers    = num_layers
        self.dropout       = dropout
        self.num_classes   = num_classes
        self.hidden_dim    = hidden_dim
        self.use_jumping   = use_jumping
        self.use_dropedge  = use_dropedge
        self.dropedge_rate = dropedge_rate

        self.convs     = nn.ModuleList()
        self.bns       = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        # 第一层：in_channels → hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim, cached=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.res_projs.append(
            nn.Linear(in_channels, hidden_dim, bias=False)
            if in_channels != hidden_dim else nn.Identity()
        )

        # 后续层：hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.res_projs.append(nn.Identity())

        # 分类头输入维度
        clf_in = hidden_dim * num_layers if use_jumping else hidden_dim

        # 3层 MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # ──────────────────────────────────────────────────────────────
    def encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """生成节点嵌入（含残差 + JK + DropEdge）。"""
        if self.use_dropedge and self.training:
            edge_index, _ = dropout_edge(
                edge_index,
                p=self.dropedge_rate,
                training=self.training,
            )

        h             = x
        layer_outputs = []

        for conv, bn, res_proj in zip(self.convs, self.bns, self.res_projs):
            residual = res_proj(h)
            h        = conv(h, edge_index)
            h        = bn(h)
            h        = F.relu(h)
            h        = F.dropout(h, p=self.dropout, training=self.training)
            h        = h + residual
            layer_outputs.append(h)

        if self.use_jumping and len(layer_outputs) > 1:
            h = torch.cat(layer_outputs, dim=-1)
        else:
            h = layer_outputs[-1]

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
        proba     = torch.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return proba
