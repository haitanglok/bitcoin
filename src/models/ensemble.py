"""
双流集成模型（SAGE + GAT）
===========================
核心思想：
  SAGE 擅长聚合邻居特征（max聚合保留异常信号）
  GAT  擅长识别重要邻居（注意力权重对异常节点敏感）
  两者输出概率加权融合，互补提升召回率

架构：
  输入特征
    ├── SAGE流：SAGEConv×5 + JK → 概率向量
    └── GAT流 ：GATConv×4      → 概率向量
         ↓
  可学习加权融合 → 最终 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import dropout_edge
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────
class SAGEStream(nn.Module):
    """SAGE 流（max聚合 + JK + 残差）。"""

    def __init__(
        self,
        in_channels:   int,
        hidden_dim:    int   = 512,
        num_layers:    int   = 5,
        dropout:       float = 0.05,
        dropedge_rate: float = 0.03,
    ):
        super().__init__()
        self.dropout       = dropout
        self.dropedge_rate = dropedge_rate
        self.num_layers    = num_layers

        self.input_bn = nn.BatchNorm1d(in_channels)
        self.convs     = nn.ModuleList()
        self.bns       = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_dim, aggr="max"))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.res_projs.append(
            nn.Linear(in_channels, hidden_dim, bias=False)
            if in_channels != hidden_dim else nn.Identity()
        )
        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="max"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.res_projs.append(nn.Identity())

        # JK 后输出维度
        self.out_dim = hidden_dim * num_layers

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_bn(x)

        if self.training:
            edge_index, _ = dropout_edge(
                edge_index, p=self.dropedge_rate, training=True
            )

        h             = x
        layer_outputs = []

        for conv, bn, res_proj in zip(
            self.convs, self.bns, self.res_projs
        ):
            residual = res_proj(h)
            h        = conv(h, edge_index)
            h        = bn(h)
            h        = F.leaky_relu(h, negative_slope=0.1)
            h        = F.dropout(
                h, p=self.dropout, training=self.training
            )
            h        = h + residual
            layer_outputs.append(h)

        # Jumping Knowledge
        return torch.cat(layer_outputs, dim=-1)


# ──────────────────────────────────────────────────────────────────
class GATStream(nn.Module):
    """GAT 流（多头注意力 + 残差）。"""

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int   = 256,
        num_layers:  int   = 4,
        num_heads:   int   = 8,
        dropout:     float = 0.05,
    ):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers

        self.input_bn  = nn.BatchNorm1d(in_channels)
        self.convs     = nn.ModuleList()
        self.bns       = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(num_layers):
            is_first = (i == 0)
            is_last  = (i == num_layers - 1)
            in_dim   = in_channels if is_first else hidden_dim * num_heads

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

        self.out_dim = hidden_dim

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_bn(x)
        h = x
        for conv, bn, res_proj in zip(
            self.convs, self.bns, self.res_projs
        ):
            residual = res_proj(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)
            h = h + residual
        return h


# ──────────────────────────────────────────────────────────────────
class EnsembleGNN(nn.Module):
    """
    SAGE + GAT 双流集成模型。

    融合策略：
      1. 两流分别生成嵌入
      2. 拼接后送入共享分类头
      3. 可学习的流权重（stream_weight）动态调整两流贡献
    """

    def __init__(
        self,
        in_channels:   int,
        sage_hidden:   int   = 512,
        gat_hidden:    int   = 256,
        sage_layers:   int   = 5,
        gat_layers:    int   = 4,
        gat_heads:     int   = 8,
        dropout:       float = 0.05,
        num_classes:   int   = 2,
        dropedge_rate: float = 0.03,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 两个流
        self.sage_stream = SAGEStream(
            in_channels   = in_channels,
            hidden_dim    = sage_hidden,
            num_layers    = sage_layers,
            dropout       = dropout,
            dropedge_rate = dropedge_rate,
        )
        self.gat_stream = GATStream(
            in_channels = in_channels,
            hidden_dim  = gat_hidden,
            num_layers  = gat_layers,
            num_heads   = gat_heads,
            dropout     = dropout,
        )

        sage_out = self.sage_stream.out_dim   # 512 * 5 = 2560
        gat_out  = self.gat_stream.out_dim    # 256
        fused_dim = sage_out + gat_out

        # 可学习流权重（初始化为等权）
        self.stream_weight = nn.Parameter(
            torch.tensor([0.5, 0.5])
        )

        # 融合后的分类头
        mid_dim = min(fused_dim // 2, 1024)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout),
            nn.Linear(mid_dim // 2, mid_dim // 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(mid_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 两流各自编码
        sage_emb = self.sage_stream(x, edge_index)   # [N, sage_out]
        gat_emb  = self.gat_stream(x, edge_index)    # [N, gat_out]

        # 可学习权重归一化
        w = torch.softmax(self.stream_weight, dim=0)

        # 按权重缩放后拼接
        sage_scaled = sage_emb * w[0]
        gat_scaled  = gat_emb  * w[1]
        fused       = torch.cat([sage_scaled, gat_scaled], dim=-1)

        logits     = self.classifier(fused)
        embeddings = fused  # 返回融合嵌入

        # ✅ 温度缩放：将概率从极端值(mean=0.93)拉回正常区间
        # T > 1 让 softmax 输出更平滑，阈值从 0.90 降至 0.40~0.55
        T = 2.5
        logits = logits / T
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
