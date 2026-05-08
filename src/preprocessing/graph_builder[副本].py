"""
区块链交易图网络构建模块
========================
将表格形式的节点特征与边列表转换为 PyTorch Geometric Data 对象。
主要任务：格式转换和索引映射
图定义：
  节点  —— 区块链交易账户（txId）
  边    —— 账户之间的交易行为（有向）
  x     —— 节点特征矩阵  [N, F]
  y     —— 节点标签向量  [N]（-1 表示无标签）
  train_mask / val_mask / test_mask —— 布尔掩码

  5个需要实现的功能：
ID 映射：将原始交易 ID (txId) 映射为 [0, N) 的连续整数索引。
特征矩阵化：将节点特征列表转换为 PyTorch 张量 [N, F]。
边索引构建：将边列表转换为 PyG 特有的 [2, E] 稀疏矩阵格式。
标签与掩码生成：构建标签向量 y 和训练/验证/测试的布尔掩码 (Masks)。
对象序列化：将所有数据打包成一个 .pt (Pickle) 文件，供模型直接加载。
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from torch_geometric.data import Data

from src.utils.config import get_data_config, ensure_dir
from src.utils.logger import get_logger


class GraphBuilder:
    """
    区块链交易图构建器。

    用法：
        builder = GraphBuilder()
        data = builder.build(transform_result)   # PyG Data 对象
    """

    def __init__(self, data_config: Optional[dict] = None):
        self.cfg    = data_config or get_data_config()
        self.logger = get_logger("graph_builder")
        self.paths  = self.cfg["paths"]

    # ──────────────────────────────────────────────────────────────
    # 1. 构建 ID 映射表 (核心：离散 -> 连续)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_id_map(node_ids: np.ndarray) -> Dict[int, int]:
        """将离散的 txId 映射到 [0, N) 的连续整数索引。
        原因：GNN 计算依赖索引查找，无法直接处理巨大的原始 ID。"""
        unique = np.unique(node_ids)
        # 返回字典：{原始ID: 连续索引}
        return {int(uid): idx for idx, uid in enumerate(unique)}

    # ──────────────────────────────────────────────────────────────
    # 2. 构建节点特征矩阵
    # ──────────────────────────────────────────────────────────────
    def _build_x(
        self,
        all_df: pd.DataFrame,
        feat_cols: list,
        id_map: Dict[int, int],
    ) -> torch.Tensor:
        """
        构建 shape = [N, F] 的节点特征张量。
       逻辑：初始化全0矩阵 -> 遍历数据 -> 根据 id_map 填入对应行。
        """
        N, F = len(id_map), len(feat_cols)
        x = np.zeros((N, F), dtype=np.float32)
        for _, row in all_df.iterrows():
            nid = int(row["txId"])
            if nid in id_map:
                x[id_map[nid]] = row[feat_cols].values.astype(np.float32)
        t = torch.tensor(x, dtype=torch.float)
        self.logger.info(f"节点特征矩阵: {list(t.shape)}")
        return t

    # ──────────────────────────────────────────────────────────────
    # 3. 构建边索引Edge Index
    # ──────────────────────────────────────────────────────────────
    def _build_edge_index(
        self,
        edge_df: pd.DataFrame,
        id_map: Dict[int, int],
    ) -> torch.Tensor:
        """
        构建 shape = [2, E] 的 edge_index 张量（PyG 格式）。

        """
        # 过滤端点不在 id_map 中的边。
        valid = edge_df[
            edge_df["txId1"].isin(id_map) & edge_df["txId2"].isin(id_map)
        ].copy()
        dropped = len(edge_df) - len(valid)
        if dropped:
            self.logger.warning(f"过滤无效边 {dropped} 条（端点不在节点集中）")
        #将原始id转换成连续索引
        src = valid["txId1"].map(id_map).values.astype(np.int64)
        dst = valid["txId2"].map(id_map).values.astype(np.int64)
        # PyG 的标准格式：[[src1, src2...], [dst1, dst2...]]
        ei  = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        self.logger.info(f"边索引: {list(ei.shape)}")
        return ei

    # ──────────────────────────────────────────────────────────────
    # 4. 构建标签向量Y
    # ──────────────────────────────────────────────────────────────
    def _build_y(
        self,
        all_df: pd.DataFrame,
        id_map: Dict[int, int],
    ) -> torch.Tensor:
        """
              构建 shape = [N] 的标签张量。
              无标签节点（如未知交易）标记为 -1，PyTorch 会自动忽略 -1 的 Loss 计算。
              """
        """构建 shape = [N] 的标签张量（无标签节点为 -1）。"""
        y = torch.full((len(id_map),), -1, dtype=torch.long)
        for _, row in all_df.iterrows():
            nid = int(row["txId"])
            if nid in id_map and "label" in row.index:
                y[id_map[nid]] = int(row["label"])
        labeled = (y >= 0).sum().item()# 有标签节点数
        illicit = (y == 1).sum().item()# 洗钱节点数
        self.logger.info(f"标签向量: 总节点 {len(id_map)}, 有标签 {labeled}, 洗钱 {illicit}")
        return y

    # ──────────────────────────────────────────────────────────────
    # 5. 构建训练/验证/测试掩码
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build_masks(
        N: int,
        train_ids: set,
        val_ids: set,
        test_ids: set,
        id_map: Dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
              生成布尔掩码张量。
              作用：告诉模型在不同阶段（训练/验证/测试）使用哪些节点。
              """
        """分别为训练/验证/测试节点生成布尔掩码张量。"""
        tm = torch.zeros(N, dtype=torch.bool)
        vm = torch.zeros(N, dtype=torch.bool)
        em = torch.zeros(N, dtype=torch.bool)
        for nid, idx in id_map.items():
            if nid in train_ids:
                tm[idx] = True
            elif nid in val_ids:
                vm[idx] = True
            elif nid in test_ids:
                em[idx] = True
        return tm, vm, em

    # ──────────────────────────────────────────────────────────────
    # 6. 主构建接口
    # ──────────────────────────────────────────────────────────────
    def build(self, transform_result: Dict) -> Data:
        """
        从 FeatureTransformer.transform() 的输出构建完整 PyG Data 对象。

        Args:
            transform_result: 特征工程输出字典

        Returns:
            torch_geometric.data.Data 对象，包含：
              x, edge_index, y, train_mask, val_mask, test_mask
        """
        self.logger.info("=" * 60)
        self.logger.info("开始构建区块链交易图网络")
        self.logger.info("=" * 60)

        train_df  = transform_result["train_df"]
        val_df    = transform_result["val_df"]
        test_df   = transform_result["test_df"]
        edge_df   = transform_result["edge_df"]
        feat_cols = transform_result["feat_cols"]

        # 合并所有有标签节点
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # 构建节点映射（仅含有标签节点）
        id_map = self._build_id_map(all_df["txId"].values)
        N      = len(id_map)

        # 构建各张量
        x          = self._build_x(all_df, feat_cols, id_map)
        edge_index = self._build_edge_index(edge_df, id_map)
        y          = self._build_y(all_df, id_map)

        train_ids = set(train_df["txId"].astype(int).tolist())
        val_ids   = set(val_df["txId"].astype(int).tolist())
        test_ids  = set(test_df["txId"].astype(int).tolist())
        train_mask, val_mask, test_mask = self._build_masks(
            N, train_ids, val_ids, test_ids, id_map
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        # 附加元信息，方便后续溯源分析
        data.id_map      = id_map
        data.num_classes = 2

        self.logger.info(
            f"图构建完成 — 节点: {data.num_nodes}, "
            f"边: {data.num_edges}, 特征维度: {data.num_node_features}"
        )

        # 保存 Data 对象
        emb_dir  = ensure_dir(self.paths["embeddings_dir"])
        save_path = os.path.join(
            ensure_dir(self.paths["processed_dir"]), "graph_data.pt"
        )
        torch.save(data, save_path)
        self.logger.info(f"图数据已保存: {save_path}")

        return data


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing.transform import FeatureTransformer
    result = FeatureTransformer().transform()
    data   = GraphBuilder().build(result)
    print(data)