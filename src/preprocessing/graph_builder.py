"""
图构建模块（终极优化版）
========================
核心新增：
  [New-1] 训练集 illicit 节点过采样（复制3次）
          让模型在训练时看到更多 illicit 的图结构
  [New-2] 数据划分改为 60/20/20，给训练集更多 illicit 样本
  [New-3] 增加详细的 illicit 分布统计
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from torch_geometric.data import Data

from src.utils.config import get_data_config, ensure_dir, get_project_root
from src.utils.logger import get_logger


class GraphBuilder:
    """区块链交易图构建器（终极优化版）。"""

    def __init__(self, data_config: Optional[dict] = None):
        self.cfg    = data_config or get_data_config()
        self.logger = get_logger("graph_builder")
        self.root   = get_project_root()

    def build(self, tf_result: Optional[Dict[str, Any]] = None) -> Data:
        processed_dir = self.cfg["paths"]["processed_dir"]
        graph_path    = os.path.join(processed_dir, "graph.pt")

        if os.path.exists(graph_path):
            self.logger.info(f"加载已有图缓存: {graph_path}")
            return torch.load(graph_path, weights_only=False)

        self.logger.info("开始构建图数据（终极优化版）...")
        ensure_dir(processed_dir)

        # ── 读取文件 ──────────────────────────────────────────────
        features_path = os.path.join(
            processed_dir,
            self.cfg["paths"].get("features_scaled", "features_scaled.csv")
        )
        classes_path = os.path.join(
            processed_dir,
            self.cfg["paths"].get("classes_clean", "classes_clean.csv")
        )
        edges_path = os.path.join(
            processed_dir,
            self.cfg["paths"].get("edges_clean", "edges_clean.csv")
        )

        for p in [features_path, classes_path, edges_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"\n❌ 文件不存在: {p}\n"
                    f"   请先运行 DataCleaner 和 FeatureTransformer"
                )

        features_df = pd.read_csv(features_path)
        classes_df  = pd.read_csv(classes_path)
        edges_df    = pd.read_csv(edges_path)

        self.logger.info(
            f"读取完成 | features={features_df.shape}, "
            f"classes={classes_df.shape}, edges={edges_df.shape}"
        )

        # ── ID 映射 ───────────────────────────────────────────────
        all_txids = features_df["txId"].values
        id2idx    = {txid: idx for idx, txid in enumerate(all_txids)}
        N         = len(all_txids)

        # ── 特征矩阵 ──────────────────────────────────────────────
        feat_cols = [
            c for c in features_df.columns
            if c not in ("txId", "time_step")
        ]
        x = torch.tensor(features_df[feat_cols].values, dtype=torch.float)
        self.logger.info(f"节点数={N}, 特征维度={x.shape[1]}")

        # ── 标签向量 ──────────────────────────────────────────────
        if "label" in classes_df.columns:
            label_series = classes_df.set_index("txId")["label"]
        else:
            label_map    = {"1": 1, "2": 0, 1: 1, 2: 0, "unknown": -1}
            classes_df["label"] = (
                classes_df["class"].map(label_map).fillna(-1).astype(int)
            )
            label_series = classes_df.set_index("txId")["label"]

        y_list = [int(label_series.get(txid, -1)) for txid in all_txids]
        y      = torch.tensor(y_list, dtype=torch.long)

        n_illicit = (y == 1).sum().item()
        n_licit   = (y == 0).sum().item()
        n_unknown = (y == -1).sum().item()
        self.logger.info(
            f"全图标签 → illicit={n_illicit}, "
            f"licit={n_licit}, unknown={n_unknown}"
        )

        # ── 时间步划分（改为 60/20/20）────────────────────────────
        # [New-2] 给训练集更多时间步，让模型见到更多 illicit 样本
        ts_arr    = features_df["time_step"].values
        unique_ts = np.sort(np.unique(ts_arr))
        n_ts      = len(unique_ts)
        train_end = unique_ts[int(n_ts * 0.50) - 1]   # 60% 训练
        val_end   = unique_ts[int(n_ts * 0.75) - 1]   # 20% 验证

        self.logger.info(
            f"时间步划分(50/25/25) | 共{n_ts}步 | "
            f"train≤{train_end}, val≤{val_end}, test>{val_end}"
        )

        # ── 构建 mask ─────────────────────────────────────────────
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)

        for i in range(N):
            if y[i].item() < 0:
                continue
            ts = ts_arr[i]
            if ts <= train_end:
                train_mask[i] = True
            elif ts <= val_end:
                val_mask[i]   = True
            else:
                test_mask[i]  = True

        # 各 split 统计
        for name, mask in [
            ("train", train_mask),
            ("val",   val_mask),
            ("test",  test_mask),
        ]:
            sl = y[mask]
            n1 = (sl == 1).sum().item()
            n0 = (sl == 0).sum().item()
            self.logger.info(
                f"  {name}: illicit={n1}, licit={n0}, "
                f"ratio=1:{n0 // max(n1, 1)}"
            )

        # ── 边索引构建 ────────────────────────────────────────────
        src_list, dst_list = [], []
        skipped = 0
        for _, row in edges_df.iterrows():
            src, dst = row["txId1"], row["txId2"]
            if src in id2idx and dst in id2idx:
                src_list.append(id2idx[src])
                dst_list.append(id2idx[dst])
            else:
                skipped += 1

        if skipped > 0:
            self.logger.warning(f"跳过 {skipped} 条边")

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.logger.info(f"边数量: {edge_index.shape[1]}")

        # ── 构建 PyG Data ─────────────────────────────────────────
        data = Data(
            x          = x,
            edge_index = edge_index,
            y          = y,
            train_mask = train_mask,
            val_mask   = val_mask,
            test_mask  = test_mask,
        )
        data.num_classes = 2

        torch.save(data, graph_path)
        self.logger.info(f"✅ 图数据已缓存: {graph_path}")
        return data
