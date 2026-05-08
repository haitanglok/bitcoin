"""
原始数据清洗模块（完整修复版）
================================
修复：
  [Fix-1] 补全被截断的 handle_missing() dropna 逻辑
  [Fix-2] 增加清晰的文件不存在错误提示
  [Fix-3] 自动创建 processed_dir
  [Fix-4] clean() 幂等：已存在则跳过
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.utils.config import get_data_config, ensure_dir, get_project_root
from src.utils.logger import get_logger


class DataCleaner:
    """Elliptic 数据集清洗器。"""

    def __init__(self, data_config: Optional[dict] = None):
        self.cfg    = data_config or get_data_config()
        self.logger = get_logger("data_cleaner")
        self.paths  = self.cfg["paths"]
        self.pp     = self.cfg["preprocessing"]

    # ──────────────────────────────────────────────────────────────
    def _check_file(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n❌ 原始数据文件不存在: {path}\n"
                f"   请从 Kaggle 下载 Elliptic Bitcoin Dataset：\n"
                f"   https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
                f"   将以下三个文件放入 data/raw/ 目录：\n"
                f"     - elliptic_txs_features.csv\n"
                f"     - elliptic_txs_classes.csv\n"
                f"     - elliptic_txs_edgelist.csv"
            )

    # ──────────────────────────────────────────────────────────────
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载三个原始 CSV 文件。"""
        raw = self.paths["raw_dir"]
        self.logger.info(f"加载原始数据，目录: {raw}")

        # 特征文件（无表头）
        feat_path = os.path.join(raw, self.paths["elliptic_features"])
        self._check_file(feat_path)
        feat_df = pd.read_csv(feat_path, header=None)
        feat_df.columns = (
            ["txId", "time_step"]
            + [f"feat_{i}" for i in range(feat_df.shape[1] - 2)]
        )
        self.logger.info(f"特征文件: {feat_df.shape}")

        # 标签文件
        cls_path = os.path.join(raw, self.paths["elliptic_classes"])
        self._check_file(cls_path)
        cls_df = pd.read_csv(cls_path)
        cls_df.columns = ["txId", "class"]
        self.logger.info(f"标签文件: {cls_df.shape}")

        # 边列表文件
        edge_path = os.path.join(raw, self.paths["elliptic_edgelist"])
        self._check_file(edge_path)
        edge_df = pd.read_csv(edge_path)
        edge_df.columns = ["txId1", "txId2"]
        self.logger.info(f"边文件: {edge_df.shape}")

        return feat_df, cls_df, edge_df

    # ──────────────────────────────────────────────────────────────
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """按配置策略填充缺失值。"""
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            self.logger.info("无缺失值，跳过")
            return df

        self.logger.info(f"缺失值总数: {total_missing}")
        strategy = self.pp["missing_value_strategy"]
        num_cols  = df.select_dtypes(include=np.number).columns.tolist()
        fill_cols = [c for c in num_cols if c not in ("txId", "time_step")]

        if strategy == "median":
            df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())
        elif strategy == "mean":
            df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())
        elif strategy == "zero":
            df[fill_cols] = df[fill_cols].fillna(0)
        elif strategy == "drop":
            before = len(df)
            # [Fix-1] 补全被截断的 dropna 逻辑
            df = df.dropna(subset=fill_cols).reset_index(drop=True)
            self.logger.info(f"删除含缺失值行: {before} → {len(df)}")
        else:
            self.logger.warning(f"未知缺失值策略 '{strategy}'，跳过")

        return df

    # ──────────────────────────────────────────────────────────────
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score / IQR 异常值裁剪。"""
        method    = self.pp.get("outlier_method", "none")
        if method == "none":
            return df

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        threshold = self.pp.get("outlier_threshold", 3.0)

        if method == "zscore":
            for col in feat_cols:
                mean = df[col].mean()
                std  = df[col].std()
                if std > 0:
                    df[col] = df[col].clip(
                        lower=mean - threshold * std,
                        upper=mean + threshold * std,
                    )
            self.logger.info(f"Z-score 异常值裁剪完成(threshold={threshold})")

        elif method == "iqr":
            for col in feat_cols:
                q1  = df[col].quantile(0.25)
                q3  = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(
                    lower=q1 - threshold * iqr,
                    upper=q3 + threshold * iqr,
                )
            self.logger.info(f"IQR 异常值裁剪完成(factor={threshold})")

        return df

    # ──────────────────────────────────────────────────────────────
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df     = df.drop_duplicates(subset=["txId"]).reset_index(drop=True)
        if len(df) < before:
            self.logger.info(f"去除重复节点: {before - len(df)} 行")
        return df

    # ──────────────────────────────────────────────────────────────
    def clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行完整清洗，输出 nodes_clean.csv 和 edges_clean.csv。
        幂等：已存在则跳过。
        """
        proc_dir = self.paths["processed_dir"]
        node_out = os.path.join(proc_dir, self.paths["processed_nodes"])
        edge_out = os.path.join(proc_dir, self.paths["processed_edges"])

        if os.path.exists(node_out) and os.path.exists(edge_out):
            self.logger.info("检测到已有清洗数据，跳过清洗步骤")
            return pd.read_csv(node_out), pd.read_csv(edge_out)

        ensure_dir(proc_dir)

        feat_df, cls_df, edge_df = self.load_raw()

        # 合并特征与标签
        node_df = feat_df.merge(cls_df, on="txId", how="left")
        node_df["class"] = node_df["class"].fillna("unknown")

        # 清洗流程
        node_df = self.handle_missing(node_df)
        node_df = self.handle_outliers(node_df)
        node_df = self.remove_duplicates(node_df)
        edge_df = edge_df.drop_duplicates().reset_index(drop=True)

        n_ill = (node_df["class"] == "1").sum()
        n_lic = (node_df["class"] == "2").sum()
        n_unk = (node_df["class"] == "unknown").sum()
        self.logger.info(
            f"清洗完成 | 节点={len(node_df)}, 边={len(edge_df)} | "
            f"illicit={n_ill}, licit={n_lic}, unknown={n_unk}"
        )

        node_df.to_csv(node_out, index=False)
        edge_df.to_csv(edge_out, index=False)
        self.logger.info(f"已保存: {node_out}")
        self.logger.info(f"已保存: {edge_out}")

        return node_df, edge_df
