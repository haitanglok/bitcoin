"""
特征工程与数据转换模块（完整修复版）
======================================
修复：
  [Fix-1] 明确输出 features_scaled.csv 和 classes_clean.csv
  [Fix-2] 补全被截断的 get_feature_cols() 方差过滤逻辑
  [Fix-3] unknown 节点保留（label=-1），由 graph_builder 过滤
  [Fix-4] transform() 幂等：已存在则跳过
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.config import get_data_config, ensure_dir, get_project_root
from src.utils.logger import get_logger


class FeatureTransformer:
    """特征工程转换器。"""

    def __init__(self, data_config: Optional[dict] = None):
        self.cfg    = data_config or get_data_config()
        self.logger = get_logger("feature_transformer")
        self.paths  = self.cfg["paths"]
        self.pp     = self.cfg["preprocessing"]
        self.fe     = self.cfg["feature_engineering"]
        self.scaler = None

    # ──────────────────────────────────────────────────────────────
    def load_cleaned(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pd_dir  = self.paths["processed_dir"]
        node_df = pd.read_csv(
            os.path.join(pd_dir, self.paths["processed_nodes"])
        )
        edge_df = pd.read_csv(
            os.path.join(pd_dir, self.paths["processed_edges"])
        )
        self.logger.info(f"加载清洗数据: 节点={len(node_df)}, 边={len(edge_df)}")
        return node_df, edge_df

    # ──────────────────────────────────────────────────────────────
    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """illicit("1")→1, licit("2")→0, unknown→-1。"""
        mapping = {
            str(k): int(v)
            for k, v in self.pp["label_mapping"].items()
        }
        df = df.copy()
        df["label"] = df["class"].astype(str).str.strip().map(mapping)
        df["label"] = df["label"].fillna(-1).astype(int)

        dist = df[df["label"] >= 0]["label"].value_counts().to_dict()
        self.logger.info(
            f"标签分布 — licit(0)={dist.get(0,0)}, "
            f"illicit(1)={dist.get(1,0)}, "
            f"unknown(-1)={(df['label']==-1).sum()}"
        )
        return df

    # ──────────────────────────────────────────────────────────────
    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """选择特征列，可选方差过滤。"""
        all_feat = [c for c in df.columns if c.startswith("feat_")]

        if not self.fe["use_aggregate_features"]:
            all_feat = all_feat[: self.fe["num_local_features"]]
            self.logger.info(f"仅使用本地特征: {len(all_feat)} 维")

        # 方差过滤（补全被截断的逻辑）
        if self.fe.get("feature_selection", False):
            thresh    = self.fe.get("variance_threshold", 0.01)
            variances = df[all_feat].var()
            before    = len(all_feat)
            all_feat  = variances[variances >= thresh].index.tolist()
            self.logger.info(
                f"方差过滤(threshold={thresh}): {before} → {len(all_feat)} 维"
            )

        self.logger.info(f"最终特征维度: {len(all_feat)}")
        return all_feat

    # ──────────────────────────────────────────────────────────────
    def scale_features(
        self,
        df:        pd.DataFrame,
        feat_cols: List[str],
        fit:       bool = True,
    ) -> pd.DataFrame:
        """特征标准化。"""
        scaler_type = self.fe.get("scaler", "standard")

        if self.scaler is None or fit:
            scaler_map = {
                "standard": StandardScaler,
                "minmax":   MinMaxScaler,
                "robust":   RobustScaler,
            }
            self.scaler = scaler_map.get(scaler_type, StandardScaler)()

        df = df.copy()
        if fit:
            df[feat_cols] = self.scaler.fit_transform(df[feat_cols].values)
            self.logger.info(
                f"特征标准化(fit+transform): {scaler_type}, {len(feat_cols)} 维"
            )
        else:
            df[feat_cols] = self.scaler.transform(df[feat_cols].values)
            self.logger.info(f"特征标准化(transform only): {scaler_type}")

        return df

    # ──────────────────────────────────────────────────────────────
    def transform(self) -> None:
        """
        执行完整特征工程，输出：
          - data/processed/features_scaled.csv
          - data/processed/classes_clean.csv
        幂等：已存在则跳过。
        """
        proc_dir     = self.paths["processed_dir"]
        features_out = os.path.join(
            proc_dir,
            self.paths.get("features_scaled", "features_scaled.csv")
        )
        classes_out  = os.path.join(
            proc_dir,
            self.paths.get("classes_clean", "classes_clean.csv")
        )

        if os.path.exists(features_out) and os.path.exists(classes_out):
            self.logger.info("检测到已有特征工程数据，跳过 transform 步骤")
            return

        ensure_dir(proc_dir)

        node_df, edge_df = self.load_cleaned()
        node_df          = self.map_labels(node_df)
        feat_cols        = self.get_feature_cols(node_df)
        node_df          = self.scale_features(node_df, feat_cols, fit=True)

        # 输出 features_scaled.csv
        keep_cols   = ["txId", "time_step"] + feat_cols
        features_df = node_df[keep_cols]
        features_df.to_csv(features_out, index=False)
        self.logger.info(
            f"已保存: {features_out}  shape={features_df.shape}"
        )

        # 输出 classes_clean.csv
        classes_df = node_df[["txId", "class", "label"]]
        classes_df.to_csv(classes_out, index=False)
        self.logger.info(
            f"已保存: {classes_out}  shape={classes_df.shape}"
        )

        self.logger.info("✅ 特征工程完成")
