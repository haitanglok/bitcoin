"""
特征工程与数据转换模块
======================
在清洗数据的基础上执行：
  - 1.标签映射（将文本标签（"illicit"/"licit"）转化为模型能计算的数字（1/0）。
  - 2.特征标准化（StandardScaler / MinMaxScaler / RobustScaler），消除不同特征之间的量纲差异，防止模型训练“爆炸”。
  - 基于方差阈值的特征选择（可选）
  - 数据集划分：分层采样划分训练 / 验证 / 测试集
  - 计算类别权重（解决洗钱样本不平衡）

输出：train_df / val_df / test_df 及完整的转换元信息
"""

#将clean.py清洗后的原始数据转化成model-ready的数据集
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from src.utils.config import get_data_config, ensure_dir
from src.utils.logger import get_logger


class FeatureTransformer:
    """
    特征工程转换器。
    用法：
        t = FeatureTransformer()
        result = t.transform()
        # result keys: train_df, val_df, test_df, edge_df,
        #              feat_cols, class_weights, scaler, label_encoder
    """

    def __init__(self, data_config: Optional[dict] = None):
        self.cfg    = data_config or get_data_config()
        self.logger = get_logger("feature_transformer")
        self.paths  = self.cfg["paths"]
        self.pp     = self.cfg["preprocessing"]
        self.fe     = self.cfg["feature_engineering"]
        self.scaler = None          # 训练后可被 GraphBuilder 复用


    # 1. 加载清洗后的数据  ---（数据加载和标签处理）
    def load_cleaned(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        从 processed_dir 加载清洗后的节点与边 CSV。

        Returns:
            (node_df, edge_df)
        """
        pd_dir = self.paths["processed_dir"]
        node_df = pd.read_csv(os.path.join(pd_dir, self.paths["processed_nodes"]))
        edge_df = pd.read_csv(os.path.join(pd_dir, self.paths["processed_edges"]))
        self.logger.info(f"加载清洗数据: 节点 {len(node_df)}, 边 {len(edge_df)}")
        return node_df, edge_df


    # 2. 标签映射-----机械学习模型只能处理数字，而不能处理字符串等。这个操作是将类别标签映射为数字。
    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将原始 class 列映射为整数 label 列。

        映射规则来自 data_config.yaml::preprocessing.label_mapping。
        filter_unknown=True 时丢弃 unknown 行；否则标记为 -1。
        """
        # 1. 读取配置中的映射字典，并确保 Key 为字符串
        mapping = {str(k): int(v) for k, v in self.pp["label_mapping"].items()}
        df = df.copy()
        # 2. 核心映射逻辑：astype(str) -> str.strip() -> map(mapping)
        df["label"] = df["class"].astype(str).str.strip().map(mapping)  #map(mapping)：Pandas的映射函数，将Series对象映射为字典对象

        n_unknown = df["label"].isna().sum()
        self.logger.info(f"标签映射: unknown 节点 {n_unknown}")

       # 3。处理未知标签的策略
        if self.pp["filter_unknown"]:
            # 直接过滤掉未知样本
            before = len(df)
            df = df.dropna(subset=["label"]).reset_index(drop=True)
            df["label"] = df["label"].astype(int)
            self.logger.info(f"过滤 unknown 后: {before} → {len(df)} 节点")
        else:
            # 保留未知样本，标记为 -1
            df["label"] = df["label"].fillna(-1).astype(int)

        dist = df[df["label"] >= 0]["label"].value_counts().to_dict()
        self.logger.info(f"标签分布 — 正常(0): {dist.get(0,0)}, 洗钱(1): {dist.get(1,0)}")
        return df





    # 3. 特征筛选 --- 选用低方差过滤的方法：如果一个特征在所有样本中都是 0 或几乎不变，对于模型判断没有用处
    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        根据配置决定使用哪些特征列，并可选地进行方差过滤。

        Returns:
            最终使用的特征列名列表
        """
# 1.获取所有以始为 feat_ 的列
        all_feat = [c for c in df.columns if c.startswith("feat_")]

        if not self.fe["use_aggregate_features"]:
 # 2.仅保留本地特征（前 num_local_features 列）
            all_feat = all_feat[: self.fe["num_local_features"]]
# 3.方差过滤 ---移除方差极地的特征列
        if self.fe.get("feature_selection", False):
            thresh = self.fe.get("variance_threshold", 0.01)
            variances = df[all_feat].var()  # 计算所有特征列的方差
            all_feat = variances[variances > thresh].index.tolist()  # 选择方差大于阈值的特征列
            self.logger.info(f"方差过滤后保留特征: {len(all_feat)}")

        self.logger.info(f"特征总维度: {len(all_feat)}")
        return all_feat




    # 4. 数据集划分（分层采样）
    # 将数据集分为训练集、验证集和测试集，用于训练模型、调整参数和最终测试。
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按 train/val/test 比例对有标签节点进行分层采样划分。

        Returns:
            (train_df, val_df, test_df)
        """
        sp    = self.pp["split"]
        seed  = sp["random_seed"]
        tr, vl, te = sp["train_ratio"], sp["val_ratio"], sp["test_ratio"]
# 进行安全校验，确保比例之和为 1.0
        assert abs(tr + vl + te - 1.0) < 1e-6, "比例之和须为 1.0"
# 仅对有明确标签的数据进行划分 (label >= 0)
        labeled = df[df["label"] >= 0].copy()

# 分离出训练集剩下的作为临时集（1次切分）
        train_df, tmp_df = train_test_split(
            labeled, test_size=(vl + te), random_state=seed,
            stratify=labeled["label"]
        )
# 将临时集分为验证集和测试集（2次切分）
        val_df, test_df = train_test_split(
            tmp_df, test_size=te / (vl + te), random_state=seed,
            stratify=tmp_df["label"]
        )
        self.logger.info(
            f"数据集划分 — 训练: {len(train_df)}, "
            f"验证: {len(val_df)}, 测试: {len(test_df)}"
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )






    # 5. 特征标准化
                #Reason：比特币交易特征中，有的是时间戳（很大），有的是交易次数（很小）。
                         # 如果不标准化，数值大的特征会淹没数值小的特征。
    def scale(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feat_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        在训练集上 fit，再分别 transform 三个子集。

        Returns:
            (train_df, val_df, test_df) — 特征列已标准化
        """
# 1.根据配置选择不同的Scaler-
        scaler_map = {
            "standard": StandardScaler,
            "minmax":   MinMaxScaler,
            "robust":   RobustScaler,
        }
        key = self.pp["scaler"]
        if key not in scaler_map:
            raise ValueError(f"不支持的标准化方式: {key}")

        self.scaler = scaler_map[key]()
        train_df = train_df.copy()
        val_df   = val_df.copy()
        test_df  = test_df.copy()
# 2. Fit on Train: 只在训练集上学习参数 (均值/方差)
        train_df[feat_cols] = self.scaler.fit_transform(train_df[feat_cols])
# 3. Transform All: 使用训练集的参数去转换验证集和测试集
        val_df[feat_cols]   = self.scaler.transform(val_df[feat_cols])
        test_df[feat_cols]  = self.scaler.transform(test_df[feat_cols])

        self.logger.info(f"特征标准化完成（{key}），维度: {len(feat_cols)}")
        return train_df, val_df, test_df







    # 6. 计算类别权重
    @staticmethod
    def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
        """
        权重 = 总样本数 / (类别数 * 该类样本数)
        结果：样本越少，权重越大
        """
        unique, counts = np.unique(labels, return_counts=True)
        n = len(labels)
        weights = {int(c): n / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
        return weights





    # 7. 主流程
    def transform(self) -> Dict:

        self.logger.info("=" * 60)
        self.logger.info("开始特征工程")
        self.logger.info("=" * 60)

        node_df, edge_df = self.load_cleaned()
        node_df  = self.map_labels(node_df)# 将标签映射为数字
        feat_cols  = self.get_feature_cols(node_df)# 获取特征列
        train_df, val_df, test_df = self.split(node_df)# 数据集划分
        train_df, val_df, test_df = self.scale(
            train_df, val_df, test_df, feat_cols
        )# 特征标准化
        class_weights = self.compute_class_weights(train_df["label"].values)# 计算类别权重
        self.logger.info(f"类别权重: {class_weights}")

        # 持久化
        pd_dir = ensure_dir(self.paths["processed_dir"])
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            df.to_csv(os.path.join(pd_dir, f"{name}_data.csv"), index=False)
        self.logger.info("特征工程完成，分割数据已保存")

        return dict(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            edge_df=edge_df,
            feat_cols=feat_cols,
            class_weights=class_weights,
            scaler=self.scaler,
        )


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    t = FeatureTransformer()
    r = t.transform()
    for split in ("train_df", "val_df", "test_df"):
        d = r[split]
        print(f"{split}: {len(d)} | 洗钱={d['label'].sum()}")