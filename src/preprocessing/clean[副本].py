"""
原始数据清洗模块
================
加载 Elliptic Bitcoin Dataset 三个原始文件，执行：
  - 缺失值处理（中位数填充 / 均值填充 / 丢弃）
  - 异常值检测与裁剪（Z-score 法）
  - 重复行去除
  - 特征列与标签列合并

Elliptic 数据集字段说明：
  elliptic_txs_features.csv  — 203 769 个交易节点，166 列（无表头）
      列 0: txId（交易 ID）
      列 1: time_step（时间步 1-49）
      列 2-95:  94 维本地特征
      列 96-165: 72 维聚合特征
  elliptic_txs_classes.csv   — 节点标签 (txId, class)
      class: "1"=illicit, "2"=licit, "unknown"
  elliptic_txs_edgelist.csv  — 有向边 (txId1, txId2)
"""
#采用提取，转化，加载的设计结构。
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.utils.config import get_data_config, ensure_dir
#进行日志导入的工具
from src.utils.logger import get_logger

# 通过DataCleaner类来组织逻辑
class DataCleaner:
    """
    Elliptic 数据集清洗器。

    用法：
        cleaner = DataCleaner()
        node_df, edge_df = cleaner.clean()
    """

    def __init__(self, data_config: Optional[dict] = None):
        #1.加载配置，如果外部导入配置，则使用外部配置
        self.cfg    = data_config or get_data_config()
        # 2. 初始化日志器：用于记录清洗过程中的信息
        self.logger = get_logger("data_cleaner")
        #3.提取路径和预处理参数
        self.paths  = self.cfg["paths"]
        self.pp     = self.cfg["preprocessing"]

        #4.原始数据缓存
        self._feat_df:  Optional[pd.DataFrame] = None
        self._class_df: Optional[pd.DataFrame] = None
        self._edge_df:  Optional[pd.DataFrame] = None





    # 1. 数据加载模块
    #列名重构：原始特征文件是纯数值矩阵，必须通过代码生成语义化的列名（feat_0）才能进行后续的特征工程。
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载三个原始 CSV 文件并完成基础列命名。

        Returns:
            (features_df, classes_df, edges_df)
        """
        raw = self.paths["raw_dir"]
        self.logger.info(f"加载原始数据，目录: {raw}")

        #1.加载特征文件（无表头）
        feat_path = os.path.join(raw, self.paths["elliptic_features"])
        self._check_file(feat_path)
        feat_df = pd.read_csv(feat_path, header=None)#header=None因为源文件没有表头
        # 命名：txId, time_step, feature_0 … feature_163
        feat_df.columns = (
            ["txId", "time_step"]
            + [f"feat_{i}" for i in range(feat_df.shape[1] - 2)]
        )
        self.logger.info(f"特征文件: {feat_df.shape}")

        #2.标签文件
        cls_path = os.path.join(raw, self.paths["elliptic_classes"])
        self._check_file(cls_path)
        cls_df = pd.read_csv(cls_path)
        cls_df.columns = ["txId", "class"]
        self.logger.info(f"标签文件: {cls_df.shape}")

        #3.加载边列表文件
        edge_path = os.path.join(raw, self.paths["elliptic_edgelist"])
        self._check_file(edge_path)
        edge_df = pd.read_csv(edge_path)
        edge_df.columns = ["txId1", "txId2"]
        self.logger.info(f"边文件: {edge_df.shape}")

        self._feat_df  = feat_df
        self._class_df = cls_df
        self._edge_df  = edge_df
        return feat_df, cls_df, edge_df







    # 2. 数据清洗模块
    # 确保矩阵没有断裂
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按配置策略填充数值型列的缺失值。

        策略：median | mean | zero | drop
        """
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            self.logger.info("无缺失值，跳过")
            return df

        self.logger.info(f"缺失值总数: {total_missing}")
        strategy = self.pp["missing_value_strategy"]
        num_cols = df.select_dtypes(include=np.number).columns

        if strategy == "median":
            #中位数填充
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif strategy == "mean":
            #平均值填充
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif strategy == "zero":
            #零填充
            df[num_cols] = df[num_cols].fillna(0)
        elif strategy == "drop":
            #删除含有缺失值的行
            df = df.dropna(subset=num_cols).reset_index(drop=True)
        else:
            raise ValueError(f"未知缺失值策略: {strategy}")

        self.logger.info(f"缺失值处理完毕，策略: {strategy}")
        return df





    # 3. 重复值处理

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, key: str) -> pd.DataFrame:
        """按关键列去除重复行，保留第一条。"""
        before = len(df)
        df = df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            import warnings
            warnings.warn(f"去除重复行 {removed} 条（基于列 '{key}'）")
        return df





    # 4. 异常值处理（Z-score 裁剪）
    """对特征列使用 Z-score 方法检测并裁剪异常值"""
    # 衡量数据点距离平均值有多少个标准差：
    # 默认阈值设为5.0，如果某个值距离平均值超过 5 个标准差，它就被视为“极端异常值”
    def handle_outliers(
        self,
        df: pd.DataFrame,
        feat_cols: list,
        z_thresh: float = 5.0,
    ) -> pd.DataFrame:
        """
        对特征列使用 Z-score 方法检测并裁剪异常值。

        Args:
            df:        数据框
            feat_cols: 需检测的特征列
            z_thresh:  Z-score 阈值，超过则视为异常值

        Returns:
            裁剪后的数据框
        """
        total_clipped = 0
        for col in feat_cols:
            std = df[col].std()
            if std == 0:#防止处以0发生错误
                continue
            mean = df[col].mean()
            # 计算阈值边界
            lo, hi = mean - z_thresh * std, mean + z_thresh * std
            # 统计超出边界的点数
            clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            if clipped:
                # 将超出边界的值裁剪为边界值：裁剪可以在保留样本的同时消除极端异常值的影响
                df[col] = df[col].clip(lo, hi)
                total_clipped += clipped
        self.logger.info(f"异常值裁剪完毕，共处理 {total_clipped} 个点（z_thresh={z_thresh}）")
        return df

    def filter_unknown_class(self, df):
        """过滤未知标签样本，仅保留1(非法),2(合法)"""
        df = df[df['class'].isin([1, 2])].copy()
        # 标签转换：非法=1，合法=0（适配模型训练）
        df['class'] = df['class'].map({1: 1, 2: 0})
        self.logger.info(f"过滤未知标签后，样本数：{len(df)}")
        return df






    # 5. 主流程
    def clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行完整清洗流水线。

        Returns:
            (node_df, edge_df)
            node_df: 包含 txId, time_step, feat_*, class 列
            edge_df: 包含 txId1, txId2 列
        """
        self.logger.info("=" * 60)
        self.logger.info("开始数据清洗")
        self.logger.info("=" * 60)
        #加载原始数据
        feat_df, cls_df, edge_df = self.load_raw()

        # 去重处理
        feat_df  = self.drop_duplicates(feat_df, "txId")
        cls_df   = self.drop_duplicates(cls_df, "txId")
        edge_df  = edge_df.drop_duplicates().reset_index(drop=True)

        # 处理缺失值
        feat_df = self.handle_missing(feat_df)

        # 处理异常值
        feat_cols = [c for c in feat_df.columns if c.startswith("feat_")]
        feat_df   = self.handle_outliers(feat_df, feat_cols)

        # 合并标签，将特征(Features)和标签(Labels)合并
        node_df = pd.merge(feat_df, cls_df, on="txId", how="left")
        self.logger.info(f"合并后节点数: {len(node_df)}，列: {node_df.shape[1]}")

        # 统计标签分布
        self.logger.info(f"标签分布:\n{node_df['class'].value_counts().to_string()}")

        # 保存清洗后的数据到磁盘
        proc_dir = ensure_dir(self.paths["processed_dir"])
        node_out = os.path.join(proc_dir, self.paths["processed_nodes"])
        edge_out = os.path.join(proc_dir, self.paths["processed_edges"])
        node_df.to_csv(node_out, index=False)
        edge_df.to_csv(edge_out, index=False)
        self.logger.info(f"节点数据已保存: {node_out}")
        self.logger.info(f"边数据已保存:   {edge_out}")
        self.logger.info("数据清洗完成")

        return node_df, edge_df

    # ──────────────────────────────────────────────────────────────
    def _check_file(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")




# 初始化和配置加载──────────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = DataCleaner()
    nodes, edges = cleaner.clean()
    print(f"节点: {len(nodes)}, 边: {len(edges)}")