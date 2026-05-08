"""
异常阈值分析与可疑账户/团伙识别模块
====================================
集成模块四（异常阈值分析）与模块五（可疑账户与团伙识别）：

模块四部分：
  - 基于分位数（默认 95%）设定洗钱风险阈值
  - 分析高误差样本特征，定位模型薄弱环节

模块五部分（进阶）：
  - 基于节点嵌入 + Louvain / Label Propagation 进行团伙聚类
  - 为每个节点计算综合风险评分
  - 生成可疑账户清单
  - 溯源分析：挖掘团伙内核心账户与交易路径
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data

from src.utils.config import get_model_config, get_project_root, ensure_dir
from src.utils.logger import get_logger


class ThresholdAnalyzer:
    """
    异常阈值分析器 —— 模块四的阈值分析功能。
    """

    def __init__(self, model_config: Optional[dict] = None):
        self.cfg     = model_config or get_model_config()
        self.logger  = get_logger("threshold_analyzer")
        self.eval_cfg = self.cfg.get("evaluation", {})
        self.report_dir = ensure_dir(
            os.path.join(get_project_root(), "reports")
        )

    def compute_threshold(
        self, y_prob: np.ndarray, percentile: Optional[float] = None
    ) -> float:
        """
        基于分位数计算洗钱风险阈值。

        默认取洗钱预测概率的 95% 分位数：
        高于此阈值的节点被标记为高风险可疑账户。

        Args:
            y_prob:     所有节点的洗钱概率 [N]
            percentile: 分位数百分比（0-100）；None 时使用配置值

        Returns:
            风险阈值（float）
        """
        p = percentile or self.eval_cfg.get("threshold_percentile", 95)
        threshold = float(np.percentile(y_prob, p))
        self.logger.info(f"风险阈值（P{p}）: {threshold:.4f}")
        return threshold

    def find_high_error_samples(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        feat_cols: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        定位高误差样本（模型预测最不确定或误分类的节点）。

        Args:
            y_true:        真实标签
            y_prob:        洗钱概率
            node_features: 节点特征矩阵（可选，用于特征分析）
            feat_cols:     特征列名（可选）
            top_n:         返回误差最大的前 N 个节点

        Returns:
            高误差节点的 DataFrame（含误差值、真实标签、预测概率）
        """
        # 误差 = |真实标签 - 预测概率| ，越大越说明模型不确定
        errors = np.abs(y_true - y_prob)
        top_indices = np.argsort(errors)[::-1][:top_n]

        records = []
        for idx in top_indices:
            record = {
                "node_index":   int(idx),
                "y_true":       int(y_true[idx]),
                "y_prob":       float(y_prob[idx]),
                "error":        float(errors[idx]),
                "is_misclassified": int(y_true[idx]) != int(y_prob[idx] >= 0.5),
            }
            if node_features is not None and feat_cols is not None:
                for col, val in zip(feat_cols[:10], node_features[idx, :10]):
                    record[f"top_{col}"] = float(val)
            records.append(record)

        df = pd.DataFrame(records)
        path = os.path.join(self.report_dir, "high_error_samples.csv")
        df.to_csv(path, index=False)
        self.logger.info(f"高误差样本已保存: {path}")

        # 分析薄弱环节：统计误分类中的类别分布
        misclf = df[df["is_misclassified"]]
        if len(misclf):
            fn = (misclf["y_true"] == 1).sum()  # 洗钱→误判为正常（漏报）
            fp = (misclf["y_true"] == 0).sum()  # 正常→误判为洗钱（误报）
            self.logger.info(
                f"模型薄弱环节分析（top-{top_n} 高误差样本）:\n"
                f"  漏报（洗钱→正常）: {fn} 个\n"
                f"  误报（正常→洗钱）: {fp} 个"
            )
        return df

    def generate_threshold_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict:
        """
        生成不同阈值下的精确率/召回率/F1 变化报告，
        辅助选择最优操作阈值。

        Returns:
            包含阈值搜索结果的字典
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        thresholds = np.arange(0.4, 0.85, 0.02)
        results = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            results.append({
                "threshold": round(float(t), 2),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall":    recall_score(y_true, y_pred, zero_division=0),
                "f1":        f1_score(y_true, y_pred, zero_division=0),
            })

            valid_results = [r for r in results if r["recall"] >= 0.90]
            if valid_results:
                optimal = max(valid_results, key=lambda x: x["f1"])
                self.logger.info(
                    f"最优阈值: {optimal['threshold']:.2f} | "
                    f"Precision: {optimal['precision']:.4f} | "
                    f"Recall: {optimal['recall']:.4f} | "
                    f"F1: {optimal['f1']:.4f}"
                )
            else:
                optimal = max(results, key=lambda x: x["f1"])
                self.logger.warning(f"未找到满足Recall≥0.90的阈值，使用最高F1阈值: {optimal['threshold']:.2f}")

            return {
                "threshold_curve": results,
                "optimal_threshold": optimal["threshold"],  # 🔧 返回最优阈值
                "current_threshold": 0.5,
            }





        df = pd.DataFrame(results)
        path = os.path.join(self.report_dir, "threshold_sweep.csv")
        df.to_csv(path, index=False)
        self.logger.info(f"阈值扫描报告已保存: {path}")

        # 最优 F1 对应的阈值
        best_row = df.loc[df["f1"].idxmax()]
        self.logger.info(
            f"最优阈值（基于 F1）: {best_row['threshold']:.2f} "
            f"→ F1={best_row['f1']:.4f}, "
            f"P={best_row['precision']:.4f}, "
            f"R={best_row['recall']:.4f}"
        )
        return {"sweep": df, "best": best_row.to_dict()}


# ─────────────────────────────────────────────────────────────────
class CommunityDetector:
    """
    可疑账户与洗钱团伙识别器 —— 模块五。

    流程：
      1. 利用训练好的节点嵌入 + 模型预测概率计算综合风险评分
      2. 将交易图转为 NetworkX 图，用 Louvain / Label Propagation 聚类
      3. 对聚类社区内洗钱风险进行评估，标记高风险团伙
      4. 对高风险团伙进行溯源：找出核心节点与交易路径
    """

    def __init__(self, model_config: Optional[dict] = None):
        self.cfg     = model_config or get_model_config()
        self.logger  = get_logger("community_detector")
        self.comm_cfg = self.cfg.get("community", {})
        self.report_dir = ensure_dir(
            os.path.join(get_project_root(), "reports")
        )

    # ──────────────────────────────────────────────────────────────
    # 1. 计算节点风险评分
    # ──────────────────────────────────────────────────────────────
    def compute_risk_scores(
        self,
        y_prob:     np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        综合预测概率与嵌入空间到洗钱类中心的距离，计算风险评分。

        风险评分 = 0.7 × 洗钱概率 + 0.3 × 归一化嵌入距离得分

        Args:
            y_prob:     洗钱预测概率 [N]
            embeddings: 节点嵌入向量 [N, D]

        Returns:
            综合风险评分 [N]，范围 [0, 1]
        """
        from sklearn.preprocessing import normalize

        # 在嵌入空间中找洗钱样本（高概率）的中心
        high_risk_mask = y_prob > 0.5
        if high_risk_mask.sum() == 0:
            self.logger.warning("无高概率洗钱节点，风险评分退化为预测概率")
            return y_prob

        illicit_center = embeddings[high_risk_mask].mean(axis=0)

        # 计算每个节点到洗钱中心的余弦相似度作为嵌入风险分
        normed_emb    = normalize(embeddings, norm="l2")
        normed_center = illicit_center / (np.linalg.norm(illicit_center) + 1e-8)
        cosine_sim    = normed_emb @ normed_center  # [N]

        # 归一化到 [0, 1]
        cosine_score = (cosine_sim - cosine_sim.min()) / (
            cosine_sim.max() - cosine_sim.min() + 1e-8
        )

        risk_scores = 0.7 * y_prob + 0.3 * cosine_score
        self.logger.info(
            f"风险评分计算完毕，高风险节点(>0.7): "
            f"{(risk_scores > 0.7).sum()}"
        )
        return risk_scores

    # ──────────────────────────────────────────────────────────────
    # 2. 构建 NetworkX 图
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def build_nx_graph(
        edge_index:  torch.Tensor,
        risk_scores: np.ndarray,
    ) -> nx.DiGraph:
        """
        从 PyG edge_index 构建带节点风险属性的 NetworkX 有向图。

        Args:
            edge_index:  [2, E] 边索引张量
            risk_scores: [N] 节点风险评分

        Returns:
            NetworkX DiGraph（节点带 risk_score 属性）
        """
        G = nx.DiGraph()

        # 添加节点
        N = len(risk_scores)
        for i in range(N):
            G.add_node(i, risk_score=float(risk_scores[i]))

        # 添加边
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        for s, d in zip(src, dst):
            G.add_edge(int(s), int(d))

        return G

    # ──────────────────────────────────────────────────────────────
    # 3. 社区检测（Louvain / Label Propagation）
    # ──────────────────────────────────────────────────────────────
    def detect_communities(
        self, G: nx.DiGraph
    ) -> Dict[int, int]:
        """
        对图执行社区检测，返回节点→社区 ID 的映射。

        Args:
            G: NetworkX 图（有向图会转为无向图进行聚类）

        Returns:
            {node_id: community_id} 字典
        """
        algo  = self.comm_cfg.get("algorithm", "louvain")
        G_und = G.to_undirected()   # Louvain/LP 通常需要无向图

        if algo == "louvain":
            try:
                import community as community_louvain
                resolution   = self.comm_cfg.get("resolution", 1.0)
                partition    = community_louvain.best_partition(
                    G_und, resolution=resolution, random_state=42
                )
                self.logger.info(
                    f"Louvain 社区检测完成，社区数: {len(set(partition.values()))}"
                )
                return partition
            except ImportError:
                self.logger.warning("python-louvain 未安装，降级使用 Label Propagation")
                algo = "label_propagation"

        if algo == "label_propagation":
            communities = nx.algorithms.community.label_propagation_communities(G_und)
            partition = {}
            for cid, comm in enumerate(communities):
                for node in comm:
                    partition[node] = cid
            self.logger.info(
                f"Label Propagation 完成，社区数: {len(set(partition.values()))}"
            )
            return partition

        raise ValueError(f"不支持的聚类算法: {algo}")

    # ──────────────────────────────────────────────────────────────
    # 4. 生成可疑账户清单
    # ──────────────────────────────────────────────────────────────
    def get_suspicious_accounts(
        self,
        risk_scores: np.ndarray,
        partition:   Dict[int, int],
        y_prob:      np.ndarray,
        threshold:   Optional[float] = None,
    ) -> pd.DataFrame:
        """
        生成可疑账户清单（含风险评分、社区 ID）。

        Args:
            risk_scores: 综合风险评分 [N]
            partition:   节点→社区 ID 映射
            y_prob:      洗钱概率 [N]
            threshold:   风险阈值；None 时使用配置值

        Returns:
            按风险评分降序排列的可疑账户 DataFrame
        """
        t = threshold or self.comm_cfg.get("risk_score_threshold", 0.7)

        records = []
        for node_id, score in enumerate(risk_scores):
            if score >= t:
                records.append({
                    "node_id":         node_id,
                    "risk_score":      round(float(score), 4),
                    "illicit_prob":    round(float(y_prob[node_id]), 4),
                    "community_id":    partition.get(node_id, -1),
                    "risk_level":      "极高" if score >= 0.9 else "高",
                })

        df = pd.DataFrame(records).sort_values("risk_score", ascending=False)
        df = df.reset_index(drop=True)

        path = os.path.join(self.report_dir, "suspicious_accounts.csv")
        df.to_csv(path, index=False)
        self.logger.info(
            f"可疑账户清单已保存: {path}（共 {len(df)} 个高风险账户）"
        )
        return df

    # ──────────────────────────────────────────────────────────────
    # 5. 识别洗钱团伙
    # ──────────────────────────────────────────────────────────────
    def identify_gangs(
        self,
        suspicious_df: pd.DataFrame,
        partition:     Dict[int, int],
        risk_scores:   np.ndarray,
    ) -> pd.DataFrame:
        """
        统计高风险社区（洗钱团伙），输出团伙报告。

        Args:
            suspicious_df: 可疑账户 DataFrame
            partition:     节点→社区 ID 映射
            risk_scores:   所有节点的风险评分

        Returns:
            洗钱团伙汇总 DataFrame
        """
        min_size = self.comm_cfg.get("min_community_size", 3)

        # 按社区聚合可疑账户
        gang_stats = (
            suspicious_df.groupby("community_id")
            .agg(
                member_count=("node_id", "count"),
                avg_risk=("risk_score", "mean"),
                max_risk=("risk_score", "max"),
                members=("node_id", list),
            )
            .reset_index()
        )

        # 过滤规模过小的团伙
        gang_stats = gang_stats[gang_stats["member_count"] >= min_size]
        gang_stats = gang_stats.sort_values("avg_risk", ascending=False)
        gang_stats = gang_stats.reset_index(drop=True)

        path = os.path.join(self.report_dir, "money_laundering_gangs.csv")
        gang_stats.drop(columns=["members"]).to_csv(path, index=False)
        self.logger.info(
            f"洗钱团伙报告已保存: {path}（共 {len(gang_stats)} 个团伙）"
        )
        return gang_stats

    # ──────────────────────────────────────────────────────────────
    # 6. 溯源分析
    # ──────────────────────────────────────────────────────────────
    def trace_gang(
        self,
        G:           nx.DiGraph,
        gang_members: List[int],
        risk_scores:  np.ndarray,
        top_n_paths:  int = 5,
    ) -> Dict:
        """
        对单个洗钱团伙进行溯源分析：
          - 找出团伙内的核心账户（PageRank 最高）
          - 挖掘成员间的最短交易路径

        Args:
            G:            完整交易图
            gang_members: 团伙成员节点 ID 列表
            risk_scores:  节点风险评分
            top_n_paths:  返回的路径数量

        Returns:
            包含核心节点与交易路径的字典
        """
        # 提取子图
        subgraph = G.subgraph(gang_members).copy()

        # 核心账户：在子图中 PageRank 最高的节点
        try:
            pr = nx.pagerank(subgraph, alpha=0.85)
            core_nodes = sorted(pr, key=pr.get, reverse=True)[:3]
        except nx.PowerIterationFailedConvergence:
            # PageRank 不收敛时退化为度数最高
            degree_dict = dict(subgraph.degree())
            core_nodes  = sorted(degree_dict, key=degree_dict.get, reverse=True)[:3]

        # 交易路径：核心账户之间的最短路径
        paths = []
        for i, src in enumerate(core_nodes):
            for dst in core_nodes[i + 1:]:
                try:
                    path = nx.shortest_path(subgraph, src, dst)
                    paths.append({
                        "source": src,
                        "target": dst,
                        "path":   path,
                        "length": len(path) - 1,
                    })
                except nx.NetworkXNoPath:
                    pass
                if len(paths) >= top_n_paths:
                    break
            if len(paths) >= top_n_paths:
                break

        result = {
            "gang_size":    len(gang_members),
            "core_nodes":   core_nodes,
            "core_risks":   [float(risk_scores[n]) for n in core_nodes],
            "paths":        paths,
            "subgraph_edges": list(subgraph.edges()),
        }

        self.logger.info(
            f"团伙溯源完成: 规模={len(gang_members)}, "
            f"核心账户={core_nodes}, 路径数={len(paths)}"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # 7. 主流程入口
    # ──────────────────────────────────────────────────────────────
    def run(
        self,
        data:        Data,
        embeddings:  torch.Tensor,
        eval_result: Dict,
    ) -> Dict:
        """
        执行完整的团伙识别与溯源分析流水线。

        Args:
            data:        PyG Data 对象
            embeddings:  节点嵌入张量 [N, D]
            eval_result: 评估结果字典（含 y_prob）

        Returns:
            包含风险评分、可疑账户、团伙、溯源结果的综合报告字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始可疑账户与团伙识别分析")
        self.logger.info("=" * 60)

        y_prob     = eval_result["y_prob"]
        emb_np     = embeddings.numpy()
        ei         = data.edge_index

        # Step 1: 计算风险评分
        risk_scores = self.compute_risk_scores(y_prob, emb_np)

        # Step 2: 计算风险阈值（P95）
        threshold = ThresholdAnalyzer(self.cfg).compute_threshold(risk_scores)

        # Step 3: 构建 NetworkX 图
        G = self.build_nx_graph(ei, risk_scores)

        # Step 4: 社区检测
        partition = self.detect_communities(G)

        # Step 5: 生成可疑账户清单
        suspicious_df = self.get_suspicious_accounts(
            risk_scores, partition, y_prob, threshold
        )

        # Step 6: 识别洗钱团伙
        gang_df = self.identify_gangs(suspicious_df, partition, risk_scores)

        # Step 7: 对规模最大的团伙进行溯源
        gang_traces = []
        if len(gang_df) > 0:
            for _, row in gang_df.head(3).iterrows():
                members = suspicious_df[
                    suspicious_df["community_id"] == row["community_id"]
                ]["node_id"].tolist()
                trace = self.trace_gang(G, members, risk_scores)
                gang_traces.append(trace)

        # 保存综合报告
        summary = {
            "n_suspicious":     len(suspicious_df),
            "n_gangs":          len(gang_df),
            "risk_threshold":   float(threshold),
            "top_gang_traces":  gang_traces,
        }
        report_path = os.path.join(self.report_dir, "community_summary.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"团伙识别综合报告已保存: {report_path}")

        return {
            "risk_scores":   risk_scores,
            "suspicious_df": suspicious_df,
            "gang_df":       gang_df,
            "gang_traces":   gang_traces,
            "partition":     partition,
            "threshold":     threshold,
        }