"""
可视化分析模块
==============
生成以下分析图：
  1. PCA 降维图   — 直观区分正常 / 洗钱账户节点在嵌入空间的分布
  2. UMAP 降维图  — 比 PCA 更能捕捉非线性结构
  3. 预测误差分布直方图 — 洗钱概率在正确/错误预测节点上的分布
  4. ROC 曲线
  5. PR 曲线（Precision-Recall，适合不平衡样本）
  6. 混淆矩阵热力图

所有图像保存到 reports/figures/ 目录。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # 非交互后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（根据系统选择）
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
from typing import Dict, Optional

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils.config   import get_model_config, get_project_root, ensure_dir
from src.utils.logger   import get_logger


class Visualizer:
    """
    GNN 评估可视化器。

    用法：
        viz = Visualizer()
        viz.plot_all(embeddings, eval_result)
    """

    def __init__(self, model_config: Optional[dict] = None):
        self.cfg = model_config or get_model_config()
        self.logger = get_logger("visualizer")
        self.fig_dir = ensure_dir(
            os.path.join(get_project_root(), "reports", "figures")
        )
        self.eval_cfg = self.cfg.get("evaluation", {})

        # ========== macOS 专属：添加中文字体配置（核心修复） ==========
        # 全局配置 matplotlib 字体，优先使用 macOS 原生中文字体
        plt.rcParams["font.family"] = ["sans-serif"]
        # 优先选苹方（PingFang SC），兜底选黑体（Heiti SC）
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
        # 解决负号（-）显示为方块的问题
        plt.rcParams["axes.unicode_minus"] = False

    # ──────────────────────────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────────────────────────
    def _save(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self.fig_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"图像已保存: {path}")

    @staticmethod
    def _color_map(labels: np.ndarray):
        """0→蓝（正常），1→红（洗钱）。"""
        colors = np.where(labels == 1, "crimson", "steelblue")
        return colors

    # ──────────────────────────────────────────────────────────────
    # 1. PCA 降维图
    # ──────────────────────────────────────────────────────────────
    def plot_pca(
        self,
        embeddings: np.ndarray,
        labels:     np.ndarray,
        title:      str = "PCA Node Embedding Dimensionality Reduction",
    ) -> None:
        """
        对节点嵌入做 PCA 降至 2D 并绘制散点图。

        Args:
            embeddings: [N, D] 节点嵌入
            labels:     [N] 节点标签（0/1）
            title:      图标题
        """
        n_comp = self.eval_cfg.get("pca_n_components", 2)
        pca    = PCA(n_components=n_comp, random_state=42)
        z      = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = self._color_map(labels)

        # 分组绘制以显示图例
        for cls, color, name in [(0, "steelblue", "Normal"), (1, "crimson", "Money Laundering")]:
            mask = labels == cls
            ax.scatter(z[mask, 0], z[mask, 1],
                       c=color, s=6, alpha=0.5, label=name, rasterized=True)

        explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        ax.set_title(title)
        ax.legend(markerscale=3)
        fig.tight_layout()
        self._save(fig, "pca_embedding")

    # ──────────────────────────────────────────────────────────────
    # 2. UMAP 降维图
    # ──────────────────────────────────────────────────────────────
    def plot_umap(
        self,
        embeddings: np.ndarray,
        labels:     np.ndarray,
        title:      str = "UMAP Node Embedding Dimensionality Reduction",
    ) -> None:
        """
        对节点嵌入做 UMAP 降至 2D 并绘制散点图。
        """
        try:
            import umap
        except ImportError:
            self.logger.warning("umap-learn not installed, skipping UMAP visualization")
            return

        n_neighbors = self.eval_cfg.get("umap_n_neighbors", 15)
        min_dist    = self.eval_cfg.get("umap_min_dist", 0.1)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            random_state=42, n_jobs=1
        )
        z = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls, color, name in [(0, "steelblue", "Normal"), (1, "crimson", "Money Laundering")]:
            mask = labels == cls
            ax.scatter(z[mask, 0], z[mask, 1],
                       c=color, s=6, alpha=0.5, label=name, rasterized=True)

        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(title)
        ax.legend(markerscale=3)
        fig.tight_layout()
        self._save(fig, "umap_embedding")

    # ──────────────────────────────────────────────────────────────
    # 3. 预测误差分布直方图
    # ──────────────────────────────────────────────────────────────
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """
        分别绘制正确预测节点与错误预测节点的洗钱概率分布直方图。

        Args:
            y_true: 真实标签
            y_prob: 洗钱类预测概率
        """
        y_pred   = (y_prob >= 0.5).astype(int)
        correct  = y_true == y_pred
        wrong    = ~correct

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, mask, title, color in [
            (axes[0], correct, "Correctly Predicted Nodes (Money Laundering Probability)", "steelblue"),
            (axes[1], wrong, "Incorrectly Predicted Nodes (Money Laundering Probability)", "crimson"),
        ]:
            ax.hist(y_prob[mask], bins=50, color=color, edgecolor="white", alpha=0.8)
            ax.set_xlabel("Money Laundering Prediction Probability")
            ax.set_ylabel("Number of Nodes")
            ax.set_title(f"{title}\n(n={mask.sum()})")
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold=0.5")
            ax.legend()

        fig.suptitle("Prediction Error Distribution Analysis", fontsize=14)
        fig.tight_layout()
        self._save(fig, "error_distribution")

    # ──────────────────────────────────────────────────────────────
    # 4. ROC 曲线
    # ──────────────────────────────────────────────────────────────
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """绘制 ROC 曲线。"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="crimson", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("ROC Curve - Money Laundering Detection")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "roc_curve")

    # ──────────────────────────────────────────────────────────────
    # 5. PR 曲线
    # ──────────────────────────────────────────────────────────────
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """绘制 Precision-Recall 曲线（更适合不平衡样本）。"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="darkorange", lw=2,
                label=f"PR Curve (AP = {pr_auc:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve - Money Laundering Detection")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "pr_curve")

    # ──────────────────────────────────────────────────────────────
    # 6. 混淆矩阵热力图
    # ──────────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        绘制带数值标注的混淆矩阵热力图。

        Args:
            cm: shape [2, 2] 的混淆矩阵
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)

        tick_labels = ["Normal (0)", "Money Laundering (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        # 在格子里标注数值
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        fig.tight_layout()
        self._save(fig, "confusion_matrix")

    # ──────────────────────────────────────────────────────────────
    # 7. 训练曲线
    # ──────────────────────────────────────────────────────────────
    def plot_training_history(self, history: Dict) -> None:
        """
        绘制训练集/验证集的损失曲线与 F1 曲线。

        Args:
            history: trainer.train() 返回的 history 字典
        """
        epochs    = range(1, len(history["train_loss"]) + 1)
        val_f1    = [m["f1"] for m in history["val_metrics"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        axes[0].plot(epochs, history["train_loss"], label="Training Loss", color="steelblue")
        axes[0].plot(epochs, history["val_loss"], label="Validation Loss", color="crimson")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training/Validation Loss Curves")
        axes[0].legend()

        # F1 曲线
        axes[1].plot(epochs, val_f1, label="Validation F1", color="darkorange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Validation Set F1 Curve")
        axes[1].legend()

        fig.tight_layout()
        self._save(fig, "training_history")

        fig.tight_layout()
        self._save(fig, "training_history")

    # ──────────────────────────────────────────────────────────────
    # 一键绘制全部图
    # ──────────────────────────────────────────────────────────────
    def plot_all(
        self,
        embeddings:  np.ndarray,
        labels:      np.ndarray,
        eval_result: Dict,
        history:     Optional[Dict] = None,
    ) -> None:
        """
        统一入口：绘制所有分析图。

        Args:
            embeddings:  节点嵌入向量 [N, D]
            labels:      节点真实标签 [N]
            eval_result: ModelEvaluator.evaluate() 的返回字典
            history:     训练历史（可选）
        """
        self.logger.info("开始生成所有可视化图")

        self.plot_pca(embeddings, labels)
        self.plot_umap(embeddings, labels)
        self.plot_error_distribution(
            eval_result["y_true"], eval_result["y_prob"]
        )
        self.plot_roc_curve(eval_result["y_true"], eval_result["y_prob"])
        self.plot_pr_curve(eval_result["y_true"], eval_result["y_prob"])
        self.plot_confusion_matrix(eval_result["confusion_matrix"])

        if history:
            self.plot_training_history(history)

        self.logger.info(f"所有图像已保存至: {self.fig_dir}")