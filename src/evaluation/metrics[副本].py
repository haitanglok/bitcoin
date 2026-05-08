"""
模型评估指标计算模块
====================
基于测试集计算洗钱检测的核心量化指标：
  - 准确率（Accuracy）
  - 精确率（Precision）
  - 召回率（Recall）
  - F1-score（binary & macro）
  - ROC-AUC
  - PR-AUC（Average Precision，更适合不平衡场景）
  - 混淆矩阵

并生成格式化的评估报告文本文件。
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Optional
from torch_geometric.data import Data

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

from src.utils.config import get_model_config, get_project_root, ensure_dir
from src.utils.logger import get_logger


class ModelEvaluator:
    """
    GNN 模型评估器。

    用法：
        evaluator = ModelEvaluator(model, data)
        report = evaluator.evaluate()
    """

    def __init__(
        self,
        model:        torch.nn.Module,
        data:         Data,
        model_config: Optional[dict] = None,
    ):
        self.model   = model
        self.data    = data
        self.cfg     = model_config or get_model_config()
        self.logger  = get_logger("evaluator")
        self.device  = next(model.parameters()).device

        self.report_dir = ensure_dir(
            os.path.join(get_project_root(), "reports")
        )

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _predict(self, mask: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        在给定掩码节点上推断，返回标签、预测值及概率。

        Returns:
            dict with keys: y_true, y_pred, y_prob (洗钱类概率)
        """
        self.model.eval()
        logits, _ = self.model(
            self.data.x.to(self.device),
            self.data.edge_index.to(self.device)
        )
        proba  = torch.softmax(logits, dim=-1)
        y_prob = proba[mask, 1].cpu().numpy()   # 洗钱类（1）的概率
        y_pred = proba[mask].argmax(dim=-1).cpu().numpy()
        y_true = self.data.y[mask].cpu().numpy()
        return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    # ──────────────────────────────────────────────────────────────
    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        计算全套评估指标。

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 洗钱类预测概率

        Returns:
            指标字典
        """
        metrics = {
            "accuracy":          accuracy_score(y_true, y_pred),
            "precision":         precision_score(y_true, y_pred, zero_division=0),
            "recall":            recall_score(y_true, y_pred, zero_division=0),
            "f1_binary":         f1_score(y_true, y_pred, average="binary",
                                          zero_division=0),
            "f1_macro":          f1_score(y_true, y_pred, average="macro",
                                          zero_division=0),
            "roc_auc":           roc_auc_score(y_true, y_prob),
            "pr_auc":            average_precision_score(y_true, y_prob),
        }
        return metrics

    def compute_confusion(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """返回混淆矩阵。"""
        return confusion_matrix(y_true, y_pred)

    # ──────────────────────────────────────────────────────────────
    def evaluate(self, split: str = "test") -> Dict:
        """
        在指定数据集上完整评估模型。

        Args:
            split: "train" | "val" | "test"

        Returns:
            包含 metrics, confusion_matrix, classification_report 的字典
        """
        mask_map = {
            "train": self.data.train_mask,
            "val":   self.data.val_mask,
            "test":  self.data.test_mask,
        }
        if split not in mask_map:
            raise ValueError(f"split 须为 train/val/test，收到: {split}")

        mask   = mask_map[split].to(self.device)
        pred   = self._predict(mask)
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]
        y_prob = pred["y_prob"]

        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        cm      = self.compute_confusion(y_true, y_pred)
        report  = classification_report(
            y_true, y_pred,
            target_names=["正常(0)", "洗钱(1)"],
            zero_division=0,
        )

        # 日志输出
        self.logger.info(f"[{split}] 评估结果:")
        for k, v in metrics.items():
            self.logger.info(f"  {k:20s}: {v:.4f}")
        self.logger.info(f"\n分类报告:\n{report}")
        self.logger.info(f"混淆矩阵:\n{cm}")

        # 保存报告
        self._save_report(split, metrics, cm.tolist(), report)

        return {
            "metrics":              metrics,
            "confusion_matrix":     cm,
            "classification_report": report,
            "y_true":               y_true,
            "y_pred":               y_pred,
            "y_prob":               y_prob,
        }

    # ──────────────────────────────────────────────────────────────
    def _save_report(
        self,
        split:   str,
        metrics: Dict[str, float],
        cm:      list,
        report:  str,
    ) -> None:
        """将评估报告保存为 JSON 和 TXT 两种格式。"""
        base = os.path.join(self.report_dir, f"eval_{split}")

        # JSON
        with open(f"{base}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "confusion_matrix": cm}, f,
                      indent=2, ensure_ascii=False)
        # TXT
        with open(f"{base}_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        self.logger.info(f"评估报告已保存: {base}_metrics.json")


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from src.preprocessing.clean         import DataCleaner
    from src.preprocessing.transform     import FeatureTransformer
    from src.preprocessing.graph_builder import GraphBuilder
    from src.models.trainer              import GNNTrainer

    DataCleaner().clean()
    result  = FeatureTransformer().transform()
    data    = GraphBuilder().build(result)
    trainer = GNNTrainer(data, result["class_weights"])
    model, _ = trainer.train()

    evaluator = ModelEvaluator(model, data)
    report    = evaluator.evaluate(split="test")
    print(f"Test ROC-AUC: {report['metrics']['roc_auc']:.4f}")