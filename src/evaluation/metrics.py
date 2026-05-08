"""
模型评估指标计算模块（完整优化版）
====================================
修复：
  [Fix-1] 接收外部传入的 threshold（训练器搜索到的最佳阈值）
  [Fix-2] 完整打印 confusion matrix（带 TN/FP/FN/TP 标注）
  [Fix-3] 完整打印 classification_report
  [Fix-4] 评估报告保存为 JSON 文件
  [Fix-5] 兼容原项目 evaluate(split="test") 调用方式
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
    GNN 模型评估器（完整优化版）。

    用法：
        evaluator = ModelEvaluator(model, data, threshold=best_thr)
        report    = evaluator.evaluate(split="test")
    """

    def __init__(
        self,
        model:        torch.nn.Module,
        data:         Data,
        model_config: Optional[dict] = None,
        threshold:    float          = 0.5,
    ):
        self.model      = model
        self.data       = data
        self.cfg        = model_config or get_model_config()
        self.logger     = get_logger("evaluator")
        self.device     = next(model.parameters()).device
        self.threshold  = threshold
        self.data       = self.data.to(self.device)
        self.report_dir = ensure_dir(
            os.path.join(get_project_root(), "reports")
        )

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _predict(self, mask: torch.Tensor) -> Dict[str, np.ndarray]:
        """在给定 mask 上推断，返回标签、预测值及概率。"""
        self.model.eval()
        logits, embeddings = self.model(
            self.data.x.to(self.device),
            self.data.edge_index.to(self.device),
        )
        proba  = torch.softmax(logits, dim=-1)
        y_prob = proba[mask, 1].cpu().numpy()
        # [Fix-1] 使用最佳阈值，而非固定 argmax
        y_pred = (y_prob >= self.threshold).astype(int)
        y_true = self.data.y[mask].cpu().numpy()
        y_emb  = embeddings[mask].cpu().numpy()
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_emb":  y_emb,
        }

    # ──────────────────────────────────────────────────────────────
    def evaluate(self, split: str = "test") -> Dict:
        """
        在指定数据集上完整评估模型。

        Args:
            split: "train" | "val" | "test"

        Returns:
            包含所有指标、confusion_matrix、classification_report 的字典
        """
        mask_map = {
            "train": self.data.train_mask,
            "val":   self.data.val_mask,
            "test":  self.data.test_mask,
        }
        if split not in mask_map:
            raise ValueError(f"split 须为 train/val/test，收到: {split}")

        mask = mask_map[split].to(self.device)
        pred = self._predict(mask)

        y_true = pred["y_true"]
        y_pred = pred["y_pred"]
        y_prob = pred["y_prob"]

        # ── 核心指标 ──────────────────────────────────────────────
        acc       = accuracy_score(y_true, y_pred)
        prec_ill  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_ill   = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_ill    = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_macro  = f1_score(y_true, y_pred, average="macro",     zero_division=0)
        f1_weight = f1_score(y_true, y_pred, average="weighted",  zero_division=0)

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = float("nan")

        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except ValueError:
            auc_pr = float("nan")

        # ── Confusion Matrix ──────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # ── Classification Report ─────────────────────────────────
        cls_report = classification_report(
            y_true, y_pred,
            target_names=["licit (0)", "illicit (1)"],
            digits=4,
            zero_division=0,
        )

        # ── 控制台打印 ────────────────────────────────────────────
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  📊  {split.upper()} SET EVALUATION RESULTS")
        print(f"  模型: {self.cfg['training']['model_type'].upper()} | "
              f"阈值: {self.threshold:.3f}")
        print(sep)

        # Confusion Matrix
        print(f"\n📌 Confusion Matrix:")
        print(f"  {'':20s}  Pred licit(0)  Pred illicit(1)")
        print(f"  {'Actual licit(0)':20s}  {tn:>13}  {fp:>15}")
        print(f"  {'Actual illicit(1)':20s}  {fn:>13}  {tp:>15}")
        print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

        # 核心指标
        print(f"\n📌 Core Metrics (illicit class, pos_label=1):")
        print(f"  Precision  (illicit) : {prec_ill:.4f}")
        print(f"  Recall     (illicit) : {rec_ill:.4f}")
        print(f"  F1-score   (illicit) : {f1_ill:.4f}")
        print(f"  F1-macro             : {f1_macro:.4f}")
        print(f"  F1-weighted          : {f1_weight:.4f}")
        print(f"  Accuracy             : {acc:.4f}")
        print(f"  ROC-AUC              : {auc_roc:.4f}")
        print(f"  PR-AUC               : {auc_pr:.4f}")

        # Classification Report
        print(f"\n📌 Classification Report:")
        print(cls_report)
        print(sep)

        # ── 日志 ──────────────────────────────────────────────────
        self.logger.info(
            f"[{split.upper()}] thr={self.threshold:.3f} | "
            f"F1={f1_ill:.4f} | Recall={rec_ill:.4f} | "
            f"Precision={prec_ill:.4f} | AUC={auc_roc:.4f}"
        )

        # ── 保存 JSON 报告 ────────────────────────────────────────
        report = {
            "split":                 split,
            "model_type":            self.cfg["training"]["model_type"],
            "threshold":             float(self.threshold),
            "accuracy":              float(acc),
            "precision_illicit":     float(prec_ill),
            "recall_illicit":        float(rec_ill),
            "f1_illicit":            float(f1_ill),
            "f1_macro":              float(f1_macro),
            "f1_weighted":           float(f1_weight),
            "roc_auc":               float(auc_roc),
            "pr_auc":                float(auc_pr),
            "confusion_matrix":      cm.tolist(),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "classification_report": cls_report,
            # 兼容原项目 threshold_analysis 使用
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
        }

        report_path = os.path.join(
            self.report_dir, "evaluation_report.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"评估报告已保存: {report_path}")

        # 保存嵌入
        emb_dir  = ensure_dir(
            os.path.join(get_project_root(), "data", "embeddings")
        )
        np.save(os.path.join(emb_dir, f"{split}_embeddings.npy"), pred["y_emb"])

        return report
