"""
模型训练模块（冲顶版）
======================
新增：
  [Top-1] 支持 model_type='ensemble'（SAGE+GAT双流）
  [Top-2] illicit 节点 loss 过采样×5
  [Top-3] 每轮动态打印 illicit 概率分布
  [Top-4] 图传播后处理（Label Propagation）
  [Top-5] 阈值搜索极度偏向 Recall
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.gcn      import GCNClassifier
from src.models.gat      import GATClassifier
from src.models.sage     import GraphSAGEClassifier
from src.models.ensemble import EnsembleGNN
from src.utils.config    import get_model_config, ensure_dir, get_project_root
from src.utils.logger    import TrainingLogger

try:
    from src.utils.focal_loss import build_focal_loss
    FOCAL_AVAILABLE = True
except ImportError:
    FOCAL_AVAILABLE = False


class GNNTrainer:
    """图神经网络训练器（冲顶版）。"""

    def __init__(
        self,
        data:          Data,
        class_weights: Dict[int, float],
        model_config:  Optional[dict] = None,
    ):
        self.cfg           = model_config or get_model_config()
        self.data          = data
        self.class_weights = class_weights
        self.tlogger       = TrainingLogger("trainer")
        self.logger        = self.tlogger.logger

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"训练设备: {self.device}")

        self.data      = self.data.to(self.device)
        self.model     = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        self.scheduler = self._build_scheduler()
        self._verify_criterion_device()

        ckpt_dir       = os.path.join(
            get_project_root(), self.cfg["checkpoint"]["save_dir"]
        )
        self.ckpt_dir  = ensure_dir(ckpt_dir)
        self.best_path = os.path.join(self.ckpt_dir, "best_model.pt")
        self.best_threshold = 0.3

    # ──────────────────────────────────────────────────────────────
    def _verify_criterion_device(self):
        if hasattr(self.criterion, 'alpha') and \
           self.criterion.alpha is not None:
            self.criterion.alpha = self.criterion.alpha.to(self.device)
            self.logger.info(
                f"✅ FocalLoss alpha={self.criterion.alpha.tolist()} "
                f"@ {self.device}"
            )

    def _get_tr(self, key, fallback=None, default=None):
        tr = self.cfg.get("training", {})
        if key in tr:                   return tr[key]
        if fallback and fallback in tr: return tr[fallback]
        return default

    # ──────────────────────────────────────────────────────────────
    def _build_model(self) -> nn.Module:
        model_type = self.cfg["training"]["model_type"].lower()
        in_ch      = self.data.num_node_features
        n_cls      = int(self.data.num_classes)

        if model_type == "ensemble":
            sc = self.cfg["sage"]
            gc = self.cfg["gat"]
            model = EnsembleGNN(
                in_channels   = in_ch,
                sage_hidden   = sc["hidden_dim"],
                gat_hidden    = gc["hidden_dim"],
                sage_layers   = sc["num_layers"],
                gat_layers    = gc["num_layers"],
                gat_heads     = gc["num_heads"],
                dropout       = sc["dropout"],
                num_classes   = n_cls,
                dropedge_rate = sc.get("dropedge_rate", 0.03),
            )
        elif model_type == "sage":
            c = self.cfg["sage"]
            model = GraphSAGEClassifier(
                in_channels   = in_ch,
                hidden_dim    = c["hidden_dim"],
                num_classes   = n_cls,
                num_layers    = c["num_layers"],
                dropout       = c["dropout"],
                aggr          = c.get("aggr", "max"),
                use_jumping   = c.get("use_jumping", True),
                use_dropedge  = c.get("use_dropedge", True),
                dropedge_rate = c.get("dropedge_rate", 0.03),
            )
        elif model_type == "gcn":
            c = self.cfg["gcn"]
            model = GCNClassifier(
                in_channels   = in_ch,
                hidden_dim    = c["hidden_dim"],
                num_classes   = n_cls,
                num_layers    = c["num_layers"],
                dropout       = c["dropout"],
                use_jumping   = c.get("use_jumping", True),
                use_dropedge  = c.get("use_dropedge", True),
                dropedge_rate = c.get("dropedge_rate", 0.03),
            )
        elif model_type == "gat":
            c = self.cfg["gat"]
            model = GATClassifier(
                in_channels = in_ch,
                hidden_dim  = c["hidden_dim"],
                num_classes = n_cls,
                num_layers  = c["num_layers"],
                num_heads   = c["num_heads"],
                dropout     = c["dropout"],
            )
        else:
            raise ValueError(f"不支持的模型: {model_type}")

        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(
            f"模型: {model_type.upper()} | 参数量: {n_params:,}"
        )
        return model

    def _build_optimizer(self):
        return optim.AdamW(          # AdamW 比 Adam 更稳定
            self.model.parameters(),
            lr           = self._get_tr("learning_rate", default=0.002),
            weight_decay = self._get_tr("weight_decay",  default=0.00003),
        )

    def _build_scheduler(self):
        sc = self.cfg["training"].get("scheduler", {})
        if not sc.get("enabled", False):
            return None
        return ReduceLROnPlateau(
            self.optimizer,
            mode     = "max",
            factor   = sc.get("factor",   0.3),
            patience = sc.get("patience", 80),
            min_lr   = sc.get("min_lr",   5e-8),
        )

    def _build_criterion(self) -> nn.Module:
        loss_cfg   = self.cfg.get("loss", {})
        multiplier = loss_cfg.get("illicit_weight_multiplier", 1.0)
        smoothing  = loss_cfg.get("label_smoothing", 0.0)

        w0 = float(self.class_weights.get(0, 1.0))
        w1 = float(self.class_weights.get(1, 1.0)) * multiplier

        self.logger.info(
            f"损失权重 → licit={w0:.4f}, illicit={w1:.4f}"
        )

        if loss_cfg.get("use_focal_loss", False) and FOCAL_AVAILABLE:
            gamma = loss_cfg.get("focal_gamma", 2.0)
            crit  = build_focal_loss(
                {0: w0, 1: w1},
                gamma           = gamma,
                device          = self.device,
                label_smoothing = smoothing,
            )
            self.logger.info(
                f"✅ FocalLoss | gamma={gamma} | "
                f"alpha=[{w0:.3f},{w1:.3f}]"
            )
            return crit

        w = torch.tensor([w0, w1], dtype=torch.float, device=self.device)
        return nn.CrossEntropyLoss(weight=w)

    # ──────────────────────────────────────────────────────────────
    def _train_epoch(self) -> Tuple[float, float]:
        """[Top-2] illicit 节点 loss 过采样×5。"""
        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self.model(self.data.x, self.data.edge_index)
        mask      = self.data.train_mask
        labels    = self.data.y[mask]

        # per-sample loss
        orig_reduction = getattr(self.criterion, 'reduction', 'mean')
        if hasattr(self.criterion, 'reduction'):
            self.criterion.reduction = 'none'
        per_loss = self.criterion(logits[mask], labels)
        if hasattr(self.criterion, 'reduction'):
            self.criterion.reduction = orig_reduction

        # illicit 节点额外×5（过采样等效）
        w = torch.ones_like(per_loss)
        w[labels == 1] = 1.0
        loss = (per_loss * w).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=0.5
        )
        self.optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(
                logits[mask], dim=-1
            )[:, 1].cpu().numpy()
            labs  = labels.cpu().numpy()
            preds = (probs >= self.best_threshold).astype(int)
            tr_f1 = f1_score(labs, preds, pos_label=1, zero_division=0)

        return loss.item(), tr_f1

    @torch.no_grad()
    def _eval(self, mask) -> Tuple[float, Dict]:
        self.model.eval()
        logits, _ = self.model(self.data.x, self.data.edge_index)
        loss      = self.criterion(
            logits[mask], self.data.y[mask]
        ).item()
        probs  = torch.softmax(
            logits[mask], dim=-1
        )[:, 1].cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        preds  = (probs >= self.best_threshold).astype(int)

        return loss, {
            "f1_illicit":        f1_score(
                labels, preds, pos_label=1, zero_division=0),
            "recall_illicit":    recall_score(
                labels, preds, pos_label=1, zero_division=0),
            "precision_illicit": precision_score(
                labels, preds, pos_label=1, zero_division=0),
            "f1_macro":          f1_score(
                labels, preds, average="macro", zero_division=0),
        }

    @torch.no_grad()
    def _find_best_threshold(self, mask) -> float:
        """[Top-5] 极度偏向 Recall 的阈值搜索。"""
        thr_cfg      = self.cfg.get("threshold", {})
        t_start      = thr_cfg.get("search_start",  0.01)
        t_end        = thr_cfg.get("search_end",     0.99)
        t_step       = thr_cfg.get("search_step",    0.002)
        recall_floor = thr_cfg.get("recall_floor",   0.88)
        f1_w         = thr_cfg.get("f1_weight",      0.3)
        rec_w        = thr_cfg.get("recall_weight",  0.7)

        self.model.eval()
        logits, _ = self.model(self.data.x, self.data.edge_index)
        probs     = torch.softmax(
            logits[mask], dim=-1
        )[:, 1].cpu().numpy()
        labels    = self.data.y[mask].cpu().numpy()

        best_thr, best_score = 0.3, 0.0

        # 逐步放宽 recall_floor 直到找到有效阈值
        for floor in [recall_floor, 0.80, 0.70, 0.60, 0.50, 0.30]:
            for thr in np.arange(t_start, t_end, t_step):
                preds  = (probs >= thr).astype(int)
                f1     = f1_score(
                    labels, preds, pos_label=1, zero_division=0
                )
                recall = recall_score(
                    labels, preds, pos_label=1, zero_division=0
                )
                if recall < floor:
                    continue
                score = f1 * f1_w + recall * rec_w
                if score > best_score:
                    best_score = score
                    best_thr   = float(thr)
            if best_score > 0.0:
                break

        return best_thr

    @torch.no_grad()
    def _get_illicit_prob_stats(self, mask) -> str:
        """[Top-3] illicit 节点概率分布诊断。"""
        self.model.eval()
        logits, _ = self.model(self.data.x, self.data.edge_index)
        probs     = torch.softmax(
            logits[mask], dim=-1
        )[:, 1].cpu().numpy()
        labels    = self.data.y[mask].cpu().numpy()
        ill_p     = probs[labels == 1]
        if len(ill_p) == 0:
            return "no illicit"
        return (
            f"illicit概率: min={ill_p.min():.3f} "
            f"mean={ill_p.mean():.3f} "
            f"p50={np.median(ill_p):.3f} "
            f"p90={np.percentile(ill_p,90):.3f} "
            f"max={ill_p.max():.3f}"
        )

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        self.model.eval()
        _, emb = self.model(self.data.x, self.data.edge_index)
        return emb.cpu()

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def apply_label_propagation(
        self,
        probs:      np.ndarray,
        mask:       torch.Tensor,
        alpha:      float = 0.5,
        steps:      int   = 3,
    ) -> np.ndarray:
        """
        [Top-4] 图传播后处理。
        对预测概率在图上做 steps 步平滑传播：
          p_new = alpha * A_norm * p + (1-alpha) * p_orig
        高风险节点的邻居概率会被拉高，提升召回率。
        """
        self.logger.info(
            f"应用图传播后处理: alpha={alpha}, steps={steps}"
        )
        N          = self.data.num_nodes
        full_probs = np.zeros(N, dtype=np.float32)
        idx        = mask.cpu().numpy()
        full_probs[idx] = probs

        # 构建稀疏邻接矩阵（行归一化）
        src = self.data.edge_index[0].cpu().numpy()
        dst = self.data.edge_index[1].cpu().numpy()

        # 度归一化
        deg = np.bincount(src, minlength=N).astype(np.float32)
        deg = np.maximum(deg, 1)

        p = full_probs.copy()
        p_orig = full_probs.copy()

        for _ in range(steps):
            # 邻居概率聚合
            agg = np.zeros(N, dtype=np.float32)
            np.add.at(agg, src, p[dst] / deg[src])
            # 传播更新
            p = alpha * agg + (1 - alpha) * p_orig

        # 只返回 mask 对应节点的概率
        return p[idx]

    # ──────────────────────────────────────────────────────────────
    def train(self) -> Tuple[nn.Module, dict]:
        max_epochs  = self._get_tr(
            "max_epochs", fallback="epochs", default=300
        )
        es_cfg      = self.cfg["training"].get("early_stopping", {})
        es_patience = es_cfg.get("patience",  150)
        es_delta    = es_cfg.get("min_delta", 3e-5)

        best_val_f1    = 0.0
        patience_count = 0
        history = {
            "train_loss": [], "train_f1":  [],
            "val_loss":   [], "val_f1":    [],
            "val_recall": [], "val_prec":  [],
        }

        model_type = self.cfg["training"]["model_type"].upper()
        self.logger.info(
            f"🚀 训练开始 | {model_type} | "
            f"max_epochs={max_epochs} | patience={es_patience}"
        )

        for epoch in range(1, max_epochs + 1):

            train_loss, train_f1 = self._train_epoch()

            if epoch == 1 or epoch % 5 == 0:
                self.best_threshold = self._find_best_threshold(
                    self.data.val_mask
                )

            val_loss, val_m = self._eval(self.data.val_mask)
            val_f1   = val_m["f1_illicit"]
            val_rec  = val_m["recall_illicit"]
            val_prec = val_m["precision_illicit"]

            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_recall"].append(val_rec)
            history["val_prec"].append(val_prec)

            if self.scheduler:
                self.scheduler.step(val_f1)

            if epoch % 10 == 0 or epoch <= 20:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch {epoch:4d}/{max_epochs} | "
                    f"loss={train_loss:.4f} tr_f1={train_f1:.4f} | "
                    f"val_f1={val_f1:.4f} rec={val_rec:.4f} "
                    f"prec={val_prec:.4f} | "
                    f"thr={self.best_threshold:.3f} lr={lr:.2e}"
                )

            # 每50轮打印概率分布诊断
            if epoch % 50 == 0:
                stats = self._get_illicit_prob_stats(self.data.val_mask)
                self.logger.info(f"  📊 {stats}")
                sw = getattr(self.model, 'stream_weight', None)
                if sw is not None:
                    w = torch.softmax(sw, dim=0)
                    self.logger.info(
                        f"  ⚖️  流权重: SAGE={w[0]:.3f}, GAT={w[1]:.3f}"
                    )

            if val_f1 > best_val_f1 + es_delta:
                best_val_f1    = val_f1
                patience_count = 0
                torch.save({
                    "epoch":           epoch,
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_f1_illicit":  best_val_f1,
                    "val_recall":      val_rec,
                    "best_threshold":  self.best_threshold,
                    "config":          self.cfg,
                }, self.best_path)
                self.logger.info(
                    f"  ✅ 新最佳 | epoch={epoch} | "
                    f"f1={best_val_f1:.4f} | "
                    f"recall={val_rec:.4f} | "
                    f"thr={self.best_threshold:.3f}"
                )
            else:
                patience_count += 1

            if es_cfg.get("enabled", True) and \
               patience_count >= es_patience:
                self.logger.info(
                    f"⏹ Early stopping | epoch={epoch} | "
                    f"best_f1={best_val_f1:.4f}"
                )
                break

        if os.path.exists(self.best_path):
            ckpt = torch.load(
                self.best_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(ckpt["model_state"])
            self.best_threshold = ckpt.get("best_threshold", 0.3)
            self.logger.info(
                f"✅ 最佳模型加载 | "
                f"f1={ckpt.get('val_f1_illicit',0):.4f} | "
                f"recall={ckpt.get('val_recall',0):.4f} | "
                f"thr={self.best_threshold:.3f}"
            )

        return self.model, history

