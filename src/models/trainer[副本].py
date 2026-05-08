"""
模型训练模块
============
实现完整的 GNN 训练流水线，包含：
  - 模型构建（GCN / GAT 可切换）
  - 类别权重损失函数（解决样本不平衡）
  - 学习率调度器
  - 早停机制：不等到完全过拟合再停止
  - 最佳模型保存（基于验证集 F1）
  - 断点续训支持
  - 训练日志记录
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple

from sklearn.metrics import f1_score

from src.models.gcn     import GCNClassifier
from src.models.gat     import GATClassifier
from src.utils.config   import get_model_config, ensure_dir, get_project_root
from src.utils.logger   import TrainingLogger


class GNNTrainer:
    """
    图神经网络训练器。

    用法：
        trainer = GNNTrainer(data, class_weights)
        model, history = trainer.train()
    """

    def __init__(
        self,
        data:          Data,
        class_weights: Dict[int, float],
        model_config:  Optional[dict] = None,
    ):
        """
        Args:
            data:          PyG Data 对象（含 train/val/test mask）
            class_weights: 类别权重字典 {0: w0, 1: w1}
            model_config:  模型配置；None 时自动加载
        """
        self.cfg           = model_config or get_model_config()
        self.data          = data
        self.class_weights = class_weights
        self.tlogger       = TrainingLogger("trainer")
        self.logger        = self.tlogger.logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"训练设备: {self.device}")

        # 将图数据移至目标设备
        self.data = self.data.to(self.device)

        # 构建模型
        self.model    = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()

        # 检查点目录
        ckpt_dir = os.path.join(
            get_project_root(),
            self.cfg["checkpoint"]["save_dir"]
        )
        self.ckpt_dir  = ensure_dir(ckpt_dir)
        self.best_path = os.path.join(self.ckpt_dir, "best_model.pt")

    # ──────────────────────────────────────────────────────────────
    # 构建辅助对象
    # ──────────────────────────────────────────────────────────────
    def _build_model(self) -> nn.Module:
        """根据配置实例化 GCN 或 GAT 模型。"""
        model_type = self.cfg["training"]["model_type"].lower()
        in_ch      = self.data.num_node_features
        n_cls      = int(self.data.num_classes)

        if model_type == "gcn":
            c = self.cfg["gcn"]
            model = GCNClassifier(
                in_channels=in_ch,
                hidden_dim=c["hidden_dim"],
                num_classes=n_cls,
                num_layers=c["num_layers"],
                dropout=c["dropout"],
            )
        elif model_type == "gat":
            c = self.cfg["gat"]
            model = GATClassifier(
                in_channels=in_ch,
                hidden_dim=c["hidden_dim"],
                num_classes=n_cls,
                num_layers=c["num_layers"],
                num_heads=c["num_heads"],
                dropout=c["dropout"],
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"模型: {model_type.upper()} | 可训练参数: {n_params:,}")
        return model

    def _build_optimizer(self) -> optim.Optimizer:
        tr = self.cfg["training"]
        return optim.Adam(
            self.model.parameters(),
            lr=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
        )

    def _build_scheduler(self) -> Optional[ReduceLROnPlateau]:
        sc = self.cfg["training"]["scheduler"]
        if not sc["enabled"]:
            return None
        return ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=sc["factor"],
            patience=sc["patience"],
            min_lr=sc["min_lr"],
        )

    def _build_criterion(self) -> nn.CrossEntropyLoss:
        """构建带类别权重的交叉熵损失。"""
        if self.cfg["loss"]["use_class_weight"]:
            w = torch.tensor(
                [self.class_weights.get(i, 1.0) for i in range(2)],
                dtype=torch.float,
                device=self.device,
            )
            self.logger.info(f"损失函数类别权重: {w.tolist()}")
            return nn.CrossEntropyLoss(weight=w)
        return nn.CrossEntropyLoss()

    # ──────────────────────────────────────────────────────────────
    # 单轮训练 / 验证
    # ──────────────────────────────────────────────────────────────
    def _train_epoch(self) -> float:
        """执行一轮训练，返回训练集损失。"""
        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self.model(self.data.x, self.data.edge_index)
        mask = self.data.train_mask
        loss = self.criterion(logits[mask], self.data.y[mask])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _eval(self, mask: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        在指定掩码节点上评估模型。

        Returns:
            (loss, metrics_dict)
        """
        self.model.eval()
        logits, _ = self.model(self.data.x, self.data.edge_index)
        loss   = self.criterion(logits[mask], self.data.y[mask]).item()
        preds  = logits[mask].argmax(dim=-1).cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()

        f1  = f1_score(labels, preds, average="binary", zero_division=0)
        acc = (preds == labels).mean()

        return loss, {"accuracy": float(acc), "f1": float(f1)}

    # ──────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────
    def train(self, resume: bool = False) -> Tuple[nn.Module, Dict]:
        """
        执行完整训练循环，支持断点续训。

        Args:
            resume: True 时从 best_model.pt 恢复并继续训练

        Returns:
            (best_model, history)
            history: dict with keys 'train_loss', 'val_loss', 'val_metrics'
        """
        cfg     = self.cfg["training"]
        epochs  = cfg["epochs"]
        patience = cfg["patience"]
        monitor  = self.cfg["checkpoint"]["monitor_metric"]

        if resume and os.path.exists(self.best_path):
            ckpt = torch.load(self.best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_metric = ckpt.get("best_metric", 0.0)
            self.logger.info(f"断点续训，从 Epoch {start_epoch} 继续")
        else:
            start_epoch = 1
            best_metric = 0.0

        history       = {"train_loss": [], "val_loss": [], "val_metrics": [], "test_loss": [], "test_metrics": []}
        patience_cnt  = 0

        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info("=" * 60)

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._eval(self.data.val_mask)
            test_loss, test_metrics = self._eval(self.data.test_mask)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            history["test_loss"].append(test_loss)
            history["test_metrics"].append(test_metrics)

            self.tlogger.log_epoch(epoch, train_loss, val_loss, val_metrics)
            self.logger.info(
                f"Epoch {epoch:03d} | Test Loss: {test_loss:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}")

            # 学习率调度
            """
ReduceLROnPlateau策略：
监控验证集F1分数
当F1连续几轮不提升时，将学习率乘以0.5
最小学习率限制为1e-5，避免学习过慢"""
            if self.scheduler:
                self.scheduler.step(val_metrics[monitor])

            # 最佳模型保存
            curr_metric = val_metrics[monitor]
            if curr_metric > best_metric:
                best_metric = curr_metric
                patience_cnt = 0
                self.tlogger.log_best(epoch, monitor, best_metric)
                self._save_checkpoint(epoch, best_metric, history)
            else:
                patience_cnt += 1

            # 早停检测
            if patience_cnt >= patience:
                self.tlogger.log_early_stop(epoch, patience)
                break

        self.tlogger.log_done(epoch)

        # 加载最佳模型权重
        best_ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(best_ckpt["model_state"])
        self.logger.info("已加载最佳模型权重")

        return self.model, history

    # ──────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, best_metric: float, history: Dict) -> None:
        """保存当前最佳模型检查点。"""
        torch.save(
            {
                "epoch":        epoch,
                "model_state":  self.model.state_dict(),
                "optim_state":  self.optimizer.state_dict(),
                "best_metric":  best_metric,
                "model_type":   self.cfg["training"]["model_type"],
                "history": history,
            },
            self.best_path,
        )

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        """
        获取全图节点嵌入向量（用于后续可视化与团伙识别）。

        Returns:
            节点嵌入张量 [N, D]，已移至 CPU
        """
        self.model.eval()
        _, embeddings = self.model(self.data.x, self.data.edge_index)
        return embeddings.cpu()


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from src.preprocessing.clean       import DataCleaner
    from src.preprocessing.transform   import FeatureTransformer
    from src.preprocessing.graph_builder import GraphBuilder

    DataCleaner().clean()
    result = FeatureTransformer().transform()
    data   = GraphBuilder().build(result)

    trainer = GNNTrainer(data, result["class_weights"])
    model, history = trainer.train()
    print(f"训练完成，最终 val_f1: {history['val_metrics'][-1]['f1']:.4f}")