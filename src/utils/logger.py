"""
日志记录模块
============
提供统一的日志配置，同时输出到控制台和文件。
封装 TrainingLogger 用于训练循环中的结构化日志记录。
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict

from .config import get_project_root, ensure_dir

# 日志格式
_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = os.path.join(get_project_root(), "logs")

# 模块级缓存，避免重复添加 Handler
_registry: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    创建并注册一个 Logger。

    Args:
        name:     Logger 名称（建议与模块同名）
        log_file: 日志文件名；None 时自动生成带时间戳的名称
        level:    日志级别
        console:  是否同时输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # ── 文件 Handler ──────────────────────────────────────────────
    ensure_dir(LOG_DIR)
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{ts}.log"
    fh = logging.FileHandler(os.path.join(LOG_DIR, log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # ── 控制台 Handler ────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    _registry[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已注册的 Logger；未注册则自动创建默认配置的实例。"""
    return _registry.get(name) or setup_logger(name)


# ─────────────────────────────────────────────────────────────────
class TrainingLogger:
    """
    训练过程专用日志记录器。
    封装 epoch 指标输出、最佳模型记录、早停通知等常用操作。
    """

    def __init__(self, name: str = "trainer"):
        self.logger = setup_logger(name)
        self.best_metric: float = 0.0
        self.best_epoch:  int   = 0

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict,
    ) -> None:
        """记录单个 epoch 的训练与验证信息。"""
        m_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss:   {val_loss:.6f} | {m_str}"
        )

    def log_best(self, epoch: int, metric: str, value: float) -> None:
        """记录新的最佳模型。"""
        self.best_metric = value
        self.best_epoch  = epoch
        self.logger.info(f"🏆 Best model at Epoch {epoch:04d} | {metric}: {value:.4f}")

    def log_early_stop(self, epoch: int, patience: int) -> None:
        """记录早停触发信息。"""
        self.logger.info(
            f"⏹  Early stopping at Epoch {epoch} "
            f"(patience={patience}, best_epoch={self.best_epoch}, "
            f"best_metric={self.best_metric:.4f})"
        )

    def log_done(self, total_epochs: int) -> None:
        """记录训练完成信息。"""
        self.logger.info(
            f"✅ Training complete — {total_epochs} epochs | "
            f"Best Epoch: {self.best_epoch} | "
            f"Best Metric: {self.best_metric:.4f}"
        )