"""
配置文件加载模块
================
统一加载 YAML 配置，提供绝对路径解析与目录自动创建功能。
所有其他模块通过此模块获取配置，避免硬编码路径或参数。
"""

import os
import yaml
from typing import Any, Dict, Optional

# 项目根目录（从当前文件向上推导两级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_MODEL_CONFIG = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
DEFAULT_DATA_CONFIG  = os.path.join(PROJECT_ROOT, "configs", "data_config.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    """
    加载并解析 YAML 文件。

    Args:
        path: YAML 文件路径（绝对或相对）

    Returns:
        解析后的配置字典

    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError:    YAML 格式错误
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_data_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载数据配置，自动将相对路径转换为基于项目根目录的绝对路径。

    Args:
        path: 自定义配置文件路径；None 时使用默认路径

    Returns:
        数据配置字典（路径已转为绝对路径）
    """
    cfg = load_yaml(path or DEFAULT_DATA_CONFIG)
    # 将 paths 下的目录路径转为绝对路径
    for key in ("raw_dir", "processed_dir", "embeddings_dir"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, cfg["paths"][key])
    return cfg


def get_model_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载模型配置。

    Args:
        path: 自定义配置文件路径；None 时使用默认路径

    Returns:
        模型配置字典
    """
    return load_yaml(path or DEFAULT_MODEL_CONFIG)


def ensure_dir(dir_path: str) -> str:
    """
    确保目录存在，不存在则递归创建。

    Args:
        dir_path: 目标目录路径

    Returns:
        目录的绝对路径
    """
    abs_path = os.path.abspath(dir_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_project_root() -> str:
    """返回项目根目录绝对路径。"""
    return PROJECT_ROOT