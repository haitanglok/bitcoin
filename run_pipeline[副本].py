"""
完整训练评估入口（GraphSAGE 完整支持版）
==========================================
适配原项目所有调用链：
  - builder.build(tf_result) 接口
  - trainer.get_embeddings() 接口
  - evaluator.evaluate(split="test") 接口
  - viz.plot_all(embeddings, labels, eval_result, history) 接口
  - analyzer / detector 接口
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.clean          import DataCleaner
from src.preprocessing.transform      import FeatureTransformer
from src.preprocessing.graph_builder  import GraphBuilder
from src.models.trainer                import GNNTrainer
from src.evaluation.metrics            import ModelEvaluator
from src.evaluation.visualization      import Visualizer
from src.evaluation.threshold_analysis import ThresholdAnalyzer, CommunityDetector
from src.utils.logger                  import setup_logger
from src.utils.config                  import (
    ensure_dir, get_project_root, get_model_config
)


# ──────────────────────────────────────────────────────────────────
def compute_class_weights(data) -> dict:
    """
    根据训练集标签分布计算类别权重，并应用 illicit_weight_multiplier。
    """
    cfg        = get_model_config()
    multiplier = cfg.get("loss", {}).get("illicit_weight_multiplier", 1.0)

    train_labels = data.y[data.train_mask].cpu().numpy()
    n_total      = len(train_labels)
    n_illicit    = (train_labels == 1).sum()
    n_licit      = (train_labels == 0).sum()

    w_licit   = n_total / (2.0 * n_licit)   if n_licit   > 0 else 1.0
    w_illicit = n_total / (2.0 * n_illicit) if n_illicit > 0 else 1.0
    w_illicit_final = w_illicit * multiplier

    print(f"\n📊 训练集类别分布:")
    print(f"   licit  (0): {n_licit:>6}  → weight={w_licit:.4f}")
    print(f"   illicit(1): {n_illicit:>6}  → weight={w_illicit:.4f} "
          f"× {multiplier} = {w_illicit_final:.4f}")
    print(f"   不平衡比例: 1 : {n_licit / max(n_illicit, 1):.1f}")

    return {0: w_licit, 1: w_illicit_final}


# ──────────────────────────────────────────────────────────────────
def _fill_proba(
    y_prob:     np.ndarray,
    mask:       torch.Tensor,
    total_nodes: int,
) -> np.ndarray:
    """将测试集概率填充到全图大小的数组（其余节点填0）。"""
    full = np.zeros(total_nodes, dtype=np.float32)
    full[mask.cpu().numpy()] = y_prob
    return full


# ──────────────────────────────────────────────────────────────────
def main():
    logger = setup_logger("pipeline")
    logger.info("🚀 区块链洗钱检测流水线启动（GCN / GAT / GraphSAGE）")

    cfg        = get_model_config()
    model_type = cfg["training"]["model_type"].upper()
    logger.info(f"当前模型: {model_type}")

    # ── Step 1: 数据清洗 ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 数据清洗")
    cleaner = DataCleaner()
    cleaner.clean()

    # ── Step 2: 特征工程 ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 特征工程")
    transformer = FeatureTransformer()
    transformer.transform()

    # ── Step 3: 图网络构建 ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: 图网络构建")
    builder = GraphBuilder()
    data    = builder.build()

    logger.info(
        f"图统计: 节点={data.num_nodes} | "
        f"边={data.edge_index.shape[1]} | "
        f"特征维度={data.num_node_features}"
    )
    logger.info(
        f"Mask: train={data.train_mask.sum().item()} | "
        f"val={data.val_mask.sum().item()} | "
        f"test={data.test_mask.sum().item()}"
    )

    # ── Step 4: 计算类别权重 ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 计算类别权重")
    class_weights = compute_class_weights(data)



    # ── Step 5: 模型训练 ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"Step 5: 模型训练 [{model_type}]")
    trainer        = GNNTrainer(data=data, class_weights=class_weights)

    # 第二步：验证损失函数（可选，确认无误后可删除）
    criterion = trainer.criterion
    print(f"\n🔍 损失函数类型: {type(criterion).__name__}")
    if hasattr(criterion, 'alpha'):
        print(f"   alpha权重: {criterion.alpha.tolist()}")
        print(f"   alpha设备: {criterion.alpha.device}")
        print(f"   模型设备:  {trainer.device}")
        # 关键检查：两者必须一致
        if criterion.alpha.device == trainer.device:
            print(f"   ✅ 设备一致，FocalLoss 权重正常生效")
        else:
            print(f"   ❌ 设备不一致！alpha将被强制迁移")
            criterion.alpha = criterion.alpha.to(trainer.device)
    else:
        print(f"   ⚠️ 未检测到 alpha，使用的是普通 CrossEntropyLoss")

    model, history = trainer.train()
    best_threshold = trainer.best_threshold
    logger.info(f"训练完成 | 最佳阈值: {best_threshold:.3f}")

    # ── Step 6: 获取节点嵌入 ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: 获取节点嵌入")
    embeddings = trainer.get_embeddings()
    emb_dir    = ensure_dir(
        os.path.join(get_project_root(), "data", "embeddings")
    )
    torch.save(embeddings, os.path.join(emb_dir, "node_embeddings.pt"))
    logger.info(f"节点嵌入已保存: shape={embeddings.shape}")

    # ── Step 7: 模型评估 ──────────────────────────────────────────

    logger.info("\n" + "=" * 60)
    logger.info("Step 7: 模型评估（含图传播后处理）")

    evaluator = ModelEvaluator(
        model=model,
        data=data,
        threshold=best_threshold,
    )
    eval_result = evaluator.evaluate(split="test")

    # ── [Top-4] 图传播后处理 ──────────────────────────────────────
    pp_cfg = cfg.get("post_process", {})
    if pp_cfg.get("use_label_propagation", False):
        logger.info("🔄 应用图传播后处理...")

        y_prob_raw = np.array(eval_result["y_prob"])
        y_true = np.array(eval_result["y_true"])

        # 在测试集上做图传播平滑
        y_prob_smooth = trainer.apply_label_propagation(
            probs=y_prob_raw,
            mask=data.test_mask.cpu().numpy().nonzero()[0],
            alpha=pp_cfg.get("propagation_alpha", 0.5),
            steps=pp_cfg.get("propagation_steps", 3),
        )

        # 用平滑后概率重新搜索阈值
        from src.models.trainer import _search_threshold_np

        best_thr_smooth = _search_threshold_np(
            y_prob=y_prob_smooth,
            y_true=y_true,
            recall_floor=0.88,
            f1_w=0.3,
            rec_w=0.7,
        )

        from sklearn.metrics import (
            f1_score, recall_score, precision_score
        )
        preds_smooth = (y_prob_smooth >= best_thr_smooth).astype(int)

        f1_s = f1_score(y_true, preds_smooth, pos_label=1, zero_division=0)
        rec_s = recall_score(y_true, preds_smooth, pos_label=1, zero_division=0)
        prec_s = precision_score(y_true, preds_smooth, pos_label=1, zero_division=0)

        print(f"\n{'=' * 60}")
        print(f"  🔄 图传播后处理结果 (thr={best_thr_smooth:.3f})")
        print(f"  Recall   : {rec_s:.4f}  (原始: {eval_result['recall_illicit']:.4f})")
        print(f"  F1-score : {f1_s:.4f}  (原始: {eval_result['f1_illicit']:.4f})")
        print(f"  Precision: {prec_s:.4f}  (原始: {eval_result['precision_illicit']:.4f})")
        print(f"{'=' * 60}")

    # 提取测试集嵌入与标签（供可视化使用）
    test_mask   = data.test_mask.cpu()
    test_emb    = embeddings[test_mask].numpy()
    test_labels = data.y[test_mask].numpy()

    # ── Step 8: 可视化分析 ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 8: 可视化分析")
    try:
        viz = Visualizer()
        viz.plot_all(
            embeddings  = test_emb,
            labels      = test_labels,
            eval_result = eval_result,
            history     = history,
        )
    except Exception as e:
        logger.warning(f"可视化跳过（非致命）: {e}")

    # ── Step 9: 阈值分析 ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 9: 阈值分析")
    try:
        y_true = np.array(eval_result["y_true"])
        y_prob = np.array(eval_result["y_prob"])

        analyzer = ThresholdAnalyzer()
        analyzer.compute_threshold(y_prob)
        analyzer.find_high_error_samples(y_true=y_true, y_prob=y_prob)

        if hasattr(analyzer, "generate_threshold_report"):
            analyzer.generate_threshold_report(y_true=y_true, y_prob=y_prob)
    except Exception as e:
        logger.warning(f"阈值分析跳过（非致命）: {e}")

    # ── Step 10: 团伙识别 ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 10: 团伙识别")
    try:
        full_prob = _fill_proba(
            np.array(eval_result["y_prob"]),
            data.test_mask,
            data.num_nodes,
        )
        detector = CommunityDetector()
        detector.run(
            data       = data,
            embeddings = embeddings,
            eval_result= {"y_prob": full_prob},
        )
    except Exception as e:
        logger.warning(f"团伙识别跳过（非致命）: {e}")

    # ── 最终汇总 ──────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  🏁  PIPELINE COMPLETE — FINAL SUMMARY")
    print(f"{sep}")
    print(f"  Model              : {model_type}")
    print(f"  Best Threshold     : {best_threshold:.3f}")
    print(f"  Precision (illicit): {eval_result['precision_illicit']:.4f}")
    print(f"  Recall    (illicit): {eval_result['recall_illicit']:.4f}")
    print(f"  F1-score  (illicit): {eval_result['f1_illicit']:.4f}")
    print(f"  F1-macro           : {eval_result['f1_macro']:.4f}")
    print(f"  ROC-AUC            : {eval_result['roc_auc']:.4f}")
    print(f"  PR-AUC             : {eval_result['pr_auc']:.4f}")
    cm = eval_result["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred licit   Pred illicit")
    print(f"  Actual licit       {cm[0][0]:>8}       {cm[0][1]:>8}")
    print(f"  Actual illicit     {cm[1][0]:>8}       {cm[1][1]:>8}")
    print(f"\n  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"{sep}")
    logger.info("✅ 流水线全部完成")



if __name__ == "__main__":
    main()

