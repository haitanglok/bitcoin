# src/preprocessing/preprocess.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# 定义保存路径（统一工程路径）
SAVE_DIR = "src/preprocessing/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_preprocessing(nodes_path="data/processed/nodes.csv"):
    # 1. 加载清洗后数据
    df = pd.read_csv(nodes_path)
    X = df.filter(regex='feat_')  # 提取所有特征
    y = df['class']
    time_step = df['time_step']

    # 2. 【关键】时序划分训练集/测试集（Elliptic必须按时间，禁止随机！）
    split_time = np.percentile(time_step, 70)  # 前70%时间为训练集
    train_mask = time_step <= split_time
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # 3. 特征标准化（适配金融数据）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. 特征筛选（保留60个核心特征，删除噪声，提速+提精度）
    top_k = 60
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    top_idx = np.argsort(scores)[-top_k:]
    X_train = X_train[:, top_idx]
    X_test = X_test[:, top_idx]


    # 保存预处理工具（供推理使用）
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    joblib.dump(top_idx, os.path.join(SAVE_DIR, "top_idx.pkl"))
    print("✅ 预处理完成，标准化器/特征索引已保存")

    return X_train, X_test, y_train, y_test