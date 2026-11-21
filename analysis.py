# -*- coding: utf-8 -*-
"""
支付交易欺诈识别建模分析
流程: 数据探索 -> 预处理 -> 逻辑回归/随机森林建模 -> 评估出图
输入: data/transactions.csv
输出: figures/ 下的图表 + 控制台打印的评估结果
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接保存图片
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

# 中文显示设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 特征列与标签列
FEATURES = [
    "trans_hour", "amount", "distance_home", "distance_last",
    "repeat_retailer", "used_chip", "used_pin", "online_order",
]
LABEL = "label"

# ========== 1. 读取数据 ==========
df = pd.read_csv("data/transactions.csv")
print("=" * 60)
print(f"样本总量: {len(df)}，正常: {(df[LABEL]==0).sum()}，欺诈: {(df[LABEL]==1).sum()}")
print(f"欺诈占比: {df[LABEL].mean():.2%}")

X = df[FEATURES]
y = df[LABEL]

# ========== 2. 数据探索 ==========

# 图1: 正常与欺诈交易的金额分布对比
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df.loc[df[LABEL]==0, "amount"], bins=40, alpha=0.6, label="正常交易", color="#4C9F70")
ax.hist(df.loc[df[LABEL]==1, "amount"], bins=40, alpha=0.6, label="欺诈交易", color="#D9534F")
ax.set_xlabel("交易金额（元）")
ax.set_ylabel("笔数")
ax.set_title("正常与欺诈交易金额分布对比")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig1_amount_dist.png", dpi=150)
plt.close(fig)

# 图2: 各特征在两类样本中的均值对比
mean_by_label = df.groupby(LABEL)[FEATURES].mean().T
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(FEATURES))
w = 0.35
ax.bar(x - w/2, mean_by_label[0], w, label="正常交易", color="#4C9F70")
ax.bar(x + w/2, mean_by_label[1], w, label="欺诈交易", color="#D9534F")
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=30, ha="right")
ax.set_ylabel("均值")
ax.set_title("各特征在正常/欺诈样本中的均值对比")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig2_feature_mean.png", dpi=150)
plt.close(fig)

# 打印两类样本的特征均值，供报告引用
print("\n两类样本特征均值:")
print(mean_by_label.round(3).to_string())

# ========== 3. 划分训练集/测试集 ==========
# 分层抽样，保证训练集和测试集欺诈比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"\n训练集: {len(X_train)} 笔，测试集: {len(X_test)} 笔")

# 标准化（逻辑回归对量纲敏感；树模型不受标准化影响）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ========== 4. 训练模型 ==========
# 逻辑回归：基线模型，类别权重平衡，对应评分卡的可解释思路
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

# 随机森林：非线性模型，类别权重平衡
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)

# ========== 5. 评估 ==========
def evaluate(model, X_test, y_test, name):
    """计算分类指标并返回结果字典"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = {
        "模型": name,
        "准确率": accuracy_score(y_test, y_pred),
        "精确率": precision_score(y_test, y_pred),
        "召回率": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
    }
    return result, y_pred

lr_result, lr_pred = evaluate(lr, X_test_s, y_test, "逻辑回归")
rf_result, rf_pred = evaluate(rf, X_test, y_test, "随机森林")

metrics_df = pd.DataFrame([lr_result, rf_result]).set_index("模型")
print("\n模型评估结果:")
print(metrics_df.round(4).to_string())

# 图3: 两个模型的混淆矩阵并排
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, y_pred, name in zip(axes, [lr_pred, rf_pred], ["逻辑回归", "随机森林"]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["正常", "欺诈"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{name} 混淆矩阵")
fig.tight_layout()
fig.savefig("figures/fig3_confusion_matrix.png", dpi=150)
plt.close(fig)

# 图4: 模型指标对比
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics_names = ["准确率", "精确率", "召回率", "F1", "AUC"]
x = np.arange(len(metrics_names))
w = 0.35
ax.bar(x - w/2, metrics_df.loc["逻辑回归", metrics_names], w, label="逻辑回归", color="#5B8FF9")
ax.bar(x + w/2, metrics_df.loc["随机森林", metrics_names], w, label="随机森林", color="#F6BD16")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("得分")
ax.set_title("两个模型评估指标对比")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig4_model_compare.png", dpi=150)
plt.close(fig)

# ========== 6. 特征重要度（随机森林） ==========
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(importance.index, importance.values, color="#5B8FF9")
ax.set_xlabel("重要度")
ax.set_title("随机森林特征重要度排序")
fig.tight_layout()
fig.savefig("figures/fig5_feature_importance.png", dpi=150)
plt.close(fig)

print("\n随机森林特征重要度:")
print(importance.sort_values(ascending=False).round(4).to_string())

print("\n图表已保存到 figures/ 目录")
