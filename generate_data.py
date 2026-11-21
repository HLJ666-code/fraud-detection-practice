# -*- coding: utf-8 -*-
"""
生成模拟的互联网支付交易数据集
正常交易与欺诈交易在金额、时间、地点、支付方式上呈现不同的分布规律
输出: data/transactions.csv
"""

import numpy as np
import pandas as pd

# 固定随机种子，保证结果可复现
rng = np.random.default_rng(42)

# 总样本数与欺诈比例（教学演示用 10%；真实场景欺诈占比通常更低，属于不平衡数据）
N_SAMPLES = 5000
FRAUD_RATIO = 0.10
N_FRAUD = int(N_SAMPLES * FRAUD_RATIO)
N_NORMAL = N_SAMPLES - N_FRAUD


def make_normal(n):
    """生成正常交易：金额中等、白天为主、距离近、老商户、多用芯片密码"""
    # 交易金额：对数正态分布，中位数约 90 元
    amount = rng.lognormal(mean=4.5, sigma=0.8, size=n)
    # 交易小时：围绕下午 13 点
    hour = np.clip(rng.normal(loc=13, scale=4, size=n), 0, 23).astype(int)
    # 与常用地点距离：指数分布，整体偏近
    distance_home = rng.exponential(scale=3.0, size=n)
    # 与上一笔交易距离：整体偏近
    distance_last = rng.exponential(scale=2.5, size=n)
    # 是否老商户：75% 是老商户
    repeat_retailer = rng.binomial(1, 0.75, size=n)
    # 是否芯片插卡：70% 用芯片
    used_chip = rng.binomial(1, 0.70, size=n)
    # 是否输密码：65% 输密码
    used_pin = rng.binomial(1, 0.65, size=n)
    # 是否线上订单：一半左右
    online_order = rng.binomial(1, 0.50, size=n)
    return pd.DataFrame({
        "trans_hour": hour,
        "amount": np.round(amount, 2),
        "distance_home": np.round(distance_home, 2),
        "distance_last": np.round(distance_last, 2),
        "repeat_retailer": repeat_retailer,
        "used_chip": used_chip,
        "used_pin": used_pin,
        "online_order": online_order,
    })


def make_fraud(n):
    """生成欺诈交易：金额偏高、深夜偏多、异地、新商户、少用芯片密码、线上居多"""
    # 交易金额：对数正态分布，中位数约 330 元，整体高于正常交易
    amount = rng.lognormal(mean=5.8, sigma=1.0, size=n)
    # 交易小时：深夜（0-4 点）和夜间（22-23 点）偏多
    night = rng.binomial(1, 0.6, size=n)
    hour = np.where(
        night == 1,
        rng.choice([0, 1, 2, 3, 4, 22, 23], size=n),
        np.clip(rng.normal(loc=13, scale=4, size=n), 0, 23),
    ).astype(int)
    # 与常用地点距离：指数分布，整体偏远
    distance_home = rng.exponential(scale=15.0, size=n)
    # 与上一笔交易距离：整体偏远
    distance_last = rng.exponential(scale=12.0, size=n)
    # 是否老商户：25% 是老商户（新商户居多）
    repeat_retailer = rng.binomial(1, 0.25, size=n)
    # 是否芯片插卡：20% 用芯片
    used_chip = rng.binomial(1, 0.20, size=n)
    # 是否输密码：20% 输密码
    used_pin = rng.binomial(1, 0.20, size=n)
    # 是否线上订单：80% 是线上订单
    online_order = rng.binomial(1, 0.80, size=n)
    return pd.DataFrame({
        "trans_hour": hour,
        "amount": np.round(amount, 2),
        "distance_home": np.round(distance_home, 2),
        "distance_last": np.round(distance_last, 2),
        "repeat_retailer": repeat_retailer,
        "used_chip": used_chip,
        "used_pin": used_pin,
        "online_order": online_order,
    })


# 拼接两类样本，打上标签
normal_df = make_normal(N_NORMAL)
normal_df["label"] = 0
fraud_df = make_fraud(N_FRAUD)
fraud_df["label"] = 1

df = pd.concat([normal_df, fraud_df], ignore_index=True)
# 打乱顺序
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 保存到 CSV
out_path = "data/transactions.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"数据集已生成: {out_path}")
print(f"总样本: {len(df)}，正常: {(df['label']==0).sum()}，欺诈: {(df['label']==1).sum()}")
print(f"欺诈占比: {df['label'].mean():.2%}")
