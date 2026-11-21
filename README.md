# 支付交易欺诈风险识别

互联网金融课程期末报告 —— 互联网支付交易中的欺诈风险识别

## 项目简介

围绕互联网支付场景下的欺诈交易识别问题，用分类模型对交易做异常检测：

- 对比正常交易与欺诈交易在金额、时间、地点、支付方式上的特征差异；
- 训练逻辑回归与随机森林两个模型，比较识别效果；
- 分析各特征对欺诈识别的贡献程度。

## 文件结构

```
fraud-detection-practice/
├── data/
│   └── transactions.csv      # 模拟交易数据集
├── figures/                   # 生成的图表
├── generate_data.py           # 模拟数据生成脚本
├── analysis.py                 # 建模分析代码
├── REPORT.md                  # 期末报告正文
└── README.md                  # 项目说明
```

## 如何运行

1. 安装依赖：`pip install pandas numpy scikit-learn matplotlib`
2. 生成数据：`python generate_data.py`
3. 建模分析：`python analysis.py`
4. 图表自动保存到 `figures/` 文件夹

## 模型与指标

- 模型：逻辑回归（基线）、随机森林；
- 不平衡处理：类别权重 `balanced`；
- 指标：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵。
