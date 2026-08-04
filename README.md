# EfficientNet-B0 图像分类实验

基于 EfficientNet-B0 的 Oxford Flowers 102 花卉分类实验，使用 PyTorch + timm 实现。

## 项目结构

```
├── train.py               # 主训练脚本（5 个阶段，全中文注释）
├── inference.py            # 独立推理脚本（单张图片 → 预测类别）
├── generate_report.py      # 可视化报告生成器（自动读取训练结果）
├── requirements.txt        # 依赖清单
├── CODE_GUIDE.html         # 完整代码解析（浏览器打开，新手必读）
├── 实验报告.html           # 可视化实验报告（浏览器打开）
└── output/
    ├── best_model.pth                     # 最优模型权重
    ├── metrics.json                        # 实验指标数据
    ├── loss_acc_curve.png                 # 训练/验证 Loss 和 Acc 曲线
    ├── confusion_matrix.png               # 混淆矩阵（计数版）
    └── confusion_matrix_normalized.png    # 混淆矩阵（归一化版）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python train.py

# 3. 生成可视化报告
python generate_report.py

# 4. 单张图片推理
python inference.py 图片路径.jpg
```

## 数据集

Oxford Flowers 102 — 102 种花卉，共 8189 张图片（训练 1020 / 验证 1020 / 测试 6149）

## 实验结果

| 指标 | 数值 |
|---|---|
| Accuracy | 86.18% |
| Precision (macro) | 87.48% |
| Recall (macro) | 86.18% |
| F1-Score (macro) | 86.02% |

## 文档

- `CODE_GUIDE.html` — 逐行代码解析，面向新手，讲解每个函数和知识点
- `实验报告.html` — 可视化课程实验报告，包含封面/目录/指标卡片/图表/架构图
