# EfficientNet-B0 图像分类实验

基于 EfficientNet-B0 的 Oxford Flowers 102 花卉分类实验，使用 PyTorch + timm 实现。

## 数据集

Oxford Flowers 102 — 102 种花卉，共 8189 张图片（训练 1020 / 验证 1020 / 测试 6149）

## 实验结果

| 指标 | 数值 |
|---|---|
| Accuracy | 86.18% |
| Precision (macro) | 87.48% |
| Recall (macro) | 86.18% |
| F1-Score (macro) | 86.02% |

## 输出文件

- `output/loss_acc_curve.png` — 训练/验证 loss 和准确率曲线
- `output/confusion_matrix.png` — 混淆矩阵（计数版）
- `output/confusion_matrix_normalized.png` — 混淆矩阵（归一化版）
- `output/best_model.pth` — 最优模型权重

## 运行方式

```bash
pip install -r requirements.txt
python train.py
```
