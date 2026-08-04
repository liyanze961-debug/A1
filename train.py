"""
================================================================================
基于EfficientNet-B0的图像分类实验
================================================================================
课程作业：使用EfficientNet在公开数据集上完成图像分类任务
重点：分类性能指标的计算与分析

框架：PyTorch  |  模型：EfficientNet-B0 (timm预训练)
数据集：牛津花卉数据集 Oxford Flowers 102（102类花卉分类）
       -> 若下载失败，自动降级为猫狗二分类数据集
================================================================================
"""

import os
import sys
import copy
import random
import warnings
from pathlib import Path

# ============================================================================
# 网络环境适配：优先使用HF镜像，解决国内下载HuggingFace模型困难的问题
# ============================================================================
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[配置] 使用 HuggingFace 镜像: https://hf-mirror.com")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，用于服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================================
# 全局配置参数
# ============================================================================
CONFIG = {
    # 数据集
    "data_dir": "./data",                 # 数据集存放目录
    "dataset_name": "flowers102",         # "flowers102" 或 "catsdogs"
    "num_classes": 102,                   # 类别数（flowers102=102, catsdogs=2）

    # 模型
    "model_name": "efficientnet_b0",      # timm模型名称
    "pretrained": True,                   # 是否使用预训练权重

    # 训练
    "batch_size": 32,
    "num_epochs": 20,                     # 训练轮数（完整30轮更好，此处用20轮做演示）
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,                     # DataLoader工作进程数（Windows建议0）

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 随机种子
    "seed": 42,

    # 输出
    "save_dir": "./output",               # 输出文件保存目录
}
print(f"设备：{CONFIG['device']}")

# 设置随机种子，确保可复现
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# 创建输出目录
os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ============================================================================
# 阶段1：数据准备
# ============================================================================
def build_transforms(train=True):
    """
    构建数据增强/预处理pipeline
    - 训练集：随机裁剪 + 翻转 + 颜色抖动 + 归一化
    - 验证集：中心裁剪 + 归一化
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet均值
                                 std=[0.229, 0.224, 0.225]),  # ImageNet标准差
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_augmented_transforms():
    """
    增强版数据增强（用于阶段5调优对比）
    增加随机旋转和随机擦除
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),            # <- 新增：随机旋转 +/-15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # <- 新增：随机擦除
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class CatDogRemapDataset(Dataset):
    """
    猫狗数据集包装器：将 Cats vs Dogs 的文件夹结构转为标准 Dataset
    目录结构需为：
    data/catsdogs/
        cats/    <- 猫图片
        dogs/    <- 狗图片
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((
                            os.path.join(cls_dir, fname),
                            self.class_to_idx[cls]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_flowers102(data_dir):
    """
    加载牛津花卉数据集 Flowers102
    返回：train_dataset, val_dataset, num_classes, class_names
    """
    print("[阶段1] 正在加载牛津花卉数据集 Flowers102...")
    try:
        # 下载并加载训练集和验证集
        train_dataset = datasets.Flowers102(
            root=data_dir, split="train",
            transform=build_transforms(train=True),
            download=True
        )
        val_dataset = datasets.Flowers102(
            root=data_dir, split="val",
            transform=build_transforms(train=False),
            download=True
        )

        num_classes = 102
        class_names = [f"flower_{i}" for i in range(1, 103)]  # 花卉数据集的无名类

        print(f"  [OK] Flowers102 加载成功")
        print(f"    训练集样本数: {len(train_dataset)}")
        print(f"    验证集样本数: {len(val_dataset)}")
        print(f"    类别数: {num_classes}")

        return train_dataset, val_dataset, num_classes, class_names

    except Exception as e:
        print(f"  [FAIL] Flowers102 加载失败: {e}")
        print(f"  -> 请检查网络连接或手动下载数据集")
        return None, None, None, None


def load_catsdogs(data_dir):
    """
    加载猫狗二分类数据集（当Flowers102不可用时的降级方案）
    期望目录结构：
    data/catsdogs/train/cats/, data/catsdogs/train/dogs/
    或 data/catsdogs/cats/, data/catsdogs/dogs/
    """
    print("[阶段1] 切换到猫狗二分类数据集...")

    catsdogs_dir = os.path.join(data_dir, "catsdogs")
    os.makedirs(catsdogs_dir, exist_ok=True)

    # 检查是否已有数据
    has_cats = os.path.isdir(os.path.join(catsdogs_dir, "cats"))
    has_dogs = os.path.isdir(os.path.join(catsdogs_dir, "dogs"))
    has_train = os.path.isdir(os.path.join(catsdogs_dir, "train"))

    if has_train:
        # 使用 torchvision ImageFolder
        train_transform = build_transforms(train=True)
        val_transform = build_transforms(train=False)
        full_dataset = datasets.ImageFolder(
            os.path.join(catsdogs_dir, "train"),
            transform=None
        )
        total = len(full_dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(CONFIG["seed"])
        )
        # 注意：random_split 的数据集需要各自设置transform
        train_dataset = _apply_transform_dataset(train_dataset, train_transform)
        val_dataset = _apply_transform_dataset(val_dataset, val_transform)
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes

    elif has_cats and has_dogs:
        # 简易目录结构
        dataset = CatDogRemapDataset(catsdogs_dir, transform=None)
        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(CONFIG["seed"])
        )
        train_dataset = _apply_transform_dataset(train_dataset, build_transforms(train=True))
        val_dataset = _apply_transform_dataset(val_dataset, build_transforms(train=False))
        num_classes = len(dataset.classes)
        class_names = dataset.classes

    else:
        print("  [FAIL] 未找到猫狗数据集文件！")
        print("  -> 请按以下结构放置数据：")
        print(f"    {catsdogs_dir}/")
        print(f"      cats/  (猫图片)")
        print(f"      dogs/  (狗图片)")
        print("  -> 或者使用 torchvision 自动下载 Flowers102")
        sys.exit(1)

    print(f"  [OK] 猫狗数据集加载成功")
    print(f"    训练集样本数: {len(train_dataset)}")
    print(f"    验证集样本数: {len(val_dataset)}")
    print(f"    类别数: {num_classes}")
    print(f"    类别名: {class_names}")

    return train_dataset, val_dataset, num_classes, class_names


class _TransformWrapper(Dataset):
    """为已有数据集动态添加/替换 transform"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _apply_transform_dataset(dataset, transform):
    """将 transform 应用到数据集"""
    return _TransformWrapper(dataset, transform)


def prepare_data(config):
    """
    数据准备主函数
    返回：train_loader, val_loader, num_classes, class_names
    """
    train_dataset, val_dataset, num_classes, class_names = \
        load_flowers102(config["data_dir"])

    if train_dataset is None:
        train_dataset, val_dataset, num_classes, class_names = \
            load_catsdogs(config["data_dir"])
        config["num_classes"] = num_classes
        config["dataset_name"] = "catsdogs"

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False,
    )

    return train_loader, val_loader, num_classes, class_names


# ============================================================================
# 阶段2：模型搭建
# ============================================================================
def build_model(num_classes, device):
    """
    使用 timm 加载 EfficientNet-B0 预训练模型
    替换最后的分类头以适配目标数据集类别数
    """
    print("[阶段2] 搭建模型 EfficientNet-B0...")

    # 加载预训练模型
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=CONFIG["pretrained"],
        num_classes=num_classes
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: {CONFIG['model_name']}")
    print(f"  类别数: {num_classes}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 损失函数：交叉熵
    criterion = nn.CrossEntropyLoss()

    # 优化器：AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"],
        eta_min=1e-6
    )

    return model, criterion, optimizer, scheduler


# ============================================================================
# 阶段3：训练循环
# ============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """单个epoch的训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练中", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="验证中", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    完整训练循环
    - 每个epoch打印loss和准确率
    - 保存最优模型权重
    - 记录训练历史用于绘图
    """
    print("[阶段3] 开始训练...")
    print(f"  设备: {config['device']}")
    print(f"  Epoch数: {config['num_epochs']}")
    print(f"  学习率: {config['learning_rate']}")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    best_val_acc = 0.0
    best_model_wts = None

    for epoch in range(config["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config["device"]
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, config["device"]
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  [*] 最佳模型已更新！(Val Acc: {best_val_acc:.2f}%)")

    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

    # 保存最佳模型权重
    save_path = os.path.join(config["save_dir"], "best_model.pth")
    torch.save(best_model_wts, save_path)
    print(f"最佳模型已保存至: {save_path}")

    return history


def plot_curves(history, save_dir):
    """
    绘制 训练/验证 的 Loss 和 Accuracy 曲线
    保存为 loss_acc_curve.png
    """
    print("[阶段3] 绘制训练曲线...")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history["train_loss"], "b-o", label="训练Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="验证Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练 & 验证 Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax2.plot(epochs, history["train_acc"], "b-o", label="训练准确率", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="验证准确率", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("训练 & 验证 准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_acc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练曲线已保存至: {save_path}")


# ============================================================================
# 阶段4：【重点】分类性能指标计算与分析
# ============================================================================
@torch.no_grad()
def get_all_predictions(model, val_loader, device):
    """
    在验证集上做全量推理，拿到所有真实标签和预测结果
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="全量推理"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred, class_names):
    """
    计算分类性能指标：
    - 准确率 (Accuracy)
    - 精确率 (Precision) - 宏平均
    - 召回率 (Recall) - 宏平均
    - F1分数 (F1-Score) - 宏平均
    - 分类报告 (Classification Report)
    """
    print("\n" + "=" * 70)
    print("【阶段4】分类性能指标计算与分析")
    print("=" * 70)

    # ---- 各项指标 ----
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*40}")
    print(f"  宏平均指标")
    print(f"{'='*40}")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision): {precision:.4f}  (宏平均)")
    print(f"  召回率 (Recall):    {recall:.4f}  (宏平均)")
    print(f"  F1分数 (F1-Score):  {f1:.4f}  (宏平均)")

    # ---- 分类报告 ----
    print(f"\n{'='*40}")
    print(f"  详细分类报告 (sklearn classification_report)")
    print(f"{'='*40}")
    if len(class_names) <= 20:
        # 类别较少时打印完整报告
        target_names = class_names
    else:
        # 类别较多时，用类别编号
        target_names = [f"c{i}" for i in range(len(class_names))]
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=3, zero_division=0
    )
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_confusion_matrix(y_true, y_pred, num_classes, class_names, save_dir):
    """
    绘制混淆矩阵并保存为图片
    - 如果类别数 <= 20：绘制完整标注版
    - 如果类别数 > 20：绘制热力图版（无单个标签，用颜色深浅区分）
    """
    print(f"\n[阶段4] 绘制混淆矩阵...")

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if num_classes <= 20:
        # ---- 小类别：完整标注版 ----
        fig, ax = plt.subplots(figsize=(12, 10))
        # 使用简短标签
        short_names = [n[:12] for n in class_names]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=short_names, yticklabels=short_names,
                    ax=ax, cbar_kws={"label": "样本数"})
        ax.set_xlabel("预测标签", fontsize=12)
        ax.set_ylabel("真实标签", fontsize=12)
        ax.set_title(f"混淆矩阵 (共 {num_classes} 类)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
    else:
        # ---- 多类别：不标注单个数字，用颜色深浅 ----
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar_kws={"label": "样本数"})
        ax.set_xlabel("预测标签", fontsize=12)
        ax.set_ylabel("真实标签", fontsize=12)
        ax.set_title(f"混淆矩阵 (共 {num_classes} 类)", fontsize=14)
        # 每隔5类标一个刻度
        ticks = list(range(0, num_classes, max(1, num_classes // 20)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, rotation=0)
        ax.set_yticklabels(ticks, rotation=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混淆矩阵已保存至: {save_path}")

    # ---- 归一化混淆矩阵 ----
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, cmap="YlOrRd", vmin=0, vmax=1,
                ax=ax, cbar_kws={"label": "比例"})
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(f"归一化混淆矩阵 (行归一化, 共 {num_classes} 类)", fontsize=14)
    ticks = list(range(0, num_classes, max(1, num_classes // 20)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, rotation=0)
    ax.set_yticklabels(ticks, rotation=0)

    plt.tight_layout()
    save_path_norm = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plt.savefig(save_path_norm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  归一化混淆矩阵已保存至: {save_path_norm}")

    return cm


def analyze_confusion_matrix(cm, class_names, top_k=10):
    """
    分析混淆矩阵：
    - 找出最容易混淆的类别对（非对角线最大值）
    - 列出 Top-K 易混淆类别对
    - 初步分析误分类原因
    """
    print(f"\n{'='*40}")
    print(f"  混淆矩阵分析")
    print(f"{'='*40}")

    num_classes = cm.shape[0]

    # 将对角线置零，找到每个真实类别最容易误分类到的类别
    cm_no_diag = cm.copy().astype(float)
    for i in range(num_classes):
        cm_no_diag[i, i] = 0.0

    # 收集所有误分类对
    error_pairs = []
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            if true_label != pred_label and cm[true_label, pred_label] > 0:
                error_pairs.append({
                    "true": true_label,
                    "pred": pred_label,
                    "true_name": class_names[true_label] if true_label < len(class_names) else f"c{true_label}",
                    "pred_name": class_names[pred_label] if pred_label < len(class_names) else f"c{pred_label}",
                    "count": cm[true_label, pred_label],
                })

    # 按误分类数量降序排列
    error_pairs.sort(key=lambda x: x["count"], reverse=True)

    # 显示 Top-K
    print(f"\n  Top-{top_k} 最容易误分类的类别对：")
    print(f"  {'排名':<6}{'真实类别':<25}{'误分类为':<25}{'误分类次数':<12}")
    print(f"  {'-'*68}")
    for rank, pair in enumerate(error_pairs[:top_k], 1):
        print(f"  {rank:<6}{pair['true_name']:<25}{pair['pred_name']:<25}{pair['count']:<12}")

    # ---- 按真实类别的混淆率分析 ----
    print(f"\n  每个类别最容易混淆到的目标类别 (Top-3)：")
    for true_label in range(num_classes):
        true_name = class_names[true_label] if true_label < len(class_names) else f"c{true_label}"
        # 找到该类别最容易被误分类到的类
        row = cm[true_label].copy()
        row[true_label] = 0  # 忽略正确分类
        top_indices = np.argsort(row)[::-1][:3]
        confusions = []
        for idx in top_indices:
            if row[idx] > 0:
                pred_name = class_names[idx] if idx < len(class_names) else f"c{idx}"
                confusions.append(f"{pred_name}({row[idx]}次)")

        if confusions:
            print(f"    {true_name:<20} -> {', '.join(confusions)}")

    # ---- 误差原因初步分析 ----
    print(f"\n  {'='*40}")
    print(f"  误差原因初步分析：")
    print(f"  {'='*40}")
    print(f"  1. 视觉相似性：花卉数据集中，同一科/属的花卉外观非常相近，")
    print(f"     如不同品种的菊花、玫瑰等，导致模型难以区分。")
    print(f"  2. 样本不均衡：某些类别训练样本较少，模型学习不充分。")
    print(f"  3. 姿态/光照变化：同一花卉的不同拍摄角度和光照条件下差异大。")
    print(f"  4. 数据增强影响：部分增强（如颜色抖动）可能模糊了关键特征。")

    # 统计每个类别的样本数（粗略通过混淆矩阵行和）
    class_counts = cm.sum(axis=1)
    min_count = class_counts.min()
    max_count = class_counts.max()
    print(f"\n  样本分布：最小 {int(min_count)} 张 / 类，最大 {int(max_count)} 张 / 类")
    if max_count > min_count * 3:
        print(f"  [WARN] 样本分布不均衡，可能影响少数类的分类效果")

    return error_pairs


def run_stage4(model, val_loader, num_classes, class_names, config):
    """阶段4主函数：指标计算与分析"""
    print(f"\n[阶段4] 在验证集上进行全量推理...")

    # 1. 全量推理
    y_true, y_pred = get_all_predictions(model, val_loader, config["device"])

    # 2. 计算指标
    metrics = compute_metrics(y_true, y_pred, class_names)

    # 3. 绘制混淆矩阵
    cm = plot_confusion_matrix(y_true, y_pred, num_classes, class_names, config["save_dir"])

    # 4. 混淆矩阵分析
    error_pairs = analyze_confusion_matrix(cm, class_names, top_k=10)

    return metrics, cm, error_pairs


# ============================================================================
# 阶段5：简单调优对比
# ============================================================================
def run_tuning_experiment(config, num_classes):
    """
    调优对比实验：
    - 基线配置：lr=1e-4, 基础数据增强
    - 调优配置：lr=5e-5, 增强数据增强（旋转+擦除）
    """
    print("\n" + "=" * 70)
    print("【阶段5】调优对比实验")
    print("=" * 70)

    results = {}

    # -------------------- 实验A：基线 --------------------
    print(f"\n{'='*40}")
    print(f"  实验A：基线配置")
    print(f"  lr={config['learning_rate']}, 基础数据增强")
    print(f"{'='*40}")

    # 重新加载数据（基础增强）
    train_dataset, val_dataset, _, class_names = load_flowers102(config["data_dir"])
    if train_dataset is None:
        train_dataset, val_dataset, _, class_names = load_catsdogs(config["data_dir"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False
    )

    # 模型
    model_a, criterion, optimizer, scheduler = build_model(num_classes, config["device"])

    # 训练（调优对比用较少epoch以节省时间）
    tuning_epochs = 10
    print(f"  调优实验训练 {tuning_epochs} epochs...")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tuning_epochs, eta_min=1e-6)

    best_val_acc_a = 0.0
    best_wts_a = None
    for epoch in range(tuning_epochs):
        train_loss, train_acc = train_one_epoch(
            model_a, train_loader, criterion, optimizer, config["device"])
        val_loss, val_acc = validate(
            model_a, val_loader, criterion, config["device"])
        scheduler.step()
        print(f"  [基线] Epoch {epoch+1}/{tuning_epochs}: "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        if val_acc > best_val_acc_a:
            best_val_acc_a = val_acc
            best_wts_a = copy.deepcopy(model_a.state_dict())

    model_a.load_state_dict(best_wts_a)

    # 推理 + 指标
    y_true, y_pred = get_all_predictions(model_a, val_loader, config["device"])
    metrics_a = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    results["基线 (lr=1e-4, 基础增强)"] = metrics_a

    # -------------------- 实验B：调优 --------------------
    print(f"\n{'='*40}")
    print(f"  实验B：调优配置")
    print(f"  lr=5e-5, 增强数据增强（旋转+擦除）")
    print(f"{'='*40}")

    # 重新加载数据（增强版增强）
    train_dataset_b, val_dataset_b, _, _ = load_flowers102(config["data_dir"])
    if train_dataset_b is None:
        train_dataset_b, val_dataset_b, _, _ = load_catsdogs(config["data_dir"])

    # 替换训练集的transform为增强版
    # Flowers102 数据集对象有 .transform 属性，直接替换即可
    if hasattr(train_dataset_b, 'transform'):
        train_dataset_b.transform = build_augmented_transforms()
    elif hasattr(train_dataset_b, 'dataset') and hasattr(train_dataset_b.dataset, 'transform'):
        # Subset(random_split) 包装的情况，修改底层数据集的transform
        train_dataset_b.dataset.transform = build_augmented_transforms()
    else:
        # 降级方案：用 TransformWrapper 包装
        train_dataset_b = _apply_transform_dataset(train_dataset_b, build_augmented_transforms())

    train_loader_b = DataLoader(
        train_dataset_b, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if config["device"] == "cuda" else False
    )

    # 模型
    model_b, criterion, optimizer_b, _ = build_model(num_classes, config["device"])
    optimizer_b = optim.AdamW(model_b.parameters(), lr=5e-5, weight_decay=1e-4)  # <- 降低学习率
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=tuning_epochs, eta_min=1e-6)

    best_val_acc_b = 0.0
    best_wts_b = None
    for epoch in range(tuning_epochs):
        train_loss, train_acc = train_one_epoch(
            model_b, train_loader_b, criterion, optimizer_b, config["device"])
        val_loss, val_acc = validate(
            model_b, val_loader, criterion, config["device"])
        scheduler_b.step()
        print(f"  [调优] Epoch {epoch+1}/{tuning_epochs}: "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        if val_acc > best_val_acc_b:
            best_val_acc_b = val_acc
            best_wts_b = copy.deepcopy(model_b.state_dict())

    model_b.load_state_dict(best_wts_b)

    # 推理 + 指标
    y_true_b, y_pred_b = get_all_predictions(model_b, val_loader, config["device"])
    metrics_b = {
        "accuracy": accuracy_score(y_true_b, y_pred_b),
        "precision": precision_score(y_true_b, y_pred_b, average="macro", zero_division=0),
        "recall": recall_score(y_true_b, y_pred_b, average="macro", zero_division=0),
        "f1": f1_score(y_true_b, y_pred_b, average="macro", zero_division=0),
    }
    results["调优 (lr=5e-5, 增强增强)"] = metrics_b

    # -------------------- 对比输出 --------------------
    print(f"\n{'='*40}")
    print(f"  调优对比结果")
    print(f"{'='*40}")
    print(f"  {'指标':<20}{'基线':<20}{'调优':<20}{'变化':<15}")
    print(f"  {'-'*75}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        base_val = results["基线 (lr=1e-4, 基础增强)"][key]
        tuned_val = results["调优 (lr=5e-5, 增强增强)"][key]
        delta = tuned_val - base_val
        arrow = "UP" if delta > 0 else "DN" if delta < 0 else "->"
        key_name = {"accuracy": "准确率", "precision": "精确率", "recall": "召回率", "f1": "F1分数"}[key]
        print(f"  {key_name:<20}{base_val:<20.4f}{tuned_val:<20.4f}{arrow} {abs(delta):.4f}")

    # 调优分析
    print(f"\n  {'='*40}")
    print(f"  调优分析结论：")
    print(f"  {'='*40}")
    if metrics_b["accuracy"] > metrics_a["accuracy"]:
        print(f"  调优后准确率提升了 {metrics_b['accuracy'] - metrics_a['accuracy']:.4f}")
        print(f"  分析：降低学习率使训练更稳定，增强数据增强（旋转+擦除）")
        print(f"        提高了模型的泛化能力，减少过拟合。")
    else:
        print(f"  本次调优准确率变化不大或略有下降。")
        print(f"  分析：可能原因 -- 训练epoch较少，增强效果未充分体现；")
        print(f"        或数据集本身较简单，基础增强已足够。")

    return results


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("  基于 EfficientNet-B0 的图像分类实验")
    print("  数据集: Oxford Flowers 102（优先）/ Cats vs Dogs（备用）")
    print("=" * 70)

    # ---- 阶段1：数据准备 ----
    train_loader, val_loader, num_classes, class_names = prepare_data(CONFIG)
    CONFIG["num_classes"] = num_classes

    # ---- 阶段2：模型搭建 ----
    model, criterion, optimizer, scheduler = build_model(num_classes, CONFIG["device"])

    # ---- 阶段3：训练 ----
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, CONFIG
    )
    plot_curves(history, CONFIG["save_dir"])

    # ---- 阶段4：指标计算与分析 ----
    metrics, cm, error_pairs = run_stage4(
        model, val_loader, num_classes, class_names, CONFIG
    )

    # ---- 阶段5：调优对比 ----
    tuning_results = run_tuning_experiment(CONFIG, num_classes)

    # ---- 实验总结 ----
    print("\n" + "=" * 70)
    print("  实验总结")
    print("=" * 70)
    print(f"""
    1. 实验环境
       - Python 3.x + PyTorch {torch.__version__}
       - 模型: EfficientNet-B0 (timm预训练)
       - 设备: {CONFIG['device']}
       - 优化器: AdamW (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})

    2. 数据集
       - 名称: {"Oxford Flowers 102" if CONFIG['dataset_name'] == 'flowers102' else 'Cats vs Dogs'}
       - 类别数: {num_classes}
       - 训练集: {len(train_loader.dataset)} 张
       - 验证集: {len(val_loader.dataset)} 张

    3. 分类指标（最优模型在验证集上）
       - 准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
       - 精确率 (Precision): {metrics['precision']:.4f} (宏平均)
       - 召回率 (Recall):    {metrics['recall']:.4f} (宏平均)
       - F1分数 (F1-Score):  {metrics['f1']:.4f} (宏平均)

    4. 混淆矩阵
       - 已保存至 {CONFIG['save_dir']}/confusion_matrix.png
       - 已保存归一化版本至 {CONFIG['save_dir']}/confusion_matrix_normalized.png

    5. 调优对比
       - 基线 Accuracy: {tuning_results['基线 (lr=1e-4, 基础增强)']['accuracy']:.4f}
       - 调优 Accuracy: {tuning_results['调优 (lr=5e-5, 增强增强)']['accuracy']:.4f}

    6. 输出文件
       - 训练曲线: {CONFIG['save_dir']}/loss_acc_curve.png
       - 混淆矩阵: {CONFIG['save_dir']}/confusion_matrix.png
       - 最佳模型: {CONFIG['save_dir']}/best_model.pth
    """)
    print("=" * 70)
    print("实验完成！请查看 {} 目录下的输出文件。".format(CONFIG['save_dir']))
    print("=" * 70)


if __name__ == "__main__":
    main()
