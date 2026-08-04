# ============================================================================
# 一键安装依赖命令（在终端中执行下面这行）：
# pip install torch torchvision timm scikit-learn matplotlib numpy pillow tqdm seaborn
# ============================================================================
# EfficientNet-B0 图像分类实验 — 生物医学工程深度学习课程作业
# 数据集：Oxford Flowers 102（102种花卉分类）
# 硬件：联想拯救者 Y7000P
# ============================================================================

import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# sklearn 用于计算分类指标和绘制混淆矩阵
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================================
# 【知识点】为什么设置 HuggingFace 镜像？
# timm 库的预训练权重默认从 huggingface.co 下载，国内直连经常超时。
# 设置 HF_ENDPOINT 环境变量指向国内镜像站，可以稳定下载模型权重。
# ============================================================================
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 尝试导入 timm（预训练模型库）
try:
    import timm
except ImportError:
    print("[ERROR] timm 库未安装！请先运行：pip install timm")
    sys.exit(1)


# ============================================================================
# 全局配置 — 所有可调参数集中在这里，方便修改
# ============================================================================
class Config:
    """
    实验配置类。
    把参数集中管理的好处：
    1. 修改参数不用到处翻代码
    2. 方便做消融实验（改一个变量看效果）
    3. 方便后续替换网络时对比实验
    """

    # ---------- 路径配置（全部使用相对路径） ----------
    data_dir: str = "./data"  # 数据集下载/存放目录
    output_dir: str = "./output"  # 输出文件保存目录（模型权重、图表）

    # ---------- 模型配置 ----------
    model_name: str = "efficientnet_b0"  # timm 中的模型名，改这里 = 换网络
    pretrained: bool = True  # 是否使用 ImageNet 预训练权重
    fine_tune_mode: str = "full"  # 微调方式："full"(全部参数训练) 或 "freeze"(冻结特征层)

    # ---------- 训练配置 ----------
    batch_size: int = 16  # 批大小，Y7000P 笔记本用 16 比较稳，显存不够改 8
    num_epochs: int = 30  # 训练轮数
    lr: float = 1e-4  # 学习率（迁移学习建议 1e-4 ~ 1e-5）
    weight_decay: float = 1e-4  # 权重衰减（L2 正则化系数），防过拟合
    num_workers: int = 2  # 数据加载子进程数，Windows 建议 0~2

    # ---------- 其他 ----------
    seed: int = 42  # 随机种子，保证实验可复现
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选 GPU/CPU


# ============================================================================
# 工具函数：固定随机种子
# ============================================================================
def set_seed(seed: int = 42):
    """
    固定 Python、NumPy、PyTorch 的随机种子。
    为什么要固定？如果每次运行的随机数不一样，实验结果就不可复现。
    你改了网络结构想对比效果时，至少要确保"随机"这个变量是一致的。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 CuDNN 使用确定性算法（会稍微变慢，但结果可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 阶段一：数据准备
# ============================================================================
def build_transforms():
    """
    构建数据预处理/增强流水线。

    【知识点】为什么需要数据增强？
    Flowers 102 每类只有 10 张训练图，模型很容易"背答案"（过拟合）。
    数据增强 = 对同一张图做随机变换，让模型每次看到的图都不一样，
    迫使模型学习"花的本质特征"而不是"某张特定图片的样子"。

    返回：
        train_transform: 训练集用的变换（包含数据增强）
        eval_transform:  验证/测试集用的变换（不增强，只做标准化）
    """

    # ---- 训练集：包含数据增强 ----
    train_transform = transforms.Compose(
        [
            # 1. 随机缩放到 256，再从里面随机裁 224×224
            #    作用：模拟花朵在画面中大小/位置不同的情况
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            # 2. 随机水平翻转
            #    作用：花朝左和朝右都应该被识别为同一种花
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. 随机旋转 ±20 度
            #    作用：拍摄角度倾斜时也能正确识别
            transforms.RandomRotation(degrees=20),
            # 4. 颜色抖动：随机微调亮度、对比度、饱和度、色相
            #    作用：模拟不同光照/拍摄设备下的颜色差异
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # 5. 转换为 Tensor（把 0-255 的像素值变成 0-1 的浮点数）
            transforms.ToTensor(),
            # 6. 用 ImageNet 的均值和标准差标准化
            #    作用：和预训练模型训练时保持一致，否则预训练权重"水土不服"
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ---- 验证/测试集：不需要数据增强，只做必要处理 ----
    eval_transform = transforms.Compose(
        [
            # 1. 缩放到 256×256
            transforms.Resize(256),
            # 2. 从中心裁出 224×224（保证每次评估裁的区域一致）
            transforms.CenterCrop(224),
            # 3. 转 Tensor
            transforms.ToTensor(),
            # 4. ImageNet 标准化（必须和训练集一致）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def load_flowers102(data_dir: str, train_transform, eval_transform, offline_root: str = None):
    """
    加载 Oxford Flowers 102 数据集。

    策略：
    1. 优先尝试 torchvision 自动下载（需要网络）
    2. 网络失败时检查本地 offline_root 下是否有已下载的文件

    Flowers 102 数据集结构（torchvision）：
        - 102 个花卉类别
        - 训练集：每类 10 张，共 1020 张
        - 验证集：每类 10 张，共 1020 张
        - 测试集：共 6149 张（类别分布不均）

    参数：
        data_dir:      数据集下载目标目录
        train_transform: 训练集预处理变换
        eval_transform:  验证/测试集预处理变换
        offline_root:  离线数据的备用路径
    """

    print(f"\n{'=' * 60}")
    print("阶段一：数据准备 - Oxford Flowers 102")
    print(f"{'=' * 60}")

    # ---- 方案1：尝试从 torchvision 在线下载 ----
    train_set = None
    val_set = None
    test_set = None

    try:
        print("[INFO] 尝试在线下载 Flowers 102 数据集...")
        print(f"[INFO] 下载目录：{data_dir}")

        # torchvision.datasets.Flowers102 会自动下载到 data_dir
        train_set = datasets.Flowers102(
            root=data_dir, split="train", transform=train_transform, download=True
        )
        val_set = datasets.Flowers102(
            root=data_dir, split="val", transform=eval_transform, download=True
        )
        test_set = datasets.Flowers102(
            root=data_dir, split="test", transform=eval_transform, download=True
        )
        print("[OK] 在线下载成功！")

    except Exception as e:
        print(f"[WARN] 在线下载失败：{e}")

        # ---- 方案2：尝试从本地离线路径加载 ----
        if offline_root is None:
            offline_root = data_dir  # 默认检测 data_dir 下是否有已下载文件

        try:
            print(f"[INFO] 尝试从本地离线路径加载：{offline_root}")
            train_set = datasets.Flowers102(
                root=offline_root, split="train", transform=train_transform, download=False
            )
            val_set = datasets.Flowers102(
                root=offline_root, split="val", transform=eval_transform, download=False
            )
            test_set = datasets.Flowers102(
                root=offline_root, split="test", transform=eval_transform, download=False
            )
            print("[OK] 本地离线加载成功！")
        except Exception as e2:
            print(f"[FATAL] 离线加载也失败了：{e2}")
            print("[HELP] 请先在有网络的环境下运行一次，让 torchvision 下载数据集。")
            print("[HELP] 数据会自动保存在 data/ 目录下，之后离线也能用。")
            raise RuntimeError("无法加载 Flowers 102 数据集，请检查网络或本地文件。") from e2

    # ---- 打印数据集信息 ----
    print(f"\n{'─' * 40}")
    print(f"[INFO] 数据集加载完毕：")
    print(f"       类别数：  {len(train_set._labels.unique())}")
    print(f"       训练集：  {len(train_set)} 张")
    print(f"       验证集：  {len(val_set)} 张")
    print(f"       测试集：  {len(test_set)} 张")
    print(f"{'─' * 40}\n")

    return train_set, val_set, test_set


def build_dataloaders(train_set, val_set, test_set, cfg: Config):
    """
    构建 DataLoader。

    【知识点】DataLoader 的作用：
    - 把数据集分成一个个 batch（批次），每次喂给模型一小批数据
    - 自动打乱训练数据（shuffle=True），防止模型记住数据顺序
    - 多进程并行加载（num_workers），提高数据读取速度
    """
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,  # 训练集要打乱，每个 epoch 的顺序都不一样
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),  # GPU 训练时加速数据传输
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,  # 验证集不打乱，保证每次评估结果一致
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# 阶段二：模型搭建
# ============================================================================
def build_model(num_classes: int, cfg: Config):
    """
    使用 timm 库搭建 EfficientNet-B0 分类网络。

    【知识点】迁移学习（Transfer Learning）：
    - 源域：ImageNet（1000 类、100 万张自然图像）
    - 目标域：Flowers 102（102 类花卉）
    - 假设：在 ImageNet 上学会的边缘/纹理/形状特征，对识别花卉也有用
    - 做法：加载 ImageNet 预训练权重 -> 替换分类头 -> 在新数据上微调

    参数：
        num_classes: 目标数据集类别数（Flowers 102 = 102）
        cfg:         全局配置

    返回：
        model:       构建好的模型
    """

    print(f"\n{'=' * 60}")
    print("阶段二：模型搭建 - EfficientNet-B0")
    print(f"{'=' * 60}")

    # ---- 创建模型 ----
    # timm.create_model 一行搞定：下载预训练权重 + 替换分类头
    print(f"[INFO] 加载模型：{cfg.model_name}")
    print(f"[INFO] 预训练权重：{'是 (ImageNet)' if cfg.pretrained else '否 (从头训练)'}")
    print(f"[INFO] 输出类别数：{num_classes}")

    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,  # True = 加载 ImageNet 预训练权重
        num_classes=num_classes,  # timm 自动替换最后的全连接层
    )

    # ---- 两种微调方式的实现 ----
    """
    【重点知识】两种微调策略的区别与适用场景：

    方式一："freeze" — 冻结特征提取层，只训练分类头
    ┌────────────────────────────────────────┐
    │  特征提取器 (EfficientNet backbone)    │ <- 冻结，不更新参数
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
    │  │ L1  │→│ L2  │→│ ... │→│ Ln  │      │
    │  └─────┘ └─────┘ └─────┘ └─────┘      │
    └────────────────────┬───────────────────┘
                         ▼
    ┌────────────────────────────────────────┐
    │  分类头 (新的全连接层，102 类)          │ <- 可训练
    └────────────────────────────────────────┘

    适用场景：
    - 目标数据集很小（如每类只有几张图）
    - 目标域和 ImageNet 比较接近（自然图像）
    - 训练速度快，显存占用少
    - 不容易过拟合

    方式二："full" — 全部参数参与训练
    ┌────────────────────────────────────────┐
    │  特征提取器 (EfficientNet backbone)    │ <- 可训练
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
    │  │ L1  │→│ L2  │→│ ... │→│ Ln  │      │
    │  └─────┘ └─────┘ └─────┘ └─────┘      │
    └────────────────────┬───────────────────┘
                         ▼
    ┌────────────────────────────────────────┐
    │  分类头 (新的全连接层，102 类)          │ <- 可训练
    └────────────────────────────────────────┘

    适用场景：
    - 目标数据集较大
    - 目标域和 ImageNet 差异较大（如医学影像、卫星图）
    - 通常能获得更好的最终性能
    - 但训练慢、显存占用大、容易过拟合（需要更多正则化）

    本项目默认使用 "full"，因为 Flowers 102 和 ImageNet 同属自然图像，
    全参数微调通常能带来几个点的提升。
    """

    if cfg.fine_tune_mode == "freeze":
        print("\n[INFO] 微调方式：冻结特征层")

        # 遍历模型的所有参数
        for name, param in model.named_parameters():
            # 分类头（classifier/head/fc）的参数保持可训练
            # 其他所有层（backbone）冻结
            if "classifier" in name or "head" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # 冻结！

        # 统计一下训练参数占比
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] 总参数量：    {total_params / 1e6:.2f}M")
        print(f"[INFO] 可训练参数：  {trainable_params / 1e6:.2f}M "
              f"({100 * trainable_params / total_params:.1f}%)")

    else:  # "full"
        print("\n[INFO] 微调方式：全部参数训练")
        # 默认所有参数 requires_grad=True，不需要额外操作
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] 可训练参数：{total_params / 1e6:.2f}M (100%)")

    model = model.to(cfg.device)
    print(f"[INFO] 模型已移至：{cfg.device}")
    print(f"[OK] 模型搭建完成！\n")

    return model


def build_loss_optimizer(model, cfg: Config):
    """
    构建损失函数和优化器。

    【知识点 1】交叉熵损失（CrossEntropyLoss）：
    - 多分类任务的标准损失函数
    - 输入：模型输出的 logits（未归一化的分数）+ 真实标签
    - 内部先做 Softmax 把 logits 变成概率，再算负对数似然
    - 公式：Loss = -log(P_true_class)，概率越低 loss 越大

    【知识点 2】AdamW 优化器：
    - Adam 的改进版，修正了权重衰减的实现方式
    - 自适应学习率：每个参数有自己的学习率，收敛更稳定
    - weight_decay = L2 正则化，防止权重过大导致过拟合
    - 迁移学习推荐 lr=1e-4 ~ 1e-5，比从头训练(1e-3)小很多
    """

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # 学习率调度器：余弦退火
    # 学习率从 lr 开始，按余弦曲线逐渐降到 0
    # 好处：训练前期快速收敛，后期精细调整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,  # 周期 = 总 epoch 数
        eta_min=cfg.lr * 0.01,  # 最低降到初始学习率的 1%
    )

    print(f"[INFO] 损失函数：CrossEntropyLoss")
    print(f"[INFO] 优化器：  AdamW (lr={cfg.lr}, weight_decay={cfg.weight_decay})")
    print(f"[INFO] 调度器：  CosineAnnealingLR\n")

    return criterion, optimizer, scheduler


# ============================================================================
# 阶段三：训练循环
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    一个 epoch 的训练过程。

    【核心知识】一次训练迭代（iteration）发生了什么？

    假设 batch_size=16，一个 batch 的处理流程：

    ╔══════════════════════════════════════════════════════════╗
    ║ 1. optimizer.zero_grad()                                ║
    ║    ↓ 清空上一轮累积的梯度（PyTorch 默认会累加梯度）      ║
    ║                                                          ║
    ║ 2. outputs = model(images)                              ║
    ║    ↓ 前向传播：16 张图经过 EfficientNet -> 102 维的输出  ║
    ║                                                          ║
    ║ 3. loss = criterion(outputs, labels)                    ║
    ║    ↓ 计算损失：比较模型预测和真实标签的差距              ║
    ║                                                          ║
    ║ 4. loss.backward()                                      ║
    ║    ↓ 反向传播：自动求每个参数对 loss 的偏导数（梯度）    ║
    ║                                                          ║
    ║ 5. optimizer.step()                                     ║
    ║    ↓ 参数更新：沿梯度反方向微调参数（让 loss 变小）      ║
    ╚══════════════════════════════════════════════════════════╝

    重复这个过程 N 次（N = 训练图片总数 / batch_size），一个 epoch 结束。
    """

    model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 等训练行为）

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # tqdm 是进度条库，让你能实时看到训练进度
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # ── 步骤 1：清零梯度 ──
        optimizer.zero_grad()

        # ── 步骤 2：前向传播 ──
        outputs = model(images)  # shape: (batch_size, 102)

        # ── 步骤 3：计算损失 ──
        loss = criterion(outputs, labels)

        # ── 步骤 4：反向传播（计算梯度）──
        loss.backward()

        # ── 步骤 5：更新参数 ──
        optimizer.step()

        # ── 统计（仅用于打印，不参与梯度计算）──
        _, preds = torch.max(outputs, 1)  # 取分数最高的类别作为预测结果
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()
        total_samples += images.size(0)

        # 更新进度条显示
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    # 计算整个 epoch 的平均 loss 和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


@torch.no_grad()  # 装饰器：验证阶段不需要计算梯度，节省显存和计算
def evaluate(model, dataloader, criterion, device):
    """
    在验证集/测试集上评估模型。

    和训练的区别：
    - model.eval()：关闭 Dropout、固定 BatchNorm 统计量
    - 没有 optimizer.zero_grad() / loss.backward() / optimizer.step()
    - torch.no_grad()：不构建计算图，大幅节省显存
    """

    model.eval()  # 切换到评估模式

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播（没有反向传播和参数更新）
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


# ============================================================================
# 阶段四：评价指标计算与分析
# ============================================================================
@torch.no_grad()
def get_all_predictions(model, dataloader, device):
    """
    在数据集上跑完整推理，收集所有真实标签和预测结果。
    后续计算混淆矩阵和各项指标都需要用到这些数据。
    """

    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in tqdm(dataloader, desc="Inference", leave=False):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred, num_classes: int):
    """
    计算分类任务的核心评价指标。

    【小白必读】四个基础指标的含义（以"识别玫瑰花"为例）：

                    预测=玫瑰          预测≠玫瑰
    真实=玫瑰          TP               FN
                   (猜对了√)        (漏掉了×)

    真实≠玫瑰          FP               TN
                   (报错了×)        (猜对了√)

    准确率 Accuracy  = (TP+TN) / 总数        <- 所有样本中猜对了多少
    精确率 Precision = TP / (TP+FP)          <- 你说"这是玫瑰"时，有多大概率真的对
    召回率 Recall    = TP / (TP+FN)          <- 所有真正的玫瑰，你找到了多少
    F1-Score         = 2*P*R / (P+R)         <- P 和 R 的调和平均，均衡考量

    "macro" 平均 = 先对每个类别各算一个指标，再取平均。
    这样小类别（图片少的花）和大类别（图片多的花）权重相同，不会"欺负"小众花。
    """

    # 总体准确率
    acc = accuracy_score(y_true, y_pred)

    # 宏平均：每个类别权重相同
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 详细分类报告
    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0,
        target_names=[f"class_{i}" for i in range(num_classes)]
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_loss_acc_curves(history: dict, save_path: str):
    """
    绘制训练/验证的 Loss 和 Accuracy 变化曲线。

    怎么读这张图：
    - Loss 曲线一直下降 -> 模型在学习
    - Loss 降不下去了 -> 接近收敛
    - 训练 Loss 降但验证 Loss 升 -> 过拟合了！
    - 两条曲线趋势一致 -> 训练健康
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：Loss 曲线 ──
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-s", markersize=4, label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── 右图：Accuracy 曲线 ──
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=4,
                 label="Train Acc", linewidth=1.5)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", markersize=4,
                 label="Val Acc", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Loss/Acc 曲线已保存至：{save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_dir: str):
    """
    绘制混淆矩阵（两张：原始计数版 + 按行归一化版）。

    怎么读混淆矩阵：
    - 第 i 行第 j 列 = 真实类别 i 被预测为类别 j 的样本数
    - 对角线上的值 = 预测正确的样本数（越亮越好）
    - 非对角线的亮块 = 容易混淆的类别对

    归一化版本：每个值 = 该行所有样本中被预测为 j 的比例。
    好处：不受各类别样本数量影响，能公平地看混淆情况。
    """

    num_classes = cm.shape[0]

    # ── 图 1：原始计数版 ──
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix ({num_classes} Classes)", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 类别太多时不显示每个格子的数字（看不清）
    if num_classes <= 30:
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=6, color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 混淆矩阵（计数版）已保存至：{save_dir}/confusion_matrix.png")

    # ── 图 2：按行归一化版（每行除以该行总和）──
    # 避免除零
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float32") / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Normalized Confusion Matrix ({num_classes} Classes)", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Recall")

    if num_classes <= 30:
        for i in range(num_classes):
            for j in range(num_classes):
                if cm_norm[i, j] > 0.05:
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 混淆矩阵（归一化版）已保存至：{save_dir}/confusion_matrix_normalized.png")


def analyze_confusion(cm: np.ndarray, num_classes: int, top_k: int = 10):
    """
    分析混淆矩阵，找出最容易混淆的类别对。

    这对实验报告的"误差分析"部分非常重要：
    - 比如 flower_9 和 flower_28 经常互相误判
    - 你可能需要去查数据：这两种花是不是长得特别像？
    - 这就是深度学习模型也会犯的错误类型
    """

    # 把对角线设为 0（只看误分类），然后找最大的 K 个值
    cm_no_diag = cm.copy().astype(float)
    for i in range(num_classes):
        cm_no_diag[i, i] = 0  # 对角线代表"猜对了"，我们只关心猜错的

    # 找出混淆最严重的类别对
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_no_diag[i, j] > 0:
                confused_pairs.append((i, j, int(cm[i, j])))

    # 按混淆次数从大到小排序
    confused_pairs.sort(key=lambda x: -x[2])

    print(f"\n{'─' * 50}")
    print(f"Top-{top_k} 最易混淆类别对（容易误分类的组合）：")
    print(f"{'─' * 50}")
    print(f"{'真实类别':<12} {'误判为':<12} {'次数':<8}")
    print(f"{'─' * 50}")

    for true_id, pred_id, count in confused_pairs[:top_k]:
        print(f"class_{true_id:<6} ->  class_{pred_id:<6} {count} 次")

    print(f"{'─' * 50}\n")

    return confused_pairs[:top_k]  # 返回混淆 Top-K，方便后续保存到 JSON


def save_best_model(model, save_path: str, is_best: bool, best_acc: float, current_acc: float):
    """
    保存最优模型权重。

    策略：只保存在验证集上准确率最高的那个 checkpoint。
    如果当前模型的准确率超过了历史最优，就覆盖保存。
    """
    if is_best:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
            },
            save_path,
        )
        print(f"[SAVE] 新最优模型已保存！Val Acc = {current_acc:.2%}")


# ============================================================================
# 主函数：把上面的所有部件拼接起来
# ============================================================================
def main():
    """
    主流程：
    数据准备 -> 模型搭建 -> 训练循环 -> 指标计算 -> 可视化 -> 报告输出
    """

    # ── 初始化配置 ──
    cfg = Config()
    set_seed(cfg.seed)

    # 创建输出目录（exist_ok=True 表示目录已存在也不报错）
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EfficientNet-B0 图像分类实验")
    print("  数据集：Oxford Flowers 102  |  框架：PyTorch + timm")
    print("  课程：生物医学工程深度学习实验")
    print("=" * 60)
    print(f"\n[INFO] 运行设备：{cfg.device}")
    if cfg.device == "cuda":
        print(f"[INFO] GPU 型号：{torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存大小：{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"[INFO] Batch Size：{cfg.batch_size}")
    print(f"[INFO] 训练轮数：{cfg.num_epochs}")
    print(f"[INFO] 微调方式：{cfg.fine_tune_mode}")

    # ── 阶段 1：数据准备 ──
    train_transform, eval_transform = build_transforms()
    train_set, val_set, test_set = load_flowers102(cfg.data_dir, train_transform, eval_transform)
    num_classes = len(train_set._labels.unique())
    train_loader, val_loader, test_loader = build_dataloaders(train_set, val_set, test_set, cfg)

    # ── 阶段 2：模型搭建 ──
    model = build_model(num_classes, cfg)
    criterion, optimizer, scheduler = build_loss_optimizer(model, cfg)

    # ── 阶段 3：训练循环 ──
    print(f"\n{'=' * 60}")
    print("阶段三：训练循环")
    print(f"{'=' * 60}\n")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"── Epoch {epoch}/{cfg.num_epochs} ", end="")

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device
        )

        # 在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 判断是否是最优模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # 打印 epoch 结果
        current_lr = optimizer.param_groups[0]["lr"]
        best_mark = " [BEST!]" if is_best else ""
        print(
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}{best_mark} | LR: {current_lr:.2e}"
        )

        # 保存最优模型
        save_best_model(model, best_model_path, is_best, best_val_acc, val_acc)

    print(f"\n[OK] 训练完成！最优验证准确率：{best_val_acc:.2%}")

    # ── 阶段 4：指标计算与分析 ──
    print(f"\n{'=' * 60}")
    print("阶段四：指标计算与分析")
    print(f"{'=' * 60}\n")

    # 加载最优模型，在测试集上做最终评估
    print("[INFO] 加载最优模型权重...")
    checkpoint = torch.load(best_model_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] 已加载最优模型（Val Acc = {checkpoint['best_acc']:.2%}）")

    # 在测试集上收集所有预测结果
    print("\n[INFO] 在测试集上运行推理...")
    y_true, y_pred = get_all_predictions(model, test_loader, cfg.device)

    # 计算所有指标
    print("\n[INFO] 计算评价指标...\n")
    results = compute_metrics(y_true, y_pred, num_classes)

    # ── 打印指标 ──
    print(f"{'=' * 50}")
    print("测试集评价指标")
    print(f"{'=' * 50}")
    print(f"  准确率 Accuracy:   {results['accuracy']:.4f} ({results['accuracy']:.2%})")
    print(f"    - 含义：测试集中被正确分类的样本占总样本的比例")
    print(f"    - 反映模型的整体分类能力")
    print()
    print(f"  精确率 Precision:  {results['precision']:.4f} (macro avg)")
    print(f"    - 含义：模型预测为某类别时，有多大概率是对的")
    print(f"    - 该值高说明误报少（宁可漏掉，也不错杀）")
    print()
    print(f"  召回率 Recall:     {results['recall']:.4f} (macro avg)")
    print(f"    - 含义：某个类别的所有样本中，被正确找出来的比例")
    print(f"    - 该值高说明漏报少（宁可错杀，也不错放）")
    print()
    print(f"  F1-Score:          {results['f1_score']:.4f} (macro avg)")
    print(f"    - 含义：精确率和召回率的调和平均数")
    print(f"    - 精确率和召回率需要同时较高，F1 才会高")
    print(f"{'=' * 50}\n")

    # ── 绘制图表 ──
    print("\n[INFO] 绘制实验结果图表...")
    loss_acc_path = os.path.join(cfg.output_dir, "loss_acc_curve.png")
    plot_loss_acc_curves(history, loss_acc_path)
    plot_confusion_matrix(results["confusion_matrix"], cfg.output_dir)

    # ── 分析混淆矩阵 ──
    top_confused = analyze_confusion(results["confusion_matrix"], num_classes)

    # ── 自动保存指标到 JSON（供 generate_report.py 读取，无需手动填数据）──
    import json
    metrics_json_path = os.path.join(cfg.output_dir, "metrics.json")
    metrics_data = {
        "best_val_acc": round(best_val_acc, 4),
        "test_accuracy": round(results["accuracy"], 4),
        "test_precision": round(results["precision"], 4),
        "test_recall": round(results["recall"], 4),
        "test_f1": round(results["f1_score"], 4),
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "fine_tune_mode": cfg.fine_tune_mode,
        "num_classes": num_classes,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "top_confused": [[str(a), str(b), int(c)] for a, b, c in top_confused],
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 实验指标已保存至：{metrics_json_path}")

    # ── 实验总结 ──
    print(f"\n{'=' * 60}")
    print("实验完成！输出文件清单：")
    print(f"{'=' * 60}")
    print(f"  {cfg.output_dir}/loss_acc_curve.png              — 损失/准确率曲线")
    print(f"  {cfg.output_dir}/confusion_matrix.png             — 混淆矩阵（计数版）")
    print(f"  {cfg.output_dir}/confusion_matrix_normalized.png  — 混淆矩阵（归一化版）")
    print(f"  {cfg.output_dir}/best_model.pth                   — 最优模型权重")
    print(f"\n  测试集准确率：{results['accuracy']:.2%}")
    print(f"  测试集 F1：    {results['f1_score']:.4f}")
    print(f"{'=' * 60}\n")

    return results, history


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    main()
