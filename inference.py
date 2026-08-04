# ============================================================================
# 独立推理脚本：输入任意一张图片，输出预测的花卉类别
# 使用方法：
#   python inference.py <图片路径>
# 示例：
#   python inference.py ./test_flower.jpg
# ============================================================================

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# HuggingFace 镜像（和训练脚本保持一致）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import timm
except ImportError:
    print("[ERROR] timm 库未安装！请先运行：pip install timm")
    sys.exit(1)


# ============================================================================
# 模型加载
# ============================================================================
def load_model(model_path: str, num_classes: int = 102, device: str = "cpu"):
    """
    加载训练好的模型权重。

    步骤：
    1. 用 timm 创建和训练时结构完全一样的模型
    2. 加载 .pth 文件中的权重
    3. 切换到评估模式
    """

    # ── 检查模型文件是否存在 ──
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在：{model_path}")
        print("[HELP] 请先运行 train.py 训练模型，或检查模型路径是否正确。")
        sys.exit(1)

    print(f"[INFO] 加载模型：{model_path}")

    # ── 步骤 1：创建模型结构 ──
    # 必须和训练时完全一致：同样的网络、同样的类别数
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,  # 推理时不需要预训练权重，我们会加载自己的
        num_classes=num_classes,
    )

    # ── 步骤 2：加载权重 ──
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] 已加载模型（该模型保存时验证集准确率 = {checkpoint.get('best_acc', 0):.2%}）")

    # ── 步骤 3：切换到评估模式并移至目标设备 ──
    model = model.to(device)
    model.eval()

    return model


# ============================================================================
# 图像预处理
# ============================================================================
def preprocess_image(image_path: str):
    """
    对输入图片做预处理，流程必须和训练时的 eval_transform 完全一致：

    图片 -> Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize

    原理：训练和推理的预处理必须一致，否则数据分布不同，模型会"水土不服"。
    """

    # ── 步骤 1：读取图片 ──
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片文件不存在：{image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")  # 确保是 RGB 三通道
    print(f"[INFO] 原始图片尺寸：{image.size}")

    # ── 步骤 2：预处理流水线（必须和训练时 eval 一致）──
    # ImageNet 的均值和标准差，不能改！
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # 缩放到 256
            transforms.CenterCrop(224),  # 中心裁剪 224×224
            transforms.ToTensor(),  # HWC -> CHW，0-255 -> 0-1
            transforms.Normalize(  # ImageNet 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 变换后形状从 (H,W,3) 变成 (3,224,224)，然后增加 batch 维度 -> (1,3,224,224)
    tensor = transform(image).unsqueeze(0)

    return tensor, image


# ============================================================================
# 推理与预测
# ============================================================================
@torch.no_grad()  # 推理不需要梯度，节省内存
def predict(model, input_tensor, device="cpu"):
    """
    执行前向推理，返回预测结果。

    流程：
    input_tensor -> EfficientNet -> 102 维 logits -> Softmax -> 102 个概率
    取概率最大的那个类别作为预测结果。
    """

    input_tensor = input_tensor.to(device)

    # ── 前向传播 ──
    outputs = model(input_tensor)  # shape: (1, 102)

    # ── Softmax 把 logits 转成概率 ──
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # ── 取概率最大的类别 ──
    top_prob, top_class = torch.max(probabilities, dim=1)

    return top_class.item(), top_prob.item()


# ============================================================================
# 类别名称映射（Flowers 102 没有官方名称，使用类别编号）
# ============================================================================
def get_class_name(class_id: int):
    """
    将类别 ID 转换为可读名称。

    Oxford Flowers 102 的类别没有官方英文名，
    这里简单的用编号表示。你可以根据需要替换为真实花名。

    提示：Flowers 102 官方提供了花名标签文件，
    在 data/flowers-102/ 目录下的 imagelabels.mat 中。
    """
    return f"flower_{class_id}"


# ============================================================================
# 主函数
# ============================================================================
def main():
    # ── 解析命令行参数 ──
    if len(sys.argv) < 2:
        print("用法：python inference.py <图片路径>")
        print("示例：python inference.py ./test_flower.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # ── 配置 ──
    # 模型文件路径（和 train.py 中保存的路径一致）
    model_path = "./output/best_model.pth"
    num_classes = 102  # Flowers 102
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 50)
    print("  EfficientNet-B0 花卉分类推理")
    print("=" * 50)
    print(f"  图片：{image_path}")
    print(f"  设备：{device}")
    print("=" * 50 + "\n")

    # ── 1. 加载模型 ──
    model = load_model(model_path, num_classes, device)

    # ── 2. 预处理图片 ──
    input_tensor, original_image = preprocess_image(image_path)

    # ── 3. 推理 ──
    class_id, confidence = predict(model, input_tensor, device)

    # ── 4. 输出结果 ──
    class_name = get_class_name(class_id)

    print(f"\n{'=' * 50}")
    print(f"  预测结果")
    print(f"{'=' * 50}")
    print(f"  类别 ID：{class_id}")
    print(f"  类别名称：{class_name}")
    print(f"  置信度：  {confidence:.4f} ({confidence:.2%})")
    print(f"{'=' * 50}\n")

    # ── 5. 如果有 GPU，显示 Top-5 预测 ──
    # 把输入再跑一次，取 Top-5 概率最高的类别（方便分析）
    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    top5_prob, top5_class = torch.topk(probs, k=min(5, num_classes), dim=1)

    print("Top-5 预测结果：")
    print(f"{'排名':<6} {'类别':<16} {'置信度':<12}")
    print("-" * 36)
    for rank, (cls_id, prob) in enumerate(zip(top5_class[0].tolist(), top5_prob[0].tolist()), 1):
        marker = " <-- 最佳" if rank == 1 else ""
        print(f"  {rank:<4} {get_class_name(cls_id):<16} {prob:.4f} ({prob:.2%}){marker}")
    print()


if __name__ == "__main__":
    main()
