# ============================================================================
# 实验报告自动生成器 — 读取训练输出，生成可视化的 HTML 实验报告
# 使用方法：python generate_report.py
# 前提：先运行 train.py 生成 output/ 目录下的图片和模型
# ============================================================================

import os
import base64
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# 辅助函数：将图片转为 base64 嵌入 HTML（报告可以独立打开，不依赖外部图片）
# ============================================================================
def image_to_base64(image_path: str) -> str:
    """将图片文件转为 base64 字符串，方便嵌入 HTML"""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# 从 train.py 自动生成的 metrics.json 读取实验数据
# 跑完 train.py 后直接运行本脚本即可，不需要手动改任何数值
# ============================================================================
def load_report_data(output_dir: str) -> dict:
    """
    自动读取 train.py 保存的 metrics.json。
    如果还没跑 train.py（没有 JSON 文件），则使用占位数据先生成报告框架。
    """

    # 默认数据（占位用，跑完 train.py 会被自动覆盖）
    defaults = {
        "best_val_acc": 0.0,
        "test_accuracy": 0.0,
        "test_precision": 0.0,
        "test_recall": 0.0,
        "test_f1": 0.0,
        "num_epochs": 30,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "fine_tune_mode": "full",
        "num_classes": 102,
        "train_samples": 1020,
        "val_samples": 1020,
        "test_samples": 6149,
        "top_confused": [],
        # 环境信息
        "gpu": "NVIDIA GeForce RTX 4060 8GB",
        "cpu": "Intel Core i7-14650HX",
        "os": "Windows 11",
        "python_version": "3.x",
        "pytorch_version": "2.x",
        "author": "",
        "student_id": "",
    }

    json_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # 用 JSON 中的值覆盖默认值
        defaults.update(saved)
        print(f"[INFO] 已从 {json_path} 自动读取实验指标")
    else:
        print(f"[WARN] 未找到 {json_path}，使用占位数据")
        print(f"[HELP] 请先运行 python train.py 训练模型")

    return defaults


def generate_html_report(data: dict, output_dir: str) -> str:
    """生成完整的 HTML 实验报告"""

    # 读取图片
    loss_acc_b64 = image_to_base64(os.path.join(output_dir, "loss_acc_curve.png"))
    cm_b64 = image_to_base64(os.path.join(output_dir, "confusion_matrix.png"))
    cm_norm_b64 = image_to_base64(os.path.join(output_dir, "confusion_matrix_normalized.png"))

    # 生成混淆表格行
    confused_rows = ""
    for i, (true_cls, pred_cls, count) in enumerate(data["top_confused"], 1):
        confused_rows += f"""
        <tr>
            <td>{i}</td>
            <td><code>{true_cls}</code></td>
            <td><code>{pred_cls}</code></td>
            <td><span class="badge badge-danger">{count} 次</span></td>
            <td>可能形态或颜色高度相似</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>基于 EfficientNet-B0 的图像分类实验报告</title>
<style>
/* ============================================================
   报告整体样式 — 学术风格，清晰易读
   ============================================================ */
:root {{
    --primary: #1a56db;
    --primary-light: #e8f0fe;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --radius: 10px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.8;
    font-size: 15px;
}}

/* ---- 封面 ---- */
.cover {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1a56db 100%);
    color: #fff;
    padding: 80px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.cover::before {{
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 70%, rgba(255,255,255,0.05) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(255,255,255,0.08) 0%, transparent 50%);
}}
.cover h1 {{
    font-size: 2.4em;
    font-weight: 700;
    margin-bottom: 12px;
    position: relative;
    letter-spacing: 1px;
}}
.cover .subtitle {{
    font-size: 1.1em;
    opacity: 0.85;
    margin-bottom: 30px;
    position: relative;
}}
.cover .meta {{
    display: flex;
    justify-content: center;
    gap: 40px;
    flex-wrap: wrap;
    position: relative;
    font-size: 0.95em;
    opacity: 0.75;
}}
.cover .meta span {{ padding: 6px 20px; border: 1px solid rgba(255,255,255,0.25); border-radius: 20px; }}

/* ---- 导航 ---- */
.toc {{
    max-width: 900px;
    margin: 30px auto;
    padding: 30px 40px;
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}}
.toc h2 {{ font-size: 1.3em; margin-bottom: 16px; color: var(--primary); }}
.toc ol {{ padding-left: 24px; }}
.toc li {{ margin: 6px 0; }}
.toc a {{ color: var(--primary); text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}

/* ---- 章节 ---- */
.container {{ max-width: 960px; margin: 0 auto; padding: 0 20px; }}
.section {{
    margin: 30px auto;
    padding: 36px 40px;
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}}
.section h2 {{
    font-size: 1.5em;
    color: var(--primary);
    border-bottom: 3px solid var(--primary);
    padding-bottom: 10px;
    margin-bottom: 24px;
}}
.section h3 {{
    font-size: 1.15em;
    color: var(--text);
    margin: 20px 0 10px;
}}
.section p {{ margin: 10px 0; color: var(--text); }}

/* ---- 概念卡片（知识点）---- */
.concept-card {{
    background: var(--primary-light);
    border-left: 4px solid var(--primary);
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.95em;
}}
.concept-card strong {{ color: var(--primary); }}

/* ---- 表格 ---- */
.table-wrap {{ overflow-x: auto; margin: 16px 0; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92em;
}}
thead th {{
    background: #f1f5f9;
    color: var(--text);
    font-weight: 600;
    padding: 12px 14px;
    text-align: left;
    border-bottom: 2px solid var(--border);
}}
tbody td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
}}
tbody tr:hover {{ background: #f8fafc; }}
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}}
.badge-success {{ background: #d1fae5; color: #065f46; }}
.badge-danger {{ background: #fee2e2; color: #991b1b; }}
.badge-info {{ background: #dbeafe; color: #1e40af; }}

/* ---- 指标仪表盘 ---- */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 20px 0;
}}
.metric-card {{
    background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    box-shadow: var(--shadow);
}}
.metric-card .value {{
    font-size: 2em;
    font-weight: 700;
    color: var(--primary);
}}
.metric-card .label {{ font-size: 0.85em; color: var(--text-secondary); margin-top: 4px; }}
.metric-card .hint {{ font-size: 0.78em; color: var(--text-secondary); margin-top: 6px; line-height: 1.4; }}

/* ---- 架构图 ---- */
.arch-diagram {{
    background: #f8fafc;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin: 20px 0;
    text-align: center;
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 0.85em;
    line-height: 2;
    overflow-x: auto;
    white-space: pre;
}}

/* ---- 流程图 ---- */
.flow-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin: 16px 0;
}}
.flow-box {{
    background: var(--primary);
    color: #fff;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9em;
    text-align: center;
}}
.flow-box.gray {{ background: #64748b; }}
.flow-box.green {{ background: #059669; }}
.flow-box.orange {{ background: #d97706; }}
.flow-arrow {{
    font-size: 1.3em;
    color: var(--text-secondary);
    font-weight: bold;
}}

/* ---- 图片区 ---- */
.figure {{
    margin: 24px 0;
    text-align: center;
}}
.figure img {{
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}}
.figure .caption {{
    margin-top: 8px;
    font-size: 0.88em;
    color: var(--text-secondary);
}}

/* ---- 提示框 ---- */
.callout {{
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 8px;
    font-size: 0.93em;
}}
.callout-info {{ background: #dbeafe; border-left: 4px solid #3b82f6; color: #1e40af; }}
.callout-warning {{ background: #fef3c7; border-left: 4px solid #f59e0b; color: #92400e; }}
.callout-success {{ background: #d1fae5; border-left: 4px solid #10b981; color: #065f46; }}

/* ---- 页脚 ---- */
.footer {{
    text-align: center;
    padding: 30px;
    color: var(--text-secondary);
    font-size: 0.85em;
}}

/* ---- 打印样式 ---- */
@media print {{
    body {{ background: #fff; }}
    .section {{ box-shadow: none; border: 1px solid #ddd; page-break-inside: avoid; }}
    .cover {{ background: #1a56db !important; -webkit-print-color-adjust: exact; }}
}}

/* ---- 响应式 ---- */
@media (max-width: 768px) {{
    .cover h1 {{ font-size: 1.5em; }}
    .section {{ padding: 20px; }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<!-- ================================================================
     封面
     ================================================================ -->
<div class="cover">
    <h1>基于 EfficientNet-B0 的图像分类实验</h1>
    <div class="subtitle">生物医学工程 · 深度学习课程实验报告</div>
    <div class="meta">
        <span>数据集：Oxford Flowers 102</span>
        <span>框架：PyTorch + timm</span>
        <span>{datetime.now().strftime('%Y 年 %m 月')}</span>
    </div>
</div>

<!-- ================================================================
     目录
     ================================================================ -->
<div class="toc">
    <h2>目 录</h2>
    <ol>
        <li><a href="#s1">实验原理 — CNN、EfficientNet、迁移学习</a></li>
        <li><a href="#s2">实验环境 — 硬件与软件配置</a></li>
        <li><a href="#s3">数据集介绍 — Oxford Flowers 102</a></li>
        <li><a href="#s4">模型讲解 — EfficientNet-B0 结构</a></li>
        <li><a href="#s5">实验参数 — 超参数配置</a></li>
        <li><a href="#s6">实验结果 — 指标、曲线、混淆矩阵</a></li>
        <li><a href="#s7">结果分析 — 误差分析与讨论</a></li>
        <li><a href="#s8">实验总结 — 收获与展望</a></li>
    </ol>
</div>

<div class="container">

<!-- ================================================================
     第一章：实验原理
     ================================================================ -->
<div class="section" id="s1">
    <h2>一、实验原理</h2>

    <h3>1.1 卷积神经网络（CNN）基本组成</h3>
    <p>卷积神经网络是深度学习在图像领域最核心的架构。它模仿人类视觉系统，从局部到全局逐层提取特征：</p>

    <!-- CNN 架构图 -->
    <div class="arch-diagram">
┌─────────────────────────────────────────────────────────────────────┐
│                        CNN 图像分类流程                               │
│                                                                     │
│  [输入图片]    [卷积层]      [池化层]      [卷积层]      [全连接层]    │
│  224×224×3  →  提取边缘   →  降采样    →  提取纹理  →   分类输出    │
│               (3×3 滤波器)  (2×2 max)    (3×3 滤波器)  (102 类)    │
│                                                                     │
│  特征层次：  低层 → 边缘/颜色  →  中层 → 纹理/形状  →  高层 → 语义   │
└─────────────────────────────────────────────────────────────────────┘
    </div>

    <div class="concept-card">
        <strong>卷积层</strong> — 用一个小的滑动窗口（如 3×3）扫描整张图，检测局部特征（边缘、纹理）。<br>
        <strong>池化层</strong> — 把特征图缩小（如 2×2 取最大值），减少计算量，同时保留重要信息。<br>
        <strong>全连接层</strong> — 把所有特征展平，映射到最终类别概率。
    </div>

    <h3>1.2 EfficientNet 简介</h3>
    <p>EfficientNet 是 Google 在 2019 年提出的高效卷积网络。它的核心思想是：<strong>不单独增加深度或宽度，而是用一个复合系数同时缩放三个维度</strong>——网络深度、特征宽度、输入分辨率。通过神经架构搜索（NAS）找到最优的缩放比例。</p>

    <div class="arch-diagram">
                     EfficientNet 复合缩放策略

    传统做法：                       EfficientNet 做法：
    只加深 ┃                        同时缩放三维度
    只加宽 ━━                       深度 × 宽度 × 分辨率
    只放大分辨率 ☐                  = 效率更高、参数更少

    B0（基准） → B1 → B2 → ... → B7（最大）
    ↑ 本实验使用 B0，最轻量版本（5.3M 参数）
    </div>

    <h3>1.3 迁移学习原理</h3>
    <p>迁移学习的核心假设：<strong>在 ImageNet（100 万张自然图像，1000 类）上学到的底层特征——边缘、纹理、形状——对花卉识别同样有用。</strong>我们不需要从零开始学"怎么看图"，只需要微调"怎么看花"。</p>

    <div class="arch-diagram">
    源域 (ImageNet)                          目标域 (Flowers 102)
    ┌─────────────────────┐                 ┌─────────────────────┐
    │ 1000 类自然图像      │    迁移        │ 102 种花卉           │
    │ 🐱 🐶 🚗 ✈️ ...   │  ==========>   │ 🌸 🌺 🌻 🌹 ...    │
    │ 100 万+ 张训练图片   │   复用特征     │ 每类仅 10 张          │
    └─────────────────────┘                 └─────────────────────┘
    </div>

    <h3>1.4 两种微调策略</h3>
    <table>
        <thead>
            <tr><th>策略</th><th>做法</th><th>适用场景</th><th>优点</th><th>缺点</th></tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="badge badge-info">冻结特征层</span></td>
                <td>只训练最后的分类头，backbone 参数不变</td>
                <td>数据极少、域相似度高</td>
                <td>快、显存少、不易过拟合</td>
                <td>模型适配能力有限</td>
            </tr>
            <tr>
                <td><span class="badge badge-success">全参数微调</span></td>
                <td>所有参数都参与训练</td>
                <td>数据较多、域有差异</td>
                <td>性能上限更高</td>
                <td>慢、需要更多正则化</td>
            </tr>
        </tbody>
    </table>
    <p><strong>本实验选择"全参数微调"</strong>，因为 Flowers 102 和 ImageNet 同属自然图像域，全参数微调能让模型更充分地适配花卉特征。</p>
</div>

<!-- ================================================================
     第二章：实验环境
     ================================================================ -->
<div class="section" id="s2">
    <h2>二、实验环境</h2>

    <table>
        <thead><tr><th>类别</th><th>配置</th><th>说明</th></tr></thead>
        <tbody>
            <tr><td>CPU</td><td>{data["cpu"]}</td><td>高性能移动处理器</td></tr>
            <tr><td>GPU</td><td>{data["gpu"]}</td><td>支持 CUDA 加速训练</td></tr>
            <tr><td>操作系统</td><td>{data["os"]}</td><td></td></tr>
            <tr><td>Python</td><td>{data["python_version"]}</td><td></td></tr>
            <tr><td>PyTorch</td><td>{data["pytorch_version"]}</td><td>深度学习框架</td></tr>
            <tr><td>timm</td><td>0.9.x</td><td>预训练模型库（提供 EfficientNet）</td></tr>
            <tr><td>scikit-learn</td><td>1.x</td><td>机器学习指标计算</td></tr>
            <tr><td>matplotlib</td><td>3.x</td><td>图表绘制</td></tr>
        </tbody>
    </table>

    <div class="callout callout-info">
        <strong>为什么用 Y7000P 笔记本？</strong> RTX 4060 8GB 显存足以运行 EfficientNet-B0 + batch_size=16 的训练。相比云端服务器，本地训练更直观，方便调试和理解每一步发生了什么。
    </div>
</div>

<!-- ================================================================
     第三章：数据集介绍
     ================================================================ -->
<div class="section" id="s3">
    <h2>三、数据集介绍 — Oxford Flowers 102</h2>

    <table>
        <thead><tr><th>属性</th><th>值</th></tr></thead>
        <tbody>
            <tr><td>类别数</td><td><strong>{data["num_classes"]}</strong> 种花卉</td></tr>
            <tr><td>总图片数</td><td>8,189 张</td></tr>
            <tr><td>训练集</td><td>{data["train_samples"]} 张（每类 10 张）</td></tr>
            <tr><td>验证集</td><td>{data["val_samples"]} 张（每类 10 张）</td></tr>
            <tr><td>测试集</td><td>{data["test_samples"]} 张（分布不均）</td></tr>
            <tr><td>图片尺寸</td><td>不统一（→ 预处理统一为 224×224）</td></tr>
            <tr><td>来源</td><td>牛津大学 VGG 实验室，2008 年</td></tr>
        </tbody>
    </table>

    <h3>3.1 数据预处理流程</h3>
    <div class="flow-row">
        <div class="flow-box gray">原始图片<br><small>尺寸不一</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box">Resize<br><small>256×256</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box">CenterCrop<br><small>224×224</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box green">数据增强<br><small>翻转/旋转/调色</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box">ToTensor<br><small>0-255→0-1</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box orange">Normalize<br><small>ImageNet 标准化</small></div>
    </div>

    <h3>3.2 数据增强方法及作用</h3>
    <table>
        <thead><tr><th>增强方法</th><th>参数</th><th>作用</th></tr></thead>
        <tbody>
            <tr><td>RandomResizedCrop</td><td>224, scale=(0.7, 1.0)</td><td>模拟花朵在画面中大小和位置不同</td></tr>
            <tr><td>RandomHorizontalFlip</td><td>p=0.5</td><td>花朵朝向不影响类别判断</td></tr>
            <tr><td>RandomRotation</td><td>±20°</td><td>模拟拍摄角度倾斜</td></tr>
            <tr><td>ColorJitter</td><td>brightness/contrast/saturation=0.3</td><td>模拟不同光照、天气、设备</td></tr>
            <tr><td>Normalize</td><td>ImageNet mean/std</td><td>与预训练模型输入分布对齐</td></tr>
        </tbody>
    </table>

    <div class="callout callout-warning">
        <strong>为什么必须做数据增强？</strong> 训练集每类只有 10 张图。如果不增强，模型运行几轮就能"背下"所有训练图片，在测试集上表现极差——这叫<strong>过拟合</strong>。增强相当于给每张图生成无数种变体，迫使模型学习"花的本质特征"而不是"某张特定照片的样子"。
    </div>
</div>

<!-- ================================================================
     第四章：模型讲解
     ================================================================ -->
<div class="section" id="s4">
    <h2>四、模型讲解 — EfficientNet-B0</h2>

    <h3>4.1 模型整体架构</h3>
    <div class="arch-diagram">
Input                    EfficientNet-B0 Backbone              Classifier
┌──────────┐    ┌──────────────────────────────────┐    ┌──────────────┐
│  Image   │    │  Stem → MBConv1 → MBConv6 × N    │    │  Global Avg  │
│ 3×224×224│───>│  (卷积 stem)  (倒残差块 × 7 组)  │───>│  Pooling     │
│          │    │  共 16 个 MBConv Block             │    │     ↓        │
│          │    │  输出 1280 维特征向量              │    │  Dropout(0.2)│
└──────────┘    └──────────────────────────────────┘    │     ↓        │
                                                        │  FC(1280→102)│
                                                        │  → Softmax   │
                                                        └──────────────┘
    </div>

    <div class="concept-card">
        <strong>MBConv（移动倒残差瓶颈卷积）</strong>是 EfficientNet 的核心模块：先用 1×1 卷积升维 → 3×3 逐通道卷积 → 压缩-激励（SE）模块 → 1×1 降维。相比普通卷积，参数量和计算量大幅减少。<br>
        <strong>EfficientNet-B0</strong> 是系列中最轻量的版本，仅 5.3M 参数，适合教学环境。
    </div>

    <h3>4.2 分类头替换</h3>
    <p>ImageNet 预训练模型的最后一层是 1000 个神经元（对应 1000 类）。Flowers 102 只有 102 类，我们通过 <code>timm.create_model(..., num_classes=102)</code> 自动替换为 102 个神经元。新层权重随机初始化，需要在花卉数据上充分训练。</p>

    <h3>4.3 训练流程中的反向传播</h3>
    <div class="arch-diagram">
    每个 batch 的 5 步循环（batch_size={data["batch_size"]}）：

    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ ① zero_grad  │───>│ ② forward    │───>│ ③ loss       │───>│ ④ backward   │───>│ ⑤ step       │
    │ 清零旧梯度    │    │ 前向传播      │    │ 计算误差      │    │ 反向求梯度    │    │ 更新参数      │
    │              │    │ model(imgs)  │    │ CrossEntropy │    │ loss.backward│    │ optimizer.   │
    │              │    │ → 102 维输出 │    │ (预测 vs 标签)│    │ (每个参数的   │    │ step()       │
    │              │    │              │    │              │    │  偏导数)     │    │ (梯度下降)    │
    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    </div>
</div>

<!-- ================================================================
     第五章：实验参数
     ================================================================ -->
<div class="section" id="s5">
    <h2>五、实验参数</h2>

    <table>
        <thead><tr><th>参数</th><th>取值</th><th>说明</th></tr></thead>
        <tbody>
            <tr><td>batch_size</td><td><strong>{data["batch_size"]}</strong></td><td>受笔记本 8GB 显存限制，16 是安全值</td></tr>
            <tr><td>num_epochs</td><td><strong>{data["num_epochs"]}</strong></td><td>30 轮后 loss 基本收敛</td></tr>
            <tr><td>learning_rate</td><td><strong>{data["learning_rate"]}</strong></td><td>迁移学习推荐范围 1e-4 ~ 1e-5</td></tr>
            <tr><td>weight_decay</td><td><strong>1e-4</strong></td><td>L2 正则化，防止权重过大</td></tr>
            <tr><td>优化器</td><td><strong>AdamW</strong></td><td>Adam 改进版 + 权重衰减修正</td></tr>
            <tr><td>损失函数</td><td><strong>CrossEntropyLoss</strong></td><td>多分类标准损失 = Softmax + NLL</td></tr>
            <tr><td>学习率调度</td><td><strong>CosineAnnealingLR</strong></td><td>余弦退火：初期快速收敛，后期精细微调</td></tr>
            <tr><td>微调方式</td><td><strong>{data["fine_tune_mode"]}（全参数）</strong></td><td>所有参数参与训练</td></tr>
            <tr><td>图片尺寸</td><td><strong>224×224</strong></td><td>EfficientNet-B0 标准输入尺寸</td></tr>
            <tr><td>预训练权重</td><td><strong>ImageNet-1K</strong></td><td>100 万张自然图像的预训练权重</td></tr>
        </tbody>
    </table>
</div>

<!-- ================================================================
     第六章：实验结果（核心章节）
     ================================================================ -->
<div class="section" id="s6">
    <h2>六、实验结果</h2>

    <h3>6.1 核心指标一览</h3>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="value">{data["test_accuracy"]:.2%}</div>
            <div class="label">准确率 Accuracy</div>
            <div class="hint">测试集中猜对了多少</div>
        </div>
        <div class="metric-card">
            <div class="value">{data["test_precision"]:.2%}</div>
            <div class="label">精确率 Precision</div>
            <div class="hint">预测"是玫瑰"时，多少真的对</div>
        </div>
        <div class="metric-card">
            <div class="value">{data["test_recall"]:.2%}</div>
            <div class="label">召回率 Recall</div>
            <div class="hint">所有真玫瑰，找到了多少</div>
        </div>
        <div class="metric-card">
            <div class="value">{data["test_f1"]:.2%}</div>
            <div class="label">F1-Score</div>
            <div class="hint">P 和 R 的调和平均，均衡考量</div>
        </div>
    </div>

    <div class="callout callout-info">
        <strong>怎么理解这四个指标？</strong> 假设模型是个安检员，要从 102 种花里找出"玫瑰"：<br>
        <strong>Accuracy</strong> = 总共猜对了多少（对玫瑰的和对非玫瑰的都算）<br>
        <strong>Precision</strong> = 说"这是玫瑰"时，多大把握（误报少 = 精确率高）<br>
        <strong>Recall</strong> = 所有真玫瑰，找到多少（漏报少 = 召回率高）<br>
        <strong>F1</strong> = Precision 和 Recall 都高时 F1 才高（综合评价指标）<br>
        本实验全部使用 <strong>macro 平均</strong>：102 个类别各算各的再取平均，不会因为某些类别图片多就"欺负"小类别。
    </div>

    <h3>6.2 训练曲线</h3>
    <div class="figure">
        <img src="data:image/png;base64,{loss_acc_b64}" alt="Loss 和 Accuracy 曲线"
             style="max-width:100%;" onerror="this.alt='请先运行 train.py 生成图片'">
        <div class="caption">图 1：训练/验证 Loss（左）与 Accuracy（右）随 Epoch 变化曲线</div>
    </div>

    <div class="callout callout-success">
        <strong>如何读这张图：</strong><br>
        ✓ Loss 持续下降 → 模型在学习<br>
        ✓ 两条 Loss 趋势一致 → 没有严重过拟合<br>
        ✓ Accuracy 趋于平稳 → 模型接近收敛<br>
        ⚠ 如果训练 Loss 降但验证 Loss 升 → 过拟合信号
    </div>

    <h3>6.3 混淆矩阵</h3>
    <div class="figure">
        <img src="data:image/png;base64,{cm_norm_b64}" alt="归一化混淆矩阵"
             style="max-width:95%;" onerror="this.alt='请先运行 train.py 生成图片'">
        <div class="caption">图 2：归一化混淆矩阵（每行和为 1，颜色越亮 = 召回率越高）</div>
    </div>

    <div class="callout callout-warning">
        <strong>如何读混淆矩阵：</strong><br>
        ✓ 对角线很亮（黄色）= 大多数类别识别准确<br>
        ⚠ 非对角线亮块 = 容易混淆的类别对<br>
        → 例如 class_9 和 class_28 之间有明显混淆，说明这两种花长得特别像
    </div>

    <h3>6.4 Top-5 最易混淆类别对</h3>
    <table>
        <thead><tr><th>排名</th><th>真实类别</th><th>误判为</th><th>次数</th><th>可能原因</th></tr></thead>
        <tbody>{confused_rows}</tbody>
    </table>

    <h3>6.5 原始计数混淆矩阵</h3>
    <div class="figure">
        <img src="data:image/png;base64,{cm_b64}" alt="混淆矩阵（计数版）"
             style="max-width:95%;" onerror="this.alt='请先运行 train.py 生成图片'">
        <div class="caption">图 3：原始计数混淆矩阵（格子里的数字 = 被误判的图片张数）</div>
    </div>
</div>

<!-- ================================================================
     第七章：结果分析
     ================================================================ -->
<div class="section" id="s7">
    <h2>七、结果分析</h2>

    <h3>7.1 模型表现评估</h3>
    <p>本实验在 Oxford Flowers 102 测试集（{data["test_samples"]} 张图）上取得了 <strong>{data["test_accuracy"]:.2%}</strong> 的准确率，F1-Score 达到 <strong>{data["test_f1"]:.2%}</strong>。考虑到训练集中每类仅有 10 张样本，这个结果是合理的。</p>

    <h3>7.2 混淆分析</h3>
    <p>从混淆矩阵可以看出：</p>
    <ul>
        <li><strong>大部分类别</strong>识别准确率高（对角线明亮），说明模型能够有效区分 102 种花卉</li>
        <li><strong>少数类别对</strong>存在明显混淆，通常是因为：花朵外观相似（颜色、形状）、拍摄角度导致特征变化、或某些类别训练样本代表性不足</li>
    </ul>

    <h3>7.3 实验局限性</h3>
    <table>
        <thead><tr><th>局限</th><th>影响</th><th>改进方向</th></tr></thead>
        <tbody>
            <tr><td>训练数据极少（每类 10 张）</td><td>模型难以学到充分的类内变化</td><td>使用更大数据集，或多任务学习</td></tr>
            <tr><td>未做系统超参数搜索</td><td>当前参数未必是最优组合</td><td>网格搜索或贝叶斯优化</td></tr>
            <tr><td>仅用基础数据增强</td><td>数据多样性有限</td><td>引入 CutMix、MixUp 等高级增强</td></tr>
            <tr><td>仅测试了 EfficientNet-B0</td><td>无法判断是否是最佳架构选择</td><td>对比 ResNet50、ViT 等其他模型</td></tr>
            <tr><td>未对比两种微调策略</td><td>不确定 freeze vs full 的差距</td><td>做对比实验，分析适用场景</td></tr>
        </tbody>
    </table>
</div>

<!-- ================================================================
     第八章：实验总结
     ================================================================ -->
<div class="section" id="s8">
    <h2>八、实验总结</h2>

    <h3>8.1 完成工作</h3>
    <p>本实验使用 ImageNet 预训练的 EfficientNet-B0 模型，在 Oxford Flowers 102 花卉数据集上完成了迁移学习。实验涵盖完整流程：数据下载与增强 → 模型搭建与微调 → 训练与验证 → 测试集评估 → 指标计算与可视化，最终在 {data["test_samples"]} 张测试图片上取得了 <strong>{data["test_accuracy"]:.2%}</strong> 的分类准确率。</p>

    <h3>8.2 核心收获</h3>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="value" style="font-size:1.1em;">迁移学习</div>
            <div class="hint">理解了"预训练+微调"的完整流程，知道何时冻结、何时全训练</div>
        </div>
        <div class="metric-card">
            <div class="value" style="font-size:1.1em;">数据增强</div>
            <div class="hint">理解了小样本场景下数据增强的必要性和各类增强的作用</div>
        </div>
        <div class="metric-card">
            <div class="value" style="font-size:1.1em;">评价指标</div>
            <div class="hint">能区分 Acc/Precision/Recall/F1 各自回答什么问题</div>
        </div>
        <div class="metric-card">
            <div class="value" style="font-size:1.1em;">混淆矩阵</div>
            <div class="hint">学会了从混淆矩阵中发现模型的弱点，定位混淆类别对</div>
        </div>
    </div>

    <h3>8.3 未来改进方向</h3>
    <div class="flow-row">
        <div class="flow-box gray">ResNet/ViT<br>多模型对比</div>
        <div class="flow-arrow">+</div>
        <div class="flow-box gray">MixUp/CutMix<br>高级增强</div>
        <div class="flow-arrow">+</div>
        <div class="flow-box gray">freeze vs full<br>策略对比</div>
        <div class="flow-arrow">+</div>
        <div class="flow-box green">更完善的<br>实验体系</div>
    </div>
</div>

</div><!-- /.container -->

<div class="footer">
    <p>本报告由 generate_report.py 自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>基于 PyTorch + timm + EfficientNet-B0  |  Oxford Flowers 102 数据集</p>
</div>

</body>
</html>"""

    return html


# ============================================================================
# 主函数
# ============================================================================
def main():
    output_dir = "./output"
    report_path = "./实验报告.html"

    # 确保 output 目录存在
    if not os.path.exists(output_dir):
        print(f"[ERROR] 未找到 {output_dir}/ 目录，请先运行 train.py")
        return

    # 检查图片文件
    required_files = ["loss_acc_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        print(f"[WARN] 缺少以下图片文件：{missing}")
        print("[INFO] 将继续生成报告，但对应图片位会留空")

    print("[INFO] 正在生成可视化 HTML 实验报告...")

    # 自动从 train.py 输出的 metrics.json 读取数据，无需手动填
    report_data = load_report_data(output_dir)
    html = generate_html_report(report_data, output_dir)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] 报告已生成：{os.path.abspath(report_path)}")
    print(f"[INFO] 用浏览器打开该文件即可查看完整报告")
    print(f"\n工作流：python train.py  →  python generate_report.py  →  浏览器打开 实验报告.html")
    print(f"指标数据自动从 output/metrics.json 读取，全程无需手动填数值。")


if __name__ == "__main__":
    main()
