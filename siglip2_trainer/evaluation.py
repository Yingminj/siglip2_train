import os
import re
import json
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from .logging_utils import NumpyEncoder
from .augmentation import apply_background_mask


def compute_and_save_class_centers(model, processor, train_dataset, config, epoch):
    """
    计算并保存每个类别的中心向量（使用图像编码的均值）

    策略：读取每个类别所有训练图片，使用图像编码器编码，计算均值作为类别中心
    同时处理 graph_info.json：复制到评估目录并更新特征中心

    Args:
        model: 当前训练的模型
        processor: processor
        train_dataset: 训练数据集（用于获取类别信息和图片路径）
        config: 配置
        epoch: 当前epoch

    Returns:
        class_centers: dict {class_name: center_vector}
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"【计算类别中心向量（图像编码均值）】Epoch {epoch}")
    print(f"{'='*60}")

    # 从训练集获取类别列表
    class_list = sorted(train_dataset.classes, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    # 按类别分组训练集样本（只用训练集图片计算中心，避免数据泄漏）
    class_image_paths = {c: [] for c in class_list}
    for sample in train_dataset.samples:
        class_image_paths[sample['class']].append(sample['image_path'])

    print(f"发现 {len(class_list)} 个类别")
    print(f"使用训练集样本计算中心（共 {len(train_dataset.samples)} 张图片）")

    class_centers = {}

    # 批量处理图片
    batch_size = 32

    with torch.no_grad():
        for category in class_list:
            # 只使用训练集中该类别的图片
            image_paths = class_image_paths[category]

            if len(image_paths) == 0:
                print(f"  警告: 类别 {category} 在训练集中没有图片")
                continue

            print(f"  处理类别 {category}: {len(image_paths)} 张图片（训练集）")

            # 批量编码图片
            features_list = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []

                # 加载图片
                mask_ratio = getattr(config, 'BACKGROUND_MASK_RATIO', 0.0)
                for img_path in batch_paths:
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            img.load()
                        if mask_ratio > 0:
                            img = apply_background_mask(img, mask_ratio)
                        batch_images.append(img)
                    except Exception as e:
                        print(f"    警告: 无法加载图片 {img_path}: {e}")
                        continue

                if len(batch_images) == 0:
                    continue

                # 使用 processor 预处理
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt"
                )
                pixel_values = inputs['pixel_values'].to(config.DEVICE)

                # V2: patch-level attention pooler
                if hasattr(model, 'encode'):
                    image_embeds = model.encode(pixel_values)  # [B, D] already L2-normalized
                elif hasattr(model, 'encode_views'):
                    image_embeds = model.encode_views(pixel_values)
                else:
                    # V1 fallback: pooler_output
                    vision_outputs = model.vision_model(pixel_values=pixel_values)
                    image_embeds = vision_outputs.pooler_output  # [B, D]
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                features_list.append(image_embeds.cpu().numpy())

                if (i // batch_size + 1) % 5 == 0:
                    print(f"    已处理 {min(i + batch_size, len(image_paths))}/{len(image_paths)} 张图片")

            # 计算该类别的中心向量（均值）
            if len(features_list) > 0:
                features = np.vstack(features_list)  # [N, D]
                center = np.mean(features, axis=0)    # [D,]
                center = center / np.linalg.norm(center)  # L2归一化

                class_centers[category] = center
                print(f"  ✓ 类别 {category} 中心特征计算完成 (特征维度: {center.shape})")
            else:
                print(f"  ✗ 类别 {category} 没有有效的特征")

    # 处理 graph_info.json
    print(f"\n{'='*60}")
    print(f"【处理 graph_info.json】")
    print(f"{'='*60}")

    # 1. 定位原始 graph_info.json 文件
    # 优先在 IMAGE_ROOT 目录下查找，找不到再去父目录
    original_graph_path = os.path.join(train_dataset.image_root, "graph_info.json")
    if not os.path.exists(original_graph_path):
        data_dir = os.path.dirname(train_dataset.image_root)
        original_graph_path = os.path.join(data_dir, "graph_info.json")

    if not os.path.exists(original_graph_path):
        print(f"  ⚠️  警告: 未找到原始 graph_info.json: {original_graph_path}")
        print(f"  跳过 graph_info.json 的处理")
        model.train()
        return class_centers

    print(f"  ✓ 找到原始 graph_info.json: {original_graph_path}")

    # 2. 创建评估结果目录
    eval_dir = os.path.join(config.MODEL_DIR, 'classification_viz')
    os.makedirs(eval_dir, exist_ok=True)

    # 3. 复制 graph_info.json 并重命名
    graph_copy_path = os.path.join(eval_dir, f'graph_info_{epoch}.json')
    import shutil
    shutil.copy2(original_graph_path, graph_copy_path)
    print(f"  ✓ 已复制 graph_info.json 到: {graph_copy_path}")

    # 4. 读取并更新 graph_info.json，添加特征中心
    try:
        with open(graph_copy_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        nodes = graph_data.get('nodes', [])

        # 为每个节点添加/更新 center_feature_siglip2
        # 建立 node_id(数字) -> 排序后类别名 的映射
        sorted_categories = sorted(class_centers.keys(),
                                   key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        updated_count = 0
        for node in nodes:
            node_id = node.get('node_id')
            if node_id is None:
                continue

            try:
                # 提取数字部分（兼容字符串和整数）
                if isinstance(node_id, str):
                    node_id_int = int(re.search(r'\d+', node_id).group())
                else:
                    node_id_int = int(node_id)

                # 按排序顺序匹配：node_id 001 -> 第1个类别, 002 -> 第2个类别
                # 同时兼容旧格式 category1, category_1 和直接文件夹名匹配
                category_name = None

                # 1. 先按 node_id 顺序索引匹配
                idx = node_id_int - 1  # node_id 从1开始
                if 0 <= idx < len(sorted_categories):
                    category_name = sorted_categories[idx]

                # 2. 回退：尝试旧格式 category1, category_1
                if category_name is None:
                    for fmt in [f"category{node_id_int}", f"category_{node_id_int}"]:
                        if fmt in class_centers:
                            category_name = fmt
                            break

                if category_name is not None and category_name in class_centers:
                    node['center_feature_siglip2'] = class_centers[category_name].tolist()
                    updated_count += 1
                    print(f"  ✓ 节点 {node_id} -> 类别 {category_name}")
                else:
                    print(f"  ⚠️  节点 {node_id}: 未找到匹配的类别")
            except Exception as e:
                print(f"  ✗ 节点 {node_id}: 处理失败 - {e}")

        print(f"  ✓ 已更新 {updated_count}/{len(nodes)} 个节点的特征中心")

        # 保存更新后的 graph_info.json（使用紧凑格式）
        temp_data = json.loads(json.dumps(graph_data))
        for node in temp_data.get('nodes', []):
            if 'center_feature_siglip2' in node:
                node['center_feature_siglip2'] = json.dumps(
                    node['center_feature_siglip2'],
                    separators=(',', ':')
                )

        with open(graph_copy_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2, ensure_ascii=False)

        print(f"✓ graph_info.json 已更新并保存: {graph_copy_path}")
        print(f"  特征字段: center_feature_siglip2")

    except Exception as e:
        print(f"  ✗ 更新 graph_info.json 失败: {e}")
        import traceback
        traceback.print_exc()

    model.train()
    return class_centers


def evaluate_with_class_centers(model, processor, test_dataset, class_centers, config, epoch):
    """
    使用类别中心向量进行分类评估（图像特征 vs 图像中心）

    Args:
        model: 模型
        processor: processor
        test_dataset: 测试数据集
        class_centers: 类别中心向量 {class_name: center_vector}（图像特征均值）
        config: 配置
        epoch: 当前epoch

    Returns:
        metrics: 评估指标
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"【分类评估】Epoch {epoch}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别中心数: {len(class_centers)}")
    print(f"{'='*60}\n")

    # 获取类别列表（保持顺序）
    class_list = sorted(class_centers.keys())
    num_classes = len(class_list)

    # 统计
    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_list}
    per_class_total = {cls: 0 for cls in class_list}

    # 混淆矩阵：{(true_class, pred_class): count}
    confusion_matrix = {(true_cls, pred_cls): 0 for true_cls in class_list for pred_cls in class_list}

    # 错误样本收集（用于错误分析）
    error_samples = []

    # 用于可视化的数据（按类别存储以确保均衡采样）
    samples_per_class = {cls: [] for cls in class_list}
    max_samples_per_class = 3

    batch_size = 32
    num_samples = len(test_dataset)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_samples = []
            batch_images = []
            batch_true_classes = []

            for idx in range(start_idx, end_idx):
                sample = test_dataset[idx]
                batch_samples.append(sample)
                batch_images.append(sample['image'])
                batch_true_classes.append(sample['class'])

            # 使用 processor 预处理
            inputs = processor(
                images=batch_images,
                return_tensors="pt"
            )

            pixel_values = inputs['pixel_values'].to(config.DEVICE)

            # V2: patch-level attention pooler
            if hasattr(model, 'encode'):
                image_embeds = model.encode(pixel_values)  # [B, D] already L2-normalized
            elif hasattr(model, 'encode_views'):
                image_embeds = model.encode_views(pixel_values)
            else:
                # V1 fallback: pooler_output
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                image_embeds = vision_outputs.pooler_output  # [B, D]
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # 计算与所有类别中心的相似度
            for i, (vec, true_cls) in enumerate(zip(image_embeds.cpu(), batch_true_classes)):
                vec_np = vec.numpy()

                # 计算与每个类中心的相似度
                similarities = {}
                for cls, center in class_centers.items():
                    sim = np.dot(vec_np, center)
                    similarities[cls] = sim

                # 预测类别
                pred_cls = max(similarities.items(), key=lambda x: x[1])[0]

                # 统计
                total += 1
                per_class_total[true_cls] += 1

                # 更新混淆矩阵
                confusion_matrix[(true_cls, pred_cls)] += 1

                if pred_cls == true_cls:
                    correct += 1
                    per_class_correct[true_cls] += 1
                else:
                    error_samples.append({
                        'true_class': true_cls,
                        'pred_class': pred_cls,
                        'similarities': similarities,
                        'sample_idx': start_idx + i
                    })

                # 保存用于可视化的样本（每个类别均衡采样）
                if len(samples_per_class[true_cls]) < max_samples_per_class:
                    samples_per_class[true_cls].append({
                        'image': batch_images[i],
                        'true_class': true_cls,
                        'pred_class': pred_cls,
                        'similarities': similarities,
                        'correct': pred_cls == true_cls
                    })

    # 计算指标
    accuracy = correct / total if total > 0 else 0
    per_class_accuracy = {
        cls: per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
        for cls in class_list
    }

    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_class_accuracy': per_class_accuracy,
        'num_classes': num_classes
    }

    # 打印结果
    print(f"  总体准确率: {accuracy:.1%} ({correct}/{total})")
    print(f"\n  各类别准确率:")
    for cls in class_list:
        acc = per_class_accuracy[cls]
        cnt = per_class_total[cls]
        print(f"    {cls}: {acc:.1%} ({per_class_correct[cls]}/{cnt})")

    # 生成可视化
    save_dir = os.path.join(config.MODEL_DIR, 'classification_viz')
    os.makedirs(save_dir, exist_ok=True)

    # 合并所有类别的样本（确保均衡表示）
    visualization_samples = []
    for cls in class_list:
        visualization_samples.extend(samples_per_class[cls])

    try:
        plot_classification_results(
            visualization_samples, class_list, metrics, save_dir, epoch
        )
        print(f"\n  ✓ 分类可视化已保存（共{len(visualization_samples)}个样本，每个类别最多{max_samples_per_class}个）")
    except Exception as e:
        print(f"  ⚠️  可视化失败: {e}")
        import traceback
        traceback.print_exc()

    # 保存结果到JSON
    result_file = os.path.join(save_dir, f'results_epoch_{epoch}.json')

    # 构建混淆矩阵（2D数组格式）
    confusion_matrix_2d = []
    for true_cls in class_list:
        row = [confusion_matrix[(true_cls, pred_cls)] for pred_cls in class_list]
        confusion_matrix_2d.append(row)

    # 转换 error_samples 中的 numpy 类型为 Python 原生类型
    error_samples_serializable = []
    for sample in error_samples[:50]:  # 最多保存50个错误样本
        similarities_serializable = {
            k: float(v) for k, v in sample['similarities'].items()
        }
        error_samples_serializable.append({
            'true_class': sample['true_class'],
            'pred_class': sample['pred_class'],
            'similarities': similarities_serializable,
            'sample_idx': int(sample['sample_idx'])
        })

    result_data = {
        'epoch': int(epoch),
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total),
            'num_classes': int(num_classes),
            'per_class_accuracy': {
                k: float(v) if not isinstance(v, str) else v
                for k, v in per_class_accuracy.items()
            }
        },
        'confusion_matrix': {
            'classes': list(class_list),
            'matrix': [[int(x) for x in row] for row in confusion_matrix_2d]
        },
        'error_analysis': {
            'total_errors': int(total - correct),
            'error_samples': error_samples_serializable
        }
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NumpyEncoder)

    print(f"  ✓ 结果已保存: {result_file}")
    print(f"  错误分析：{total - correct} 个错误样本（已保存前{min(len(error_samples), 50)}个）")

    # 绘制混淆矩阵
    try:
        plot_confusion_matrix(
            np.array(confusion_matrix_2d),
            class_list,
            save_dir,
            epoch
        )
    except Exception as e:
        print(f"  ⚠️  混淆矩阵绘制失败: {e}")
        import traceback
        traceback.print_exc()

    model.train()
    print(f"{'='*60}\n")

    return metrics


def plot_confusion_matrix(confusion_matrix_2d, class_list, save_dir, epoch):
    """
    绘制混淆矩阵热力图

    Args:
        confusion_matrix_2d: 2D混淆矩阵数组 [num_classes, num_classes]
        class_list: 类别列表
        save_dir: 保存目录
        epoch: 当前epoch
    """
    plt.figure(figsize=(10, 8))

    # 归一化（按行）
    cm_normalized = confusion_matrix_2d.astype('float') / confusion_matrix_2d.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_list,
        yticklabels=class_list,
        cbar_kws={'label': 'Normalized Count'}
    )

    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(f'Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 混淆矩阵已保存: {save_path}")


def plot_classification_results(samples, class_list, metrics, save_dir, epoch):
    """
    可视化分类结果 - 展示每个测试样本与所有类别中心的相似度
    """
    if not samples:
        print("  没有可视化样本")
        return

    num_samples = len(samples)

    # 创建子图
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 2.5))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        # 左列：显示图像
        ax_img = axes[idx, 0]
        ax_img.imshow(sample['image'])
        ax_img.axis('off')

        # 标题：真实类别 vs 预测类别
        true_cls = sample['true_class']
        pred_cls = sample['pred_class']
        correct = sample['correct']

        status = "✓" if correct else "✗"
        color = "green" if correct else "red"
        title = f"{status} True: {true_cls} | Pred: {pred_cls}"
        ax_img.set_title(title, fontsize=10, fontweight='bold', color=color)

        # 右列：相似度柱状图
        ax_bar = axes[idx, 1]

        similarities = sample['similarities']

        # 按照类别序号排序（不按相似度）
        classes_sorted = class_list
        values_sorted = [similarities[cls] for cls in class_list]

        # 根据是否正确设置颜色
        colors = []
        for cls in classes_sorted:
            if cls == true_cls:
                colors.append('green')  # 真实类别
            elif cls == pred_cls:
                colors.append('red')    # 错误预测类别
            else:
                colors.append('lightgray')  # 其他类别

        ax_bar.barh(range(len(classes_sorted)), values_sorted, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for i, (cls, val) in enumerate(zip(classes_sorted, values_sorted)):
            ax_bar.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

        ax_bar.set_yticks(range(len(classes_sorted)))
        ax_bar.set_yticklabels(classes_sorted)
        ax_bar.set_xlabel('Cosine Similarity', fontsize=9)
        ax_bar.set_xlim(-0.2, 1.0)
        ax_bar.grid(True, alpha=0.3, axis='x')

        # 高亮真实类别和预测类别
        if true_cls in similarities:
            true_idx = list(classes_sorted).index(true_cls)
            ax_bar.axhline(true_idx - 0.5, color='green', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle(f'Classification Results - Epoch {epoch} (Acc: {metrics["accuracy"]:.1%})',
                fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_classification.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    {save_path}")
