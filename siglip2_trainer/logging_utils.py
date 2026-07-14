import os
import json
from datetime import datetime

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，自动转换 numpy 类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        return super().default(obj)


def save_training_config(config, dataset_info, model_info, save_dir):
    """
    保存完整的训练配置信息

    Args:
        config: Config 对象
        dataset_info: 数据集信息字典
        model_info: 模型信息字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 保存为 JSON 格式（机器可读）
    json_path = os.path.join(save_dir, "training_config.json")

    config_data = {
        'metadata': {
            'timestamp': timestamp,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'script_name': 'fine_tuning_siglip2_fixed_eval.py',
        },
        'training_parameters': config.to_dict(),
        'dataset_info': dataset_info,
        'model_info': model_info,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Training config saved (JSON): {json_path}")

    # 2. 保存为可读文本格式（人类可读）
    txt_path = os.path.join(save_dir, "training_config.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SigLIP2 Training Configuration\n")
        f.write("="*80 + "\n\n")

        # 元数据
        f.write("[Metadata]\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Working Directory: {os.getcwd()}\n")
        f.write(f"Python Version:    {os.sys.version.split()[0]}\n")
        f.write(f"PyTorch Version:   {torch.__version__}\n")
        f.write("\n")

        # 模型配置
        f.write("[Model Configuration]\n")
        f.write("-"*80 + "\n")
        f.write(f"Base Model:        {config.SIGLIP_MODEL}\n")
        f.write(f"Model Type:        {model_info.get('model_type', 'N/A')}\n")
        f.write(f"Total Parameters:  {model_info.get('total_params', 0):,}\n")
        f.write(f"Trainable Params:  {model_info.get('trainable_params', 0):,}\n")
        f.write(f"Trainable Ratio:   {model_info.get('trainable_ratio', 0):.2f}%\n")
        finetune_strategy = getattr(config, 'FINETUNE_STRATEGY', 'frozen')
        f.write(f"Fine-tune Strategy: {finetune_strategy}\n")
        if finetune_strategy == "last_n_blocks":
            f.write(f"  - Visual Blocks:  {getattr(config, 'NUM_LAST_VISUAL_BLOCKS', 'N/A')}\n")
            f.write(f"  - Text Blocks:    {getattr(config, 'NUM_LAST_TEXT_BLOCKS', 'N/A')}\n")
        if hasattr(config, 'TRAIN_VISUAL_PROJ'):
            f.write(f"Train Visual Proj: {config.TRAIN_VISUAL_PROJ}\n")
        if hasattr(config, 'TRAIN_TEXT_PROJ'):
            f.write(f"Train Text Proj:   {config.TRAIN_TEXT_PROJ}\n")
        if hasattr(config, 'MAX_TEXT_LENGTH'):
            f.write(f"Max Text Length:   {config.MAX_TEXT_LENGTH}\n")
        if hasattr(config, 'NUM_QUERY_TOKENS'):
            f.write(f"Query Tokens:      {config.NUM_QUERY_TOKENS}\n")
            f.write(f"Pooler Layers:     {config.POOLER_NUM_LAYERS}\n")
            f.write(f"Pooler Heads:      {config.POOLER_NUM_HEADS}\n")
        f.write("\n")

        # 训练超参数
        f.write("[Training Hyperparameters]\n")
        f.write("-"*80 + "\n")
        f.write(f"Epochs:            {config.EPOCHS}\n")
        f.write(f"Batch Size:        {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate:     {config.LEARNING_RATE}\n")
        f.write(f"Weight Decay:      {config.WEIGHT_DECAY}\n")
        f.write(f"Optimizer:         Adam (betas=(0.9, 0.98), eps=1e-6)\n")
        f.write(f"Device:            {config.DEVICE}\n")
        f.write("\n")

        # 学习率调度
        f.write("[Learning Rate Scheduler]\n")
        f.write("-"*80 + "\n")
        f.write(f"Scheduler Type:    {config.SCHEDULER_TYPE}\n")
        if config.SCHEDULER_TYPE == "warmup_cosine":
            f.write(f"Warmup Epochs:     {config.WARMUP_EPOCHS}\n")
        f.write(f"Min Learning Rate: {config.MIN_LR}\n")
        f.write("\n")

        # 数据集信息
        f.write("[Dataset Information]\n")
        f.write("-"*80 + "\n")
        f.write(f"Image Root:        {config.IMAGE_ROOT}\n")
        f.write(f"Text Root:         {getattr(config, 'TEXT_ROOT', 'N/A')}\n")
        f.write(f"Number of Classes: {dataset_info.get('num_classes', 0)}\n")
        f.write(f"Samples per Class: {dataset_info.get('samples_per_class', 0)}\n")
        f.write(f"Total Samples:     {dataset_info.get('total_samples', 0)}\n")
        f.write(f"Dataset Length:    {dataset_info.get('dataset_length', 0)}\n")

        # 类别详情
        if 'class_details' in dataset_info:
            f.write("\nClass Details:\n")
            for class_info in dataset_info['class_details']:
                f.write(f"  {class_info['name']:20s} - "
                       f"{class_info['num_images']:3d} images, "
                       f"{class_info['num_texts']:3d} texts\n")
        f.write("\n")

        # 数据增强
        f.write("[Data Augmentation]\n")
        f.write("-"*80 + "\n")
        f.write(f"Use Augmentation:  {config.USE_AUGMENTATION}\n")
        if config.USE_AUGMENTATION:
            aug_config = config.AUGMENTATION_CONFIG
            f.write(f"  - Horizontal Flip Prob: {aug_config['horizontal_flip_prob']}\n")
            f.write(f"  - Rotation Degrees:     {aug_config['rotation_degrees']}\n")
            f.write(f"  - Crop Scale:           {aug_config['crop_scale']}\n")
            f.write(f"  - Crop Ratio:           {aug_config['crop_ratio']}\n")
            f.write(f"  - Brightness:           {aug_config['brightness']}\n")
            f.write(f"  - Contrast:             {aug_config['contrast']}\n")
            f.write(f"  - Saturation:           {aug_config['saturation']}\n")
            f.write(f"  - Hue:                  {aug_config['hue']}\n")
            f.write(f"  - Grayscale Prob:       {aug_config['grayscale_prob']}\n")
            f.write(f"  - Use Blur:             {aug_config['use_blur']}\n")
        f.write("\n")

        # Checkpoint 和早停
        f.write("[Checkpoint & Early Stopping]\n")
        f.write("-"*80 + "\n")
        f.write(f"Save Directory:    {config.MODEL_DIR}\n")
        f.write(f"Model Name:        {config.MODEL_NAME}\n")
        f.write(f"Save Every N Epochs: {config.SAVE_EVERY_N_EPOCHS}\n")
        f.write(f"Max Checkpoints:   {config.MAX_CHECKPOINTS}\n")
        f.write(f"Early Stopping:    {config.EARLY_STOPPING}\n")
        if config.EARLY_STOPPING:
            f.write(f"  - Patience:        {config.PATIENCE}\n")
            f.write(f"  - Min Delta:       {config.MIN_DELTA}\n")
        f.write("\n")

        # 训练策略
        f.write("[Training Strategy]\n")
        f.write("-"*80 + "\n")
        f.write(f"DataLoader Workers: 4\n")
        f.write(f"Pin Memory:        {config.DEVICE != 'cpu'}\n")
        f.write(f"Drop Last Batch:   True\n")
        f.write(f"Shuffle:           True\n")
        f.write("\n")

        f.write("="*80 + "\n")

    print(f"✓ Training config saved (TXT):  {txt_path}")

    return json_path, txt_path


def collect_dataset_info(dataset, image_root, text_root):
    """
    收集数据集的详细信息
    兼容 GroupedDataset 和 SimpleDataset

    Args:
        dataset: GroupedDataset 或 SimpleDataset 对象
        image_root: 图像根目录
        text_root: 文本根目录

    Returns:
        dataset_info: 数据集信息字典
    """
    # 检测 dataset 类型
    is_grouped = hasattr(dataset, 'num_classes') and hasattr(dataset, 'img_dict')

    if is_grouped:
        # GroupedDataset
        dataset_info = {
            'dataset_type': 'GroupedDataset',
            'num_classes': dataset.num_classes,
            'samples_per_class': dataset.num_samples_per_class,
            'total_samples': dataset.num_classes * dataset.num_samples_per_class,
            'dataset_length': len(dataset),
            'image_root': image_root,
            'text_root': text_root,
            'class_names': dataset.classes,
            'class_details': []
        }

        # 每个类别的详细信息
        for class_name in dataset.classes:
            num_images = len(dataset.img_dict.get(class_name, []))
            num_texts = len(dataset.txt_dict.get(class_name, []))

            dataset_info['class_details'].append({
                'name': class_name,
                'index': dataset.class_to_idx[class_name],
                'num_images': num_images,
                'num_texts': num_texts,
            })
    else:
        # SimpleDataset
        dataset_info = {
            'dataset_type': 'SimpleDataset',
            'num_classes': len(dataset.classes),
            'samples_per_class': 'variable',
            'total_samples': len(dataset.samples),
            'dataset_length': len(dataset),
            'image_root': image_root,
            'text_root': text_root,
            'class_names': dataset.classes,
            'class_details': []
        }

        # 统计每个类别的样本数
        class_counts = {}
        for sample in dataset.samples:
            class_name = sample['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 每个类别的详细信息
        for class_name in dataset.classes:
            dataset_info['class_details'].append({
                'name': class_name,
                'index': dataset.class_to_idx[class_name],
                'num_images': class_counts.get(class_name, 0),
                'num_texts': class_counts.get(class_name, 0),
            })

    return dataset_info


def collect_model_info(model):
    """
    收集模型的详细信息

    Args:
        model: 模型对象

    Returns:
        model_info: 模型信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    model_info = {
        'model_type': type(model).__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': 100 * trainable_params / total_params if total_params > 0 else 0,
        'trainable_param_names': trainable_param_names,
    }

    return model_info
