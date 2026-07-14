import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .augmentation import SigLIPAugmentation, apply_background_mask


def load_label_file(label_path):
    """读取帧级别标注文件，返回 {frame_idx: class_id(0-based)} dict"""
    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_idx        = int(parts[0])
                class_id         = int(parts[1]) - 1  # 转为 0-based
                labels[frame_idx] = class_id
    return labels


class VideoSequenceDataset(Dataset):
    """
    从预提取的视频特征缓存中构建滑动窗口序列，用于 LSTM Stage 2 训练。
    每条样本: T 帧特征 [T, D] + 最后一帧的类别标签。
    """
    def __init__(self, features_cache_path, seq_len=16, stride=4):
        self.seq_len = seq_len

        print(f"  加载特征缓存: {features_cache_path}")
        cache = torch.load(features_cache_path, map_location='cpu')

        self.clips = []  # list of (video_name, start_frame)
        self.cache = cache

        for video_name, data in cache.items():
            n_frames = data['features'].shape[0]
            for start in range(0, n_frames - seq_len, stride):
                window_labels = data['labels'][start:start + seq_len]
                # 只保留纯窗口：所有帧有效且标签相同（无跨状态过渡）
                if window_labels.min() >= 0 and window_labels.min() == window_labels.max():
                    self.clips.append((video_name, start))

        print(f"  VideoSequenceDataset: {len(self.clips)} clips，来自 {len(cache)} 个视频")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_name, start = self.clips[idx]
        data  = self.cache[video_name]
        end   = start + self.seq_len
        seq   = data['features'][start:end]           # [T, D]
        label = data['labels'][end - 1].long()        # scalar (最后帧标签)
        return {'seq_features': seq, 'label': label}


class GroupedDataset(Dataset):
    def __init__(self, image_root, text_root, use_augmentation=False,
                 augmentation_config=None, is_training=True):
        self.image_root       = image_root
        self.text_root        = text_root
        self.use_augmentation = use_augmentation and is_training

        def extract_number(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else 0

        self.classes = sorted(
            [d for d in os.listdir(image_root)
             if os.path.isdir(os.path.join(image_root, d))],
            key=extract_number
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.img_dict = {}
        self.txt_dict = {}

        for c in self.classes:
            img_dir = os.path.join(image_root, c)
            txt_dir = os.path.join(text_root, c)

            self.img_dict[c] = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith('.jpg')
            ])
            self.txt_dict[c] = sorted([
                os.path.join(txt_dir, f)
                for f in os.listdir(txt_dir)
                if f.endswith('.txt')
            ])

        self.num_classes           = len(self.classes)
        self.num_samples_per_class = len(next(iter(self.img_dict.values())))
        self.length                = self.num_samples_per_class

        print(f"Dataset loaded: {self.num_classes} classes, "
              f"{self.num_samples_per_class} samples per class")

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        images = []
        texts  = []
        labels = []

        for c in self.classes:
            rand_idx = np.random.randint(0, len(self.img_dict[c]))
            with Image.open(self.img_dict[c][rand_idx]) as img_file:
                img = img_file.convert('RGB')
                img.load()
            with open(self.txt_dict[c][rand_idx], 'r', encoding='utf-8') as f:
                text = f.read().strip()

            images.append(img)
            texts.append(text)
            labels.append(self.class_to_idx[c])

        perm   = np.random.permutation(self.num_classes)
        images = [images[i] for i in perm]
        texts  = [texts[i]  for i in perm]
        labels = [labels[i] for i in perm]

        return {
            'P': images,
            'T': texts,
            'L': torch.tensor(labels, dtype=torch.long)
        }


class SimpleDataset(Dataset):
    """
    简单数据集：每次返回单个样本，支持train/val/test三份划分
    避免GroupedDataset的同类别文本共享问题
    """
    def __init__(self, image_root, text_root, use_augmentation=False,
                 augmentation_config=None, is_training=True,
                 val_ratio=0.2, test_ratio=0.2, split='train',
                 background_mask_ratio=0.0):
        """
        Args:
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            split: 'train', 'val', 或 'test'
            background_mask_ratio: 顶部背景mask比例（0=不启用）
        """
        self.image_root = image_root
        self.text_root = text_root
        self.use_augmentation = use_augmentation and is_training
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split = split
        self.background_mask_ratio = background_mask_ratio

        # 设置数据增强
        if self.use_augmentation and augmentation_config:
            self.augment_fn = SigLIPAugmentation(
                is_training=True,
                augment_config=augmentation_config
            )
        else:
            self.augment_fn = None

        # 提取数字排序
        def extract_number(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else 0

        # 加载所有类别
        self.classes = sorted(
            [d for d in os.listdir(image_root)
             if os.path.isdir(os.path.join(image_root, d))],
            key=extract_number
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 收集所有样本
        all_samples = []
        for c in self.classes:
            img_dir = os.path.join(image_root, c)

            img_files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ])

            if text_root is not None:
                txt_dir = os.path.join(text_root, c)
                for img_file in img_files:
                    base_name = os.path.splitext(img_file)[0]
                    txt_file = base_name + '.txt'
                    img_path = os.path.join(img_dir, img_file)
                    txt_path = os.path.join(txt_dir, txt_file)

                    if os.path.exists(txt_path):
                        all_samples.append({
                            'class': c,
                            'image_path': img_path,
                            'text_path': txt_path
                        })
            else:
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    all_samples.append({
                        'class': c,
                        'image_path': img_path,
                        'text_path': None
                    })

        # 划分数据集
        self.samples = self._split_samples(all_samples)

        print(f"SimpleDataset ({split}): {len(self.samples)} samples from {len(self.classes)} classes")

    def _split_samples(self, all_samples):
        """按类别划分数据集，使用连续时间段切分避免视频帧数据泄露。

        视频帧数据中相邻帧高度相似，随机切分会导致训练集和验证集包含几乎
        相同的帧（数据泄露）。改为连续段切分：
          - 训练集取前面的连续帧
          - 中间留 gap 帧作为缓冲（避免边界帧泄露）
          - 验证集取末尾的连续帧
        """
        split_samples = []
        GAP_FRAMES = 10  # 训练集和验证集之间的间隔帧数

        # 按类别分组
        class_samples = {}
        for sample in all_samples:
            c = sample['class']
            if c not in class_samples:
                class_samples[c] = []
            class_samples[c].append(sample)

        # 均匀采样：将所有类别降采样到最少类别的数量（均匀间隔采样）
        min_count = min(len(s) for s in class_samples.values())
        for c in class_samples:
            samples = class_samples[c]
            if len(samples) > min_count:
                # 均匀间隔采样，保证覆盖整个序列
                indices = np.round(np.linspace(0, len(samples) - 1, min_count)).astype(int)
                class_samples[c] = [samples[i] for i in indices]
                print(f"  类别平衡: {c} 从 {len(samples)} 降采样到 {min_count}")

        # 对每个类别进行连续段切分
        for c_name, samples in class_samples.items():
            n_total = len(samples)
            n_val = int(n_total * self.val_ratio)
            n_test = int(n_total * self.test_ratio)

            # 确保 gap 不超过剩余帧数
            effective_gap = min(GAP_FRAMES, max(0, n_total - n_val - n_test - 1))

            # 布局: [train] [gap] [val] [test(末尾)]
            # test 取最末尾，val 在 test 前面，gap 在 val 前面，train 取最前面
            test_start = n_total - n_test
            val_start = test_start - n_val
            gap_start = val_start - effective_gap
            n_train = gap_start  # train 取 0 ~ gap_start

            if self.split == 'train':
                selected = samples[:n_train]
            elif self.split == 'val':
                selected = samples[val_start:val_start + n_val]
            else:  # test
                selected = samples[test_start:]

            if self.split == 'train':
                print(f"  {c_name}: total={n_total}, train={n_train}, gap={effective_gap}, val={n_val}, test={n_test}")

            split_samples.extend(selected)

        return split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        with Image.open(sample['image_path']) as img_file:
            image = img_file.convert('RGB')
            image.load()

        # 背景mask（在数据增强之前，确保mask区域不影响裁剪等操作）
        if self.background_mask_ratio > 0:
            image = apply_background_mask(image, self.background_mask_ratio)

        # 应用数据增强
        if self.augment_fn is not None:
            image = self.augment_fn(image)

        # 加载文本（text_root=None 时跳过）
        text = ''
        if sample['text_path'] is not None:
            with open(sample['text_path'], 'r', encoding='utf-8') as f:
                text = f.read().strip()

        return {
            'image': image,
            'text': text,
            'label': self.class_to_idx[sample['class']],
            'class': sample['class']
        }


def make_collate_fn(processor, config, use_augmentation=False, augmentation_config=None):
    """
    [FIX 1] 使用完整 processor 一次性调用，而不是分开调用 image_processor 和 tokenizer。
    naflex 模型需要 spatial_shapes，只有完整 processor 调用才会自动生成该字段。
    分开调用 processor.image_processor() 不会产生 spatial_shapes，导致 model forward 报错。
    """
    augment_fn = None
    if use_augmentation and augmentation_config:
        augment_fn = SigLIPAugmentation(is_training=True, augment_config=augmentation_config)

    def collate_fn(batch):
        all_images = [img  for item in batch for img  in item['P']]
        all_texts  = [text for item in batch for text in item['T']]
        labels     = torch.stack([item['L'] for item in batch]).view(-1)

        if augment_fn is not None:
            all_images = [augment_fn(img) for img in all_images]

        inputs = processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH
        )

        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = (inputs['input_ids'] != 0).long()

        return {
            'pixel_values':          inputs['pixel_values'],
            'input_ids':             inputs['input_ids'],
            'attention_mask':        attention_mask,
            'spatial_shapes':        inputs.get('spatial_shapes'),        # naflex 必需
            'pixel_attention_mask':  inputs.get('pixel_attention_mask'),  # 可选
            'L':                     labels
        }

    return collate_fn


def make_vision_only_collate_fn(processor):
    """
    Vision-only collate function：只处理图像，不处理文本。
    用于 ArcFace 等纯视觉训练模式。
    """
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        inputs = processor(images=images, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'],
            'labels': labels,
        }

    return collate_fn
