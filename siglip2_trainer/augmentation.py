import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms


def apply_background_mask(image, top_ratio=0.20):
    """对 PIL 图像顶部区域打黑色 mask，遮盖不相关背景。

    Args:
        image: PIL Image
        top_ratio: 顶部遮盖比例（0.20 = 上方20%）
    Returns:
        PIL Image（已 mask）
    """
    if top_ratio <= 0:
        return image
    image = image.copy()
    w, h = image.size
    mask_h = int(h * top_ratio)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w - 1, mask_h - 1], fill=(0, 0, 0))
    return image


class RandomErasingPIL:
    """在 PIL 图像上随机遮挡矩形区域（用黑色填充）。"""

    def __init__(self, p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        area = w * h
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            ew = int(round((target_area * aspect_ratio) ** 0.5))
            eh = int(round((target_area / aspect_ratio) ** 0.5))
            if ew < w and eh < h:
                x = random.randint(0, w - ew)
                y = random.randint(0, h - eh)
                img = img.copy()
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + ew, y + eh], fill=(0, 0, 0))
                return img
        return img


class GaussianNoisePIL:
    """向 PIL 图像添加高斯噪声。"""

    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class SigLIPAugmentation:
    def __init__(self, is_training=True, augment_config=None):
        self.is_training = is_training

        default_config = {
            'horizontal_flip_prob': 0.5,
            'rotation_degrees': 5,
            'crop_scale': (0.9, 1.0),
            'crop_ratio': (0.95, 1.05),
            'brightness': 0.15,
            'contrast': 0.15,
            'saturation': 0.1,
            'hue': 0.05,
            'grayscale_prob': 0.1,
            'use_blur': False,
            'blur_kernel': 3,
            'blur_sigma': (0.1, 0.3),
        }

        self.config = {**default_config, **(augment_config or {})}
        aug_transforms = []

        if self.is_training:
            aug_transforms.append(
                transforms.RandomHorizontalFlip(p=self.config['horizontal_flip_prob'])
            )
            aug_transforms.append(
                transforms.RandomRotation(degrees=self.config['rotation_degrees'], fill=0)
            )
            aug_transforms.append(
                transforms.RandomResizedCrop(
                    size=224,
                    scale=self.config['crop_scale'],
                    ratio=self.config['crop_ratio'],
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
            aug_transforms.append(
                transforms.ColorJitter(
                    brightness=self.config['brightness'],
                    contrast=self.config['contrast'],
                    saturation=self.config['saturation'],
                    hue=self.config['hue']
                )
            )
            if self.config.get('perspective_prob', 0) > 0:
                aug_transforms.append(
                    transforms.RandomPerspective(
                        distortion_scale=self.config.get('perspective_scale', 0.1),
                        p=self.config.get('perspective_prob', 0.3)
                    )
                )
            if self.config.get('use_blur', False):
                aug_transforms.append(
                    transforms.GaussianBlur(
                        kernel_size=self.config.get('blur_kernel', 3),
                        sigma=self.config.get('blur_sigma', (0.1, 0.3))
                    )
                )
            erasing_prob = self.config.get('random_erasing_prob', 0)
            if erasing_prob > 0:
                aug_transforms.append(
                    RandomErasingPIL(
                        p=erasing_prob,
                        scale=self.config.get('random_erasing_scale', (0.02, 0.15))
                    )
                )
            noise_std = self.config.get('gaussian_noise_std', 0)
            if noise_std > 0:
                aug_transforms.append(GaussianNoisePIL(std=noise_std))

        self.augment_transform = transforms.Compose(aug_transforms) if aug_transforms else None

    def __call__(self, image):
        if self.augment_transform is not None:
            image = self.augment_transform(image)
        return image
