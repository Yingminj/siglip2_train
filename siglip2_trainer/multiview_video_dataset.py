"""Frame-labelled multi-view video dataset for SigLIP2 training."""

import os
from collections import OrderedDict
from pathlib import Path

import cv2
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import SigLIPAugmentation
from .multiview_models import split_image_to_views


class MultiViewClassFolderVideoDataset(Dataset):
    """Read uniformly sampled frames from ``split/class/*.mp4`` folders.

    The train/val/test split is defined by directories, so frames from one
    video can never leak into another split.  A fixed number of frames is used
    per video to stop longer states from dominating training.
    """

    def __init__(self, split_root, split="train", frames_per_video=16,
                 use_augmentation=False, augmentation_config=None,
                 edge_fraction=0.05, max_open_videos=8):
        self.split_root = Path(split_root).expanduser().resolve()
        self.image_root = str(self.split_root)
        self.split = split
        self.frames_per_video = int(frames_per_video)
        self.max_open_videos = max_open_videos
        self._captures = OrderedDict()
        if self.frames_per_video < 1:
            raise ValueError("frames_per_video must be >= 1")
        if not 0 <= edge_fraction < 0.5:
            raise ValueError("edge_fraction must be in [0, 0.5)")

        split_dir = self.split_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split directory does not exist: {split_dir}")
        class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
        class_dirs.sort(
            key=lambda path: (
                0, int(path.name[1:])
            ) if path.name.startswith("M") and path.name[1:].isdigit()
            else (1, path.name)
        )
        if not class_dirs:
            raise RuntimeError(f"no class directories in {split_dir}")

        self.classes = [path.name for path in class_dirs]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.augment_fn = None
        if use_augmentation and augmentation_config:
            self.augment_fn = SigLIPAugmentation(
                is_training=True, augment_config=augmentation_config
            )

        self.samples = []
        for class_dir in class_dirs:
            for video_path in sorted(class_dir.glob("*.mp4")):
                capture = cv2.VideoCapture(str(video_path))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                if frame_count < 1:
                    print(f"Warning: skipping unreadable video {video_path}")
                    continue
                first = int(frame_count * edge_fraction)
                last = max(first, frame_count - 1 - first)
                indices = [
                    round(first + i * (last - first) /
                          max(1, self.frames_per_video - 1))
                    for i in range(self.frames_per_video)
                ]
                for frame_idx in indices:
                    self.samples.append({
                        "class": class_dir.name,
                        "label": self.class_to_idx[class_dir.name],
                        "video_path": str(video_path),
                        "frame_idx": int(frame_idx),
                    })

        if not self.samples:
            raise RuntimeError(f"no readable videos in {split_dir}")
        self.video_count = len({sample["video_path"] for sample in self.samples})
        print(
            f"MultiViewClassFolderVideoDataset ({split}): {len(self.samples)} "
            f"frames from {self.video_count} videos, "
            f"{self.frames_per_video} frames/video, classes={self.classes}"
        )

    def __len__(self):
        return len(self.samples)

    def _get_capture(self, video_path):
        capture = self._captures.pop(video_path, None)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {video_path}")
        self._captures[video_path] = capture
        while len(self._captures) > self.max_open_videos:
            _, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        return capture

    def load_views(self, sample, apply_augmentation=False):
        capture = self._get_capture(sample["video_path"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_idx"])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"failed reading {sample['video_path']} frame={sample['frame_idx']}"
            )
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        views = split_image_to_views(image)
        if apply_augmentation and self.augment_fn is not None:
            views = [self.augment_fn(view) for view in views]
        return views

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "views": self.load_views(sample, apply_augmentation=True),
            "label": sample["label"],
            "class": sample["class"],
        }

    def close(self):
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self):
        if hasattr(self, "_captures"):
            self.close()


class MultiViewVideoDataset(Dataset):
    """Read labelled frames from horizontal three-view MP4 files.

    Videos and labels are paired as ``name.mp4`` and ``name_lable.txt``.
    Labels are 1-based and exposed as classes M1, M2, ... . Splitting is done
    by video rather than frame to prevent adjacent-frame leakage.
    """

    def __init__(self, video_root, label_root, split="train", val_ratio=0.1,
                 frame_stride=5, max_samples_per_class=None,
                 use_augmentation=False,
                 augmentation_config=None, max_open_videos=8):
        self.video_root = Path(video_root).expanduser().resolve()
        self.label_root = Path(label_root).expanduser().resolve()
        self.image_root = os.path.commonpath(
            [str(self.video_root), str(self.label_root)]
        )
        self.split = split
        self.frame_stride = frame_stride
        self.max_samples_per_class = max_samples_per_class
        self.max_open_videos = max_open_videos
        self._captures = OrderedDict()

        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if max_samples_per_class is not None and max_samples_per_class < 1:
            raise ValueError("max_samples_per_class must be at least 1")
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"视频目录不存在: {self.video_root}")
        if not self.label_root.is_dir():
            raise FileNotFoundError(f"标签目录不存在: {self.label_root}")

        self.augment_fn = None
        if use_augmentation and augmentation_config:
            self.augment_fn = SigLIPAugmentation(
                is_training=True, augment_config=augmentation_config
            )

        videos = sorted(self.video_root.glob("*.mp4"))
        if not videos:
            raise RuntimeError(f"没有找到 MP4 视频: {self.video_root}")

        pairs = []
        all_class_ids = set()
        for video_path in videos:
            label_path = self.label_root / f"{video_path.stem}_lable.txt"
            if not label_path.is_file():
                label_path = self.label_root / f"{video_path.stem}_label.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"视频缺少同名标签: {video_path.name}")
            labels = self._read_labels(label_path)
            all_class_ids.update(labels.values())
            pairs.append((video_path, label_path, labels))

        self.classes = [f"M{class_id}" for class_id in sorted(all_class_ids)]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        if not self.classes:
            raise RuntimeError("标签文件中没有有效标注")

        n_val = max(1, round(len(pairs) * val_ratio)) if val_ratio > 0 else 0
        if split == "val":
            selected_pairs = pairs[-n_val:] if n_val else []
        else:
            selected_pairs = pairs[:-n_val] if n_val else pairs
        if not selected_pairs:
            raise RuntimeError(f"{split} 划分没有视频，请调整 val_ratio")

        self.samples = []
        for video_path, label_path, labels in selected_pairs:
            for frame_idx, class_id in sorted(labels.items()):
                if frame_idx % frame_stride != 0:
                    continue
                class_name = f"M{class_id}"
                self.samples.append({
                    "class": class_name,
                    "label": self.class_to_idx[class_name],
                    "video_path": str(video_path),
                    "label_path": str(label_path),
                    "frame_idx": frame_idx,
                })

        if max_samples_per_class is not None:
            train_limit = round(max_samples_per_class * (1 - val_ratio))
            per_class_limit = (
                max_samples_per_class - train_limit
                if split == "val" else train_limit
            )
            self.samples = self._uniform_limit_by_class(
                self.samples, per_class_limit
            )

        print(
            f"MultiViewVideoDataset ({split}): {len(self.samples)} frames from "
            f"{len(selected_pairs)} videos, stride={frame_stride}, "
            f"classes={self.classes}"
        )

    @staticmethod
    def _uniform_limit_by_class(samples, limit):
        """Uniformly retain at most ``limit`` chronological samples/class."""
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample["class"], []).append(sample)

        selected = []
        for class_name, class_samples in grouped.items():
            count = len(class_samples)
            if count <= limit:
                print(
                    f"  Warning: {class_name} only has {count}/{limit} "
                    "available samples"
                )
                selected.extend(class_samples)
                continue
            if limit == 1:
                selected.append(class_samples[count // 2])
                continue
            indices = [
                round(i * (count - 1) / (limit - 1))
                for i in range(limit)
            ]
            selected.extend(class_samples[index] for index in indices)

        return sorted(
            selected,
            key=lambda sample: (sample["video_path"], sample["frame_idx"]),
        )

    @staticmethod
    def _read_labels(label_path):
        labels = {}
        with label_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 2:
                    raise ValueError(f"无效标签 {label_path}:{line_number}: {line.rstrip()}")
                frame_idx, class_id = int(parts[0]), int(parts[1])
                if frame_idx < 0 or class_id < 0:
                    raise ValueError(f"无效标签 {label_path}:{line_number}: {line.rstrip()}")
                # State 0 means unlabeled/invalid in the existing project
                # loaders and must not become a trainable class.
                if class_id == 0:
                    continue
                labels[frame_idx] = class_id
        return labels

    def __len__(self):
        return len(self.samples)

    def _get_capture(self, video_path):
        capture = self._captures.pop(video_path, None)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")
        self._captures[video_path] = capture
        while len(self._captures) > self.max_open_videos:
            _, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        return capture

    def load_views(self, sample, apply_augmentation=False):
        capture = self._get_capture(sample["video_path"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_idx"])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"读取视频帧失败: {sample['video_path']} frame={sample['frame_idx']}"
            )
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        views = split_image_to_views(image)
        if apply_augmentation and self.augment_fn is not None:
            views = [self.augment_fn(view) for view in views]
        return views

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "views": self.load_views(sample, apply_augmentation=True),
            "label": sample["label"],
            "class": sample["class"],
        }

    def close(self):
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self):
        if hasattr(self, "_captures"):
            self.close()
