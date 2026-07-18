"""Frame-labelled multi-view video dataset for SigLIP2 training.

Supports three data modes:
  - ``video_class_folders``:  pre-split train/val/test/M1...M6 directories with MP4 files.
  - ``video``:                labelled MP4 files with frame-level TXT labels.
  - ``video_class_folders_transition``:  class-folder videos + dense transition-frame
    sampling from continuous labelled videos (best for tackling boundary effects).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from .augmentation import SigLIPAugmentation
from .multiview_models import split_image_to_views


# =============================================================================
# helper: non-uniform frame sampling with boundary emphasis
# =============================================================================

def _sample_indices_boundary_emphasis(
    frame_count: int,
    num_samples: int,
    edge_fraction: float = 0.0,
    boundary_emphasis: float = 0.0,
) -> list[int]:
    """Return *num_samples* frame indices with optional boundary emphasis.

    Parameters
    ----------
    frame_count:
        Total frames in the video.
    num_samples:
        Desired number of sampled frames.
    edge_fraction:
        Fraction of frames to trim from each end (0.0 = keep all).
    boundary_emphasis:
        Fraction of *num_samples* that are allocated near the start and end.
        0.0 = uniform; 0.3 = 30% of the samples go near the two boundaries.
    """
    first = int(frame_count * edge_fraction)
    last = max(first, frame_count - 1 - first)
    span = last - first

    if span <= 0 or num_samples <= 0:
        return [first]

    boundary_emphasis = float(np.clip(boundary_emphasis, 0.0, 0.45))
    if boundary_emphasis <= 0.0:
        # uniform
        indices = [
            round(first + i * span / max(1, num_samples - 1))
            for i in range(num_samples)
        ]
        return sorted(set(indices))

    n_boundary = max(1, round(num_samples * boundary_emphasis))
    n_head = n_boundary // 2
    n_tail = n_boundary - n_head
    n_middle = num_samples - n_head - n_tail

    boundary_span = int(span * 0.2)  # boundary region covers 20% of span
    head_end = first + boundary_span

    if n_head >= 1:
        head_indices = [
            round(first + i * (head_end - first) / max(1, n_head))
            for i in range(n_head)
        ]
    else:
        head_indices = []

    if n_tail >= 1:
        tail_start = last - boundary_span
        tail_indices = [
            round(tail_start + i * (last - tail_start) / max(1, n_tail))
            for i in range(n_tail)
        ]
    else:
        tail_indices = []

    if n_middle >= 1:
        middle_indices = [
            round(head_end + i * (tail_start - head_end) / max(1, n_middle))
            for i in range(n_middle)
        ]
    else:
        middle_indices = []

    return sorted(set(head_indices + middle_indices + tail_indices))


# =============================================================================
# MultiViewClassFolderVideoDataset
# =============================================================================

class MultiViewClassFolderVideoDataset(Dataset):
    """Read uniformly sampled frames from ``split/class/*.mp4`` folders.

    The train/val/test split is defined by directories, so frames from one
    video can never leak into another split.  A fixed number of frames is used
    per video to stop longer states from dominating training.

    .. versionchanged::
        Default ``edge_fraction`` is now **0.0** (was 0.05) so that frames
        near video boundaries — which carry transition-like visual features —
        are included in training.  Set ``boundary_emphasis > 0`` to allocate
        more samples near start/end of each video.
    """

    def __init__(
        self,
        split_root,
        split="train",
        frames_per_video=24,
        use_augmentation=False,
        augmentation_config=None,
        edge_fraction=0.0,           # ← was 0.05, now 0.0
        boundary_emphasis=0.0,       # ← NEW: 0.0=uniform, 0.3=30% near edges
        max_open_videos=8,
    ):
        self.split_root = Path(split_root).expanduser().resolve()
        self.image_root = str(self.split_root)
        self.split = split
        self.frames_per_video = int(frames_per_video)
        self.edge_fraction = float(edge_fraction)
        self.boundary_emphasis = float(boundary_emphasis)
        self.max_open_videos = max_open_videos
        self._captures = OrderedDict()

        if self.frames_per_video < 1:
            raise ValueError("frames_per_video must be >= 1")
        if not 0 <= self.edge_fraction < 0.5:
            raise ValueError("edge_fraction must be in [0, 0.5)")
        if not 0 <= self.boundary_emphasis < 0.5:
            raise ValueError("boundary_emphasis must be in [0, 0.5)")

        split_dir = self.split_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split directory does not exist: {split_dir}")
        class_dirs = [
            path for path in split_dir.iterdir() if path.is_dir()
        ]
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

                indices = _sample_indices_boundary_emphasis(
                    frame_count=frame_count,
                    num_samples=self.frames_per_video,
                    edge_fraction=self.edge_fraction,
                    boundary_emphasis=self.boundary_emphasis,
                )
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

        # count boundary samples for logging
        boundary_samples = 0
        for sample in self.samples:
            capture = cv2.VideoCapture(sample["video_path"])
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            first = int(total * self.edge_fraction)
            last = max(first, total - 1 - first)
            head_region = first + int((last - first) * 0.2)
            tail_region = last - int((last - first) * 0.2)
            if sample["frame_idx"] <= head_region or sample["frame_idx"] >= tail_region:
                boundary_samples += 1

        print(
            f"MultiViewClassFolderVideoDataset ({split}): {len(self.samples)} "
            f"frames from {self.video_count} videos, "
            f"{self.frames_per_video} frames/video, "
            f"edge_fraction={self.edge_fraction:.2f}, "
            f"boundary_emphasis={self.boundary_emphasis:.2f} "
            f"(boundary frames: {boundary_samples}/{len(self.samples)} = "
            f"{boundary_samples/max(1,len(self.samples))*100:.1f}%), "
            f"classes={self.classes}"
        )

    # ---- remaining methods unchanged ----
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


# =============================================================================
# MultiViewVideoDataset  (连续标注视频)
# =============================================================================

class MultiViewVideoDataset(Dataset):
    """Read labelled frames from horizontal three-view MP4 files.

    Videos and labels are paired as ``name.mp4`` and ``name_lable.txt``.
    Labels are 1-based and exposed as classes M1, M2, ... . Splitting is done
    by video rather than frame to prevent adjacent-frame leakage.

    .. versionadded:: transition-aware sampling
        ``transition_window`` and ``transition_dense_stride`` allow dense
        sampling near label transition points while keeping the normal
        ``frame_stride`` in stable regions.
    """

    def __init__(
        self,
        video_root,
        label_root,
        split="train",
        val_ratio=0.1,
        frame_stride=5,
        max_samples_per_class=None,
        use_augmentation=False,
        augmentation_config=None,
        max_open_videos=8,
        # ---- transition-aware sampling ----
        transition_window=0,          # ±frames around label transitions for dense sampling
        transition_dense_stride=1,     # stride inside the transition window (1=every frame)
    ):
        self.video_root = Path(video_root).expanduser().resolve()
        self.label_root = Path(label_root).expanduser().resolve()
        self.image_root = os.path.commonpath(
            [str(self.video_root), str(self.label_root)]
        )
        self.split = split
        self.frame_stride = frame_stride
        self.max_samples_per_class = max_samples_per_class
        self.max_open_videos = max_open_videos
        self.transition_window = int(transition_window)
        self.transition_dense_stride = int(transition_dense_stride)
        self._captures = OrderedDict()

        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.transition_window < 0:
            raise ValueError("transition_window must be >= 0")
        if self.transition_dense_stride < 1:
            raise ValueError("transition_dense_stride must be >= 1")
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

        # ---- build samples with transition-aware stride ----
        self.samples = []
        transition_added = 0
        normal_added = 0

        for video_path, label_path, labels in selected_pairs:
            all_frames = sorted(labels.keys())
            if not all_frames:
                continue

            # find transition points
            transitions: set[int] = set()
            prev_cat = labels[all_frames[0]]
            for fi in all_frames[1:]:
                cur_cat = labels[fi]
                if cur_cat != prev_cat:
                    transitions.add(fi)
                    prev_cat = cur_cat

            # build transition-region frame set
            transition_frames: set[int] = set()
            if self.transition_window > 0:
                for tp in transitions:
                    for offset in range(-self.transition_window,
                                        self.transition_window + 1):
                        candidate = tp + offset
                        if candidate in labels:
                            transition_frames.add(candidate)

            for frame_idx in all_frames:
                in_transition = frame_idx in transition_frames
                if in_transition:
                    if frame_idx % self.transition_dense_stride != 0:
                        continue
                    transition_added += 1
                else:
                    if frame_idx % self.frame_stride != 0:
                        continue
                    normal_added += 1

                class_name = f"M{labels[frame_idx]}"
                if class_name not in self.class_to_idx:
                    continue
                self.samples.append({
                    "class": class_name,
                    "label": self.class_to_idx[class_name],
                    "video_path": str(video_path),
                    "label_path": str(label_path),
                    "frame_idx": frame_idx,
                    "is_transition": in_transition,
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

        n_trans = sum(1 for s in self.samples if s.get("is_transition"))

        print(
            f"MultiViewVideoDataset ({split}): {len(self.samples)} frames from "
            f"{len(selected_pairs)} videos, "
            f"stride={frame_stride}, "
            f"trans_window=±{self.transition_window}, "
            f"trans_dense_stride={self.transition_dense_stride}, "
            f"transition_frames={n_trans}/{len(self.samples)} "
            f"({n_trans/max(1,len(self.samples))*100:.1f}%), "
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
                    raise ValueError(
                        f"无效标签 {label_path}:{line_number}: {line.rstrip()}"
                    )
                frame_idx, class_id = int(parts[0]), int(parts[1])
                if frame_idx < 0 or class_id < 0:
                    raise ValueError(
                        f"无效标签 {label_path}:{line_number}: {line.rstrip()}"
                    )
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
