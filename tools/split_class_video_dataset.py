#!/usr/bin/env python3
"""Create deterministic train/val/test class-folder splits using symlinks.

The source tree is expected to contain one directory per state (``1`` ...
``6``), each containing MP4 files.  Videos are never copied or modified.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true",
                        help="允许输出目录只包含本脚本创建的链接时重新生成")
    args = parser.parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser()
    if not source.is_dir():
        raise SystemExit(f"source does not exist: {source}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {output}; use --force to regenerate")
        # Only remove the split directories and manifest, never source files.
        for child in output.iterdir():
            if child.name in {"train", "val", "test", "split_manifest.json"}:
                if child.is_dir() and not child.is_symlink():
                    for p in sorted(child.rglob("*"), reverse=True):
                        if p.is_symlink() or p.is_file(): p.unlink()
                        elif p.is_dir(): p.rmdir()
                    child.rmdir()
                else:
                    child.unlink()
            else:
                raise SystemExit(f"refusing to remove unrelated output: {child}")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {"source": str(source), "seed": args.seed,
                "ratios": {"train": args.train_ratio, "val": args.val_ratio,
                            "test": args.test_ratio}, "files": []}
    class_dirs = sorted((p for p in source.iterdir() if p.is_dir()),
                        key=lambda p: (int(p.name) if p.name.isdigit() else p.name))
    if not class_dirs:
        raise SystemExit("no class directories found")
    for class_dir in class_dirs:
        videos = sorted(class_dir.glob("*.mp4"))
        if not videos:
            continue
        rng.shuffle(videos)
        n = len(videos)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        # Put rounding remainder in train, keeping val/test held out.
        assignments = (["train"] * n_train + ["val"] * n_val +
                       ["test"] * (n - n_train - n_val))
        for video, split in zip(videos, assignments):
            class_name = f"M{class_dir.name}" if class_dir.name.isdigit() else class_dir.name
            target_dir = output / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            link = target_dir / video.name
            link.symlink_to(video)
            manifest["files"].append({"split": split, "class": class_name,
                                      "source": str(video), "link": str(link)})
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("train", "val", "test"):
        counts = {}
        for item in manifest["files"]:
            if item["split"] == split:
                counts[item["class"]] = counts.get(item["class"], 0) + 1
        print(f"{split}: {sum(counts.values())} videos | {counts}")
    print(f"Created symlink split at {output}")


if __name__ == "__main__":
    main()
