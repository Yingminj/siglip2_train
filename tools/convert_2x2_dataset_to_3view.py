#!/usr/bin/env python3
"""Batch-convert a 2x2 video dataset to horizontal three-view videos.

Observed input layout (1280x960):

    center (top-left) | left (top-right)
    right  (bottom-left) | unused/black (bottom-right)

Output layout (1920x480): ``center | left | right``.
The input directory structure (for example train/M1/*.mp4) is preserved.
"""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="输入根目录，可直接传 video_split")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="输出根目录，例如 video_split_3view")
    parser.add_argument("--workers", type=int, default=2,
                        help="并行转换数，默认 2")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 质量，越小越清晰，默认 18")
    parser.add_argument("--preset", default="fast",
                        choices=("ultrafast", "fast", "medium", "slow"))
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出视频")
    return parser.parse_args()


def convert_one(source, target, ffmpeg, crf, preset, overwrite):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return "skipped", source, "already exists"

    # Robot-centric camera names: tl=center, tr=left, bl=right.
    # The bottom-right quadrant is unused/black.
    filter_graph = (
        "[0:v]crop=iw/2:ih/2:0:0[center];"
        "[0:v]crop=iw/2:ih/2:iw/2:0[left];"
        "[0:v]crop=iw/2:ih/2:0:ih/2[right];"
        "[center][left][right]hstack=inputs=3[outv]"
    )
    command = [
        ffmpeg, "-y" if overwrite else "-n", "-loglevel", "error",
        "-i", str(source), "-filter_complex", filter_graph,
        "-map", "[outv]", "-an", "-c:v", "libx264",
        "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if target.exists():
            target.unlink()
        return "failed", source, result.stderr.strip().splitlines()[-1]
    return "converted", source, None


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("找不到 ffmpeg，请先安装 ffmpeg")
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("输入和输出目录不能相同")

    videos = sorted(input_dir.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"没有找到 MP4: {input_dir}")
    print(f"找到 {len(videos)} 个视频")
    print("输出视角顺序: center | left | right")

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for source in videos:
            target = output_dir / source.relative_to(input_dir)
            future = executor.submit(
                convert_one, source, target, ffmpeg, args.crf,
                args.preset, args.overwrite,
            )
            futures[future] = target
        for index, future in enumerate(as_completed(futures), 1):
            status, source, error = future.result()
            target = futures[future]
            records.append({"status": status, "source": str(source),
                            "output": str(target), "error": error})
            mark = "✓" if status in {"converted", "skipped"} else "✗"
            print(f"[{index}/{len(videos)}] {mark} {status}: {source.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "conversion_manifest.json"
    manifest.write_text(json.dumps({
        "layout": ["center", "left", "right"],
        "quadrants": ["top-left", "top-right", "bottom-left"],
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {name: sum(r["status"] == name for r in records)
              for name in ("converted", "skipped", "failed")}
    print(f"完成: {counts}; 清单: {manifest}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
