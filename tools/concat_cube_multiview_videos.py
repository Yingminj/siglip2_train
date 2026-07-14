#!/usr/bin/env python3
"""将 cube_data 中的三路同步视频横向拼接。

默认输入目录结构：

    cube_data/
    ├── data_eyes/
    ├── data_left/
    └── data_right/

输出布局为 ``eyes | left | right``。对于 960x540 的原视频，输出尺寸为
2880x540，符合 multiview_models.split_image_to_views() 的三等分规则。

示例：
    python tools/concat_cube_multiview_videos.py
    python tools/concat_cube_multiview_videos.py --workers 4 --crf 18
    python tools/concat_cube_multiview_videos.py --overwrite
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VIEW_DIRS = ("data_eyes", "data_left", "data_right")


def parse_args():
    parser = argparse.ArgumentParser(description="横向拼接 cube_data 三视角视频")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/liuqian/Aqcy/cube_data"),
        help="包含 data_eyes、data_left、data_right 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认是 <input-dir>/merged",
    )
    parser.add_argument("--crf", type=int, default=18, help="H.264 质量，默认 18")
    parser.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"),
        default="medium",
        help="编码速度/压缩率，默认 medium",
    )
    parser.add_argument("--workers", type=int, default=2, help="并行任务数，默认 2")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的输出；默认跳过",
    )
    return parser.parse_args()


def collect_videos(input_dir):
    """检查三个目录存在且 MP4 文件名集合完全一致。"""
    file_sets = {}
    for view_dir in VIEW_DIRS:
        directory = input_dir / view_dir
        if not directory.is_dir():
            raise SystemExit(f"错误：视角目录不存在：{directory}")
        file_sets[view_dir] = {
            path.name for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        }

    reference = file_sets[VIEW_DIRS[0]]
    errors = []
    for view_dir in VIEW_DIRS[1:]:
        missing = sorted(reference - file_sets[view_dir])
        extra = sorted(file_sets[view_dir] - reference)
        if missing:
            errors.append(f"{view_dir} 缺少：{', '.join(missing[:10])}")
        if extra:
            errors.append(f"{view_dir} 多出：{', '.join(extra[:10])}")
    if errors:
        raise SystemExit("错误：三路视频无法一一对应\n" + "\n".join(errors))
    if not reference:
        raise SystemExit("错误：没有找到 MP4 视频")
    return sorted(reference)


def merge_one(ffmpeg, input_dir, output_dir, filename, crf, preset, overwrite):
    output_path = output_dir / filename
    if output_path.exists() and not overwrite:
        return filename, "skipped", "输出已存在"

    inputs = [input_dir / view_dir / filename for view_dir in VIEW_DIRS]
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(inputs[0]),
        "-i", str(inputs[1]),
        "-i", str(inputs[2]),
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        return filename, "failed", message[-1] if message else "ffmpeg 执行失败"
    return filename, "success", ""


def main():
    args = parse_args()
    if not 0 <= args.crf <= 51:
        raise SystemExit("错误：--crf 必须在 0 到 51 之间")
    if args.workers < 1:
        raise SystemExit("错误：--workers 必须大于等于 1")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("错误：系统中没有找到 ffmpeg")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "merged"
    )
    filenames = collect_videos(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print("视角顺序：data_eyes | data_left | data_right")
    print(f"视频数量：{len(filenames)}")
    print(f"编码参数：libx264, CRF={args.crf}, preset={args.preset}")

    counts = {"success": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                merge_one,
                ffmpeg,
                input_dir,
                output_dir,
                filename,
                args.crf,
                args.preset,
                args.overwrite,
            )
            for filename in filenames
        ]
        for index, future in enumerate(as_completed(futures), 1):
            filename, status, message = future.result()
            counts[status] += 1
            symbol = {"success": "✓", "skipped": "-", "failed": "✗"}[status]
            detail = f"（{message}）" if message else ""
            print(f"[{index:03d}/{len(filenames):03d}] {symbol} {filename}{detail}")

    print(
        f"完成：成功 {counts['success']}，跳过 {counts['skipped']}，"
        f"失败 {counts['failed']}"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
