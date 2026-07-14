#!/usr/bin/env python3
"""Convert a 2x2 camera video into a 3-view horizontal video.

Input:  1280x960 (four 640x480 quadrants)
Output: 1920x480 ([view1 | view2 | view3])

Example:
    python tools/convert_2x2_video_to_3view.py input.mp4 output.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path


QUADRANTS = {
    "tl": (0, 0),
    "tr": (1, 0),
    "bl": (0, 1),
    "br": (1, 1),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入 2x2 视频")
    parser.add_argument("output", type=Path, help="输出三视角视频")
    parser.add_argument(
        "--views", nargs=3, default=["tl", "tr", "bl"], choices=sorted(QUADRANTS),
        metavar=("VIEW1", "VIEW2", "VIEW3"),
        help="三个视角的象限，默认 tl tr bl；可选 tl/tr/bl/br",
    )
    parser.add_argument("--fps", type=float, default=None,
                        help="输出帧率，默认沿用输入帧率")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 质量参数，数值越小质量越高，默认 18")
    return parser.parse_args()


def main():
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("错误：未找到 ffmpeg，请先安装 ffmpeg。")
    if not args.input.is_file():
        raise SystemExit(f"错误：输入视频不存在：{args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    filters = []
    labels = []
    for i, name in enumerate(args.views):
        x, y = QUADRANTS[name]
        label = f"v{i}"
        filters.append(f"[0:v]crop=iw/2:ih/2:{x}*iw/2:{y}*ih/2[{label}]")
        labels.append(f"[{label}]")
    filters.append("".join(labels) + "hstack=inputs=3[outv]")

    command = [
        "ffmpeg", "-y", "-i", str(args.input),
        "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
    ]
    if args.fps is not None:
        command += ["-r", str(args.fps)]
    command.append(str(args.output))

    print("执行：", " ".join(command))
    subprocess.run(command, check=True)
    print(f"转换完成：{args.output}")
    print("输出布局：", " | ".join(args.views), "，尺寸应为 1920x480")


if __name__ == "__main__":
    main()
