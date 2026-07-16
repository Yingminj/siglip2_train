#!/usr/bin/env python3
"""Concatenate one complete M1->M6 sequence and generate frame labels.

Example:
  python tools/concat_labeled_state_videos.py \
    --m1 /path/M1/video.mp4 --m2 /path/M2/video.mp4 \
    --m3 /path/M3/video.mp4 --m4 /path/M4/video.mp4 \
    --m5 /path/M5/video.mp4 --m6 /path/M6/video.mp4 \
    --output /path/sequence_01.mp4
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for state in range(1, 7):
        parser.add_argument(f"--m{state}", required=True, type=Path,
                            help=f"M{state} 视频路径")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    videos = [getattr(args, f"m{state}").expanduser().resolve()
              for state in range(1, 7)]
    for path in videos:
        if not path.is_file():
            raise SystemExit(f"视频不存在: {path}")

    frame_counts = []
    fps_values = []
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if count < 1 or width < 1 or height < 1:
            raise SystemExit(f"无法读取视频信息: {path}")
        frame_counts.append(count)
        fps_values.append(fps)

    if max(fps_values) - min(fps_values) > 0.1:
        print(f"警告: 输入视频帧率不同，将统一为 {fps_values[0]:.3f} FPS")
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_path = args.output.with_name(args.output.stem + "_lable.txt")

    # concat demuxer avoids placing all six videos in memory. Re-encoding
    # makes frame boundaries and the generated labels deterministic.
    with tempfile.TemporaryDirectory() as temp_dir:
        list_path = Path(temp_dir) / "concat.txt"
        concat_lines = []
        for path in videos:
            escaped = str(path).replace("'", "'\\''")
            concat_lines.append("file '" + escaped + "'\n")
        list_path.write_text("".join(concat_lines), encoding="utf-8")
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
            "-safe", "0", "-i", str(list_path), "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-r", f"{fps_values[0]:.6f}", str(args.output),
        ]
        subprocess.run(command, check=True)

    # Labels are one line per output frame: frame_index state_id.
    with label_path.open("w", encoding="utf-8") as handle:
        frame_index = 0
        for state_id, count in enumerate(frame_counts, 1):
            for _ in range(count):
                handle.write(f"{frame_index} {state_id}\n")
                frame_index += 1

    print(f"视频已生成: {args.output}")
    print(f"标签已生成: {label_path}")
    print("状态顺序: M1 -> M2 -> M3 -> M4 -> M5 -> M6")
    print(f"原始帧数: {frame_counts}; 标签总帧数: {sum(frame_counts)}")


if __name__ == "__main__":
    main()
