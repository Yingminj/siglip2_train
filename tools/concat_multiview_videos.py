"""
concat_multiview_videos.py
===========================
将三个视角的视频横向拼接为一个多视角视频 (1920x480)。
拼接顺序: wristleft | wristright | head

使用 ffmpeg hstack 滤镜，默认 libx264 -crf 0 无损 H.264 MP4。
只拼接三个视角目录中都存在的同名 .mp4 文件。

用法:
    python tools/concat_multiview_videos.py
    python tools/concat_multiview_videos.py --input_dir /path/to/data --output_dir /path/to/output
    python tools/concat_multiview_videos.py --crf 18   # 视觉无损，体积小很多
"""

import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 系统 ffmpeg 带 libx264，conda 的不带
FFMPEG = "/usr/bin/ffmpeg"


def parse_args():
    parser = argparse.ArgumentParser(description='横向拼接三视角视频')
    parser.add_argument('--input_dir', type=str,
                        default="/home/kewei/spatial_encoder/clip/data/unlableddata/0623gift",
                        help='输入根目录（包含 wristleft/wristright/head 子目录）')
    parser.add_argument('--output_dir', type=str,
                        default="/home/kewei/spatial_encoder/clip/data/unlableddata/0623gift_multi",
                        help='输出目录')
    parser.add_argument('--crf', type=int, default=18,
                        help='编码质量 (0=数学无损, 17=视觉无损, 23=默认)')
    parser.add_argument('--preset', type=str, default='ultrafast',
                        help='编码速度 (ultrafast/fast/medium/slow)')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行处理数')
    return parser.parse_args()


VIEW_ORDER = ['wristleft', 'wristright', 'head']


def find_common_videos(input_dir):
    """找到三个视角目录中都存在的 .mp4 文件名。"""
    sets = []
    for view in VIEW_ORDER:
        view_dir = os.path.join(input_dir, view)
        if not os.path.isdir(view_dir):
            raise FileNotFoundError(f"视角目录不存在: {view_dir}")
        mp4s = {f for f in os.listdir(view_dir) if f.lower().endswith('.mp4')}
        sets.append(mp4s)

    common = sorted(sets[0] & sets[1] & sets[2])
    return common


def concat_video(input_dir, output_path, filename, crf, preset):
    """用 ffmpeg hstack 拼接单个视频，libx264 编码输出 MP4。"""
    inputs = []
    for view in VIEW_ORDER:
        inputs.append(os.path.join(input_dir, view, filename))

    cmd = [
        FFMPEG, '-y',
        '-i', inputs[0],
        '-i', inputs[1],
        '-i', inputs[2],
        '-filter_complex', 'hstack=inputs=3',
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', 'yuv420p',
        '-an',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return filename, False, result.stderr.strip().split('\n')[-1]
    return filename, True, None


def main():
    args = parse_args()

    quality_desc = '无损' if args.crf == 0 else ('视觉无损' if args.crf <= 17 else '有损')

    print("=" * 60)
    print("三视角视频横向拼接")
    print("=" * 60)
    print(f"输入目录  : {args.input_dir}")
    print(f"输出目录  : {args.output_dir}")
    print(f"拼接顺序  : {' | '.join(VIEW_ORDER)}")
    print(f"编码      : libx264  CRF={args.crf} ({quality_desc})  preset={args.preset}")
    print(f"并行数    : {args.workers}")
    print("=" * 60)

    common = find_common_videos(args.input_dir)
    print(f"\n共找到 {len(common)} 个三视角匹配的视频")

    if not common:
        print("无匹配视频，退出。")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 同时拷贝 head 目录下的 label 文件
    head_dir = os.path.join(args.input_dir, 'head')
    label_count = 0
    for f in os.listdir(head_dir):
        if f.endswith('_lable.txt') or f.endswith('_label.txt'):
            src = os.path.join(head_dir, f)
            dst = os.path.join(args.output_dir, f)
            subprocess.run(['cp', src, dst])
            label_count += 1
    if label_count > 0:
        print(f"已复制 {label_count} 个标签文件到输出目录")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for filename in common:
            output_path = os.path.join(args.output_dir, filename)
            future = executor.submit(
                concat_video, args.input_dir, output_path, filename,
                args.crf, args.preset)
            futures[future] = filename

        for i, future in enumerate(as_completed(futures)):
            fname, ok, err = future.result()
            if ok:
                success_count += 1
                print(f"  [{i+1}/{len(common)}] {fname} ✓")
            else:
                fail_count += 1
                print(f"  [{i+1}/{len(common)}] {fname} ✗ {err}")

    print(f"\n完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出: {args.output_dir}")


if __name__ == "__main__":
    main()
