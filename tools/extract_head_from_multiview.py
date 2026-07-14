"""
extract_head_from_multiview.py
================================
从多视角拼接图像(1920x480)中截取头部视角(最右640x480)。

多视角图像布局: [left_wrist(640x480) | right_wrist(640x480) | head(640x480)]
本脚本提取最右边的 head 视角并保存到目标目录，保持原有的子目录结构。

用法示例:
    python tools/extract_head_from_multiview.py
    python tools/extract_head_from_multiview.py --input_dir /path/to/multi --output_dir /path/to/head
"""

import os
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description='从多视角图像中截取头部视角')
    parser.add_argument('--input_dir', type=str,
                        default="/home/kewei/spatial_encoder/clip/data/test_data/0622gift_picture_test10",
                        help='输入目录（包含多视角拼接图像）')
    parser.add_argument('--output_dir', type=str,
                        default="/home/kewei/spatial_encoder/clip/data/test_data/0622gift_picture_test10_head",
                        help='输出目录（头部视角图像）')
    parser.add_argument('--view', type=str, default='head',
                        choices=['left_wrist', 'right_wrist', 'head'],
                        help='要截取的视角 (default: head)')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行工作线程数')
    return parser.parse_args()


# 1920x480 三视角布局的裁剪区域
VIEW_CROPS = {
    'left_wrist':  (0, 0, 640, 480),
    'right_wrist': (640, 0, 1280, 480),
    'head':        (1280, 0, 1920, 480),
}


def process_image(input_path, output_path, crop_box):
    """裁剪单张图像并保存。"""
    try:
        with Image.open(input_path) as img:
            # 验证尺寸
            if img.size != (1920, 480):
                return input_path, False, f"unexpected size {img.size}"
            cropped = img.crop(crop_box)
            cropped.save(output_path, quality=95)
        return input_path, True, None
    except Exception as e:
        return input_path, False, str(e)


def main():
    args = parse_args()
    crop_box = VIEW_CROPS[args.view]

    print("=" * 60)
    print(f"从多视角图像截取 [{args.view}] 视角")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"裁剪区域: {crop_box}")
    print("=" * 60)

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")

    # 收集所有待处理的图片
    tasks = []
    for root, dirs, files in os.walk(args.input_dir):
        dirs.sort()
        rel_dir = os.path.relpath(root, args.input_dir)
        out_dir = os.path.join(args.output_dir, rel_dir)

        image_files = [f for f in sorted(files)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        if image_files:
            os.makedirs(out_dir, exist_ok=True)
            for fname in image_files:
                input_path = os.path.join(root, fname)
                output_path = os.path.join(out_dir, fname)
                tasks.append((input_path, output_path))

    print(f"\n共发现 {len(tasks)} 张图片待处理")

    # 并行处理
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, inp, out, crop_box): (inp, out)
            for inp, out in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            path, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  Error: {os.path.basename(path)}: {err}")

            if (i + 1) % 200 == 0:
                print(f"  进度: {i+1}/{len(tasks)}")

    print(f"\n完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
