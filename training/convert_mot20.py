"""
MOT20 数据集 → YOLO 格式转换脚本

MOT20 下载地址: https://motchallenge.net/data/MOT20/
下载后的目录结构:
  MOT20/
  ├── train/
  │   ├── MOT20-01/
  │   │   ├── gt/gt.txt      ← 标注文件
  │   │   ├── img1/           ← 图像帧
  │   │   └── seqinfo.ini
  │   └── ...
  └── test/
      └── ...

使用方法:
  python convert_mot20.py --mot_root E:/datasets/MOT20/train --output_dir E:/datasets/MOT20_yolo
"""
import os
import shutil
import argparse
from pathlib import Path
from configparser import ConfigParser


def convert_mot20(mot_root, output_dir, val_ratio=0.1):
    """
    将 MOT20 训练集转为 YOLO 格式
    Args:
        mot_root: MOT20/train 目录路径
        output_dir: 输出目录
        val_ratio: 每个序列中用于验证的帧比例（取末尾部分）
    """
    mot_root = Path(mot_root)
    output_dir = Path(output_dir)

    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 处理所有子目录
    sequences = sorted([d for d in mot_root.iterdir() if d.is_dir()])

    if not sequences:
        print(f"❌ 在 {mot_root} 下找不到任何序列目录")
        return

    total_train = 0
    total_val = 0
    total_boxes = 0

    for seq_dir in sequences:
        seq_name = seq_dir.name
        gt_file = seq_dir / 'gt' / 'gt.txt'
        img_dir = seq_dir / 'img1'
        ini_file = seq_dir / 'seqinfo.ini'

        if not gt_file.exists() or not img_dir.exists():
            print(f"⚠️  跳过 {seq_name}: 缺少 gt 或图像目录")
            continue

        # 读取序列信息（图像尺寸）
        config = ConfigParser()
        config.read(str(ini_file))
        img_w = int(config['Sequence']['imWidth'])
        img_h = int(config['Sequence']['imHeight'])

        # 解析 gt.txt，按帧分组
        frame_annotations = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue

                frame_id = int(parts[0])
                conf_flag = int(parts[6])  # 0 = 忽略

                if conf_flag == 0:
                    continue

                # MOT20 class 标注: 1=行人
                cls = int(parts[7]) if len(parts) > 7 else 1
                if cls != 1:
                    continue

                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_w = float(parts[4])
                bb_h = float(parts[5])

                if bb_w <= 0 or bb_h <= 0:
                    continue

                # 转为 YOLO 归一化格式
                x_center = (bb_left + bb_w / 2) / img_w
                y_center = (bb_top + bb_h / 2) / img_h
                w_norm = bb_w / img_w
                h_norm = bb_h / img_h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0.001, min(1, w_norm))
                h_norm = max(0.001, min(1, h_norm))

                if frame_id not in frame_annotations:
                    frame_annotations[frame_id] = []
                frame_annotations[frame_id].append(
                    f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

        if not frame_annotations:
            print(f"⚠️  跳过 {seq_name}: 无有效标注")
            continue

        # 按帧号排序，后 val_ratio 部分作为验证集
        frames = sorted(frame_annotations.keys())
        split_idx = int(len(frames) * (1 - val_ratio))
        train_frames = set(frames[:split_idx])
        val_frames = set(frames[split_idx:])

        seq_boxes = 0
        for frame_id in frames:
            split = 'train' if frame_id in train_frames else 'val'

            # 查找图像文件
            img_name = f"{frame_id:06d}.jpg"
            src_img = img_dir / img_name
            if not src_img.exists():
                continue

            # 使用 seq_name 前缀避免文件名冲突
            dst_name = f"{seq_name}_{img_name}"
            shutil.copy2(str(src_img), str(output_dir / 'images' / split / dst_name))

            label_name = dst_name.replace('.jpg', '.txt')
            with open(output_dir / 'labels' / split / label_name, 'w') as f:
                f.write('\n'.join(frame_annotations[frame_id]))

            box_count = len(frame_annotations[frame_id])
            seq_boxes += box_count

            if split == 'train':
                total_train += 1
            else:
                total_val += 1

        total_boxes += seq_boxes
        print(f"✅ {seq_name}: {len(train_frames)} train + {len(val_frames)} val 帧, {seq_boxes} 个标注框")

    # 生成 data.yaml
    yaml_path = output_dir / 'data.yaml'
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: 1
names: ['person']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n🎉 转换完成！train={total_train}, val={total_val}, 共 {total_boxes} 个标注框")
    print(f"   data.yaml: {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOT20 → YOLO 格式转换')
    parser.add_argument('--mot_root', type=str, required=True,
                        help='MOT20/train 目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='YOLO 格式输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (默认 0.1)')

    args = parser.parse_args()
    convert_mot20(args.mot_root, args.output_dir, args.val_ratio)
