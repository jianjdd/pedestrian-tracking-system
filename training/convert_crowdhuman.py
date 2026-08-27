"""
CrowdHuman 数据集 → YOLO 格式转换脚本

CrowdHuman 下载地址: https://www.crowdhuman.org/
下载后需要的文件:
  - annotation_train.odgt  (训练集标注)
  - annotation_val.odgt    (验证集标注)
  - Images/                (所有图像)

使用方法:
  python convert_crowdhuman.py --crowdhuman_root E:/datasets/CrowdHuman --output_dir E:/datasets/CrowdHuman_yolo

可选参数:
  --box_type  fbox | vbox (默认 vbox)
      fbox = 完整人体框 (full body box)，包含被遮挡部分
      vbox = 可见部分框 (visible box)，仅标注可见区域
    建议密集场景使用 vbox，能让模型学到"只标可见部分"的能力
"""
import json
import os
import shutil
import argparse
from pathlib import Path
from PIL import Image


def parse_odgt(odgt_path):
    """解析 CrowdHuman 的 .odgt 标注文件"""
    annotations = []
    with open(odgt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def convert_crowdhuman(crowdhuman_root, output_dir, box_type='vbox'):
    """
    将 CrowdHuman 转为 YOLO 格式
    Args:
        crowdhuman_root: CrowdHuman 数据集根目录
        output_dir: 输出目录
        box_type: 'fbox' 或 'vbox'
    """
    crowdhuman_root = Path(crowdhuman_root)
    output_dir = Path(output_dir)

    # 创建输出目录
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 处理训练集和验证集
    splits = {
        'train': crowdhuman_root / 'annotation_train.odgt',
        'val': crowdhuman_root / 'annotation_val.odgt',
    }

    # CrowdHuman 图像可能在多个子目录中
    image_dirs = []
    for candidate in ['Images', 'images', 'CrowdHuman_train', 'CrowdHuman_val']:
        p = crowdhuman_root / candidate
        if p.is_dir():
            image_dirs.append(p)
    if not image_dirs:
        image_dirs = [crowdhuman_root]  # fallback: 直接在根目录找

    total_images = 0
    total_boxes = 0

    for split, odgt_path in splits.items():
        if not odgt_path.exists():
            print(f"⚠️  跳过 {split}: 找不到 {odgt_path}")
            continue

        annotations = parse_odgt(odgt_path)
        split_images = 0
        split_boxes = 0

        for anno in annotations:
            image_id = anno['ID']

            # 查找图像文件
            src_img = None
            for ext in ['.jpg', '.png', '.jpeg']:
                for img_dir in image_dirs:
                    candidate = img_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
                if src_img:
                    break

            if src_img is None:
                continue

            # 获取图像尺寸
            try:
                with Image.open(src_img) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

            # 解析标注
            labels = []
            for gt in anno.get('gtboxes', []):
                # 忽略 "mask" 类型（非行人）
                if gt.get('tag', '') != 'person':
                    continue

                # 获取选定类型的框
                box = gt.get(box_type) or gt.get('fbox')
                if box is None:
                    continue

                # CrowdHuman 格式: [x, y, w, h] (左上角 + 宽高，像素值)
                x, y, w, h = box

                # 过滤无效框
                if w <= 0 or h <= 0:
                    continue

                # 转为 YOLO 归一化格式
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # 裁剪到 [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.001, min(1.0, w_norm))
                h_norm = max(0.001, min(1.0, h_norm))

                labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if not labels:
                continue

            # 复制图像
            dst_img = output_dir / 'images' / split / f"{image_id}.jpg"
            shutil.copy2(str(src_img), str(dst_img))

            # 写入标注
            dst_label = output_dir / 'labels' / split / f"{image_id}.txt"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(labels))

            split_images += 1
            split_boxes += len(labels)

        total_images += split_images
        total_boxes += split_boxes
        print(f"✅ {split}: {split_images} 张图像, {split_boxes} 个标注框")

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

    print(f"\n🎉 转换完成！共 {total_images} 张图像, {total_boxes} 个标注框")
    print(f"   data.yaml: {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrowdHuman → YOLO 格式转换')
    parser.add_argument('--crowdhuman_root', type=str, required=True,
                        help='CrowdHuman 数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='YOLO 格式输出目录')
    parser.add_argument('--box_type', type=str, default='vbox', choices=['fbox', 'vbox'],
                        help='使用哪种框: fbox(全身) 或 vbox(可见部分), 默认 vbox')

    args = parser.parse_args()
    convert_crowdhuman(args.crowdhuman_root, args.output_dir, args.box_type)
