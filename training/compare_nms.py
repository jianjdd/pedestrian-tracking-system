"""
Standard NMS vs Soft-NMS 对比脚本

功能:
  - 验证模式 (--mode val): 在数据集上对比两种 NMS 的检测精度 (mAP, Precision, Recall)
  - 预测模式 (--mode predict): 在图片/视频上对比两种 NMS 的可视化结果

用法:
  # 精度对比（需要数据集 yaml）
  python training/compare_nms.py --mode val --model runs/exp4_full/weights/best.pt --data data/CrowdHuman_YOLO/data.yaml

  # 可视化对比（单张图）
  python training/compare_nms.py --mode predict --model runs/exp4_full/weights/best.pt --source path/to/image.jpg

  # 可视化对比（图片文件夹）
  python training/compare_nms.py --mode predict --model runs/exp4_full/weights/best.pt --source path/to/folder

  # 同时跑两种模式
  python training/compare_nms.py --mode all --model runs/exp4_full/weights/best.pt --data data/CrowdHuman_YOLO/data.yaml --source path/to/image.jpg
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ultralytics_path = PROJECT_ROOT / "ultralytics"
if str(ultralytics_path) not in sys.path:
    sys.path.insert(0, str(ultralytics_path))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS


def benchmark_nms_speed(n_boxes=5000, n_rounds=100, device="cpu"):
    """测试 Standard NMS 与 Soft-NMS 的纯后处理速度（不含模型推理）。

    Args:
        n_boxes: 模拟候选框数量
        n_rounds: 测试轮数
        device: 测试设备

    Returns:
        dict: 各方法平均延迟 (ms)
    """
    torch.manual_seed(42)
    boxes = torch.rand(n_boxes, 4, device=device) * 640
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.rand(n_boxes, device=device)

    results = {}

    # Warm up
    for _ in range(10):
        TorchNMS.nms(boxes, scores, 0.5)
        TorchNMS.soft_nms(boxes, scores, 0.5, sigma=0.5, method="gaussian")
        TorchNMS.soft_nms(boxes, scores, 0.5, method="linear")
    if device == "cuda":
        torch.cuda.synchronize()

    # Standard NMS
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        TorchNMS.nms(boxes, scores, 0.5)
    if device == "cuda":
        torch.cuda.synchronize()
    results["Standard NMS"] = (time.perf_counter() - t0) / n_rounds * 1000

    # Soft-NMS Gaussian
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        TorchNMS.soft_nms(boxes, scores, 0.5, sigma=0.5, method="gaussian")
    if device == "cuda":
        torch.cuda.synchronize()
    results["Soft-NMS Gaussian"] = (time.perf_counter() - t0) / n_rounds * 1000

    # Soft-NMS Linear
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        TorchNMS.soft_nms(boxes, scores, 0.5, method="linear")
    if device == "cuda":
        torch.cuda.synchronize()
    results["Soft-NMS Linear"] = (time.perf_counter() - t0) / n_rounds * 1000

    return results


def compare_val(model_path, data_yaml, device, imgsz=640, conf=0.001, iou=0.7):
    """在验证集上对比 Standard NMS 与 Soft-NMS 的检测精度。

    Args:
        model_path: 模型权重路径
        data_yaml: 数据集 yaml 路径
        device: 计算设备
        imgsz: 推理尺寸
        conf: 置信度阈值
        iou: NMS IoU 阈值

    Returns:
        dict: key 为方法名，value 为 metrics 对象
    """
    print("\n" + "=" * 60)
    print("  验证集精度对比: Standard NMS vs Soft-NMS")
    print("=" * 60)

    results = {}

    # ---------- Standard NMS ----------
    print("\n[1/2] 正在运行 Standard NMS 验证...")
    model = YOLO(model_path)
    metrics_std = model.val(
        data=data_yaml,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        soft_nms=False,
        split="val",
        verbose=True,
    )
    results["Standard NMS"] = metrics_std

    # ---------- Soft-NMS ----------
    print("\n[2/2] 正在运行 Soft-NMS (Gaussian) 验证...")
    model = YOLO(model_path)
    metrics_soft = model.val(
        data=data_yaml,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        soft_nms=True,
        soft_nms_sigma=0.5,
        soft_nms_method="gaussian",
        split="val",
        verbose=True,
    )
    results["Soft-NMS Gaussian"] = metrics_soft

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("  检测精度对比结果")
    print("=" * 60)
    print(f"{'Method':<25} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>10}")
    print("-" * 69)
    for name, m in results.items():
        rd = m.results_dict
        print(
            f"{name:<25} {rd.get('metrics/mAP50(B)', 0):>10.4f} "
            f"{rd.get('metrics/mAP50-95(B)', 0):>14.4f} "
            f"{rd.get('metrics/precision(B)', 0):>10.4f} "
            f"{rd.get('metrics/recall(B)', 0):>10.4f}"
        )

    return results


def compare_predict(model_path, source, device, conf=0.25, iou=0.7, save_dir=None):
    """在单张/多张图片上对比 Standard NMS 与 Soft-NMS 的检测结果并保存并排图。

    Args:
        model_path: 模型权重路径
        source: 图片路径或图片文件夹路径
        device: 计算设备
        conf: 置信度阈值
        iou: NMS IoU 阈值
        save_dir: 输出目录（默认 training/output/nms_compare/）
    """
    print("\n" + "=" * 60)
    print("  可视化对比: Standard NMS vs Soft-NMS")
    print("=" * 60)

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "output" / "nms_compare"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片路径
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = sorted([p for p in source_path.iterdir() if p.suffix.lower() in exts])
    else:
        print(f"❌ 无效的 source: {source}")
        return

    if not image_paths:
        print(f"❌ 未找到图片文件")
        return

    image_paths = image_paths[:20]  # 限制最多 20 张

    # 模型只加载一次，通过 predict kwargs 切换 soft_nms
    model = YOLO(model_path)

    idx = 0
    for img_path in image_paths:
        print(f"\n[{idx + 1}/{len(image_paths)}] 处理: {img_path.name}")

        # Standard NMS
        results_std = model.predict(
            source=str(img_path),
            device=device,
            conf=conf,
            iou=iou,
            soft_nms=False,
            verbose=False,
        )

        # Soft-NMS
        results_soft = model.predict(
            source=str(img_path),
            device=device,
            conf=conf,
            iou=iou,
            soft_nms=True,
            soft_nms_sigma=0.5,
            soft_nms_method="gaussian",
            verbose=False,
        )

        # 绘制并排对比图
        img_std = results_std[0].plot(conf=False)
        img_soft = results_soft[0].plot(conf=False)

        n_std = len(results_std[0].boxes)
        n_soft = len(results_soft[0].boxes)
        diff = n_soft - n_std

        # 拼接：左 Standard | 右 Soft-NMS
        h, w = img_std.shape[:2]
        panel = np.zeros((h + 40, w * 2, 3), dtype=np.uint8)

        # 标题栏
        cv2.putText(
            panel, f"Standard NMS ({n_std} detections)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            panel, f"Soft-NMS Gaussian ({n_soft} detections, diff={diff:+d})",
            (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        panel[40:, :w] = img_std
        panel[40:, w:] = img_soft

        out_path = save_dir / f"compare_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), panel)
        print(f"  → 保存: {out_path}")
        print(f"  Standard NMS: {n_std} 框 | Soft-NMS: {n_soft} 框 | 差异: {diff:+d}")

        idx += 1

    print(f"\n✅ 可视化对比完成，结果保存至: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standard NMS vs Soft-NMS 对比")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["val", "predict", "speed", "all"],
                        help="对比模式: val(精度), predict(可视化), speed(速度), all(全部)")
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的 best.pt 权重路径")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集 yaml 路径（val 模式必需）")
    parser.add_argument("--source", type=str, default=None,
                        help="图片/文件夹路径（predict 模式）")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: cpu, cuda:0 等（默认自动选择）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理尺寸（默认 640）")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="val 模式置信度阈值（默认 0.001）")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值（默认 0.7）")
    parser.add_argument("--nms-speed-boxes", type=int, default=5000,
                        help="speed 模式模拟候选框数（默认 5000）")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="predict 模式输出目录")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- 速度对比 ----
    if args.mode in ("speed", "all"):
        print("\n" + "=" * 60)
        print(f"  NMS 后处理速度对比 (候选框={args.nms_speed_boxes}, 设备={args.device})")
        print("=" * 60)
        speed = benchmark_nms_speed(
            n_boxes=args.nms_speed_boxes, n_rounds=100, device=args.device
        )
        print(f"\n{'Method':<25} {'平均延迟/ms':>12}")
        print("-" * 37)
        for name, lat in speed.items():
            print(f"{name:<25} {lat:>12.2f}")

    # ---- 精度对比 ----
    if args.mode in ("val", "all"):
        if args.data is None:
            print("❌ val 模式需要 --data 参数指定数据集 yaml")
            sys.exit(1)
        compare_val(args.model, args.data, args.device, args.imgsz, args.conf, args.iou)

    # ---- 可视化对比 ----
    if args.mode in ("predict", "all"):
        if args.source is None:
            print("❌ predict 模式需要 --source 参数（图片路径或文件夹）")
        else:
            compare_predict(args.model, args.source, args.device, save_dir=args.save_dir)
