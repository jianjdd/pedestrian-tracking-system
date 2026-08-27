"""
YOLO 检测 + StrongSORT 追踪 脚本

使用 YOLO 进行行人检测，StrongSORT (boxmot) 进行多目标追踪。
支持视频文件、摄像头输入，可保存带追踪结果的视频。

依赖:
  pip install boxmot opencv-python

使用方法:
  # 基本用法（视频文件）
  python track_strongsort.py --source video.mp4

  # 指定检测模型权重
  python track_strongsort.py --source video.mp4 --yolo-weights ../weights/yolov8n.pt

  # 使用摄像头
  python track_strongsort.py --source 0

  # 保存输出视频
  python track_strongsort.py --source video.mp4 --save --save-path output.mp4

  # 调整检测置信度和 NMS IoU
  python track_strongsort.py --source video.mp4 --conf 0.5 --iou 0.45
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 将本地 ultralytics 源码仓库加入搜索路径
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
_repo_root = os.path.normpath(_repo_root)
if os.path.isdir(os.path.join(_repo_root, 'ultralytics')):
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']

from ultralytics import YOLO

# 尝试导入 boxmot
try:
    from boxmot import StrongSORT
    HAS_BOXMOT = True
except ImportError:
    HAS_BOXMOT = False
    print("⚠️  未检测到 boxmot 库，请运行: pip install boxmot")


# ─── 颜色工具 ─────────────────────────────────────────────
def id_to_color(track_id: int):
    """根据 track_id 生成稳定的颜色"""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


# ─── 核心逻辑 ─────────────────────────────────────────────
def run_tracking(args):
    """执行 YOLO 检测 + StrongSORT 追踪"""

    # ---------- 0. 前置检查 ----------
    if not HAS_BOXMOT:
        print("❌ boxmot 未安装，无法使用 StrongSORT。")
        return

    import torch
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- 1. 加载 YOLO 模型 ----------
    print(f"📦 加载 YOLO 模型: {args.yolo_weights}")
    model = YOLO(args.yolo_weights)

    # ---------- 2. 初始化 StrongSORT ----------
    # 权重路径：优先使用参数，否则在 weights/ 目录下查找
    reid_weights = Path(args.reid_weights)
    if not reid_weights.exists():
        reid_weights = Path(__file__).resolve().parent.parent / 'weights' / args.reid_weights
    if not reid_weights.exists():
        print(f"❌ ReID 权重文件未找到: {args.reid_weights}")
        print("   请将 osnet_x0_25_msmt17.pt 放入项目 weights/ 目录")
        return

    # 密集场景模式：覆盖关键参数
    if args.dense_mode:
        args.max_dist = 0.4
        args.max_iou_dist = 0.9
        args.max_age = 50
        args.n_init = 2
        print("🔧 密集场景模式已启用 (max_dist=0.4, max_iou_dist=0.9, max_age=50, n_init=2)")

    print(f"🔗 加载 StrongSORT (ReID 权重: {reid_weights}, 设备: {device})")
    # StrongSORT++ 密集场景优化初始化
    tracker = StrongSORT(
        model_weights=reid_weights,
        device=device,
        fp16=True,                              # FP16 加速
        max_dist=args.max_dist,                 # 外观距离阈值
        max_iou_dist=args.max_iou_dist,         # IoU 距离阈值
        max_age=args.max_age,                   # 丢失目标保留帧数
        n_init=args.n_init,                     # 新目标确认帧数
    )

    # ---------- 3. 打开视频源 ----------
    source = args.source
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)

    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print(f"📹 视频信息: {w}x{h} @ {fps:.1f}fps", end='')
    if total_frames > 0:
        print(f", 共 {total_frames} 帧")
    else:
        print()

    # ---------- 4. 设置视频写入器 ----------
    writer = None
    if args.save:
        save_path = args.save_path or 'output_strongsort.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"💾 结果将保存到: {save_path}")

    # ---------- 5. 逐帧处理 ----------
    frame_idx = 0
    total_time = 0

    print("\n🚀 开始追踪... (按 'q' 退出)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # ---- 5a. YOLO 检测 ----
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            imgsz=args.imgsz,
            verbose=False,
        )

        # 解析检测结果 → [x1, y1, x2, y2, conf, cls]
        det = results[0].boxes
        if det is not None and len(det) > 0:
            dets_np = np.hstack([
                det.xyxy.cpu().numpy(),
                det.conf.cpu().numpy().reshape(-1, 1),
                det.cls.cpu().numpy().reshape(-1, 1),
            ])  # shape: (N, 6)
        else:
            dets_np = np.empty((0, 6))

        # ---- 5b. StrongSORT 更新 ----
        tracks = tracker.update(dets_np, frame)
        # tracks 格式: [x1, y1, x2, y2, id, conf, cls, ...]

        dt = time.time() - t0
        total_time += dt
        current_fps = 1.0 / dt if dt > 0 else 0

        # ---- 5c. 绘制结果 ----
        n_tracks = 0
        if len(tracks) > 0:
            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk[:4])
                track_id = int(trk[4])
                conf = trk[5]
                cls = int(trk[6])
                n_tracks += 1

                color = id_to_color(track_id)

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"ID:{track_id} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # 绘制中心点
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, color, -1)

        # 状态栏
        status = f"Frame {frame_idx}"
        if total_frames > 0:
            status += f"/{total_frames}"
        status += f" | Tracks: {n_tracks} | Dets: {len(dets_np)} | FPS: {current_fps:.1f}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 进度打印
        if frame_idx % 50 == 0 or frame_idx == 1:
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            progress = f"  帧 {frame_idx}"
            if total_frames > 0:
                progress += f"/{total_frames} ({100 * frame_idx / total_frames:.1f}%)"
            progress += f" | 检测: {len(dets_np)} | 追踪: {n_tracks} | 平均FPS: {avg_fps:.1f}"
            print(progress)

        # ---- 5d. 显示 / 保存 ----
        if not args.nosave_display:
            cv2.imshow('YOLO + StrongSORT Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹  用户中断")
                break

        if writer is not None:
            writer.write(frame)

    # ---------- 6. 收尾 ----------
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"\n✅ 追踪完成!")
    print(f"   总帧数: {frame_idx}")
    print(f"   平均FPS: {avg_fps:.1f}")
    if args.save:
        print(f"   输出文件: {args.save_path or 'output_strongsort.mp4'}")


# ─── 入口 ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='YOLO 检测 + StrongSORT 追踪',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 视频文件追踪
  python track_strongsort.py --source video.mp4

  # 使用自定义检测权重
  python track_strongsort.py --source video.mp4 --yolo-weights best.pt

  # 只检测行人（COCO class 0 = person）
  python track_strongsort.py --source video.mp4 --classes 0

  # 保存结果视频
  python track_strongsort.py --source video.mp4 --save --save-path result.mp4
"""
    )

    # 输入/输出
    default_video = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads', 'TownCentreXVID.mp4')
    parser.add_argument('--source', type=str, default=default_video,
                        help='视频路径 (默认使用 uploads 文件夹下的 TownCentre 视频)')
    parser.add_argument('--save', action='store_true',
                        help='保存输出视频')
    parser.add_argument('--save-path', type=str, default=None,
                        help='输出视频路径 (默认 output_strongsort.mp4)')
    parser.add_argument('--nosave-display', action='store_true',
                        help='不显示窗口 (无头模式)')

    # YOLO 参数
    parser.add_argument('--yolo-weights', type=str, default='../runs/dense_pedestrian4/weights/best.pt',
                        help='YOLO 模型权重路径 (默认 ../weights/yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='检测置信度阈值 (默认 0.45)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值 (默认 0.45)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理分辨率 (默认 640)')
    parser.add_argument('--classes', type=int, nargs='+', default=[0],
                        help='检测类别 (默认 [0]=person)')

    # StrongSORT 参数
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt',
                        help='StrongSORT ReID 权重 (建议使用轻量级 osnet_x0_25_msmt17.pt)')
    parser.add_argument('--max-dist', type=float, default=0.2,
                        help='外观特征最大距离阈值 (默认 0.2, 密集场景建议 0.4)')
    parser.add_argument('--max-iou-dist', type=float, default=0.7,
                        help='IoU 距离阈值 (默认 0.7, 密集场景建议 0.9)')
    parser.add_argument('--max-age', type=int, default=30,
                        help='丢失目标保留帧数 (默认 30, 密集场景建议 50)')
    parser.add_argument('--n-init', type=int, default=3,
                        help='新目标确认帧数 (默认 3, 密集场景建议 2)')
    parser.add_argument('--dense-mode', action='store_true',
                        help='启用密集场景预设参数 (自动设置最优参数组合)')

    # 设备
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda:0 / cpu, 默认自动检测)')

    args = parser.parse_args()
    run_tracking(args)


if __name__ == '__main__':
    main()
