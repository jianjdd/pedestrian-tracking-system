"""
YOLO 检测 + BoTSORT 追踪 脚本（使用 ultralytics 内置追踪器）

与 StrongSORT 版本的区别:
  - StrongSORT: 每帧对每个检测框额外跑一次 ReID 网络 → 精度高但慢
  - BoTSORT:    卡尔曼滤波 + IoU 匹配 + 光流补偿 → 速度快，无额外模型开销
  - ByteTrack:  与 BoTSORT 类似，但不含光流补偿，更轻量

使用方法:
  # 默认使用 BoTSORT
  python track_botsort.py

  # 切换为 ByteTrack
  python track_botsort.py --tracker bytetrack

  # 指定视频和模型
  python track_botsort.py --source video.mp4 --yolo-weights best.pt

  # 保存结果
  python track_botsort.py --save --save-path result.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from collections import defaultdict, deque

# 将本地 ultralytics 源码仓库加入搜索路径
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
_repo_root = os.path.normpath(_repo_root)
if os.path.isdir(os.path.join(_repo_root, 'ultralytics')):
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']

from ultralytics import YOLO


# ─── 颜色工具 ─────────────────────────────────────────────
def id_to_color(track_id: int):
    """根据 track_id 生成稳定的颜色"""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


# ─── 核心逻辑 ─────────────────────────────────────────────
def run_tracking(args):
    """执行 YOLO 检测 + BoTSORT/ByteTrack 追踪"""

    # ---------- 1. 加载 YOLO 模型 ----------
    yolo_path = args.yolo_weights
    if not os.path.isabs(yolo_path):
        yolo_path = os.path.join(os.path.dirname(__file__), yolo_path)
    yolo_path = os.path.normpath(yolo_path)

    print(f"📦 加载 YOLO 模型: {yolo_path}")
    model = YOLO(yolo_path)

    # ---------- 2. 确定 tracker 配置文件 ----------
    # ultralytics 内置的 botsort.yaml / bytetrack.yaml
    tracker_yaml = os.path.join(
        os.path.dirname(__file__), '..', 'ultralytics', 'ultralytics',
        'cfg', 'trackers', f'{args.tracker}.yaml'
    )
    tracker_yaml = os.path.normpath(tracker_yaml)

    if not os.path.exists(tracker_yaml):
        # 如果本地没有，让 ultralytics 自己找默认的
        tracker_yaml = f'{args.tracker}.yaml'

    print(f"🔗 追踪器: {args.tracker} ({tracker_yaml})")

    # ---------- 3. 打开视频源 ----------
    source = args.source
    # 移除错误的路径叠加逻辑
    source = os.path.normpath(source)

    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else source)

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
        save_path = args.save_path or 'output_botsort.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"💾 结果将保存到: {save_path}")

    # ---------- 5. 逐帧处理 ----------
    frame_idx = 0
    total_time = 0
    track_history = defaultdict(lambda: deque(maxlen=30))  # 保存每个 ID 的历史轨迹中心点

    print(f"\n🚀 开始追踪 ({args.tracker})... (按 'q' 退出)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # ---- 5a. YOLO 检测 + 内置追踪（一步到位） ----
        results = model.track(
            frame,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            imgsz=args.imgsz,
            tracker=tracker_yaml,
            persist=True,        # 跨帧保持追踪状态
            soft_nms=True,
            verbose=False,
        )

        dt = time.time() - t0
        total_time += dt
        current_fps = 1.0 / dt if dt > 0 else 0

        # ---- 5b. 解析结果并绘制 ----
        result = results[0]
        boxes = result.boxes
        n_dets = len(boxes) if boxes is not None else 0
        n_tracks = 0

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            for box in boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
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

                # 绘制中心点并记录轨迹
                cx, cy = int((x1 + x2) / 2), int(y2)  # 使用底边中心(脚部)作为轨迹点更合理
                cv2.circle(frame, (cx, cy), 4, color, -1)

                track = track_history[track_id]
                track.append((cx, cy))

                # 绘制历史运动轨迹连线
                if len(track) > 1:
                    points = np.array(track).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

        # 状态栏
        status = f"Frame {frame_idx}"
        if total_frames > 0:
            status += f"/{total_frames}"
        status += f" | Tracks: {n_tracks} | Dets: {n_dets} | FPS: {current_fps:.1f}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 进度打印
        if frame_idx % 50 == 0 or frame_idx == 1:
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            progress = f"  帧 {frame_idx}"
            if total_frames > 0:
                progress += f"/{total_frames} ({100 * frame_idx / total_frames:.1f}%)"
            progress += f" | 检测: {n_dets} | 追踪: {n_tracks} | 平均FPS: {avg_fps:.1f}"
            print(progress)

        # ---- 5c. 显示 / 保存 ----
        if not args.nosave_display:
            cv2.imshow(f'YOLO + {args.tracker.upper()} Tracking', frame)
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
    print(f"   追踪器: {args.tracker}")
    print(f"   总帧数: {frame_idx}")
    print(f"   平均FPS: {avg_fps:.1f}")
    if args.save:
        print(f"   输出文件: {args.save_path or 'output_botsort.mp4'}")


# ─── 入口 ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='YOLO 检测 + BoTSORT/ByteTrack 追踪（ultralytics 内置）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置运行
  python track_botsort.py

  # 切换 ByteTrack
  python track_botsort.py --tracker bytetrack

  # 自定义检测权重 + 保存
  python track_botsort.py --yolo-weights ../runs/dense_pedestrian4/weights/best.pt --save
"""
    )

    # 输入/输出
    default_video = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads', '2121-155244120_small.mp4')
    parser.add_argument('--source', type=str, default=default_video,
                        help='视频路径 (默认使用 uploads 文件夹下的 TownCentre 视频)')
    parser.add_argument('--save', action='store_true',
                        help='保存输出视频')
    parser.add_argument('--save-path', type=str, default=None,
                        help='输出视频路径 (默认 output_botsort.mp4)')
    parser.add_argument('--nosave-display', action='store_true',
                        help='不显示窗口 (无头模式)')

    # YOLO 参数
    parser.add_argument('--yolo-weights', type=str, default='../runs/dense_pedestrian4/weights/best.pt',
                        help='YOLO 模型权重路径')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='检测置信度阈值 (默认 0.45)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值 (默认 0.45)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理分辨率 (默认 640)')
    parser.add_argument('--classes', type=int, nargs='+', default=[0],
                        help='检测类别 (默认 [0]=person)')

    # 追踪器选择
    parser.add_argument('--tracker', type=str, default='botsort',
                        choices=['botsort', 'bytetrack'],
                        help='追踪器类型 (默认 botsort)')

    # 设备
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda:0 / cpu, 默认自动检测)')

    args = parser.parse_args()
    run_tracking(args)


if __name__ == '__main__':
    main()
