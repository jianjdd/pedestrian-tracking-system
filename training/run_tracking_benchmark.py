"""
MOT20 行人追踪对比试验 — 两阶段分离式评估

架构:
  阶段1: YOLO 检测（每个模型只跑一次）→ 保存检测结果
  阶段2: 追踪器用同一份检测结果独立追踪（BoT-SORT / StrongSORT）
  阶段3: 统一评估 CLEAR 指标（MOTA, IDF1, MOTP, MT, ML, IDSW, FP, FN）

特点:
  - 检测与追踪完全解耦，保证不同追踪器输入完全一致
  - 内置 MOT 指标计算（纯 numpy+scipy，零外部评估依赖）
  - 自动汇总对比表，支持 CSV 导出

依赖（tracker conda 环境）:
  numpy, scipy, opencv-python, ultralytics
  (可选) pip install boxmot  # StrongSORT 需要

使用方法:
  conda activate tracker
  cd training

  # 计划中的 4 组实验
  python run_tracking_benchmark.py \
      --mot_root D:/Datasets/MOT20/train \
      --models \
          ../runs/Sampled_100/exp0_baseline/weights/best.pt \
          ../runs/Sampled_100/exp4_full/weights/best.pt \
      --trackers botsort strongsort
"""

import argparse
import os
import sys
import time
import csv
from collections import defaultdict
from pathlib import Path
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ── 劫持导入: 强制使用本地 ultralytics（含自定义模块 C2f_DCNv2, BiFPN 等）──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_local_ultralytics_root = str(PROJECT_ROOT / 'ultralytics')  # 仓库根目录
_local_ultralytics_pkg = str(PROJECT_ROOT / 'ultralytics' / 'ultralytics')  # Python 包目录

# 1. 清除已缓存的 ultralytics 模块（避免安装版干扰）
for mod_name in list(sys.modules.keys()):
    if mod_name == 'ultralytics' or mod_name.startswith('ultralytics.'):
        del sys.modules[mod_name]

# 2. 将本地仓库和包目录都放入 sys.path 最前面
for p in [_local_ultralytics_root, _local_ultralytics_pkg]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 3. 预导入本地自定义模块（确保 torch.load 反序列化时能找到 C2f_DCNv2 等类）
import ultralytics
import ultralytics.nn.modules.block      # 含 C2f_DCNv2
import ultralytics.nn.modules.conv       # 含 BiFPN_Concat, CoordAtt
import ultralytics.nn.modules.head       # 可能含自定义检测头

from ultralytics import YOLO
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Detection:
    """单帧单目标检测结果"""
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0


@dataclass
class TrackResult:
    """单帧单目标追踪结果（MOT 格式）"""
    frame: int
    track_id: int
    x: float       # bb_left
    y: float       # bb_top
    w: float       # bb_width
    h: float       # bb_height
    conf: float


@dataclass
class SeqMetrics:
    """单个序列的评估指标"""
    seq_name: str
    num_frames: int
    num_gt: int
    num_pred: int
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    mt: float
    ml: float
    fp: int
    fn: int
    id_switches: int
    fragments: int
    fps: float


# ============================================================
# MOT 格式 I/O
# ============================================================

def read_gt(gt_path: str) -> List[TrackResult]:
    """读取 MOT Challenge 格式 ground truth。"""
    results = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame, tid = int(parts[0]), int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf_flag = int(parts[6])
            cls = int(parts[7])
            if conf_flag == 0 or cls != 1 or w <= 0 or h <= 0:
                continue
            results.append(TrackResult(
                frame=frame, track_id=tid, x=x, y=y, w=w, h=h, conf=1.0
            ))
    return results


def write_mot_format(results: List[TrackResult], output_path: str):
    """写入 MOT Challenge 格式结果文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for r in results:
            f.write(
                f"{r.frame},{r.track_id},{r.x:.1f},{r.y:.1f},"
                f"{r.w:.1f},{r.h:.1f},{r.conf:.4f},-1,-1,-1\n"
            )


# ============================================================
# CLEAR MOT 指标计算（纯 numpy+scipy 实现）
# ============================================================

def box_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算两组 xywh boxes 之间的 IoU 矩阵。
    boxes 格式: [N, 4] = [x, y, w, h]（左上角坐标 + 宽高）
    返回: [N, M] IoU 矩阵
    """
    # 转为 [x1, y1, x2, y2]
    b1 = boxes1.copy()
    b1[:, 2] = boxes1[:, 0] + boxes1[:, 2]
    b1[:, 3] = boxes1[:, 1] + boxes1[:, 3]
    b2 = boxes2.copy()
    b2[:, 2] = boxes2[:, 0] + boxes2[:, 2]
    b2[:, 3] = boxes2[:, 1] + boxes2[:, 3]

    inter_x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union_area = area1[:, None] + area2[None, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0)


class MOTEvaluator:
    """
    CLEAR MOT 评估器。

    实现指标:
      - MOTA (Multi-Object Tracking Accuracy)
      - MOTP (Multi-Object Tracking Precision)
      - IDF1 / IDP / IDR (Identification F1 / Precision / Recall)
      - MT (Mostly Tracked), ML (Mostly Lost)
      - ID Switches, Fragments, FP, FN

    参考文献:
      - Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
        Performance: The CLEAR MOT Metrics", 2008
      - Ristani et al., "Performance Measures and a Data Set for
        Multi-Target, Multi-Camera Tracking", ECCV 2016
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate_sequence(
        self,
        preds: List[TrackResult],
        gts: List[TrackResult],
        seq_name: str = "",
    ) -> SeqMetrics:
        """评估单个序列。"""
        # 按帧分组
        pred_by_frame: Dict[int, List[TrackResult]] = defaultdict(list)
        for p in preds:
            pred_by_frame[p.frame].append(p)

        gt_by_frame: Dict[int, List[TrackResult]] = defaultdict(list)
        for g in gts:
            gt_by_frame[g.frame].append(g)

        all_frames = sorted(set(list(pred_by_frame.keys()) + list(gt_by_frame.keys())))

        # 累计统计
        total_fp, total_fn, total_idsw, total_frag = 0, 0, 0, 0
        total_gt = 0
        sum_iou, num_matches = 0.0, 0
        id_tp, id_fp, id_fn = 0, 0, 0
        gt_track_frames: Dict[int, int] = {}
        gt_track_matched: Dict[int, int] = {}
        prev_id_map: Dict[int, int] = {}  # GT ID → 上一帧匹配到的预测 ID
        prev_matched_pred_ids: Dict[int, set] = defaultdict(set)  # 上一帧预测 ID → 匹配的 GT ID 集合（用于检测 fragment）

        for g in gts:
            gt_track_frames[g.track_id] = gt_track_frames.get(g.track_id, 0) + 1

        num_frames = len(all_frames)

        for frame_id in all_frames:
            frame_preds = pred_by_frame.get(frame_id, [])
            frame_gts = gt_by_frame.get(frame_id, [])

            total_gt += len(frame_gts)

            if len(frame_preds) == 0 and len(frame_gts) == 0:
                continue

            if len(frame_preds) == 0:
                total_fn += len(frame_gts)
                id_fn += len(frame_gts)
                continue

            if len(frame_gts) == 0:
                total_fp += len(frame_preds)
                id_fp += len(frame_preds)
                continue

            # 构建 IoU 矩阵并匈牙利匹配
            pred_boxes = np.array([[p.x, p.y, p.w, p.h] for p in frame_preds])
            gt_boxes = np.array([[g.x, g.y, g.w, g.h] for g in frame_gts])
            iou_matrix = box_iou_xywh(pred_boxes, gt_boxes)

            row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

            matched_preds = set()
            matched_gts = set()
            curr_matched_pred_ids: Dict[int, int] = {}  # pred_id → gt_id for this frame

            for p_idx, g_idx in zip(row_ind, col_ind):
                iou = iou_matrix[p_idx, g_idx]
                if iou >= self.iou_threshold:
                    matched_preds.add(p_idx)
                    matched_gts.add(g_idx)

                    sum_iou += iou
                    num_matches += 1

                    pred_tid = frame_preds[p_idx].track_id
                    gt_tid = frame_gts[g_idx].track_id
                    curr_matched_pred_ids[pred_tid] = gt_tid

                    # ID switch 检测
                    if gt_tid in prev_id_map and prev_id_map[gt_tid] != pred_tid:
                        total_idsw += 1
                    prev_id_map[gt_tid] = pred_tid

                    # GT 覆盖
                    gt_track_matched[gt_tid] = gt_track_matched.get(gt_tid, 0) + 1

                    # ID 级别 TP
                    id_tp += 1

            # Fragment 检测：上一帧匹配到的预测 ID 在当前帧出现了但不匹配同一 GT
            for gt_tid, prev_pid in prev_id_map.items():
                if prev_pid in curr_matched_pred_ids:
                    if curr_matched_pred_ids[prev_pid] != gt_tid:
                        total_frag += 1
                        break  # 一个 GT 只计一次 fragment

            # FP / FN
            fp_count = len(frame_preds) - len(matched_preds)
            fn_count = len(frame_gts) - len(matched_gts)
            total_fp += fp_count
            total_fn += fn_count
            id_fp += fp_count
            id_fn += fn_count

        # ── 计算指标 ──
        mota = max(0, (1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)) * 100)
        motp = (sum_iou / max(num_matches, 1)) * 100

        idp_val = id_tp / max(id_tp + id_fp, 1)
        idr_val = id_tp / max(id_tp + id_fn, 1)
        idf1_val = 2 * idp_val * idr_val / max(idp_val + idr_val, 1e-9)

        # MT / ML
        num_gt_tracks = len(gt_track_frames)
        mt_count = sum(
            1 for gt_tid, total in gt_track_frames.items()
            if gt_track_matched.get(gt_tid, 0) / max(total, 1) >= 0.8
        )
        ml_count = sum(
            1 for gt_tid, total in gt_track_frames.items()
            if gt_track_matched.get(gt_tid, 0) / max(total, 1) <= 0.2
        )
        mt = mt_count / max(num_gt_tracks, 1) * 100
        ml = ml_count / max(num_gt_tracks, 1) * 100

        return SeqMetrics(
            seq_name=seq_name,
            num_frames=num_frames,
            num_gt=total_gt,
            num_pred=id_tp + id_fp,
            mota=mota,
            motp=motp,
            idf1=idf1_val * 100,
            idp=idp_val * 100,
            idr=idr_val * 100,
            mt=mt,
            ml=ml,
            fp=total_fp,
            fn=total_fn,
            id_switches=total_idsw,
            fragments=total_frag,
            fps=0.0,
        )


# ============================================================
# 阶段 1: 检测
# ============================================================

def run_detection(
    seq_dir: str,
    model: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
) -> Tuple[Dict[int, np.ndarray], float]:
    """
    对单个序列的每一帧运行 YOLO 检测。

    Args:
        seq_dir: 序列目录（含 img1/）
        model: YOLO 模型
        conf: 置信度阈值
        iou: NMS IoU 阈值
        imgsz: 推理分辨率

    Returns:
        {frame_id: detections_array}  其中 detections_array shape [N, 6] = [x1,y1,x2,y2,conf,cls]
        avg_fps
    """
    img_dir = os.path.join(seq_dir, 'img1')
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    all_dets: Dict[int, np.ndarray] = {}
    total_time = 0.0

    for idx, img_name in enumerate(img_files):
        frame_num = idx + 1  # MOT 帧号从 1 开始
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        t0 = time.time()
        results = model.predict(
            frame, conf=conf, iou=iou, classes=[0],
            imgsz=imgsz, verbose=False,
        )
        dt = time.time() - t0
        total_time += dt

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # ultralytics boxes: xyxy + conf + cls
            dets = boxes.data.cpu().numpy()  # [N, 6] = [x1,y1,x2,y2,conf,cls]
        else:
            dets = np.empty((0, 6))

        all_dets[frame_num] = dets

    avg_fps = len(img_files) / total_time if total_time > 0 else 0
    return all_dets, avg_fps


# ============================================================
# 阶段 2: 追踪
# ============================================================

def _detections_to_tracker_input(all_dets: Dict[int, np.ndarray], frame_num: int) -> np.ndarray:
    """获取某帧的检测结果，转为 tracker.update() 要求的格式 [N, 6] = [x1,y1,x2,y2,conf,cls]"""
    return all_dets.get(frame_num, np.empty((0, 6)))


def run_ultralytics_tracking(
    seq_dir: str,
    all_dets: Dict[int, np.ndarray],
    tracker_yaml: str,
    tracker_type: str = 'botsort',
) -> Tuple[List[TrackResult], float]:
    """
    用 ultralytics 追踪器（BoT-SORT / ByteTrack）对已有检测结果进行追踪。

    Args:
        seq_dir: 序列目录
        all_dets: {frame_id: detections_array}
        tracker_yaml: 追踪器配置文件路径
        tracker_type: 'botsort' 或 'bytetrack'

    Returns:
        (追踪结果列表, FPS)
    """
    import torch
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    img_dir = os.path.join(seq_dir, 'img1')
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    cfg = IterableSimpleNamespace(**YAML.load(tracker_yaml))
    if tracker_type == 'botsort':
        tracker = BOTSORT(args=cfg, frame_rate=30)
        if hasattr(tracker, 'encoder') and tracker.encoder is not None:
            if hasattr(tracker.encoder, 'model'):
                tracker.encoder.model.to(device)
    else:
        tracker = BYTETracker(args=cfg, frame_rate=30)

    all_results: List[TrackResult] = []
    total_time = 0.0

    for idx, img_name in enumerate(img_files):
        frame_num = idx + 1
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        dets = _detections_to_tracker_input(all_dets, frame_num)

        t0 = time.time()
        if len(dets) > 0:
            dets_obj = Boxes(dets, frame.shape[:2])
            tracks = tracker.update(dets_obj, frame)
        else:
            dets_obj = Boxes(np.empty((0, 6)), frame.shape[:2])
            tracks = tracker.update(dets_obj, frame)
        dt = time.time() - t0
        total_time += dt

        if tracks is not None and len(tracks) > 0:
            for trk in tracks:
                x1, y1, x2, y2 = trk[:4]
                tid = int(trk[4])
                conf_val = float(trk[5]) if len(trk) > 5 else 1.0
                all_results.append(TrackResult(
                    frame=frame_num, track_id=tid,
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    conf=conf_val,
                ))

    avg_fps = len(img_files) / total_time if total_time > 0 else 0
    return all_results, avg_fps


def run_strongsort_tracking(
    seq_dir: str,
    all_dets: Dict[int, np.ndarray],
    reid_weights: str,
    dense_mode: bool = False,
) -> Tuple[List[TrackResult], float]:
    """
    用 StrongSORT 对已有检测结果进行追踪。

    Args:
        seq_dir: 序列目录
        all_dets: {frame_id: detections_array}
        reid_weights: ReID 模型权重路径
        dense_mode: 密集场景参数预设

    Returns:
        (追踪结果列表, FPS)
    """
    from boxmot.trackers.strongsort.strongsort import StrongSort
    import torch

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    img_dir = os.path.join(seq_dir, 'img1')
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if dense_mode:
        tracker = StrongSort(
            reid_weights=reid_weights, device=device, half=True,
            max_cos_dist=0.4, max_iou_dist=0.9, max_age=50, n_init=2,
        )
    else:
        tracker = StrongSort(
            reid_weights=reid_weights, device=device, half=True,
        )

    all_results: List[TrackResult] = []
    total_time = 0.0

    for idx, img_name in enumerate(img_files):
        frame_num = idx + 1
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        dets = _detections_to_tracker_input(all_dets, frame_num)

        t0 = time.time()
        tracks = tracker.update(dets, frame)
        dt = time.time() - t0
        total_time += dt

        if tracks is not None and len(tracks) > 0:
            for trk in tracks:
                x1, y1, x2, y2 = trk[:4]
                tid = int(trk[4])
                conf_val = float(trk[5]) if len(trk) > 5 else 1.0
                all_results.append(TrackResult(
                    frame=frame_num, track_id=tid,
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    conf=conf_val,
                ))

    avg_fps = len(img_files) / total_time if total_time > 0 else 0
    return all_results, avg_fps


# ============================================================
# 流程编排
# ============================================================

@dataclass
class ExperimentResult:
    """一组实验（model + tracker + all sequences）的结果"""
    model_name: str
    tracker_name: str
    seq_metrics: List[SeqMetrics] = field(default_factory=list)

    @property
    def overall(self) -> dict:
        """汇总指标（按 GT 数量加权）"""
        total_gt = sum(s.num_gt for s in self.seq_metrics)
        if total_gt == 0:
            return {
                'mota': 0.0, 'idf1': 0.0, 'motp': 0.0,
                'mt': 0.0, 'ml': 0.0, 'idsw': 0,
                'fp': 0, 'fn': 0, 'fps': 0.0,
            }
        return {
            'mota': sum(s.mota * s.num_gt for s in self.seq_metrics) / total_gt,
            'idf1': sum(s.idf1 * s.num_gt for s in self.seq_metrics) / total_gt,
            'motp': sum(s.motp * s.num_gt for s in self.seq_metrics) / total_gt,
            'mt': sum(s.mt * s.num_gt for s in self.seq_metrics) / total_gt,
            'ml': sum(s.ml * s.num_gt for s in self.seq_metrics) / total_gt,
            'idsw': sum(s.id_switches for s in self.seq_metrics),
            'fp': sum(s.fp for s in self.seq_metrics),
            'fn': sum(s.fn for s in self.seq_metrics),
            'fps': sum(s.fps for s in self.seq_metrics) / max(len(self.seq_metrics), 1),
        }


def discover_sequences(mot_root: str) -> List[str]:
    """自动发现 MOT 根目录下的序列。"""
    seqs = []
    for d in sorted(os.listdir(mot_root)):
        full = os.path.join(mot_root, d)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, 'img1')):
            if os.path.isfile(os.path.join(full, 'gt', 'gt.txt')):
                seqs.append(full)
    return seqs


def print_results(experiments: List[ExperimentResult]):
    """打印汇总对比表。"""
    if not experiments:
        print("无实验结果。")
        return

    # 列宽
    print()
    print("=" * 72)
    print("  MOT20 追踪性能对比试验结果")
    print("=" * 72)
    header = (
        f"{'检测器':<20} {'追踪器':<12} {'MOTA↑':>8} {'IDF1↑':>8} {'IDs↓':>7} "
        f"{'MOTP↑':>8} {'MT↑':>7} {'ML↓':>7} {'FPS↑':>7}"
    )
    sep = "─" * 72
    print(header)
    print(sep)

    for exp in experiments:
        o = exp.overall
        print(
            f"{exp.model_name:<20} {exp.tracker_name:<12} "
            f"{o['mota']:>7.1f}% {o['idf1']:>7.1f}% {o['idsw']:>7d} "
            f"{o['motp']:>7.1f}% {o['mt']:>6.1f}% {o['ml']:>6.1f}% {o['fps']:>6.1f}"
        )

    print(sep)
    print("  MOTA↑/IDF1↑/MOTP↑/MT↑/FPS↑ 越高越好; ML↓/IDs↓ 越低越好")
    print()

    # 按序列详细
    all_seqs = sorted(set(
        s.seq_name for exp in experiments for s in exp.seq_metrics
    ))
    for seq_name in all_seqs:
        print(f"  📹 {seq_name}")
        print(f"  {'─' * 60}")
        print(f"  {'检测器':<20} {'追踪器':<12} {'MOTA↑':>8} {'IDF1↑':>8} {'IDs↓':>7} {'FP↓':>7} {'FN↓':>7}")
        for exp in experiments:
            sm = next((s for s in exp.seq_metrics if s.seq_name == seq_name), None)
            if sm:
                print(
                    f"  {exp.model_name:<20} {exp.tracker_name:<12} "
                    f"{sm.mota:>7.1f}% {sm.idf1:>7.1f}% {sm.id_switches:>7d} "
                    f"{sm.fp:>7d} {sm.fn:>7d}"
                )
        print()


def save_csv_results(experiments: List[ExperimentResult], output_path: str):
    """保存结果为 CSV 文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        # 汇总
        w.writerow(['检测器', '追踪器', 'MOTA(%)', 'IDF1(%)', 'MOTP(%)',
                     'MT(%)', 'ML(%)', 'IDs', 'FP', 'FN', 'FPS'])
        for exp in experiments:
            o = exp.overall
            w.writerow([
                exp.model_name, exp.tracker_name,
                f"{o['mota']:.1f}", f"{o['idf1']:.1f}", f"{o['motp']:.1f}",
                f"{o['mt']:.1f}", f"{o['ml']:.1f}",
                o['idsw'], o['fp'], o['fn'], f"{o['fps']:.1f}",
            ])
        # 按序列
        w.writerow([])
        w.writerow(['检测器', '追踪器', '序列', 'MOTA(%)', 'IDF1(%)', 'MOTP(%)',
                     'MT(%)', 'ML(%)', 'IDs', 'FP', 'FN', 'FPS'])
        for exp in experiments:
            for sm in exp.seq_metrics:
                w.writerow([
                    exp.model_name, exp.tracker_name, sm.seq_name,
                    f"{sm.mota:.1f}", f"{sm.idf1:.1f}", f"{sm.motp:.1f}",
                    f"{sm.mt:.1f}", f"{sm.ml:.1f}",
                    sm.id_switches, sm.fp, sm.fn, f"{sm.fps:.1f}",
                ])
    print(f"📄 CSV 已保存: {output_path}")


# ============================================================
# 配置加载
# ============================================================

def _load_config_module(config_path: str):
    """动态加载 Python 配置文件，返回其命名空间 dict。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("benchmark_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in module.__dict__.items() if not k.startswith('_') and not k.startswith('"""')}


def _resolve_path(base_dir: str, p: str) -> str:
    """将相对路径转为绝对路径。"""
    p = os.path.normpath(p)
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    return os.path.normpath(p)


# ============================================================
# 主入口
# ============================================================

def main():
    # ── 加载默认配置文件 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'benchmark_config.py')

    parser = argparse.ArgumentParser(
        description='MOT20 行人追踪对比试验 — 两阶段分离式评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式:
  # 方式1（推荐）: 编辑 benchmark_config.py 后直接运行
  python run_tracking_benchmark.py

  # 方式2: 指定其他配置文件
  python run_tracking_benchmark.py --config my_config.py

  # 方式3: 纯命令行（覆盖配置文件）
  python run_tracking_benchmark.py --mot_root D:/Datasets/MOT20/train \\
      --models ../runs/Sampled_100/exp0_baseline/weights/best.pt \\
      --trackers botsort strongsort
"""
    )

    # 配置文件
    parser.add_argument('--config', type=str, default=default_config,
                        help=f'配置文件路径（默认 {default_config}）')

    # 数据集
    parser.add_argument('--mot_root', type=str, default='D:/Datasets/MOT20/train',
                        help='MOT20/train 目录路径')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='指定序列（默认自动发现全部）')

    # 模型（命令行模型用 --model-paths，格式: path1 path2 ...）
    parser.add_argument('--model-paths', type=str, nargs='+', default=None,
                        help='直接指定模型路径（命令行模式），覆盖配置文件中的 MODELS')
    parser.add_argument('--models', type=str, nargs='+', default=['D:/Downloads/临时下载/all_first_end/all_first_end/exp0_baseline/weights/exp0.pt',
        'D:/Downloads/临时下载/all_first_end/all_first_end/exp4_full/weights/exp4.pt'],
                        help=argparse.SUPPRESS)  # 兼容旧版，同 --model-paths

    # 追踪器
    parser.add_argument('--trackers', type=str, nargs='+', default=['bytetrack'],
                        choices=['botsort', 'strongsort', 'bytetrack'],
                        help='追踪器类型')

    # 检测参数
    parser.add_argument('--conf', type=float, default=None,
                        help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=None,
                        help='NMS IoU 阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理分辨率')

    # StrongSORT 专用
    parser.add_argument('--reid-weights', type=str, default=None,
                        help='StrongSORT ReID 权重路径')
    parser.add_argument('--dense-mode', action='store_true', default=None,
                        help='StrongSORT 密集场景参数')

    # 输出
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果输出目录')
    parser.add_argument('--save-tracks', action='store_true', default=None,
                        help='保存每帧追踪结果为 MOT 格式')

    # 评估
    parser.add_argument('--iou-threshold', type=float, default=None,
                        help='评估 IoU 匹配阈值')

    args = parser.parse_args()

    # ── 加载配置文件 ──
    if os.path.exists(args.config):
        cfg = _load_config_module(args.config)
        print(f"📋 加载配置: {args.config}")
    else:
        cfg = {}

    # ── 合并参数：命令行 > 配置文件 > 硬编码默认值 ──
    mot_root       = args.mot_root       or cfg.get('MOT_ROOT')
    sequences      = args.sequences      or cfg.get('SEQUENCES')
    trackers       = args.trackers       or cfg.get('TRACKERS', ['botsort', 'bytetrack'])
    conf           = args.conf           if args.conf is not None           else cfg.get('CONF', 0.25)
    iou            = args.iou            if args.iou is not None            else cfg.get('IOU', 0.45)
    imgsz          = args.imgsz          if args.imgsz is not None          else cfg.get('IMGSZ', 640)
    reid_weights   = args.reid_weights   or cfg.get('REID_WEIGHTS', 'osnet_x0_25_msmt17.pt')
    dense_mode     = args.dense_mode     if args.dense_mode is not None     else cfg.get('DENSE_MODE', False)
    output_dir     = args.output_dir     or cfg.get('OUTPUT_DIR', './benchmark_results')
    save_tracks    = args.save_tracks    if args.save_tracks is not None    else cfg.get('SAVE_TRACKS', False)
    iou_threshold  = args.iou_threshold  if args.iou_threshold is not None  else cfg.get('IOU_THRESHOLD', 0.5)

    # 模型列表：命令行 model-paths 优先，其次 --models（兼容），最后从配置文件
    model_paths_raw = args.model_paths or args.models
    if model_paths_raw:
        # 命令行模式：路径列表，自动生成名称
        models = [{"name": os.path.basename(p).replace('.pt', ''), "path": p}
                  for p in model_paths_raw]
    else:
        models = cfg.get('MODELS', [])

    # ── 校验 ──
    if not mot_root:
        print("❌ 未指定数据集路径。请在 benchmark_config.py 中设置 MOT_ROOT，或用 --mot_root 指定。")
        return
    if not models:
        print("❌ 未指定检测模型。请在 benchmark_config.py 中设置 MODELS，或用 --model-paths 指定。")
        return
    if not trackers:
        print("❌ 未指定追踪器。请在 benchmark_config.py 中设置 TRACKERS，或用 --trackers 指定。")
        return

    # ── 打印本次运行配置 ──
    print(f"📂 数据集:  {mot_root}")
    print(f"📹 序列:    {sequences if sequences else '自动发现全部'}")
    print(f"🔍 模型:    {', '.join(m['name'] for m in models)}")
    print(f"🏃 追踪器:  {', '.join(trackers)}")
    print(f"⚙️  检测参数: conf={conf}, iou={iou}, imgsz={imgsz}")
    print(f"📏 评估 IoU: {iou_threshold}")
    print()

    # ── 1. 发现序列 ──
    if sequences:
        seq_dirs = [
            os.path.join(mot_root, s) for s in sequences
            if os.path.isdir(os.path.join(mot_root, s))
        ]
    else:
        seq_dirs = discover_sequences(mot_root)

    if not seq_dirs:
        print(f"❌ 未在 {mot_root} 下发现有效序列（需要 img1/ 和 gt/gt.txt）")
        return

    seq_names = [os.path.basename(d) for d in seq_dirs]
    print(f"📂 发现 {len(seq_dirs)} 个序列: {', '.join(seq_names)}")

    # ── 2. 确定 tracker 配置 ──
    tracker_yamls = {}
    for t in ['botsort', 'bytetrack']:
        yaml_path = os.path.normpath(os.path.join(
            script_dir, '..', 'ultralytics', 'ultralytics', 'ultralytics',
            'cfg', 'trackers', f'{t}.yaml'
        ))
        if not os.path.exists(yaml_path):
            # 回退到 ultralytics 包内置配置
            from ultralytics.utils.checks import check_yaml
            yaml_path = check_yaml(f'{t}.yaml')
        tracker_yamls[t] = yaml_path

    # ── 3. 逐模型 → 检测一次，再喂给所有追踪器 ──
    experiments: List[ExperimentResult] = []

    for model_info in models:
        model_path = _resolve_path(script_dir, model_info['path'])
        model_name = model_info.get('name', os.path.basename(model_path))

        print(f"\n{'=' * 60}")
        print(f"📦 加载检测模型: {model_name}")
        print(f"   {model_path}")
        print(f"{'=' * 60}")

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            continue

        model = YOLO(model_path)

        # 每个序列检测一次（自动缓存，避免重复跑）
        seq_detections: Dict[str, Dict[int, np.ndarray]] = {}
        cache_dir = os.path.join(output_dir, '.det_cache', model_name)
        os.makedirs(cache_dir, exist_ok=True)

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            cache_file = os.path.join(cache_dir, f'{seq_name}.npz')

            if os.path.exists(cache_file):
                print(f"\n  🔍 检测 {seq_name} ... (从缓存加载)", end='', flush=True)
                data = np.load(cache_file, allow_pickle=True)
                dets = {int(k): data[k] for k in data.files}
                det_fps = 0  # 缓存不记 FPS
            else:
                print(f"\n  🔍 检测 {seq_name} ... ", end='', flush=True)
                dets, det_fps = run_detection(seq_dir, model, conf, iou, imgsz)
                # 保存缓存
                np.savez_compressed(cache_file, **{str(k): v for k, v in dets.items()})

            seq_detections[seq_name] = dets
            total_dets = sum(len(d) for d in dets.values())
            print(f"{total_dets} 个检测框, {det_fps:.1f} FPS")

        # 每个追踪器用同一份检测结果
        for tracker_type in trackers:
            print(f"\n  🏃 追踪 [{tracker_type}] ...")
            exp = ExperimentResult(model_name=model_name, tracker_name=tracker_type)

            for seq_dir in seq_dirs:
                seq_name = os.path.basename(seq_dir)
                gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')

                if not os.path.isfile(gt_file):
                    print(f"    ⚠️  {seq_name}: 无 GT，跳过")
                    continue

                dets = seq_detections[seq_name]

                # 运行追踪
                if tracker_type in ('botsort', 'bytetrack'):
                    yaml_path = tracker_yamls[tracker_type]
                    tracks, trk_fps = run_ultralytics_tracking(seq_dir, dets, yaml_path, tracker_type)
                elif tracker_type == 'strongsort':
                    try:
                        tracks, trk_fps = run_strongsort_tracking(
                            seq_dir, dets, reid_weights, dense_mode
                        )
                    except ImportError as e:
                        print(f"    ❌ StrongSORT 不可用: {e}")
                        print(f"    请先安装: pip install boxmot")
                        continue
                else:
                    continue

                print(f"    {seq_name}: {len(tracks)} 条轨迹, {trk_fps:.1f} FPS", end='')

                # 评估
                gt = read_gt(gt_file)
                evaluator = MOTEvaluator(iou_threshold=iou_threshold)
                metrics = evaluator.evaluate_sequence(tracks, gt, seq_name)
                metrics.fps = trk_fps
                exp.seq_metrics.append(metrics)

                print(f" → MOTA={metrics.mota:.1f}% IDF1={metrics.idf1:.1f}% IDs={metrics.id_switches}")

                # 可选保存追踪结果
                if save_tracks:
                    track_out = os.path.join(
                        output_dir, model_name, tracker_type, f'{seq_name}.txt'
                    )
                    write_mot_format(tracks, track_out)

            experiments.append(exp)

    # ── 4. 输出 ──
    print_results(experiments)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'benchmark_summary.csv')
    save_csv_results(experiments, csv_path)


if __name__ == '__main__':
    main()
