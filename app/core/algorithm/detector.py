"""
Video processing and tracking module (Web).
Uses YOLOv8 + BoT-SORT for person detection and multi-object tracking.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch

from .counter import LineCrossingCounter
from .tracker_botsort import YOLOTrackerAdapter
from app.core.config import settings

# Prefer bundled ultralytics source at repo_root/ultralytics/ultralytics
_repo_root = str(Path(settings.BASE_DIR) / "ultralytics")
if os.path.isdir(os.path.join(_repo_root, "ultralytics")):
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    if "ultralytics" in sys.modules:
        del sys.modules["ultralytics"]

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
    ULTRALYTICS_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    HAS_ULTRALYTICS = False
    ULTRALYTICS_IMPORT_ERROR = e
    print(f"Warning: failed to import ultralytics: {e}")


class VideoDetector:
    """Video detector with start/pause/resume/stop lifecycle."""

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()

        self.model = None
        self.model_type = None  # 'yolov8'
        self.video_source = None

        self.tracker = YOLOTrackerAdapter()
        self.tracker_yaml = self._resolve_tracker_yaml("botsort")
        self.counter = LineCrossingCounter()

        self.is_running = False
        self.is_paused = False

        self.confidence_threshold = 0.6
        self.nms_iou = 0.75
        self.max_det = 300

        self.infer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.runtime_device = self.infer_device
        self.infer_imgsz = 640 if self.infer_device.startswith("cuda") else 512

        # Runtime acceleration flags (quality-preserving)
        self.fp16_enabled = self.infer_device.startswith("cuda")
        self.tensor_rt_enabled = False
        self.model_backend = "none"  # none | pytorch | tensorrt
        self.loaded_model_path = None
        self.loaded_runtime_path = None

        self.display_invisible_threshold = 1

        self.callbacks = {
            "on_frame": None,
            "on_stats": None,
            "on_error": None,
            "on_finished": None,
        }

        self.class_names_map = {}
        self.enabled_classes = set()

        self.cap = None

        self.show_bbox = True
        self.show_label = True
        self.show_center = True
        self.show_trajectory = True

        self.stats_font_scale = 0.8
        self.line_thickness = 3
        self.bbox_thickness = 2
        self.label_font_scale = 0.6
        self.center_size = 4

        self.stats_font_color = (0, 255, 0)
        self.label_font_color = (255, 0, 0)
        self.bbox_color = (255, 0, 0)
        self.center_color = (0, 255, 0)
        self.trajectory_color = (0, 255, 255)
        self.line_color = (0, 255, 0)

        self.avg_frame_window = 30
        self.frame_object_counts = []

        self.gpu_cleanup_interval = 50
        self.processed_frame_count = 0

        if self.infer_device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True

    def set_callback(self, name, func):
        if name in self.callbacks:
            self.callbacks[name] = func

    def _call_callback(self, name, *args):
        callback = self.callbacks.get(name)
        if callback:
            try:
                callback(*args)
            except Exception as e:
                print(f"Callback {name} error: {e}")

    def load_model(self, model_path):
        """Load model and initialize runtime backend/device state."""
        try:
            model_loaded = False
            load_error = None

            if HAS_ULTRALYTICS:
                try:
                    requested_device = self.infer_device
                    runtime_model_path, backend = self._select_runtime_model_path(model_path)
                    temp_model = YOLO(runtime_model_path)

                    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    temp_model(
                        test_img,
                        verbose=False,
                        device=requested_device,
                        half=self.fp16_enabled,
                    )

                    self.model = temp_model
                    self.model_type = "yolov8"
                    self.model_backend = backend
                    self.tensor_rt_enabled = backend == "tensorrt"
                    self.loaded_model_path = str(model_path)
                    self.loaded_runtime_path = str(runtime_model_path)
                    self.runtime_device = self._get_model_runtime_device(temp_model) or requested_device
                    if not str(self.runtime_device).startswith("cuda"):
                        self.fp16_enabled = False

                    if hasattr(self.model, "names") and self.model.names:
                        new_map = {}
                        for k, v in self.model.names.items():
                            try:
                                new_map[int(k)] = v
                            except ValueError:
                                pass
                        self.class_names_map = {k: v for k, v in new_map.items() if v == "person"}
                        self.enabled_classes = set(self.class_names_map.values())
                        model_loaded = True

                        if str(self.runtime_device) != requested_device:
                            self._call_callback(
                                "on_error",
                                f"inference device fallback: requested={requested_device}, actual={self.runtime_device}",
                            )
                        return True
                except Exception as e:
                    print(f"Ultralytics model load failed: {e}")
                    load_error = e
                    model_loaded = False

            if not model_loaded:
                if not HAS_ULTRALYTICS:
                    import_error = str(ULTRALYTICS_IMPORT_ERROR) if ULTRALYTICS_IMPORT_ERROR else "unknown"
                    reason = f"ultralytics unavailable ({import_error})"
                else:
                    reason = f"model init failed ({load_error or 'unknown'})"
                self._call_callback("on_error", f"model load failed: {reason}")
                return False

        except Exception as e:
            self._call_callback("on_error", f"model load failed: {str(e)}")
            return False

    def _select_runtime_model_path(self, model_path):
        """Prefer TensorRT engine on CUDA when sibling .engine exists."""
        requested = Path(model_path)
        if self.infer_device.startswith("cuda"):
            if requested.suffix.lower() == ".engine" and requested.exists():
                return str(requested), "tensorrt"
            engine_candidate = requested.with_suffix(".engine")
            if engine_candidate.exists():
                return str(engine_candidate), "tensorrt"
        return str(requested), "pytorch"

    def _get_model_runtime_device(self, model):
        """Best-effort runtime device detection from ultralytics internals."""
        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            d = getattr(predictor, "device", None)
            if d is not None:
                return str(d)

        inner_model = getattr(model, "model", None)
        if inner_model is not None:
            d = getattr(inner_model, "device", None)
            if d is not None:
                return str(d)
        return None

    def set_video_source(self, source):
        self.video_source = source

    def set_enabled_classes(self, class_names):
        self.enabled_classes = set(class_names)

    def set_counting_line(self, point1, point2, reset_counts=False):
        self.counter.set_line(point1, point2, reset_counts=reset_counts)

    def clear_counting_line(self):
        self.counter.clear_line()

    def reset_counter(self):
        self.counter.reset_counts()
        self.tracker.reset()
        self._reset_yolo_internal_trackers()

    def _resolve_tracker_yaml(self, tracker_name="botsort"):
        local_yaml = (
            Path(settings.BASE_DIR)
            / "ultralytics"
            / "ultralytics"
            / "cfg"
            / "trackers"
            / f"{tracker_name}.yaml"
        )
        if local_yaml.exists():
            return str(local_yaml)
        return f"{tracker_name}.yaml"

    def _reset_yolo_internal_trackers(self):
        try:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    if hasattr(tracker, "reset"):
                        tracker.reset()
        except Exception:
            pass

    def start_detection(self):
        if not self.is_running:
            self.tracker.reset()
            self._reset_yolo_internal_trackers()
            self.is_running = True
            self.is_paused = False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop_detection(self):
        self.is_running = False
        self.is_paused = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    def pause_detection(self):
        self.is_paused = True

    def resume_detection(self):
        self.is_paused = False

    def _run_loop(self):
        if self.model is None:
            self._call_callback("on_error", "Please load model first")
            return
        if self.video_source is None:
            self._call_callback("on_error", "Please select video source first")
            return

        if isinstance(self.video_source, int):
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.video_source)

        is_file = isinstance(self.video_source, str)

        if not cap.isOpened():
            self._call_callback("on_error", "Unable to open video source")
            return

        self.cap = cap
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 120:
            fps = 30

        self.counter.video_fps = float(fps)
        self.counter.use_video_time = is_file

        frame_interval = 1.0 / fps
        next_frame_time = time.monotonic()
        consecutive_failures = 0

        try:
            while self.is_running and not self._stop_event.is_set():
                if self.is_paused:
                    next_frame_time = time.monotonic()
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                wait = next_frame_time - now
                if wait > 0:
                    time.sleep(wait)
                next_frame_time = max(time.monotonic(), next_frame_time) + frame_interval

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        if not is_file:
                            self._call_callback("on_error", "Video source connection lost")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                processed_frame = self.process_frame(frame)
                self._call_callback("on_frame", processed_frame)
                self._call_callback("on_stats", self.counter.get_statistics())

        except Exception as e:
            self._call_callback("on_error", f"Detection loop error: {str(e)}")
        finally:
            self.is_running = False
            cap.release()
            self.cap = None
            self._call_callback("on_finished")

    def detect_and_track(self, frame):
        enabled_class_ids = [
            cid
            for cid, cname in self.class_names_map.items()
            if cname in self.enabled_classes
        ]

        if self.model_type == "yolov8":
            if not self.enabled_classes:
                return []

            with torch.no_grad():
                results = self.model.track(
                    frame,
                    device=self.infer_device,
                    imgsz=self.infer_imgsz,
                    half=self.fp16_enabled,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou,
                    max_det=self.max_det,
                    classes=enabled_class_ids if enabled_class_ids else None,
                    tracker=self.tracker_yaml,
                    persist=True,
                    verbose=False,
                )

            if results and len(results) > 0:
                tracks = self.tracker.update_from_yolo_results(results[0])
            else:
                tracks = self.tracker.update_from_yolo_results(SimpleNamespace(boxes=None, names={}))
        else:
            tracks = []

        self.processed_frame_count += 1
        if self.processed_frame_count % self.gpu_cleanup_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return tracks

    def process_frame(self, frame):
        try:
            tracks = self.detect_and_track(frame)
            self.counter.update(tracks)
            return self.draw_results(frame, tracks)
        except Exception as e:
            print(f"process_frame error: {e}")
            return frame

    def _get_track_display_color(self, track):
        invisible = int(getattr(track, "consecutive_invisible_count", 0) or 0)
        visible_hits = int(getattr(track, "total_visible_count", 0) or 0)
        min_hits = int(getattr(self.tracker, "min_hits", 3) or 3)
        if invisible > 0:
            return (0, 165, 255)
        if visible_hits <= max(2, min_hits):
            return (0, 255, 255)
        return (0, 200, 0)

    def _draw_crossing_effect(self, frame, x1, y1, x2, y2, direction, progress):
        h, w = frame.shape[:2]
        pulse = 1.0 - float(progress)
        effect_color = (255, 0, 255) if direction == "a_to_b" else (255, 255, 0)

        pad = max(4, int(10 * pulse))
        thick = max(self.bbox_thickness + 1, int(self.bbox_thickness + 3 * pulse))
        alpha = min(0.75, max(0.25, 0.25 + 0.45 * pulse))

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w - 1, x2 + pad)
        y2p = min(h - 1, y2 + pad)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1p, y1p), (x2p, y2p), effect_color, thick)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        fx_text = "A->B" if direction == "a_to_b" else "B->A"
        text_pos = (x1p, max(15, y1p - 8))
        cv2.putText(frame, fx_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, effect_color, 2, cv2.LINE_AA)

    def draw_results(self, frame, tracks):
        frame = self.counter.draw_line(frame, line_thickness=self.line_thickness, line_color=self.line_color)

        for track in tracks:
            if track.consecutive_invisible_count > self.display_invisible_threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            center = track.get_center()
            cx, cy = int(center[0]), int(center[1])
            track_color = self._get_track_display_color(track)
            is_crossing_fx, crossing_dir, crossing_progress = self.counter.get_crossing_effect(track.track_id)
            if is_crossing_fx:
                track_color = (255, 0, 255) if crossing_dir == "a_to_b" else (255, 255, 0)
                self._draw_crossing_effect(frame, x1, y1, x2, y2, crossing_dir, crossing_progress)

            if self.show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, self.bbox_thickness)

            if self.show_label:
                label = f"P#{track.track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), track_color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.label_font_scale,
                    self.label_font_color,
                    1,
                    cv2.LINE_AA,
                )

            if self.show_center:
                cv2.circle(frame, (cx, cy), self.center_size, self.center_color, -1)

            if self.show_trajectory and len(track.history) > 1:
                points = list(track.history)
                for i in range(1, len(points)):
                    p1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                    p2 = (int(points[i][0]), int(points[i][1]))
                    alpha = i / len(points)
                    thickness = max(1, int(self.bbox_thickness * alpha))
                    cv2.line(frame, p1, p2, self.trajectory_color, thickness)

        stats = self.counter.get_statistics()
        self.frame_object_counts.append(len(tracks))
        if len(self.frame_object_counts) > self.avg_frame_window:
            self.frame_object_counts.pop(0)
        avg_objects = int(sum(self.frame_object_counts) / len(self.frame_object_counts)) if self.frame_object_counts else 0

        info_lines = [
            f"A->B: {stats.get('count_a_to_b', 0)}",
            f"B->A: {stats.get('count_b_to_a', 0)}",
            f"Total: {stats.get('total', 0)}",
            f"Current: {avg_objects}",
        ]
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.stats_font_scale,
                self.stats_font_color,
                2,
                cv2.LINE_AA,
            )
            y_offset += int(30 * self.stats_font_scale + 10)

        return frame


def get_available_cameras(max_cameras=10):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_frame_from_source(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        is_cam = True
    else:
        cap = cv2.VideoCapture(source)
        is_cam = False
    try:
        if not cap.isOpened():
            return None
        if is_cam:
            for _ in range(20):
                cap.read()
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
        return None
    finally:
        cap.release()
