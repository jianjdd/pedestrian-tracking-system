"""
StrongSORT tracker adapter.

This module wraps boxmot.StrongSORT and adapts outputs to the local Track type
used by the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.core.algorithm.tracker import Track

try:
    from boxmot import StrongSORT

    HAS_BOXMOT = True
except ImportError:
    HAS_BOXMOT = False
    print("Warning: boxmot is not installed, StrongSORT is unavailable.")


class StrongSORTTrackerAdapter:
    """StrongSORT adapter that exposes an interface compatible with DeepSORTTracker."""

    def __init__(
        self,
        reid_weights: str = "osnet_x0_25_msmt17.pt",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        max_age: int = 50,
        min_hits: int = 4,
        iou_threshold: float = 0.3,
        max_dist: float = 0.4,
        max_iou_dist: float = 0.9,
        n_init: int = 2,
    ):
        self.tracks_dict: dict[int, Track] = {}
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracker = None
        if not HAS_BOXMOT:
            return

        weights_path = self._resolve_weights_path(reid_weights)
        if not weights_path:
            print(f"Warning: StrongSORT ReID weights not found: {reid_weights}")
            return

        try:
            self.tracker = StrongSORT(
                model_weights=weights_path,
                device=device,
                fp16=True,
                max_dist=max_dist,
                max_iou_dist=max_iou_dist,
                max_age=max_age,
                n_init=n_init,
            )
            print(
                "StrongSORT initialized "
                f"(device={device}, max_dist={max_dist}, max_iou_dist={max_iou_dist}, "
                f"max_age={max_age}, n_init={n_init})"
            )
        except Exception as exc:
            print(f"StrongSORT initialization failed: {exc}")
            self.tracker = None

    def _resolve_weights_path(self, reid_weights: str) -> Path | None:
        candidates = [
            Path(reid_weights),
            Path("weights") / reid_weights,
            Path(settings.BASE_DIR) / "weights" / reid_weights,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def update(self, detections, frame=None):
        """Update tracking state and return local Track objects."""
        if self.tracker is None or frame is None:
            return []

        self.frame_count += 1

        if len(detections) > 0:
            dets_list = []
            class_map = {}
            for det in detections:
                x1, y1, x2, y2, conf, cls_id, cls_name = det
                dets_list.append([x1, y1, x2, y2, conf, cls_id])
                class_map[int(cls_id)] = cls_name
            dets_np = np.array(dets_list)
        else:
            dets_np = np.empty((0, 6))
            class_map = {}

        try:
            results = self.tracker.update(dets_np, frame)
        except Exception as exc:
            print(f"StrongSORT update error: {exc}")
            return []

        current_ids = set()
        if len(results) > 0:
            for res in results:
                x1, y1, x2, y2 = res[:4]
                track_id = int(res[4])
                conf = float(res[5])
                cls_id = int(res[6])
                class_name = class_map.get(cls_id, "unknown")

                bbox = [x1, y1, x2, y2]
                current_ids.add(track_id)

                if track_id in self.tracks_dict:
                    self.tracks_dict[track_id].update(bbox, conf, cls_id, class_name)
                else:
                    self.tracks_dict[track_id] = Track(track_id, bbox, conf, cls_id, class_name)

        all_known_ids = set(self.tracks_dict.keys())
        missed_ids = all_known_ids - current_ids
        for track_id in missed_ids:
            self.tracks_dict[track_id].mark_missed()

        expired_ids = [
            track_id
            for track_id, track in self.tracks_dict.items()
            if track.consecutive_invisible_count >= self.max_age
        ]
        for track_id in expired_ids:
            del self.tracks_dict[track_id]

        return list(self.tracks_dict.values())

    def reset(self):
        """Reset local tracking cache and underlying tracker state when available."""
        self.tracks_dict = {}
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            self.tracker.reset()
