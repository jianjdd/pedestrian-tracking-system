"""
BoT-SORT / ByteTrack 追踪器适配模块
YOLO11 提供了内置的强大的追踪算法。该模块用于将 YOLO11 的追踪结果适配到系统原有的 Track 类结构中
这样无需修改下层绘制、计数的逻辑。
"""
import numpy as np
from collections import deque
from .tracker import Track

class YOLOTrackerAdapter:
    """
    YOLO 原生追踪器结果适配器
    用于接收 YOLO `model.track()` 返回的 Results 对象，并将其转为现有的 Track 对象列表。
    """
    def __init__(self, max_age=5, min_hits=4):
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks_dict = {}  # {track_id: Track对象}
        self.frame_count = 0

    def update_from_yolo_results(self, result):
        """
        接收原生的 YOLO Results，从中提取出 track_id 并组装成 Track。
        Args:
            result: YOLO 推理出的 result 对象 (通常为 results[0])
        Returns:
            list of Track objects
        """
        self.frame_count += 1
        current_ids = set()
        
        # 1. 解析 YOLO 检测结果
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().tolist()
            
            # 获取名字映射词典（如果存在）
            names = getattr(result, 'names', {})
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, class_ids):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bbox = [x1, y1, x2, y2]
                conf = float(conf)
                class_name = names.get(cls_id, 'unknown')
                
                current_ids.add(track_id)
                
                # 2. 更新或新建我们自己的 Track 对象以保留历史轨迹 history
                if track_id in self.tracks_dict:
                    track = self.tracks_dict[track_id]
                    track.update(bbox, conf, cls_id, class_name)
                    # 重置不可见计数，确保活跃
                    track.consecutive_invisible_count = 0
                else:
                    new_track = Track(track_id, bbox, conf, cls_id, class_name)
                    # 由于是 YOLO 内部帮我们过滤和确认的目标，直接放进去
                    self.tracks_dict[track_id] = new_track
        
        # 3. 处理当前帧中丢失但在本地历史字典中的目标
        all_known_ids = set(self.tracks_dict.keys())
        missed_ids = all_known_ids - current_ids
        for track_id in missed_ids:
            self.tracks_dict[track_id].mark_missed()
            
        # 4. 清理连续丢失达到 max_age 阈值的幽灵框拖尾
        expired_ids = []
        for track_id, track in self.tracks_dict.items():
            if track.consecutive_invisible_count >= self.max_age:
                expired_ids.append(track_id)
                
        for track_id in expired_ids:
            del self.tracks_dict[track_id]
            
        # 返回仍存活的追踪列表供外界画线计数
        return list(self.tracks_dict.values())

    def update(self, detections, frame=None):
        """
        这个方法保留仅仅为了兼容。
        如果 detector 没有正确使用 update_from_yolo_results 而是传入了切片好的 detections，则无法复原ID。
        正常应抛出异常或返回空，提示切换逻辑。
        """
        print("警告: YOLOTrackerAdapter 应直接接收 YOLO 的 Result 对象，不应使用基础的 update。")
        return []

    def reset(self):
        """重置追踪器的缓冲字典"""
        self.tracks_dict.clear()
        self.frame_count = 0
