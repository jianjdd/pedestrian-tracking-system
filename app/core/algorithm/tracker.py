"""
DeepSORT追踪器模块
用于追踪检测到的人员目标
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


class Track:
    """单个追踪目标类"""
    def __init__(self, track_id, bbox, confidence, class_id=0, class_name='unknown'):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id  # 类别ID
        self.class_name = class_name  # 类别名称
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.history = deque(maxlen=30)  # 保存历史中心点
        self.history.append(self.get_center())
        
    def get_center(self):
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def update(self, bbox, confidence, class_id=None, class_name=None):
        """更新追踪目标"""
        self.bbox = bbox
        self.confidence = confidence
        if class_id is not None:
            self.class_id = class_id
        if class_name is not None:
            self.class_name = class_name
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        self.history.append(self.get_center())
    
    def mark_missed(self):
        """标记为未匹配"""
        self.age += 1
        self.consecutive_invisible_count += 1


class DeepSORTTracker:
    """简化版DeepSORT追踪器"""
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, max_id=500):
        self.max_age = max_age  # 最大丢失帧数（降低到5帧，约0.17秒）
        self.min_hits = min_hits  # 最小命中次数
        self.iou_threshold = iou_threshold  # IOU阈值
        self.max_id = max_id  # ID上限，达到后重置
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """
        更新追踪器
        Args:
            detections: list of [x1, y1, x2, y2, confidence]
        Returns:
            list of Track objects
        """
        # 预测所有追踪目标的位置（简化版：保持原位置）
        for track in self.tracks:
            track.age += 1
        
        # 如果有检测结果，进行匹配
        if len(detections) > 0:
            matched_indices, unmatched_detections, unmatched_tracks = self._match(detections)
            
            # 更新匹配的追踪目标
            for track_idx, det_idx in matched_indices:
                det = detections[det_idx]
                bbox = det[:4]
                confidence = det[4]
                class_id = det[5] if len(det) > 5 else 0
                class_name = det[6] if len(det) > 6 else 'unknown'
                self.tracks[track_idx].update(bbox, confidence, class_id, class_name)
            
            # 为未匹配的检测创建新追踪
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                bbox = det[:4]
                confidence = det[4]
                class_id = det[5] if len(det) > 5 else 0
                class_name = det[6] if len(det) > 6 else 'unknown'
                new_track = Track(self.next_id, bbox, confidence, class_id, class_name)
                self.next_id += 1
                
                # ID达到上限时重置
                if self.next_id > self.max_id:
                    self.next_id = 1
                
                self.tracks.append(new_track)
            
            # 标记未匹配的追踪为丢失
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
        else:
            # 没有检测结果，所有追踪标记为丢失
            for track in self.tracks:
                track.mark_missed()
        
        # 移除过期的追踪
        self.tracks = [t for t in self.tracks 
                      if t.consecutive_invisible_count < self.max_age]
        
        # 返回确认的追踪（命中次数足够）
        confirmed_tracks = [t for t in self.tracks 
                           if t.total_visible_count >= self.min_hits]
        
        return confirmed_tracks
    
    def _match(self, detections):
        """匹配检测和追踪"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 计算IOU矩阵
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track.bbox, det[:4])
        
        # 使用匈牙利算法进行匹配
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for t, d in zip(track_indices, det_indices):
            if iou_matrix[t, d] < self.iou_threshold:
                continue
            matched_indices.append((t, d))
            if d in unmatched_detections:
                unmatched_detections.remove(d)
            if t in unmatched_tracks:
                unmatched_tracks.remove(t)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集区域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集区域
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """重置追踪器"""
        self.tracks = []
        self.next_id = 1

