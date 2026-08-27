"""
目标过线计数模块
用于统计多类别目标穿过计数线的数量
"""
import numpy as np
import time
from collections import defaultdict


class LineCrossingCounter:
    """过线计数器（支持多类别）"""
    def __init__(self):
        self.line_points = None  # 计数线的两个端点 [(x1, y1), (x2, y2)]
        
        # 总计数（不区分类别）
        self.count_a_to_b = 0  # 从A侧到B侧穿过的总数量
        self.count_b_to_a = 0  # 从B侧到A侧穿过的总数量
        
        # 分类别计数
        self.class_counts_a_to_b = defaultdict(int)  # {'person': 5, 'car': 3, ...}
        self.class_counts_b_to_a = defaultdict(int)
        
        # 当前屏幕目标数
        self.current_objects_count = 0  # 当前屏幕上的目标总数
        
        # 追踪状态
        self.crossed_ids = {}  # 记录每个ID最后的位置状态: {track_id: (stable_position, class_name, has_crossed, last_frame)}
        self.position_history = {}  # 记录每个ID的位置历史: {track_id: [(position, frame_count), ...]}
        self.last_position = {}  # 记录每个ID最后一次的原始位置: {track_id: position}
        self.frame_count = 0  # 帧计数器
        self.recent_crossings = {}  # {track_id: (direction, frame_count)}
        self.crossing_effect_frames = 12
        self.max_inactive_frames = 30  # 最大不活跃帧数（约1秒 @ 30fps，加快清理）
        self.max_tracked_ids = 200  # 最大跟踪ID数量，防止内存无限增长
        self.cleanup_interval = 100  # 每100帧执行一次强制清理
        self.start_time = None
        self.paused_time = 0  # 累计暂停时间
        self.pause_start_time = None  # 暂停开始时间
        
        # 过线检测参数
        self.history_length = 4  # 保留最近N帧的位置历史
        self.min_stable_frames = 2  # 至少连续N帧确认在同一侧才认为稳定
        self.position_threshold = 3.0  # 点到线的距离阈值（像素）
        
        # 分钟级统计数据记录（基于视频时间）
        self.minute_records = []  # 记录每分钟的统计数据
        self.last_minute_frame = 0  # 上一次记录时的帧数
        self.current_minute_counts = {'total': 0}  # 当前分钟内各类别的计数
        self.video_fps = 30.0  # 视频帧率（默认30fps，会在开始时更新）
        self.use_video_time = False  # 是否使用视频时间（视频文件为True，摄像头为False）
        
    def set_line(self, point1, point2, reset_counts=False):
        """
        设置计数线
        Args:
            point1: 第一个端点
            point2: 第二个端点
            reset_counts: 是否重置计数（默认False，只在开始计数时重置）
        """
        self.line_points = [point1, point2]
        # 清除crossed_ids和位置历史，因为线条位置变了，之前的位置记录无效
        self.crossed_ids = {}
        self.position_history = {}
        self.last_position = {}
        self.recent_crossings = {}
        
        # 只在明确要求时才重置计数
        if reset_counts:
            self.reset_counts()
        
    def reset_counts(self):
        """重置计数"""
        self.count_a_to_b = 0
        self.count_b_to_a = 0
        self.class_counts_a_to_b = defaultdict(int)
        self.class_counts_b_to_a = defaultdict(int)
        self.crossed_ids = {}
        self.position_history = {}
        self.last_position = {}
        self.recent_crossings = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.paused_time = 0
        self.pause_start_time = None
        
        # 重置分钟级记录
        self.minute_records = []
        self.last_minute_frame = 0
        self.current_minute_counts = {'total': 0}
    
    def pause(self):
        """暂停计数（开始记录暂停时间）"""
        if self.pause_start_time is None:
            self.pause_start_time = time.time()
    
    def resume(self):
        """恢复计数（累加暂停时间）"""
        if self.pause_start_time is not None:
            self.paused_time += time.time() - self.pause_start_time
            self.pause_start_time = None
        
    def clear_line(self):
        """清除计数线"""
        self.line_points = None
        self.reset_counts()
    
    def update(self, tracks):
        """
        更新计数
        Args:
            tracks: Track对象列表（每个Track有class_name属性）
        """
        # 更新当前屏幕目标总数
        self.current_objects_count = len(tracks)
        
        if self.line_points is None or len(self.line_points) != 2:
            return
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frame_count += 1
        
        # 获取当前活跃的track_id集合
        active_track_ids = {track.track_id for track in tracks}
        
        # 清理长时间未更新的ID
        self._cleanup_inactive_ids(active_track_ids)
        
        for track in tracks:
            track_id = track.track_id
            class_name = getattr(track, 'class_name', 'unknown')
            center = track.get_center()
            
            # 计算当前点相对于线的位置
            current_position = self._point_position(center)
            
            # 更新位置历史
            if track_id not in self.position_history:
                self.position_history[track_id] = []
            
            # 添加当前位置到历史记录
            self.position_history[track_id].append((current_position, self.frame_count))
            
            # 只保留最近N帧的历史
            if len(self.position_history[track_id]) > self.history_length:
                self.position_history[track_id].pop(0)
            
            # 检查位置是否稳定（连续多帧在同一侧）
            stable_position = self._get_stable_position(track_id)
            
            # 过线检测逻辑
            if track_id in self.crossed_ids:
                last_stable_position, last_class, has_crossed, _ = self.crossed_ids[track_id]
                
                # 当前位置稳定
                if stable_position != 0:
                    # 上次也稳定，检测是否过线
                    if last_stable_position != 0:
                        if stable_position != last_stable_position and not has_crossed:
                            # 从A侧到B侧
                            if last_stable_position < 0 < stable_position:
                                self.count_a_to_b += 1
                                self.class_counts_a_to_b[class_name] += 1
                                self._record_crossing(class_name, 'a_to_b')
                                self.recent_crossings[track_id] = ('a_to_b', self.frame_count)
                                self.crossed_ids[track_id] = (stable_position, class_name, True, self.frame_count)
                            # 从B侧到A侧
                            elif last_stable_position > 0 > stable_position:
                                self.count_b_to_a += 1
                                self.class_counts_b_to_a[class_name] += 1
                                self._record_crossing(class_name, 'b_to_a')
                                self.recent_crossings[track_id] = ('b_to_a', self.frame_count)
                                self.crossed_ids[track_id] = (stable_position, class_name, True, self.frame_count)
                            else:
                                # 位置变化但不是过线
                                self.crossed_ids[track_id] = (stable_position, class_name, has_crossed, self.frame_count)
                        else:
                            # 位置未变化或已计数，更新帧号
                            self.crossed_ids[track_id] = (stable_position, class_name, has_crossed, self.frame_count)
                    else:
                        # 上次不稳定，现在稳定了，记录稳定位置（重置has_crossed）
                        self.crossed_ids[track_id] = (stable_position, class_name, False, self.frame_count)
                # 当前不稳定，更新帧号但保持其他状态
                else:
                    self.crossed_ids[track_id] = (last_stable_position, last_class, has_crossed, self.frame_count)
            else:
                # 新ID，如果稳定则初始化
                if stable_position != 0:
                    self.crossed_ids[track_id] = (stable_position, class_name, False, self.frame_count)
        
        # 检查是否需要记录新的一分钟
        self._check_minute_record()
    
    def _cleanup_inactive_ids(self, active_track_ids):
        """
        清理长时间未更新的ID
        Args:
            active_track_ids: 当前帧中活跃的track_id集合
        """
        # 清理 crossed_ids 中不活跃的ID
        ids_to_remove = []
        for track_id, (_, _, _, last_frame) in self.crossed_ids.items():
            # 如果ID不在当前活跃列表中，且超过最大不活跃帧数
            if track_id not in active_track_ids:
                if self.frame_count - last_frame > self.max_inactive_frames:
                    ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.crossed_ids[track_id]
            if track_id in self.position_history:
                del self.position_history[track_id]
            if track_id in self.last_position:
                del self.last_position[track_id]
            if track_id in self.recent_crossings:
                del self.recent_crossings[track_id]
        
        # 周期性强制清理：每隔一定帧数执行更彻底的清理
        if self.frame_count % self.cleanup_interval == 0:
            self._force_cleanup(active_track_ids)

        # 清理过线特效历史，避免字典无界增长
        stale_ids = []
        for track_id, (_, cross_frame) in self.recent_crossings.items():
            if self.frame_count - cross_frame > self.crossing_effect_frames:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.recent_crossings[track_id]
    
    def _force_cleanup(self, active_track_ids):
        """
        强制清理，确保字典不会无限增长
        Args:
            active_track_ids: 当前帧中活跃的track_id集合
        """
        # 清理position_history中不在crossed_ids中的ID
        ph_to_remove = [tid for tid in self.position_history if tid not in self.crossed_ids]
        for tid in ph_to_remove:
            del self.position_history[tid]
        
        # 清理last_position中不在crossed_ids中的ID
        lp_to_remove = [tid for tid in self.last_position if tid not in self.crossed_ids]
        for tid in lp_to_remove:
            del self.last_position[tid]
        
        # 如果字典仍然过大，清理最旧的记录
        if len(self.crossed_ids) > self.max_tracked_ids:
            # 按最后更新帧排序，保留最新的
            sorted_ids = sorted(self.crossed_ids.items(), 
                              key=lambda x: x[1][3], reverse=True)
            # 只保留最新的max_tracked_ids个
            ids_to_keep = {item[0] for item in sorted_ids[:self.max_tracked_ids]}
            ids_to_remove = [tid for tid in self.crossed_ids if tid not in ids_to_keep]
            
            for tid in ids_to_remove:
                del self.crossed_ids[tid]
                if tid in self.position_history:
                    del self.position_history[tid]
                if tid in self.last_position:
                    del self.last_position[tid]
    
    def _get_stable_position(self, track_id):
        """
        获取目标的稳定位置（需要连续多帧在同一侧）
        Args:
            track_id: 追踪ID
        Returns:
            稳定的位置值（1, -1, 或 0）
        """
        if track_id not in self.position_history:
            return 0
        
        history = self.position_history[track_id]
        
        # 历史记录不足，认为不稳定
        if len(history) < self.min_stable_frames:
            return 0
        
        # 检查最近N帧是否都在同一侧
        recent_positions = [pos for pos, _ in history[-self.min_stable_frames:]]
        
        # 如果有任何一帧位置为0（在线上或无效），认为不稳定
        if 0 in recent_positions:
            return 0
        
        # 检查是否所有位置都相同（都是正数或都是负数）
        first_pos = recent_positions[0]
        if all(pos * first_pos > 0 for pos in recent_positions):
            # 所有位置同号，返回统一的符号
            return 1 if first_pos > 0 else -1
        
        # 位置不一致，不稳定
        return 0
    
    def _point_position(self, point):
        """
        判断点相对于线段的位置
        Returns:
            > 0: 点在线段一侧的有效区域
            < 0: 点在线段另一侧的有效区域
            = 0: 点在线上、不在线段延伸范围内、或太近
        """
        if self.line_points is None:
            return 0
        
        p1, p2 = self.line_points
        px, py = point
        
        # 步骤1: 检查点是否在线段的"垂直延伸范围"内
        # 计算点到线段的投影是否落在线段上
        # 使用点积判断投影位置
        
        # 线段向量
        line_vec = (p2[0] - p1[0], p2[1] - p1[1])
        # 点到起点的向量
        point_vec = (px - p1[0], py - p1[1])
        
        # 线段长度的平方
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        if line_len_sq == 0:
            return 0  # 线段长度为0
        
        # 计算投影比例 t (0 <= t <= 1 表示在线段范围内)
        t = (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq
        
        # 投影点不在线段上，而是在延长线上
        if t < 0 or t > 1:
            return 0  # 不在线段的垂直延伸范围内
        
        # 步骤2: 点在线段范围内，使用叉积判断在哪一侧
        # (p2.x - p1.x) * (py - p1.y) - (p2.y - p1.y) * (px - p1.x)
        cross_product = line_vec[0] * (py - p1[1]) - line_vec[1] * (px - p1[0])
        
        # 使用可配置的阈值，避免在线附近的点被误判
        if abs(cross_product) < self.position_threshold:
            return 0
        
        return 1 if cross_product > 0 else -1

    def get_crossing_effect(self, track_id):
        """
        获取指定目标是否处于“刚过线”的特效窗口。
        Returns:
            (is_active, direction, progress)
            progress: 0~1, 0 表示刚过线，1 表示特效接近结束
        """
        item = self.recent_crossings.get(track_id)
        if item is None:
            return False, None, 1.0

        direction, cross_frame = item
        age = self.frame_count - cross_frame
        duration = max(1, int(self.crossing_effect_frames))
        if age < 0 or age > duration:
            return False, None, 1.0

        progress = min(1.0, max(0.0, age / float(duration)))
        return True, direction, progress
    
    def get_statistics(self):
        """
        获取统计数据（总计）
        Returns:
            dict: 包含总体统计信息
        """
        total = self.count_a_to_b + self.count_b_to_a
        
        # 计算实际运行时间（总时间 - 暂停时间）
        if self.start_time:
            total_elapsed = time.time() - self.start_time
            # 如果当前正在暂停，需要加上当前暂停段的时间
            current_pause = 0
            if self.pause_start_time is not None:
                current_pause = time.time() - self.pause_start_time
            # 实际运行时间 = 总时间 - 累计暂停时间 - 当前暂停时间
            elapsed_time = total_elapsed - self.paused_time - current_pause
        else:
            elapsed_time = 0
        
        elapsed_minutes = elapsed_time / 60.0
        avg_per_minute = total / elapsed_minutes if elapsed_minutes > 0 else 0
        
        return {
            'count_a_to_b': self.count_a_to_b,
            'count_b_to_a': self.count_b_to_a,
            'total': total,
            'avg_per_minute': avg_per_minute,
            'elapsed_time': elapsed_time,
            'current_objects': self.current_objects_count  # 当前屏幕目标总数
        }
    
    def get_detailed_statistics(self):
        """
        获取详细统计数据（分类别）
        Returns:
            dict: {
                'class_stats': {
                    'person': {'a_to_b': 5, 'b_to_a': 3, 'total': 8},
                    'car': {'a_to_b': 10, 'b_to_a': 8, 'total': 18},
                    ...
                },
                'total_stats': {'a_to_b': 15, 'b_to_a': 11, 'total': 26},
                'elapsed_time': 120.5,
                'avg_per_minute': 13.0
            }
        """
        # 获取所有出现过的类别
        all_classes = set(self.class_counts_a_to_b.keys()) | set(self.class_counts_b_to_a.keys())
        
        class_stats = {}
        for class_name in all_classes:
            a_to_b = self.class_counts_a_to_b.get(class_name, 0)
            b_to_a = self.class_counts_b_to_a.get(class_name, 0)
            total = a_to_b + b_to_a
            class_stats[class_name] = {
                'a_to_b': a_to_b,
                'b_to_a': b_to_a,
                'total': total
            }
        
        # 总计统计（使用扣除暂停时间的实际运行时间）
        total = self.count_a_to_b + self.count_b_to_a
        
        # 计算实际运行时间（总时间 - 暂停时间）
        if self.start_time:
            total_elapsed = time.time() - self.start_time
            # 如果当前正在暂停，需要加上当前暂停段的时间
            current_pause = 0
            if self.pause_start_time is not None:
                current_pause = time.time() - self.pause_start_time
            # 实际运行时间 = 总时间 - 累计暂停时间 - 当前暂停时间
            elapsed_time = total_elapsed - self.paused_time - current_pause
        else:
            elapsed_time = 0
        
        elapsed_minutes = elapsed_time / 60.0
        avg_per_minute = total / elapsed_minutes if elapsed_minutes > 0 else 0
        
        return {
            'class_stats': class_stats,
            'total_stats': {
                'a_to_b': self.count_a_to_b,
                'b_to_a': self.count_b_to_a,
                'total': total
            },
            'current_objects': self.current_objects_count,
            'elapsed_time': elapsed_time,
            'avg_per_minute': avg_per_minute
        }
    
    def draw_line(self, image, draw_debug_info=False, tracks=None, line_thickness=3, line_color=(0, 255, 0)):
        """
        在图像上绘制计数线
        Args:
            image: OpenCV图像
            draw_debug_info: 是否绘制调试信息（显示目标的位置状态）
            tracks: Track对象列表（用于调试显示）
            line_thickness: 线条粗细
            line_color: 线条颜色（BGR格式）
        Returns:
            绘制了线的图像
        """
        import cv2
        if self.line_points is None or len(self.line_points) != 2:
            return image
        
        p1, p2 = self.line_points
        # 绘制计数线（可自定义颜色和粗细）
        cv2.line(image, 
                (int(p1[0]), int(p1[1])), 
                (int(p2[0]), int(p2[1])), 
                line_color, line_thickness)
        
        # 在线的两端绘制圆点并标注A和B
        # A端（起点）- 使用配置的颜色
        circle_radius = max(5, int(line_thickness * 2.5))
        cv2.circle(image, (int(p1[0]), int(p1[1])), circle_radius, line_color, -1)
        cv2.putText(image, 'A', (int(p1[0]) - 20, int(p1[1]) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, max(1, line_thickness - 1))
        
        # B端（终点）- 使用配置的颜色
        cv2.circle(image, (int(p2[0]), int(p2[1])), circle_radius, line_color, -1)
        cv2.putText(image, 'B', (int(p2[0]) - 20, int(p2[1]) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, max(1, line_thickness - 1))
        
        # 绘制调试信息
        if draw_debug_info and tracks is not None:
            for track in tracks:
                track_id = track.track_id
                center = track.get_center()
                
                # 获取当前位置状态
                position = self._point_position(center)
                stable_position = self._get_stable_position(track_id)
                
                # 获取历史信息
                history_len = len(self.position_history.get(track_id, []))
                has_crossed = self.crossed_ids.get(track_id, (0, '', False, 0))[2]
                
                # 在目标中心绘制位置信息
                if position > 0:
                    pos_text = "B"
                    pos_color = (255, 0, 255)  # 品红色
                elif position < 0:
                    pos_text = "A"
                    pos_color = (0, 255, 255)  # 黄色
                else:
                    pos_text = "?"
                    pos_color = (128, 128, 128)  # 灰色
                
                # 稳定状态标记
                stable_mark = "✓" if stable_position != 0 else "✗"
                crossed_mark = "✓" if has_crossed else ""
                
                debug_text = f"{pos_text}{stable_mark} H:{history_len} {crossed_mark}"
                cv2.putText(image, debug_text,
                           (int(center[0]) - 30, int(center[1]) + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, pos_color, 2)
        
        return image
    
    def is_point_near_line_endpoint(self, point, threshold=15):
        """
        检查点是否靠近线的端点
        Args:
            point: (x, y) 坐标
            threshold: 距离阈值
        Returns:
            0: 不靠近任何端点
            1: 靠近第一个端点
            2: 靠近第二个端点
        """
        if self.line_points is None:
            return 0
        
        p1, p2 = self.line_points
        
        dist1 = np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
        dist2 = np.sqrt((point[0] - p2[0])**2 + (point[1] - p2[1])**2)
        
        if dist1 < threshold:
            return 1
        elif dist2 < threshold:
            return 2
        else:
            return 0
    
    def _record_crossing(self, class_name, direction='a_to_b'):
        """
        记录过线事件（用于分钟级统计）
        Args:
            class_name: 类别名称
            direction: 方向 'a_to_b' 或 'b_to_a'
        """
        # 更新当前分钟的计数
        self.current_minute_counts['total'] = self.current_minute_counts.get('total', 0) + 1
        key = f'{class_name}_{direction}'
        self.current_minute_counts[key] = self.current_minute_counts.get(key, 0) + 1
    
    def _check_minute_record(self):
        """检查是否需要记录新的一分钟数据（基于帧数和视频FPS）"""
        if self.pause_start_time is not None:
            return
        
        interval_seconds = 5  # 每5秒记录一次

        if self.use_video_time:
            # 使用视频时间（基于帧数和FPS）
            frames_per_interval = int(self.video_fps * interval_seconds)
            frames_since_last = self.frame_count - self.last_minute_frame

            # 每到间隔帧数就记录一次
            if frames_since_last >= frames_per_interval:
                intervals_passed = int(frames_since_last / frames_per_interval)

                for i in range(intervals_passed):
                    minute_index = len(self.minute_records) + 1
                    # 计算视频时间戳
                    video_time_seconds = (self.last_minute_frame + (i + 1) * frames_per_interval) / self.video_fps

                    record = {
                        'minute': minute_index,
                        'video_time': video_time_seconds,  # 视频时间（秒）
                        'counts': self.current_minute_counts.copy(),
                        'objects_count': self.current_objects_count,
                    }
                    self.minute_records.append(record)

                    # 重置当前间隔计数
                    self.current_minute_counts = {'total': 0}

                # 更新上次记录的帧数
                self.last_minute_frame += int(intervals_passed * frames_per_interval)
        else:
            # 使用实际时间（摄像头模式）
            if self.start_time is None:
                return

            current_time = time.time()
            elapsed = current_time - self.start_time - self.paused_time
            last_elapsed = self.last_minute_frame  # 这里存储的是上次的秒数

            # 每到间隔秒数就记录一次
            if elapsed - last_elapsed >= interval_seconds:
                intervals_passed = int((elapsed - last_elapsed) / interval_seconds)

                for i in range(intervals_passed):
                    minute_index = len(self.minute_records) + 1

                    record = {
                        'minute': minute_index,
                        'video_time': last_elapsed + (i + 1) * interval_seconds,
                        'counts': self.current_minute_counts.copy(),
                        'objects_count': self.current_objects_count,
                    }
                    self.minute_records.append(record)

                    # 重置当前间隔计数
                    self.current_minute_counts = {'total': 0}

                # 更新上次记录的时间
                self.last_minute_frame = last_elapsed + intervals_passed * interval_seconds
    
    def get_minute_records(self):
        """
        获取分钟级统计记录
        Returns:
            list: 分钟级记录列表
        """
        return self.minute_records
    
    def finalize_current_minute(self):
        """
        完成当前分钟的记录（停止计数时调用）
        """
        # 如果当前分钟有数据，记录下来（避免重复记录）
        if self.current_minute_counts.get('total', 0) > 0:
            minute_index = len(self.minute_records) + 1
            
            if self.use_video_time:
                # 使用视频时间
                video_time_seconds = self.frame_count / self.video_fps
            else:
                # 使用实际时间
                if self.start_time:
                    video_time_seconds = time.time() - self.start_time - self.paused_time
                else:
                    video_time_seconds = 0
            
            record = {
                'minute': minute_index,
                'video_time': video_time_seconds,
                'counts': self.current_minute_counts.copy(),
                'objects_count': self.current_objects_count,
            }
            self.minute_records.append(record)
            
            # 清空当前计数（避免再次记录）
            self.current_minute_counts = {'total': 0}
