"""
CameraManager: FastAPI 与 VideoDetector 的桥接层 (纯 Python 版)
处理视频检测、参数管理、日志生成等业务逻辑
"""
import os
import cv2
import threading
import time
import json

# 导入重构后的 core 模块
from app.core.algorithm.detector import VideoDetector, get_available_cameras, get_frame_from_source
from app.core.config import settings

class CameraManager:
    """单例模式的相机管理器"""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.detector = VideoDetector()
        self.current_frame = None
        self.current_stats = {}
        self.is_running = False
        self.is_paused = False
        self.lock = threading.Lock()
        self.frame_bytes = None
        self.preview_bytes = None
        self.cached_cameras = []
        self.camera_scan_in_progress = False
        self.camera_scan_last_ts = 0.0

        # 快速分析相关
        self.fast_analysis_running = False
        self.fast_analysis_progress = 0
        self.fast_analysis_total = 0
        self.fast_analysis_status = ''

        # 统计历史
        self.stats_history = []
        self.max_history = 120

        # 设置回调替代 PyQt 信号
        self.detector.set_callback('on_frame', self._on_frame)
        self.detector.set_callback('on_stats', self._on_stats)
        self.detector.set_callback('on_error', self._on_error)
        self.detector.set_callback('on_finished', self._on_finished)

        # 配置文件路径
        self.config_file = str(settings.BASE_DIR / "config.json")

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraManager()
            return cls._instance

    # ==================== 回调函数 ====================

    def _on_frame(self, frame):
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with self.lock:
                    self.frame_bytes = buffer.tobytes()
                    self.current_frame = frame
        except Exception as e:
            print(f"Frame encode error: {e}")

    def _on_stats(self, stats):
        with self.lock:
            self.current_stats = stats
            entry = {
                'time': time.time(),
                'count_a_to_b': stats.get('count_a_to_b', 0),
                'count_b_to_a': stats.get('count_b_to_a', 0),
                'total': stats.get('total', 0),
            }
            self.stats_history.append(entry)
            if len(self.stats_history) > self.max_history:
                self.stats_history = self.stats_history[-self.max_history:]

    def _on_error(self, msg):
        print(f"[Detector Error] {msg}")

    def _on_finished(self):
        with self.lock:
            self.is_running = False

    # ==================== 业务逻辑 (保持原有接口兼容) ====================

    def get_frame(self):
        with self.lock: return self.frame_bytes

    def get_stats(self):
        with self.lock: return dict(self.current_stats)

    def get_stats_history(self):
        with self.lock: return list(self.stats_history)

    def load_model(self, model_path):
        return self.detector.load_model(model_path)

    def get_model_info(self):
        if self.detector.model is None: return {'loaded': False}
        return {
            'loaded': True,
            'type': self.detector.model_type,
            'backend': self.detector.model_backend,
            'loaded_model_path': self.detector.loaded_model_path,
            'loaded_runtime_path': self.detector.loaded_runtime_path,
            'requested_device': self.detector.infer_device,
            'runtime_device': self.detector.runtime_device,
            'fp16_enabled': self.detector.fp16_enabled,
            'tensor_rt_enabled': self.detector.tensor_rt_enabled,
            'classes': list(self.detector.class_names_map.values()),
            'enabled_classes': list(self.detector.enabled_classes),
        }

    def _camera_scan_worker(self):
        try:
            cameras = get_available_cameras()
            with self.lock:
                self.cached_cameras = cameras
                self.camera_scan_last_ts = time.time()
        except Exception as e:
            print(f"Camera scan error: {e}")
        finally:
            with self.lock:
                self.camera_scan_in_progress = False

    def start_camera_scan_async(self, force=False):
        with self.lock:
            if self.camera_scan_in_progress:
                return False
            if not force and self.cached_cameras and (time.time() - self.camera_scan_last_ts) < 15:
                return False
            self.camera_scan_in_progress = True

        threading.Thread(target=self._camera_scan_worker, daemon=True, name="camera-scan-worker").start()
        return True

    def is_camera_scan_in_progress(self):
        with self.lock:
            return self.camera_scan_in_progress

    def detect_cameras(self, blocking=False, force_refresh=False):
        if blocking:
            cameras = get_available_cameras()
            with self.lock:
                self.cached_cameras = cameras
                self.camera_scan_last_ts = time.time()
                self.camera_scan_in_progress = False
            return cameras

        self.start_camera_scan_async(force=force_refresh or not self.cached_cameras)
        with self.lock:
            return list(self.cached_cameras)

    def set_video_source(self, source):
        self.detector.set_video_source(source)
        frame = get_frame_from_source(source)
        if frame is not None:
            with self.lock: self.current_frame = frame
            frame_with_line = self.detector.counter.draw_line(frame.copy(), 
                line_thickness=self.detector.line_thickness, line_color=self.detector.line_color)
            ret, buffer = cv2.imencode('.jpg', frame_with_line, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                with self.lock:
                    self.preview_bytes = buffer.tobytes()
                    self.frame_bytes = buffer.tobytes()
            return True
        return False

    def get_preview(self):
        with self.lock: return self.preview_bytes

    def get_classes(self):
        return {'all': list(self.detector.class_names_map.values()), 'enabled': list(self.detector.enabled_classes)}

    def set_enabled_classes(self, class_names):
        self.detector.set_enabled_classes(class_names)

    def set_line(self, point1, point2, reset_counts=True):
        self.detector.set_counting_line(tuple(point1), tuple(point2), reset_counts=reset_counts)
        if self.current_frame is not None:
            frame_with_line = self.detector.counter.draw_line(self.current_frame.copy(), 
                line_thickness=self.detector.line_thickness, line_color=self.detector.line_color)
            ret, buffer = cv2.imencode('.jpg', frame_with_line, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                with self.lock:
                    self.preview_bytes = buffer.tobytes()
                    if not self.is_running: self.frame_bytes = buffer.tobytes()

    def clear_line(self): self.detector.clear_counting_line()

    def get_line_points(self):
        pts = self.detector.counter.line_points
        return {'point1': list(pts[0]), 'point2': list(pts[1])} if pts else None

    def start_detection(self, source=None):
        if source: self.detector.set_video_source(source)
        if not self.is_running:
            self.stats_history.clear()
            self.detector.start_detection()
            self.is_running = True
            self.is_paused = False

    def pause_detection(self):
        if self.is_running:
            if self.is_paused:
                self.detector.resume_detection()
                self.detector.counter.resume()
                self.is_paused = False
            else:
                self.detector.pause_detection()
                self.detector.counter.pause()
                self.is_paused = True

    def stop_detection(self):
        if self.is_running:
            self.detector.counter.finalize_current_minute()
            self.detector.stop_detection()
            self.is_running, self.is_paused = False, False

    def reset_counter(self):
        self.detector.reset_counter()
        self.stats_history.clear()
        # 重新抓取视频第一帧，让画面回到初始状态（无检测框）
        source = self.detector.video_source
        if source is not None:
            from app.core.algorithm.detector import get_frame_from_source
            frame = get_frame_from_source(source)
            if frame is not None:
                with self.lock:
                    self.current_frame = frame
                frame_with_line = self.detector.counter.draw_line(
                    frame.copy(),
                    line_thickness=self.detector.line_thickness,
                    line_color=self.detector.line_color,
                )
                ret, buffer = cv2.imencode('.jpg', frame_with_line, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    raw = buffer.tobytes()
                    with self.lock:
                        self.preview_bytes = raw
                        self.frame_bytes = raw

    def get_status(self):
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'model_loaded': self.detector.model is not None,
            'has_source': self.detector.video_source is not None,
            'has_line': self.detector.counter.line_points is not None,
            'camera_scan_in_progress': self.is_camera_scan_in_progress(),
            'infer_device': self.detector.infer_device,
            'runtime_device': self.detector.runtime_device,
            'infer_imgsz': self.detector.infer_imgsz,
            'fp16_enabled': self.detector.fp16_enabled,
            'tensor_rt_enabled': self.detector.tensor_rt_enabled,
            'model_backend': self.detector.model_backend,
        }

    def get_detailed_stats(self): return self.detector.counter.get_detailed_statistics()

    def get_settings(self):
        return {
            'yolo': {'confidence': self.detector.confidence_threshold},
            'tracking': {
                'max_age': self.detector.tracker.max_age,
                'min_hits': self.detector.tracker.min_hits,
            },
            'display': {
                'invisible_threshold': self.detector.display_invisible_threshold,
                'show_bbox': self.detector.show_bbox,
                'show_label': self.detector.show_label,
                'show_center': self.detector.show_center,
                'show_trajectory': self.detector.show_trajectory,
                'stats_font_scale': self.detector.stats_font_scale,
                'line_thickness': self.detector.line_thickness,
                'bbox_thickness': self.detector.bbox_thickness,
                'label_font_scale': self.detector.label_font_scale,
                'center_size': self.detector.center_size,
                'avg_frame_window': self.detector.avg_frame_window,
                'stats_font_color': list(self.detector.stats_font_color),
                'label_font_color': list(self.detector.label_font_color),
                'bbox_color': list(self.detector.bbox_color),
                'center_color': list(self.detector.center_color),
                'trajectory_color': list(self.detector.trajectory_color),
                'line_color': list(self.detector.line_color),
            }
        }

    def update_settings(self, settings):
        d = self.detector
        if 'yolo' in settings: d.confidence_threshold = float(settings['yolo'].get('confidence', d.confidence_threshold))
        if 'tracking' in settings:
            ts = settings['tracking']
            d.tracker.max_age = int(ts.get('max_age', d.tracker.max_age))
            d.tracker.min_hits = int(ts.get('min_hits', d.tracker.min_hits))
        if 'display' in settings:
            disp = settings['display']
            d.display_invisible_threshold = int(disp.get('invisible_threshold', d.display_invisible_threshold))
            for k in ['show_bbox', 'show_label', 'show_center', 'show_trajectory', 'line_thickness', 
                      'bbox_thickness', 'center_size', 'avg_frame_window']:
                if k in disp: setattr(d, k, disp[k])
            for k in ['stats_font_scale', 'label_font_scale']:
                if k in disp: setattr(d, k, float(disp[k]))
            for k in ['stats_font_color', 'label_font_color', 'bbox_color', 'center_color', 'trajectory_color', 'line_color']:
                if k in disp: setattr(d, k, tuple(disp[k]))

    def load_preset(self, preset_name):
        presets = {
            'standard': {'yolo': {'confidence': 0.6}, 'tracking': {'max_age': 5, 'min_hits': 4}, 'display': {'invisible_threshold': 1}},
            'crowded': {'yolo': {'confidence': 0.65}, 'tracking': {'max_age': 3, 'min_hits': 5}, 'display': {'invisible_threshold': 0}},
            'sparse': {'yolo': {'confidence': 0.35}, 'tracking': {'max_age': 15, 'min_hits': 2}, 'display': {'invisible_threshold': 4}}
        }
        if preset_name in presets: self.update_settings(presets[preset_name]); return True
        return False

    def save_log(self, save_dir):
        """保存日志CSV和折线图"""
        minute_records = self.detector.counter.get_minute_records()
        if not minute_records:
            return {'success': False, 'message': '没有可保存的统计数据'}

        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"counting_log_{timestamp}.csv"
            chart_filename = f"counting_chart_{timestamp}.png"
            csv_path = os.path.join(save_dir, csv_filename)
            chart_path = os.path.join(save_dir, chart_filename)

            enabled_classes = list(self.detector.enabled_classes)
            data_rows = []
            cum_a_to_b = 0
            cum_b_to_a = 0

            for record in minute_records:
                v_time = record['video_time']
                t_str = f"{int(v_time // 60):02d}:{int(v_time % 60):02d}"
                counts = record['counts']

                # 计算本间隔各方向合计
                interval_a_to_b = sum(
                    counts.get(f'{cn}_a_to_b', 0) for cn in enabled_classes
                )
                interval_b_to_a = sum(
                    counts.get(f'{cn}_b_to_a', 0) for cn in enabled_classes
                )
                interval_total = interval_a_to_b + interval_b_to_a

                # 累计值
                cum_a_to_b += interval_a_to_b
                cum_b_to_a += interval_b_to_a
                cum_total = cum_a_to_b + cum_b_to_a

                # 过线速率（/分钟）：按5秒间隔折算
                rate_per_min = interval_total / 5.0 * 60.0 if v_time > 0 else 0

                row = {
                    '序号': record['minute'],
                    '时间': t_str,
                    '累计秒数': round(v_time, 1),
                    'A→B': interval_a_to_b,
                    'B→A': interval_b_to_a,
                    '合计': interval_total,
                    '累计A→B': cum_a_to_b,
                    '累计B→A': cum_b_to_a,
                    '累计合计': cum_total,
                    '过线速率(/分钟)': round(rate_per_min, 1),
                    '画面人数': record.get('objects_count', 0),
                }
                for cn in enabled_classes:
                    row[f'{cn}_A→B'] = counts.get(f'{cn}_a_to_b', 0)
                    row[f'{cn}_B→A'] = counts.get(f'{cn}_b_to_a', 0)
                data_rows.append(row)

            df = pd.DataFrame(data_rows)
            # 列顺序：基本信息 → 合计 → 累计 → 速率 → 画面人数 → 分类明细
            base_cols = ['序号', '时间', '累计秒数', 'A→B', 'B→A', '合计',
                         '累计A→B', '累计B→A', '累计合计', '过线速率(/分钟)', '画面人数']
            class_cols = []
            for cn in enabled_classes:
                class_cols.append(f'{cn}_A→B')
                class_cols.append(f'{cn}_B→A')
            df = df[base_cols + class_cols]
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # ---- 绘图 ----
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
            x = df['序号'].tolist()
            x_labels = df['时间'].tolist()
            colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

            multi_class = len(enabled_classes) > 1

            # 子图1：A→B方向
            for i, cn in enumerate(enabled_classes):
                col = f'{cn}_A→B'
                if col in df.columns:
                    ax1.plot(x, df[col].tolist(), marker='o', linewidth=2 if not multi_class else 1.5,
                             label=cn, color=colors[i % len(colors)],
                             alpha=1.0 if not multi_class else 0.7)
            if multi_class:
                ax1.plot(x, df['A→B'].tolist(), marker='D', linewidth=2.5,
                         label='合计', color='black', zorder=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(x_labels, rotation=45, fontsize=8)
            ax1.set_ylabel('过线数量')
            ax1.set_title('A→B 方向统计', fontweight='bold')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 子图2：B→A方向
            for i, cn in enumerate(enabled_classes):
                col = f'{cn}_B→A'
                if col in df.columns:
                    ax2.plot(x, df[col].tolist(), marker='s', linewidth=2 if not multi_class else 1.5,
                             label=cn, color=colors[i % len(colors)],
                             alpha=1.0 if not multi_class else 0.7)
            if multi_class:
                ax2.plot(x, df['B→A'].tolist(), marker='D', linewidth=2.5,
                         label='合计', color='black', zorder=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=45, fontsize=8)
            ax2.set_ylabel('过线数量')
            ax2.set_title('B→A 方向统计', fontweight='bold')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 子图3：累计趋势
            ax3.plot(x, df['累计A→B'].tolist(), marker='o', linewidth=2,
                     label='累计 A→B', color='#2980b9')
            ax3.plot(x, df['累计B→A'].tolist(), marker='s', linewidth=2,
                     label='累计 B→A', color='#c0392b')
            ax3.plot(x, df['累计合计'].tolist(), marker='D', linewidth=2.5,
                     label='累计合计', color='black')
            ax3.set_xticks(x)
            ax3.set_xticklabels(x_labels, rotation=45, fontsize=8)
            ax3.set_ylabel('累计过线数量')
            ax3.set_title('累计趋势', fontweight='bold')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            return {'success': True, 'csv': csv_filename, 'chart': chart_filename, 'path': save_dir}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def run_fast_analysis(self, video_path, save_dir):
        def _run():
            self.fast_analysis_running, self.fast_analysis_progress, self.fast_analysis_status = True, 0, '正在初始化...'
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): self.fast_analysis_status, self.fast_analysis_running = '无法打开视频文件', False; return
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                self.fast_analysis_total, d = total, self.detector
                d.counter.video_fps, d.counter.use_video_time = float(fps), True
                d.reset_counter()
                f_count = 0
                while cap.isOpened() and self.fast_analysis_running:
                    ret, frame = cap.read()
                    if not ret: break
                    d.counter.update(d.detect_and_track(frame))
                    f_count += 1
                    self.fast_analysis_progress = f_count
                    if f_count % 100 == 0: self.fast_analysis_status = f'已处理 {f_count}/{total} 帧 ({int(f_count/total*100)}%)'
                cap.release()
                d.counter.finalize_current_minute()
                self.fast_analysis_status = f'分析完成！已保存日志到 {save_dir}'
            except Exception as e: self.fast_analysis_status = f'分析失败: {e}'
            finally: self.fast_analysis_running = False
        threading.Thread(target=_run, daemon=True).start()

    def get_analysis_progress(self):
        return {
            'running': self.fast_analysis_running, 'progress': self.fast_analysis_progress, 
            'total': self.fast_analysis_total, 'status': self.fast_analysis_status,
            'percentage': int(self.fast_analysis_progress / self.fast_analysis_total * 100) if self.fast_analysis_total > 0 else 0
        }

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f: config = json.load(f)
                d = self.detector
                if 'yolo_params' in config: d.confidence_threshold = config['yolo_params'].get('confidence', 0.5)
                tracking_cfg = config.get('tracking_params') or config.get('deepsort_params')
                if tracking_cfg:
                    d.tracker.max_age = tracking_cfg.get('max_age', 5)
                    d.tracker.min_hits = tracking_cfg.get('min_hits', 3)
                if 'display_params' in config:
                    dsp = config['display_params']
                    d.display_invisible_threshold = dsp.get('invisible_threshold', d.display_invisible_threshold)
                    for k in ['show_bbox', 'show_label', 'show_center', 'show_trajectory', 'stats_font_scale', 'line_thickness', 'bbox_thickness', 'label_font_scale', 'center_size', 'avg_frame_window']:
                        if k in dsp: setattr(d, k, dsp[k])
                    for k in ['stats_font_color', 'label_font_color', 'bbox_color', 'center_color', 'trajectory_color', 'line_color']:
                        if k in dsp: setattr(d, k, tuple(dsp[k]))
                return config
            except Exception as e: print(f"加载配置失败: {e}")
        return {}

    def save_config(self):
        d = self.detector
        config = {
            'yolo_params': {'confidence': d.confidence_threshold},
            'tracking_params': {'max_age': d.tracker.max_age, 'min_hits': d.tracker.min_hits},
            'display_params': {
                'invisible_threshold': d.display_invisible_threshold, 'show_bbox': d.show_bbox, 'show_label': d.show_label,
                'show_center': d.show_center, 'show_trajectory': d.show_trajectory, 'stats_font_scale': d.stats_font_scale,
                'line_thickness': d.line_thickness, 'bbox_thickness': d.bbox_thickness, 'label_font_scale': d.label_font_scale,
                'center_size': d.center_size, 'avg_frame_window': d.avg_frame_window,
                'stats_font_color': list(d.stats_font_color), 'label_font_color': list(d.label_font_color),
                'bbox_color': list(d.bbox_color), 'center_color': list(d.center_color),
                'trajectory_color': list(d.trajectory_color), 'line_color': list(d.line_color)
            }
        }
        if d.counter.line_points: config['last_line_points'] = [list(d.counter.line_points[0]), list(d.counter.line_points[1])]
        try:
            with open(self.config_file, 'w') as f: json.dump(config, f, indent=4)
        except Exception as e: print(f"保存配置失败: {e}")


def get_camera_manager() -> CameraManager:
    return CameraManager.get_instance()
