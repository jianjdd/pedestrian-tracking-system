"""
自适应网络流读取器
支持两种模式：实时模式（低延迟）和完整模式（不漏帧）
"""
import cv2
import threading
import time
import queue


class AdaptiveStreamReader:
    """
    自适应网络流读取器
    
    两种模式：
    1. 实时模式(realtime)：优先显示最新帧，可能跳帧
       - 适合：实时监控、快速响应
       - 缺点：可能漏计快速移动的目标
       
    2. 完整模式(complete)：处理每一帧，不跳帧
       - 适合：精确计数、不能漏计
       - 缺点：画面会有延迟累积
    """
    
    def __init__(self, source, mode='realtime', buffer_size=5):
        """
        Args:
            source: 视频源
            mode: 'realtime' 或 'complete'
            buffer_size: 缓冲区大小（仅complete模式）
        """
        self.source = source
        self.mode = mode
        self.stopped = False
        self.lock = threading.Lock()
        
        # 实时模式：单帧缓存
        self.frame = None
        self.ret = False
        
        # 完整模式：队列缓存
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # 统计信息
        self.total_read = 0  # 读取的总帧数
        self.total_processed = 0  # 处理的总帧数（由外部更新）
        self.dropped_frames = 0  # 丢弃的帧数
        
        # 判断是否为网络流
        self.is_network_stream = isinstance(source, str) and (
            source.startswith('rtsp://') or
            source.startswith('rtmp://') or
            source.startswith('http://') or
            source.startswith('https://')
        )
        
        # 打开视频捕获
        self.cap = None
        self._open_capture()
        
        # 启动读取线程
        self.thread = None
        if self.cap and self.cap.isOpened():
            if self.mode == 'realtime':
                self.thread = threading.Thread(target=self._update_realtime, daemon=True)
            else:
                self.thread = threading.Thread(target=self._update_complete, daemon=True)
            self.thread.start()
            print(f"✅ StreamReader启动 - 模式: {self.mode}")
    
    def _open_capture(self):
        """打开视频捕获"""
        import os
        
        if self.is_network_stream:
            # 网络流优化配置
            if self.mode == 'realtime':
                # 实时模式：极致低延迟
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'
                    'buffer_size;1024000|'  # 1MB缓冲
                    'max_delay;100000|'  # 100ms最大延迟
                    'reorder_queue_size;0|'
                    'fflags;nobuffer+fastseek+flush_packets'
                )
            else:
                # 完整模式：稳定性优先
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'
                    'buffer_size;4096000|'  # 4MB缓冲
                    'max_delay;500000|'  # 500ms最大延迟
                    'fflags;nobuffer'
                )
            
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _update_realtime(self):
        """
        实时模式更新线程：主动丢弃旧帧，只保留最新
        """
        consecutive_failures = 0
        max_failures = 30
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                break
            
            try:
                # 对于网络流，使用grab()快速跳帧
                if self.is_network_stream:
                    # 丢弃2帧，读取第3帧
                    for i in range(3):
                        grabbed = self.cap.grab()
                        if not grabbed:
                            break
                        self.total_read += 1
                        if i < 2:  # 前2帧丢弃
                            self.dropped_frames += 1
                    
                    if grabbed:
                        ret, frame = self.cap.retrieve()
                    else:
                        ret, frame = False, None
                else:
                    # 本地摄像头直接读取
                    ret, frame = self.cap.read()
                    self.total_read += 1
                
                if ret and frame is not None:
                    # 更新最新帧（覆盖旧帧）
                    with self.lock:
                        if self.frame is not None:
                            self.dropped_frames += 1  # 旧帧被覆盖，算作丢弃
                        self.ret = True
                        self.frame = frame.copy()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("⚠️ 视频源连接丢失")
                        break
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"读取帧错误: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.05)
        
        if self.cap:
            self.cap.release()
    
    def _update_complete(self):
        """
        完整模式更新线程：不丢帧，按顺序放入队列
        """
        consecutive_failures = 0
        max_failures = 30
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                break
            
            try:
                ret, frame = self.cap.read()
                self.total_read += 1
                
                if ret and frame is not None:
                    # 放入队列（如果队列满了会阻塞）
                    try:
                        self.frame_queue.put((True, frame.copy()), timeout=1.0)
                        consecutive_failures = 0
                    except queue.Full:
                        # 队列满了，说明处理太慢
                        # 可以选择丢弃或等待，这里选择丢弃
                        self.dropped_frames += 1
                        print("⚠️ 处理速度过慢，队列已满，丢弃帧")
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("⚠️ 视频源连接丢失")
                        break
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"读取帧错误: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.05)
        
        if self.cap:
            self.cap.release()
    
    def read(self):
        """
        获取帧
        Returns:
            (bool, numpy.ndarray): 成功标志和图像帧
        """
        if self.mode == 'realtime':
            # 实时模式：返回最新帧
            with self.lock:
                if self.frame is None:
                    return False, None
                self.total_processed += 1
                return self.ret, self.frame.copy()
        else:
            # 完整模式：从队列获取帧（按顺序）
            try:
                ret, frame = self.frame_queue.get(timeout=1.0)
                self.total_processed += 1
                return ret, frame
            except queue.Empty:
                return False, None
    
    def isOpened(self):
        """检查是否已打开"""
        return self.cap is not None and self.cap.isOpened()
    
    def get(self, prop_id):
        """获取属性"""
        if self.cap:
            return self.cap.get(prop_id)
        return 0
    
    def get_statistics(self):
        """
        获取统计信息
        Returns:
            dict: 包含读取、处理、丢帧信息
        """
        return {
            'mode': self.mode,
            'total_read': self.total_read,
            'total_processed': self.total_processed,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / self.total_read if self.total_read > 0 else 0
        }
    
    def release(self):
        """释放资源"""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None

