"""
网络流读取优化模块
使用独立线程持续读取最新帧，避免缓冲区积压导致的卡顿
"""
import cv2
import threading
import time
import numpy as np


class StreamReader:
    """
    网络流读取器 - 使用独立线程持续获取最新帧
    """
    def __init__(self, source, skip_frames=3):
        """
        Args:
            source: 视频源
            skip_frames: 跳帧数量，默认3表示grab 3次只取最后1帧
                        设置为1表示不跳帧，2表示grab 2次取最后1帧
        """
        self.source = source
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.skip_frames = max(1, skip_frames)  # 至少为1
        
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
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
    
    def _open_capture(self):
        """打开视频捕获"""
        import os
        
        if self.is_network_stream:
            # 先使用默认配置打开，检测分辨率后再调整
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'  # 使用TCP传输
                'buffer_size;8192000|'  # 初始8MB缓冲区（适合高分辨率）
                'max_delay;200000|'  # 最大延迟200ms（高分辨率需要更多时间）
                'reorder_queue_size;0|'  # 禁用重排序
                'fflags;nobuffer+fastseek+flush_packets'  # 无缓冲+快速定位+刷新包
            )
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                # 最小化内部缓冲
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 尝试读取一帧以检测分辨率
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    # 根据分辨率动态调整缓冲区大小
                    # 计算公式：缓冲区 = 宽度 * 高度 * 3 * 压缩比 * 帧数
                    # 压缩比约0.3-0.5，考虑3-5帧缓冲
                    frame_size_bytes = w * h * 3
                    # 对于高分辨率，使用更大的缓冲区
                    if frame_size_bytes > 5000000:  # 大于5MB的帧（约1920x1080以上）
                        buffer_size = max(16384000, int(frame_size_bytes * 0.4 * 3))  # 至少16MB，或按帧大小计算
                        max_delay = 300000  # 300ms
                    elif frame_size_bytes > 2000000:  # 大于2MB的帧（约1280x720以上）
                        buffer_size = max(8192000, int(frame_size_bytes * 0.4 * 3))  # 至少8MB
                        max_delay = 200000  # 200ms
                    else:
                        buffer_size = 4096000  # 4MB
                        max_delay = 100000  # 100ms
                    
                    # 重新设置环境变量（需要重新打开）
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                        f'rtsp_transport;tcp|'
                        f'buffer_size;{buffer_size}|'
                        f'max_delay;{max_delay}|'
                        f'reorder_queue_size;0|'
                        f'fflags;nobuffer+fastseek+flush_packets'
                    )
                    # 重新打开以应用新配置
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        print(f"✅ 网络流已连接（分辨率: {w}x{h}, 缓冲区: {buffer_size//1024//1024}MB）")
                else:
                    print("✅ 网络流已连接（独立线程读取模式，无法检测分辨率）")
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _update(self):
        """
        后台线程持续读取最新帧
        """
        consecutive_failures = 0
        max_failures = 30
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                break
            
            try:
                # 对于网络流，使用grab()快速丢弃旧帧
                if self.is_network_stream:
                    # 连续grab多次，只retrieve最后一次
                    # 这样可以跳过缓冲区中的旧帧
                    # skip_frames控制丢帧数量：3表示丢2帧读1帧，1表示不丢帧
                    grabbed = False
                    for i in range(self.skip_frames):
                        grabbed = self.cap.grab()
                        if not grabbed:
                            break
                    
                    if grabbed:
                        ret, frame = self.cap.retrieve()
                    else:
                        ret, frame = False, None
                else:
                    # 本地摄像头直接读取
                    ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # 验证帧有效性（防止花屏）
                    if frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        # 检查帧数据是否异常（均值过低可能表示损坏）
                        frame_mean = np.mean(frame)
                        if frame_mean > 1.0:  # 有效帧的均值应该大于1
                            # 更新最新帧（使用copy确保线程安全）
                            with self.lock:
                                self.ret = True
                                self.frame = frame.copy()
                            consecutive_failures = 0
                        else:
                            # 帧数据异常，跳过
                            consecutive_failures += 1
                            if consecutive_failures < 5:  # 前几次异常不报错
                                continue
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("⚠️ 视频源连接丢失或帧数据异常")
                    break
                if consecutive_failures > 0:
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"读取帧错误: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.05)
        
        # 清理
        if self.cap:
            self.cap.release()
    
    def read(self):
        """
        获取最新帧
        Returns:
            (bool, numpy.ndarray): 成功标志和图像帧
        """
        with self.lock:
            if self.frame is None:
                return False, None
            # 返回帧的副本以确保线程安全
            # 注意：对于高分辨率帧，这个copy可能较慢，但为了线程安全是必要的
            return self.ret, self.frame.copy()
    
    def isOpened(self):
        """检查是否已打开"""
        return self.cap is not None and self.cap.isOpened()
    
    def get(self, prop_id):
        """获取属性"""
        if self.cap:
            return self.cap.get(prop_id)
        return 0
    
    def set_skip_frames(self, skip_frames):
        """
        动态设置跳帧数量
        Args:
            skip_frames: 跳帧数量，1表示不跳帧，3表示grab 3次只取最后1帧
        """
        with self.lock:
            self.skip_frames = max(1, skip_frames)
            print(f"✓ StreamReader跳帧已更新: {skip_frames}")
    
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

