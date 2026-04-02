"""
数据采集模块 - 负责从ADRV9009采集数据
"""
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict
import numpy as np

try:
    import adi
    import iio
    ADI_AVAILABLE = True
except ImportError:
    ADI_AVAILABLE = False
    print("警告: adi/iio 模块不可用，将使用模拟数据")

from src.models.signal_data import SignalData
from src.utils.logger import get_logger
from src.utils.data_validator import validate_signal_data

class DataAcquisition:
    """数据采集器"""
    
    def __init__(self, config: Dict[str, Any], db_manager=None):
        self.config = config
        self.db = db_manager
        self.logger = get_logger(__name__)
        self.sdr = None
        self.is_running = False
        self.callbacks = []
        self.data_queue = queue.Queue(maxsize=1000)
        
        # 统计信息
        self.stats = {
            'samples_received': 0,
            'bytes_received': 0,
            'last_sample_time': None,
            'errors': 0
        }
        
    def initialize_hardware(self):
        """初始化硬件连接"""
        if not ADI_AVAILABLE:
            self.logger.warning("ADI库不可用，使用模拟模式")
            return False
        
        try:
            # 尝试不同的设备URI
            uris = self.config.get('device_uris', [
                'ip:192.168.1.10',
                'ip:192.168.2.1',
                'usb:',
                'local:'
            ])
            
            for uri in uris:
                try:
                    self.logger.info(f"尝试连接设备: {uri}")
                    self.sdr = adi.adrv9009(uri=uri)
                    
                    # 配置设备参数
                    self.configure_device()
                    self.logger.info(f"成功连接到设备: {uri}")
                    return True
                    
                except Exception as e:
                    self.logger.debug(f"连接失败 {uri}: {e}")
                    continue
            
            self.logger.error("无法连接到任何设备")
            return False
            
        except Exception as e:
            self.logger.error(f"硬件初始化失败: {e}")
            return False
    
    def configure_device(self):
        """配置设备参数"""
        if not self.sdr:
            return
        
        # 配置采样率
        sample_rate = self.config.get('sample_rate', 1000000)
        self.sdr.sample_rate = sample_rate
        
        # 配置RX LO
        rx_lo = self.config.get('rx_lo', 1000000000)
        self.sdr.rx_lo = rx_lo
        
        # 配置TX LO
        tx_lo = self.config.get('tx_lo', 1000000000)
        self.sdr.tx_lo = tx_lo
        
        # 配置增益
        rx_gain = self.config.get('rx_gain', 30)
        self.sdr.gain_control_mode = "manual"
        self.sdr.rx_hardwaregain = rx_gain
        
        # 配置通道
        self.sdr.rx_enabled_channels = [0]  # 只启用通道0
        
        self.logger.info(f"设备配置完成: SR={sample_rate}, LO={rx_lo/1e6}MHz")
    
    def start_continuous_acquisition(self, duration_hours=24, 
                                    callback_on_complete=None):
        """启动连续采集"""
        if not self.initialize_hardware():
            self.logger.error("硬件初始化失败，无法开始采集")
            return False
        
        self.is_running = True
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        # 启动数据处理线程
        processing_thread = threading.Thread(
            target=self._process_data_queue,
            daemon=True
        )
        processing_thread.start()
        
        # 启动采集线程
        acquisition_thread = threading.Thread(
            target=self._continuous_acquisition,
            args=(end_time, callback_on_complete),
            daemon=True
        )
        acquisition_thread.start()
        
        self.logger.info(f"开始连续数据采集，持续时间: {duration_hours}小时")
        return True
    
    def _continuous_acquisition(self, end_time, callback):
        """连续采集线程"""
        batch_size = self.config.get('batch_size', 1024)
        
        try:
            while self.is_running and datetime.now() < end_time:
                try:
                    # 采集数据
                    samples = self._acquire_samples(batch_size)
                    
                    if samples is not None:
                        # 创建数据包
                        signal_data = SignalData(
                            timestamp=datetime.now(),
                            iq_data=samples,
                            sample_rate=self.sdr.sample_rate,
                            center_freq=self.sdr.rx_lo,
                            metadata={
                                'channel': 0,
                                'gain': self.sdr.rx_hardwaregain
                            }
                        )
                        
                        # 验证数据
                        if validate_signal_data(signal_data):
                            # 放入队列
                            self.data_queue.put(signal_data, timeout=1)
                            self._update_stats(len(samples), signal_data.size_bytes())
                        else:
                            self.logger.warning("数据验证失败")
                    
                    # 短暂休眠，防止过度占用CPU
                    time.sleep(0.001)
                    
                except queue.Full:
                    self.logger.warning("数据队列已满，丢弃数据")
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"采集错误: {e}")
                    self.stats['errors'] += 1
                    time.sleep(0.5)  # 错误后等待
            
            # 采集完成
            self.logger.info("数据采集完成")
            if callback:
                callback()
                
        except Exception as e:
            self.logger.error(f"采集线程异常: {e}", exc_info=True)
    
    def _acquire_samples(self, num_samples):
        """采集样本"""
        if not self.sdr and ADI_AVAILABLE:
            # 模拟模式
            t = np.linspace(0, 0.001, num_samples)
            signal = np.exp(2j * np.pi * 1000 * t)  # 1kHz 信号
            noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            return signal + noise
        
        try:
            return self.sdr.rx()
        except Exception as e:
            self.logger.error(f"采集样本失败: {e}")
            return None
    
    def _process_data_queue(self):
        """处理数据队列"""
        while self.is_running:
            try:
                # 从队列获取数据
                signal_data = self.data_queue.get(timeout=1)
                
                # 执行所有回调
                for callback in self.callbacks:
                    try:
                        callback(signal_data)
                    except Exception as e:
                        self.logger.error(f"回调执行失败: {e}")
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理队列错误: {e}")
    
    def _update_stats(self, num_samples, bytes_received):
        """更新统计信息"""
        self.stats['samples_received'] += num_samples
        self.stats['bytes_received'] += bytes_received
        self.stats['last_sample_time'] = datetime.now()
    
    def add_callback(self, callback: Callable):
        """添加数据处理回调"""
        self.callbacks.append(callback)
        self.logger.debug(f"添加回调: {callback.__name__}")
    
    def stop(self):
        """停止采集"""
        self.is_running = False
        self.logger.info("停止数据采集")
    
    def get_status(self) -> Dict[str, Any]:
        """获取采集器状态"""
        return {
            'running': self.is_running,
            'stats': self.stats,
            'queue_size': self.data_queue.qsize(),
            'callbacks_count': len(self.callbacks)
        }