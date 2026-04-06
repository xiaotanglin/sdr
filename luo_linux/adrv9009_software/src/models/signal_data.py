"""
信号数据模型模块
定义原始信号数据、处理结果、检测结果等数据模型
"""
import uuid
import json
import hashlib
import pickle
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, TYPE_CHECKING
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, validator, confloat, conint, root_validator
from scipy import signal as scipy_signal
import warnings
import struct
import zlib
import base64
import copy
from pathlib import Path


if TYPE_CHECKING:
    from .detection_result import DetectionResult


class SignalType(str, Enum):
    """信号类型枚举"""
    RAW = "raw"                 # 原始信号
    PROCESSED = "processed"     # 处理后的信号
    FEATURE = "feature"         # 特征信号
    SYNTHETIC = "synthetic"     # 合成信号
    NOISE = "noise"            # 噪声信号
    UNKNOWN = "unknown"         # 未知信号


class ModulationType(str, Enum):
    """调制类型枚举"""
    AM = "AM"                   # 幅度调制
    FM = "FM"                   # 频率调制
    PM = "PM"                   # 相位调制
    ASK = "ASK"                 # 幅移键控
    FSK = "FSK"                 # 频移键控
    PSK = "PSK"                 # 相移键控
    QAM = "QAM"                 # 正交幅度调制
    OFDM = "OFDM"              # 正交频分复用
    BPSK = "BPSK"               # 二进制相移键控
    QPSK = "QPSK"               # 四相相移键控
    GFSK = "GFSK"               # 高斯频移键控
    UNKNOWN = "unknown"         # 未知调制


class SignalQuality(str, Enum):
    """信号质量枚举"""
    EXCELLENT = "excellent"     # 优秀
    GOOD = "good"              # 良好
    FAIR = "fair"              # 一般
    POOR = "poor"              # 差
    BAD = "bad"                # 很差


class DataFormat(str, Enum):
    """数据格式枚举"""
    COMPLEX64 = "complex64"     # 复数64位
    COMPLEX128 = "complex128"   # 复数128位
    FLOAT32 = "float32"         # 浮点32位
    FLOAT64 = "float64"         # 浮点64位
    INT16 = "int16"             # 整数16位
    INT32 = "int32"             # 整数32位
    UINT8 = "uint8"             # 无符号整数8位
    UINT16 = "uint16"           # 无符号整数16位


class CoordinateSystem(str, Enum):
    """坐标系枚举"""
    CARTESIAN = "cartesian"     # 笛卡尔坐标系
    POLAR = "polar"            # 极坐标系
    SPHERICAL = "spherical"     # 球坐标系


class ProcessingStage(str, Enum):
    """处理阶段枚举"""
    RAW_CAPTURE = "raw_capture"      # 原始采集
    PREPROCESSING = "preprocessing"  # 预处理
    FEATURE_EXTRACTION = "feature_extraction"  # 特征提取
    DETECTION = "detection"          # 检测
    CLASSIFICATION = "classification"  # 分类
    IDENTIFICATION = "identification"  # 识别
    POST_PROCESSING = "post_processing"  # 后处理


@dataclass
class FrequencyInfo:
    """频率信息"""
    
    center_frequency: float = 0.0           # 中心频率 (Hz)
    bandwidth: float = 0.0                  # 带宽 (Hz)
    start_frequency: float = 0.0            # 起始频率 (Hz)
    stop_frequency: float = 0.0             # 终止频率 (Hz)
    sample_rate: float = 0.0                # 采样率 (Hz)
    nyquist_zone: int = 1                   # 奈奎斯特区
    
    def __post_init__(self):
        """初始化后处理"""
        if self.start_frequency == 0.0 and self.stop_frequency == 0.0:
            half_bandwidth = self.bandwidth / 2.0
            self.start_frequency = self.center_frequency - half_bandwidth
            self.stop_frequency = self.center_frequency + half_bandwidth
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """获取频率范围"""
        return (self.start_frequency, self.stop_frequency)
    
    def contains_frequency(self, frequency: float, tolerance: float = 0.0) -> bool:
        """检查是否包含指定频率"""
        return (self.start_frequency - tolerance <= frequency <= 
                self.stop_frequency + tolerance)
    
    def get_normalized_frequency(self, frequency: float) -> float:
        """获取归一化频率（相对于采样率）"""
        if self.sample_rate == 0:
            return 0.0
        return (frequency - self.center_frequency) / (self.sample_rate / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class TimeInfo:
    """时间信息"""
    
    timestamp: datetime                      # 时间戳
    duration: float = 0.0                    # 持续时间 (秒)
    sample_interval: float = 0.0             # 采样间隔 (秒)
    start_time: Optional[datetime] = None    # 开始时间
    end_time: Optional[datetime] = None      # 结束时间
    
    def __post_init__(self):
        """初始化后处理"""
        if self.start_time is None:
            self.start_time = self.timestamp
        
        if self.end_time is None and self.duration > 0:
            self.end_time = self.start_time + timedelta(seconds=self.duration)
        
        if self.sample_interval == 0 and self.duration > 0:
            # 估算采样间隔
            self.sample_interval = self.duration
    
    def get_time_vector(self, num_samples: int) -> np.ndarray:
        """获取时间向量"""
        if self.sample_interval > 0:
            return np.arange(num_samples) * self.sample_interval
        elif self.duration > 0 and num_samples > 0:
            return np.linspace(0, self.duration, num_samples)
        else:
            return np.arange(num_samples)
    
    def get_elapsed_time(self, reference_time: datetime = None) -> float:
        """获取经过的时间"""
        if reference_time is None:
            reference_time = datetime.now()
        return (reference_time - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换datetime为字符串
        for key in ['timestamp', 'start_time', 'end_time']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


@dataclass
class SignalMetrics:
    """信号指标"""
    
    # 功率指标
    power_db: float = 0.0                    # 功率 (dB)
    power_db_per_hz: float = 0.0             # 功率谱密度 (dB/Hz)
    peak_power_db: float = 0.0               # 峰值功率 (dB)
    average_power_db: float = 0.0            # 平均功率 (dB)
    
    # 信噪比
    snr_db: float = 0.0                      # 信噪比 (dB)
    snr_linear: float = 0.0                  # 信噪比 (线性)
    
    # 动态范围
    dynamic_range_db: float = 0.0            # 动态范围 (dB)
    
    # 噪声指标
    noise_floor_db: float = 0.0              # 噪声基底 (dB)
    noise_power_db: float = 0.0              # 噪声功率 (dB)
    
    # 质量指标
    evm_percent: float = 0.0                 # 误差矢量幅度 (%)
    mer_db: float = 0.0                      # 调制误差率 (dB)
    constellation_quality: float = 0.0        # 星座图质量
    
    # 频谱指标
    spectral_flatness: float = 0.0           # 频谱平坦度
    spectral_centroid: float = 0.0           # 频谱质心
    spectral_spread: float = 0.0             # 频谱扩展
    spectral_skewness: float = 0.0           # 频谱偏度
    spectral_kurtosis: float = 0.0           # 频谱峰度
    
    # 统计指标
    mean: complex = 0+0j                     # 均值
    variance: float = 0.0                    # 方差
    skewness: float = 0.0                    # 偏度
    kurtosis: float = 0.0                    # 峰度
    
    def calculate_from_signal(self, signal_data: 'SignalData') -> 'SignalMetrics':
        """从信号数据计算指标"""
        if signal_data.iq_data is None or len(signal_data.iq_data) == 0:
            return self
        
        iq_data = signal_data.iq_data
        
        # 计算功率
        power = np.mean(np.abs(iq_data) ** 2)
        self.power_db = 10 * np.log10(power) if power > 0 else -np.inf
        self.average_power_db = self.power_db
        
        # 计算峰值功率
        peak_power = np.max(np.abs(iq_data) ** 2)
        self.peak_power_db = 10 * np.log10(peak_power) if peak_power > 0 else -np.inf
        
        # 计算统计指标
        self.mean = np.mean(iq_data)
        
        # 计算SNR（简化估计）
        if hasattr(signal_data, 'noise_floor_db'):
            self.noise_floor_db = signal_data.noise_floor_db
            self.snr_db = self.power_db - self.noise_floor_db
            self.snr_linear = 10 ** (self.snr_db / 10)
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换复数为字典
        if isinstance(data['mean'], complex):
            data['mean'] = {'real': data['mean'].real, 'imag': data['mean'].imag}
        return data


@dataclass
class SignalFeatures:
    """信号特征"""
    
    # 时域特征
    time_features: Dict[str, float] = field(default_factory=dict)
    
    # 频域特征
    frequency_features: Dict[str, float] = field(default_factory=dict)
    
    # 时频特征
    time_frequency_features: Dict[str, float] = field(default_factory=dict)
    
    # 统计特征
    statistical_features: Dict[str, float] = field(default_factory=dict)
    
    # 调制特征
    modulation_features: Dict[str, float] = field(default_factory=dict)
    
    # 自定义特征
    custom_features: Dict[str, Any] = field(default_factory=dict)
    
    def add_feature(self, category: str, name: str, value: Any):
        """添加特征"""
        if category == "time":
            self.time_features[name] = value
        elif category == "frequency":
            self.frequency_features[name] = value
        elif category == "time_frequency":
            self.time_frequency_features[name] = value
        elif category == "statistical":
            self.statistical_features[name] = value
        elif category == "modulation":
            self.modulation_features[name] = value
        elif category == "custom":
            self.custom_features[name] = value
        else:
            raise ValueError(f"不支持的分类: {category}")
    
    def get_feature(self, category: str, name: str, default: Any = None) -> Any:
        """获取特征"""
        if category == "time":
            return self.time_features.get(name, default)
        elif category == "frequency":
            return self.frequency_features.get(name, default)
        elif category == "time_frequency":
            return self.time_frequency_features.get(name, default)
        elif category == "statistical":
            return self.statistical_features.get(name, default)
        elif category == "modulation":
            return self.modulation_features.get(name, default)
        elif category == "custom":
            return self.custom_features.get(name, default)
        else:
            return default
    
    def to_vector(self, feature_names: List[str] = None) -> np.ndarray:
        """转换为特征向量"""
        all_features = {}
        all_features.update(self.time_features)
        all_features.update(self.frequency_features)
        all_features.update(self.time_frequency_features)
        all_features.update(self.statistical_features)
        all_features.update(self.modulation_features)
        
        # 处理自定义特征
        for key, value in self.custom_features.items():
            if isinstance(value, (int, float)):
                all_features[f"custom_{key}"] = value
        
        if feature_names:
            # 只提取指定的特征
            vector = []
            for name in feature_names:
                vector.append(all_features.get(name, 0.0))
        else:
            # 提取所有特征
            vector = list(all_features.values())
        
        return np.array(vector, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ProcessingMetadata:
    """处理元数据"""
    
    # 处理链
    processing_chain: List[ProcessingStage] = field(default_factory=list)
    
    # 算法信息
    algorithms_used: List[str] = field(default_factory=list)
    
    # 参数配置
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 性能指标
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    # 质量指标
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 错误信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_processing_step(self, stage: ProcessingStage, algorithm: str = None, 
                          processing_time: float = None, parameters: Dict = None):
        """添加处理步骤"""
        self.processing_chain.append(stage)
        
        if algorithm:
            self.algorithms_used.append(algorithm)
        
        if processing_time is not None:
            key = f"{stage.value}_{len(self.processing_chain)}"
            if algorithm:
                key = f"{algorithm}_{key}"
            self.processing_times[key] = processing_time
        
        if parameters:
            self.parameters.update(parameters)
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """添加警告信息"""
        self.warnings.append(warning)
    
    def get_total_processing_time(self) -> float:
        """获取总处理时间"""
        return sum(self.processing_times.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DeviceInfo:
    """设备信息"""
    
    device_id: str = ""                      # 设备ID
    device_type: str = ""                    # 设备类型
    manufacturer: str = ""                   # 制造商
    model: str = ""                          # 型号
    serial_number: str = ""                  # 序列号
    firmware_version: str = ""               # 固件版本
    driver_version: str = ""                 # 驱动版本
    
    # 硬件配置
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    
    # 校准信息
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    calibration_date: Optional[datetime] = None
    calibration_valid_until: Optional[datetime] = None
    
    # 位置信息
    location_lat: Optional[float] = None     # 纬度
    location_lon: Optional[float] = None     # 经度
    location_alt: Optional[float] = None     # 海拔
    location_accuracy: Optional[float] = None  # 定位精度
    
    def is_calibrated(self) -> bool:
        """检查是否已校准"""
        if not self.calibration_date:
            return False
        
        if self.calibration_valid_until:
            return datetime.now() <= self.calibration_valid_until
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换datetime为字符串
        for key in ['calibration_date', 'calibration_valid_until']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


class SignalData:
    """信号数据基类"""
    
    def __init__(
        self,
        # 标识信息
        id: Optional[str] = None,
        name: str = "",
        description: str = "",
        signal_type: SignalType = SignalType.RAW,
        
        # 数据
        iq_data: Optional[np.ndarray] = None,
        data_format: DataFormat = DataFormat.COMPLEX64,
        
        # 频率信息
        frequency_info: Optional[FrequencyInfo] = None,
        center_freq: float = 0.0,
        sample_rate: float = 0.0,
        bandwidth: float = 0.0,
        
        # 时间信息
        time_info: Optional[TimeInfo] = None,
        timestamp: datetime = None,
        duration: float = 0.0,
        
        # 元数据
        metadata: Optional[Dict[str, Any]] = None,
        
        # 设备信息
        device_info: Optional[DeviceInfo] = None,
        device_id: str = "",
        
        # 通道信息
        channel_index: int = 0,
        num_channels: int = 1,
        
        # 处理信息
        processing_metadata: Optional[ProcessingMetadata] = None,
        parent_signal_id: Optional[str] = None,
        source_signal_ids: List[str] = field(default_factory=list),
        
        # 质量信息
        quality: SignalQuality = SignalQuality.UNKNOWN,
        noise_floor_db: float = -100.0,
        gain_db: float = 0.0,
        
        # 存储信息
        storage_path: Optional[str] = None,
        compressed: bool = False,
        encryption_key: Optional[str] = None,
    ):
        """初始化信号数据"""
        # 标识信息
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.signal_type = signal_type
        
        # 数据
        self.iq_data = iq_data
        self.data_format = data_format
        
        # 频率信息
        if frequency_info is None:
            self.frequency_info = FrequencyInfo(
                center_frequency=center_freq,
                bandwidth=bandwidth,
                sample_rate=sample_rate
            )
        else:
            self.frequency_info = frequency_info
        
        # 时间信息
        if time_info is None:
            if timestamp is None:
                timestamp = datetime.now()
            self.time_info = TimeInfo(
                timestamp=timestamp,
                duration=duration
            )
        else:
            self.time_info = time_info
        
        # 元数据
        self.metadata = metadata or {}
        
        # 设备信息
        if device_info is None:
            self.device_info = DeviceInfo(device_id=device_id)
        else:
            self.device_info = device_info
        
        # 通道信息
        self.channel_index = channel_index
        self.num_channels = num_channels
        
        # 处理信息
        if processing_metadata is None:
            self.processing_metadata = ProcessingMetadata()
        else:
            self.processing_metadata = processing_metadata
        
        self.parent_signal_id = parent_signal_id
        self.source_signal_ids = source_signal_ids
        
        # 质量信息
        self.quality = quality
        self.noise_floor_db = noise_floor_db
        self.gain_db = gain_db
        
        # 存储信息
        self.storage_path = storage_path
        self.compressed = compressed
        self.encryption_key = encryption_key
        
        # 计算指标
        self.metrics = SignalMetrics()
        if self.iq_data is not None:
            self.metrics.calculate_from_signal(self)
        
        # 特征
        self.features = SignalFeatures()
        
        # 缓存
        self._spectrum_cache = None
        self._spectrogram_cache = None
    
    @property
    def center_freq(self) -> float:
        """获取中心频率"""
        return self.frequency_info.center_frequency
    
    @center_freq.setter
    def center_freq(self, value: float):
        """设置中心频率"""
        self.frequency_info.center_frequency = value
    
    @property
    def sample_rate(self) -> float:
        """获取采样率"""
        return self.frequency_info.sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: float):
        """设置采样率"""
        self.frequency_info.sample_rate = value
    
    @property
    def bandwidth(self) -> float:
        """获取带宽"""
        return self.frequency_info.bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value: float):
        """设置带宽"""
        self.frequency_info.bandwidth = value
    
    @property
    def timestamp(self) -> datetime:
        """获取时间戳"""
        return self.time_info.timestamp
    
    @timestamp.setter
    def timestamp(self, value: datetime):
        """设置时间戳"""
        self.time_info.timestamp = value
    
    @property
    def duration(self) -> float:
        """获取持续时间"""
        return self.time_info.duration
    
    @duration.setter
    def duration(self, value: float):
        """设置持续时间"""
        self.time_info.duration = value
    
    @property
    def num_samples(self) -> int:
        """获取样本数"""
        if self.iq_data is None:
            return 0
        return len(self.iq_data)
    
    @property
    def sample_interval(self) -> float:
        """获取采样间隔"""
        if self.sample_rate > 0:
            return 1.0 / self.sample_rate
        return 0.0
    
    @property
    def time_vector(self) -> np.ndarray:
        """获取时间向量"""
        return self.time_info.get_time_vector(self.num_samples)
    
    @property
    def frequency_vector(self) -> np.ndarray:
        """获取频率向量"""
        if self.sample_rate == 0:
            return np.array([])
        
        n = self.num_samples
        if n == 0:
            return np.array([])
        
        freqs = np.fft.fftfreq(n, 1.0/self.sample_rate)
        freqs = np.fft.fftshift(freqs) + self.center_freq
        return freqs
    
    def size_bytes(self) -> int:
        """获取数据大小（字节）"""
        if self.iq_data is None:
            return 0
        
        if self.iq_data.dtype == np.complex64:
            item_size = 8  # 2 * 4 bytes
        elif self.iq_data.dtype == np.complex128:
            item_size = 16  # 2 * 8 bytes
        elif self.iq_data.dtype == np.float32:
            item_size = 4
        elif self.iq_data.dtype == np.float64:
            item_size = 8
        elif self.iq_data.dtype == np.int16:
            item_size = 2
        elif self.iq_data.dtype == np.int32:
            item_size = 4
        elif self.iq_data.dtype == np.uint8:
            item_size = 1
        elif self.iq_data.dtype == np.uint16:
            item_size = 2
        else:
            item_size = 8  # 默认
        
        return self.num_samples * item_size
    
    def calculate_spectrum(self, window: str = 'hann', nfft: int = None, 
                          db_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """计算频谱"""
        if self.iq_data is None or len(self.iq_data) == 0:
            return np.array([]), np.array([])
        
        if nfft is None:
            nfft = len(self.iq_data)
        
        # 应用窗函数
        if window == 'hann':
            window_func = np.hanning(len(self.iq_data))
        elif window == 'hamming':
            window_func = np.hamming(len(self.iq_data))
        elif window == 'blackman':
            window_func = np.blackman(len(self.iq_data))
        else:
            window_func = np.ones(len(self.iq_data))
        
        windowed_data = self.iq_data * window_func
        
        # 计算FFT
        spectrum = np.fft.fft(windowed_data, nfft)
        spectrum = np.fft.fftshift(spectrum)
        
        # 转换为dB
        if db_scale:
            spectrum = 20 * np.log10(np.abs(spectrum) + 1e-10)
        else:
            spectrum = np.abs(spectrum)
        
        # 频率向量
        freqs = self.frequency_vector
        
        # 缓存结果
        self._spectrum_cache = (freqs, spectrum)
        
        return freqs, spectrum
    
    def calculate_spectrogram(self, window_size: int = 1024, overlap: int = 512,
                            window: str = 'hann', nfft: int = None,
                            db_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算频谱图"""
        if self.iq_data is None or len(self.iq_data) < window_size:
            return np.array([]), np.array([]), np.array([])
        
        if nfft is None:
            nfft = window_size
        
        # 计算STFT
        f, t, Zxx = scipy_signal.stft(
            self.iq_data,
            fs=self.sample_rate,
            window=window,
            nperseg=window_size,
            noverlap=overlap,
            nfft=nfft,
            return_onesided=False
        )
        
        # 转换为dB
        if db_scale:
            Zxx = 20 * np.log10(np.abs(Zxx) + 1e-10)
        else:
            Zxx = np.abs(Zxx)
        
        # 调整频率轴
        f = np.fft.fftshift(f) + self.center_freq
        Zxx = np.fft.fftshift(Zxx, axes=0)
        
        # 时间轴
        t = t + (self.timestamp - datetime(1970, 1, 1)).total_seconds()
        
        # 缓存结果
        self._spectrogram_cache = (f, t, Zxx)
        
        return f, t, Zxx
    
    def extract_features(self, feature_types: List[str] = None) -> SignalFeatures:
        """提取特征"""
        if self.iq_data is None or len(self.iq_data) == 0:
            return self.features
        
        if feature_types is None:
            feature_types = ['time', 'frequency', 'statistical']
        
        iq_data = self.iq_data
        n = len(iq_data)
        
        # 时域特征
        if 'time' in feature_types:
            amplitude = np.abs(iq_data)
            phase = np.angle(iq_data)
            
            self.features.add_feature('time', 'amplitude_mean', np.mean(amplitude))
            self.features.add_feature('time', 'amplitude_std', np.std(amplitude))
            self.features.add_feature('time', 'amplitude_skewness', scipy_signal.skew(amplitude))
            self.features.add_feature('time', 'amplitude_kurtosis', scipy_signal.kurtosis(amplitude))
            self.features.add_feature('time', 'phase_mean', np.mean(phase))
            self.features.add_feature('time', 'phase_std', np.std(phase))
        
        # 频域特征
        if 'frequency' in feature_types:
            freqs, spectrum = self.calculate_spectrum(db_scale=False)
            if len(spectrum) > 0:
                self.features.add_feature('frequency', 'spectral_centroid', 
                                        np.sum(freqs * spectrum) / np.sum(spectrum))
                self.features.add_feature('frequency', 'spectral_spread', 
                                        np.sqrt(np.sum((freqs - self.features.get_feature('frequency', 'spectral_centroid'))**2 * spectrum) / 
                                               np.sum(spectrum)))
                self.features.add_feature('frequency', 'spectral_flatness', 
                                        scipy_signal.spectral_flatness(spectrum))
        
        # 统计特征
        if 'statistical' in feature_types:
            real_part = iq_data.real
            imag_part = iq_data.imag
            
            self.features.add_feature('statistical', 'real_mean', np.mean(real_part))
            self.features.add_feature('statistical', 'real_std', np.std(real_part))
            self.features.add_feature('statistical', 'imag_mean', np.mean(imag_part))
            self.features.add_feature('statistical', 'imag_std', np.std(imag_part))
            self.features.add_feature('statistical', 'iq_correlation', 
                                    np.corrcoef(real_part, imag_part)[0, 1])
        
        return self.features
    
    def resample(self, new_sample_rate: float, method: str = 'polyphase') -> 'SignalData':
        """重采样信号"""
        if self.iq_data is None:
            return self
        
        if new_sample_rate == self.sample_rate:
            return self.copy()
        
        # 计算重采样因子
        up = int(new_sample_rate)
        down = int(self.sample_rate)
        gcd = np.gcd(up, down)
        up //= gcd
        down //= gcd
        
        # 重采样
        if method == 'polyphase':
            from scipy import signal
            resampled = signal.resample_poly(self.iq_data, up, down)
        else:
            # 使用FFT重采样
            num_samples = int(len(self.iq_data) * new_sample_rate / self.sample_rate)
            resampled = scipy_signal.resample(self.iq_data, num_samples)
        
        # 创建新信号
        new_signal = self.copy()
        new_signal.iq_data = resampled
        new_signal.sample_rate = new_sample_rate
        new_signal.duration = len(resampled) / new_sample_rate
        
        # 更新处理元数据
        new_signal.processing_metadata.add_processing_step(
            ProcessingStage.PREPROCESSING,
            algorithm=f"resample_{method}",
            parameters={'old_rate': self.sample_rate, 'new_rate': new_sample_rate}
        )
        
        return new_signal
    
    def filter(self, filter_type: str = 'lowpass', cutoff_freq: float = None, 
              order: int = 4) -> 'SignalData':
        """滤波信号"""
        if self.iq_data is None or cutoff_freq is None:
            return self
        
        from scipy import signal
        
        # 计算归一化截止频率
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        # 设计滤波器
        if filter_type == 'lowpass':
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        elif filter_type == 'highpass':
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        elif filter_type == 'bandpass':
            if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                low, high = cutoff_freq
                normal_cutoff = [low/nyquist, high/nyquist]
                b, a = signal.butter(order, normal_cutoff, btype='band', analog=False)
            else:
                raise ValueError("带通滤波需要两个截止频率")
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        # 应用滤波器
        filtered = signal.filtfilt(b, a, self.iq_data)
        
        # 创建新信号
        new_signal = self.copy()
        new_signal.iq_data = filtered
        
        # 更新处理元数据
        new_signal.processing_metadata.add_processing_step(
            ProcessingStage.PREPROCESSING,
            algorithm=f"filter_{filter_type}",
            parameters={'cutoff_freq': cutoff_freq, 'order': order}
        )
        
        return new_signal
    
    def decimate(self, factor: int, method: str = 'iir') -> 'SignalData':
        """降采样信号"""
        if self.iq_data is None or factor <= 1:
            return self
        
        from scipy import signal
        
        if method == 'iir':
            decimated = signal.decimate(self.iq_data, factor, ftype='iir')
        else:  # fir
            decimated = signal.decimate(self.iq_data, factor, ftype='fir')
        
        # 创建新信号
        new_signal = self.copy()
        new_signal.iq_data = decimated
        new_signal.sample_rate = self.sample_rate / factor
        new_signal.duration = len(decimated) * factor / self.sample_rate
        
        # 更新处理元数据
        new_signal.processing_metadata.add_processing_step(
            ProcessingStage.PREPROCESSING,
            algorithm=f"decimate_{method}",
            parameters={'factor': factor}
        )
        
        return new_signal
    
    def normalize(self, method: str = 'amplitude', target_power: float = 0.0) -> 'SignalData':
        """归一化信号"""
        if self.iq_data is None:
            return self
        
        if method == 'amplitude':
            # 幅度归一化
            amplitude = np.abs(self.iq_data)
            max_amplitude = np.max(amplitude)
            if max_amplitude > 0:
                normalized = self.iq_data / max_amplitude
            else:
                normalized = self.iq_data
        elif method == 'power':
            # 功率归一化
            power = np.mean(np.abs(self.iq_data) ** 2)
            if power > 0:
                scale = np.sqrt(target_power / power) if target_power > 0 else 1.0 / np.sqrt(power)
                normalized = self.iq_data * scale
            else:
                normalized = self.iq_data
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 创建新信号
        new_signal = self.copy()
        new_signal.iq_data = normalized
        
        # 更新处理元数据
        new_signal.processing_metadata.add_processing_step(
            ProcessingStage.PREPROCESSING,
            algorithm=f"normalize_{method}",
            parameters={'target_power': target_power}
        )
        
        return new_signal
    
    def segment(self, segment_length: int, overlap: int = 0) -> List['SignalData']:
        """分割信号"""
        if self.iq_data is None or segment_length <= 0:
            return []
        
        n = len(self.iq_data)
        if segment_length >= n:
            return [self.copy()]
        
        step = segment_length - overlap
        segments = []
        
        for i in range(0, n - segment_length + 1, step):
            segment_data = self.iq_data[i:i + segment_length]
            
            # 创建新信号
            segment_signal = self.copy()
            segment_signal.iq_data = segment_data
            segment_signal.duration = len(segment_data) / self.sample_rate
            
            # 调整时间戳
            segment_offset = i / self.sample_rate
            segment_signal.timestamp = self.timestamp + timedelta(seconds=segment_offset)
            
            # 更新元数据
            segment_signal.name = f"{self.name}_segment_{i//step}"
            segment_signal.description = f"Segment {i//step} of {self.name}"
            segment_signal.metadata['segment_index'] = i // step
            segment_signal.metadata['segment_offset'] = segment_offset
            
            segments.append(segment_signal)
        
        return segments
    
    def copy(self) -> 'SignalData':
        """创建副本"""
        return copy.deepcopy(self)
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'signal_type': self.signal_type.value,
            'data_format': self.data_format.value,
            'frequency_info': self.frequency_info.to_dict(),
            'time_info': self.time_info.to_dict(),
            'metadata': self.metadata,
            'device_info': self.device_info.to_dict(),
            'channel_index': self.channel_index,
            'num_channels': self.num_channels,
            'processing_metadata': self.processing_metadata.to_dict(),
            'parent_signal_id': self.parent_signal_id,
            'source_signal_ids': self.source_signal_ids,
            'quality': self.quality.value,
            'noise_floor_db': self.noise_floor_db,
            'gain_db': self.gain_db,
            'metrics': self.metrics.to_dict(),
            'features': self.features.to_dict(),
            'storage_path': self.storage_path,
            'compressed': self.compressed,
            'num_samples': self.num_samples,
            'size_bytes': self.size_bytes(),
        }
        
        if include_data and self.iq_data is not None:
            # 序列化数据
            if self.compressed:
                data_buffer = zlib.compress(self.iq_data.tobytes(), level=6)
                data['iq_data'] = base64.b64encode(data_buffer).decode('ascii')
            else:
                data['iq_data'] = self.iq_data.tolist()
        
        return data
    
    def to_json(self, include_data: bool = False, indent: int = 2) -> str:
        """转换为JSON字符串"""
        data = self.to_dict(include_data=include_data)
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    def save(self, filepath: str, include_data: bool = True, compress: bool = False):
        """保存到文件"""
        data = self.to_dict(include_data=include_data)
        
        if compress:
            data_str = json.dumps(data, ensure_ascii=False, default=str)
            compressed = zlib.compress(data_str.encode('utf-8'), level=6)
            with open(filepath, 'wb') as f:
                f.write(compressed)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalData':
        """从字典创建"""
        # 解析数据
        iq_data = None
        if 'iq_data' in data:
            if isinstance(data['iq_data'], list):
                iq_data = np.array(data['iq_data'], dtype=np.complex64)
            elif isinstance(data['iq_data'], str):
                # 解码base64和压缩数据
                data_buffer = base64.b64decode(data['iq_data'])
                if data.get('compressed', False):
                    data_buffer = zlib.decompress(data_buffer)
                iq_data = np.frombuffer(data_buffer, dtype=np.complex64)
        
        # 解析枚举
        signal_type = SignalType(data.get('signal_type', 'raw'))
        data_format = DataFormat(data.get('data_format', 'complex64'))
        quality = SignalQuality(data.get('quality', 'unknown'))
        
        # 解析嵌套对象
        frequency_info = FrequencyInfo(**data.get('frequency_info', {}))
        
        time_info_data = data.get('time_info', {})
        if 'timestamp' in time_info_data and isinstance(time_info_data['timestamp'], str):
            time_info_data['timestamp'] = datetime.fromisoformat(time_info_data['timestamp'])
        time_info = TimeInfo(**time_info_data)
        
        device_info = DeviceInfo(**data.get('device_info', {}))
        processing_metadata = ProcessingMetadata(**data.get('processing_metadata', {}))
        
        # 创建信号
        signal = cls(
            id=data.get('id'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            signal_type=signal_type,
            iq_data=iq_data,
            data_format=data_format,
            frequency_info=frequency_info,
            time_info=time_info,
            metadata=data.get('metadata', {}),
            device_info=device_info,
            channel_index=data.get('channel_index', 0),
            num_channels=data.get('num_channels', 1),
            processing_metadata=processing_metadata,
            parent_signal_id=data.get('parent_signal_id'),
            source_signal_ids=data.get('source_signal_ids', []),
            quality=quality,
            noise_floor_db=data.get('noise_floor_db', -100.0),
            gain_db=data.get('gain_db', 0.0),
            storage_path=data.get('storage_path'),
            compressed=data.get('compressed', False),
        )
        
        # 恢复指标和特征
        if 'metrics' in data:
            signal.metrics = SignalMetrics(**data['metrics'])
        
        if 'features' in data:
            signal.features = SignalFeatures(**data['features'])
        
        return signal
    
    @classmethod
    def from_file(cls, filepath: str) -> 'SignalData':
        """从文件加载"""
        with open(filepath, 'rb') as f:
            # 尝试解压
            try:
                data = f.read()
                try:
                    data = zlib.decompress(data)
                    data = json.loads(data.decode('utf-8'))
                except zlib.error:
                    f.seek(0)
                    data = json.load(f)
            except Exception as e:
                raise ValueError(f"无法解析文件 {filepath}: {e}")
        
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"SignalData(id={self.id}, name={self.name}, "
                f"type={self.signal_type.value}, samples={self.num_samples}, "
                f"sample_rate={self.sample_rate:.1f}Hz, "
                f"center_freq={self.center_freq/1e6:.1f}MHz)")


# 工厂函数
def create_signal_from_iq(
    iq_data: np.ndarray,
    sample_rate: float,
    center_freq: float = 0.0,
    timestamp: datetime = None,
    name: str = "",
    description: str = "",
    **kwargs
) -> SignalData:
    """从IQ数据创建信号"""
    return SignalData(
        iq_data=iq_data,
        sample_rate=sample_rate,
        center_freq=center_freq,
        timestamp=timestamp or datetime.now(),
        name=name,
        description=description,
        **kwargs
    )


def create_synthetic_signal(
    signal_type: str = 'sine',
    frequency: float = 1e6,
    sample_rate: float = 10e6,
    duration: float = 0.001,
    amplitude: float = 1.0,
    phase: float = 0.0,
    noise_snr_db: float = None,
    **kwargs
) -> SignalData:
    """创建合成信号"""
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    
    if signal_type == 'sine':
        # 正弦波
        iq_data = amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))
    elif signal_type == 'chirp':
        # 线性调频
        f0 = frequency * 0.8
        f1 = frequency * 1.2
        iq_data = amplitude * np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    elif signal_type == 'noise':
        # 高斯噪声
        iq_data = amplitude * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    else:
        raise ValueError(f"不支持的信号类型: {signal_type}")
    
    # 添加噪声
    if noise_snr_db is not None:
        signal_power = np.mean(np.abs(iq_data) ** 2)
        noise_power = signal_power / (10 ** (noise_snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        iq_data = iq_data + noise
    
    return create_signal_from_iq(
        iq_data=iq_data,
        sample_rate=sample_rate,
        center_freq=frequency,
        name=f"synthetic_{signal_type}",
        description=f"Synthetic {signal_type} signal at {frequency/1e6:.1f}MHz",
        signal_type=SignalType.SYNTHETIC,
        **kwargs
    )


# 使用示例
if __name__ == "__main__":
    # 创建合成信号
    synthetic = create_synthetic_signal(
        signal_type='sine',
        frequency=100e6,
        sample_rate=10e6,
        duration=0.001,
        noise_snr_db=20
    )
    
    print(f"创建信号: {synthetic}")
    print(f"样本数: {synthetic.num_samples}")
    print(f"持续时间: {synthetic.duration*1000:.1f}ms")
    print(f"中心频率: {synthetic.center_freq/1e6:.1f}MHz")
    print(f"采样率: {synthetic.sample_rate/1e6:.1f}MHz")
    
    # 计算频谱
    freqs, spectrum = synthetic.calculate_spectrum()
    print(f"频谱长度: {len(spectrum)}")
    
    # 提取特征
    features = synthetic.extract_features()
    print(f"提取特征数: {len(features.to_vector())}")
    
    # 转换为字典
    data_dict = synthetic.to_dict(include_data=True)
    print(f"字典大小: {len(str(data_dict))} 字符")
    
    # 保存到文件
    synthetic.save("test_signal.json", compress=False)
    print("保存到 test_signal.json")
    
    # 从文件加载
    loaded = SignalData.from_file("test_signal.json")
    print(f"加载信号: {loaded}")
    
    # 测试处理操作
    filtered = synthetic.filter('lowpass', cutoff_freq=1e6)
    print(f"滤波后信号: {filtered}")
    
    resampled = synthetic.resample(5e6)
    print(f"重采样后: {resampled}")
    
    normalized = synthetic.normalize('amplitude')
    print(f"归一化后: {normalized}")