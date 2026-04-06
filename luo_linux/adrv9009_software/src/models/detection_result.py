"""
检测结果模型模块
定义信号检测、识别、分类的结果数据模型
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
import warnings
import copy
from pathlib import Path


if TYPE_CHECKING:
    from .signal_data import SignalData


class DetectionType(str, Enum):
    """检测类型枚举"""
    # 信号检测
    SIGNAL_PRESENCE = "signal_presence"           # 信号存在
    SIGNAL_ABSENCE = "signal_absence"             # 信号消失
    SIGNAL_CHANGE = "signal_change"               # 信号变化
    
    # 调制检测
    MODULATION_DETECTION = "modulation_detection"  # 调制检测
    MODULATION_CLASSIFICATION = "modulation_classification"  # 调制分类
    
    # 参数估计
    FREQUENCY_ESTIMATION = "frequency_estimation"  # 频率估计
    BANDWIDTH_ESTIMATION = "bandwidth_estimation"  # 带宽估计
    POWER_ESTIMATION = "power_estimation"         # 功率估计
    SNR_ESTIMATION = "snr_estimation"             # 信噪比估计
    TIMING_ESTIMATION = "timing_estimation"       # 定时估计
    
    # 方向估计
    DIRECTION_FINDING = "direction_finding"       # 方向估计
    DOA_ESTIMATION = "doa_estimation"             # 波达方向估计
    
    # 信号识别
    SIGNAL_RECOGNITION = "signal_recognition"     # 信号识别
    SIGNAL_IDENTIFICATION = "signal_identification"  # 信号辨识
    
    # 异常检测
    ANOMALY_DETECTION = "anomaly_detection"       # 异常检测
    INTERFERENCE_DETECTION = "interference_detection"  # 干扰检测
    
    # 事件检测
    EVENT_DETECTION = "event_detection"           # 事件检测
    PATTERN_DETECTION = "pattern_detection"       # 模式检测
    
    # 未知类型
    UNKNOWN = "unknown"                           # 未知类型


class DetectionStatus(str, Enum):
    """检测状态枚举"""
    DETECTED = "detected"                         # 已检测
    CONFIRMED = "confirmed"                       # 已确认
    VERIFIED = "verified"                         # 已验证
    FALSE_ALARM = "false_alarm"                   # 虚警
    MISSED = "missed"                             # 漏检
    PENDING = "pending"                           # 待处理
    PROCESSING = "processing"                     # 处理中
    ERROR = "error"                               # 错误


class ConfidenceLevel(str, Enum):
    """置信度级别枚举"""
    VERY_HIGH = "very_high"                      # 非常高 (>90%)
    HIGH = "high"                                # 高 (80-90%)
    MEDIUM = "medium"                            # 中等 (70-80%)
    LOW = "low"                                  # 低 (60-70%)
    VERY_LOW = "very_low"                        # 非常低 (<60%)
    UNKNOWN = "unknown"                          # 未知


class AlertLevel(str, Enum):
    """告警级别枚举"""
    CRITICAL = "critical"                        # 严重
    HIGH = "high"                                # 高
    MEDIUM = "medium"                            # 中
    LOW = "low"                                  # 低
    INFO = "info"                                # 信息
    DEBUG = "debug"                              # 调试


class ProcessingMethod(str, Enum):
    """处理方法枚举"""
    ENERGY_DETECTION = "energy_detection"        # 能量检测
    MATCHED_FILTER = "matched_filter"            # 匹配滤波器
    CYCLOSTATIONARY = "cyclostationary"          # 循环平稳
    MACHINE_LEARNING = "machine_learning"        # 机器学习
    DEEP_LEARNING = "deep_learning"              # 深度学习
    STATISTICAL = "statistical"                  # 统计方法
    THRESHOLD_BASED = "threshold_based"          # 基于阈值
    FEATURE_BASED = "feature_based"             # 基于特征
    CORRELATION = "correlation"                  # 相关检测
    COHERENT = "coherent"                        # 相干检测


class VerificationMethod(str, Enum):
    """验证方法枚举"""
    MANUAL = "manual"                            # 人工验证
    AUTOMATIC = "automatic"                      # 自动验证
    CROSS_VALIDATION = "cross_validation"        # 交叉验证
    MULTIPLE_ALGORITHMS = "multiple_algorithms"  # 多算法验证
    GROUND_TRUTH = "ground_truth"                # 真实验证
    SIMULATION = "simulation"                    # 仿真验证
    UNVERIFIED = "unverified"                    # 未验证


@dataclass
class DetectionParameters:
    """检测参数"""
    
    # 算法参数
    algorithm_name: str = ""                     # 算法名称
    algorithm_version: str = ""                  # 算法版本
    parameters: Dict[str, Any] = field(default_factory=dict)  # 参数配置
    
    # 检测阈值
    detection_threshold: float = 0.0             # 检测阈值
    confidence_threshold: float = 0.0            # 置信度阈值
    false_alarm_rate: float = 0.0                # 虚警率
    
    # 处理参数
    window_size: int = 0                         # 窗口大小
    overlap: int = 0                             # 重叠大小
    fft_size: int = 0                            # FFT大小
    
    # 性能参数
    processing_time_ms: float = 0.0              # 处理时间(ms)
    memory_usage_mb: float = 0.0                 # 内存使用(MB)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DetectionMetrics:
    """检测指标"""
    
    # 检测性能
    probability_of_detection: float = 0.0        # 检测概率
    probability_of_false_alarm: float = 0.0      # 虚警概率
    probability_of_miss: float = 0.0             # 漏检概率
    receiver_operating_characteristic: Dict[str, float] = field(default_factory=dict)  # ROC曲线
    
    # 准确性指标
    accuracy: float = 0.0                        # 准确率
    precision: float = 0.0                       # 精确率
    recall: float = 0.0                          # 召回率
    f1_score: float = 0.0                        # F1分数
    
    # 置信度指标
    confidence_score: float = 0.0                # 置信度分数
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间
    uncertainty: float = 0.0                     # 不确定性
    
    # 质量指标
    signal_to_noise_ratio: float = 0.0           # 信噪比
    signal_to_interference_ratio: float = 0.0    # 信干比
    signal_to_clutter_ratio: float = 0.0         # 信杂比
    quality_factor: float = 0.0                  # 质量因子
    
    def calculate_from_detection(self, detection: 'DetectionResult') -> 'DetectionMetrics':
        """从检测结果计算指标"""
        # 计算置信度指标
        self.confidence_score = detection.confidence
        if detection.confidence_interval:
            self.confidence_interval = detection.confidence_interval
        
        # 计算质量指标
        if detection.snr_db is not None:
            self.signal_to_noise_ratio = detection.snr_db
        
        # 如果有验证信息，计算性能指标
        if detection.verification_method != VerificationMethod.UNVERIFIED:
            if detection.is_verified:
                self.accuracy = 1.0
                self.precision = 1.0
                self.recall = 1.0
                self.f1_score = 1.0
                self.probability_of_detection = 1.0
            else:
                self.probability_of_false_alarm = 1.0
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换元组为列表
        if isinstance(data['confidence_interval'], tuple):
            data['confidence_interval'] = list(data['confidence_interval'])
        return data


@dataclass
class LocationInfo:
    """位置信息"""
    
    # 地理坐标
    latitude: Optional[float] = None             # 纬度
    longitude: Optional[float] = None            # 经度
    altitude: Optional[float] = None             # 海拔
    
    # 位置精度
    accuracy_horizontal: Optional[float] = None  # 水平精度(m)
    accuracy_vertical: Optional[float] = None    # 垂直精度(m)
    
    # 坐标系
    coordinate_system: str = "WGS84"             # 坐标系
    projection: str = ""                         # 投影
    
    # 参考点
    reference_point: Optional[str] = None        # 参考点
    distance_to_reference: Optional[float] = None  # 到参考点距离
    
    def is_valid(self) -> bool:
        """检查位置是否有效"""
        return (self.latitude is not None and self.longitude is not None and
                -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180)
    
    def calculate_distance(self, other: 'LocationInfo') -> Optional[float]:
        """计算两点距离（使用Haversine公式）"""
        if not self.is_valid() or not other.is_valid():
            return None
        
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # 地球半径(m)
        
        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(other.latitude), radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DirectionInfo:
    """方向信息"""
    
    # 角度
    azimuth: Optional[float] = None              # 方位角(度)
    elevation: Optional[float] = None            # 俯仰角(度)
    
    # 角度精度
    azimuth_accuracy: Optional[float] = None     # 方位角精度(度)
    elevation_accuracy: Optional[float] = None   # 俯仰角精度(度)
    
    # 坐标系
    coordinate_system: str = "ENU"               # 坐标系(东-北-天)
    
    # 到达方向
    angle_of_arrival: Optional[float] = None     # 到达角
    direction_of_arrival: Tuple[float, float] = (0.0, 0.0)  # 到达方向(方位, 俯仰)
    
    # 信号源位置估计
    estimated_location: Optional[LocationInfo] = None  # 估计位置
    
    def is_valid(self) -> bool:
        """检查方向是否有效"""
        return (self.azimuth is not None and
                0 <= self.azimuth <= 360)
    
    def to_cartesian(self) -> Optional[Tuple[float, float, float]]:
        """转换为笛卡尔坐标"""
        if not self.is_valid():
            return None
        
        from math import radians, sin, cos
        
        az = radians(self.azimuth)
        el = radians(self.elevation) if self.elevation is not None else 0
        
        x = cos(el) * sin(az)
        y = cos(el) * cos(az)
        z = sin(el)
        
        return (x, y, z)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.estimated_location:
            data['estimated_location'] = self.estimated_location.to_dict()
        return data


@dataclass
class SignalParameters:
    """信号参数"""
    
    # 频率参数
    center_frequency: float = 0.0                # 中心频率(Hz)
    bandwidth: float = 0.0                       # 带宽(Hz)
    start_frequency: float = 0.0                 # 起始频率(Hz)
    stop_frequency: float = 0.0                  # 终止频率(Hz)
    
    # 时间参数
    duration: float = 0.0                        # 持续时间(s)
    pulse_width: Optional[float] = None          # 脉冲宽度(s)
    pulse_repetition_interval: Optional[float] = None  # 脉冲重复间隔(s)
    
    # 功率参数
    peak_power_db: float = 0.0                   # 峰值功率(dB)
    average_power_db: float = 0.0                # 平均功率(dB)
    power_spectral_density: float = 0.0          # 功率谱密度(dB/Hz)
    
    # 调制参数
    modulation_type: str = ""                    # 调制类型
    modulation_rate: Optional[float] = None      # 调制速率(Hz)
    modulation_index: Optional[float] = None     # 调制指数
    deviation: Optional[float] = None            # 频偏/相偏
    
    # 编码参数
    coding_scheme: str = ""                      # 编码方案
    symbol_rate: Optional[float] = None          # 符号率
    chip_rate: Optional[float] = None            # 码片率
    
    def is_valid(self) -> bool:
        """检查参数是否有效"""
        return (self.center_frequency > 0 and 
                self.bandwidth >= 0 and 
                self.duration > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class VerificationInfo:
    """验证信息"""
    
    # 验证方法
    verification_method: VerificationMethod = VerificationMethod.UNVERIFIED
    verification_algorithm: str = ""             # 验证算法
    
    # 验证结果
    is_verified: bool = False                    # 是否已验证
    verification_score: float = 0.0              # 验证分数
    verification_confidence: float = 0.0         # 验证置信度
    
    # 验证者信息
    verified_by: str = ""                        # 验证者
    verified_at: Optional[datetime] = None       # 验证时间
    
    # 验证依据
    verification_evidence: List[str] = field(default_factory=list)  # 验证依据
    verification_notes: str = ""                 # 验证说明
    
    def verify(self, method: VerificationMethod, algorithm: str = "",
               score: float = 1.0, confidence: float = 1.0,
               verifier: str = "system", evidence: List[str] = None,
               notes: str = ""):
        """验证检测结果"""
        self.verification_method = method
        self.verification_algorithm = algorithm
        self.verification_score = score
        self.verification_confidence = confidence
        self.is_verified = score >= 0.5
        self.verified_by = verifier
        self.verified_at = datetime.now()
        
        if evidence:
            self.verification_evidence = evidence
        self.verification_notes = notes
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.verified_at:
            data['verified_at'] = self.verified_at.isoformat()
        return data


@dataclass
class AlertInfo:
    """告警信息"""
    
    # 告警级别
    alert_level: AlertLevel = AlertLevel.INFO
    alert_severity: int = 0                      # 告警严重度(0-10)
    
    # 告警内容
    alert_message: str = ""                      # 告警消息
    alert_description: str = ""                  # 告警描述
    
    # 告警状态
    is_acknowledged: bool = False                # 是否已确认
    acknowledged_by: str = ""                    # 确认者
    acknowledged_at: Optional[datetime] = None   # 确认时间
    
    is_resolved: bool = False                    # 是否已解决
    resolved_by: str = ""                        # 解决者
    resolved_at: Optional[datetime] = None       # 解决时间
    
    # 告警动作
    required_actions: List[str] = field(default_factory=list)  # 需要采取的动作
    taken_actions: List[str] = field(default_factory=list)     # 已采取的动作
    
    def acknowledge(self, acknowledged_by: str = "operator", notes: str = ""):
        """确认告警"""
        self.is_acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now()
        if notes:
            self.alert_description += f"\n确认说明: {notes}"
    
    def resolve(self, resolved_by: str = "operator", actions: List[str] = None, notes: str = ""):
        """解决告警"""
        self.is_resolved = True
        self.resolved_by = resolved_by
        self.resolved_at = datetime.now()
        
        if actions:
            self.taken_actions = actions
        if notes:
            self.alert_description += f"\n解决说明: {notes}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        for time_field in ['acknowledged_at', 'resolved_at']:
            if data[time_field]:
                data[time_field] = data[time_field].isoformat()
        return data


@dataclass
class RelatedDetection:
    """相关检测"""
    
    detection_id: str                           # 检测ID
    relation_type: str = "related"              # 关系类型
    similarity_score: float = 0.0               # 相似度分数
    time_difference_ms: float = 0.0             # 时间差(ms)
    frequency_difference_hz: float = 0.0        # 频率差(Hz)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class DetectionResult:
    """检测结果基类"""
    
    def __init__(
        self,
        # 标识信息
        id: Optional[str] = None,
        detection_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        
        # 检测类型
        detection_type: DetectionType = DetectionType.UNKNOWN,
        detection_subtype: str = "",
        
        # 源信号信息
        signal_id: str = "",                     # 源信号ID
        signal_timestamp: datetime = None,       # 信号时间戳
        signal_data: Optional['SignalData'] = None,  # 信号数据引用
        
        # 检测时间
        detection_time: datetime = None,         # 检测时间
        processing_time_ms: float = 0.0,         # 处理时间(ms)
        
        # 检测参数
        parameters: Optional[DetectionParameters] = None,
        
        # 信号参数
        signal_parameters: Optional[SignalParameters] = None,
        center_freq: float = 0.0,
        bandwidth: float = 0.0,
        duration_ms: float = 0.0,
        
        # 位置信息
        location: Optional[LocationInfo] = None,
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
        
        # 方向信息
        direction: Optional[DirectionInfo] = None,
        direction_angle: Optional[float] = None,
        direction_confidence: float = 0.0,
        
        # 功率信息
        power_db: Optional[float] = None,
        snr_db: Optional[float] = None,
        
        # 置信度
        confidence: float = 0.0,
        confidence_interval: Optional[Tuple[float, float]] = None,
        confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN,
        
        # 状态
        status: DetectionStatus = DetectionStatus.DETECTED,
        
        # 验证信息
        verification: Optional[VerificationInfo] = None,
        verification_method: VerificationMethod = VerificationMethod.UNVERIFIED,
        is_verified: bool = False,
        
        # 告警信息
        alert: Optional[AlertInfo] = None,
        alert_level: AlertLevel = AlertLevel.INFO,
        
        # 识别信息
        is_known: bool = False,                  # 是否已知信号
        known_signal_id: Optional[str] = None,   # 已知信号ID
        classification: str = "",                # 分类
        category: str = "",                      # 类别
        
        # 匹配信息
        match_score: float = 0.0,                # 匹配分数
        match_features: Optional[Dict[str, Any]] = None,  # 匹配特征
        match_algorithm: str = "",               # 匹配算法
        match_parameters: Optional[Dict[str, Any]] = None,  # 匹配参数
        
        # 特征
        features: Optional[Dict[str, Any]] = None,
        
        # 处理元数据
        algorithm_name: str = "",                # 算法名称
        algorithm_version: str = "",             # 算法版本
        processing_method: ProcessingMethod = ProcessingMethod.ENERGY_DETECTION,
        
        # 指标
        metrics: Optional[DetectionMetrics] = None,
        
        # 关联检测
        related_detections: List[RelatedDetection] = field(default_factory=list),
        parent_detection_id: Optional[str] = None,
        child_detection_ids: List[str] = field(default_factory=list),
        
        # 元数据
        metadata: Optional[Dict[str, Any]] = None,
        tags: List[str] = field(default_factory=list),
        
        # 存储信息
        storage_path: Optional[str] = None,
        created_at: datetime = None,
        updated_at: datetime = None,
    ):
        """初始化检测结果"""
        # 标识信息
        self.id = id or str(uuid.uuid4())
        self.detection_id = detection_id or self.id
        self.name = name
        self.description = description
        
        # 检测类型
        self.detection_type = detection_type
        self.detection_subtype = detection_subtype
        
        # 源信号信息
        self.signal_id = signal_id
        self.signal_timestamp = signal_timestamp
        self.signal_data = signal_data
        
        # 检测时间
        self.detection_time = detection_time or datetime.now()
        self.processing_time_ms = processing_time_ms
        
        # 检测参数
        if parameters is None:
            self.parameters = DetectionParameters()
        else:
            self.parameters = parameters
        
        # 信号参数
        if signal_parameters is None:
            self.signal_parameters = SignalParameters(
                center_frequency=center_freq,
                bandwidth=bandwidth,
                duration=duration_ms / 1000.0
            )
        else:
            self.signal_parameters = signal_parameters
        
        # 位置信息
        if location is None and (location_lat is not None or location_lon is not None):
            self.location = LocationInfo(
                latitude=location_lat,
                longitude=location_lon
            )
        else:
            self.location = location
        
        # 方向信息
        if direction is None and direction_angle is not None:
            self.direction = DirectionInfo(azimuth=direction_angle)
        else:
            self.direction = direction
        self.direction_confidence = direction_confidence
        
        # 功率信息
        self.power_db = power_db
        self.snr_db = snr_db
        
        # 置信度
        self.confidence = max(0.0, min(1.0, confidence))
        self.confidence_interval = confidence_interval
        self.confidence_level = self._calculate_confidence_level(confidence)
        
        # 状态
        self.status = status
        
        # 验证信息
        if verification is None:
            self.verification = VerificationInfo(
                verification_method=verification_method,
                is_verified=is_verified
            )
        else:
            self.verification = verification
        
        # 告警信息
        if alert is None and alert_level != AlertLevel.INFO:
            self.alert = AlertInfo(alert_level=alert_level)
        else:
            self.alert = alert
        
        # 识别信息
        self.is_known = is_known
        self.known_signal_id = known_signal_id
        self.classification = classification
        self.category = category
        
        # 匹配信息
        self.match_score = match_score
        self.match_features = match_features or {}
        self.match_algorithm = match_algorithm
        self.match_parameters = match_parameters or {}
        
        # 特征
        self.features = features or {}
        
        # 处理元数据
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self.processing_method = processing_method
        
        # 指标
        if metrics is None:
            self.metrics = DetectionMetrics()
            self.metrics.calculate_from_detection(self)
        else:
            self.metrics = metrics
        
        # 关联检测
        self.related_detections = related_detections
        self.parent_detection_id = parent_detection_id
        self.child_detection_ids = child_detection_ids
        
        # 元数据
        self.metadata = metadata or {}
        self.tags = tags
        
        # 存储信息
        self.storage_path = storage_path
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or self.created_at
    
    @property
    def center_freq(self) -> float:
        """获取中心频率"""
        return self.signal_parameters.center_frequency
    
    @center_freq.setter
    def center_freq(self, value: float):
        """设置中心频率"""
        self.signal_parameters.center_frequency = value
    
    @property
    def bandwidth(self) -> float:
        """获取带宽"""
        return self.signal_parameters.bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value: float):
        """设置带宽"""
        self.signal_parameters.bandwidth = value
    
    @property
    def duration_ms(self) -> float:
        """获取持续时间(ms)"""
        return self.signal_parameters.duration * 1000.0
    
    @duration_ms.setter
    def duration_ms(self, value: float):
        """设置持续时间(ms)"""
        self.signal_parameters.duration = value / 1000.0
    
    @property
    def location_lat(self) -> Optional[float]:
        """获取纬度"""
        if self.location:
            return self.location.latitude
        return None
    
    @location_lat.setter
    def location_lat(self, value: Optional[float]):
        """设置纬度"""
        if self.location is None:
            self.location = LocationInfo()
        self.location.latitude = value
    
    @property
    def location_lon(self) -> Optional[float]:
        """获取经度"""
        if self.location:
            return self.location.longitude
        return None
    
    @location_lon.setter
    def location_lon(self, value: Optional[float]):
        """设置经度"""
        if self.location is None:
            self.location = LocationInfo()
        self.location.longitude = value
    
    @property
    def direction_angle(self) -> Optional[float]:
        """获取方向角"""
        if self.direction:
            return self.direction.azimuth
        return None
    
    @direction_angle.setter
    def direction_angle(self, value: Optional[float]):
        """设置方向角"""
        if self.direction is None:
            self.direction = DirectionInfo()
        self.direction.azimuth = value
    
    @property
    def is_verified(self) -> bool:
        """获取是否已验证"""
        return self.verification.is_verified
    
    @is_verified.setter
    def is_verified(self, value: bool):
        """设置是否已验证"""
        self.verification.is_verified = value
    
    @property
    def verification_method(self) -> VerificationMethod:
        """获取验证方法"""
        return self.verification.verification_method
    
    @verification_method.setter
    def verification_method(self, value: VerificationMethod):
        """设置验证方法"""
        self.verification.verification_method = value
    
    @property
    def alert_level(self) -> AlertLevel:
        """获取告警级别"""
        if self.alert:
            return self.alert.alert_level
        return AlertLevel.INFO
    
    @alert_level.setter
    def alert_level(self, value: AlertLevel):
        """设置告警级别"""
        if self.alert is None and value != AlertLevel.INFO:
            self.alert = AlertInfo()
        if self.alert:
            self.alert.alert_level = value
    
    def _calculate_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """根据置信度分数计算置信度级别"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.6:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def update_confidence(self, confidence: float, interval: Tuple[float, float] = None):
        """更新置信度"""
        self.confidence = max(0.0, min(1.0, confidence))
        self.confidence_level = self._calculate_confidence_level(confidence)
        
        if interval:
            self.confidence_interval = interval
        
        # 更新指标
        self.metrics.confidence_score = self.confidence
        if self.confidence_interval:
            self.metrics.confidence_interval = self.confidence_interval
        
        self.updated_at = datetime.now()
    
    def verify(self, method: VerificationMethod = VerificationMethod.AUTOMATIC, 
              algorithm: str = "", score: float = 1.0, confidence: float = 1.0,
              verifier: str = "system", evidence: List[str] = None, notes: str = ""):
        """验证检测结果"""
        self.verification.verify(
            method=method,
            algorithm=algorithm,
            score=score,
            confidence=confidence,
            verifier=verifier,
            evidence=evidence,
            notes=notes
        )
        
        # 更新状态
        if self.verification.is_verified:
            self.status = DetectionStatus.VERIFIED
        else:
            self.status = DetectionStatus.FALSE_ALARM
        
        self.updated_at = datetime.now()
    
    def create_alert(self, level: AlertLevel, message: str, description: str = "",
                    severity: int = None, required_actions: List[str] = None):
        """创建告警"""
        if self.alert is None:
            self.alert = AlertInfo()
        
        self.alert.alert_level = level
        self.alert.alert_message = message
        self.alert.alert_description = description
        
        if severity is not None:
            self.alert.alert_severity = severity
        
        if required_actions:
            self.alert.required_actions = required_actions
        
        # 如果告警级别高，标记为待确认
        if level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
            self.status = DetectionStatus.PENDING
        
        self.updated_at = datetime.now()
    
    def add_related_detection(self, detection: 'DetectionResult', 
                            relation_type: str = "related", 
                            similarity_score: float = 0.0,
                            time_diff_ms: float = None,
                            freq_diff_hz: float = None):
        """添加相关检测"""
        if time_diff_ms is None and detection.detection_time:
            time_diff_ms = (self.detection_time - detection.detection_time).total_seconds() * 1000
        
        if freq_diff_hz is None:
            freq_diff_hz = abs(self.center_freq - detection.center_freq)
        
        related = RelatedDetection(
            detection_id=detection.id,
            relation_type=relation_type,
            similarity_score=similarity_score,
            time_difference_ms=time_diff_ms or 0.0,
            frequency_difference_hz=freq_diff_hz or 0.0
        )
        
        self.related_detections.append(related)
        self.updated_at = datetime.now()
    
    def calculate_similarity(self, other: 'DetectionResult', 
                           weights: Dict[str, float] = None) -> float:
        """计算与另一个检测的相似度"""
        if weights is None:
            weights = {
                'frequency': 0.4,
                'time': 0.3,
                'type': 0.2,
                'confidence': 0.1
            }
        
        similarity = 0.0
        
        # 频率相似度
        freq_similarity = 1.0 - min(1.0, abs(self.center_freq - other.center_freq) / max(self.bandwidth, other.bandwidth, 1e6))
        similarity += weights['frequency'] * freq_similarity
        
        # 时间相似度
        if self.detection_time and other.detection_time:
            time_diff = abs((self.detection_time - other.detection_time).total_seconds())
            max_duration = max(self.duration_ms, other.duration_ms) / 1000.0
            time_similarity = 1.0 - min(1.0, time_diff / max(max_duration, 1.0))
            similarity += weights['time'] * time_similarity
        
        # 类型相似度
        type_similarity = 1.0 if self.detection_type == other.detection_type else 0.5
        similarity += weights['type'] * type_similarity
        
        # 置信度相似度
        conf_similarity = 1.0 - abs(self.confidence - other.confidence)
        similarity += weights['confidence'] * conf_similarity
        
        return similarity
    
    def to_detection_event(self) -> Dict[str, Any]:
        """转换为检测事件字典"""
        event = {
            'event_id': self.id,
            'event_type': 'signal_detection',
            'detection_type': self.detection_type.value,
            'detection_subtype': self.detection_subtype,
            'timestamp': self.detection_time.isoformat(),
            'signal_id': self.signal_id,
            'frequency': self.center_freq,
            'bandwidth': self.bandwidth,
            'duration_ms': self.duration_ms,
            'power_db': self.power_db,
            'snr_db': self.snr_db,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'status': self.status.value,
            'is_known': self.is_known,
            'known_signal_id': self.known_signal_id,
            'classification': self.classification,
            'alert_level': self.alert_level.value if self.alert else 'info',
        }
        
        # 添加位置信息
        if self.location and self.location.is_valid():
            event.update({
                'latitude': self.location.latitude,
                'longitude': self.location.longitude,
                'accuracy': self.location.accuracy_horizontal,
            })
        
        # 添加方向信息
        if self.direction and self.direction.is_valid():
            event.update({
                'azimuth': self.direction.azimuth,
                'elevation': self.direction.elevation,
                'direction_confidence': self.direction_confidence,
            })
        
        return event
    
    def copy(self) -> 'DetectionResult':
        """创建副本"""
        return copy.deepcopy(self)
    
    def to_dict(self, include_signal_data: bool = False) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            'id': self.id,
            'detection_id': self.detection_id,
            'name': self.name,
            'description': self.description,
            'detection_type': self.detection_type.value,
            'detection_subtype': self.detection_subtype,
            'signal_id': self.signal_id,
            'signal_timestamp': self.signal_timestamp.isoformat() if self.signal_timestamp else None,
            'detection_time': self.detection_time.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'parameters': self.parameters.to_dict(),
            'signal_parameters': self.signal_parameters.to_dict(),
            'confidence': self.confidence,
            'confidence_interval': list(self.confidence_interval) if self.confidence_interval else None,
            'confidence_level': self.confidence_level.value,
            'status': self.status.value,
            'verification': self.verification.to_dict(),
            'is_known': self.is_known,
            'known_signal_id': self.known_signal_id,
            'classification': self.classification,
            'category': self.category,
            'match_score': self.match_score,
            'match_features': self.match_features,
            'match_algorithm': self.match_algorithm,
            'match_parameters': self.match_parameters,
            'features': self.features,
            'algorithm_name': self.algorithm_name,
            'algorithm_version': self.algorithm_version,
            'processing_method': self.processing_method.value,
            'metrics': self.metrics.to_dict(),
            'related_detections': [rd.to_dict() for rd in self.related_detections],
            'parent_detection_id': self.parent_detection_id,
            'child_detection_ids': self.child_detection_ids,
            'metadata': self.metadata,
            'tags': self.tags,
            'storage_path': self.storage_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
        
        # 添加位置信息
        if self.location:
            data['location'] = self.location.to_dict()
        
        # 添加方向信息
        if self.direction:
            data['direction'] = self.direction.to_dict()
            data['direction_confidence'] = self.direction_confidence
        
        # 添加功率信息
        if self.power_db is not None:
            data['power_db'] = self.power_db
        if self.snr_db is not None:
            data['snr_db'] = self.snr_db
        
        # 添加告警信息
        if self.alert:
            data['alert'] = self.alert.to_dict()
        
        # 添加信号数据
        if include_signal_data and self.signal_data:
            data['signal_data'] = self.signal_data.to_dict(include_data=True)
        
        return data
    
    def to_json(self, include_signal_data: bool = False, indent: int = 2) -> str:
        """转换为JSON字符串"""
        data = self.to_dict(include_signal_data=include_signal_data)
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    def save(self, filepath: str, include_signal_data: bool = False, compress: bool = False):
        """保存到文件"""
        data = self.to_dict(include_signal_data=include_signal_data)
        
        if compress:
            data_str = json.dumps(data, ensure_ascii=False, default=str)
            compressed = zlib.compress(data_str.encode('utf-8'), level=6)
            with open(filepath, 'wb') as f:
                f.write(compressed)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """从字典创建"""
        # 解析枚举
        detection_type = DetectionType(data.get('detection_type', 'unknown'))
        confidence_level = ConfidenceLevel(data.get('confidence_level', 'unknown'))
        status = DetectionStatus(data.get('status', 'detected'))
        processing_method = ProcessingMethod(data.get('processing_method', 'energy_detection'))
        
        # 解析时间
        detection_time = None
        if 'detection_time' in data and data['detection_time']:
            detection_time = datetime.fromisoformat(data['detection_time'].replace('Z', '+00:00'))
        
        signal_timestamp = None
        if 'signal_timestamp' in data and data['signal_timestamp']:
            signal_timestamp = datetime.fromisoformat(data['signal_timestamp'].replace('Z', '+00:00'))
        
        created_at = None
        if 'created_at' in data and data['created_at']:
            created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        
        updated_at = None
        if 'updated_at' in data and data['updated_at']:
            updated_at = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        
        # 解析嵌套对象
        parameters = DetectionParameters(**data.get('parameters', {}))
        signal_parameters = SignalParameters(**data.get('signal_parameters', {}))
        verification = VerificationInfo(**data.get('verification', {}))
        metrics = DetectionMetrics(**data.get('metrics', {}))
        
        # 解析可选对象
        location = None
        if 'location' in data and data['location']:
            location = LocationInfo(**data['location'])
        
        direction = None
        if 'direction' in data and data['direction']:
            direction = DirectionInfo(**data['direction'])
        
        alert = None
        if 'alert' in data and data['alert']:
            alert_data = data['alert']
            # 解析时间字段
            for time_field in ['acknowledged_at', 'resolved_at']:
                if time_field in alert_data and alert_data[time_field]:
                    alert_data[time_field] = datetime.fromisoformat(alert_data[time_field].replace('Z', '+00:00'))
            alert = AlertInfo(**alert_data)
        
        # 解析相关检测
        related_detections = []
        if 'related_detections' in data:
            for rd_data in data['related_detections']:
                related_detections.append(RelatedDetection(**rd_data))
        
        # 解析置信区间
        confidence_interval = None
        if 'confidence_interval' in data and data['confidence_interval']:
            confidence_interval = tuple(data['confidence_interval'])
        
        # 创建检测结果
        detection = cls(
            id=data.get('id'),
            detection_id=data.get('detection_id'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            detection_type=detection_type,
            detection_subtype=data.get('detection_subtype', ''),
            signal_id=data.get('signal_id', ''),
            signal_timestamp=signal_timestamp,
            detection_time=detection_time,
            processing_time_ms=data.get('processing_time_ms', 0.0),
            parameters=parameters,
            signal_parameters=signal_parameters,
            location=location,
            direction=direction,
            direction_confidence=data.get('direction_confidence', 0.0),
            power_db=data.get('power_db'),
            snr_db=data.get('snr_db'),
            confidence=data.get('confidence', 0.0),
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            status=status,
            verification=verification,
            alert=alert,
            is_known=data.get('is_known', False),
            known_signal_id=data.get('known_signal_id'),
            classification=data.get('classification', ''),
            category=data.get('category', ''),
            match_score=data.get('match_score', 0.0),
            match_features=data.get('match_features', {}),
            match_algorithm=data.get('match_algorithm', ''),
            match_parameters=data.get('match_parameters', {}),
            features=data.get('features', {}),
            algorithm_name=data.get('algorithm_name', ''),
            algorithm_version=data.get('algorithm_version', ''),
            processing_method=processing_method,
            metrics=metrics,
            related_detections=related_detections,
            parent_detection_id=data.get('parent_detection_id'),
            child_detection_ids=data.get('child_detection_ids', []),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            storage_path=data.get('storage_path'),
            created_at=created_at,
            updated_at=updated_at,
        )
        
        return detection
    
    @classmethod
    def from_file(cls, filepath: str) -> 'DetectionResult':
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
        return (f"DetectionResult(id={self.id}, type={self.detection_type.value}, "
                f"confidence={self.confidence:.2f}, freq={self.center_freq/1e6:.1f}MHz, "
                f"status={self.status.value})")


# 工厂函数
def create_detection_from_signal(
    signal_data: 'SignalData',
    detection_type: DetectionType = DetectionType.SIGNAL_PRESENCE,
    confidence: float = 0.8,
    algorithm_name: str = "energy_detector",
    **kwargs
) -> DetectionResult:
    """从信号数据创建检测结果"""
    return DetectionResult(
        signal_id=signal_data.id,
        signal_timestamp=signal_data.timestamp,
        signal_data=signal_data,
        detection_type=detection_type,
        signal_parameters=SignalParameters(
            center_frequency=signal_data.center_freq,
            bandwidth=signal_data.bandwidth,
            duration=signal_data.duration
        ),
        confidence=confidence,
        algorithm_name=algorithm_name,
        **kwargs
    )


def create_frequency_detection(
    signal_data: 'SignalData',
    center_freq: float,
    bandwidth: float,
    power_db: float,
    confidence: float = 0.9,
    **kwargs
) -> DetectionResult:
    """创建频率检测结果"""
    detection = create_detection_from_signal(
        signal_data=signal_data,
        detection_type=DetectionType.FREQUENCY_ESTIMATION,
        confidence=confidence,
        algorithm_name="frequency_estimator",
        **kwargs
    )
    
    detection.center_freq = center_freq
    detection.bandwidth = bandwidth
    detection.power_db = power_db
    detection.name = f"频率检测@{center_freq/1e6:.1f}MHz"
    detection.description = f"检测到信号在{center_freq/1e6:.1f}MHz ±{bandwidth/1e6/2:.1f}MHz"
    
    return detection


def create_direction_detection(
    signal_data: 'SignalData',
    azimuth: float,
    elevation: float = None,
    confidence: float = 0.85,
    **kwargs
) -> DetectionResult:
    """创建方向检测结果"""
    detection = create_detection_from_signal(
        signal_data=signal_data,
        detection_type=DetectionType.DIRECTION_FINDING,
        confidence=confidence,
        algorithm_name="direction_finder",
        **kwargs
    )
    
    detection.direction = DirectionInfo(
        azimuth=azimuth,
        elevation=elevation
    )
    detection.direction_confidence = confidence
    detection.name = f"方向检测@{azimuth:.1f}°"
    detection.description = f"信号来自方位角{azimuth:.1f}°"
    if elevation is not None:
        detection.description += f", 俯仰角{elevation:.1f}°"
    
    return detection


def create_modulation_detection(
    signal_data: 'SignalData',
    modulation_type: str,
    confidence: float = 0.8,
    modulation_params: Dict[str, Any] = None,
    **kwargs
) -> DetectionResult:
    """创建调制检测结果"""
    detection = create_detection_from_signal(
        signal_data=signal_data,
        detection_type=DetectionType.MODULATION_CLASSIFICATION,
        confidence=confidence,
        algorithm_name="modulation_classifier",
        **kwargs
    )
    
    detection.classification = modulation_type
    detection.name = f"调制检测:{modulation_type}"
    detection.description = f"检测到{modulation_type}调制信号"
    
    if modulation_params:
        detection.features.update(modulation_params)
    
    return detection


def create_signal_recognition(
    signal_data: 'SignalData',
    known_signal_id: str,
    match_score: float,
    classification: str = "",
    confidence: float = 0.9,
    **kwargs
) -> DetectionResult:
    """创建信号识别结果"""
    detection = create_detection_from_signal(
        signal_data=signal_data,
        detection_type=DetectionType.SIGNAL_RECOGNITION,
        confidence=confidence,
        algorithm_name="signal_recognizer",
        **kwargs
    )
    
    detection.is_known = True
    detection.known_signal_id = known_signal_id
    detection.match_score = match_score
    detection.classification = classification
    detection.name = f"信号识别:{known_signal_id}"
    detection.description = f"识别为已知信号: {known_signal_id}, 匹配度: {match_score:.2f}"
    
    return detection


def create_anomaly_detection(
    signal_data: 'SignalData',
    anomaly_type: str,
    anomaly_score: float,
    confidence: float = 0.7,
    alert_level: AlertLevel = AlertLevel.MEDIUM,
    **kwargs
) -> DetectionResult:
    """创建异常检测结果"""
    detection = create_detection_from_signal(
        signal_data=signal_data,
        detection_type=DetectionType.ANOMALY_DETECTION,
        confidence=confidence,
        algorithm_name="anomaly_detector",
        **kwargs
    )
    
    detection.detection_subtype = anomaly_type
    detection.features['anomaly_score'] = anomaly_score
    detection.name = f"异常检测:{anomaly_type}"
    detection.description = f"检测到{anomaly_type}异常, 异常分数: {anomaly_score:.2f}"
    
    # 创建告警
    detection.create_alert(
        level=alert_level,
        message=f"检测到{anomaly_type}异常",
        description=detection.description,
        severity=int(alert_level == AlertLevel.CRITICAL) * 10
    )
    
    return detection


# 使用示例
if __name__ == "__main__":
    from src.models.signal_data import create_synthetic_signal, SignalData
    
    # 创建合成信号
    synthetic_signal = create_synthetic_signal(
        signal_type='sine',
        frequency=100e6,
        sample_rate=10e6,
        duration=0.001
    )
    
    # 创建频率检测
    freq_detection = create_frequency_detection(
        signal_data=synthetic_signal,
        center_freq=100.1e6,
        bandwidth=1e6,
        power_db=-45.5,
        confidence=0.92
    )
    
    print(f"频率检测: {freq_detection}")
    print(f"中心频率: {freq_detection.center_freq/1e6:.2f}MHz")
    print(f"置信度: {freq_detection.confidence:.2f}")
    
    # 创建方向检测
    dir_detection = create_direction_detection(
        signal_data=synthetic_signal,
        azimuth=45.5,
        elevation=10.2,
        confidence=0.88
    )
    
    print(f"\n方向检测: {dir_detection}")
    print(f"方位角: {dir_detection.direction_angle:.1f}°")
    
    # 创建调制检测
    mod_detection = create_modulation_detection(
        signal_data=synthetic_signal,
        modulation_type="FM",
        confidence=0.85,
        modulation_params={"deviation": 75e3, "modulation_index": 2.5}
    )
    
    print(f"\n调制检测: {mod_detection}")
    print(f"调制类型: {mod_detection.classification}")
    
    # 验证检测结果
    freq_detection.verify(
        method=VerificationMethod.AUTOMATIC,
        algorithm="cross_validation",
        score=0.95,
        confidence=0.90,
        verifier="system",
        evidence=["频谱分析", "功率估计"],
        notes="通过交叉验证"
    )
    
    print(f"\n验证状态: {freq_detection.is_verified}")
    print(f"验证分数: {freq_detection.verification.verification_score:.2f}")
    
    # 转换为字典
    detection_dict = freq_detection.to_dict(include_signal_data=True)
    print(f"\n字典大小: {len(str(detection_dict))} 字符")
    
    # 保存到文件
    freq_detection.save("test_detection.json", compress=False)
    print("保存到 test_detection.json")
    
    # 从文件加载
    loaded = DetectionResult.from_file("test_detection.json")
    print(f"加载检测: {loaded}")