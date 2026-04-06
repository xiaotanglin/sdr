"""
测试配置文件
提供测试所需的fixture、配置、工具函数等
"""
import os
import sys
import json
import tempfile
import asyncio
import shutil
import time
import uuid
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Generator, AsyncGenerator
from datetime import datetime, timedelta
import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.app_config import AppConfig, get_current_config
from config.web_config import WebConfig, get_current_config as get_web_config
from config.db_config import DatabaseConfig, get_current_config as get_db_config
from config.signal_config import SignalConfig, get_current_config as get_signal_config
from config.logging_config import LoggingConfig, setup_logging
from config.alert_config import AlertConfig, get_current_config as get_alert_config

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq, SignalType, SignalMetrics
from src.models.detection_result import DetectionResult, create_detection_from_signal, DetectionType, ConfidenceLevel
from src.models.alert_models import AlertLevel, AlertStatus, AlertType, NotificationMethod

# 导入数据库模型
from src.models.database_models import (
    Base, 
    RawSignal, 
    Detection, 
    KnownSignal,
    DetectionMatch,
    ProcessedSignal,
    AlgorithmResult,
    SystemEvent,
    AlertRule,
    AlertHistory,
    AlertRecipient,
    AlertTemplate,
    SystemStatus
)

# 导入核心组件
from src.core.database_manager import DatabaseManager
from src.core.signal_processor import SignalProcessor
from src.core.detection_engine import DetectionEngine
from src.core.algorithm_manager import AlgorithmManager
from src.core.alert_manager import AlertManager
from src.core.system_monitor import SystemMonitor
from src.core.data_acquisition import DataAcquisition
from src.core.event_manager import EventManager
from src.core.scheduler import Scheduler

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult, ValidationLevel
from src.utils.file_utils import ensure_directory, delete_file, get_file_info
from src.utils.signal_utils import SignalUtils
from src.utils.performance_monitor import PerformanceMonitor

# 导入Web组件
from src.web.app import create_web_app, WebApp
from src.web.api import create_api_router
from src.web.websocket_manager import WebSocketManager
from src.web.auth import create_access_token, verify_access_token, get_current_user

# 导入测试数据生成器
from src.tests.test_data_generator import TestDataGenerator


# 测试配置
class TestConfig:
    """测试配置类"""
    
    # 基本配置
    TEST_ENVIRONMENT = "testing"
    TEST_DB_TYPE = "sqlite"
    TEST_DB_NAME = ":memory:"  # 使用内存数据库
    
    # 测试数据
    TEST_DATA_DIR = Path(tempfile.gettempdir()) / "adrv9009_test_data"
    TEST_LOG_DIR = TEST_DATA_DIR / "logs"
    TEST_TEMP_DIR = TEST_DATA_DIR / "temp"
    TEST_BACKUP_DIR = TEST_DATA_DIR / "backup"
    TEST_UPLOAD_DIR = TEST_DATA_DIR / "upload"
    
    # 信号测试参数
    TEST_SIGNAL_SAMPLES = 1000
    TEST_SAMPLE_RATE = 10e6
    TEST_CENTER_FREQ = 100e6
    TEST_BANDWIDTH = 1e6
    
    # 测试用户
    TEST_USERNAME = "test_user"
    TEST_PASSWORD = "test_password"
    TEST_EMAIL = "test@example.com"
    
    # Web测试配置
    TEST_WEB_HOST = "127.0.0.1"
    TEST_WEB_PORT = 8888
    TEST_API_PREFIX = "/api/v1"
    
    # 测试超时
    TEST_TIMEOUT = 30
    TEST_ASYNC_TIMEOUT = 10
    
    @classmethod
    def setup_test_environment(cls):
        """设置测试环境"""
        # 设置环境变量
        os.environ["ENVIRONMENT"] = cls.TEST_ENVIRONMENT
        os.environ["TESTING"] = "true"
        
        # 创建测试目录
        ensure_directory(cls.TEST_DATA_DIR)
        ensure_directory(cls.TEST_LOG_DIR)
        ensure_directory(cls.TEST_TEMP_DIR)
        ensure_directory(cls.TEST_BACKUP_DIR)
        ensure_directory(cls.TEST_UPLOAD_DIR)
        
        # 配置日志
        logging_config = LoggingConfig(
            root_level="DEBUG",
            log_dir=str(cls.TEST_LOG_DIR),
            log_to_console=True
        )
        setup_logging(logging_config)
    
    @classmethod
    def cleanup_test_environment(cls):
        """清理测试环境"""
        if cls.TEST_DATA_DIR.exists():
            try:
                shutil.rmtree(cls.TEST_DATA_DIR)
            except Exception as e:
                print(f"清理测试目录失败: {e}")


# 初始化测试环境
TestConfig.setup_test_environment()


# 测试数据生成器
@pytest.fixture
def test_data_generator():
    """测试数据生成器fixture"""
    return TestDataGenerator()


# 信号数据fixtures
@pytest.fixture
def mock_signal_data() -> SignalData:
    """创建模拟信号数据"""
    t = np.linspace(0, 0.001, TestConfig.TEST_SIGNAL_SAMPLES)
    iq_data = np.exp(1j * 2 * np.pi * 1e6 * t)  # 1MHz正弦波
    
    signal = create_signal_from_iq(
        iq_data=iq_data,
        sample_rate=TestConfig.TEST_SAMPLE_RATE,
        center_freq=TestConfig.TEST_CENTER_FREQ,
        name="test_signal"
    )
    
    signal.metadata = {
        "test": True,
        "generated_at": datetime.now().isoformat()
    }
    
    return signal


@pytest.fixture
def mock_signal_dict() -> Dict[str, Any]:
    """创建模拟信号字典"""
    t = np.linspace(0, 0.001, TestConfig.TEST_SIGNAL_SAMPLES)
    iq_data = np.exp(1j * 2 * np.pi * 1e6 * t)
    
    return {
        "signal_id": f"test_signal_{uuid.uuid4()}",
        "timestamp": datetime.now(),
        "center_frequency": TestConfig.TEST_CENTER_FREQ,
        "sample_rate": TestConfig.TEST_SAMPLE_RATE,
        "bandwidth": TestConfig.TEST_BANDWIDTH,
        "iq_data": iq_data,
        "metadata": {"test": True},
        "device_id": "test_device",
        "channel_index": 0,
        "gain": 30.0,
        "noise_floor": -100.0,
        "duration_ms": 0.1
    }


@pytest.fixture
def mock_signal_array() -> np.ndarray:
    """创建模拟信号数组"""
    t = np.linspace(0, 0.001, TestConfig.TEST_SIGNAL_SAMPLES)
    return np.exp(1j * 2 * np.pi * 1e6 * t)


@pytest.fixture
def mock_noisy_signal() -> SignalData:
    """创建带噪声的模拟信号"""
    t = np.linspace(0, 0.001, TestConfig.TEST_SIGNAL_SAMPLES)
    signal = np.exp(1j * 2 * np.pi * 1e6 * t)
    noise = 0.1 * (np.random.randn(TestConfig.TEST_SIGNAL_SAMPLES) + 1j * np.random.randn(TestConfig.TEST_SIGNAL_SAMPLES))
    
    noisy_signal = create_signal_from_iq(
        iq_data=signal + noise,
        sample_rate=TestConfig.TEST_SAMPLE_RATE,
        center_freq=TestConfig.TEST_CENTER_FREQ,
        name="noisy_signal"
    )
    
    return noisy_signal


# 检测结果fixtures
@pytest.fixture
def mock_detection_result() -> DetectionResult:
    """创建模拟检测结果"""
    detection = DetectionResult(
        detection_id=f"test_detection_{uuid.uuid4()}",
        signal_id=f"test_signal_{uuid.uuid4()}",
        detection_type=DetectionType.SIGNAL_PRESENCE,
        confidence=0.85,
        detection_time=datetime.now()
    )
    
    detection.center_freq = TestConfig.TEST_CENTER_FREQ
    detection.bandwidth = TestConfig.TEST_BANDWIDTH
    detection.power_db = -45.5
    detection.snr_db = 15.2
    detection.duration_ms = 50.0
    detection.is_known = True
    detection.known_signal_id = "known_fm_100mhz"
    detection.classification = "FM广播"
    detection.modulation_type = "FM"
    
    detection.features = {
        "modulation_index": 2.5,
        "deviation": 75e3,
        "bandwidth_estimate": 150e3
    }
    
    detection.metadata = {
        "test": True,
        "algorithm": "energy_detector"
    }
    
    return detection


@pytest.fixture
def mock_detection_dict() -> Dict[str, Any]:
    """创建模拟检测结果字典"""
    return {
        "detection_id": f"test_detection_{uuid.uuid4()}",
        "signal_id": f"test_signal_{uuid.uuid4()}",
        "detection_type": "signal_presence",
        "detection_time": datetime.now(),
        "center_frequency": TestConfig.TEST_CENTER_FREQ,
        "bandwidth": TestConfig.TEST_BANDWIDTH,
        "power_db": -45.5,
        "snr_db": 15.2,
        "confidence": 0.85,
        "is_known": True,
        "known_signal_id": "known_fm_100mhz",
        "classification": "FM广播",
        "modulation_type": "FM",
        "features": {
            "modulation_index": 2.5,
            "deviation": 75e3
        },
        "metadata": {"test": True}
    }


# 配置fixtures
@pytest.fixture
def app_config() -> AppConfig:
    """创建应用配置"""
    config = AppConfig(
        environment=TestConfig.TEST_ENVIRONMENT,
        debug=True,
        data_dir=str(TestConfig.TEST_DATA_DIR),
        log_dir=str(TestConfig.TEST_LOG_DIR),
        temp_dir=str(TestConfig.TEST_TEMP_DIR),
        backup_dir=str(TestConfig.TEST_BACKUP_DIR),
        upload_dir=str(TestConfig.TEST_UPLOAD_DIR),
        version="1.0.0-test",
        build="test_build"
    )
    
    return config


@pytest.fixture
def db_config() -> DatabaseConfig:
    """创建数据库配置"""
    config = DatabaseConfig(
        db_type=TestConfig.TEST_DB_TYPE,
        connection={
            "host": "localhost",
            "port": 3306,
            "username": "test_user",
            "password": "test_password",
            "database": TestConfig.TEST_DB_NAME,
            "charset": "utf8mb4"
        },
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
        auto_create_tables=True,
        retention={
            "raw_signals_retention_days": 7,
            "detections_retention_days": 30,
            "raw_signals_auto_cleanup": False
        },
        backup={
            "backup_enabled": False,
            "backup_schedule": "0 2 * * *",
            "backup_retention_days": 7
        }
    )
    
    return config


@pytest.fixture
def web_config() -> WebConfig:
    """创建Web配置"""
    config = WebConfig(
        server={
            "enabled": True,
            "host": TestConfig.TEST_WEB_HOST,
            "port": TestConfig.TEST_WEB_PORT,
            "workers": 1,
            "ssl_enabled": False
        },
        api={
            "title": "ADRV9009 Signal Processing System - Test",
            "description": "Test API",
            "version": "1.0.0-test",
            "api_prefix": TestConfig.TEST_API_PREFIX,
            "enabled": True,
            "database": {
                "enabled": True,
                "required": False
            }
        },
        auth={
            "require_authentication": False,  # 测试中禁用认证
            "jwt_secret_key": "test_secret_key",
            "jwt_algorithm": "HS256",
            "jwt_expire_minutes": 30
        },
        cors={
            "enabled": True,
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        },
        websocket={
            "enabled": True,
            "ws_endpoint": "/ws",
            "ws_broadcast_enabled": True
        }
    )
    
    return config


@pytest.fixture
def signal_config() -> SignalConfig:
    """创建信号配置"""
    config = SignalConfig(
        acquisition={
            "enabled": True,
            "device_type": "adrv9009",
            "sample_rate": 10e6,
            "center_frequency": 100e6,
            "bandwidth": 1e6,
            "gain": 30.0,
            "channels": [0],
            "buffer_size": 1024,
            "acquisition_interval_ms": 100
        },
        processing={
            "enabled": True,
            "window_type": "hann",
            "fft_size": 1024,
            "overlap": 0.5,
            "normalize": True
        },
        detection={
            "enabled": True,
            "threshold_db": -50.0,
            "min_snr_db": 10.0,
            "min_duration_ms": 10.0,
            "max_duration_ms": 1000.0
        },
        algorithms={
            "default_algorithm": "energy_detector",
            "enabled_algorithms": ["energy_detector", "spectrum_analyzer", "modulation_detector"]
        }
    )
    
    return config


@pytest.fixture
def alert_config() -> AlertConfig:
    """创建告警配置"""
    config = AlertConfig(
        email={
            "enabled": False,
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "smtp_username": "test@example.com",
            "smtp_password": "test_password",
            "from_email": "alerts@example.com",
            "use_tls": True
        },
        webhook={
            "enabled": False,
            "url": "http://localhost:8000/webhook"
        },
        slack={
            "enabled": False,
            "webhook_url": "https://hooks.slack.com/services/xxx"
        },
        telegram={
            "enabled": False,
            "bot_token": "xxx",
            "chat_id": "xxx"
        },
        rules={
            "default_alert_level": "medium",
            "suppress_duplicate_minutes": 5
        }
    )
    
    return config


@pytest.fixture
def logging_config() -> LoggingConfig:
    """创建日志配置"""
    config = LoggingConfig(
        root_level="DEBUG",
        log_dir=str(TestConfig.TEST_LOG_DIR),
        log_to_console=True,
        log_to_file=False,  # 测试中不写入文件
        max_file_size_mb=10,
        backup_count=5
    )
    
    return config


# 数据库fixtures
@pytest.fixture
def test_db_url() -> str:
    """创建测试数据库URL"""
    if TestConfig.TEST_DB_TYPE == "sqlite":
        return f"sqlite:///{TestConfig.TEST_DB_NAME}"
    elif TestConfig.TEST_DB_TYPE == "mysql":
        return (f"mysql+pymysql://{TestConfig.TEST_USERNAME}:{TestConfig.TEST_PASSWORD}@"
                f"localhost:3306/{TestConfig.TEST_DB_NAME}")
    else:
        return f"sqlite:///:memory:"


@pytest.fixture
def database_manager(test_db_url: str) -> DatabaseManager:
    """创建数据库管理器"""
    manager = DatabaseManager(test_db_url, echo=False)
    
    # 创建表
    manager.create_tables()
    
    yield manager
    
    # 清理
    manager.close()


@pytest.fixture
def db_session(database_manager: DatabaseManager):
    """创建数据库会话"""
    session = database_manager.SessionLocal()
    
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
async def async_db_session(database_manager: DatabaseManager):
    """创建异步数据库会话"""
    session = database_manager.SessionLocal()
    
    try:
        yield session
    finally:
        session.close()


# 组件fixtures
@pytest.fixture
def event_manager():
    """创建事件管理器"""
    manager = EventManager()
    
    yield manager
    
    manager.stop()


@pytest.fixture
def algorithm_manager(signal_config: SignalConfig, event_manager: EventManager):
    """创建算法管理器"""
    manager = AlgorithmManager(
        config=signal_config.algorithms,
        event_manager=event_manager
    )
    
    yield manager
    
    manager.stop()


@pytest.fixture
def signal_processor(signal_config: SignalConfig, algorithm_manager: AlgorithmManager, event_manager: EventManager):
    """创建信号处理器"""
    processor = SignalProcessor(
        config=signal_config.processing,
        algorithm_manager=algorithm_manager,
        event_manager=event_manager
    )
    
    yield processor
    
    processor.stop()


@pytest.fixture
def detection_engine(signal_config: SignalConfig, algorithm_manager: AlgorithmManager, event_manager: EventManager):
    """创建检测引擎"""
    engine = DetectionEngine(
        config=signal_config.detection,
        algorithm_manager=algorithm_manager,
        event_manager=event_manager
    )
    
    yield engine
    
    engine.stop()


@pytest.fixture
def alert_manager(alert_config: AlertConfig, event_manager: EventManager):
    """创建告警管理器"""
    manager = AlertManager(
        config=alert_config,
        event_manager=event_manager
    )
    
    yield manager
    
    manager.stop()


@pytest.fixture
def system_monitor(app_config: AppConfig, event_manager: EventManager):
    """创建系统监控"""
    monitor = SystemMonitor(
        config=app_config.monitoring,
        event_manager=event_manager
    )
    
    yield monitor
    
    monitor.stop()


@pytest.fixture
def data_acquisition(signal_config: SignalConfig, signal_processor: SignalProcessor, event_manager: EventManager):
    """创建数据采集器"""
    # 模拟数据采集
    with patch('src.core.data_acquisition.ADRV9009Device') as mock_device:
        mock_device.return_value.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        acquisition = DataAcquisition(
            config=signal_config.acquisition,
            signal_processor=signal_processor,
            event_manager=event_manager
        )
        
        yield acquisition
        
        acquisition.stop()


@pytest.fixture
def scheduler(event_manager: EventManager):
    """创建调度器"""
    scheduler = Scheduler(event_manager=event_manager)
    
    yield scheduler
    
    scheduler.stop()


# 工具fixtures
@pytest.fixture
def data_validator():
    """创建数据验证器"""
    return DataValidator(level=ValidationLevel.STRICT)


@pytest.fixture
def performance_monitor():
    """创建性能监控器"""
    monitor = PerformanceMonitor()
    
    yield monitor
    
    monitor.stop()


@pytest.fixture
def signal_utils():
    """创建信号工具"""
    return SignalUtils()


# Web fixtures
@pytest.fixture
def web_app():
    """创建Web应用"""
    app = create_web_app(
        title="Test App",
        description="Test Description",
        version="1.0.0-test"
    )
    
    return app


@pytest.fixture
async def test_client():
    """创建测试客户端"""
    from fastapi.testclient import TestClient
    from src.web.app import create_web_app
    
    app = create_web_app()
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_test_client():
    """创建异步测试客户端"""
    from httpx import AsyncClient
    from src.web.app import create_web_app
    
    app = create_web_app()
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_auth_token():
    """创建测试认证令牌"""
    token = create_access_token(
        data={"sub": TestConfig.TEST_USERNAME, "email": TestConfig.TEST_EMAIL},
        expires_delta=timedelta(minutes=30)
    )
    
    return token


@pytest.fixture
def authenticated_headers(test_auth_token: str):
    """创建认证头"""
    return {
        "Authorization": f"Bearer {test_auth_token}",
        "Content-Type": "application/json"
    }


# 模拟设备fixtures
@pytest.fixture
def mock_adrv9009_device():
    """创建模拟ADRV9009设备"""
    mock_device = Mock()
    
    # 配置模拟方法
    mock_device.initialize.return_value = True
    mock_device.is_initialized.return_value = True
    mock_device.get_sample_rate.return_value = TestConfig.TEST_SAMPLE_RATE
    mock_device.get_center_frequency.return_value = TestConfig.TEST_CENTER_FREQ
    mock_device.get_bandwidth.return_value = TestConfig.TEST_BANDWIDTH
    mock_device.get_gain.return_value = 30.0
    
    # 模拟读取样本
    def mock_read_samples(num_samples):
        t = np.linspace(0, num_samples/TestConfig.TEST_SAMPLE_RATE, num_samples)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        return signal + noise
    
    mock_device.read_samples.side_effect = mock_read_samples
    
    return mock_device


@pytest.fixture
def mock_adrv9009_device_factory(mock_adrv9009_device):
    """创建模拟ADRV9009设备工厂"""
    def factory():
        return mock_adrv9009_device
    
    return factory


# 测试信号生成fixtures
@pytest.fixture
def generate_test_signal():
    """生成测试信号的fixture"""
    def _generate_test_signal(
        freq: float = 1e6,
        sample_rate: float = TestConfig.TEST_SAMPLE_RATE,
        num_samples: int = TestConfig.TEST_SIGNAL_SAMPLES,
        noise_level: float = 0.1,
        modulation: str = None
    ) -> np.ndarray:
        """生成测试信号
        
        Args:
            freq: 信号频率(Hz)
            sample_rate: 采样率(Hz)
            num_samples: 样本数
            noise_level: 噪声级别
            modulation: 调制类型
            
        Returns:
            IQ信号数组
        """
        t = np.linspace(0, num_samples/sample_rate, num_samples)
        
        if modulation == "fm":
            # 频率调制信号
            modulation_index = 5.0
            modulating_signal = np.sin(2 * np.pi * 10e3 * t)
            phase = 2 * np.pi * freq * t + modulation_index * np.cumsum(modulating_signal) / sample_rate
            signal = np.exp(1j * phase)
        
        elif modulation == "am":
            # 幅度调制信号
            carrier = np.exp(1j * 2 * np.pi * freq * t)
            modulating_signal = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * 10e3 * t))
            signal = carrier * modulating_signal
        
        elif modulation == "qpsk":
            # QPSK信号
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_samples//10)
            symbols_upsampled = np.repeat(symbols, 10)
            signal = symbols_upsampled[:num_samples]
        
        else:
            # 简单正弦波
            signal = np.exp(1j * 2 * np.pi * freq * t)
        
        # 添加噪声
        if noise_level > 0:
            noise = noise_level * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            signal = signal + noise
        
        return signal
    
    return _generate_test_signal


# 测试数据插入fixtures
@pytest.fixture
def insert_test_signals(db_session, test_data_generator: TestDataGenerator):
    """插入测试信号数据的fixture"""
    def _insert_test_signals(count: int = 10, **kwargs):
        """插入测试信号
        
        Args:
            count: 信号数量
            **kwargs: 信号参数
            
        Returns:
            插入的信号ID列表
        """
        signals = test_data_generator.generate_signals(count=count, **kwargs)
        signal_ids = []
        
        for signal in signals:
            db_session.add(signal)
            signal_ids.append(signal.signal_id)
        
        db_session.commit()
        return signal_ids
    
    return _insert_test_signals


@pytest.fixture
def insert_test_detections(db_session, test_data_generator: TestDataGenerator):
    """插入测试检测结果的fixture"""
    def _insert_test_detections(signal_ids: List[str] = None, count: int = 10, **kwargs):
        """插入测试检测结果
        
        Args:
            signal_ids: 信号ID列表
            count: 检测结果数量
            **kwargs: 检测参数
            
        Returns:
            插入的检测ID列表
        """
        if signal_ids is None:
            # 生成一些信号
            from src.tests.conftest import insert_test_signals
            signal_ids = insert_test_signals(count=count)
        
        detections = test_data_generator.generate_detections(signal_ids=signal_ids, count=count, **kwargs)
        detection_ids = []
        
        for detection in detections:
            db_session.add(detection)
            detection_ids.append(detection.detection_id)
        
        db_session.commit()
        return detection_ids
    
    return _insert_test_detections


# 测试事件监听fixtures
@pytest.fixture
def event_listener(event_manager: EventManager):
    """创建事件监听器"""
    class EventListener:
        def __init__(self):
            self.events = []
            self.event_manager = event_manager
        
        def subscribe(self, event_type: str):
            self.event_manager.subscribe(event_type, self._handle_event)
        
        def _handle_event(self, event: Dict[str, Any]):
            self.events.append(event)
        
        def get_events(self, event_type: str = None) -> List[Dict[str, Any]]:
            if event_type:
                return [e for e in self.events if e.get('type') == event_type]
            return self.events
        
        def clear(self):
            self.events.clear()
    
    listener = EventListener()
    return listener


# 测试性能监控fixtures
@pytest.fixture
def performance_metrics_collector(performance_monitor: PerformanceMonitor):
    """创建性能指标收集器"""
    class PerformanceMetricsCollector:
        def __init__(self, monitor: PerformanceMonitor):
            self.monitor = monitor
            self.metrics_history = []
        
        def collect(self, duration: float = 1.0, interval: float = 0.1):
            """收集性能指标"""
            import time
            start_time = time.time()
            
            while time.time() - start_time < duration:
                metrics = self.monitor.collect_metrics()
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                time.sleep(interval)
        
        def get_metrics(self) -> List[Dict[str, Any]]:
            """获取收集的指标"""
            return self.metrics_history
        
        def get_average_metrics(self) -> Dict[str, float]:
            """获取平均指标"""
            if not self.metrics_history:
                return {}
            
            avg_metrics = {}
            metric_keys = self.metrics_history[0]['metrics'].keys()
            
            for key in metric_keys:
                values = [m['metrics'].get(key, 0) for m in self.metrics_history if isinstance(m['metrics'].get(key), (int, float))]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
            
            return avg_metrics
    
    collector = PerformanceMetricsCollector(performance_monitor)
    return collector


# 测试工具函数
def assert_signal_equal(signal1: SignalData, signal2: SignalData, rtol: float = 1e-5, atol: float = 1e-8):
    """断言两个信号相等"""
    assert signal1.signal_id == signal2.signal_id
    assert signal1.sample_rate == signal2.sample_rate
    assert signal1.center_freq == signal2.center_freq
    assert signal1.name == signal2.name
    
    if signal1.iq_data is not None and signal2.iq_data is not None:
        np.testing.assert_allclose(signal1.iq_data, signal2.iq_data, rtol=rtol, atol=atol)


def assert_detection_equal(detection1: DetectionResult, detection2: DetectionResult, rtol: float = 1e-5, atol: float = 1e-8):
    """断言两个检测结果相等"""
    assert detection1.detection_id == detection2.detection_id
    assert detection1.signal_id == detection2.signal_id
    assert detection1.detection_type == detection2.detection_type
    assert detection1.confidence == pytest.approx(detection2.confidence, rel=rtol, abs=atol)
    assert detection1.center_freq == pytest.approx(detection2.center_freq, rel=rtol, abs=atol)


def create_test_file(content: str = None, suffix: str = ".txt") -> Path:
    """创建测试文件"""
    if content is None:
        content = f"Test content {uuid.uuid4()}"
    
    temp_dir = TestConfig.TEST_TEMP_DIR
    ensure_directory(temp_dir)
    
    file_path = temp_dir / f"test_file_{uuid.uuid4()}{suffix}"
    file_path.write_text(content)
    
    return file_path


def create_test_iq_file(samples: int = 1000) -> Path:
    """创建测试IQ文件"""
    signal = np.exp(1j * 2 * np.pi * 1e6 * np.linspace(0, 0.001, samples))
    
    temp_dir = TestConfig.TEST_TEMP_DIR
    ensure_directory(temp_dir)
    
    file_path = temp_dir / f"test_iq_{uuid.uuid4()}.bin"
    signal.tofile(file_path)
    
    return file_path


# 测试装饰器
def skip_if_no_database(func):
    """如果数据库不可用则跳过测试"""
    return pytest.mark.skipif(
        TestConfig.TEST_DB_TYPE == "none",
        reason="Database testing disabled"
    )(func)


def skip_if_no_adrv9009(func):
    """如果没有ADRV9009设备则跳过测试"""
    return pytest.mark.skipif(
        not os.getenv("ADRV9009_AVAILABLE", False),
        reason="ADRV9009 device not available"
    )(func)


def slow_test(func):
    """标记为慢测试"""
    return pytest.mark.slow(func)


def integration_test(func):
    """标记为集成测试"""
    return pytest.mark.integration(func)


def performance_test(func):
    """标记为性能测试"""
    return pytest.mark.performance(func)


# 测试类基类
class BaseTest:
    """测试基类"""
    
    @classmethod
    def setup_class(cls):
        """测试类设置"""
        cls.test_config = TestConfig
        cls.test_data_dir = cls.test_config.TEST_DATA_DIR
        
        # 确保测试目录存在
        ensure_directory(cls.test_data_dir)
    
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        # 可选：清理测试目录
        # if cls.test_data_dir.exists():
        #     shutil.rmtree(cls.test_data_dir)
        pass
    
    def setup_method(self):
        """测试方法设置"""
        self.start_time = time.time()
    
    def teardown_method(self):
        """测试方法清理"""
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:  # 记录耗时较长的测试
            print(f"\nTest {self.__class__.__name__}.{self._testMethodName} took {elapsed:.2f}s")
    
    def assert_dict_contains(self, actual: Dict, expected: Dict):
        """断言字典包含"""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            if isinstance(value, dict):
                self.assert_dict_contains(actual[key], value)
            else:
                assert actual[key] == value, f"Value mismatch for key '{key}': expected {value}, got {actual[key]}"
    
    def assert_list_of_dicts_contains(self, actual_list: List[Dict], expected: Dict):
        """断言字典列表包含"""
        found = False
        for actual in actual_list:
            try:
                self.assert_dict_contains(actual, expected)
                found = True
                break
            except AssertionError:
                continue
        assert found, f"Expected dict not found in list: {expected}"


# 异步测试fixtures
@pytest_asyncio.fixture
async def async_event_manager():
    """创建异步事件管理器"""
    manager = EventManager()
    
    yield manager
    
    await manager.stop()


@pytest_asyncio.fixture
async def async_algorithm_manager(signal_config: SignalConfig, async_event_manager: EventManager):
    """创建异步算法管理器"""
    manager = AlgorithmManager(
        config=signal_config.algorithms,
        event_manager=async_event_manager
    )
    
    yield manager
    
    await manager.stop()


# WebSocket测试fixtures
@pytest_asyncio.fixture
async def websocket_manager():
    """创建WebSocket管理器"""
    from src.web.websocket_manager import WebSocketManager
    from config.web_config import WebConfig
    
    config = WebConfig()
    config.websocket.enabled = True
    
    manager = WebSocketManager(config=config.websocket)
    
    await manager.start()
    
    yield manager
    
    await manager.stop()


@pytest_asyncio.fixture
async def websocket_client():
    """创建WebSocket客户端"""
    from websockets.client import connect
    from websockets.exceptions import ConnectionClosed
    
    websocket = None
    
    async def _connect():
        nonlocal websocket
        websocket = await connect(f"ws://{TestConfig.TEST_WEB_HOST}:{TestConfig.TEST_WEB_PORT}/ws")
        return websocket
    
    try:
        yield _connect
    finally:
        if websocket:
            await websocket.close()


# 清理fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_environment():
    """清理测试环境"""
    yield
    
    # 测试结束后清理
    TestConfig.cleanup_test_environment()


# 测试标记注册
def pytest_configure(config):
    """配置pytest"""
    # 注册自定义标记
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "database: test requires database")
    config.addinivalue_line("markers", "adrv9009: test requires ADRV9009 device")
    config.addinivalue_line("markers", "web: test requires web server")
    config.addinivalue_line("markers", "websocket: test requires websocket")


# 测试配置选项
def pytest_addoption(parser):
    """添加pytest命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="运行性能测试"
    )
    parser.addoption(
        "--with-adrv9009",
        action="store_true",
        default=False,
        help="运行需要ADRV9009设备的测试"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试集合"""
    skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项")
    skip_integration = pytest.mark.skip(reason="需要 --run-integration 选项")
    skip_performance = pytest.mark.skip(reason="需要 --run-performance 选项")
    skip_adrv9009 = pytest.mark.skip(reason="需要 --with-adrv9009 选项")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "performance" in item.keywords and not config.getoption("--run-performance"):
            item.add_marker(skip_performance)
        if "adrv9009" in item.keywords and not config.getoption("--with-adrv9009"):
            item.add_marker(skip_adrv9009)


# 测试报告生成
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """生成测试报告"""
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call" and rep.failed:
        # 测试失败时保存额外的调试信息
        if hasattr(item, "funcargs"):
            # 可以在这里保存测试状态
            pass


# 测试覆盖率配置
def pytest_sessionfinish(session, exitstatus):
    """测试会话结束"""
    # 可以在这里生成测试报告
    pass