"""
完整流水线集成测试模块
测试从数据采集到信号处理、检测、告警的完整流水线
"""
import os
import sys
import time
import json
import asyncio
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Generator
from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import multiprocessing as mp
from scipy import signal as scipy_signal
from sqlalchemy.orm import Session
import threading

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig, skip_if_no_adrv9009

# 导入系统组件
from src.core.data_acquisition import DataAcquisition, AcquisitionConfig, DeviceStatus
from src.core.signal_processor import SignalProcessor, ProcessingConfig
from src.core.detection_engine import DetectionEngine, DetectionConfig
from src.core.algorithm_manager import AlgorithmManager, AlgorithmType
from src.core.alert_manager import AlertManager, AlertConfig
from src.core.system_monitor import SystemMonitor
from src.core.event_manager import EventManager
from src.core.scheduler import Scheduler
from src.core.database_manager import DatabaseManager

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq, SignalType, SignalMetrics
from src.models.detection_result import DetectionResult, create_detection_from_signal, DetectionType, ConfidenceLevel
from src.models.algorithm_result import AlgorithmMetrics, AlgorithmPerformance
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

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult, ValidationLevel
from src.utils.signal_utils import SignalUtils
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.file_utils import ensure_directory, get_file_info

# 导入Web组件
from src.web.app import create_web_app, WebApp
from src.web.websocket_manager import WebSocketManager
from src.web.api.data_api import get_db_session
from src.web.api.status_api import get_system_status
from src.web.api.alert_api import get_alert_history

# 导入测试fixtures
from src.tests.conftest import (
    app_config,
    db_config,
    web_config,
    signal_config,
    alert_config,
    logging_config,
    database_manager,
    db_session,
    event_manager,
    algorithm_manager,
    signal_processor,
    detection_engine,
    alert_manager,
    system_monitor,
    scheduler,
    mock_signal_data,
    mock_signal_dict,
    mock_noisy_signal,
    mock_detection_result,
    mock_detection_dict,
    test_data_generator,
    insert_test_signals,
    insert_test_detections
)

# 设置测试日志
import logging
logger = logging.getLogger(__name__)


class TestFullPipeline:
    """完整流水线测试类"""
    
    @pytest.fixture
    def pipeline_components(self, signal_config, event_manager, tmp_path):
        """创建完整流水线组件"""
        # 模拟ADRV9009设备
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建测试信号
            def generate_test_signal(num_samples):
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                # 模拟FM广播信号
                carrier_freq = 1e6
                modulation_freq = 10e3
                modulation_index = 5.0
                
                phase = 2 * np.pi * carrier_freq * t + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
                signal = np.exp(1j * phase)
                
                # 添加噪声
                noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            mock_device.read_samples.side_effect = generate_test_signal
            mock_device.get_temperature.return_value = 35.5
            mock_device.get_power.return_value = 3.3
            MockDevice.return_value = mock_device
            
            # 创建算法管理器
            algorithm_manager = AlgorithmManager(
                config=signal_config.algorithms,
                event_manager=event_manager
            )
            
            # 创建信号处理器
            signal_processor = SignalProcessor(
                config=signal_config.processing,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            # 创建检测引擎
            detection_engine = DetectionEngine(
                config=signal_config.detection,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            # 创建告警管理器
            alert_config_obj = AlertConfig()
            alert_manager = AlertManager(
                config=alert_config_obj,
                event_manager=event_manager
            )
            
            # 创建系统监控
            system_monitor = SystemMonitor(event_manager=event_manager)
            
            # 创建调度器
            scheduler = Scheduler(event_manager=event_manager)
            
            # 创建数据采集器
            data_acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 创建数据库管理器
            db_url = f"sqlite:///{tmp_path}/test_pipeline.db"
            database_manager = DatabaseManager(db_url, echo=False)
            database_manager.create_tables()
            
            # 事件收集器
            class EventCollector:
                def __init__(self):
                    self.events = []
                    self.event_counts = {}
                
                def collect(self, event):
                    self.events.append(event)
                    event_type = event.get('type', 'unknown')
                    self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
                
                def get_events(self, event_type=None):
                    if event_type:
                        return [e for e in self.events if e.get('type') == event_type]
                    return self.events
                
                def get_count(self, event_type):
                    return self.event_counts.get(event_type, 0)
                
                def clear(self):
                    self.events.clear()
                    self.event_counts.clear()
            
            collector = EventCollector()
            
            # 订阅所有事件
            for event_type in [
                'signal_acquired', 'signal_processed', 'signal_detected',
                'algorithm_started', 'algorithm_completed',
                'alert_triggered', 'alert_resolved',
                'system_status_updated', 'database_updated'
            ]:
                event_manager.subscribe(event_type, collector.collect)
            
            return {
                'data_acquisition': data_acquisition,
                'signal_processor': signal_processor,
                'detection_engine': detection_engine,
                'algorithm_manager': algorithm_manager,
                'alert_manager': alert_manager,
                'system_monitor': system_monitor,
                'scheduler': scheduler,
                'database_manager': database_manager,
                'event_manager': event_manager,
                'event_collector': collector,
                'mock_device': mock_device
            }
    
    def test_pipeline_initialization(self, pipeline_components):
        """测试流水线初始化"""
        components = pipeline_components
        
        # 验证所有组件初始化成功
        assert components['data_acquisition'] is not None
        assert components['signal_processor'] is not None
        assert components['detection_engine'] is not None
        assert components['algorithm_manager'] is not None
        assert components['alert_manager'] is not None
        assert components['system_monitor'] is not None
        assert components['scheduler'] is not None
        assert components['database_manager'] is not None
        assert components['event_manager'] is not None
        
        # 验证数据采集器状态
        assert components['data_acquisition'].status == DeviceStatus.IDLE
        assert not components['data_acquisition'].running
        
        # 验证设备初始化
        components['mock_device'].initialize.assert_called_once()
    
    def test_pipeline_start_stop(self, pipeline_components):
        """测试流水线启动和停止"""
        components = pipeline_components
        
        # 启动所有组件
        logger.info("启动流水线组件...")
        
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['alert_manager'].start()
        components['system_monitor'].start()
        components['scheduler'].start()
        components['data_acquisition'].start()
        
        # 验证启动状态
        assert components['data_acquisition'].running is True
        assert components['data_acquisition'].status == DeviceStatus.RUNNING
        
        # 运行一段时间
        logger.info("流水线运行中...")
        time.sleep(2.0)
        
        # 收集事件
        collector = components['event_collector']
        logger.info(f"收集到 {len(collector.events)} 个事件")
        
        # 验证事件流
        assert collector.get_count('signal_acquired') > 0
        assert collector.get_count('signal_processed') > 0
        assert collector.get_count('algorithm_started') > 0
        assert collector.get_count('algorithm_completed') > 0
        
        # 停止所有组件
        logger.info("停止流水线组件...")
        
        components['data_acquisition'].stop()
        components['scheduler'].stop()
        components['system_monitor'].stop()
        components['alert_manager'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证停止状态
        assert components['data_acquisition'].running is False
        assert components['data_acquisition'].status == DeviceStatus.IDLE
    
    def test_pipeline_data_flow(self, pipeline_components):
        """测试流水线数据流"""
        components = pipeline_components
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行采集
        time.sleep(3.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 获取事件统计
        collector = components['event_collector']
        
        # 验证数据流
        signal_acquired = collector.get_count('signal_acquired')
        signal_processed = collector.get_count('signal_processed')
        signal_detected = collector.get_count('signal_detected')
        
        logger.info(f"信号采集: {signal_acquired}")
        logger.info(f"信号处理: {signal_processed}")
        logger.info(f"信号检测: {signal_detected}")
        
        # 验证数据流连续性
        assert signal_acquired > 0, "没有采集到信号"
        assert signal_processed > 0, "没有处理信号"
        
        # 验证处理率
        if signal_acquired > 0:
            processing_rate = signal_processed / signal_acquired
            logger.info(f"处理率: {processing_rate:.2%}")
            
            # 应该处理大部分采集的信号
            assert processing_rate > 0.7, f"处理率过低: {processing_rate:.2%}"
        
        # 验证检测率
        if signal_processed > 0:
            detection_rate = signal_detected / signal_processed
            logger.info(f"检测率: {detection_rate:.2%}")
    
    def test_pipeline_signal_quality(self, pipeline_components):
        """测试流水线信号质量"""
        components = pipeline_components
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行采集
        time.sleep(2.0)
        
        # 获取信号质量事件
        collector = components['event_collector']
        signal_events = collector.get_events('signal_acquired')
        
        # 分析信号质量
        snr_values = []
        power_values = []
        
        for event in signal_events:
            signal_data = event.get('signal_data')
            if signal_data and hasattr(signal_data, 'metrics'):
                metrics = signal_data.metrics
                if hasattr(metrics, 'snr_db'):
                    snr_values.append(metrics.snr_db)
                if hasattr(metrics, 'average_power_db'):
                    power_values.append(metrics.average_power_db)
        
        # 计算统计
        if snr_values:
            avg_snr = np.mean(snr_values)
            min_snr = np.min(snr_values)
            max_snr = np.max(snr_values)
            
            logger.info(f"平均SNR: {avg_snr:.1f} dB")
            logger.info(f"最小SNR: {min_snr:.1f} dB")
            logger.info(f"最大SNR: {max_snr:.1f} dB")
            
            # 验证信号质量
            assert avg_snr > 10, f"平均SNR过低: {avg_snr:.1f} dB"
        
        if power_values:
            avg_power = np.mean(power_values)
            logger.info(f"平均功率: {avg_power:.1f} dB")
            
            # 验证功率范围
            assert -80 < avg_power < 0, f"平均功率异常: {avg_power:.1f} dB"
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
    
    def test_pipeline_detection_accuracy(self, pipeline_components):
        """测试流水线检测准确率"""
        components = pipeline_components
        
        # 创建已知信号数据库
        db_manager = components['database_manager']
        session = db_manager.SessionLocal()
        
        try:
            # 添加已知信号
            known_signal = KnownSignal(
                signal_id="known_fm_100mhz",
                name="FM广播测试信号",
                center_frequency=100e6 + 1e6,  # 101MHz
                bandwidth=200e3,
                modulation_type="FM",
                classification="广播",
                power_db=-45.0,
                description="测试FM广播信号"
            )
            session.add(known_signal)
            session.commit()
            
        finally:
            session.close()
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行采集
        time.sleep(3.0)
        
        # 获取检测事件
        collector = components['event_collector']
        detection_events = collector.get_events('signal_detected')
        
        # 分析检测结果
        detections = []
        for event in detection_events:
            detection = event.get('detection')
            if detection:
                detections.append(detection)
        
        logger.info(f"检测到 {len(detections)} 个信号")
        
        # 验证检测质量
        if detections:
            confidences = [d.confidence for d in detections if hasattr(d, 'confidence')]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            logger.info(f"平均置信度: {avg_confidence:.3f}")
            
            # 验证高置信度检测
            high_confidence = [c for c in confidences if c > 0.7]
            if confidences:
                high_confidence_rate = len(high_confidence) / len(confidences)
                logger.info(f"高置信度检测率: {high_confidence_rate:.2%}")
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
    
    def test_pipeline_performance(self, pipeline_components):
        """测试流水线性能"""
        components = pipeline_components
        
        # 性能监控
        performance_data = {
            '采集延迟': [],
            '处理延迟': [],
            '检测延迟': [],
            '总延迟': []
        }
        
        # 事件时间戳跟踪
        event_timestamps = {}
        
        def track_event(event):
            event_type = event.get('type')
            timestamp = event.get('timestamp', datetime.now())
            
            if event_type == 'signal_acquired':
                event_timestamps['acquired'] = timestamp
            elif event_type == 'signal_processed':
                if 'acquired' in event_timestamps:
                    acquired_time = event_timestamps['acquired']
                    if isinstance(acquired_time, datetime) and isinstance(timestamp, datetime):
                        processing_delay = (timestamp - acquired_time).total_seconds() * 1000
                        performance_data['采集延迟'].append(processing_delay)
            elif event_type == 'signal_detected':
                if 'acquired' in event_timestamps:
                    acquired_time = event_timestamps['acquired']
                    if isinstance(acquired_time, datetime) and isinstance(timestamp, datetime):
                        total_delay = (timestamp - acquired_time).total_seconds() * 1000
                        performance_data['总延迟'].append(total_delay)
        
        # 订阅事件
        event_manager = components['event_manager']
        event_manager.subscribe('signal_acquired', track_event)
        event_manager.subscribe('signal_processed', track_event)
        event_manager.subscribe('signal_detected', track_event)
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行性能测试
        logger.info("开始性能测试...")
        time.sleep(5.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 分析性能数据
        logger.info("性能测试结果:")
        for metric, values in performance_data.items():
            if values:
                avg_value = np.mean(values)
                min_value = np.min(values)
                max_value = np.max(values)
                std_value = np.std(values)
                
                logger.info(f"{metric}: {avg_value:.1f} ± {std_value:.1f} ms "
                           f"(min: {min_value:.1f}, max: {max_value:.1f})")
                
                # 验证性能要求
                if metric == '总延迟':
                    assert avg_value < 1000, f"总延迟过高: {avg_value:.1f} ms"
                    assert max_value < 2000, f"最大延迟过高: {max_value:.1f} ms"
        
        # 验证吞吐量
        total_signals = len(performance_data.get('采集延迟', []))
        if total_signals > 0:
            throughput = total_signals / 5.0  # 信号/秒
            logger.info(f"吞吐量: {throughput:.1f} 信号/秒")
            
            # 验证最小吞吐量
            assert throughput > 1.0, f"吞吐量过低: {throughput:.1f} 信号/秒"
    
    def test_pipeline_error_handling(self, pipeline_components):
        """测试流水线错误处理"""
        components = pipeline_components
        
        # 模拟设备错误
        mock_device = components['mock_device']
        call_count = 0
        
        original_read = mock_device.read_samples
        
        def failing_read(num_samples):
            nonlocal call_count
            call_count += 1
            
            if call_count == 5:  # 第5次调用时失败
                raise Exception("模拟设备读取错误")
            
            return original_read(num_samples)
        
        mock_device.read_samples.side_effect = failing_read
        
        # 收集错误事件
        error_events = []
        
        def collect_errors(event):
            if event.get('type') in ['device_error', 'processing_error', 'algorithm_error']:
                error_events.append(event)
        
        components['event_manager'].subscribe('device_error', collect_errors)
        components['event_manager'].subscribe('processing_error', collect_errors)
        components['event_manager'].subscribe('algorithm_error', collect_errors)
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(3.0)
        
        # 验证错误处理
        assert len(error_events) > 0, "应该检测到错误"
        
        for error_event in error_events:
            logger.info(f"检测到错误: {error_event.get('type')} - {error_event.get('message', '')}")
        
        # 验证流水线继续运行
        collector = components['event_collector']
        signals_after_error = collector.get_count('signal_acquired')
        
        logger.info(f"错误后采集的信号: {signals_after_error}")
        assert signals_after_error > 0, "错误后流水线应该继续运行"
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
    
    def test_pipeline_alert_generation(self, pipeline_components):
        """测试流水线告警生成"""
        components = pipeline_components
        
        # 配置告警规则
        alert_manager = components['alert_manager']
        
        # 创建高功率告警规则
        high_power_rule = {
            'name': '高功率信号告警',
            'description': '检测到高功率信号时触发告警',
            'conditions': [
                {
                    'field': 'power_db',
                    'operator': 'gt',
                    'value': -30.0
                }
            ],
            'actions': [
                {
                    'action_type': 'web',
                    'target': 'dashboard'
                }
            ],
            'alert_level': 'high',
            'alert_type': 'performance',
            'enabled': True
        }
        
        # 创建无信号告警规则
        no_signal_rule = {
            'name': '无信号告警',
            'description': '长时间无信号时触发告警',
            'conditions': [
                {
                    'field': 'signal_count',
                    'operator': 'lt',
                    'value': 1,
                    'duration': 5  # 5秒内无信号
                }
            ],
            'actions': [
                {
                    'action_type': 'email',
                    'target': 'admin@example.com'
                }
            ],
            'alert_level': 'critical',
            'alert_type': 'availability',
            'enabled': True
        }
        
        # 添加告警规则
        alert_manager.add_rule(high_power_rule)
        alert_manager.add_rule(no_signal_rule)
        
        # 收集告警事件
        alert_events = []
        
        def collect_alerts(event):
            if event.get('type') == 'alert_triggered':
                alert_events.append(event)
        
        components['event_manager'].subscribe('alert_triggered', collect_alerts)
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['alert_manager'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(4.0)
        
        # 验证告警生成
        logger.info(f"生成的告警数量: {len(alert_events)}")
        
        for alert_event in alert_events:
            alert = alert_event.get('alert')
            if alert:
                logger.info(f"告警: {alert.get('title')} - 级别: {alert.get('level')}")
        
        # 停止组件
        components['data_acquisition'].stop()
        components['alert_manager'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
    
    def test_pipeline_database_integration(self, pipeline_components, tmp_path):
        """测试流水线数据库集成"""
        components = pipeline_components
        
        # 创建测试数据库
        db_path = tmp_path / "pipeline_test.db"
        db_url = f"sqlite:///{db_path}"
        
        db_manager = DatabaseManager(db_url, echo=False)
        db_manager.create_tables()
        
        # 替换数据库管理器
        components['database_manager'] = db_manager
        
        # 数据库事件处理器
        class DatabaseHandler:
            def __init__(self, db_manager):
                self.db_manager = db_manager
                self.saved_signals = []
                self.saved_detections = []
            
            def handle_signal(self, event):
                signal_data = event.get('signal_data')
                if signal_data:
                    # 保存信号到数据库
                    session = self.db_manager.SessionLocal()
                    try:
                        signal = RawSignal(
                            signal_id=signal_data.signal_id,
                            timestamp=signal_data.timestamp,
                            center_frequency=signal_data.center_freq,
                            sample_rate=signal_data.sample_rate,
                            iq_data=signal_data.iq_data,
                            metadata=signal_data.metadata
                        )
                        session.add(signal)
                        session.commit()
                        self.saved_signals.append(signal)
                    except Exception as e:
                        logger.error(f"保存信号失败: {e}")
                        session.rollback()
                    finally:
                        session.close()
            
            def handle_detection(self, event):
                detection = event.get('detection')
                if detection:
                    # 保存检测结果到数据库
                    session = self.db_manager.SessionLocal()
                    try:
                        db_detection = Detection(
                            detection_id=detection.detection_id,
                            signal_id=detection.signal_id,
                            detection_type=detection.detection_type.value,
                            detection_time=detection.detection_time,
                            center_frequency=detection.center_freq,
                            confidence=detection.confidence,
                            power_db=detection.power_db,
                            snr_db=detection.snr_db,
                            metadata=detection.metadata
                        )
                        session.add(db_detection)
                        session.commit()
                        self.saved_detections.append(db_detection)
                    except Exception as e:
                        logger.error(f"保存检测结果失败: {e}")
                        session.rollback()
                    finally:
                        session.close()
        
        db_handler = DatabaseHandler(db_manager)
        
        # 订阅数据库事件
        event_manager = components['event_manager']
        event_manager.subscribe('signal_acquired', db_handler.handle_signal)
        event_manager.subscribe('signal_detected', db_handler.handle_detection)
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(3.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证数据库保存
        logger.info(f"保存的信号数量: {len(db_handler.saved_signals)}")
        logger.info(f"保存的检测结果数量: {len(db_handler.saved_detections)}")
        
        assert len(db_handler.saved_signals) > 0, "应该保存信号到数据库"
        
        # 验证数据库内容
        session = db_manager.SessionLocal()
        try:
            # 查询信号
            signals_in_db = session.query(RawSignal).count()
            detections_in_db = session.query(Detection).count()
            
            logger.info(f"数据库中的信号: {signals_in_db}")
            logger.info(f"数据库中的检测结果: {detections_in_db}")
            
            assert signals_in_db == len(db_handler.saved_signals)
            assert detections_in_db == len(db_handler.saved_detections)
            
            # 验证信号数据
            if signals_in_db > 0:
                signal = session.query(RawSignal).first()
                assert signal.signal_id is not None
                assert signal.timestamp is not None
                assert signal.center_frequency > 0
                assert signal.sample_rate > 0
        
        finally:
            session.close()
    
    def test_pipeline_resource_monitoring(self, pipeline_components):
        """测试流水线资源监控"""
        components = pipeline_components
        
        # 资源使用数据
        resource_data = {
            'cpu_percent': [],
            'memory_mb': [],
            'disk_usage': [],
            'network_io': []
        }
        
        # 系统监控回调
        def monitor_resources(event):
            if event.get('type') == 'system_status_updated':
                status = event.get('status', {})
                
                if 'cpu_percent' in status:
                    resource_data['cpu_percent'].append(status['cpu_percent'])
                if 'memory_used_mb' in status:
                    resource_data['memory_mb'].append(status['memory_used_mb'])
                if 'disk_usage_percent' in status:
                    resource_data['disk_usage'].append(status['disk_usage_percent'])
        
        event_manager = components['event_manager']
        event_manager.subscribe('system_status_updated', monitor_resources)
        
        # 启动组件（包括系统监控）
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['system_monitor'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(5.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['system_monitor'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 分析资源使用
        logger.info("资源使用统计:")
        
        for resource, values in resource_data.items():
            if values:
                avg_value = np.mean(values)
                max_value = np.max(values)
                
                logger.info(f"{resource}: 平均 {avg_value:.1f}, 最大 {max_value:.1f}")
                
                # 验证资源限制
                if resource == 'cpu_percent':
                    assert max_value < 90, f"CPU使用率过高: {max_value:.1f}%"
                elif resource == 'memory_mb':
                    assert max_value < 1024, f"内存使用过高: {max_value:.1f} MB"
                elif resource == 'disk_usage':
                    assert max_value < 90, f"磁盘使用率过高: {max_value:.1f}%"
    
    def test_pipeline_scheduler_integration(self, pipeline_components):
        """测试流水线调度器集成"""
        components = pipeline_components
        
        # 调度任务执行计数
        task_counts = {
            'maintenance': 0,
            'backup': 0,
            'report': 0
        }
        
        # 定义调度任务
        def maintenance_task():
            task_counts['maintenance'] += 1
            logger.info("执行维护任务")
            return {"status": "completed", "task": "maintenance"}
        
        def backup_task():
            task_counts['backup'] += 1
            logger.info("执行备份任务")
            return {"status": "completed", "task": "backup"}
        
        def report_task():
            task_counts['report'] += 1
            logger.info("执行报告任务")
            return {"status": "completed", "task": "report"}
        
        # 添加调度任务
        scheduler = components['scheduler']
        scheduler.add_task(
            name="maintenance",
            function=maintenance_task,
            interval_seconds=2.0
        )
        
        scheduler.add_task(
            name="backup",
            function=backup_task,
            interval_seconds=3.0
        )
        
        scheduler.add_task(
            name="report",
            function=report_task,
            interval_seconds=5.0
        )
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['scheduler'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(7.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['scheduler'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证调度任务执行
        logger.info("调度任务执行统计:")
        for task, count in task_counts.items():
            logger.info(f"{task}: {count} 次")
            
            # 验证任务执行
            if task == 'maintenance':
                assert count >= 3, f"维护任务执行次数不足: {count}"
            elif task == 'backup':
                assert count >= 2, f"备份任务执行次数不足: {count}"
            elif task == 'report':
                assert count >= 1, f"报告任务执行次数不足: {count}"
    
    def test_pipeline_websocket_integration(self, pipeline_components):
        """测试流水线WebSocket集成"""
        components = pipeline_components
        
        # 模拟WebSocket管理器
        class MockWebSocketManager:
            def __init__(self):
                self.connections = []
                self.messages = []
                self.broadcast_enabled = True
            
            async def connect(self, websocket):
                self.connections.append(websocket)
            
            async def disconnect(self, websocket):
                if websocket in self.connections:
                    self.connections.remove(websocket)
            
            async def broadcast(self, message):
                if self.broadcast_enabled:
                    self.messages.append(message)
                    logger.debug(f"WebSocket广播: {message['type']}")
            
            def get_message_count(self, message_type=None):
                if message_type:
                    return len([m for m in self.messages if m.get('type') == message_type])
                return len(self.messages)
        
        ws_manager = MockWebSocketManager()
        
        # WebSocket广播处理器
        async def broadcast_to_ws(event):
            message = {
                'type': event.get('type'),
                'timestamp': datetime.now().isoformat(),
                'data': event
            }
            await ws_manager.broadcast(message)
        
        # 创建异步事件处理器
        def async_event_handler(event):
            asyncio.create_task(broadcast_to_ws(event))
        
        # 订阅关键事件
        event_manager = components['event_manager']
        for event_type in ['signal_acquired', 'signal_detected', 'alert_triggered']:
            event_manager.subscribe(event_type, async_event_handler)
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(3.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证WebSocket消息
        logger.info(f"WebSocket消息总数: {ws_manager.get_message_count()}")
        logger.info(f"信号采集消息: {ws_manager.get_message_count('signal_acquired')}")
        logger.info(f"信号检测消息: {ws_manager.get_message_count('signal_detected')}")
        
        # 验证消息发送
        assert ws_manager.get_message_count() > 0, "应该发送WebSocket消息"
        assert ws_manager.get_message_count('signal_acquired') > 0, "应该发送信号采集消息"
    
    def test_pipeline_configuration_update(self, pipeline_components):
        """测试流水线配置更新"""
        components = pipeline_components
        
        # 初始配置
        initial_sample_rate = components['data_acquisition'].config.sample_rate
        initial_center_freq = components['data_acquisition'].config.center_frequency
        
        logger.info(f"初始采样率: {initial_sample_rate/1e6:.1f} MHz")
        logger.info(f"初始中心频率: {initial_center_freq/1e6:.1f} MHz")
        
        # 启动组件
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行一段时间
        time.sleep(2.0)
        
        # 更新配置
        new_sample_rate = 20e6  # 20 MHz
        new_center_freq = 200e6  # 200 MHz
        
        # 创建新配置
        new_config = components['data_acquisition'].config.copy()
        new_config.sample_rate = new_sample_rate
        new_config.center_frequency = new_center_freq
        
        # 更新配置
        components['data_acquisition'].update_config(new_config)
        
        logger.info(f"更新采样率: {new_sample_rate/1e6:.1f} MHz")
        logger.info(f"更新中心频率: {new_center_freq/1e6:.1f} MHz")
        
        # 验证配置更新
        assert components['data_acquisition'].config.sample_rate == new_sample_rate
        assert components['data_acquisition'].config.center_frequency == new_center_freq
        
        # 验证设备配置更新
        mock_device = components['mock_device']
        mock_device.set_sample_rate.assert_called_with(new_sample_rate)
        mock_device.set_center_frequency.assert_called_with(new_center_freq)
        
        # 继续运行
        time.sleep(2.0)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证流水线在配置更新后继续运行
        collector = components['event_collector']
        total_signals = collector.get_count('signal_acquired')
        
        logger.info(f"总采集信号: {total_signals}")
        assert total_signals > 0, "配置更新后应该继续采集信号"
    
    def test_pipeline_restart_recovery(self, pipeline_components):
        """测试流水线重启恢复"""
        components = pipeline_components
        
        # 第一次启动
        logger.info("第一次启动流水线...")
        
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        time.sleep(2.0)
        
        # 获取第一次运行的数据
        collector1 = components['event_collector'].copy()
        signals_before = collector1.get_count('signal_acquired')
        
        logger.info(f"第一次运行采集信号: {signals_before}")
        
        # 停止
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 清理事件收集器
        components['event_collector'].clear()
        
        # 第二次启动
        logger.info("第二次启动流水线...")
        
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        time.sleep(2.0)
        
        # 获取第二次运行的数据
        signals_after = components['event_collector'].get_count('signal_acquired')
        
        logger.info(f"第二次运行采集信号: {signals_after}")
        
        # 停止
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证重启恢复
        assert signals_before > 0, "第一次运行应该采集信号"
        assert signals_after > 0, "第二次运行应该采集信号"
        
        logger.info(f"重启恢复测试: 第一次 {signals_before} 信号, 第二次 {signals_after} 信号")
    
    def test_pipeline_long_running(self, pipeline_components):
        """测试流水线长时间运行"""
        components = pipeline_components
        
        # 性能数据
        performance_history = []
        
        def monitor_performance():
            # 每秒记录一次性能
            start_time = time.time()
            
            while True:
                time.sleep(1.0)
                
                # 获取组件状态
                acquisition_status = components['data_acquisition'].status
                acquisition_running = components['data_acquisition'].running
                
                # 获取事件计数
                collector = components['event_collector']
                signals = collector.get_count('signal_acquired')
                detections = collector.get_count('signal_detected')
                
                # 记录性能
                performance_history.append({
                    'timestamp': time.time(),
                    'acquisition_status': acquisition_status,
                    'acquisition_running': acquisition_running,
                    'signals_acquired': signals,
                    'signals_detected': detections
                })
                
                # 运行30秒
                if time.time() - start_time > 30:
                    break
        
        # 启动性能监控线程
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()
        
        # 启动组件
        logger.info("开始长时间运行测试 (30秒)...")
        
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['system_monitor'].start()
        components['data_acquisition'].start()
        
        # 等待监控线程完成
        monitor_thread.join(timeout=35)
        
        # 停止组件
        components['data_acquisition'].stop()
        components['system_monitor'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 分析长时间运行性能
        if performance_history:
            # 计算统计
            durations = [p['timestamp'] for p in performance_history]
            total_duration = durations[-1] - durations[0]
            
            signals_over_time = [p['signals_acquired'] for p in performance_history]
            detections_over_time = [p['signals_detected'] for p in performance_history]
            
            # 计算吞吐量
            if len(signals_over_time) > 1:
                signal_rate = (signals_over_time[-1] - signals_over_time[0]) / total_duration
                detection_rate = (detections_over_time[-1] - detections_over_time[0]) / total_duration
                
                logger.info(f"运行时间: {total_duration:.1f} 秒")
                logger.info(f"信号采集率: {signal_rate:.1f} 信号/秒")
                logger.info(f"信号检测率: {detection_rate:.1f} 检测/秒")
                
                # 验证稳定性
                assert signal_rate > 0.5, f"信号采集率过低: {signal_rate:.1f} 信号/秒"
                assert components['data_acquisition'].status == DeviceStatus.IDLE, "停止后状态应为IDLE"
        
        logger.info("长时间运行测试完成")
    
    def test_pipeline_with_different_signal_types(self, pipeline_components):
        """测试流水线处理不同类型信号"""
        components = pipeline_components
        
        # 定义不同类型信号
        signal_types = [
            {
                'name': 'CW信号',
                'generator': lambda t: np.exp(1j * 2 * np.pi * 1e6 * t)
            },
            {
                'name': 'AM信号',
                'generator': lambda t: np.exp(1j * 2 * np.pi * 1e6 * t) * (1 + 0.5 * np.sin(2 * np.pi * 10e3 * t))
            },
            {
                'name': 'FM信号',
                'generator': lambda t: np.exp(1j * (2 * np.pi * 1e6 * t + 5.0 * np.sin(2 * np.pi * 10e3 * t)))
            },
            {
                'name': 'QPSK信号',
                'generator': lambda t: np.repeat(np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 100), 10)[:len(t)]
            }
        ]
        
        results = {}
        
        for signal_info in signal_types:
            logger.info(f"测试 {signal_info['name']}...")
            
            # 重置事件收集器
            components['event_collector'].clear()
            
            # 创建信号生成器
            def signal_generator(num_samples):
                t = np.linspace(0, num_samples/10e6, num_samples)
                signal = signal_info['generator'](t)
                # 添加噪声
                noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            # 更新设备读取函数
            mock_device = components['mock_device']
            mock_device.read_samples.side_effect = signal_generator
            
            # 启动组件
            components['algorithm_manager'].start()
            components['signal_processor'].start()
            components['detection_engine'].start()
            components['data_acquisition'].start()
            
            # 运行测试
            time.sleep(2.0)
            
            # 停止组件
            components['data_acquisition'].stop()
            components['detection_engine'].stop()
            components['signal_processor'].stop()
            components['algorithm_manager'].stop()
            
            # 收集结果
            collector = components['event_collector']
            signals = collector.get_count('signal_acquired')
            detections = collector.get_count('signal_detected')
            
            results[signal_info['name']] = {
                'signals': signals,
                'detections': detections,
                'detection_rate': detections / signals if signals > 0 else 0
            }
            
            logger.info(f"{signal_info['name']}: {signals} 信号, {detections} 检测, "
                       f"检测率 {results[signal_info['name']]['detection_rate']:.1%}")
        
        # 分析结果
        logger.info("不同信号类型测试结果:")
        for signal_name, result in results.items():
            logger.info(f"{signal_name}: {result['signals']} 信号, "
                       f"检测率 {result['detection_rate']:.1%}")
            
            # 验证基本功能
            assert result['signals'] > 0, f"{signal_name}: 没有采集到信号"
    
    def test_pipeline_with_varying_snr(self, pipeline_components):
        """测试流水线在不同SNR下的表现"""
        components = pipeline_components
        
        # 测试不同SNR水平
        snr_levels = [20, 10, 5, 0, -5]  # dB
        
        results = {}
        
        for snr_db in snr_levels:
            logger.info(f"测试 SNR = {snr_db} dB...")
            
            # 重置事件收集器
            components['event_collector'].clear()
            
            # 计算噪声水平
            signal_power = 1.0  # 信号功率
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = np.sqrt(noise_power)
            
            # 创建信号生成器
            def signal_generator(num_samples):
                t = np.linspace(0, num_samples/10e6, num_samples)
                signal = np.exp(1j * 2 * np.pi * 1e6 * t)
                noise = noise_std * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            # 更新设备读取函数
            mock_device = components['mock_device']
            mock_device.read_samples.side_effect = signal_generator
            
            # 启动组件
            components['algorithm_manager'].start()
            components['signal_processor'].start()
            components['detection_engine'].start()
            components['data_acquisition'].start()
            
            # 运行测试
            time.sleep(2.0)
            
            # 停止组件
            components['data_acquisition'].stop()
            components['detection_engine'].stop()
            components['signal_processor'].stop()
            components['algorithm_manager'].stop()
            
            # 收集结果
            collector = components['event_collector']
            signals = collector.get_count('signal_acquired')
            detections = collector.get_count('signal_detected')
            
            # 计算检测率
            detection_rate = detections / signals if signals > 0 else 0
            
            results[snr_db] = {
                'signals': signals,
                'detections': detections,
                'detection_rate': detection_rate
            }
            
            logger.info(f"SNR {snr_db} dB: {signals} 信号, {detections} 检测, "
                       f"检测率 {detection_rate:.1%}")
        
        # 分析结果
        logger.info("不同SNR测试结果:")
        for snr_db, result in results.items():
            logger.info(f"SNR {snr_db} dB: 检测率 {result['detection_rate']:.1%}")
            
            # 验证SNR与检测率的关系
            if snr_db >= 10:
                # 高SNR应该有高检测率
                assert result['detection_rate'] > 0.5, f"SNR {snr_db} dB 检测率过低"
            elif snr_db <= 0:
                # 低SNR检测率可能较低
                # 但流水线应该仍然工作
                assert result['signals'] > 0, f"SNR {snr_db} dB 没有采集到信号"


class TestFullPipelineWithRealDatabase:
    """完整流水线数据库集成测试"""
    
    @pytest.fixture
    def pipeline_with_db(self, tmp_path, signal_config, event_manager):
        """创建带数据库的流水线"""
        # 创建数据库
        db_path = tmp_path / "integration_test.db"
        db_url = f"sqlite:///{db_path}"
        
        db_manager = DatabaseManager(db_url, echo=False)
        db_manager.create_tables()
        
        # 模拟设备
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            def signal_generator(num_samples):
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                signal = np.exp(1j * 2 * np.pi * 1e6 * t)
                noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            mock_device.read_samples.side_effect = signal_generator
            MockDevice.return_value = mock_device
            
            # 创建组件
            algorithm_manager = AlgorithmManager(
                config=signal_config.algorithms,
                event_manager=event_manager
            )
            
            signal_processor = SignalProcessor(
                config=signal_config.processing,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            detection_engine = DetectionEngine(
                config=signal_config.detection,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            data_acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 数据库处理器
            class DatabaseProcessor:
                def __init__(self, db_manager):
                    self.db_manager = db_manager
                    self.saved_data = {
                        'signals': 0,
                        'detections': 0,
                        'events': 0
                    }
                
                def save_signal(self, event):
                    signal_data = event.get('signal_data')
                    if signal_data:
                        session = self.db_manager.SessionLocal()
                        try:
                            signal = RawSignal(
                                signal_id=signal_data.signal_id,
                                timestamp=signal_data.timestamp,
                                center_frequency=signal_data.center_freq,
                                sample_rate=signal_data.sample_rate,
                                iq_data=signal_data.iq_data,
                                metadata=signal_data.metadata
                            )
                            session.add(signal)
                            session.commit()
                            self.saved_data['signals'] += 1
                        except Exception as e:
                            logger.error(f"保存信号失败: {e}")
                            session.rollback()
                        finally:
                            session.close()
                
                def save_detection(self, event):
                    detection = event.get('detection')
                    if detection:
                        session = self.db_manager.SessionLocal()
                        try:
                            db_detection = Detection(
                                detection_id=detection.detection_id,
                                signal_id=detection.signal_id,
                                detection_type=detection.detection_type.value,
                                detection_time=detection.detection_time,
                                center_frequency=detection.center_freq,
                                confidence=detection.confidence,
                                power_db=detection.power_db,
                                snr_db=detection.snr_db
                            )
                            session.add(db_detection)
                            session.commit()
                            self.saved_data['detections'] += 1
                        except Exception as e:
                            logger.error(f"保存检测失败: {e}")
                            session.rollback()
                        finally:
                            session.close()
                
                def save_event(self, event):
                    session = self.db_manager.SessionLocal()
                    try:
                        db_event = SystemEvent(
                            event_type=event.get('type', 'unknown'),
                            source=event.get('source', 'pipeline'),
                            message=json.dumps(event, default=str),
                            timestamp=datetime.now()
                        )
                        session.add(db_event)
                        session.commit()
                        self.saved_data['events'] += 1
                    except Exception as e:
                        logger.error(f"保存事件失败: {e}")
                        session.rollback()
                    finally:
                        session.close()
            
            db_processor = DatabaseProcessor(db_manager)
            
            # 订阅事件
            event_manager.subscribe('signal_acquired', db_processor.save_signal)
            event_manager.subscribe('signal_detected', db_processor.save_detection)
            event_manager.subscribe('algorithm_completed', db_processor.save_event)
            event_manager.subscribe('alert_triggered', db_processor.save_event)
            
            return {
                'db_manager': db_manager,
                'db_processor': db_processor,
                'algorithm_manager': algorithm_manager,
                'signal_processor': signal_processor,
                'detection_engine': detection_engine,
                'data_acquisition': data_acquisition,
                'event_manager': event_manager,
                'mock_device': mock_device
            }
    
    def test_database_persistence(self, pipeline_with_db):
        """测试数据库持久化"""
        components = pipeline_with_db
        
        # 启动流水线
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        # 运行测试
        time.sleep(3.0)
        
        # 停止流水线
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 验证数据库保存
        db_processor = components['db_processor']
        
        logger.info(f"保存的数据: {db_processor.saved_data}")
        
        assert db_processor.saved_data['signals'] > 0, "应该保存信号到数据库"
        
        # 验证数据库内容
        session = components['db_manager'].SessionLocal()
        try:
            # 查询信号
            signals = session.query(RawSignal).all()
            detections = session.query(Detection).all()
            events = session.query(SystemEvent).all()
            
            logger.info(f"数据库中的信号: {len(signals)}")
            logger.info(f"数据库中的检测结果: {len(detections)}")
            logger.info(f"数据库中的事件: {len(events)}")
            
            assert len(signals) == db_processor.saved_data['signals']
            assert len(detections) == db_processor.saved_data['detections']
            assert len(events) == db_processor.saved_data['events']
            
            # 验证信号数据完整性
            for signal in signals:
                assert signal.signal_id is not None
                assert signal.timestamp is not None
                assert signal.center_frequency > 0
                assert signal.sample_rate > 0
            
            # 验证检测结果
            for detection in detections:
                assert detection.detection_id is not None
                assert detection.signal_id is not None
                assert 0 <= detection.confidence <= 1
            
        finally:
            session.close()
    
    def test_database_query_performance(self, pipeline_with_db):
        """测试数据库查询性能"""
        components = pipeline_with_db
        
        # 启动流水线并生成数据
        components['algorithm_manager'].start()
        components['signal_processor'].start()
        components['detection_engine'].start()
        components['data_acquisition'].start()
        
        time.sleep(2.0)
        
        components['data_acquisition'].stop()
        components['detection_engine'].stop()
        components['signal_processor'].stop()
        components['algorithm_manager'].stop()
        
        # 测试查询性能
        session = components['db_manager'].SessionLocal()
        
        try:
            # 测试各种查询
            query_times = {}
            
            # 1. 简单计数查询
            start = time.time()
            signal_count = session.query(RawSignal).count()
            end = time.time()
            query_times['count_signals'] = (end - start) * 1000
            
            # 2. 条件查询
            start = time.time()
            recent_signals = session.query(RawSignal).filter(
                RawSignal.timestamp > datetime.now() - timedelta(minutes=5)
            ).all()
            end = time.time()
            query_times['filter_signals'] = (end - start) * 1000
            
            # 3. 连接查询
            start = time.time()
            from sqlalchemy.orm import joinedload
            signals_with_detections = session.query(RawSignal).options(
                joinedload(RawSignal.detections)
            ).limit(10).all()
            end = time.time()
            query_times['join_query'] = (end - start) * 1000
            
            # 4. 聚合查询
            start = time.time()
            from sqlalchemy import func
            avg_power = session.query(func.avg(Detection.power_db)).scalar()
            end = time.time()
            query_times['aggregate_query'] = (end - start) * 1000
            
            # 记录查询性能
            logger.info("数据库查询性能:")
            for query_name, query_time in query_times.items():
                logger.info(f"{query_name}: {query_time:.1f} ms")
                
                # 验证查询性能
                if query_name in ['count_signals', 'filter_signals']:
                    assert query_time < 100, f"{query_name} 查询过慢: {query_time:.1f} ms"
                elif query_name in ['join_query', 'aggregate_query']:
                    assert query_time < 200, f"{query_name} 查询过慢: {query_time:.1f} ms"
        
        finally:
            session.close()


class TestFullPipelineEndToEnd:
    """完整流水线端到端测试"""
    
    def test_complete_workflow(self, tmp_path, signal_config, event_manager):
        """测试完整工作流：从采集到Web展示"""
        import tempfile
        
        # 创建临时数据库
        db_file = tmp_path / "e2e_test.db"
        db_url = f"sqlite:///{db_file}"
        
        # 初始化数据库
        db_manager = DatabaseManager(db_url, echo=False)
        db_manager.create_tables()
        
        # 创建测试目录
        data_dir = tmp_path / "data"
        log_dir = tmp_path / "logs"
        ensure_directory(data_dir)
        ensure_directory(log_dir)
        
        # 模拟ADRV9009设备
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 生成测试信号
            def generate_test_signals(num_samples):
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                
                # 生成不同类型的信号
                signal_type = np.random.choice(['cw', 'am', 'fm'])
                
                if signal_type == 'cw':
                    signal = np.exp(1j * 2 * np.pi * 1e6 * t)
                elif signal_type == 'am':
                    carrier = np.exp(1j * 2 * np.pi * 1e6 * t)
                    modulating = 1 + 0.5 * np.sin(2 * np.pi * 10e3 * t)
                    signal = carrier * modulating
                else:  # 'fm'
                    phase = 2 * np.pi * 1e6 * t + 5.0 * np.sin(2 * np.pi * 10e3 * t)
                    signal = np.exp(1j * phase)
                
                # 添加噪声
                noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            mock_device.read_samples.side_effect = generate_test_signals
            MockDevice.return_value = mock_device
            
            # 创建所有组件
            algorithm_manager = AlgorithmManager(
                config=signal_config.algorithms,
                event_manager=event_manager
            )
            
            signal_processor = SignalProcessor(
                config=signal_config.processing,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            detection_engine = DetectionEngine(
                config=signal_config.detection,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            # 创建告警配置
            alert_config_obj = AlertConfig()
            alert_manager = AlertManager(
                config=alert_config_obj,
                event_manager=event_manager
            )
            
            # 添加告警规则
            alert_manager.add_rule({
                'name': '高功率告警',
                'conditions': [{'field': 'power_db', 'operator': 'gt', 'value': -30}],
                'actions': [{'action_type': 'log'}],
                'alert_level': 'high',
                'enabled': True
            })
            
            system_monitor = SystemMonitor(event_manager=event_manager)
            scheduler = Scheduler(event_manager=event_manager)
            
            data_acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 数据库处理器
            class E2EDatabaseHandler:
                def __init__(self, db_manager):
                    self.db_manager = db_manager
                    self.stats = {
                        'signals': 0,
                        'detections': 0,
                        'alerts': 0
                    }
                
                def handle_signal(self, event):
                    signal_data = event.get('signal_data')
                    if signal_data:
                        session = self.db_manager.SessionLocal()
                        try:
                            signal = RawSignal(
                                signal_id=signal_data.signal_id,
                                timestamp=signal_data.timestamp,
                                center_frequency=signal_data.center_freq,
                                sample_rate=signal_data.sample_rate,
                                iq_data=signal_data.iq_data,
                                metadata=signal_data.metadata
                            )
                            session.add(signal)
                            session.commit()
                            self.stats['signals'] += 1
                        finally:
                            session.close()
                
                def handle_detection(self, event):
                    detection = event.get('detection')
                    if detection:
                        session = self.db_manager.SessionLocal()
                        try:
                            db_detection = Detection(
                                detection_id=detection.detection_id,
                                signal_id=detection.signal_id,
                                detection_type=detection.detection_type.value,
                                detection_time=detection.detection_time,
                                center_frequency=detection.center_freq,
                                confidence=detection.confidence,
                                power_db=detection.power_db
                            )
                            session.add(db_detection)
                            session.commit()
                            self.stats['detections'] += 1
                        finally:
                            session.close()
                
                def handle_alert(self, event):
                    alert = event.get('alert')
                    if alert:
                        session = self.db_manager.SessionLocal()
                        try:
                            db_alert = AlertHistory(
                                alert_id=alert.get('alert_id'),
                                title=alert.get('title'),
                                message=alert.get('message'),
                                alert_type=alert.get('alert_type'),
                                alert_level=alert.get('alert_level'),
                                source=alert.get('source'),
                                status='active',
                                metadata=alert
                            )
                            session.add(db_alert)
                            session.commit()
                            self.stats['alerts'] += 1
                        finally:
                            session.close()
            
            db_handler = E2EDatabaseHandler(db_manager)
            
            # 订阅事件
            event_manager.subscribe('signal_acquired', db_handler.handle_signal)
            event_manager.subscribe('signal_detected', db_handler.handle_detection)
            event_manager.subscribe('alert_triggered', db_handler.handle_alert)
            
            # 启动完整流水线
            logger.info("启动完整流水线...")
            
            start_time = time.time()
            
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            alert_manager.start()
            system_monitor.start()
            scheduler.start()
            data_acquisition.start()
            
            # 运行端到端测试
            logger.info("运行端到端测试...")
            time.sleep(5.0)
            
            # 停止流水线
            data_acquisition.stop()
            scheduler.stop()
            system_monitor.stop()
            alert_manager.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 验证端到端工作流
            logger.info(f"端到端测试完成，总时间: {total_time:.1f} 秒")
            logger.info(f"统计数据: {db_handler.stats}")
            
            # 验证数据流
            assert db_handler.stats['signals'] > 0, "应该采集并保存信号"
            assert db_handler.stats['detections'] > 0, "应该检测信号"
            
            # 验证数据库内容
            session = db_manager.SessionLocal()
            try:
                # 验证信号
                signals = session.query(RawSignal).all()
                assert len(signals) == db_handler.stats['signals']
                
                # 验证检测
                detections = session.query(Detection).all()
                assert len(detections) == db_handler.stats['detections']
                
                # 验证告警
                alerts = session.query(AlertHistory).all()
                assert len(alerts) == db_handler.stats['alerts']
                
                # 验证数据关系
                for detection in detections:
                    # 每个检测应该对应一个信号
                    signal = session.query(RawSignal).filter_by(signal_id=detection.signal_id).first()
                    assert signal is not None, f"检测 {detection.detection_id} 没有对应的信号"
                
                logger.info(f"端到端验证: {len(signals)} 信号, {len(detections)} 检测, {len(alerts)} 告警")
                
            finally:
                session.close()
            
            # 验证性能
            signals_per_second = db_handler.stats['signals'] / total_time
            detections_per_second = db_handler.stats['detections'] / total_time
            
            logger.info(f"性能: {signals_per_second:.1f} 信号/秒, {detections_per_second:.1f} 检测/秒")
            
            assert signals_per_second > 0.5, f"吞吐量过低: {signals_per_second:.1f} 信号/秒"


if __name__ == "__main__":
    """直接运行测试"""
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])