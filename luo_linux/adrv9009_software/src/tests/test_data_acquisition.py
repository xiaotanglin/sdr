"""
数据采集测试模块
测试DataAcquisition类的功能
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
import queue

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig, skip_if_no_adrv9009

# 导入被测试模块
from src.core.data_acquisition import (
    DataAcquisition,
    AcquisitionMode,
    DeviceStatus,
    BufferStatus,
    AcquisitionConfig,
    AcquisitionMetrics
)

# 导入相关模块
from src.core.signal_processor import SignalProcessor
from src.core.event_manager import EventManager
from src.models.signal_data import SignalData, create_signal_from_iq
from src.utils.data_validator import DataValidator, ValidationResult
from src.utils.signal_utils import SignalUtils

# 导入测试fixtures
from src.tests.conftest import (
    signal_config,
    event_manager,
    signal_processor,
    mock_adrv9009_device,
    mock_adrv9009_device_factory
)

# 设置测试日志
import logging
logger = logging.getLogger(__name__)


class TestDataAcquisition:
    """数据采集测试类"""
    
    def test_initialization(self, signal_config, event_manager, signal_processor):
        """测试数据采集器初始化"""
        # 模拟设备
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            MockDevice.return_value = mock_device
            
            # 创建数据采集器
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 验证初始化
            assert acquisition.config == signal_config.acquisition
            assert acquisition.signal_processor == signal_processor
            assert acquisition.event_manager == event_manager
            assert acquisition.device is not None
            assert acquisition.status == DeviceStatus.IDLE
            assert acquisition.running is False
            
            # 验证设备初始化
            mock_device.initialize.assert_called_once()
            
            # 清理
            acquisition.stop()
    
    def test_initialization_without_device(self, signal_config, event_manager, signal_processor):
        """测试无设备初始化"""
        # 配置禁用设备
        config = signal_config.acquisition.copy()
        config.enabled = False
        
        acquisition = DataAcquisition(
            config=config,
            signal_processor=signal_processor,
            event_manager=event_manager
        )
        
        # 验证
        assert acquisition.device is None
        assert acquisition.status == DeviceStatus.DISABLED
        assert not acquisition.running
        
        # 清理
        acquisition.stop()
    
    def test_initialization_failure(self, signal_config, event_manager, signal_processor):
        """测试初始化失败"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = False
            MockDevice.return_value = mock_device
            
            with pytest.raises(RuntimeError, match="设备初始化失败"):
                DataAcquisition(
                    config=signal_config.acquisition,
                    signal_processor=signal_processor,
                    event_manager=event_manager
                )
    
    def test_start_stop(self, signal_config, event_manager, signal_processor):
        """测试启动和停止"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            assert acquisition.running is True
            assert acquisition.status == DeviceStatus.RUNNING
            
            # 等待采集线程启动
            time.sleep(0.1)
            
            # 停止
            acquisition.stop()
            assert acquisition.running is False
            assert acquisition.status == DeviceStatus.IDLE
            
            # 验证设备停止
            mock_device.stop.assert_called_once()
    
    def test_acquisition_cycle(self, signal_config, event_manager, signal_processor):
        """测试采集周期"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟读取样本
            test_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
            mock_device.read_samples.return_value = test_samples
            MockDevice.return_value = mock_device
            
            # 模拟信号处理器
            mock_processor = Mock(spec=SignalProcessor)
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=mock_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)  # 等待一个采集周期
            
            # 停止
            acquisition.stop()
            
            # 验证读取样本被调用
            mock_device.read_samples.assert_called()
            
            # 验证信号处理被调用
            mock_processor.process_signal.assert_called()
            
            # 验证事件发布
            assert event_manager.has_event("signal_acquired")
    
    def test_acquisition_with_real_device(self, signal_config, event_manager, signal_processor):
        """测试真实设备采集"""
        # 这个测试需要真实的ADRV9009设备
        pytest.skip("需要真实的ADRV9009设备")
    
    def test_acquisition_modes(self, signal_config, event_manager, signal_processor):
        """测试不同采集模式"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 测试连续模式
            config = signal_config.acquisition.copy()
            config.mode = "continuous"
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            acquisition.start()
            time.sleep(0.1)
            acquisition.stop()
            
            # 测试单次模式
            config.mode = "single"
            
            acquisition2 = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            acquisition2.start()
            time.sleep(0.1)
            acquisition2.stop()
    
    def test_buffer_management(self, signal_config, event_manager, signal_processor):
        """测试缓冲区管理"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建大量测试样本
            test_samples = []
            for i in range(10):
                test_samples.append(np.random.randn(1024) + 1j * np.random.randn(1024))
            
            mock_device.read_samples.side_effect = test_samples
            MockDevice.return_value = mock_device
            
            # 创建小缓冲区测试
            config = signal_config.acquisition.copy()
            config.buffer_size = 1024
            config.buffer_capacity = 5  # 小容量
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.3)  # 采集多个样本
            
            # 检查缓冲区状态
            buffer_status = acquisition.get_buffer_status()
            assert buffer_status is not None
            assert buffer_status.current_size <= config.buffer_capacity
            
            # 停止
            acquisition.stop()
    
    def test_signal_validation(self, signal_config, event_manager, signal_processor):
        """测试信号验证"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建有效信号
            valid_samples = np.exp(1j * 2 * np.pi * 1e6 * np.linspace(0, 0.001, 1024))
            mock_device.read_samples.return_value = valid_samples
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 获取验证结果
            metrics = acquisition.get_metrics()
            assert metrics.signals_validated > 0
            
            # 停止
            acquisition.stop()
    
    def test_error_handling(self, signal_config, event_manager, signal_processor):
        """测试错误处理"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟设备错误
            mock_device.read_samples.side_effect = Exception("设备读取错误")
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 验证错误处理
            metrics = acquisition.get_metrics()
            assert metrics.errors > 0
            
            # 验证状态
            assert acquisition.status == DeviceStatus.ERROR
            
            # 停止
            acquisition.stop()
    
    def test_configuration_update(self, signal_config, event_manager, signal_processor):
        """测试配置更新"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 更新配置
            new_config = signal_config.acquisition.copy()
            new_config.sample_rate = 20e6
            new_config.center_frequency = 200e6
            
            result = acquisition.update_config(new_config)
            assert result is True
            
            # 验证设备配置更新
            mock_device.set_sample_rate.assert_called_with(20e6)
            mock_device.set_center_frequency.assert_called_with(200e6)
            
            # 清理
            acquisition.stop()
    
    def test_metrics_collection(self, signal_config, event_manager, signal_processor):
        """测试指标收集"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 获取指标
            metrics = acquisition.get_metrics()
            
            # 验证指标
            assert metrics is not None
            assert metrics.samples_acquired > 0
            assert metrics.signals_processed > 0
            assert metrics.acquisition_time_ms > 0
            assert metrics.sample_rate_hz == signal_config.acquisition.sample_rate
            
            # 获取详细指标
            detailed_metrics = acquisition.get_detailed_metrics()
            assert detailed_metrics is not None
            assert "buffer_usage" in detailed_metrics
            assert "signal_quality" in detailed_metrics
            
            # 停止
            acquisition.stop()
    
    def test_signal_quality_monitoring(self, signal_config, event_manager, signal_processor):
        """测试信号质量监控"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建高质量信号
            t = np.linspace(0, 0.001, 1024)
            high_quality_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
            mock_device.read_samples.return_value = high_quality_signal
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 获取质量指标
            metrics = acquisition.get_metrics()
            assert hasattr(metrics, 'average_snr_db')
            assert hasattr(metrics, 'average_power_db')
            
            # 停止
            acquisition.stop()
    
    def test_multiple_channels(self, signal_config, event_manager, signal_processor):
        """测试多通道采集"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟多通道数据
            channels = [0, 1]
            test_data = {}
            for channel in channels:
                t = np.linspace(0, 0.001, 1024)
                test_data[channel] = np.exp(1j * 2 * np.pi * (1e6 + channel * 0.5e6) * t)
            
            mock_device.read_samples.side_effect = lambda n, channel=0: test_data[channel]
            MockDevice.return_value = mock_device
            
            # 配置多通道
            config = signal_config.acquisition.copy()
            config.channels = channels
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 验证多通道采集
            metrics = acquisition.get_metrics()
            assert metrics.channels_active == len(channels)
            
            # 停止
            acquisition.stop()
    
    def test_event_publication(self, signal_config, event_manager, signal_processor):
        """测试事件发布"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 创建事件监听器
            events_received = []
            
            def event_handler(event):
                events_received.append(event)
            
            event_manager.subscribe("signal_acquired", event_handler)
            event_manager.subscribe("acquisition_started", event_handler)
            event_manager.subscribe("acquisition_stopped", event_handler)
            event_manager.subscribe("device_error", event_handler)
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证事件
            event_types = [e.get('type') for e in events_received]
            assert "acquisition_started" in event_types
            assert "signal_acquired" in event_types
            
            # 停止
            acquisition.stop()
            time.sleep(0.1)
            
            event_types = [e.get('type') for e in events_received]
            assert "acquisition_stopped" in event_types
            
            # 验证信号采集事件
            signal_events = [e for e in events_received if e.get('type') == 'signal_acquired']
            assert len(signal_events) > 0
            
            for event in signal_events:
                assert 'signal_data' in event
                assert 'timestamp' in event
    
    def test_resource_cleanup(self, signal_config, event_manager, signal_processor):
        """测试资源清理"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 停止
            acquisition.stop()
            
            # 验证资源清理
            mock_device.stop.assert_called_once()
            mock_device.close.assert_called_once()
            
            # 验证状态
            assert acquisition.running is False
            assert acquisition.status == DeviceStatus.IDLE
            assert acquisition.acquisition_thread is None
    
    def test_concurrent_access(self, signal_config, event_manager, signal_processor):
        """测试并发访问"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            
            # 并发访问
            import threading
            
            results = []
            errors = []
            
            def get_metrics():
                try:
                    metrics = acquisition.get_metrics()
                    results.append(metrics)
                except Exception as e:
                    errors.append(str(e))
            
            # 创建多个线程并发访问
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=get_metrics)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证
            assert len(errors) == 0, f"并发访问出错: {errors}"
            assert len(results) == 10
            
            # 停止
            acquisition.stop()
    
    def test_signal_metadata(self, signal_config, event_manager, signal_processor):
        """测试信号元数据"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            mock_device.get_temperature.return_value = 45.5
            mock_device.get_power.return_value = 3.3
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证信号包含元数据
            events = event_manager.get_events("signal_acquired")
            assert len(events) > 0
            
            for event in events:
                signal_data = event.get('signal_data')
                if signal_data and hasattr(signal_data, 'metadata'):
                    metadata = signal_data.metadata
                    assert 'device_temperature' in metadata
                    assert 'device_power' in metadata
                    assert 'acquisition_time' in metadata
            
            # 停止
            acquisition.stop()
    
    def test_performance_monitoring(self, signal_config, event_manager, signal_processor):
        """测试性能监控"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.5)  # 采集一段时间
            
            # 获取性能指标
            metrics = acquisition.get_metrics()
            
            # 验证性能指标
            assert metrics.acquisition_rate_hz > 0
            assert metrics.processing_time_ms >= 0
            assert metrics.buffer_usage_percent >= 0
            
            # 验证实时性能
            real_time_metrics = acquisition.get_real_time_metrics()
            assert real_time_metrics is not None
            assert 'current_sample_rate' in real_time_metrics
            assert 'current_latency_ms' in real_time_metrics
            
            # 停止
            acquisition.stop()
    
    def test_signal_synchronization(self, signal_config, event_manager, signal_processor):
        """测试信号同步"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建时间同步的信号
            sample_rate = signal_config.acquisition.sample_rate
            t_base = np.linspace(0, 1.0, int(sample_rate))
            
            signals = []
            for i in range(5):
                phase = 2 * np.pi * 1e6 * t_base + i * 0.1
                signal = np.exp(1j * phase)
                signals.append(signal)
            
            signal_iter = iter(signals)
            mock_device.read_samples.side_effect = lambda n: next(signal_iter)[:n]
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证时间戳连续性
            events = event_manager.get_events("signal_acquired")
            timestamps = []
            
            for event in events:
                signal_data = event.get('signal_data')
                if signal_data and hasattr(signal_data, 'timestamp'):
                    timestamps.append(signal_data.timestamp)
            
            # 验证时间戳是连续的
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    # 考虑到处理延迟，时间差应该在合理范围内
                    assert 0.09 <= time_diff <= 0.11  # 100ms间隔±10%
            
            # 停止
            acquisition.stop()
    
    def test_config_validation(self, signal_config, event_manager, signal_processor):
        """测试配置验证"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            MockDevice.return_value = mock_device
            
            # 测试无效配置
            invalid_config = signal_config.acquisition.copy()
            invalid_config.sample_rate = 0  # 无效采样率
            
            with pytest.raises(ValueError, match="采样率必须为正数"):
                DataAcquisition(
                    config=invalid_config,
                    signal_processor=signal_processor,
                    event_manager=event_manager
                )
            
            # 测试无效缓冲区大小
            invalid_config = signal_config.acquisition.copy()
            invalid_config.buffer_size = 0
            
            with pytest.raises(ValueError, match="缓冲区大小必须为正数"):
                DataAcquisition(
                    config=invalid_config,
                    signal_processor=signal_processor,
                    event_manager=event_manager
                )
            
            # 测试无效通道
            invalid_config = signal_config.acquisition.copy()
            invalid_config.channels = []  # 空通道列表
            
            with pytest.raises(ValueError, match="必须指定至少一个通道"):
                DataAcquisition(
                    config=invalid_config,
                    signal_processor=signal_processor,
                    event_manager=event_manager
                )
    
    def test_signal_compression(self, signal_config, event_manager, signal_processor):
        """测试信号压缩"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 配置压缩
            config = signal_config.acquisition.copy()
            config.compress_signals = True
            config.compression_level = 6
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证压缩
            events = event_manager.get_events("signal_acquired")
            assert len(events) > 0
            
            for event in events:
                signal_data = event.get('signal_data')
                if signal_data and hasattr(signal_data, 'metadata'):
                    metadata = signal_data.metadata
                    assert 'compressed' in metadata
                    assert metadata['compressed'] is True
            
            # 停止
            acquisition.stop()
    
    def test_signal_filtering(self, signal_config, event_manager, signal_processor):
        """测试信号滤波"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建带噪声的信号
            t = np.linspace(0, 0.001, 1024)
            clean_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
            noise = 0.5 * (np.random.randn(1024) + 1j * np.random.randn(1024))
            noisy_signal = clean_signal + noise
            
            mock_device.read_samples.return_value = noisy_signal
            MockDevice.return_value = mock_device
            
            # 配置滤波
            config = signal_config.acquisition.copy()
            config.apply_filter = True
            config.filter_type = "bandpass"
            config.filter_cutoff_low = 0.9e6
            config.filter_cutoff_high = 1.1e6
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证滤波
            events = event_manager.get_events("signal_acquired")
            assert len(events) > 0
            
            for event in events:
                signal_data = event.get('signal_data')
                if signal_data and hasattr(signal_data, 'metadata'):
                    metadata = signal_data.metadata
                    assert 'filter_applied' in metadata
                    assert metadata['filter_applied'] is True
            
            # 停止
            acquisition.stop()
    
    def test_data_persistence(self, signal_config, event_manager, signal_processor, tmp_path):
        """测试数据持久化"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 配置持久化
            config = signal_config.acquisition.copy()
            config.save_to_disk = True
            config.data_directory = str(tmp_path)
            config.max_file_size_mb = 1
            config.max_files = 5
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 验证文件保存
            data_files = list(tmp_path.glob("*.h5"))
            assert len(data_files) > 0
            
            # 验证文件内容
            for data_file in data_files:
                assert data_file.stat().st_size > 0
            
            # 停止
            acquisition.stop()
    
    def test_realtime_visualization(self, signal_config, event_manager, signal_processor):
        """测试实时可视化数据"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 获取可视化数据
            viz_data = acquisition.get_visualization_data()
            
            # 验证可视化数据
            assert viz_data is not None
            assert 'spectrum' in viz_data
            assert 'time_domain' in viz_data
            assert 'constellation' in viz_data
            assert 'metrics' in viz_data
            
            # 验证数据格式
            assert isinstance(viz_data['spectrum'], dict)
            assert 'frequencies' in viz_data['spectrum']
            assert 'magnitude' in viz_data['spectrum']
            
            assert isinstance(viz_data['time_domain'], dict)
            assert 'time' in viz_data['time_domain']
            assert 'amplitude' in viz_data['time_domain']
            
            # 停止
            acquisition.stop()
    
    def test_adaptive_sampling(self, signal_config, event_manager, signal_processor):
        """测试自适应采样"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟信号变化
            def varying_signal(num_samples):
                freq = 1e6 + 0.5e6 * np.sin(time.time())
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * freq * t)
            
            mock_device.read_samples.side_effect = varying_signal
            MockDevice.return_value = mock_device
            
            # 配置自适应采样
            config = signal_config.acquisition.copy()
            config.adaptive_sampling = True
            config.min_sample_rate = 5e6
            config.max_sample_rate = 20e6
            config.snr_target_db = 20.0
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.3)  # 运行一段时间
            
            # 验证自适应调整
            metrics = acquisition.get_metrics()
            assert hasattr(metrics, 'current_sample_rate')
            
            # 获取配置历史
            config_history = acquisition.get_config_history()
            assert len(config_history) > 0
            
            # 停止
            acquisition.stop()
    
    def test_signal_calibration(self, signal_config, event_manager, signal_processor):
        """测试信号校准"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟需要校准的信号
            t = np.linspace(0, 0.001, 1024)
            uncalibrated_signal = 0.8 * np.exp(1j * (2 * np.pi * 1e6 * t + 0.1))  # 幅度和相位偏移
            mock_device.read_samples.return_value = uncalibrated_signal
            MockDevice.return_value = mock_device
            
            # 配置校准
            config = signal_config.acquisition.copy()
            config.calibrate_signals = True
            config.calibration_interval_minutes = 1
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 执行校准
            calibration_result = acquisition.perform_calibration()
            assert calibration_result is not None
            assert 'success' in calibration_result
            assert calibration_result['success'] is True
            
            # 启动
            acquisition.start()
            time.sleep(0.1)
            
            # 验证校准应用
            events = event_manager.get_events("signal_acquired")
            assert len(events) > 0
            
            for event in events:
                signal_data = event.get('signal_data')
                if signal_data and hasattr(signal_data, 'metadata'):
                    metadata = signal_data.metadata
                    assert 'calibrated' in metadata
                    assert metadata['calibrated'] is True
            
            # 停止
            acquisition.stop()
    
    def test_signal_triggering(self, signal_config, event_manager, signal_processor):
        """测试信号触发"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟触发信号
            def triggered_signal(num_samples):
                # 生成一个脉冲信号
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                signal = np.zeros(num_samples, dtype=np.complex64)
                
                # 在中间添加一个脉冲
                pulse_start = num_samples // 2
                pulse_end = pulse_start + 100
                if pulse_end < num_samples:
                    signal[pulse_start:pulse_end] = np.exp(1j * 2 * np.pi * 1e6 * t[:100])
                
                return signal
            
            mock_device.read_samples.side_effect = triggered_signal
            MockDevice.return_value = mock_device
            
            # 配置触发
            config = signal_config.acquisition.copy()
            config.trigger_enabled = True
            config.trigger_type = "amplitude"
            config.trigger_threshold = 0.5
            config.trigger_delay_samples = 100
            config.pre_trigger_samples = 500
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 验证触发事件
            events = event_manager.get_events("trigger_activated")
            assert len(events) > 0
            
            for event in events:
                assert 'trigger_type' in event
                assert 'trigger_time' in event
                assert 'signal_data' in event
            
            # 停止
            acquisition.stop()
    
    def test_error_recovery(self, signal_config, event_manager, signal_processor):
        """测试错误恢复"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 模拟间歇性错误
            call_count = 0
            
            def intermittent_error(num_samples):
                nonlocal call_count
                call_count += 1
                
                if call_count % 3 == 0:  # 每3次调用失败一次
                    raise Exception("模拟设备错误")
                
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            
            mock_device.read_samples.side_effect = intermittent_error
            MockDevice.return_value = mock_device
            
            # 配置错误恢复
            config = signal_config.acquisition.copy()
            config.auto_recover = True
            config.max_recovery_attempts = 3
            config.recovery_delay_ms = 100
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.5)  # 运行一段时间
            
            # 验证错误恢复
            metrics = acquisition.get_metrics()
            assert metrics.recovery_attempts > 0
            assert metrics.recovery_successes >= 0
            
            # 验证采集继续运行
            assert acquisition.running is True
            assert acquisition.status == DeviceStatus.RUNNING
            
            # 停止
            acquisition.stop()
    
    def test_performance_benchmark(self, signal_config, event_manager, signal_processor):
        """测试性能基准"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 运行性能测试
            benchmark_result = acquisition.run_performance_benchmark(duration_seconds=1.0)
            
            # 验证性能指标
            assert benchmark_result is not None
            assert 'average_sample_rate' in benchmark_result
            assert 'max_sample_rate' in benchmark_result
            assert 'min_latency_ms' in benchmark_result
            assert 'max_latency_ms' in benchmark_result
            assert 'cpu_usage_percent' in benchmark_result
            assert 'memory_usage_mb' in benchmark_result
            
            # 验证性能
            assert benchmark_result['average_sample_rate'] > 0
            assert benchmark_result['max_sample_rate'] >= benchmark_result['average_sample_rate']
            assert benchmark_result['min_latency_ms'] >= 0
            assert benchmark_result['max_latency_ms'] >= benchmark_result['min_latency_ms']
            
            # 清理
            acquisition.stop()
    
    def test_integration_with_signal_processor(self, signal_config, event_manager):
        """测试与信号处理器的集成"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 创建真实的信号处理器
            from src.core.signal_processor import SignalProcessor
            from src.core.algorithm_manager import AlgorithmManager
            
            algorithm_manager = AlgorithmManager(
                config=signal_config.algorithms,
                event_manager=event_manager
            )
            
            signal_processor = SignalProcessor(
                config=signal_config.processing,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 验证集成
            events = event_manager.get_events("signal_processed")
            assert len(events) > 0
            
            # 验证数据流
            signal_events = event_manager.get_events("signal_acquired")
            processed_events = event_manager.get_events("signal_processed")
            
            assert len(signal_events) > 0
            assert len(processed_events) > 0
            assert len(processed_events) <= len(signal_events)
            
            # 停止
            acquisition.stop()
            signal_processor.stop()
            algorithm_manager.stop()
    
    def test_thread_safety(self, signal_config, event_manager, signal_processor):
        """测试线程安全"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动采集线程
            acquisition.start()
            
            # 创建多个线程并发访问
            import threading
            import queue as thread_queue
            
            results_queue = thread_queue.Queue()
            errors = []
            
            def test_operation(operation_id):
                try:
                    # 执行各种操作
                    if operation_id % 4 == 0:
                        result = acquisition.get_metrics()
                    elif operation_id % 4 == 1:
                        result = acquisition.get_status()
                    elif operation_id % 4 == 2:
                        result = acquisition.get_buffer_status()
                    else:
                        result = acquisition.get_config()
                    
                    results_queue.put((operation_id, result))
                except Exception as e:
                    errors.append((operation_id, str(e)))
            
            # 启动多个测试线程
            threads = []
            for i in range(20):
                thread = threading.Thread(target=test_operation, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 收集结果
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            # 验证
            assert len(errors) == 0, f"线程安全错误: {errors}"
            assert len(results) == 20
            
            # 停止
            acquisition.stop()
    
    def test_signal_quality_metrics(self, signal_config, event_manager, signal_processor):
        """测试信号质量指标"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建不同质量的信号
            def varying_quality_signal(num_samples):
                quality = np.sin(time.time()) * 0.5 + 0.5  # 0到1之间变化
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                
                # 高质量信号
                clean_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
                
                # 添加噪声
                noise_level = 1.0 - quality
                noise = noise_level * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                
                return clean_signal + noise
            
            mock_device.read_samples.side_effect = varying_quality_signal
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.3)
            
            # 获取质量指标
            quality_metrics = acquisition.get_signal_quality_metrics()
            
            # 验证质量指标
            assert quality_metrics is not None
            assert 'average_snr_db' in quality_metrics
            assert 'average_power_db' in quality_metrics
            assert 'iq_balance_db' in quality_metrics
            assert 'dc_offset_db' in quality_metrics
            assert 'phase_noise_db' in quality_quality_metrics
            
            # 验证指标范围
            assert -200 <= quality_metrics['average_power_db'] <= 0
            assert 0 <= quality_metrics['average_snr_db'] <= 100
            
            # 停止
            acquisition.stop()
    
    def test_configuration_persistence(self, signal_config, event_manager, signal_processor, tmp_path):
        """测试配置持久化"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 保存配置
            config_file = tmp_path / "acquisition_config.json"
            acquisition.save_config(str(config_file))
            
            # 验证文件存在
            assert config_file.exists()
            assert config_file.stat().st_size > 0
            
            # 加载配置
            loaded_config = acquisition.load_config(str(config_file))
            
            # 验证配置
            assert loaded_config is not None
            assert loaded_config.sample_rate == signal_config.acquisition.sample_rate
            assert loaded_config.center_frequency == signal_config.acquisition.center_frequency
            
            # 清理
            acquisition.stop()
    
    def test_signal_statistics(self, signal_config, event_manager, signal_processor):
        """测试信号统计"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建统计变化的信号
            def statistical_signal(num_samples):
                # 随时间变化的统计特性
                mean_i = np.sin(time.time()) * 0.1
                mean_q = np.cos(time.time()) * 0.1
                variance = 0.5 + 0.3 * np.sin(time.time() * 0.5)
                
                signal = (mean_i + 1j * mean_q) + np.sqrt(variance) * (
                    np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
                )
                return signal
            
            mock_device.read_samples.side_effect = statistical_signal
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.3)
            
            # 获取统计
            statistics = acquisition.get_signal_statistics()
            
            # 验证统计
            assert statistics is not None
            assert 'mean_i' in statistics
            assert 'mean_q' in statistics
            assert 'variance_i' in statistics
            assert 'variance_q' in statistics
            assert 'covariance' in statistics
            assert 'kurtosis' in statistics
            assert 'skewness' in statistics
            
            # 验证统计范围
            assert -1 <= statistics['mean_i'] <= 1
            assert -1 <= statistics['mean_q'] <= 1
            assert statistics['variance_i'] >= 0
            assert statistics['variance_q'] >= 0
            
            # 停止
            acquisition.stop()
    
    def test_signal_archive(self, signal_config, event_manager, signal_processor, tmp_path):
        """测试信号归档"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 配置归档
            config = signal_config.acquisition.copy()
            config.archive_signals = True
            config.archive_directory = str(tmp_path)
            config.archive_retention_days = 7
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.2)
            
            # 执行归档
            archive_result = acquisition.archive_signals()
            
            # 验证归档
            assert archive_result is not None
            assert 'archived_files' in archive_result
            assert 'total_size_bytes' in archive_result
            assert archive_result['success'] is True
            
            # 验证归档文件
            archive_files = list(tmp_path.glob("*.h5"))
            assert len(archive_files) > 0
            
            # 停止
            acquisition.stop()


class TestDataAcquisitionPerformance:
    """数据采集性能测试类"""
    
    @pytest.mark.performance
    def test_high_sample_rate_performance(self, signal_config, event_manager, signal_processor):
        """测试高采样率性能"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 高采样率配置
            config = signal_config.acquisition.copy()
            config.sample_rate = 100e6  # 100 MHz
            config.buffer_size = 8192
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            # 生成高速数据
            def high_speed_data(num_samples):
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            
            mock_device.read_samples.side_effect = high_speed_data
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 运行性能测试
            import time
            start_time = time.time()
            
            acquisition.start()
            time.sleep(1.0)  # 运行1秒
            acquisition.stop()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 获取性能指标
            metrics = acquisition.get_metrics()
            
            # 验证性能
            sample_rate_hz = metrics.samples_acquired / duration
            logger.info(f"实际采样率: {sample_rate_hz/1e6:.2f} MHz")
            
            # 目标：达到配置采样率的80%
            target_rate = config.sample_rate * 0.8
            assert sample_rate_hz >= target_rate, f"采样率不足: {sample_rate_hz/1e6:.2f} MHz < {target_rate/1e6:.2f} MHz"
    
    @pytest.mark.performance
    def test_low_latency_performance(self, signal_config, event_manager, signal_processor):
        """测试低延迟性能"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 低延迟配置
            config = signal_config.acquisition.copy()
            config.buffer_size = 256  # 小缓冲区
            config.acquisition_interval_ms = 1  # 1ms间隔
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            mock_device.read_samples.return_value = np.random.randn(config.buffer_size) + 1j * np.random.randn(config.buffer_size)
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 测量延迟
            latencies = []
            
            def measure_latency(event):
                if 'signal_data' in event and hasattr(event['signal_data'], 'timestamp'):
                    acquisition_time = event['signal_data'].timestamp
                    processing_time = datetime.now()
                    latency = (processing_time - acquisition_time).total_seconds() * 1000  # 毫秒
                    latencies.append(latency)
            
            event_manager.subscribe("signal_acquired", measure_latency)
            
            # 运行测试
            acquisition.start()
            time.sleep(0.5)  # 运行0.5秒
            acquisition.stop()
            
            # 分析延迟
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                
                logger.info(f"平均延迟: {avg_latency:.2f} ms")
                logger.info(f"最小延迟: {min_latency:.2f} ms")
                logger.info(f"最大延迟: {max_latency:.2f} ms")
                
                # 目标：平均延迟 < 10ms
                assert avg_latency < 10, f"延迟过高: {avg_latency:.2f} ms"
                assert max_latency < 50, f"最大延迟过高: {max_latency:.2f} ms"
            else:
                pytest.skip("没有采集到信号数据")
    
    @pytest.mark.performance
    def test_memory_usage(self, signal_config, event_manager, signal_processor):
        """测试内存使用"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 大缓冲区配置
            config = signal_config.acquisition.copy()
            config.buffer_size = 65536  # 64k样本
            config.buffer_capacity = 100  # 100个缓冲区
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            mock_device.read_samples.return_value = np.random.randn(config.buffer_size) + 1j * np.random.randn(config.buffer_size)
            MockDevice.return_value = mock_device
            
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 运行测试
            acquisition.start()
            time.sleep(0.5)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            logger.info(f"内存增加: {memory_increase:.2f} MB")
            
            # 停止
            acquisition.stop()
            
            # 验证内存释放
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_released = current_memory - final_memory
            
            logger.info(f"内存释放: {memory_released:.2f} MB")
            
            # 目标：内存增加合理
            expected_memory = (config.buffer_size * config.buffer_capacity * 16) / 1024 / 1024  # 字节到MB
            assert memory_increase <= expected_memory * 1.5, f"内存使用过高: {memory_increase:.2f} MB"
    
    @pytest.mark.performance
    def test_cpu_usage(self, signal_config, event_manager, signal_processor):
        """测试CPU使用"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 高负载配置
            config = signal_config.acquisition.copy()
            config.buffer_size = 4096
            config.acquisition_interval_ms = 1  # 1ms间隔，高频率
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            mock_device.read_samples.return_value = np.random.randn(config.buffer_size) + 1j * np.random.randn(config.buffer_size)
            MockDevice.return_value = mock_device
            
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 测量基线CPU使用
            initial_cpu = process.cpu_percent(interval=0.1)
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 运行测试
            acquisition.start()
            time.sleep(1.0)  # 运行1秒
            
            # 测量运行中CPU使用
            running_cpu = process.cpu_percent(interval=0.5)
            
            # 停止
            acquisition.stop()
            
            cpu_increase = running_cpu - initial_cpu
            
            logger.info(f"CPU使用增加: {cpu_increase:.2f}%")
            logger.info(f"运行中CPU使用: {running_cpu:.2f}%")
            
            # 目标：CPU使用 < 50%
            assert running_cpu < 50, f"CPU使用过高: {running_cpu:.2f}%"
    
    @pytest.mark.performance
    def test_throughput_performance(self, signal_config, event_manager, signal_processor):
        """测试吞吐量性能"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 高吞吐量配置
            config = signal_config.acquisition.copy()
            config.sample_rate = 50e6  # 50 MHz
            config.buffer_size = 16384
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            samples_generated = 0
            
            def high_throughput_data(num_samples):
                nonlocal samples_generated
                samples_generated += num_samples
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            
            mock_device.read_samples.side_effect = high_throughput_data
            MockDevice.return_value = mock_device
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 测量吞吐量
            import time
            start_time = time.time()
            
            acquisition.start()
            time.sleep(2.0)  # 运行2秒
            acquisition.stop()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 计算吞吐量
            samples_per_second = samples_generated / duration
            bytes_per_second = samples_per_second * 16  # 复数，每个样本16字节
            
            logger.info(f"样本吞吐量: {samples_per_second/1e6:.2f} MS/s")
            logger.info(f"数据吞吐量: {bytes_per_second/1024/1024:.2f} MB/s")
            
            # 目标：达到理论值的70%
            theoretical_samples = config.sample_rate
            efficiency = samples_per_second / theoretical_samples
            
            logger.info(f"效率: {efficiency*100:.1f}%")
            
            assert efficiency >= 0.7, f"吞吐量效率低: {efficiency*100:.1f}%"


class TestDataAcquisitionIntegration:
    """数据采集集成测试类"""
    
    @pytest.mark.integration
    def test_integration_with_full_pipeline(self, signal_config, event_manager):
        """测试与完整管道的集成"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 创建完整管道
            from src.core.algorithm_manager import AlgorithmManager
            from src.core.signal_processor import SignalProcessor
            from src.core.detection_engine import DetectionEngine
            from src.core.alert_manager import AlertManager
            
            # 初始化所有组件
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
            
            alert_manager = AlertManager(
                config=get_alert_config(),
                event_manager=event_manager
            )
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动所有组件
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            alert_manager.start()
            acquisition.start()
            
            # 运行一段时间
            time.sleep(1.0)
            
            # 验证数据流
            events = {
                'signal_acquired': 0,
                'signal_processed': 0,
                'signal_detected': 0,
                'alert_triggered': 0
            }
            
            for event_type in events.keys():
                events[event_type] = len(event_manager.get_events(event_type))
            
            logger.info(f"事件统计: {events}")
            
            # 验证数据流完整性
            assert events['signal_acquired'] > 0
            assert events['signal_processed'] > 0
            # 检测和告警可能为0，取决于信号内容
            
            # 停止所有组件
            acquisition.stop()
            alert_manager.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
    
    @pytest.mark.integration
    def test_integration_with_database(self, signal_config, event_manager, signal_processor, database_manager):
        """测试与数据库的集成"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 创建数据库监听器
            from src.models.database_models import RawSignal
            
            db_signals = []
            
            def save_to_database(event):
                signal_data = event.get('signal_data')
                if signal_data:
                    # 模拟保存到数据库
                    signal = RawSignal(
                        signal_id=signal_data.signal_id,
                        timestamp=signal_data.timestamp,
                        center_frequency=signal_data.center_freq,
                        sample_rate=signal_data.sample_rate,
                        iq_data=signal_data.iq_data
                    )
                    db_signals.append(signal)
            
            event_manager.subscribe("signal_acquired", save_to_database)
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.5)
            acquisition.stop()
            
            # 验证数据库保存
            assert len(db_signals) > 0
            
            for signal in db_signals:
                assert signal.signal_id is not None
                assert signal.timestamp is not None
                assert signal.center_frequency == signal_config.acquisition.center_frequency
                assert signal.sample_rate == signal_config.acquisition.sample_rate
                assert signal.iq_data is not None
    
    @pytest.mark.integration
    def test_integration_with_websocket(self, signal_config, event_manager, signal_processor):
        """测试与WebSocket的集成"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            mock_device.read_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
            MockDevice.return_value = mock_device
            
            # 模拟WebSocket广播
            websocket_messages = []
            
            def broadcast_to_websocket(event):
                if event.get('type') == 'signal_acquired':
                    message = {
                        'type': 'signal_update',
                        'signal_id': event.get('signal_data').signal_id,
                        'timestamp': event.get('timestamp'),
                        'frequency': event.get('signal_data').center_freq
                    }
                    websocket_messages.append(message)
            
            event_manager.subscribe("signal_acquired", broadcast_to_websocket)
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 启动
            acquisition.start()
            time.sleep(0.5)
            acquisition.stop()
            
            # 验证WebSocket消息
            assert len(websocket_messages) > 0
            
            for message in websocket_messages:
                assert message['type'] == 'signal_update'
                assert 'signal_id' in message
                assert 'timestamp' in message
                assert 'frequency' in message


if __name__ == "__main__":
    """直接运行测试"""
    pytest.main([__file__, "-v", "--tb=short"])