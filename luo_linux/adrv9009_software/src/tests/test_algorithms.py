"""
算法测试模块
测试AlgorithmManager、信号处理算法、检测算法的功能
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
from scipy import fft

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig

# 导入被测试模块
from src.core.algorithm_manager import (
    AlgorithmManager,
    AlgorithmType,
    AlgorithmStatus,
    AlgorithmResult,
    AlgorithmConfig
)
from src.core.signal_processor import SignalProcessor, ProcessingConfig
from src.core.detection_engine import DetectionEngine, DetectionConfig
from src.core.event_manager import EventManager

# 导入算法实现
from src.algorithms.base_algorithm import BaseAlgorithm
from src.algorithms.energy_detector import EnergyDetector
from src.algorithms.spectrum_analyzer import SpectrumAnalyzer
from src.algorithms.modulation_detector import ModulationDetector
from src.algorithms.anomaly_detector import AnomalyDetector
from src.algorithms.signal_classifier import SignalClassifier
from src.algorithms.pulse_detector import PulseDetector
from src.algorithms.ofdm_detector import OFDMDetector
from src.algorithms.noise_estimator import NoiseEstimator
from src.algorithms.snr_estimator import SNREstimator
from src.algorithms.beamforming_processor import BeamformingProcessor

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq, SignalType, SignalMetrics
from src.models.detection_result import DetectionResult, create_detection_from_signal, DetectionType, ConfidenceLevel
from src.models.algorithm_result import AlgorithmMetrics, AlgorithmPerformance

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult, ValidationLevel
from src.utils.signal_utils import SignalUtils
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.file_utils import ensure_directory, get_file_info

# 导入测试fixtures
from src.tests.conftest import (
    signal_config,
    event_manager,
    mock_signal_data,
    mock_signal_dict,
    mock_signal_array,
    mock_noisy_signal,
    mock_detection_result,
    mock_detection_dict
)

# 设置测试日志
import logging
logger = logging.getLogger(__name__)


class TestBaseAlgorithm:
    """基础算法测试类"""
    
    def test_base_algorithm_initialization(self):
        """测试基础算法初始化"""
        # 创建基础算法实例
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION,
            version="1.0.0"
        )
        
        # 验证初始化
        assert algorithm.name == "test_algorithm"
        assert algorithm.algorithm_type == AlgorithmType.DETECTION
        assert algorithm.version == "1.0.0"
        assert algorithm.status == AlgorithmStatus.INITIALIZED
        assert algorithm.enabled is True
        assert algorithm.config == {}
    
    def test_base_algorithm_configuration(self):
        """测试基础算法配置"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 设置配置
        config = {
            "threshold": 0.5,
            "window_size": 1024,
            "overlap": 0.5
        }
        
        algorithm.configure(config)
        assert algorithm.config == config
        
        # 更新配置
        new_config = {"threshold": 0.7}
        algorithm.configure(new_config, merge=True)
        assert algorithm.config["threshold"] == 0.7
        assert algorithm.config["window_size"] == 1024
    
    def test_base_algorithm_status(self):
        """测试基础算法状态"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 验证初始状态
        assert algorithm.status == AlgorithmStatus.INITIALIZED
        assert algorithm.is_ready() is True
        
        # 更改状态
        algorithm.set_status(AlgorithmStatus.RUNNING)
        assert algorithm.status == AlgorithmStatus.RUNNING
        assert algorithm.is_running() is True
        
        # 重置状态
        algorithm.reset()
        assert algorithm.status == AlgorithmStatus.INITIALIZED
    
    def test_base_algorithm_execution(self):
        """测试基础算法执行"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 测试执行（应该被覆盖）
        with pytest.raises(NotImplementedError):
            algorithm.execute(None)
        
        # 测试异步执行
        with pytest.raises(NotImplementedError):
            algorithm.execute_async(None)
    
    def test_base_algorithm_validation(self):
        """测试基础算法验证"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 验证输入
        input_data = {"signal": np.array([1, 2, 3])}
        result = algorithm.validate_input(input_data)
        assert result.is_valid is True
        
        # 验证输出
        output_data = {"result": 0.5}
        result = algorithm.validate_output(output_data)
        assert result.is_valid is True
    
    def test_base_algorithm_metrics(self):
        """测试基础算法指标"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 获取指标
        metrics = algorithm.get_metrics()
        assert metrics is not None
        assert metrics.algorithm_name == "test_algorithm"
        assert metrics.execution_count == 0
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        
        # 记录执行
        algorithm.record_execution(success=True, execution_time=0.1)
        metrics = algorithm.get_metrics()
        assert metrics.execution_count == 1
        assert metrics.success_count == 1
        assert metrics.average_execution_time_ms == 100.0
    
    def test_base_algorithm_performance(self):
        """测试基础算法性能"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 记录性能
        for i in range(5):
            algorithm.record_execution(success=True, execution_time=0.1 + i * 0.01)
        
        # 获取性能指标
        performance = algorithm.get_performance()
        assert performance is not None
        assert performance.total_executions == 5
        assert performance.success_rate == 1.0
        assert 100 <= performance.average_time_ms <= 140
        
        # 获取历史记录
        history = algorithm.get_execution_history(limit=3)
        assert len(history) == 3
    
    def test_base_algorithm_reset(self):
        """测试基础算法重置"""
        algorithm = BaseAlgorithm(
            name="test_algorithm",
            algorithm_type=AlgorithmType.DETECTION
        )
        
        # 记录一些执行
        algorithm.record_execution(success=True, execution_time=0.1)
        algorithm.set_status(AlgorithmStatus.RUNNING)
        
        # 重置
        algorithm.reset()
        
        # 验证重置
        assert algorithm.status == AlgorithmStatus.INITIALIZED
        metrics = algorithm.get_metrics()
        assert metrics.execution_count == 0
        assert metrics.success_count == 0


class TestEnergyDetector:
    """能量检测器测试类"""
    
    def test_energy_detector_initialization(self):
        """测试能量检测器初始化"""
        detector = EnergyDetector()
        
        assert detector.name == "energy_detector"
        assert detector.algorithm_type == AlgorithmType.DETECTION
        assert detector.version is not None
        assert detector.enabled is True
        
        # 验证默认配置
        assert "threshold_db" in detector.config
        assert "window_size" in detector.config
        assert "fft_size" in detector.config
    
    def test_energy_detector_configuration(self):
        """测试能量检测器配置"""
        detector = EnergyDetector()
        
        # 自定义配置
        config = {
            "threshold_db": -50.0,
            "window_size": 2048,
            "fft_size": 4096,
            "overlap": 0.5
        }
        
        detector.configure(config)
        assert detector.config["threshold_db"] == -50.0
        assert detector.config["window_size"] == 2048
    
    def test_energy_detector_simple_signal(self, mock_signal_data):
        """测试能量检测器处理简单信号"""
        detector = EnergyDetector()
        
        # 处理信号
        result = detector.execute(mock_signal_data)
        
        # 验证结果
        assert result is not None
        assert "detected" in result
        assert "power_db" in result
        assert "snr_db" in result
        assert "confidence" in result
        
        # 验证检测结果
        assert isinstance(result["detected"], bool)
        assert isinstance(result["power_db"], float)
        assert isinstance(result["snr_db"], float)
        assert 0 <= result["confidence"] <= 1
    
    def test_energy_detector_no_signal(self):
        """测试能量检测器处理无信号"""
        detector = EnergyDetector()
        
        # 创建噪声信号
        noise = np.random.randn(1000) + 1j * np.random.randn(1000)
        noise_signal = create_signal_from_iq(
            iq_data=noise,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 处理信号
        result = detector.execute(noise_signal)
        
        # 验证结果
        assert result is not None
        # 纯噪声应该检测不到信号
        if result["detected"]:
            assert result["confidence"] < 0.5
    
    def test_energy_detector_strong_signal(self):
        """测试能量检测器处理强信号"""
        detector = EnergyDetector()
        
        # 创建强信号
        t = np.linspace(0, 0.001, 1000)
        strong_signal = 10.0 * np.exp(1j * 2 * np.pi * 1e6 * t)  # 20dB更强
        signal = create_signal_from_iq(
            iq_data=strong_signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 处理信号
        result = detector.execute(signal)
        
        # 验证结果
        assert result["detected"] is True
        assert result["confidence"] > 0.8
        assert result["power_db"] > -20  # 强信号功率较高
    
    def test_energy_detector_threshold_adjustment(self):
        """测试能量检测器阈值调整"""
        detector = EnergyDetector()
        
        # 创建中等强度信号
        t = np.linspace(0, 0.001, 1000)
        medium_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal = create_signal_from_iq(
            iq_data=medium_signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 低阈值检测
        detector.configure({"threshold_db": -80.0})
        result_low = detector.execute(signal)
        
        # 高阈值检测
        detector.configure({"threshold_db": -20.0})
        result_high = detector.execute(signal)
        
        # 验证阈值影响
        assert result_low["detected"] is True
        assert result_low["confidence"] > 0.5
        # 高阈值可能检测不到
        if not result_high["detected"]:
            assert result_high["confidence"] < 0.5
    
    def test_energy_detector_metrics(self, mock_signal_data):
        """测试能量检测器指标"""
        detector = EnergyDetector()
        
        # 处理多个信号
        for _ in range(3):
            result = detector.execute(mock_signal_data)
            assert result is not None
        
        # 获取指标
        metrics = detector.get_metrics()
        assert metrics.execution_count == 3
        assert metrics.success_count == 3
        assert metrics.average_execution_time_ms > 0
    
    def test_energy_detector_batch_processing(self):
        """测试能量检测器批处理"""
        detector = EnergyDetector()
        
        # 创建多个信号
        signals = []
        for i in range(5):
            t = np.linspace(0, 0.001, 1000)
            freq = 1e6 + i * 0.1e6
            iq_data = np.exp(1j * 2 * np.pi * freq * t)
            signal = create_signal_from_iq(
                iq_data=iq_data,
                sample_rate=10e6,
                center_freq=100e6
            )
            signals.append(signal)
        
        # 批处理
        results = []
        for signal in signals:
            result = detector.execute(signal)
            results.append(result)
        
        # 验证结果
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert "detected" in result
            assert "confidence" in result
    
    def test_energy_detector_real_time(self, mock_signal_data):
        """测试能量检测器实时处理"""
        detector = EnergyDetector()
        
        # 模拟实时处理
        processing_times = []
        for _ in range(10):
            start_time = time.time()
            result = detector.execute(mock_signal_data)
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            assert result is not None
        
        # 验证实时性能
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        logger.info(f"平均处理时间: {avg_time*1000:.2f} ms")
        logger.info(f"最大处理时间: {max_time*1000:.2f} ms")
        
        # 实时处理应该快速
        assert avg_time < 0.1  # 平均小于100ms
        assert max_time < 0.2  # 最大小于200ms


class TestSpectrumAnalyzer:
    """频谱分析器测试类"""
    
    def test_spectrum_analyzer_initialization(self):
        """测试频谱分析器初始化"""
        analyzer = SpectrumAnalyzer()
        
        assert analyzer.name == "spectrum_analyzer"
        assert analyzer.algorithm_type == AlgorithmType.ANALYSIS
        assert analyzer.version is not None
        
        # 验证默认配置
        assert "fft_size" in analyzer.config
        assert "window_type" in analyzer.config
        assert "overlap" in analyzer.config
    
    def test_spectrum_analyzer_configuration(self):
        """测试频谱分析器配置"""
        analyzer = SpectrumAnalyzer()
        
        # 自定义配置
        config = {
            "fft_size": 4096,
            "window_type": "hamming",
            "overlap": 0.75,
            "average_count": 10
        }
        
        analyzer.configure(config)
        assert analyzer.config["fft_size"] == 4096
        assert analyzer.config["window_type"] == "hamming"
        assert analyzer.config["overlap"] == 0.75
    
    def test_spectrum_analyzer_simple_signal(self, mock_signal_data):
        """测试频谱分析器处理简单信号"""
        analyzer = SpectrumAnalyzer()
        
        # 分析信号
        result = analyzer.execute(mock_signal_data)
        
        # 验证结果
        assert result is not None
        assert "spectrum" in result
        assert "frequencies" in result
        assert "peak_frequency" in result
        assert "peak_power" in result
        assert "bandwidth" in result
        
        # 验证数据类型
        spectrum = result["spectrum"]
        frequencies = result["frequencies"]
        
        assert isinstance(spectrum, np.ndarray)
        assert isinstance(frequencies, np.ndarray)
        assert len(spectrum) == len(frequencies)
        assert spectrum.ndim == 1
        
        # 验证峰值
        assert isinstance(result["peak_frequency"], float)
        assert isinstance(result["peak_power"], float)
        assert isinstance(result["bandwidth"], float)
    
    def test_spectrum_analyzer_multitone_signal(self):
        """测试频谱分析器处理多音信号"""
        analyzer = SpectrumAnalyzer()
        
        # 创建多音信号
        t = np.linspace(0, 0.001, 1000)
        signal1 = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal2 = 0.5 * np.exp(1j * 2 * np.pi * 1.5e6 * t)
        signal3 = 0.3 * np.exp(1j * 2 * np.pi * 2e6 * t)
        combined = signal1 + signal2 + signal3
        
        test_signal = create_signal_from_iq(
            iq_data=combined,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 分析信号
        result = analyzer.execute(test_signal)
        
        # 验证结果
        assert result is not None
        
        # 验证检测到多个峰值
        if "peak_frequencies" in result:
            peaks = result["peak_frequencies"]
            assert len(peaks) >= 2  # 应该检测到多个频率
    
    def test_spectrum_analyzer_window_functions(self, mock_signal_data):
        """测试频谱分析器不同窗函数"""
        analyzer = SpectrumAnalyzer()
        
        window_types = ["rectangular", "hann", "hamming", "blackman"]
        results = {}
        
        for window in window_types:
            analyzer.configure({"window_type": window})
            result = analyzer.execute(mock_signal_data)
            results[window] = result
        
        # 验证所有窗函数都有效
        for window, result in results.items():
            assert result is not None
            assert "spectrum" in result
            assert len(result["spectrum"]) > 0
        
        # 验证不同窗函数产生不同结果
        spectra = [r["spectrum"] for r in results.values()]
        for i in range(len(spectra) - 1):
            # 频谱形状应该相似但不完全相同
            assert spectra[i].shape == spectra[i + 1].shape
            assert not np.array_equal(spectra[i], spectra[i + 1])
    
    def test_spectrum_analyzer_fft_sizes(self, mock_signal_data):
        """测试频谱分析器不同FFT大小"""
        analyzer = SpectrumAnalyzer()
        
        fft_sizes = [256, 512, 1024, 2048]
        results = {}
        
        for fft_size in fft_sizes:
            analyzer.configure({"fft_size": fft_size})
            result = analyzer.execute(mock_signal_data)
            results[fft_size] = result
        
        # 验证所有FFT大小都有效
        for fft_size, result in results.items():
            assert result is not None
            spectrum = result["spectrum"]
            assert len(spectrum) == fft_size // 2 + 1
        
        # 验证频率分辨率
        resolutions = []
        for fft_size, result in results.items():
            freqs = result["frequencies"]
            if len(freqs) > 1:
                resolution = freqs[1] - freqs[0]
                resolutions.append(resolution)
        
        # 更大的FFT应该提供更好的频率分辨率
        assert all(resolutions[i] <= resolutions[i + 1] for i in range(len(resolutions) - 1))
    
    def test_spectrum_analyzer_spectrum_metrics(self, mock_signal_data):
        """测试频谱分析器频谱指标"""
        analyzer = SpectrumAnalyzer()
        
        result = analyzer.execute(mock_signal_data)
        
        # 验证频谱指标
        spectrum = result["spectrum"]
        
        assert np.all(np.isfinite(spectrum))
        assert np.all(spectrum <= 0)  # dB值应为负数或零
        
        # 验证动态范围
        dynamic_range = np.max(spectrum) - np.min(spectrum)
        assert dynamic_range > 0
        
        # 验证噪声基底
        if "noise_floor" in result:
            noise_floor = result["noise_floor"]
            assert noise_floor < 0  # dB值应为负数
    
    def test_spectrum_analyzer_peak_detection(self):
        """测试频谱分析器峰值检测"""
        analyzer = SpectrumAnalyzer()
        
        # 创建有明显峰值的信号
        t = np.linspace(0, 0.001, 1000)
        clean_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        test_signal = create_signal_from_iq(
            iq_data=clean_signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        result = analyzer.execute(test_signal)
        
        # 验证峰值检测
        peak_freq = result["peak_frequency"]
        peak_power = result["peak_power"]
        
        # 峰值频率应该在1MHz附近
        expected_freq = 1e6
        tolerance = 0.1e6  # 100kHz容差
        assert abs(peak_freq - expected_freq) <= tolerance
        
        # 峰值功率应该相对较高
        assert peak_power > -50  # 大于-50dB
    
    def test_spectrum_analyzer_bandwidth_estimation(self):
        """测试频谱分析器带宽估计"""
        analyzer = SpectrumAnalyzer()
        
        # 创建带限信号
        t = np.linspace(0, 0.001, 1000)
        
        # 添加多个频率分量模拟带宽
        base_freq = 1e6
        bandwidth = 0.2e6  # 200kHz带宽
        
        signal = np.zeros_like(t, dtype=np.complex128)
        for freq_offset in np.linspace(-bandwidth/2, bandwidth/2, 5):
            freq = base_freq + freq_offset
            signal += np.exp(1j * 2 * np.pi * freq * t)
        
        test_signal = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        result = analyzer.execute(test_signal)
        
        # 验证带宽估计
        if "bandwidth" in result:
            estimated_bw = result["bandwidth"]
            # 估计带宽应该在真实带宽附近
            assert 0.1e6 <= estimated_bw <= 0.3e6
    
    def test_spectrum_analyzer_averaging(self, mock_signal_data):
        """测试频谱分析器平均"""
        analyzer = SpectrumAnalyzer()
        
        # 配置平均
        analyzer.configure({
            "average_count": 10,
            "average_type": "moving"
        })
        
        # 多次处理同一个信号
        results = []
        for _ in range(20):
            result = analyzer.execute(mock_signal_data)
            results.append(result)
        
        # 验证平均效果
        spectra = [r["spectrum"] for r in results]
        
        # 计算方差
        variances = [np.var(spectrum) for spectrum in spectra]
        
        # 平均应该减少方差
        logger.info(f"频谱方差: {np.mean(variances):.2f}")
    
    def test_spectrum_analyzer_performance(self, mock_signal_data):
        """测试频谱分析器性能"""
        analyzer = SpectrumAnalyzer()
        
        # 测试不同FFT大小的性能
        fft_sizes = [256, 512, 1024, 2048, 4096]
        
        performance_results = {}
        for fft_size in fft_sizes:
            analyzer.configure({"fft_size": fft_size})
            
            # 计时
            start_time = time.time()
            for _ in range(10):
                analyzer.execute(mock_signal_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            performance_results[fft_size] = avg_time
        
        # 记录性能
        for fft_size, avg_time in performance_results.items():
            logger.info(f"FFT大小 {fft_size}: {avg_time*1000:.2f} ms")
        
        # 验证性能趋势
        sizes = list(performance_results.keys())
        times = list(performance_results.values())
        
        # 更大的FFT应该需要更多时间
        for i in range(len(sizes) - 1):
            if sizes[i] < sizes[i + 1]:
                assert times[i] <= times[i + 1] * 1.5  # 允许一些波动


class TestModulationDetector:
    """调制检测器测试类"""
    
    def test_modulation_detector_initialization(self):
        """测试调制检测器初始化"""
        detector = ModulationDetector()
        
        assert detector.name == "modulation_detector"
        assert detector.algorithm_type == AlgorithmType.CLASSIFICATION
        assert detector.version is not None
        
        # 验证默认配置
        assert "min_confidence" in detector.config
        assert "feature_thresholds" in detector.config
    
    def test_modulation_detector_am_signal(self):
        """测试调制检测器AM信号检测"""
        detector = ModulationDetector()
        
        # 创建AM信号
        t = np.linspace(0, 0.001, 1000)
        carrier_freq = 1e6
        modulation_freq = 10e3
        modulation_index = 0.8
        
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        modulating = 1 + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
        am_signal = carrier * modulating
        
        test_signal = create_signal_from_iq(
            iq_data=am_signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 检测调制
        result = detector.execute(test_signal)
        
        # 验证结果
        assert result is not None
        assert "modulation_type" in result
        assert "confidence" in result
        assert "features" in result
        
        # 应该是AM调制
        if result["confidence"] > 0.5:
            assert result["modulation_type"] == "AM"
    
    def test_modulation_detector_fm_signal(self):
        """测试调制检测器FM信号检测"""
        detector = ModulationDetector()
        
        # 创建FM信号
        t = np.linspace(0, 0.001, 1000)
        carrier_freq = 1e6
        modulation_freq = 10e3
        modulation_index = 5.0
        
        phase = 2 * np.pi * carrier_freq * t + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
        fm_signal = np.exp(1j * phase)
        
        test_signal = create_signal_from_iq(
            iq_data=fm_signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 检测调制
        result = detector.execute(test_signal)
        
        # 验证结果
        if result["confidence"] > 0.5:
            assert result["modulation_type"] in ["FM", "PM"]
    
    def test_modulation_detector_qpsk_signal(self):
        """测试调制检测器QPSK信号检测"""
        detector = ModulationDetector()
        
        # 创建QPSK信号
        num_symbols = 100
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
        
        # 升采样
        samples_per_symbol = 10
        symbols_upsampled = np.repeat(symbols, samples_per_symbol)
        
        # 添加脉冲成形
        from scipy import signal
        beta = 0.5
        sps = samples_per_symbol
        t = np.linspace(-2, 2, 4*sps + 1)
        h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
        h = h / np.sqrt(np.sum(h**2))
        
        # 滤波
        shaped = signal.convolve(symbols_upsampled, h, mode='same')
        
        test_signal = create_signal_from_iq(
            iq_data=shaped[:1000],
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 检测调制
        result = detector.execute(test_signal)
        
        # 验证结果
        if result["confidence"] > 0.5:
            assert result["modulation_type"] in ["QPSK", "PSK"]
    
    def test_modulation_detector_constellation_features(self):
        """测试调制检测器星座图特征"""
        detector = ModulationDetector()
        
        # 创建不同调制类型的信号
        modulations = ["AM", "FM", "QPSK"]
        
        for mod_type in modulations:
            # 创建测试信号
            t = np.linspace(0, 0.001, 1000)
            
            if mod_type == "AM":
                signal = np.exp(1j * 2 * np.pi * 1e6 * t) * (1 + 0.5 * np.sin(2 * np.pi * 10e3 * t))
            elif mod_type == "FM":
                phase = 2 * np.pi * 1e6 * t + 5.0 * np.sin(2 * np.pi * 10e3 * t)
                signal = np.exp(1j * phase)
            else:  # QPSK
                symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 100)
                signal = np.repeat(symbols, 10)[:1000]
            
            test_signal = create_signal_from_iq(
                iq_data=signal,
                sample_rate=10e6,
                center_freq=100e6
            )
            
            # 检测调制
            result = detector.execute(test_signal)
            
            # 验证特征提取
            features = result["features"]
            assert isinstance(features, dict)
            assert "constellation_variance" in features
            assert "phase_variance" in features
            assert "amplitude_variance" in features
    
    def test_modulation_detector_confidence_calibration(self):
        """测试调制检测器置信度校准"""
        detector = ModulationDetector()
        
        # 创建不同质量的信号
        qualities = [0.2, 0.5, 0.8]  # SNR水平
        
        for quality in qualities:
            t = np.linspace(0, 0.001, 1000)
            
            # 创建FM信号
            phase = 2 * np.pi * 1e6 * t + 5.0 * np.sin(2 * np.pi * 10e3 * t)
            clean_signal = np.exp(1j * phase)
            
            # 添加噪声
            noise_level = 1.0 - quality
            noise = noise_level * (np.random.randn(1000) + 1j * np.random.randn(1000))
            noisy_signal = clean_signal + noise
            
            test_signal = create_signal_from_iq(
                iq_data=noisy_signal,
                sample_rate=10e6,
                center_freq=100e6
            )
            
            # 检测调制
            result = detector.execute(test_signal)
            
            # 验证置信度与质量相关
            confidence = result["confidence"]
            
            if quality > 0.5:
                assert confidence > 0.5
            else:
                assert confidence < 0.7
    
    def test_modulation_detector_unknown_modulation(self, mock_signal_data):
        """测试调制检测器未知调制"""
        detector = ModulationDetector()
        
        # 处理简单正弦波（无调制）
        result = detector.execute(mock_signal_data)
        
        # 验证结果
        assert result is not None
        
        # 可能是未知或CW
        if result["modulation_type"] == "unknown":
            assert result["confidence"] < 0.5
        else:
            # 如果检测到某种调制，置信度应该较低
            assert result["confidence"] < 0.7


class TestAlgorithmManager:
    """算法管理器测试类"""
    
    def test_algorithm_manager_initialization(self, signal_config, event_manager):
        """测试算法管理器初始化"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 验证初始化
        assert manager.config == signal_config.algorithms
        assert manager.event_manager == event_manager
        assert manager.initialized is True
        
        # 验证算法加载
        algorithms = manager.get_algorithms()
        assert len(algorithms) > 0
        
        # 验证默认算法
        assert manager.default_algorithm is not None
    
    def test_algorithm_manager_algorithm_registration(self, signal_config, event_manager):
        """测试算法管理器算法注册"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建自定义算法
        class CustomAlgorithm(BaseAlgorithm):
            def __init__(self):
                super().__init__("custom_algorithm", AlgorithmType.CUSTOM)
            
            def execute(self, data):
                return {"result": "custom"}
        
        # 注册算法
        custom_algo = CustomAlgorithm()
        manager.register_algorithm(custom_algo)
        
        # 验证注册
        algorithms = manager.get_algorithms()
        assert "custom_algorithm" in algorithms
        
        # 执行算法
        result = manager.execute_algorithm("custom_algorithm", {})
        assert result["result"] == "custom"
    
    def test_algorithm_manager_algorithm_execution(self, signal_config, event_manager, mock_signal_data):
        """测试算法管理器算法执行"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 执行默认算法
        result = manager.execute_default(mock_signal_data)
        assert result is not None
        
        # 执行特定算法
        result = manager.execute_algorithm("energy_detector", mock_signal_data)
        assert result is not None
        
        # 执行不存在的算法
        with pytest.raises(ValueError, match="算法不存在"):
            manager.execute_algorithm("non_existent", mock_signal_data)
    
    def test_algorithm_manager_batch_execution(self, signal_config, event_manager):
        """测试算法管理器批处理执行"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建多个信号
        signals = []
        for i in range(3):
            t = np.linspace(0, 0.001, 1000)
            freq = 1e6 + i * 0.1e6
            iq_data = np.exp(1j * 2 * np.pi * freq * t)
            signal = create_signal_from_iq(
                iq_data=iq_data,
                sample_rate=10e6,
                center_freq=100e6
            )
            signals.append(signal)
        
        # 批处理
        results = manager.execute_batch("energy_detector", signals)
        
        # 验证结果
        assert len(results) == len(signals)
        for result in results:
            assert result is not None
            assert "detected" in result
    
    def test_algorithm_manager_parallel_execution(self, signal_config, event_manager):
        """测试算法管理器并行执行"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建多个信号
        num_signals = 10
        signals = []
        for i in range(num_signals):
            t = np.linspace(0, 0.001, 1000)
            freq = 1e6 + i * 0.1e6
            iq_data = np.exp(1j * 2 * np.pi * freq * t)
            signal = create_signal_from_iq(
                iq_data=iq_data,
                sample_rate=10e6,
                center_freq=100e6
            )
            signals.append(signal)
        
        # 串行执行计时
        start_serial = time.time()
        serial_results = []
        for signal in signals:
            result = manager.execute_algorithm("energy_detector", signal)
            serial_results.append(result)
        end_serial = time.time()
        serial_time = end_serial - start_serial
        
        # 并行执行计时
        start_parallel = time.time()
        parallel_results = manager.execute_parallel("energy_detector", signals, max_workers=4)
        end_parallel = time.time()
        parallel_time = end_parallel - start_parallel
        
        # 验证结果
        assert len(parallel_results) == len(serial_results)
        
        # 记录性能
        logger.info(f"串行时间: {serial_time:.3f}s")
        logger.info(f"并行时间: {parallel_time:.3f}s")
        logger.info(f"加速比: {serial_time/parallel_time:.2f}x")
        
        # 并行应该更快（对于足够多的任务）
        if num_signals >= 4:
            assert parallel_time < serial_time * 1.5
    
    def test_algorithm_manager_algorithm_status(self, signal_config, event_manager):
        """测试算法管理器算法状态"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 获取算法状态
        status = manager.get_algorithm_status("energy_detector")
        assert status is not None
        assert "name" in status
        assert "status" in status
        assert "execution_count" in status
        
        # 获取所有算法状态
        all_status = manager.get_all_algorithm_status()
        assert len(all_status) > 0
        assert "energy_detector" in all_status
    
    def test_algorithm_manager_algorithm_enable_disable(self, signal_config, event_manager):
        """测试算法管理器启用/禁用算法"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 禁用算法
        manager.disable_algorithm("energy_detector")
        
        # 验证禁用
        status = manager.get_algorithm_status("energy_detector")
        assert status["enabled"] is False
        
        # 尝试执行禁用的算法
        with pytest.raises(ValueError, match="算法已禁用"):
            manager.execute_algorithm("energy_detector", {})
        
        # 启用算法
        manager.enable_algorithm("energy_detector")
        
        # 验证启用
        status = manager.get_algorithm_status("energy_detector")
        assert status["enabled"] is True
        
        # 应该可以执行
        result = manager.execute_algorithm("energy_detector", {})
        assert result is not None
    
    def test_algorithm_manager_algorithm_configuration(self, signal_config, event_manager):
        """测试算法管理器算法配置"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 配置算法
        new_config = {"threshold_db": -60.0, "window_size": 2048}
        manager.configure_algorithm("energy_detector", new_config)
        
        # 验证配置
        config = manager.get_algorithm_config("energy_detector")
        assert config["threshold_db"] == -60.0
        assert config["window_size"] == 2048
    
    def test_algorithm_manager_algorithm_chain(self, signal_config, event_manager, mock_signal_data):
        """测试算法管理器算法链"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 定义算法链
        chain = ["energy_detector", "spectrum_analyzer"]
        
        # 执行算法链
        results = manager.execute_chain(chain, mock_signal_data)
        
        # 验证结果
        assert isinstance(results, dict)
        assert len(results) == len(chain)
        
        for algo_name in chain:
            assert algo_name in results
            assert results[algo_name] is not None
    
    def test_algorithm_manager_algorithm_metrics(self, signal_config, event_manager, mock_signal_data):
        """测试算法管理器算法指标"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 执行多次算法
        for _ in range(5):
            manager.execute_algorithm("energy_detector", mock_signal_data)
        
        # 获取算法指标
        metrics = manager.get_algorithm_metrics("energy_detector")
        assert metrics is not None
        assert metrics.execution_count >= 5
        
        # 获取所有算法指标
        all_metrics = manager.get_all_algorithm_metrics()
        assert len(all_metrics) > 0
    
    def test_algorithm_manager_performance_monitoring(self, signal_config, event_manager, mock_signal_data):
        """测试算法管理器性能监控"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 执行算法多次以收集性能数据
        execution_times = []
        for _ in range(20):
            start_time = time.time()
            manager.execute_algorithm("energy_detector", mock_signal_data)
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        # 获取性能统计
        performance = manager.get_algorithm_performance("energy_detector")
        assert performance is not None
        assert performance.total_executions >= 20
        assert performance.average_time_ms > 0
        
        # 验证性能一致性
        avg_time = sum(execution_times) / len(execution_times)
        std_time = np.std(execution_times)
        
        logger.info(f"平均执行时间: {avg_time*1000:.2f} ms")
        logger.info(f"执行时间标准差: {std_time*1000:.2f} ms")
        
        # 执行时间应该相对稳定
        assert std_time < avg_time * 0.5  # 标准差小于平均值的50%
    
    def test_algorithm_manager_error_handling(self, signal_config, event_manager):
        """测试算法管理器错误处理"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建会抛出异常的算法
        class FailingAlgorithm(BaseAlgorithm):
            def __init__(self):
                super().__init__("failing_algorithm", AlgorithmType.CUSTOM)
            
            def execute(self, data):
                raise ValueError("模拟算法错误")
        
        # 注册并执行
        failing_algo = FailingAlgorithm()
        manager.register_algorithm(failing_algo)
        
        # 执行应该捕获异常
        result = manager.execute_algorithm("failing_algorithm", {})
        
        # 验证错误处理
        assert result is not None
        assert "error" in result
        assert "模拟算法错误" in result["error"]
        
        # 验证错误计数
        metrics = manager.get_algorithm_metrics("failing_algorithm")
        assert metrics.error_count > 0
    
    def test_algorithm_manager_event_publication(self, signal_config, event_manager, mock_signal_data):
        """测试算法管理器事件发布"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 监听事件
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        event_manager.subscribe("algorithm_started", event_handler)
        event_manager.subscribe("algorithm_completed", event_handler)
        event_manager.subscribe("algorithm_failed", event_handler)
        
        # 执行算法
        manager.execute_algorithm("energy_detector", mock_signal_data)
        
        # 验证事件
        assert len(events_received) >= 2
        
        event_types = [e.get("type") for e in events_received]
        assert "algorithm_started" in event_types
        assert "algorithm_completed" in event_types
        
        # 验证事件数据
        for event in events_received:
            if event["type"] == "algorithm_started":
                assert event["algorithm"] == "energy_detector"
            elif event["type"] == "algorithm_completed":
                assert event["algorithm"] == "energy_detector"
                assert "result" in event
    
    def test_algorithm_manager_resource_management(self, signal_config, event_manager):
        """测试算法管理器资源管理"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 获取资源使用
        resources = manager.get_resource_usage()
        assert resources is not None
        assert "memory_mb" in resources
        assert "cpu_percent" in resources
        assert "thread_count" in resources
        
        # 验证资源限制
        assert resources["memory_mb"] >= 0
        assert 0 <= resources["cpu_percent"] <= 100
    
    def test_algorithm_manager_algorithm_validation(self, signal_config, event_manager):
        """测试算法管理器算法验证"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 验证算法
        validation = manager.validate_algorithm("energy_detector")
        assert validation is not None
        assert "valid" in validation
        assert validation["valid"] is True
        
        # 验证所有算法
        all_validation = manager.validate_all_algorithms()
        assert len(all_validation) > 0
        
        for algo_name, validation_result in all_validation.items():
            assert "valid" in validation_result
            if not validation_result["valid"]:
                assert "errors" in validation_result


class TestSignalProcessingAlgorithms:
    """信号处理算法测试类"""
    
    def test_anomaly_detector(self, mock_noisy_signal):
        """测试异常检测器"""
        detector = AnomalyDetector()
        
        # 配置检测器
        detector.configure({
            "threshold": 3.0,  # 3倍标准差
            "window_size": 100
        })
        
        # 检测异常
        result = detector.execute(mock_noisy_signal)
        
        # 验证结果
        assert result is not None
        assert "anomalies_detected" in result
        assert "anomaly_locations" in result
        assert "anomaly_scores" in result
        
        # 对于正常信号，应该没有异常
        anomalies = result["anomalies_detected"]
        assert isinstance(anomalies, bool)
    
    def test_signal_classifier(self, mock_signal_data):
        """测试信号分类器"""
        classifier = SignalClassifier()
        
        # 分类信号
        result = classifier.execute(mock_signal_data)
        
        # 验证结果
        assert result is not None
        assert "signal_type" in result
        assert "confidence" in result
        assert "features" in result
        
        # 验证分类
        assert isinstance(result["signal_type"], str)
        assert 0 <= result["confidence"] <= 1
        
        # 验证特征
        features = result["features"]
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_pulse_detector(self):
        """测试脉冲检测器"""
        detector = PulseDetector()
        
        # 创建脉冲信号
        t = np.linspace(0, 0.001, 1000)
        
        # 添加脉冲
        signal = np.zeros_like(t, dtype=np.complex128)
        pulse_start = 300
        pulse_end = 350
        signal[pulse_start:pulse_end] = np.exp(1j * 2 * np.pi * 1e6 * t[:50])
        
        # 添加一些噪声
        noise = 0.1 * (np.random.randn(1000) + 1j * np.random.randn(1000))
        signal += noise
        
        test_signal = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 检测脉冲
        result = detector.execute(test_signal)
        
        # 验证结果
        assert result is not None
        assert "pulses_detected" in result
        assert "pulse_count" in result
        assert "pulse_durations" in result
        assert "pulse_powers" in result
        
        if result["pulses_detected"]:
            assert result["pulse_count"] > 0
            assert len(result["pulse_durations"]) == result["pulse_count"]
            assert len(result["pulse_powers"]) == result["pulse_count"]
    
    def test_ofdm_detector(self):
        """测试OFDM检测器"""
        detector = OFDMDetector()
        
        # 创建OFDM-like信号
        t = np.linspace(0, 0.001, 1000)
        
        # 多个子载波
        num_subcarriers = 10
        signal = np.zeros_like(t, dtype=np.complex128)
        
        for i in range(num_subcarriers):
            freq = 1e6 + i * 0.1e6
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            signal += amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))
        
        test_signal = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 检测OFDM
        result = detector.execute(test_signal)
        
        # 验证结果
        assert result is not None
        assert "is_ofdm" in result
        assert "confidence" in result
        assert "estimated_subcarriers" in result
        assert "cyclic_prefix_ratio" in result
        
        # 对于这个简单信号，可能检测不到OFDM
        # 但算法应该运行完成
    
    def test_noise_estimator(self, mock_noisy_signal):
        """测试噪声估计器"""
        estimator = NoiseEstimator()
        
        # 估计噪声
        result = estimator.execute(mock_noisy_signal)
        
        # 验证结果
        assert result is not None
        assert "noise_power_db" in result
        assert "noise_floor_db" in result
        assert "noise_variance" in result
        assert "noise_distribution" in result
        
        # 验证噪声功率
        assert result["noise_power_db"] < 0  # dB值为负
        assert result["noise_floor_db"] < 0
        
        # 验证噪声方差
        assert result["noise_variance"] > 0
    
    def test_snr_estimator(self, mock_noisy_signal):
        """测试SNR估计器"""
        estimator = SNREstimator()
        
        # 估计SNR
        result = estimator.execute(mock_noisy_signal)
        
        # 验证结果
        assert result is not None
        assert "snr_db" in result
        assert "signal_power_db" in result
        assert "noise_power_db" in result
        assert "estimation_method" in result
        
        # 验证SNR
        assert isinstance(result["snr_db"], float)
        
        # 验证功率
        signal_power = result["signal_power_db"]
        noise_power = result["noise_power_db"]
        
        # SNR = 信号功率 - 噪声功率
        calculated_snr = signal_power - noise_power
        assert abs(result["snr_db"] - calculated_snr) < 1.0  # 允许1dB误差
    
    def test_beamforming_processor(self):
        """测试波束形成处理器"""
        processor = BeamformingProcessor()
        
        # 创建多通道信号
        num_channels = 4
        num_samples = 1000
        
        # 生成信号
        t = np.linspace(0, 0.001, num_samples)
        signals = []
        
        for ch in range(num_channels):
            # 每个通道有轻微延迟
            delay = ch * 1e-7  # 微小延迟
            freq = 1e6
            
            signal = np.exp(1j * 2 * np.pi * freq * (t - delay))
            noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            signals.append(signal + noise)
        
        # 转换为多通道信号
        multi_channel_signal = np.array(signals).T  # 形状: (samples, channels)
        
        # 创建信号数据
        from src.models.signal_data import SignalData
        signal_data = SignalData(
            signal_id="test_multi_channel",
            iq_data=multi_channel_signal,
            sample_rate=10e6,
            center_freq=100e6,
            timestamp=datetime.now()
        )
        signal_data.metadata["channels"] = num_channels
        
        # 配置波束形成
        processor.configure({
            "direction": 30.0,  # 30度方向
            "method": "MVDR"  # 最小方差无失真响应
        })
        
        # 执行波束形成
        result = processor.execute(signal_data)
        
        # 验证结果
        assert result is not None
        assert "beamformed_signal" in result
        assert "direction_response" in result
        assert "weights" in result
        
        # 验证波束形成信号
        beamformed = result["beamformed_signal"]
        assert isinstance(beamformed, np.ndarray)
        assert beamformed.ndim == 1
        assert len(beamformed) == num_samples
        
        # 验证权重
        weights = result["weights"]
        assert isinstance(weights, np.ndarray)
        assert len(weights) == num_channels


class TestAlgorithmIntegration:
    """算法集成测试类"""
    
    def test_algorithm_pipeline(self, signal_config, event_manager, mock_signal_data):
        """测试算法流水线"""
        # 创建算法管理器
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 定义处理流水线
        pipeline = [
            ("energy_detector", {"threshold_db": -50.0}),
            ("spectrum_analyzer", {"fft_size": 1024}),
            ("modulation_detector", {"min_confidence": 0.3})
        ]
        
        # 执行流水线
        results = {}
        for algo_name, algo_config in pipeline:
            # 配置算法
            if algo_config:
                manager.configure_algorithm(algo_name, algo_config)
            
            # 执行算法
            result = manager.execute_algorithm(algo_name, mock_signal_data)
            results[algo_name] = result
        
        # 验证结果
        assert len(results) == len(pipeline)
        
        for algo_name, result in results.items():
            assert result is not None
            
            # 验证算法特定结果
            if algo_name == "energy_detector":
                assert "detected" in result
            elif algo_name == "spectrum_analyzer":
                assert "spectrum" in result
            elif algo_name == "modulation_detector":
                assert "modulation_type" in result
    
    def test_algorithm_fusion(self, signal_config, event_manager, mock_signal_data):
        """测试算法融合"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 使用多个算法检测同一信号
        algorithms = ["energy_detector", "anomaly_detector", "signal_classifier"]
        
        # 并行执行
        import concurrent.futures
        
        def execute_algorithm(algo_name):
            return manager.execute_algorithm(algo_name, mock_signal_data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(algorithms)) as executor:
            future_to_algo = {
                executor.submit(execute_algorithm, algo_name): algo_name 
                for algo_name in algorithms
            }
            
            results = {}
            for future in concurrent.futures.as_completed(future_to_algo):
                algo_name = future_to_algo[future]
                try:
                    result = future.result()
                    results[algo_name] = result
                except Exception as e:
                    results[algo_name] = {"error": str(e)}
        
        # 验证所有算法都完成
        assert len(results) == len(algorithms)
        
        # 融合结果
        detection_results = []
        for algo_name, result in results.items():
            if "error" not in result:
                if algo_name == "energy_detector" and "detected" in result:
                    detection_results.append({
                        "algorithm": algo_name,
                        "detected": result["detected"],
                        "confidence": result.get("confidence", 0.5)
                    })
                elif algo_name == "anomaly_detector" and "anomalies_detected" in result:
                    detection_results.append({
                        "algorithm": algo_name,
                        "detected": result["anomalies_detected"],
                        "confidence": 0.7 if result["anomalies_detected"] else 0.3
                    })
        
        # 基于融合结果做决策
        if detection_results:
            total_confidence = sum(r["confidence"] for r in detection_results if r["detected"])
            avg_confidence = total_confidence / len(detection_results)
            
            logger.info(f"融合置信度: {avg_confidence:.3f}")
    
    def test_algorithm_performance_comparison(self, signal_config, event_manager):
        """测试算法性能比较"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建测试信号
        t = np.linspace(0, 0.001, 1000)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        test_signal = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 测试算法性能
        algorithms = ["energy_detector", "spectrum_analyzer", "modulation_detector"]
        
        performance_results = {}
        for algo_name in algorithms:
            execution_times = []
            
            # 热身
            for _ in range(5):
                manager.execute_algorithm(algo_name, test_signal)
            
            # 正式测试
            for _ in range(20):
                start_time = time.time()
                manager.execute_algorithm(algo_name, test_signal)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # 计算统计
            avg_time = sum(execution_times) / len(execution_times) * 1000  # 转换为ms
            min_time = min(execution_times) * 1000
            max_time = max(execution_times) * 1000
            std_time = np.std(execution_times) * 1000
            
            performance_results[algo_name] = {
                "avg_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "std_ms": std_time
            }
        
        # 记录性能比较
        for algo_name, perf in performance_results.items():
            logger.info(f"{algo_name}: {perf['avg_ms']:.2f} ± {perf['std_ms']:.2f} ms")
    
    def test_algorithm_resource_usage_comparison(self, signal_config, event_manager, mock_signal_data):
        """测试算法资源使用比较"""
        import psutil
        import os
        
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        process = psutil.Process(os.getpid())
        
        algorithms = ["energy_detector", "spectrum_analyzer", "modulation_detector"]
        
        resource_results = {}
        for algo_name in algorithms:
            # 测量内存使用
            initial_memory = process.memory_info().rss
            
            # 执行算法多次
            for _ in range(10):
                manager.execute_algorithm(algo_name, mock_signal_data)
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
            
            # 测量CPU使用
            initial_cpu = process.cpu_percent(interval=0.1)
            
            # 执行算法
            for _ in range(5):
                manager.execute_algorithm(algo_name, mock_signal_data)
            
            current_cpu = process.cpu_percent(interval=0.1)
            cpu_increase = current_cpu - initial_cpu
            
            resource_results[algo_name] = {
                "memory_increase_mb": memory_increase,
                "cpu_increase_percent": cpu_increase
            }
        
        # 记录资源使用
        for algo_name, resources in resource_results.items():
            logger.info(f"{algo_name}: 内存增加 {resources['memory_increase_mb']:.2f} MB, "
                       f"CPU增加 {resources['cpu_increase_percent']:.1f}%")
    
    def test_algorithm_error_recovery(self, signal_config, event_manager):
        """测试算法错误恢复"""
        manager = AlgorithmManager(
            config=signal_config.algorithms,
            event_manager=event_manager
        )
        
        # 创建会随机失败的算法
        class FlakyAlgorithm(BaseAlgorithm):
            def __init__(self):
                super().__init__("flaky_algorithm", AlgorithmType.CUSTOM)
                self.call_count = 0
            
            def execute(self, data):
                self.call_count += 1
                if self.call_count % 3 == 0:  # 每3次失败一次
                    raise ValueError(f"第{self.call_count}次调用失败")
                return {"success": True, "call": self.call_count}
        
        # 注册算法
        flaky_algo = FlakyAlgorithm()
        manager.register_algorithm(flaky_algo)
        
        # 执行多次，应该能恢复
        results = []
        for i in range(10):
            try:
                result = manager.execute_algorithm("flaky_algorithm", {})
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # 验证错误恢复
        success_count = sum(1 for r in results if "error" not in r)
        error_count = sum(1 for r in results if "error" in r)
        
        logger.info(f"成功: {success_count}, 失败: {error_count}")
        
        # 应该有成功也有失败
        assert success_count > 0
        assert error_count > 0


if __name__ == "__main__":
    """直接运行测试"""
    pytest.main([__file__, "-v", "--tb=short"])