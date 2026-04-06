"""
性能测试模块
测试系统的性能、吞吐量、延迟、资源使用等关键指标
"""
import os
import sys
import time
import json
import asyncio
import tempfile
import warnings
import statistics
import psutil
import resource
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Generator, Tuple
from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import gc
import tracemalloc
from memory_profiler import memory_usage
import logging
from logging.handlers import RotatingFileHandler

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig

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

# 导入算法
from src.algorithms.energy_detector import EnergyDetector
from src.algorithms.spectrum_analyzer import SpectrumAnalyzer
from src.algorithms.modulation_detector import ModulationDetector
from src.algorithms.anomaly_detector import AnomalyDetector
from src.algorithms.signal_classifier import SignalClassifier

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult, ValidationLevel
from src.utils.signal_utils import SignalUtils
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.file_utils import ensure_directory, get_file_info

# 设置测试日志
logger = logging.getLogger(__name__)

# 性能测试配置
class PerformanceConfig:
    """性能测试配置"""
    
    # 测试持续时间
    SHORT_TEST_DURATION = 5.0  # 秒
    MEDIUM_TEST_DURATION = 30.0  # 秒
    LONG_TEST_DURATION = 120.0  # 秒
    
    # 信号参数
    SMALL_SIGNAL_SIZE = 1024  # 样本
    MEDIUM_SIGNAL_SIZE = 8192  # 样本
    LARGE_SIGNAL_SIZE = 65536  # 样本
    VERY_LARGE_SIGNAL_SIZE = 262144  # 样本
    
    # 采样率
    LOW_SAMPLE_RATE = 1e6  # 1 MHz
    MEDIUM_SAMPLE_RATE = 10e6  # 10 MHz
    HIGH_SAMPLE_RATE = 50e6  # 50 MHz
    
    # 性能阈值
    MAX_PROCESSING_LATENCY_MS = 1000  # 最大处理延迟
    MAX_ACQUISITION_LATENCY_MS = 100  # 最大采集延迟
    MAX_CPU_USAGE_PERCENT = 80  # 最大CPU使用率
    MAX_MEMORY_INCREASE_MB = 500  # 最大内存增加
    MIN_THROUGHPUT_SIGNALS_PER_SEC = 1.0  # 最小吞吐量
    
    # 测试重复次数
    WARMUP_ITERATIONS = 5
    MEASUREMENT_ITERATIONS = 20
    
    # 并发级别
    LOW_CONCURRENCY = 2
    MEDIUM_CONCURRENCY = 8
    HIGH_CONCURRENCY = 32


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, name: str):
        """初始化性能基准测试
        
        Args:
            name: 测试名称
        """
        self.name = name
        self.results = []
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.cpu_samples = []
        self.memory_samples = []
        
    def start(self):
        """开始测试"""
        self.start_time = time.time()
        self.performance_monitor.start()
        
        # 开始采样
        self._start_sampling()
        
    def stop(self):
        """停止测试"""
        self.end_time = time.time()
        self.performance_monitor.stop()
        self._stop_sampling()
        
    def _start_sampling(self):
        """开始资源采样"""
        self.cpu_samples = []
        self.memory_samples = []
        
        # 创建采样线程
        self.sampling_running = True
        self.sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampling_thread.start()
        
    def _stop_sampling(self):
        """停止资源采样"""
        self.sampling_running = False
        if hasattr(self, 'sampling_thread'):
            self.sampling_thread.join(timeout=2.0)
        
    def _sampling_loop(self):
        """资源采样循环"""
        process = psutil.Process(os.getpid())
        while self.sampling_running:
            try:
                # CPU使用率
                cpu_percent = process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
                # 内存使用
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                
                time.sleep(0.5)  # 0.5秒采样间隔
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
    def add_result(self, result: Dict[str, Any]):
        """添加测试结果"""
        self.results.append(result)
        
    def calculate_metrics(self):
        """计算性能指标"""
        if not self.results:
            return {}
        
        # 计算统计
        if hasattr(self, 'results') and self.results:
            result_keys = self.results[0].keys()
            for key in result_keys:
                if all(key in r for r in self.results):
                    values = [r[key] for r in self.results if r[key] is not None]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        self.metrics[key] = {
                            'min': min(values),
                            'max': max(values),
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                            'count': len(values)
                        }
        
        # 计算时间
        if self.start_time and self.end_time:
            self.metrics['duration_seconds'] = self.end_time - self.start_time
            
        # 计算资源使用
        if self.cpu_samples:
            self.metrics['cpu_percent'] = {
                'min': min(self.cpu_samples),
                'max': max(self.cpu_samples),
                'mean': statistics.mean(self.cpu_samples)
            }
            
        if self.memory_samples:
            self.metrics['memory_mb'] = {
                'min': min(self.memory_samples),
                'max': max(self.memory_samples),
                'mean': statistics.mean(self.memory_samples)
            }
            
        return self.metrics
    
    def print_report(self):
        """打印性能报告"""
        print(f"\n{'='*60}")
        print(f"性能测试报告: {self.name}")
        print(f"{'='*60}")
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"测试持续时间: {duration:.2f} 秒")
            
        for metric_name, metric_data in self.metrics.items():
            if isinstance(metric_data, dict):
                print(f"\n{metric_name}:")
                for stat_name, stat_value in metric_data.items():
                    if isinstance(stat_value, float):
                        print(f"  {stat_name}: {stat_value:.4f}")
                    else:
                        print(f"  {stat_name}: {stat_value}")
            else:
                print(f"{metric_name}: {metric_data}")
                
    def save_report(self, filepath: str):
        """保存性能报告到文件"""
        report = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"性能报告已保存到: {filepath}")


class TestAlgorithmPerformance:
    """算法性能测试类"""
    
    def test_energy_detector_performance(self):
        """测试能量检测器性能"""
        detector = EnergyDetector()
        benchmark = PerformanceBenchmark("能量检测器性能测试")
        
        # 生成测试信号
        signal_sizes = [
            PerformanceConfig.SMALL_SIGNAL_SIZE,
            PerformanceConfig.MEDIUM_SIGNAL_SIZE,
            PerformanceConfig.LARGE_SIGNAL_SIZE
        ]
        
        benchmark.start()
        
        for signal_size in signal_sizes:
            # 生成信号
            t = np.linspace(0, signal_size/10e6, signal_size)
            signal = np.exp(1j * 2 * np.pi * 1e6 * t)
            signal_data = create_signal_from_iq(
                iq_data=signal,
                sample_rate=10e6,
                center_freq=100e6
            )
            
            # 预热
            for _ in range(PerformanceConfig.WARMUP_ITERATIONS):
                detector.execute(signal_data)
            
            # 性能测试
            execution_times = []
            for _ in range(PerformanceConfig.MEASUREMENT_ITERATIONS):
                start_time = time.perf_counter()
                result = detector.execute(signal_data)
                end_time = time.perf_counter()
                
                execution_times.append((end_time - start_time) * 1000)  # 转换为ms
                assert result is not None
            
            # 记录结果
            benchmark.add_result({
                'signal_size': signal_size,
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'mean_time_ms': statistics.mean(execution_times),
                'median_time_ms': statistics.median(execution_times),
                'std_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            })
            
            logger.info(f"信号大小 {signal_size}: 平均 {statistics.mean(execution_times):.2f} ms")
        
        benchmark.stop()
        metrics = benchmark.calculate_metrics()
        benchmark.print_report()
        
        # 验证性能
        for result in benchmark.results:
            assert result['mean_time_ms'] < PerformanceConfig.MAX_PROCESSING_LATENCY_MS, \
                f"处理延迟过高: {result['mean_time_ms']:.1f} ms"
    
    def test_spectrum_analyzer_performance(self):
        """测试频谱分析器性能"""
        analyzer = SpectrumAnalyzer()
        benchmark = PerformanceBenchmark("频谱分析器性能测试")
        
        # 测试不同FFT大小
        fft_sizes = [256, 512, 1024, 2048, 4096, 8192]
        
        # 生成测试信号
        signal_size = 16384
        t = np.linspace(0, signal_size/10e6, signal_size)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal_data = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        benchmark.start()
        
        for fft_size in fft_sizes:
            # 配置FFT大小
            analyzer.configure({'fft_size': fft_size})
            
            # 预热
            for _ in range(PerformanceConfig.WARMUP_ITERATIONS):
                analyzer.execute(signal_data)
            
            # 性能测试
            execution_times = []
            for _ in range(PerformanceConfig.MEASUREMENT_ITERATIONS):
                start_time = time.perf_counter()
                result = analyzer.execute(signal_data)
                end_time = time.perf_counter()
                
                execution_times.append((end_time - start_time) * 1000)
                assert result is not None
            
            # 记录结果
            benchmark.add_result({
                'fft_size': fft_size,
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'mean_time_ms': statistics.mean(execution_times),
                'median_time_ms': statistics.median(execution_times)
            })
            
            logger.info(f"FFT大小 {fft_size}: 平均 {statistics.mean(execution_times):.2f} ms")
        
        benchmark.stop()
        metrics = benchmark.calculate_metrics()
        benchmark.print_report()
        
        # 验证性能
        for result in benchmark.results:
            assert result['mean_time_ms'] < PerformanceConfig.MAX_PROCESSING_LATENCY_MS, \
                f"处理延迟过高: {result['mean_time_ms']:.1f} ms"
    
    def test_modulation_detector_performance(self):
        """测试调制检测器性能"""
        detector = ModulationDetector()
        benchmark = PerformanceBenchmark("调制检测器性能测试")
        
        # 生成不同调制类型的信号
        modulations = [
            ('CW', lambda t: np.exp(1j * 2 * np.pi * 1e6 * t)),
            ('AM', lambda t: np.exp(1j * 2 * np.pi * 1e6 * t) * (1 + 0.5 * np.sin(2 * np.pi * 10e3 * t))),
            ('FM', lambda t: np.exp(1j * (2 * np.pi * 1e6 * t + 5.0 * np.sin(2 * np.pi * 10e3 * t))))
        ]
        
        signal_size = 4096
        benchmark.start()
        
        for mod_name, mod_func in modulations:
            # 生成信号
            t = np.linspace(0, signal_size/10e6, signal_size)
            signal = mod_func(t)
            signal_data = create_signal_from_iq(
                iq_data=signal,
                sample_rate=10e6,
                center_freq=100e6
            )
            
            # 预热
            for _ in range(PerformanceConfig.WARMUP_ITERATIONS):
                detector.execute(signal_data)
            
            # 性能测试
            execution_times = []
            for _ in range(PerformanceConfig.MEASUREMENT_ITERATIONS):
                start_time = time.perf_counter()
                result = detector.execute(signal_data)
                end_time = time.perf_counter()
                
                execution_times.append((end_time - start_time) * 1000)
                assert result is not None
            
            # 记录结果
            benchmark.add_result({
                'modulation': mod_name,
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'mean_time_ms': statistics.mean(execution_times),
                'median_time_ms': statistics.median(execution_times)
            })
            
            logger.info(f"调制类型 {mod_name}: 平均 {statistics.mean(execution_times):.2f} ms")
        
        benchmark.stop()
        metrics = benchmark.calculate_metrics()
        benchmark.print_report()
        
        # 验证性能
        for result in benchmark.results:
            assert result['mean_time_ms'] < PerformanceConfig.MAX_PROCESSING_LATENCY_MS, \
                f"处理延迟过高: {result['mean_time_ms']:.1f} ms"
    
    def test_algorithm_memory_usage(self):
        """测试算法内存使用"""
        algorithms = [
            ('energy_detector', EnergyDetector()),
            ('spectrum_analyzer', SpectrumAnalyzer()),
            ('modulation_detector', ModulationDetector()),
            ('anomaly_detector', AnomalyDetector()),
            ('signal_classifier', SignalClassifier())
        ]
        
        # 生成测试信号
        signal_size = 8192
        t = np.linspace(0, signal_size/10e6, signal_size)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal_data = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        memory_results = []
        
        for algo_name, algorithm in algorithms:
            logger.info(f"测试 {algo_name} 内存使用...")
            
            # 强制垃圾回收
            gc.collect()
            
            # 记录初始内存
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行算法多次
            execution_times = []
            for _ in range(50):
                start_time = time.perf_counter()
                result = algorithm.execute(signal_data)
                end_time = time.perf_counter()
                execution_times.append((end_time - start_time) * 1000)
                assert result is not None
            
            # 强制垃圾回收
            gc.collect()
            
            # 记录最终内存
            final_memory = process.memory_info().rss / 1024 / 1024
            
            memory_increase = final_memory - initial_memory
            avg_time = statistics.mean(execution_times)
            
            memory_results.append({
                'algorithm': algo_name,
                'memory_increase_mb': memory_increase,
                'avg_time_ms': avg_time,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory
            })
            
            logger.info(f"{algo_name}: 内存增加 {memory_increase:.2f} MB, "
                       f"平均时间 {avg_time:.2f} ms")
        
        # 验证内存使用
        for result in memory_results:
            assert result['memory_increase_mb'] < PerformanceConfig.MAX_MEMORY_INCREASE_MB, \
                f"{result['algorithm']} 内存增加过多: {result['memory_increase_mb']:.1f} MB"
    
    def test_algorithm_concurrent_performance(self):
        """测试算法并发性能"""
        algorithm = EnergyDetector()
        
        # 生成测试信号
        signal_size = 4096
        t = np.linspace(0, signal_size/10e6, signal_size)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal_data = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 测试不同并发级别
        concurrency_levels = [1, 2, 4, 8, 16]
        
        results = []
        
        for concurrency in concurrency_levels:
            logger.info(f"测试并发级别 {concurrency}...")
            
            def process_signal(_):
                start_time = time.perf_counter()
                result = algorithm.execute(signal_data)
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000, result is not None
            
            # 串行执行
            serial_times = []
            serial_start = time.time()
            for i in range(concurrency * 10):
                exec_time, success = process_signal(i)
                serial_times.append(exec_time)
                assert success
            serial_duration = time.time() - serial_start
            
            # 并发执行
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                concurrent_start = time.time()
                futures = [executor.submit(process_signal, i) for i in range(concurrency * 10)]
                concurrent_times = []
                for future in futures:
                    exec_time, success = future.result()
                    concurrent_times.append(exec_time)
                    assert success
                concurrent_duration = time.time() - concurrent_start
            
            # 计算加速比
            speedup = serial_duration / concurrent_duration if concurrent_duration > 0 else 0
            
            results.append({
                'concurrency': concurrency,
                'serial_duration': serial_duration,
                'concurrent_duration': concurrent_duration,
                'speedup': speedup,
                'serial_avg_time_ms': statistics.mean(serial_times),
                'concurrent_avg_time_ms': statistics.mean(concurrent_times)
            })
            
            logger.info(f"并发 {concurrency}: 串行 {serial_duration:.2f}s, "
                       f"并发 {concurrent_duration:.2f}s, 加速 {speedup:.2f}x")
        
        # 验证并发性能
        for result in results:
            if result['concurrency'] > 1:
                # 并发应该比串行快（对于CPU密集型任务）
                assert result['speedup'] > 0.5, \
                    f"并发级别 {result['concurrency']} 加速比过低: {result['speedup']:.2f}"
    
    def test_algorithm_batch_processing_performance(self):
        """测试算法批处理性能"""
        detector = EnergyDetector()
        
        # 生成批量信号
        batch_sizes = [1, 10, 50, 100, 200]
        
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"测试批处理大小 {batch_size}...")
            
            # 生成信号
            signals = []
            for i in range(batch_size):
                signal_size = 4096
                t = np.linspace(0, signal_size/10e6, signal_size)
                freq = 1e6 + i * 0.1e6
                signal = np.exp(1j * 2 * np.pi * freq * t)
                signal_data = create_signal_from_iq(
                    iq_data=signal,
                    sample_rate=10e6,
                    center_freq=100e6
                )
                signals.append(signal_data)
            
            # 串行处理
            serial_start = time.time()
            serial_results = []
            for signal in signals:
                result = detector.execute(signal)
                serial_results.append(result)
            serial_duration = time.time() - serial_start
            
            # 验证结果
            assert len(serial_results) == batch_size
            for result in serial_results:
                assert result is not None
            
            # 计算吞吐量
            serial_throughput = batch_size / serial_duration
            
            results.append({
                'batch_size': batch_size,
                'serial_duration': serial_duration,
                'serial_throughput': serial_throughput,
                'avg_time_per_signal_ms': (serial_duration * 1000) / batch_size
            })
            
            logger.info(f"批大小 {batch_size}: {serial_duration:.2f}s, "
                       f"吞吐量 {serial_throughput:.1f} 信号/秒")
        
        # 验证批处理性能
        for result in results:
            assert result['serial_throughput'] > PerformanceConfig.MIN_THROUGHPUT_SIGNALS_PER_SEC, \
                f"批大小 {result['batch_size']} 吞吐量过低: {result['serial_throughput']:.1f} 信号/秒"


class TestDataAcquisitionPerformance:
    """数据采集性能测试类"""
    
    def test_acquisition_throughput(self, signal_config, event_manager):
        """测试数据采集吞吐量"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            mock_device.get_sample_rate.return_value = signal_config.acquisition.sample_rate
            mock_device.get_center_frequency.return_value = signal_config.acquisition.center_frequency
            
            # 创建信号生成器
            samples_generated = 0
            
            def signal_generator(num_samples):
                nonlocal samples_generated
                samples_generated += num_samples
                t = np.linspace(0, num_samples/signal_config.acquisition.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * 1e6 * t)
            
            mock_device.read_samples.side_effect = signal_generator
            MockDevice.return_value = mock_device
            
            # 创建处理器
            mock_processor = Mock()
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=mock_processor,
                event_manager=event_manager
            )
            
            benchmark = PerformanceBenchmark("数据采集吞吐量测试")
            benchmark.start()
            
            # 启动采集
            acquisition.start()
            time.sleep(PerformanceConfig.MEDIUM_TEST_DURATION)
            
            # 停止采集
            acquisition.stop()
            benchmark.stop()
            
            # 计算吞吐量
            duration = benchmark.end_time - benchmark.start_time
            samples_per_second = samples_generated / duration
            bytes_per_second = samples_per_second * 16  # 复数，每个样本16字节
            
            metrics = benchmark.calculate_metrics()
            benchmark.add_result({
                'duration_seconds': duration,
                'samples_generated': samples_generated,
                'samples_per_second': samples_per_second,
                'megasamples_per_second': samples_per_second / 1e6,
                'bytes_per_second': bytes_per_second,
                'megabytes_per_second': bytes_per_second / 1024 / 1024
            })
            
            benchmark.print_report()
            
            # 验证吞吐量
            target_sample_rate = signal_config.acquisition.sample_rate
            efficiency = samples_per_second / target_sample_rate
            
            logger.info(f"采集效率: {efficiency*100:.1f}%")
            logger.info(f"吞吐量: {samples_per_second/1e6:.2f} MS/s, "
                       f"{bytes_per_second/1024/1024:.2f} MB/s")
            
            assert efficiency > 0.5, f"采集效率过低: {efficiency*100:.1f}%"
            assert samples_per_second > target_sample_rate * 0.5, \
                f"吞吐量过低: {samples_per_second/1e6:.2f} MS/s < {target_sample_rate/1e6/2:.2f} MS/s"
    
    def test_acquisition_latency(self, signal_config, event_manager):
        """测试数据采集延迟"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 配置小缓冲区以减少延迟
            config = signal_config.acquisition.copy()
            config.buffer_size = 256
            config.acquisition_interval_ms = 1
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            def fast_signal_generator(num_samples):
                t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * 1e6 * t)
            
            mock_device.read_samples.side_effect = fast_signal_generator
            MockDevice.return_value = mock_device
            
            # 收集延迟数据
            latencies = []
            
            def measure_latency(event):
                if event.get('type') == 'signal_acquired':
                    signal_data = event.get('signal_data')
                    if signal_data and hasattr(signal_data, 'timestamp'):
                        acquisition_time = signal_data.timestamp
                        processing_time = datetime.now()
                        latency = (processing_time - acquisition_time).total_seconds() * 1000
                        latencies.append(latency)
            
            event_manager.subscribe('signal_acquired', measure_latency)
            
            mock_processor = Mock()
            acquisition = DataAcquisition(
                config=config,
                signal_processor=mock_processor,
                event_manager=event_manager
            )
            
            # 启动采集
            acquisition.start()
            time.sleep(PerformanceConfig.SHORT_TEST_DURATION)
            acquisition.stop()
            
            # 分析延迟
            if latencies:
                avg_latency = statistics.mean(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
                
                logger.info(f"采集延迟统计: {len(latencies)} 个样本")
                logger.info(f"平均延迟: {avg_latency:.2f} ms")
                logger.info(f"最小延迟: {min_latency:.2f} ms")
                logger.info(f"最大延迟: {max_latency:.2f} ms")
                logger.info(f"P95延迟: {p95_latency:.2f} ms")
                logger.info(f"P99延迟: {p99_latency:.2f} ms")
                
                # 验证延迟
                assert avg_latency < PerformanceConfig.MAX_ACQUISITION_LATENCY_MS, \
                    f"平均延迟过高: {avg_latency:.2f} ms"
                assert p95_latency < PerformanceConfig.MAX_ACQUISITION_LATENCY_MS * 2, \
                    f"P95延迟过高: {p95_latency:.2f} ms"
            else:
                pytest.skip("没有采集到信号数据")
    
    def test_acquisition_resource_usage(self, signal_config, event_manager):
        """测试数据采集资源使用"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 配置高吞吐量
            config = signal_config.acquisition.copy()
            config.buffer_size = 8192
            config.acquisition_interval_ms = 1
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            def high_throughput_generator(num_samples):
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            
            mock_device.read_samples.side_effect = high_throughput_generator
            MockDevice.return_value = mock_device
            
            mock_processor = Mock()
            acquisition = DataAcquisition(
                config=config,
                signal_processor=mock_processor,
                event_manager=event_manager
            )
            
            # 监控资源使用
            process = psutil.Process(os.getpid())
            
            # 初始资源
            initial_cpu = process.cpu_percent(interval=0.5)
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # 启动采集
            acquisition.start()
            
            # 监控运行中资源
            cpu_samples = []
            memory_samples = []
            
            for _ in range(10):  # 监控10秒
                cpu_samples.append(process.cpu_percent(interval=0.1))
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                time.sleep(0.9)  # 总共1秒
            
            acquisition.stop()
            
            # 最终资源
            final_cpu = process.cpu_percent(interval=0.5)
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # 分析资源使用
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            avg_memory = statistics.mean(memory_samples)
            max_memory = max(memory_samples)
            memory_increase = max_memory - initial_memory
            
            logger.info(f"CPU使用: 平均 {avg_cpu:.1f}%, 最大 {max_cpu:.1f}%")
            logger.info(f"内存使用: 平均 {avg_memory:.1f} MB, 最大 {max_memory:.1f} MB")
            logger.info(f"内存增加: {memory_increase:.1f} MB")
            
            # 验证资源使用
            assert avg_cpu < PerformanceConfig.MAX_CPU_USAGE_PERCENT, \
                f"CPU使用率过高: {avg_cpu:.1f}%"
            assert memory_increase < PerformanceConfig.MAX_MEMORY_INCREASE_MB, \
                f"内存增加过多: {memory_increase:.1f} MB"
    
    def test_acquisition_stability(self, signal_config, event_manager):
        """测试数据采集稳定性"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            config = signal_config.acquisition.copy()
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            # 创建稳定的信号生成器
            def stable_signal_generator(num_samples):
                t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * 1e6 * t)
            
            mock_device.read_samples.side_effect = stable_signal_generator
            MockDevice.return_value = mock_device
            
            mock_processor = Mock()
            acquisition = DataAcquisition(
                config=config,
                signal_processor=mock_processor,
                event_manager=event_manager
            )
            
            # 监控事件率
            event_counts = []
            
            def count_events(event):
                if event.get('type') == 'signal_acquired':
                    event_counts.append(time.time())
            
            event_manager.subscribe('signal_acquired', count_events)
            
            # 长时间运行测试
            logger.info("开始长时间稳定性测试...")
            
            acquisition.start()
            
            # 分阶段监控
            stage_durations = []
            for stage in range(6):  # 6个阶段，每个10秒
                start_count = len(event_counts)
                time.sleep(10)
                end_count = len(event_counts)
                events_in_stage = end_count - start_count
                rate = events_in_stage / 10.0
                stage_durations.append(rate)
                logger.info(f"阶段 {stage+1}: {rate:.1f} 信号/秒")
            
            acquisition.stop()
            
            # 分析稳定性
            avg_rate = statistics.mean(stage_durations)
            std_rate = statistics.stdev(stage_durations) if len(stage_durations) > 1 else 0
            cv_rate = (std_rate / avg_rate * 100) if avg_rate > 0 else 0  # 变异系数
            
            logger.info(f"平均事件率: {avg_rate:.1f} 信号/秒")
            logger.info(f"事件率标准差: {std_rate:.2f}")
            logger.info(f"事件率变异系数: {cv_rate:.1f}%")
            
            # 验证稳定性
            assert cv_rate < 20, f"事件率不稳定: 变异系数 {cv_rate:.1f}%"
            assert avg_rate > 0, "没有采集到信号"


class TestPipelinePerformance:
    """流水线性能测试类"""
    
    def test_full_pipeline_throughput(self, tmp_path, signal_config, event_manager):
        """测试完整流水线吞吐量"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            config = signal_config.acquisition.copy()
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            # 高性能信号生成
            samples_generated = 0
            
            def high_perf_generator(num_samples):
                nonlocal samples_generated
                samples_generated += num_samples
                t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * 1e6 * t)
            
            mock_device.read_samples.side_effect = high_perf_generator
            MockDevice.return_value = mock_device
            
            # 创建完整流水线
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
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 事件计数器
            event_counts = {
                'signal_acquired': 0,
                'signal_processed': 0,
                'signal_detected': 0
            }
            
            def count_event(event):
                event_type = event.get('type')
                if event_type in event_counts:
                    event_counts[event_type] += 1
            
            for event_type in event_counts.keys():
                event_manager.subscribe(event_type, count_event)
            
            benchmark = PerformanceBenchmark("完整流水线吞吐量测试")
            benchmark.start()
            
            # 启动流水线
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            acquisition.start()
            
            # 运行测试
            time.sleep(PerformanceConfig.MEDIUM_TEST_DURATION)
            
            # 停止流水线
            acquisition.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
            
            benchmark.stop()
            
            # 计算性能指标
            duration = benchmark.end_time - benchmark.start_time
            
            # 吞吐量
            signals_per_second = event_counts['signal_acquired'] / duration
            processing_per_second = event_counts['signal_processed'] / duration
            detection_per_second = event_counts['signal_detected'] / duration
            
            # 处理率
            processing_rate = event_counts['signal_processed'] / event_counts['signal_acquired'] \
                if event_counts['signal_acquired'] > 0 else 0
            detection_rate = event_counts['signal_detected'] / event_counts['signal_processed'] \
                if event_counts['signal_processed'] > 0 else 0
            
            # 数据吞吐量
            data_rate_mbps = (samples_generated * 16 * 8) / duration / 1e6  # Mbps
            
            benchmark.add_result({
                'duration_seconds': duration,
                'signals_acquired': event_counts['signal_acquired'],
                'signals_processed': event_counts['signal_processed'],
                'signals_detected': event_counts['signal_detected'],
                'signals_per_second': signals_per_second,
                'processing_per_second': processing_per_second,
                'detection_per_second': detection_per_second,
                'processing_rate': processing_rate,
                'detection_rate': detection_rate,
                'data_rate_mbps': data_rate_mbps,
                'samples_generated': samples_generated
            })
            
            benchmark.print_report()
            
            logger.info(f"流水线吞吐量: {signals_per_second:.1f} 信号/秒")
            logger.info(f"数据处理率: {processing_rate:.1%}")
            logger.info(f"数据检测率: {detection_rate:.1%}")
            logger.info(f"数据速率: {data_rate_mbps:.1f} Mbps")
            
            # 验证性能
            assert signals_per_second > PerformanceConfig.MIN_THROUGHPUT_SIGNALS_PER_SEC, \
                f"吞吐量过低: {signals_per_second:.1f} 信号/秒"
            assert processing_rate > 0.7, f"处理率过低: {processing_rate:.1%}"
    
    def test_pipeline_latency(self, tmp_path, signal_config, event_manager):
        """测试流水线延迟"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 配置小缓冲区以减少延迟
            config = signal_config.acquisition.copy()
            config.buffer_size = 512
            config.acquisition_interval_ms = 1
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            def low_latency_generator(num_samples):
                t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                return np.exp(1j * 2 * np.pi * 1e6 * t)
            
            mock_device.read_samples.side_effect = low_latency_generator
            MockDevice.return_value = mock_device
            
            # 创建流水线
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
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 延迟跟踪
            latency_data = []
            
            def track_latency(event):
                if event.get('type') == 'signal_detected':
                    detection = event.get('detection')
                    if detection and hasattr(detection, 'detection_time'):
                        # 获取信号时间戳
                        signal_data = event.get('signal_data')
                        if signal_data and hasattr(signal_data, 'timestamp'):
                            acquisition_time = signal_data.timestamp
                            detection_time = detection.detection_time
                            
                            if isinstance(acquisition_time, datetime) and isinstance(detection_time, datetime):
                                latency = (detection_time - acquisition_time).total_seconds() * 1000
                                latency_data.append(latency)
            
            event_manager.subscribe('signal_detected', track_latency)
            
            # 启动流水线
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            acquisition.start()
            
            # 运行测试
            time.sleep(PerformanceConfig.SHORT_TEST_DURATION)
            
            # 停止流水线
            acquisition.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
            
            # 分析延迟
            if latency_data:
                avg_latency = statistics.mean(latency_data)
                min_latency = min(latency_data)
                max_latency = max(latency_data)
                p95_latency = np.percentile(latency_data, 95)
                p99_latency = np.percentile(latency_data, 99)
                
                logger.info(f"流水线延迟统计: {len(latency_data)} 个样本")
                logger.info(f"端到端延迟: 平均 {avg_latency:.2f} ms")
                logger.info(f"最小延迟: {min_latency:.2f} ms")
                logger.info(f"最大延迟: {max_latency:.2f} ms")
                logger.info(f"P95延迟: {p95_latency:.2f} ms")
                logger.info(f"P99延迟: {p99_latency:.2f} ms")
                
                # 验证延迟
                assert avg_latency < PerformanceConfig.MAX_PROCESSING_LATENCY_MS, \
                    f"端到端延迟过高: {avg_latency:.2f} ms"
                assert p95_latency < PerformanceConfig.MAX_PROCESSING_LATENCY_MS * 1.5, \
                    f"P95延迟过高: {p95_latency:.2f} ms"
            else:
                pytest.skip("没有检测到信号")
    
    def test_pipeline_concurrent_throughput(self, tmp_path, signal_config, event_manager):
        """测试流水线并发吞吐量"""
        # 测试多个并行流水线
        num_pipelines = 3
        results = []
        
        for pipeline_id in range(num_pipelines):
            with patch(f'src.core.data_acquisition.ADRV9009Device') as MockDevice:
                mock_device = Mock()
                mock_device.initialize.return_value = True
                
                config = signal_config.acquisition.copy()
                # 每个流水线使用不同频率
                config.center_frequency = 100e6 + pipeline_id * 10e6
                
                mock_device.get_sample_rate.return_value = config.sample_rate
                mock_device.get_center_frequency.return_value = config.center_frequency
                
                def pipeline_generator(num_samples):
                    t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                    freq = 1e6 + pipeline_id * 0.5e6
                    return np.exp(1j * 2 * np.pi * freq * t)
                
                mock_device.read_samples.side_effect = pipeline_generator
                MockDevice.return_value = mock_device
                
                # 为每个流水线创建独立的事件管理器
                pipeline_event_manager = EventManager()
                
                # 创建流水线
                algorithm_manager = AlgorithmManager(
                    config=signal_config.algorithms,
                    event_manager=pipeline_event_manager
                )
                
                signal_processor = SignalProcessor(
                    config=signal_config.processing,
                    algorithm_manager=algorithm_manager,
                    event_manager=pipeline_event_manager
                )
                
                detection_engine = DetectionEngine(
                    config=signal_config.detection,
                    algorithm_manager=algorithm_manager,
                    event_manager=pipeline_event_manager
                )
                
                acquisition = DataAcquisition(
                    config=config,
                    signal_processor=signal_processor,
                    event_manager=pipeline_event_manager
                )
                
                # 事件计数器
                event_counts = {'signal_acquired': 0}
                
                def count_event(event):
                    if event.get('type') == 'signal_acquired':
                        event_counts['signal_acquired'] += 1
                
                pipeline_event_manager.subscribe('signal_acquired', count_event)
                
                # 启动流水线
                algorithm_manager.start()
                signal_processor.start()
                detection_engine.start()
                acquisition.start()
                
                results.append({
                    'pipeline_id': pipeline_id,
                    'acquisition': acquisition,
                    'event_counts': event_counts,
                    'event_manager': pipeline_event_manager
                })
        
        # 并行运行
        logger.info(f"启动 {num_pipelines} 个并行流水线...")
        time.sleep(PerformanceConfig.SHORT_TEST_DURATION)
        
        # 停止所有流水线并收集结果
        total_signals = 0
        for result in results:
            result['acquisition'].stop()
            total_signals += result['event_counts']['signal_acquired']
        
        # 计算总吞吐量
        throughput = total_signals / PerformanceConfig.SHORT_TEST_DURATION
        
        logger.info(f"并行流水线总吞吐量: {throughput:.1f} 信号/秒")
        logger.info(f"单个流水线平均吞吐量: {throughput/num_pipelines:.1f} 信号/秒")
        
        # 验证并发性能
        assert throughput > PerformanceConfig.MIN_THROUGHPUT_SIGNALS_PER_SEC * num_pipelines * 0.5, \
            f"并发吞吐量过低: {throughput:.1f} 信号/秒"
    
    def test_pipeline_scalability(self, tmp_path, signal_config, event_manager):
        """测试流水线可扩展性"""
        # 测试不同信号大小的可扩展性
        signal_sizes = [1024, 2048, 4096, 8192, 16384]
        
        scalability_results = []
        
        for signal_size in signal_sizes:
            with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
                mock_device = Mock()
                mock_device.initialize.return_value = True
                
                config = signal_config.acquisition.copy()
                config.buffer_size = signal_size
                
                mock_device.get_sample_rate.return_value = config.sample_rate
                mock_device.get_center_frequency.return_value = config.center_frequency
                
                def scalable_generator(num_samples):
                    t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                    return np.exp(1j * 2 * np.pi * 1e6 * t)
                
                mock_device.read_samples.side_effect = scalable_generator
                MockDevice.return_value = mock_device
                
                # 创建流水线
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
                
                acquisition = DataAcquisition(
                    config=config,
                    signal_processor=signal_processor,
                    event_manager=event_manager
                )
                
                # 事件计数器
                event_counts = {'signal_acquired': 0}
                
                def count_event(event):
                    if event.get('type') == 'signal_acquired':
                        event_counts['signal_acquired'] += 1
                
                event_manager.subscribe('signal_acquired', count_event)
                
                # 清空计数器
                event_counts['signal_acquired'] = 0
                
                # 启动流水线
                algorithm_manager.start()
                signal_processor.start()
                detection_engine.start()
                acquisition.start()
                
                # 运行测试
                time.sleep(PerformanceConfig.SHORT_TEST_DURATION)
                
                # 停止流水线
                acquisition.stop()
                detection_engine.stop()
                signal_processor.stop()
                algorithm_manager.stop()
                
                # 计算性能
                throughput = event_counts['signal_acquired'] / PerformanceConfig.SHORT_TEST_DURATION
                data_rate_mbps = (event_counts['signal_acquired'] * signal_size * 16 * 8) / \
                                PerformanceConfig.SHORT_TEST_DURATION / 1e6
                
                scalability_results.append({
                    'signal_size': signal_size,
                    'throughput_signals_per_second': throughput,
                    'data_rate_mbps': data_rate_mbps,
                    'events': event_counts['signal_acquired']
                })
                
                logger.info(f"信号大小 {signal_size}: {throughput:.1f} 信号/秒, "
                           f"{data_rate_mbps:.1f} Mbps")
        
        # 分析可扩展性
        logger.info("可扩展性分析:")
        for i in range(1, len(scalability_results)):
            prev = scalability_results[i-1]
            curr = scalability_results[i]
            
            size_ratio = curr['signal_size'] / prev['signal_size']
            throughput_ratio = curr['throughput_signals_per_second'] / prev['throughput_signals_per_second']
            data_ratio = curr['data_rate_mbps'] / prev['data_rate_mbps']
            
            logger.info(f"大小 {prev['signal_size']} -> {curr['signal_size']} "
                       f"(x{size_ratio:.1f}): "
                       f"吞吐量比率 {throughput_ratio:.2f}, 数据比率 {data_ratio:.2f}")
            
            # 验证可扩展性
            # 当信号大小增加时，吞吐量应该减少，但数据率可能增加
            assert throughput_ratio < 1.5, f"信号大小增加时吞吐量异常增加"
    
    def test_pipeline_under_load(self, tmp_path, signal_config, event_manager):
        """测试高负载下的流水线性能"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            
            # 配置高负载
            config = signal_config.acquisition.copy()
            config.buffer_size = 16384
            config.acquisition_interval_ms = 0.5  # 高频率
            
            mock_device.get_sample_rate.return_value = config.sample_rate
            mock_device.get_center_frequency.return_value = config.center_frequency
            
            def high_load_generator(num_samples):
                # 生成复杂信号增加处理负载
                t = np.linspace(0, num_samples/config.sample_rate, num_samples)
                
                # 多个信号叠加
                signal = np.zeros_like(t, dtype=np.complex128)
                for i in range(5):  # 5个信号
                    freq = 1e6 + i * 0.2e6
                    amplitude = 1.0 / (i + 1)
                    signal += amplitude * np.exp(1j * 2 * np.pi * freq * t)
                
                # 添加噪声
                noise = 0.2 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
                return signal + noise
            
            mock_device.read_samples.side_effect = high_load_generator
            MockDevice.return_value = mock_device
            
            # 创建流水线
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
            
            acquisition = DataAcquisition(
                config=config,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 监控资源使用
            process = psutil.Process(os.getpid())
            cpu_samples = []
            memory_samples = []
            
            def monitor_resources():
                for _ in range(20):  # 监控20秒
                    cpu_samples.append(process.cpu_percent(interval=0.5))
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            
            # 事件计数器
            event_counts = {'signal_acquired': 0, 'signal_processed': 0}
            
            def count_event(event):
                event_type = event.get('type')
                if event_type in event_counts:
                    event_counts[event_type] += 1
            
            event_manager.subscribe('signal_acquired', count_event)
            event_manager.subscribe('signal_processed', count_event)
            
            logger.info("开始高负载测试...")
            
            # 启动监控
            monitor_thread.start()
            
            # 启动流水线
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            acquisition.start()
            
            # 高负载运行
            time.sleep(PerformanceConfig.MEDIUM_TEST_DURATION)
            
            # 停止流水线
            acquisition.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
            
            monitor_thread.join(timeout=5)
            
            # 分析性能
            duration = PerformanceConfig.MEDIUM_TEST_DURATION
            throughput = event_counts['signal_acquired'] / duration
            processing_rate = event_counts['signal_processed'] / event_counts['signal_acquired'] \
                if event_counts['signal_acquired'] > 0 else 0
            
            # 资源使用
            avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
            max_cpu = max(cpu_samples) if cpu_samples else 0
            avg_memory = statistics.mean(memory_samples) if memory_samples else 0
            max_memory = max(memory_samples) if memory_samples else 0
            
            logger.info(f"高负载测试结果:")
            logger.info(f"吞吐量: {throughput:.1f} 信号/秒")
            logger.info(f"处理率: {processing_rate:.1%}")
            logger.info(f"CPU使用: 平均 {avg_cpu:.1f}%, 最大 {max_cpu:.1f}%")
            logger.info(f"内存使用: 平均 {avg_memory:.1f} MB, 最大 {max_memory:.1f} MB")
            
            # 验证高负载性能
            assert throughput > PerformanceConfig.MIN_THROUGHPUT_SIGNALS_PER_SEC, \
                f"高负载下吞吐量过低: {throughput:.1f} 信号/秒"
            assert processing_rate > 0.5, f"高负载下处理率过低: {processing_rate:.1%}"
            assert avg_cpu < PerformanceConfig.MAX_CPU_USAGE_PERCENT, \
                f"高负载下CPU使用率过高: {avg_cpu:.1f}%"


class TestSystemPerformance:
    """系统性能测试类"""
    
    def test_system_startup_time(self, signal_config, event_manager):
        """测试系统启动时间"""
        startup_times = []
        
        for _ in range(5):  # 多次测试
            with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
                mock_device = Mock()
                mock_device.initialize.return_value = True
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
                
                # 测量启动时间
                start_time = time.time()
                
                algorithm_manager.start()
                signal_processor.start()
                detection_engine.start()
                
                startup_duration = time.time() - start_time
                startup_times.append(startup_duration)
                
                # 停止
                detection_engine.stop()
                signal_processor.stop()
                algorithm_manager.stop()
        
        # 分析启动时间
        avg_startup = statistics.mean(startup_times)
        min_startup = min(startup_times)
        max_startup = max(startup_times)
        
        logger.info(f"系统启动时间: 平均 {avg_startup:.2f}s, "
                   f"最小 {min_startup:.2f}s, 最大 {max_startup:.2f}s")
        
        # 验证启动性能
        assert avg_startup < 5.0, f"系统启动时间过长: {avg_startup:.2f} 秒"
        assert max_startup < 10.0, f"最大启动时间过长: {max_startup:.2f} 秒"
    
    def test_system_memory_footprint(self, tmp_path, signal_config, event_manager):
        """测试系统内存占用"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            MockDevice.return_value = mock_device
            
            # 创建所有组件
            components = []
            
            # 1. 算法管理器
            algorithm_manager = AlgorithmManager(
                config=signal_config.algorithms,
                event_manager=event_manager
            )
            components.append(('algorithm_manager', algorithm_manager))
            
            # 2. 信号处理器
            signal_processor = SignalProcessor(
                config=signal_config.processing,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            components.append(('signal_processor', signal_processor))
            
            # 3. 检测引擎
            detection_engine = DetectionEngine(
                config=signal_config.detection,
                algorithm_manager=algorithm_manager,
                event_manager=event_manager
            )
            components.append(('detection_engine', detection_engine))
            
            # 4. 告警管理器
            alert_config = AlertConfig()
            alert_manager = AlertManager(
                config=alert_config,
                event_manager=event_manager
            )
            components.append(('alert_manager', alert_manager))
            
            # 5. 系统监控
            system_monitor = SystemMonitor(event_manager=event_manager)
            components.append(('system_monitor', system_monitor))
            
            # 6. 调度器
            scheduler = Scheduler(event_manager=event_manager)
            components.append(('scheduler', scheduler))
            
            # 7. 数据采集器
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            components.append(('data_acquisition', acquisition))
            
            # 测量内存占用
            process = psutil.Process(os.getpid())
            
            # 初始内存
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            memory_usage_by_component = []
            
            # 逐个启动组件并测量内存
            for comp_name, component in components:
                # 启动前强制GC
                gc.collect()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # 启动组件
                if hasattr(component, 'start'):
                    component.start()
                    time.sleep(0.5)  # 等待稳定
                
                # 启动后强制GC
                gc.collect()
                memory_after = process.memory_info().rss / 1024 / 1024
                
                memory_increase = memory_after - memory_before
                memory_usage_by_component.append({
                    'component': comp_name,
                    'memory_increase_mb': memory_increase,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after
                })
                
                logger.info(f"{comp_name}: 内存增加 {memory_increase:.2f} MB")
            
            # 总内存占用
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory
            
            # 停止所有组件
            for comp_name, component in reversed(components):
                if hasattr(component, 'stop'):
                    component.stop()
            
            # 分析内存占用
            logger.info(f"系统总内存占用: {total_increase:.2f} MB")
            logger.info(f"初始内存: {initial_memory:.2f} MB")
            logger.info(f"最终内存: {final_memory:.2f} MB")
            
            # 验证内存占用
            assert total_increase < PerformanceConfig.MAX_MEMORY_INCREASE_MB, \
                f"系统内存占用过多: {total_increase:.1f} MB"
    
    def test_system_cpu_efficiency(self, tmp_path, signal_config, event_manager):
        """测试系统CPU效率"""
        with patch('src.core.data_acquisition.ADRV9009Device') as MockDevice:
            mock_device = Mock()
            mock_device.initialize.return_value = True
            MockDevice.return_value = mock_device
            
            # 创建系统
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
            
            acquisition = DataAcquisition(
                config=signal_config.acquisition,
                signal_processor=signal_processor,
                event_manager=event_manager
            )
            
            # 监控CPU使用
            process = psutil.Process(os.getpid())
            cpu_samples_idle = []
            cpu_samples_active = []
            
            # 空闲状态CPU使用
            logger.info("测量空闲状态CPU使用...")
            algorithm_manager.start()
            signal_processor.start()
            detection_engine.start()
            
            for _ in range(10):
                cpu_samples_idle.append(process.cpu_percent(interval=0.5))
            
            # 活跃状态CPU使用
            logger.info("测量活跃状态CPU使用...")
            acquisition.start()
            
            for _ in range(10):
                cpu_samples_active.append(process.cpu_percent(interval=0.5))
            
            acquisition.stop()
            detection_engine.stop()
            signal_processor.stop()
            algorithm_manager.stop()
            
            # 分析CPU效率
            avg_cpu_idle = statistics.mean(cpu_samples_idle) if cpu_samples_idle else 0
            avg_cpu_active = statistics.mean(cpu_samples_active) if cpu_samples_active else 0
            cpu_increase = avg_cpu_active - avg_cpu_idle
            
            logger.info(f"空闲状态CPU: {avg_cpu_idle:.1f}%")
            logger.info(f"活跃状态CPU: {avg_cpu_active:.1f}%")
            logger.info(f"CPU增加: {cpu_increase:.1f}%")
            
            # 验证CPU效率
            assert avg_cpu_active < PerformanceConfig.MAX_CPU_USAGE_PERCENT, \
                f"活跃状态CPU使用率过高: {avg_cpu_active:.1f}%"
            assert cpu_increase > 0, "活跃状态应该比空闲状态使用更多CPU"
    
    def test_system_io_performance(self, tmp_path, signal_config, event_manager):
        """测试系统I/O性能"""
        # 创建测试文件
        test_file = tmp_path / "test_signal.bin"
        
        # 生成大信号文件
        signal_size = 10 * 1024 * 1024  # 10M样本
        signal = np.random.randn(signal_size) + 1j * np.random.randn(signal_size)
        
        # 测试写入性能
        write_start = time.time()
        signal.astype(np.complex64).tofile(test_file)
        write_duration = time.time() - write_start
        write_speed = (signal_size * 8) / write_duration / 1e6  # MB/s
        
        logger.info(f"写入性能: {write_speed:.1f} MB/s, 时间 {write_duration:.2f}s")
        
        # 测试读取性能
        read_start = time.time()
        data = np.fromfile(test_file, dtype=np.complex64)
        read_duration = time.time() - read_start
        read_speed = (signal_size * 8) / read_duration / 1e6
        
        logger.info(f"读取性能: {read_speed:.1f} MB/s, 时间 {read_duration:.2f}s")
        
        # 验证I/O性能
        assert write_speed > 10, f"写入速度过低: {write_speed:.1f} MB/s"
        assert read_speed > 20, f"读取速度过低: {read_speed:.1f} MB/s"
        
        # 清理
        test_file.unlink()
    
    def test_system_network_performance(self):
        """测试系统网络性能"""
        import socket
        import select
        
        # 创建本地Socket测试
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定到本地端口
        server_socket.bind(('127.0.0.1', 0))
        server_socket.listen(1)
        
        port = server_socket.getsockname()[1]
        
        # 客户端Socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 测试连接建立时间
        connect_start = time.time()
        client_socket.connect(('127.0.0.1', port))
        connect_duration = time.time() - connect_start
        
        # 接受连接
        conn, addr = server_socket.accept()
        
        # 测试小数据发送
        small_data = b'test' * 256  # 1KB
        send_start = time.time()
        client_socket.sendall(small_data)
        send_duration = time.time() - send_start
        
        # 接收数据
        conn.setblocking(0)
        ready = select.select([conn], [], [], 1.0)
        if ready[0]:
            received = conn.recv(4096)
        
        # 计算性能
        small_data_speed = len(small_data) / send_duration / 1024  # KB/s
        
        logger.info(f"连接建立时间: {connect_duration*1000:.1f} ms")
        logger.info(f"小数据发送速度: {small_data_speed:.1f} KB/s")
        
        # 验证网络性能
        assert connect_duration < 0.1, f"连接建立时间过长: {connect_duration*1000:.1f} ms"
        
        # 清理
        client_socket.close()
        conn.close()
        server_socket.close()


class TestPerformanceRegression:
    """性能回归测试类"""
    
    def test_performance_baseline_comparison(self):
        """测试性能基线比较"""
        # 加载性能基线
        baseline_file = Path(__file__).parent / "performance_baseline.json"
        
        if not baseline_file.exists():
            logger.warning("性能基线文件不存在，创建新基线")
            # 运行测试并创建基线
            self._create_performance_baseline(baseline_file)
            pytest.skip("已创建性能基线，需要重新运行测试进行比较")
        
        # 加载基线
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # 运行当前测试
        current_results = self._run_performance_tests()
        
        # 比较性能
        regression_detected = False
        regression_details = []
        
        for test_name, baseline_metrics in baseline.items():
            if test_name in current_results:
                current_metrics = current_results[test_name]
                
                # 比较关键指标
                for metric_name, baseline_value in baseline_metrics.items():
                    if metric_name in current_metrics:
                        current_value = current_metrics[metric_name]
                        
                        # 计算性能变化
                        if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                            if baseline_value > 0:
                                change_percent = (current_value - baseline_value) / baseline_value * 100
                                
                                # 检查回归
                                if change_percent > 10:  # 性能下降超过10%
                                    regression_detected = True
                                    regression_details.append({
                                        'test': test_name,
                                        'metric': metric_name,
                                        'baseline': baseline_value,
                                        'current': current_value,
                                        'change_percent': change_percent
                                    })
        
        # 报告结果
        if regression_detected:
            logger.error("检测到性能回归:")
            for detail in regression_details:
                logger.error(f"{detail['test']}.{detail['metric']}: "
                           f"{detail['baseline']:.2f} -> {detail['current']:.2f} "
                           f"(+{detail['change_percent']:.1f}%)")
            
            # 保存当前结果作为新基线（如果需要）
            # self._update_performance_baseline(baseline_file, current_results)
            
            # 失败测试
            pytest.fail(f"检测到性能回归: {len(regression_details)} 个指标")
        else:
            logger.info("性能测试通过，无回归检测")
    
    def _create_performance_baseline(self, baseline_file: Path):
        """创建性能基线"""
        baseline = {}
        
        # 运行基本算法测试
        logger.info("创建性能基线...")
        
        # 能量检测器
        detector = EnergyDetector()
        signal_size = 4096
        t = np.linspace(0, signal_size/10e6, signal_size)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        signal_data = create_signal_from_iq(
            iq_data=signal,
            sample_rate=10e6,
            center_freq=100e6
        )
        
        # 测量性能
        execution_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            result = detector.execute(signal_data)
            end_time = time.perf_counter()
            execution_times.append((end_time - start_time) * 1000)
        
        baseline['energy_detector'] = {
            'mean_time_ms': statistics.mean(execution_times),
            'min_time_ms': min(execution_times),
            '