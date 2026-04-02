"""
数据处理模块 - 实现各种算法处理
"""
import threading
import time
from typing import Dict, Any, Callable, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.algorithms.algorithm_factory import AlgorithmFactory
from src.models.signal_data import SignalData
from src.models.detection_result import DetectionResult
from src.utils.logger import get_logger

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Dict[str, Any], db_manager=None):
        self.config = config
        self.db = db_manager
        self.logger = get_logger(__name__)
        
        # 算法工厂
        self.algorithm_factory = AlgorithmFactory()
        
        # 启用的算法
        self.algorithms = self._initialize_algorithms()
        
        # 线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        # 回调
        self.result_callbacks = []
        
        # 统计
        self.processing_stats = {
            'processed_batches': 0,
            'detections': 0,
            'avg_processing_time': 0,
            'errors': 0
        }
    
    def _initialize_algorithms(self) -> List:
        """初始化算法列表"""
        algorithms = []
        algorithm_configs = self.config.get('algorithms', {})
        
        for algo_name, algo_config in algorithm_configs.items():
            if algo_config.get('enabled', False):
                try:
                    algorithm = self.algorithm_factory.create_algorithm(
                        algo_name, algo_config
                    )
                    algorithms.append(algorithm)
                    self.logger.info(f"启用算法: {algo_name}")
                except Exception as e:
                    self.logger.error(f"初始化算法 {algo_name} 失败: {e}")
        
        return algorithms
    
    def process_data(self, signal_data: SignalData):
        """处理数据（异步）"""
        if not self.algorithms:
            self.logger.warning("没有启用的算法")
            return
        
        # 提交处理任务
        future = self.executor.submit(
            self._process_data_sync, signal_data
        )
        
        # 可选：添加回调
        future.add_done_callback(self._on_processing_complete)
    
    def _process_data_sync(self, signal_data: SignalData) -> List[DetectionResult]:
        """同步处理数据"""
        start_time = time.time()
        all_results = []
        
        try:
            # 应用所有算法
            for algorithm in self.algorithms:
                try:
                    results = algorithm.process(signal_data)
                    
                    if results:
                        all_results.extend(results)
                        
                        # 如果是识别算法，与数据库对比
                        if algorithm.name == "recognition":
                            self._compare_with_database(results)
                            
                except Exception as e:
                    self.logger.error(f"算法 {algorithm.name} 处理失败: {e}")
                    continue
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(len(all_results), processing_time)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            self.processing_stats['errors'] += 1
            return []
    
    def _compare_with_database(self, results: List[DetectionResult]):
        """与数据库对比"""
        if not self.db:
            return
        
        for result in results:
            if result.confidence > self.config.get('recognition_threshold', 0.7):
                try:
                    # 查询数据库
                    match = self.db.query_similar_signals(
                        features=result.features,
                        threshold=0.8
                    )
                    
                    if match:
                        result.database_match = match
                        result.is_known = True
                        
                        # 触发事件
                        self._trigger_event("signal_recognized", result)
                    else:
                        result.is_known = False
                        self._trigger_event("unknown_signal_detected", result)
                        
                except Exception as e:
                    self.logger.error(f"数据库查询失败: {e}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """触发事件"""
        # 记录事件
        self.logger.info(f"事件触发: {event_type}")
        
        # 通知Web服务器
        for callback in self.result_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"事件回调失败: {e}")
        
        # 存储到数据库
        if self.db:
            self.db.store_event(event_type, data)
    
    def _on_processing_complete(self, future):
        """处理完成回调"""
        try:
            results = future.result()
            
            # 通知结果回调
            for callback in self.result_callbacks:
                try:
                    callback(results)
                except Exception as e:
                    self.logger.error(f"结果回调失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"处理任务失败: {e}")
    
    def _update_stats(self, num_detections: int, processing_time: float):
        """更新统计信息"""
        self.processing_stats['processed_batches'] += 1
        self.processing_stats['detections'] += num_detections
        
        # 更新平均处理时间
        n = self.processing_stats['processed_batches']
        current_avg = self.processing_stats['avg_processing_time']
        new_avg = (current_avg * (n-1) + processing_time) / n
        self.processing_stats['avg_processing_time'] = new_avg
    
    def add_result_callback(self, callback: Callable):
        """添加结果回调"""
        self.result_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return {
            'algorithms': [algo.name for algo in self.algorithms],
            'stats': self.processing_stats,
            'thread_pool': {
                'max_workers': self.executor._max_workers,
                'active_threads': threading.active_count() - 1
            }
        }
    
    def shutdown(self):
        """关闭处理器"""
        self.logger.info("关闭数据处理器...")
        self.executor.shutdown(wait=True)