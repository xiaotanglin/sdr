"""
高级日志管理器 - 支持多级别日志、文件轮转、结构化日志、日志上下文
"""
import os
import sys
import logging
import json
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, TextIO, Callable
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from enum import Enum
import inspect
from dataclasses import dataclass, asdict
import hashlib
from contextvars import ContextVar
from functools import wraps

# 日志级别枚举
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# 日志上下文变量
log_context_var = ContextVar('log_context', default={})

@dataclass
class LogContext:
    """日志上下文信息"""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    algorithm: Optional[str] = None
    signal_id: Optional[str] = None
    detection_id: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = {}
        for field, value in self.__dataclass__fields__.items():
            field_value = getattr(self, field)
            if field_value is not None:
                if field == 'custom_fields' and isinstance(field_value, dict):
                    data.update(field_value)
                else:
                    data[field] = field_value
        return data

class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def __init__(self, include_context: bool = True, **kwargs):
        """
        初始化JSON格式化器
        
        Args:
            include_context: 是否包含日志上下文
        """
        super().__init__(**kwargs)
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为JSON
        
        Args:
            record: 日志记录
            
        Returns:
            str: JSON格式的日志字符串
        """
        # 获取上下文信息
        context = getattr(record, 'context', {})
        
        # 构建日志数据
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        }
        
        # 添加上下文信息
        if self.include_context and context:
            log_data['context'] = context
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # 添加额外字段
        extra_fields = {k: v for k, v in record.__dict__.items() 
                       if k not in logging.LogRecord.__dict__ and not k.startswith('_')}
        if extra_fields:
            log_data['extra'] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)

class ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[94m',     # 蓝色
        'INFO': '\033[92m',      # 绿色
        'WARNING': '\033[93m',   # 黄色
        'ERROR': '\033[91m',     # 红色
        'CRITICAL': '\033[95m',  # 紫色
        'RESET': '\033[0m',      # 重置
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, style: str = '%', 
                 show_context: bool = True, show_module: bool = True):
        """
        初始化彩色格式化器
        
        Args:
            fmt: 格式字符串
            datefmt: 日期格式
            style: 格式样式
            show_context: 是否显示上下文
            show_module: 是否显示模块名
        """
        if fmt is None:
            fmt = self._get_default_format(show_context, show_module)
        
        super().__init__(fmt, datefmt, style)
        self.show_context = show_context
        self.show_module = show_module
    
    def _get_default_format(self, show_context: bool, show_module: bool) -> str:
        """获取默认格式字符串"""
        parts = []
        
        # 时间
        parts.append('%(asctime)s')
        
        # 日志级别（带颜色）
        parts.append('%(levelname_color)s%(levelname)-8s%(color_reset)s')
        
        # 模块和函数
        if self.show_module:
            parts.append('[%(module)s.%(funcName)s:%(lineno)d]')
        
        # 消息
        parts.append('%(message)s')
        
        # 上下文
        if self.show_context:
            parts.append('%(context_str)s')
        
        return ' '.join(parts)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            str: 格式化后的日志字符串
        """
        # 添加上下文字符串
        context = getattr(record, 'context', {})
        if context:
            record.context_str = f"[{', '.join(f'{k}={v}' for k, v in context.items())}]"
        else:
            record.context_str = ""
        
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname_color = self.COLORS[levelname]
            record.color_reset = self.COLORS['RESET']
        else:
            record.levelname_color = ''
            record.color_reset = ''
        
        return super().format(record)

class StructuredLogger:
    """结构化日志管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化日志管理器"""
        if getattr(self, '_initialized', False):
            return
        
        # 配置
        self.config = {
            'default_level': LogLevel.INFO.value,
            'log_dir': 'logs',
            'max_file_size_mb': 100,  # 100MB
            'backup_count': 10,  # 保留10个备份文件
            'rotate_daily': True,
            'enable_console': True,
            'enable_file': True,
            'enable_json_file': False,
            'enable_metrics': True,
            'log_performance': False,
        }
        
        # 日志记录器字典
        self.loggers: Dict[str, logging.Logger] = {}
        
        # 日志处理器字典
        self.handlers: Dict[str, List[logging.Handler]] = {}
        
        # 性能统计
        self.metrics = {
            'logs_written': 0,
            'errors_logged': 0,
            'warnings_logged': 0,
            'log_files_rotated': 0,
            'performance_logs': 0,
        }
        
        # 确保日志目录存在
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 上下文管理器
        self.context_manager = LogContextManager()
        
        self._initialized = True
    
    def configure(self, config: Dict[str, Any]):
        """
        配置日志管理器
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
        
        # 确保日志目录存在
        if 'log_dir' in config:
            os.makedirs(config['log_dir'], exist_ok=True)
        
        # 重新配置所有日志记录器
        for logger_name, logger in self.loggers.items():
            self._configure_logger(logger_name, logger)
    
    def _configure_logger(self, logger_name: str, logger: logging.Logger):
        """配置单个日志记录器"""
        # 移除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 设置日志级别
        logger.setLevel(getattr(logging, self.config['default_level']))
        
        # 防止日志传播到根记录器
        logger.propagate = False
        
        # 添加处理器
        handlers = []
        
        # 控制台处理器
        if self.config.get('enable_console', True):
            console_handler = self._create_console_handler()
            handlers.append(console_handler)
        
        # 文件处理器
        if self.config.get('enable_file', True):
            file_handler = self._create_file_handler(logger_name)
            handlers.append(file_handler)
        
        # JSON文件处理器
        if self.config.get('enable_json_file', False):
            json_handler = self._create_json_file_handler(logger_name)
            handlers.append(json_handler)
        
        # 添加处理器到记录器
        for handler in handlers:
            logger.addHandler(handler)
        
        # 保存处理器引用
        self.handlers[logger_name] = handlers
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        
        # 设置格式化器
        formatter = ColoredFormatter(
            show_context=True,
            show_module=True
        )
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_file_handler(self, logger_name: str) -> logging.Handler:
        """创建文件处理器"""
        # 生成日志文件名
        log_file = os.path.join(
            self.config['log_dir'], 
            f"{logger_name.replace('.', '_')}.log"
        )
        
        if self.config.get('rotate_daily', True):
            # 按时间轮转
            handler = TimedRotatingFileHandler(
                filename=log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.get('backup_count', 10),
                encoding='utf-8'
            )
        else:
            # 按大小轮转
            max_bytes = self.config.get('max_file_size_mb', 100) * 1024 * 1024
            handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=self.config.get('backup_count', 10),
                encoding='utf-8'
            )
        
        # 设置格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(module)s.%(funcName)s:%(lineno)d]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_json_file_handler(self, logger_name: str) -> logging.Handler:
        """创建JSON文件处理器"""
        # 生成JSON日志文件名
        json_log_file = os.path.join(
            self.config['log_dir'], 
            f"{logger_name.replace('.', '_')}.json.log"
        )
        
        # 按天轮转
        handler = TimedRotatingFileHandler(
            filename=json_log_file,
            when='midnight',
            interval=1,
            backupCount=self.config.get('backup_count', 10),
            encoding='utf-8'
        )
        
        # 设置JSON格式化器
        formatter = JsonFormatter(
            include_context=True
        )
        handler.setFormatter(formatter)
        
        return handler
    
    def get_logger(self, name: str, level: str = None) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 记录器名称
            level: 日志级别
            
        Returns:
            logging.Logger: 日志记录器
        """
        with self._lock:
            if name not in self.loggers:
                # 创建新记录器
                logger = logging.getLogger(name)
                self.loggers[name] = logger
                
                # 配置记录器
                self._configure_logger(name, logger)
            
            # 设置日志级别
            if level:
                level_name = level.upper()
                if level_name in LogLevel.__members__:
                    self.loggers[name].setLevel(getattr(logging, level_name))
            
            return EnhancedLogger(self.loggers[name], self.context_manager)
    
    def get_log_file_path(self, logger_name: str) -> Optional[str]:
        """
        获取日志文件路径
        
        Args:
            logger_name: 记录器名称
            
        Returns:
            Optional[str]: 日志文件路径
        """
        for handler in self.handlers.get(logger_name, []):
            if isinstance(handler, (TimedRotatingFileHandler, RotatingFileHandler)):
                return handler.baseFilename
        return None
    
    def rotate_all_logs(self):
        """轮转所有日志文件"""
        for logger_name, handlers in self.handlers.items():
            for handler in handlers:
                if hasattr(handler, 'doRollover'):
                    try:
                        handler.doRollover()
                        self.metrics['log_files_rotated'] += 1
                    except Exception as e:
                        self.error(f"轮转日志文件失败 {logger_name}: {e}")
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        清理旧日志文件
        
        Args:
            days_to_keep: 保留天数
        """
        log_dir = Path(self.config['log_dir'])
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in log_dir.glob("*.log*"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    log_file.unlink()
                    self.info(f"删除旧日志文件: {log_file.name}")
            except Exception as e:
                self.error(f"删除日志文件失败 {log_file}: {e}")
    
    def log_performance(self, operation: str, duration_ms: float, 
                       context: Dict = None, level: str = "INFO"):
        """
        记录性能日志
        
        Args:
            operation: 操作名称
            duration_ms: 持续时间（毫秒）
            context: 上下文信息
            level: 日志级别
        """
        if not self.config.get('log_performance', False):
            return
        
        # 构建上下文
        perf_context = context.copy() if context else {}
        perf_context.update({
            'operation': operation,
            'duration_ms': duration_ms,
            'duration_s': duration_ms / 1000.0,
        })
        
        # 记录日志
        if duration_ms > 1000:  # 超过1秒
            self.log(f"性能警告: {operation} 耗时 {duration_ms:.2f}ms", 
                    level="WARNING", context=perf_context)
        else:
            self.log(f"性能: {operation} 耗时 {duration_ms:.2f}ms", 
                    level=level, context=perf_context)
        
        self.metrics['performance_logs'] += 1
    
    def log(self, message: str, level: str = "INFO", context: Dict = None, 
           exc_info: bool = False, logger_name: str = "root"):
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别
            context: 上下文信息
            exc_info: 是否记录异常信息
            logger_name: 记录器名称
        """
        logger = self.get_logger(logger_name)
        
        # 添加上下文
        ctx = context.copy() if context else {}
        current_context = self.context_manager.get_context()
        if current_context:
            ctx.update(current_context)
        
        # 记录日志
        level_method = getattr(logger, level.lower(), logger.info)
        level_method(message, extra={'context': ctx}, exc_info=exc_info)
        
        # 更新统计
        self.metrics['logs_written'] += 1
        if level.upper() == LogLevel.ERROR.value:
            self.metrics['errors_logged'] += 1
        elif level.upper() == LogLevel.WARNING.value:
            self.metrics['warnings_logged'] += 1
    
    def debug(self, message: str, context: Dict = None, logger_name: str = "root"):
        """记录DEBUG级别日志"""
        self.log(message, "DEBUG", context, logger_name=logger_name)
    
    def info(self, message: str, context: Dict = None, logger_name: str = "root"):
        """记录INFO级别日志"""
        self.log(message, "INFO", context, logger_name=logger_name)
    
    def warning(self, message: str, context: Dict = None, logger_name: str = "root"):
        """记录WARNING级别日志"""
        self.log(message, "WARNING", context, logger_name=logger_name)
    
    def error(self, message: str, context: Dict = None, 
             exc_info: bool = False, logger_name: str = "root"):
        """记录ERROR级别日志"""
        self.log(message, "ERROR", context, exc_info, logger_name=logger_name)
    
    def critical(self, message: str, context: Dict = None, 
                exc_info: bool = False, logger_name: str = "root"):
        """记录CRITICAL级别日志"""
        self.log(message, "CRITICAL", context, exc_info, logger_name=logger_name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        return {
            'metrics': self.metrics.copy(),
            'loggers_count': len(self.loggers),
            'config': self.config,
            'log_dir': self.config['log_dir']
        }
    
    def shutdown(self):
        """关闭日志管理器"""
        for logger_name, handlers in self.handlers.items():
            for handler in handlers:
                handler.flush()
                handler.close()

class EnhancedLogger:
    """增强的日志记录器，支持上下文和性能监控"""
    
    def __init__(self, logger: logging.Logger, context_manager: 'LogContextManager'):
        self.logger = logger
        self.context_manager = context_manager
    
    def _add_context(self, kwargs: Dict) -> Dict:
        """添加上下文到日志参数"""
        kwargs = kwargs.copy()
        context = kwargs.pop('context', {})
        
        # 合并当前上下文
        current_context = self.context_manager.get_context()
        if current_context:
            if context:
                context = {**current_context, **context}
            else:
                context = current_context.copy()
        
        if context:
            kwargs['extra'] = {'context': context}
        
        return kwargs
    
    def debug(self, msg: str, *args, **kwargs):
        """记录DEBUG级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """记录INFO级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """记录WARNING级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """记录ERROR级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """记录CRITICAL级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, exc_info: bool = True, **kwargs):
        """记录异常日志"""
        kwargs = self._add_context(kwargs)
        kwargs['exc_info'] = exc_info
        self.logger.exception(msg, *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs):
        """记录指定级别日志"""
        kwargs = self._add_context(kwargs)
        self.logger.log(level, msg, *args, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, 
                   context: Dict = None, level: str = "INFO"):
        """记录性能日志"""
        perf_context = context.copy() if context else {}
        perf_context.update({
            'operation': operation,
            'duration_ms': duration_ms,
            'duration_s': duration_ms / 1000.0,
        })
        
        if duration_ms > 1000:  # 超过1秒
            self.warning(f"性能警告: {operation} 耗时 {duration_ms:.2f}ms", 
                        context=perf_context)
        else:
            level_method = getattr(self, level.lower(), self.info)
            level_method(f"性能: {operation} 耗时 {duration_ms:.2f}ms", 
                        context=perf_context)
    
    def with_context(self, **context_kwargs):
        """返回一个带有上下文的记录器"""
        return ContextLogger(self, context_kwargs)
    
    def __getattr__(self, name):
        """代理其他属性和方法到原始记录器"""
        return getattr(self.logger, name)

class ContextLogger:
    """上下文日志记录器"""
    
    def __init__(self, enhanced_logger: EnhancedLogger, context: Dict):
        self.enhanced_logger = enhanced_logger
        self.context = context
    
    def __getattr__(self, name):
        """代理日志方法，添加上下文"""
        if hasattr(self.enhanced_logger, name) and callable(getattr(self.enhanced_logger, name)):
            method = getattr(self.enhanced_logger, name)
            
            def wrapper(*args, **kwargs):
                # 合并上下文
                kwargs_context = kwargs.pop('context', {})
                merged_context = {**self.context, **kwargs_context}
                if merged_context:
                    kwargs['context'] = merged_context
                return method(*args, **kwargs)
            
            return wrapper
        return getattr(self.enhanced_logger, name)

class LogContextManager:
    """日志上下文管理器"""
    
    def __init__(self):
        self.local = threading.local()
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        if not hasattr(self.local, 'context_stack'):
            self.local.context_stack = []
        
        if not self.local.context_stack:
            return {}
        
        # 合并所有上下文
        merged_context = {}
        for context in self.local.context_stack:
            if context:
                merged_context.update(context)
        
        return merged_context
    
    def set_context(self, **kwargs):
        """设置上下文"""
        if not hasattr(self.local, 'context_stack'):
            self.local.context_stack = []
        
        self.local.context_stack.append(kwargs)
    
    def update_context(self, **kwargs):
        """更新当前上下文"""
        if not hasattr(self.local, 'context_stack'):
            self.local.context_stack = [kwargs]
        elif self.local.context_stack:
            self.local.context_stack[-1].update(kwargs)
        else:
            self.local.context_stack.append(kwargs)
    
    def clear_context(self):
        """清除当前上下文"""
        if hasattr(self.local, 'context_stack'):
            if self.local.context_stack:
                self.local.context_stack.pop()
    
    def push_context(self, **kwargs):
        """推入新的上下文"""
        if not hasattr(self.local, 'context_stack'):
            self.local.context_stack = []
        
        self.local.context_stack.append(kwargs)
    
    def pop_context(self) -> Dict[str, Any]:
        """弹出上下文"""
        if hasattr(self.local, 'context_stack') and self.local.context_stack:
            return self.local.context_stack.pop()
        return {}
    
    def context_decorator(self, **context_kwargs):
        """上下文装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.push_context(**context_kwargs)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.pop_context()
            return wrapper
        return decorator

# 全局日志管理器实例
_log_manager = None

def get_log_manager() -> StructuredLogger:
    """获取全局日志管理器实例"""
    global _log_manager
    if _log_manager is None:
        _log_manager = StructuredLogger()
    return _log_manager

def setup_logger(name: str, level: str = None) -> EnhancedLogger:
    """
    设置日志记录器
    
    Args:
        name: 记录器名称
        level: 日志级别
        
    Returns:
        EnhancedLogger: 增强的日志记录器
    """
    return get_log_manager().get_logger(name, level)

def get_logger(name: str) -> EnhancedLogger:
    """
    获取日志记录器
    
    Args:
        name: 记录器名称
        
    Returns:
        EnhancedLogger: 增强的日志记录器
    """
    return get_log_manager().get_logger(name)

def configure_logging(config: Dict[str, Any]):
    """
    配置日志系统
    
    Args:
        config: 配置字典
    """
    get_log_manager().configure(config)

def log_with_context(func=None, **context_kwargs):
    """
    带上下文的日志装饰器
    
    Args:
        func: 被装饰的函数
        **context_kwargs: 上下文参数
        
    Returns:
        装饰后的函数
    """
    if func is None:
        return lambda f: log_with_context(f, **context_kwargs)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        context_manager = LogContextManager()
        context_manager.push_context(**context_kwargs)
        try:
            return func(*args, **kwargs)
        finally:
            context_manager.pop_context()
    
    return wrapper

def log_performance(operation: str = None):
    """
    性能日志装饰器
    
    Args:
        operation: 操作名称，如果为None则使用函数名
        
    Returns:
        装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                logger = get_logger(func.__module__)
                logger.performance(op_name, duration_ms)
        
        return wrapper
    return decorator

class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, operation: str, logger: EnhancedLogger = None, 
                 level: str = "INFO", context: Dict = None):
        """
        初始化计时器
        
        Args:
            operation: 操作名称
            logger: 日志记录器
            level: 日志级别
            context: 上下文
        """
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.level = level
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        # 添加上下文
        perf_context = self.context.copy()
        perf_context.update({
            'operation': self.operation,
            'duration_ms': duration_ms,
            'duration_s': duration_ms / 1000.0,
        })
        
        # 记录性能日志
        if exc_type is None:  # 没有异常
            self.logger.performance(self.operation, duration_ms, perf_context, self.level)
        else:  # 有异常
            self.logger.error(f"{self.operation} 执行失败，耗时 {duration_ms:.2f}ms", 
                            context=perf_context, exc_info=True)
    
    def get_elapsed_ms(self) -> float:
        """
        获取已用时间（毫秒）
        
        Returns:
            float: 已用时间（毫秒）
        """
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000

# 常用快捷函数
def debug(msg: str, context: Dict = None, logger_name: str = "root"):
    """DEBUG级别日志"""
    get_log_manager().debug(msg, context, logger_name)

def info(msg: str, context: Dict = None, logger_name: str = "root"):
    """INFO级别日志"""
    get_log_manager().info(msg, context, logger_name)

def warning(msg: str, context: Dict = None, logger_name: str = "root"):
    """WARNING级别日志"""
    get_log_manager().warning(msg, context, logger_name)

def error(msg: str, context: Dict = None, exc_info: bool = False, 
         logger_name: str = "root"):
    """ERROR级别日志"""
    get_log_manager().error(msg, context, exc_info, logger_name)

def critical(msg: str, context: Dict = None, exc_info: bool = False, 
            logger_name: str = "root"):
    """CRITICAL级别日志"""
    get_log_manager().critical(msg, context, exc_info, logger_name)

def performance(operation: str, duration_ms: float, context: Dict = None, 
               level: str = "INFO"):
    """记录性能日志"""
    get_log_manager().log_performance(operation, duration_ms, context, level)

# 默认配置
DEFAULT_CONFIG = {
    'default_level': 'INFO',
    'log_dir': 'logs',
    'max_file_size_mb': 100,
    'backup_count': 10,
    'rotate_daily': True,
    'enable_console': True,
    'enable_file': True,
    'enable_json_file': False,
    'enable_metrics': True,
    'log_performance': False,
}

# 初始化默认配置
configure_logging(DEFAULT_CONFIG)