"""
信号处理系统日志配置文件
支持结构化日志、日志轮转、多级别日志、日志聚合
"""
import os
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
import time
import threading
import queue
import sys
import warnings
from datetime import datetime, timedelta
import hashlib
import re
import socket
import traceback
from contextlib import contextmanager


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """从字符串获取日志级别"""
        level_str = level_str.upper()
        for level in cls:
            if level.value == level_str:
                return level
        raise ValueError(f"无效的日志级别: {level_str}")


class LogFormat(str, Enum):
    """日志格式枚举"""
    SIMPLE = "simple"          # 简单格式
    DETAILED = "detailed"      # 详细格式
    JSON = "json"              # JSON格式
    GELF = "gelf"              # Graylog扩展日志格式
    CUSTOM = "custom"          # 自定义格式


class LogHandler(str, Enum):
    """日志处理器枚举"""
    CONSOLE = "console"        # 控制台处理器
    FILE = "file"              # 文件处理器
    ROTATING_FILE = "rotating_file"  # 轮转文件处理器
    TIMED_ROTATING_FILE = "timed_rotating_file"  # 时间轮转文件处理器
    SYSLOG = "syslog"          # Syslog处理器
    HTTP = "http"              # HTTP处理器
    QUEUE = "queue"            # 队列处理器
    NULL = "null"              # 空处理器（用于测试）


class CompressionMethod(str, Enum):
    """压缩方法枚举"""
    NONE = "none"              # 不压缩
    GZIP = "gzip"              # Gzip压缩
    BZIP2 = "bzip2"            # Bzip2压缩
    LZMA = "lzma"              # LZMA压缩


class LogAggregationMethod(str, Enum):
    """日志聚合方法枚举"""
    NONE = "none"              # 不聚合
    TIME_BASED = "time_based"  # 基于时间聚合
    COUNT_BASED = "count_based"  # 基于数量聚合
    SIZE_BASED = "size_based"  # 基于大小聚合


class LogFilterType(str, Enum):
    """日志过滤器类型枚举"""
    LEVEL = "level"            # 级别过滤
    MODULE = "module"          # 模块过滤
    MESSAGE = "message"        # 消息过滤
    REGEX = "regex"            # 正则表达式过滤
    CUSTOM = "custom"          # 自定义过滤


@dataclass
class LogFilterConfig:
    """日志过滤器配置"""
    
    # 过滤器类型
    filter_type: LogFilterType = LogFilterType.LEVEL
    
    # 过滤条件
    level: LogLevel = LogLevel.INFO
    module_pattern: str = ".*"
    message_pattern: str = ".*"
    regex_pattern: str = ".*"
    
    # 过滤行为
    include: bool = True        # True=包含匹配的，False=排除匹配的
    case_sensitive: bool = False  # 是否大小写敏感
    
    def create_filter(self) -> logging.Filter:
        """创建日志过滤器"""
        if self.filter_type == LogFilterType.LEVEL:
            return LevelFilter(self.level, self.include)
        elif self.filter_type == LogFilterType.MODULE:
            return ModuleFilter(self.module_pattern, self.include, self.case_sensitive)
        elif self.filter_type == LogFilterType.MESSAGE:
            return MessageFilter(self.message_pattern, self.include, self.case_sensitive)
        elif self.filter_type == LogFilterType.REGEX:
            return RegexFilter(self.regex_pattern, self.include, self.case_sensitive)
        else:
            raise ValueError(f"不支持的过滤器类型: {self.filter_type}")


class LevelFilter(logging.Filter):
    """级别过滤器"""
    
    def __init__(self, level: LogLevel, include: bool = True):
        super().__init__()
        self.level = getattr(logging, level.value)
        self.include = include
    
    def filter(self, record: logging.LogRecord) -> bool:
        if self.include:
            return record.levelno >= self.level
        else:
            return record.levelno < self.level


class ModuleFilter(logging.Filter):
    """模块过滤器"""
    
    def __init__(self, pattern: str, include: bool = True, case_sensitive: bool = False):
        super().__init__()
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)
        self.include = include
    
    def filter(self, record: logging.LogRecord) -> bool:
        matches = bool(self.pattern.search(record.name))
        return matches if self.include else not matches


class MessageFilter(logging.Filter):
    """消息过滤器"""
    
    def __init__(self, pattern: str, include: bool = True, case_sensitive: bool = False):
        super().__init__()
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)
        self.include = include
    
    def filter(self, record: logging.LogRecord) -> bool:
        matches = bool(self.pattern.search(record.getMessage()))
        return matches if self.include else not matches


class RegexFilter(logging.Filter):
    """正则表达式过滤器"""
    
    def __init__(self, pattern: str, include: bool = True, case_sensitive: bool = False):
        super().__init__()
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(pattern, flags)
        self.include = include
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 检查模块名
        module_match = bool(self.pattern.search(record.name))
        
        # 检查消息
        message_match = bool(self.pattern.search(record.getMessage()))
        
        # 检查异常信息
        exc_match = False
        if record.exc_info and record.exc_text:
            exc_match = bool(self.pattern.search(record.exc_text))
        
        matches = module_match or message_match or exc_match
        return matches if self.include else not matches


@dataclass
class LogFormatterConfig:
    """日志格式化器配置"""
    
    # 格式类型
    format_type: LogFormat = LogFormat.DETAILED
    
    # 格式字符串
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
    
    # 日期格式
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # JSON格式配置
    json_fields: List[str] = field(default_factory=lambda: [
        "timestamp", "level", "logger", "message", "module", "function", 
        "line", "process", "thread", "exception"
    ])
    json_indent: Optional[int] = None
    json_ensure_ascii: bool = False
    
    # 颜色配置（控制台）
    use_colors: bool = True
    color_map: Dict[str, str] = field(default_factory=lambda: {
        "DEBUG": "\033[94m",     # 蓝色
        "INFO": "\033[92m",      # 绿色
        "WARNING": "\033[93m",   # 黄色
        "ERROR": "\033[91m",     # 红色
        "CRITICAL": "\033[95m",  # 紫色
    })
    
    def create_formatter(self) -> logging.Formatter:
        """创建日志格式化器"""
        if self.format_type == LogFormat.JSON:
            return JSONFormatter(
                fields=self.json_fields,
                indent=self.json_indent,
                ensure_ascii=self.json_ensure_ascii,
                datefmt=self.date_format
            )
        elif self.format_type == LogFormat.GELF:
            return GELFFormatter(datefmt=self.date_format)
        else:
            if self.format_type == LogFormat.DETAILED:
                fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
            elif self.format_type == LogFormat.SIMPLE:
                fmt = "%(levelname)s - %(message)s"
            else:  # CUSTOM
                fmt = self.format_string
            
            if self.use_colors and sys.stdout.isatty():
                return ColoredFormatter(fmt, self.date_format, color_map=self.color_map)
            else:
                return logging.Formatter(fmt, self.date_format)


class ColoredFormatter(logging.Formatter):
    """彩色格式化器"""
    
    RESET = "\033[0m"
    
    def __init__(self, fmt: str, datefmt: str, color_map: Dict[str, str]):
        super().__init__(fmt, datefmt)
        self.color_map = color_map
    
    def format(self, record: logging.LogRecord) -> str:
        # 保存原始级别名
        original_levelname = record.levelname
        
        # 添加颜色
        if original_levelname in self.color_map:
            color = self.color_map[original_levelname]
            record.levelname = f"{color}{original_levelname}{self.RESET}"
        
        result = super().format(record)
        
        # 恢复原始级别名
        record.levelname = original_levelname
        
        return result


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self, fields: List[str], indent: Optional[int] = None, 
                 ensure_ascii: bool = False, datefmt: str = None):
        super().__init__(datefmt=datefmt)
        self.fields = fields
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {}
        
        for field in self.fields:
            if field == "timestamp":
                log_data[field] = self.formatTime(record, self.datefmt)
            elif field == "level":
                log_data[field] = record.levelname
            elif field == "logger":
                log_data[field] = record.name
            elif field == "message":
                log_data[field] = record.getMessage()
            elif field == "module":
                log_data[field] = record.module
            elif field == "function":
                log_data[field] = record.funcName
            elif field == "line":
                log_data[field] = record.lineno
            elif field == "process":
                log_data[field] = record.process
            elif field == "thread":
                log_data[field] = record.thread
            elif field == "threadName":
                log_data[field] = record.threadName
            elif field == "pathname":
                log_data[field] = record.pathname
            elif field == "filename":
                log_data[field] = record.filename
            elif field == "exception":
                if record.exc_info:
                    log_data[field] = self.formatException(record.exc_info)
            else:
                # 额外字段
                if hasattr(record, field):
                    log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, indent=self.indent, ensure_ascii=self.ensure_ascii)


class GELFFormatter(logging.Formatter):
    """GELF格式化器（用于Graylog）"""
    
    def __init__(self, datefmt: str = None):
        super().__init__(datefmt=datefmt)
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        gelf_data = {
            "version": "1.1",
            "host": self.hostname,
            "short_message": record.getMessage(),
            "full_message": self.formatException(record.exc_info) if record.exc_info else "",
            "timestamp": time.time(),
            "level": self._map_level(record.levelno),
            "_logger": record.name,
            "_module": record.module,
            "_function": record.funcName,
            "_line": record.lineno,
            "_process": record.process,
            "_thread": record.thread,
        }
        
        # 添加额外字段
        for key, value in record.__dict__.items():
            if key.startswith("_") and key not in gelf_data:
                gelf_data[key] = value
        
        return json.dumps(gelf_data, ensure_ascii=False)
    
    def _map_level(self, levelno: int) -> int:
        """映射Python日志级别到Syslog级别"""
        if levelno >= logging.CRITICAL:
            return 2
        elif levelno >= logging.ERROR:
            return 3
        elif levelno >= logging.WARNING:
            return 4
        elif levelno >= logging.INFO:
            return 6
        else:  # DEBUG
            return 7


@dataclass
class LogHandlerConfig:
    """日志处理器配置"""
    
    # 处理器类型
    handler_type: LogHandler = LogHandler.CONSOLE
    
    # 通用配置
    level: LogLevel = LogLevel.INFO
    formatter: LogFormatterConfig = field(default_factory=LogFormatterConfig)
    filters: List[LogFilterConfig] = field(default_factory=list)
    
    # 控制台处理器配置
    console_stream: str = "stdout"  # stdout/stderr
    
    # 文件处理器配置
    filename: str = "logs/app.log"
    mode: str = "a"  # 文件模式
    encoding: str = "utf-8"
    
    # 轮转文件处理器配置
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # 时间轮转文件处理器配置
    when: str = "midnight"  # S, M, H, D, midnight, W0-W6
    interval: int = 1
    backup_count_timed: int = 30
    utc: bool = False
    at_time: Optional[str] = None  # HH:MM格式
    
    # 压缩配置
    compression: CompressionMethod = CompressionMethod.GZIP
    compression_level: int = 6
    
    # Syslog处理器配置
    syslog_address: str = "/dev/log"  # Unix socket
    syslog_facility: str = "user"
    
    # HTTP处理器配置
    http_url: str = "http://localhost:8080/log"
    http_method: str = "POST"
    http_headers: Dict[str, str] = field(default_factory=dict)
    http_timeout: int = 5
    
    # 队列处理器配置
    queue_maxsize: int = 10000
    queue_respect_handler_level: bool = True
    
    def create_handler(self) -> logging.Handler:
        """创建日志处理器"""
        # 创建基础处理器
        if self.handler_type == LogHandler.CONSOLE:
            if self.console_stream == "stdout":
                stream = sys.stdout
            else:
                stream = sys.stderr
            handler = logging.StreamHandler(stream)
            
        elif self.handler_type == LogHandler.FILE:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            handler = logging.FileHandler(
                filename=self.filename,
                mode=self.mode,
                encoding=self.encoding
            )
            
        elif self.handler_type == LogHandler.ROTATING_FILE:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                filename=self.filename,
                mode=self.mode,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding
            )
            
        elif self.handler_type == LogHandler.TIMED_ROTATING_FILE:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=self.filename,
                when=self.when,
                interval=self.interval,
                backupCount=self.backup_count_timed,
                encoding=self.encoding,
                utc=self.utc,
                atTime=datetime.strptime(self.at_time, "%H:%M").time() if self.at_time else None
            )
            
        elif self.handler_type == LogHandler.SYSLOG:
            if self.syslog_address.startswith("/"):
                # Unix socket
                address = self.syslog_address
            else:
                # TCP/UDP
                if ":" in self.syslog_address:
                    host, port = self.syslog_address.split(":", 1)
                    address = (host, int(port))
                else:
                    address = (self.syslog_address, 514)
            
            facility = getattr(logging.handlers.SysLogHandler, f"LOG_{self.syslog_facility.upper()}", 
                             logging.handlers.SysLogHandler.LOG_USER)
            handler = logging.handlers.SysLogHandler(
                address=address,
                facility=facility
            )
            
        elif self.handler_type == LogHandler.HTTP:
            handler = logging.handlers.HTTPHandler(
                host=self.http_url,
                url="",  # URL在http_url中指定
                method=self.http_method,
                secure=False  # 通过URL的scheme判断
            )
            
        elif self.handler_type == LogHandler.QUEUE:
            handler = logging.handlers.QueueHandler(
                queue.Queue(maxsize=self.queue_maxsize)
            )
            
        elif self.handler_type == LogHandler.NULL:
            handler = logging.NullHandler()
            
        else:
            raise ValueError(f"不支持的处理器类型: {self.handler_type}")
        
        # 设置级别
        handler.setLevel(getattr(logging, self.level.value))
        
        # 设置格式化器
        handler.setFormatter(self.formatter.create_formatter())
        
        # 添加过滤器
        for filter_config in self.filters:
            handler.addFilter(filter_config.create_filter())
        
        return handler


@dataclass
class LogAggregationConfig:
    """日志聚合配置"""
    
    # 是否启用聚合
    enabled: bool = False
    
    # 聚合方法
    method: LogAggregationMethod = LogAggregationMethod.TIME_BASED
    
    # 时间聚合配置
    time_window_seconds: int = 60  # 时间窗口（秒）
    
    # 数量聚合配置
    max_messages: int = 100  # 最大消息数
    
    # 大小聚合配置
    max_size_bytes: int = 10 * 1024  # 10KB
    
    # 聚合键
    group_by: List[str] = field(default_factory=lambda: ["level", "logger"])
    
    # 聚合格式
    format_string: str = "聚合了 {count} 条{level}日志，来自{logger}，时间范围: {start_time} - {end_time}"
    
    def should_aggregate(self, buffer: List[logging.LogRecord]) -> bool:
        """检查是否应该聚合"""
        if not self.enabled or not buffer:
            return False
        
        if self.method == LogAggregationMethod.TIME_BASED:
            if len(buffer) < 2:
                return False
            first_time = buffer[0].created
            last_time = buffer[-1].created
            return (last_time - first_time) >= self.time_window_seconds
            
        elif self.method == LogAggregationMethod.COUNT_BASED:
            return len(buffer) >= self.max_messages
            
        elif self.method == LogAggregationMethod.SIZE_BASED:
            total_size = sum(len(json.dumps(record.__dict__)) for record in buffer)
            return total_size >= self.max_size_bytes
            
        else:
            return False
    
    def create_aggregated_record(self, buffer: List[logging.LogRecord]) -> logging.LogRecord:
        """创建聚合日志记录"""
        if not buffer:
            return None
        
        # 使用第一条记录作为基础
        base_record = buffer[0]
        
        # 计算聚合信息
        count = len(buffer)
        levels = {}
        loggers = {}
        
        for record in buffer:
            levels[record.levelname] = levels.get(record.levelname, 0) + 1
            loggers[record.name] = loggers.get(record.name, 0) + 1
        
        # 构建聚合消息
        level_str = "/".join(sorted(levels.keys()))
        logger_str = "/".join(sorted(loggers.keys())[:3])  # 只显示前3个
        if len(loggers) > 3:
            logger_str += f" 等{len(loggers)}个"
        
        start_time = datetime.fromtimestamp(buffer[0].created)
        end_time = datetime.fromtimestamp(buffer[-1].created)
        
        message = self.format_string.format(
            count=count,
            level=level_str,
            logger=logger_str,
            start_time=start_time.strftime("%H:%M:%S"),
            end_time=end_time.strftime("%H:%M:%S"),
            levels=levels,
            loggers=loggers
        )
        
        # 创建新记录
        aggregated_record = logging.LogRecord(
            name="aggregated",
            level=base_record.levelno,
            pathname=base_record.pathname,
            lineno=base_record.lineno,
            msg=message,
            args=(),
            exc_info=None,
            func=base_record.funcName
        )
        
        # 添加聚合元数据
        aggregated_record.aggregated_count = count
        aggregated_record.aggregated_levels = levels
        aggregated_record.aggregated_loggers = loggers
        aggregated_record.aggregated_time_range = (start_time, end_time)
        
        return aggregated_record


@dataclass
class LogRotationConfig:
    """日志轮转配置"""
    
    # 是否启用轮转
    enabled: bool = True
    
    # 轮转策略
    strategy: str = "time"  # time/size/both
    
    # 时间轮转
    rotate_when: str = "midnight"  # S, M, H, D, W0-W6, midnight
    rotate_interval: int = 1
    
    # 大小轮转
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_files: int = 30  # 保留文件数
    
    # 压缩
    compress: bool = True
    compression_method: CompressionMethod = CompressionMethod.GZIP
    compression_level: int = 6
    
    # 清理旧日志
    cleanup_old_logs: bool = True
    max_age_days: int = 30  # 最大保留天数
    cleanup_cron: str = "0 2 * * *"  # 清理时间（cron表达式）
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.enabled:
            if self.max_file_size <= 0:
                errors.append("最大文件大小必须大于0")
            
            if self.max_files <= 0:
                errors.append("最大文件数必须大于0")
            
            if self.max_age_days <= 0:
                errors.append("最大保留天数必须大于0")
        
        return errors


@dataclass
class PerformanceMonitoringConfig:
    """性能监控配置"""
    
    # 是否启用性能监控
    enabled: bool = True
    
    # 监控间隔
    monitor_interval_seconds: int = 60
    
    # 监控指标
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_disk: bool = True
    monitor_queue: bool = True
    monitor_throughput: bool = True
    
    # 阈值告警
    memory_threshold_mb: int = 100  # 内存使用阈值
    cpu_threshold_percent: float = 80.0  # CPU使用阈值
    disk_threshold_percent: float = 90.0  # 磁盘使用阈值
    queue_threshold: int = 1000  # 队列长度阈值
    
    # 监控日志
    log_performance_metrics: bool = True
    log_interval_seconds: int = 300  # 5分钟
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        import psutil
        import threading
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.monitor_memory:
            process = psutil.Process()
            metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        
        if self.monitor_cpu:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            process = psutil.Process()
            metrics["process_cpu_percent"] = process.cpu_percent()
        
        if self.monitor_disk:
            # 获取当前日志目录的磁盘使用情况
            log_dirs = ["logs", "data/logs"]
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    disk_usage = psutil.disk_usage(log_dir)
                    metrics["disk_usage_percent"] = disk_usage.percent
                    metrics["disk_free_gb"] = disk_usage.free / 1024 / 1024 / 1024
                    break
        
        if self.monitor_queue:
            # 获取活动线程数
            metrics["active_threads"] = threading.active_count()
            
            # 获取队列大小（如果有的话）
            for handler in logging.root.handlers:
                if hasattr(handler, 'queue'):
                    metrics["queue_size"] = handler.queue.qsize()
                    break
        
        if self.monitor_throughput:
            # 这里需要应用自己记录吞吐量指标
            pass
        
        return metrics


@dataclass
class ModuleLogLevelConfig:
    """模块日志级别配置"""
    
    # 模块日志级别映射
    levels: Dict[str, LogLevel] = field(default_factory=lambda: {
        "__main__": LogLevel.INFO,
        "data_acquisition": LogLevel.INFO,
        "algorithms": LogLevel.DEBUG,
        "database": LogLevel.INFO,
        "web": LogLevel.INFO,
        "utils": LogLevel.WARNING,
        "tests": LogLevel.DEBUG,
    })
    
    # 模块匹配模式
    patterns: List[Tuple[str, LogLevel]] = field(default_factory=lambda: [
        (r"^src\.core\.", LogLevel.INFO),
        (r"^src\.algorithms\.", LogLevel.DEBUG),
        (r"^src\.utils\.", LogLevel.WARNING),
        (r"^src\.web\.", LogLevel.INFO),
        (r"^tests\.", LogLevel.DEBUG),
    ])
    
    def get_module_level(self, module_name: str) -> Optional[LogLevel]:
        """获取模块的日志级别"""
        # 首先检查精确匹配
        if module_name in self.levels:
            return self.levels[module_name]
        
        # 然后检查模式匹配
        for pattern, level in self.patterns:
            if re.match(pattern, module_name):
                return level
        
        return None


@dataclass
class LoggingConfig:
    """日志主配置"""
    
    # 环境标识
    environment: str = "development"  # development, testing, production
    
    # 根日志配置
    root_level: LogLevel = LogLevel.INFO
    disable_existing_loggers: bool = False
    propagate: bool = True
    
    # 处理器配置
    handlers: List[LogHandlerConfig] = field(default_factory=list)
    
    # 模块级别配置
    module_levels: ModuleLogLevelConfig = field(default_factory=ModuleLogLevelConfig)
    
    # 日志聚合配置
    aggregation: LogAggregationConfig = field(default_factory=LogAggregationConfig)
    
    # 日志轮转配置
    rotation: LogRotationConfig = field(default_factory=LogRotationConfig)
    
    # 性能监控配置
    performance: PerformanceMonitoringConfig = field(default_factory=PerformanceMonitoringConfig)
    
    # 高级配置
    capture_warnings: bool = True
    capture_unhandled_exceptions: bool = True
    log_thread_names: bool = True
    log_process_names: bool = True
    log_task_names: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 根据环境调整配置
        self._adjust_for_environment()
        
        # 如果没有处理器，添加默认处理器
        if not self.handlers:
            self._add_default_handlers()
        
        # 确保日志目录存在
        self._ensure_log_dirs()
    
    def _adjust_for_environment(self):
        """根据环境调整配置"""
        if self.environment == "production":
            # 生产环境配置
            self.root_level = LogLevel.INFO
            self.capture_unhandled_exceptions = True
            
            # 生产环境使用JSON格式和文件处理器
            json_formatter = LogFormatterConfig(
                format_type=LogFormat.JSON,
                json_fields=["timestamp", "level", "logger", "message", "module", "function", "line"]
            )
            
            self.handlers = [
                LogHandlerConfig(
                    handler_type=LogHandler.CONSOLE,
                    level=LogLevel.WARNING,
                    formatter=json_formatter
                ),
                LogHandlerConfig(
                    handler_type=LogHandler.TIMED_ROTATING_FILE,
                    filename="/var/log/signal_processing/app.log",
                    level=LogLevel.INFO,
                    formatter=json_formatter,
                    when="midnight",
                    interval=1,
                    backup_count_timed=30
                ),
                LogHandlerConfig(
                    handler_type=LogHandler.TIMED_ROTATING_FILE,
                    filename="/var/log/signal_processing/error.log",
                    level=LogLevel.ERROR,
                    formatter=json_formatter,
                    when="midnight",
                    interval=1,
                    backup_count_timed=90
                )
            ]
            
        elif self.environment == "testing":
            # 测试环境配置
            self.root_level = LogLevel.WARNING
            self.disable_existing_loggers = False
            
            # 测试环境只记录错误
            self.handlers = [
                LogHandlerConfig(
                    handler_type=LogHandler.CONSOLE,
                    level=LogLevel.WARNING,
                    formatter=LogFormatterConfig(format_type=LogFormat.SIMPLE)
                )
            ]
            
        else:  # development
            # 开发环境配置
            self.root_level = LogLevel.DEBUG
            
            # 开发环境使用详细格式和控制台输出
            detailed_formatter = LogFormatterConfig(
                format_type=LogFormat.DETAILED,
                use_colors=True
            )
            
            self.handlers = [
                LogHandlerConfig(
                    handler_type=LogHandler.CONSOLE,
                    level=LogLevel.DEBUG,
                    formatter=detailed_formatter
                ),
                LogHandlerConfig(
                    handler_type=LogHandler.ROTATING_FILE,
                    filename="logs/app.log",
                    level=LogLevel.INFO,
                    formatter=detailed_formatter,
                    max_bytes=10 * 1024 * 1024,  # 10MB
                    backup_count=5
                )
            ]
    
    def _add_default_handlers(self):
        """添加默认处理器"""
        if self.environment == "development":
            formatter = LogFormatterConfig(
                format_type=LogFormat.DETAILED,
                use_colors=True
            )
            self.handlers = [
                LogHandlerConfig(
                    handler_type=LogHandler.CONSOLE,
                    level=LogLevel.DEBUG,
                    formatter=formatter
                )
            ]
        else:
            formatter = LogFormatterConfig(format_type=LogFormat.JSON)
            self.handlers = [
                LogHandlerConfig(
                    handler_type=LogHandler.CONSOLE,
                    level=LogLevel.INFO,
                    formatter=formatter
                )
            ]
    
    def _ensure_log_dirs(self):
        """确保日志目录存在"""
        for handler_config in self.handlers:
            if handler_config.handler_type in [LogHandler.FILE, LogHandler.ROTATING_FILE, 
                                              LogHandler.TIMED_ROTATING_FILE]:
                log_dir = os.path.dirname(handler_config.filename)
                if log_dir:  # 空字符串表示当前目录
                    os.makedirs(log_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为logging.config.dictConfig兼容的字典"""
        config_dict = {
            'version': 1,
            'disable_existing_loggers': self.disable_existing_loggers,
            'formatters': {},
            'filters': {},
            'handlers': {},
            'loggers': {},
            'root': {
                'level': self.root_level.value,
                'handlers': [],
                'propagate': self.propagate
            }
        }
        
        # 添加工厂函数
        config_dict['filters'] = {
            'module_filter': {
                '()': 'config.logging_config.ModuleFilter',
                'pattern': '.*',
                'include': True,
                'case_sensitive': False
            }
        }
        
        # 添加格式化器
        for i, handler_config in enumerate(self.handlers):
            formatter_name = f'formatter_{i}'
            if handler_config.formatter.format_type == LogFormat.JSON:
                config_dict['formatters'][formatter_name] = {
                    '()': 'config.logging_config.JSONFormatter',
                    'fields': handler_config.formatter.json_fields,
                    'indent': handler_config.formatter.json_indent,
                    'ensure_ascii': handler_config.formatter.json_ensure_ascii,
                    'datefmt': handler_config.formatter.date_format
                }
            elif handler_config.formatter.format_type == LogFormat.GELF:
                config_dict['formatters'][formatter_name] = {
                    '()': 'config.logging_config.GELFFormatter',
                    'datefmt': handler_config.formatter.date_format
                }
            else:
                config_dict['formatters'][formatter_name] = {
                    'format': handler_config.formatter.format_string,
                    'datefmt': handler_config.formatter.date_format
                }
            
            # 添加处理器
            handler_name = f'handler_{i}'
            handler_dict = {
                'class': self._get_handler_class(handler_config.handler_type),
                'level': handler_config.level.value,
                'formatter': formatter_name,
            }
            
            # 添加处理器特定配置
            if handler_config.handler_type == LogHandler.CONSOLE:
                handler_dict['stream'] = f'sys.{handler_config.console_stream}'
            elif handler_config.handler_type in [LogHandler.FILE, LogHandler.ROTATING_FILE, 
                                                LogHandler.TIMED_ROTATING_FILE]:
                handler_dict['filename'] = handler_config.filename
                handler_dict['encoding'] = handler_config.encoding
                
                if handler_config.handler_type == LogHandler.ROTATING_FILE:
                    handler_dict['maxBytes'] = handler_config.max_bytes
                    handler_dict['backupCount'] = handler_config.backup_count
                elif handler_config.handler_type == LogHandler.TIMED_ROTATING_FILE:
                    handler_dict['when'] = handler_config.when
                    handler_dict['interval'] = handler_config.interval
                    handler_dict['backupCount'] = handler_config.backup_count_timed
                    handler_dict['utc'] = handler_config.utc
                    if handler_config.at_time:
                        handler_dict['atTime'] = handler_config.at_time
            
            config_dict['handlers'][handler_name] = handler_dict
            config_dict['root']['handlers'].append(handler_name)
        
        # 添加模块特定配置
        for module_name, level in self.module_levels.levels.items():
            config_dict['loggers'][module_name] = {
                'level': level.value,
                'handlers': config_dict['root']['handlers'].copy(),
                'propagate': self.propagate
            }
        
        return config_dict
    
    def _get_handler_class(self, handler_type: LogHandler) -> str:
        """获取处理器类名"""
        if handler_type == LogHandler.CONSOLE:
            return "logging.StreamHandler"
        elif handler_type == LogHandler.FILE:
            return "logging.FileHandler"
        elif handler_type == LogHandler.ROTATING_FILE:
            return "logging.handlers.RotatingFileHandler"
        elif handler_type == LogHandler.TIMED_ROTATING_FILE:
            return "logging.handlers.TimedRotatingFileHandler"
        elif handler_type == LogHandler.SYSLOG:
            return "logging.handlers.SysLogHandler"
        elif handler_type == LogHandler.HTTP:
            return "logging.handlers.HTTPHandler"
        elif handler_type == LogHandler.QUEUE:
            return "logging.handlers.QueueHandler"
        elif handler_type == LogHandler.NULL:
            return "logging.NullHandler"
        else:
            raise ValueError(f"不支持的处理器类型: {handler_type}")
    
    def configure_logging(self):
        """配置日志系统"""
        # 配置标准logging
        logging.config.dictConfig(self.to_dict())
        
        # 捕获警告
        if self.capture_warnings:
            logging.captureWarnings(True)
        
        # 捕获未处理异常
        if self.capture_unhandled_exceptions:
            self._setup_exception_handling()
        
        # 设置线程和进程名称
        if self.log_thread_names:
            threading.current_thread().name = "MainThread"
        
        # 记录配置信息
        logger = logging.getLogger(__name__)
        logger.info(f"日志系统已配置，环境: {self.environment}")
        logger.info(f"根日志级别: {self.root_level}")
        logger.info(f"处理器数量: {len(self.handlers)}")
    
    def _setup_exception_handling(self):
        """设置异常处理"""
        def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
            """处理未捕获的异常"""
            if issubclass(exc_type, KeyboardInterrupt):
                # 不记录键盘中断
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger = logging.getLogger(__name__)
            logger.critical(
                "未处理的异常",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        
        sys.excepthook = handle_unhandled_exception
        
        # 设置线程异常处理
        threading.excepthook = lambda args: handle_unhandled_exception(
            args.exc_type, args.exc_value, args.exc_traceback
        )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # 验证处理器配置
        for i, handler_config in enumerate(self.handlers):
            if handler_config.handler_type in [LogHandler.FILE, LogHandler.ROTATING_FILE, 
                                              LogHandler.TIMED_ROTATING_FILE]:
                log_dir = os.path.dirname(handler_config.filename)
                if log_dir and not os.path.exists(log_dir):
                    try:
                        os.makedirs(log_dir, exist_ok=True)
                    except Exception as e:
                        errors.append(f"无法创建日志目录 {log_dir}: {e}")
        
        # 验证轮转配置
        rotation_errors = self.rotation.validate()
        errors.extend(rotation_errors)
        
        return len(errors) == 0, errors
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_yaml(self) -> str:
        """转换为YAML字符串"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    def save_to_file(self, filepath: str, format: str = "json"):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'LoggingConfig':
        """从文件加载配置"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """从字典创建配置"""
        # 转换嵌套的dataclass
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                field_value = data[field_name]
                
                if hasattr(field_type, '__origin__') and field_type.__origin__ == type:
                    # 处理dataclass字段
                    field_class = field_type.__args__[0]
                    kwargs[field_name] = field_class(**field_value)
                elif isinstance(field_value, dict) and hasattr(field_type, '__dataclass_fields__'):
                    # 处理dataclass字段
                    kwargs[field_name] = field_type(**field_value)
                else:
                    kwargs[field_name] = field_value
        
        return cls(**kwargs)


# 配置管理函数
def get_default_config(environment: str = None) -> LoggingConfig:
    """
    获取默认配置
    
    Args:
        environment: 环境标识
        
    Returns:
        LoggingConfig: 默认配置
    """
    config = LoggingConfig()
    
    if environment:
        config.environment = environment
        config._adjust_for_environment()
    
    return config


def get_production_config() -> LoggingConfig:
    """获取生产环境配置"""
    config = LoggingConfig(environment="production")
    return config


def get_development_config() -> LoggingConfig:
    """获取开发环境配置"""
    config = LoggingConfig(environment="development")
    return config


def get_testing_config() -> LoggingConfig:
    """获取测试环境配置"""
    config = LoggingConfig(environment="testing")
    return config


def get_config(environment: str = None) -> LoggingConfig:
    """
    获取当前环境配置
    
    Args:
        environment: 环境标识，如果为None则从环境变量读取
        
    Returns:
        LoggingConfig: 日志配置
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return get_production_config()
    elif environment == "testing":
        return get_testing_config()
    else:
        return get_development_config()


def get_config_path() -> str:
    """获取配置文件路径"""
    env = os.getenv("ENVIRONMENT", "development")
    config_dir = Path(__file__).parent
    
    # 查找配置文件
    config_files = [
        config_dir / f"logging_config_{env}.yaml",
        config_dir / f"logging_config_{env}.json",
        config_dir / "logging_config.yaml",
        config_dir / "logging_config.json",
    ]
    
    for config_file in config_files:
        if config_file.exists():
            return str(config_file)
    
    return str(config_dir / f"logging_config_{env}.yaml")


def load_config(config_file: str = None) -> LoggingConfig:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径，如果为None则自动查找
        
    Returns:
        LoggingConfig: 日志配置
    """
    if config_file is None:
        config_file = get_config_path()
    
    if os.path.exists(config_file):
        try:
            return LoggingConfig.load_from_file(config_file)
        except Exception as e:
            warnings.warn(f"加载配置文件失败 {config_file}: {e}")
    
    # 使用默认配置
    env = os.getenv("ENVIRONMENT", "development")
    return get_config(env)


def save_config(config: LoggingConfig, config_file: str = None, format: str = "yaml"):
    """
    保存配置
    
    Args:
        config: 日志配置
        config_file: 配置文件路径
        format: 文件格式 (json/yaml)
    """
    if config_file is None:
        config_file = get_config_path()
    
    config.save_to_file(config_file, format)


# 当前配置实例
_current_config: Optional[LoggingConfig] = None

def get_current_config() -> LoggingConfig:
    """获取当前配置实例"""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config


# 便捷函数
def setup_logging(config: LoggingConfig = None):
    """
    设置日志系统
    
    Args:
        config: 日志配置，如果为None则使用当前配置
    """
    if config is None:
        config = get_current_config()
    
    # 验证配置
    is_valid, errors = config.validate()
    if not is_valid:
        print(f"日志配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("日志配置验证失败")
    
    # 配置日志
    config.configure_logging()
    
    return config


def get_logger(name: str, config: LoggingConfig = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 记录器名称
        config: 日志配置，如果为None则使用当前配置
        
    Returns:
        logging.Logger: 日志记录器
    """
    if config is None:
        config = get_current_config()
    
    logger = logging.getLogger(name)
    
    # 设置模块特定级别
    module_level = config.module_levels.get_module_level(name)
    if module_level:
        logger.setLevel(getattr(logging, module_level.value))
    
    return logger


@contextmanager
def temporary_log_level(level: LogLevel, logger_name: str = None):
    """
    临时修改日志级别上下文管理器
    
    Args:
        level: 临时日志级别
        logger_name: 记录器名称，如果为None则修改根记录器
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    original_level = logger.level
    
    try:
        logger.setLevel(getattr(logging, level.value))
        yield
    finally:
        logger.setLevel(original_level)


# 性能监控装饰器
def log_performance(operation_name: str = None, level: LogLevel = LogLevel.INFO):
    """
    记录函数性能的装饰器
    
    Args:
        operation_name: 操作名称
        level: 日志级别
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                op_name = operation_name or func.__name__
                
                logger = get_logger(func.__module__)
                logger.log(
                    getattr(logging, level.value),
                    f"{op_name} 耗时: {duration:.3f}s"
                )
        
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    # 获取当前配置
    config = get_current_config()
    
    # 验证配置
    is_valid, errors = config.validate()
    if not is_valid:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过")
    
    # 设置日志
    setup_logging(config)
    
    # 测试日志
    logger = get_logger(__name__)
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    
    # 测试异常处理
    try:
        1 / 0
    except Exception as e:
        logger.exception("发生异常")
    
    # 保存配置
    config_path = get_config_path()
    print(f"\n保存配置到: {config_path}")
    save_config(config, config_path)
    
    # 打印配置摘要
    print(f"\n日志配置摘要:")
    print(f"环境: {config.environment}")
    print(f"根日志级别: {config.root_level}")
    print(f"处理器数量: {len(config.handlers)}")
    for i, handler in enumerate(config.handlers):
        print(f"  处理器{i}: {handler.handler_type.value}, 级别: {handler.level.value}")