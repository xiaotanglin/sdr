"""
时间工具模块 - 提供时间处理、转换、计时和调度相关功能
"""
import time
import datetime
import calendar
import pytz
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio
from functools import wraps
import json
from contextlib import contextmanager
import dateutil.parser
import numpy as np


class TimeUnit(Enum):
    """时间单位枚举"""
    NANOSECONDS = "ns"
    MICROSECONDS = "µs"
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "d"
    WEEKS = "w"


class TimeFormat(Enum):
    """时间格式枚举"""
    ISO_8601 = "iso"               # ISO 8601 格式: 2024-01-15T10:30:00.123456+08:00
    RFC_3339 = "rfc3339"          # RFC 3339 格式: 2024-01-15T10:30:00.123Z
    RFC_2822 = "rfc2822"          # RFC 2822 格式: Mon, 15 Jan 2024 10:30:00 +0800
    TIMESTAMP = "timestamp"       # Unix 时间戳（秒）
    TIMESTAMP_MS = "timestamp_ms" # Unix 时间戳（毫秒）
    DATE_ONLY = "date"            # 仅日期: 2024-01-15
    TIME_ONLY = "time"            # 仅时间: 10:30:00
    DATETIME = "datetime"         # 日期时间: 2024-01-15 10:30:00
    CUSTOM = "custom"             # 自定义格式


@dataclass
class TimeRange:
    """时间范围"""
    start: datetime.datetime
    end: datetime.datetime
    
    def duration(self, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """
        计算时间范围长度
        
        Args:
            unit: 时间单位
            
        Returns:
            float: 时间长度
        """
        delta = self.end - self.start
        return convert_time_delta(delta, unit)
    
    def contains(self, dt: datetime.datetime) -> bool:
        """
        检查时间是否在范围内
        
        Args:
            dt: 要检查的时间
            
        Returns:
            bool: 是否在范围内
        """
        return self.start <= dt <= self.end
    
    def overlaps(self, other: 'TimeRange') -> bool:
        """
        检查时间范围是否重叠
        
        Args:
            other: 另一个时间范围
            
        Returns:
            bool: 是否重叠
        """
        return self.start <= other.end and other.start <= self.end
    
    def intersection(self, other: 'TimeRange') -> Optional['TimeRange']:
        """
        计算交集
        
        Args:
            other: 另一个时间范围
            
        Returns:
            Optional[TimeRange]: 交集时间范围，如果没有交集返回None
        """
        if not self.overlaps(other):
            return None
        
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return TimeRange(start, end)
    
    def union(self, other: 'TimeRange') -> 'TimeRange':
        """
        计算并集
        
        Args:
            other: 另一个时间范围
            
        Returns:
            TimeRange: 并集时间范围
        """
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return TimeRange(start, end)
    
    def split(self, interval: datetime.timedelta) -> List['TimeRange']:
        """
        将时间范围分割为多个子范围
        
        Args:
            interval: 分割间隔
            
        Returns:
            List[TimeRange]: 分割后的时间范围列表
        """
        ranges = []
        current = self.start
        
        while current < self.end:
            next_time = min(current + interval, self.end)
            ranges.append(TimeRange(current, next_time))
            current = next_time
        
        return ranges
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            'start': format_datetime(self.start, TimeFormat.ISO_8601),
            'end': format_datetime(self.end, TimeFormat.ISO_8601)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TimeRange':
        """从字典创建"""
        return cls(
            start=parse_datetime(data['start']),
            end=parse_datetime(data['end'])
        )


class TimeZoneManager:
    """时区管理器"""
    
    def __init__(self, default_timezone: str = 'UTC'):
        """
        初始化时区管理器
        
        Args:
            default_timezone: 默认时区
        """
        self.default_timezone = pytz.timezone(default_timezone)
        self._timezone_cache = {}
    
    def get_timezone(self, timezone_str: str) -> pytz.tzinfo:
        """
        获取时区对象
        
        Args:
            timezone_str: 时区字符串（如 'Asia/Shanghai'）
            
        Returns:
            pytz.tzinfo: 时区对象
        """
        if timezone_str not in self._timezone_cache:
            try:
                self._timezone_cache[timezone_str] = pytz.timezone(timezone_str)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"未知时区: {timezone_str}")
        
        return self._timezone_cache[timezone_str]
    
    def convert_timezone(self, dt: datetime.datetime, 
                        from_tz: str = None, 
                        to_tz: str = None) -> datetime.datetime:
        """
        转换时区
        
        Args:
            dt: 日期时间对象
            from_tz: 原始时区，如果为None则使用dt的时区
            to_tz: 目标时区，如果为None则使用默认时区
            
        Returns:
            datetime.datetime: 转换后的日期时间
        """
        if dt.tzinfo is None and from_tz is None:
            # 如果dt没有时区信息，且没有指定原始时区，则认为是本地时区
            from_tz = 'UTC'
        
        if from_tz:
            from_tz_obj = self.get_timezone(from_tz)
            dt = from_tz_obj.localize(dt) if dt.tzinfo is None else dt.astimezone(from_tz_obj)
        
        if to_tz:
            to_tz_obj = self.get_timezone(to_tz)
            dt = dt.astimezone(to_tz_obj)
        
        return dt
    
    def get_current_time(self, timezone: str = None) -> datetime.datetime:
        """
        获取当前时间
        
        Args:
            timezone: 时区，如果为None则使用默认时区
            
        Returns:
            datetime.datetime: 当前时间
        """
        if timezone:
            tz = self.get_timezone(timezone)
        else:
            tz = self.default_timezone
        
        return datetime.datetime.now(tz)
    
    def list_timezones(self, country_code: str = None) -> List[str]:
        """
        列出时区
        
        Args:
            country_code: 国家代码（如 'CN'），如果为None则返回所有时区
            
        Returns:
            List[str]: 时区列表
        """
        if country_code:
            return pytz.country_timezones.get(country_code.upper(), [])
        return pytz.all_timezones
    
    def get_timezone_offset(self, timezone: str, 
                           dt: datetime.datetime = None) -> str:
        """
        获取时区偏移量
        
        Args:
            timezone: 时区
            dt: 日期时间，如果为None则使用当前时间
            
        Returns:
            str: 时区偏移量（如 '+08:00'）
        """
        tz = self.get_timezone(timezone)
        dt = dt or datetime.datetime.now()
        
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        
        offset = dt.utcoffset()
        if offset is None:
            return 'Z'
        
        total_seconds = int(offset.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        return f"{hours:+03d}:{minutes:02d}"


# 全局时区管理器实例
_timezone_manager = TimeZoneManager()


# ==================== 时间获取函数 ====================
def get_timestamp(unit: TimeUnit = TimeUnit.SECONDS) -> float:
    """
    获取当前时间戳
    
    Args:
        unit: 时间单位
        
    Returns:
        float: 时间戳
    """
    current_time = time.time()
    
    if unit == TimeUnit.NANOSECONDS:
        return current_time * 1e9
    elif unit == TimeUnit.MICROSECONDS:
        return current_time * 1e6
    elif unit == TimeUnit.MILLISECONDS:
        return current_time * 1e3
    elif unit == TimeUnit.SECONDS:
        return current_time
    elif unit == TimeUnit.MINUTES:
        return current_time / 60
    elif unit == TimeUnit.HOURS:
        return current_time / 3600
    elif unit == TimeUnit.DAYS:
        return current_time / 86400
    elif unit == TimeUnit.WEEKS:
        return current_time / 604800
    else:
        raise ValueError(f"不支持的时间单位: {unit}")


def get_current_datetime(timezone: str = None) -> datetime.datetime:
    """
    获取当前日期时间
    
    Args:
        timezone: 时区
        
    Returns:
        datetime.datetime: 当前日期时间
    """
    return _timezone_manager.get_current_time(timezone)


def get_current_date(timezone: str = None) -> datetime.date:
    """
    获取当前日期
    
    Args:
        timezone: 时区
        
    Returns:
        datetime.date: 当前日期
    """
    return get_current_datetime(timezone).date()


def get_current_time(timezone: str = None) -> datetime.time:
    """
    获取当前时间
    
    Args:
        timezone: 时区
        
    Returns:
        datetime.time: 当前时间
    """
    return get_current_datetime(timezone).time()


def get_microsecond_timestamp() -> int:
    """
    获取微秒级时间戳
    
    Returns:
        int: 微秒级时间戳
    """
    return int(time.time() * 1e6)


def get_nanosecond_timestamp() -> int:
    """
    获取纳秒级时间戳
    
    Returns:
        int: 纳秒级时间戳
    """
    return int(time.time() * 1e9)


def get_iso_timestamp() -> str:
    """
    获取ISO 8601格式时间戳
    
    Returns:
        str: ISO 8601格式时间戳
    """
    return format_datetime(datetime.datetime.utcnow(), TimeFormat.ISO_8601)


# ==================== 时间格式化函数 ====================
def format_datetime(dt: datetime.datetime, 
                   fmt: Union[TimeFormat, str] = TimeFormat.ISO_8601,
                   timezone: str = None) -> str:
    """
    格式化日期时间
    
    Args:
        dt: 日期时间对象
        fmt: 格式类型或自定义格式字符串
        timezone: 目标时区
        
    Returns:
        str: 格式化后的时间字符串
    """
    # 转换时区
    if timezone:
        dt = _timezone_manager.convert_timezone(dt, to_tz=timezone)
    
    # 处理时区
    if dt.tzinfo is None:
        dt = _timezone_manager.default_timezone.localize(dt)
    
    if isinstance(fmt, TimeFormat):
        if fmt == TimeFormat.ISO_8601:
            return dt.isoformat()
        elif fmt == TimeFormat.RFC_3339:
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        elif fmt == TimeFormat.RFC_2822:
            return dt.strftime('%a, %d %b %Y %H:%M:%S %z')
        elif fmt == TimeFormat.TIMESTAMP:
            return str(int(dt.timestamp()))
        elif fmt == TimeFormat.TIMESTAMP_MS:
            return str(int(dt.timestamp() * 1000))
        elif fmt == TimeFormat.DATE_ONLY:
            return dt.strftime('%Y-%m-%d')
        elif fmt == TimeFormat.TIME_ONLY:
            return dt.strftime('%H:%M:%S')
        elif fmt == TimeFormat.DATETIME:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            raise ValueError(f"不支持的格式: {fmt}")
    else:
        # 自定义格式字符串
        return dt.strftime(fmt)


def format_timestamp(timestamp: float, 
                    fmt: Union[TimeFormat, str] = TimeFormat.ISO_8601,
                    unit: TimeUnit = TimeUnit.SECONDS,
                    timezone: str = None) -> str:
    """
    格式化时间戳
    
    Args:
        timestamp: 时间戳
        fmt: 格式类型
        unit: 时间戳单位
        timezone: 目标时区
        
    Returns:
        str: 格式化后的时间字符串
    """
    # 转换为秒
    seconds = convert_time(timestamp, unit, TimeUnit.SECONDS)
    
    # 创建datetime对象
    dt = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    
    # 格式化
    return format_datetime(dt, fmt, timezone)


def format_timedelta(td: datetime.timedelta, 
                    fmt: str = "{days}d {hours}h {minutes}m {seconds}s",
                    precision: int = 3) -> str:
    """
    格式化时间间隔
    
    Args:
        td: 时间间隔
        fmt: 格式字符串
        precision: 秒的小数位数精度
        
    Returns:
        str: 格式化后的时间间隔
    """
    total_seconds = td.total_seconds()
    
    # 计算各个部分
    days = td.days
    hours = int(total_seconds // 3600) % 24
    minutes = int(total_seconds // 60) % 60
    seconds = total_seconds % 60
    
    # 格式化秒
    if precision > 0:
        seconds_str = f"{seconds:.{precision}f}"
    else:
        seconds_str = str(int(seconds))
    
    # 替换占位符
    result = fmt.format(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds_str,
        total_seconds=total_seconds,
        total_minutes=total_seconds / 60,
        total_hours=total_seconds / 3600,
        total_days=total_seconds / 86400
    )
    
    return result


# ==================== 时间解析函数 ====================
def parse_datetime(dt_str: str, 
                  timezone: str = None) -> datetime.datetime:
    """
    解析时间字符串
    
    Args:
        dt_str: 时间字符串
        timezone: 时区
        
    Returns:
        datetime.datetime: 解析后的日期时间
    """
    try:
        # 尝试使用dateutil解析
        dt = dateutil.parser.parse(dt_str)
    except Exception as e:
        # 如果dateutil解析失败，尝试常见格式
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y%m%d%H%M%S',
            '%Y-%m-%d',
            '%H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(dt_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"无法解析时间字符串: {dt_str}")
    
    # 处理时区
    if timezone:
        tz = _timezone_manager.get_timezone(timezone)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        else:
            dt = dt.astimezone(tz)
    
    return dt


def parse_timestamp(timestamp: Union[int, float, str], 
                   unit: TimeUnit = TimeUnit.SECONDS) -> datetime.datetime:
    """
    解析时间戳
    
    Args:
        timestamp: 时间戳
        unit: 时间戳单位
        
    Returns:
        datetime.datetime: 解析后的日期时间
    """
    # 转换为浮点数
    if isinstance(timestamp, str):
        try:
            timestamp = float(timestamp)
        except ValueError:
            raise ValueError(f"无效的时间戳: {timestamp}")
    
    # 转换为秒
    seconds = convert_time(timestamp, unit, TimeUnit.SECONDS)
    
    return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)


# ==================== 时间转换函数 ====================
def convert_time(value: float, 
                from_unit: TimeUnit, 
                to_unit: TimeUnit) -> float:
    """
    转换时间单位
    
    Args:
        value: 时间值
        from_unit: 原始单位
        to_unit: 目标单位
        
    Returns:
        float: 转换后的值
    """
    # 先转换为秒
    if from_unit == TimeUnit.NANOSECONDS:
        seconds = value / 1e9
    elif from_unit == TimeUnit.MICROSECONDS:
        seconds = value / 1e6
    elif from_unit == TimeUnit.MILLISECONDS:
        seconds = value / 1e3
    elif from_unit == TimeUnit.SECONDS:
        seconds = value
    elif from_unit == TimeUnit.MINUTES:
        seconds = value * 60
    elif from_unit == TimeUnit.HOURS:
        seconds = value * 3600
    elif from_unit == TimeUnit.DAYS:
        seconds = value * 86400
    elif from_unit == TimeUnit.WEEKS:
        seconds = value * 604800
    else:
        raise ValueError(f"不支持的单位: {from_unit}")
    
    # 从秒转换为目标单位
    if to_unit == TimeUnit.NANOSECONDS:
        return seconds * 1e9
    elif to_unit == TimeUnit.MICROSECONDS:
        return seconds * 1e6
    elif to_unit == TimeUnit.MILLISECONDS:
        return seconds * 1e3
    elif to_unit == TimeUnit.SECONDS:
        return seconds
    elif to_unit == TimeUnit.MINUTES:
        return seconds / 60
    elif to_unit == TimeUnit.HOURS:
        return seconds / 3600
    elif to_unit == TimeUnit.DAYS:
        return seconds / 86400
    elif to_unit == TimeUnit.WEEKS:
        return seconds / 604800
    else:
        raise ValueError(f"不支持的单位: {to_unit}")


def convert_time_delta(td: datetime.timedelta, 
                      unit: TimeUnit = TimeUnit.SECONDS) -> float:
    """
    转换时间间隔
    
    Args:
        td: 时间间隔
        unit: 目标单位
        
    Returns:
        float: 转换后的值
    """
    total_seconds = td.total_seconds()
    
    if unit == TimeUnit.NANOSECONDS:
        return total_seconds * 1e9
    elif unit == TimeUnit.MICROSECONDS:
        return total_seconds * 1e6
    elif unit == TimeUnit.MILLISECONDS:
        return total_seconds * 1e3
    elif unit == TimeUnit.SECONDS:
        return total_seconds
    elif unit == TimeUnit.MINUTES:
        return total_seconds / 60
    elif unit == TimeUnit.HOURS:
        return total_seconds / 3600
    elif unit == TimeUnit.DAYS:
        return total_seconds / 86400
    elif unit == TimeUnit.WEEKS:
        return total_seconds / 604800
    else:
        raise ValueError(f"不支持的单位: {unit}")


def datetime_to_timestamp(dt: datetime.datetime, 
                         unit: TimeUnit = TimeUnit.SECONDS) -> float:
    """
    将datetime转换为时间戳
    
    Args:
        dt: 日期时间
        unit: 时间戳单位
        
    Returns:
        float: 时间戳
    """
    if dt.tzinfo is None:
        # 假设是本地时间
        timestamp = dt.timestamp()
    else:
        # 转换为UTC时间戳
        timestamp = dt.timestamp()
    
    return convert_time(timestamp, TimeUnit.SECONDS, unit)


# ==================== 时间计算函数 ====================
def add_time(dt: datetime.datetime, 
            value: float, 
            unit: TimeUnit = TimeUnit.SECONDS) -> datetime.datetime:
    """
    添加时间
    
    Args:
        dt: 原始时间
        value: 要添加的值
        unit: 时间单位
        
    Returns:
        datetime.datetime: 添加后的时间
    """
    seconds = convert_time(value, unit, TimeUnit.SECONDS)
    return dt + datetime.timedelta(seconds=seconds)


def subtract_time(dt: datetime.datetime, 
                 value: float, 
                 unit: TimeUnit = TimeUnit.SECONDS) -> datetime.datetime:
    """
    减去时间
    
    Args:
        dt: 原始时间
        value: 要减去的值
        unit: 时间单位
        
    Returns:
        datetime.datetime: 减去后的时间
    """
    return add_time(dt, -value, unit)


def time_difference(dt1: datetime.datetime, 
                   dt2: datetime.datetime, 
                   unit: TimeUnit = TimeUnit.SECONDS) -> float:
    """
    计算时间差
    
    Args:
        dt1: 时间1
        dt2: 时间2
        unit: 时间单位
        
    Returns:
        float: 时间差
    """
    td = dt2 - dt1
    return convert_time_delta(td, unit)


def is_within_range(dt: datetime.datetime, 
                   start: datetime.datetime, 
                   end: datetime.datetime) -> bool:
    """
    检查时间是否在范围内
    
    Args:
        dt: 要检查的时间
        start: 范围开始时间
        end: 范围结束时间
        
    Returns:
        bool: 是否在范围内
    """
    return start <= dt <= end


def get_time_ranges(start: datetime.datetime, 
                   end: datetime.datetime, 
                   interval: datetime.timedelta) -> List[TimeRange]:
    """
    获取时间范围列表
    
    Args:
        start: 开始时间
        end: 结束时间
        interval: 间隔
        
    Returns:
        List[TimeRange]: 时间范围列表
    """
    tr = TimeRange(start, end)
    return tr.split(interval)


def round_time(dt: datetime.datetime, 
              round_to: str = '5min') -> datetime.datetime:
    """
    四舍五入时间
    
    Args:
        dt: 原始时间
        round_to: 四舍五入到（如 '5min', '1hour', '15min'）
        
    Returns:
        datetime.datetime: 四舍五入后的时间
    """
    if round_to == '5min':
        minutes = (dt.minute // 5) * 5
        return dt.replace(minute=minutes, second=0, microsecond=0)
    elif round_to == '10min':
        minutes = (dt.minute // 10) * 10
        return dt.replace(minute=minutes, second=0, microsecond=0)
    elif round_to == '15min':
        minutes = (dt.minute // 15) * 15
        return dt.replace(minute=minutes, second=0, microsecond=0)
    elif round_to == '30min':
        minutes = (dt.minute // 30) * 30
        return dt.replace(minute=minutes, second=0, microsecond=0)
    elif round_to == '1hour':
        return dt.replace(minute=0, second=0, microsecond=0)
    elif round_to == 'day':
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif round_to == 'week':
        # 四舍五入到周的开始（周一）
        days_since_monday = dt.weekday()
        return dt.replace(hour=0, minute=0, second=0, microsecond=0) - \
               datetime.timedelta(days=days_since_monday)
    elif round_to == 'month':
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"不支持的舍入单位: {round_to}")


# ==================== 性能计时器 ====================
class Timer:
    """高性能计时器"""
    
    def __init__(self, name: str = None, auto_start: bool = True):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            auto_start: 是否自动开始计时
        """
        self.name = name
        self._start_time = None
        self._elapsed_time = 0.0
        self._is_running = False
        
        if auto_start:
            self.start()
    
    def start(self):
        """开始计时"""
        if self._is_running:
            raise RuntimeError("计时器已经在运行")
        
        self._start_time = time.perf_counter()
        self._is_running = True
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            float: 经过的时间（秒）
        """
        if not self._is_running:
            raise RuntimeError("计时器没有在运行")
        
        end_time = time.perf_counter()
        self._elapsed_time = end_time - self._start_time
        self._is_running = False
        
        return self._elapsed_time
    
    def pause(self):
        """暂停计时"""
        if not self._is_running:
            raise RuntimeError("计时器没有在运行")
        
        end_time = time.perf_counter()
        self._elapsed_time += end_time - self._start_time
        self._is_running = False
    
    def resume(self):
        """恢复计时"""
        if self._is_running:
            raise RuntimeError("计时器已经在运行")
        
        self._start_time = time.perf_counter()
        self._is_running = True
    
    def elapsed(self, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """
        获取经过的时间
        
        Args:
            unit: 时间单位
            
        Returns:
            float: 经过的时间
        """
        if self._is_running:
            current_time = time.perf_counter()
            total_elapsed = self._elapsed_time + (current_time - self._start_time)
        else:
            total_elapsed = self._elapsed_time
        
        return convert_time(total_elapsed, TimeUnit.SECONDS, unit)
    
    def reset(self):
        """重置计时器"""
        self._start_time = None
        self._elapsed_time = 0.0
        self._is_running = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __str__(self) -> str:
        """字符串表示"""
        elapsed = self.elapsed()
        name_str = f" '{self.name}'" if self.name else ""
        return f"Timer{name_str}: {elapsed:.6f}s"


class MultiTimer:
    """多计时器管理器"""
    
    def __init__(self):
        self.timers = {}
    
    def start(self, name: str) -> Timer:
        """
        开始计时器
        
        Args:
            name: 计时器名称
            
        Returns:
            Timer: 计时器实例
        """
        if name in self.timers:
            self.timers[name].reset()
            self.timers[name].start()
        else:
            self.timers[name] = Timer(name, auto_start=True)
        
        return self.timers[name]
    
    def stop(self, name: str) -> float:
        """
        停止计时器
        
        Args:
            name: 计时器名称
            
        Returns:
            float: 经过的时间（秒）
        """
        if name not in self.timers:
            raise KeyError(f"计时器不存在: {name}")
        
        return self.timers[name].stop()
    
    def get_elapsed(self, name: str, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """
        获取计时器经过的时间
        
        Args:
            name: 计时器名称
            unit: 时间单位
            
        Returns:
            float: 经过的时间
        """
        if name not in self.timers:
            raise KeyError(f"计时器不存在: {name}")
        
        return self.timers[name].elapsed(unit)
    
    def get_all_stats(self) -> Dict[str, float]:
        """
        获取所有计时器统计
        
        Returns:
            Dict[str, float]: 计时器统计字典
        """
        stats = {}
        for name, timer in self.timers.items():
            stats[name] = timer.elapsed()
        
        return stats
    
    def reset(self, name: str = None):
        """
        重置计时器
        
        Args:
            name: 计时器名称，如果为None则重置所有计时器
        """
        if name is None:
            for timer in self.timers.values():
                timer.reset()
        elif name in self.timers:
            self.timers[name].reset()
        else:
            raise KeyError(f"计时器不存在: {name}")


# ==================== 装饰器函数 ====================
def timeit(func=None, *, unit: TimeUnit = TimeUnit.SECONDS, 
          print_result: bool = True, logger=None):
    """
    函数计时装饰器
    
    Args:
        func: 被装饰的函数
        unit: 时间单位
        print_result: 是否打印结果
        logger: 日志记录器
        
    Returns:
        装饰后的函数
    """
    if func is None:
        return lambda f: timeit(f, unit=unit, print_result=print_result, logger=logger)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(f"Function: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = timer.elapsed(unit)
            
            if print_result:
                message = f"{func.__name__} 执行时间: {elapsed:.6f} {unit.value}"
                if logger:
                    logger.debug(message)
                else:
                    print(message)
            
            return result
        finally:
            timer.stop()
    
    return wrapper


async def async_timeit(func=None, *, unit: TimeUnit = TimeUnit.SECONDS, 
                      print_result: bool = True, logger=None):
    """
    异步函数计时装饰器
    
    Args:
        func: 被装饰的函数
        unit: 时间单位
        print_result: 是否打印结果
        logger: 日志记录器
        
    Returns:
        装饰后的函数
    """
    if func is None:
        return lambda f: async_timeit(f, unit=unit, print_result=print_result, logger=logger)
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        timer = Timer(f"Async Function: {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            elapsed = timer.elapsed(unit)
            
            if print_result:
                message = f"{func.__name__} 执行时间: {elapsed:.6f} {unit.value}"
                if logger:
                    logger.debug(message)
                else:
                    print(message)
            
            return result
        finally:
            timer.stop()
    
    return wrapper


@contextmanager
def timing_context(name: str = None, unit: TimeUnit = TimeUnit.SECONDS, 
                  print_result: bool = True, logger=None):
    """
    计时上下文管理器
    
    Args:
        name: 上下文名称
        unit: 时间单位
        print_result: 是否打印结果
        logger: 日志记录器
        
    Yields:
        Timer: 计时器实例
    """
    timer_name = name or "Context"
    timer = Timer(timer_name)
    
    try:
        yield timer
    finally:
        elapsed = timer.elapsed(unit)
        
        if print_result:
            message = f"{timer_name} 执行时间: {elapsed:.6f} {unit.value}"
            if logger:
                logger.debug(message)
            else:
                print(message)


# ==================== 调度相关函数 ====================
def calculate_next_run(base_time: datetime.datetime, 
                      interval: datetime.timedelta) -> datetime.datetime:
    """
    计算下一次运行时间
    
    Args:
        base_time: 基准时间
        interval: 运行间隔
        
    Returns:
        datetime.datetime: 下一次运行时间
    """
    now = datetime.datetime.now(tz=base_time.tzinfo)
    
    if now <= base_time:
        return base_time
    
    # 计算距离基准时间的间隔数
    delta = now - base_time
    intervals_passed = delta.total_seconds() / interval.total_seconds()
    intervals_to_add = int(intervals_passed) + 1
    
    return base_time + interval * intervals_to_add


def is_time_to_run(last_run: datetime.datetime, 
                  interval: datetime.timedelta) -> bool:
    """
    检查是否应该运行
    
    Args:
        last_run: 上次运行时间
        interval: 运行间隔
        
    Returns:
        bool: 是否应该运行
    """
    now = datetime.datetime.now(tz=last_run.tzinfo)
    next_run = last_run + interval
    
    return now >= next_run


def schedule_daily(time_str: str, timezone: str = 'UTC') -> datetime.datetime:
    """
    安排每日运行时间
    
    Args:
        time_str: 时间字符串（格式: HH:MM:SS）
        timezone: 时区
        
    Returns:
        datetime.datetime: 下一次运行时间
    """
    today = get_current_date(timezone)
    run_time = datetime.datetime.strptime(time_str, '%H:%M:%S').time()
    run_datetime = datetime.datetime.combine(today, run_time)
    
    tz = _timezone_manager.get_timezone(timezone)
    run_datetime = tz.localize(run_datetime)
    
    now = get_current_datetime(timezone)
    
    if run_datetime > now:
        return run_datetime
    else:
        # 明天运行
        tomorrow = today + datetime.timedelta(days=1)
        run_datetime = datetime.datetime.combine(tomorrow, run_time)
        return tz.localize(run_datetime)


# ==================== 日期时间工具函数 ====================
def get_start_of_day(dt: datetime.datetime = None) -> datetime.datetime:
    """
    获取一天的开始时间（00:00:00）
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        datetime.datetime: 一天的开始时间
    """
    if dt is None:
        dt = get_current_datetime()
    
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime.datetime = None) -> datetime.datetime:
    """
    获取一天的结束时间（23:59:59.999999）
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        datetime.datetime: 一天的结束时间
    """
    if dt is None:
        dt = get_current_datetime()
    
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_week(dt: datetime.datetime = None, 
                     week_start: int = 0) -> datetime.datetime:
    """
    获取一周的开始时间
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        week_start: 周开始日（0=周一，6=周日）
        
    Returns:
        datetime.datetime: 一周的开始时间
    """
    if dt is None:
        dt = get_current_datetime()
    
    # 计算距离周开始日的天数
    days_since_start = (dt.weekday() - week_start) % 7
    
    start_date = dt - datetime.timedelta(days=days_since_start)
    return get_start_of_day(start_date)


def get_start_of_month(dt: datetime.datetime = None) -> datetime.datetime:
    """
    获取一月的开始时间
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        datetime.datetime: 一月的开始时间
    """
    if dt is None:
        dt = get_current_datetime()
    
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_start_of_year(dt: datetime.datetime = None) -> datetime.datetime:
    """
    获取一年的开始时间
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        datetime.datetime: 一年的开始时间
    """
    if dt is None:
        dt = get_current_datetime()
    
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def is_weekend(dt: datetime.datetime = None) -> bool:
    """
    检查是否是周末
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        bool: 是否是周末
    """
    if dt is None:
        dt = get_current_datetime()
    
    return dt.weekday() >= 5  # 5=周六, 6=周日


def is_workday(dt: datetime.datetime = None) -> bool:
    """
    检查是否是工作日
    
    Args:
        dt: 日期时间，如果为None则使用当前时间
        
    Returns:
        bool: 是否是工作日
    """
    return not is_weekend(dt)


def is_leap_year(year: int) -> bool:
    """
    检查是否是闰年
    
    Args:
        year: 年份
        
    Returns:
        bool: 是否是闰年
    """
    return calendar.isleap(year)


def get_days_in_month(year: int, month: int) -> int:
    """
    获取月份的天数
    
    Args:
        year: 年份
        month: 月份
        
    Returns:
        int: 天数
    """
    return calendar.monthrange(year, month)[1]


# ==================== 时间序列相关函数 ====================
def generate_time_series(start: datetime.datetime, 
                        end: datetime.datetime, 
                        interval: datetime.timedelta) -> List[datetime.datetime]:
    """
    生成时间序列
    
    Args:
        start: 开始时间
        end: 结束时间
        interval: 时间间隔
        
    Returns:
        List[datetime.datetime]: 时间序列
    """
    time_series = []
    current = start
    
    while current <= end:
        time_series.append(current)
        current += interval
    
    return time_series


def resample_time_series(timestamps: List[datetime.datetime], 
                        values: List[Any],
                        interval: datetime.timedelta,
                        agg_func: Callable = np.mean) -> Tuple[List[datetime.datetime], List[Any]]:
    """
    重采样时间序列
    
    Args:
        timestamps: 时间戳列表
        values: 值列表
        interval: 目标间隔
        agg_func: 聚合函数
        
    Returns:
        Tuple[List[datetime.datetime], List[Any]]: 重采样后的时间序列和值
    """
    if len(timestamps) != len(values):
        raise ValueError("时间戳和值的长度必须相同")
    
    if not timestamps:
        return [], []
    
    # 排序
    sorted_data = sorted(zip(timestamps, values), key=lambda x: x[0])
    timestamps = [x[0] for x in sorted_data]
    values = [x[1] for x in sorted_data]
    
    # 重采样
    resampled_timestamps = []
    resampled_values = []
    
    current_bin_start = timestamps[0]
    current_bin_end = current_bin_start + interval
    current_bin_values = []
    
    for ts, val in zip(timestamps, values):
        if ts < current_bin_end:
            current_bin_values.append(val)
        else:
            if current_bin_values:
                resampled_timestamps.append(current_bin_start)
                resampled_values.append(agg_func(current_bin_values))
            
            # 移动时间窗口
            while ts >= current_bin_end:
                current_bin_start = current_bin_end
                current_bin_end += interval
            
            current_bin_values = [val]
    
    # 处理最后一个时间窗口
    if current_bin_values:
        resampled_timestamps.append(current_bin_start)
        resampled_values.append(agg_func(current_bin_values))
    
    return resampled_timestamps, resampled_values


# ==================== 默认配置 ====================
# 设置默认时区
_timezone_manager = TimeZoneManager('UTC')

# 多计时器实例
_multi_timer = MultiTimer()