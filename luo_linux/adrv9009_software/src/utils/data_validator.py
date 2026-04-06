"""
数据验证器模块
提供信号数据、检测结果、配置等的数据验证功能
"""
import re
import json
import yaml
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, get_type_hints
from enum import Enum
import inspect
import hashlib
import struct
import math
from pathlib import Path
import warnings
import jsonschema
from jsonschema import validate, ValidationError
from pydantic import BaseModel, validator, root_validator, Field, create_model
from pydantic.error_wrappers import ValidationError as PydanticValidationError
import pandas as pd
import scipy
from scipy import signal as scipy_signal


class ValidationLevel(str, Enum):
    """验证级别枚举"""
    STRICT = "strict"      # 严格验证，失败时抛出异常
    WARNING = "warning"    # 警告验证，失败时记录警告
    LENIENT = "lenient"    # 宽松验证，失败时返回默认值
    SKIP = "skip"         # 跳过验证


class ValidationResult:
    """验证结果类"""
    
    def __init__(self, is_valid: bool = True, message: str = "", 
                 errors: List[str] = None, warnings: List[str] = None,
                 data: Any = None):
        """
        初始化验证结果
        
        Args:
            is_valid: 是否验证通过
            message: 验证消息
            errors: 错误列表
            warnings: 警告列表
            data: 验证后的数据
        """
        self.is_valid = is_valid
        self.message = message
        self.errors = errors or []
        self.warnings = warnings or []
        self.data = data
        self.timestamp = datetime.now()
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
        self.message = f"验证失败: {error}"
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """合并验证结果"""
        result = ValidationResult()
        result.is_valid = self.is_valid and other.is_valid
        result.errors = self.errors + other.errors
        result.warnings = self.warnings + other.warnings
        result.message = f"合并验证: {len(self.errors)+len(other.errors)} 错误, {len(self.warnings)+len(other.warnings)} 警告"
        result.data = other.data if other.data is not None else self.data
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'message': self.message,
            'errors': self.errors,
            'warnings': self.warnings,
            'data': self.data if isinstance(self.data, (dict, list, str, int, float, bool)) else str(self.data),
            'timestamp': self.timestamp.isoformat(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def __bool__(self) -> bool:
        """布尔值转换"""
        return self.is_valid
    
    def __str__(self) -> str:
        """字符串表示"""
        status = "通过" if self.is_valid else "失败"
        return f"验证{status}: {self.message}"


class BaseValidator:
    """基础验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        """
        初始化验证器
        
        Args:
            level: 验证级别
        """
        self.level = level
        self.schemas = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """加载默认JSON Schema"""
        # 基础信号数据Schema
        self.schemas['signal_data'] = {
            "type": "object",
            "required": ["signal_id", "timestamp", "center_frequency", "sample_rate"],
            "properties": {
                "signal_id": {"type": "string", "minLength": 1},
                "timestamp": {"type": "string", "format": "date-time"},
                "center_frequency": {"type": "number", "minimum": 0},
                "sample_rate": {"type": "number", "minimum": 0},
                "bandwidth": {"type": "number", "minimum": 0},
                "duration_ms": {"type": "number", "minimum": 0},
                "metadata": {"type": "object"},
                "device_id": {"type": "string"},
                "channel_index": {"type": "integer", "minimum": 0}
            }
        }
        
        # 检测结果Schema
        self.schemas['detection_result'] = {
            "type": "object",
            "required": ["detection_id", "signal_id", "detection_type", "detection_time", "confidence"],
            "properties": {
                "detection_id": {"type": "string", "minLength": 1},
                "signal_id": {"type": "string", "minLength": 1},
                "detection_type": {"type": "string", "minLength": 1},
                "detection_time": {"type": "string", "format": "date-time"},
                "center_frequency": {"type": "number", "minimum": 0},
                "bandwidth": {"type": "number", "minimum": 0},
                "power_db": {"type": "number"},
                "snr_db": {"type": "number"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "is_known": {"type": "boolean"},
                "classification": {"type": "string"},
                "metadata": {"type": "object"}
            }
        }
        
        # 配置Schema
        self.schemas['config'] = {
            "type": "object",
            "required": ["environment"],
            "properties": {
                "environment": {"type": "string", "enum": ["development", "testing", "production"]},
                "debug": {"type": "boolean"},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}
            }
        }
    
    def validate_with_schema(self, data: Dict[str, Any], schema_name: str, 
                           level: ValidationLevel = None) -> ValidationResult:
        """
        使用JSON Schema验证数据
        
        Args:
            data: 要验证的数据
            schema_name: Schema名称
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                message=f"未知的Schema: {schema_name}",
                errors=[f"Schema '{schema_name}' 不存在"]
            )
        
        try:
            validate(instance=data, schema=self.schemas[schema_name])
            return ValidationResult(
                is_valid=True,
                message=f"数据符合Schema '{schema_name}'",
                data=data
            )
        except ValidationError as e:
            if level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    message=f"Schema验证失败: {e.message}",
                    errors=[str(e)],
                    data=data
                )
            elif level == ValidationLevel.WARNING:
                return ValidationResult(
                    is_valid=True,
                    message=f"Schema验证警告: {e.message}",
                    warnings=[str(e)],
                    data=data
                )
            else:  # LENIENT or SKIP
                return ValidationResult(
                    is_valid=True,
                    message=f"跳过Schema验证: {e.message}",
                    data=data
                )
    
    def register_schema(self, schema_name: str, schema: Dict[str, Any]):
        """
        注册新的JSON Schema
        
        Args:
            schema_name: Schema名称
            schema: JSON Schema定义
        """
        self.schemas[schema_name] = schema
    
    def validate_type(self, value: Any, expected_type: Type, 
                     level: ValidationLevel = None) -> ValidationResult:
        """
        验证类型
        
        Args:
            value: 要验证的值
            expected_type: 期望的类型
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        
        if expected_type is Any:
            return ValidationResult(is_valid=True, message="类型Any总是有效", data=value)
        
        # 处理可选类型
        origin = getattr(expected_type, '__origin__', None)
        args = getattr(expected_type, '__args__', [])
        
        if origin is Union:
            # 联合类型
            for arg in args:
                result = self.validate_type(value, arg, ValidationLevel.LENIENT)
                if result.is_valid:
                    return ValidationResult(
                        is_valid=True,
                        message=f"值符合联合类型之一: {expected_type}",
                        data=value
                    )
            
            error_msg = f"值 {type(value).__name__} 不符合任何联合类型: {expected_type}"
            if level == ValidationLevel.STRICT:
                return ValidationResult(is_valid=False, message=error_msg, errors=[error_msg])
            elif level == ValidationLevel.WARNING:
                return ValidationResult(is_valid=True, message=error_msg, warnings=[error_msg], data=value)
            else:
                return ValidationResult(is_valid=True, message=f"跳过类型验证: {error_msg}", data=value)
        
        elif origin is list and len(args) == 1:
            # 列表类型
            if not isinstance(value, list):
                error_msg = f"期望列表，得到 {type(value).__name__}"
                if level == ValidationLevel.STRICT:
                    return ValidationResult(is_valid=False, message=error_msg, errors=[error_msg])
                elif level == ValidationLevel.WARNING:
                    return ValidationResult(is_valid=True, message=error_msg, warnings=[error_msg], data=value)
                else:
                    return ValidationResult(is_valid=True, message=f"跳过类型验证: {error_msg}", data=value)
            
            # 验证列表元素
            errors = []
            validated_items = []
            for i, item in enumerate(value):
                result = self.validate_type(item, args[0], level)
                if not result.is_valid:
                    errors.append(f"元素[{i}]: {result.message}")
                validated_items.append(result.data)
            
            if errors and level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    message=f"列表元素验证失败: {len(errors)} 错误",
                    errors=errors
                )
            
            return ValidationResult(
                is_valid=not errors or level != ValidationLevel.STRICT,
                message=f"列表验证: {len(errors)} 错误",
                errors=errors if errors and level == ValidationLevel.STRICT else [],
                warnings=errors if errors and level == ValidationLevel.WARNING else [],
                data=validated_items
            )
        
        elif origin is dict and len(args) == 2 and args[0] is str:
            # 字典类型 Dict[str, T]
            if not isinstance(value, dict):
                error_msg = f"期望字典，得到 {type(value).__name__}"
                if level == ValidationLevel.STRICT:
                    return ValidationResult(is_valid=False, message=error_msg, errors=[error_msg])
                elif level == ValidationLevel.WARNING:
                    return ValidationResult(is_valid=True, message=error_msg, warnings=[error_msg], data=value)
                else:
                    return ValidationResult(is_valid=True, message=f"跳过类型验证: {error_msg}", data=value)
            
            # 验证字典值
            errors = []
            validated_dict = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    errors.append(f"键类型错误: 期望str, 得到{type(key).__name__}")
                    continue
                
                result = self.validate_type(item, args[1], level)
                if not result.is_valid:
                    errors.append(f"键 '{key}': {result.message}")
                validated_dict[key] = result.data
            
            if errors and level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    message=f"字典值验证失败: {len(errors)} 错误",
                    errors=errors
                )
            
            return ValidationResult(
                is_valid=not errors or level != ValidationLevel.STRICT,
                message=f"字典验证: {len(errors)} 错误",
                errors=errors if errors and level == ValidationLevel.STRICT else [],
                warnings=errors if errors and level == ValidationLevel.WARNING else [],
                data=validated_dict
            )
        
        else:
            # 基本类型检查
            if not isinstance(value, expected_type):
                error_msg = f"期望 {expected_type.__name__}，得到 {type(value).__name__}"
                if level == ValidationLevel.STRICT:
                    return ValidationResult(is_valid=False, message=error_msg, errors=[error_msg])
                elif level == ValidationLevel.WARNING:
                    return ValidationResult(is_valid=True, message=error_msg, warnings=[error_msg], data=value)
                else:
                    return ValidationResult(is_valid=True, message=f"跳过类型验证: {error_msg}", data=value)
            
            return ValidationResult(is_valid=True, message=f"类型 {expected_type.__name__} 验证通过", data=value)


class SignalDataValidator(BaseValidator):
    """信号数据验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        """初始化信号数据验证器"""
        super().__init__(level)
        self._setup_signal_schemas()
    
    def _setup_signal_schemas(self):
        """设置信号数据Schema"""
        # IQ数据Schema
        self.schemas['iq_data'] = {
            "type": "object",
            "required": ["data", "sample_rate"],
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "sample_rate": {"type": "number", "minimum": 1},
                "center_frequency": {"type": "number", "minimum": 0},
                "timestamp": {"type": "string", "format": "date-time"}
            }
        }
    
    def validate_signal_data(self, signal_data: Any, 
                           level: ValidationLevel = None) -> ValidationResult:
        """
        验证信号数据
        
        Args:
            signal_data: 信号数据
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        
        # 如果没有数据
        if signal_data is None:
            if level == ValidationLevel.STRICT:
                return ValidationResult(
                    is_valid=False,
                    message="信号数据为空",
                    errors=["信号数据不能为空"]
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message="信号数据为空，跳过验证",
                    warnings=["信号数据为空"]
                )
        
        result = ValidationResult(is_valid=True, message="开始信号数据验证")
        
        # 检查是否为字典格式
        if isinstance(signal_data, dict):
            dict_result = self._validate_signal_dict(signal_data, level)
            result = result.merge(dict_result)
        
        # 检查是否为NumPy数组
        elif isinstance(signal_data, np.ndarray):
            array_result = self._validate_iq_array(signal_data, level)
            result = result.merge(array_result)
        
        # 检查是否为列表格式
        elif isinstance(signal_data, (list, tuple)):
            list_result = self._validate_iq_list(signal_data, level)
            result = result.merge(list_result)
        
        else:
            error_msg = f"不支持的信号数据类型: {type(signal_data).__name__}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        if result.is_valid:
            result.message = "信号数据验证通过"
        
        return result
    
    def _validate_signal_dict(self, data: Dict[str, Any], 
                            level: ValidationLevel) -> ValidationResult:
        """验证信号字典数据"""
        result = ValidationResult(is_valid=True, message="开始字典验证")
        
        # 验证必需字段
        required_fields = ['timestamp', 'center_frequency', 'sample_rate']
        for field in required_fields:
            if field not in data:
                error_msg = f"缺少必需字段: {field}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证时间戳
        if 'timestamp' in data:
            ts_result = self._validate_timestamp(data['timestamp'], level)
            result = result.merge(ts_result)
        
        # 验证频率
        if 'center_frequency' in data:
            freq_result = self._validate_frequency(data['center_frequency'], level)
            result = result.merge(freq_result)
        
        # 验证采样率
        if 'sample_rate' in data:
            sr_result = self._validate_sample_rate(data['sample_rate'], level)
            result = result.merge(sr_result)
        
        # 验证IQ数据
        if 'iq_data' in data and data['iq_data'] is not None:
            iq_data = data['iq_data']
            if isinstance(iq_data, np.ndarray):
                iq_result = self._validate_iq_array(iq_data, level)
            elif isinstance(iq_data, (list, tuple)):
                iq_result = self._validate_iq_list(iq_data, level)
            else:
                error_msg = f"不支持的IQ数据类型: {type(iq_data).__name__}"
                if level == ValidationLevel.STRICT:
                    iq_result = ValidationResult(is_valid=False, errors=[error_msg])
                else:
                    iq_result = ValidationResult(is_valid=True, warnings=[error_msg])
            result = result.merge(iq_result)
        
        # 验证元数据
        if 'metadata' in data and data['metadata'] is not None:
            if not isinstance(data['metadata'], dict):
                error_msg = f"元数据必须是字典，得到 {type(data['metadata']).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        return result
    
    def _validate_iq_array(self, iq_array: np.ndarray, 
                          level: ValidationLevel) -> ValidationResult:
        """验证IQ数组数据"""
        result = ValidationResult(is_valid=True, message="开始IQ数组验证")
        
        # 检查数组维度
        if iq_array.ndim != 1:
            error_msg = f"IQ数组必须是1维，得到 {iq_array.ndim} 维"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        # 检查数组大小
        if len(iq_array) == 0:
            error_msg = "IQ数组不能为空"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        # 检查数组类型
        if not np.iscomplexobj(iq_array):
            error_msg = f"IQ数组必须是复数类型，得到 {iq_array.dtype}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        # 检查NaN和Inf
        if np.any(np.isnan(iq_array)):
            error_msg = "IQ数组包含NaN值"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        if np.any(np.isinf(iq_array)):
            error_msg = "IQ数组包含无穷大值"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        # 计算统计信息
        amplitude = np.abs(iq_array)
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        
        # 检查幅度范围
        if mean_amp > 1e6:  # 非常大
            result.add_warning(f"平均幅度异常大: {mean_amp:.2e}")
        
        if std_amp / mean_amp > 10:  # 标准差过大
            result.add_warning(f"幅度波动过大: 标准差/均值 = {std_amp/mean_amp:.2f}")
        
        result.data = {
            'array': iq_array,
            'sample_count': len(iq_array),
            'mean_amplitude': mean_amp,
            'std_amplitude': std_amp,
            'max_amplitude': np.max(amplitude),
            'min_amplitude': np.min(amplitude)
        }
        
        return result
    
    def _validate_iq_list(self, iq_list: List, level: ValidationLevel) -> ValidationResult:
        """验证IQ列表数据"""
        result = ValidationResult(is_valid=True, message="开始IQ列表验证")
        
        if len(iq_list) == 0:
            error_msg = "IQ列表不能为空"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        # 转换为NumPy数组
        try:
            iq_array = np.array(iq_list, dtype=np.complex64)
            array_result = self._validate_iq_array(iq_array, level)
            result = result.merge(array_result)
        except Exception as e:
            error_msg = f"无法将IQ列表转换为数组: {e}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def _validate_timestamp(self, timestamp: Any, level: ValidationLevel) -> ValidationResult:
        """验证时间戳"""
        result = ValidationResult(is_valid=True, message="时间戳验证")
        
        if isinstance(timestamp, datetime):
            # 已经是datetime对象
            if timestamp > datetime.now() + timedelta(days=1):
                result.add_warning(f"时间戳在未来: {timestamp}")
            if timestamp < datetime(2000, 1, 1):
                result.add_warning(f"时间戳过于久远: {timestamp}")
            
            result.data = timestamp
        
        elif isinstance(timestamp, (int, float)):
            # Unix时间戳
            try:
                dt = datetime.fromtimestamp(timestamp)
                result.data = dt
            except Exception as e:
                error_msg = f"无效的时间戳: {timestamp}, 错误: {e}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        elif isinstance(timestamp, str):
            # 字符串时间戳
            try:
                # 尝试多种格式
                for fmt in ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z', 
                          '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                          '%Y%m%d%H%M%S']:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        result.data = dt
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"无法解析时间戳: {timestamp}")
            except Exception as e:
                error_msg = f"无效的时间戳字符串: {timestamp}, 错误: {e}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        else:
            error_msg = f"不支持的timestamp类型: {type(timestamp).__name__}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def _validate_frequency(self, frequency: Any, level: ValidationLevel) -> ValidationResult:
        """验证频率"""
        result = ValidationResult(is_valid=True, message="频率验证")
        
        try:
            freq = float(frequency)
            
            if freq < 0:
                error_msg = f"频率不能为负: {freq}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            
            if freq > 100e9:  # 100GHz
                result.add_warning(f"频率异常高: {freq:.1f} Hz")
            
            if freq < 1:  # 1Hz
                result.add_warning(f"频率异常低: {freq:.1f} Hz")
            
            result.data = freq
            
        except (ValueError, TypeError) as e:
            error_msg = f"无效的频率值: {frequency}, 错误: {e}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def _validate_sample_rate(self, sample_rate: Any, level: ValidationLevel) -> ValidationResult:
        """验证采样率"""
        result = ValidationResult(is_valid=True, message="采样率验证")
        
        try:
            sr = float(sample_rate)
            
            if sr <= 0:
                error_msg = f"采样率必须为正: {sr}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            
            if sr > 10e9:  # 10GS/s
                result.add_warning(f"采样率异常高: {sr:.1f} Hz")
            
            if sr < 1:  # 1Hz
                result.add_warning(f"采样率异常低: {sr:.1f} Hz")
            
            result.data = sr
            
        except (ValueError, TypeError) as e:
            error_msg = f"无效的采样率: {sample_rate}, 错误: {e}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def validate_signal_quality(self, iq_data: np.ndarray, 
                              sample_rate: float = None) -> ValidationResult:
        """
        验证信号质量
        
        Args:
            iq_data: IQ数据数组
            sample_rate: 采样率
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True, message="信号质量验证")
        
        if iq_data is None or len(iq_data) == 0:
            result.add_warning("IQ数据为空")
            return result
        
        # 基本数组验证
        array_result = self._validate_iq_array(iq_data, ValidationLevel.WARNING)
        if not array_result:
            return array_result
        
        amplitude = np.abs(iq_data)
        
        # 1. 检查直流偏移
        dc_offset = np.abs(np.mean(iq_data))
        if dc_offset > 0.1 * np.mean(amplitude):  # DC偏移超过10%
            result.add_warning(f"直流偏移较大: {dc_offset:.2e}")
        
        # 2. 检查幅度平坦度
        amplitude_variation = np.std(amplitude) / np.mean(amplitude)
        if amplitude_variation > 2.0:  # 幅度变化过大
            result.add_warning(f"幅度变化过大: 变异系数 = {amplitude_variation:.2f}")
        
        # 3. 检查相位连续性
        phase = np.angle(iq_data)
        phase_diff = np.diff(phase)
        
        # 处理相位跳变
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_jumps = np.sum(np.abs(phase_diff) > np.pi / 2)  # 大于90度的跳变
        
        if phase_jumps > len(iq_data) * 0.1:  # 超过10%的相位跳变
            result.add_warning(f"相位跳变过多: {phase_jumps} 次")
        
        # 4. 计算信噪比（简化估计）
        if sample_rate and len(iq_data) > 100:
            # 计算频谱
            f, pxx = scipy_signal.welch(iq_data, fs=sample_rate, nperseg=min(1024, len(iq_data)))
            
            # 找到主峰
            peak_idx = np.argmax(pxx)
            peak_freq = f[peak_idx]
            peak_power = pxx[peak_idx]
            
            # 计算噪声基底（排除主峰附近）
            mask = np.ones_like(pxx, dtype=bool)
            mask[max(0, peak_idx-5):min(len(pxx), peak_idx+6)] = False
            
            if np.sum(mask) > 0:
                noise_floor = np.median(pxx[mask])
                if noise_floor > 0:
                    snr_estimate = 10 * np.log10(peak_power / noise_floor)
                    if snr_estimate < 10:  # SNR < 10dB
                        result.add_warning(f"信噪比较低: {snr_estimate:.1f} dB")
                else:
                    result.add_warning("噪声基底为0，无法计算信噪比")
        
        # 5. 检查IQ平衡
        I = iq_data.real
        Q = iq_data.imag
        
        I_mean, I_std = np.mean(I), np.std(I)
        Q_mean, Q_std = np.mean(Q), np.std(Q)
        
        # 检查均值平衡
        mean_imbalance = abs(I_mean - Q_mean) / (abs(I_mean) + abs(Q_mean) + 1e-10)
        if mean_imbalance > 0.1:  # 均值不平衡超过10%
            result.add_warning(f"IQ均值不平衡: {mean_imbalance:.2f}")
        
        # 检查方差平衡
        std_imbalance = abs(I_std - Q_std) / (I_std + Q_std + 1e-10)
        if std_imbalance > 0.1:  # 方差不平衡超过10%
            result.add_warning(f"IQ方差不平衡: {std_imbalance:.2f}")
        
        # 计算质量分数
        quality_score = self._calculate_quality_score(result.warnings, result.errors)
        result.data = {
            'quality_score': quality_score,
            'dc_offset': dc_offset,
            'amplitude_variation': amplitude_variation,
            'phase_jumps': phase_jumps,
            'I_mean': I_mean, 'I_std': I_std,
            'Q_mean': Q_mean, 'Q_std': Q_std
        }
        
        if 'snr_estimate' in locals():
            result.data['snr_estimate'] = snr_estimate
        
        return result
    
    def _calculate_quality_score(self, warnings: List[str], errors: List[str]) -> float:
        """计算质量分数"""
        # 基础分数
        score = 100.0
        
        # 每个警告扣5分
        score -= len(warnings) * 5
        
        # 每个错误扣20分
        score -= len(errors) * 20
        
        # 确保分数在0-100之间
        return max(0.0, min(100.0, score))
    
    def validate_signal_parameters(self, parameters: Dict[str, Any], 
                                 level: ValidationLevel = None) -> ValidationResult:
        """
        验证信号参数
        
        Args:
            parameters: 信号参数字典
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        result = ValidationResult(is_valid=True, message="信号参数验证")
        
        if not parameters:
            result.add_warning("信号参数为空")
            return result
        
        # 验证频率参数
        if 'center_frequency' in parameters:
            freq_result = self._validate_frequency(parameters['center_frequency'], level)
            result = result.merge(freq_result)
        
        if 'bandwidth' in parameters and parameters['bandwidth'] is not None:
            bw = parameters['bandwidth']
            if bw < 0:
                error_msg = f"带宽不能为负: {bw}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证功率参数
        if 'power_db' in parameters and parameters['power_db'] is not None:
            power = parameters['power_db']
            if not isinstance(power, (int, float)):
                error_msg = f"功率必须是数字: {type(power).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证信噪比
        if 'snr_db' in parameters and parameters['snr_db'] is not None:
            snr = parameters['snr_db']
            if not isinstance(snr, (int, float)):
                error_msg = f"信噪比必须是数字: {type(snr).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        return result


class DetectionResultValidator(BaseValidator):
    """检测结果验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        """初始化检测结果验证器"""
        super().__init__(level)
        self._setup_detection_schemas()
    
    def _setup_detection_schemas(self):
        """设置检测结果Schema"""
        # 检测结果详细Schema
        self.schemas['detection_detail'] = {
            "type": "object",
            "required": ["detection_id", "signal_id", "detection_type", "confidence"],
            "properties": {
                "detection_id": {"type": "string", "minLength": 1},
                "signal_id": {"type": "string", "minLength": 1},
                "detection_type": {"type": "string", "minLength": 1},
                "detection_time": {"type": "string", "format": "date-time"},
                "center_frequency": {"type": "number", "minimum": 0},
                "bandwidth": {"type": "number", "minimum": 0},
                "power_db": {"type": "number"},
                "snr_db": {"type": "number"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "is_known": {"type": "boolean"},
                "known_signal_id": {"type": "string"},
                "classification": {"type": "string"},
                "modulation_type": {"type": "string"},
                "direction_angle": {"type": "number", "minimum": 0, "maximum": 360},
                "location_lat": {"type": "number", "minimum": -90, "maximum": 90},
                "location_lon": {"type": "number", "minimum": -180, "maximum": 180},
                "features": {"type": "object"},
                "metadata": {"type": "object"}
            }
        }
    
    def validate_detection_result(self, detection: Any, 
                                level: ValidationLevel = None) -> ValidationResult:
        """
        验证检测结果
        
        Args:
            detection: 检测结果
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        result = ValidationResult(is_valid=True, message="开始检测结果验证")
        
        if detection is None:
            error_msg = "检测结果为空"
            if level == ValidationLevel.STRICT:
                return ValidationResult(is_valid=False, errors=[error_msg])
            else:
                return ValidationResult(is_valid=True, warnings=[error_msg])
        
        # 检查是否为字典格式
        if isinstance(detection, dict):
            dict_result = self._validate_detection_dict(detection, level)
            result = result.merge(dict_result)
        
        # 检查是否为对象（有to_dict方法）
        elif hasattr(detection, 'to_dict'):
            try:
                detection_dict = detection.to_dict()
                dict_result = self._validate_detection_dict(detection_dict, level)
                result = result.merge(dict_result)
            except Exception as e:
                error_msg = f"无法转换检测结果为字典: {e}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        else:
            error_msg = f"不支持的检测结果类型: {type(detection).__name__}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        if result.is_valid:
            result.message = "检测结果验证通过"
        
        return result
    
    def _validate_detection_dict(self, detection: Dict[str, Any], 
                               level: ValidationLevel) -> ValidationResult:
        """验证检测结果字典"""
        result = ValidationResult(is_valid=True, message="检测字典验证")
        
        # 验证必需字段
        required_fields = ['detection_id', 'signal_id', 'detection_type', 'confidence']
        for field in required_fields:
            if field not in detection:
                error_msg = f"缺少必需字段: {field}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证检测ID
        if 'detection_id' in detection:
            if not detection['detection_id'] or not isinstance(detection['detection_id'], str):
                error_msg = f"无效的detection_id: {detection['detection_id']}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证信号ID
        if 'signal_id' in detection:
            if not detection['signal_id'] or not isinstance(detection['signal_id'], str):
                error_msg = f"无效的signal_id: {detection['signal_id']}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证检测时间
        if 'detection_time' in detection and detection['detection_time']:
            signal_validator = SignalDataValidator(level)
            ts_result = signal_validator._validate_timestamp(detection['detection_time'], level)
            result = result.merge(ts_result)
        
        # 验证置信度
        if 'confidence' in detection:
            conf_result = self._validate_confidence(detection['confidence'], level)
            result = result.merge(conf_result)
        
        # 验证频率
        if 'center_frequency' in detection and detection['center_frequency'] is not None:
            signal_validator = SignalDataValidator(level)
            freq_result = signal_validator._validate_frequency(detection['center_frequency'], level)
            result = result.merge(freq_result)
        
        # 验证功率
        if 'power_db' in detection and detection['power_db'] is not None:
            power = detection['power_db']
            if not isinstance(power, (int, float)):
                error_msg = f"功率必须是数字: {type(power).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            elif power > 100:  # 异常高功率
                result.add_warning(f"功率异常高: {power} dB")
            elif power < -200:  # 异常低功率
                result.add_warning(f"功率异常低: {power} dB")
        
        # 验证位置
        if 'location_lat' in detection and detection['location_lat'] is not None:
            lat = detection['location_lat']
            if not isinstance(lat, (int, float)):
                error_msg = f"纬度必须是数字: {type(lat).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            elif not (-90 <= lat <= 90):
                error_msg = f"纬度必须在-90到90之间: {lat}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        if 'location_lon' in detection and detection['location_lon'] is not None:
            lon = detection['location_lon']
            if not isinstance(lon, (int, float)):
                error_msg = f"经度必须是数字: {type(lon).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            elif not (-180 <= lon <= 180):
                error_msg = f"经度必须在-180到180之间: {lon}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        # 验证方向
        if 'direction_angle' in detection and detection['direction_angle'] is not None:
            angle = detection['direction_angle']
            if not isinstance(angle, (int, float)):
                error_msg = f"方向角必须是数字: {type(angle).__name__}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            elif not (0 <= angle <= 360):
                error_msg = f"方向角必须在0到360度之间: {angle}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        return result
    
    def _validate_confidence(self, confidence: Any, level: ValidationLevel) -> ValidationResult:
        """验证置信度"""
        result = ValidationResult(is_valid=True, message="置信度验证")
        
        try:
            conf = float(confidence)
            
            if not (0 <= conf <= 1):
                error_msg = f"置信度必须在0到1之间: {conf}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
            
            # 分类置信度级别
            if conf >= 0.9:
                conf_level = "非常高"
            elif conf >= 0.8:
                conf_level = "高"
            elif conf >= 0.7:
                conf_level = "中等"
            elif conf >= 0.6:
                conf_level = "低"
            else:
                conf_level = "非常低"
                result.add_warning(f"置信度较低: {conf:.2f}")
            
            result.data = {
                'confidence': conf,
                'confidence_level': conf_level
            }
            
        except (ValueError, TypeError) as e:
            error_msg = f"无效的置信度: {confidence}, 错误: {e}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def validate_detection_consistency(self, detection: Dict[str, Any], 
                                     reference_data: Dict[str, Any] = None) -> ValidationResult:
        """
        验证检测结果一致性
        
        Args:
            detection: 检测结果
            reference_data: 参考数据（如信号数据）
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True, message="检测一致性验证")
        
        if reference_data is None:
            result.add_warning("无参考数据，跳过一致性验证")
            return result
        
        # 验证频率一致性
        if 'center_frequency' in detection and 'center_frequency' in reference_data:
            det_freq = detection.get('center_frequency')
            ref_freq = reference_data.get('center_frequency')
            
            if det_freq is not None and ref_freq is not None:
                freq_diff = abs(det_freq - ref_freq)
                if freq_diff > 1e6:  # 1MHz差异
                    result.add_warning(f"频率不一致: 检测 {det_freq/1e6:.1f}MHz, 参考 {ref_freq/1e6:.1f}MHz, 差异 {freq_diff/1e6:.1f}MHz")
        
        # 验证时间一致性
        if 'detection_time' in detection and 'timestamp' in reference_data:
            det_time = detection.get('detection_time')
            ref_time = reference_data.get('timestamp')
            
            if det_time and ref_time:
                try:
                    if isinstance(det_time, str):
                        from dateutil.parser import parse
                        det_dt = parse(det_time)
                    else:
                        det_dt = det_time
                    
                    if isinstance(ref_time, str):
                        from dateutil.parser import parse
                        ref_dt = parse(ref_time)
                    else:
                        ref_dt = ref_time
                    
                    time_diff = abs((det_dt - ref_dt).total_seconds())
                    if time_diff > 60:  # 60秒差异
                        result.add_warning(f"时间不一致: 检测 {det_dt}, 信号 {ref_dt}, 差异 {time_diff:.1f}秒")
                except Exception as e:
                    result.add_warning(f"无法比较时间: {e}")
        
        return result
    
    def validate_detection_quality(self, detection: Dict[str, Any]) -> ValidationResult:
        """
        验证检测结果质量
        
        Args:
            detection: 检测结果
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True, message="检测质量验证")
        
        quality_score = 100.0
        
        # 1. 置信度评分
        confidence = detection.get('confidence', 0.0)
        if confidence < 0.6:
            quality_score -= 30
            result.add_warning(f"置信度较低: {confidence:.2f}")
        elif confidence < 0.8:
            quality_score -= 15
            result.add_warning(f"置信度中等: {confidence:.2f}")
        
        # 2. 信噪比评分
        snr_db = detection.get('snr_db')
        if snr_db is not None:
            if snr_db < 10:
                quality_score -= 20
                result.add_warning(f"信噪比较低: {snr_db:.1f} dB")
            elif snr_db < 20:
                quality_score -= 10
                result.add_warning(f"信噪比中等: {snr_db:.1f} dB")
        
        # 3. 功率评分
        power_db = detection.get('power_db')
        if power_db is not None:
            if power_db < -100:
                quality_score -= 15
                result.add_warning(f"功率过低: {power_db:.1f} dB")
            elif power_db > 0:
                quality_score -= 10
                result.add_warning(f"功率异常高: {power_db:.1f} dB")
        
        # 4. 特征完整性评分
        features = detection.get('features', {})
        if not features or len(features) < 3:
            quality_score -= 10
            result.add_warning("特征数据不足")
        
        # 5. 分类信息评分
        classification = detection.get('classification')
        if not classification or classification == 'unknown':
            quality_score -= 5
            result.add_warning("分类信息缺失或未知")
        
        # 确保分数在0-100之间
        quality_score = max(0.0, min(100.0, quality_score))
        
        # 质量等级
        if quality_score >= 90:
            quality_level = "优秀"
        elif quality_score >= 80:
            quality_level = "良好"
        elif quality_score >= 70:
            quality_level = "一般"
        elif quality_score >= 60:
            quality_level = "较差"
        else:
            quality_level = "差"
        
        result.data = {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'confidence': confidence,
            'snr_db': snr_db,
            'power_db': power_db,
            'feature_count': len(features) if features else 0
        }
        
        return result


class ConfigValidator(BaseValidator):
    """配置验证器"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        """初始化配置验证器"""
        super().__init__(level)
        self._setup_config_schemas()
    
    def _setup_config_schemas(self):
        """设置配置Schema"""
        # 数据库配置Schema
        self.schemas['db_config'] = {
            "type": "object",
            "required": ["host", "port", "username", "database"],
            "properties": {
                "host": {"type": "string", "minLength": 1},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "username": {"type": "string", "minLength": 1},
                "password": {"type": "string"},
                "database": {"type": "string", "minLength": 1},
                "charset": {"type": "string", "default": "utf8mb4"},
                "pool_size": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }
        
        # Web配置Schema
        self.schemas['web_config'] = {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string", "minLength": 1},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "debug": {"type": "boolean", "default": False},
                "workers": {"type": "integer", "minimum": 1, "maximum": 32},
                "ssl_enabled": {"type": "boolean", "default": False}
            }
        }
        
        # 日志配置Schema
        self.schemas['logging_config'] = {
            "type": "object",
            "required": ["level", "log_dir"],
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "log_dir": {"type": "string", "minLength": 1},
                "max_file_size_mb": {"type": "integer", "minimum": 1, "maximum": 1024},
                "backup_count": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }
    
    def validate_config(self, config: Any, config_type: str = None, 
                       level: ValidationLevel = None) -> ValidationResult:
        """
        验证配置
        
        Args:
            config: 配置数据
            config_type: 配置类型
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        result = ValidationResult(is_valid=True, message="开始配置验证")
        
        if config is None:
            error_msg = "配置为空"
            if level == ValidationLevel.STRICT:
                return ValidationResult(is_valid=False, errors=[error_msg])
            else:
                return ValidationResult(is_valid=True, warnings=[error_msg])
        
        # 如果是字典，使用Schema验证
        if isinstance(config, dict):
            if config_type and config_type in self.schemas:
                schema_result = self.validate_with_schema(config, config_type, level)
                result = result.merge(schema_result)
            else:
                # 通用字典验证
                dict_result = self._validate_config_dict(config, level)
                result = result.merge(dict_result)
        
        # 如果是字符串，尝试解析为JSON或YAML
        elif isinstance(config, str):
            str_result = self._validate_config_string(config, level)
            result = result.merge(str_result)
        
        # 如果是文件路径
        elif isinstance(config, (str, Path)):
            path_result = self._validate_config_file(config, level)
            result = result.merge(path_result)
        
        else:
            error_msg = f"不支持的配置类型: {type(config).__name__}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        if result.is_valid:
            result.message = "配置验证通过"
        
        return result
    
    def _validate_config_dict(self, config: Dict[str, Any], 
                            level: ValidationLevel) -> ValidationResult:
        """验证配置字典"""
        result = ValidationResult(is_valid=True, message="配置字典验证")
        
        if not config:
            result.add_warning("配置字典为空")
            return result
        
        # 检查常见配置字段
        if 'environment' in config:
            env = config['environment']
            valid_envs = ['development', 'testing', 'production']
            if env not in valid_envs:
                error_msg = f"无效的环境: {env}，有效值: {valid_envs}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        if 'debug' in config and not isinstance(config['debug'], bool):
            error_msg = f"debug必须是布尔值: {type(config['debug']).__name__}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result
    
    def _validate_config_string(self, config_str: str, 
                              level: ValidationLevel) -> ValidationResult:
        """验证配置字符串"""
        result = ValidationResult(is_valid=True, message="配置字符串验证")
        
        if not config_str.strip():
            result.add_warning("配置字符串为空")
            return result
        
        try:
            # 尝试解析为JSON
            config_data = json.loads(config_str)
            dict_result = self._validate_config_dict(config_data, level)
            result = result.merge(dict_result)
            result.data = config_data
            
        except json.JSONDecodeError:
            try:
                # 尝试解析为YAML
                config_data = yaml.safe_load(config_str)
                dict_result = self._validate_config_dict(config_data, level)
                result = result.merge(dict_result)
                result.data = config_data
                
            except yaml.YAMLError as e:
                error_msg = f"无法解析配置字符串: {e}"
                if level == ValidationLevel.STRICT:
                    result.add_error(error_msg)
                else:
                    result.add_warning(error_msg)
        
        return result
    
    def _validate_config_file(self, filepath: Union[str, Path], 
                            level: ValidationLevel) -> ValidationResult:
        """验证配置文件"""
        result = ValidationResult(is_valid=True, message="配置文件验证")
        
        path = Path(filepath)
        if not path.exists():
            error_msg = f"配置文件不存在: {filepath}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        if not path.is_file():
            error_msg = f"不是文件: {filepath}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
            return result
        
        try:
            # 根据扩展名选择解析器
            if path.suffix.lower() in ['.json', '.json5']:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                # 尝试自动检测
                content = path.read_text(encoding='utf-8')
                str_result = self._validate_config_string(content, level)
                result = result.merge(str_result)
                return result
            
            dict_result = self._validate_config_dict(config_data, level)
            result = result.merge(dict_result)
            result.data = config_data
            
        except Exception as e:
            error_msg = f"无法读取或解析配置文件 {filepath}: {e}"
            if level == ValidationLevel.STRICT:
                result.add_error(error_msg)
            else:
                result.add_warning(error_msg)
        
        return result


# 组合验证器
class DataValidator:
    """数据验证器（组合所有验证器）"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        """
        初始化数据验证器
        
        Args:
            level: 验证级别
        """
        self.level = level
        self.signal_validator = SignalDataValidator(level)
        self.detection_validator = DetectionResultValidator(level)
        self.config_validator = ConfigValidator(level)
        self.base_validator = BaseValidator(level)
    
    def validate_signal(self, signal_data: Any, 
                       validate_quality: bool = False,
                       level: ValidationLevel = None) -> ValidationResult:
        """
        验证信号数据
        
        Args:
            signal_data: 信号数据
            validate_quality: 是否验证质量
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        
        # 基本验证
        result = self.signal_validator.validate_signal_data(signal_data, level)
        
        # 质量验证
        if validate_quality and result.is_valid and result.data:
            if isinstance(signal_data, dict) and 'iq_data' in signal_data:
                iq_data = signal_data['iq_data']
                if isinstance(iq_data, np.ndarray):
                    sample_rate = signal_data.get('sample_rate')
                    quality_result = self.signal_validator.validate_signal_quality(
                        iq_data, sample_rate
                    )
                    result = result.merge(quality_result)
        
        return result
    
    def validate_detection(self, detection: Any, 
                          reference_signal: Any = None,
                          validate_quality: bool = False,
                          level: ValidationLevel = None) -> ValidationResult:
        """
        验证检测结果
        
        Args:
            detection: 检测结果
            reference_signal: 参考信号数据
            validate_quality: 是否验证质量
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        level = level or self.level
        
        # 基本验证
        result = self.detection_validator.validate_detection_result(detection, level)
        
        # 一致性验证
        if reference_signal and result.is_valid:
            if isinstance(detection, dict):
                detection_dict = detection
            elif hasattr(detection, 'to_dict'):
                detection_dict = detection.to_dict()
            else:
                detection_dict = {}
            
            if isinstance(reference_signal, dict):
                signal_dict = reference_signal
            elif hasattr(reference_signal, 'to_dict'):
                signal_dict = reference_signal.to_dict()
            else:
                signal_dict = {}
            
            consistency_result = self.detection_validator.validate_detection_consistency(
                detection_dict, signal_dict
            )
            result = result.merge(consistency_result)
        
        # 质量验证
        if validate_quality and result.is_valid:
            if isinstance(detection, dict):
                detection_dict = detection
            elif hasattr(detection, 'to_dict'):
                detection_dict = detection.to_dict()
            else:
                detection_dict = {}
            
            quality_result = self.detection_validator.validate_detection_quality(detection_dict)
            result = result.merge(quality_result)
        
        return result
    
    def validate_config(self, config: Any, config_type: str = None, 
                       level: ValidationLevel = None) -> ValidationResult:
        """
        验证配置
        
        Args:
            config: 配置数据
            config_type: 配置类型
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        return self.config_validator.validate_config(config, config_type, level or self.level)
    
    def validate_with_schema(self, data: Dict[str, Any], schema_name: str, 
                           level: ValidationLevel = None) -> ValidationResult:
        """
        使用Schema验证数据
        
        Args:
            data: 要验证的数据
            schema_name: Schema名称
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        return self.base_validator.validate_with_schema(data, schema_name, level or self.level)
    
    def validate_type(self, value: Any, expected_type: Type, 
                     level: ValidationLevel = None) -> ValidationResult:
        """
        验证类型
        
        Args:
            value: 要验证的值
            expected_type: 期望的类型
            level: 验证级别
            
        Returns:
            ValidationResult: 验证结果
        """
        return self.base_validator.validate_type(value, expected_type, level or self.level)


# 工厂函数
def create_validator(level: ValidationLevel = ValidationLevel.STRICT) -> DataValidator:
    """
    创建数据验证器
    
    Args:
        level: 验证级别
        
    Returns:
        DataValidator: 数据验证器实例
    """
    return DataValidator(level)


# 使用示例
if __name__ == "__main__":
    # 创建验证器
    validator = create_validator(ValidationLevel.WARNING)
    
    # 测试信号数据验证
    print("=== 测试信号数据验证 ===")
    
    # 创建测试信号
    test_signal = {
        'signal_id': 'test_signal_001',
        'timestamp': '2024-01-15T10:30:00',
        'center_frequency': 100e6,
        'sample_rate': 10e6,
        'bandwidth': 1e6,
        'device_id': 'ADRV9009_001',
        'channel_index': 0,
        'iq_data': np.exp(1j * 2 * np.pi * 1e6 * np.linspace(0, 0.001, 10000))
    }
    
    signal_result = validator.validate_signal(test_signal, validate_quality=True)
    print(f"信号验证: {signal_result}")
    print(f"错误: {signal_result.errors}")
    print(f"警告: {signal_result.warnings}")
    
    if signal_result.data and 'quality_score' in signal_result.data:
        print(f"质量分数: {signal_result.data['quality_score']:.1f}")
    
    # 测试检测结果验证
    print("\n=== 测试检测结果验证 ===")
    
    test_detection = {
        'detection_id': 'test_detection_001',
        'signal_id': 'test_signal_001',
        'detection_type': 'signal_presence',
        'detection_time': '2024-01-15T10:30:01',
        'center_frequency': 100.1e6,
        'bandwidth': 0.5e6,
        'power_db': -45.5,
        'snr_db': 15.2,
        'confidence': 0.85,
        'is_known': True,
        'known_signal_id': 'known_fm_100mhz',
        'classification': 'FM广播',
        'location_lat': 39.9042,
        'location_lon': 116.4074
    }
    
    detection_result = validator.validate_detection(
        test_detection, 
        reference_signal=test_signal,
        validate_quality=True
    )
    
    print(f"检测验证: {detection_result}")
    print(f"错误: {detection_result.errors}")
    print(f"警告: {detection_result.warnings}")
    
    if detection_result.data and 'quality_score' in detection_result.data:
        print(f"质量分数: {detection_result.data['quality_score']:.1f}")
        print(f"质量等级: {detection_result.data['quality_level']}")
    
    # 测试配置验证
    print("\n=== 测试配置验证 ===")
    
    test_config = {
        'environment': 'development',
        'debug': True,
        'log_level': 'INFO',
        'database': {
            'host': 'localhost',
            'port': 3306,
            'username': 'root',
            'password': 'password123',
            'database': 'signal_processing'
        }
    }
    
    config_result = validator.validate_config(test_config)
    print(f"配置验证: {config_result}")
    print(f"错误: {config_result.errors}")
    print(f"警告: {config_result.warnings}")
    
    # 测试Schema验证
    print("\n=== 测试Schema验证 ===")
    
    schema_result = validator.validate_with_schema(
        test_detection, 
        'detection_result'
    )
    print(f"Schema验证: {schema_result}")