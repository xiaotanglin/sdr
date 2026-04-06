"""
数据API模块
提供信号数据、检测结果、算法结果的数据访问接口
"""
import os
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
import logging
import traceback
from dataclasses import asdict
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path as FPath
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, and_, or_, not_, between
from pydantic import BaseModel, Field, validator, conint, confloat
import aiofiles
import aiofiles.os

# 导入配置
from config.app_config import get_current_config as get_app_config
from config.web_config import get_current_config as get_web_config
from config.db_config import get_current_config as get_db_config
from config.signal_config import get_current_config as get_signal_config

# 导入认证
from src.web.auth import (
    get_current_user, 
    get_current_active_user, 
    get_current_admin_user
)

# 导入数据库
from src.core.database_manager import DatabaseManager
from src.models.database_models import (
    SessionLocal, 
    RawSignal, 
    Detection, 
    KnownSignal,
    DetectionMatch,
    ProcessedSignal,
    AlgorithmResult,
    SystemEvent
)

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq, SignalType, SignalMetrics
from src.models.detection_result import DetectionResult, create_detection_from_signal, DetectionType, ConfidenceLevel
from src.models.database_models import DatabaseManager as ORMDatabaseManager

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult
from src.utils.file_utils import (
    ensure_directory, 
    get_file_info, 
    calculate_file_hash,
    read_json_file,
    write_json_file,
    compress_file,
    decompress_file
)
from src.utils.signal_utils import SignalUtils
from src.utils.performance_monitor import PerformanceMonitor

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# 数据模型
class TimeRange(BaseModel):
    """时间范围"""
    start_time: datetime
    end_time: datetime
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        """验证时间范围"""
        if 'start_time' in values and v < values['start_time']:
            raise ValueError('结束时间必须晚于开始时间')
        return v


class FrequencyRange(BaseModel):
    """频率范围"""
    center_frequency: Optional[float] = None
    min_frequency: Optional[float] = None
    max_frequency: Optional[float] = None
    bandwidth: Optional[float] = None
    
    @validator('max_frequency')
    def validate_frequency_range(cls, v, values):
        """验证频率范围"""
        if 'min_frequency' in values and v is not None and values['min_frequency'] is not None:
            if v <= values['min_frequency']:
                raise ValueError('最大频率必须大于最小频率')
        return v


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Query(1, ge=1, description="页码")
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """计算限制数量"""
        return self.page_size


class SignalQueryParams(PaginationParams):
    """信号查询参数"""
    device_id: Optional[str] = None
    channel_index: Optional[int] = None
    min_frequency: Optional[float] = Query(None, ge=0, description="最小频率(Hz)")
    max_frequency: Optional[float] = Query(None, ge=0, description="最大频率(Hz)")
    min_sample_rate: Optional[float] = Query(None, ge=0, description="最小采样率(Hz)")
    max_sample_rate: Optional[float] = Query(None, ge=0, description="最大采样率(Hz)")
    min_duration_ms: Optional[float] = Query(None, ge=0, description="最小持续时间(ms)")
    max_duration_ms: Optional[float] = Query(None, ge=0, description="最大持续时间(ms)")
    status: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sort_by: str = Query("timestamp", description="排序字段")
    sort_order: str = Query("desc", description="排序顺序")


class DetectionQueryParams(PaginationParams):
    """检测查询参数"""
    signal_id: Optional[str] = None
    detection_type: Optional[str] = None
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="最小置信度")
    max_confidence: Optional[float] = Query(None, ge=0, le=1, description="最大置信度")
    is_known: Optional[bool] = None
    known_signal_id: Optional[str] = None
    classification: Optional[str] = None
    modulation_type: Optional[str] = None
    min_power_db: Optional[float] = Query(None, description="最小功率(dB)")
    max_power_db: Optional[float] = Query(None, description="最大功率(dB)")
    min_snr_db: Optional[float] = Query(None, description="最小信噪比(dB)")
    max_snr_db: Optional[float] = Query(None, description="最大信噪比(dB)")
    min_frequency: Optional[float] = Query(None, ge=0, description="最小频率(Hz)")
    max_frequency: Optional[float] = Query(None, ge=0, description="最大频率(Hz)")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sort_by: str = Query("detection_time", description="排序字段")
    sort_order: str = Query("desc", description="排序顺序")


class SignalUploadRequest(BaseModel):
    """信号上传请求"""
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    device_id: str = Field(..., description="设备ID")
    channel_index: int = Field(0, description="通道索引")
    center_frequency: float = Field(..., ge=0, description="中心频率(Hz)")
    sample_rate: float = Field(..., gt=0, description="采样率(Hz)")
    bandwidth: Optional[float] = Field(None, ge=0, description="带宽(Hz)")
    gain_db: Optional[float] = Field(None, description="增益(dB)")
    noise_floor_db: Optional[float] = Field(None, description="噪声基底(dB)")
    duration_ms: Optional[float] = Field(None, gt=0, description="持续时间(ms)")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    compress: bool = Field(True, description="是否压缩")
    validate: bool = Field(True, description="是否验证")


class SignalBatchRequest(BaseModel):
    """信号批量请求"""
    signals: List[SignalUploadRequest] = Field(..., description="信号列表")
    compress: bool = Field(True, description="是否压缩")
    validate: bool = Field(True, description="是否验证")
    batch_id: Optional[str] = Field(None, description="批次ID")


class DetectionCreateRequest(BaseModel):
    """检测创建请求"""
    signal_id: str = Field(..., description="信号ID")
    detection_type: str = Field(..., description="检测类型")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    center_frequency: float = Field(..., ge=0, description="中心频率(Hz)")
    bandwidth: Optional[float] = Field(None, ge=0, description="带宽(Hz)")
    power_db: Optional[float] = Field(None, description="功率(dB)")
    snr_db: Optional[float] = Field(None, description="信噪比(dB)")
    duration_ms: Optional[float] = Field(None, gt=0, description="持续时间(ms)")
    modulation_type: Optional[str] = Field(None, description="调制类型")
    is_known: bool = Field(False, description="是否已知信号")
    known_signal_id: Optional[str] = Field(None, description="已知信号ID")
    classification: Optional[str] = Field(None, description="分类")
    features: Dict[str, Any] = Field(default_factory=dict, description="特征")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    detection_time: datetime = Field(default_factory=datetime.now, description="检测时间")
    validate: bool = Field(True, description="是否验证")


class SignalUpdateRequest(BaseModel):
    """信号更新请求"""
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None


class DetectionUpdateRequest(BaseModel):
    """检测更新请求"""
    confidence: Optional[float] = Field(None, ge=0, le=1, description="置信度")
    is_verified: Optional[bool] = Field(None, description="是否已验证")
    verification_notes: Optional[str] = Field(None, description="验证说明")
    classification: Optional[str] = Field(None, description="分类")
    features: Optional[Dict[str, Any]] = Field(None, description="特征")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    tags: Optional[List[str]] = Field(None, description="标签")


class ExportRequest(BaseModel):
    """导出请求"""
    format: str = Field("json", description="导出格式")
    compress: bool = Field(True, description="是否压缩")
    include_iq_data: bool = Field(False, description="是否包含IQ数据")
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    frequency_range: Optional[FrequencyRange] = Field(None, description="频率范围")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤器")


class DataStatisticsRequest(BaseModel):
    """数据统计请求"""
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    frequency_range: Optional[FrequencyRange] = Field(None, description="频率范围")
    group_by: Optional[List[str]] = Field(None, description="分组字段")
    metrics: Optional[List[str]] = Field(None, description="统计指标")


class DataSyncRequest(BaseModel):
    """数据同步请求"""
    source: str = Field(..., description="数据源")
    destination: str = Field(..., description="目标")
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    frequency_range: Optional[FrequencyRange] = Field(None, description="频率范围")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤器")
    batch_size: int = Field(1000, description="批处理大小")


# 依赖项
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_database_manager() -> DatabaseManager:
    """获取数据库管理器"""
    try:
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        
        # 尝试从应用状态获取
        pass
    except:
        pass
    
    # 创建新的数据库管理器
    db_config = get_db_config()
    app_config = get_app_config()
    
    if db_config.db_type == "mysql":
        conn = db_config.connection
        db_url = f"mysql+pymysql://{conn.username}:{conn.password}@{conn.host}:{conn.port}/{conn.database}"
    else:
        db_url = f"sqlite:///{app_config.data_dir}/signal_processing.db"
    
    return DatabaseManager(db_url)


def get_data_validator() -> DataValidator:
    """获取数据验证器"""
    return DataValidator(level="warning")


# 信号数据API
@router.get("/signals", tags=["signals"])
async def get_signals(
    query_params: SignalQueryParams = Depends(),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号数据列表
    
    Args:
        query_params: 查询参数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        信号数据列表
    """
    logger.info(f"用户 {current_user['username']} 查询信号数据")
    
    try:
        # 构建查询
        query = db.query(RawSignal)
        
        # 应用过滤器
        if query_params.device_id:
            query = query.filter(RawSignal.device_id == query_params.device_id)
        
        if query_params.channel_index is not None:
            query = query.filter(RawSignal.channel_index == query_params.channel_index)
        
        if query_params.min_frequency is not None:
            query = query.filter(RawSignal.center_frequency >= query_params.min_frequency)
        
        if query_params.max_frequency is not None:
            query = query.filter(RawSignal.center_frequency <= query_params.max_frequency)
        
        if query_params.min_sample_rate is not None:
            query = query.filter(RawSignal.sample_rate >= query_params.min_sample_rate)
        
        if query_params.max_sample_rate is not None:
            query = query.filter(RawSignal.sample_rate <= query_params.max_sample_rate)
        
        if query_params.min_duration_ms is not None:
            query = query.filter(RawSignal.duration_ms >= query_params.min_duration_ms)
        
        if query_params.max_duration_ms is not None:
            query = query.filter(RawSignal.duration_ms <= query_params.max_duration_ms)
        
        if query_params.status:
            query = query.filter(RawSignal.status == query_params.status)
        
        if query_params.start_time:
            query = query.filter(RawSignal.timestamp >= query_params.start_time)
        
        if query_params.end_time:
            query = query.filter(RawSignal.timestamp <= query_params.end_time)
        
        # 获取总数
        total_count = query.count()
        
        # 应用排序
        sort_field = getattr(RawSignal, query_params.sort_by, RawSignal.timestamp)
        if query_params.sort_order.lower() == "desc":
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(asc(sort_field))
        
        # 应用分页
        signals = query.offset(query_params.offset).limit(query_params.limit).all()
        
        # 转换为字典
        signals_data = [signal.to_dict() for signal in signals]
        
        return {
            "signals": signals_data,
            "pagination": {
                "page": query_params.page,
                "page_size": query_params.page_size,
                "total": total_count,
                "total_pages": (total_count + query_params.page_size - 1) // query_params.page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取信号数据失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取信号数据失败: {str(e)}"
        )


@router.get("/signals/{signal_id}", tags=["signals"])
async def get_signal(
    signal_id: str = FPath(..., description="信号ID"),
    include_iq_data: bool = Query(False, description="是否包含IQ数据"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取单个信号数据
    
    Args:
        signal_id: 信号ID
        include_iq_data: 是否包含IQ数据
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        信号数据
    """
    logger.info(f"用户 {current_user['username']} 获取信号 {signal_id}")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 转换为字典
        signal_data = signal.to_dict(include_iq_data=include_iq_data)
        
        return signal_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取信号 {signal_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取信号失败: {str(e)}"
        )


@router.post("/signals", tags=["signals"], status_code=status.HTTP_201_CREATED)
async def create_signal(
    request: SignalUploadRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    创建信号数据
    
    Args:
        request: 信号创建请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        创建的信号数据
    """
    logger.info(f"用户 {current_user['username']} 创建信号数据")
    
    try:
        # 验证信号数据
        if request.validate:
            validator = DataValidator(level="strict")
            signal_dict = request.dict()
            
            # 移除不需要验证的字段
            signal_dict.pop("validate", None)
            signal_dict.pop("compress", None)
            
            validation_result = validator.validate_signal(signal_dict, validate_quality=False)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"信号数据验证失败: {validation_result.errors}"
                )
        
        # 创建信号记录
        signal = RawSignal(
            signal_id=str(uuid.uuid4()),
            timestamp=request.timestamp,
            center_frequency=request.center_frequency,
            sample_rate=request.sample_rate,
            bandwidth=request.bandwidth,
            metadata=request.metadata,
            device_id=request.device_id,
            channel_index=request.channel_index,
            gain=request.gain_db,
            noise_floor=request.noise_floor_db,
            duration_ms=request.duration_ms,
            created_at=datetime.now()
        )
        
        # 设置空IQ数据
        signal.set_iq_data(None, compress=request.compress)
        
        # 保存到数据库
        db.add(signal)
        db.commit()
        db.refresh(signal)
        
        logger.info(f"信号 {signal.signal_id} 创建成功")
        
        return {
            "message": "信号创建成功",
            "signal_id": signal.signal_id,
            "signal": signal.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建信号失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建信号失败: {str(e)}"
        )


@router.post("/signals/batch", tags=["signals"], status_code=status.HTTP_201_CREATED)
async def create_signals_batch(
    request: SignalBatchRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    批量创建信号数据
    
    Args:
        request: 批量信号请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        批量创建结果
    """
    logger.info(f"用户 {current_user['username']} 批量创建 {len(request.signals)} 个信号")
    
    try:
        results = {
            "total": len(request.signals),
            "success": 0,
            "failed": 0,
            "errors": [],
            "signal_ids": []
        }
        
        validator = DataValidator(level="strict" if request.validate else "warning")
        
        for i, signal_request in enumerate(request.signals):
            try:
                # 验证信号数据
                if request.validate:
                    signal_dict = signal_request.dict()
                    signal_dict.pop("validate", None)
                    signal_dict.pop("compress", None)
                    
                    validation_result = validator.validate_signal(signal_dict, validate_quality=False)
                    if not validation_result.is_valid:
                        raise ValueError(f"验证失败: {validation_result.errors}")
                
                # 创建信号记录
                signal = RawSignal(
                    signal_id=str(uuid.uuid4()),
                    timestamp=signal_request.timestamp,
                    center_frequency=signal_request.center_frequency,
                    sample_rate=signal_request.sample_rate,
                    bandwidth=signal_request.bandwidth,
                    metadata=signal_request.metadata,
                    device_id=signal_request.device_id,
                    channel_index=signal_request.channel_index,
                    gain=signal_request.gain_db,
                    noise_floor=signal_request.noise_floor_db,
                    duration_ms=signal_request.duration_ms,
                    created_at=datetime.now()
                )
                
                # 设置空IQ数据
                signal.set_iq_data(None, compress=request.compress)
                
                # 保存到数据库
                db.add(signal)
                results["success"] += 1
                results["signal_ids"].append(signal.signal_id)
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "index": i,
                    "error": str(e)
                })
        
        # 提交事务
        db.commit()
        
        logger.info(f"批量创建完成: 成功 {results['success']}, 失败 {results['failed']}")
        
        return results
        
    except Exception as e:
        db.rollback()
        logger.error(f"批量创建信号失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量创建信号失败: {str(e)}"
        )


@router.put("/signals/{signal_id}", tags=["signals"])
async def update_signal(
    signal_id: str = FPath(..., description="信号ID"),
    request: SignalUpdateRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    更新信号数据
    
    Args:
        signal_id: 信号ID
        request: 信号更新请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        更新后的信号数据
    """
    logger.info(f"用户 {current_user['username']} 更新信号 {signal_id}")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                setattr(signal, field, value)
        
        # 更新时间戳
        signal.updated_at = datetime.now()
        
        # 保存到数据库
        db.commit()
        db.refresh(signal)
        
        logger.info(f"信号 {signal_id} 更新成功")
        
        return {
            "message": "信号更新成功",
            "signal": signal.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新信号 {signal_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新信号失败: {str(e)}"
        )


@router.delete("/signals/{signal_id}", tags=["signals"])
async def delete_signal(
    signal_id: str = FPath(..., description="信号ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    删除信号数据
    
    Args:
        signal_id: 信号ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    logger.info(f"用户 {current_user['username']} 删除信号 {signal_id}")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 删除信号
        db.delete(signal)
        db.commit()
        
        logger.info(f"信号 {signal_id} 删除成功")
        
        return {
            "message": "信号删除成功",
            "signal_id": signal_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除信号 {signal_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除信号失败: {str(e)}"
        )


@router.post("/signals/{signal_id}/upload-iq", tags=["signals"])
async def upload_signal_iq(
    signal_id: str = FPath(..., description="信号ID"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    上传信号IQ数据
    
    Args:
        signal_id: 信号ID
        file: IQ数据文件
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        上传结果
    """
    logger.info(f"用户 {current_user['username']} 上传信号 {signal_id} 的IQ数据")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 读取文件内容
        content = await file.read()
        
        # 解析IQ数据
        try:
            # 尝试解析为JSON
            iq_data = json.loads(content.decode('utf-8'))
            if isinstance(iq_data, list):
                # 转换为NumPy数组
                iq_array = np.array(iq_data, dtype=np.complex64)
            else:
                raise ValueError("IQ数据格式无效")
        except:
            # 尝试解析为二进制格式
            try:
                iq_array = np.frombuffer(content, dtype=np.complex64)
            except:
                raise ValueError("无法解析IQ数据文件")
        
        # 设置IQ数据
        signal.set_iq_data(iq_array, compress=True)
        
        # 更新持续时间
        if signal.sample_rate > 0 and len(iq_array) > 0:
            signal.duration_ms = len(iq_array) / signal.sample_rate * 1000
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"信号 {signal_id} IQ数据上传成功: {len(iq_array)} 个样本")
        
        return {
            "message": "IQ数据上传成功",
            "signal_id": signal_id,
            "samples": len(iq_array),
            "duration_ms": signal.duration_ms
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"上传IQ数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传IQ数据失败: {str(e)}"
        )


@router.get("/signals/{signal_id}/iq", tags=["signals"])
async def get_signal_iq(
    signal_id: str = FPath(..., description="信号ID"),
    format: str = Query("json", description="返回格式"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="限制样本数"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号IQ数据
    
    Args:
        signal_id: 信号ID
        format: 返回格式
        limit: 限制样本数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        IQ数据
    """
    logger.info(f"用户 {current_user['username']} 获取信号 {signal_id} 的IQ数据")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 获取IQ数据
        iq_data = signal.get_iq_data()
        if iq_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 没有IQ数据"
            )
        
        # 限制样本数
        if limit is not None and len(iq_data) > limit:
            iq_data = iq_data[:limit]
        
        # 根据格式返回
        if format.lower() == "binary":
            # 二进制格式
            return StreamingResponse(
                iter([iq_data.tobytes()]),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={signal_id}.bin",
                    "X-Signal-Samples": str(len(iq_data)),
                    "X-Signal-Sample-Rate": str(signal.sample_rate)
                }
            )
        elif format.lower() == "csv":
            # CSV格式
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入表头
            writer.writerow(["sample_index", "I", "Q", "amplitude", "phase"])
            
            # 写入数据
            for i, sample in enumerate(iq_data):
                amplitude = abs(sample)
                phase = np.angle(sample)
                writer.writerow([i, sample.real, sample.imag, amplitude, phase])
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={signal_id}.csv"
                }
            )
        else:
            # JSON格式（默认）
            iq_list = []
            for sample in iq_data[:100]:  # 限制前100个样本
                iq_list.append({
                    "I": float(sample.real),
                    "Q": float(sample.imag),
                    "amplitude": float(abs(sample)),
                    "phase": float(np.angle(sample))
                })
            
            return {
                "signal_id": signal_id,
                "samples": len(iq_data),
                "sample_rate": float(signal.sample_rate),
                "center_frequency": float(signal.center_frequency),
                "iq_data_preview": iq_list
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取IQ数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取IQ数据失败: {str(e)}"
        )


@router.get("/signals/{signal_id}/spectrum", tags=["signals"])
async def get_signal_spectrum(
    signal_id: str = FPath(..., description="信号ID"),
    window: str = Query("hann", description="窗函数"),
    nfft: Optional[int] = Query(None, description="FFT点数"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号频谱
    
    Args:
        signal_id: 信号ID
        window: 窗函数
        nfft: FFT点数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        频谱数据
    """
    logger.info(f"用户 {current_user['username']} 获取信号 {signal_id} 的频谱")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 获取IQ数据
        iq_data = signal.get_iq_data()
        if iq_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 没有IQ数据"
            )
        
        # 计算频谱
        from scipy import signal as scipy_signal
        
        if window == "hann":
            window_func = np.hanning(len(iq_data))
        elif window == "hamming":
            window_func = np.hamming(len(iq_data))
        elif window == "blackman":
            window_func = np.blackman(len(iq_data))
        else:
            window_func = np.ones(len(iq_data))
        
        # 计算FFT
        if nfft is None:
            nfft = len(iq_data)
        
        spectrum = np.fft.fft(iq_data * window_func, nfft)
        spectrum = np.fft.fftshift(spectrum)
        
        # 频率向量
        freqs = np.fft.fftfreq(nfft, 1.0/signal.sample_rate)
        freqs = np.fft.fftshift(freqs) + signal.center_frequency
        
        # 转换为dB
        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        
        # 限制数据量
        if len(freqs) > 1000:
            step = len(freqs) // 1000
            freqs = freqs[::step]
            spectrum_db = spectrum_db[::step]
        
        return {
            "signal_id": signal_id,
            "frequencies": freqs.tolist(),
            "spectrum_db": spectrum_db.tolist(),
            "max_power_db": float(np.max(spectrum_db)),
            "min_power_db": float(np.min(spectrum_db)),
            "center_frequency": float(signal.center_frequency)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算频谱失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"计算频谱失败: {str(e)}"
        )


@router.get("/signals/{signal_id}/metrics", tags=["signals"])
async def get_signal_metrics(
    signal_id: str = FPath(..., description="信号ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号指标
    
    Args:
        signal_id: 信号ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        信号指标
    """
    logger.info(f"用户 {current_user['username']} 获取信号 {signal_id} 的指标")
    
    try:
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 不存在"
            )
        
        # 获取IQ数据
        iq_data = signal.get_iq_data()
        if iq_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {signal_id} 没有IQ数据"
            )
        
        # 计算信号指标
        metrics = SignalMetrics()
        signal_data = create_signal_from_iq(
            iq_data=iq_data,
            sample_rate=signal.sample_rate,
            center_freq=signal.center_frequency
        )
        metrics = metrics.calculate_from_signal(signal_data)
        
        return metrics.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算信号指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"计算信号指标失败: {str(e)}"
        )


# 检测结果API
@router.get("/detections", tags=["detections"])
async def get_detections(
    query_params: DetectionQueryParams = Depends(),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取检测结果列表
    
    Args:
        query_params: 查询参数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        检测结果列表
    """
    logger.info(f"用户 {current_user['username']} 查询检测结果")
    
    try:
        # 构建查询
        query = db.query(Detection)
        
        # 应用过滤器
        if query_params.signal_id:
            query = query.filter(Detection.signal_id == query_params.signal_id)
        
        if query_params.detection_type:
            query = query.filter(Detection.detection_type == query_params.detection_type)
        
        if query_params.min_confidence is not None:
            query = query.filter(Detection.confidence >= query_params.min_confidence)
        
        if query_params.max_confidence is not None:
            query = query.filter(Detection.confidence <= query_params.max_confidence)
        
        if query_params.is_known is not None:
            query = query.filter(Detection.is_known == query_params.is_known)
        
        if query_params.known_signal_id:
            query = query.filter(Detection.known_signal_id == query_params.known_signal_id)
        
        if query_params.classification:
            query = query.filter(Detection.classification == query_params.classification)
        
        if query_params.modulation_type:
            query = query.filter(Detection.modulation_type == query_params.modulation_type)
        
        if query_params.min_power_db is not None:
            query = query.filter(Detection.power_db >= query_params.min_power_db)
        
        if query_params.max_power_db is not None:
            query = query.filter(Detection.power_db <= query_params.max_power_db)
        
        if query_params.min_snr_db is not None:
            query = query.filter(Detection.snr_db >= query_params.min_snr_db)
        
        if query_params.max_snr_db is not None:
            query = query.filter(Detection.snr_db <= query_params.max_snr_db)
        
        if query_params.min_frequency is not None:
            query = query.filter(Detection.center_frequency >= query_params.min_frequency)
        
        if query_params.max_frequency is not None:
            query = query.filter(Detection.center_frequency <= query_params.max_frequency)
        
        if query_params.start_time:
            query = query.filter(Detection.detection_time >= query_params.start_time)
        
        if query_params.end_time:
            query = query.filter(Detection.detection_time <= query_params.end_time)
        
        # 获取总数
        total_count = query.count()
        
        # 应用排序
        sort_field = getattr(Detection, query_params.sort_by, Detection.detection_time)
        if query_params.sort_order.lower() == "desc":
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(asc(sort_field))
        
        # 应用分页
        detections = query.offset(query_params.offset).limit(query_params.limit).all()
        
        # 转换为字典
        detections_data = [detection.to_dict() for detection in detections]
        
        return {
            "detections": detections_data,
            "pagination": {
                "page": query_params.page,
                "page_size": query_params.page_size,
                "total": total_count,
                "total_pages": (total_count + query_params.page_size - 1) // query_params.page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取检测结果失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取检测结果失败: {str(e)}"
        )


@router.get("/detections/{detection_id}", tags=["detections"])
async def get_detection(
    detection_id: str = FPath(..., description="检测ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取单个检测结果
    
    Args:
        detection_id: 检测ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        检测结果
    """
    logger.info(f"用户 {current_user['username']} 获取检测结果 {detection_id}")
    
    try:
        # 查找检测结果
        detection = db.query(Detection).filter(Detection.detection_id == detection_id).first()
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"检测结果 {detection_id} 不存在"
            )
        
        # 转换为字典
        detection_data = detection.to_dict()
        
        return detection_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取检测结果 {detection_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取检测结果失败: {str(e)}"
        )


@router.post("/detections", tags=["detections"], status_code=status.HTTP_201_CREATED)
async def create_detection(
    request: DetectionCreateRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    创建检测结果
    
    Args:
        request: 检测创建请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        创建的检测结果
    """
    logger.info(f"用户 {current_user['username']} 创建检测结果")
    
    try:
        # 验证检测结果
        if request.validate:
            validator = DataValidator(level="strict")
            detection_dict = request.dict()
            detection_dict.pop("validate", None)
            
            validation_result = validator.validate_detection(detection_dict, validate_quality=False)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"检测结果验证失败: {validation_result.errors}"
                )
        
        # 验证信号存在
        signal = db.query(RawSignal).filter(RawSignal.signal_id == request.signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {request.signal_id} 不存在"
            )
        
        # 创建检测记录
        detection = Detection(
            detection_id=str(uuid.uuid4()),
            signal_id=request.signal_id,
            detection_type=request.detection_type,
            detection_time=request.detection_time,
            center_frequency=request.center_frequency,
            bandwidth=request.bandwidth,
            power_db=request.power_db,
            snr_db=request.snr_db,
            duration_ms=request.duration_ms,
            modulation_type=request.modulation_type,
            is_known=request.is_known,
            known_signal_id=request.known_signal_id,
            classification=request.classification,
            confidence=request.confidence,
            features=request.features,
            metadata=request.metadata,
            created_at=datetime.now()
        )
        
        # 保存到数据库
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        logger.info(f"检测结果 {detection.detection_id} 创建成功")
        
        return {
            "message": "检测结果创建成功",
            "detection_id": detection.detection_id,
            "detection": detection.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建检测结果失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建检测结果失败: {str(e)}"
        )


@router.put("/detections/{detection_id}", tags=["detections"])
async def update_detection(
    detection_id: str = FPath(..., description="检测ID"),
    request: DetectionUpdateRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    更新检测结果
    
    Args:
        detection_id: 检测ID
        request: 检测更新请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        更新后的检测结果
    """
    logger.info(f"用户 {current_user['username']} 更新检测结果 {detection_id}")
    
    try:
        # 查找检测结果
        detection = db.query(Detection).filter(Detection.detection_id == detection_id).first()
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"检测结果 {detection_id} 不存在"
            )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                setattr(detection, field, value)
        
        # 保存到数据库
        db.commit()
        db.refresh(detection)
        
        logger.info(f"检测结果 {detection_id} 更新成功")
        
        return {
            "message": "检测结果更新成功",
            "detection": detection.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新检测结果 {detection_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新检测结果失败: {str(e)}"
        )


@router.delete("/detections/{detection_id}", tags=["detections"])
async def delete_detection(
    detection_id: str = FPath(..., description="检测ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    删除检测结果
    
    Args:
        detection_id: 检测ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    logger.info(f"用户 {current_user['username']} 删除检测结果 {detection_id}")
    
    try:
        # 查找检测结果
        detection = db.query(Detection).filter(Detection.detection_id == detection_id).first()
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"检测结果 {detection_id} 不存在"
            )
        
        # 删除检测结果
        db.delete(detection)
        db.commit()
        
        logger.info(f"检测结果 {detection_id} 删除成功")
        
        return {
            "message": "检测结果删除成功",
            "detection_id": detection_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除检测结果 {detection_id} 失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除检测结果失败: {str(e)}"
        )


@router.get("/detections/{detection_id}/signal", tags=["detections"])
async def get_detection_signal(
    detection_id: str = FPath(..., description="检测ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取检测结果对应的信号
    
    Args:
        detection_id: 检测ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        对应的信号数据
    """
    logger.info(f"用户 {current_user['username']} 获取检测结果 {detection_id} 的信号")
    
    try:
        # 查找检测结果
        detection = db.query(Detection).filter(Detection.detection_id == detection_id).first()
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"检测结果 {detection_id} 不存在"
            )
        
        # 查找信号
        signal = db.query(RawSignal).filter(RawSignal.signal_id == detection.signal_id).first()
        if not signal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"信号 {detection.signal_id} 不存在"
            )
        
        return signal.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取检测结果信号失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取检测结果信号失败: {str(e)}"
        )


@router.post("/detections/{detection_id}/verify", tags=["detections"])
async def verify_detection(
    detection_id: str = FPath(..., description="检测ID"),
    is_verified: bool = Body(True, description="是否验证"),
    notes: str = Body(None, description="验证说明"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    验证检测结果
    
    Args:
        detection_id: 检测ID
        is_verified: 是否验证
        notes: 验证说明
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        验证结果
    """
    logger.info(f"用户 {current_user['username']} 验证检测结果 {detection_id}: {is_verified}")
    
    try:
        # 查找检测结果
        detection = db.query(Detection).filter(Detection.detection_id == detection_id).first()
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"检测结果 {detection_id} 不存在"
            )
        
        # 更新验证状态
        detection.is_verified = is_verified
        detection.verification_notes = notes
        
        if is_verified:
            detection.verified_by = current_user['username']
            detection.verified_at = datetime.now()
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"检测结果 {detection_id} 验证成功")
        
        return {
            "message": f"检测结果已{'验证' if is_verified else '取消验证'}",
            "detection_id": detection_id,
            "is_verified": is_verified,
            "verified_by": current_user['username'] if is_verified else None,
            "verified_at": detection.verified_at.isoformat() if is_verified else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"验证检测结果失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"验证检测结果失败: {str(e)}"
        )


# 数据统计API
@router.get("/statistics/signals", tags=["statistics"])
async def get_signals_statistics(
    time_range: TimeRange = None,
    frequency_range: FrequencyRange = None,
    group_by: List[str] = Query(None, description="分组字段"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号数据统计
    
    Args:
        time_range: 时间范围
        frequency_range: 频率范围
        group_by: 分组字段
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        信号统计
    """
    logger.info(f"用户 {current_user['username']} 获取信号统计")
    
    try:
        # 构建查询
        query = db.query(RawSignal)
        
        # 应用过滤器
        if time_range:
            query = query.filter(
                RawSignal.timestamp >= time_range.start_time,
                RawSignal.timestamp <= time_range.end_time
            )
        
        if frequency_range:
            if frequency_range.min_frequency is not None:
                query = query.filter(RawSignal.center_frequency >= frequency_range.min_frequency)
            if frequency_range.max_frequency is not None:
                query = query.filter(RawSignal.center_frequency <= frequency_range.max_frequency)
        
        # 获取统计数据
        stats = {
            "total": query.count(),
            "time_range": {
                "start": time_range.start_time if time_range else None,
                "end": time_range.end_time if time_range else None
            },
            "frequency_range": {
                "min": frequency_range.min_frequency if frequency_range else None,
                "max": frequency_range.max_frequency if frequency_range else None
            }
        }
        
        # 如果有分组字段
        if group_by:
            stats["group_by"] = {}
            
            for field in group_by:
                if hasattr(RawSignal, field):
                    # 获取分组统计
                    subquery = query.group_by(getattr(RawSignal, field))
                    result = subquery.with_entities(
                        getattr(RawSignal, field),
                        func.count(RawSignal.id)
                    ).all()
                    
                    stats["group_by"][field] = {
                        str(value): count for value, count in result
                    }
        
        # 添加设备统计
        device_stats = query.group_by(RawSignal.device_id).with_entities(
            RawSignal.device_id,
            func.count(RawSignal.id)
        ).all()
        
        stats["by_device"] = {
            device_id: count for device_id, count in device_stats
        }
        
        # 添加时间统计
        if query.count() > 0:
            time_stats = query.with_entities(
                func.min(RawSignal.timestamp),
                func.max(RawSignal.timestamp)
            ).first()
            
            stats["time_span"] = {
                "earliest": time_stats[0].isoformat() if time_stats[0] else None,
                "latest": time_stats[1].isoformat() if time_stats[1] else None
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"获取信号统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取信号统计失败: {str(e)}"
        )


@router.get("/statistics/detections", tags=["statistics"])
async def get_detections_statistics(
    time_range: TimeRange = None,
    frequency_range: FrequencyRange = None,
    group_by: List[str] = Query(None, description="分组字段"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取检测结果统计
    
    Args:
        time_range: 时间范围
        frequency_range: 频率范围
        group_by: 分组字段
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        检测统计
    """
    logger.info(f"用户 {current_user['username']} 获取检测统计")
    
    try:
        # 构建查询
        query = db.query(Detection)
        
        # 应用过滤器
        if time_range:
            query = query.filter(
                Detection.detection_time >= time_range.start_time,
                Detection.detection_time <= time_range.end_time
            )
        
        if frequency_range:
            if frequency_range.min_frequency is not None:
                query = query.filter(Detection.center_frequency >= frequency_range.min_frequency)
            if frequency_range.max_frequency is not None:
                query = query.filter(Detection.center_frequency <= frequency_range.max_frequency)
        
        # 获取统计数据
        stats = {
            "total": query.count(),
            "time_range": {
                "start": time_range.start_time if time_range else None,
                "end": time_range.end_time if time_range else None
            },
            "frequency_range": {
                "min": frequency_range.min_frequency if frequency_range else None,
                "max": frequency_range.max_frequency if frequency_range else None
            }
        }
        
        # 如果有分组字段
        if group_by:
            stats["group_by"] = {}
            
            for field in group_by:
                if hasattr(Detection, field):
                    # 获取分组统计
                    subquery = query.group_by(getattr(Detection, field))
                    result = subquery.with_entities(
                        getattr(Detection, field),
                        func.count(Detection.id)
                    ).all()
                    
                    stats["group_by"][field] = {
                        str(value): count for value, count in result
                    }
        
        # 添加检测类型统计
        type_stats = query.group_by(Detection.detection_type).with_entities(
            Detection.detection_type,
            func.count(Detection.id)
        ).all()
        
        stats["by_type"] = {
            detection_type: count for detection_type, count in type_stats
        }
        
        # 添加置信度统计
        confidence_stats = query.with_entities(
            func.min(Detection.confidence),
            func.max(Detection.confidence),
            func.avg(Detection.confidence)
        ).first()
        
        stats["confidence"] = {
            "min": float(confidence_stats[0]) if confidence_stats[0] else None,
            "max": float(confidence_stats[1]) if confidence_stats[1] else None,
            "avg": float(confidence_stats[2]) if confidence_stats[2] else None
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"获取检测统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取检测统计失败: {str(e)}"
        )


# 数据导出API
@router.post("/export/signals", tags=["export"])
async def export_signals(
    request: ExportRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    导出信号数据
    
    Args:
        request: 导出请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        导出的数据
    """
    logger.info(f"用户 {current_user['username']} 导出信号数据")
    
    try:
        # 构建查询
        query = db.query(RawSignal)
        
        # 应用过滤器
        if request.time_range:
            query = query.filter(
                RawSignal.timestamp >= request.time_range.start_time,
                RawSignal.timestamp <= request.time_range.end_time
            )
        
        if request.frequency_range:
            if request.frequency_range.min_frequency is not None:
                query = query.filter(RawSignal.center_frequency >= request.frequency_range.min_frequency)
            if request.frequency_range.max_frequency is not None:
                query = query.filter(RawSignal.center_frequency <= request.frequency_range.max_frequency)
        
        # 获取数据
        signals = query.all()
        
        # 转换为字典
        signals_data = []
        for signal in signals:
            signal_dict = signal.to_dict(include_iq_data=request.include_iq_data)
            signals_data.append(signal_dict)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            if request.format.lower() == "json":
                json.dump(signals_data, f, indent=2, default=str)
            else:
                # 默认JSON格式
                json.dump(signals_data, f, indent=2, default=str)
            
            temp_file = f.name
        
        # 压缩文件
        if request.compress:
            compressed_file = compress_file(temp_file, method="gzip")
            os.unlink(temp_file)
            export_file = compressed_file
            content_type = "application/gzip"
            filename = f"signals_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
        else:
            export_file = temp_file
            content_type = "application/json"
            filename = f"signals_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 返回文件
        return FileResponse(
            path=export_file,
            filename=filename,
            media_type=content_type,
            headers={
                "X-Total-Signals": str(len(signals_data)),
                "X-Export-Time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"导出信号数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出信号数据失败: {str(e)}"
        )


@router.post("/export/detections", tags=["export"])
async def export_detections(
    request: ExportRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    导出检测结果
    
    Args:
        request: 导出请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        导出的数据
    """
    logger.info(f"用户 {current_user['username']} 导出检测结果")
    
    try:
        # 构建查询
        query = db.query(Detection)
        
        # 应用过滤器
        if request.time_range:
            query = query.filter(
                Detection.detection_time >= request.time_range.start_time,
                Detection.detection_time <= request.time_range.end_time
            )
        
        if request.frequency_range:
            if request.frequency_range.min_frequency is not None:
                query = query.filter(Detection.center_frequency >= request.frequency_range.min_frequency)
            if request.frequency_range.max_frequency is not None:
                query = query.filter(Detection.center_frequency <= request.frequency_range.max_frequency)
        
        # 获取数据
        detections = query.all()
        
        # 转换为字典
        detections_data = [detection.to_dict() for detection in detections]
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            if request.format.lower() == "json":
                json.dump(detections_data, f, indent=2, default=str)
            elif request.format.lower() == "csv":
                # 转换为DataFrame
                df = pd.DataFrame(detections_data)
                df.to_csv(f, index=False)
            else:
                # 默认JSON格式
                json.dump(detections_data, f, indent=2, default=str)
            
            temp_file = f.name
        
        # 压缩文件
        if request.compress:
            compressed_file = compress_file(temp_file, method="gzip")
            os.unlink(temp_file)
            export_file = compressed_file
            content_type = "application/gzip"
            filename = f"detections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}.gz"
        else:
            export_file = temp_file
            content_type = "application/json" if request.format.lower() == "json" else "text/csv"
            filename = f"detections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}"
        
        # 返回文件
        return FileResponse(
            path=export_file,
            filename=filename,
            media_type=content_type,
            headers={
                "X-Total-Detections": str(len(detections_data)),
                "X-Export-Time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"导出检测结果失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出检测结果失败: {str(e)}"
        )


# 数据清理API
@router.delete("/cleanup/signals", tags=["cleanup"])
async def cleanup_signals(
    days_to_keep: int = Query(30, ge=1, description="保留天数"),
    dry_run: bool = Query(False, description="试运行"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    清理旧信号数据
    
    Args:
        days_to_keep: 保留天数
        dry_run: 试运行
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        清理结果
    """
    logger.info(f"管理员 {current_user['username']} 清理信号数据，保留 {days_to_keep} 天")
    
    try:
        # 计算截止时间
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        # 查找要删除的信号
        signals_to_delete = db.query(RawSignal).filter(
            RawSignal.timestamp < cutoff_time
        ).all()
        
        # 试运行
        if dry_run:
            return {
                "message": "试运行完成",
                "dry_run": True,
                "cutoff_time": cutoff_time.isoformat(),
                "signals_to_delete": len(signals_to_delete),
                "signal_ids": [signal.signal_id for signal in signals_to_delete]
            }
        
        # 实际删除
        deleted_count = 0
        for signal in signals_to_delete:
            db.delete(signal)
            deleted_count += 1
        
        db.commit()
        
        logger.info(f"清理了 {deleted_count} 个信号数据")
        
        return {
            "message": "信号数据清理完成",
            "dry_run": False,
            "cutoff_time": cutoff_time.isoformat(),
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"清理信号数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理信号数据失败: {str(e)}"
        )


@router.delete("/cleanup/detections", tags=["cleanup"])
async def cleanup_detections(
    days_to_keep: int = Query(30, ge=1, description="保留天数"),
    dry_run: bool = Query(False, description="试运行"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    清理旧检测结果
    
    Args:
        days_to_keep: 保留天数
        dry_run: 试运行
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        清理结果
    """
    logger.info(f"管理员 {current_user['username']} 清理检测结果，保留 {days_to_keep} 天")
    
    try:
        # 计算截止时间
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        # 查找要删除的检测结果
        detections_to_delete = db.query(Detection).filter(
            Detection.detection_time < cutoff_time
        ).all()
        
        # 试运行
        if dry_run:
            return {
                "message": "试运行完成",
                "dry_run": True,
                "cutoff_time": cutoff_time.isoformat(),
                "detections_to_delete": len(detections_to_delete),
                "detection_ids": [detection.detection_id for detection in detections_to_delete]
            }
        
        # 实际删除
        deleted_count = 0
        for detection in detections_to_delete:
            db.delete(detection)
            deleted_count += 1
        
        db.commit()
        
        logger.info(f"清理了 {deleted_count} 个检测结果")
        
        return {
            "message": "检测结果清理完成",
            "dry_run": False,
            "cutoff_time": cutoff_time.isoformat(),
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"清理检测结果失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理检测结果失败: {str(e)}"
        )


# 数据同步API
@router.post("/sync", tags=["sync"])
async def sync_data(
    request: DataSyncRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    同步数据
    
    Args:
        request: 同步请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        同步结果
    """
    logger.info(f"管理员 {current_user['username']} 同步数据: {request.source} -> {request.destination}")
    
    try:
        # 这里实现数据同步逻辑
        # 实际实现将取决于具体的数据源和目标
        
        return {
            "message": "数据同步功能待实现",
            "request": request.dict(),
            "status": "not_implemented"
        }
        
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据同步失败: {str(e)}"
        )


# 搜索API
@router.get("/search", tags=["search"])
async def search_data(
    query: str = Query(..., description="搜索查询"),
    search_type: str = Query("all", description="搜索类型"),
    limit: int = Query(20, description="限制数量"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    搜索数据
    
    Args:
        query: 搜索查询
        search_type: 搜索类型
        limit: 限制数量
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        搜索结果
    """
    logger.info(f"用户 {current_user['username']} 搜索: {query}")
    
    try:
        results = {
            "query": query,
            "search_type": search_type,
            "results": {
                "signals": [],
                "detections": []
            }
        }
        
        if search_type in ["all", "signals"]:
            # 搜索信号
            signal_results = db.query(RawSignal).filter(
                or_(
                    RawSignal.signal_id.ilike(f"%{query}%"),
                    RawSignal.device_id.ilike(f"%{query}%"),
                    RawSignal.metadata.like(f"%{query}%")
                )
            ).limit(limit).all()
            
            results["results"]["signals"] = [
                {
                    "id": signal.signal_id,
                    "timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
                    "device_id": signal.device_id,
                    "frequency": float(signal.center_frequency) if signal.center_frequency else None,
                    "sample_rate": float(signal.sample_rate) if signal.sample_rate else None
                }
                for signal in signal_results
            ]
        
        if search_type in ["all", "detections"]:
            # 搜索检测结果
            detection_results = db.query(Detection).filter(
                or_(
                    Detection.detection_id.ilike(f"%{query}%"),
                    Detection.signal_id.ilike(f"%{query}%"),
                    Detection.detection_type.ilike(f"%{query}%"),
                    Detection.classification.ilike(f"%{query}%"),
                    Detection.modulation_type.ilike(f"%{query}%"),
                    Detection.metadata.like(f"%{query}%")
                )
            ).limit(limit).all()
            
            results["results"]["detections"] = [
                {
                    "id": detection.detection_id,
                    "signal_id": detection.signal_id,
                    "detection_type": detection.detection_type,
                    "confidence": detection.confidence,
                    "frequency": float(detection.center_frequency) if detection.center_frequency else None,
                    "classification": detection.classification
                }
                for detection in