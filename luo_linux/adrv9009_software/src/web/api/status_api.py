"""
状态API模块
提供系统状态、组件状态、性能指标、健康检查等接口
"""
import os
import sys
import time
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
import logging
import traceback
import psutil
import platform
import socket
import multiprocessing
from dataclasses import asdict
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path as FPath
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator, conint, confloat
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import aiofiles
import aiofiles.os

# 导入配置
from config.app_config import get_current_config as get_app_config
from config.web_config import get_current_config as get_web_config
from config.db_config import get_current_config as get_db_config
from config.signal_config import get_current_config as get_signal_config
from config.logging_config import get_current_config as get_logging_config

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
    SystemStatus
)

# 导入工具
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.file_utils import get_file_info, calculate_file_hash, get_directory_size
from src.utils.data_validator import DataValidator

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# 数据模型
class SystemStatusRequest(BaseModel):
    """系统状态请求"""
    component: str = Field(..., description="组件名称")
    status: str = Field(..., description="状态")
    message: Optional[str] = Field(None, description="状态消息")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU使用率")
    memory_usage: Optional[float] = Field(None, ge=0, le=100, description="内存使用率")
    disk_usage: Optional[float] = Field(None, ge=0, le=100, description="磁盘使用率")
    network_rx: Optional[int] = Field(None, ge=0, description="网络接收字节数")
    network_tx: Optional[int] = Field(None, ge=0, description="网络发送字节数")


class HealthCheckRequest(BaseModel):
    """健康检查请求"""
    component: Optional[str] = Field(None, description="组件名称")
    timeout: int = Field(5, ge=1, le=30, description="超时时间(秒)")


class PerformanceQueryParams(BaseModel):
    """性能查询参数"""
    metrics: List[str] = Field(["cpu", "memory", "disk"], description="指标列表")
    duration: int = Field(60, ge=1, le=3600, description="持续时间(秒)")
    interval: int = Field(1, ge=1, le=60, description="采样间隔(秒)")


class SystemDiagnosticRequest(BaseModel):
    """系统诊断请求"""
    include_logs: bool = Field(True, description="是否包含日志")
    include_configs: bool = Field(True, description="是否包含配置")
    include_metrics: bool = Field(True, description="是否包含指标")
    include_database: bool = Field(True, description="是否包含数据库")
    compress: bool = Field(True, description="是否压缩")


class ResourceUsage(BaseModel):
    """资源使用情况"""
    cpu_percent: float = Field(0.0, description="CPU使用率(%)")
    memory_percent: float = Field(0.0, description="内存使用率(%)")
    disk_percent: float = Field(0.0, description="磁盘使用率(%)")
    network_rx_bytes: int = Field(0, description="网络接收字节数")
    network_tx_bytes: int = Field(0, description="网络发送字节数")
    process_count: int = Field(0, description="进程数")
    thread_count: int = Field(0, description="线程数")
    open_files: int = Field(0, description="打开文件数")


class ComponentStatus(BaseModel):
    """组件状态"""
    name: str = Field(..., description="组件名称")
    status: str = Field(..., description="状态")
    alive: bool = Field(..., description="是否存活")
    uptime_seconds: Optional[float] = Field(None, description="运行时间(秒)")
    last_error: Optional[str] = Field(None, description="最后错误")
    performance: Optional[Dict[str, Any]] = Field(None, description="性能指标")
    metrics: Optional[Dict[str, Any]] = Field(None, description="指标数据")


class SystemHealth(BaseModel):
    """系统健康状态"""
    status: str = Field(..., description="整体状态")
    timestamp: datetime = Field(..., description="时间戳")
    uptime_seconds: float = Field(..., description="系统运行时间(秒)")
    version: str = Field(..., description="版本")
    environment: str = Field(..., description="环境")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="检查项")
    resources: ResourceUsage = Field(..., description="资源使用")
    components: Dict[str, ComponentStatus] = Field(..., description="组件状态")


class SystemInfo(BaseModel):
    """系统信息"""
    system: Dict[str, Any] = Field(..., description="系统信息")
    python: Dict[str, Any] = Field(..., description="Python信息")
    hardware: Dict[str, Any] = Field(..., description="硬件信息")
    network: Dict[str, Any] = Field(..., description="网络信息")
    paths: Dict[str, str] = Field(..., description="路径信息")
    config: Dict[str, Any] = Field(..., description="配置信息")


# 依赖项
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_system():
    """获取系统实例"""
    try:
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        # 在实际应用中，从请求中获取
        pass
    except:
        pass
    
    return None


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """获取性能监控器"""
    try:
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        # 在实际应用中，从请求中获取
        pass
    except:
        pass
    
    return None


# 系统状态API
@router.get("/health", tags=["status"])
async def health_check():
    """
    健康检查端点
    
    Returns:
        健康状态
    """
    logger.debug("健康检查请求")
    
    try:
        # 基本健康状态
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": get_app_config().version,
            "environment": get_app_config().environment
        }
        
        return status
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"健康检查失败: {str(e)}"
        )


@router.get("/ready", tags=["status"])
async def readiness_check():
    """
    就绪检查端点
    
    Returns:
        就绪状态
    """
    logger.debug("就绪检查请求")
    
    try:
        checks = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "status": "ready"
        }
        
        all_passed = True
        
        # 1. 检查数据库连接
        try:
            db_config = get_db_config()
            if db_config.connection.host:
                import mysql.connector
                from mysql.connector import Error
                
                conn = mysql.connector.connect(
                    host=db_config.connection.host,
                    port=db_config.connection.port,
                    user=db_config.connection.username,
                    password=db_config.connection.password,
                    database=db_config.connection.database
                )
                conn.ping(reconnect=True, attempts=1, delay=0)
                conn.close()
                checks["checks"]["database"] = {
                    "status": "healthy",
                    "message": "数据库连接正常"
                }
            else:
                checks["checks"]["database"] = {
                    "status": "healthy",
                    "message": "使用SQLite数据库"
                }
        except Exception as e:
            checks["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            all_passed = False
        
        # 2. 检查磁盘空间
        try:
            disk_usage = psutil.disk_usage("/")
            disk_percent = disk_usage.percent
            checks["checks"]["disk"] = {
                "status": "healthy" if disk_percent < 90 else "warning",
                "usage_percent": disk_percent,
                "free_gb": disk_usage.free / (1024**3),
                "total_gb": disk_usage.total / (1024**3)
            }
            if disk_percent >= 90:
                all_passed = False
        except Exception as e:
            checks["checks"]["disk"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            all_passed = False
        
        # 3. 检查内存
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            checks["checks"]["memory"] = {
                "status": "healthy" if memory_percent < 90 else "warning",
                "usage_percent": memory_percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3)
            }
            if memory_percent >= 90:
                all_passed = False
        except Exception as e:
            checks["checks"]["memory"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            all_passed = False
        
        # 4. 检查CPU
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            checks["checks"]["cpu"] = {
                "status": "healthy" if cpu_percent < 90 else "warning",
                "usage_percent": cpu_percent,
                "core_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            if cpu_percent >= 90:
                all_passed = False
        except Exception as e:
            checks["checks"]["cpu"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            all_passed = False
        
        checks["status"] = "ready" if all_passed else "not_ready"
        
        return checks
        
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"就绪检查失败: {str(e)}"
        )


@router.get("/status", tags=["status"])
async def get_system_status(
    detailed: bool = Query(False, description="是否详细"),
    current_user: dict = Depends(get_current_user)
):
    """
    获取系统状态
    
    Args:
        detailed: 是否详细
        current_user: 当前用户
        
    Returns:
        系统状态
    """
    logger.info(f"用户 {current_user['username']} 获取系统状态")
    
    try:
        status_info = {
            "timestamp": datetime.now().isoformat(),
            "version": get_app_config().version,
            "environment": get_app_config().environment,
            "hostname": socket.gethostname(),
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "machine": platform.machine()
            }
        }
        
        if detailed:
            # 添加详细系统信息
            status_info.update({
                "resources": _get_resource_usage(),
                "processes": _get_process_info(),
                "network": _get_network_info(),
                "disk": _get_disk_info()
            })
        
        return status_info
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )


@router.get("/status/components", tags=["status"])
async def get_component_status(
    component_name: Optional[str] = Query(None, description="组件名称"),
    current_user: dict = Depends(get_current_user)
):
    """
    获取组件状态
    
    Args:
        component_name: 组件名称
        current_user: 当前用户
        
    Returns:
        组件状态
    """
    logger.info(f"用户 {current_user['username']} 获取组件状态")
    
    try:
        system = get_system()
        if not system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="系统未初始化"
            )
        
        if component_name:
            # 获取单个组件状态
            component_status = system.get_component_status(component_name)
            if not component_status:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"组件 {component_name} 不存在"
                )
            
            return {
                "component": component_name,
                "status": component_status
            }
        else:
            # 获取所有组件状态
            all_status = system.get_all_component_status()
            
            return {
                "components": all_status,
                "total": len(all_status),
                "alive": sum(1 for status in all_status.values() if status.get("alive", False)),
                "failed": sum(1 for status in all_status.values() if not status.get("alive", False))
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取组件状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取组件状态失败: {str(e)}"
        )


@router.post("/status/components", tags=["status"])
async def update_component_status(
    request: SystemStatusRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    更新组件状态
    
    Args:
        request: 状态更新请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        更新结果
    """
    logger.info(f"用户 {current_user['username']} 更新组件状态: {request.component}")
    
    try:
        # 创建状态记录
        status_record = SystemStatus(
            component=request.component,
            status=request.status,
            message=request.message,
            cpu_usage=request.cpu_usage,
            memory_usage=request.memory_usage,
            disk_usage=request.disk_usage,
            network_rx=request.network_rx,
            network_tx=request.network_tx,
            timestamp=datetime.now()
        )
        
        # 保存到数据库
        db.add(status_record)
        db.commit()
        db.refresh(status_record)
        
        logger.info(f"组件 {request.component} 状态已更新: {request.status}")
        
        return {
            "message": "组件状态已更新",
            "component": request.component,
            "status": request.status,
            "record_id": status_record.id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"更新组件状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新组件状态失败: {str(e)}"
        )


@router.get("/status/history", tags=["status"])
async def get_status_history(
    component: Optional[str] = Query(None, description="组件名称"),
    hours: int = Query(24, ge=1, le=168, description="小时数"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取状态历史
    
    Args:
        component: 组件名称
        hours: 小时数
        limit: 限制数量
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        状态历史
    """
    logger.info(f"用户 {current_user['username']} 获取状态历史")
    
    try:
        # 计算时间范围
        start_time = datetime.now() - timedelta(hours=hours)
        
        # 构建查询
        query = db.query(SystemStatus).filter(
            SystemStatus.timestamp >= start_time
        ).order_by(SystemStatus.timestamp.desc())
        
        if component:
            query = query.filter(SystemStatus.component == component)
        
        # 获取数据
        records = query.limit(limit).all()
        
        # 转换为字典
        history = []
        for record in records:
            history.append({
                "id": record.id,
                "component": record.component,
                "status": record.status,
                "message": record.message,
                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                "cpu_usage": record.cpu_usage,
                "memory_usage": record.memory_usage,
                "disk_usage": record.disk_usage,
                "network_rx": record.network_rx,
                "network_tx": record.network_tx
            })
        
        return {
            "component": component or "all",
            "hours": hours,
            "total": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"获取状态历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取状态历史失败: {str(e)}"
        )


# 性能指标API
@router.get("/metrics", tags=["metrics"])
async def get_metrics():
    """
    获取Prometheus指标
    
    Returns:
        Prometheus指标
    """
    logger.debug("获取Prometheus指标")
    
    try:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取指标失败: {str(e)}"
        )


@router.get("/metrics/system", tags=["metrics"])
async def get_system_metrics(
    detailed: bool = Query(False, description="是否详细"),
    current_user: dict = Depends(get_current_user)
):
    """
    获取系统指标
    
    Args:
        detailed: 是否详细
        current_user: 当前用户
        
    Returns:
        系统指标
    """
    logger.info(f"用户 {current_user['username']} 获取系统指标")
    
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "resources": _get_resource_usage(),
            "system": _get_system_info()
        }
        
        if detailed:
            metrics.update({
                "processes": _get_process_info(),
                "network": _get_network_info(),
                "disk": _get_disk_info(),
                "memory": _get_memory_info(),
                "cpu": _get_cpu_info()
            })
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统指标失败: {str(e)}"
        )


@router.get("/metrics/performance", tags=["metrics"])
async def get_performance_metrics(
    duration: int = Query(60, ge=1, le=3600, description="持续时间(秒)"),
    interval: int = Query(1, ge=1, le=60, description="采样间隔(秒)"),
    metrics: List[str] = Query(["cpu", "memory", "disk"], description="指标列表"),
    current_user: dict = Depends(get_current_user)
):
    """
    获取性能指标
    
    Args:
        duration: 持续时间
        interval: 采样间隔
        metrics: 指标列表
        current_user: 当前用户
        
    Returns:
        性能指标
    """
    logger.info(f"用户 {current_user['username']} 获取性能指标")
    
    try:
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="性能监控未启用"
            )
        
        # 获取历史数据
        history_data = performance_monitor.get_history(duration)
        
        # 过滤指标
        filtered_data = {}
        for metric_name in metrics:
            if metric_name in history_data:
                filtered_data[metric_name] = history_data[metric_name]
        
        return {
            "duration": duration,
            "interval": interval,
            "metrics": metrics,
            "data": filtered_data,
            "sample_count": len(history_data.get("cpu", [])) if history_data.get("cpu") else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )


@router.get("/metrics/real-time", tags=["metrics"])
async def get_real_time_metrics(
    metrics: List[str] = Query(["cpu", "memory", "disk"], description="指标列表"),
    current_user: dict = Depends(get_current_user)
):
    """
    获取实时指标
    
    Args:
        metrics: 指标列表
        current_user: 当前用户
        
    Returns:
        实时指标
    """
    logger.info(f"用户 {current_user['username']} 获取实时指标")
    
    try:
        real_time_data = {
            "timestamp": datetime.now().isoformat()
        }
        
        for metric in metrics:
            if metric == "cpu":
                real_time_data["cpu"] = {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "percent_per_core": psutil.cpu_percent(interval=0.1, percpu=True),
                    "count": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None
                }
            elif metric == "memory":
                memory = psutil.virtual_memory()
                real_time_data["memory"] = {
                    "percent": memory.percent,
                    "available": memory.available,
                    "total": memory.total,
                    "used": memory.used,
                    "free": memory.free
                }
            elif metric == "disk":
                disk = psutil.disk_usage("/")
                real_time_data["disk"] = {
                    "percent": disk.percent,
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free
                }
            elif metric == "network":
                net_io = psutil.net_io_counters()
                real_time_data["network"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            elif metric == "process":
                process = psutil.Process()
                real_time_data["process"] = {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "memory_percent": process.memory_percent(),
                    "memory_info": process.memory_info()._asdict() if hasattr(process.memory_info(), '_asdict') else {}
                }
        
        return real_time_data
        
    except Exception as e:
        logger.error(f"获取实时指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实时指标失败: {str(e)}"
        )


@router.get("/metrics/database", tags=["metrics"])
async def get_database_metrics(
    detailed: bool = Query(False, description="是否详细"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取数据库指标
    
    Args:
        detailed: 是否详细
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        数据库指标
    """
    logger.info(f"用户 {current_user['username']} 获取数据库指标")
    
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "connection": {
                "alive": True
            }
        }
        
        if detailed:
            # 获取表统计
            from sqlalchemy import inspect, func, text
            
            inspector = inspect(db.get_bind())
            tables = inspector.get_table_names()
            
            table_stats = {}
            for table in tables:
                try:
                    # 获取行数
                    row_count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    table_stats[table] = {
                        "row_count": row_count
                    }
                    
                    # 获取表大小（MySQL特定）
                    if get_db_config().db_type == "mysql":
                        size_query = text(f"""
                        SELECT 
                            data_length as data_size,
                            index_length as index_size
                        FROM information_schema.TABLES 
                        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name
                        """)
                        result = db.execute(size_query, {"table_name": table}).fetchone()
                        if result:
                            table_stats[table]["data_size"] = result[0] or 0
                            table_stats[table]["index_size"] = result[1] or 0
                            table_stats[table]["total_size"] = (result[0] or 0) + (result[1] or 0)
                
                except Exception as e:
                    table_stats[table] = {
                        "error": str(e)
                    }
            
            metrics["tables"] = table_stats
            metrics["table_count"] = len(tables)
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取数据库指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据库指标失败: {str(e)}"
        )


# 系统信息API
@router.get("/info", tags=["info"])
async def get_system_info(
    current_user: dict = Depends(get_current_user)
):
    """
    获取系统信息
    
    Args:
        current_user: 当前用户
        
    Returns:
        系统信息
    """
    logger.info(f"用户 {current_user['username']} 获取系统信息")
    
    try:
        info = {
            "system": _get_system_info(),
            "python": _get_python_info(),
            "hardware": _get_hardware_info(),
            "network": _get_network_info(),
            "paths": _get_paths_info(),
            "config": _get_config_info(),
            "timestamp": datetime.now().isoformat()
        }
        
        return info
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统信息失败: {str(e)}"
        )


@router.get("/info/version", tags=["info"])
async def get_version():
    """
    获取版本信息
    
    Returns:
        版本信息
    """
    try:
        app_config = get_app_config()
        
        return {
            "name": "ADRV9009 Signal Processing System",
            "version": app_config.version,
            "build": app_config.build,
            "environment": app_config.environment,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取版本信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取版本信息失败: {str(e)}"
        )


@router.get("/info/dependencies", tags=["info"])
async def get_dependencies(
    current_user: dict = Depends(get_current_user)
):
    """
    获取依赖信息
    
    Args:
        current_user: 当前用户
        
    Returns:
        依赖信息
    """
    logger.info(f"用户 {current_user['username']} 获取依赖信息")
    
    try:
        dependencies = []
        
        # 主要依赖
        main_deps = [
            "fastapi", "uvicorn", "sqlalchemy", "pydantic",
            "numpy", "scipy", "pandas", "matplotlib",
            "psutil", "prometheus-client", "aiofiles"
        ]
        
        for dep in main_deps:
            try:
                module = __import__(dep)
                dependencies.append({
                    "name": dep,
                    "version": getattr(module, "__version__", "unknown"),
                    "path": getattr(module, "__file__", "unknown")
                })
            except ImportError:
                dependencies.append({
                    "name": dep,
                    "version": "not installed",
                    "path": None
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "dependencies": dependencies
        }
        
    except Exception as e:
        logger.error(f"获取依赖信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取依赖信息失败: {str(e)}"
        )


@router.get("/info/config", tags=["info"])
async def get_configuration(
    include_secrets: bool = Query(False, description="是否包含密钥"),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    获取配置信息
    
    Args:
        include_secrets: 是否包含密钥
        current_user: 当前用户
        
    Returns:
        配置信息
    """
    logger.info(f"管理员 {current_user['username']} 获取配置信息")
    
    try:
        config_info = {
            "app": _get_app_config_safe(include_secrets),
            "web": _get_web_config_safe(include_secrets),
            "database": _get_db_config_safe(include_secrets),
            "signal": _get_signal_config_safe(include_secrets),
            "logging": _get_logging_config_safe(include_secrets)
        }
        
        return config_info
        
    except Exception as e:
        logger.error(f"获取配置信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取配置信息失败: {str(e)}"
        )


# 诊断API
@router.post("/diagnose", tags=["diagnose"])
async def diagnose_system(
    request: SystemDiagnosticRequest = Body(...),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    系统诊断
    
    Args:
        request: 诊断请求
        current_user: 当前用户
        
    Returns:
        诊断结果
    """
    logger.info(f"管理员 {current_user['username']} 执行系统诊断")
    
    try:
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "diagnostic_id": f"diagnosis_{int(time.time())}"
        }
        
        # 1. 系统信息
        diagnosis["system_info"] = _get_system_info()
        
        # 2. 资源使用
        diagnosis["resource_usage"] = _get_resource_usage()
        
        # 3. 健康检查
        diagnosis["health_check"] = await readiness_check()
        
        # 4. 组件状态
        system = get_system()
        if system:
            diagnosis["component_status"] = system.get_all_component_status()
        
        # 5. 日志（如果启用）
        if request.include_logs:
            diagnosis["logs"] = await _get_recent_logs()
        
        # 6. 配置（如果启用）
        if request.include_configs:
            diagnosis["configs"] = await get_configuration(include_secrets=False)
        
        # 7. 指标（如果启用）
        if request.include_metrics:
            diagnosis["metrics"] = await get_system_metrics(detailed=True)
        
        # 8. 数据库（如果启用）
        if request.include_database:
            try:
                from src.core.database_manager import DatabaseManager
                diagnosis["database"] = await get_database_metrics(detailed=True)
            except Exception as e:
                diagnosis["database_error"] = str(e)
        
        # 创建诊断文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(diagnosis, f, indent=2, default=str)
            temp_file = f.name
        
        # 压缩文件
        if request.compress:
            import gzip
            compressed_file = temp_file + ".gz"
            with open(temp_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.unlink(temp_file)
            export_file = compressed_file
            content_type = "application/gzip"
            filename = f"diagnosis_{diagnosis['diagnostic_id']}.json.gz"
        else:
            export_file = temp_file
            content_type = "application/json"
            filename = f"diagnosis_{diagnosis['diagnostic_id']}.json"
        
        # 返回文件
        return FileResponse(
            path=export_file,
            filename=filename,
            media_type=content_type,
            headers={
                "X-Diagnostic-ID": diagnosis["diagnostic_id"],
                "X-Diagnostic-Time": diagnosis["timestamp"]
            }
        )
        
    except Exception as e:
        logger.error(f"系统诊断失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"系统诊断失败: {str(e)}"
        )


@router.get("/diagnose/logs", tags=["diagnose"])
async def get_system_logs(
    lines: int = Query(100, ge=1, le=10000, description="行数"),
    level: str = Query("INFO", description="日志级别"),
    search: Optional[str] = Query(None, description="搜索内容"),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    获取系统日志
    
    Args:
        lines: 行数
        level: 日志级别
        search: 搜索内容
        current_user: 当前用户
        
    Returns:
        系统日志
    """
    logger.info(f"管理员 {current_user['username']} 获取系统日志")
    
    try:
        return await _get_recent_logs(lines, level, search)
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统日志失败: {str(e)}"
        )


# 实用函数
def _get_resource_usage() -> Dict[str, Any]:
    """获取资源使用情况"""
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        
        # 进程信息
        process = psutil.Process()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "network_rx_bytes": net_io.bytes_recv,
            "network_tx_bytes": net_io.bytes_sent,
            "process_count": len(psutil.pids()),
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
        }
    except Exception as e:
        logger.warning(f"获取资源使用失败: {e}")
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "network_rx_bytes": 0,
            "network_tx_bytes": 0,
            "process_count": 0,
            "thread_count": 0,
            "open_files": 0
        }


def _get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    try:
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat() if hasattr(psutil, 'boot_time') else None
        }
    except Exception as e:
        logger.warning(f"获取系统信息失败: {e}")
        return {}


def _get_python_info() -> Dict[str, Any]:
    """获取Python信息"""
    try:
        return {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": platform.python_build(),
            "executable": sys.executable,
            "path": sys.path
        }
    except Exception as e:
        logger.warning(f"获取Python信息失败: {e}")
        return {}


def _get_hardware_info() -> Dict[str, Any]:
    """获取硬件信息"""
    try:
        info = {
            "cpu": {
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if hasattr(psutil.cpu_freq(), '_asdict') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "free": psutil.disk_usage("/").free
            }
        }
        
        # 添加交换内存信息
        if hasattr(psutil, 'swap_memory'):
            swap = psutil.swap_memory()
            info["swap"] = {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free
            }
        
        return info
    except Exception as e:
        logger.warning(f"获取硬件信息失败: {e}")
        return {}


def _get_network_info() -> Dict[str, Any]:
    """获取网络信息"""
    try:
        info = {
            "hostname": socket.gethostname(),
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "interfaces": []
        }
        
        # 网络接口
        net_if_addrs = psutil.net_if_addrs()
        for interface, addresses in net_if_addrs.items():
            interface_info = {
                "name": interface,
                "addresses": []
            }
            for addr in addresses:
                interface_info["addresses"].append({
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast
                })
            info["interfaces"].append(interface_info)
        
        return info
    except Exception as e:
        logger.warning(f"获取网络信息失败: {e}")
        return {}


def _get_paths_info() -> Dict[str, str]:
    """获取路径信息"""
    try:
        app_config = get_app_config()
        return {
            "current_dir": os.getcwd(),
            "script_dir": os.path.dirname(os.path.abspath(__file__)),
            "data_dir": str(app_config.data_dir),
            "log_dir": str(app_config.log_dir),
            "temp_dir": str(app_config.temp_dir),
            "backup_dir": str(app_config.backup_dir),
            "upload_dir": str(app_config.upload_dir)
        }
    except Exception as e:
        logger.warning(f"获取路径信息失败: {e}")
        return {}


def _get_config_info() -> Dict[str, Any]:
    """获取配置信息"""
    try:
        return {
            "environment": get_app_config().environment,
            "debug": get_app_config().debug,
            "strict_mode": get_app_config().strict_mode
        }
    except Exception as e:
        logger.warning(f"获取配置信息失败: {e}")
        return {}


async def _get_recent_logs(
    lines: int = 100,
    level: str = "INFO",
    search: str = None
) -> Dict[str, Any]:
    """获取最近的日志"""
    try:
        log_config = get_logging_config()
        log_file = Path(log_config.log_dir) / "web.log"
        
        if not log_file.exists():
            raise FileNotFoundError(f"日志文件不存在: {log_file}")
        
        # 读取日志文件
        async with aiofiles.open(log_file, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        log_lines = content.split('\n')
        
        # 过滤日志
        filtered_lines = []
        for line in reversed(log_lines):  # 从最新开始
            if not line.strip():
                continue
            
            # 级别过滤
            if level.upper() not in line:
                continue
            
            # 搜索过滤
            if search and search.lower() not in line.lower():
                continue
            
            filtered_lines.append(line)
            if len(filtered_lines) >= lines:
                break
        
        return {
            "log_file": str(log_file),
            "total_lines": len(log_lines),
            "filtered_lines": len(filtered_lines),
            "level": level,
            "search": search,
            "logs": list(reversed(filtered_lines))  # 恢复时间顺序
        }
        
    except Exception as e:
        logger.error(f"读取日志失败: {e}")
        raise


def _get_app_config_safe(include_secrets: bool = False) -> Dict[str, Any]:
    """安全获取应用配置"""
    try:
        config = get_app_config()
        config_dict = asdict(config)
        
        if not include_secrets:
            # 移除敏感信息
            secrets = ['secret_key', 'password', 'token', 'key', 'credential']
            for secret in secrets:
                if secret in config_dict:
                    config_dict[secret] = "***REDACTED***"
        
        return config_dict
    except Exception as e:
        logger.warning(f"获取应用配置失败: {e}")
        return {}


def _get_web_config_safe(include_secrets: bool = False) -> Dict[str, Any]:
    """安全获取Web配置"""
    try:
        config = get_web_config()
        config_dict = asdict(config)
        
        if not include_secrets:
            # 移除敏感信息
            secrets = ['secret_key', 'password', 'token', 'key', 'credential']
            for secret in secrets:
                if secret in config_dict:
                    config_dict[secret] = "***REDACTED***"
        
        return config_dict
    except Exception as e:
        logger.warning(f"获取Web配置失败: {e}")
        return {}


def _get_db_config_safe(include_secrets: bool = False) -> Dict[str, Any]:
    """安全获取数据库配置"""
    try:
        config = get_db_config()
        config_dict = asdict(config)
        
        if not include_secrets:
            # 移除密码
            if 'connection' in config_dict and 'password' in config_dict['connection']:
                config_dict['connection']['password'] = "***REDACTED***"
        
        return config_dict
    except Exception as e:
        logger.warning(f"获取数据库配置失败: {e}")
        return {}


def _get_signal_config_safe(include_secrets: bool = False) -> Dict[str, Any]:
    """安全获取信号配置"""
    try:
        config = get_signal_config()
        config_dict = asdict(config)
        return config_dict
    except Exception as e:
        logger.warning(f"获取信号配置失败: {e}")
        return {}


def _get_logging_config_safe(include_secrets: bool = False) -> Dict[str, Any]:
    """安全获取日志配置"""
    try:
        config = get_logging_config()
        config_dict = asdict(config)
        return config_dict
    except Exception as e:
        logger.warning(f"获取日志配置失败: {e}")
        return {}


def _get_process_info() -> Dict[str, Any]:
    """获取进程信息"""
    try:
        process = psutil.Process()
        return {
            "pid": process.pid,
            "name": process.name(),
            "status": process.status(),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat() if hasattr(process, 'create_time') else None,
            "cpu_times": process.cpu_times()._asdict() if hasattr(process.cpu_times(), '_asdict') else {},
            "memory_info": process.memory_info()._asdict() if hasattr(process.memory_info(), '_asdict') else {},
            "open_files": [f.path for f in process.open_files()] if hasattr(process, 'open_files') else [],
            "connections": [c._asdict() for c in process.connections()] if hasattr(process, 'connections') else []
        }
    except Exception as e:
        logger.warning(f"获取进程信息失败: {e}")
        return {}


def _get_disk_info() -> Dict[str, Any]:
    """获取磁盘信息"""
    try:
        partitions = psutil.disk_partitions()
        disk_info = []
        
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                })
            except Exception as e:
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "error": str(e)
                })
        
        return {"partitions": disk_info}
    except Exception as e:
        logger.warning(f"获取磁盘信息失败: {e}")
        return {}


def _get_memory_info() -> Dict[str, Any]:
    """获取内存信息"""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory() if hasattr(psutil, 'swap_memory') else None
        
        info = {
            "virtual": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
                "active": memory.active if hasattr(memory, 'active') else None,
                "inactive": memory.inactive if hasattr(memory, 'inactive') else None
            }
        }
        
        if swap:
            info["swap"] = {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent
            }
        
        return info
    except Exception as e:
        logger.warning(f"获取内存信息失败: {e}")
        return {}


def _get_cpu_info() -> Dict[str, Any]:
    """获取CPU信息"""
    try:
        info = {
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "times": psutil.cpu_times()._asdict() if hasattr(psutil.cpu_times(), '_asdict') else {},
            "stats": psutil.cpu_stats()._asdict() if hasattr(psutil.cpu_stats(), '_asdict') else {},
            "freq": psutil.cpu_freq()._asdict() if hasattr(psutil.cpu_freq(), '_asdict') else {}
        }
        return info
    except Exception as e:
        logger.warning(f"获取CPU信息失败: {e}")
        return {}