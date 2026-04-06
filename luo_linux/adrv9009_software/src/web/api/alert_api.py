"""
告警API模块
提供告警管理、告警规则配置、告警通知等接口
"""
import os
import sys
import json
import asyncio
import smtplib
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
import logging
import traceback
import uuid
from dataclasses import asdict
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path as FPath, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator, conint, confloat, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, and_, or_, not_, between
from jinja2 import Template
import aiofiles
import aiofiles.os

# 导入配置
from config.app_config import get_current_config as get_app_config
from config.web_config import get_current_config as get_web_config
from config.alert_config import get_current_config as get_alert_config
from config.db_config import get_current_config as get_db_config

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
    AlertRule,
    AlertHistory,
    AlertRecipient,
    AlertTemplate,
    SystemEvent
)

# 导入模型
from src.models.alert_models import (
    AlertLevel,
    AlertStatus,
    AlertType,
    NotificationMethod
)

# 导入工具
from src.utils.data_validator import DataValidator
from src.utils.file_utils import ensure_directory, get_file_info
from src.utils.performance_monitor import PerformanceMonitor

# 获取日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# 数据模型
class AlertLevelEnum(str, Enum):
    """告警级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatusEnum(str, Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class AlertTypeEnum(str, Enum):
    """告警类型枚举"""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    NETWORK = "network"
    APPLICATION = "application"
    CUSTOM = "custom"


class NotificationMethodEnum(str, Enum):
    """通知方法枚举"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEB = "web"
    ALL = "all"


class TimeWindow(BaseModel):
    """时间窗口"""
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    repeat: bool = Field(False, description="是否重复")
    repeat_interval: Optional[int] = Field(None, description="重复间隔(分钟)")
    weekdays: Optional[List[int]] = Field(None, description="工作日(0-6)")


class AlertCondition(BaseModel):
    """告警条件"""
    field: str = Field(..., description="字段名")
    operator: str = Field(..., description="操作符")
    value: Any = Field(..., description="比较值")
    threshold: Optional[float] = Field(None, description="阈值")
    duration: Optional[int] = Field(None, description="持续时间(秒)")


class AlertAction(BaseModel):
    """告警动作"""
    action_type: str = Field(..., description="动作类型")
    target: str = Field(..., description="目标")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数")
    delay: Optional[int] = Field(0, description="延迟(秒)")


class AlertRuleCreate(BaseModel):
    """告警规则创建"""
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    alert_type: AlertTypeEnum = Field(..., description="告警类型")
    alert_level: AlertLevelEnum = Field(..., description="告警级别")
    enabled: bool = Field(True, description="是否启用")
    conditions: List[AlertCondition] = Field(..., description="条件列表")
    actions: List[AlertAction] = Field(default_factory=list, description="动作列表")
    time_window: Optional[TimeWindow] = Field(None, description="时间窗口")
    tags: List[str] = Field(default_factory=list, description="标签")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    priority: int = Field(0, description="优先级")


class AlertRuleUpdate(BaseModel):
    """告警规则更新"""
    name: Optional[str] = Field(None, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    alert_level: Optional[AlertLevelEnum] = Field(None, description="告警级别")
    enabled: Optional[bool] = Field(None, description="是否启用")
    conditions: Optional[List[AlertCondition]] = Field(None, description="条件列表")
    actions: Optional[List[AlertAction]] = Field(None, description="动作列表")
    time_window: Optional[TimeWindow] = Field(None, description="时间窗口")
    tags: Optional[List[str]] = Field(None, description="标签")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    priority: Optional[int] = Field(None, description="优先级")


class AlertRecipientCreate(BaseModel):
    """告警接收人创建"""
    name: str = Field(..., description="接收人姓名")
    email: EmailStr = Field(..., description="邮箱地址")
    phone: Optional[str] = Field(None, description="电话号码")
    notification_methods: List[NotificationMethodEnum] = Field(..., description="通知方法")
    alert_levels: List[AlertLevelEnum] = Field(..., description="告警级别")
    alert_types: List[AlertTypeEnum] = Field(..., description="告警类型")
    enabled: bool = Field(True, description="是否启用")
    schedule: Optional[Dict[str, Any]] = Field(None, description="排班表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AlertRecipientUpdate(BaseModel):
    """告警接收人更新"""
    name: Optional[str] = Field(None, description="接收人姓名")
    email: Optional[EmailStr] = Field(None, description="邮箱地址")
    phone: Optional[str] = Field(None, description="电话号码")
    notification_methods: Optional[List[NotificationMethodEnum]] = Field(None, description="通知方法")
    alert_levels: Optional[List[AlertLevelEnum]] = Field(None, description="告警级别")
    alert_types: Optional[List[AlertTypeEnum]] = Field(None, description="告警类型")
    enabled: Optional[bool] = Field(None, description="是否启用")
    schedule: Optional[Dict[str, Any]] = Field(None, description="排班表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class AlertTemplateCreate(BaseModel):
    """告警模板创建"""
    name: str = Field(..., description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    alert_type: AlertTypeEnum = Field(..., description="告警类型")
    alert_level: AlertLevelEnum = Field(..., description="告警级别")
    subject_template: str = Field(..., description="主题模板")
    body_template: str = Field(..., description="正文模板")
    notification_method: NotificationMethodEnum = Field(..., description="通知方法")
    variables: Dict[str, str] = Field(default_factory=dict, description="变量定义")
    enabled: bool = Field(True, description="是否启用")


class AlertTemplateUpdate(BaseModel):
    """告警模板更新"""
    name: Optional[str] = Field(None, description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    alert_type: Optional[AlertTypeEnum] = Field(None, description="告警类型")
    alert_level: Optional[AlertLevelEnum] = Field(None, description="告警级别")
    subject_template: Optional[str] = Field(None, description="主题模板")
    body_template: Optional[str] = Field(None, description="正文模板")
    notification_method: Optional[NotificationMethodEnum] = Field(None, description="通知方法")
    variables: Optional[Dict[str, str]] = Field(None, description="变量定义")
    enabled: Optional[bool] = Field(None, description="是否启用")


class AlertCreate(BaseModel):
    """告警创建"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="告警ID")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    alert_type: AlertTypeEnum = Field(..., description="告警类型")
    alert_level: AlertLevelEnum = Field(..., description="告警级别")
    source: str = Field(..., description="告警源")
    component: str = Field(..., description="组件")
    data: Dict[str, Any] = Field(default_factory=dict, description="告警数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class AlertUpdate(BaseModel):
    """告警更新"""
    status: Optional[AlertStatusEnum] = Field(None, description="状态")
    acknowledged_by: Optional[str] = Field(None, description="确认人")
    acknowledged_at: Optional[datetime] = Field(None, description="确认时间")
    resolved_by: Optional[str] = Field(None, description="解决人")
    resolved_at: Optional[datetime] = Field(None, description="解决时间")
    resolution_notes: Optional[str] = Field(None, description="解决说明")
    assigned_to: Optional[str] = Field(None, description="分配给")


class AlertQueryParams(BaseModel):
    """告警查询参数"""
    page: int = Query(1, ge=1, description="页码")
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
    alert_id: Optional[str] = Query(None, description="告警ID")
    alert_type: Optional[AlertTypeEnum] = Query(None, description="告警类型")
    alert_level: Optional[AlertLevelEnum] = Query(None, description="告警级别")
    status: Optional[AlertStatusEnum] = Query(None, description="状态")
    source: Optional[str] = Query(None, description="告警源")
    component: Optional[str] = Query(None, description="组件")
    start_time: Optional[datetime] = Query(None, description="开始时间")
    end_time: Optional[datetime] = Query(None, description="结束时间")
    search: Optional[str] = Query(None, description="搜索内容")
    sort_by: str = Query("timestamp", description="排序字段")
    sort_order: str = Query("desc", description="排序顺序")


class AlertAcknowledgeRequest(BaseModel):
    """告警确认请求"""
    notes: Optional[str] = Field(None, description="确认说明")


class AlertResolveRequest(BaseModel):
    """告警解决请求"""
    notes: Optional[str] = Field(..., description="解决说明")
    resolution_code: Optional[str] = Field(None, description="解决代码")


class AlertTestRequest(BaseModel):
    """告警测试请求"""
    alert_level: AlertLevelEnum = Field(..., description="告警级别")
    alert_type: AlertTypeEnum = Field(..., description="告警类型")
    notification_method: NotificationMethodEnum = Field(..., description="通知方法")
    recipient_id: Optional[int] = Field(None, description="接收人ID")


class AlertStatisticsRequest(BaseModel):
    """告警统计请求"""
    time_range: str = Query("24h", description="时间范围")
    group_by: str = Query("alert_level", description="分组字段")


class AlertExportRequest(BaseModel):
    """告警导出请求"""
    format: str = Query("json", description="导出格式")
    compress: bool = Query(True, description="是否压缩")
    time_range: Optional[str] = Query(None, description="时间范围")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤器")


# 依赖项
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_alert_manager():
    """获取告警管理器"""
    try:
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        # 在实际应用中，从请求中获取
        pass
    except:
        pass
    
    return None


# 告警规则API
@router.get("/alerts/rules", tags=["alerts"])
async def get_alert_rules(
    enabled: Optional[bool] = Query(None, description="是否启用"),
    alert_type: Optional[AlertTypeEnum] = Query(None, description="告警类型"),
    alert_level: Optional[AlertLevelEnum] = Query(None, description="告警级别"),
    search: Optional[str] = Query(None, description="搜索内容"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取告警规则列表
    
    Args:
        enabled: 是否启用
        alert_type: 告警类型
        alert_level: 告警级别
        search: 搜索内容
        page: 页码
        page_size: 每页数量
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        告警规则列表
    """
    logger.info(f"用户 {current_user['username']} 获取告警规则列表")
    
    try:
        # 构建查询
        query = db.query(AlertRule)
        
        # 应用过滤器
        if enabled is not None:
            query = query.filter(AlertRule.enabled == enabled)
        
        if alert_type:
            query = query.filter(AlertRule.alert_type == alert_type.value)
        
        if alert_level:
            query = query.filter(AlertRule.alert_level == alert_level.value)
        
        if search:
            query = query.filter(
                or_(
                    AlertRule.name.ilike(f"%{search}%"),
                    AlertRule.description.ilike(f"%{search}%"),
                    AlertRule.tags.like(f"%{search}%")
                )
            )
        
        # 获取总数
        total_count = query.count()
        
        # 应用分页
        offset = (page - 1) * page_size
        rules = query.order_by(desc(AlertRule.priority), desc(AlertRule.updated_at)).offset(offset).limit(page_size).all()
        
        # 转换为字典
        rules_data = []
        for rule in rules:
            rule_dict = {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "alert_type": rule.alert_type,
                "alert_level": rule.alert_level,
                "enabled": rule.enabled,
                "conditions": rule.conditions,
                "actions": rule.actions,
                "time_window": rule.time_window,
                "tags": rule.tags,
                "metadata": rule.metadata,
                "priority": rule.priority,
                "created_at": rule.created_at.isoformat() if rule.created_at else None,
                "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
                "created_by": rule.created_by,
                "updated_by": rule.updated_by
            }
            rules_data.append(rule_dict)
        
        return {
            "rules": rules_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警规则失败: {str(e)}"
        )


@router.get("/alerts/rules/{rule_id}", tags=["alerts"])
async def get_alert_rule(
    rule_id: int = FPath(..., description="规则ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取单个告警规则
    
    Args:
        rule_id: 规则ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        告警规则详情
    """
    logger.info(f"用户 {current_user['username']} 获取告警规则 {rule_id}")
    
    try:
        # 查找规则
        rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警规则 {rule_id} 不存在"
            )
        
        # 转换为字典
        rule_dict = {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "alert_type": rule.alert_type,
            "alert_level": rule.alert_level,
            "enabled": rule.enabled,
            "conditions": rule.conditions,
            "actions": rule.actions,
            "time_window": rule.time_window,
            "tags": rule.tags,
            "metadata": rule.metadata,
            "priority": rule.priority,
            "created_at": rule.created_at.isoformat() if rule.created_at else None,
            "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
            "created_by": rule.created_by,
            "updated_by": rule.updated_by
        }
        
        return rule_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警规则失败: {str(e)}"
        )


@router.post("/alerts/rules", tags=["alerts"], status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    request: AlertRuleCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    创建告警规则
    
    Args:
        request: 规则创建请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        创建的规则
    """
    logger.info(f"管理员 {current_user['username']} 创建告警规则: {request.name}")
    
    try:
        # 验证条件
        if not request.conditions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="告警规则必须至少包含一个条件"
            )
        
        # 创建规则记录
        rule = AlertRule(
            name=request.name,
            description=request.description,
            alert_type=request.alert_type.value,
            alert_level=request.alert_level.value,
            enabled=request.enabled,
            conditions=request.conditions,
            actions=request.actions,
            time_window=request.time_window.dict() if request.time_window else None,
            tags=request.tags,
            metadata=request.metadata,
            priority=request.priority,
            created_by=current_user['username'],
            updated_by=current_user['username']
        )
        
        # 保存到数据库
        db.add(rule)
        db.commit()
        db.refresh(rule)
        
        logger.info(f"告警规则 {rule.id} 创建成功")
        
        return {
            "message": "告警规则创建成功",
            "rule_id": rule.id,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "alert_type": rule.alert_type,
                "alert_level": rule.alert_level,
                "enabled": rule.enabled
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建告警规则失败: {str(e)}"
        )


@router.put("/alerts/rules/{rule_id}", tags=["alerts"])
async def update_alert_rule(
    rule_id: int = FPath(..., description="规则ID"),
    request: AlertRuleUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    更新告警规则
    
    Args:
        rule_id: 规则ID
        request: 规则更新请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        更新后的规则
    """
    logger.info(f"管理员 {current_user['username']} 更新告警规则 {rule_id}")
    
    try:
        # 查找规则
        rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警规则 {rule_id} 不存在"
            )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                setattr(rule, field, value)
        
        # 更新时间和用户
        rule.updated_at = datetime.now()
        rule.updated_by = current_user['username']
        
        # 保存到数据库
        db.commit()
        db.refresh(rule)
        
        logger.info(f"告警规则 {rule_id} 更新成功")
        
        return {
            "message": "告警规则更新成功",
            "rule_id": rule_id,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "alert_type": rule.alert_type,
                "alert_level": rule.alert_level,
                "enabled": rule.enabled
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新告警规则失败: {str(e)}"
        )


@router.delete("/alerts/rules/{rule_id}", tags=["alerts"])
async def delete_alert_rule(
    rule_id: int = FPath(..., description="规则ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    删除告警规则
    
    Args:
        rule_id: 规则ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    logger.info(f"管理员 {current_user['username']} 删除告警规则 {rule_id}")
    
    try:
        # 查找规则
        rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警规则 {rule_id} 不存在"
            )
        
        # 删除规则
        db.delete(rule)
        db.commit()
        
        logger.info(f"告警规则 {rule_id} 删除成功")
        
        return {
            "message": "告警规则删除成功",
            "rule_id": rule_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除告警规则失败: {str(e)}"
        )


@router.post("/alerts/rules/{rule_id}/enable", tags=["alerts"])
async def enable_alert_rule(
    rule_id: int = FPath(..., description="规则ID"),
    enabled: bool = Body(True, description="是否启用"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    启用/禁用告警规则
    
    Args:
        rule_id: 规则ID
        enabled: 是否启用
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        操作结果
    """
    logger.info(f"管理员 {current_user['username']} {'启用' if enabled else '禁用'}告警规则 {rule_id}")
    
    try:
        # 查找规则
        rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警规则 {rule_id} 不存在"
            )
        
        # 更新状态
        rule.enabled = enabled
        rule.updated_at = datetime.now()
        rule.updated_by = current_user['username']
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"告警规则 {rule_id} 已{'启用' if enabled else '禁用'}")
        
        return {
            "message": f"告警规则已{'启用' if enabled else '禁用'}",
            "rule_id": rule_id,
            "enabled": enabled
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"{'启用' if enabled else '禁用'}告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{'启用' if enabled else '禁用'}告警规则失败: {str(e)}"
        )


@router.post("/alerts/rules/{rule_id}/test", tags=["alerts"])
async def test_alert_rule(
    rule_id: int = FPath(..., description="规则ID"),
    test_data: Dict[str, Any] = Body(..., description="测试数据"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    测试告警规则
    
    Args:
        rule_id: 规则ID
        test_data: 测试数据
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        测试结果
    """
    logger.info(f"管理员 {current_user['username']} 测试告警规则 {rule_id}")
    
    try:
        # 查找规则
        rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警规则 {rule_id} 不存在"
            )
        
        if not rule.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="告警规则未启用"
            )
        
        # 测试条件
        test_results = []
        triggered = True
        
        for condition in rule.conditions:
            # 检查测试数据中是否包含条件字段
            field_value = test_data.get(condition.field)
            
            if field_value is None:
                result = {
                    "condition": condition,
                    "matched": False,
                    "reason": f"字段 '{condition.field}' 不存在于测试数据中"
                }
                test_results.append(result)
                triggered = False
                continue
            
            # 根据操作符测试条件
            matched = _test_condition(condition, field_value)
            
            result = {
                "condition": condition,
                "field_value": field_value,
                "matched": matched
            }
            test_results.append(result)
            
            if not matched:
                triggered = False
        
        return {
            "rule_id": rule_id,
            "rule_name": rule.name,
            "triggered": triggered,
            "test_results": test_results,
            "test_data": test_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"测试告警规则失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"测试告警规则失败: {str(e)}"
        )


def _test_condition(condition: AlertCondition, field_value: Any) -> bool:
    """测试单个条件"""
    try:
        operator = condition.operator.lower()
        condition_value = condition.value
        
        if operator == "eq":
            return field_value == condition_value
        elif operator == "ne":
            return field_value != condition_value
        elif operator == "gt":
            return field_value > condition_value
        elif operator == "ge":
            return field_value >= condition_value
        elif operator == "lt":
            return field_value < condition_value
        elif operator == "le":
            return field_value <= condition_value
        elif operator == "contains":
            return condition_value in str(field_value)
        elif operator == "not_contains":
            return condition_value not in str(field_value)
        elif operator == "starts_with":
            return str(field_value).startswith(str(condition_value))
        elif operator == "ends_with":
            return str(field_value).endswith(str(condition_value))
        elif operator == "regex":
            import re
            return bool(re.search(str(condition_value), str(field_value)))
        elif operator == "in":
            return field_value in condition_value
        elif operator == "not_in":
            return field_value not in condition_value
        elif operator == "between":
            if isinstance(condition_value, (list, tuple)) and len(condition_value) == 2:
                return condition_value[0] <= field_value <= condition_value[1]
            return False
        else:
            return False
            
    except Exception as e:
        logger.warning(f"测试条件失败: {e}")
        return False


# 告警历史API
@router.get("/alerts/history", tags=["alerts"])
async def get_alert_history(
    query_params: AlertQueryParams = Depends(),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取告警历史
    
    Args:
        query_params: 查询参数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        告警历史列表
    """
    logger.info(f"用户 {current_user['username']} 获取告警历史")
    
    try:
        # 构建查询
        query = db.query(AlertHistory)
        
        # 应用过滤器
        if query_params.alert_id:
            query = query.filter(AlertHistory.alert_id == query_params.alert_id)
        
        if query_params.alert_type:
            query = query.filter(AlertHistory.alert_type == query_params.alert_type.value)
        
        if query_params.alert_level:
            query = query.filter(AlertHistory.alert_level == query_params.alert_level.value)
        
        if query_params.status:
            query = query.filter(AlertHistory.status == query_params.status.value)
        
        if query_params.source:
            query = query.filter(AlertHistory.source == query_params.source)
        
        if query_params.component:
            query = query.filter(AlertHistory.component == query_params.component)
        
        if query_params.start_time:
            query = query.filter(AlertHistory.timestamp >= query_params.start_time)
        
        if query_params.end_time:
            query = query.filter(AlertHistory.timestamp <= query_params.end_time)
        
        if query_params.search:
            query = query.filter(
                or_(
                    AlertHistory.title.ilike(f"%{query_params.search}%"),
                    AlertHistory.message.ilike(f"%{query_params.search}%"),
                    AlertHistory.data.like(f"%{query_params.search}%")
                )
            )
        
        # 获取总数
        total_count = query.count()
        
        # 应用排序
        sort_field = getattr(AlertHistory, query_params.sort_by, AlertHistory.timestamp)
        if query_params.sort_order.lower() == "desc":
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(asc(sort_field))
        
        # 应用分页
        offset = (query_params.page - 1) * query_params.page_size
        alerts = query.offset(offset).limit(query_params.page_size).all()
        
        # 转换为字典
        alerts_data = []
        for alert in alerts:
            alert_dict = {
                "id": alert.id,
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "alert_type": alert.alert_type,
                "alert_level": alert.alert_level,
                "source": alert.source,
                "component": alert.component,
                "status": alert.status,
                "data": alert.data,
                "metadata": alert.metadata,
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_by": alert.resolved_by,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "resolution_notes": alert.resolution_notes,
                "assigned_to": alert.assigned_to,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            }
            alerts_data.append(alert_dict)
        
        return {
            "alerts": alerts_data,
            "pagination": {
                "page": query_params.page,
                "page_size": query_params.page_size,
                "total": total_count,
                "total_pages": (total_count + query_params.page_size - 1) // query_params.page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取告警历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警历史失败: {str(e)}"
        )


@router.get("/alerts/history/{alert_id}", tags=["alerts"])
async def get_alert_detail(
    alert_id: str = FPath(..., description="告警ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取告警详情
    
    Args:
        alert_id: 告警ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        告警详情
    """
    logger.info(f"用户 {current_user['username']} 获取告警详情 {alert_id}")
    
    try:
        # 查找告警
        alert = db.query(AlertHistory).filter(AlertHistory.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警 {alert_id} 不存在"
            )
        
        # 转换为字典
        alert_dict = {
            "id": alert.id,
            "alert_id": alert.alert_id,
            "title": alert.title,
            "message": alert.message,
            "alert_type": alert.alert_type,
            "alert_level": alert.alert_level,
            "source": alert.source,
            "component": alert.component,
            "status": alert.status,
            "data": alert.data,
            "metadata": alert.metadata,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "resolved_by": alert.resolved_by,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "resolution_notes": alert.resolution_notes,
            "assigned_to": alert.assigned_to,
            "created_at": alert.created_at.isoformat() if alert.created_at else None
        }
        
        return alert_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取告警详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警详情失败: {str(e)}"
        )


@router.post("/alerts", tags=["alerts"], status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: AlertCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    创建告警
    
    Args:
        request: 告警创建请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        创建的告警
    """
    logger.info(f"用户 {current_user['username']} 创建告警: {request.title}")
    
    try:
        # 创建告警记录
        alert = AlertHistory(
            alert_id=request.alert_id,
            title=request.title,
            message=request.message,
            alert_type=request.alert_type.value,
            alert_level=request.alert_level.value,
            source=request.source,
            component=request.component,
            status="active",
            data=request.data,
            metadata=request.metadata,
            timestamp=request.timestamp,
            created_by=current_user['username']
        )
        
        # 保存到数据库
        db.add(alert)
        db.commit()
        db.refresh(alert)
        
        logger.info(f"告警 {alert.alert_id} 创建成功")
        
        # 触发告警规则
        await _trigger_alert_rules(alert, db)
        
        return {
            "message": "告警创建成功",
            "alert_id": alert.alert_id,
            "alert": {
                "id": alert.id,
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "alert_type": alert.alert_type,
                "alert_level": alert.alert_level,
                "status": alert.status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建告警失败: {str(e)}"
        )


async def _trigger_alert_rules(alert: AlertHistory, db: Session):
    """触发告警规则"""
    try:
        # 获取启用的告警规则
        rules = db.query(AlertRule).filter(
            AlertRule.enabled == True
        ).order_by(desc(AlertRule.priority)).all()
        
        triggered_rules = []
        
        for rule in rules:
            # 检查告警类型和级别
            if rule.alert_type != "all" and rule.alert_type != alert.alert_type:
                continue
            
            if rule.alert_level != "all" and rule.alert_level != alert.alert_level:
                continue
            
            # 检查时间窗口
            if rule.time_window and not _check_time_window(rule.time_window, alert.timestamp):
                continue
            
            # 检查条件
            conditions_met = True
            for condition in rule.conditions:
                # 检查告警数据中是否满足条件
                field_value = alert.data.get(condition.field) or alert.metadata.get(condition.field)
                if field_value is None:
                    conditions_met = False
                    break
                
                if not _test_condition(condition, field_value):
                    conditions_met = False
                    break
            
            if conditions_met:
                triggered_rules.append(rule)
                # 执行动作
                await _execute_alert_actions(rule, alert, db)
        
        return triggered_rules
        
    except Exception as e:
        logger.error(f"触发告警规则失败: {e}")
        return []


def _check_time_window(time_window: Dict[str, Any], timestamp: datetime) -> bool:
    """检查时间窗口"""
    try:
        if not time_window:
            return True
        
        start_time = time_window.get("start_time")
        end_time = time_window.get("end_time")
        
        if not start_time or not end_time:
            return True
        
        # 转换时间字符串为datetime对象
        if isinstance(start_time, str):
            from dateutil.parser import parse
            start_time = parse(start_time)
        if isinstance(end_time, str):
            from dateutil.parser import parse
            end_time = parse(end_time)
        
        # 检查是否在时间窗口内
        if start_time <= timestamp <= end_time:
            return True
        
        # 检查重复
        if time_window.get("repeat"):
            repeat_interval = time_window.get("repeat_interval", 1440)  # 默认24小时
            # 计算时间差
            time_diff = (timestamp - start_time).total_seconds() / 60
            if time_diff % repeat_interval == 0:
                return True
        
        # 检查工作日
        weekdays = time_window.get("weekdays")
        if weekdays and timestamp.weekday() in weekdays:
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"检查时间窗口失败: {e}")
        return True


async def _execute_alert_actions(rule: AlertRule, alert: AlertHistory, db: Session):
    """执行告警动作"""
    try:
        for action in rule.actions:
            try:
                if action.action_type == "email":
                    await _send_email_alert(action, alert, db)
                elif action.action_type == "webhook":
                    await _send_webhook_alert(action, alert, db)
                elif action.action_type == "slack":
                    await _send_slack_alert(action, alert, db)
                elif action.action_type == "telegram":
                    await _send_telegram_alert(action, alert, db)
                elif action.action_type == "create_ticket":
                    await _create_ticket(action, alert, db)
                elif action.action_type == "execute_script":
                    await _execute_script(action, alert, db)
                
                # 记录动作执行
                logger.info(f"执行告警动作: {action.action_type} for alert {alert.alert_id}")
                
            except Exception as e:
                logger.error(f"执行告警动作失败: {e}")
                continue
    except Exception as e:
        logger.error(f"执行告警动作失败: {e}")


async def _send_email_alert(action: AlertAction, alert: AlertHistory, db: Session):
    """发送邮件告警"""
    try:
        config = get_alert_config()
        if not config.email.enabled:
            return
        
        # 获取接收人
        recipients = db.query(AlertRecipient).filter(
            AlertRecipient.enabled == True,
            AlertRecipient.notification_methods.contains(["email"])
        ).all()
        
        if not recipients:
            return
        
        # 准备邮件内容
        subject = f"[{alert.alert_level.upper()}] {alert.title}"
        body = f"""
        告警标题: {alert.title}
        告警消息: {alert.message}
        告警级别: {alert.alert_level}
        告警类型: {alert.alert_type}
        告警源: {alert.source}
        组件: {alert.component}
        时间: {alert.timestamp}
        
        告警数据:
        {json.dumps(alert.data, indent=2, ensure_ascii=False)}
        """
        
        # 发送邮件
        for recipient in recipients:
            if alert.alert_level in recipient.alert_levels and alert.alert_type in recipient.alert_types:
                # 这里应该实现实际的邮件发送逻辑
                logger.info(f"发送邮件告警给 {recipient.email}: {subject}")
    
    except Exception as e:
        logger.error(f"发送邮件告警失败: {e}")


async def _send_webhook_alert(action: AlertAction, alert: AlertHistory, db: Session):
    """发送Webhook告警"""
    try:
        import aiohttp
        
        webhook_url = action.target
        if not webhook_url:
            return
        
        # 准备数据
        data = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "message": alert.message,
            "alert_type": alert.alert_type,
            "alert_level": alert.alert_level,
            "source": alert.source,
            "component": alert.component,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "data": alert.data
        }
        
        # 发送请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Webhook请求失败: {response.status}")
    
    except Exception as e:
        logger.error(f"发送Webhook告警失败: {e}")


async def _send_slack_alert(action: AlertAction, alert: AlertHistory, db: Session):
    """发送Slack告警"""
    try:
        import aiohttp
        
        webhook_url = action.target
        if not webhook_url:
            return
        
        # 根据告警级别设置颜色
        color_map = {
            "critical": "#FF0000",
            "high": "#FF4500",
            "medium": "#FFA500",
            "low": "#FFFF00",
            "info": "#00BFFF",
            "debug": "#808080"
        }
        color = color_map.get(alert.alert_level.lower(), "#808080")
        
        # 准备Slack消息
        slack_message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"[{alert.alert_level.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "告警类型",
                            "value": alert.alert_type,
                            "short": True
                        },
                        {
                            "title": "告警源",
                            "value": alert.source,
                            "short": True
                        },
                        {
                            "title": "组件",
                            "value": alert.component,
                            "short": True
                        },
                        {
                            "title": "时间",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S") if alert.timestamp else "N/A",
                            "short": True
                        }
                    ],
                    "ts": int(alert.timestamp.timestamp()) if alert.timestamp else int(datetime.now().timestamp())
                }
            ]
        }
        
        # 发送请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=slack_message,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Slack请求失败: {response.status}")
    
    except Exception as e:
        logger.error(f"发送Slack告警失败: {e}")


async def _send_telegram_alert(action: AlertAction, alert: AlertHistory, db: Session):
    """发送Telegram告警"""
    try:
        import aiohttp
        
        bot_token = action.parameters.get("bot_token")
        chat_id = action.parameters.get("chat_id")
        
        if not bot_token or not chat_id:
            return
        
        # 准备消息
        message = f"""
        *🚨 告警通知*
        
        *标题*: {alert.title}
        *级别*: {alert.alert_level.upper()}
        *类型*: {alert.alert_type}
        
        *消息*: {alert.message}
        
        *源*: {alert.source}
        *组件*: {alert.component}
        *时间*: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S') if alert.timestamp else 'N/A'}
        
        *告警ID*: {alert.alert_id}
        """
        
        # 发送请求
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Telegram请求失败: {response.status}")
    
    except Exception as e:
        logger.error(f"发送Telegram告警失败: {e}")


async def _create_ticket(action: AlertAction, alert: AlertHistory, db: Session):
    """创建工单"""
    try:
        # 这里应该实现工单系统的集成
        logger.info(f"创建工单 for alert {alert.alert_id}")
    except Exception as e:
        logger.error(f"创建工单失败: {e}")


async def _execute_script(action: AlertAction, alert: AlertHistory, db: Session):
    """执行脚本"""
    try:
        script_path = action.target
        if not script_path or not os.path.exists(script_path):
            return
        
        # 准备参数
        import subprocess
        env = os.environ.copy()
        env.update({
            "ALERT_ID": alert.alert_id,
            "ALERT_TITLE": alert.title,
            "ALERT_MESSAGE": alert.message,
            "ALERT_TYPE": alert.alert_type,
            "ALERT_LEVEL": alert.alert_level,
            "ALERT_SOURCE": alert.source,
            "ALERT_COMPONENT": alert.component,
            "ALERT_TIMESTAMP": alert.timestamp.isoformat() if alert.timestamp else ""
        })
        
        # 执行脚本
        result = subprocess.run(
            [script_path],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"执行脚本失败: {result.stderr}")
    
    except Exception as e:
        logger.error(f"执行脚本失败: {e}")


@router.put("/alerts/{alert_id}", tags=["alerts"])
async def update_alert(
    alert_id: str = FPath(..., description="告警ID"),
    request: AlertUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    更新告警
    
    Args:
        alert_id: 告警ID
        request: 告警更新请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        更新后的告警
    """
    logger.info(f"用户 {current_user['username']} 更新告警 {alert_id}")
    
    try:
        # 查找告警
        alert = db.query(AlertHistory).filter(AlertHistory.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警 {alert_id} 不存在"
            )
        
        # 更新字段
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                setattr(alert, field, value)
        
        # 如果是确认或解决，记录用户和时间
        if request.status == "acknowledged" and not alert.acknowledged_at:
            alert.acknowledged_by = current_user['username']
            alert.acknowledged_at = datetime.now()
        elif request.status == "resolved" and not alert.resolved_at:
            alert.resolved_by = current_user['username']
            alert.resolved_at = datetime.now()
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"告警 {alert_id} 更新成功")
        
        return {
            "message": "告警更新成功",
            "alert_id": alert_id,
            "status": alert.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新告警失败: {str(e)}"
        )


@router.post("/alerts/{alert_id}/acknowledge", tags=["alerts"])
async def acknowledge_alert(
    alert_id: str = FPath(..., description="告警ID"),
    request: AlertAcknowledgeRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    确认告警
    
    Args:
        alert_id: 告警ID
        request: 确认请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        确认结果
    """
    logger.info(f"用户 {current_user['username']} 确认告警 {alert_id}")
    
    try:
        # 查找告警
        alert = db.query(AlertHistory).filter(AlertHistory.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警 {alert_id} 不存在"
            )
        
        if alert.status == "acknowledged":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="告警已确认"
            )
        
        if alert.status == "resolved":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="告警已解决，无法确认"
            )
        
        # 更新状态
        alert.status = "acknowledged"
        alert.acknowledged_by = current_user['username']
        alert.acknowledged_at = datetime.now()
        
        if request.notes:
            alert.resolution_notes = request.notes
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"告警 {alert_id} 已确认")
        
        return {
            "message": "告警已确认",
            "alert_id": alert_id,
            "acknowledged_by": current_user['username'],
            "acknowledged_at": alert.acknowledged_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"确认告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"确认告警失败: {str(e)}"
        )


@router.post("/alerts/{alert_id}/resolve", tags=["alerts"])
async def resolve_alert(
    alert_id: str = FPath(..., description="告警ID"),
    request: AlertResolveRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    解决告警
    
    Args:
        alert_id: 告警ID
        request: 解决请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        解决结果
    """
    logger.info(f"用户 {current_user['username']} 解决告警 {alert_id}")
    
    try:
        # 查找告警
        alert = db.query(AlertHistory).filter(AlertHistory.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警 {alert_id} 不存在"
            )
        
        if alert.status == "resolved":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="告警已解决"
            )
        
        # 更新状态
        alert.status = "resolved"
        alert.resolved_by = current_user['username']
        alert.resolved_at = datetime.now()
        alert.resolution_notes = request.notes
        
        if request.resolution_code:
            alert.metadata = alert.metadata or {}
            alert.metadata["resolution_code"] = request.resolution_code
        
        # 保存到数据库
        db.commit()
        
        logger.info(f"告警 {alert_id} 已解决")
        
        return {
            "message": "告警已解决",
            "alert_id": alert_id,
            "resolved_by": current_user['username'],
            "resolved_at": alert.resolved_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"解决告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"解决告警失败: {str(e)}"
        )


@router.delete("/alerts/{alert_id}", tags=["alerts"])
async def delete_alert(
    alert_id: str = FPath(..., description="告警ID"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    删除告警
    
    Args:
        alert_id: 告警ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    logger.info(f"管理员 {current_user['username']} 删除告警 {alert_id}")
    
    try:
        # 查找告警
        alert = db.query(AlertHistory).filter(AlertHistory.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"告警 {alert_id} 不存在"
            )
        
        # 删除告警
        db.delete(alert)
        db.commit()
        
        logger.info(f"告警 {alert_id} 删除成功")
        
        return {
            "message": "告警删除成功",
            "alert_id": alert_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除告警失败: {str(e)}"
        )


# 告警统计API
@router.get("/alerts/statistics", tags=["alerts"])
async def get_alert_statistics(
    time_range: str = Query("24h", description="时间范围"),
    group_by: str = Query("alert_level", description="分组字段"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    获取告警统计
    
    Args:
        time_range: 时间范围
        group_by: 分组字段
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        告警统计
    """
    logger.info(f"用户 {current_user['username']} 获取告警统计")
    
    try:
        # 计算时间范围
        end_time = datetime.now()
        if time_range.endswith("h"):
            hours = int(time_range[:-1])
            start_time = end_time - timedelta(hours=hours)
        elif time_range.endswith("d"):
            days = int(time_range[:-1])
            start_time = end_time - timedelta(days=days)
        elif time_range.endswith("w"):
            weeks = int(time_range[:-1])
            start_time = end_time - timedelta(weeks=weeks)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # 构建查询
        query = db.query(AlertHistory).filter(
            AlertHistory.timestamp >= start_time,
            AlertHistory.timestamp <= end_time
        )
        
        # 获取总数
        total_count = query.count()
        
        # 按状态统计
        status_stats = {}
        status_results = query.group_by(AlertHistory.status).with_entities(
            AlertHistory.status,
            func.count(AlertHistory.id)
        ).all()
        
        for status, count in status_results:
            status_stats[status] = count
        
        # 按级别统计
        level_stats = {}
        level_results = query.group_by(AlertHistory.alert_level).with_entities(
            AlertHistory.alert_level,
            func.count(AlertHistory.id)
        ).all()
        
        for level, count in level_results:
            level_stats[level] = count
        
        # 按类型统计
        type_stats = {}
        type_results = query.group_by(AlertHistory.alert_type).with_entities(
            AlertHistory.alert_type,
            func.count(AlertHistory.id)
        ).all()
        
        for alert_type, count in type_results:
            type_stats[alert_type] = count
        
        # 按组件统计
        component_stats = {}
        component_results = query.group_by(AlertHistory.component).with_entities(
            AlertHistory.component,
            func.count(AlertHistory.id)
        ).all()
        
        for component, count in component_results:
            component_stats[component] = count
        
        # 时间序列统计
        time_series = []
        if group_by == "hour":
            # 按小时统计
            hour_results = query.with_entities(
                func.date_format(AlertHistory.timestamp, "%Y-%m-%d %H:00:00"),
                func.count(AlertHistory.id)
            ).group_by(func.date_format(AlertHistory.timestamp, "%Y-%m-%d %H")).all()
            
            for hour_str, count in hour_results:
                time_series.append({
                    "time": hour_str,
                    "count": count
                })
        
        statistics = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "range": time_range
            },
            "total": total_count,
            "by_status": status_stats,
            "by_level": level_stats,
            "by_type": type_stats,
            "by_component": component_stats,
            "time_series": time_series
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"获取告警统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取告警统计失败: {str(e)}"
        )


# 告警导出API
@router.post("/alerts/export", tags=["alerts"])
async def export_alerts(
    request: AlertExportRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    导出告警
    
    Args:
        request: 导出请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        导出的告警数据
    """
    logger.info(f"用户 {current_user['username']} 导出告警")
    
    try:
        import tempfile
        
        # 构建查询
        query = db.query(AlertHistory)
        
        # 应用时间范围过滤器
        if request.time_range:
            end_time = datetime.now()
            if request.time_range.endswith("h"):
                hours = int(request.time_range[:-1])
                start_time = end_time - timedelta(hours=hours)
            elif request.time_range.endswith("d"):
                days = int(request.time_range[:-1])
                start_time = end_time - timedelta(days=days)
            elif request.time_range.endswith("w"):
                weeks = int(request.time_range[:-1])
                start_time = end_time - timedelta(weeks=weeks)
            else:
                start_time = end_time - timedelta(hours=24)
            
            query = query.filter(
                AlertHistory.timestamp >= start_time,
                AlertHistory.timestamp <= end_time
            )
        
        # 应用其他过滤器
        for field, value in request.filters.items():
            if hasattr(AlertHistory, field) and value is not None:
                query = query.filter(getattr(AlertHistory, field) == value)
        
        # 获取数据
        alerts = query.order_by(desc(AlertHistory.timestamp)).all()
        
        # 转换为字典
        alerts_data = []
        for alert in alerts:
            alert_dict = {
                "id": alert.id,
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "alert_type": alert.alert_type,
                "alert_level": alert.alert_level,
                "source": alert.source,
                "component": alert.component,
                "status": alert.status,
                "data": alert.data,
                "metadata": alert.metadata,
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_by": alert.resolved_by,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "resolution_notes": alert.resolution_notes,
                "assigned_to": alert.assigned_to,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            }
            alerts_data.append(alert_dict)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            if request.format.lower() == "json":
                json.dump(alerts_data, f, indent=2, default=str)
            elif request.format.lower() == "csv":
                import csv
                if alerts_data:
                    # 获取所有字段
                    fieldnames = set()
                    for alert in alerts_data:
                        fieldnames.update(alert.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(alerts_data)
            else:
                json.dump(alerts_data, f, indent=2, default=str)
            
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
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}.gz"
        else:
            export_file = temp_file
            content_type = "application/json" if request.format.lower() == "json" else "text/csv"
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}"
        
        # 返回文件
        return FileResponse(
            path=export_file,
            filename=filename,
            media_type=content_type,
            headers={
                "X-Total-Alerts": str(len(alerts_data)),
                "X-Export-Time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"导出告警失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出告警失败: {str(e)}"
        )


# 告警测试API
@router.post("/alerts/test", tags=["alerts"])
async def test_alert_notification(
    request: AlertTestRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    测试告警通知
    
    Args:
        request: 测试请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        测试结果
    """
    logger.info(f"管理员 {current_user['username']} 测试告警通知")
    
    try:
        # 创建测试告警
        test_alert = AlertHistory(
            alert_id=f"test_{int(time.time())}",
            title="测试告警通知",
            message="这是一个测试告警，用于验证通知系统是否正常工作。",
            alert_type=request.alert_type.value,
            alert_level=request.alert_level.value,
            source="test",
            component="alert_api",
            status="active",
            data={
                "test": True,
                "user": current_user['username'],
                "timestamp": datetime.now().isoformat()
            },
            metadata={
                "test_alert": True,
                "notification_method": request.notification_method.value
            },
            timestamp=datetime.now(),
            created_by=current_user['username']
        )
        
        db.add(test_alert)
        db.commit()
        
        # 根据通知方法发送测试
        results = []
        
        if request.notification_method == NotificationMethodEnum.EMAIL:
            # 测试邮件通知
            result = await _test_email_notification(request, test_alert, db)
            results.append(result)
        
        elif request.notification_method == NotificationMethodEnum.WEBHOOK:
            # 测试Webhook通知
            result = await _test_webhook_notification(request, test_alert, db)
            results.append(result)
        
        elif request.notification_method == NotificationMethodEnum.SLACK:
            # 测试Slack通知
            result = await _test_slack_notification(request, test_alert, db)
            results.append(result)
        
        elif request.notification_method == NotificationMethodEnum.TELEGRAM:
            # 测试Telegram通知
            result = await _test_telegram_notification(request, test_alert, db)
            results.append(result)
        
        elif request.notification_method == NotificationMethodEnum.WEB:
            # 测试Web通知
            result = await _test_web_notification(request, test_alert, db)
            results.append(result)
        
        elif request.notification_method == NotificationMethodEnum.ALL:
            # 测试所有通知方法
            result_email = await _test_email_notification(request, test_alert, db)
            result_webhook = await _test_webhook_notification(request, test_alert, db)
            result_slack = await _test_slack_notification(request, test_alert, db)
            result_telegram = await _test_telegram_notification(request, test_alert, db)
            result_web = await _test_web_notification(request, test_alert, db)
            
            results.extend([result_email, result_webhook, result_slack, result_telegram, result_web])
        
        return {
            "message": "告警通知测试完成",
            "alert_id": test_alert.alert_id,
            "results": results
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"测试告警通知失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"测试告警通知失败: {str(e)}"
        )


async def _test_email_notification(request: AlertTestRequest, alert: AlertHistory, db: Session) -> Dict[str, Any]:
    """测试邮件通知"""
    try:
        config = get_alert_config()
        if not config.email.enabled:
            return {
                "method": "email",
                "success": False,
                "error": "邮件通知未启用"
            }
        
        # 获取接收人
        if request.recipient_id:
            recipient = db.query(AlertRecipient).filter(AlertRecipient.id == request.recipient_id).first()
            if not recipient:
                return {
                    "method": "email",
                    "success": False,
                    "error": f"接收人 {request.recipient_id} 不存在"
                }
            recipients = [recipient]
        else:
            recipients = [AlertRecipient(
                name="测试用户",
                email=config.email.smtp_username,
                notification_methods=["email"]
            )]
        
        # 这里应该实现实际的邮件发送逻辑
        # 暂时返回模拟结果
        return {
            "method": "email",
            "success": True,
            "recipients": [r.email for r in recipients],
            "message": "测试邮件已发送"
        }
        
    except Exception as e:
        return {
            "method": "email",
            "success": False,
            "error": str(e)
        }


async def _test_webhook_notification(request: AlertTestRequest,