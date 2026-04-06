"""
数据库测试模块
测试DatabaseManager、数据库模型、ORM操作等功能
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
import sqlite3
import hashlib
import pickle
import gzip
from sqlalchemy import create_engine, text, func, and_, or_, not_
from sqlalchemy.orm import sessionmaker, Session, Query
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
import mysql.connector
from mysql.connector import Error as MySQLError

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig

# 导入被测试模块
from src.core.database_manager import (
    DatabaseManager,
    DatabaseConfig,
    ConnectionPool,
    BackupManager,
    DatabaseMetrics
)
from src.models.database_models import (
    Base,
    RawSignal,
    Detection,
    KnownSignal,
    DetectionMatch,
    ProcessedSignal,
    AlgorithmResult,
    SystemEvent,
    AlertRule,
    AlertHistory,
    AlertRecipient,
    AlertTemplate,
    SystemStatus
)

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq, SignalType, SignalMetrics
from src.models.detection_result import DetectionResult, create_detection_from_signal, DetectionType, ConfidenceLevel
from src.models.alert_models import AlertLevel, AlertStatus, AlertType, NotificationMethod

# 导入工具
from src.utils.data_validator import DataValidator, ValidationResult, ValidationLevel
from src.utils.file_utils import ensure_directory, delete_file, get_file_info
from src.utils.performance_monitor import PerformanceMonitor

# 导入测试fixtures
from src.tests.conftest import (
    db_config,
    database_manager,
    db_session,
    async_db_session,
    mock_signal_data,
    mock_signal_dict,
    mock_detection_result,
    mock_detection_dict,
    test_data_generator
)

# 设置测试日志
import logging
logger = logging.getLogger(__name__)


class TestDatabaseManager:
    """数据库管理器测试类"""
    
    def test_database_manager_initialization(self, db_config):
        """测试数据库管理器初始化"""
        # 创建数据库管理器
        manager = DatabaseManager(
            db_url="sqlite:///:memory:",
            config=db_config
        )
        
        # 验证初始化
        assert manager.db_url == "sqlite:///:memory:"
        assert manager.config == db_config
        assert manager.engine is not None
        assert manager.SessionLocal is not None
        assert manager.initialized is True
        
        # 验证表创建
        inspector = manager.engine.inspector
        tables = inspector.get_table_names()
        
        # 应该创建了基础表
        expected_tables = ["raw_signals", "detections", "known_signals", "system_events"]
        for table in expected_tables:
            assert table in tables
        
        # 清理
        manager.close()
    
    def test_database_manager_mysql_connection(self):
        """测试MySQL数据库连接"""
        # 跳过测试，除非配置了MySQL
        if TestConfig.TEST_DB_TYPE != "mysql":
            pytest.skip("MySQL测试需要MySQL数据库")
        
        try:
            # 测试MySQL连接
            manager = DatabaseManager(
                db_url=f"mysql+pymysql://{TestConfig.TEST_USERNAME}:{TestConfig.TEST_PASSWORD}@"
                      f"localhost:3306/{TestConfig.TEST_DB_NAME}",
                echo=False
            )
            
            # 验证连接
            assert manager.test_connection() is True
            
            # 清理
            manager.close()
            
        except Exception as e:
            pytest.skip(f"无法连接到MySQL数据库: {e}")
    
    def test_database_manager_sqlite_connection(self):
        """测试SQLite数据库连接"""
        # 使用内存数据库
        manager = DatabaseManager(
            db_url="sqlite:///:memory:",
            echo=False
        )
        
        # 验证连接
        assert manager.test_connection() is True
        
        # 清理
        manager.close()
    
    def test_database_manager_create_tables(self, db_config):
        """测试数据库表创建"""
        # 使用临时文件数据库
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_file = f.name
        
        try:
            # 创建数据库管理器
            manager = DatabaseManager(
                db_url=f"sqlite:///{db_file}",
                config=db_config
            )
            
            # 创建表
            manager.create_tables()
            
            # 验证表存在
            inspector = manager.engine.inspector
            tables = inspector.get_table_names()
            
            # 检查所有表
            expected_tables = [
                "raw_signals", "detections", "known_signals", 
                "detection_matches", "processed_signals", "algorithm_results",
                "system_events", "alert_rules", "alert_history", 
                "alert_recipients", "alert_templates", "system_status"
            ]
            
            for table in expected_tables:
                assert table in tables
            
            # 验证表结构
            for table_name in ["raw_signals", "detections"]:
                columns = inspector.get_columns(table_name)
                assert len(columns) > 0
            
        finally:
            # 清理
            if 'manager' in locals():
                manager.close()
            if os.path.exists(db_file):
                os.unlink(db_file)
    
    def test_database_manager_drop_tables(self, db_config):
        """测试数据库表删除"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_file = f.name
        
        try:
            # 创建数据库管理器
            manager = DatabaseManager(
                db_url=f"sqlite:///{db_file}",
                config=db_config
            )
            
            # 创建表
            manager.create_tables()
            
            # 删除表
            manager.drop_tables()
            
            # 验证表被删除
            inspector = manager.engine.inspector
            tables = inspector.get_table_names()
            assert len(tables) == 0
            
        finally:
            #