"""
MySQL 数据库管理模块 - 负责信号数据的存储、查询和管理
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from pathlib import Path
from contextlib import contextmanager
import threading
import uuid
import hashlib
import numpy as np
import pickle
from dataclasses import asdict

import pymysql
from pymysql import connections
from pymysql.cursors import DictCursor
import pymysql.pool as pool

from src.models.signal_data import SignalData
from src.models.detection_result import DetectionResult
from src.utils.logger import get_logger
from src.utils.time_utils import get_timestamp, format_datetime

class MySQLDatabaseManager:
    """MySQL 数据库管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MySQL 数据库管理器
        
        Args:
            config: 数据库配置字典，包含以下字段：
                - host: MySQL 服务器地址
                - port: MySQL 服务器端口
                - username: 用户名
                - password: 密码
                - database_name: 数据库名称
                - pool_size: 连接池大小（默认10）
                - charset: 字符集（默认utf8mb4）
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.initialized = False
        self.lock = threading.RLock()
        
        # 连接池
        self.connection_pool = None
        
        # 数据库连接参数
        self.connection_params = {
            'host': config.get('host', 'localhost'),
            'port': int(config.get('port', 3306)),
            'user': config.get('username', 'root'),
            'password': config.get('password', ''),
            'database': config.get('database_name', 'signal_db'),
            'charset': config.get('charset', 'utf8mb4'),
            'cursorclass': DictCursor,
            'autocommit': False
        }
        
        # 连接池配置
        self.pool_size = int(config.get('pool_size', 10))
        self.pool_name = f"signal_db_pool_{id(self)}"
        
        # 统计信息
        self.stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'inserts_performed': 0,
            'updates_performed': 0,
            'errors': 0,
            'last_error': None,
            'last_operation_time': None
        }
        
        # 缓存最近的操作
        self.recent_operations = []
        self.max_recent_ops = 1000
        
    def initialize(self) -> bool:
        """
        初始化数据库，创建必要的表结构
        
        Returns:
            bool: 初始化是否成功
        """
        with self.lock:
            if self.initialized:
                return True
                
            self.logger.info("初始化 MySQL 数据库管理器...")
            
            try:
                # 创建数据库（如果不存在）
                self._create_database_if_not_exists()
                
                # 初始化连接池
                self._init_connection_pool()
                
                # 创建表结构
                self._create_tables()
                
                # 创建索引
                self._create_indexes()
                
                # 创建存储过程
                self._create_stored_procedures()
                
                self.initialized = True
                self.logger.info("MySQL 数据库初始化完成")
                
                # 记录初始化完成
                self._log_operation("initialize", "Database initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"数据库初始化失败: {e}", exc_info=True)
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
                return False
    
    def _create_database_if_not_exists(self):
        """创建数据库（如果不存在）"""
        # 先连接到 MySQL 服务器（不指定数据库）
        temp_params = self.connection_params.copy()
        temp_params.pop('database', None)
        
        try:
            connection = pymysql.connect(**temp_params)
            cursor = connection.cursor()
            
            # 创建数据库
            db_name = self.connection_params['database']
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            
            cursor.close()
            connection.close()
            self.logger.info(f"数据库 {db_name} 已创建或已存在")
            
        except Exception as e:
            self.logger.error(f"创建数据库失败: {e}")
            raise
    
    def _init_connection_pool(self):
        """初始化连接池"""
        try:
            self.connection_pool = pool.QueuePool(
                creator=pymysql.connect,
                maxsize=self.pool_size,
                timeout=10,
                **self.connection_params
            )
            self.logger.info(f"MySQL 连接池初始化完成，大小: {self.pool_size}")
        except Exception as e:
            self.logger.error(f"连接池初始化失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        从连接池获取连接
        
        Yields:
            pymysql.Connection: 数据库连接
        """
        connection = None
        try:
            if self.connection_pool is None:
                raise RuntimeError("连接池未初始化")
            
            connection = self.connection_pool.get_connection()
            self.stats['connections_created'] += 1
            yield connection
            
        except Exception as e:
            self.logger.error(f"获取数据库连接失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            raise
            
        finally:
            if connection:
                # 将连接返回到连接池
                self.connection_pool.release(connection)
    
    @contextmanager
    def get_cursor(self, connection=None):
        """
        获取数据库游标
        
        Args:
            connection: 可选的数据库连接，如果为None则从连接池获取
            
        Yields:
            pymysql.cursors.DictCursor: 数据库游标
        """
        own_connection = False
        
        try:
            if connection is None:
                connection = self.connection_pool.get_connection()
                own_connection = True
                self.stats['connections_created'] += 1
            
            cursor = connection.cursor()
            yield cursor
            
        except Exception as e:
            self.logger.error(f"获取游标失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            raise
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if own_connection and connection:
                self.connection_pool.release(connection)
    
    def _create_tables(self):
        """创建数据库表结构"""
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS system_status (
                id INT AUTO_INCREMENT PRIMARY KEY,
                component VARCHAR(100) NOT NULL,
                status ENUM('running', 'stopped', 'error', 'warning') NOT NULL,
                message TEXT,
                cpu_usage FLOAT,
                memory_usage FLOAT,
                disk_usage FLOAT,
                network_rx BIGINT,
                network_tx BIGINT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_component (component),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS raw_signals (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                signal_id VARCHAR(64) NOT NULL UNIQUE,
                timestamp TIMESTAMP NOT NULL,
                center_frequency DOUBLE NOT NULL,
                sample_rate DOUBLE NOT NULL,
                bandwidth DOUBLE,
                iq_data BLOB,  -- 存储序列化的IQ数据
                iq_data_hash VARCHAR(64),  -- IQ数据的哈希值，用于去重
                metadata JSON,  -- 存储元数据
                device_id VARCHAR(100),
                channel_index INT DEFAULT 0,
                gain FLOAT,
                noise_floor FLOAT,
                duration_ms FLOAT,
                raw_data_size INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_signal_id (signal_id),
                INDEX idx_timestamp (timestamp),
                INDEX idx_frequency (center_frequency),
                INDEX idx_device (device_id),
                INDEX idx_data_hash (iq_data_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS processed_signals (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                raw_signal_id VARCHAR(64) NOT NULL,
                process_id VARCHAR(64) NOT NULL,
                algorithm_name VARCHAR(100) NOT NULL,
                parameters JSON,
                input_features JSON,
                output_features JSON,
                confidence FLOAT,
                processing_time_ms FLOAT,
                status ENUM('success', 'failed', 'pending') DEFAULT 'success',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_raw_signal (raw_signal_id),
                INDEX idx_process_id (process_id),
                INDEX idx_algorithm (algorithm_name),
                INDEX idx_created_at (created_at),
                FOREIGN KEY (raw_signal_id) REFERENCES raw_signals(signal_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS detections (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                detection_id VARCHAR(64) NOT NULL UNIQUE,
                signal_id VARCHAR(64) NOT NULL,
                detection_type VARCHAR(50) NOT NULL,
                detection_time TIMESTAMP NOT NULL,
                center_frequency DOUBLE NOT NULL,
                bandwidth DOUBLE,
                power_db FLOAT,
                snr_db FLOAT,
                duration_ms FLOAT,
                modulation_type VARCHAR(50),
                modulation_params JSON,
                direction_angle FLOAT,
                direction_confidence FLOAT,
                location_lat DOUBLE,
                location_lon DOUBLE,
                location_accuracy FLOAT,
                features JSON,
                confidence FLOAT NOT NULL,
                is_known BOOLEAN DEFAULT FALSE,
                known_signal_id VARCHAR(64),
                classification VARCHAR(100),
                tags JSON,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_detection_id (detection_id),
                INDEX idx_signal_id (signal_id),
                INDEX idx_detection_time (detection_time),
                INDEX idx_frequency (center_frequency),
                INDEX idx_detection_type (detection_type),
                INDEX idx_is_known (is_known),
                INDEX idx_confidence (confidence),
                INDEX idx_created_at (created_at),
                FOREIGN KEY (signal_id) REFERENCES raw_signals(signal_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS known_signals (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                known_id VARCHAR(64) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                signal_type VARCHAR(50) NOT NULL,
                center_frequency DOUBLE,
                frequency_range_low DOUBLE,
                frequency_range_high DOUBLE,
                bandwidth DOUBLE,
                modulation_type VARCHAR(50),
                modulation_params JSON,
                typical_power_db FLOAT,
                typical_duration_ms FLOAT,
                features JSON,
                feature_hash VARCHAR(64),
                source VARCHAR(200),
                source_confidence FLOAT,
                is_active BOOLEAN DEFAULT TRUE,
                tags JSON,
                metadata JSON,
                created_by VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_known_id (known_id),
                INDEX idx_name (name),
                INDEX idx_signal_type (signal_type),
                INDEX idx_frequency_range (frequency_range_low, frequency_range_high),
                INDEX idx_feature_hash (feature_hash),
                INDEX idx_is_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS detection_matches (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                detection_id VARCHAR(64) NOT NULL,
                known_id VARCHAR(64) NOT NULL,
                match_score FLOAT NOT NULL,
                match_features JSON,
                match_algorithm VARCHAR(100),
                match_parameters JSON,
                is_verified BOOLEAN DEFAULT FALSE,
                verified_by VARCHAR(100),
                verified_at TIMESTAMP,
                verification_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_detection_id (detection_id),
                INDEX idx_known_id (known_id),
                INDEX idx_match_score (match_score),
                INDEX idx_is_verified (is_verified),
                UNIQUE KEY uk_detection_known (detection_id, known_id),
                FOREIGN KEY (detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE,
                FOREIGN KEY (known_id) REFERENCES known_signals(known_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                event_id VARCHAR(64) NOT NULL UNIQUE,
                event_type VARCHAR(100) NOT NULL,
                event_source VARCHAR(100),
                severity ENUM('info', 'warning', 'error', 'critical') DEFAULT 'info',
                title VARCHAR(500) NOT NULL,
                description TEXT,
                data JSON,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by VARCHAR(100),
                acknowledged_at TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_by VARCHAR(100),
                resolved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_event_id (event_id),
                INDEX idx_event_type (event_type),
                INDEX idx_severity (severity),
                INDEX idx_created_at (created_at),
                INDEX idx_acknowledged (acknowledged),
                INDEX idx_resolved (resolved)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS algorithm_results (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                result_id VARCHAR(64) NOT NULL UNIQUE,
                algorithm_type VARCHAR(100) NOT NULL,
                algorithm_name VARCHAR(100) NOT NULL,
                input_data_ids JSON,
                parameters JSON,
                result_data JSON,
                performance_metrics JSON,
                status ENUM('success', 'failed', 'running') DEFAULT 'success',
                error_message TEXT,
                processing_time_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_result_id (result_id),
                INDEX idx_algorithm_type (algorithm_type),
                INDEX idx_algorithm_name (algorithm_name),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS web_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(128) NOT NULL UNIQUE,
                user_id VARCHAR(100),
                user_agent TEXT,
                ip_address VARCHAR(45),
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                data JSON,
                INDEX idx_session_id (session_id),
                INDEX idx_user_id (user_id),
                INDEX idx_last_activity (last_activity)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        ]
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                for sql in tables_sql:
                    cursor.execute(sql)
                    self.stats['queries_executed'] += 1
                
                conn.commit()
        
        self.logger.info("数据库表结构创建完成")
    
    def _create_indexes(self):
        """创建额外的索引"""
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_raw_signals_composite ON raw_signals (timestamp, center_frequency, device_id)",
            "CREATE INDEX IF NOT EXISTS idx_detections_composite ON detections (detection_time, center_frequency, confidence)",
            "CREATE INDEX IF NOT EXISTS idx_detections_freq_time ON detections (center_frequency, detection_time)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_composite ON system_events (created_at, severity, event_type)",
            "CREATE INDEX IF NOT EXISTS idx_known_signals_freq ON known_signals (center_frequency, bandwidth)",
        ]
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                for sql in indexes_sql:
                    try:
                        cursor.execute(sql)
                        self.stats['queries_executed'] += 1
                    except Exception as e:
                        self.logger.warning(f"创建索引失败（可能已存在）: {e}")
                
                conn.commit()
    
    def _create_stored_procedures(self):
        """创建存储过程"""
        procedures = [
            """
            CREATE PROCEDURE IF NOT EXISTS sp_cleanup_old_data(
                IN days_to_keep INT,
                IN table_name VARCHAR(100)
            )
            BEGIN
                DECLARE rows_deleted INT;
                
                IF table_name = 'raw_signals' THEN
                    DELETE FROM raw_signals 
                    WHERE timestamp < DATE_SUB(NOW(), INTERVAL days_to_keep DAY)
                    AND id NOT IN (
                        SELECT DISTINCT raw_signal_id FROM detections 
                        WHERE detection_time > DATE_SUB(NOW(), INTERVAL 7 DAY)
                    );
                    
                ELSEIF table_name = 'detections' THEN
                    DELETE FROM detections 
                    WHERE detection_time < DATE_SUB(NOW(), INTERVAL days_to_keep DAY)
                    AND is_known = FALSE
                    AND confidence < 0.7;
                    
                ELSEIF table_name = 'system_events' THEN
                    DELETE FROM system_events 
                    WHERE created_at < DATE_SUB(NOW(), INTERVAL days_to_keep DAY)
                    AND severity = 'info';
                END IF;
                
                SET rows_deleted = ROW_COUNT();
                SELECT rows_deleted;
            END
            """,
            """
            CREATE PROCEDURE IF NOT EXISTS sp_get_detection_stats(
                IN start_time TIMESTAMP,
                IN end_time TIMESTAMP
            )
            BEGIN
                SELECT 
                    COUNT(*) as total_detections,
                    SUM(CASE WHEN is_known = TRUE THEN 1 ELSE 0 END) as known_detections,
                    SUM(CASE WHEN is_known = FALSE THEN 1 ELSE 0 END) as unknown_detections,
                    AVG(confidence) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence,
                    COUNT(DISTINCT detection_type) as unique_types,
                    COUNT(DISTINCT modulation_type) as unique_modulations
                FROM detections
                WHERE detection_time BETWEEN start_time AND end_time;
            END
            """
        ]
        
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                for proc in procedures:
                    try:
                        cursor.execute(proc)
                        self.stats['queries_executed'] += 1
                    except Exception as e:
                        self.logger.warning(f"创建存储过程失败: {e}")
                
                conn.commit()
    
    def store_raw_signal(self, signal_data: SignalData, store_iq_data: bool = True) -> Optional[str]:
        """
        存储原始信号数据
        
        Args:
            signal_data: 信号数据对象
            store_iq_data: 是否存储IQ数据（如果为False，只存储元数据）
            
        Returns:
            str: 信号ID，如果存储失败则返回None
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return None
        
        signal_id = signal_data.id or self._generate_id()
        
        try:
            # 计算IQ数据的哈希值
            iq_data_hash = None
            iq_data_blob = None
            
            if store_iq_data and signal_data.iq_data is not None:
                iq_data_hash = self._calculate_iq_hash(signal_data.iq_data)
                iq_data_blob = pickle.dumps(signal_data.iq_data)
            
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # 检查是否已存在相同哈希的数据
                    if iq_data_hash:
                        cursor.execute(
                            "SELECT signal_id FROM raw_signals WHERE iq_data_hash = %s LIMIT 1",
                            (iq_data_hash,)
                        )
                        existing = cursor.fetchone()
                        if existing:
                            self.logger.debug(f"重复的IQ数据，哈希: {iq_data_hash}")
                            return existing['signal_id']
                    
                    # 插入数据
                    cursor.execute("""
                        INSERT INTO raw_signals (
                            signal_id, timestamp, center_frequency, sample_rate, bandwidth,
                            iq_data, iq_data_hash, metadata, device_id, channel_index,
                            gain, noise_floor, duration_ms, raw_data_size
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        signal_id,
                        signal_data.timestamp,
                        signal_data.center_freq,
                        signal_data.sample_rate,
                        signal_data.bandwidth,
                        iq_data_blob,
                        iq_data_hash,
                        json.dumps(signal_data.metadata) if signal_data.metadata else None,
                        signal_data.metadata.get('device_id') if signal_data.metadata else None,
                        signal_data.metadata.get('channel_index', 0) if signal_data.metadata else 0,
                        signal_data.metadata.get('gain') if signal_data.metadata else None,
                        signal_data.metadata.get('noise_floor') if signal_data.metadata else None,
                        signal_data.metadata.get('duration_ms') if signal_data.metadata else None,
                        signal_data.size_bytes()
                    ))
                    
                    conn.commit()
                    self.stats['inserts_performed'] += 1
                    self.stats['queries_executed'] += 1
                    
                    self._log_operation("store_raw_signal", f"Stored signal {signal_id}")
                    return signal_id
                    
        except Exception as e:
            self.logger.error(f"存储原始信号失败: {e}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def store_detection_result(self, detection: DetectionResult) -> Optional[str]:
        """
        存储检测结果
        
        Args:
            detection: 检测结果对象
            
        Returns:
            str: 检测结果ID，如果存储失败则返回None
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return None
        
        detection_id = detection.id or self._generate_id()
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # 插入检测结果
                    cursor.execute("""
                        INSERT INTO detections (
                            detection_id, signal_id, detection_type, detection_time,
                            center_frequency, bandwidth, power_db, snr_db, duration_ms,
                            modulation_type, modulation_params, direction_angle,
                            direction_confidence, location_lat, location_lon,
                            location_accuracy, features, confidence, is_known,
                            known_signal_id, classification, tags, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        detection_id,
                        detection.signal_id,
                        detection.detection_type,
                        detection.timestamp,
                        detection.center_freq,
                        detection.bandwidth,
                        detection.power_db,
                        detection.snr_db,
                        detection.duration_ms,
                        detection.modulation_type,
                        json.dumps(detection.modulation_params) if detection.modulation_params else None,
                        detection.direction_angle,
                        detection.direction_confidence,
                        detection.location_lat,
                        detection.location_lon,
                        detection.location_accuracy,
                        json.dumps(detection.features) if detection.features else None,
                        detection.confidence,
                        detection.is_known,
                        detection.known_signal_id,
                        detection.classification,
                        json.dumps(detection.tags) if detection.tags else None,
                        json.dumps(detection.metadata) if detection.metadata else None
                    ))
                    
                    # 如果有已知信号匹配，存储匹配关系
                    if detection.known_signal_id and detection.match_score:
                        match_id = self._generate_id()
                        cursor.execute("""
                            INSERT INTO detection_matches (
                                detection_id, known_id, match_score, match_features,
                                match_algorithm, match_parameters
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            detection_id,
                            detection.known_signal_id,
                            detection.match_score,
                            json.dumps(detection.match_features) if detection.match_features else None,
                            detection.match_algorithm,
                            json.dumps(detection.match_parameters) if detection.match_parameters else None
                        ))
                    
                    conn.commit()
                    self.stats['inserts_performed'] += 1
                    self.stats['queries_executed'] += 1
                    
                    self._log_operation("store_detection", f"Stored detection {detection_id}")
                    return detection_id
                    
        except Exception as e:
            self.logger.error(f"存储检测结果失败: {e}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def query_similar_signals(self, features: Dict[str, Any], 
                             threshold: float = 0.8, 
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        查询相似信号
        
        Args:
            features: 特征字典
            threshold: 相似度阈值
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 相似信号列表
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return []
        
        try:
            # 计算特征哈希
            feature_hash = self._calculate_feature_hash(features)
            
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # 首先尝试精确哈希匹配
                    cursor.execute("""
                        SELECT 
                            ks.known_id, ks.name, ks.description, ks.signal_type,
                            ks.center_frequency, ks.bandwidth, ks.modulation_type,
                            ks.features, ks.source_confidence
                        FROM known_signals ks
                        WHERE ks.feature_hash = %s 
                        AND ks.is_active = TRUE
                        LIMIT %s
                    """, (feature_hash, limit))
                    
                    results = cursor.fetchall()
                    
                    if results:
                        return results
                    
                    # 如果没有精确匹配，进行特征相似度匹配
                    # 这里使用简单的余弦相似度，实际可以使用更复杂的算法
                    cursor.execute("""
                        SELECT 
                            ks.known_id, ks.name, ks.description, ks.signal_type,
                            ks.center_frequency, ks.bandwidth, ks.modulation_type,
                            ks.features, ks.source_confidence
                        FROM known_signals ks
                        WHERE ks.is_active = TRUE
                        AND ABS(ks.center_frequency - %s) <= %s
                        LIMIT %s
                    """, (
                        features.get('center_freq', 0),
                        features.get('bandwidth', 1000000) * 2,  # 频率容限
                        limit
                    ))
                    
                    results = cursor.fetchall()
                    self.stats['queries_executed'] += 1
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"查询相似信号失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return []
    
    def store_system_event(self, event_type: str, title: str, 
                          description: str = None, data: Dict = None,
                          severity: str = 'info') -> Optional[str]:
        """
        存储系统事件
        
        Args:
            event_type: 事件类型
            title: 事件标题
            description: 事件描述
            data: 事件数据
            severity: 严重程度 (info, warning, error, critical)
            
        Returns:
            str: 事件ID
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return None
        
        event_id = self._generate_id()
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute("""
                        INSERT INTO system_events (
                            event_id, event_type, event_source, severity,
                            title, description, data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event_id,
                        event_type,
                        'system',
                        severity,
                        title,
                        description,
                        json.dumps(data) if data else None
                    ))
                    
                    conn.commit()
                    self.stats['inserts_performed'] += 1
                    self.stats['queries_executed'] += 1
                    
                    self._log_operation("store_event", f"Stored event {event_id}: {title}")
                    return event_id
                    
        except Exception as e:
            self.logger.error(f"存储系统事件失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return None
    
    def get_recent_detections(self, limit: int = 100, 
                             start_time: datetime = None,
                             end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        获取最近的检测结果
        
        Args:
            limit: 返回数量限制
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return []
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if start_time and end_time:
                        cursor.execute("""
                            SELECT d.*, ks.name as known_signal_name
                            FROM detections d
                            LEFT JOIN known_signals ks ON d.known_signal_id = ks.known_id
                            WHERE d.detection_time BETWEEN %s AND %s
                            ORDER BY d.detection_time DESC
                            LIMIT %s
                        """, (start_time, end_time, limit))
                    else:
                        cursor.execute("""
                            SELECT d.*, ks.name as known_signal_name
                            FROM detections d
                            LEFT JOIN known_signals ks ON d.known_signal_id = ks.known_id
                            ORDER BY d.detection_time DESC
                            LIMIT %s
                        """, (limit,))
                    
                    results = cursor.fetchall()
                    self.stats['queries_executed'] += 1
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"获取检测结果失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return []
    
    def get_detection_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Args:
            hours: 统计的小时数
            
        Returns:
            Dict: 统计信息
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return {}
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # 调用存储过程
                    cursor.callproc('sp_get_detection_stats', [
                        datetime.now() - timedelta(hours=hours),
                        datetime.now()
                    ])
                    
                    result = cursor.fetchone()
                    self.stats['queries_executed'] += 1
                    
                    if result:
                        return dict(result)
                    return {}
                    
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
            
        Returns:
            Dict: 清理结果
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return {}
        
        try:
            result = {}
            
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # 清理原始信号
                    cursor.callproc('sp_cleanup_old_data', [days_to_keep, 'raw_signals'])
                    cursor.nextset()
                    result['raw_signals_deleted'] = cursor.fetchone()['rows_deleted']
                    
                    # 清理检测结果
                    cursor.callproc('sp_cleanup_old_data', [days_to_keep, 'detections'])
                    cursor.nextset()
                    result['detections_deleted'] = cursor.fetchone()['rows_deleted']
                    
                    # 清理系统事件
                    cursor.callproc('sp_cleanup_old_data', [days_to_keep, 'system_events'])
                    cursor.nextset()
                    result['events_deleted'] = cursor.fetchone()['rows_deleted']
                    
                    conn.commit()
                    self.stats['queries_executed'] += 3
                    
                    self.logger.info(f"数据清理完成: {result}")
                    return result
                    
        except Exception as e:
            self.logger.error(f"数据清理失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return {}
    
    def update_system_status(self, component: str, status: str, 
                            metrics: Dict[str, Any] = None) -> bool:
        """
        更新系统状态
        
        Args:
            component: 组件名称
            status: 状态 (running, stopped, error, warning)
            metrics: 性能指标
            
        Returns:
            bool: 是否成功
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return False
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute("""
                        INSERT INTO system_status (
                            component, status, message, cpu_usage, memory_usage,
                            disk_usage, network_rx, network_tx
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            status = VALUES(status),
                            message = VALUES(message),
                            cpu_usage = VALUES(cpu_usage),
                            memory_usage = VALUES(memory_usage),
                            disk_usage = VALUES(disk_usage),
                            network_rx = VALUES(network_rx),
                            network_tx = VALUES(network_tx),
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        component,
                        status,
                        metrics.get('message') if metrics else None,
                        metrics.get('cpu_usage') if metrics else None,
                        metrics.get('memory_usage') if metrics else None,
                        metrics.get('disk_usage') if metrics else None,
                        metrics.get('network_rx') if metrics else None,
                        metrics.get('network_tx') if metrics else None
                    ))
                    
                    conn.commit()
                    self.stats['updates_performed'] += 1
                    self.stats['queries_executed'] += 1
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"更新系统状态失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return False
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())
    
    def _calculate_iq_hash(self, iq_data: np.ndarray) -> str:
        """计算IQ数据的哈希值"""
        if iq_data is None:
            return None
        
        # 将IQ数据转换为字节串
        if isinstance(iq_data, np.ndarray):
            data_bytes = iq_data.tobytes()
        else:
            data_bytes = str(iq_data).encode('utf-8')
        
        # 计算SHA256哈希
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _calculate_feature_hash(self, features: Dict[str, Any]) -> str:
        """计算特征哈希"""
        if not features:
            return None
        
        # 将特征字典转换为排序的JSON字符串
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(features_str.encode('utf-8')).hexdigest()
    
    def _log_operation(self, operation: str, message: str):
        """记录操作日志"""
        log_entry = {
            'timestamp': datetime.now(),
            'operation': operation,
            'message': message
        }
        
        self.recent_operations.append(log_entry)
        if len(self.recent_operations) > self.max_recent_ops:
            self.recent_operations.pop(0)
    
    def get_status(self) -> Dict[str, Any]:
        """获取数据库管理器状态"""
        return {
            'initialized': self.initialized,
            'db_type': 'mysql',
            'pool_size': self.pool_size,
            'stats': self.stats.copy(),
            'recent_operations_count': len(self.recent_operations),
            'connection_info': {
                'host': self.connection_params['host'],
                'port': self.connection_params['port'],
                'database': self.connection_params['database']
            }
        }
    
    def execute_raw_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        执行原始SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            List[Dict]: 查询结果
        """
        if not self.initialized:
            self.logger.error("数据库未初始化")
            return []
        
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(query, params or ())
                    self.stats['queries_executed'] += 1
                    
                    if cursor.description:  # 有返回结果
                        return cursor.fetchall()
                    else:  # 无返回结果（如INSERT, UPDATE）
                        conn.commit()
                        return [{'rows_affected': cursor.rowcount}]
                    
        except Exception as e:
            self.logger.error(f"执行原始查询失败: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return []
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            self.logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def shutdown(self):
        """关闭数据库连接池"""
        if self.connection_pool:
            self.connection_pool.close()
            self.logger.info("MySQL 连接池已关闭")
        
        self.initialized = False