"""
信号处理系统数据库配置文件 - MySQL专用配置
针对ADRV9009信号处理系统优化
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
from datetime import datetime, timedelta
import hashlib
import secrets


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConnectionPoolType(str, Enum):
    SIMPLE = "simple"      # 简单连接池
    QUEUE = "queue"        # 队列连接池
    PERSISTENT = "persistent"  # 持久连接


class BackupType(str, Enum):
    FULL = "full"          # 完整备份
    INCREMENTAL = "incremental"  # 增量备份
    LOGICAL = "logical"    # 逻辑备份
    PHYSICAL = "physical"  # 物理备份


class IndexType(str, Enum):
    BTREE = "BTREE"
    HASH = "HASH"
    FULLTEXT = "FULLTEXT"
    SPATIAL = "SPATIAL"


class PartitionType(str, Enum):
    RANGE = "RANGE"
    LIST = "LIST"
    HASH = "HASH"
    KEY = "KEY"


@dataclass
class MySQLConnectionConfig:
    """MySQL连接配置"""
    
    # 基本连接参数
    host: str = "localhost"
    port: int = 3306
    username: str = "signal_user"
    password: str = ""
    database: str = "signal_processing"
    
    # 连接选项
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"
    use_unicode: bool = True
    connect_timeout: int = 10           # 连接超时（秒）
    read_timeout: int = 30              # 读取超时（秒）
    write_timeout: int = 30             # 写入超时（秒）
    client_flag: int = 0
    
    # SSL配置
    ssl_enabled: bool = False
    ssl_ca: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_verify_cert: bool = False
    
    # 连接行为
    autocommit: bool = False
    buffered: bool = True
    cursorclass: str = "DictCursor"
    
    def get_connection_params(self) -> Dict[str, Any]:
        """获取连接参数字典"""
        params = {
            'host': self.host,
            'port': self.port,
            'user': self.username,
            'password': self.password,
            'database': self.database,
            'charset': self.charset,
            'use_unicode': self.use_unicode,
            'connect_timeout': self.connect_timeout,
            'read_timeout': self.read_timeout,
            'write_timeout': self.write_timeout,
            'autocommit': self.autocommit,
        }
        
        if self.client_flag:
            params['client_flag'] = self.client_flag
            
        if self.ssl_enabled and self.ssl_ca:
            params['ssl'] = {
                'ca': self.ssl_ca,
                'cert': self.ssl_cert,
                'key': self.ssl_key,
                'check_hostname': self.ssl_verify_cert
            }
            
        return params


@dataclass
class ConnectionPoolConfig:
    """连接池配置"""
    
    # 连接池类型
    pool_type: ConnectionPoolType = ConnectionPoolType.QUEUE
    
    # 连接池大小
    pool_size: int = 20                    # 连接池最大连接数
    max_overflow: int = 10                 # 最大溢出连接数
    pool_timeout: int = 30                 # 获取连接超时（秒）
    
    # 连接管理
    pool_recycle: int = 3600               # 连接回收时间（秒）
    pool_pre_ping: bool = True             # 连接前ping检查
    pool_reset_on_return: str = "rollback"  # 连接返回时重置
    
    # 连接生命周期
    max_lifetime: int = 7200               # 连接最大生命周期（秒）
    idle_timeout: int = 600                # 空闲超时（秒）
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.pool_size <= 0:
            errors.append("连接池大小必须大于0")
            
        if self.max_overflow < 0:
            errors.append("最大溢出连接数不能为负数")
            
        if self.pool_timeout <= 0:
            errors.append("连接池超时时间必须大于0")
            
        if self.pool_recycle <= 0:
            errors.append("连接回收时间必须大于0")
            
        if self.max_lifetime <= 0:
            errors.append("连接最大生命周期必须大于0")
            
        if self.idle_timeout <= 0:
            errors.append("空闲超时时间必须大于0")
            
        return errors


@dataclass
class QueryLoggingConfig:
    """查询日志配置"""
    
    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    slow_query_threshold_ms: int = 1000    # 慢查询阈值（毫秒）
    
    # 日志文件配置
    log_dir: str = "logs/db"
    log_file: str = "mysql_queries.log"
    max_file_size_mb: int = 100            # 最大文件大小（MB）
    backup_count: int = 10                 # 备份文件数量
    
    # 日志内容
    log_connections: bool = True           # 记录连接信息
    log_disconnections: bool = True        # 记录断开连接
    log_queries: bool = True               # 记录查询
    log_errors: bool = True                # 记录错误
    log_slow_queries: bool = True          # 记录慢查询
    log_transactions: bool = False         # 记录事务
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return os.path.join(self.log_dir, self.log_file)


@dataclass
class DatabaseRetryConfig:
    """数据库重试配置"""
    
    enabled: bool = True
    max_retries: int = 3                   # 最大重试次数
    retry_delay: float = 1.0               # 重试延迟（秒）
    exponential_backoff: bool = True       # 指数退避
    backoff_factor: float = 2.0            # 退避因子
    
    # 重试的错误类型
    retry_on_timeout: bool = True          # 超时重试
    retry_on_deadlock: bool = True         # 死锁重试
    retry_on_lost_connection: bool = True  # 连接丢失重试
    
    # 重试的错误码
    retry_error_codes: List[int] = field(default_factory=lambda: [
        1205,  # ER_LOCK_WAIT_TIMEOUT
        1213,  # ER_LOCK_DEADLOCK
        2006,  # CR_SERVER_GONE_ERROR
        2013,  # CR_SERVER_LOST
    ])
    
    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟时间"""
        if not self.exponential_backoff:
            return self.retry_delay
        return self.retry_delay * (self.backoff_factor ** (attempt - 1))


@dataclass
class TablePartitionConfig:
    """表分区配置"""
    
    # 原始信号表分区
    raw_signals_partition: bool = True
    raw_signals_partition_type: PartitionType = PartitionType.RANGE
    raw_signals_partition_key: str = "DATE(created_at)"
    raw_signals_partition_interval: str = "DAY"  # DAY/MONTH/YEAR
    raw_signals_partition_count: int = 30        # 保留30个分区
    
    # 检测结果表分区
    detections_partition: bool = True
    detections_partition_type: PartitionType = PartitionType.RANGE
    detections_partition_key: str = "DATE(created_at)"
    detections_partition_interval: str = "DAY"
    detections_partition_count: int = 90
    
    # 系统事件表分区
    system_events_partition: bool = True
    system_events_partition_type: PartitionType = PartitionType.RANGE
    system_events_partition_key: str = "DATE(created_at)"
    system_events_partition_interval: str = "MONTH"
    system_events_partition_count: int = 12


@dataclass
class IndexConfig:
    """索引配置"""
    
    # 原始信号表索引
    raw_signals_indexes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "idx_timestamp", "columns": ["timestamp"], "type": "BTREE"},
        {"name": "idx_frequency", "columns": ["center_frequency"], "type": "BTREE"},
        {"name": "idx_device", "columns": ["device_id"], "type": "BTREE"},
        {"name": "idx_created_at", "columns": ["created_at"], "type": "BTREE"},
        {"name": "idx_timestamp_frequency", "columns": ["timestamp", "center_frequency"], "type": "BTREE"},
    ])
    
    # 检测结果表索引
    detections_indexes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "idx_detection_time", "columns": ["detection_time"], "type": "BTREE"},
        {"name": "idx_frequency", "columns": ["center_frequency"], "type": "BTREE"},
        {"name": "idx_type", "columns": ["detection_type"], "type": "BTREE"},
        {"name": "idx_confidence", "columns": ["confidence"], "type": "BTREE"},
        {"name": "idx_is_known", "columns": ["is_known"], "type": "BTREE"},
        {"name": "idx_created_at", "columns": ["created_at"], "type": "BTREE"},
        {"name": "idx_time_frequency", "columns": ["detection_time", "center_frequency"], "type": "BTREE"},
        {"name": "idx_type_confidence", "columns": ["detection_type", "confidence"], "type": "BTREE"},
    ])
    
    # 已知信号表索引
    known_signals_indexes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "idx_frequency_range", "columns": ["frequency_range_low", "frequency_range_high"], "type": "BTREE"},
        {"name": "idx_signal_type", "columns": ["signal_type"], "type": "BTREE"},
        {"name": "idx_is_active", "columns": ["is_active"], "type": "BTREE"},
        {"name": "idx_name", "columns": ["name"], "type": "BTREE"},
    ])
    
    # 系统事件表索引
    system_events_indexes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "idx_event_type", "columns": ["event_type"], "type": "BTREE"},
        {"name": "idx_severity", "columns": ["severity"], "type": "BTREE"},
        {"name": "idx_created_at", "columns": ["created_at"], "type": "BTREE"},
        {"name": "idx_acknowledged", "columns": ["acknowledged"], "type": "BTREE"},
        {"name": "idx_resolved", "columns": ["resolved"], "type": "BTREE"},
    ])


@dataclass
class DataRetentionConfig:
    """数据保留配置"""
    
    # 原始信号数据保留
    raw_signals_retention_days: int = 30           # 保留30天
    raw_signals_auto_cleanup: bool = True          # 自动清理
    raw_signals_cleanup_batch_size: int = 1000     # 清理批次大小
    raw_signals_cleanup_interval_hours: int = 6    # 清理间隔（小时）
    
    # 检测结果保留
    detections_retention_days: int = 90            # 保留90天
    detections_auto_cleanup: bool = True
    detections_cleanup_batch_size: int = 1000
    detections_cleanup_interval_hours: int = 12
    
    # 系统事件保留
    system_events_retention_days: int = 180        # 保留180天
    system_events_auto_cleanup: bool = True
    system_events_cleanup_batch_size: int = 100
    system_events_cleanup_interval_hours: int = 24
    
    # 算法结果保留
    algorithm_results_retention_days: int = 7      # 保留7天
    algorithm_results_auto_cleanup: bool = True
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.raw_signals_retention_days <= 0:
            errors.append("原始信号保留天数必须大于0")
            
        if self.detections_retention_days <= 0:
            errors.append("检测结果保留天数必须大于0")
            
        if self.system_events_retention_days <= 0:
            errors.append("系统事件保留天数必须大于0")
            
        if self.algorithm_results_retention_days <= 0:
            errors.append("算法结果保留天数必须大于0")
            
        return errors


@dataclass
class BackupConfig:
    """备份配置"""
    
    # 备份类型
    backup_type: BackupType = BackupType.INCREMENTAL
    
    # 备份计划
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # cron表达式，每天凌晨2点
    backup_retention_days: int = 30      # 备份保留天数
    
    # 备份目录
    backup_dir: str = "data/backups"
    backup_prefix: str = "signal_db_backup"
    
    # 压缩和加密
    compress_backup: bool = True
    compression_level: int = 6
    encrypt_backup: bool = False
    encryption_key: Optional[str] = None
    
    # 备份内容
    backup_schema: bool = True           # 备份表结构
    backup_data: bool = True             # 备份数据
    backup_triggers: bool = True         # 备份触发器
    backup_routines: bool = True         # 备份存储过程
    
    def get_backup_file_name(self, timestamp: datetime = None) -> str:
        """获取备份文件名"""
        if timestamp is None:
            timestamp = datetime.now()
        
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        suffix = ".sql.gz" if self.compress_backup else ".sql"
        
        return f"{self.backup_prefix}_{date_str}{suffix}"
    
    def get_backup_path(self, timestamp: datetime = None) -> str:
        """获取备份文件路径"""
        filename = self.get_backup_file_name(timestamp)
        return os.path.join(self.backup_dir, filename)


@dataclass
class PerformanceConfig:
    """性能配置"""
    
    # 查询缓存
    query_cache_enabled: bool = True
    query_cache_size_mb: int = 64
    query_cache_limit_mb: int = 2
    query_cache_min_res_unit: int = 4096
    
    # InnoDB缓冲池
    innodb_buffer_pool_size_mb: int = 1024
    innodb_buffer_pool_instances: int = 8
    innodb_log_file_size_mb: int = 512
    
    # 连接配置
    max_connections: int = 200
    max_user_connections: int = 50
    thread_cache_size: int = 8
    
    # 超时配置
    wait_timeout: int = 28800      # 8小时
    interactive_timeout: int = 28800
    net_read_timeout: int = 30
    net_write_timeout: int = 30
    
    # 事务配置
    transaction_isolation: str = "REPEATABLE-READ"
    innodb_flush_log_at_trx_commit: int = 2
    sync_binlog: int = 0
    
    # 临时表
    tmp_table_size_mb: int = 64
    max_heap_table_size_mb: int = 64


@dataclass
class MonitoringConfig:
    """监控配置"""
    
    # 监控启用
    enabled: bool = True
    monitor_interval_seconds: int = 60
    
    # 监控指标
    monitor_connections: bool = True
    monitor_queries: bool = True
    monitor_slow_queries: bool = True
    monitor_locks: bool = True
    monitor_buffer_pool: bool = True
    
    # 告警阈值
    connection_usage_threshold: float = 0.8    # 80%
    slow_query_threshold_ms: int = 1000
    lock_wait_threshold_ms: int = 5000
    buffer_pool_usage_threshold: float = 0.9   # 90%
    
    # 指标保留
    metrics_retention_days: int = 7
    metrics_aggregation_hours: int = 1
    
    def get_monitoring_queries(self) -> Dict[str, str]:
        """获取监控查询语句"""
        return {
            'connections': """
                SELECT 
                    COUNT(*) as total_connections,
                    COUNT(CASE WHEN COMMAND != 'Sleep' THEN 1 END) as active_connections,
                    MAX(TIME) as max_query_time
                FROM information_schema.PROCESSLIST
            """,
            'slow_queries': """
                SELECT 
                    COUNT(*) as slow_queries,
                    AVG(query_time) as avg_slow_time,
                    MAX(query_time) as max_slow_time
                FROM mysql.slow_log
                WHERE start_time > DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """,
            'buffer_pool': """
                SELECT 
                    SUM(DATA_LENGTH + INDEX_LENGTH) as total_size,
                    SUM(DATA_LENGTH) as data_size,
                    SUM(INDEX_LENGTH) as index_size
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
            """,
            'table_sizes': """
                SELECT 
                    TABLE_NAME,
                    TABLE_ROWS,
                    DATA_LENGTH,
                    INDEX_LENGTH,
                    DATA_LENGTH + INDEX_LENGTH as TOTAL_SIZE
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TOTAL_SIZE DESC
                LIMIT 10
            """
        }


@dataclass
class SecurityConfig:
    """安全配置"""
    
    # 密码策略
    password_validation_enabled: bool = True
    password_lifetime_days: int = 90
    password_history_count: int = 5
    password_min_length: int = 12
    
    # 访问控制
    bind_address: str = "127.0.0.1"
    skip_name_resolve: bool = True
    local_infile: bool = False
    
    # SSL/TLS
    require_secure_transport: bool = False
    tls_version: str = "TLSv1.2,TLSv1.3"
    
    # 审计日志
    audit_log_enabled: bool = False
    audit_log_format: str = "JSON"
    audit_log_file: str = "logs/mysql_audit.log"
    
    # 用户权限
    minimal_privileges: bool = True
    revoke_public_privileges: bool = True


@dataclass
class DatabaseConfig:
    """数据库主配置"""
    
    # 环境标识
    environment: str = "development"  # development, testing, production
    
    # 数据库类型
    db_type: str = "mysql"
    
    # 连接配置
    connection: MySQLConnectionConfig = field(default_factory=MySQLConnectionConfig)
    
    # 连接池配置
    pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    
    # 查询日志配置
    query_logging: QueryLoggingConfig = field(default_factory=QueryLoggingConfig)
    
    # 重试配置
    retry: DatabaseRetryConfig = field(default_factory=DatabaseRetryConfig)
    
    # 表分区配置
    partitions: TablePartitionConfig = field(default_factory=TablePartitionConfig)
    
    # 索引配置
    indexes: IndexConfig = field(default_factory=IndexConfig)
    
    # 数据保留配置
    retention: DataRetentionConfig = field(default_factory=DataRetentionConfig)
    
    # 备份配置
    backup: BackupConfig = field(default_factory=BackupConfig)
    
    # 性能配置
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 监控配置
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 安全配置
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保日志目录存在
        os.makedirs(self.query_logging.log_dir, exist_ok=True)
        
        # 确保备份目录存在
        os.makedirs(self.backup.backup_dir, exist_ok=True)
        
        # 根据环境调整配置
        self._adjust_for_environment()
    
    def _adjust_for_environment(self):
        """根据环境调整配置"""
        if self.environment == "production":
            # 生产环境配置
            self.connection.host = os.getenv("DB_HOST", "localhost")
            self.connection.username = os.getenv("DB_USER", "signal_user")
            self.connection.password = os.getenv("DB_PASSWORD", "")
            self.connection.database = os.getenv("DB_NAME", "signal_processing")
            
            # 生产环境使用连接池
            self.pool.pool_size = 50
            self.pool.max_overflow = 20
            
            # 生产环境更严格的保留策略
            self.retention.raw_signals_retention_days = 7
            self.retention.detections_retention_days = 30
            
            # 生产环境启用SSL
            self.security.require_secure_transport = True
            self.connection.ssl_enabled = True
            
        elif self.environment == "testing":
            # 测试环境配置
            self.connection.database = f"test_{self.connection.database}"
            self.pool.pool_size = 5
            self.pool.max_overflow = 2
            
            # 测试环境不保留数据
            self.retention.raw_signals_retention_days = 1
            self.retention.detections_retention_days = 1
            
        else:  # development
            # 开发环境配置
            self.connection.host = "localhost"
            self.connection.username = "root"
            self.connection.password = ""
            self.pool.pool_size = 10
            self.pool.max_overflow = 5
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # 验证连接配置
        if not self.connection.host:
            errors.append("数据库主机不能为空")
            
        if not 1 <= self.connection.port <= 65535:
            errors.append("数据库端口必须在1-65535之间")
            
        if not self.connection.username:
            errors.append("数据库用户名不能为空")
            
        if not self.connection.database:
            errors.append("数据库名称不能为空")
            
        # 验证连接池配置
        pool_errors = self.pool.validate()
        errors.extend(pool_errors)
        
        # 验证数据保留配置
        retention_errors = self.retention.validate()
        errors.extend(retention_errors)
        
        # 验证备份配置
        if self.backup.backup_enabled and not self.backup.backup_dir:
            errors.append("备份目录不能为空")
            
        if self.backup.encrypt_backup and not self.backup.encryption_key:
            errors.append("启用备份加密时必须提供加密密钥")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（排除敏感信息）"""
        data = asdict(self)
        
        # 隐藏密码
        if 'connection' in data and 'password' in data['connection']:
            data['connection']['password'] = '***'
            
        if 'backup' in data and 'encryption_key' in data['backup']:
            data['backup']['encryption_key'] = '***'
            
        return data
    
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
    def load_from_file(cls, filepath: str) -> 'DatabaseConfig':
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
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConfig':
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


# 默认配置实例
def get_default_config(environment: str = None) -> DatabaseConfig:
    """
    获取默认配置
    
    Args:
        environment: 环境标识
        
    Returns:
        DatabaseConfig: 默认配置
    """
    config = DatabaseConfig()
    
    if environment:
        config.environment = environment
        config._adjust_for_environment()
    
    return config


# 生产环境配置
def get_production_config() -> DatabaseConfig:
    """获取生产环境配置"""
    config = DatabaseConfig(environment="production")
    return config


# 开发环境配置
def get_development_config() -> DatabaseConfig:
    """获取开发环境配置"""
    config = DatabaseConfig(environment="development")
    return config


# 测试环境配置
def get_testing_config() -> DatabaseConfig:
    """获取测试环境配置"""
    config = DatabaseConfig(environment="testing")
    return config


# 当前环境配置
def get_config(environment: str = None) -> DatabaseConfig:
    """
    获取当前环境配置
    
    Args:
        environment: 环境标识，如果为None则从环境变量读取
        
    Returns:
        DatabaseConfig: 数据库配置
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return get_production_config()
    elif environment == "testing":
        return get_testing_config()
    else:
        return get_development_config()


# 配置文件路径
def get_config_path() -> str:
    """获取配置文件路径"""
    env = os.getenv("ENVIRONMENT", "development")
    config_dir = Path(__file__).parent
    
    # 查找配置文件
    config_files = [
        config_dir / f"db_config_{env}.yaml",
        config_dir / f"db_config_{env}.json",
        config_dir / "db_config.yaml",
        config_dir / "db_config.json",
    ]
    
    for config_file in config_files:
        if config_file.exists():
            return str(config_file)
    
    return str(config_dir / f"db_config_{env}.yaml")


# 加载配置
def load_config(config_file: str = None) -> DatabaseConfig:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径，如果为None则自动查找
        
    Returns:
        DatabaseConfig: 数据库配置
    """
    if config_file is None:
        config_file = get_config_path()
    
    if os.path.exists(config_file):
        try:
            return DatabaseConfig.load_from_file(config_file)
        except Exception as e:
            logging.warning(f"加载配置文件失败 {config_file}: {e}")
    
    # 使用默认配置
    env = os.getenv("ENVIRONMENT", "development")
    return get_config(env)


# 保存配置
def save_config(config: DatabaseConfig, config_file: str = None, format: str = "yaml"):
    """
    保存配置
    
    Args:
        config: 数据库配置
        config_file: 配置文件路径
        format: 文件格式 (json/yaml)
    """
    if config_file is None:
        config_file = get_config_path()
    
    config.save_to_file(config_file, format)


# 当前配置实例
_current_config: Optional[DatabaseConfig] = None

def get_current_config() -> DatabaseConfig:
    """获取当前配置实例"""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config


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
    
    # 打印配置
    print("\n数据库配置:")
    print(config.to_yaml())
    
    # 保存配置
    config_path = get_config_path()
    print(f"\n保存配置到: {config_path}")
    save_config(config, config_path)
    
    # 测试连接参数
    print("\n连接参数:")
    print(json.dumps(config.connection.get_connection_params(), indent=2))