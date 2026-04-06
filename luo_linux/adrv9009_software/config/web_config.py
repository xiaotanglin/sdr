"""
信号处理系统Web服务配置文件
提供API接口、WebSocket、静态文件服务配置
"""
import os
import json
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
from datetime import timedelta
import ssl
from urllib.parse import urlparse


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class AuthMethod(str, Enum):
    NONE = "none"              # 无需认证
    BASIC = "basic"            # Basic认证
    JWT = "jwt"                # JWT认证
    OAUTH2 = "oauth2"          # OAuth2认证
    API_KEY = "api_key"        # API密钥认证


class WebSocketMode(str, Enum):
    ASYNC = "async"            # 异步模式
    THREADING = "threading"    # 线程模式
    MULTIPROCESSING = "multiprocessing"  # 多进程模式


class CacheBackend(str, Enum):
    MEMORY = "memory"          # 内存缓存
    REDIS = "redis"            # Redis缓存
    MEMCACHED = "memcached"    # Memcached缓存
    FILESYSTEM = "filesystem"  # 文件系统缓存


class RateLimitAlgorithm(str, Enum):
    FIXED_WINDOW = "fixed_window"     # 固定窗口
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    TOKEN_BUCKET = "token_bucket"     # 令牌桶
    LEAKY_BUCKET = "leaky_bucket"     # 漏桶


@dataclass
class ServerConfig:
    """服务器配置"""
    
    # 服务器监听
    host: str = "0.0.0.0"                 # 监听地址
    port: int = 8000                      # 监听端口
    debug: bool = False                   # 调试模式
    
    # 工作进程
    workers: int = 1                      # 工作进程数
    threads: int = 4                      # 工作线程数
    worker_class: str = "uvicorn.workers.UvicornWorker"  # worker类
    
    # 连接管理
    max_connections: int = 1000           # 最大连接数
    backlog: int = 2048                   # 待处理连接队列长度
    keepalive_timeout: int = 5            # Keep-Alive超时（秒）
    graceful_timeout: int = 30            # 优雅关闭超时（秒）
    
    # 性能优化
    limit_concurrency: Optional[int] = None  # 并发限制
    limit_max_requests: Optional[int] = None  # 最大请求数限制
    timeout: int = 30                     # 请求超时（秒）
    
    # SSL配置
    ssl_enabled: bool = False             # 启用SSL
    ssl_certfile: Optional[str] = None    # SSL证书文件
    ssl_keyfile: Optional[str] = None     # SSL密钥文件
    ssl_keyfile_password: Optional[str] = None  # 密钥密码
    ssl_version: int = ssl.PROTOCOL_TLS   # SSL版本
    
    def get_server_url(self, scheme: str = "http") -> str:
        """获取服务器URL"""
        if scheme == "https" and not self.ssl_enabled:
            raise ValueError("HTTPS需要启用SSL")
        
        if self.host == "0.0.0.0":
            host = "localhost"
        else:
            host = self.host
            
        return f"{scheme}://{host}:{self.port}"
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """获取SSL上下文"""
        if not self.ssl_enabled:
            return None
        
        if not self.ssl_certfile or not self.ssl_keyfile:
            raise ValueError("启用SSL需要提供证书和密钥文件")
        
        ssl_context = ssl.SSLContext(self.ssl_version)
        ssl_context.load_cert_chain(
            certfile=self.ssl_certfile,
            keyfile=self.ssl_keyfile,
            password=self.ssl_keyfile_password
        )
        
        return ssl_context


@dataclass
class CORSConfig:
    """CORS跨域配置"""
    
    enabled: bool = True                  # 启用CORS
    
    # 允许的来源
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_origin_regex: Optional[str] = None
    
    # 允许的方法
    allow_methods: List[HTTPMethod] = field(default_factory=lambda: [
        HTTPMethod.GET,
        HTTPMethod.POST,
        HTTPMethod.PUT,
        HTTPMethod.DELETE,
        HTTPMethod.OPTIONS
    ])
    
    # 允许的头部
    allow_headers: List[str] = field(default_factory=lambda: [
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With"
    ])
    
    # 允许的凭据
    allow_credentials: bool = True
    
    # 暴露的头部
    expose_headers: List[str] = field(default_factory=list)
    
    # 最大年龄
    max_age: int = 600                    # 预检请求缓存时间（秒）
    
    def to_fastapi_kwargs(self) -> Dict[str, Any]:
        """转换为FastAPI CORS参数"""
        return {
            "allow_origins": self.allow_origins,
            "allow_origin_regex": self.allow_origin_regex,
            "allow_methods": [m.value for m in self.allow_methods],
            "allow_headers": self.allow_headers,
            "allow_credentials": self.allow_credentials,
            "expose_headers": self.expose_headers,
            "max_age": self.max_age,
        }


@dataclass
class AuthenticationConfig:
    """认证配置"""
    
    # 认证方法
    method: AuthMethod = AuthMethod.JWT
    
    # JWT配置
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    jwt_token_url: str = "/api/v1/auth/token"
    
    # API密钥配置
    api_key_header: str = "X-API-Key"
    api_key_param: str = "api_key"
    api_keys: List[str] = field(default_factory=list)
    
    # OAuth2配置
    oauth2_token_url: str = "/api/v1/auth/oauth2/token"
    oauth2_authorization_url: str = "/api/v1/auth/oauth2/authorize"
    oauth2_scopes: Dict[str, str] = field(default_factory=dict)
    
    # 会话配置
    session_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    session_cookie_name: str = "session_id"
    session_max_age: int = 3600  # 会话有效期（秒）
    session_same_site: str = "lax"
    session_secure: bool = False
    
    # 用户管理
    require_authentication: bool = True
    default_username: str = "admin"
    default_password: str = "admin123"
    password_hash_algorithm: str = "bcrypt"
    password_hash_rounds: int = 12
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.api_keys:
            # 生成默认API密钥
            self.api_keys = [secrets.token_urlsafe(32) for _ in range(3)]
        
        if not self.oauth2_scopes:
            self.oauth2_scopes = {
                "read:signals": "读取信号数据",
                "write:signals": "写入信号数据",
                "read:detections": "读取检测结果",
                "write:detections": "写入检测结果",
                "admin": "管理员权限"
            }


@dataclass
class RateLimitConfig:
    """限流配置"""
    
    enabled: bool = True                  # 启用限流
    
    # 限流算法
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.FIXED_WINDOW
    
    # 全局限流
    global_rate_limit: str = "100/minute"  # 全局请求限制
    
    # IP限流
    ip_rate_limit: str = "10/second"      # IP请求限制
    ip_whitelist: List[str] = field(default_factory=list)  # IP白名单
    ip_blacklist: List[str] = field(default_factory=list)  # IP黑名单
    
    # 用户限流
    user_rate_limit: str = "60/minute"    # 用户请求限制
    
    # 存储配置
    storage_url: str = "memory://"        # 存储URL
    storage_options: Dict[str, Any] = field(default_factory=dict)
    
    # 错误处理
    retry_after_header: bool = True       # 返回Retry-After头部
    enable_headers: bool = True           # 启用限流头部
    headers_enabled: str = "X-RateLimit"  # 限流头部前缀
    
    def parse_rate_limit(self, rate_limit: str) -> Tuple[int, int]:
        """
        解析速率限制字符串
        
        Args:
            rate_limit: 格式如 "100/minute"
            
        Returns:
            Tuple[int, int]: (请求数, 秒数)
        """
        try:
            count, unit = rate_limit.split("/")
            count = int(count)
            
            if unit == "second":
                seconds = 1
            elif unit == "minute":
                seconds = 60
            elif unit == "hour":
                seconds = 3600
            elif unit == "day":
                seconds = 86400
            else:
                raise ValueError(f"不支持的限流单位: {unit}")
                
            return count, seconds
        except Exception as e:
            raise ValueError(f"无效的限流格式: {rate_limit}") from e


@dataclass
class CacheConfig:
    """缓存配置"""
    
    enabled: bool = True                  # 启用缓存
    
    # 缓存后端
    backend: CacheBackend = CacheBackend.MEMORY
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Memcached配置
    memcached_servers: List[str] = field(default_factory=lambda: ["localhost:11211"])
    
    # 文件系统缓存
    cache_dir: str = "data/cache"
    cache_file_size_limit: int = 1024 * 1024  # 1MB
    
    # 缓存策略
    default_timeout: int = 300            # 默认缓存时间（秒）
    key_prefix: str = "signal_web"        # 缓存键前缀
    
    # 内存缓存
    memory_cache_maxsize: int = 1000      # 最大缓存条目数
    memory_cache_ttl: int = 300           # 内存缓存TTL（秒）
    
    def get_cache_url(self) -> str:
        """获取缓存URL"""
        if self.backend == CacheBackend.REDIS:
            return self.redis_url
        elif self.backend == CacheBackend.MEMCACHED:
            return f"memcached://{','.join(self.memcached_servers)}"
        elif self.backend == CacheBackend.FILESYSTEM:
            return f"file://{self.cache_dir}"
        else:
            return "memory://"


@dataclass
class StaticFilesConfig:
    """静态文件配置"""
    
    # 静态文件目录
    static_dir: str = "src/web/static"
    static_url: str = "/static"
    
    # 静态文件选项
    html: bool = True                     # 提供HTML文件
    check_dir: bool = True                # 检查目录
    follow_symlink: bool = False          # 跟随符号链接
    
    # 缓存控制
    cache_control: str = "public, max-age=3600"  # 缓存控制头部
    expires: int = 3600                   # 过期时间（秒）
    
    # 压缩
    gzip_enabled: bool = True             # 启用Gzip压缩
    gzip_min_size: int = 500              # 最小压缩大小（字节）
    brotli_enabled: bool = False          # 启用Brotli压缩
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if not os.path.exists(self.static_dir):
            errors.append(f"静态文件目录不存在: {self.static_dir}")
            
        if not self.static_url.startswith("/"):
            errors.append("静态文件URL必须以/开头")
            
        return errors


@dataclass
class TemplateConfig:
    """模板配置"""
    
    # 模板目录
    template_dir: str = "src/web/templates"
    
    # 模板引擎
    template_engine: str = "jinja2"       # 模板引擎
    
    # Jinja2配置
    jinja2_autoescape: bool = True        # 自动转义
    jinja2_auto_reload: bool = True       # 自动重载
    jinja2_cache_size: int = 400          # 缓存大小
    jinja2_trim_blocks: bool = True       # 修剪块
    jinja2_lstrip_blocks: bool = True     # 去除左侧空白
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if not os.path.exists(self.template_dir):
            errors.append(f"模板目录不存在: {self.template_dir}")
            
        return errors


@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    
    enabled: bool = True                  # 启用WebSocket
    
    # WebSocket端点
    ws_endpoint: str = "/ws"
    ws_ping_interval: int = 20            # Ping间隔（秒）
    ws_ping_timeout: int = 20             # Ping超时（秒）
    ws_max_size: int = 16 * 1024 * 1024   # 最大消息大小（16MB）
    
    # 连接管理
    ws_max_connections: int = 1000        # 最大连接数
    ws_queue_size: int = 100              # 消息队列大小
    ws_send_timeout: int = 10             # 发送超时（秒）
    
    # 运行模式
    ws_mode: WebSocketMode = WebSocketMode.ASYNC
    
    # 广播配置
    ws_broadcast_enabled: bool = True     # 启用广播
    ws_broadcast_rooms: List[str] = field(default_factory=lambda: [
        "realtime",
        "alerts",
        "status"
    ])
    
    # 重连配置
    ws_reconnect_enabled: bool = True     # 启用重连
    ws_reconnect_timeout: int = 5         # 重连超时（秒）
    ws_max_reconnect_attempts: int = 10   # 最大重连尝试次数


@dataclass
class APIConfig:
    """API配置"""
    
    # API前缀
    api_prefix: str = "/api/v1"
    
    # 文档配置
    docs_enabled: bool = True             # 启用API文档
    docs_url: str = "/docs"               # Swagger UI地址
    redoc_url: str = "/redoc"             # ReDoc地址
    openapi_url: str = "/openapi.json"    # OpenAPI JSON地址
    
    # 版本控制
    version: str = "1.0.0"                # API版本
    title: str = "信号处理系统API"         # API标题
    description: str = "ADRV9009信号处理系统REST API"
    terms_of_service: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    license_name: Optional[str] = "MIT"
    license_url: Optional[str] = None
    
    # API端点
    endpoints: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "signals": {
            "enabled": True,
            "path": "/signals",
            "methods": ["GET", "POST"],
            "require_auth": True,
            "rate_limit": "60/minute"
        },
        "detections": {
            "enabled": True,
            "path": "/detections",
            "methods": ["GET", "POST", "DELETE"],
            "require_auth": True,
            "rate_limit": "100/minute"
        },
        "realtime": {
            "enabled": True,
            "path": "/realtime",
            "methods": ["GET", "POST"],
            "require_auth": True,
            "rate_limit": "10/second"
        },
        "alerts": {
            "enabled": True,
            "path": "/alerts",
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "require_auth": True,
            "rate_limit": "30/minute"
        },
        "system": {
            "enabled": True,
            "path": "/system",
            "methods": ["GET"],
            "require_auth": True,
            "require_admin": True,
            "rate_limit": "10/minute"
        },
        "auth": {
            "enabled": True,
            "path": "/auth",
            "methods": ["POST"],
            "require_auth": False,
            "rate_limit": "5/minute"
        }
    })
    
    # 响应格式
    default_response_format: str = "json"  # 默认响应格式
    supported_formats: List[str] = field(default_factory=lambda: ["json", "xml"])
    
    # 分页配置
    default_page_size: int = 20           # 默认分页大小
    max_page_size: int = 100              # 最大分页大小
    page_size_param: str = "page_size"    # 分页大小参数
    page_param: str = "page"              # 页码参数


@dataclass
class LoggingConfig:
    """日志配置"""
    
    # 日志级别
    level: LogLevel = LogLevel.INFO
    
    # 日志文件
    log_dir: str = "logs/web"
    access_log: str = "access.log"
    error_log: str = "error.log"
    
    # 日志格式
    access_log_format: str = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
    error_log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 日志轮转
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10                 # 备份文件数
    encoding: str = "utf-8"
    
    # 日志处理器
    enable_console: bool = True           # 启用控制台日志
    enable_file: bool = True              # 启用文件日志
    enable_json: bool = False             # 启用JSON格式日志
    
    def get_access_log_path(self) -> str:
        """获取访问日志路径"""
        return os.path.join(self.log_dir, self.access_log)
    
    def get_error_log_path(self) -> str:
        """获取错误日志路径"""
        return os.path.join(self.log_dir, self.error_log)


@dataclass
class MetricsConfig:
    """指标监控配置"""
    
    enabled: bool = True                  # 启用指标监控
    
    # Prometheus配置
    prometheus_enabled: bool = True
    prometheus_endpoint: str = "/metrics"
    prometheus_port: int = 9090
    
    # 健康检查
    health_endpoint: str = "/health"
    health_check_interval: int = 30       # 健康检查间隔（秒）
    
    # 就绪检查
    ready_endpoint: str = "/ready"
    
    # 性能指标
    request_duration_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    ])
    
    # 自定义指标
    custom_metrics: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "http_requests_total",
            "help": "Total HTTP requests",
            "type": "counter"
        },
        {
            "name": "http_request_duration_seconds",
            "help": "HTTP request duration in seconds",
            "type": "histogram"
        },
        {
            "name": "active_websocket_connections",
            "help": "Active WebSocket connections",
            "type": "gauge"
        },
        {
            "name": "signal_processing_requests",
            "help": "Signal processing requests",
            "type": "counter"
        }
    ])


@dataclass
class SecurityConfig:
    """安全配置"""
    
    # 头部安全
    enable_secure_headers: bool = True
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000          # 1年
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False
    
    # 内容安全策略
    csp_enabled: bool = True
    csp_directives: Dict[str, List[str]] = field(default_factory=lambda: {
        "default-src": ["'self'"],
        "script-src": ["'self'", "'unsafe-inline'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "https:"],
        "connect-src": ["'self'"],
        "font-src": ["'self'"],
        "object-src": ["'none'"],
        "media-src": ["'self'"],
        "frame-src": ["'none'"],
    })
    
    # 点击劫持保护
    x_frame_options: str = "DENY"
    
    # 内容类型选项
    x_content_type_options: str = "nosniff"
    
    # XSS保护
    x_xss_protection: str = "1; mode=block"
    
    # 推荐人策略
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    # 权限策略
    permissions_policy: Dict[str, List[str]] = field(default_factory=lambda: {
        "camera": ["none"],
        "microphone": ["none"],
        "geolocation": ["none"],
        "payment": ["none"],
    })
    
    # 请求限制
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    max_upload_size: int = 100 * 1024 * 1024  # 100MB


@dataclass
class WebConfig:
    """Web服务主配置"""
    
    # 环境标识
    environment: str = "development"  # development, testing, production
    
    # 服务器配置
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # CORS配置
    cors: CORSConfig = field(default_factory=CORSConfig)
    
    # 认证配置
    auth: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    
    # 限流配置
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    
    # 缓存配置
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # 静态文件配置
    static: StaticFilesConfig = field(default_factory=StaticFilesConfig)
    
    # 模板配置
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    
    # WebSocket配置
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    
    # API配置
    api: APIConfig = field(default_factory=APIConfig)
    
    # 日志配置
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 指标配置
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # 安全配置
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保日志目录存在
        os.makedirs(self.logging.log_dir, exist_ok=True)
        
        # 确保静态文件目录存在
        os.makedirs(self.static.static_dir, exist_ok=True)
        
        # 确保模板目录存在
        os.makedirs(self.templates.template_dir, exist_ok=True)
        
        # 确保缓存目录存在
        if self.cache.backend == CacheBackend.FILESYSTEM:
            os.makedirs(self.cache.cache_dir, exist_ok=True)
        
        # 根据环境调整配置
        self._adjust_for_environment()
    
    def _adjust_for_environment(self):
        """根据环境调整配置"""
        if self.environment == "production":
            # 生产环境配置
            self.server.debug = False
            self.server.workers = 4
            self.server.threads = 8
            self.server.ssl_enabled = True
            
            # 生产环境使用更强的安全配置
            self.auth.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self.auth.jwt_secret_key)
            self.auth.session_secret_key = os.getenv("SESSION_SECRET_KEY", self.auth.session_secret_key)
            self.auth.session_secure = True
            
            # 生产环境使用Redis缓存
            self.cache.backend = CacheBackend.REDIS
            self.cache.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            # 生产环境更严格的限流
            self.rate_limit.global_rate_limit = "1000/minute"
            self.rate_limit.ip_rate_limit = "5/second"
            
            # 生产环境日志配置
            self.logging.level = LogLevel.INFO
            self.logging.log_dir = "/var/log/signal_processing/web"
            
        elif self.environment == "testing":
            # 测试环境配置
            self.server.debug = True
            self.server.port = 8001
            
            # 测试环境禁用认证
            self.auth.require_authentication = False
            
            # 测试环境禁用限流
            self.rate_limit.enabled = False
            
            # 测试环境禁用缓存
            self.cache.enabled = False
            
        else:  # development
            # 开发环境配置
            self.server.debug = True
            self.server.host = "127.0.0.1"
            self.auth.require_authentication = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # 验证服务器配置
        if not 1 <= self.server.port <= 65535:
            errors.append("服务器端口必须在1-65535之间")
            
        if self.server.ssl_enabled and (not self.server.ssl_certfile or not self.server.ssl_keyfile):
            errors.append("启用SSL需要提供证书和密钥文件")
        
        # 验证静态文件配置
        static_errors = self.static.validate()
        errors.extend(static_errors)
        
        # 验证模板配置
        template_errors = self.templates.validate()
        errors.extend(template_errors)
        
        # 验证缓存配置
        if self.cache.enabled and self.cache.backend == CacheBackend.FILESYSTEM:
            if not os.path.exists(self.cache.cache_dir):
                try:
                    os.makedirs(self.cache.cache_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"无法创建缓存目录: {e}")
        
        return len(errors) == 0, errors
    
    def get_fastapi_config(self) -> Dict[str, Any]:
        """获取FastAPI配置"""
        return {
            "title": self.api.title,
            "description": self.api.description,
            "version": self.api.version,
            "docs_url": self.api.docs_url if self.api.docs_enabled else None,
            "redoc_url": self.api.redoc_url if self.api.docs_enabled else None,
            "openapi_url": self.api.openapi_url if self.api.docs_enabled else None,
            "debug": self.server.debug,
        }
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """获取Uvicorn配置"""
        config = {
            "host": self.server.host,
            "port": self.server.port,
            "debug": self.server.debug,
            "reload": self.server.debug,  # 开发环境自动重载
            "workers": self.server.workers,
            "log_level": self.logging.level.lower(),
            "access_log": self.logging.enable_file,
            "timeout_keep_alive": self.server.keepalive_timeout,
            "limit_concurrency": self.server.limit_concurrency,
            "limit_max_requests": self.server.limit_max_requests,
        }
        
        if self.server.ssl_enabled:
            config.update({
                "ssl_keyfile": self.server.ssl_keyfile,
                "ssl_certfile": self.server.ssl_certfile,
                "ssl_keyfile_password": self.server.ssl_keyfile_password,
            })
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（排除敏感信息）"""
        data = asdict(self)
        
        # 隐藏敏感信息
        if 'auth' in data:
            if 'jwt_secret_key' in data['auth']:
                data['auth']['jwt_secret_key'] = '***'
            if 'session_secret_key' in data['auth']:
                data['auth']['session_secret_key'] = '***'
            if 'api_keys' in data['auth']:
                data['auth']['api_keys'] = ['***' for _ in data['auth']['api_keys']]
                
        if 'server' in data and 'ssl_keyfile_password' in data['server']:
            data['server']['ssl_keyfile_password'] = '***'
            
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
    def load_from_file(cls, filepath: str) -> 'WebConfig':
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
    def from_dict(cls, data: Dict[str, Any]) -> 'WebConfig':
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
def get_default_config(environment: str = None) -> WebConfig:
    """
    获取默认配置
    
    Args:
        environment: 环境标识
        
    Returns:
        WebConfig: 默认配置
    """
    config = WebConfig()
    
    if environment:
        config.environment = environment
        config._adjust_for_environment()
    
    return config


def get_production_config() -> WebConfig:
    """获取生产环境配置"""
    config = WebConfig(environment="production")
    return config


def get_development_config() -> WebConfig:
    """获取开发环境配置"""
    config = WebConfig(environment="development")
    return config


def get_testing_config() -> WebConfig:
    """获取测试环境配置"""
    config = WebConfig(environment="testing")
    return config


def get_config(environment: str = None) -> WebConfig:
    """
    获取当前环境配置
    
    Args:
        environment: 环境标识，如果为None则从环境变量读取
        
    Returns:
        WebConfig: Web服务配置
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
        config_dir / f"web_config_{env}.yaml",
        config_dir / f"web_config_{env}.json",
        config_dir / "web_config.yaml",
        config_dir / "web_config.json",
    ]
    
    for config_file in config_files:
        if config_file.exists():
            return str(config_file)
    
    return str(config_dir / f"web_config_{env}.yaml")


def load_config(config_file: str = None) -> WebConfig:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径，如果为None则自动查找
        
    Returns:
        WebConfig: Web服务配置
    """
    if config_file is None:
        config_file = get_config_path()
    
    if os.path.exists(config_file):
        try:
            return WebConfig.load_from_file(config_file)
        except Exception as e:
            logging.warning(f"加载配置文件失败 {config_file}: {e}")
    
    # 使用默认配置
    env = os.getenv("ENVIRONMENT", "development")
    return get_config(env)


def save_config(config: WebConfig, config_file: str = None, format: str = "yaml"):
    """
    保存配置
    
    Args:
        config: Web服务配置
        config_file: 配置文件路径
        format: 文件格式 (json/yaml)
    """
    if config_file is None:
        config_file = get_config_path()
    
    config.save_to_file(config_file, format)


# 当前配置实例
_current_config: Optional[WebConfig] = None

def get_current_config() -> WebConfig:
    """获取当前配置实例"""
    global _current_config
    if _current_config is None:
        _current_config = load_config()
    return _current_config


# 工具函数
def generate_ssl_certificates(cert_dir: str = "config/ssl", 
                            common_name: str = "localhost",
                            days_valid: int = 365) -> Tuple[str, str]:
    """
    生成自签名SSL证书
    
    Args:
        cert_dir: 证书目录
        common_name: 通用名称
        days_valid: 有效天数
        
    Returns:
        Tuple[str, str]: (证书文件路径, 密钥文件路径)
    """
    import subprocess
    import tempfile
    
    os.makedirs(cert_dir, exist_ok=True)
    
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file
    
    # 生成私钥
    subprocess.run([
        "openssl", "genrsa", "-out", key_file, "2048"
    ], check=True)
    
    # 生成证书签名请求
    csr_file = os.path.join(cert_dir, "csr.pem")
    subprocess.run([
        "openssl", "req", "-new", "-key", key_file, "-out", csr_file,
        "-subj", f"/C=CN/ST=Beijing/L=Beijing/O=SignalProcessing/CN={common_name}"
    ], check=True)
    
    # 生成自签名证书
    subprocess.run([
        "openssl", "x509", "-req", "-days", str(days_valid), 
        "-in", csr_file, "-signkey", key_file, "-out", cert_file
    ], check=True)
    
    # 清理临时文件
    if os.path.exists(csr_file):
        os.remove(csr_file)
    
    return cert_file, key_file


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
    print("\nWeb服务配置:")
    print(config.to_yaml())
    
    # 获取服务器URL
    scheme = "https" if config.server.ssl_enabled else "http"
    try:
        server_url = config.server.get_server_url(scheme)
        print(f"\n服务器地址: {server_url}")
        
        if config.api.docs_enabled:
            docs_url = f"{server_url}{config.api.docs_url}"
            print(f"API文档: {docs_url}")
            
        if config.metrics.enabled and config.metrics.prometheus_enabled:
            metrics_url = f"{server_url}{config.metrics.prometheus_endpoint}"
            print(f"指标监控: {metrics_url}")
            
    except ValueError as e:
        print(f"\n服务器URL错误: {e}")
    
    # 保存配置
    config_path = get_config_path()
    print(f"\n保存配置到: {config_path}")
    save_config(config, config_path)