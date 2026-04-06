"""
Web应用模块
创建FastAPI应用，配置中间件、路由、异常处理等
"""
import os
import sys
import json
import time
import traceback
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
import warnings

from fastapi import FastAPI, Request, Response, status, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import make_asgi_app, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import uvicorn
from sqlalchemy.orm import Session

# 导入配置
from config.web_config import get_current_config, WebConfig
from config.logging_config import get_current_config as get_logging_config
from config.app_config import get_current_config as get_app_config
from config.db_config import get_current_config as get_db_config

# 导入工具
from src.utils.data_validator import DataValidator
from src.utils.file_utils import ensure_directory, get_file_info
from src.utils.performance_monitor import PerformanceMonitor

# 导入认证
from src.web.auth import (
    create_access_token, 
    verify_access_token,
    get_current_user,
    get_current_active_user,
    get_current_admin_user
)

# 导入数据库
from src.core.database_manager import DatabaseManager
from src.models.database_models import SessionLocal, Base, RawSignal, Detection, KnownSignal

# 导入API路由
from src.web.api import api_router
from src.web.api.v1 import (
    auth_router,
    signals_router,
    detections_router,
    algorithms_router,
    system_router,
    events_router,
    files_router
)

# 导入WebSocket管理器
from src.web.websocket_manager import WebSocketManager

# 导入中间件
from src.web.middleware import (
    LoggingMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    ExceptionHandlerMiddleware
)

# 获取日志记录器
logger = logging.getLogger(__name__)


class WebApp:
    """Web应用管理器"""
    
    def __init__(self, config: WebConfig = None, system=None):
        """
        初始化Web应用
        
        Args:
            config: Web配置
            system: 信号处理系统实例
        """
        self.config = config or get_current_config()
        self.system = system
        
        # 应用实例
        self.app = None
        
        # 组件
        self.templates = None
        self.websocket_manager = None
        self.performance_monitor = None
        self.validator = None
        
        # 数据库
        self.database = None
        self.session_local = None
        
        # 限流器
        self.limiter = None
        
        # 指标
        self.metrics_app = None
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化Web应用"""
        logger.info("初始化Web应用...")
        
        # 创建必要的目录
        self._create_directories()
        
        # 创建模板引擎
        self.templates = self._create_templates()
        
        # 创建数据验证器
        self.validator = DataValidator(level="warning")
        
        # 创建性能监控
        if self.config.metrics.enabled:
            self.performance_monitor = PerformanceMonitor()
        
        # 创建WebSocket管理器
        if self.config.websocket.enabled:
            self.websocket_manager = WebSocketManager(
                config=self.config.websocket,
                event_manager=getattr(self.system, 'event_manager', None) if self.system else None
            )
        
        logger.info("Web应用初始化完成")
    
    def _create_directories(self):
        """创建必要目录"""
        # 静态文件目录
        ensure_directory(self.config.static.static_dir)
        
        # 模板目录
        ensure_directory(self.config.templates.template_dir)
        
        # 上传目录
        upload_dir = Path(self.config.static.static_dir) / "uploads"
        ensure_directory(upload_dir)
    
    def _create_templates(self) -> Jinja2Templates:
        """创建模板引擎"""
        return Jinja2Templates(
            directory=self.config.templates.template_dir,
            autoescape=self.config.templates.jinja2_autoescape,
            auto_reload=self.config.templates.jinja2_auto_reload
        )
    
    def create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        logger.info("创建FastAPI应用...")
        
        # 应用生命周期
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """应用生命周期管理"""
            # 启动
            logger.info("Web应用启动中...")
            
            # 初始化数据库
            if self.system and hasattr(self.system, 'database'):
                self.database = self.system.database
                self.session_local = self.system.orm_database.SessionLocal
            else:
                self._init_database()
            
            # 启动WebSocket管理器
            if self.websocket_manager:
                await self.websocket_manager.start()
            
            # 启动性能监控
            if self.performance_monitor:
                self.performance_monitor.start()
            
            logger.info("Web应用启动完成")
            yield
            
            # 关闭
            logger.info("Web应用关闭中...")
            
            # 关闭WebSocket管理器
            if self.websocket_manager:
                await self.websocket_manager.stop()
            
            # 关闭性能监控
            if self.performance_monitor:
                self.performance_monitor.stop()
            
            # 关闭数据库连接
            if self.session_local:
                self.session_local.remove()
            
            logger.info("Web应用关闭完成")
        
        # 创建FastAPI应用
        app_config = self.config.get_fastapi_config()
        self.app = FastAPI(
            title=app_config["title"],
            description=app_config["description"],
            version=app_config["version"],
            docs_url=app_config["docs_url"],
            redoc_url=app_config["redoc_url"],
            openapi_url=app_config["openapi_url"],
            lifespan=lifespan
        )
        
        # 配置应用
        self._configure_app(self.app)
        
        # 添加中间件
        self._add_middleware(self.app)
        
        # 添加路由
        self._add_routes(self.app)
        
        # 添加异常处理器
        self._add_exception_handlers(self.app)
        
        # 自定义OpenAPI
        self._customize_openapi(self.app)
        
        logger.info("FastAPI应用创建完成")
        return self.app
    
    def _init_database(self):
        """初始化数据库"""
        if not self.config.api.database.enabled:
            return
        
        try:
            db_config = get_db_config()
            db_url = self._get_database_url(db_config)
            
            # 创建数据库管理器
            self.database = DatabaseManager(db_url)
            self.session_local = self.database.SessionLocal
            
            # 测试连接
            self.database.test_connection()
            logger.info("数据库连接成功")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            if self.config.api.database.required:
                raise
            else:
                logger.warning("数据库初始化失败，但Web应用将继续运行（某些功能可能受限）")
    
    def _get_database_url(self, db_config) -> str:
        """获取数据库URL"""
        conn = db_config.connection
        
        if db_config.db_type == "mysql":
            return (f"mysql+pymysql://{conn.username}:{conn.password}@"
                   f"{conn.host}:{conn.port}/{conn.database}?charset={conn.charset}")
        elif db_config.db_type == "sqlite":
            app_config = get_app_config()
            return f"sqlite:///{app_config.data_dir}/signal_processing.db"
        else:
            raise ValueError(f"不支持的数据库类型: {db_config.db_type}")
    
    def _configure_app(self, app: FastAPI):
        """配置应用"""
        # 添加系统实例到app状态
        app.state.system = self.system
        app.state.config = self.config
        app.state.validator = self.validator
        app.state.websocket_manager = self.websocket_manager
        app.state.performance_monitor = self.performance_monitor
        app.state.database = self.database
        app.state.session_local = self.session_local
        
        # 添加模板引擎
        app.state.templates = self.templates
        
        # 配置静态文件
        if self.config.static.enabled:
            app.mount(
                self.config.static.static_url,
                StaticFiles(directory=self.config.static.static_dir),
                name="static"
            )
    
    def _add_middleware(self, app: FastAPI):
        """添加中间件"""
        # 1. 信任主机中间件
        if self.config.security.enable_trusted_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.security.allowed_hosts
            )
        
        # 2. HTTPS重定向中间件
        if self.config.server.ssl_enabled and self.config.security.https_redirect:
            app.add_middleware(HTTPSRedirectMiddleware)
        
        # 3. CORS中间件
        if self.config.cors.enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors.allow_origins,
                allow_credentials=self.config.cors.allow_credentials,
                allow_methods=self.config.cors.allow_methods,
                allow_headers=self.config.cors.allow_headers,
                expose_headers=self.config.cors.expose_headers,
                max_age=self.config.cors.max_age
            )
        
        # 4. GZIP压缩中间件
        if self.config.static.gzip_enabled:
            app.add_middleware(
                GZipMiddleware,
                minimum_size=self.config.static.gzip_min_size
            )
        
        # 5. 安全头部中间件
        if self.config.security.enable_secure_headers:
            app.add_middleware(
                SecurityHeadersMiddleware,
                config=self.config.security
            )
        
        # 6. 请求ID中间件
        if self.config.logging.enable_request_id:
            app.add_middleware(RequestIDMiddleware)
        
        # 7. 日志中间件
        if self.config.logging.enable_access_log:
            app.add_middleware(
                LoggingMiddleware,
                config=self.config.logging
            )
        
        # 8. 异常处理中间件
        app.add_middleware(ExceptionHandlerMiddleware)
    
    def _add_routes(self, app: FastAPI):
        """添加路由"""
        # 健康检查端点
        self._add_health_routes(app)
        
        # 指标端点
        if self.config.metrics.enabled:
            self._add_metrics_routes(app)
        
        # API路由
        if self.config.api.enabled:
            self._add_api_routes(app)
        
        # WebSocket端点
        if self.config.websocket.enabled:
            self._add_websocket_routes(app)
        
        # 静态页面路由
        if self.config.web_pages.enabled:
            self._add_web_page_routes(app)
        
        # 自定义路由
        self._add_custom_routes(app)
    
    def _add_health_routes(self, app: FastAPI):
        """添加健康检查路由"""
        
        @app.get("/health", tags=["system"])
        async def health_check():
            """健康检查"""
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.api.version,
                "environment": get_app_config().environment
            }
            
            # 检查数据库连接
            if self.database:
                try:
                    self.database.test_connection()
                    status["database"] = "connected"
                except Exception as e:
                    status["database"] = "disconnected"
                    status["database_error"] = str(e)
                    status["status"] = "degraded"
            
            # 检查系统状态
            if self.system and hasattr(self.system, 'running'):
                status["system"] = "running" if self.system.running else "stopped"
            
            return status
        
        @app.get("/ready", tags=["system"])
        async def readiness_check():
            """就绪检查"""
            checks = {
                "timestamp": datetime.now().isoformat(),
                "checks": {}
            }
            
            all_passed = True
            
            # 检查数据库
            if self.database:
                try:
                    self.database.test_connection()
                    checks["checks"]["database"] = {"status": "healthy"}
                except Exception as e:
                    checks["checks"]["database"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    all_passed = False
            else:
                checks["checks"]["database"] = {"status": "disabled"}
            
            # 检查WebSocket
            if self.websocket_manager:
                checks["checks"]["websocket"] = {
                    "status": "healthy" if self.websocket_manager.is_running else "unhealthy"
                }
                if not self.websocket_manager.is_running:
                    all_passed = False
            else:
                checks["checks"]["websocket"] = {"status": "disabled"}
            
            # 检查系统
            if self.system:
                if hasattr(self.system, 'running') and self.system.running:
                    checks["checks"]["system"] = {"status": "healthy"}
                else:
                    checks["checks"]["system"] = {"status": "unhealthy"}
                    all_passed = False
            else:
                checks["checks"]["system"] = {"status": "not_available"}
            
            checks["status"] = "ready" if all_passed else "not_ready"
            return checks
        
        @app.get("/status", tags=["system"])
        async def system_status():
            """系统状态"""
            status = {
                "timestamp": datetime.now().isoformat(),
                "version": self.config.api.version,
                "environment": get_app_config().environment
            }
            
            # 添加系统信息
            if self.system:
                try:
                    system_info = self.system.get_system_info()
                    status.update(system_info)
                except Exception as e:
                    status["system_info_error"] = str(e)
            
            # 添加Web应用信息
            status["web"] = {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "ssl": self.config.server.ssl_enabled,
                "workers": self.config.server.workers
            }
            
            # 添加组件状态
            if self.system:
                try:
                    component_status = self.system.get_all_component_status()
                    status["components"] = component_status
                except Exception as e:
                    status["component_status_error"] = str(e)
            
            return status
    
    def _add_metrics_routes(self, app: FastAPI):
        """添加指标路由"""
        
        @app.get(self.config.metrics.prometheus_endpoint, tags=["system"])
        async def metrics():
            """Prometheus指标"""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @app.get("/metrics/json", tags=["system"])
        async def metrics_json():
            """JSON格式指标"""
            if not self.performance_monitor:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Performance monitoring is not enabled"
                )
            
            metrics = self.performance_monitor.collect_metrics()
            return metrics
        
        # 创建ASGI应用用于Prometheus
        self.metrics_app = make_asgi_app()
        app.mount("/metrics/prometheus", self.metrics_app)
    
    def _add_api_routes(self, app: FastAPI):
        """添加API路由"""
        # 包含所有API路由
        app.include_router(
            auth_router,
            prefix=self.config.api.api_prefix,
            tags=["authentication"]
        )
        
        app.include_router(
            signals_router,
            prefix=self.config.api.api_prefix,
            tags=["signals"]
        )
        
        app.include_router(
            detections_router,
            prefix=self.config.api.api_prefix,
            tags=["detections"]
        )
        
        app.include_router(
            algorithms_router,
            prefix=self.config.api.api_prefix,
            tags=["algorithms"]
        )
        
        app.include_router(
            system_router,
            prefix=self.config.api.api_prefix,
            tags=["system"]
        )
        
        app.include_router(
            events_router,
            prefix=self.config.api.api_prefix,
            tags=["events"]
        )
        
        app.include_router(
            files_router,
            prefix=self.config.api.api_prefix,
            tags=["files"]
        )
    
    def _add_websocket_routes(self, app: FastAPI):
        """添加WebSocket路由"""
        from src.web.websocket import create_websocket_endpoint
        
        if not self.websocket_manager:
            logger.warning("WebSocket管理器未初始化，无法添加WebSocket路由")
            return
        
        # 创建WebSocket端点
        websocket_endpoint = create_websocket_endpoint(self.websocket_manager)
        
        # 添加WebSocket路由
        app.websocket(self.config.websocket.ws_endpoint)(websocket_endpoint)
        
        # 添加WebSocket状态端点
        @app.get("/ws/status", tags=["websocket"])
        async def websocket_status():
            """WebSocket状态"""
            if not self.websocket_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="WebSocket is not enabled"
                )
            
            return {
                "enabled": True,
                "endpoint": self.config.websocket.ws_endpoint,
                "connections": len(self.websocket_manager.connections),
                "rooms": list(self.websocket_manager.rooms.keys()),
                "broadcast_enabled": self.config.websocket.ws_broadcast_enabled
            }
    
    def _add_web_page_routes(self, app: FastAPI):
        """添加Web页面路由"""
        
        @app.get("/", response_class=HTMLResponse, tags=["web"])
        async def index(request: Request):
            """首页"""
            return self.templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": self.config.api.title,
                    "version": self.config.api.version,
                    "environment": get_app_config().environment
                }
            )
        
        @app.get("/dashboard", response_class=HTMLResponse, tags=["web"])
        async def dashboard(request: Request):
            """仪表板"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "title": f"{self.config.api.title} - Dashboard",
                    "api_prefix": self.config.api.api_prefix,
                    "ws_endpoint": self.config.websocket.ws_endpoint
                }
            )
        
        @app.get("/signals", response_class=HTMLResponse, tags=["web"])
        async def signals_page(request: Request):
            """信号页面"""
            return self.templates.TemplateResponse(
                "signals.html",
                {
                    "request": request,
                    "title": f"{self.config.api.title} - Signals"
                }
            )
        
        @app.get("/detections", response_class=HTMLResponse, tags=["web"])
        async def detections_page(request: Request):
            """检测页面"""
            return self.templates.TemplateResponse(
                "detections.html",
                {
                    "request": request,
                    "title": f"{self.config.api.title} - Detections"
                }
            )
        
        @app.get("/system", response_class=HTMLResponse, tags=["web"])
        async def system_page(request: Request):
            """系统页面"""
            return self.templates.TemplateResponse(
                "system.html",
                {
                    "request": request,
                    "title": f"{self.config.api.title} - System"
                }
            )
        
        @app.get("/api", response_class=HTMLResponse, tags=["web"])
        async def api_docs_redirect():
            """API文档重定向"""
            return HTMLResponse(f"""
            <html>
                <head>
                    <meta http-equiv="refresh" content="0; url={self.config.api.docs_url}" />
                </head>
                <body>
                    <p>Redirecting to <a href="{self.config.api.docs_url}">API Documentation</a>...</p>
                </body>
            </html>
            """)
    
    def _add_custom_routes(self, app: FastAPI):
        """添加自定义路由"""
        
        @app.get("/config", tags=["system"])
        async def get_config():
            """获取配置（敏感信息已过滤）"""
            config_dict = self.config.to_dict()
            return config_dict
        
        @app.get("/info", tags=["system"])
        async def get_info():
            """获取系统信息"""
            info = {
                "name": self.config.api.title,
                "description": self.config.api.description,
                "version": self.config.api.version,
                "environment": get_app_config().environment,
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "api": self.config.api.api_prefix,
                    "docs": self.config.api.docs_url,
                    "redoc": self.config.api.redoc_url,
                    "openapi": self.config.api.openapi_url,
                    "health": "/health",
                    "ready": "/ready",
                    "status": "/status",
                    "metrics": self.config.metrics.prometheus_endpoint
                },
                "features": {
                    "authentication": self.config.auth.require_authentication,
                    "websocket": self.config.websocket.enabled,
                    "cors": self.config.cors.enabled,
                    "rate_limiting": self.config.rate_limit.enabled,
                    "compression": self.config.static.gzip_enabled
                }
            }
            
            if self.system:
                info["system_id"] = getattr(self.system, 'system_id', 'unknown')
                info["system_running"] = getattr(self.system, 'running', False)
            
            return info
        
        @app.post("/reload", tags=["system"])
        async def reload_config():
            """重载配置"""
            if not self.system:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="System not available"
                )
            
            try:
                self.system.reload_config()
                return {"message": "Configuration reloaded successfully"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to reload configuration: {str(e)}"
                )
        
        @app.post("/shutdown", tags=["system"])
        async def shutdown_system():
            """关闭系统"""
            if not self.system:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="System not available"
                )
            
            # 在后台任务中关闭系统
            asyncio.create_task(self._shutdown_background())
            
            return {"message": "System shutdown initiated"}
        
        @app.get("/logs", tags=["system"])
        async def get_logs(
            lines: int = 100,
            level: str = "INFO",
            component: str = None
        ):
            """获取日志"""
            log_file = Path(self.config.logging.log_dir) / "web.log"
            
            if not log_file.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Log file not found"
                )
            
            # 读取日志文件
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                
                # 过滤和限制
                filtered_lines = []
                for line in reversed(log_lines):  # 从最新开始
                    if level and level.upper() not in line:
                        continue
                    if component and component.lower() not in line.lower():
                        continue
                    
                    filtered_lines.append(line)
                    if len(filtered_lines) >= lines:
                        break
                
                return {
                    "file": str(log_file),
                    "total_lines": len(log_lines),
                    "filtered_lines": len(filtered_lines),
                    "logs": list(reversed(filtered_lines))  # 恢复时间顺序
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to read logs: {str(e)}"
                )
        
        @app.get("/upload", tags=["files"])
        async def upload_page(request: Request):
            """上传页面"""
            return self.templates.TemplateResponse(
                "upload.html",
                {"request": request}
            )
        
        @app.post("/upload", tags=["files"])
        async def upload_file(
            file: UploadFile = File(...),
            file_type: str = "signal"
        ):
            """上传文件"""
            try:
                # 验证文件类型
                allowed_types = ["signal", "config", "algorithm", "other"]
                if file_type not in allowed_types:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid file type. Allowed: {allowed_types}"
                    )
                
                # 创建上传目录
                upload_dir = Path(self.config.static.static_dir) / "uploads" / file_type
                ensure_directory(upload_dir)
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_name = Path(file.filename).stem
                extension = Path(file.filename).suffix
                filename = f"{original_name}_{timestamp}{extension}"
                filepath = upload_dir / filename
                
                # 保存文件
                content = await file.read()
                with open(filepath, "wb") as f:
                    f.write(content)
                
                # 获取文件信息
                file_info = get_file_info(filepath)
                
                return {
                    "message": "File uploaded successfully",
                    "filename": filename,
                    "filepath": str(filepath),
                    "type": file_type,
                    "size": file_info["size"],
                    "uploaded_at": timestamp
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to upload file: {str(e)}"
                )
    
    async def _shutdown_background(self):
        """后台关闭系统"""
        await asyncio.sleep(1)  # 给响应一些时间
        if self.system:
            self.system.shutdown()
    
    def _add_exception_handlers(self, app: FastAPI):
        """添加异常处理器"""
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """HTTP异常处理器"""
            logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "timestamp": datetime.now().isoformat(),
                        "path": request.url.path
                    }
                }
            )
        
        @app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            """验证异常处理器"""
            logger.warning(f"验证异常: {exc.errors()}")
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "message": "Validation error",
                        "details": exc.errors(),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
        
        @app.exception_handler(RateLimitExceeded)
        async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
            """限流异常处理器"""
            logger.warning(f"限流异常: {request.client.host}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": status.HTTP_429_TOO_MANY_REQUESTS,
                        "message": "Rate limit exceeded",
                        "retry_after": exc.retry_after,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                headers={"Retry-After": str(exc.retry_after)}
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """通用异常处理器"""
            logger.error(f"未处理异常: {str(exc)}", exc_info=True)
            
            # 在生产环境中隐藏详细错误信息
            if get_app_config().environment == "production":
                detail = "Internal server error"
            else:
                detail = str(exc)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                        "message": detail,
                        "timestamp": datetime.now().isoformat(),
                        "path": request.url.path
                    }
                }
            )
    
    def _customize_openapi(self, app: FastAPI):
        """自定义OpenAPI文档"""
        
        def custom_openapi():
            """自定义OpenAPI文档"""
            if app.openapi_schema:
                return app.openapi_schema
            
            # 生成基础OpenAPI
            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )
            
            # 添加安全方案
            if self.config.auth.require_authentication:
                openapi_schema["components"]["securitySchemes"] = {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            
            # 添加服务器信息
            scheme = "https" if self.config.server.ssl_enabled else "http"
            host = self.config.server.host if self.config.server.host != "0.0.0.0" else "localhost"
            port = self.config.server.port
            
            openapi_schema["servers"] = [
                {
                    "url": f"{scheme}://{host}:{port}",
                    "description": f"{get_app_config().environment.capitalize()} server"
                }
            ]
            
            # 添加标签
            openapi_schema["tags"] = [
                {"name": "authentication", "description": "用户认证和授权"},
                {"name": "signals", "description": "信号数据管理"},
                {"name": "detections", "description": "检测结果管理"},
                {"name": "algorithms", "description": "算法管理"},
                {"name": "system", "description": "系统管理"},
                {"name": "events", "description": "事件管理"},
                {"name": "files", "description": "文件管理"},
                {"name": "websocket", "description": "WebSocket连接"},
                {"name": "web", "description": "Web页面"}
            ]
            
            # 添加联系信息
            if self.config.api.contact_name or self.config.api.contact_email:
                openapi_schema["contact"] = {}
                if self.config.api.contact_name:
                    openapi_schema["contact"]["name"] = self.config.api.contact_name
                if self.config.api.contact_email:
                    openapi_schema["contact"]["email"] = self.config.api.contact_email
            
            # 添加许可证信息
            if self.config.api.license_name:
                openapi_schema["license"] = {
                    "name": self.config.api.license_name
                }
                if self.config.api.license_url:
                    openapi_schema["license"]["url"] = self.config.api.license_url
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
        
        # 自定义Swagger UI
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html(request: Request):
            """自定义Swagger UI"""
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} - Swagger UI",
                oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
                swagger_ui_parameters={
                    "defaultModelsExpandDepth": -1,
                    "docExpansion": "none",
                    "filter": True,
                    "showExtensions": True,
                    "showCommonExtensions": True
                }
            )
        
        # 自定义ReDoc
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html(request: Request):
            """自定义ReDoc"""
            return get_redoc_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} - ReDoc"
            )
    
    def run(self):
        """运行Web应用"""
        if not self.app:
            self.app = self.create_app()
        
        logger.info("=" * 60)
        logger.info(f"启动Web服务: {self.config.api.title}")
        logger.info(f"版本: {self.config.api.version}")
        logger.info(f"环境: {get_app_config().environment}")
        logger.info(f"主机: {self.config.server.host}")
        logger.info(f"端口: {self.config.server.port}")
        logger.info(f"Workers: {self.config.server.workers}")
        logger.info(f"SSL: {'启用' if self.config.server.ssl_enabled else '禁用'}")
        logger.info("=" * 60)
        
        # 获取Uvicorn配置
        uvicorn_config = self.config.get_uvicorn_config()
        
        # 运行Uvicorn
        uvicorn.run(
            self.app,
            **uvicorn_config
        )


# 工具函数
def get_system_from_request(request: Request):
    """从请求中获取系统实例"""
    return request.app.state.system


def get_config_from_request(request: Request):
    """从请求中获取配置"""
    return request.app.state.config


def get_validator_from_request(request: Request):
    """从请求中获取验证器"""
    return request.app.state.validator


def get_websocket_manager_from_request(request: Request):
    """从请求中获取WebSocket管理器"""
    return request.app.state.websocket_manager


def get_database_from_request(request: Request):
    """从请求中获取数据库管理器"""
    return request.app.state.database


def get_db_session() -> Session:
    """获取数据库会话"""
    session_local = None
    
    # 尝试从应用状态获取
    try:
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        
        # 这是一个hack，实际上应该在依赖注入中使用
        pass
    except:
        pass
    
    if not session_local:
        # 创建新的会话
        db_config = get_db_config()
        from src.core.database_manager import DatabaseManager
        database = DatabaseManager(_get_database_url(db_config))
        session_local = database.SessionLocal
    
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def _get_database_url(db_config) -> str:
    """获取数据库URL"""
    conn = db_config.connection
    
    if db_config.db_type == "mysql":
        return (f"mysql+pymysql://{conn.username}:{conn.password}@"
               f"{conn.host}:{conn.port}/{conn.database}?charset={conn.charset}")
    elif db_config.db_type == "sqlite":
        app_config = get_app_config()
        return f"sqlite:///{app_config.data_dir}/signal_processing.db"
    else:
        raise ValueError(f"不支持的数据库类型: {db_config.db_type}")


# 创建应用的快捷函数
def create_web_app(
    title: str = None,
    description: str = None,
    version: str = None,
    system=None
) -> FastAPI:
    """
    创建Web应用
    
    Args:
        title: 应用标题
        description: 应用描述
        version: 应用版本
        system: 信号处理系统实例
        
    Returns:
        FastAPI: FastAPI应用实例
    """
    config = get_current_config()
    
    # 覆盖配置
    if title:
        config.api.title = title
    if description:
        config.api.description = description
    if version:
        config.api.version = version
    
    # 创建Web应用管理器
    web_app = WebApp(config=config, system=system)
    
    # 创建并返回FastAPI应用
    return web_app.create_app()


# 主函数
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动Web服务")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="端口号"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="工作进程数"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    
    # 创建Web应用
    app = create_web_app()
    
    # 获取配置
    config = get_current_config()
    
    # 覆盖配置
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.workers:
        config.server.workers = args.workers
    
    # 获取Uvicorn配置
    uvicorn_config = config.get_uvicorn_config()
    
    # 覆盖重载设置
    if args.reload:
        uvicorn_config["reload"] = True
    
    # 运行Uvicorn
    uvicorn.run(
        app,
        **uvicorn_config
    )


if __name__ == "__main__":
    main()