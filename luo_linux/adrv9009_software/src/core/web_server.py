"""
Web服务模块 - 提供Web接口和前端
"""
import json
import asyncio
from typing import Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from src.utils.logger import get_logger
from src.web.app import create_app
from src.web.websocket.ws_handler import WebSocketManager

class WebServer:
    """Web服务器"""
    
    def __init__(self, config: Dict[str, Any], db_manager=None):
        self.config = config
        self.db = db_manager
        self.logger = get_logger(__name__)
        
        # FastAPI 应用
        self.app = create_app(self.config, self.db)
        
        # WebSocket 管理器
        self.ws_manager = WebSocketManager()
        
        # 服务器实例
        self.server = None
        
        # 事件订阅
        self.subscribed_events = []
    
    def start(self, host=None, port=None):
        """启动Web服务器"""
        host = host or self.config.get('host', '0.0.0.0')
        port = port or self.config.get('port', 8000)
        
        self.logger.info(f"启动Web服务器: {host}:{port}")
        
        # 配置uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info" if self.config.get('debug', False) else "warning"
        )
        
        self.server = uvicorn.Server(config)
        
        # 在后台线程中运行
        import threading
        server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        server_thread.start()
        
        return True
    
    def _run_server(self):
        """运行服务器（在独立线程中）"""
        try:
            self.server.run()
        except Exception as e:
            self.logger.error(f"Web服务器运行错误: {e}")
    
    def register_event_callback(self, event_type: str, callback: callable):
        """注册事件回调"""
        self.subscribed_events.append((event_type, callback))
        self.logger.debug(f"注册事件回调: {event_type}")
    
    def broadcast_event(self, event_type: str, data: Any):
        """广播事件到所有WebSocket客户端"""
        try:
            message = {
                'type': 'event',
                'event': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # 通过WebSocket广播
            self.ws_manager.broadcast(json.dumps(message))
            
            # 执行事件回调
            for subscribed_type, callback in self.subscribed_events:
                if subscribed_type == event_type or subscribed_type == '*':
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"事件回调执行失败: {e}")
                        
        except Exception as e:
            self.logger.error(f"广播事件失败: {e}")
    
    def get_realtime_data(self, limit: int = 100):
        """获取实时数据"""
        if not self.db:
            return []
        
        try:
            return self.db.get_recent_detections(limit)
        except Exception as e:
            self.logger.error(f"获取实时数据失败: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'web_server': {
                'running': self.server is not None,
                'clients': len(self.ws_manager.active_connections)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def stop(self):
        """停止Web服务器"""
        if self.server:
            self.logger.info("停止Web服务器...")
            self.server.should_exit = True
            self.ws_manager.disconnect_all()
    
    def get_status(self) -> Dict[str, Any]:
        """获取Web服务器状态"""
        return {
            'running': self.server is not None,
            'active_connections': len(self.ws_manager.active_connections),
            'subscribed_events': len(self.subscribed_events)
        }