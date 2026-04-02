#!/usr/bin/env python3
"""
主程序入口 - 协调各个模块运行
"""
import signal
import sys
import time
from threading import Event
from typing import Dict, Any

from src.core.data_acquisition import DataAcquisition
from src.core.data_processor import DataProcessor
from src.core.database_manager import DatabaseManager
from src.core.web_server import WebServer
from src.utils.logger import setup_logger
from config.settings import Config

class Application:
    """主应用程序类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or Config.load()
        self.logger = setup_logger("main", self.config.logging.level)
        self.shutdown_event = Event()
        
        # 初始化各模块
        self.modules = {}
        
    def initialize(self):
        """初始化所有模块"""
        self.logger.info("初始化应用程序...")
        
        try:
            # 1. 初始化数据库管理器
            self.modules['db'] = DatabaseManager(self.config.database)
            self.modules['db'].initialize()
            
            # 2. 初始化数据采集模块
            self.modules['acquisition'] = DataAcquisition(
                config=self.config.acquisition,
                db_manager=self.modules['db']
            )
            
            # 3. 初始化数据处理模块
            self.modules['processor'] = DataProcessor(
                config=self.config.processing,
                db_manager=self.modules['db']
            )
            
            # 4. 初始化Web服务器
            self.modules['web'] = WebServer(
                config=self.config.web,
                db_manager=self.modules['db']
            )
            
            # 连接模块间数据流
            self.setup_data_pipeline()
            
            self.logger.info("应用程序初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}", exc_info=True)
            return False
    
    def setup_data_pipeline(self):
        """设置数据处理流水线"""
        # 采集 -> 处理 -> 存储
        self.modules['acquisition'].set_callback(
            self.modules['processor'].process_data
        )
        
        # 处理结果 -> 数据库存储
        self.modules['processor'].set_result_callback(
            self.modules['db'].store_processing_result
        )
    
    def run(self):
        """运行应用程序"""
        if not self.initialize():
            self.logger.error("初始化失败，退出")
            return
        
        self.logger.info("启动应用程序...")
        
        try:
            # 启动Web服务器
            self.modules['web'].start()
            
            # 启动数据处理模块
            self.modules['processor'].start()
            
            # 启动数据采集（24小时运行）
            self.modules['acquisition'].start_continuous_acquisition(
                duration_hours=24,
                callback_on_complete=self.on_acquisition_complete
            )
            
            # 等待关闭信号
            self.wait_for_shutdown()
            
        except KeyboardInterrupt:
            self.logger.info("接收到中断信号")
        except Exception as e:
            self.logger.error(f"运行时错误: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def on_acquisition_complete(self):
        """数据采集完成回调"""
        self.logger.info("24小时数据采集完成")
        # 可以在这里触发后续处理或重启采集
    
    def wait_for_shutdown(self):
        """等待关闭信号"""
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("接收到键盘中断")
    
    def shutdown(self):
        """优雅关闭"""
        self.logger.info("正在关闭应用程序...")
        
        # 逆序关闭模块
        for name in reversed(list(self.modules.keys())):
            try:
                if hasattr(self.modules[name], 'shutdown'):
                    self.modules[name].shutdown()
                elif hasattr(self.modules[name], 'stop'):
                    self.modules[name].stop()
                self.logger.debug(f"已关闭模块: {name}")
            except Exception as e:
                self.logger.warning(f"关闭模块 {name} 时出错: {e}")
        
        self.logger.info("应用程序已关闭")


def main():
    """主函数"""
    app = Application()
    
    # 注册信号处理
    def signal_handler(signum, frame):
        app.logger.info(f"接收到信号 {signum}")
        app.shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行应用
    app.run()


if __name__ == "__main__":
    main()