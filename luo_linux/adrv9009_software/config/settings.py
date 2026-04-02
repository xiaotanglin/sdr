"""
应用程序配置
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml

BASE_DIR = Path(__file__).parent.parent

class Config:
    """配置类"""
    
    # 默认配置
    DEFAULTS = {
        'app': {
            'name': 'ADRV9009信号处理系统',
            'version': '1.0.0',
            'debug': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': str(BASE_DIR / 'logs' / 'app.log'),
            'max_size': 10485760,  # 10MB
            'backup_count': 5
        },
        'acquisition': {
            'device_uris': ['ip:192.168.1.10', 'ip:192.168.2.1', 'usb:', 'local:'],
            'sample_rate': 1000000,
            'rx_lo': 1000000000,  # 1GHz
            'tx_lo': 1000000000,
            'rx_gain': 30,
            'batch_size': 1024,
            'buffer_size': 1000
        },
        'processing': {
            'max_workers': 4,
            'algorithms': {
                'direction_finding': {
                    'enabled': True,
                    'method': 'music',
                    'threshold': 0.8
                },
                'sorting': {
                    'enabled': True,
                    'algorithm': 'dbscan',
                    'parameters': {'eps': 0.5, 'min_samples': 5}
                },
                'recognition': {
                    'enabled': True,
                    'model_path': str(BASE_DIR / 'models' / 'recognition.pkl'),
                    'threshold': 0.7
                }
            },
            'recognition_threshold': 0.8
        },
        'database': {
            'type': 'sqlite',  # sqlite, postgresql, mysql
            'path': str(BASE_DIR / 'data' / 'database' / 'signals.db'),
            'host': 'localhost',
            'port': 5432,
            'username': '',
            'password': '',
            'database_name': 'signal_db'
        },
        'web': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'static_dir': str(BASE_DIR / 'src' / 'web' / 'static'),
            'template_dir': str(BASE_DIR / 'src' / 'web' / 'templates'),
            'cors_origins': ['*'],
            'api_prefix': '/api/v1'
        },
        'alerts': {
            'enabled': True,
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': ''
            },
            'webhook': {
                'enabled': False,
                'url': ''
            },
            'notification_events': [
                'unknown_signal_detected',
                'signal_recognized',
                'system_error'
            ]
        }
    }
    
    @classmethod
    def load(cls, config_file: str = None) -> Dict[str, Any]:
        """加载配置"""
        config = cls.DEFAULTS.copy()
        
        # 从文件加载
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                cls._deep_update(config, file_config)
        
        # 从环境变量加载
        cls._load_from_env(config)
        
        return config
    
    @staticmethod
    def _deep_update(original: Dict, update: Dict):
        """深度更新字典"""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                Config._deep_update(original[key], value)
            else:
                original[key] = value
    
    @staticmethod
    def _load_from_env(config: Dict):
        """从环境变量加载配置"""
        # 示例：数据库配置
        if os.getenv('DB_HOST'):
            config['database']['host'] = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config['database']['port'] = int(os.getenv('DB_PORT'))