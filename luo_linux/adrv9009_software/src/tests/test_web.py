"""
Web测试模块
测试Web应用、API接口、认证授权、WebSocket等Web相关功能
"""
import os
import sys
import json
import time
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import aiohttp
import aiofiles
import jwt
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import websockets
from websockets.exceptions import ConnectionClosed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试配置
from src.tests.conftest import TestConfig

# 导入被测试模块
from src.web.app import create_web_app, WebApp
from src.web.auth import (
    create_access_token,
    verify_access_token,
    get_password_hash,
    verify_password,
    get_current_user,
    get_current_active_user,
    get_current_admin_user
)
from src.web.api.data_api import router as data_router
from src.web.api.status_api import router as status_router
from src.web.api.alert_api import router as alert_router
from src.web.websocket_manager import WebSocketManager
from src.web.websocket import create_websocket_endpoint

# 导入模型
from src.models.signal_data import SignalData, create_signal_from_iq
from src.models.detection_result import DetectionResult, create_detection_from_signal
from src.models.database_models import RawSignal, Detection, User

# 导入工具
from src.utils.data_validator import DataValidator
from src.utils.file_utils import ensure_directory, get_file_info

# 导入测试fixtures
from src.tests.conftest import (
    app_config,
    web_config,
    db_config,
    signal_config,
    test_db_url,
    database_manager,
    db_session,
    async