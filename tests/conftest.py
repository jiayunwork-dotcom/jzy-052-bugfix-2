import os
import tempfile

# 在导入 app.main 之前指定默认存储路径，避免模块级 app 在仓库根目录落文件
os.environ.setdefault(
    "PRESET_STORE_PATH",
    os.path.join(tempfile.gettempdir(), "welding_service_tests_default_presets.json"),
)

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "presets.json")))


@pytest.fixture()
def demo_process():
    return {
        "mode": "thick",
        "travel_speed": 0.005,
        "conductivity": 50.0,
        "diffusivity": 1.2e-5,
        "efficiency": 0.8,
        "initial_temperature": 25.0,
        "power": 3000.0,
    }
