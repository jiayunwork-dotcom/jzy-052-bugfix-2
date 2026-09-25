"""具名工艺档的持久化：JSON 文件 + 进程内锁 + 原子写，进程重启后仍可点名取回。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

DEMO_PRESET_NAME = "demo-butt-thick"

DEMO_PRESET = {
    "description": (
        "对接接头厚板焊示范工艺（结构钢）：总功率 3 kW、热效率 0.8、焊速 5 mm/s、"
        "初始温度 25 °C。热源后方轴线（y = z = 0，x < 0）上温升有闭式解 "
        "ΔT = q/(2πk|x|)，q = 2400 W，可用于手算核对积分与衰减是否可信。"
    ),
    "process": {
        "mode": "thick",
        "travel_speed": 0.005,
        "conductivity": 50.0,
        "diffusivity": 1.2e-5,
        "efficiency": 0.8,
        "initial_temperature": 25.0,
        "power": 3000.0,
    },
}


class PresetStore:
    """以 JSON 文件为后端的工艺档存储；读写均在锁内，写操作原子替换。"""

    def __init__(self, path: str | os.PathLike):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})
        self._seed_demo()

    def _read(self) -> dict:
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        tmp = self._path.parent / (self._path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)  # 原子替换，避免留下半截文件

    def _seed_demo(self) -> None:
        with self._lock:
            data = self._read()
            if DEMO_PRESET_NAME not in data:
                data[DEMO_PRESET_NAME] = DEMO_PRESET
                self._write(data)

    def list(self) -> dict:
        with self._lock:
            return self._read()

    def get(self, name: str) -> dict | None:
        with self._lock:
            return self._read().get(name)

    def put(self, name: str, record: dict) -> None:
        with self._lock:
            data = self._read()
            data[name] = record
            self._write(data)

    def delete(self, name: str) -> bool:
        with self._lock:
            data = self._read()
            if name not in data:
                return False
            del data[name]
            self._write(data)
            return True
