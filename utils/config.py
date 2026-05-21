from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class Config:
    """
    โหลดและเข้าถึง config จาก YAML file
    รองรับ dot-notation เช่น config.camera.fps
    """

    def __init__(self, path: str = "configs/system.yaml") -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ config: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        เข้าถึงค่าด้วย dot-notation
        เช่น get('camera.fps') คืน 30
        """
        keys = key.split(".")
        val = self._data
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        return val

    def __getattr__(self, name: str) -> "_Section":
        if name.startswith("_"):
            raise AttributeError(name)
        section = self._data.get(name)
        if section is None:
            raise AttributeError(f"ไม่พบ section '{name}' ใน config")
        return _Section(section)


class _Section:
    """ห่อ dict section ให้เข้าถึงด้วย attribute ได้"""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(f"ไม่พบ key '{name}' ใน config section")
        return self._data[name]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)