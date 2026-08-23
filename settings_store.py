"""Small, dependency-free persistence layer for local Studio preferences."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from config import (
    LANGUAGE_AUTO, LANGUAGES,
    MAX_BATCH_SIZE, MIN_BATCH_SIZE,
)
from state import AppSettings
from ui import strings as S


SETTINGS_VERSION = 1
SETTINGS_PATH = str(Path(__file__).resolve().parents[1] / "config" / "settings.json")

PERSISTED_SETTING_KEYS = (
    "model_size",
    "quantization",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "max_tokens",
    "timeout",
    "batch_size",
    "output_dir",
    "autosave",
    "jit_compile",
    "default_language",
    "export_format",
    "mp3_bitrate",
    "loudnorm",
    "trim_silence",
    "denoise_ref",
)

_DISPLAY_NAMES = {
    "model_size": "模型大小",
    "quantization": "量化方式",
    "temperature": "温度",
    "top_k": "Top-K",
    "top_p": "Top-P",
    "repetition_penalty": "重复惩罚",
    "max_tokens": "最大长度",
    "timeout": "自动停止时间",
    "batch_size": "批量大小",
    "output_dir": "输出目录",
    "autosave": "自动保存",
    "jit_compile": "加速重复运行",
    "default_language": "默认语言",
    "export_format": "音频格式",
    "mp3_bitrate": "MP3 比特率",
    "loudnorm": "标准化响度",
    "trim_silence": "裁剪静音",
    "denoise_ref": "降低参考音频噪声",
}


@dataclass
class SettingsLoadResult:
    settings: AppSettings
    warnings: list[str]


def _warning(key: str) -> str:
    return S.SETTINGS_INVALID_VALUE.format(key=_DISPLAY_NAMES.get(key, key))


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated_value(key: str, value):
    if key == "model_size":
        return value if value in ("0.6B", "1.7B") else None
    if key == "quantization":
        return value if value in ("4bit", "6bit", "8bit", "bf16") else None
    if key in {"temperature", "top_p", "repetition_penalty"}:
        if not _finite_number(value):
            return None
        lower, upper = {
            "temperature": (0.0, 1.5),
            "top_p": (0.0, 1.0),
            "repetition_penalty": (1.0, 2.0),
        }[key]
        return float(value) if lower <= float(value) <= upper else None
    if key in {"top_k", "max_tokens", "timeout", "batch_size", "mp3_bitrate"}:
        if not _finite_number(value) or int(value) != value:
            return None
        lower, upper = {
            "top_k": (0, 100),
            "max_tokens": (512, 8192),
            "timeout": (30, 300),
            "batch_size": (MIN_BATCH_SIZE, MAX_BATCH_SIZE),
            "mp3_bitrate": (64, 320),
        }[key]
        return int(value) if lower <= int(value) <= upper else None
    if key == "output_dir":
        return value.strip() if isinstance(value, str) and value.strip() else None
    if key in {"autosave", "jit_compile", "loudnorm", "trim_silence", "denoise_ref"}:
        return value if isinstance(value, bool) else None
    if key == "default_language":
        return value if value in [LANGUAGE_AUTO] + LANGUAGES else None
    if key == "export_format":
        return value if value in ("wav", "mp3", "ogg") else None
    raise KeyError(key)


def sanitize_settings_mapping(data: dict) -> tuple[AppSettings, list[str]]:
    """Build settings from untrusted JSON/form data, using safe defaults."""
    defaults = AppSettings()
    values = asdict(defaults)
    warnings = []
    for key in PERSISTED_SETTING_KEYS:
        if key not in data:
            continue
        normalized = _validated_value(key, data[key])
        if normalized is None:
            warnings.append(_warning(key))
        else:
            values[key] = normalized
    return AppSettings(**values), warnings


def sanitize_settings(settings: AppSettings) -> tuple[AppSettings, list[str]]:
    return sanitize_settings_mapping(asdict(settings))


def settings_to_dict(settings: AppSettings) -> dict:
    values = {key: getattr(settings, key) for key in PERSISTED_SETTING_KEYS}
    return {"version": SETTINGS_VERSION, **values}


def load_settings(path: str | os.PathLike | None = None) -> SettingsLoadResult:
    """Load local preferences without ever making startup depend on the file."""
    path = Path(SETTINGS_PATH if path is None else path)
    if not path.exists():
        return SettingsLoadResult(AppSettings(), [])
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("JSON 根对象必须是对象")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return SettingsLoadResult(
            AppSettings(), [S.SETTINGS_LOAD_FAILED.format(err=str(exc))]
        )

    warnings = []
    version = data.get("version", SETTINGS_VERSION)
    if version != SETTINGS_VERSION:
        warnings.append(S.SETTINGS_UNSUPPORTED_VERSION)
    settings, value_warnings = sanitize_settings_mapping(data)
    warnings.extend(value_warnings)
    return SettingsLoadResult(settings, warnings)


def save_settings(
    settings: AppSettings,
    path: str | os.PathLike | None = None,
) -> tuple[AppSettings, list[str]]:
    """Atomically write only the allow-listed application preferences."""
    normalized, warnings = sanitize_settings(settings)
    path = Path(SETTINGS_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(settings_to_dict(normalized), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return normalized, warnings
