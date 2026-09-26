"""Ollama 本地模型管理辅助 —— 移植 ``routes/aiConfigs.ts`` 里的小工具。

识别本机 Ollama 可执行文件位置 → 探测可达性 → 需要时拉起服务。

⚠️ 这是本项目里少见的**子进程**依赖（另一个是 media 域的 ffmpeg）。Python 侧用
``subprocess`` 完全等价：``where ollama`` 走 cmd（与原 TS 的 ``shell: 'cmd.exe'`` 同形），
启动用 ``Popen`` + 脱离进程组（对应 JS 的 ``spawn(..., {detached:true})`` + ``child.unref()``）。

⚠️ 启动等待是**最坏 20 秒的轮询**（40 × 500ms）—— 它是接口内的阻塞等待，
真机测试时要注意别误判成卡死。
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from typing import Any

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def normalize_ollama_url(base_url: str | None = None) -> str:
    return re.sub(r"/+$", "", str(base_url or DEFAULT_OLLAMA_URL))


def format_bytes(size: int | None = None) -> str:
    """字节数 → 人类可读（``toFixed(1)`` ⇒ 恒一位小数）。"""
    if not size or size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.1f} {units[index]}"


def find_ollama_exe() -> str | None:
    """探测 Ollama 可执行文件位置：PATH → 用户声明 → **标准安装位置**（找不到 ⇒ ``None``）。

    ⚠️ 顺序与理由（2026-09-26 用户口径 ✓：**扫描/探测类路径一律不许写死** ✗）：
    1. ``where ollama`` —— 用户把哪个装进了 PATH，就以哪个为准 ✓（最权威 ✓ 不用我们猜 ✗）；
    2. 环境变量 ``OLLAMA_EXE`` ✓ —— 装到**非标准位置**时的唯一正解 ✓（显式声明 ✓，与
       ``COMFYUI_PATH`` / ``EXTRA_MODEL_PATHS`` 同一套习惯 ✓）。⚠️ 声明了就以声明为准 ✓
       **不回退**：路径写错时启动会**当场报错** ✓（免得变成「我明明设了变量」这种最难查的静默错 ✗）；
    3. **标准安装位置**（官方 per-user 安装 → ``Program Files``）✓ —— 这两处**不是本机专有** ✗
       （任何机器装上都在同一处 ✓）⇒ 这是 ``tests/local_models_test.py`` 里唯一带理由的豁免 ✓；
    ✗ **不写死**「某盘某自建目录」✗（曾有一条本机自建目录的候选 ✗ 已删 ✓）：换台机器就悬空 ✗，
    且会踩 ``app/`` 的静态守卫 ✗。返回 ``None`` 时调用方给明示错误（「请先安装（ollama.com）」✓），
    不静默兜底 ✗。
    """
    try:
        proc = subprocess.run(
            "where ollama",
            capture_output=True,
            text=True,
            timeout=3,
            shell=True,
        )
        first = next((line.strip() for line in (proc.stdout or "").splitlines() if line.strip()), None)
        if first:
            return first
    except Exception:  # noqa: BLE001 - 不在 PATH 中
        pass

    declared = (os.environ.get("OLLAMA_EXE") or "").strip()
    if declared:
        return declared

    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = [
        f"{local_app_data}\\Programs\\Ollama\\ollama.exe" if local_app_data else "",
        "C:\\Program Files\\Ollama\\ollama.exe",
        "C:\\Program Files (x86)\\Ollama\\ollama.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


async def is_ollama_reachable(base_url: str) -> bool:
    """探测 ``/api/tags``（2.5s 超时，任何异常都视为不可达）。"""
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(f"{base_url}/api/tags")
            return response.is_success
    except Exception:  # noqa: BLE001
        return False


async def try_start_ollama() -> dict[str, Any]:
    """尝试启动本地 Ollama 服务并等待就绪（最坏轮询 20 秒）。"""
    if await is_ollama_reachable(DEFAULT_OLLAMA_URL):
        return {"started": False, "message": "Ollama 已在运行", "exe": find_ollama_exe()}

    exe = find_ollama_exe()
    if not exe:
        return {"started": False, "message": "未找到 Ollama，请先安装（ollama.com）", "exe": None}

    try:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            # 对应 spawn(detached:true)：脱离控制台，父进程退出后仍存活
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "message": f"启动失败: {exc}", "exe": exe}

    for _ in range(40):
        await asyncio.sleep(0.5)
        if await is_ollama_reachable(DEFAULT_OLLAMA_URL):
            return {"started": True, "message": "Ollama 已启动并就绪", "exe": exe}

    return {
        "started": True,
        "message": "启动命令已发出，若 10 秒内未就绪请检查 Ollama 安装",
        "exe": exe,
    }


#: 本地四大运行时（与 ``LOCAL_PRESET_SERVICES`` 的 baseUrl 保持一致）
LOCAL_RUNTIMES = (
    {"key": "ollama", "label": "文本 · Ollama", "service_type": "text", "provider": "ollama", "base_url": "http://localhost:11434", "probe_path": "/api/tags"},
    {"key": "local-sd", "label": "图像 · Stable Diffusion", "service_type": "image", "provider": "local-sd", "base_url": "http://localhost:7860", "probe_path": "/sdapi/v1/samplers"},
    {"key": "h3", "label": "视频 · MiniMax H3", "service_type": "video", "provider": "minimax", "base_url": "http://localhost:8765", "probe_path": "/"},
    {"key": "cosyvoice", "label": "语音 · CosyVoice", "service_type": "audio", "provider": "cosyvoice", "base_url": "http://localhost:9880", "probe_path": "/"},
)


async def probe_local_runtime(
    base_url: str, probe_path: str, timeout_ms: int = 2500
) -> dict[str, Any]:
    """探测单个本地运行时。**任何 HTTP 状态码都算可达**（只有连不上才算不可达）。

    错误文案与原 TS 同形：超时 → ``连接超时``；连接被拒/域名解析失败 → ``未启动``。
    """
    started_at = asyncio.get_event_loop().time()
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            response = await client.get(f"{base_url}{probe_path}")
            return {
                "reachable": True,
                "httpStatus": response.status_code,
                "latencyMs": int((asyncio.get_event_loop().time() - started_at) * 1000),
                "error": "",
            }
    except httpx.TimeoutException:
        error = "连接超时"
    except httpx.ConnectError:
        error = "未启动"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return {
        "reachable": False,
        "httpStatus": None,
        "latencyMs": int((asyncio.get_event_loop().time() - started_at) * 1000),
        "error": error,
    }
