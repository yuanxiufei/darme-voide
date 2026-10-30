"""**环境变量真源** —— 上游 ``envconfig/config.go`` 的对等实现。

为什么值得单独搬：本项目里 ``ollama_store.models_root()``、``local_model_scan``、
adapter 的 base_url **各自**猜过一次 ``OLLAMA_MODELS`` / ``OLLAMA_HOST``，一旦用户设了
带引号、带 ``https://`` 前缀、或只写 ``0.0.0.0:11434`` 这种值，各处行为就分叉了。
这里把上游那套解析（含 ``Host()`` 的端口默认值推导、``Var()`` 的去引号）统一固化。

⚠️ 三处刻意与 Go 不同，都是因为「照抄会引入隐形 bug」：

* ``KeepAlive()``/``LoadTimeout()`` 返回 **float 秒**（``math.inf`` 表示无穷），
  而不是 ``timedelta`` —— 调用方要直接拿去算 ``asyncio.wait_for`` 的超时；
* Go 的 ``BoolWithDefault(k)`` 返回一个函数，Python 侧同样返回闭包以保持「读环境变量
  的时机」一致（本仓有测试会在 import 后再改 ``os.environ``，必须每次现读）；
* ``Values()`` 的字符串化按 Go ``%v`` 的形状拼（``5m0s`` / ``http://…``），
  因为它是 ``ollama config`` 的对外可见输出，形状变了就等于契约变了。
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "AllowedOrigins",
    "AsMap",
    "Bool",
    "BoolWithDefault",
    "ConnectableHost",
    "EnvVar",
    "Host",
    "HostInfo",
    "KeepAlive",
    "LoadTimeout",
    "LogLevel",
    "Models",
    "NoCloud",
    "NoCloudSource",
    "Remotes",
    "String",
    "Uint",
    "Uint64",
    "Values",
    "Var",
    "format_duration",
    "parse_bool",
    "parse_duration",
]


# --------------------------------------------------------------------------- #
# 基础解析
# --------------------------------------------------------------------------- #

_TRUE = {"1", "t", "T", "true", "TRUE", "True"}
_FALSE = {"0", "f", "F", "false", "FALSE", "False"}


def Var(key: str) -> str:
    """对应 ``envconfig.Var`` —— 先 trim 空白，再剥掉首尾的引号。

    ⚠️ 剥引号是上游的真实行为（``strings.Trim(s, "\\"'")``），Windows 用户用
    ``set OLLAMA_MODELS="<盘符>:\\models"`` 时会带进引号；不剥就会拼出
    ``"<盘符>:\\models"/blobs`` 这种不存在的路径。
    """
    return os.environ.get(key, "").strip().strip("\"'")


def parse_bool(text: str) -> bool:
    """对等 ``strconv.ParseBool``（只认 Go 那十个字面量，其他一律报错）。"""
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"invalid syntax: {text!r}")


_DURATION_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}
_DURATION_TOKEN = re.compile(r"([0-9]*\.?[0-9]+)(ns|us|µs|μs|ms|s|m|h)")


def parse_duration(text: str) -> float:
    """对等 ``time.ParseDuration``，返回**秒**。

    支持 ``"300ms"``/``"1.5h"``/``"2h45m"``/``"-5m"``/``"0"``。
    不支持的写法（``"5 minutes"``、``"1d"``）**报错而不猜** —— 上游 ParseDuration 也报错，
    而调用方 ``KeepAlive()`` 在报错后会退回「当作纯数字秒」再退回默认值。
    """
    original = text
    if text in ("0", "+0", "-0"):
        return 0.0
    sign = 1.0
    if text.startswith(("+", "-")):
        if text[0] == "-":
            sign = -1.0
        text = text[1:]
    if not text:
        raise ValueError(f"invalid duration {original!r}")
    total = 0.0
    position = 0
    while position < len(text):
        match = _DURATION_TOKEN.match(text, position)
        if not match:
            raise ValueError(f"invalid duration {original!r}")
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        position = match.end()
    return sign * total


def format_duration(seconds: float) -> str:
    """按 Go ``time.Duration.String()`` 的形状格式化（``300`` → ``"5m0s"``）。"""
    if seconds == math.inf:
        return "2562047h47m16.854775807s"   # math.MaxInt64 纳秒，与 Go 的溢出格式化一致
    if seconds == 0:
        return "0s"
    sign = "-" if seconds < 0 else ""
    nanoseconds = int(round(abs(seconds) * 1e9))
    if nanoseconds < 1000:
        return f"{sign}{nanoseconds}ns"
    if nanoseconds < 1_000_000:
        return f"{sign}{_trim(nanoseconds / 1000)}µs"
    if nanoseconds < 1_000_000_000:
        return f"{sign}{_trim(nanoseconds / 1_000_000)}ms"
    hours, rest = divmod(nanoseconds, 3_600_000_000_000)
    minutes, rest = divmod(rest, 60_000_000_000)
    secs = rest / 1e9
    out = sign
    if hours:
        out += f"{hours}h"
    if hours or minutes:
        out += f"{minutes}m"
    return out + f"{_trim(secs)}s"


def _trim(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text or "0"


# --------------------------------------------------------------------------- #
# Host
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HostInfo:
    """``url.URL`` 的极简对等物（只要 scheme/host/port/path 四个人用得到的字段）。"""

    scheme: str = "http"
    host: str = "127.0.0.1"
    port: str = "11434"
    path: str = ""

    @property
    def netloc(self) -> str:
        """``net.JoinHostPort`` —— IPv6 带方括号，否则 iPad 那种地址会拼坏。"""
        if ":" in self.host:
            return f"[{self.host}]:{self.port}"
        return f"{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        # Go 的 url.URL.String() 在 Host 非空时会给不以 "/" 开头的 Path 补一个斜杠，
        # 这里必须照做：`http://[::1]:8080/x` 少一个斜杠就变成端口 8080x 的非法 URL
        path = self.path
        if path and not path.startswith("/"):
            path = "/" + path
        return f"{self.scheme}://{self.netloc}{path}"

    def __str__(self) -> str:
        return self.base_url


def _parse_ip(text: str) -> str | None:
    """``net.ParseIP`` → 规范化字符串（``0:0:0:0:0:0:0:1`` → ``::1``）。"""
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def _split_host_port(hostport: str) -> tuple[str, str] | None:
    """对等 ``net.SplitHostPort``：失败返回 ``None``（含「有端口无主机」「冒号过多」）。"""
    if not hostport:
        return None
    if hostport.startswith("["):
        end = hostport.find("]")
        if end == -1:
            return None
        host = hostport[1:end]
        rest = hostport[end + 1:]
        if not rest.startswith(":"):
            return None
        port = rest[1:]
        if not port:
            return None
        return host, port
    if hostport.count(":") != 1:
        return None             # 0 个（无端口）或多个（裸 IPv6）都算失败
    host, _, port = hostport.partition(":")
    if not port:
        return None
    return host, port


def Host() -> HostInfo:
    """对应 ``envconfig.Host`` —— 默认 ``http://127.0.0.1:11434``。

    端口默认值随 scheme 走（``http`` → 80、``https`` → 443），这是上游的隐含行为；
    漏掉它的话 ``OLLAMA_HOST=https://example.com`` 会连到 ``example.com:11434``。
    """
    default_port = "11434"
    raw = Var("OLLAMA_HOST")

    if "://" not in raw:
        scheme, hostport = "http", raw
        if raw == "ollama.com":
            scheme, hostport = "https", "ollama.com:443"
    else:
        scheme, _, hostport = raw.partition("://")
        if scheme == "http":
            default_port = "80"
        elif scheme == "https":
            default_port = "443"

    hostport, _, path = hostport.partition("/")

    parsed = _split_host_port(hostport)
    if parsed is None:
        host, port = "127.0.0.1", default_port
        ip = _parse_ip(hostport.strip("[]"))
        if ip is not None:
            host = ip
        elif hostport:
            host = hostport
    else:
        host, port = parsed

    try:
        number = int(port)
        if number > 65535 or number < 0:
            raise ValueError
    except (TypeError, ValueError):
        port = default_port       # 上游此处 slog.Warn 后回落，不抛

    return HostInfo(scheme=scheme, host=host, port=port, path=path)


def ConnectableHost() -> HostInfo:
    """对应 ``envconfig.ConnectableHost`` —— 绑定地址换成回环地址。

    ``OLLAMA_HOST=0.0.0.0:11434`` 是**服务端**写法；客户端拿它去连在 Windows 上必失败
    （Windows 不接受把通配地址当目的地），所以连之前必须换成 ``127.0.0.1``。
    """
    info = Host()
    try:
        address = ipaddress.ip_address(info.host)
    except ValueError:
        return info
    if not address.is_unspecified:
        return info
    info.host = "127.0.0.1" if address.version == 4 else "::1"
    return info


def AllowedOrigins() -> list[str]:
    """对应 ``envconfig.AllowedOrigins``（CORS 白名单，本仓 FastAPI 侧可直接用）。"""
    origins: list[str] = []
    raw = Var("OLLAMA_ORIGINS")
    if raw:
        origins.extend(raw.split(","))
    for origin in ("localhost", "127.0.0.1", "0.0.0.0"):
        host = f"[{origin}]:*" if ":" in origin else f"{origin}:*"
        origins.extend([
            f"http://{origin}", f"https://{origin}",
            f"http://{host}", f"https://{host}",
        ])
    origins.extend(["app://*", "file://*", "tauri://*", "vscode-webview://*", "vscode-file://*"])
    return origins


# --------------------------------------------------------------------------- #
# 路径与时长
# --------------------------------------------------------------------------- #


def Models() -> str:
    """对应 ``envconfig.Models`` —— 默认 ``~/.ollama/models``。

    ⚠️ 这是 ollama **本地仓库**的根（下面还有 ``blobs/`` 与 ``manifests/``），不是本仓
    ``models/`` 那个推理权重目录。两者不要混。
    """
    raw = Var("OLLAMA_MODELS")
    if raw:
        return raw
    return str(Path.home() / ".ollama" / "models")


def _duration_from_env(key: str, default: float) -> float:
    raw = Var(key)
    if not raw:
        return default
    try:
        return parse_duration(raw)
    except ValueError:
        pass
    try:
        return float(int(raw))      # 上游允许裸数字，单位是秒
    except ValueError:
        return default


def KeepAlive() -> float:
    """模型在显存里的驻留时长（秒）。负值视为无穷，``0`` 视为不留驻。"""
    value = _duration_from_env("OLLAMA_KEEP_ALIVE", 5 * 60.0)
    return math.inf if value < 0 else value


def LoadTimeout() -> float:
    """加载阶段判定「卡住」的时长（秒）。非正值视为无穷。"""
    value = _duration_from_env("OLLAMA_LOAD_TIMEOUT", 5 * 60.0)
    return math.inf if value <= 0 else value


def Remotes() -> list[str]:
    """允许的远程推理主机（对应 ``OLLAMA_REMOTES``，默认只有 ``ollama.com``）。"""
    raw = Var("OLLAMA_REMOTES").strip()
    return raw.split(",") if raw else ["ollama.com"]


# --------------------------------------------------------------------------- #
# 取值的闭包（保持「每次现读」的语义）
# --------------------------------------------------------------------------- #


def BoolWithDefault(key: str) -> Callable[[bool], bool]:
    """对应 ``BoolWithDefault`` —— 返回闭包，**每次调用都重读环境变量**。

    ⚠️ 上游 ``ParseBool`` 失败时返回 ``true``，照抄：显式写了非法值等于「开启」，
    因为用户写这个变量本身的意图就是开。
    """

    def read(default_value: bool) -> bool:
        raw = Var(key)
        if not raw:
            return default_value
        try:
            return parse_bool(raw)
        except ValueError:
            return True

    return read


def Bool(key: str) -> Callable[[], bool]:
    with_default = BoolWithDefault(key)
    return lambda: with_default(False)


def String(key: str) -> Callable[[], str]:
    return lambda: Var(key)


def Uint(key: str, default_value: int) -> Callable[[], int]:
    def read() -> int:
        raw = Var(key)
        if not raw:
            return default_value
        try:
            number = int(raw)
        except ValueError:
            return default_value
        if number < 0:
            return default_value
        return number

    return read


def Uint64(key: str, default_value: int) -> Callable[[], int]:
    return Uint(key, default_value)


# --------------------------------------------------------------------------- #
# 日志级别
# --------------------------------------------------------------------------- #

LOG_LEVELS = {"info": 0, "debug": -4, "warn": 4, "error": 8, "trace": -8}


def LogLevel() -> str:
    """对应 ``OLLAMA_DEBUG`` 的映射：``1``/``true`` → debug，``2`` → trace，``0`` → info。"""
    raw = Var("OLLAMA_DEBUG")
    if not raw:
        return "info"
    try:
        enabled = parse_bool(raw)
    except ValueError:
        enabled = False
    if enabled:
        return "debug"
    try:
        number = int(raw)
    except ValueError:
        return "info"
    if number == 2:
        return "trace"
    if number != 0:
        return "debug"
    return "info"


# --------------------------------------------------------------------------- #
# ~/.ollama/server.json
# --------------------------------------------------------------------------- #


def _server_config_path() -> Path:
    return Path.home() / ".ollama" / "server.json"


_server_cache: dict[str, Any] = {"loaded": False, "value": {}}


def _load_server_config() -> dict[str, Any]:
    if _server_cache["loaded"]:
        return _server_cache["value"]
    value: dict[str, Any] = {}
    try:
        value = json.loads(_server_config_path().read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        value = {}
    _server_cache["value"] = value
    _server_cache["loaded"] = True
    return value


def ReloadServerConfig() -> None:
    """刷新 ``~/.ollama/server.json`` 的缓存（对应 ``ReloadServerConfig``）。"""
    _server_cache["loaded"] = False
    _server_cache["value"] = {}
    _load_server_config()


def NoCloud() -> bool:
    """云功能是否被关掉 —— 环境变量 ``OLLAMA_NO_CLOUD`` **或** ``server.json`` 任一为真。"""
    if Var("OLLAMA_NO_CLOUD") and _safe_bool(Var("OLLAMA_NO_CLOUD")):
        return True
    return bool(_load_server_config().get("disable_ollama_cloud"))


def NoCloudSource() -> str:
    """对应 ``NoCloudSource``：``"none"`` / ``"env"`` / ``"config"`` / ``"both"``。"""
    raw = Var("OLLAMA_NO_CLOUD")
    env_disabled = bool(raw) and _safe_bool(raw)
    config_disabled = bool(_load_server_config().get("disable_ollama_cloud"))
    if env_disabled and config_disabled:
        return "both"
    if env_disabled:
        return "env"
    if config_disabled:
        return "config"
    return "none"


def _safe_bool(raw: str) -> bool:
    try:
        return parse_bool(raw)
    except ValueError:
        return True


# --------------------------------------------------------------------------- #
# 一览表
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EnvVar:
    """对应 ``envconfig.EnvVar``（``ollama config`` 的对外形状）。"""

    name: str
    value: Any
    description: str


def AsMap() -> dict[str, EnvVar]:
    """对应 ``AsMap`` —— 每项的 ``value`` 是**求值后**的结果。"""
    entries = [
        EnvVar("OLLAMA_DEBUG", LogLevel(), "Show additional debug information (e.g. OLLAMA_DEBUG=1)"),
        EnvVar("OLLAMA_DEBUG_LOG_REQUESTS", Bool("OLLAMA_DEBUG_LOG_REQUESTS")(),
               "Log inference request bodies and replay curl commands to a temp directory"),
        EnvVar("OLLAMA_GO_TEMPLATE", BoolWithDefault("OLLAMA_GO_TEMPLATE")(True),
               "Enable Modelfile TEMPLATE based rendering when available"),
        EnvVar("OLLAMA_FLASH_ATTENTION", BoolWithDefault("OLLAMA_FLASH_ATTENTION")(False),
               "Enabled flash attention"),
        EnvVar("OLLAMA_KV_CACHE_TYPE", Var("OLLAMA_KV_CACHE_TYPE"),
               "Quantization type for the K/V cache (default: f16)"),
        EnvVar("OLLAMA_GPU_OVERHEAD", Uint64("OLLAMA_GPU_OVERHEAD", 0)(),
               "Reserve a portion of VRAM per GPU (bytes)"),
        EnvVar("OLLAMA_IGPU_ENABLE", Var("OLLAMA_IGPU_ENABLE"), "Enable integrated GPUs"),
        EnvVar("LLAMA_ARG_FIT", Var("LLAMA_ARG_FIT"),
               'Enable llama.cpp automatic fit of unset memory options (default "on")'),
        EnvVar("LLAMA_ARG_FIT_TARGET", Var("LLAMA_ARG_FIT_TARGET"),
               "Target free VRAM margin per device for llama.cpp fit (MiB)"),
        EnvVar("OLLAMA_HOST", Host(), "IP Address for the ollama server (default 127.0.0.1:11434)"),
        EnvVar("OLLAMA_KEEP_ALIVE", KeepAlive(),
               'The duration that models stay loaded in memory (default "5m")'),
        EnvVar("OLLAMA_LLM_LIBRARY", Var("OLLAMA_LLM_LIBRARY"),
               "Set LLM library to bypass autodetection"),
        EnvVar("OLLAMA_LOAD_TIMEOUT", LoadTimeout(),
               'How long to allow model loads to stall before giving up (default "5m")'),
        EnvVar("OLLAMA_MAX_LOADED_MODELS", Uint("OLLAMA_MAX_LOADED_MODELS", 0)(),
               "Maximum number of loaded models per GPU"),
        EnvVar("OLLAMA_MAX_TRANSFER_STREAMS", Uint("OLLAMA_MAX_TRANSFER_STREAMS", 4)(),
               "Maximum parallel transfer streams for safetensors model pulls/pushes (default 4)"),
        EnvVar("OLLAMA_MAX_QUEUE", Uint("OLLAMA_MAX_QUEUE", 512)(),
               "Maximum number of queued requests"),
        EnvVar("OLLAMA_MODELS", Models(), "The path to the models directory"),
        EnvVar("OLLAMA_NO_CLOUD", NoCloud(),
               "Disable Ollama cloud features (remote inference and web search)"),
        EnvVar("OLLAMA_NOHISTORY", Bool("OLLAMA_NOHISTORY")(), "Do not preserve readline history"),
        EnvVar("OLLAMA_NOPRUNE", Bool("OLLAMA_NOPRUNE")(), "Do not prune model blobs on startup"),
        EnvVar("OLLAMA_NUM_PARALLEL", Uint("OLLAMA_NUM_PARALLEL", 1)(),
               "Maximum number of parallel requests"),
        EnvVar("OLLAMA_ORIGINS", AllowedOrigins(), "A comma separated list of allowed origins"),
        EnvVar("OLLAMA_SCHED_SPREAD", Bool("OLLAMA_SCHED_SPREAD")(),
               "Always schedule model across all GPUs"),
        EnvVar("OLLAMA_CONTEXT_LENGTH", Uint("OLLAMA_CONTEXT_LENGTH", 0)(),
               "Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)"),
        EnvVar("OLLAMA_CREATE_REMOTE", Bool("OLLAMA_CREATE_REMOTE")(),
               "Force model creation through the server API even when the server is local"),
        EnvVar("OLLAMA_EDITOR", Var("OLLAMA_EDITOR"),
               "Path to editor for interactive prompt editing (Ctrl+G)"),
        EnvVar("OLLAMA_REMOTES", Remotes(), 'Allowed hosts for remote models (default "ollama.com")'),
        EnvVar("HTTP_PROXY", Var("HTTP_PROXY"), "HTTP proxy"),
        EnvVar("HTTPS_PROXY", Var("HTTPS_PROXY"), "HTTPS proxy"),
        EnvVar("NO_PROXY", Var("NO_PROXY"), "No proxy"),
        EnvVar("CUDA_VISIBLE_DEVICES", Var("CUDA_VISIBLE_DEVICES"), "Set which NVIDIA devices are visible"),
        EnvVar("HIP_VISIBLE_DEVICES", Var("HIP_VISIBLE_DEVICES"),
               "Set which AMD devices are visible by numeric ID"),
        EnvVar("ROCR_VISIBLE_DEVICES", Var("ROCR_VISIBLE_DEVICES"),
               "Set which AMD devices are visible by UUID or numeric ID"),
        EnvVar("GGML_VK_VISIBLE_DEVICES", Var("GGML_VK_VISIBLE_DEVICES"),
               "Set which Vulkan devices are visible by numeric ID"),
        EnvVar("GPU_DEVICE_ORDINAL", Var("GPU_DEVICE_ORDINAL"),
               "Set which AMD devices are visible by numeric ID"),
        EnvVar("HSA_OVERRIDE_GFX_VERSION", Var("HSA_OVERRIDE_GFX_VERSION"),
               "Override the gfx used for all detected AMD GPUs"),
        EnvVar("OLLAMA_VULKAN", BoolWithDefault("OLLAMA_VULKAN")(True), "Enable Vulkan support"),
    ]
    return {entry.name: entry for entry in entries}


def Values() -> dict[str, str]:
    """对应 ``Values`` —— 全部值字符串化（供 ``GET /api/config`` 之类直接吐）。"""
    out: dict[str, str] = {}
    for key, entry in AsMap().items():
        value = entry.value
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, float):
            out[key] = format_duration(value)
        elif isinstance(value, (list, tuple)):
            out[key] = ",".join(str(item) for item in value)
        elif isinstance(value, HostInfo):
            out[key] = value.base_url
        else:
            out[key] = "" if value is None else str(value)
    return out
