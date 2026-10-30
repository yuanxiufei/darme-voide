"""**运行时体检** —— 回答一句话：这条流水线**现在**能不能真的跑 ✓（零依赖 ✓ 2026-10-30）。

## 它解决什么真问题

:mod:`.production_preflight` 查的是**内容与资产**（提示词落位、连续性、资产门 ✓），
:mod:`.engine_readiness` 查的是**引擎权重**（哪件在盘上、缺多少 GiB ✓）——
两者都**不看**「服务到底通不通」✗✗。

而本机的真实形状是：四类 AI 服务**全部**指向本机常驻进程（见 ``ai_service_configs``），
任何一类没起，自动管线就会**跑到一半**才崩 ✗（出图阶段挂在 HTTP 连接上 ✓，
而用户看到的只是 `auto:failed` ✓ —— 报错与真因隔着五层 ✓）。

所以本模块把「**配了没有 / 服务活着没有 / 依赖满不满足 / 不通怎么修**」
压成一条命令 ✓：``python -m app.services.runtime_doctor`` ✓

## 事实来源（不猜 ✓）

* 配置：``ai_service_configs`` 表（经 :func:`app.services.ai_providers.get_active_config`
  取「生效那条」✓ —— 优先序、``is_active``、``model`` 解析**只有那一份口径** ✗）；
* 探活请求：:func:`app.services.provider_probe.build_probe`（按厂商拼「最小可判活请求」✓
  —— 本地服务的前缀与路径也由它一处决定 ✓，本模块**不自己拼 URL** ✗）；
* provider 是否存在：:mod:`app.services.adapters.registry`（**未知 provider 显式抛错** ✓）；
* 进程内引擎：:func:`app.services.engine.bridge.is_engine_provider`（它**不是 HTTP 服务** ✓
  ⇒ 探端口会永远得出「离线」这种假结论 ✗，必须单独分叉 ✓）。

## 判定口径（三态，别混 ✓）

* ``ok``        —— 端口通、端点应答 ✓（4xx 也算 **活着** ✓：说明服务在、只是这个探测端点不认 ✓）；
* ``offline``   —— 连接被拒/超时 ⇒ **服务没起** ✓（最常见 ✓）；
* ``missing``   —— 库里没有生效配置 ⇒ 根本没配 ✓（与「配了但没起」是两件事 ✓）。

⚠️ 依赖只做 **warning** 不做 blocker ✓：视频阶段优先走图生视频 ✓，但路由里有 T2V 分支
（:mod:`.shot_router`）⇒ 图像离线时视频仍可能跑起来，只是质量掉一档 ✓ —— 把它判成
「阻断」会撒谎 ✗。
"""
from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import select

from ..core.db import engine as db_engine
from ..core.models import ai_service_configs
from .adapters import registry as adapter_registry
from .ai_providers import get_active_config
from .provider_probe import build_probe, redact_url

__all__ = [
    "DEPENDS_ON",
    "FIX_HINTS",
    "SERVICE_LABEL",
    "SERVICE_ORDER",
    "DoctorReport",
    "check_service",
    "main",
    "run_doctor",
]

#: 体检顺序 **按管线依赖排** ✓（文本最前 ✓ 视频最后 ✓ —— 与 ``auto_pipeline.STAGE_SEQUENCE`` 同向）
SERVICE_ORDER: tuple[str, ...] = ("text", "image", "audio", "video")

#: 人读标签（报告里用 ✓）
SERVICE_LABEL: dict[str, str] = {
    "text": "文本（剧本 / 提取 / 分镜拆解）",
    "image": "图像（角色板 / 场景卡 / 分镜首帧）",
    "audio": "音频（配音 / 音色克隆）",
    "video": "视频（分镜片段 / 首尾帧连接）",
}

#: 依赖（**只报 warning** ✓ 理由见模块头）
DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "text": (),
    "image": (),
    "audio": (),
    "video": ("image",),
}

#: 各类 provider 的启动 / 配置指引（**只在本地服务上给命令** ✓ 云端只提示填 key ✗ 不乱猜）
FIX_HINTS: dict[str, str] = {
    "ollama": "启动 Ollama：`ollama serve`（默认 11434）；缺模型：`ollama pull <模型名>`",
    "local-sd": "启动 SD WebUI 并**带 `--api`**（默认 7860）；未开 --api 时 /sdapi/* 恒 404",
    "cosyvoice": "启动 CosyVoice 服务（默认 9880）；确认 baseUrl 的端口与它一致",
    "minimax": "baseUrl 指向本机时是 H3 薄封装：启动 backend-py/app/local_services/h3（默认 8765）；"
               "指向云端时检查 apiKey",
    "engine": "进程内自研引擎，**不需要独立端口**；权重是否齐备："
              "`python -m app.services.engine readiness`",
    "chatfire": "云端聚合站：检查 apiKey 与余额",
    "openai": "云端：检查 apiKey",
    "openrouter": "云端：检查 apiKey",
    "gemini": "云端：检查 apiKey",
    "volcengine": "云端方舟：检查 apiKey 与推理接入点",
    "ali": "云端百炼：检查 apiKey",
    "vidu": "云端：检查 apiKey",
}

#: 默认探测超时（秒）—— 本机服务应当毫秒级应答 ✓ 超过 2 秒等于没起 ✓
DEFAULT_TIMEOUT = 2.0


@dataclass
class ServiceCheck:
    """一类服务的体检结论 ✓（``state`` 只有 ok/offline/missing/degraded 四态 ✓）。"""

    service: str
    label: str
    provider: str = ""
    model: str = ""
    base_url: str = ""
    state: str = "missing"
    status_code: int | None = None
    probe_url: str = ""
    note: str = ""
    fix: str = ""
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "baseUrl": self.base_url,
            "state": self.state,
            "statusCode": self.status_code,
            "probeUrl": self.probe_url,
            "note": self.note,
            "fix": self.fix,
            "dependencies": list(self.dependencies),
        }


@dataclass
class DoctorReport:
    ready: bool = False
    services: list[ServiceCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nextActions: list[str] = field(default_factory=list)  # noqa: N815

    def to_dict(self) -> dict[str, Any]:
        online = [s.service for s in self.services if s.state == "ok"]
        return {
            "ready": self.ready,
            "online": online,
            "services": [s.to_dict() for s in self.services],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "nextActions": self.nextActions,
            "summary": {
                "total": len(self.services),
                "online": len(online),
                "offline": len([s for s in self.services if s.state == "offline"]),
                "missing": len([s for s in self.services if s.state == "missing"]),
                "degraded": len([s for s in self.services if s.state == "degraded"]),
            },
        }


def _is_local_url(base_url: str) -> bool:
    """``baseUrl`` 是否指向本机（本机服务的「没起」与云端「key 错」是两种修法 ✓）。"""
    try:
        parts = urlsplit(base_url or "")
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _adapter_table(service_type: str) -> dict[str, Any]:
    """该类服务可用的适配器表 ✓（**注册表是唯一口径** ✗ 本模块不另立清单 ✓）。"""
    return {
        "text": adapter_registry.text_adapters,
        "image": adapter_registry.image_adapters,
        "audio": adapter_registry.tts_adapters,
        "video": adapter_registry.video_adapters,
    }.get(service_type) or {}


def _provider_known(service_type: str, provider: str) -> bool:
    """该 provider 在本类服务里**有适配器**吗 ✓。"""
    return str(provider or "").lower() in _adapter_table(service_type)


def _engine_provider(provider: str) -> bool:
    """是否进程内引擎（**不是 HTTP 服务** ✓ ⇒ 绝不能靠探端口判活 ✗）。"""
    try:
        from .engine import bridge
    except Exception:  # noqa: BLE001 —— 引擎可选依赖缺失时不该让体检整体崩 ✓
        return str(provider or "").lower() == "engine"
    return bridge.is_engine_provider(provider)


def _ipv4_variant(url: str) -> str | None:
    """``localhost`` ⇒ ``127.0.0.1`` 的等价 URL ✓（**Windows 专项** ✓ 见 :func:`_probe`）。

    Windows 上 ``localhost`` 常**先解析到 ``::1``**，而多数本地服务只监听 IPv4 ``0.0.0.0``
    ⇒ ``urllib`` 会一路白等到 timeout，报出来是「超时」而不是「连接被拒」✗ ——
    两种错的修法完全不同（前者像服务卡死、后者明确是没起 ✓），所以必须先纠回 IPv4 再探 ✓。
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if (parts.hostname or "").lower() not in ("localhost", "::1"):
        return None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return urlunsplit((parts.scheme, f"127.0.0.1:{port}", parts.path, parts.query, parts.fragment))


def _describe_error(err: BaseException) -> str:
    """把网络异常翻成**能直接对症**的一句话 ✓（别只回显 ``URLError: timed out`` ✗）。"""
    reason = getattr(err, "reason", None)
    root = reason if isinstance(reason, BaseException) else err
    if isinstance(root, ConnectionRefusedError):
        return "连接被拒 ⇒ 端口没有进程在监听（服务没起 ✓）"
    if isinstance(root, (socket.timeout, TimeoutError)):
        return "连接超时 ⇒ 端口无响应（进程卡死 / 防火墙拦 / 只监听别的地址 ✓）"
    if isinstance(root, OSError) and getattr(root, "errno", None) in (10061, 111):
        return "连接被拒 ⇒ 端口没有进程在监听（服务没起 ✓）"
    if isinstance(root, OSError) and getattr(root, "errno", None) in (10060, 110):
        return "连接超时 ⇒ 端口无响应（进程卡死 / 防火墙拦 ✓）"
    return f"{type(root).__name__}: {root}"


def _tcp_ping(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """只做 TCP 握手 ⇒ 秒级判「端口有没有进程」✓（**不走 HTTP** ✗ 见 :func:`_probe`）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except ConnectionRefusedError:
        return False, "连接被拒 ⇒ 端口没有进程在监听（服务没起 ✓）"
    except (socket.timeout, TimeoutError):
        return False, "连接超时 ⇒ 端口无响应（安全软件 DROP / 只监听别的地址 ✓）"
    except OSError as err:
        return False, _describe_error(err)


def _probe(desc: dict[str, Any], timeout: float) -> tuple[str, int | None, str]:
    """发一次探测请求 ⇒ ``(state, status_code, note)`` ✓（**本函数绝不抛** ✗）。"""
    method = str(desc.get("method") or "GET").upper()
    headers = {str(k): str(v) for k, v in (desc.get("headers") or {}).items()}
    body = desc.get("body")
    data = None
    if body is not None and method != "GET":
        # 紧凑分隔符：与 Node 的 ``JSON.stringify`` 逐字同形（探针会打服务端，格式要对齐）
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    raw_url = str(desc.get("url") or "")

    # ① 本机地址先做 TCP 预连（0.8s 上限）⇒ 端口没进程就**立刻**收工 ✓
    #    理由：HTTP 层失败要白等满 timeout 才知道没连上 ✗，而「端口通不通」是更便宜的问题 ✓
    try:
        parts = urlsplit(raw_url)
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        host, port = "", 0
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") and port:
        reachable, why = _tcp_ping("127.0.0.1", port, min(timeout, 0.8))
        if not reachable:
            return "offline", None, why

    # 本机地址先试 IPv4 等价形（见 ``_ipv4_variant``），原 URL 留作兜底 ✓
    candidates = [c for c in (_ipv4_variant(raw_url), raw_url) if c]
    note = ""
    for url in candidates:
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return "ok", int(getattr(response, "status", 200)), ""
        except urllib.error.HTTPError as err:
            # 4xx：**服务是活的** ✓ 只是这个探测端点不认 ✓（cosyvoice 的 `/` 常见 404）
            code = int(err.code)
            if code < 500:
                return "ok", code, f"服务在线，但探测端点返回 {code}（多数本地服务的 `/` 不实现 ✓）"
            return "degraded", code, f"服务在线但报 {code}（进程活着、内部异常 ✓）"
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as err:
            note = _describe_error(err)
    return "offline", None, note


def check_service(service_type: str, *, timeout: float = DEFAULT_TIMEOUT,
                  probe: bool = True) -> ServiceCheck:
    """体检**一类**服务 ✓（配置 → provider 存在性 → 探活 → 依赖 ✓）。"""
    check = ServiceCheck(
        service=service_type,
        label=SERVICE_LABEL.get(service_type, service_type),
        dependencies=DEPENDS_ON.get(service_type, ()),
    )
    with db_engine.begin() as conn:
        config = get_active_config(conn, service_type)

    if not config:
        check.state = "missing"
        check.note = f"``ai_service_configs`` 里没有生效的 {service_type} 配置 ✗"
        check.fix = f"在「设置 → AI 服务」里添加一条 {service_type} 配置并设为生效"
        return check

    provider = str(config.get("provider") or "")
    check.provider = provider
    check.model = str(config.get("model") or "")
    check.base_url = str(config.get("baseUrl") or "")
    check.fix = FIX_HINTS.get(provider.lower(), "")

    # ① 进程内引擎：**不探端口** ✓
    if _engine_provider(provider):
        check.state = "ok"
        check.note = "进程内自研引擎（不依赖外部服务 ✓）"
        return check

    # ② provider 拼写/未实现：配置错就要当场说 ✗ 别让它走到生成阶段才炸 ✓
    if not _provider_known(service_type, provider):
        check.state = "missing"
        check.note = f"provider ``{provider}`` 在 {service_type} 适配器注册表里不存在 ✗"
        known = ", ".join(sorted(_adapter_table(service_type))) or "（该表为空 ✗）"
        check.fix = f"把 provider 改成已实现的之一：{known}"
        return check

    # ③ 探活（不探就只报配置存在 ✓ —— ``--no-probe`` 用）
    desc = build_probe(service_type, provider, check.base_url, check.model or None,
                       config.get("apiKey"))
    check.probe_url = redact_url(str(desc.get("url") or ""))
    if not probe:
        check.state = "ok"
        check.note = "已配置（本次未探活 ✓）"
        return check

    state, code, note = _probe(desc, timeout)
    # comfyui 的 404 是**真问题**（插件没加载 ⇒ 渲染层不可用），不能按通用规则算「活着」✓
    if state == "ok" and provider.lower() == "comfyui" and code == 404:
        state = "degraded"
        note = ("ComfyUI 进程在线，但 /comfy-studio/renders 返回 404 ⇒ "
                "comfy_studio 插件未加载，渲染层不可用 ✗")
    check.state = state
    check.status_code = code
    check.note = note
    if state == "offline" and _is_local_url(check.base_url):
        check.fix = check.fix or "本机服务没起 —— 见 baseUrl 的端口，启动对应进程"
    return check


def run_doctor(*, timeout: float = DEFAULT_TIMEOUT, probe: bool = True) -> DoctorReport:
    """全量体检 ⇒ 结论 + 阻断 + 可执行顺序 ✓（顺序＝先便宜后贵 ✓ 见模块头）。"""
    report = DoctorReport()
    report.services = [check_service(name, timeout=timeout, probe=probe)
                       for name in SERVICE_ORDER]

    by_service = {item.service: item for item in report.services}

    for item in report.services:
        if item.state == "missing":
            report.blockers.append(f"{item.label}：{item.note}")
        elif item.state == "offline":
            report.blockers.append(f"{item.label}：服务离线 —— {item.note}")
        elif item.state == "degraded":
            report.warnings.append(f"{item.label}：{item.note}")

    # 依赖降级（**只 warning** ✓ 见模块头）
    for item in report.services:
        for dep in item.dependencies:
            dep_item = by_service.get(dep)
            if dep_item is not None and dep_item.state in ("offline", "missing"):
                report.warnings.append(
                    f"{item.label} 依赖 {dep_item.label}，而它当前 {dep_item.state} ⇒ "
                    f"{item.service} 阶段会退化（仍可能跑，但产出质量下降 ✓）")

    report.ready = not report.blockers
    report.nextActions = _next_actions(report)
    return report


def _next_actions(report: DoctorReport) -> list[str]:
    """把阻断翻成**可执行顺序** ✓（按 ``SERVICE_ORDER`` —— 文本最先、视频最后 ✓）。"""
    actions: list[str] = []
    for item in report.services:
        if item.state == "missing":
            actions.append(f"① {item.label} 缺配置 / provider 不存在 ⇒ {item.fix}")
        elif item.state == "offline":
            actions.append(f"② {item.label} 服务没起 ⇒ {item.fix or '启动该服务'}")
    if not actions and report.warnings:
        actions.append("服务都能用 ✓；仅有降级建议，见 warnings ✓")
    if report.ready and not report.warnings:
        actions.append("✅ 四类服务全通 ⇒ 可以直接跑 `auto_pipeline` 全流程 ✓")
    return actions


def format_report(report: DoctorReport) -> str:
    """人读报告（终端直出 ✓ 不依赖前端 ✓）。"""
    lines: list[str] = []
    payload = report.to_dict()
    summary = payload["summary"]
    lines.append(f"运行时体检：{summary['online']}/{summary['total']} 在线"
                 f"（离线 {summary['offline']} / 未配 {summary['missing']}"
                 f" / 降级 {summary['degraded']}）"
                 f" ⇒ {'可跑' if report.ready else '有阻断'}")
    lines.append("-" * 68)
    icon = {"ok": "OK  ", "offline": "离线", "missing": "未配", "degraded": "降级"}
    for item in report.services:
        lines.append(f"[{icon.get(item.state, item.state)}] {item.label}")
        lines.append(f"       provider={item.provider or '-'}  model={item.model or '-'}")
        lines.append(f"       baseUrl={item.base_url or '-'}")
        if item.probe_url:
            lines.append(f"       probe={item.probe_url}"
                         + (f"  → {item.status_code}" if item.status_code else ""))
        if item.note:
            lines.append(f"       {item.note}")
        if item.fix and item.state in ("offline", "missing"):
            lines.append(f"       修：{item.fix}")
    if report.blockers:
        lines.append("-" * 68)
        lines.append("阻断：")
        lines.extend(f"  ✗ {line}" for line in report.blockers)
    if report.warnings:
        lines.append("降级（不阻断）：")
        lines.extend(f"  ! {line}" for line in report.warnings)
    if report.nextActions:
        lines.append("-" * 68)
        lines.append("下一步：")
        lines.extend(f"  → {line}" for line in report.nextActions)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI ⇒ ``python -m app.services.runtime_doctor [--json] [--no-probe]`` ✓（有阻断退出 1）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    probe = "--no-probe" not in args
    report = run_doctor(probe=probe)
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=1, separators=(",", ": ")))
    else:
        print(format_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
