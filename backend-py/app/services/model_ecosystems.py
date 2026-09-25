"""本机模型生态落点 —— 「世界各大模型」在电脑里分别落在哪。

为什么需要（2026-09-25 用户口径 ✓）：扫描要覆盖**世界各大模型** ✓，不能只认 Ollama ✗ ——
``local_model_scan`` 原来只扫本仓 ``models/``、``local_services/`` 与 ComfyUI ✗ ⇒
HuggingFace / ModelScope / LM Studio / GPT4All / Jan / llama.cpp 里躺着的模型**扫不到**（而且不报错 ✗），
用户会以为「本机没模型」✗✗。

本模块把「各生态的本地落点」收成**一张表** ✓，供扫描器当默认根目录，并给每个命中文件标注来源生态 ✓。

三条口径（都在扫描结果里如实体现 ✓，不许含糊 ✗）：

1. **零外部依赖** ✓：全部是**读目录** ✓ —— 不要求任何生态的服务在跑（``ollama serve`` /
   LM Studio / vLLM…）✗，这正是用户口径的原话 ✓；
2. **缓存 ≠ 能推理** ✓：读到的只是权重文件 ✓ ⇒ 默认 ``callable=False`` + ``note`` 说清**怎么才真能用** ✓
   （本仓铁律：**没查 ≠ 通过** ✓）；
3. **只读** ✓：只列文件，不删/不改别人生态的库 ✓（删除要由那个生态自己的服务做 ✓）。

⚠️ 与 ``_RULES``（启发式识别）**分工不同** ✗：规则回答「这是**什么**权重」✓，
本模块回答「它**在哪**、能不能**直接调**」✓。两者冲突时**规则优先** ✓（见 :func:`apply_ecosystem`）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from . import ollama_store

__all__ = [
    "ECOSYSTEMS",
    "OVERRIDE_RUNTIMES",
    "Ecosystem",
    "all_roots",
    "apply_ecosystem",
    "build_suggestion",
    "by_ecosystem",
    "get",
    "labels",
    "roots",
]

#: 根目录探测用的环境变量模板（``VAR`` 或 ``VAR/子路径``，见 :func:`_env_path`）
_ENV_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%|\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class Ecosystem:
    """一个生态的本地落点与「怎么用起来」。

    ``candidates`` / ``env`` 里的路径模板支持 ``~``、``%VAR%``、``$VAR`` ✓。
    ``walkable=False`` 表示**文件遍历发现不了**（权重名不是模型扩展名，如 Ollama 的无扩展名 blob ✓），
    需要专用读法（清单）✓ —— 但仍参与 :func:`detect` 的归属判定 ✓。
    """

    id: str
    label: str
    runtime: str
    note: str
    provider: str = ""
    base_url: str = ""
    runnable: bool = False
    env: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    skip_prefixes: tuple[str, ...] = ()
    walkable: bool = True
    kind_hint: str = ""


#: ⭐ 生态表（**声明顺序即优先级** ✓：同一个目录只归**先声明**的生态 ✓）。
#: ⚠️ 只列「模型库/模型缓存」⇒ 不放**代码**目录 ✗；ComfyUI 不在这里 ——
#: 它由 ``local_model_scan.COMFYUI_CANDIDATES`` 单独管（避免循环 import ✓），
#: 且其模型 runtime 本就是 ``comfyui`` ✓，无需生态再标注 ✓。
ECOSYSTEMS: tuple[Ecosystem, ...] = (
    Ecosystem(
        id="ollama",
        label="Ollama 模型库",
        runtime="ollama",
        provider="openai",
        base_url="http://localhost:11434",
        runnable=True,
        note="已在本机 Ollama 模型库里（服务没起也看得见）；调用需 Ollama 服务在跑",
        env=("OLLAMA_MODELS",),
        candidates=tuple(ollama_store.models_root_candidates()),
        # ⚠️ 权重是无扩展名的 blob ⇒ 文件遍历**发现不了** ✗，改由 manifests 清单读出 ✓
        walkable=False,
        kind_hint="text",
    ),
    Ecosystem(
        id="huggingface",
        label="HuggingFace 缓存",
        runtime="huggingface",
        note="HuggingFace 本地缓存（只读）；要调用得用 transformers/diffusers/vLLM 等加载，"
             "或先转成 ComfyUI / Ollama 能用的格式",
        env=("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
             "HF_HOME/hub", "XDG_CACHE_HOME/huggingface/hub"),
        candidates=("~/.cache/huggingface/hub", "~/.huggingface/hub"),
        # ⚠️ `datasets--*` 是**数据集**不是模型 ✗；`.no_exist` 是「查过、不存在」的负缓存标记 ✗
        skip_prefixes=("datasets--", ".no_exist"),
    ),
    Ecosystem(
        id="modelscope",
        label="ModelScope 缓存",
        runtime="modelscope",
        note="ModelScope（魔搭）本地缓存（只读）；要调用得用 modelscope/transformers 加载",
        env=("MODELSCOPE_CACHE", "MODELSCOPE_CACHE/hub", "MODELSCOPE_HOME/hub"),
        candidates=("~/.cache/modelscope/hub", "~/.modelscope/hub"),
        skip_prefixes=("datasets--", "._____temp"),
    ),
    Ecosystem(
        id="lmstudio",
        label="LM Studio",
        runtime="lmstudio",
        provider="openai",
        base_url="http://localhost:1234/v1",
        note="LM Studio 模型库（只读，多为 GGUF）；需在 LM Studio 里启动本地服务"
             "（默认 http://localhost:1234/v1）后才能调用",
        candidates=("~/.lmstudio/models", "~/.cache/lm-studio/models"),
        skip_prefixes=(".internal",),
    ),
    Ecosystem(
        id="gpt4all",
        label="GPT4All",
        runtime="gpt4all",
        provider="openai",
        base_url="http://localhost:4891/v1",
        note="GPT4All 模型库（只读，GGUF）；需在 GPT4All 里开启 API Server（默认 4891）后才能调用",
        candidates=("%LOCALAPPDATA%/nomic.ai/GPT4All", "~/.local/share/nomic.ai/GPT4All",
                    "~/Library/Application Support/nomic.ai/GPT4All"),
    ),
    Ecosystem(
        id="jan",
        label="Jan",
        runtime="jan",
        provider="openai",
        base_url="http://localhost:1337/v1",
        note="Jan 模型库（只读）；需在 Jan 里启动本地 API（默认 1337）后才能调用",
        candidates=("%APPDATA%/Jan/data/models", "~/Library/Application Support/Jan/data/models",
                    "~/.config/Jan/data/models", "~/jan/models"),
    ),
    Ecosystem(
        id="llamacpp",
        label="llama.cpp / GGUF 目录",
        runtime="llamacpp",
        provider="openai",
        base_url="http://localhost:8080/v1",
        note="GGUF 权重（只读）；需用 llama-server 起服务（默认 8080），"
             "或 `ollama create <name> -f Modelfile` 导入后才能调用",
        candidates=("~/llama.cpp/models", "~/gguf"),
        kind_hint="text",
    ),
    Ecosystem(
        id="textgen",
        label="text-generation-webui",
        runtime="textgen",
        provider="openai",
        base_url="http://localhost:5000/v1",
        note="text-generation-webui 模型目录（只读）；需以 --api 启动（默认 5000）后才能调用",
        candidates=("~/text-generation-webui/models", "~/oobabooga/text-generation-webui/models"),
    ),
    Ecosystem(
        id="sdwebui",
        label="Stable Diffusion WebUI",
        runtime="local-sd",
        provider="local-sd",
        base_url="http://localhost:7860",
        # ⚠️ WebUI 的 `--api` 是**可调用**的（与本仓既有 local-sd 口径一致 ✓），但得先启起来 ✓
        runnable=True,
        note="Stable Diffusion WebUI 模型目录；以 --api 启动后可直接调用（默认 http://localhost:7860）",
        candidates=("~/stable-diffusion-webui/models", "~/automatic/models"),
    ),
    Ecosystem(
        id="stability-matrix",
        label="Stability Matrix",
        runtime="stability-matrix",
        note="Stability Matrix 的共享模型库（只读）；由它内部的 ComfyUI / InvokeAI 加载",
        candidates=("%APPDATA%/StabilityMatrix/Models", "~/.local/share/StabilityMatrix/Models",
                    "~/Library/Application Support/StabilityMatrix/Models"),
    ),
    Ecosystem(
        id="invokeai",
        label="InvokeAI",
        runtime="invokeai",
        provider="local-sd",
        base_url="http://localhost:9090",
        note="InvokeAI 模型库（只读）；需启动 InvokeAI 服务（默认 9090）后才能调用",
        candidates=("~/invokeai/models",),
    ),
    Ecosystem(
        id="torchhub",
        label="PyTorch Hub 缓存",
        runtime="torch-hub",
        note="PyTorch Hub 缓存的辅助权重（CLIP / SAM 等），一般由其他程序引用，不单独注册",
        env=("TORCH_HOME/hub/checkpoints",),
        candidates=("~/.cache/torch/hub/checkpoints", "%LOCALAPPDATA%/torch/hub/checkpoints"),
    ),
)

#: 生态**可以覆盖**的 runtime ✓：这三个都是「弱命中」—— ``comfyui`` 只说明「ComfyUI 能读」✗、
#: ``unknown`` 什么都没说 ✗、``ollama`` 只是「GGUF 的默认去处」✗。
#: ⚠️ ``h3`` / ``local-sd`` / ``cosyvoice`` 这类**强命中不覆盖** ✗：那是「这是什么权重」的**事实** ✓
#: （HF 缓存里的 H3 DiT 依然是 H3 权重 ⇒ 仍按 H3 注册 ✓）。
OVERRIDE_RUNTIMES = frozenset({"comfyui", "unknown", "ollama"})

_BY_ID: dict[str, Ecosystem] = {spec.id: spec for spec in ECOSYSTEMS}


def get(ecosystem_id: str) -> Ecosystem | None:
    """按 id 取生态（未知 id ⇒ ``None`` ✓）。"""
    return _BY_ID.get(ecosystem_id or "")


def labels() -> dict[str, str]:
    """``{id: 中文标签}``（供前端渲染角标 ✓，避免前后端各写一份 ✗）。"""
    return {spec.id: spec.label for spec in ECOSYSTEMS}


# ============================================================
# 落点发现
# ============================================================

def _env_path(entry: str) -> str | None:
    """把 ``VAR`` / ``VAR/子路径`` 展开成路径 ✓；变量**没设** ⇒ ``None``（**不猜** ✓）。"""
    name, _, suffix = entry.partition("/")
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    parts = [p for p in suffix.split("/") if p]
    return os.path.join(value, *parts) if parts else value


def _expand(template: str) -> str | None:
    """展开 ``~`` / ``%VAR%`` / ``$VAR`` ✓；**有变量没设 ⇒ None**（宁可漏一个落点，不编一个路径 ✓）。

    ⚠️ 自己实现而不用 ``os.path.expandvars`` ✗：后者在 POSIX 上**不认** ``%VAR%`` ⇒
    跨平台测试里会静默留下 ``%APPDATA%`` 字面量 ✗（那是个**总会失败**的候选，还看不出来 ✗）。
    """
    missing = False

    def repl(match: re.Match[str]) -> str:
        nonlocal missing
        name = match.group(1) or match.group(2) or match.group(3) or ""
        value = os.environ.get(name, "").strip()
        if not value:
            missing = True
            return ""
        return value

    text = _ENV_RE.sub(repl, template)
    if missing:
        return None
    return os.path.expanduser(text)


def _candidates(spec: Ecosystem) -> list[str]:
    """某生态的候选落点（**保序** ✓：环境变量 > 平台约定落点 ✓；不存在的也返回 ✓）。"""
    out: list[str] = []
    for entry in spec.env:
        value = _env_path(entry)
        if value:
            out.append(value)
    for template in spec.candidates:
        value = _expand(template)
        if value:
            out.append(value)

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in out:
        resolved = os.path.abspath(raw)
        key = os.path.normcase(resolved)
        if key not in seen:
            seen.add(key)
            cleaned.append(resolved)
    return cleaned


def all_roots() -> list[tuple[str, str]]:
    """``[(生态 id, 落点)]`` —— **全部候选**（含不存在的 ✓，供 :func:`detect` 判归属 ✓）。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in ECOSYSTEMS:
        for path in _candidates(spec):
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            out.append((spec.id, path))
    return out


def roots(walkable_only: bool = True) -> list[tuple[str, str]]:
    """``[(生态 id, 落点)]`` —— 只返回**真实存在**的目录 ✓（扫描器的默认根目录来源 ✓）。

    ``walkable_only=True`` 时排除 ``walkable=False`` 的生态 ✓（它们的权重文件对文件遍历不可见 ✓，
    硬扫只是白跑 ✓）；但 ``detect`` 用 :func:`all_roots` ✓，所以归属判定仍认它们 ✓。
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in ECOSYSTEMS:
        if walkable_only and not spec.walkable:
            continue
        for path in _candidates(spec):
            if not os.path.isdir(path):
                continue
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            out.append((spec.id, path))
    return out


def detect(path: str, roots_list: list[tuple[str, str]] | None = None) -> str:
    """某路径属于哪个生态 ✓（**最长匹配** ✓；都不匹配 ⇒ ``""`` ✓）。

    ⚠️ 用**最长匹配**而不是「先声明先赢」✗：落点会嵌套（``~/.cache/huggingface/hub`` 在
    ``~/.cache`` 之下 ✓），取最具体的那个才对 ✓。
    """
    if not path:
        return ""
    target = os.path.normcase(os.path.abspath(path))
    best = ""
    best_len = -1
    for ecosystem_id, root in (roots_list if roots_list is not None else all_roots()):
        base = os.path.normcase(os.path.abspath(root)).rstrip("\\/")
        if not base:
            continue
        if target == base or target.startswith(base + os.sep):
            if len(base) > best_len:
                best, best_len = ecosystem_id, len(base)
    return best


def skip_prefixes() -> tuple[str, ...]:
    """所有生态的「非模型目录」前缀合集 ✓（供扫描器跳过 HF 数据集等 ✓）。"""
    out: list[str] = []
    for spec in ECOSYSTEMS:
        for prefix in spec.skip_prefixes:
            if prefix not in out:
                out.append(prefix)
    return tuple(out)


# ============================================================
# 命中结果的生态修正
# ============================================================

def apply_ecosystem(hit: dict[str, Any], ecosystem_id: str) -> dict[str, Any]:
    """把**弱命中**改判为「所在生态」✓（``hit`` 是 ``local_model_scan.classify`` 的结果 ✓）。

    ⚠️ 强命中（``h3`` / ``local-sd`` / ``cosyvoice``…）**原样返回** ✗：规则说的是「这是什么」✓，
    生态只说「它在哪、能不能直接调」✓ ⇒ 不能拿「它在 HF 缓存里」把「它是 H3 的 DiT」抹掉 ✗✗。
    """
    spec = get(ecosystem_id)
    if spec is None or hit.get("runtime") not in OVERRIDE_RUNTIMES:
        return hit
    out = dict(hit)
    out["runtime"] = spec.runtime
    out["matchedBy"] = [*hit.get("matchedBy", []), f"ecosystem:{spec.id}"]
    return out


def build_suggestion(ecosystem_id: str, hit: dict[str, Any], name: str, ext: str) -> dict[str, Any] | None:
    """生态版的注册建议 ✓；**只有真被生态改判过**才给（否则 ``None`` ⇒ 用规则原建议 ✓）。

    与 :func:`apply_ecosystem` 严格互补 ✓：条件同为「runtime 已被改成该生态」✓。
    """
    spec = get(ecosystem_id)
    if spec is None or hit.get("runtime") != spec.runtime:
        return None
    role = hit.get("role") or "standalone"
    kind = hit.get("kind") or "unknown"
    note = spec.note
    runnable = spec.runnable
    if role == "component":
        # 组件权重（VAE / CLIP / 文本编码器…）：照样列出但**不单独注册** ✓（与原 TS 口径一致 ✓）
        note = f"{note}；⚠️ 这是**组件权重**，通常由主模型引用，不单独注册"
        runnable = False
    return {
        "serviceType": "image" if kind == "unknown" else kind,
        "provider": spec.provider,
        "baseUrl": spec.base_url,
        "model": re.sub(r"\.[^.]+$", "", name),
        "runtime": spec.runtime,
        "role": role,
        "callable": runnable,
        "note": note,
    }


def build_ollama_manifest_suggestion(name: str, size_bytes: int, missing_blobs: int) -> dict[str, Any]:
    """Ollama **清单**读出的模型的注册建议 ✓（它已经在 Ollama 里了 ⇒ 不用再导入 ✓）。

    ⚠️ blob 缺失要**说出来** ✓（清单在、权重没了 ≠ 能推理 ✓）—— 但不因此判它不可注册 ✓：
    到底能不能加载**只有 Ollama 服务端说了算** ✓（本仓铁律：没查 ≠ 通过 ✓）。
    """
    spec = _BY_ID["ollama"]
    note = spec.note
    if missing_blobs:
        note = (f"⚠️ 清单里有 {missing_blobs} 个 blob 不在盘上（可能已被清理），"
                f"未必加载得起来；{note}")
    return {
        "serviceType": "text",
        "provider": spec.provider,
        "baseUrl": spec.base_url,
        "model": name,
        "runtime": spec.runtime,
        "role": "standalone",
        "callable": True,
        "note": note,
    }


def by_ecosystem(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按生态计数 ✓（**声明顺序**输出 ✓；无生态归 ``""`` ⇒ 标签「其他位置」✓）。"""
    counts: dict[str, int] = {}
    for model in models:
        key = model.get("ecosystem") or ""
        counts[key] = counts.get(key, 0) + 1
    out: list[dict[str, Any]] = []
    for spec in ECOSYSTEMS:
        if counts.get(spec.id):
            out.append({"id": spec.id, "label": spec.label, "count": counts[spec.id]})
    if counts.get(""):
        out.append({"id": "", "label": "其他位置", "count": counts[""]})
    return out
