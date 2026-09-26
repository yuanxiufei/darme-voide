"""**训练工具脚本的定位** —— 训练真正执行的那几个 ``.py`` 在哪 ✓、用什么解释器跑 ✓、环境变量怎么拼 ✓。

## 事实来源（别凭记忆改 ✗）

训练本体不由本仓实现 ✓，而由两个**上游开源工具**提供 ✓：

* **musubi-tuner**（kohya-ss ✓）—— Wan / HunyuanVideo / Qwen-Image / FLUX-Kontext 走它 ✓；
* **sd-scripts**（kohya-ss ✓）—— **FLUX（非 Kontext）走它** ✓。

⚠️ 「FLUX 走 sd-scripts」这条是**实测事实**、不是笔误 ✓：参考实现 ``flux_lora_train.py`` 里
拼接的是 ``os.path.join(base_dir, 'sd-scripts')`` + ``flux_train_network.py`` ✓，
而其余四个脚本拼接的是 ``musubi-tuner/src/musubi_tuner`` ✓。⇒ 本仓按模型分别指路 ✓。

## 两种目录布局都真实存在 ✓（所以是候选表，不是单一路径 ✗）

* **官方仓库布局**（本仓 ``reference/lora/musubi-tuner`` 实测 = musubi-tuner **0.3.5** ✓）：
  根目录是**薄壳** —— ``wan_train_network.py`` 全文只有一行
  ``from musubi_tuner.wan_train_network import main`` ✓，真实现在 ``src/musubi_tuner/`` ✓
  ⇒ 跑那层壳需要 ``PYTHONPATH`` 含 ``<root>/src`` ✓。
* **参考实现的内置布局**：脚本平铺在 ``<root>/src/musubi_tuner/`` ✓，``PYTHONPATH`` 同样是 ``<root>/src`` ✓
  （脚本 import 的兄弟模块就在这个包里 ✓）。

``sd-scripts`` 侧无需猜测 ✓：``flux_train_network.py`` 用的是**同目录**导入
（``from library import ...`` ✓ ``import train_network`` ✓，第 10~29 行实测 ✓）⇒
``PYTHONPATH`` 要给 **sd-scripts 根** ✓（Python 也会把脚本自身目录放进 ``sys.path[0]`` ✓，
两条都留着更稳 ✓；⚠️ 参考实现在这一处给的是**上一层目录** ✗，靠的正是 ``sys.path[0]`` 兜着 ✓）。

## 找不到就报错 ✗

⚠️ 本模块**不做「找不到就换一个」的兜底** ✗✗：候选全部不存在 ⇒
:class:`~app.services.lora_train.errors.LoraTrainEnvironmentError`，
报文里**列出试过的每一个绝对路径** ✓（这类错最常见的死法是「报了个看不懂的
``No such file``，而真正的信息量是『你以为装在这儿、其实不在』」✓）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core import config as core_config

from .errors import LoraTrainEnvironmentError

#: 仓库根 ✓（唯一权威是 ``app/core/config.py`` ✗ —— 本包**不自己数** ``parents[N]`` ✓）
REPO_ROOT = core_config.PROJECT_ROOT

#: 两套工具 ✓ —— 见模块头「事实来源」✓
TOOLKIT_MUSUBI = "musubi-tuner"
TOOLKIT_SD_SCRIPTS = "sd-scripts"

#: ``configs/model-paths.json`` ✓（与 ``models_dir`` 等键并列 ✓）
PATHS_JSON = REPO_ROOT / "configs" / "model-paths.json"

#: 工具根：JSON 键名 / 环境变量名 / 本仓默认落点 ✓ —— 三个对齐着一处一处写清 ✓。
_TOOL_ROOTS: dict[str, tuple[str, str, Path]] = {
    TOOLKIT_MUSUBI: ("musubi_tuner_root", "MUSUBI_TUNER_ROOT",
                     REPO_ROOT / "reference" / "lora" / "musubi-tuner"),
    TOOLKIT_SD_SCRIPTS: ("sd_scripts_root", "SD_SCRIPTS_ROOT",
                         REPO_ROOT / "reference" / "lora" / "sd-scripts"),
}

#: 环境变量：显式指定跑训练用的 python ✓（默认 ``sys.executable`` ✓ = 当前后端那个 venv ✓，
#: 与参考实现的 ``python_executable = sys.executable`` 同一口径 ✓）。
ENV_PYTHON = "LORA_TRAIN_PYTHON"


def _read_paths_json() -> dict[str, object]:
    """读 ``configs/model-paths.json`` ✓；缺失/坏文件 ⇒ 空字典 ✓（**不阻断** ✓，后面还有默认落点 ✓）。"""
    try:
        raw = PATHS_JSON.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def tool_root(toolkit: str) -> Path:
    """工具根目录 ✓。优先级：环境变量 > ``model-paths.json`` > 本仓 ``reference/lora/`` ✓。

    ⚠️ 返回的是**解析结果**，不保证存在 ✗ —— 存在性由 :func:`resolve_tool` 报错 ✓
    （分开是为了让「路径算得出来、但没装」这种状态能被单独说清楚 ✓）。
    """
    if toolkit not in _TOOL_ROOTS:
        raise LoraTrainEnvironmentError(f"不认识的工具 {toolkit!r} ✗；合法值：{sorted(_TOOL_ROOTS)}")
    json_key, env_key, fallback = _TOOL_ROOTS[toolkit]

    override = (os.environ.get(env_key) or "").strip()
    if override:
        return Path(override)
    configured = _read_paths_json().get(json_key)
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip())
    return fallback


def python_executable() -> str:
    """跑训练/预缓存用的解释器 ✓。

    ⚠️ 默认 ``sys.executable`` **就是这个后端进程的解释器** ✓（本机后端 venv 已装 torch ✓
    —— 训练依赖（``torch``/``accelerate``/``transformers`` 等）应当与它同一套 ✓）。
    显式给 ``LORA_TRAIN_PYTHON`` 时以它为准 ✓（训练环境与后端环境分开装的机器 ✓）。
    """
    override = (os.environ.get(ENV_PYTHON) or "").strip()
    return override or sys.executable


@dataclass(frozen=True)
class ResolvedTool:
    """一个**已确认存在**的训练工具脚本 ✓。"""

    #: 脚本绝对路径 ✓
    path: Path
    #: 跑它时要**前导**进 ``PYTHONPATH`` 的目录 ✓（可多个 ✓）
    pythonpath: tuple[Path, ...]
    #: 命中的候选相对路径 ✓（便于日志里说清"用的哪一层壳"✓）
    matched: str
    #: 属于哪套工具 ✓
    toolkit: str
    #: 工具根 ✓
    root: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "pythonpath": [str(p) for p in self.pythonpath],
            "matched": self.matched,
            "toolkit": self.toolkit,
        }


def _candidates(toolkit: str, script: str) -> tuple[tuple[str, str], ...]:
    """``(脚本相对路径, PYTHONPATH 相对路径)`` 候选 ✓ —— 见模块头「两种目录布局」✓。"""
    name = script if script.endswith(".py") else f"{script}.py"
    if toolkit == TOOLKIT_MUSUBI:
        return (
            (name, "src"),
            (f"src/musubi_tuner/{name}", "src"),
        )
    if toolkit == TOOLKIT_SD_SCRIPTS:
        return ((name, "."),)
    raise LoraTrainEnvironmentError(f"不认识的工具 {toolkit!r} ✗；合法值：{sorted(_TOOL_ROOTS)}")


def resolve_tool(script: str, toolkit: str = TOOLKIT_MUSUBI) -> ResolvedTool:
    """定位训练脚本 ✓。全部候选都不存在 ⇒ :class:`LoraTrainEnvironmentError` ✓。

    ``script`` 传**不带 ``.py``** 的名字也行 ✓。
    """
    root = tool_root(toolkit)
    tried: list[str] = []
    for relative, py_relative in _candidates(toolkit, script):
        candidate = root / relative
        tried.append(str(candidate))
        if candidate.is_file():
            return ResolvedTool(
                path=candidate,
                pythonpath=(root / py_relative,),
                matched=relative,
                toolkit=toolkit,
                root=root,
            )
    json_key, env_key, fallback = _TOOL_ROOTS[toolkit]
    raise LoraTrainEnvironmentError(
        f"找不到训练脚本 {script}（工具：{toolkit} ✗）\n"
        f"工具根：{root}（存在={root.is_dir()}）\n"
        "试过的路径：\n  - " + "\n  - ".join(tried) + "\n"
        f"⇒ 装法：把 {toolkit} 放到 {fallback} ✓，"
        f"或设环境变量 {env_key} / 在 configs/model-paths.json 里写 \"{json_key}\" ✓。"
    )


def tool_available(script: str, toolkit: str = TOOLKIT_MUSUBI) -> bool:
    """只问「在不在」✓，**不抛异常** ✗ —— 给体检接口用 ✓（启动路径请直接用 :func:`resolve_tool` ✓）。"""
    root = tool_root(toolkit)
    return any((root / relative).is_file() for relative, _ in _candidates(toolkit, script))


def build_env(tool: ResolvedTool, extra: dict[str, str] | None = None) -> dict[str, str]:
    """子进程环境 ✓ —— ``PYTHONPATH`` 前导 + 编码 + 日志级别 ✓。

    * ``PYTHONPATH``：把工具的导入根放**最前** ✓，再拼上调用方原本的 ✓（参考实现同样把这两段拼起来 ✓）；
    * ``PYTHONIOENCODING=utf-8`` ✓：Windows 控制台默认 GBK ✓，训练进程里打中文/表情符号会
      ``UnicodeEncodeError`` ✗（参考实现同样显式设了 ✓）；
    * ``LOG_LEVEL=DEBUG`` ✓：musubi-tuner 读这个变量决定日志级别 ✓ ——
      不给的话关键阶段（缓存命不命中、显存分配 ✓）会被压掉，训练日志就没法排查了 ✓。
    """
    env = os.environ.copy()
    parts = [str(p) for p in tool.pythonpath]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("LOG_LEVEL", "DEBUG")
    if extra:
        env.update(extra)
    return env


def accelerate_available(timeout: int = 60) -> bool:
    """``accelerate`` 在**目标解释器**里是否可用 ✓（用 ``-c "import accelerate"`` 实测 ✓，不猜 ✓）。

    ⚠️ 这里**不用** ``shutil.which("accelerate")`` ✗：那查的是 PATH ✓，而启动用的是
    ``-m accelerate.commands.launch`` ✓ —— 两件事，实测差距就是「拿 A 环境启动、脚本在 B 环境里跑」✓✗。
    ⚠️ 子进程有开销 ✓ ⇒ 只给体检接口用 ✓，训练启动路径**不调**它 ✗
    （启动失败会由子进程自己的 stderr 说清楚 ✓，不必提前拦 ✓）。
    """
    try:
        proc = subprocess.run(
            [python_executable(), "-c", "import accelerate"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


_ACCELERATE_CACHE: list[bool] = []


def accelerate_available_cached() -> bool:
    """:func:`accelerate_available` 的**进程内缓存** ✓（体检接口会被前端反复调 ✓）。"""
    if not _ACCELERATE_CACHE:
        _ACCELERATE_CACHE.append(accelerate_available())
    return _ACCELERATE_CACHE[0]


def reset_caches() -> None:
    """清缓存 ✓ —— 给测试用 ✓（换了环境变量后要重新实测 ✓）。"""
    _ACCELERATE_CACHE.clear()


def environment_report() -> dict[str, object]:
    """**环境体检** ✓ —— 「这台机器现在能不能跑训练」 ✓。

    ⚠️ ``checked`` 三态语义（与本仓 ``services/engine_readiness`` 同一口径 ✓）：
    ``False`` = **没查**✗，不是「不行」✓；要让前端能区分「没证据」与「证据是坏」✓。
    """
    exe = python_executable()
    return {
        "python": exe,
        "pythonExists": Path(exe).is_file() if os.path.sep in exe else bool(shutil.which(exe)),
        "accelerate": accelerate_available_cached(),
        "pythonPathEntries": [str(p) for p in _all_pythonpath_entries()],
        "tools": {
            toolkit: {
                "root": str(tool_root(toolkit)),
                "rootExists": tool_root(toolkit).is_dir(),
            }
            for toolkit in _TOOL_ROOTS
        },
    }


def _all_pythonpath_entries() -> tuple[Path, ...]:
    """两套工具各自的导入根 ✓（去重 ✓）—— 体检里列出来，方便对「装是装了、导入路径不对」✓。"""
    found: list[Path] = []
    for toolkit in _TOOL_ROOTS:
        for _script, py_relative in _candidates(toolkit, "x"):
            entry = tool_root(toolkit) / py_relative
            if entry not in found:
                found.append(entry)
    return tuple(found)
