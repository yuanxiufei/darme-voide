"""ComfyUI 渲染目标目录 + 「图片记录 → target」路由（**单一真源**）。

为什么需要这一层
----------------
盘上 ComfyUI 的 ``GET /comfy-studio/renders`` 自述 19 个目标，短剧线的三类图
（角色定妆板 / 场景卡 / 分镜帧）**各有自己的工作流**，参数口径也不同：

==============  ====================  ==============  ==================
板型             target 后缀          负向口          备注
==============  ====================  ==============  ==================
角色定妆板      ``-krea2`` / 无       krea2 有         Qwen 档的 #6 是
                                      Qwen 档**没有**  「三视图布局约束」
场景卡          ``-krea2`` / ``-zimage`` / 无  有        —
分镜帧          ``-krea2`` / 无       有              ``-ref``/``-dualref`` 是尾帧
==============  ====================  ==============  ==================

两条**必须由本模块守住的坑**（都由实测或驱动明文确认）：

1. **负向不能盲送**：``01_角色定妆板_Qwen2512.json`` 的 ``#6`` 是布局约束口，
   往它送负向会整段冲掉工作流自带的 6 条负向 ⇒ 只有本表声明
   ``negative=True`` 的 target 才允许带负向。
2. **档位属于工作流**：steps / cfg / seed / width / height 都是「不传就用图上原值」。
   Qwen 档是 30 步 / cfg 5.5，Krea2 档是 8 步 / cfg 2.0，分镜帧是 720×1280 ——
   适配器若自己拍一个默认值送上去，就会把设计师调好的档位压坏。所以这些参数
   **只在调用方显式给出时才送**（``None`` 就是不送）。
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "CHARACTER",
    "DEFAULT_LINE",
    "LINES",
    "SCENE",
    "STORYBOARD",
    "TARGETS",
    "line_of",
    "plate_of_record",
    "resolve_target",
    "supports_negative",
]

#: 板型（短剧线的三类图）
CHARACTER = "character"
SCENE = "scene"
STORYBOARD = "storyboard"

#: 可选底模线（同一个板型在不同底模上各有一份工作流）
LINES = ("krea2", "qwen", "zimage")

#: 未显式指定时的默认线（Krea2 在本机 24G 卡上 8 步可跑、量化全通，实测首选）
DEFAULT_LINE = "krea2"

#: target 目录：id → {板型, 线, 是否吃负向}
#: ⚠️ 只登记**图片类**目标；video-* / music / video-interpolate 等不属于本层。
TARGETS: dict[str, dict[str, Any]] = {
    # ── 角色定妆板 ────────────────────────────────────────────────────────
    "character-sheet-krea2": {"plate": CHARACTER, "line": "krea2", "negative": True},
    "character-sheet": {"plate": CHARACTER, "line": "qwen", "negative": False},
    "character-sheet-lightning": {"plate": CHARACTER, "line": "qwen", "negative": False, "fast": True},
    "character-sheet-zimage": {"plate": CHARACTER, "line": "zimage", "negative": False},
    # ── 场景卡 ────────────────────────────────────────────────────────────
    "scene-card-krea2": {"plate": SCENE, "line": "krea2", "negative": True},
    "scene-card": {"plate": SCENE, "line": "qwen", "negative": True},
    "scene-card-zimage": {"plate": SCENE, "line": "zimage", "negative": True},
    # ── 分镜帧 ────────────────────────────────────────────────────────────
    "storyboard-frame-krea2": {"plate": STORYBOARD, "line": "krea2", "negative": True},
    "storyboard-frame": {"plate": STORYBOARD, "line": "qwen", "negative": True},
    "storyboard-frame-krea2-ref": {"plate": STORYBOARD, "line": "krea2", "negative": True, "ref": True},
    "storyboard-frame-krea2-dualref": {
        "plate": STORYBOARD, "line": "krea2", "negative": True, "ref": True,
    },
    "storyboard-frame-ref": {"plate": STORYBOARD, "line": "qwen", "negative": True, "ref": True},
}


def _targets_of(plate: str, line: str) -> str | None:
    """在某板型 / 某线里找**不带 ref 的**主目标。"""
    for target, meta in TARGETS.items():
        if meta["plate"] == plate and meta["line"] == line and not meta.get("ref"):
            return target
    return None


def line_of(target: str) -> str | None:
    meta = TARGETS.get(target)
    return meta["line"] if meta else None


def supports_negative(target: str) -> bool:
    """该 target 的工作流是否真有负向口（Qwen 档定妆板的 #6 是布局约束，不是负向）。"""
    meta = TARGETS.get(target)
    return bool(meta and meta.get("negative"))


def _setting(settings: Any, key: str) -> Any:
    """从 ``config["settings"]`` 取值；settings 可能是 dict，也可能是 JSON 字符串。"""
    if isinstance(settings, dict):
        return settings.get(key)
    if isinstance(settings, str) and settings.strip():
        import json

        try:
            parsed = json.loads(settings)
        except ValueError:
            return None
        return parsed.get(key) if isinstance(parsed, dict) else None
    return None


def plate_of_record(record: dict[str, Any]) -> str | None:
    """按记录语义判板型（**不猜**：判不出来就返回 None，由调用方要求显式指定）。"""
    if record.get("storyboardId") or record.get("frameType"):
        return STORYBOARD
    if record.get("characterId"):
        return CHARACTER
    if record.get("sceneId"):
        return SCENE
    return None


def resolve_target(config: dict[str, Any], record: dict[str, Any]) -> str:
    """选渲染目标。优先级从高到低：

    1. ``record["target"]`` —— 单条记录显式点名（作业单/测试用）
    2. ``config["settings"]["renderTarget"]`` —— 服务配置里钉死（整条线只出一种图时用）
    3. ``config["model"]`` 本身就是目录里的 target id —— 等价于钉死
    4. 按板型 + 线推断：``settings["line"]``（默认 ``krea2``）

    判不出板型时**抛错**，绝不静默落到某个默认工作流上（那会拿定妆板工作流去出分镜帧）。
    """
    explicit = str(record.get("target") or "").strip()
    if explicit:
        if explicit not in TARGETS:
            raise ValueError(
                f"comfyui：未知渲染目标 {explicit!r}；本层支持：{', '.join(sorted(TARGETS))}"
            )
        return explicit

    pinned = str(_setting(config.get("settings"), "renderTarget") or "").strip()
    if pinned:
        if pinned not in TARGETS:
            raise ValueError(f"comfyui：settings.renderTarget={pinned!r} 不在本层目录里")
        return pinned

    model = str(config.get("model") or "").strip()
    if model in TARGETS:
        return model

    plate = plate_of_record(record)
    if not plate:
        if record.get("propId"):
            raise ValueError(
                "comfyui：盘上 19 个渲染目标里**没有道具板工作流**（只有角色定妆板 / 场景卡 / 分镜帧）"
                "—— 道具图请改用其他 provider，或先给 ComfyUI 补一件道具板工作流再登记进本表"
            )
        raise ValueError(
            "comfyui：判不出这张图属于哪一类（角色定妆板 / 场景卡 / 分镜帧）——"
            "请在记录里带上 characterId / sceneId / storyboardId，"
            "或用 settings.renderTarget 钉死一个渲染目标"
        )

    line = str(_setting(config.get("settings"), "line") or DEFAULT_LINE).strip()
    if line not in LINES:
        raise ValueError(f"comfyui：settings.line={line!r} 不在 {LINES} 里")
    target = _targets_of(plate, line)
    if not target:
        raise ValueError(f"comfyui：没有「{plate} × {line}」这一档工作流（见 comfyui_targets.TARGETS）")
    return target
