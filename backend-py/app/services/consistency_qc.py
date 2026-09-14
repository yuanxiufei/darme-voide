"""图像连续性 QC —— 移植 ``backend/src/services/consistency-qc.ts``（227 行，**整域关闭**）。

概念：内容审片。对同一集内**相邻分镜**的画面做**纯数值相似度**检测（dHash + 汉明距离，
零新增模型依赖 —— 注意：这**不是**视觉模型/LLM 任务，早先待办表里把它归成「视觉模型 + 多图」
是误记，实际全靠本地图像哈希）。

* 同场景相邻镜头相似度过低 → ``warning``（可能场景/光照/机位穿帮）
* 相似度介于 warn 与 ok 之间 → ``info``（建议人工复核）
* 跨场景相邻镜头 → **一律** ``info``（场景切换是正常行为；只有相似度过高会额外提示「场景图复用」）

策略：**report 模式**（对齐 Comic-drama 的 ``CONSISTENCY_POLICY_MODE=report``）。
结果并入各分镜**最新** ``video_quality_checks`` 的 ``continuity_vision`` 维度，只做展示与留痕，
**不参与总分/状态判定**（避免误伤），未来可升级 block 模式。

⚠️ 与 TS 版的两处**有意差异**：

1. **dHash 用 ffmpeg 而非 ``sharp``**：Node 侧走 libvips（``resize(9,8).greyscale().raw()``），
   Python 侧 venv **刻意不装 Pillow/numpy/cv2**（依赖保持最小），改用已在用的 ffmpeg：
   ``scale=9:8:flags=lanczos,format=gray -f rawvideo -`` ⇒ 正好 **9×8 = 72 字节**。
   两者都是「拉伸到 9×8 灰度 + lanczos 重采样」，同族算法；但**不同实现的最小差异可能让
   个别 hash 位翻转** ⇒ ``similarity`` 末位偶有 0.0x 级偏差。本域是 report（留痕）语义、
   阈值判定只影响提示文案，故接受。
2. **``dimensions`` 解析只接受对象**：TS 是 ``JSON.parse`` 的结果直接赋值（若存的是 ``5`` 这类
   非对象，后面 ``.continuity_vision = ...`` 静默失效、落库反把整列写成 ``"5"``）；
   Python 侧非 dict 一律当 ``{}``（**正常数据行为一致**，只在存量脏数据上「救回来」而不是抹掉）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..models import storyboards, video_quality_checks
from ..response import js_round, now
from .file_storage import get_absolute_path
from .task_logger import log_task_warn

__all__ = ["CONSISTENCY_QC_THRESHOLDS", "run_episode_consistency_qc"]

#: 一致性判定阈值（可调）
#: ⚠️ **必须与 ``consistency-qc.ts`` 的同名常量逐值一致**（守卫 ``route_parity_test`` 会比对）
CONSISTENCY_QC_THRESHOLDS: dict[str, float] = {
    # 同场景相邻镜头画面相似度下限：低于此值判 warning
    "sameSceneWarn": 0.55,
    # 相似度低于此值判 info（介于 warn 与 ok 之间）
    "sameSceneInfo": 0.65,
    # 同场景相邻镜头画面相似度上限：高于此值判 ok
    "sameSceneOk": 0.9,
    # 跨场景相邻镜头：相似度过高反而可疑（场景变了画面几乎一样）
    "crossSceneInfo": 0.9,
}

#: dHash 的目标尺寸：9 列（多一列用于相邻比较）× 8 行 ⇒ 64 bit
_HASH_W = 9
_HASH_H = 8


def _js_num_str(value: Any) -> str:
    """JS 模板字面量里的 ``${x}``：**整数不带 ``.0``**（相似度 1.0 在 JS 里渲染成 ``1``）。

    这条不是洁癖：两张完全相同的图（复用素材）相似度就是 1 ⇒ 文案必须与 Node 完全相同。
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


async def _compute_dhash(abs_path: str) -> bytes | None:
    """计算图片 dHash（9×8 灰度 → 8 字节），失败返回 ``None``（对齐 TS 的裸 catch）。"""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", abs_path,
            "-vf", f"scale={_HASH_W}:{_HASH_H}:flags=lanczos,format=gray",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await process.communicate()
    except Exception:  # noqa: BLE001 —— 找不到 ffmpeg / 路径异常
        return None
    if process.returncode != 0 or len(stdout) < _HASH_W * _HASH_H:
        return None

    data = stdout
    bits = bytearray(8)
    for y in range(_HASH_H):
        row = y * _HASH_W
        for x in range(_HASH_H):
            if data[row + x + 1] > data[row + x]:
                bits[y] |= 1 << (7 - x)
    return bytes(bits)


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _similarity(left: bytes, right: bytes) -> float:
    """汉明距离 → 相似度（0~1）。"""
    distance = sum(_popcount(left[i] ^ right[i]) for i in range(8))
    return 1 - distance / 64


def _pick_image(sb: Any) -> str | None:
    """从分镜行挑选用于比对的画面（composed 优先，其次首帧/关键帧）。"""
    for candidate in (sb.composed_image, sb.first_frame_image, sb.keyframe_image):
        # ⚠️ TS 是 `if (c && String(c).trim()) return String(c).trim()` ⇒ 返回**去空白后**的值
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    """``JSON.parse`` 后只认对象（见模块 docstring 第 2 条差异）。"""
    try:
        parsed = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001 —— 与 TS 的空 catch 等价
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(raw: Any) -> list[Any]:
    try:
        parsed = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001
        return []
    return parsed if isinstance(parsed, list) else []


async def run_episode_consistency_qc(conn: Connection, episode_id: Any,
                                    drama_id: Any = None) -> dict[str, Any]:
    """对一集所有分镜做相邻画面连续性检测。

    结果以 report 模式写回各分镜最新 ``video_quality_checks`` 的 ``continuity_vision`` 维度。
    """
    # ⚠️ 原 TS **不过滤软删**（`where(episodeId)` 而已），照抄
    rows = conn.execute(
        select(storyboards).where(storyboards.c.episode_id == episode_id)
    ).all()
    # `sort((a,b) => (a.storyboardNumber ?? 0) - (b.storyboardNumber ?? 0))`：稳定排序、空值当 0
    ordered = sorted(rows, key=lambda sb: sb.storyboard_number or 0)

    report: dict[str, Any] = {
        "episodeId": episode_id,
        "dramaId": drama_id if drama_id is not None else None,
        "checkedPairs": 0,
        "warningCount": 0,
        "pairs": [],
    }
    if len(ordered) < 2:
        return report

    # 每张图只算一次 hash（避免相邻对重复计算）；键是**相对路径**
    hash_cache: dict[str, bytes | None] = {}

    async def get_hash(rel_path: str) -> bytes | None:
        if rel_path in hash_cache:
            return hash_cache[rel_path]
        digest: bytes | None = None
        try:
            absolute = get_absolute_path(rel_path)
            digest = await _compute_dhash(absolute)
        except Exception:  # noqa: BLE001 —— 与 TS 的裸 catch 等价
            digest = None
        hash_cache[rel_path] = digest
        return digest

    # storyboardId → 最近 QC 记录（写回用）；**顺序敏感**（末尾按插入序落库，与 TS 的 Map 同序）
    latest_qc: dict[int, dict[str, Any]] = {}

    def load_qc(sb_id: int) -> dict[str, Any]:
        if sb_id in latest_qc:
            return latest_qc[sb_id]
        # ⚠️ 取的是 **id 最大**（不是 created_at 最新）的那条
        row = conn.execute(
            select(video_quality_checks)
            .where(video_quality_checks.c.storyboard_id == sb_id)
            .order_by(video_quality_checks.c.id.desc())
        ).first()
        state: dict[str, Any] = {"id": 0, "dimensions": {}, "issues": []}
        if row is not None:
            state = {
                "id": row.id,
                "dimensions": _parse_json_dict(row.dimensions),
                "issues": _parse_json_list(row.issues),
            }
        latest_qc[sb_id] = state
        return state

    thresholds = CONSISTENCY_QC_THRESHOLDS
    for index in range(1, len(ordered)):
        prev = ordered[index - 1]
        cur = ordered[index]
        prev_image = _pick_image(prev)
        cur_image = _pick_image(cur)
        if not prev_image or not cur_image:
            continue

        prev_hash, cur_hash = await asyncio.gather(get_hash(prev_image), get_hash(cur_image))
        if prev_hash is None or cur_hash is None:
            continue

        sim = js_round(_similarity(prev_hash, cur_hash) * 1000) / 1000
        same_scene = bool(prev.scene_id) and prev.scene_id == cur.scene_id

        severity = "ok"
        if same_scene:
            if sim < thresholds["sameSceneWarn"]:
                severity = "warning"
                message = (f"同场景（#{prev.scene_id}）相邻镜头画面相似度 {_js_num_str(sim)} 过低"
                           f"（标准 ≥{_js_num_str(thresholds['sameSceneWarn'])}），"
                           f"可能场景/光照/机位穿帮")
            elif sim < thresholds["sameSceneInfo"]:
                severity = "info"
                message = f"同场景相邻镜头画面相似度 {_js_num_str(sim)} 偏低，建议人工复核"
            else:
                message = f"同场景相邻镜头画面相似度 {_js_num_str(sim)}，正常"
        else:
            if sim > thresholds["crossSceneInfo"]:
                # ⚠️ 这一支 TS 也是 `info`（跨场景**不**升级为 warning）
                severity = "info"
                message = (f"跨场景切换但画面相似度 {_js_num_str(sim)} 偏高，"
                           f"可能场景图复用/未切换")
            else:
                severity = "info"
                message = f"跨场景相邻镜头，相似度 {_js_num_str(sim)}（仅记录）"

        report["pairs"].append({
            "prevStoryboardId": prev.id,
            "curStoryboardId": cur.id,
            "prevImage": prev_image,
            "curImage": cur_image,
            "sameScene": same_scene,
            "similarity": sim,
            "severity": severity,
            "message": message,
        })
        report["checkedPairs"] += 1
        if severity == "warning":
            report["warningCount"] += 1

        # 写回当前镜头最新 QC 记录（report 模式，不改总分/状态）
        state = load_qc(cur.id)
        if state["id"]:
            state["dimensions"]["continuity_vision"] = {
                "checked": True,
                "severity": severity,
                "similarity": sim,
                "sameScene": same_scene,
                "prevStoryboardId": prev.id,
                "notes": [message],
            }
            if not any(
                isinstance(item, dict) and item.get("dimension") == "continuity_vision"
                and item.get("message") == message
                for item in state["issues"]
            ):
                state["issues"].append(
                    {"dimension": "continuity_vision", "severity": severity, "message": message})

    # 统一落库（仅对有 QC 记录的分镜）；单条失败只 warn，不影响其它
    stamp = now()
    for state in latest_qc.values():
        if not state["id"]:
            continue
        try:
            conn.execute(
                video_quality_checks.update()
                .where(video_quality_checks.c.id == state["id"])
                .values(dimensions=json.dumps(state["dimensions"], ensure_ascii=False,
                                              separators=(",", ":")),
                        issues=json.dumps(state["issues"], ensure_ascii=False,
                                          separators=(",", ":")),
                        updated_at=stamp)
            )
        except Exception as exc:  # noqa: BLE001 —— 与 TS 的 per-record try/catch 等价
            log_task_warn("ConsistencyQc", "writeback-failed",
                          {"qcId": state["id"], "error": str(exc)})

    return report
