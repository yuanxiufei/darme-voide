"""视频**识别** —— 抽帧 + 视觉模型逐帧读，串成一份可用的描述。

⭐⭐ 先把一件事说清楚（这是本模块全部设计的前提）：**Ollama 没有视频模型**。
它没有视频解码层、没有视频输入类型、也没有任何 ``video`` 能力名。所谓「用 Ollama 读视频」
在工程上只有一条真实路径：**ffmpeg 抽 N 帧 → 每帧转 base64 → 塞进一条带多图的
``/api/chat`` 消息 → 视觉模型逐帧描述**。任何号称「Ollama 原生视频理解」的实现，
底下都是这件事（有些只是把它藏起来了）。

⚠️ 由此推出三条硬约束，缺一条结果就会**看起来成功其实没用**：
1. **必须真带图**：图片只能挂 ``messages[i].images``（裸 base64，无 ``data:`` 前缀）。
   放请求顶层会被服务端静默忽略 ⇒ 模型凭空编描述（最坏的一种「成功」✗）。
2. **必须挑到 ``vision`` 模型**：拿纯文本模型硬塞图，服务端多数版本会直接报错，少数
   版本会把图**丢掉**继续答 —— 后者更坏。所以这里先用能力面筛，筛不到就明说 ✗。
3. **必须限帧数/限宽**：一帧 1080p 的 JPEG ≈ 300KB，6 帧进上下文轻松吃掉几千 token；
   超过模型上下文窗口时服务端会截断（截断的是**图**，不是文字，同样静默 ✗）。

⚠️ 抽帧是 ``ffmpeg`` 子进程（本仓第二个子进程依赖，第一个在 ``ollama.py``）。这里沿用
``frame_extractor`` 的既有习惯：相对路径走 ``to_abs_media_path`` ✓、``ffmpeg`` 从 PATH 取 ✓。
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from typing import Any, Mapping, Sequence

from .ollama import resolve_ollama_url
from .ollama_client import OllamaClient, OllamaError, user_message
from .ollama_native import capabilities as capability_rules

__all__ = [
    "DEFAULT_FRAMES",
    "DEFAULT_PROMPT",
    "MAX_FRAME_WIDTH",
    "analyze_video",
    "decode_frames",
    "pick_reader",
    "probe_duration",
    "sample_frames",
]

#: 默认抽 6 帧：覆盖「开场/中段/收尾」三类信息量，又不至于把上下文吃穿（经验值 ✓）
DEFAULT_FRAMES = 6
#: 帧宽上限 768：视觉塔输入普遍 336~1024，再宽只是白烧 token（且多数模型内部会缩 ✗）
MAX_FRAME_WIDTH = 768

DEFAULT_PROMPT = """这是一段视频按时间顺序抽取的 {count} 帧（第 1 帧最早）。
请只依据这些画面回答：
1. 场景与主体：在哪里、有谁/有什么；
2. 动作与变化：主体在做什么，帧与帧之间发生了什么变化；
3. 镜头：景别（远景/中景/近景）与运动（推拉摇移/固定）；
4. 光线与色彩：基调、光源方向、冷暖；
5. 画面文字：有则逐字抄出，没有就写「无」。
不要推测剧情与台词，不要评价画质。"""

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _abs_path(video_url: str) -> str:
    """相对路径按媒体库根解析（与 ``frame_extractor`` 同一套，别自己拼盘符 ✗）。"""
    text = str(video_url or "").strip()
    if not text:
        raise ValueError("视频路径为空")
    if os.path.isabs(text):
        return text
    try:
        from .frame_extractor import to_abs_media_path

        return to_abs_media_path(text)
    except Exception:                     # noqa: BLE001 - 拿不到根就按进程 cwd 兜底 ✓
        return os.path.abspath(text)


async def _run_ffmpeg(args: list[str], *, capture: bool = False) -> tuple[int, bytes, bytes]:
    """跑一次 ffmpeg。**独立成函数是刻意留的测试缝**（自检换掉它就能在没装 ffmpeg 的机器上跑 ✓）。"""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout if capture else b"", stderr or b""


async def ffmpeg_available() -> bool:
    """``ffmpeg -version`` 能不能跑（找不到可执行文件 ⇒ ``False``，不抛 ✓）。"""
    try:
        code, _, _ = await _run_ffmpeg(["-version"])
    except (FileNotFoundError, OSError):
        return False
    return code == 0


async def probe_duration(video_url: str) -> float:
    """时长（秒）。⚠️ 用 ``ffmpeg -i`` 的 stderr 里那行 ``Duration:`` 解析 —— 本仓不假设装了
    ``ffprobe``（两者常分开打包）；解析不到返回 ``0.0``（调用方据此退化成「只抽首帧」✓）。"""
    path = _abs_path(video_url)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频不存在：{path}")
    try:
        _, _, stderr = await _run_ffmpeg(["-hide_banner", "-i", path])
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError("找不到 ffmpeg（请先安装并加入 PATH）") from exc
    match = _DURATION_RE.search(stderr.decode("utf-8", "replace"))
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


async def sample_frames(
    video_url: str,
    *,
    count: int = DEFAULT_FRAMES,
    max_width: int = MAX_FRAME_WIDTH,
) -> list[dict[str, Any]]:
    """等间隔抽 ``count`` 帧（JPEG，宽度 ≤ ``max_width``），返回 ``[{index, time, bytes, base64}]``。

    ⚠️ 逐帧**单次 seek**（``-ss`` 在 ``-i`` 之前 ⇒ 关键帧级快速定位 ✓），不是「一口气解码
    全片再挑帧」：后者对 10 分钟视频要解码几分钟，前者 6 次 seek 通常 < 1 秒 ✓。
    ``-ss`` 放在 ``-i`` 前会落在**关键帧**上（时间有 ±1~2 秒误差），对「描述画面」没有影响；
    要逐帧精确对齐就别用这条路径（那是剪辑线的事，不是识别线的事 ✓）。
    """
    path = _abs_path(video_url)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频不存在：{path}")
    wanted = max(1, int(count))

    duration = await probe_duration(path)
    if duration <= 0:
        stamps = [0.0] * wanted        # 时长未知 ⇒ 只能取首帧，重复占位（调用方按 index 去重 ✓）
    else:
        stamps = [duration * (i + 0.5) / wanted for i in range(wanted)]

    frames: list[dict[str, Any]] = []
    scale = f"scale='min({int(max_width)},iw)':-2"
    for index, stamp in enumerate(stamps):
        try:
            code, stdout, _ = await _run_ffmpeg([
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{max(0.0, stamp):.3f}", "-i", path,
                "-frames:v", "1", "-vf", scale,
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3",
                "pipe:1",
            ], capture=True)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError("找不到 ffmpeg（请先安装并加入 PATH）") from exc
        if code != 0 or not stdout:
            # 单帧抽不出来（seek 越过结尾/坏帧）⇒ 跳过这一帧，不让整条链断 ✓
            continue
        frames.append({
            "index": len(frames),
            "time": round(max(0.0, stamp), 3),
            "bytes": len(stdout),
            "base64": base64.b64encode(stdout).decode("ascii"),
        })
    return frames


def decode_frames(payloads: Sequence[str]) -> list[dict[str, Any]]:
    """把「已经是 base64 的帧」规整成同形状（供外部已经抽好帧的调用方复用 ✓）。

    ⚠️ 会**剥掉** ``data:image/jpeg;base64,`` 前缀：上游只认裸 base64，带前缀解出来是坏图，
    而坏图不会报错、只会让模型描述成「模糊的画面」—— 静默降级，最难查 ✗。
    """
    frames: list[dict[str, Any]] = []
    for index, raw in enumerate(payloads):
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("data:") and "," in text:
            text = text.split(",", 1)[1]
        frames.append({"index": len(frames), "time": 0.0, "base64": text})
    return frames


def pick_reader(
    models: Sequence[Mapping[str, Any]],
    *,
    prefer: str | None = None,
) -> dict[str, Any] | None:
    """从目录里挑一个能读视频的模型：显式 ``prefer`` 优先（得真具备 vision ✓），否则最大的 ✓。"""
    readers = [m for m in models if m.get("videoReader")]
    if not readers:
        return None
    if prefer:
        wanted = str(prefer)
        for model in readers:
            if str(model.get("name") or "") == wanted:
                return dict(model)
        # 给了不存在的名字 ⇒ 回落默认而不是报错（前端传的还是旧名字是常见情况 ✓）
    readers.sort(key=lambda m: (-int(m.get("size") or 0), str(m.get("name") or "")))
    return dict(readers[0])


async def analyze_video(
    video_url: str,
    *,
    model: str | None = None,
    frames: int = DEFAULT_FRAMES,
    prompt: str | None = None,
    max_width: int = MAX_FRAME_WIDTH,
    base_url: str | None = None,
    precomputed_images: Sequence[str] | None = None,
) -> dict[str, Any]:
    """读一段视频：抽帧 → 视觉模型 → 结构化描述（**全流程只有这一条路** ✓）。

    返回 ``{ok, model, frames, prompt, summary, capabilities, baseUrl}``；失败时
    ``ok=False`` 且 ``reason`` 是人话（不是异常栈 ✓）—— 这类操作是「用户点一下等 10 秒」，
    抛异常会让前端只能显示 500。
    """
    from . import ollama_catalog

    client = OllamaClient(base_url)
    try:
        report = await ollama_catalog.inventory(client.base_url, include_details=True)
        picked = pick_reader(report.get("models") or [], prefer=model)
        if picked is None:
            return {
                "ok": False,
                "model": model or "",
                "baseUrl": client.base_url,
                "reason": (
                    "没有可用的视觉模型（Ollama 里没有具备 vision 能力的模型）"
                    "——请先 `ollama pull` 一个视觉模型（如 qwen2.5vl / llava / gemma3）"
                ),
                "frames": 0,
            }
        name = str(picked.get("name") or "")
        caps = list(picked.get("capabilities") or [])
        missing = capability_rules.check_capabilities(caps, [capability_rules.CAPABILITY_VISION])
        if missing:
            return {
                "ok": False,
                "model": name,
                "baseUrl": client.base_url,
                "reason": f"模型 {name} 不具备读图能力（缺 {', '.join(missing)}）",
                "frames": 0,
            }

        if precomputed_images:
            sampled = decode_frames(precomputed_images)
        else:
            sampled = await sample_frames(video_url, count=frames, max_width=max_width)
        if not sampled:
            return {
                "ok": False,
                "model": name,
                "baseUrl": client.base_url,
                "reason": "没抽到任何帧（视频为空、编码不支持，或 ffmpeg 不可用）",
                "frames": 0,
            }

        question = (prompt or DEFAULT_PROMPT).format(count=len(sampled))
        message = user_message(question, [f["base64"] for f in sampled])
        try:
            payload = await client.chat(name, [message], stream=False)
        except OllamaError as exc:
            return {
                "ok": False,
                "model": name,
                "baseUrl": client.base_url,
                "reason": str(exc),
                "frames": len(sampled),
            }

        text = ""
        reply = payload.get("message") if isinstance(payload, Mapping) else None
        if isinstance(reply, Mapping):
            text = str(reply.get("content") or "")
        return {
            "ok": bool(text),
            "model": name,
            "baseUrl": client.base_url,
            "capabilities": caps,
            "frames": len(sampled),
            "frameTimes": [f["time"] for f in sampled],
            "prompt": question,
            "summary": text,
            "reason": "" if text else "模型没有返回内容",
            "metrics": {
                key: payload.get(key)
                for key in ("total_duration", "eval_count", "prompt_eval_count")
                if isinstance(payload, Mapping) and payload.get(key) is not None
            },
        }
    finally:
        await client.close()


def resolve_url(base_url: str | None = None) -> str:
    """当前要连的地址（唯一真源在 ``ollama.resolve_ollama_url`` ✓，这里只是转发）。"""
    return resolve_ollama_url(base_url)
