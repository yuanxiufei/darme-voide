"""FFmpeg 多镜头拼接（移植自 ``backend/src/services/ffmpeg-merge.ts``，277 行）。

把一集里**所有**已合成镜头（``composed_video_url``）按镜号串接为整集成片，写 ``episodes.video_url``。

流水线（``do_merge``）：

1. ``concat`` 列表文件（``<storage>/temp/<uuid>.txt``，每行 ``file '<绝对路径>'``）；
2. ffmpeg concat 拼接：``-f concat -safe 0 -fflags +genpts -c:v libx264 -preset medium
   -crf 23 -c:a aac -ar 48000 -b:a 192k -movflags +faststart``，随后删列表文件；
3. **统一配乐**（``mix_bgm``，一条 BGM 贯穿全片）：``-stream_loop -1`` 循环 + 音量/首尾淡入淡出
   + ``amix`` 叠加 + ``alimiter`` 限幅；失败**只告警**、回退无 BGM 版本；
4. **响度归一化**（``normalize_loudness``，I=-14 LUFS / TP=-1.5dB / LRA=11，对齐社媒验收）；
   无音轨直接原样返回；失败只告警；
5. 探测时长 → 更新 ``video_merges``（completed + merged_url + duration）与 ``episodes.video_url``。

⚠️ 三处保真点：

* **必须全部镜头都已合成**才能拼（否则抛
  ``Only composed storyboards can be merged (n/m ready)``）；
* 切片顺序按 ``storyboard_number`` 升序（不是 id）；
* 中间文件**用完即删**：concat 列表、无 BGM 的中间成片、响度归一化前的文件。

✅ ``run_consistency_qc_before_merge`` **已接线**（2026-09-15 校正）：Node 侧动态 import
``consistency-qc.js`` 做穿帮筛查，Python 侧直接复用 ``services.consistency_qc``
（此前是空实现 —— 即「合并前从不做穿帮筛查」，而 Node 会做）。
**独立事务 + 失败静默**：写入不受拼接主流程成败影响（对齐原 fire-and-forget 语义）。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update

from ..config import get_storage_root
from ..db import engine
from ..models import episodes, storyboards, video_merges
from ..response import now
from .file_storage import get_absolute_path
from .task_logger import log_task_error, log_task_start, log_task_success
from .video_probe import probe_video_duration

__all__ = ["merge_episode_videos"]


async def run_consistency_qc_before_merge(episode_id: int, drama_id: int) -> None:
    """合并前的图像连续性检测（穿帮筛查）—— 复用 ``services.consistency_qc``。

    与 Node 的动态 import + ``.catch(() => {})`` 等价：**用独立事务跑**（这样即使拼接主流程
    随后失败，筛查结果也已落库），**任何异常都吞掉**（不阻塞拼接）。
    """
    try:
        from .consistency_qc import run_episode_consistency_qc  # noqa: PLC0415 —— 惰性导入避免环

        with engine.begin() as conn:
            await run_episode_consistency_qc(conn, episode_id, drama_id)
    except Exception as exc:  # noqa: BLE001 —— 与原实现的静默失败一致
        log_task_error("MergeTask", "consistency-qc-failed",
                       {"episodeId": episode_id, "dramaId": drama_id, "error": str(exc)})


def _to_abs_path(relative_path: str) -> str:
    return get_absolute_path(relative_path)


async def _run_ffmpeg(args: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {tail[-2000:]}")


async def _video_has_audio(file_path: str) -> bool:
    """探测视频是否存在音轨（ffprobe 失败按「没有」处理）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "json", file_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return False
        streams = (json.loads(stdout.decode("utf-8", errors="replace") or "{}")
                   .get("streams") or [])
        return any(stream.get("codec_type") == "audio" for stream in streams)
    except Exception:  # noqa: BLE001 —— 没装 ffprobe / 解析失败
        return False


async def _mix_bgm(video_path: str, bgm: dict[str, Any]) -> str:
    """把统一配乐混入成片（BGM 循环贯穿 + 压低音量 + 首尾淡入淡出 + 限幅）。"""
    bgm_abs = _to_abs_path(bgm["bgmUrl"])
    if not Path(bgm_abs).exists():
        raise FileNotFoundError(f"BGM file not found: {bgm['bgmUrl']}")

    duration = await probe_video_duration(video_path)
    volume = max(0.0, min(1.0, bgm.get("bgmVolume") if bgm.get("bgmVolume") is not None else 0.3))
    # 淡入淡出时长不超过成片一半，避免短片段超界
    half = max(0.0, duration / 2)
    fade_in = min(max(0.0, bgm.get("bgmFadeIn") if bgm.get("bgmFadeIn") is not None else 1.5), half)
    fade_out = min(max(0.0, bgm.get("bgmFadeOut") if bgm.get("bgmFadeOut") is not None else 2.0), half)
    fade_out_start = max(0.0, duration - fade_out)

    out_path = str(Path(video_path).parent / f"{uuid4()}.mp4")
    complex_filter = (
        f"[1:a]volume={volume:.3f},afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}[bgm];"
        "[0:a][bgm]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]"
    )
    await _run_ffmpeg([
        "ffmpeg", "-i", video_path,
        # `-stream_loop -1` 属于**后加入的 BGM 输入**（fluent-ffmpeg 的 inputOptions 语义）
        "-stream_loop", "-1", "-i", bgm_abs,
        "-filter_complex", complex_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
        "-movflags", "+faststart", out_path,
    ])

    # 成功后删除无 BGM 的中间文件
    if out_path != video_path:
        try:
            Path(video_path).unlink()
        except OSError:
            pass
    return out_path


async def _normalize_loudness(video_path: str) -> str:
    """成片响度归一化（I=-14 LUFS / TP=-1.5dB / LRA=11）；无音轨直接原样返回。"""
    if not await _video_has_audio(video_path):
        return video_path

    out_path = str(Path(video_path).parent / f"{uuid4()}.mp4")
    await _run_ffmpeg([
        "ffmpeg", "-i", video_path,
        "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
        "-movflags", "+faststart", out_path,
    ])
    if out_path != video_path:
        try:
            Path(video_path).unlink()
        except OSError:
            pass
    return out_path


async def _do_merge(
    merge_id: int, episode_id: int, videos: list[str], bgm: dict[str, Any] | None
) -> None:
    # 生成 concat 列表文件
    list_dir = Path(get_storage_root()) / "temp"
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = str(list_dir / f"{uuid4()}.txt")
    Path(list_path).write_text(
        "\n".join(f"file '{_to_abs_path(v)}'" for v in videos), encoding="utf-8"
    )

    output_dir = Path(get_storage_root()) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{uuid4()}.mp4"
    output_path = str(output_dir / output_filename)

    try:
        await _run_ffmpeg([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
            "-fflags", "+genpts", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-movflags", "+faststart",
            output_path,
        ])
    finally:
        # 清理临时文件（**无论成败**：Node 只在成功后删，这里更保守一点，不留垃圾）
        try:
            Path(list_path).unlink()
        except OSError:
            pass

    # 统一配乐混音（失败回退无 BGM 版本）
    final_path = output_path
    if bgm:
        try:
            final_path = await _mix_bgm(output_path, bgm)
        except Exception as err:  # noqa: BLE001
            log_task_error("MergeTask", "bgm-mix", {
                "mergeId": merge_id, "episodeId": episode_id, "error": str(err),
            })

    # 响度归一化（失败保持原样）
    try:
        final_path = await _normalize_loudness(final_path)
    except Exception as err:  # noqa: BLE001
        log_task_error("MergeTask", "loudnorm", {
            "mergeId": merge_id, "episodeId": episode_id, "error": str(err),
        })

    duration = await probe_video_duration(final_path)
    merged_relative = f"static/merged/{Path(final_path).name}"

    with engine.begin() as conn:
        conn.execute(
            update(video_merges).where(video_merges.c.id == merge_id).values(
                status="completed", merged_url=merged_relative,
                duration=duration, completed_at=now(),
            )
        )
        conn.execute(
            update(episodes).where(episodes.c.id == episode_id).values(
                video_url=merged_relative, updated_at=now()
            )
        )

    log_task_success("MergeTask", "episode-merge", {
        "mergeId": merge_id, "episodeId": episode_id, "output": merged_relative,
        "duration": duration, "clips": len(videos),
    })


async def _do_merge_guarded(
    merge_id: int, episode_id: int, videos: list[str], bgm: dict[str, Any] | None
) -> None:
    """包一层错误处理：失败时把 merge 记录标成 failed + error_msg。"""
    try:
        await _do_merge(merge_id, episode_id, videos, bgm)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 .catch 等价
        log_task_error("MergeTask", "episode-merge", {
            "mergeId": merge_id, "episodeId": episode_id, "error": str(err),
        })
        with engine.begin() as conn:
            conn.execute(
                update(video_merges).where(video_merges.c.id == merge_id).values(
                    status="failed", error_msg=str(err)
                )
            )


def merge_episode_videos(episode_id: int, drama_id: int) -> int:
    """串接一集所有合成镜头；返回 ``video_merges.id``（**立即返回，拼接在后台跑**）。

    ⚠️ 要求**全部**分镜都有 ``composed_video_url`` —— 少一个就拒绝，避免拼出残缺成片。
    """
    with engine.begin() as conn:
        rows = conn.execute(
            select(storyboards.c.id, storyboards.c.composed_video_url)
            .where(storyboards.c.episode_id == episode_id)
            .order_by(storyboards.c.storyboard_number)
        ).all()
        episode = conn.execute(
            select(
                episodes.c.bgm_url, episodes.c.bgm_volume,
                episodes.c.bgm_fade_in, episodes.c.bgm_fade_out,
            ).where(episodes.c.id == episode_id)
        ).first()

        composed = [row for row in rows if row[1]]
        if len(composed) != len(rows):
            raise ValueError(
                "Only composed storyboards can be merged "
                f"({len(composed)}/{len(rows)} ready)"
            )
        videos = [row[1] for row in composed]
        if not videos:
            raise ValueError("No videos to merge")

        # 本集统一配乐方案：一条 BGM 贯穿全片（而非逐镜头切换），跨镜头自然过渡
        bgm = None
        if episode is not None and episode[0]:
            bgm = {
                "bgmUrl": episode[0],
                "bgmVolume": episode[1] if episode[1] is not None else 0.3,
                "bgmFadeIn": episode[2] if episode[2] is not None else 1.5,
                "bgmFadeOut": episode[3] if episode[3] is not None else 2.0,
            }

        log_task_start("MergeTask", "episode-merge", {
            "episodeId": episode_id, "dramaId": drama_id,
            "clips": len(videos), "bgm": bool(bgm),
        })

        # 合并前自动做图像连续性检测（穿帮筛查，不影响拼接主流程）
        # ⚠️ **fire-and-forget**：与下面起后台拼接同一手法（本函数虽同步，但由异步端点调用，
        #    事件循环在跑）。任务自带独立事务 ⇒ 可以安全 detach；且它内部吞掉所有异常，
        #    不会产生「task exception was never retrieved」告警。
        asyncio.create_task(run_consistency_qc_before_merge(episode_id, drama_id))

        merge_id = int(conn.execute(
            video_merges.insert().values(
                episode_id=episode_id,
                drama_id=drama_id,
                title=f"Episode {episode_id} Merge",
                provider="ffmpeg",
                model="ffmpeg-concat-h264-aac",
                status="processing",
                # ⚠️ 紧凑分隔符：这一列与 Node 共用（JSON.stringify 无空格）
                scenes=json.dumps(videos, ensure_ascii=False, separators=(",", ":")),
                created_at=now(),
            )
        ).lastrowid)

    asyncio.create_task(_do_merge_guarded(merge_id, episode_id, videos, bgm))
    return merge_id
