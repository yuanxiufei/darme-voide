"""技术维度 QC —— 移植 ``backend/src/services/technical-qc.ts``（281 行，**整域关闭**）。

对本地成品视频做**硬性数值检测**（对齐 MiniMax-H3-Codex-Drama 的 QC 标准）：

* 黑场间隔 > 0.25s ⇒ 缺陷（ffmpeg ``blackdetect``）
* 冻帧 > 1.0s ⇒ 缺陷（ffmpeg ``freezedetect``）
* 音频真峰 > -1dBTP ⇒ 缺陷（ffmpeg ``ebur128``）
* 集成响度偏离 -14 LUFS ±1 ⇒ 缺陷（ffmpeg ``ebur128``）
* 帧率 < 15 ⇒ 缺陷；时长偏差 > 1s（与分镜声明比）⇒ 缺陷

结果写回该分镜**最新** ``video_quality_checks``：合并 ``video_spec`` 维度 + 追加 ``issues``，
并**重算总分与状态**（``round(规则三维均值 * 0.7 + 技术分 * 0.3)``；**任一 error 级 issue 即 failed**，
注意这里读的是合并后的**全部** issues，规则维度的 error 也会让它变 failed —— 与原 TS 一致）。

⚠️ 与原 TS 的一处**有意差异（阻塞 vs fire-and-forget）**：Node 是 ``runTechnicalQc(...).catch()``
**不等待**；Python 侧选择**在同一事务内同步跑完**。理由：写回必须用调用方的连接，而 detached task
极可能在调用方提交/关闭连接之后才执行（SQLAlchemy 连接不是为跨任务共享设计的）⇒ 用「等待」换正确性。
代价是这一步会占用调用方的执行时间（ffmpeg 4~5 趟，短视频通常 1~3s）；失败仍只 warn，不向上抛。

⚠️ ``execFileP`` 的判失败条件很特别：**只有「非零退出且 stderr 为空」才算失败** ——
ffmpeg 的滤镜输出全在 stderr，正常结束也可能非零退出。这里逐字镜像（``_run``）✓，**不要改成非零即失败** ✗。

⚠️⚠️ 但这条语义有个**静默绿灯**的连带后果 ✗：坏文件/空文件走到这里 ⇒ ffprobe 非零退出但打了 stderr
⇒ **不报错** ✓、各滤镜读不出任何特征 ⇒ 段数全空 ⇒ **最终 100 分、零 issue** ✗✗
（比报错更坏：它看起来**干净** ✓）。2026-09-20 收口：**规格侧**补一条判据 ——
``duration`` 与 ``fps`` **同时**读不出来 ⇒ 报 warning 并扣 15 分（"后面那些检测都是空转" ✓）。
``_run`` 的语义**一个字没动** ✓（改的是"读不出规格"这件事的**报出** ✓，不是退出码判定 ✓）。

## ✅ 三项检测曾**两边都死**，现已修（2026-09-18，Python 侧单边修）

历史：2026-09-15 实测发现 **冻帧 / 音频真峰 / 集成响度**三项在 Node 与 Python **都永远判不出来**
（正则与 ffmpeg 实际输出格式不符；黑场、帧率、时长三项正常）。当时约定「要修必须两侧一起改」
⇒ 于是**没修** ✗，只用「真实格式 ⇒ 解析为空」的用例把缺陷钉住当信号 ✓。

现状：Node 侧**已删**（2026-09-15 ✓）⇒「两边一起改」这个约束随之消失 ✓；本机装了
**ffmpeg 9.0.1** ✓ ⇒ 按**实测输出**修掉 ✓（不靠文档猜 ✗ —— 当初正是因为猜不准才没修 ✓）：

* ⚠️ ``freezedetect`` 打的是**带前缀的三行**（``lavfi.freezedetect.freeze_start`` /
  ``freeze_duration`` / ``freeze_end``，**各占一行、顺序为 start→duration→end**）✗
  ⇒ 改成**事件流配对** ✓（不再要求相邻 ✓）。**且**：视频一直冻到片尾时**只打 ``freeze_start``**
  ✗✗ ⇒ 用 ffprobe 的总时长收尾并标 ``open``（初版对这种片**整段漏报** ✗）；
* ⚠️ ``ebur128`` **是有 summary 的** ✓，但值是**标题的下一行**（``Integrated loudness:`` →
  ``I: … LUFS``；``True peak:`` → ``Peak: … dBFS``）✗ ⇒ 改成跨行取 ✓；
  无 summary 的构建退化为**取最后一条**逐帧进度行 ✓ —— ``I:`` 是**滚动累计值**，
  **首行恒为 ``-70.0 LUFS``**（起手静音）✗✗ ⇒ 取首行会造出「永远 -70」的假读数 ✓✗（比不解析更坏 ✓）；
* ``Peak: … dBFS`` 是 ebur128 的**真峰**（与 ``dBTP`` 同一标度 ✓）⇒ ``TECH_QC_THRESHOLDS``
  的语义**不变** ✓（值也与 ``technical-qc.ts`` 逐字一致 ✓，不触发镜像常量守卫 ✓）。
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection

from ..core.models import storyboards, video_generations, video_quality_checks
from ..core.response import now
from .file_storage import get_absolute_path
from .task_logger import log_task_warn

__all__ = ["TECH_QC_THRESHOLDS", "probe_video_technical_spec", "run_technical_qc"]

#: 硬性数值标准（可调）
#: ⚠️ **必须与 ``technical-qc.ts`` 的同名常量逐值一致**（守卫 ``route_parity_test`` 会比对）
TECH_QC_THRESHOLDS: dict[str, float] = {
    "blackMinDuration": 0.25,   # 单段黑场最小判定时长（s）
    "blackPixThreshold": 0.1,   # 黑场像素阈值
    "freezeMinDuration": 1.0,   # 单段冻帧最小判定时长（s）
    "freezeNoiseThreshold": -60,  # 冻帧噪声阈值（dB）
    "truePeakDb": -1,           # 音频真峰上限（dBTP）
    "loudnessTarget": -14,      # 集成响度目标（LUFS）
    "loudnessTolerance": 1,     # 集成响度容差（LUFS）
    "fpsMin": 15,               # 合理帧率下限
    "durationToleranceSec": 1,  # 时长偏差容差（s）
}

_BLACK_RE = re.compile(
    r"black_start:\s*([0-9.]+)\s*black_end:\s*([0-9.]+)\s*black_duration:\s*([0-9.]+)")
#: freezedetect：实测 ffmpeg 9 是**三行、各带前缀** ⇒ 按**事件**取（顺序 start→duration→end）
_FREEZE_EVENT_RE = re.compile(
    r"freezedetect\.(freeze_start|freeze_duration|freeze_end):\s*(-?[0-9.]+)")
#: 兼容**同一行**的老形态（``freeze_start: 0.8 freeze_duration: 1.7``）
_FREEZE_INLINE_RE = re.compile(r"freeze_start:\s*([0-9.]+)\s*freeze_duration:\s*([0-9.]+)")
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[0-9.]+) dB")
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?[0-9.]+) dB")
#: ebur128 summary：实测 ffmpeg 9 的**值是标题的下一行**
_INTEGRATED_SUMMARY_RE = re.compile(
    r"Integrated loudness:\s*\r?\n\s*I:\s*(-?[0-9.]+)\s*LUFS")
_TRUE_PEAK_SUMMARY_RE = re.compile(
    r"True peak:\s*\r?\n\s*Peak:\s*(-?[0-9.]+)\s*dBFS")
#: 老形态：值与标题**同一行**
_INTEGRATED_INLINE_RE = re.compile(r"Integrated loudness:\s*(-?[0-9.]+)\s*LUFS")
_TRUE_PEAK_INLINE_RE = re.compile(r"True peak:\s*(-?[0-9.]+)\s*dBTP")
#: 退化路径：逐帧进度行 ⇒ 必须取**最后**一条（``I:`` 是滚动累计值，首行恒 -70 LUFS）
_INTEGRATED_PROGRESS_RE = re.compile(r"\bI:\s*(-?[0-9.]+)\s*LUFS")
_TRUE_PEAK_PROGRESS_RE = re.compile(r"\bTPK:\s*(-?[0-9.]+)\s*dBFS")


def _js_to_fixed(value: float, digits: int) -> str:
    """JS ``Number.toFixed(n)``：**half-away-from-zero**（Python 的 ``f"{x:.2f}"`` 是 banker's）。

    ⚠️ 与 ``qc_report._js_to_fixed`` 同源（都映射 JS 的 ``toFixed``）——按本仓「每个 service 自带
    兜底」的结构各留一份，别合并成公共件后忘了 ``round`` 的差异。
    """
    from decimal import ROUND_HALF_UP, Decimal  # noqa: PLC0415

    quant = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(value))).quantize(quant, rounding=ROUND_HALF_UP))


def _js_num_str(value: Any) -> str:
    """JS 模板字面量里的 ``${x}``：整数**不带** ``.0``（阈值 1.0 渲染成 ``1``）。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _run(binary: str, args: list[str]) -> tuple[str, str]:
    """``execFileP``：返回 ``(stdout, stderr)``；**只有「非零退出且 stderr 为空」才抛**。

    ffmpeg 的滤镜统计全写在 stderr，且正常结束也可能非零退出 ⇒ 照抄 TS 的判定。
    """
    completed = subprocess.run([binary, *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=180)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0 and not stderr:
        raise RuntimeError(f"{binary} 退出码 {completed.returncode}（无 stderr 输出）")
    return stdout, stderr


def parse_black_segments(stderr: str) -> list[dict[str, float]]:
    """解析 ``black_start/black_end/black_duration`` 行。"""
    return [{"start": float(m[0]), "duration": float(m[2])} for m in _BLACK_RE.findall(stderr)]


def _freeze_segment(start: float, duration: float | None, end: float | None,
                    total_duration: float | None) -> dict[str, Any]:
    """补齐一段冻帧的已知量。

    ``open`` 的含义**只有一个**：该段**只能靠总时长收尾** ⇒ 也就是「**一直冻到片尾**」✓
    （初版把「ffmpeg 没打 ``freeze_end``」也归成 ``open`` ✗ ⇒ 文案会对一段"其实结束了"的
    冻帧说「持续到片尾」✓✗ —— 自检当场红 ✓）。三项来源的优先级：
    ``freeze_end`` ✓ → ``freeze_duration`` 推 ✓ → 总时长收尾 ✓；
    **都没有就如实留 ``None``**（不猜 ✓）。
    """
    if end is None and duration is not None:
        end = start + duration
    if duration is None and end is not None:
        duration = end - start
    open_ended = False
    if end is None and total_duration is not None and total_duration > start:
        end = total_duration
        duration = total_duration - start
        open_ended = True
    return {"start": start, "duration": duration, "end": end, "open": open_ended}


def parse_freeze_segments(stderr: str,
                          total_duration: float | None = None) -> list[dict[str, Any]]:
    """解析 ``freezedetect`` 段 ⇒ ``[{"start", "duration", "end", "open"}]``。

    ⚠️ 实测 ffmpeg 9 打的是**三行**（带 ``[Parsed_freezedetect_0 @ …]`` 前缀）::

        lavfi.freezedetect.freeze_start: 0
        lavfi.freezedetect.freeze_duration: 3
        lavfi.freezedetect.freeze_end: 3

    ⇒ 必须按**事件流配对**（初版要求 ``start``/``duration`` 相邻 ⇒ **永不匹配** ✗）。
    ⚠️ 视频**一直冻到片尾**时只打 ``freeze_start`` ✗ ⇒ 用 ``total_duration`` 收尾并标 ``open``
    （初版对这种片**整段漏报** ✗✗）。
    """
    events = [(kind, float(value)) for kind, value in _FREEZE_EVENT_RE.findall(stderr)]
    if not events:
        return [_freeze_segment(float(start), float(duration), None, total_duration)
                for start, duration in _FREEZE_INLINE_RE.findall(stderr)]

    segments: list[dict[str, Any]] = []
    start: float | None = None
    duration: float | None = None
    for kind, value in events:
        if kind == "freeze_start":
            if start is not None:  # 上一段还没闭合就又开始了 ⇒ 先收掉 ✓
                segments.append(_freeze_segment(start, duration, None, total_duration))
            start, duration = value, None
        elif kind == "freeze_duration" and start is not None:
            duration = value
        elif kind == "freeze_end" and start is not None:
            segments.append(_freeze_segment(start, duration, value, total_duration))
            start, duration = None, None
    if start is not None:
        segments.append(_freeze_segment(start, duration, None, total_duration))
    return segments


def parse_volume(stderr: str) -> dict[str, float | None]:
    """解析 ``volumedetect`` 的 ``mean_volume/max_volume``。"""
    mean = _MEAN_VOLUME_RE.search(stderr)
    max_volume = _MAX_VOLUME_RE.search(stderr)
    return {"mean": float(mean.group(1)) if mean else None,
            "max": float(max_volume.group(1)) if max_volume else None}


def _first_group(pattern: re.Pattern[str], text: str) -> float | None:
    """取**第一条**匹配（summary 只会出现一次）。"""
    found = pattern.search(text)
    return float(found.group(1)) if found else None


def _last_group(pattern: re.Pattern[str], text: str) -> float | None:
    """取**最后一条**匹配（逐帧进度行必须取末条 ⇒ 见 ``parse_ebur128`` 的警告）。"""
    found = None
    for found in pattern.finditer(text):  # noqa: B007 —— 故意留最后一个
        pass
    return float(found.group(1)) if found else None


def parse_ebur128(stderr: str) -> dict[str, float | None]:
    """解析 ``ebur128`` 的**集成响度**与**真峰**。

    ⚠️ 实测 ffmpeg 9 的 summary 是**跨两行**的（值在标题下一行）::

          Integrated loudness:
            I:         -21.8 LUFS
          True peak:
            Peak:      -16.1 dBFS

    ⇒ 先取跨行 summary ✓、再取同行老形态 ✓、最后退化为**最后一条**逐帧进度行 ✓。
    ⚠️ **绝不能取首条进度行** ✗：``I:`` 是**滚动累计值**，首行恒为 ``-70.0 LUFS``（起手静音）
    ⇒ 取首行会得到一个「永远 -70」的假读数 ✗✗（比解析不出更坏）。
    """
    integrated = _first_group(_INTEGRATED_SUMMARY_RE, stderr)
    if integrated is None:
        integrated = _first_group(_INTEGRATED_INLINE_RE, stderr)
    if integrated is None:
        integrated = _last_group(_INTEGRATED_PROGRESS_RE, stderr)

    peak = _first_group(_TRUE_PEAK_SUMMARY_RE, stderr)
    if peak is None:
        peak = _first_group(_TRUE_PEAK_INLINE_RE, stderr)
    if peak is None:
        peak = _last_group(_TRUE_PEAK_PROGRESS_RE, stderr)
    return {"integrated": integrated, "truePeak": peak}


def _empty_result(error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "duration": None, "fps": None, "hasAudio": False, "maxVolumeDb": None,
        "integratedLoudness": None, "blackSegments": [], "freezeSegments": [],
        "issues": [], "score": 100,
    }
    if error is not None:
        result["error"] = error
    return result


def probe_video_technical_spec(local_path: str,
                              expected_duration: float | None = None) -> dict[str, Any]:
    """对本地视频文件执行技术规格检测（同步；失败按原 TS 语义降级或早退）。"""
    try:
        absolute = get_absolute_path(local_path)
    except Exception as exc:  # noqa: BLE001 —— 原 TS 在 getAbsolutePath 抛错时**带上 error 早退**
        return _empty_result(str(exc))

    result = _empty_result()
    thresholds = TECH_QC_THRESHOLDS

    # ===== 1. ffprobe：时长 / 帧率 =====
    try:
        stdout, _stderr = _run("ffprobe", [
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate:format=duration",
            "-of", "json", absolute,
        ])
        meta = json.loads(stdout or "{}")
        streams = meta.get("streams") if isinstance(meta.get("streams"), list) else []
        stream = streams[0] if streams else None
        fmt = meta.get("format") if isinstance(meta.get("format"), dict) else {}
        # `Number(x) || null`：0/NaN/空串都要落到 null
        duration = _js_number_or_none(fmt.get("duration"))
        result["duration"] = duration
        rate = (stream or {}).get("avg_frame_rate")
        if rate and rate != "0/0":
            parts = str(rate).split("/")
            num = _js_number_or_none(parts[0]) if parts else None
            den = _js_number_or_none(parts[1]) if len(parts) > 1 else None
            if num and den:
                result["fps"] = float(_js_to_fixed(num / den, 3))

        # ⚠️ **两条都读不出来** ⇒ 必须报出来 ✗（初版一声不吭地给 100 分 ✗✗ —— 坏文件看起来"干净" ✓）。
        #    ``_run`` 的判定**不动** ✓：ffmpeg 滤镜正常结束也会非零退出 ✓，
        #    所以这里判的是"**规格读不出来**"这件事本身 ✓，不是退出码 ✓。
        #    注意不能只看 stdout 是否为空 ✗（有些容器/裸流本来就没有那些字段 ✓）；
        #    也不能只缺一个就报警 ✗（例如无音轨、或某些容器没有 avg_frame_rate ✓）。
        if result["duration"] is None and result["fps"] is None:
            result["score"] -= 15
            result["issues"].append({
                "severity": "warning",
                "message": "无法读取视频规格（ffprobe 未给出时长与帧率）"
                           "⇒ 黑场/冻帧/响度这些检测会全部空转 ✗",
            })
    except Exception:  # noqa: BLE001
        result["score"] -= 15
        result["issues"].append({"severity": "warning",
                                 "message": "无法读取视频规格（ffprobe 失败），跳过技术检测"})
        return result

    # ===== 2. 黑场检测 =====
    try:
        _stdout, stderr = _run("ffmpeg", [
            "-i", absolute,
            "-vf", f"blackdetect=d={_js_num_str(thresholds['blackMinDuration'])}"
                   f":pix_th={_js_num_str(thresholds['blackPixThreshold'])}",
            "-an", "-f", "null", "-",
        ])
        result["blackSegments"] = parse_black_segments(stderr)
        for segment in result["blackSegments"]:
            result["score"] -= 20
            result["issues"].append({
                "severity": "error",
                "message": f"画面黑场 {_js_to_fixed(segment['duration'], 2)}s"
                           f"（第 {_js_to_fixed(segment['start'], 1)}s 起，"
                           f"标准 >{_js_num_str(thresholds['blackMinDuration'])}s 即判缺陷）",
            })
    except Exception:  # noqa: BLE001 —— 忽略检测失败，不重复扣分
        pass

    # ===== 3. 冻帧检测 =====
    # ⚠️ 传 total_duration：视频**一直冻到片尾**时 ffmpeg 只打 `freeze_start` ✗
    #    ⇒ 用 ffprobe 的总时长收尾；拿不到就不猜 ✓（文案如实说「时长未知」✓）。
    try:
        _stdout, stderr = _run("ffmpeg", [
            "-i", absolute,
            "-vf", f"freezedetect=n={_js_num_str(thresholds['freezeNoiseThreshold'])}dB"
                   f":d={_js_num_str(thresholds['freezeMinDuration'])}",
            "-map", "0:v", "-f", "null", "-",
        ])
        result["freezeSegments"] = parse_freeze_segments(stderr, result["duration"])
        for segment in result["freezeSegments"]:
            result["score"] -= 20
            duration = segment["duration"]
            head = (f"画面冻帧 {_js_to_fixed(duration, 2)}s" if duration is not None
                    else "画面冻帧（时长未知 ⇒ ffmpeg 未给结束时刻）")
            tail = "，该段一直持续到片尾" if segment["open"] else ""
            result["issues"].append({
                "severity": "error",
                "message": f"{head}（第 {_js_to_fixed(segment['start'], 1)}s 起{tail}，"
                           f"标准 >{_js_num_str(thresholds['freezeMinDuration'])}s 即判缺陷）",
            })
    except Exception:  # noqa: BLE001
        pass

    # ===== 4. 音频响度（volumedetect 近似真峰 + ebur128 集成响度） =====
    try:
        _stdout, stderr = _run("ffmpeg", ["-i", absolute, "-af", "volumedetect",
                                          "-f", "null", "-"])
        max_volume = parse_volume(stderr)["max"]
        result["maxVolumeDb"] = max_volume
        if max_volume is not None:
            result["hasAudio"] = True
    except Exception:  # noqa: BLE001 —— 无音轨或检测失败
        pass

    if result["hasAudio"]:
        try:
            _stdout, stderr = _run("ffmpeg", ["-i", absolute, "-af", "ebur128=peak=true",
                                              "-f", "null", "-"])
            loudness = parse_ebur128(stderr)
            result["integratedLoudness"] = loudness["integrated"]
            true_peak = loudness["truePeak"]
            if true_peak is not None and true_peak > thresholds["truePeakDb"]:
                result["score"] -= 15
                result["issues"].append({
                    "severity": "warning",
                    "message": f"音频真峰 {_js_to_fixed(true_peak, 1)}dBTP 超过 "
                               f"{_js_num_str(thresholds['truePeakDb'])}dBTP 标准（防削波）",
                })
            integrated = loudness["integrated"]
            if integrated is not None and abs(integrated - thresholds["loudnessTarget"]) \
                    > thresholds["loudnessTolerance"]:
                result["score"] -= 10
                result["issues"].append({
                    "severity": "info",
                    "message": f"集成响度 {_js_to_fixed(integrated, 1)}LUFS 偏离标准 "
                               f"{_js_num_str(thresholds['loudnessTarget'])}"
                               f"±{_js_num_str(thresholds['loudnessTolerance'])}LUFS",
                })
        except Exception:  # noqa: BLE001 —— ebur128 可能因多声道失败
            pass

    # ===== 5. 帧率 / 时长合理性 =====
    if result["fps"] is not None and result["fps"] < thresholds["fpsMin"]:
        result["score"] -= 10
        result["issues"].append({
            "severity": "warning",
            "message": f"帧率 {_js_num_str(result['fps'])}fps 过低"
                       f"（标准 ≥{_js_num_str(thresholds['fpsMin'])}fps）",
        })
    # ⚠️ `expectedDuration &&`：0 / None 都不做时长判定
    if expected_duration and result["duration"] is not None:
        difference = abs(result["duration"] - expected_duration)
        if difference > thresholds["durationToleranceSec"]:
            result["score"] -= 10
            result["issues"].append({
                "severity": "warning",
                "message": f"视频实际时长 {_js_to_fixed(result['duration'], 1)}s 与期望 "
                           f"{_js_num_str(expected_duration)}s 偏差 "
                           f"{_js_to_fixed(difference, 1)}s"
                           f"（标准 ≤{_js_num_str(thresholds['durationToleranceSec'])}s）",
            })

    result["score"] = max(0, min(100, round(result["score"])))
    return result


def _js_number_or_none(value: Any) -> float | None:
    """``Number(x) || null`` 的语义（``None``/``''``/非数字 ⇒ ``None``，``0`` 也落 ``None``）。"""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number == 0:  # NaN / 0
        return None
    return number


def _json_dict(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(raw: Any) -> list[Any]:
    try:
        parsed = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001
        return []
    return parsed if isinstance(parsed, list) else []


def run_technical_qc(conn: Connection, storyboard_id: Any,
                     video_generation_id: Any) -> None:
    """技术 QC：对本地成品做硬性规格检测，结果并入最近一条 QC 记录（**失败只 warn**）。"""
    try:
        gen = conn.execute(
            select(video_generations).where(and_(
                video_generations.c.id == video_generation_id,
                video_generations.c.status == "completed"))
        ).first()
        if gen is None or not gen.local_path:
            log_task_warn("TechQc", "no-local-file",
                          {"storyboardId": storyboard_id,
                           "videoGenerationId": video_generation_id})
            return

        sb = conn.execute(
            select(storyboards).where(storyboards.c.id == storyboard_id)).first()
        spec = probe_video_technical_spec(
            gen.local_path, (sb.duration if sb is not None else None))

        # 找该分镜**最新**（id 最大）的 QC 记录，读-改-写 video_spec 维度
        qc = conn.execute(
            select(video_quality_checks)
            .where(video_quality_checks.c.storyboard_id == storyboard_id)
            .order_by(video_quality_checks.c.id.desc())
        ).first()
        if qc is None:
            return

        dims = _json_dict(qc.dimensions)
        dims["video_spec"] = {
            "score": spec["score"],
            "checked": True,
            "notes": ([issue["message"] for issue in spec["issues"]]
                      if spec["issues"] else ["技术规格检测通过"]),
            "duration": spec["duration"],
            "fps": spec["fps"],
            "hasAudio": spec["hasAudio"],
            "maxVolumeDb": spec["maxVolumeDb"],
            "integratedLoudness": spec["integratedLoudness"],
            "blackSegments": len(spec["blackSegments"]),
            "freezeSegments": len(spec["freezeSegments"]),
        }

        issues = _json_list(qc.issues)
        for issue in spec["issues"]:
            if not any(isinstance(item, dict) and item.get("dimension") == "video_spec"
                       and item.get("message") == issue["message"] for item in issues):
                issues.append({"dimension": "video_spec", "severity": issue["severity"],
                               "message": issue["message"]})

        # 重算总分：规则三维平均 70% + 技术维度 30%
        rule_scores = [
            (dims.get("lip_sync") or {}).get("score") or 0,
            (dims.get("character_consistency") or {}).get("score") or 0,
            (dims.get("continuity") or {}).get("score") or 0,
        ]
        rule_avg = sum(rule_scores) / 3
        overall = round(rule_avg * 0.7 + spec["score"] * 0.3)
        # ⚠️ 读的是**合并后全部** issues：规则维度的 error 也会把状态压成 failed（原 TS 如此）
        has_error = any(isinstance(item, dict) and item.get("severity") == "error"
                        for item in issues)
        status = "failed" if has_error else "passed"

        conn.execute(
            video_quality_checks.update()
            .where(video_quality_checks.c.id == qc.id)
            .values(dimensions=json.dumps(dims, ensure_ascii=False, separators=(",", ":")),
                    issues=json.dumps(issues, ensure_ascii=False, separators=(",", ":")),
                    overall_score=overall, status=status, updated_at=now())
        )
    except Exception as exc:  # noqa: BLE001 —— 与原 TS 的 catch 等价
        log_task_warn("TechQc", "run-failed",
                      {"storyboardId": storyboard_id,
                       "videoGenerationId": video_generation_id, "error": str(exc)})
