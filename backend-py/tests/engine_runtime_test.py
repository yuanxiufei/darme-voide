"""S? 自检：**引擎运行时** ✓ —— 「真跑一次」那条路终于有入口了（2026-09-26 ✓）。

这一套钉的是「**运行时真的把引擎跑起来了**」✓，不是"接口返回 200"✗：

1. **装载** 用 `load_weights` 的**真路径** ✓（造缩小版 H3 形态权重 ✓ ⇒ 判形态那条路照走 ✓）；
2. **排队** 真串行 ✓（两个任务：后一个的 `startedAt` **不早于**前一个的 `finishedAt` ✓）；
3. **产物** 真 mp4 + 真 wav ✓（`result.outputs.videoPath` 落盘 ✓、文件真在 ✓）；
4. **诚实** `synthetic` **必须**是 `True` ✗（参考 TE / VAE 未训练 ⇒ 是噪声画面 ✓；
   若哪天它变成 `False` 而实现没换 ⇒ **这条自检就该红** ✓✗）；
5. **卸载** 真丢张量 ✓（之后 `loaded.dit` 必须回到 `False` ✓）且**正忙时拒卸** ✓；
6. **回收缓存** 还的只是"空着的块" ✓ —— `memory_allocated` 前后必须**一致** ✓、装好的东西必须**还在** ✓
   （⚠️ 回收 ≠ 卸载 ✗；不还的话缓存会"看着占满整卡"✗ —— 实测见 `case_reclaim` 的注释 ✓）；
7. **收尾顺序** 读到「终结」的那一刻，"这一张还了多少"必须**已经在事件里** ✓
   （⚠️ 读者不止"被唤醒"那种 ✗：`bridge.run_job` 是**带超时轮询**的 ✓ ⇒ 一到终结就收工走人 ✗
   ⇒ 顺序错了那条事实就永远漏掉 ✓✗ —— 2026-09-26 被实测证伪过一次 ✓，见 `case_terminal_order` ✓）。

⚠️ 「不依赖外部」的**反向证明**也在这里 ✓：`status()["externalDependencies"]` 必须是空列表 ✓
（它一旦不是空的，就说明这条路上又接回了 ComfyUI / SD WebUI / Ollama 之类外部服务 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_runtime_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import h3_form  # noqa: E402
from app.services.engine import media as media_mod  # noqa: E402
from app.services.engine import pipeline as pipe  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402
from app.services.engine.runtime import EngineBusy, engine_runtime  # noqa: E402
from app.services.engine.torch_backend import TorchBackendUnavailable  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


#: **缩小版** H3 主干 ✓（只改尺寸 ✓ —— 模块名 / 键名与真权重一致 ✓ ⇒ 形态判别照样认它 ✓）
TRUNK: dict[str, Any] = {
    "hidden": 32, "layers": 2, "heads": 4, "head_dim": 16, "ffn": 64,
    "text_dim": 16, "latents_dim": 4, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
    "time_input_dim": 16, "time_hidden": 32, "time_dim": 32, "inv_freq_len": 2,
    "refiner_layers": 1,
}
#: 请求：1:1 ⇒ 224×224 ✓（/16 = 14×14 ✓ CPU 秒级 ✓）；3 步 ✓；双流取整口径显式给 ✓
REQUEST: dict[str, Any] = {"prompt": "雨夜霓虹街头", "seed": 7, "steps": 3, "seconds": 0.2,
                           "ratio": "1:1", "megapixels": 0.05, "temporal_compression": 1}


def _request(out_dir: Path) -> pipe.GenerationRequest:
    return pipe.GenerationRequest(**{**REQUEST, "outputs_dir": str(out_dir)})


# ══════════════════════════════════════════════════════════════════════════
# ① 自述：缺什么报什么 + **不许依赖外部** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_status() -> None:
    status = engine_runtime.status()
    check("自述: 五个必需键都在（available/reason/loaded/busy/tasks）",
          all(key in status for key in ("available", "reason", "loaded", "busy", "queued", "tasks")),
          sorted(status))
    check("自述: 逐组件的 loaded 是**四个布尔**（dit / textEncoder / vae / audioVae）",
          set(status["loaded"]) == {"dit", "textEncoder", "vae", "audioVae"}, status["loaded"])
    check("⭐ 不依赖外部: externalDependencies 是**空列表**（接回任何外部服务这条就该红）",
          status["externalDependencies"] == [], status["externalDependencies"])
    check("自述: synthetic 恒 True（参考 TE / VAE 未训练 ⇒ 噪声画面）",
          status["synthetic"] is True, status["synthetic"])
    check("自述: 没装也能答「将会在哪跑」（deviceSource=probe ⇒ cpu 还是 cuda 要一眼看见）",
          (status["device"] in ("cpu", "cuda") and status["deviceSource"] in ("probe", "loaded", None)),
          (status["device"], status["deviceSource"]))


# ══════════════════════════════════════════════════════════════════════════
# ② 队列：真串行 + 真取消 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_queue(out_dir: Path) -> None:
    first = engine_runtime.submit(_request(out_dir), assembly={"dryRun": True})
    second = engine_runtime.submit(_request(out_dir), assembly={"dryRun": True})
    one = engine_runtime.wait(first, timeout=120)
    two = engine_runtime.wait(second, timeout=120)
    check("队列: 两个干跑任务都跑完（done）", one["status"] == "done" and two["status"] == "done",
          (one["status"], two["status"]))
    check("⭐ 队列: **真串行** —— 后一个的 startedAt 不早于前一个的 finishedAt",
          two["startedAt"] >= one["finishedAt"], (one["finishedAt"], two["startedAt"]))
    check("队列: 事件是逐条留存的（干跑 > 0 条，前端按它画进度）",
          one["eventCount"] > 0 and isinstance(one["events"], list), one["eventCount"])

    cancelled = engine_runtime.submit(_request(out_dir), assembly={"dryRun": True})
    blocked = engine_runtime.submit(_request(out_dir), assembly={"dryRun": True})
    engine_runtime.cancel(blocked)
    check("取消: 排队中的**立刻**终结（cancelled）",
          engine_runtime.get_task(blocked, events_limit=0)["status"] == "cancelled",
          engine_runtime.get_task(blocked, events_limit=0)["status"])
    engine_runtime.wait(cancelled, timeout=120)


# ══════════════════════════════════════════════════════════════════════════
# ②b 收尾顺序：**终结态可见 ⇒ "还了多少"已经在事件里** ✓（踩过一次的坑 ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_terminal_order(tmp: Path) -> None:
    """⚠️ 这条钉的是**一个实测踩过的窗口** ✗（2026-09-26 ✓）：

    原来那句「先还、后唤醒 ⇒ 读任务的人一定看得见」**不成立** ✗✗ —— 它只保证了**被唤醒**的读者 ✓，
    而 `bridge.run_job` 是**带超时轮询**的 ✓ ⇒ 终结态一可见，它当场收工走人 ✓ ⇒ 若"还了多少"
    那条事件还没写进去，就**永远**看不到 ✓✗。实测：同进程连跑两张，**第 1 张 `reclaimEvent`
    为空、第 2 张有** ✓✗（第 1 张要还的量最大 ⇒ 窗口最宽 ✓）。

    这里把"还"**故意放慢** ✗（300 ms ✓）⇒ 窗口被拉宽 ⇒ 旧写法**必红** ✓、修好的写法必绿 ✓。
    用**干跑**任务 ✓：不碰真权重、不依赖这台机器有没有卡 ✓（假回收自己报 ``released=True`` ✓，
    所以最后那条断言在没卡的机器上也照跑 ✓）。
    """
    import time as time_mod  # noqa: PLC0415

    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    real_reclaim = runtime_mod._reclaim_cuda_cache  # noqa: SLF001

    def slow_reclaim() -> dict[str, Any]:
        """**假装有卡 + 放慢** ✓：就是要把"终结已可见、事件还没写"的窗口拉宽 ✓。"""
        time_mod.sleep(0.3)
        return {"released": True, "reclaimedGiB": 1.5, "reservedGiB": 1.0,
                "allocatedGiB": 0.5, "freeGiB": 20.0, "totalGiB": 22.49}

    runtime_mod._reclaim_cuda_cache = slow_reclaim  # noqa: SLF001
    try:
        task_id = engine_runtime.submit(_request(tmp / "order"), assembly={"dryRun": True})
        # ⚠️ 按 `bridge.run_job` 的口径读 ✓：**带超时地轮询** ✓（不是干等被唤醒 ✓）
        snapshots: list[dict[str, Any]] = []
        deadline = time_mod.monotonic() + 120
        while time_mod.monotonic() < deadline:
            snapshot = engine_runtime.wait(task_id, timeout=0.01)
            if snapshot is None:
                break
            snapshots.append(snapshot)
            if snapshot.get("terminal"):
                break
    finally:
        runtime_mod._reclaim_cuda_cache = real_reclaim  # noqa: SLF001

    last = snapshots[-1] if snapshots else {}
    events = last.get("events") or []
    kinds = [event.get("kind") for event in events]
    reclaim_events = [event for event in events if event.get("kind") == "reclaim"]
    check("⭐ 收尾顺序: 读到**终结**的那一刻，「还了多少」已经在事件里、且是最后一条"
          "（封口与 append 同一段临界区 ⇒ 一个窗口都不留）",
          last.get("status") == "done" and kinds[-1:] == ["reclaim"],
          (last.get("status"), kinds[-3:]))
    check("收尾顺序: 事件里带的就是「还」那一步返回的数字（口径一致，不另编一份）",
          bool(reclaim_events) and reclaim_events[-1].get("reclaimedGiB") == 1.5,
          reclaim_events)


def case_missing_weights(tmp: Path) -> None:
    """⚠️ 权重不在 ⇒ 必须**报出缺哪个文件** ✓（不是"引擎不可用"一句糊过去 ✗）。"""
    try:
        engine_runtime.ensure_loaded(dit_path=str(tmp / "根本不存在.safetensors"))
    except TorchBackendUnavailable as err:
        check("缺权重: reason=pending 且报错里带**路径**（可行动 ✓）",
              getattr(err, "reason", "") == "pending" and "根本不存在" in str(err), str(err)[:120])
        return
    except Exception as err:  # noqa: BLE001
        check("缺权重: 抛的必须是 TorchBackendUnavailable", False, repr(err))
        return
    check("缺权重: 居然装上了（这条该红 ⇒ 检查装载是不是没核存在性 ✗）", False, "no raise")


# ══════════════════════════════════════════════════════════════════════════
# ③ 真装载 + 真产物 ✓（走运行时的 submit，不是直接调管线 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_real_run(tmp: Path) -> None:
    checkpoint = weights_mod.save_module_weights(h3_form.H3FormTrunk(**TRUNK),
                                                 tmp / "h3_runtime.safetensors")
    out_dir = tmp / "out"
    task_id = engine_runtime.submit(
        _request(out_dir),
        assembly={"dit_path": str(checkpoint), "audio_latent_mode": "round"})
    task = engine_runtime.wait(task_id, timeout=600)
    result = task.get("result") or {}
    outputs = result.get("outputs") or {}
    check("真跑: 任务 done", task["status"] == "done", (task["status"], task["error"]))
    check("真跑: 管线 ok", result.get("ok") is True, result.get("error"))
    check("⭐ 真跑: synthetic **仍为 True**（参考 TE / VAE 未训练 ⇒ 不许改口说是真画面）",
          result.get("synthetic") is True, result.get("synthetic"))
    check("真跑: 装了哪几个组件可回看（loadReport.components）",
          sorted((task.get("loadReport") or {}).get("components") or []) ==
          ["audioVae", "dit", "textEncoder", "vae"],
          (task.get("loadReport") or {}).get("components"))
    video = outputs.get("videoPath")
    audio = outputs.get("audioPath")
    check("真跑: **真 mp4 落盘**（文件存在且非空）",
          bool(video) and Path(video).exists() and Path(video).stat().st_size > 0, video)
    check("真跑: 双流 ⇒ **真 wav 也落盘**（audioLatentMode=round + 挂了音频 VAE）",
          bool(audio) and Path(audio).exists() and Path(audio).stat().st_size > 0, audio)
    check("真跑: 产物落在**请求指定的目录**里（不许偷偷落临时目录）",
          bool(video) and Path(video).parent == out_dir, (video, str(out_dir)))
    check("真跑: status 自述里 dit 已装（loaded.dit=True）",
          engine_runtime.status()["loaded"]["dit"] is True,
          engine_runtime.status()["loaded"])
    load_report = task.get("loadReport") or {}
    cache = load_report.get("cache") or {}
    check("⭐ 真跑: 装载报告里带**回收事实**（cache.released / reclaimedGiB ⇒ 显存口径可见不静默）",
          "released" in cache and "reclaimedGiB" in cache, cache)

    import torch  # noqa: PLC0415 —— 只为"这台机器有没有卡"这一句 ✓

    events = task.get("events") or []
    reclaim_events = [event for event in events if event.get("kind") == "reclaim"]
    if torch.cuda.is_available():
        check("⭐ 真跑: 任务**末尾**留了一条回收事件（先还后唤醒 ⇒ 读任务的人一定看得见）",
              bool(reclaim_events) and events[-1].get("kind") == "reclaim",
              [event.get("kind") for event in events[-3:]])
    else:
        check("真跑: 没卡 ⇒ 如实**不留**回收事件（不编数字）✗",
              not reclaim_events, reclaim_events)


# ══════════════════════════════════════════════════════════════════════════
# ③b 回收缓存：还的是"空着的"，**不许**把在用的还掉 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_reclaim() -> None:
    """⚠️ 回收 **≠ 卸载** ✗ —— 这条就是钉住这个区别的 ✓。

    ⚠️ 为什么要还 ✗（2026-09-26 实测 ✓，A5000 22.49 GiB ✓）：装完 SDXL **真占用 6.73 GiB** ✓
    而缓存**保留 13.88 GiB** ✗ ⇒ 驱动视角只剩 7.37 GiB ✗；再采样一张 ⇒ 保留 **15.76 GiB** ✗
    ⇒ 只剩 5.47 GiB ✗✗；还一次 ⇒ 保留 **8.24 GiB** ✓、驱动视角 **12.98 GiB** ✓ —— 而在用的
    6.73 GiB **一点没少** ✓。不还就等于「缓存看着占满整卡」✗，同机 ComfyUI 先撞它 ✓✗。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    before = runtime_mod._vram_facts()  # noqa: SLF001 —— 白盒：就为这条不变量 ✓
    allocated_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    key_before = engine_runtime._loadKeyTuple  # noqa: SLF001

    report = runtime_mod._reclaim_cuda_cache()  # noqa: SLF001

    allocated_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    check("回收: 台账口径是**四个键**（allocated/reserved/free/total，单位 GiB）",
          before is None or set(before) == {"allocatedGiB", "reservedGiB", "freeGiB", "totalGiB"},
          before)
    if before is None:
        check("回收: 没卡 ⇒ 如实说「没得还」（released=False + note ✗ 不抛 ✓）",
              report.get("released") is False and "note" in report, report)
    else:
        check("回收: 有卡 ⇒ released=True 且还回的量**非负**",
              report.get("released") is True and float(report.get("reclaimedGiB", -1)) >= 0,
              report)
        check("回收: 还完的账**只减不增**（reserved 不涨 / free 不降）",
              float(report.get("reservedGiB") or 0) <= float(before["reservedGiB"]) + 1e-6
              and float(report.get("freeGiB") or 0) >= float(before["freeGiB"]) - 1e-6,
              (before, report))
    check("⭐ 回收 **≠** 卸载: 在用的张量**一点没动**（memory_allocated 前后必须一致）",
          allocated_after == allocated_before, (allocated_before, allocated_after))
    check("⭐ 回收 **≠** 卸载: 装好的东西**还在**（loadKeyTuple 不变 + loaded.dit=True）",
          engine_runtime._loadKeyTuple == key_before  # noqa: SLF001
          and engine_runtime.status()["loaded"]["dit"] is True,
          (key_before, engine_runtime.status()["loaded"]))


# ══════════════════════════════════════════════════════════════════════════
# ④ 卸载：真丢 + 正忙拒卸 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_unload() -> None:
    report = engine_runtime.unload()
    check("卸载: 真装了才会报 unloaded=True", report["unloaded"] is True, report)
    flags = engine_runtime.status()["loaded"]
    check("⭐ 卸载: 四个组件**全部回到 False**（丢张量 + 清缓存 ⇒ 不是只改了个标记）",
          flags == {"dit": False, "textEncoder": False, "vae": False, "audioVae": False}, flags)

    # ⚠️ **白盒**（手工把 current 置上）✓：真任务可能几毫秒就跑完 ✗ ⇒ 靠它跑到来撞守卫会是**抖动** ✓✗
    engine_runtime._current = "fake-task"          # noqa: SLF001 —— 就为这条守卫 ✓
    try:
        try:
            engine_runtime.unload()
        except EngineBusy:
            check("守卫: 正忙时**拒卸**（EngineBusy ⇒ 不会把在跑的模型从底下抽走）", True)
        else:
            check("守卫: 正忙时居然卸成功了（这条该红）", False, "no raise")
        check("守卫: force=true 可强卸（逃生门留着 ✓）",
              engine_runtime.unload(force=True)["unloaded"] is False)
    finally:
        engine_runtime._current = None             # noqa: SLF001 —— 别把假状态留给后面的用例 ✓


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="engine_runtime_"))
    case_status()
    case_queue(tmp)
    case_terminal_order(tmp)
    case_missing_weights(tmp)
    if not _have_torch():
        skip("缺 torch ⇒ 真装载 / 真产物 / 真卸载 跳过 ✓")
    elif not media_mod.have_ffmpeg():
        skip("缺 ffmpeg ⇒ 落盘那几条跳过 ✓")
    else:
        case_real_run(tmp)
        case_reclaim()
        case_unload()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
