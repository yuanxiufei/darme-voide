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
8. **取消必须当场给裁决** ✓（E3.2 ✓）：`accepted`（排队中 ⇒ 已当场终结）/ `too_late`（已经在跑 ⇒
   只置旗标、**不保证**停）/ `completed`（早已终结 ⇒ 取消无从生效）；裁决在**同一把锁内**定 ✓
   —— 调用方事后比 `status` 读的是另一瞬间的事实 ✗（二次取消会谎称"又停了一次"，反向验证过 ✓）。

⚠️ 「不依赖外部」的**反向证明**也在这里 ✓：`status()["externalDependencies"]` 必须是空列表 ✓
（它一旦不是空的，就说明这条路上又接回了 ComfyUI / SD WebUI / Ollama 之类外部服务 ✓✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_runtime_test.py
"""
from __future__ import annotations

import inspect
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
    first = engine_runtime.cancel(blocked) or {}
    check("取消: 排队中的**立刻**终结（cancelled）",
          engine_runtime.get_task(blocked, events_limit=0)["status"] == "cancelled",
          engine_runtime.get_task(blocked, events_limit=0)["status"])
    first_block = first.get("cancel") or {}
    check("取消裁决: 排队中被取消 ⇒ ``accepted`` + ``stopped: true``（保证停了 ✓）",
          first_block.get("verdict") == "accepted" and first_block.get("stopped") is True,
          first_block)
    again = engine_runtime.cancel(blocked) or {}
    again_block = again.get("cancel") or {}
    check("取消裁决: 同一个任务点第二次 ⇒ ``completed`` + ``stopped: false``（**不许**谎称又停了一次 ✓✗）",
          again_block.get("verdict") == "completed" and again_block.get("stopped") is False,
          again_block)
    done = engine_runtime.wait(cancelled, timeout=120) or {}
    late_done = engine_runtime.cancel(cancelled) or {}
    late_done_block = late_done.get("cancel") or {}
    check("取消裁决: 已经跑完 ⇒ ``completed`` + ``stopped: false``（来晚了，但**不是错误** ✓✗）",
          done.get("status") == "done" and late_done_block.get("verdict") == "completed"
          and late_done_block.get("stopped") is False, late_done_block)

    # ⚠️ **白盒**造一个"正在跑"的任务 ✓：真等它跑起来再点取消会是**抖动** ✓✗（干跑几毫秒就完 ✓）
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    running = runtime_mod.EngineTask(id="t-running", stage="h3", request=_request(out_dir),
                                     status="running", createdAt=runtime_mod._now_ms())
    engine_runtime._tasks["t-running"] = running        # noqa: SLF001 —— 就为这条裁决 ✓
    try:
        too_late_report = engine_runtime.cancel("t-running") or {}
        too_late_block = too_late_report.get("cancel") or {}
        check("取消裁决: 已经在跑 ⇒ ``too_late`` + ``stopped: false``（**不保证**停 ✗，但也不许翻成报错 ✓✗）",
              too_late_block.get("verdict") == "too_late" and too_late_block.get("stopped") is False,
              too_late_block)
        check("取消裁决: ``too_late`` 时快照**仍是 running** 且旗标已置"
              "（不许为了让前端看着舒服就把状态改成 cancelled ✓✗）",
              too_late_report.get("status") == "running" and running.cancelEvent.is_set(),
              (too_late_report.get("status"), running.cancelEvent.is_set()))
    finally:
        engine_runtime._tasks.pop("t-running", None)    # noqa: SLF001 —— 别把假任务留给后面的用例 ✓
    check("取消裁决: 未知任务 ⇒ ``None``（路由那边才是 400 ✓）",
          engine_runtime.cancel("nope") is None, engine_runtime.cancel("nope"))


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


class _FakeBackend:
    """白盒假后端 ✓ —— 只够 `status()` 把自述拼出来 ✓（不碰 torch / 不碰权重 ✓）。"""

    def describe(self) -> dict[str, Any]:
        return {"device": "cpu", "name": "fake"}


# ══════════════════════════════════════════════════════════════════════════
# ②c 换装配：**先断旧的、再造新的** ✓（同时压两份 = OOM ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_replace_order() -> None:
    """⚠️ H3 主干 int8 是 **17.94 GiB** ✓（offload 时在 CPU 锁页 ✓），而装配期还要临时建一份
    **fp32 主干 78 GiB** ✓✗（见 `_prepare_h3_module` ✓）⇒ 「旧的还活着就装新的」= 同时压两份 ✓✗
    （96 GB 的机器也炸 ✓；老路更硬：17.94 GiB 还在显存里 ⇒ 22.49 GiB 的卡**必 OOM** ✗）。

    出图那条路早就按「先卸后装」写了 ✓（`_ensure_image_loaded` ✓），**主路原本是装完才覆盖** ✗
    —— 这条判据架在**新后端构造的那一刻** ✓：那时旧后端必须已经断了 ✓。
    """
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    seen: dict[str, Any] = {}

    class _Probe:
        """假后端：**被构造的那一刻**记下"旧后端还在不在" ✓，然后故意装不上 ✓。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen["stale"] = engine_runtime._backend  # noqa: SLF001 —— 就是这条判据要看的东西 ✓
            raise TorchBackendUnavailable("探针：故意装不上 ✗（不改真模型 ✓）", reason="pending")

    real_backend_cls = runtime_mod.TorchBackend
    engine_runtime._backend = _FakeBackend()                    # noqa: SLF001 —— 摆一个"旧的还活着" ✓
    engine_runtime._loadKeyTuple = ("stale",)                   # noqa: SLF001
    engine_runtime._loadReport = {"components": ["dit"]}        # noqa: SLF001
    setattr(runtime_mod, "TorchBackend", _Probe)                # noqa: B010 —— 只在这一条里换掉构造 ✓
    try:
        try:
            engine_runtime.ensure_loaded(dit_path="探针.safetensors", force=True)
        except TorchBackendUnavailable:
            check("⭐ 换装配: 新后端**构造那一刻**旧后端已断（`_backend is None`）"
                  "⇒ 不会两份大模型同时在 ✓（装完才覆盖的话这里会看到那个哨兵 ✗✗）",
                  "stale" in seen and seen["stale"] is None, seen)
            check("换装配: 装失败后 `_backend` 保持 None ✓（**不半装** ✗、也不留着旧的骗人 ✓）",
                  engine_runtime._backend is None, engine_runtime._backend)  # noqa: SLF001
        except Exception as err:  # noqa: BLE001
            check("换装配: 抛的必须是 TorchBackendUnavailable", False, repr(err))
        else:
            check("换装配: 居然装上了（这条该红 ⇒ 探针没生效 ✗）", False, "no raise")
    finally:
        setattr(runtime_mod, "TorchBackend", real_backend_cls)  # noqa: B010 —— 立刻复原 ✓
        engine_runtime._backend = None                          # noqa: SLF001
        engine_runtime._loadKeyTuple = None                     # noqa: SLF001
        engine_runtime._loadReport = None                       # noqa: SLF001


# ══════════════════════════════════════════════════════════════════════════
# ②c' **不半装**（E2.6 ✓）：装到第三个组件才坏 ⇒ 已挂上的两个也不许留下 ✓
# ══════════════════════════════════════════════════════════════════════════
def _load_projection(status: dict[str, Any]) -> dict[str, Any]:
    """只取「装了什么」那几项 ✓ —— 不拿整个 `status()` 比 ✗：
    里面还有量计 / 任务史 / LoRA 账本（它们按设计就会变 ✓✗）。"""
    return {key: status.get(key) for key in ("loaded", "backend", "loadReport", "deviceSource")}


def case_half_load() -> None:
    """E2.6：装载失败 ⇒ 组件状态与调用前**逐项相同** ✓✗。

    `case_replace_order` 钉的是「**新后端构造那一刻**旧的就断了」✓（那一支在挂任何组件之前就失败 ✓）；
    这一条钉的是**更靠后**的一支：DiT 与 TE **已经挂上**，第三个组件（VAE）才坏 ✓✗ ——
    这时若只把 `_backend = None` 而少了 `del backend`，半装的那一份会被异常回溯/局部变量吊着 ✓：
    张量不回收、`status()` 还报着 `loaded.dit=True`，而用户看到的是「装失败了」✓✗。

    所以判据要**三件一起看** ✓：① 半装的证据（组件确实挂上过 ⇒ 否则这条在验一个还没开始的装载 ✓✗）；
    ② 引用真断了（`weakref` 取不到 ⇒ 张量可回收 ✓）；③ `status()` 与装载前逐项相同，
    且**同一运行时随后还能装成功** ✓（有残留就会在这里红 ✓）。
    """
    import gc  # noqa: PLC0415
    import weakref  # noqa: PLC0415

    from app.services.engine import lora_cache  # noqa: PLC0415
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    engine_runtime._backend = None                # noqa: SLF001 —— 从干净态起判 ✓
    engine_runtime._loadKeyTuple = None           # noqa: SLF001
    engine_runtime._loadReport = None             # noqa: SLF001
    before = _load_projection(engine_runtime.status())
    lora_before = int(lora_cache.shared().facts().get("refTotal") or 0)

    made: list[Any] = []

    class _HalfBackend:
        """假后端：DiT 与 TE 真挂上 ✓，之后由被替换的 `_attach_video_vae` 炸 ✓（不碰 torch ✓）。

        ⚠️ 必须把 `_model` / `_textEncoder` / `_vae` / `_audioVae` 真的挂到自身上 ✗✗ ——
        `status().loaded` 读的就是这四个属性 ✓（见 `_loaded_flags` ✓）；不挂上去，
        "逐项相同"那条判据会因为"真后端也报 False"而**永远绿** ✓✗。
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.components: set[str] = set()
            self._model = self._textEncoder = self._vae = self._audioVae = None
            made.append(self)

        def load_weights(self, *, path: Any = None, loras: Any = None) -> dict[str, Any]:
            self.components.add("dit")
            self._model = object()
            return {"path": "假主干 ✓", "synthetic": True}

        def describe(self) -> dict[str, Any]:
            return {"device": "cpu", "name": "half", "config": {"text_dim": 64}}

        def attach_text_encoder(self, *, tokenizer_path: Any = None) -> bool:
            self.components.add("textEncoder")
            self._textEncoder = object()
            return True

    def _fake_video_vae(backend: Any, trunk: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        backend.components.add("vae")
        backend._vae = object()                                    # noqa: SLF001
        return {"kind": "synthetic"}

    def _fake_audio_vae(backend: Any, trunk: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        backend.components.add("audioVae")
        backend._audioVae = object()                               # noqa: SLF001
        return {"kind": "synthetic"}

    def _boom_vae(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("第二个组件坏了 ✓（故意 ✓）")

    real_backend_cls = runtime_mod.TorchBackend
    real_video_vae = runtime_mod._attach_video_vae                  # noqa: SLF001
    real_audio_vae = runtime_mod._attach_reference_audio_vae        # noqa: SLF001
    setattr(runtime_mod, "TorchBackend", _HalfBackend)
    setattr(runtime_mod, "_attach_video_vae", _boom_vae)
    ref = None
    try:
        try:
            engine_runtime.ensure_loaded(dit_path="假主干.safetensors", force=True)
        except TorchBackendUnavailable as err:
            check("不半装: 抛 `TorchBackendUnavailable` 且 `reason=pending`、报错点名**哪个组件坏了** ✓"
                  "（一句「引擎不可用」等于不说 ✓✗）",
                  getattr(err, "reason", "") == "pending" and "第二个组件坏了" in str(err),
                  str(err)[:160])
            half = made[0] if made else None
            check("不半装: 失败前 DiT 与 TE **确实挂上过** ✓（否则这条判据在验一个还没开始的装载 ✓✗）",
                  half is not None and half.components == {"dit", "textEncoder"},
                  getattr(half, "components", None))
            ref = weakref.ref(half) if half is not None else None
            del half
        except Exception as err:  # noqa: BLE001
            check("不半装: 抛的必须是 TorchBackendUnavailable", False, repr(err))
        else:
            check("不半装: 居然装上了（这条该红 ⇒ 探针没生效 ✗）", False, "no raise")

        # ⚠️ 这条判据只能问「**运行时自己**留没留」✗✗ —— 两个实测踩过的坑：
        #    ① 异常还绑在 `except` 的名字上时，回溯帧的**待调用参数元组**（`(半装实例, trunk)`）
        #       会把它吊住 ⇒ 必须出了 `except` 再判，否则测的是"异常对象在不在手上"而不是"运行时放没放" ✓✗；
        #    ② **异常被上层留住**（记日志 / 再抛）时，那份半装的照样活着 ✓✗ —— 实测：把 `del backend`
        #       改回留个局部变量，判据仍 PASS ✓；而把异常留在 `held` 里再判，**改了 `del backend` 也照样红** ✓。
        #       ⇒ 那是 CPython 回溯自己钉住的（`_attach_video_vae(backend, trunk, …)` 那次调用的参数元组），
        #       **不是运行时能控制的** ✗；本仓不做「`from None` / 清 `__traceback__`」来省这份峰值 ✓
        #       （诊断信息比那点峰值值钱 ✓），所以判据只钉运行时这一侧 ✓。
        made.clear()
        gc.collect()
        check("⭐ 不半装: **运行时自己不留**半装的那一份 ✓（放开异常后 `weakref` 取不到 ⇒ 张量可回收 ✓✗）"
              "—— `self._backend = backend` 若挪到失败分支里、或漏了 `del backend` 之后又被自身引用，这里就红 ✓",
              ref is not None and ref() is None, "半装的那份还被运行时吊着 ✓✗")

        after = _load_projection(engine_runtime.status())
        check("不半装: `status()` 与装载前**逐项相同** ✓（loaded / backend / loadReport / deviceSource）"
              "—— 报着 `loaded.dit=True` 就是假装载 ✓✗",
              after == before, {"before": before, "after": after})
        lora_after = int(lora_cache.shared().facts().get("refTotal") or 0)
        check("不半装: LoRA 账本**没多出引用** ✓（装到一半崩了也不许吊着租约 ✓✗）",
              lora_after == lora_before, {"before": lora_before, "after": lora_after})

        # ③ 崩过一次之后**同一运行时还能装成功** ✓（有残留挡住就会红 ✓✗）
        setattr(runtime_mod, "_attach_video_vae", _fake_video_vae)
        setattr(runtime_mod, "_attach_reference_audio_vae", _fake_audio_vae)
        report = engine_runtime.ensure_loaded(dit_path="假主干.safetensors", force=True)
        loaded = engine_runtime.status()
        check("不半装: 崩过一次之后**同一运行时还能装成功** ✓（残留挡住、或装载键被脏值占住，都会在这里红 ✓✗）",
              report.get("reused") is False and loaded.get("loaded") == {
                  "dit": True, "textEncoder": True, "vae": True, "audioVae": True},
              {"reused": report.get("reused"), "loaded": loaded.get("loaded")})
    finally:
        setattr(runtime_mod, "TorchBackend", real_backend_cls)
        setattr(runtime_mod, "_attach_video_vae", real_video_vae)          # noqa: SLF001
        setattr(runtime_mod, "_attach_reference_audio_vae", real_audio_vae)  # noqa: SLF001
        engine_runtime._backend = None                            # noqa: SLF001
        engine_runtime._loadKeyTuple = None                       # noqa: SLF001
        engine_runtime._loadReport = None                         # noqa: SLF001
        made.clear()


# ══════════════════════════════════════════════════════════════════════════
# ②d 装配量计：**这一程**复用 / 真装 / 丢弃各几次 ✓（E3.4 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_cache_meter() -> None:
    """`status().cache` ✓（E3.4 ✓，2026-10-29 ✓）。

    以前只有 `loadReport.reused` ✗✗ —— 那是**最后一份**装载报告 ✓，只讲"这一次" ✓：
    「这一程复用了几次、真装了几次、丢了几次」没有出口 ⇒ 调参时说不清"慢是重装还是别的"✓✗，
    "怎么越跑越卡"也只能靠猜 ✓✗。

    ⚠️ 口径**只到装配复用门** ✓：`cache_guard` / `latent_container` 是**纯函数模块**、一个计数器
    都没有 ✗（方案里"取自这两个模块的命中/淘汰计数"落不了地 ✓✗ —— 与其编两个假计数，不如把
    口径写在 `scope` 里讲明白 ✓）。

    ⚠️ 白盒摆状态 ✗：真装一份要几十秒 + 真权重 ✓✗ ⇒ 命中 / 卸载 / 计数这些**记账**的事直接摆
    `_backend` 与 `_loadKeyTuple` ✓（这里要验的是"记了几笔、记在哪"✓；"装得对不对"归 `case_real_run` ✓）。
    """
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    def _fresh() -> None:
        """把量计归零 ✓（每个子断言只看自己这一段 ✓ 不受前面用例影响 ✓）。"""
        engine_runtime._cacheHits = 0                      # noqa: SLF001
        engine_runtime._cacheMisses = 0                    # noqa: SLF001
        engine_runtime._cacheEvictions = 0                 # noqa: SLF001
        engine_runtime._cacheEvents.clear()                # noqa: SLF001
        engine_runtime._cacheSince = runtime_mod._now_ms()  # noqa: SLF001

    def _key(dit_path: str) -> tuple[Any, ...]:
        """与 `ensure_loaded` **同一份**键 ✓ —— 默认值**现取** ✓（手抄参数清单迟早对不上 ✗✗）。"""
        params = inspect.signature(engine_runtime.ensure_loaded).parameters
        kwargs = {name: param.default for name, param in params.items()
                  if name not in ("self", "force") and param.default is not inspect.Parameter.empty}
        return runtime_mod._load_key(**{**kwargs, "dit_path": dit_path,
                                        "loras": runtime_mod._lora_key(kwargs.get("loras"))})

    _fresh()
    cache = engine_runtime.status()["cache"]
    check("量计: `status().cache` 三数齐（hits / misses / evictions ✓ E3.4 ✓）",
          all(key in cache for key in ("hits", "misses", "evictions")), sorted(cache))
    check("量计: 口径写明白（**不含** torch allocator / hybrid 合并缓存 ✓✗）",
          "torch" in str(cache.get("scope")) and "hybrid" in str(cache.get("scope")),
          cache.get("scope"))
    check("量计: 「重启即归零」写明白（读到 0 只说明这一程没复用 ✓ 不是从没复用 ✓✗）",
          "归零" in str(cache.get("note")), cache.get("note"))

    try:
        # ① 同键再要一次 ⇒ **复用**（一次都不用花 ✓）
        _fresh()
        engine_runtime._backend = _FakeBackend()                                # noqa: SLF001
        engine_runtime._loadKeyTuple = _key("D:\\m\\dit.safetensors")           # noqa: SLF001
        engine_runtime._loadReport = {"components": ["dit"], "device": "cpu"}    # noqa: SLF001
        hit = engine_runtime.ensure_loaded(dit_path="D:\\m\\dit.safetensors")
        after_hit = engine_runtime.status()["cache"]
        check("量计: 同键再要一次 ⇒ hit（reused ✓ 且 misses **不动** ✓）",
              hit.get("reused") is True and after_hit["hits"] == 1 and after_hit["misses"] == 0,
              (hit.get("reused"), after_hit["hits"], after_hit["misses"]))
        last = (after_hit["recent"] or [{}])[-1]
        check("量计: hit 可追溯（`recent` 末条 kind=hit、带 12 位键摘要与非空理由 ✓）",
              last.get("kind") == "hit" and len(str(last.get("key") or "")) == 12
              and bool(last.get("reason")), last)

        # ② `force=True` 但键**相同** ⇒ 内部当场复用 ⇒ 账必须记 hit ✗✗（记 miss+eviction 是两条假账 ✓✗）
        _fresh()
        forced = engine_runtime.ensure_loaded(dit_path="D:\\m\\dit.safetensors", force=True)
        forced_facts = engine_runtime.status()["cache"]
        check("量计: force=True 但键相同 ⇒ 实际是**复用** ⇒ 只涨 hits（misses / evictions 不许涨 ✓✗）",
              forced.get("reused") is True and forced_facts["hits"] == 1
              and forced_facts["misses"] == 0 and forced_facts["evictions"] == 0,
              (forced.get("reused"), forced_facts["hits"], forced_facts["misses"],
               forced_facts["evictions"]))

        # ③ 忙到拒装 ⇒ **什么都没发生** ⇒ 一笔都不许记 ✗✗
        _fresh()
        engine_runtime._current = "t-busy"                                     # noqa: SLF001
        try:
            engine_runtime.ensure_loaded(dit_path="别的装配", force=True)
        except EngineBusy:
            pass
        except Exception as err:  # noqa: BLE001
            check("量计: 忙时抛的必须是 EngineBusy", False, repr(err))
        else:
            check("量计: 忙时居然装上了（这条该红）", False, "no raise")
        finally:
            engine_runtime._current = None                                     # noqa: SLF001
        busy = engine_runtime.status()["cache"]
        check("量计: 忙到拒装 **不进计数**（三数全 0、`recent` 空 ✓）",
              (busy["hits"], busy["misses"], busy["evictions"]) == (0, 0, 0) and not busy["recent"],
              (busy["hits"], busy["misses"], busy["evictions"], busy["recent"]))

        # ④ 换装配 ⇒ 真装一次（装不上 ✓，但"决定要装"和"丢旧的"都**真发生了** ✓）
        _fresh()
        try:
            engine_runtime.ensure_loaded(dit_path="D:\\m\\没有这个文件.safetensors")
        except TorchBackendUnavailable:
            pass
        except Exception as err:  # noqa: BLE001
            check("量计: 装失败抛的必须是 TorchBackendUnavailable", False, repr(err))
        else:
            check("量计: 装不存在的权重居然装上了（这条该红 ✗）", False, "no raise")
        miss = engine_runtime.status()["cache"]
        check("量计: 真装一次 ⇒ miss +1、evictions +1（旧的先被放掉 ✓）",
              miss["misses"] == 1 and miss["evictions"] == 1,
              (miss["misses"], miss["evictions"]))
        check("量计: 顺序是 miss → eviction（先决定要装 ✓ 再丢旧的 ✓；反了就是记账写错地方 ✓✗）",
              [event.get("kind") for event in miss["recent"]] == ["miss", "eviction"], miss["recent"])

        # ⑤ 显式卸载 ⇒ eviction（且**不清零** —— 这是进程量计 ✗ 不是"当前装了几个" ✓✗）
        _fresh()
        since = engine_runtime.status()["cache"]["since"]
        engine_runtime._backend = _FakeBackend()                              # noqa: SLF001
        engine_runtime._loadKeyTuple = _key("D:\\m\\dit.safetensors")          # noqa: SLF001
        engine_runtime._loadReport = {"components": ["dit"]}                   # noqa: SLF001
        unloaded = engine_runtime.unload()
        after_unload = engine_runtime.status()["cache"]
        check("量计: 显式卸载 ⇒ evictions +1 且可追溯（`recent` 末条 kind=eviction ✓）",
              unloaded.get("unloaded") is True and after_unload["evictions"] == 1
              and (after_unload["recent"] or [{}])[-1].get("kind") == "eviction", after_unload)
        check("量计: 卸完**不清零**（这是**进程量计** ✗ 不是「当前装了几个」✓✗）",
              after_unload["evictions"] == 1 and after_unload["since"] == since, after_unload)

        # ⑥ 本来就没装 ⇒ `unload` **不记** eviction（"没丢过"不许报成"丢了一次" ✓✗）
        _fresh()
        engine_runtime.unload()
        empty = engine_runtime.status()["cache"]
        check("量计: 没装东西时 `unload` **不记** eviction（本来就没丢过 ✓）",
              (empty["hits"], empty["misses"], empty["evictions"]) == (0, 0, 0) and not empty["recent"],
              empty)

        # ⑦ 环形缓冲：`recent` 有上限（否则长跑进程的内存只涨不落 ✓✗）
        _fresh()
        keep = runtime_mod.EngineRuntime._CACHE_EVENT_KEEP  # noqa: SLF001
        for _ in range(keep + 4):
            engine_runtime._note_cache("hit", key="k", reason="压力测试 ✓")  # noqa: SLF001
        bounded = engine_runtime.status()["cache"]["recent"]
        check(f"量计: `recent` 只留最近 {keep} 条（长跑进程不许涨 ✓✗）",
              len(bounded) == keep and bounded[-1].get("kind") == "hit", len(bounded))

        # ⑧ 打错字 ⇒ **当场报**（静默不计数 = 量看着正常但根本没有这一档 ✓✗）
        try:
            engine_runtime._note_cache("ヒット", key="打错字 ✓", reason="打错字 ✓")  # noqa: SLF001
        except ValueError:
            check("量计: 不认识的种类**当场报**（打错字不许静默不计数 ✓✗）", True)
        else:
            check("量计: 不认识的种类居然不报（这条该红）", False, "no raise")
    finally:
        # 复原：假状态一个字都不许留给后面的用例 ✓（量计也清了 —— 上面那些是摆出来的 ✓✗）
        engine_runtime._backend = None          # noqa: SLF001
        engine_runtime._loadKeyTuple = None     # noqa: SLF001
        engine_runtime._loadReport = None       # noqa: SLF001
        engine_runtime._current = None          # noqa: SLF001
        _fresh()


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
    case_replace_order()
    case_half_load()
    case_cache_meter()
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
