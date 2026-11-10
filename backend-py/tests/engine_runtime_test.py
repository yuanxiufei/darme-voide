"""S? 自检：**引擎运行时** ✓ —— 「真跑一次」那条路终于有入口了（2026-09-26 ✓）。

这一套钉的是「**运行时真的把引擎跑起来了**」✓，不是"接口返回 200"✗：

1. **装载** 用 `load_weights` 的**真路径** ✓（造缩小版 H3 形态权重 ✓ ⇒ 判形态那条路照走 ✓）；
2. **排队** 真串行 ✓（两个任务：后一个的 `startedAt` **不早于**前一个的 `finishedAt` ✓）；
3. **产物** 真 mp4 + 真 wav ✓（`result.outputs.videoPath` 落盘 ✓、文件真在 ✓）；
4. **诚实** `synthetic` 必须**按已装件算** ✓✗（2026-11-01 ✓）：参考 TE / VAE 未训练 ⇒ `True` ✓；
   真权重 VAE（`REAL_VAE_KINDS`）挂上 ⇒ `False` ✓。⚠️ 它曾经是**类属性写死 `True`** ✓✗ ——
   那时这条自检只能钉住"写死的值"，钉不住"说的是不是实话"✗；同理 `canGenerate` 曾写死 `False`
   ⇒ 装完整链、真出过 mp4 也照样报「不能出片」✓✗（见 `case_generation_facts` ①⁶ 与 `case_status` ①⁵）；
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
import textwrap
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ 干跑报告隔离到临时树 ✓✗（同 ``engine_pipeline_test`` 的理由）：
#: 不设它 ⇒ `data/engine-dryrun/engine_runtime_xxxx/` 每次跑都多一个 ✓。
os.environ.setdefault("DARME_ENGINE_DRYRUN", tempfile.mkdtemp(prefix="dryrun_rt_"))
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
    check("自述: 没建后端时 synthetic 按「一件都没装」算 True（DiT / 真权重 VAE 都没有 ⇒ 只可能是噪声）",
          status["synthetic"] is True, status["synthetic"])
    check("自述: 没建后端时 canGenerate 必须是 False（**且这一格是算出来的**，见 `case_generation_facts`）",
          status["canGenerate"] is False, status["canGenerate"])
    check("自述: 没装也能答「将会在哪跑」（deviceSource=probe ⇒ cpu 还是 cuda 要一眼看见）",
          (status["device"] in ("cpu", "cuda") and status["deviceSource"] in ("probe", "loaded", None)),
          (status["device"], status["deviceSource"]))
    #: ⭐ ①⁵ 两个「缺口」面**必须不同名、不同形** ✓✗（2026-11-01 ✓）：
    #: `capabilities().knownGaps` 是**环境**缺口（`[{id, severity, summary, facts, fix}]` ✓），
    #: `status().codeGaps` 是**代码**缺口（`[str]` ✓ —— 还没写前向的族）。
    #: 同名（都叫 `knownGaps`）的后果：前端按 `.summary` 读代码缺口 ⇒ 一串 `undefined` ✓✗；
    #: 反向按「有 codeGaps」判「要修代码」又会被环境缺口顶掉 ✓✗ —— 两件事都会读错 ✓。
    from app.services.engine import facade as facade_mod            # noqa: PLC0415
    from app.services.engine import torch_backend as tb_mod        # noqa: PLC0415

    caps = facade_mod.capabilities()
    gaps = status.get("codeGaps")
    check("①⁵ 自述: 代码缺口叫 `codeGaps`（字符串数组、与 `torch_backend.CODE_GAPS` 同源）"
          "且 `status` 里**不许**出现 `knownGaps` ✗✗（那是环境缺口的名字 ✓）",
          isinstance(gaps, list) and bool(gaps)
          and all(isinstance(item, str) and item for item in gaps)
          and list(gaps) == list(tb_mod.CODE_GAPS)
          and "knownGaps" not in status,
          {"codeGaps": gaps, "statusKeys": sorted(status)})
    #: ⚠️ 这里**不能**断言「环境缺口非空」✗：它按设计就是「空数组 = 没有既知缺口」✓
    #: （本机 torch 正常时正是空 ✓）⇒ 那样写会在健康的机器上报假红 ✓✗。
    check("①⁵ 两个缺口面的**形状必须不同**（环境缺口 dict / 代码缺口 str —— 同名会读成 undefined）",
          isinstance(caps.get("knownGaps"), list)
          and all(isinstance(item, dict) for item in caps["knownGaps"])
          and not any(item in caps["knownGaps"] for item in gaps),
          {"capabilities.knownGaps": caps.get("knownGaps"), "codeGaps": gaps})

    #: ⭐ ①⁶ 物化缓存的账（§4.4 ✓ 2026-11-07 ✓）：**体积**与**次数**必须**同时**报 ✓✗。
    #: 只报次数（"命中 3 次"）⇒ 用户不知道盘上压了 8 GiB ✓✗；只报体积 ⇒ 看不出这层有没有在起作用 ✓✗。
    #: 三处易读错的默认值也要钉住：`enabled=False` 是**总开关关着** ✓（不是"没命中"✗）、
    #: `limitGiB=None` 是**不限** ✓（不是"限了 0"✗）、`hits` 是**本进程**的 ✓（重启清零 ✓）。
    cache = status.get("diskCache") or {}
    check("①⁶ 自述: `diskCache` 同时给**体积**与**次数**（缺一个就变成半句话）",
          all(key in cache for key in ("dir", "enabled", "files", "sidecars", "diskGiB",
                                       "limitGiB", "freeGiB", "hits", "misses", "refused")),
          sorted(cache))
    check("①⁶ 自述: 读不到目录也**不编 0 GiB 假装量过**（`dir` 非空 + 数值是数字或 None）",
          isinstance(cache.get("dir"), str) and bool(cache.get("dir"))
          and isinstance(cache.get("diskGiB"), (int, float))
          and isinstance(cache.get("hits"), int) and isinstance(cache.get("misses"), int),
          cache)
    check("①⁶ 自述: `enabled` 是**布尔**（关掉时必须能一眼读出是关着，而不是「0 次命中」）",
          isinstance(cache.get("enabled"), bool), cache.get("enabled"))


# ══════════════════════════════════════════════════════════════════════════
# ①⁶ 「此刻能不能出片」必须**算**出来，不许写死 ✓✗（2026-11-01 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_generation_facts() -> None:
    """⭐ `TorchBackend._generation_facts`：装了什么 ⇒ 报什么 ✓✗。

    ⚠️ 这条钉的是**实测踩过的误导** ✗：`describe()["canGenerate"]` 原先**写死 `False`** ✓、
    `pendingParts` 塞的是 75 行「当时缺什么、后来怎么补的」**沿革散文** ✓✗ ⇒ H3 / Z-Image /
    Krea2 / Qwen-Image 2.1 / Flux 全都端到端出过片之后，`GET /engine/backends`（UI 唯一的自述口 ✓）
    仍报「不能出片 ＋ 一堆没实现」✓✗ —— 一台能用的机器被说成不能用 ✓。
    ⇒ 判据：**三件齐 + 真权重 VAE ⇒ True**；反向验证（本轮做过 ✓）：把那格改回 `False`
    （或把 `synthetic` 改回类属性 `True`），本条第二/四条立刻红 ✓。

    ⚠️ 走 `object.__new__` **不跑 `__init__`** ✗：不为这条判据 import torch ✓（判据零依赖 ✓），
    只钉语义 —— 「按已装件算」这件事本身与 torch 无关 ✓。
    """
    from app.services.engine import torch_backend as tb_mod            # noqa: PLC0415

    backend = object.__new__(tb_mod.TorchBackend)                     # noqa: SLF001
    backend._available = True                                         # noqa: SLF001
    backend._reason = ""                                              # noqa: SLF001
    backend._model = backend._textEncoder = backend._vae = None       # noqa: SLF001
    backend._audioVae = None                                          # noqa: SLF001
    backend._vaeKind = None                                           # noqa: SLF001
    backend._form = None                                              # noqa: SLF001

    empty = backend._generation_facts()                               # noqa: SLF001
    check("①⁶ 一件没装 ⇒ canGenerate=False 且**逐件**点名（DiT / 文本编码器 / 画面 VAE），"
          "不按「机制」列 ✗（机制层缺口是 `codeGaps` ✓）",
          empty["canGenerate"] is False and len(empty["pendingParts"]) == 3
          and "DiT" in empty["pendingParts"][0]
          and "文本编码器" in empty["pendingParts"][1]
          and "画面 VAE" in empty["pendingParts"][2]
          and empty["synthetic"] is True,
          empty)

    backend._model = object()                                         # noqa: SLF001
    backend._textEncoder = object()                                   # noqa: SLF001
    backend._vae = object()                                           # noqa: SLF001
    backend._vaeKind = "flux-real"                                    # noqa: SLF001
    full = backend._generation_facts()                                # noqa: SLF001
    check("①⁶ 三件齐 ＋ 真权重 VAE ⇒ canGenerate=True / pendingParts 空 / synthetic=False"
          "（**这就是 2026-11-01 之前恒 False 的那一格** ✓✗）",
          full["canGenerate"] is True and full["pendingParts"] == []
          and full["synthetic"] is False, full)

    backend._vaeKind = "synthetic"                                    # noqa: SLF001
    soft = backend._generation_facts()                                # noqa: SLF001
    check("①⁶ VAE 是**合成**的 ⇒ 能出片但必须标 synthetic=True（参考 TE/VAE 未训练时不许改口）",
          soft["canGenerate"] is True and soft["synthetic"] is True, soft)

    backend._form = tb_mod.TorchBackend.H3_FORM_NAME                  # noqa: SLF001
    h3 = backend._generation_facts()                                  # noqa: SLF001
    check("①⁶ H3 形态**多要一件**音频 VAE（反向陷阱：拿它判单流族 ⇒ Z-Image 永远「缺一件」✓✗）",
          h3["canGenerate"] is False
          and any("音频 VAE" in part for part in h3["pendingParts"]), h3)


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

    # ⭐⭐ **``elapsedMs`` 由哪几段拼成** ✗✗（2026-11-08 补 ✗）：``elapsedMs`` 是真墙钟 ✓（含装载 ✓），
    #    而 ``result.totalMs`` 只是管线各阶段之和 ✓（**不含装载** ✗）—— 两个数躺在**同一份**回执里 ✓✗，
    #    差的那一截原先**没有任何一格解释** ✓（实测 H3：94.7 s vs 73.3 s，差的是装载 20 s ✓）。
    #    现在 ``timing`` 一次给出三格 ✓，``unaccountedMs`` 是剩下的排队与收尾 ✓。
    timing_task = runtime_mod.EngineTask(
        id="t-timing", stage="h3", request=_request(out_dir), status="done",
        createdAt=1_000, startedAt=1_000, finishedAt=6_000,
        loadMs=2_000, result={"totalMs": 2_800, "wallMs": 3_000})
    timing_block = timing_task.to_summary().get("timing") or {}
    check("⓪¹ ⭐⭐ ``timing`` 逐格给 ``elapsedMs``／``loadMs``／``pipelineMs``／``pipelineWallMs``／"
          "``unaccountedMs`` ✓（5 s 墙钟 ＝ 装载 2 s ＋ 管线墙钟 3 s ✓，未入账 0 ✓）"
          "—— 一个叫 ``totalMs`` 的数**不含装载** ✗✗，不点明就只能在回执里互相矛盾 ✓",
          timing_block.get("elapsedMs") == 5_000 and timing_block.get("loadMs") == 2_000
          and timing_block.get("pipelineMs") == 2_800
          and timing_block.get("pipelineWallMs") == 3_000
          and timing_block.get("unaccountedMs") == 0, timing_block)
    check("⓪² 干跑（**没装载**）⇒ ``loadMs=None`` ✓ 而不是编个 0 ✗（0 会被读成「装载瞬时完成」✓✗）",
          runtime_mod.EngineTask(id="t-dry", stage="h3", request=_request(out_dir))
          .to_summary().get("loadMs") is None, None)


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
    from app.services.engine import torch_backend as tb_mod  # noqa: PLC0415

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

        ⚠️ 也必须带上 `REAL_VAE_KINDS` ✓✗：`runtime` 判定 ``synthetic`` 时读的是
        ``TorchBackend.REAL_VAE_KINDS`` ✓，而这个名字在测试里**已经被换成替身** ✓ ⇒
        替身缺这个类属性时会 ``AttributeError`` 把整条判据炸掉 ✓（不是"判红"、是根本没跑到 ✓）。
        """

        REAL_VAE_KINDS = tb_mod.TorchBackend.REAL_VAE_KINDS

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

        # ④ `force=True` ＋ **同键** ⇒ 真的再装一遍 ✓✗（2026-11-09 兑现 ✓）。
        #    ⚠️ 这一半只能用假装配验（「装配函数被真调了一次」✓）；量的那两条账归量计那组 ✓ ——
        #    两组各验一半，谁也别替对方作证 ✓。
        made.clear()
        forced = engine_runtime.ensure_loaded(dit_path="假主干.safetensors", force=True)
        check("force: 同键 ＋ force=True ⇒ 后端被**重新构造一次**（reused=false、forced=true ✓✗）",
              len(made) == 1 and forced.get("reused") is False and forced.get("forced") is True,
              (len(made), forced.get("reused"), forced.get("forced")))
        check("force: 重装后装载账指向**新的那一份** ✓（旧的已丢、不是两份并存 ✓✗）",
              bool(made) and engine_runtime._backend is made[0],  # noqa: SLF001
              (len(made), engine_runtime._backend))  # noqa: SLF001
        again = engine_runtime.ensure_loaded(dit_path="假主干.safetensors")
        check("force: 紧接着**不传** force ⇒ 回到复用 ✓（reused=true、不再构造 ✗），"
              "且**不带** forced ✗✗（它只属那一次回执，写进缓存报告就会被读成「正在重装」✓✗）",
              again.get("reused") is True and len(made) == 1 and again.get("forced") is None,
              (again.get("reused"), len(made), again.get("forced")))
    finally:
        setattr(runtime_mod, "TorchBackend", real_backend_cls)
        setattr(runtime_mod, "_attach_video_vae", real_video_vae)          # noqa: SLF001
        setattr(runtime_mod, "_attach_reference_audio_vae", real_audio_vae)  # noqa: SLF001
        engine_runtime._backend = None                            # noqa: SLF001
        engine_runtime._loadKeyTuple = None                       # noqa: SLF001
        engine_runtime._loadReport = None                         # noqa: SLF001
        made.clear()


# ══════════════════════════════════════════════════════════════════════════
# ②c′ 装载账三格：**只许两个助手改** ✓✗（2026-10-31 收口 ✓）
# ══════════════════════════════════════════════════════════════════════════
#: 装载账的三格 ✓（`_backend` / `_loadKeyTuple` / `_loadReport`）—— 少一格就静默坏 ✓✗：
#: 漏 key ⇒ 每次调用**白装一遍**（Z-Image 40 秒／19 GiB）；漏 report ⇒ `/load` 回执空 ✓✗；
#: 清空时只清 `_backend` ⇒ 回执答的是**已经不存在的模型** ✓✗（SDXL 那条线的历史原样 ✓）。
_LEDGER_LINES = ("self._backend =", "self._loadKeyTuple =", "self._loadReport =")
_LEDGER_KEEPERS = frozenset({"__init__", "_reset_loaded", "_commit_load", "unload"})


def _method_bodies(src: str) -> dict[str, str]:
    """把类源码按 4 空格缩进的 ``def`` 切成 ``{方法名: 该方法的源码}`` ✓。"""
    import re as _re  # noqa: PLC0415

    bodies: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in src.splitlines():
        hit = _re.match(r"    def (\w+)\(", line)
        if hit:
            if name:
                bodies[name] = "\n".join(buf)
            name, buf = hit.group(1), [line]
        elif name is not None:
            buf.append(line)
    if name:
        bodies[name] = "\n".join(buf)
    return bodies


def _ledger_offenders(src: str) -> list[tuple[str, str]]:
    """哪些方法在**自己**改那三格 ✓✗（白名单见 :data:`_LEDGER_KEEPERS` ✓）。"""
    found: list[tuple[str, str]] = []
    for name, body in _method_bodies(src).items():
        if name in _LEDGER_KEEPERS:
            continue
        for line in body.splitlines():
            if any(key in line for key in _LEDGER_LINES):
                found.append((name, line.strip()))
    return found


def case_load_ledger() -> None:
    """⚠️ 装载账三格必须只能由 ``_reset_loaded`` / ``_commit_load`` 改 ✓✗。

    收口前这五行 ``with self._lock:`` 在**四条装配线**里逐字抄了四遍 ✓✗，且 SDXL 那条
    清空时**只抄了一格** ✓✗ —— 装失败之后回执还在讲上一个模型 ✓。它的坏处正是"不报错"：
    `reused` 判据同时看 `_backend is not None` ✓ ⇒ 假复用不会发生，所以这个错**永远不会自己暴露** ✓✗。
    """
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    # ① 自检：扫描器必须先**抓得住**历史那行坏写法，否则下面那条就是永绿的摆设 ✓✗
    bad_sample = (
        "class Fake:\n"
        "    def _ensure_image_loaded(self, device):\n"
        "        with self._lock:\n"
        "            previous = self._backend\n"
        "            self._backend = None\n"
        "        del previous\n"
        "        return previous\n"
        "    def _commit_load(self, key, backend, report):\n"
        "        with self._lock:\n"
        "            self._backend = backend\n"
        "            self._loadKeyTuple = key\n"
        "            self._loadReport = report\n"
    )
    sample_hits = _ledger_offenders(bad_sample)
    check("②c′ 自检: 扫描器抓得住「只清 `_backend`、漏清另两格」的历史写法 ✓✗，"
          "且**放行** `_commit_load` 里的三行 ✓（不验 ⇒ 这条判据可能永绿 ✓✗）",
          [name for name, _ in sample_hits] == ["_ensure_image_loaded"], sample_hits)

    # ② 真源码：四条装配线自己不许再写那三行 ✓
    src = inspect.getsource(runtime_mod.EngineRuntime)
    offenders = _ledger_offenders(src)
    check("②c′ 装载账: 装配线**不再各抄一遍**三格（只许 `_reset_loaded` / `_commit_load` 改 ✓✗）",
          offenders == [], offenders)

    # ③ **五条**装配线都要报 `replaced` ✓（SDXL 原来漏报 ⇒ 换装看不出来 ✓✗；
    #    2026-10-31 加 `qwen_image` 后 4 → 5 ✓ —— 这条判据**本来就该跟着条数走** ✓✗：
    #    它数的是"有多少条线记得报" ✓，写死 4 会让新增的那条线**漏报也过** ✓✗）。
    #    ⚠️ 认**代码形态** ``"replaced": had_previous`` ✓ 而不是裸键名 ✗：文档里也写着
    #    ``report["replaced"]`` ✓ ⇒ 数裸键名会得多一个、判据变成"6 才算对"那种莫名数字 ✓✗。
    assembly_lines = ("_ensure_loaded_internal", "_ensure_image_loaded", "_ensure_zimage_loaded",
                      "_ensure_qwen21_loaded", "_ensure_qwen_image_loaded",
                      "_ensure_krea2_loaded", "_ensure_seedvr2_loaded",
                      "_ensure_wan22_loaded", "_ensure_music3_loaded")
    replaced_hits = src.count('"replaced": had_previous')
    #: ⚠️ 这条**数字必须跟着装配线走** ✗✗（2026-11-01 加第六套 Krea2 时它当场红 ✓；
    #: 同日加第七套 SEEDVR2 又红了一次 ✓；2026-11-02 加第八套 Wan2.2 第三次红 ✓；
    #: 2026-11-06 加第九套 Music3 第四次红 ✓ —— 每加一条线都要来这里加名字 ✓）：
    #: 「新加一条线、忘了报 `replaced`」的症状是**换装看不见** ✓（看着像"内存莫名不够" ✓✗）。
    check("②c′ 报告: **每条**装配线都报 `replaced` ✓（漏一条 ⇒ 那条线的换装看不见 ✓✗）",
          replaced_hits == len(assembly_lines), replaced_hits)

    # ③b 缺格收集也一样 ✓✗：两条装配线各抄过 23 行「missing.append + if missing: raise」✓，
    #     而这段的**纪律**是"首行只列真缺的那几格" ✓（测试也只读首行 ✓）⇒ 必须只有一份实现 ✓
    bodies = _method_bodies(src)
    slot_users = ("_ensure_zimage_loaded", "_ensure_qwen21_loaded", "_ensure_qwen_image_loaded",
                  "_ensure_krea2_loaded")
    slot_copies = [(name, line.strip())
                   for name in slot_users
                   for line in bodies.get(name, "").splitlines() if "missing.append(" in line]
    check("②c′ 缺格: **四条**装配线不再各抄一遍「收集 + 首行只列真缺」（只许 `_require_slots` ✓✗）",
          slot_copies == [], slot_copies)
    slot_callers = [name for name in slot_users if "_require_slots(" in bodies.get(name, "")]
    check("②c′ 缺格: **四条**装配线都真调 `_require_slots` ✓"
          "（上一条只查'不再手抄' ⇒ 把这段整段删掉也能过 ✓✗）",
          slot_callers == list(slot_users), slot_callers)
    check("②c′ 覆盖面: 这份测试点名的装配线**与实现里的条数一致** ✓✗"
          "（漏一条 ⇒ 新增那条线的 `replaced`/缺格/收尾三条约束全都没被查 ✓✗）",
          all(f"def {name}(" in src for name in assembly_lines), assembly_lines)

    # ③c 失败收尾也一样 ✓✗：`del backend` + 还缓存 + **保住 reason** 三件事 ✓
    #     其中 SDXL 那条原来只有一段 `except BaseException` ✓✗ ⇒ 内部的 reason="deps" 被重包成
    #     "pending" ✓（缺依赖被报成"权重没就绪" ⇒ 照文档去补权重白忙 ✓✗，比不报更贵 ✓）
    del_copies = [(name, line.strip()) for name, body in bodies.items()
                  for line in body.splitlines() if line.strip() == "del backend"]
    check("②c′ 失败收尾: `del backend` 只许出现在 `_abandon_assembly` 里"
          "（装配线不再各抄一遍 ✓✗；漏这一句 ⇒ 半装的后端还挂在 traceback 上 ✓）",
          del_copies == [], del_copies)

    try:
        runtime_mod._abandon_assembly(  # noqa: SLF001
            object(), TorchBackendUnavailable("缺 torch", reason="deps"))
    except TorchBackendUnavailable as err:
        raised: Any = err
    else:
        raised = None
    check('②c′ 失败收尾: `reason="deps"` **原样**传出 ✓✗'
          "（SDXL 原来重包成 pending ⇒ 用户去补权重、而真正缺的是依赖 ✓✗）",
          raised is not None and getattr(raised, "reason", None) == "deps"
          and "缺 torch" in str(raised), getattr(raised, "reason", None))

    try:
        runtime_mod._abandon_assembly(object(), ValueError("boom"), what="SDXL")  # noqa: SLF001
    except TorchBackendUnavailable as err:
        raised = err
    else:
        raised = None
    check("②c′ 失败收尾: 普通异常 ⇒ 「SDXL 装配失败：ValueError: boom」＋ reason=pending ✓"
          "（前缀丢了 ⇒ 四条线的报错分不出是谁装的 ✓✗）",
          raised is not None and getattr(raised, "reason", None) == "pending"
          and str(raised) == "SDXL 装配失败：ValueError: boom", str(raised))

    # ④ 行为：复位真清三格、提交真写三格 ✓（不依赖 torch ✓）
    rt = runtime_mod.EngineRuntime()
    rt._backend = "OLD"                      # noqa: SLF001 —— 白盒：就为验"清了几格" ✓
    rt._loadKeyTuple = ("old",)              # noqa: SLF001
    rt._loadReport = {"stage": "old"}         # noqa: SLF001
    had = rt._reset_loaded()
    check("②c′ 复位: 三格**一起**清 且答出「原来有装」✓",
          had is True and rt._backend is None and rt._loadKeyTuple is None
          and rt._loadReport is None,
          {"had": had, "backend": rt._backend, "key": rt._loadKeyTuple,
           "report": rt._loadReport})
    check("②c′ 复位: 空转时答 False ✓（否则 `replaced` 会对着空气报 true ✓✗）",
          rt._reset_loaded() is False)

    rt._commit_load(("k",), "NEW", {"stage": "new"})  # noqa: SLF001
    check("②c′ 提交: 一次写满三格 ✓（少一格 ⇒ 白装一遍或回执空 ✓✗）",
          rt._backend == "NEW" and rt._loadKeyTuple == ("k",)
          and rt._loadReport == {"stage": "new"},
          {"backend": rt._backend, "key": rt._loadKeyTuple, "report": rt._loadReport})
    rt._reset_loaded()                       # noqa: SLF001 —— 别把假账留给后面的用例 ✓


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

    def _normalizers() -> dict[str, Any]:
        """`ensure_loaded` 进键前**真的**对哪几项做了归一 ✓ —— 从它自己的源码里读 ✓（不手抄 ✗）。

        ⚠️ 为什么不能按「参数名 ＋ `_key`」硬拼 ✗✗：`loras` 的归一函数叫 `_lora_key` ✓
        （没有那个 s ✓），硬拼 `_loras_key` 会**静默**取不到 ⇒ 这一项就漏归一 ✓✗。
        漏了不会报错 ✓，只会让本文件摆的"同键"和真实键**不同形** ⇒ 假 miss ⇒ 真去装权重 ✓✗
        （2026-11-02 实况：`vae_tiling` 漏归一后，这条用例报的是「DiT 权重未到位」✓✗
        —— 看起来完全像环境问题，其实是测试本身在算命 ✓）。

        ⚠️ 读源码而不是再抄一张表 ✗✗：抄表的话，下次 `ensure_loaded` 里新加一项归一，
        这里不跟着改又是同一个坑 ✓✗（而它只在**恰好**摆白盒状态时才露头 ✓）。
        源码里没写的归一这里也照实不做 ✓ —— 那才是 `ensure_loaded` 自己的问题 ✓，
        不该由测试替它补齐 ✓（补齐等于把缺陷抹平 ✓✗）。
        """
        import ast  # noqa: PLC0415

        source = inspect.getsource(engine_runtime.ensure_loaded)
        table: dict[str, Any] = {}
        for node in ast.walk(ast.parse(textwrap.dedent(source))):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "_load_key":
                continue
            for keyword in node.keywords:
                value = keyword.value
                if (keyword.arg and isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name) and value.func.id.endswith("_key")):
                    coerce = getattr(runtime_mod, value.func.id, None)
                    if callable(coerce):
                        table[keyword.arg] = coerce
        return table

    def _key(dit_path: str) -> tuple[Any, ...]:
        """与 `ensure_loaded` **同一份**键 ✓ —— 默认值**现取** ✓、归一**照源码** ✓。

        ⚠️ 归一必须跟着做 ✗✗：`ensure_loaded` 进键前会把 `loras` / `vae_tiling` 这类
        「多种写法同一个意思」的项先过一层快照函数 ✓（`None` 与 `{}`、`512` 与 `"512"` 都归一 ✓）。
        只搬签名默认值、不过那一层 ⇒ 键的**形**不一样 ⇒ 同键被算成"换了装配" ✓✗。
        """
        params = inspect.signature(engine_runtime.ensure_loaded).parameters
        kwargs = {name: param.default for name, param in params.items()
                  if name not in ("self", "force") and param.default is not inspect.Parameter.empty}
        for name, coerce in _normalizers().items():
            if name in kwargs:
                kwargs[name] = coerce(kwargs[name])
        return runtime_mod._load_key(**{**kwargs, "dit_path": dit_path})

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

        # ② `force=True` ＋ 键**相同** ⇒ **真重装一次** ✓✗（2026-11-09 改判 ✓）。
        #    旧口径是「内部当场复用 ⇒ 只记 hit」，理由是「记 miss+eviction 会是两条假账」✓ ——
        #    那个理由在当时成立（当时确实没真装 ✗），但它让 `force` 这个参数名**纯装饰** ✓✗：
        #    调用方显式要求重装，实际什么都没发生、还回 `reused=True` ✓✗。现在内部那道同键短路
        #    被 `force_reload` 跳过 ⇒ 旧的**真丢了**、新的**真装了一遍** ⇒ 这两条账**不再是假账** ✓。
        _fresh()
        engine_runtime._backend = _FakeBackend()                                # noqa: SLF001
        engine_runtime._loadKeyTuple = _key("D:\\m\\dit.safetensors")           # noqa: SLF001
        engine_runtime._loadReport = {"components": ["dit"], "device": "cpu"}    # noqa: SLF001
        try:
            engine_runtime.ensure_loaded(dit_path="D:\\m\\dit.safetensors", force=True)
        except TorchBackendUnavailable:
            pass                      # 真去装了 ⇒ 这台机器上没有真权重 ⇒ 抛才对 ✓（装不上正是证据 ✓）
        except Exception as err:  # noqa: BLE001
            check("量计: force＋同键真重装时抛的必须是 TorchBackendUnavailable", False, repr(err))
        forced_facts = engine_runtime.status()["cache"]
        check("量计: force=True ＋ 键相同 ⇒ **真重装**（miss / eviction 各 1、hits 0 ✓✗）",
              (forced_facts["hits"], forced_facts["misses"],
               forced_facts["evictions"]) == (0, 1, 1),
              (forced_facts["hits"], forced_facts["misses"], forced_facts["evictions"]))
        forced_recent = forced_facts["recent"] or []
        check("量计: 那两笔的**理由**点名 force ✓（别让下一个人读成「换装配」✓✗）",
              [e.get("kind") for e in forced_recent] == ["miss", "eviction"]
              and all("force" in str(e.get("reason")) for e in forced_recent),
              [e.get("reason") for e in forced_recent])
        check("量计: 重装失败后旧后端**已丢**（不半装 ✗、也不留着旧的骗人 ✓）",
              engine_runtime._backend is None, engine_runtime._backend)  # noqa: SLF001

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
        #: ⚠️ 这里必须**自己摆一份旧的** ✗✗：② 改成「force 真重装」后会顺手把 `_backend` 清成 None ✓，
        #: 不摆的话「evictions +1」这条会红得莫名其妙 ✓（而它本来要验的正是"装新的之前先丢旧的"✓）。
        engine_runtime._backend = _FakeBackend()                                # noqa: SLF001
        engine_runtime._loadKeyTuple = _key("D:\\m\\旧的那份.safetensors")      # noqa: SLF001
        engine_runtime._loadReport = {"components": ["dit"], "device": "cpu"}    # noqa: SLF001
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
        #: ⚠️ `freeGiB` 是**整张卡**的读数 ✓✗（含别的进程 ✓）⇒ 严格「一点不降」在多进程机器上
        #: 本来就站不住 ✓✗（2026-10-31 实测：本进程 `memory_allocated` 前后完全一致 ✓、
        #: `reserved` 也没涨 ✓，但 free 被外部占了 10 MiB ⇒ 原式判负 ✗）。
        #: 能钉的是两条 ✓：**reserved 不涨** ✓ ＋ **free 抖动在 256 MiB 内** ✓；
        #: 「一点没动」由下面那条 `memory_allocated` 前后相等来钉 ✓✗（那才是本进程自己的账 ✓）。
        FREE_JITTER_GIB = 0.25
        check("回收: 还完的账**只减不增**（reserved 不涨 / free 不降，容外部进程 256 MiB 抖动）",
              float(report.get("reservedGiB") or 0) <= float(before["reservedGiB"]) + 1e-6
              and float(report.get("freeGiB") or 0) >= float(before["freeGiB"]) - FREE_JITTER_GIB,
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


def case_vae_tiling_wiring() -> None:
    """VAE 解码旋钮（``vaeTiling``）**接线**是不是每一条装配线都拿到了 ✓✗（纯静态判 ✓，不装 torch ✓）。

    ⚠️ 为什么这条要独立成一条 ✗✗：装配白名单那条守卫（路由契约 ⑪′）只保证
    ``ensure_loaded`` 的**签名**有这一项 ✓ —— 而真正把它送到后端的是**八处**手写调用 ✓
    （七条 ``_ensure_*_loaded`` ＋ ``_ensure_loaded_internal`` 里 H3 那条 ✓）。少写一处
    ⇒ 那一档「配了没生效」✓✗，症状是同一个开关在某些档位管用、在另一些档位静默无效 ✓✗
    （``int8_offload`` 2026-10-29 就是这么丢的 ✓），而且**任何**运行期测试都撞不到 ✓✗
    （它们不会去验"每个档位的构造器都收了它"✓）。
    """
    import ast  # noqa: PLC0415

    from app.services.engine import runtime as rt_mod  # noqa: PLC0415

    path = Path(rt_mod.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "EngineRuntime")
    funcs = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    loaders = {name: node for name, node in funcs.items()
               if name == "_ensure_loaded_internal"
               or (name.startswith("_ensure_") and name.endswith("_loaded"))}
    no_param: list[str] = []
    no_pass: list[str] = []
    for name, node in sorted(loaders.items()):
        args = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        if "vae_tiling" not in args:
            no_param.append(name)
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", "") != "TorchBackend":
                continue          #: 出图那条走 ``SdxlBackend`` ✓ —— 它没有这两个旋钮 ✓（另有声明 ✓）
            if not any(kw.arg == "vae_tiling" for kw in call.keywords):
                no_pass.append(name)
                break
    check("旋钮接线: 每条装配线**都收** `vae_tiling` 且**都传给** `TorchBackend(...)`"
          "（少一处 ⇒ 那一档「配了没生效」，运行期测试撞不到）",
          not no_param and not no_pass, {"少参数": no_param, "收了没传": no_pass})

    internal = loaders.get("_ensure_loaded_internal")
    dispatch = [call for call in (ast.walk(internal) if internal is not None else [])
                if isinstance(call, ast.Call)
                and getattr(call.func, "attr", "").startswith("_ensure_")
                and getattr(call.func, "attr", "").endswith("_loaded")]
    missing = sorted({getattr(call.func, "attr", "?") for call in dispatch
                      if not any(kw.arg == "vae_tiling" for kw in call.keywords)})
    check("旋钮分派: `_ensure_loaded_internal` 往七条装配线**每一处**都带上了它"
          "（签名改了、调用点没改 ⇒ 默认 None ⇒ 又是「配了没生效」）",
          bool(dispatch) and not missing, {"分派数": len(dispatch), "漏": missing})

    key_calls = [call for call in ast.walk(cls) if isinstance(call, ast.Call)
                 and getattr(call.func, "id", "") == "_load_key"]
    key_missing = [i for i, call in enumerate(key_calls)
                   if not any(kw.arg == "vae_tiling" for kw in call.keywords)]
    check("旋钮进装载键: **两处** `_load_key` 都带 `vae_tiling`"
          "（只改一处 ⇒ 一处判「已装」、另一处判「要重装」⇒ 抖动 ✓✗）",
          len(key_calls) == 2 and not key_missing,
          {"调用点": len(key_calls), "漏": key_missing})

    commit = funcs.get("_commit_load")
    commit_src = ast.get_source_segment(source, commit) if commit is not None else ""
    check("旋钮回执: `_commit_load` 用 `vae_tiling()` 回吐实值 ✓"
          "（挂在所有装配线共用的人口上 ✓；各线自己写会漏 ✗）",
          "vaeTiling" in (commit_src or "") and "vae_tiling" in (commit_src or ""),
          (commit_src or "")[:200])

    #: ⭐ **`temporalChunk` 的最后一公里**（2026-11-02 晚 ✓）：空间档走 `vae_wan` 自己的分块器 ✓，
    #: 时间档却是后端在 SEEDVR2 那条解码分支上**显式**取出、传给 `decode_latents` ✓ ——
    #: 这一行漏了，开关就是「收下了、存住了、从没用过」✓✗（运行期同样撞不到 ✓，症状＝
    #: 回执里 `temporalChunk` 明明有值、解码却还整段跑 ✓✗）。
    tb_src = (path.parent / "torch_backend.py").read_text(encoding="utf-8")
    tb_tree = ast.parse(tb_src)
    decode_fn = next((node for node in ast.walk(tb_tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "decode"), None)
    chunk_calls = [call for call in (ast.walk(decode_fn) if decode_fn is not None else [])
                   if isinstance(call, ast.Call)
                   and getattr(call.func, "attr", "") == "decode_latents"
                   #: 只看 ``vae_seedvr2`` 那一族 ✓（flux / wan21 的 ``decode_latents`` 是**同名
                   #: 不同模块** ✓，它们没有时间分块这条旋钮 ✓ —— 一律要求带参就是在冤枉它们 ✗）。
                   and getattr(call.func.value, "id", "") == "vae_seedvr2_mod"]
    wired = [any(kw.arg == "temporal_chunk" for kw in call.keywords) for call in chunk_calls]
    #: 读旋钮**两种写法都算** ✓✗：``self._vaeTiling.get("temporalChunk")`` 与
    #: ``_knobs = dict(self._vaeTiling); _knobs["temporalChunk"]`` ✓ —— 只认前一种就是把
    #: 「先取一份快照再逐项读」这种**更稳**的写法判红 ✗（实测就是这么误红过一次 ✓）。
    #: 三种**等价**写法都要认 ✓✗：``self._vaeTiling.get("x")`` ／ ``self._vaeTiling["x"]`` ／
    #: 先 ``_knobs = dict(self._vaeTiling)`` 再 ``_knobs["x"]`` ✓ —— 只认第一种会把「取一份
    #: 快照再逐项读」这种**更稳**的写法判红 ✗（实测就是这么误红过一次 ✓✗）。
    snapshots = {
        target.id
        for node in (ast.walk(decode_fn) if decode_fn is not None else [])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "dict"
        and any(getattr(arg, "attr", "") == "_vaeTiling" for arg in node.value.args)
        for target in node.targets if isinstance(target, ast.Name)
    }
    read_knob: list = []
    for node in (ast.walk(decode_fn) if decode_fn is not None else []):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "get"
                and getattr(node.func.value, "attr", "") == "_vaeTiling"
                and any(getattr(arg, "value", None) == "temporalChunk" for arg in node.args)):
            read_knob.append(node)
        elif (isinstance(node, ast.Subscript)
              and (getattr(node.value, "attr", "") == "_vaeTiling"
                   or getattr(node.value, "id", "") in snapshots)
              and getattr(node.slice, "value", None) == "temporalChunk"):
            read_knob.append(node)
    check("旋钮落地: `decode` 的 SEEDVR2 分支**读** `_vaeTiling[\"temporalChunk\"]`，"
          "并**真的传给** `decode_latents(...)`（漏这一行 ⇒ 旋钮进得了键、进不了解码器）",
          bool(chunk_calls) and all(wired) and bool(read_knob),
          {"调用数": len(chunk_calls), "带参": wired, "读旋钮": len(read_knob)})


def case_gpu_layers_wiring() -> None:
    """``gpuLayers`` 这条旋钮的**接线**与 `vae_tiling` 同形 ✓✗（2026-11-07 加 ✓，§4.1 ✓）。

    ⚠️ 为什么必须静态判（不真装）✗✗：它坏起来**一声不响** ✓ —— 少写一处 `gpu_layers=gpu_layers`
    ⇒ 那一档收下参数、构造后端时不传 ⇒ 后端 `_gpuLayers=None` ⇒ **按预算自算** ✓✗，
    而回执里 `requested` 明明有你给的那个数 ✓ ⇒ 症状是「回执说切了、实际没切」✓✗。
    真跑一遍也撞不到（除非恰好去看层数 ✓）。

    判据五件：

      ① 每条走 `TorchBackend` 的装配线**都收** `gpu_layers` 且**都传给**构造器 ✓；
      ② 唯一的例外是出图那条（`SdxlBackend` 没有可切的层 ✓）—— 必须**恰好**只有它 ✗✗
         （多一条就是在某个档位静默丢参 ✓）；
      ③ `_ensure_loaded_internal` 往每一条装配线**都带上了**它 ✓；
      ④ **两处** `_load_key` 都在键里 ✓（只改一处 ⇒ 一处判「已装」、另一处判「要重装」⇒ 抖动 ✓）；
      ⑤ `ensure_loaded` 里**先归一、后拼键** ✓✗（顺序反了 ⇒ `"3"` 与 `3` 是两个键 ✓，
         同一份布局白装两遍 ✓）。
    """
    import ast  # noqa: PLC0415

    from app.services.engine import runtime as rt_mod  # noqa: PLC0415

    path = Path(rt_mod.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "EngineRuntime")
    funcs = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    loaders = {name: node for name, node in funcs.items()
               if name == "_ensure_loaded_internal"
               or (name.startswith("_ensure_") and name.endswith("_loaded"))}
    #: ① 走 `TorchBackend` 的那几条：收了就必须传 ✓
    no_param: list[str] = []
    no_pass: list[str] = []
    for name, node in sorted(loaders.items()):
        args = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        ctors = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                 and getattr(call.func, "id", "") == "TorchBackend"]
        if not ctors:
            continue                       #: 没构造 TorchBackend 的（出图那条）② 单独判 ✓
        if "gpu_layers" not in args:
            no_param.append(name)
            continue
        if not all(any(kw.arg == "gpu_layers" for kw in call.keywords) for call in ctors):
            no_pass.append(name)
    check("gpuLayers 接线: 每条走 TorchBackend 的装配线**都收** `gpu_layers` 且**都传给**构造器"
          "（少一处 ⇒ 回执说切了、实际按 auto ✓✗）",
          not no_param and not no_pass, {"少参数": no_param, "收了没传": no_pass})

    #: ② 例外必须**只有**出图那条，而且是 `SdxlBackend` ✓
    no_ctor = sorted(name for name, node in loaders.items()
                     if not any(isinstance(call, ast.Call)
                                and getattr(call.func, "id", "") == "TorchBackend"
                                for call in ast.walk(node)))
    sdxl = loaders.get("_ensure_image_loaded")
    sdxl_src = ast.get_source_segment(source, sdxl) if sdxl is not None else ""
    check("gpuLayers 例外只有出图那条（SdxlBackend 没有可切的层）",
          no_ctor == ["_ensure_image_loaded"] and "SdxlBackend" in (sdxl_src or ""),
          {"无 TorchBackend 的": no_ctor})

    #: ③ 分派：每一处都要带 ✓
    internal = loaders.get("_ensure_loaded_internal")
    dispatch = [call for call in (ast.walk(internal) if internal is not None else [])
                if isinstance(call, ast.Call)
                and getattr(call.func, "attr", "").startswith("_ensure_")
                and getattr(call.func, "attr", "").endswith("_loaded")]
    missing = sorted({getattr(call.func, "attr", "?") for call in dispatch
                      if not any(kw.arg == "gpu_layers" for kw in call.keywords)})
    #: ⚠️ 例外**只有**出图那条（八处里少一处）✗✗ —— 若哪天出图也接上 residency 切分，
    #: 这条会先红 ✓：那时把 `["_ensure_image_loaded"]` 改成 `[]` 就行 ✓（改之前先想清楚
    #: `SdxlBackend` 有没有「层」这个概念 ✗）。
    check("gpuLayers 分派: 八处里**只有**出图那条不带它（其余七条都带）",
          len(dispatch) == 8 and missing == ["_ensure_image_loaded"],
          {"分派数": len(dispatch), "漏": missing})

    #: ④ 两处 `_load_key` 都要在键里 ✓
    key_calls = [call for call in ast.walk(cls) if isinstance(call, ast.Call)
                 and getattr(call.func, "id", "") == "_load_key"]
    key_missing = [i for i, call in enumerate(key_calls)
                   if not any(kw.arg == "gpu_layers" for kw in call.keywords)]
    check("gpuLayers 进装载键: **两处** `_load_key` 都带 `gpu_layers`"
          "（只改一处 ⇒ 抖动 ✓✗）",
          len(key_calls) == 2 and not key_missing,
          {"调用点": len(key_calls), "漏": key_missing})

    #: ⑤ 先归一、后拼键 —— 靠**行号**判 ✓✗（顺序反了就是两个键 ✓，纯静态也能判 ✓）
    ensure = funcs.get("ensure_loaded")
    normalize_at: list[int] = []
    key_at: list[int] = []
    for node in (ast.walk(ensure) if ensure is not None else []):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
        if name == "normalize_gpu_layers":
            normalize_at.append(node.lineno)
        if name == "_load_key":
            key_at.append(node.lineno)
    check("gpuLayers **先归一、后拼键**（顺序反了 ⇒ '3' 与 3 是两个键、同一布局白装两遍 ✓✗）",
          bool(normalize_at) and bool(key_at) and min(normalize_at) < min(key_at),
          {"归一": normalize_at, "拼键": key_at})

    #: ⑥ 出图那条给了数要**如实报** granted=null（不许静默吞掉 ✗、不许报 0 ✗）
    check("gpuLayers 出图那条：给了数时回执报 `granted=None`（**不是** 0 —— 0 是「一层都不上卡」）",
          '"granted": None' in (source or "") and "没有可切的层" in (source or ""), None)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="engine_runtime_"))
    case_status()
    case_generation_facts()
    case_queue(tmp)
    case_terminal_order(tmp)
    case_missing_weights(tmp)
    case_replace_order()
    case_half_load()
    case_load_ledger()
    case_cache_meter()
    case_vae_tiling_wiring()
    case_gpu_layers_wiring()
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
