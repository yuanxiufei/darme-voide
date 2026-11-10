"""Wan 2.2 **I2V** 接线自检 ✓：plan 口径 → 36 通道硬约束 → 双专家 σ 换装 → 装配四格 ✓。

这一族与前面七条线最大的不同不是「更难」，而是**三处都是错的、且全都不报错** ✗✗：

① **plan 少一帧就白跑** ✓ —— 16ch VAE 的时间压缩是 **4 倍** ✓ ⇒ 像素帧必须 ``4n+1`` ✓；
   写 80 帧会在解码时丢掉最后几帧的残差 ✓（报错指向形状 ✓、指不到「帧数算错」✓）；
② **入口通道差一位就出「与首帧无关的片子」** ✓ —— I2V 主干吃 **36 = 噪声 16 ＋ 掩码 4 ＋
   条件图 16** ✓；盘上同族还有 T2V 那份（**16** ✓）。装 T2V 跑 I2V 会当场炸 ✓（还好 ✓）；
   真危险的是**反过来**：装上 I2V 却只喂 16、或零填充凑 36 ⇒ 形状对 ✓、画面与首帧无关 ✓、
   **一条报错都没有** ✓✗；
③ **双专家分界早/晚一步没人看得出来** ✓ —— 官方是两个 ``KSamplerAdvanced`` 各 4 步 ✓、
   切换在 **50%** ✓；判据必须是 **σ 阈值**（不是「数第几次 ``denoise``」✓：引导 2 次、
   heun 再翻倍 ✓✗），且边界要写 ``<=`` ✓（写 ``<`` ⇒ 4 步退化成 3/1 ✓，只影响后段细节 ✓）。

⚠️ 本节**不装真权重** ✓：判据都在 plan / 纯函数 / 报错文本 / 源码结构上 ✓
（真权重那一段由 ``engine_dit_wan22_test`` 与 ``engine_vae_wan_test`` 各管一半 ✓）。
"""

import ast
import inspect
import sys
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # cp936 控制台下首行 ✓ 直接 UnicodeEncodeError ✓

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import dit_build as db  # noqa: E402
from app.services.engine import dit_wan22 as dw  # noqa: E402
from app.services.engine import inventory  # noqa: E402
from app.services.engine import pipeline as pl  # noqa: E402
from app.services.engine import registry  # noqa: E402
from app.services.engine import runtime as rt  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402
from app.services.engine import vae_wan as vw  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}　{detail}")


def _err(fn) -> str:
    """跑一下，把异常压成一行 ✓（判据只看**说了什么** ✓）。"""
    try:
        fn()
    except BaseException as err:  # noqa: BLE001 —— 这里就是要看所有失败形态 ✓
        return f"{type(err).__name__}: {err}"
    return ""


def case_plan() -> None:
    print("① plan：I2V 必带首帧 ＋ 尺寸/帧数/fps/步数吸附 ✓")

    rejected = _err(lambda: pl.build_plan(pl.GenerationRequest(
        prompt="雨夜街头", stage=pl.WAN22_STAGE)))
    check("⭐ 没给 `firstFrame` ⇒ **plan 期就拒** ✗✗（绝不退化成文生视频 ✓）",
          "firstFrame" in rejected and "36" in rejected, rejected)
    check("拒信里**指明改走哪条线** ✓（纯文本 ⇒ `stage=` 后接 h3 ✓ —— 只报「缺参数」没用 ✓✗）",
          '"h3"' in rejected, rejected)

    req = pl.GenerationRequest(prompt="雨夜街头", stage=pl.WAN22_STAGE,
                               first_frame="假首帧.png", seconds=1.5, ratio="16:9")
    plan = pl.build_plan(req)
    check("尺寸吸附到 **16** 网格 ✓（VAE 8× × patch 2× ✓；不整除 ⇒ `_PatchIn` 当场抛 ✓）",
          plan.width % pl.WAN22_DIM_GRID == 0 and plan.height % pl.WAN22_DIM_GRID == 0,
          (plan.width, plan.height))
    check("帧数吸附到 ``4k+1`` ✓（1.5s × 16fps ⇒ 24 ⇒ **25** 帧 ✓）且**向上**吸 ✓",
          plan.frames == 25 and plan.frames % pl.WAN22_FRAME_GRID == 1, plan.frames)
    check("潜帧 =(25−1)/4+1 = **7** ✓（4 倍时间压缩 ✓，写成 25 会在解码炸 ✓）",
          plan.latent_frames == 7, plan.latent_frames)
    check("默认 `fps=24`（别的族的口径 ✗）⇒ 换 **16** ✓ 并**说出来** ✓",
          plan.fps == pl.WAN22_FPS
          and any("24fps" in w for w in plan.warnings), (plan.fps, plan.warnings[:2]))
    check("默认 `steps=30`（别的族的口径 ✗）⇒ 换官方 **4** 步 ✓ 并**说出来** ✓",
          plan.steps == pl.WAN22_OFFICIAL_STEPS
          and any("30" in w for w in plan.warnings), plan.steps)
    check("双专家分界 = ``round(4 × 0.5)`` = **2** ✓（`wanHighSteps` ✓）",
          plan.wan_high_steps == 2, plan.wan_high_steps)
    check("σ 是 flow 梯子 ✓：**shift=5.0**（不是 H3 的 12 / Krea2 的 μ1.15 ✗）、"
          "5 个端点、首值 **1.0**、末值 **0** ✓",
          len(plan.sigmas) == plan.steps + 1
          and abs(plan.sigmas[0] - 1.0) < 1e-6 and abs(plan.sigmas[-1]) < 1e-9,
          plan.sigmas)
    check("`timesteps` 与 σ **逐位相同** ✓（flow 域里 timestep 就是 σ ✓ 不乘 1000 ✗）",
          [float(v) for v in plan.timesteps] == [float(v) for v in plan.sigmas])
    check("plan 回执里带 ``wanHighSteps`` ✓（默认 0 ✓ —— 其余七条线不受影响 ✓）",
          plan.to_dict().get("wanHighSteps") == 2)

    # ── 显式给的帧率/步数：**保留**，只警告 ✓（不静默改写 ✓）──
    plan2 = pl.build_plan(pl.GenerationRequest(
        prompt="p", stage=pl.WAN22_STAGE, first_frame="假首帧.png",
        seconds=5.0, fps=24, steps=8))
    check("⚠️ 收到 `fps=24` ⇒ 仍是 **16** ✓✗（24 恰是默认值 ✓，**分不出**「没写」与「写了 24」✓）"
          "且**不谎称**显式 24 会被保留 ✓ —— 警告里写明这一档给不了 24fps ✓",
          plan2.fps == pl.WAN22_FPS
          and any("给不了" in w and "24" in w for w in plan2.warnings),
          (plan2.fps, plan2.warnings[:3]))
    check("显式 `steps=8` ⇒ **保留 8** ✓ 只警告 ✓；`wanHighSteps` 跟着算 **4** ✓"
          "（分界跟着步数走 ✓，不写死 2 ✓）",
          plan2.steps == 8 and plan2.wan_high_steps == 4, (plan2.steps, plan2.wan_high_steps))
    check("5.0s × **16fps** = 80 ⇒ 吸到 **81** 帧 ✓（`4k+1` ✓）、潜帧 **21** ✓"
          "（按 24fps 算会得 121/31 ✓✗ —— 帧率变了帧数必须跟着变 ✓）",
          plan2.frames == 81 and plan2.latent_frames == 21,
          (plan2.frames, plan2.latent_frames))

    # ── 本阶段不认的输入：**说清**（忽略 ✓）或**当场拒**（互斥 ✗），绝不含糊 ✓ ──
    plan3 = pl.build_plan(pl.GenerationRequest(
        prompt="p", stage=pl.WAN22_STAGE, first_frame="假首帧.png",
        last_frame="假尾帧.png", reference_frames=("假参考.png",)))
    joined = "\n".join(plan3.warnings)
    check("⭐ `lastFrame` / `referenceFrames` **明确说被忽略** ✓（静默丢掉＝用户以为生效了 ✓✗）",
          "lastFrame" in joined and "referenceFrames" in joined, joined[-400:])

    rejected = _err(lambda: pl.build_plan(pl.GenerationRequest(
        prompt="p", stage=pl.WAN22_STAGE, first_frame="假首帧.png",
        init_image="假起点.png")))
    check("⭐ `initImage` ⇒ **当场拒** ✗✗（不是静默忽略 ✓✗）—— 它与 `firstFrame` **互斥** ✓："
          "前者是「编码后按 σ0 掺一次噪声当起点」✓、后者是 36 通道逐步条件 ✓；"
          "静默丢掉 ⇒ 用户以为起点是他那张图 ✓，实际是纯噪声 ✓",
          "initImage" in rejected and "firstFrame" in rejected and "不收" in rejected,
          rejected)

    rejected = _err(lambda: pl.build_plan(pl.GenerationRequest(
        prompt="p", stage=pl.WAN22_STAGE, first_frame="假首帧.png",
        init_image="假起点.png", denoise=0.5)))
    check("⭐ ``denoise<1`` ⇒ **当场拒** ✗✗（I2V 的起点是首帧条件 ✓，"
          "不是从参考 latent 起跑 ✓；两者语义不同 ✓）",
          "denoise" in rejected and "Wan 2.2 I2V" in rejected, rejected)

    unsupported = _err(lambda: pl.build_plan(pl.GenerationRequest(
        prompt="p", stage="wan21")))
    check("没认过的 stage ⇒ 拒信里**列出 wan22** ✓（新档要能被发现 ✓✗）",
          "wan22" in unsupported, unsupported)


def case_expert_switch() -> None:
    print("② 双专家：**σ 阈值**换装 ✓（不是数调用次数 ✗）、边界 ``<=`` ✓、跨 run 复位 ✓")

    def stub(sigmas: list, high_steps: int):
        backend = tb.TorchBackend.__new__(tb.TorchBackend)
        backend._wanExperts = {"high": "高噪.safetensors", "low": "低噪.safetensors"}
        backend._wanActive = "high"
        backend._wanSwitchSigma = sigmas[high_steps]
        backend._wanHighSteps = high_steps
        calls = []

        def fake(self, want):           # noqa: ANN001 —— 替身 ✓
            calls.append(want)
            self._wanActive = want
            return {"expert": want}

        backend._wan_switch_expert = types.MethodType(fake, backend)
        return backend, calls

    sig = [1.0, 0.5, 0.25, 0.1, 0.0]     # 四步 flow 梯子的形状 ✓（首值 1 ✓ 末值 0 ✓）
    backend, calls = stub(sig, 2)
    backend._wan_expert_tick(1.0)
    backend._wan_expert_tick(0.5)
    check("分界之上（σ=1.0 / 0.5）⇒ **不换** ✓", calls == [], calls)
    backend._wan_expert_tick(0.25)
    check("⭐ σ **恰好等于**阈值（``sigmas[wanHighSteps]`` ✓）⇒ **就该换了** ✓"
          "（严格 ``<`` 会让 4 步退化成 3/1 ✓✗，而官方是 2/2 ✓）",
          calls == ["low"], calls)
    backend._wan_expert_tick(0.1)
    backend._wan_expert_tick(0.0)
    check("换过就**不回头** ✓（σ 单调 ✓，不必再判 ✓）", calls == ["low"], calls)

    # ── 同一步的正负两趟：同一 σ ⇒ 同一专家 ✓（这才是「按 σ 而不是按次数」的理由 ✓）──
    backend2, calls2 = stub(sig, 2)
    for _ in range(2):                   # 引导：正向一趟 ＋ 负向一趟 ✓
        backend2._wan_expert_tick(0.25)
    check("同一步被调 **2 次**（引导正/负 ✓）⇒ 只在第 1 次换 ✓，两次都落在低噪 ✓"
          "（数次数会换在半路 ✓✗）",
          calls2 == ["low"], calls2)

    # ── 只登记一份 ⇒ 不接力 ✓，且**不做任何换装** ✓ ──
    solo = tb.TorchBackend.__new__(tb.TorchBackend)
    solo._wanExperts = {"high": "高噪.safetensors"}
    solo._wanActive = "high"
    solo._wanSwitchSigma = 0.25
    solo_calls = []
    solo._wan_switch_expert = types.MethodType(
        lambda self, want: solo_calls.append(want), solo)
    solo._wan_expert_tick(0.0)
    check("只给一份专家 ⇒ **不换** ✓（不报错 ✓、不猜另一份路径 ✓；后段质量掉要如实标 ✓）",
          solo_calls == [], solo_calls)

    # ── `bind_plan`：阈值从 plan 取 ✓、每轮复位到高噪 ✓ ──
    plan = pl.build_plan(pl.GenerationRequest(
        prompt="p", stage=pl.WAN22_STAGE, first_frame="假首帧.png", seconds=1.0))
    fresh = tb.TorchBackend.__new__(tb.TorchBackend)
    fresh._wanActive = "low"             # 上一轮换到低噪的残留 ✓
    fresh.bind_plan(plan)
    check("⭐ `bind_plan` 把阈值取自 ``plan.sigmas[wanHighSteps]`` ✓（跟着 schedule 走 ✓、"
          "不写死数 ✓）",
          fresh._wanSwitchSigma == plan.sigmas[plan.wan_high_steps]
          and fresh._wanHighSteps == plan.wan_high_steps,
          (fresh._wanSwitchSigma, plan.sigmas, plan.wan_high_steps))
    check("⭐ 每轮 `bind_plan` 都**复位**到 ``high`` ✓✗（不复位 ⇒ 第二次跑从第一步就用低噪 ✓、"
          "画面糊且**没有任何提示** ✓✗）",
          fresh._wanActive == "high", fresh._wanActive)

    # ── 两份给成同一个文件 ⇒ 当场拒 ✓（那不是接力 ✓）──
    same = tb.TorchBackend.__new__(tb.TorchBackend)
    rejected = _err(lambda: tb.TorchBackend.register_wan_experts(
        same, "D:/同一个.safetensors", "D:/同一个.safetensors"))
    check("高噪/低噪**同一个文件** ⇒ 当场拒 ✗（登记成同一份＝假装接力 ✓✗）",
          "同一个文件" in rejected, rejected)


def case_assembly() -> None:
    print("③ 装配：四格缺格点名 ✓、36 通道硬约束 ✓、三件分时 ✓、装载键含 `ditPath2` ✓")

    runtime = rt.EngineRuntime()
    rejected = _err(lambda: runtime.ensure_loaded(
        stage=pl.WAN22_STAGE, dit_path="D:/高噪.safetensors"))
    first_line = rejected.split("\n")[0]
    check("⭐ 缺格 ⇒ **首行只列真缺的那几格** ✓（`ditPath` 给了就不该出现在缺项里 ✓✗ —— "
          "全列会把真缺的埋进说明里 ✓）",
          "textWeights" in first_line and "imageVaeWeights" in first_line
          and "ditPath" not in first_line, first_line)

    source = inspect.getsource(rt.EngineRuntime._ensure_wan22_loaded)  # noqa: SLF001
    check("⭐ 装配期**显式核 36 通道** ✓✗（Wan 族的 T2V / I2V 同名同族 ⇒ "
          "只看族名会装错 ✓ 而形状要到采样才炸 ✓）",
          "WAN22_I2V_IN_CHANNELS" in source and "in_channels" in source, "")
    check("⭐ 双专家走 `register_wan_experts` ✓（**只登记不装** ✓ —— 两份并存 ≈53 GiB ✗✗）",
          "register_wan_experts" in source)
    check("⭐ 三件**分时** ✓：encode = text+vae ✓ / sample = **只有 dit** ✓ / decode = vae ✓"
          "（首帧条件必须在 encode 段做 ✓ —— 那时 VAE 还在 ✓）",
          '"encode": ("text", "vae")' in source
          and '"sample": ("dit",)' in source and '"decode": ("vae",)' in source, "")
    # ⚠️ 这条原先盯字面串 `vae_family="wan21"` ✗ —— 后来装配线改成**从契约读**
    #    （`vae_family=dit_build.contract_of("wan").latent_format` ✓，见 `dit_build` 那套
    #    「口径只许有一个真源」✓）⇒ 字面串不见了，但口径**没变** ✓。
    #    ⇒ 判据改成「走契约 + 契约真解析成 wan21」✓（盯实现写法会让正确重构变成红 ✗✗）。
    from app.services.engine import dit_build as _db

    check("VAE 口径**从契约读**且解析成 **wan21** ✓、文本塔写明 **umt5** ✓"
          "（写错族 ⇒ 宽度/通道报错，报错指不到借错了族 ✓✗）",
          'vae_family=dit_build.contract_of("wan").latent_format' in source
          and _db.contract_of("wan").latent_format == "wan21"
          and 'text_kind="umt5"' in source,
          (_db.contract_of("wan").latent_format, 'text_kind="umt5"' in source))

    ensure_src = inspect.getsource(rt.EngineRuntime.ensure_loaded)
    check("⭐ **第二份主干进了装载键** ✓✗（不进键 ⇒ 换了低噪却复用上一套 ✓、一声不响 ✓）",
          "dit_path2=dit_path2" in ensure_src, "")

    #: ⚠️ 首帧条件那一拍**不在任何阶段里** ✓（管线里它夹在 init 与 sample 之间 ✓）——
    #: 所以必须**显式**要一次 encode 那一拍 ✓，否则 VAE 在不在场全靠「上一拍恰好是谁」✓✗。
    #: ⚠️ 2026-11-10：`run_sync` 已是**薄壳** ✓✗（阶段执行体是 `_run_stages` ✓）⇒ 扫它得 ``(-1, -1)`` ✓，
    #: 与「压根没要那一拍」的读数一模一样 ✓。改扫**整份 pipeline.py** ＋ `── ③b condition` 当锚点 ✓
    #: （那句锚在模块内唯一 ✓；`_backend_phase(backend, "encode")` 模块里有两处 ✓ —— encode 段开头
    #: 与 ③b 之前 ✓，所以必须从锚点之后开始找 ✓✗，从文件头找会拿到 encode 段那处、判据永远绿 ✓）。
    pipeline_src = (Path(__file__).resolve().parents[1] / "app" / "services" / "engine"
                    / "pipeline.py").read_text(encoding="utf-8")
    anchor = pipeline_src.find("── ③b condition")
    phase_at = pipeline_src.find('_backend_phase(backend, "encode")', anchor)
    #: ⚠️ 别拿 `condition_first_frame` 当锚点 ✗✗ —— ③b 那段**注释里**先提到过一次这个名字 ✓
    #: （说的是"双流首帧不走这条路"✓），`find()` 会命中那句注释 ✓ ⇒ 判出一条**假红** ✓✗。
    hook_at = pipeline_src.find('hook = getattr(backend, "condition_first_frame"', anchor)
    check("⭐ 首帧条件**之前显式要 encode 那一拍** ✓✗（wan22 的条件要 VAE ✓；不显式 ⇒ "
          "VAE 在不在场取决于上一拍 ✓，报错会指成「没挂 VAE」✓✗）",
          phase_at > 0 and hook_at > 0 and phase_at < hook_at, (phase_at, hook_at))

    stub_src = inspect.getsource(tb.TorchBackend._wan_concat_latents)  # noqa: SLF001
    check("⭐ 没做首帧条件就 `denoise` ⇒ **拒跑** ✗✗（零填充凑 36 会出一段与首帧无关的片子 ✓、"
          "且不报错 ✓）",
          "i2v" in stub_src and ("raise" in stub_src or "拒" in stub_src), "")


def case_wiring_attributes() -> None:
    print("③c 装配线引用的**类属性**都真的存在 ✓（真跑抓到的第一处 ✓✗）")

    #: ⭐⭐ 2026-11-02 真跑第一次装配就崩在这里 ✓✗：
    #: ``runtime._ensure_wan22_loaded`` 写着 ``dict(TorchBackend.WAN22_PART_NEED_GIB)`` ✓，
    #: 而 ``TorchBackend`` **没有**这个属性 ✓（文档里写了三格数字 ✓、类体里没建 ✓✗）⇒
    #: ``AttributeError`` 一步都没装 ✓。**40 条源文本判据全绿** ✗✗ —— 它们判的是
    #: 「源码里有没有这句」✓，不是「这句引用的东西存不存在」✓ ⇒ 这一整类都漏 ✓。
    check("⭐ `WAN22_PART_NEED_GIB` 真的**建了** ✓✗（只写进文档不建属性 ⇒ 装配第一步 AttributeError ✓）",
          hasattr(tb.TorchBackend, "WAN22_PART_NEED_GIB"),
          getattr(tb.TorchBackend, "WAN22_PART_NEED_GIB", "<缺失>"))
    table = getattr(tb.TorchBackend, "WAN22_PART_NEED_GIB", {})
    check("三格口径：一份专家 **26.62** ✓（两份相加 53.24 是错的悲观 ✗✗）、UMT5 **10.58** ✓、VAE **0.24** ✓",
          dict(table) == {"dit": 26.62, "text": 10.58, "vae": 0.24}, table)
    check("⭐ 单份 DiT **大于整张卡** ✓（22.49 ✓）⇒ `decide_for` 除了 `split` 没别的选择 ✓"
          "（表里若写成 13.26 那个文件字节 ⇒ 判 `off` ⇒ 走到 sample 当场 OOM ✓✗）",
          float(table.get("dit", 0)) > 22.49, table.get("dit"))

    #: ⚠️ 泛化这一类：**`runtime` 里凡写 ``TorchBackend.<大写字面>`` 的，属性必须真存在** ✓✗
    #: （四种族各引用一张表 ✓，任何一张漏建都只会在**真装到那一步**才炸 ✓）。
    import re  # noqa: PLC0415
    runtime_src = inspect.getsource(rt)
    missing = sorted({name for name in re.findall(r"TorchBackend\.([A-Z][A-Z0-9_]{2,})", runtime_src)
                      if not hasattr(tb.TorchBackend, name)})
    check("⭐ `runtime` 引用的 `TorchBackend.<大写属性>` **一个都不缺** ✓✗（缺的只会在真跑到那一步"
          " AttributeError ✓；本判据是**通用**的 ✓，新加族漏建也拦得住 ✓）",
          not missing, missing)

    #: ⭐⭐ **真跑抓到的第二处** ✓✗：装配线拿 ``config["in_channels"]`` 判入口 ✓，
    #: 而 Wan 族的 ``in_channels`` **按契约**就是潜宽 **16** ✓（`dit_build` 那行
    #: ``shortcuts=(("in_channels","out_dim"),)`` ✓ —— `latent_shape` 造的是噪声 ✓）
    #: ⇒ **每一份正经 I2V 都会被判成 T2V** ✓✗（盘上那份 36 通道的真文件也照拒 ✓）。
    #: 入口宽的真名是 ``in_dim`` ✓（``patch_embedding.weight`` 第二维 ✓）。
    check("I2V 布局 **16 ＋ 4 ＋ 16 = 36** ✓（掩码 **4** 不是 1 ✓✗ —— 写 1 ⇒ 拼出来 33 ✓ 不报错 ✓ 也不对 ✓）",
          sum(width for _name, width in dw.I2V_LAYOUT) == 36
          and dict(dw.I2V_LAYOUT)["mask"] == 4
          and rt.WAN22_I2V_IN_CHANNELS == 36,
          (dw.I2V_LAYOUT, rt.WAN22_I2V_IN_CHANNELS))
    check("Wan 族的 `in_channels` 契约上就是**潜宽 16** ✓ ⇒ 装配线读它判入口必假红 ✓✗"
          "（这条判据把「为什么不能读它」钉在表里 ✓）",
          dict(db.BUILDERS["wan"].shortcuts).get("in_channels") == "out_dim",
          dict(db.BUILDERS["wan"].shortcuts))
    guard_src = inspect.getsource(rt.EngineRuntime._ensure_wan22_loaded)  # noqa: SLF001
    at_in_dim = guard_src.find('config.get("in_dim"')
    at_in_ch = guard_src.find('config.get("in_channels"')
    check("⭐ 装配线读 **`in_dim`**（36）✓ 且**先于** `in_channels`（16）✓✗ —— 读错这一格 ⇒ "
          "正经 I2V 权重全被拒成「这是 T2V」✓（自检 40 条也拦不住 ✓）",
          at_in_dim > 0 and at_in_ch > at_in_dim, (at_in_dim, at_in_ch))
    check("读不到入口通道 ⇒ **当场拒并写明「读不到」** ✓（不是 `or 0` 然后报「入口是 0 通道」✓✗"
          "、更不是放行 ✓）",
          "读不到这份 Wan 权重的**入口通道**" in guard_src, "")

    #: ⭐⭐ **真跑抓到的第三处** ✓✗：``declare_chain(text_kind="umt5")`` 传下去了 ✓，
    #: 但 ``enter_phase`` 的分派表里**没有 umt5 那一支** ✓ ⇒ 落到 ``qwen35`` 兜底 ✓ ⇒
    #: 报「缺 ``model.embed_tokens.weight`` ⇒ 这不是 Qwen3.5 的文本塔」✓（措辞对 ✓、真因错 ✓✗）。
    #: ⚠️ 别用 `try/except` 兜 ✗✗：UMT5 与 Qwen3.5 宽度都可能是 4096 ✓，试错能"跑起来"✓
    #: 而隐状态来源完全不同 ✓（encoder 末层过 final norm、不套 chat 模板 ✓）⇒ 条件全错且不报错 ✓✗。
    phase_src = inspect.getsource(tb.TorchBackend.enter_phase)
    check("⭐ 分派表里有 **umt5** 那一支 ✓✗（漏了 ⇒ 落到 `qwen35` 兜底 ⇒ 报「缺 "
          "`model.embed_tokens.weight`」✓，读起来像权重下坏了 ✓）",
          "self.attach_umt5_te" in phase_src
          and phase_src.find("self.attach_umt5_te") < phase_src.find("self.attach_qwen35_te"),
          (phase_src.find("self.attach_umt5_te"), phase_src.find("self.attach_qwen35_te")))
    #: ⚠️ 泛化：**`runtime` 里每个 `text_kind="…"` 都得在分派表里有一支** ✓✗
    #: （四种族各传一个 ✓，漏一个只会在**真跑到 encode** 时才炸 ✓，且报错指不到分派表 ✓）。
    import re as _re  # noqa: PLC0415
    kinds = sorted(set(_re.findall(r'text_kind="([a-z0-9-]+)"', inspect.getsource(rt))))
    unresolved = [k for k in kinds if f'"{k}"' not in phase_src]
    check("⭐ `runtime` 传的**每一个** `text_kind` 在分派表里都认 ✓✗（新加族漏登记 ⇒ 静默落到 "
          "`qwen35` 兜底 ✓，报错指向「缺某权重」✓）", not unresolved, (kinds, unresolved))

    #: ⭐⭐ **真跑抓到的第四处** ✓✗：``self._device()`` —— ``_device`` 是**字符串属性** ✓
    #: （``"cuda"`` ✓，见 ``__init__`` 与 ``device`` 属性 ✓），它**不是方法** ✓ ⇒
    #: ``TypeError: 'str' object is not callable`` ✓，报在「文本编码失败」那句里 ✓
    #: （读起来像分词器或文本塔坏了 ✓✗）。正确取法 :meth:`TorchBackend._device_torch` ✓
    #: （它顺手补卡号 ✓）。⚠️ 判据必须走 **AST** ✗：扫文本会把**注释里**那句
    #: ``self._device()`` 也算进调用 ✓ ⇒ 假红（本轮注释里就写了它 ✓）。
    tree = ast.parse(inspect.getsource(tb))
    called: set[str] = set()
    returned: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
            called.add(node.func.attr)
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "self"):
            returned.add(node.value.attr)
    value_fields_called = sorted(called & returned)
    check("⭐ 被 `return self.X` **当值**给出的字段，一个都不许被 `self.X()` **当方法**调 ✓✗"
          "（`_device` 就是这么错成 `'str' object is not callable` 的 ✓；本条是**通用**的 ✓）",
          not value_fields_called, value_fields_called)


def case_registration() -> None:
    print("④ 登记：`wan22` 单列一档 ✓（不与 h3 / seedvr2 混 ✓）、清单四件 ✓")

    spec = registry.STAGES.get(pl.WAN22_STAGE)
    check("`registry.STAGES` 里有 `wan22` ✓、`category='video'` ✓、"
          "能力写 `video.generate` ✓",
          spec is not None and spec.category == "video"
          and "video.generate" in spec.capabilities, spec)
    check("`STAGE_FILTERS['wan22']` 与 registry **双向核** ✓（只登记一处 ⇒ "
          "`readiness` 会列空集却报就绪 ✓✗）",
          inventory.STAGE_FILTERS.get(pl.WAN22_STAGE, {}).get("category") == spec.category)
    check("筛选标签是 **`wan22`** ✓ 而不是 `wan` ✓✗（清单里 `wan26_i2v_14b_gguf` 也带 `wan` ✓，"
          "卷进来会多算十几 GiB ✓✗）",
          inventory.STAGE_FILTERS[pl.WAN22_STAGE]["tags"] == ["wan22"])

    entries = inventory.stage_entries(pl.WAN22_STAGE)
    keys = sorted(str(entry["key"]) for entry in entries)
    check("清单里四件都被 **`wan22`** 认领 ✓（高噪/低噪/UMT5/VAE ✓）",
          keys == ["dit_wan22_i2v_high", "dit_wan22_i2v_low",
                   "te_wan22_umt5_xxl", "vae_wan22"], keys)
    check("⭐ 四件**全部** `required=True` ✓（`requireAll` ✓）—— 标成可选 ⇒ "
          "一份没有也说就绪 ✓✗",
          all(bool(entry["required"]) for entry in entries),
          [entry["required"] for entry in entries])
    #: ⚠️ 别去判「h3 档里没有 wan22」✗✗ —— ``h3`` 的筛选是 ``{"category": "video"}`` ✓
    #: （**全 video 目录视图** ✓，SEEDVR2 也一直在里面 ✓），它**不是**"本链必需集"✓。
    #: 该判的是**反向**：``readiness("wan22")`` 的**必需集**只有这四件 ✓✗。
    ready = inventory.readiness(pl.WAN22_STAGE)
    required = sorted(str(c["key"]) for c in ready["components"] if c.get("required"))
    check("⭐ `readiness('wan22')` 的**必需集 = 那四件** ✓（把 H3 的 `dit_fl2va_*` / `te_nvfp4` "
          "卷进来 ⇒ 凭空报要 60 GiB ✓✗）",
          required == ["dit_wan22_i2v_high", "dit_wan22_i2v_low",
                       "te_wan22_umt5_xxl", "vae_wan22"], required)
    check("⭐ 就绪档**如实报「装不下」** ✓（33.14 GiB 权重 > 24 GiB 卡 ⇒ ``fits=False`` ✓）"
          "—— 报「装得下」会让用户以为不用 split ✓✗",
          ready["vram"]["fits"] is False
          and abs(float(ready["requiredWeightsGiB"]) - 33.14) < 0.1,
          (ready["requiredWeightsGiB"], ready["vram"]["fits"]))


def case_real_run_truth() -> None:
    print("⑤ 真跑（2026-11-02）抓到的三处「形状/字段都对、跑起来才炸」✓✗")

    #: ⭐ **第五处**：``dit_wan22.WanDiT.forward`` 旧写法
    #: ``x = conv(latents.to(pw.dtype)).to(latents.dtype)`` ✓ ⇒ conv 走 bf16 ✓、
    #: 紧接着第一个 ``Linear`` 吃回 fp32 的 x ✓ ⇒ 采样第 1 步当场
    #: ``RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`` ✓
    #: （读起来像"装配精度混了" ✓✗，其实是**同一个前向里两种精度** ✓）。
    #: 采样起点是 fp32 噪声 ✓（``init_latents`` 就是 fp32 ✓）、盘上 fp8_scaled 反量化后是 bf16 ✓
    #: ⇒ 这两者只能在**前向入口**对齐 ✓，且**整条前向只留一种精度** ✓。
    import torch  # noqa: PLC0415 —— 这条判据要真跑一次前向 ✓
    shape = dw.WanShape(dim=32, ffn_dim=64, in_dim=36, out_dim=16, text_dim=8, freq_dim=16,
                        heads=2, layers=1, patch=(1, 2, 2))
    model = dw.WanDiT(shape, dtype=torch.bfloat16).eval()
    latents = torch.zeros(1, 36, 2, 4, 4, dtype=torch.float32)
    context = torch.zeros(1, 3, shape.text_dim, dtype=torch.float32)
    out, failure = None, ""
    try:
        with torch.no_grad():
            out = model(latents, torch.tensor([0.5]), context)
    except BaseException as err:  # noqa: BLE001 —— 就是要看它炸成什么样 ✓
        failure = f"{type(err).__name__}: {err}"
    check("⭐ fp32 latent ＋ bf16 权重 ⇒ 前向**跑得通** ✓✗（旧写法第 1 步就 `mat1 Float / mat2 "
          "BFloat16` ✓；这是真跑炸出来的，41 条静态判据一条都没拦住 ✓）",
          not failure, failure)
    check("前向精度**跟权重走** ✓（输出 bf16 而不是 latent 的 fp32 ✓ —— 与 `dit_qwen21` 同一口径 ✓）",
          out is not None and out.dtype == torch.bfloat16,
          None if out is None else out.dtype)
    #: ⚠️ 只能扫**非注释**行 ✓：这条缺陷的来龙去脉就写在旁边那段 ``#:`` 注释里 ✓✗（含旧写法原文 ✓）。
    fwd_lines = [line for line in inspect.getsource(dw.WanDiT.forward).splitlines()
                 if not line.strip().startswith("#")]
    check("反向：`forward` 的**代码**里不许再出现 `.to(latents.dtype)`（那就是「两种精度」的写法 ✓✗；"
          "注释里保留旧写法原文不算 ✓）",
          not any(".to(latents.dtype)" in line for line in fwd_lines), "")

    #: ⭐ **第六处**：管线首帧条件的潜帧数原先**只**问请求的 ``temporalCompression`` ✓ ⇒
    #: Wan 2.2 的请求不带它 ✓（4 是族常量 ✓，`_fill_wan22_plan` 早算好了 `plan.latent_frames`
    #: ＝5 ✓）⇒ 当场报「首帧条件需要 temporalCompression ✗」，读起来像**用户少填了字段** ✓✗。
    #: ⚠️ 2026-11-10：`run_sync` 已拆成**薄壳** ✓✗（阶段执行体搬进 `_run_stages` ✓）⇒
    #: `inspect.getsource(pl.run_sync)` 里这两句都找不到 ✓✗，得到 ``(-1, -1)`` ✓ ——
    #: 与「顺序真写反了」的读数**一模一样** ✓（假红长得像真缺陷 ✓）。
    #: 改扫**整份 pipeline.py** ✓：这两句全模块各只有一处 ✓（下面那条「各只出现一次」钉住它 ✓✗，
    #: 将来别的族也写同样的句子时，`find` 会取错 → 那条先红、提示来精化扫描面 ✓）。
    pipeline_src = (Path(__file__).resolve().parents[1] / "app" / "services" / "engine"
                    / "pipeline.py").read_text(encoding="utf-8")
    at_plan = pipeline_src.find('getattr(plan, "latent_frames"')
    at_req = pipeline_src.find("latent_frames_for(plan.frames")
    check("⭐ 首帧条件的潜帧数**先问 plan**（`getattr(plan, \"latent_frames\"` ✓）、plan 上没有才按"
          "请求压缩比现算 ✓✗（顺序反了 ⇒ wan22 报「需要 temporalCompression」✓，指不到真因 ✓；"
          "扫的是整个模块而不只是 `run_sync` ✓ —— 后者 2026-11-10 起已是薄壳 ✓✗）",
          at_plan > 0 and at_req > at_plan, (at_plan, at_req))
    check("反向：这两句在 `pipeline.py` 里**各只出现一次** ✓✗（多于一处 ⇒ 上面那条的 `find` "
          "可能取到别的族的同文句、判成「顺序对」而实际没看 wan22 那段 ✓✗）",
          pipeline_src.count('getattr(plan, "latent_frames"') == 1
          and pipeline_src.count("latent_frames_for(plan.frames") == 1,
          (pipeline_src.count('getattr(plan, "latent_frames"'),
           pipeline_src.count("latent_frames_for(plan.frames")))

    #: ⭐ **第七处**（不在本文件、但它是 wan22 的前提 ✓）：Wan 2.1 VAE 的 **4 倍时间压缩** ✓。
    #: 旧 ``Resample`` 从不真跑 ``time_conv`` ✓ ⇒ 17 帧原样 17 潜帧 ✓、且**不报错** ✓✗。
    #: 细判据在 ``engine_vae_wan_test.case_temporal`` ✓；这里只钉 wan22 依赖的那条不变量 ✓。
    vae_shape = vw.WanVaeShape(dim=8, dec_dim=12, z_dim=4, dim_mult=(1, 2, 2), num_res_blocks=1,
                               temperal_downsample=(True, True), image_channels=3,
                               patch_size=1, temporal_kernel=3)
    vae = vw.build(vae_shape).eval()
    with torch.no_grad():
        z = vae.encode(torch.zeros(1, 3, 17, 32, 32))
        y = vae.decode(torch.zeros(1, vae_shape.z_dim, 5, 4, 4))
    check("⭐ wan22 的帧数口径**靠 VAE 真做 4 倍时间压缩** ✓：17 像素帧 ⇒ **5 潜帧** ✓"
          "（＝`plan.latent_frames` ✓、`frame_grid=4` ✓；旧实现给 17 ✓✗）",
          z.shape[2] == 5 and pl.WAN22_FRAME_GRID == 4, (tuple(z.shape), pl.WAN22_FRAME_GRID))
    check("反向：5 潜帧解码回 **17 像素帧** ＝ `4(T−1)+1` ✓（旧实现给 5 ✓✗ —— 出片短 4 倍且不报错 ✓）",
          y.shape[2] == 4 * (5 - 1) + 1, tuple(y.shape))


def main() -> int:
    print("=" * 78)
    print("Wan 2.2 I2V 接线自检（plan / 双专家 / 装配 / 登记 ✓）")
    print("=" * 78)
    case_plan()
    case_expert_switch()
    case_assembly()
    case_wiring_attributes()
    case_real_run_truth()
    case_registration()
    print("-" * 78)
    print(f"通过 {PASSED} / 失败 {FAILED}")
    print(f"SUMMARY: {PASSED}/{PASSED + FAILED} passed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
