"""自检：**Music3 全链接线**（plan → 装配 → 条件 → 采样 → 解码 → 落 wav）✓。

跑法：``.venv/Scripts/python.exe tests/engine_music3_wiring_test.py``

这里判的不是"公式对不对" ✗（那归 `engine_music3_test.py` ✓ 与 `engine_music3_keys_test.py` ✓），
而是**零件装没装上** ✓：一条链上每两个相邻零件之间的口径（参数名 ✓、几何 ✓、单位 ✓）
都各自"看起来对" ✓，拼起来对不对**只有真跑或逐处对账才判得出** ✓✗ —— 本族至今踩到的三处
（``audioVaeWeights`` 塞进 ``imageVaeWeights`` ✓ / ``lyrics`` 掉在路由键表外 ✓ /
``latent_shape`` 先判 ``vae_scale`` 再认 3 维 ✓）**全都属于这一类** ✓：不报错、不崩、
只是"歌不对"或者"根本没唱" ✓✗。

判据分七组：
A **常量同源**（25 Hz / 上限 9000 只有一份真源 ✓；潜帧换算与主干逐帧对拍 ✓；
  **DAV 键名归一表只有一份** ✓ —— 2026-11-06 首次真跑就栽在「对账归一了、装载没归一」✓✗；
  A14–A16 是**同一趟真跑的后两处** ✓：解码取采样率读的是配置/事实表 ✓、且
  ``torch_backend`` 引用的每个 ``vae_music3_mod.*`` 名字都真的存在 ✓ —— 写错属性名不会在
  import 期报 ✗，只在跑到那一行时炸 ✓，而那一行往往正好在整链最末尾 ✓；
  A17 是**对着参考实现逐处对账**才抓到的真差异 ✓✗：σ 走 `normal`（σ 等距 ✓、端点 1.0/0.001 ✓）
  而不是 flow 的 t 等分 ✓ —— 差 33 倍的末段跳幅，不报错、只是尾巴残噪没清 ✓）
B **plan 算术**（真跑 ``build_plan`` ✓：秒 → 条件帧 → 潜帧 ✓ 三档互核 ✓）
C **散文不撒谎**（引导警告不再说「更平」✓、缺歌词必须明说纯器乐 ✓）
D **清单口径**（`readiness("music3")` 真有 13.3 GiB ✓；装配格是 `audio_vae_weights` ✓；不收词表 ✓）
E **接线顺序**（三处"必须排在谁前面" ✓ —— 顺序换了不报错 ✓ 只是静默走错路 ✓✗）
F **装配线拒绝面**（给 `tokenizerPath` 当场拒 ✓；缺格首行只列真缺的 ✓）
G **路由键表**（`lyrics` 在册 ✓、`music3` 在阶段表里 ✓）
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp936 下 `✓` 会 UnicodeEncodeError
except Exception:                              # pragma: no cover
    pass

_RESULTS: list[tuple[bool, str, object]] = []


def check(name: str, condition: object, detail: object = "") -> bool:
    ok = bool(condition)
    _RESULTS.append((ok, name, detail))
    print(f"{'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail and not ok else ""))
    return ok


def _error_of(call) -> str:
    """跑一段必炸的调用，返回异常文本 ✓（判据要「炸得对」✓）。"""
    try:
        call()
    except Exception as err:            # noqa: BLE001 —— 这里就是要接住任何异常 ✓
        return f"{type(err).__name__}: {err}"
    return ""


from app.services.engine import inventory as inv                     # noqa: E402
from app.services.engine import pipeline as pipe                     # noqa: E402
from app.services.engine import runtime as rt                        # noqa: E402
from app.services.engine import te_music3 as te                      # noqa: E402
from app.services.engine import torch_backend as tb                  # noqa: E402
from app.services.engine import vae_music3 as vae                    # noqa: E402
from app.routers import engine as router_engine                      # noqa: E402


def _request(seconds: float = 20.0, **kw):
    kw.setdefault("prompt", "轻柔的钢琴与弦乐，女声哼唱")
    kw.setdefault("stage", "music3")
    return pipe.GenerationRequest(seconds=seconds, **kw)


# ───────────────────────── A 常量同源 ─────────────────────────


def case_constants() -> None:
    """★ **两把时间尺子都只有一份真源** ✓✗。

    本族的时间轴与出图/出片**完全不同** ✓：出片是「帧 = 秒 × fps」✓（24/16 ✓，几十的量级 ✓）；
    音乐是「**条件帧** = 秒 × **25**」✓ 与「**潜帧** = 条件帧 × 3.4453125」✓（86.13 Hz ✓，
    几百上千的量级 ✓）。⚠️ 三个数**手抄一遍就够出一次事故** ✓✗：抄成 24 或 30 ⇒ 条件比歌长 ✓、
    尾部一段越来越空 ✓（不报错 ✓）；抄成「乘 25 再除 25」⇒ 直接不对齐 ✓。
    """
    check("A1 pipeline 的条件帧率**就是** `te_music3.AUDIO_FRAMES_PER_SECOND`（同一对象 ✓）",
          pipe.MUSIC3_COND_FPS == te.AUDIO_FRAMES_PER_SECOND,
          f"pipeline={getattr(pipe, 'MUSIC3_COND_FPS', None)}{{}} te={te.AUDIO_FRAMES_PER_SECOND}")
    check("A2 上限也同源 ✓（`MUSIC3_MAX_AUDIO_FRAMES` 不是只手抄的 9000）",
          getattr(pipe, "MUSIC3_MAX_AUDIO_FRAMES", None) == te.MAX_AUDIO_FRAMES,
          f"pipeline={getattr(pipe, 'MUSIC3_MAX_AUDIO_FRAMES', None)} te={te.MAX_AUDIO_FRAMES}")
    check("A3 25 帧/秒（这一条是**事实判据** ✓，不是「跟着常量走」✓：数字本身错了上面两条也自洽 ✗）",
          te.AUDIO_FRAMES_PER_SECOND == 25, te.AUDIO_FRAMES_PER_SECOND)
    facts = vae.MUSIC3_DAV_FACTS
    check("A4 潜帧率 ≈ 86.13 Hz（44100/512 ✓）", abs(float(facts["latentHz"]) - 86.1328125) < 1e-9,
          facts["latentHz"])
    check("A5 DAV 采样率 44.1 kHz / 2 声道 / 128 潜通道",
          (facts["sampleRate"], facts["stereo"], facts["latentChannels"]) == (44100, 2, 128),
          (facts["sampleRate"], facts["stereo"], facts["latentChannels"]))
    check("A6 潜帧率**不是**从条件帧率推的（两把尺子各自独立 ✓：25 Hz 与 86.13 Hz）",
          abs(float(facts["latentHz"]) - float(te.AUDIO_FRAMES_PER_SECOND)) > 1.0,
          (facts["latentHz"], te.AUDIO_FRAMES_PER_SECOND))
    #: ⚠️ 下面两组是**两个模块各算一遍**的同一件事 ✓✗：`pipeline` 为了方便抄了一份潜帧比，
    #:  `dit_music3.latent_length` 是主干真正用来切 window 的那份 ⇒ 差 1 帧不会有任何报错 ✓✗。
    from app.services.engine import dit_music3 as dit3
    check("A7 抄自 `dit_music3` 的两个窗口数逐值一致（200 / 100 ✓）",
          (pipe.MUSIC3_MAX_COND_FRAMES, pipe.MUSIC3_COND_HOP_FRAMES)
          == (dit3.MAX_CONDITION_FRAMES, dit3.CONDITION_HOP_FRAMES),
          (pipe.MUSIC3_MAX_COND_FRAMES, pipe.MUSIC3_COND_HOP_FRAMES,
           dit3.MAX_CONDITION_FRAMES, dit3.CONDITION_HOP_FRAMES))
    bad = [(n, pipe.music3_latent_frames(n), dit3.latent_length(n))
           for n in range(1, 9001) if pipe.music3_latent_frames(n) != dit3.latent_length(n)]
    check("A8 潜帧换算与主干**逐帧对拍**（1..9000 全部一致 —— 抄成 `× 3.4453125` 会在个别帧数上差 1 ✓✗）",
          not bad, bad[:6])
    #: ⭐ **2026-11-06 首次真跑抓到的真事故** ✓✗：DAV 检查点里 weight_norm 是**旧式**
    #: ``…weight_g / …weight_v`` ✓，`build_music3_dav` 造出的模块 ``state_dict`` 是**新式**
    #: ``…parametrizations.weight.original0/1`` ✓ ⇒ ``shape_problems`` 会归一（所以它全绿 ✓）、
    #: 而 ``attach_music3_dav`` 走通用装载器**按原键硬对** ⇒ 报「缺 60 键 / 形状不符 0 处」✓✗
    #: （读起来像拿错了文件 ✓，其实是同一份文件两套写法 ✓）。判据两条：表能真的映射 ✓、
    #: 装载口**真的传了**这张表 ✓（只加表不传 ⇒ 事故原样复现 ✓✗）。
    shape_key = "decoder.model.0.weight_g"
    mapped = vae._canonical_key(shape_key)  # noqa: SLF001 —— 对账口 ✓
    check("A9 键名归一表真的把旧式映射到新式（`…weight_g` ⇒ `…parametrizations.weight.original0`）",
          mapped == "decoder.model.0.parametrizations.weight.original0"
          and vae.as_legacy_name(mapped) == shape_key, (mapped, vae.as_legacy_name(mapped)))
    attach_src = inspect.getsource(tb.TorchBackend.attach_music3_dav)
    check("A10 装载口**真的传了** `key_map`（`shape_problems` 归一 ≠ 装载器归一 ✗✗）",
          "key_map=vae_music3_mod.KEY_MAP" in attach_src.replace(" ", "").replace("\n", ""),
          attach_src[-500:])
    check("A11 两张表**是同一张**（`KEY_MAP` 只有一份真源 ✓，别再手写 `endswith` ✗）",
          vae.KEY_MAP and all(isinstance(k, str) and isinstance(v, str) for k, v in vae.KEY_MAP.items())
          and len(vae.KEY_MAP) == 2, vae.KEY_MAP)
    #: ⭐ **首次真跑栽的第二处** ✓✗：`te_music3.generate` **有意**把条件留在 CPU ✓（生成时腾卡 ✓），
    #: `denoise` 进门又核「condition 与后端同设备」✓ ⇒ 中间**必须**有一处搬卡 ✓。
    #: 两条判据成对 ✓：上游确实交 CPU ✓ ＋ 下游确实搬 ✓（只改一头就复现 `在 cpu，后端在 cuda:0` ✗）。
    gen_tail = inspect.getsource(te.Music3TextTower.generate).rstrip().splitlines()[-1].strip()
    check("A12 `te_music3.generate` 的返回**仍是有意留 CPU** ✓（改它要先想清楚采样期显存 ✓）",
          'device="cpu"' in gen_tail or "device='cpu'" in gen_tail, gen_tail)
    cond_src = inspect.getsource(tb.TorchBackend._encode_music3_condition)
    check("A13 条件编码**在边界搬一次卡** ✓（上游交 CPU ＋ 下游不搬 ⇒ `condition 在 cpu` ✓✗）",
          "frames.to(device=self._device_torch()" in cond_src.replace("frames = ", ""),
          cond_src[-420:])
    #: ⭐ **首次真跑栽的第四处** ✓✗：`_decode_music3` 算时长要读采样率 ✓，回落值原本写
    #: ``vae_music3_mod.SAMPLE_RATE`` ✓ —— 那个名字**从来不存在** ✓（真身在
    #: :data:`vae.MUSIC3_DAV_FACTS["sampleRate"]` ✓）⇒ 前面四环全绿 ✓、最后一步炸
    #: AttributeError ✓。（同类病的通用面见 A15 ✓。）
    dec_src = inspect.getsource(tb.TorchBackend._decode_music3)
    #: 判据刻意盯**表达式**而不是盯字面名 ✓✗：这段源码的注释里就写着事故名 ✓，
    #: 盯字面会把「解释这段事故的注释」当成事故本身 ✓（A16 才是「这个名字存不存在」的守门 ✓）。
    dec_flat = dec_src.replace(" ", "").replace("\t", "")
    check("A14 解码取采样率的回落值**读 `MUSIC3_DAV_FACTS`** ✓（`vae_music3.SAMPLE_RATE` 不存在 ✓✗）",
          'MUSIC3_DAV_FACTS["sampleRate"]' in dec_flat
          and "orvae_music3_mod.SAMPLE_RATE" not in dec_flat,
          dec_flat[-600:])
    facts_src = inspect.getsource(tb.TorchBackend.attach_music3_dav)
    check("A15 装配口把采样率/声道**记进 `_vaeConfig`** ✓（`_decode_music3` 读的是配置 ✓，不记就回落）",
          "sample_rate=int(spec[" in facts_src.replace("'", '"'), facts_src[-700:])
    #: ⚠️ **通用面** ✓✗：`vae_music3_mod.<NAME>` 里的每个名字都必须在模块上**真的存在** ✓ ——
    #: 写错一个属性名不会在 import 期报 ✗（模块级属性是延迟解析的 ✓），只在**跑到那一行**时炸 ✓，
    #: 而那一行往往正好在整链最末尾 ✓（解码 ✓）。用 AST 扫，不执行 ✓。
    import ast as _ast                                       # noqa: PLC0415
    tb_path = Path(inspect.getfile(tb.TorchBackend))
    names: set[str] = set()
    for node in _ast.walk(_ast.parse(tb_path.read_text(encoding="utf-8"))):
        if (isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name)
                and node.value.id == "vae_music3_mod"):
            names.add(node.attr)
    absent = sorted(n for n in names if not hasattr(vae, n))
    check("A16 `torch_backend` 引用 `vae_music3_mod.*` 的每个名字都真的存在 "
          "（写错属性名只在跑到那一行才炸 ✓✗）", names and not absent, absent)
    #: ⭐ **对参考实现抓到的真差异** ✓✗（2026-11-06 第二轮）：本族是 ``ModelType.FLOW`` ⇒
    #: ``ModelSamplingDiscreteFlow`` 取 ``shift=1.0``（``sampling_settings`` 没写 shift ✓）＋
    #: ``multiplier=1.0`` ⇒ ``time_snr_shift(1.0, t) = t`` **恒等** ✓，而官方 ``KSampler`` 的默认
    #: ``scheduler`` 是 ``normal`` ✓ ⇒ 落到 σ 上就是 **σ 等距**、端点 ``(1.0, 0.001)``、末尾 ``0`` ✓。
    #: 旧写法走 ``flow_sigmas_for_steps`` ✓（**t** 等分、端点 ``(1.0, 0.0)`` ✓）⇒ 30 步时末段是
    #: ``0.0333 → 0`` 对 ``0.001 → 0`` ✓ —— **噪声跳幅差 33 倍** ✓，症状只是"尾巴那点残噪没清"✓
    #: （不报错、不看波形根本发现不了 ✓✗）。三条判据成对：逐值对拍 ✓ ＋ 端点 ✓ ＋ 源码没走回头路 ✓。
    from app.services.engine import schedules as sch          # noqa: PLC0415
    plan = pipe.build_plan(_request(20.0, lyrics="月亮升起来"))
    want = sch.normal_sigmas(plan.steps, shift=pipe.MUSIC3_FLOW_SHIFT)
    check("A17a Music3 的 σ **逐值等于 `normal_sigmas`**（σ 等距 ✓，不是 flow 的 t 等分 ✗）",
          [round(v, 9) for v in plan.sigmas] == [round(v, 9) for v in want],
          (plan.sigmas[:4], want[:4]))
    check("A17b 端点 ``1.0 → 0.001 → 0``（末段差 33 倍的那一格 ✓）",
          abs(plan.sigmas[0] - 1.0) < 1e-9 and abs(plan.sigmas[-2] - 0.001) < 1e-9
          and plan.sigmas[-1] == 0.0, plan.sigmas[-3:])
    music_src = inspect.getsource(pipe._fill_music3_plan).replace(" ", "").replace("\n", "")
    check("A17c `_fill_music3_plan` 源码**不再**调 `_image_flow_sigmas`（改回去不报错、只静默错 ✗）",
          "normal_sigmas(" in music_src and "_image_flow_sigmas(" not in music_src,
          music_src[-260:])


# ───────────────────────── B plan 算术 ─────────────────────────


def case_plan_math() -> None:
    """★ 三档**互核** ✓✗：条件帧（25 Hz ✓）⇒ 潜帧（× 3.4453125 ✓）⇒ 秒数回推 ✓。

    只核一件是不够的 ✓：``frames = round(秒×25)`` 与 ``latent = ceil(条件×3.4453125)``
    两处各自"看着对" ✓，但 ``_decode_music3`` 最后按
    ``样本数 / 44100`` 报"实际几秒" ✓ ⇒ 三处**必须自洽** ✓，差 1 帧就是几十毫秒 ✓
    （听起来像"尾巴被切了"✓，查起来却是三个数各对 ✓）。
    """
    for seconds, want_cond, want_latent in ((1.0, 25, 86), (20.0, 500, 1722), (360.0, 9000, 31007)):
        plan = pipe.build_plan(_request(seconds, lyrics="月亮升起来"))
        check(f"B1 {seconds:g}s ⇒ 条件 {want_cond} 帧 / 潜帧 {want_latent}（**向下取整** ✓）",
              (plan.frames, plan.latent_frames) == (want_cond, want_latent),
              (plan.frames, plan.latent_frames))
        check(f"B2 {seconds:g}s 的 plan.fps 是条件帧率 25（不是视频的 24/16）",
              float(plan.fps) == float(te.AUDIO_FRAMES_PER_SECOND), plan.fps)
        back = vae.seconds_for_frames(want_latent)
        #: 潜帧 → 样本 → 秒 ✓：解码侧按这个数报"实际几秒" ✓，与请求的秒数差 ≤ 一个潜帧 ✓。
        check(f"B3 {seconds:g}s：潜帧回推 {back:.4f}s 与请求差 ≤ 一个潜帧（{1.0 / 86.1328125:.4f}s）",
              abs(back - seconds) <= 1.0 / 86.1328125 + 1e-9,
              (back, seconds))
    over = _error_of(lambda: pipe.build_plan(_request(400.0)))
    check("B4 超过上限（360 秒 / 9000 条件帧）⇒ **拒**，且报出「不是慢而是编不出」",
          "编不出" in over and "360" in over and "9000" in over, over[:160])
    zero = _error_of(lambda: pipe.build_plan(_request(0.0)))
    check("B5 0 秒 ⇒ 拒（不是「给个默认 5 秒」✗：音频没有可猜的默认 ✓）",
          "seconds" in zero and ">0" in zero, zero[:120])
    #: ⚠️ 时长**不能**从 ``frames``/``fps`` 反推着改 ✓：提示串里没有"帧"这回事 ✓。
    plan = pipe.build_plan(_request(20.0))
    check("B6 width/height 在音频档是 0（不是「顺手给个 1024」✗ —— 那会让代价估算凭空乘像素）",
          (plan.width, plan.height) == (0, 0), (plan.width, plan.height))


# ───────────────────────── C 散文不撒谎 ─────────────────────────


def case_prose() -> None:
    """★ 本族有**两层** CFG ✓✗，散文把它们混成一层就变成了错的说明书 ✓。

    自回归内部固定 ``cfg 1.5 / top-50`` ✓（真引导在这儿 ✓）；DiT 那层的 CFG 要**负向条件** ✓，
    而本仓恒空 ⇒ ``guidance`` 这一格**调多少都不变** ✓✗。旧散文写「给 1.0 等于只留一条支路、
    会明显更平」✗ —— 它把「用户调的这一格」说成了「真正的那个旋钮」✓✗，
    用户会去调高它 ✓、然后发现一点没变 ✓（这比报错更费时 ✓）。
    """
    plan = pipe.build_plan(_request(20.0, lyrics="月亮升起来"))
    text = "\n".join(plan.warnings)
    check("C1 引导警告**不再**说「会明显更平」（那是错的 ✓✗）", "更平" not in text, text[:200])
    check("C2 引导警告写清了「对出歌没有作用」＋「自回归内部」这层真旋钮",
          "对出歌没有作用" in text and "自回归" in text, text[:260])
    #: 给了歌词 ⇒ 必须**明说**走独立支路 ✓（不能静默按纯器乐 ✓✗）。
    check("C3 有歌词时明说「独立支路」", "独立支路" in text, text[:200])
    bare = "\n".join(pipe.build_plan(_request(20.0)).warnings)
    check("C4 没给歌词 ⇒ 明说「纯器乐」（**不报错** ✓，但不能不说 ✓✗）",
          "纯器乐" in bare, bare[:200])


# ───────────────────────── D 清单口径 ─────────────────────────


def case_inventory() -> None:
    """★ 三件**逐件必需** ＋ **VAE 那一格叫 `audio_vae_weights`** ✓✗。

    ⚠️ 这组的价值在**第二条** ✓：`assembly_for_stage()` 原来只有"视频 VAE / 图像 VAE"两选一 ✓✗
    ⇒ Music3 的 DAV 会被塞进 `image_vae_weights` ✓、而装配线要 `audio_vae_weights` ✓
    ⇒ 报「缺 `audioVaeWeights`」✓，读起来像"清单里没有 DAV"✓✗ —— 其实清单里**有** ✓。
    """
    ready = inv.readiness("music3")
    check("D1 `readiness(\"music3\")` 三件齐 ✓ 且 ready=True",
          ready["ready"] is True and not ready["missingRequired"] and not ready["brokenRequired"],
          {k: ready.get(k) for k in ("ready", "missingRequired", "brokenRequired")})
    check("D2 必需件总重 ≈ 13.34 GiB（4.58 ＋ 8.57 ＋ 0.20 ✓，容 ±0.05）",
          abs(float(ready["requiredWeightsGiB"]) - 13.34) < 0.05, ready["requiredWeightsGiB"])
    required = [c for c in ready.get("components") or [] if c.get("required")]
    check("D3 `requireAll` 生效：三件**全部**被判成必需（不是 0 件）",
          len(required) == 3 and {c.get("key") for c in required} == {"dit_music3", "te_music3", "vae_music3"},
          [c.get("key") for c in required])
    assembly = inv.assembly_for_stage("music3")
    check("D4 装配格是 `audio_vae_weights` ✗不是 `image_vae_weights`",
          "audio_vae_weights" in assembly and "image_vae_weights" not in assembly,
          sorted(assembly))
    check("D5 装配**不收** `tokenizer_path`（词表焊在语义塔里 ✓）",
          "tokenizer_path" not in assembly, sorted(assembly))
    check("D6 槽位表：music3→audio ✓ / h3→video ✓ / 其余回 image ✓",
          (inv._SLOT_BY_STAGE_VAE.get("music3"), inv._SLOT_BY_STAGE_VAE.get("h3"),
           inv._SLOT_BY_STAGE_VAE.get("wan22", inv._DEFAULT_VAE_SLOT)) ==
          ("audio_vae_weights", "video_vae_weights", "image_vae_weights"),
          (inv._SLOT_BY_STAGE_VAE, inv._DEFAULT_VAE_SLOT))


# ───────────────────────── E 接线顺序 ─────────────────────────


def case_wiring_order() -> None:
    """★ 三处**顺序**判据 ✓✗ —— 顺序换了**不报错** ✓，只是静默走错路 ✓。

    ① ``latent_shape`` 的 3 维分支必须在 ``vae_scale`` 闸门**之前** ✓：Music3 结构里
       ``vae_scale`` 万一缺 ⇒ 先判它会让整族掉进 ``None`` ✓ ⇒ 一维占位 ✓ ⇒
       采样第 1 步炸 ``latents 要 (B,C,L)，实际 (512,)`` ✓（报错指向 DiT ✓、根因在形状函数 ✓）。
    ② ``decode`` 的 ``music3-real`` 早退必须在**开瓦片/reshape 那段之前** ✓：那整段是
       ``(B,T,3,H,W)`` 视频口径 ✓，音频**没有帧** ✓。
    ③ ``write`` 的纯音频分支必须在 **5 维之后**（H3 是「mp4 ＋ 另落 wav」✓）
       且**在 JSON 兜底之前** ✓：不然真解出了波形 ✓、回执照样写"占位产物"✓、盘上只有一个
       ``torch_stub_seed*.json`` ✓✗（与 4 维 PNG 那次同一个坑 ✓）。
    """
    shape_src = inspect.getsource(tb.TorchBackend.latent_shape)
    i3 = shape_src.find("latent_dims == 3")
    iscale = shape_src.find('_config_value("vae_scale"')
    check("E1 `latent_shape`：3 维分支排在 `vae_scale` 之前 ✓",
          -1 < i3 < iscale, (i3, iscale))
    mode_src = inspect.getsource(tb.TorchBackend._latent_mode)
    check("E2 `_latent_mode` 也认 3 维为真形状（自述不许与做法相反 ✓）",
          "latent_dims == 3" in mode_src and "placeholder-1d" in mode_src, mode_src[:200])
    dec_src = inspect.getsource(tb.TorchBackend.decode)
    i_audio = dec_src.find('self._vaeKind == "music3-real"')
    i_tiling = dec_src.find("alpha_dropped")
    check("E3 `decode`：music3 早退排在视频口径那段之前 ✓",
          -1 < i_audio < i_tiling, (i_audio, i_tiling))
    write_src = inspect.getsource(tb.TorchBackend.write)
    i_5d = write_src.find("len(tuple(frames.shape)) == 5")
    i_wav = write_src.find("这就是成品音频")
    i_stub = write_src.find('f"torch_stub_seed{int(request.seed)}.json"')
    check("E4 `write`：纯音频分支在 5 维之后 ✓、JSON 兜底之前 ✓",
          -1 < i_5d < i_wav < i_stub, (i_5d, i_wav, i_stub))
    enc_src = inspect.getsource(tb.TorchBackend.encode_text)
    check("E5 `encode_text` 认 `_textKind == \"music3-ar\"`（不按 `family ==` 判 ✓）",
          'music3-ar' in enc_src and "_textKind" in enc_src, enc_src[-400:])
    cond_src = inspect.getsource(tb.TorchBackend._encode_music3_condition)
    check("E6 条件帧数**从 te 的常量算**（代码里没有手抄的 25 / 32768）",
          "AUDIO_FRAMES_PER_SECOND" in cond_src and "* 25" not in cond_src
          and "32768" not in cond_src.replace("32768 宽", "").replace("(F, 32768)", ""),
          cond_src[:200])
    check("E7 条件编完必须**当场核帧数**（≠ 就拒，不许零填充 ✓✗）",
          "got != cond" in cond_src and "补零" in cond_src, cond_src[:160])


# ───────────────────────── F 装配线拒绝面 ─────────────────────────


def case_assembly_refusals() -> None:
    """★ 装配线**只碰得到三格** ✓，两处拒绝面必须"拒得对、说得清" ✓。

    ⚠️ 这里用 ``object.__new__(EngineRuntime)`` ✓：这两条判据都在**函数最前面** ✓
    （给词表就拒 ✓ / 缺格先报 ✓），一次属性都不用摸 ✓ ⇒ 不必真起服务、也不必装 13 GiB ✓。
    """
    ghost = object.__new__(rt.EngineRuntime)
    err = _error_of(lambda: ghost._ensure_music3_loaded(
        key=("music3",), dit_path="a.safetensors", text_weights="b.safetensors",
        audio_vae_weights="c.safetensors", tokenizer_path="tokenizer.json"))
    check("F1 给 `tokenizerPath` ⇒ 当场拒 ✓（不许「收下但不用」✗）",
          "不收" in err and "tokenizerPath" in err, err[:200])
    err2 = _error_of(lambda: ghost._ensure_music3_loaded(
        key=("music3",), dit_path=None, text_weights=None, audio_vae_weights=None))
    check("F2 三格全缺 ⇒ 首行**只列真缺的三格** ✓ 且不点名 `tokenizerPath` ✗",
          "audioVaeWeights" in err2 and "ditPath" in err2 and "textWeights" in err2
          and "tokenizerPath" not in err2.split("\n")[0], err2.split("\n")[0])
    err3 = _error_of(lambda: ghost._ensure_music3_loaded(
        key=("music3",), dit_path="a.safetensors", text_weights=None,
        audio_vae_weights="c.safetensors"))
    check("F3 只缺语义塔 ⇒ 首行只点名它 ✓（不把给过的也列一遍 ✗）",
          "textWeights" in err3.split("\n")[0] and "ditPath" not in err3.split("\n")[0], err3.split("\n")[0])
    src = inspect.getsource(rt.EngineRuntime._ensure_music3_loaded)
    #: 注释里点名这一格是**允许的**（那正是"为什么不回读"的说明 ✓）—— 判据只盯**代码**行 ✓。
    code_lines = [ln for ln in src.splitlines()
                  if "real_audio_vae" in ln and not ln.strip().startswith("#")]
    check("F4 这一族**不回读** `real_audio_vae`（默认就是 False ✗，按它拒会拦掉每次正常调用 ✓✗）",
          not code_lines, code_lines[:3])
    check("F5 装配期把 DiT 的潜通道与 DAV 的通道**对一次**（两处各自都对、凑起来不对才最贵 ✗）",
          "dit_channels" in src and "vae_channels" in src, "")
    check("F6 这一族**不声明** phase / chain（13.34 GiB 同驻 ✓，多一层换装就多一层失败面 ✗）",
          "declare_chain(" not in src and "phaseParts" not in src.replace("不登记 `phaseParts`", ""), "")


# ───────────────────────── G 路由键表 ─────────────────────────


def case_router() -> None:
    """★ ``lyrics`` 曾经**不在** `_REQUEST_KEYS` 里 ✓✗ —— 前端送了、后端**静默丢掉** ✓✗。

    症状是最坏的一类 ✓：不报错 ✓、不崩 ✓、只是**没唱** ✓，而且 ``plan.warnings``
    还会理直气壮地写「没给歌词 ⇒ 按纯器乐出」✓✗ —— 用户以为自己送了歌词 ✓。
    """
    keys = router_engine._REQUEST_KEYS
    check("G1 `lyrics` 在路由键表里 ✓", keys.get("lyrics") == "lyrics", keys.get("lyrics"))
    check("G2 `seconds` 也在（音频没有「像素帧」可数 ✓，时长只认它 ✓）",
          keys.get("seconds") == "seconds", keys.get("seconds"))
    #: `/engine/stages` 的档位表是函数内的一段字面量 ✓（没有模块级常量可读 ✓）⇒ 读源码判 ✓。
    stages_src = inspect.getsource(router_engine.engine_stages)
    check("G3 阶段表里有 `music3` ✓（`pipe.MUSIC3_STAGE` 出现在那张表里 ✓）",
          "pipe.MUSIC3_STAGE" in stages_src, stages_src[:200])
    label_line = next((ln.strip() for ln in stages_src.splitlines()
                       if "pipe.MUSIC3_STAGE" in ln and "音乐" in ln), "")
    check("G4 标签**写明产物是音频**（前端默认「有 stage 就有画面」✗ ⇒ 必须能一眼看出 ✓）",
          "音频" in label_line, label_line)
    reqs = router_engine._REQUEST_KEYS
    check("G5 别族字段没被顺手改（`stage` 仍在 ✓、`prompt` 仍在 ✓）",
          reqs.get("stage") == "stage" and reqs.get("prompt") == "prompt", (reqs.get("stage"), reqs.get("prompt")))


def main() -> int:
    for case in (case_constants, case_plan_math, case_prose, case_inventory,
                 case_wiring_order, case_assembly_refusals, case_router):
        print(f"\n=== {case.__name__} ===")
        case()
    passed = sum(1 for ok, _, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\nSUMMARY: {passed}/{total} passed")
    for ok, name, detail in _RESULTS:
        if not ok:
            print(f"  ✗ {name}\n    {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
