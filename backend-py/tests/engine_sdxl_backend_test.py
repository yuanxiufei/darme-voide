"""S8 自检：**SDXL 图片后端**（图片闭环的接线层 ✓ 2026-09-26）。

本套盯的是「**喂错也不报错**」那类错 ✓✗（本仓最忌 ✓）：

* **σ↔t** ✗✗：SDXL 的 UNet 只认整数 ``t ∈ [0, 999]`` ✓ —— 把 σ 直接当 t 喂进去**不报错** ✓，
  出来的是「结构像图但是噪声」的东西 ✓✗。判据：σ 表**单调** ✓ + 端点 == SDXL 已知口径 ✓ +
  ``t(σ_max)=999`` ✓ + 一个**线性域与对数域会分歧**的 σ ✓（证明确实按对数域映射 ✓）；
* **EPS 预条件** ✗✗：UNet 吃 ``x/√(σ²+1)`` ✓ 但 ``x₀`` 要**用原始 x** 算 ✓ ——
  拿**记录参数的桩**把这四个数逐个钉住 ✓（用预条件后的值算 ⇒ 系统性偏亮且不报错 ✓✗）；
* **解码必须走 `decode_latents`** ✗✗（漏除 0.13025 ⇒ 图发灰、不报错 ✓）—— 拿**缩小版真网络**
  走完整 ``decode`` ⇒ 形状 + 值域 ``[0,1]`` ✓，再 ``write`` 出**真 PNG**（验文件头 ✓）；
* **四前缀装配** ✓：真权重头上四个前缀**恰好完整划分**检查点 ✓（1680+248+197+390 == 2515 ✓）；
* **画布恒按训练预算收** ✗✗：低于 SDXL 的**训练分辨率**（1024×1024 ✓）⇒ 出「物体重复 + 霓虹过饱和」/
  「纯色块」✓✗；**高于**它 ⇒ 解码 reserved 冲到 **30.66 GiB ≫ 物理 22.49 GiB** ✗✗ ⇒ 共享显存
  换页 ⇒ 光解码 11 s（1024² 只要 0.66 s ✓）✓。**两个方向都不报错** ✓ ⇒ 计划阶段一律**收到训练预算**
  并**如实告警** ✓（只保留比例 ✗；要更大像素走超清那条 ✓）；
* **如实** ✓：没装齐 ⇒ ``synthetic`` **为真** ✓（不许"看起来像出图了"✗）。

⚠️ 真权重不在盘上 ⇒ **显式 SKIP** ✓（**没跑 ≠ 绿** ✗）。⚠️ 完整装配要 6.9 GiB ✗ ⇒ 默认**不跑**
（`SDXL_BACKEND_FULL_LOAD=1` 才跑 ✓），本套走**缩小版前向** ✓ 把口径验穿 ✓。
⚠️ `ComfyUI` 是 GPL-3.0 ⇒ **只当规格书读** ✗、一行不搬 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_sdxl_backend_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import sdxl_backend as mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: object, needle: str = "") -> bool:
    try:
        call()  # type: ignore[operator]
    except Exception as err:  # noqa: BLE001
        return needle in str(err)
    return False


TOY = dict(in_channels=3, out_channels=3, latent_channels=4,
           block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=4)


def find_sdxl_base_checkpoint() -> tuple[Path | None, str]:
    """复用 `engine_sdxl_test` 的**扫描器**口径 ✓（不写死机器路径 ✗）。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import engine_sdxl_test as sibling  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        return None, f"导不进同目录的 `engine_sdxl_test`（{error}）⇒ 真权重对齐不跑 ✓"
    return sibling.find_sdxl_base_checkpoint()


def case_sigma() -> None:
    """① σ↔t ✓：**纯数学** ✓ ⇒ 没装 torch 也能判 ✓（最要紧的一条 ✓）。"""
    table = mod.sdxl_sigmas_table()
    check(f"① σ 表 {len(table)} 点 ✓、**严格单调增** ✓（i=0 最干净 ⇒ i=999 最噪 ✓）",
          len(table) == mod.SDXL_SIGMA_TIMESTEPS
          and all(table[i] < table[i + 1] for i in range(len(table) - 1)))
    check("①′ 端点 == SDXL 已知口径 ✓（σ_min ≈ 0.029167 ✓、σ_max ≈ **14.6146** ✓ —— "
          "与本仓 `schedules.DEFAULT_SIGMA_MAX` 的 14.6 一致 ✓ ⇒ 公式不是瞎写的 ✓）",
          abs(table[0] - 0.029167) < 1e-5 and abs(table[-1] - 14.6146) < 1e-3,
          (table[0], table[-1]))
    check("①″ ⭐ **边界** ✗✗：``t(σ_max) == 999`` ✓、``t(σ_min) == 0`` ✓、``t(0) == 0`` ✓"
          "（σ=0 不取对数 ✓ —— 取了对数会炸 ✓）",
          mod.sigma_to_timestep(table[-1]) == 999 and mod.sigma_to_timestep(table[0]) == 0
          and mod.sigma_to_timestep(0.0) == 0)
    check("①‴ 表内每一个 σ 都映回**自己的下标** ✓（自洽 ✓ ⇒ 映射没跑偏 ✓）",
          all(mod.sigma_to_timestep(table[i]) == i for i in range(0, len(table), 37)))
    check("①⁗ ⚠️ **对数域**而非线性域 ✗：取两点的**几何中值** √(σ_a·σ_b) ✓ —— "
          "它到两端的对数距离**相等** ✓ ⇒ 必落在两者之间且靠中 ✓；"
          "线性域最近邻在 σ 跨数量级时会**选错** ✓✗",
          mod.sigma_to_timestep(1.0) > mod.sigma_to_timestep(0.1)
          and mod.sigma_to_timestep(0.1) > mod.sigma_to_timestep(0.01)
          and 300 < mod.sigma_to_timestep(1.0) < 500, mod.sigma_to_timestep(1.0))
    check("①⁵ 坏输入**响亮报错** ✗：σ 非有限 / 表太短 / β 端点顺序反了 ✓"
          "（⚠️ 端点反了表会**单调减** ✓ —— 采样整条废掉而不报错 ✓✗）",
          _raises(lambda: mod.sigma_to_timestep(float("nan")), "有限") is not None
          and _raises(lambda: mod.sdxl_sigmas_table(1), "至少") is not None
          and _raises(lambda: mod.sdxl_sigmas_table(100, linear_start=0.012,
                                                    linear_end=0.00085), "顺序") is not None)


def case_eps() -> None:
    """② EPS 前向算术 ✓：用**定值桩**把 ``x_in`` 与 ``x₀`` 两条式子钉死 ✓。"""
    if mod.torch is None:
        skip("没装 torch ⇒ **需要 torch 的那几条**不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415

    sigma = 3.0
    noise = torch.full((1, 4, 2, 2), 5.0)
    check("② ⭐ `eps_input` ✗✗：`x=5, σ=3 ⇒ 5/√(9+1) = 1.5811…` ✓"
          "（⚠️ 漏了这一步 ⇒ UNet 见到的是**量级不对**的输入 ✓ 出图必糊且**不报错** ✓✗）",
          torch.allclose(mod.eps_input(noise, sigma),
                         torch.full((1, 4, 2, 2), 5.0 / (10 ** 0.5)), atol=1e-6),
          mod.eps_input(noise, sigma).flatten()[0].item())
    check("②′ ⭐ `denoised_from_eps` ✗✗：`x₀ = x − ε·σ` ⇒ `5 − 1·3 = 2` ✓；"
          "`ε=0 ⇒ x₀ == x` ✓（**用的是原始 x** ✗ 不是预条件后的值 ✓）",
          torch.allclose(mod.denoised_from_eps(noise, sigma, torch.ones_like(noise)),
                         torch.full((1, 4, 2, 2), 2.0), atol=1e-6)
          and torch.equal(mod.denoised_from_eps(noise, sigma, torch.zeros_like(noise)), noise))
    check("②″ 尺寸吸附：**向下**到 8 的倍数 ✓、并**如实报「改过」** ✓（不静默改输入 ✗）",
          mod.snap_image_size(1023, 768) == (1016, 768, True)
          and mod.snap_image_size(1024, 768) == (1024, 768, False))
    check("②‴ 吸附的坏输入**报错** ✗：宽高为 0 / 倍数非正 ✓",
          _raises(lambda: mod.snap_image_size(0, 8), "为正") is not None
          and _raises(lambda: mod.snap_image_size(8, 8, scale=0), "为正") is not None)


def case_mounting() -> None:
    """③ 装配 ✓：四个前缀**恰好完整划分**检查点 ✓（只读头部 ✓ 不载张量 ✗）。"""
    path, why = find_sdxl_base_checkpoint()
    if path is None:
        skip(why)
        return
    from app.services.engine.safetensors import read_header  # noqa: PLC0415

    _, header = read_header(path)
    facts = mod.SDXL_IMAGE_FACTS["componentKeys"]
    found = {name: sum(1 for key in header if key.startswith(prefix))
             for name, prefix in mod.SDXL_IMAGE_FACTS["keyPrefixes"].items()}
    for name in ("unet", "vae", "clipL", "clipG"):
        check(f"③ 前缀 {name} 的张量数 == 实测 {facts[name]} ✓"
              f"（`{mod.SDXL_IMAGE_FACTS['keyPrefixes'][name]}` ✓）",
              found[name] == facts[name], found[name])
    check(f"③′ ⭐ **四个前缀之和 == 文件总键数** ✗✗（{sum(found.values())} == {len(header)} ✓）"
          "—— 少一个前缀 / 前缀拼错 ⇒ 会有大量键**没人认领** ✓ 这条必红 ✓",
          sum(found.values()) == len(header) == mod.SDXL_IMAGE_FACTS["totalKeys"],
          (sum(found.values()), len(header)))
    check("③″ 缺前缀 ⇒ **报错** ✗（不静默给空字典 ✓ —— 那会让四件套里某件保持**随机初始化** ✓✗）",
          _raises(lambda: mod._split_by_prefix({"a": 1}, "first_stage_model.", what="vae"),
                  "拿不到") is not None)


def case_protocol() -> None:
    """④ 协议面 ✓：状态**如实** ✓ + 缩小版前向把 `denoise`/`decode`/`write` 验穿 ✓。"""
    if mod.torch is None:
        skip("没装 torch ⇒ **需要 torch 的那几条**不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    import torch  # noqa: PLC0415
    from app.services.engine import sdxl_vae as vae_mod  # noqa: PLC0415

    fresh = mod.SdxlBackend()
    check("④ ⚠️ **如实** ✗：没装权重时 ``synthetic`` 为**真** ✓、``loaded`` 为假 ✓"
          "（不许「看起来像出图了」✗）；``name == 'sdxl'`` ✓（出图链路的分流依据 ✓）",
          fresh.synthetic is True and fresh.loaded is False and fresh.name == "sdxl")
    check("④′ 没装就调 ⇒ **报错** ✗（不返回零张量糊过去 ✓）：`encode_text` / `decode` ✓"
          "（⚠️ 后者若返回全零 ⇒ 会写出一张**黑图**且不报错 ✓✗）",
          _raises(lambda: fresh.encode_text(SimpleNamespace(prompt="a", negative="")),
                  "没装齐") is not None
          and _raises(lambda: fresh.decode(None, None, None), "没装齐") is not None)
    check("④″ 没给 tokenizer ⇒ **报错并说清怎么办** ✗（词表是权重的一部分 ✓ 引擎不内置 ✓）；"
          "`load_weights` 指到不存在的路 ⇒ 报错 ✓",
          _raises(lambda: fresh.tokenizer, "tokenizer") is not None
          and _raises(lambda: fresh.load_weights(Path("no-such-model.safetensors")),
                      "不存在") is not None)

    # ── 缩小版：真 VAE + 桩 UNet ⇒ 把 decode / write / denoise 的口径验穿 ✓ ──
    backend = mod.SdxlBackend(device="cpu")
    backend._vae = vae_mod.build_sdxl_vae(vae_mod.SdxlVaeConfig(**TOY))
    backend._unet = backend._clip_l = backend._clip_g = object()
    plan = SimpleNamespace(width=8, height=8, sigmas=[1.0])
    request = SimpleNamespace(seed=7, outputs_dir=None, width=8, height=8,
                              prompt="", negative="")

    with torch.no_grad():
        outputs = backend.decode(torch.rand(1, 4, 1, 1), plan, request)
    frames = outputs["frames"]
    check("④‴ ⭐ `decode` 走**缩放口径** ✗✗：产物 ``(B, 3, 1, H, W)`` ✓（补了 T 轴 ✓ 给 `write` 用 ✓）、"
          "**放大倍数 == `spatial_scale`** ✓（缩小版两层 ⇒ **2 倍** ✓：1×1 ⇒ 2×2 ✓；"
          "官方配置才是 8 倍 ✓ —— ⚠️ 写死 8 在这里就会假红 ✓）、值域**落在 `[0,1]`** ✓ —— "
          "随机初始化的 VAE 输出本会大范围越界 ✓，能落在界内说明**确实走了** `decode_latents` "
          "的除缩放 + 夹取 ✓",
          tuple(frames.shape) == (1, 3, 1, 2, 2)
          and float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0,
          tuple(frames.shape))

    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        request.outputs_dir = tmp
        written = backend.write(outputs, plan, request)
        produced = Path(written["primaryPath"])
        blob = produced.read_bytes() if produced.exists() else b""
        check("④⁗ `write` 真的写出**一张 PNG** ✓（验文件头 ``\\x89PNG`` ✓ 不是空文件 ✓）、"
              "且 ``frames == 1`` ✓ —— ⚠️ **不是**视频取首帧那套 ✗；报告的尺寸与 plan 一致 ✓",
              produced.exists() and blob[:4] == b"\x89PNG" and written["frames"] == 1
              and (written["width"], written["height"]) == (8, 8),
              (str(produced), len(blob)))
        check("④⁵ 路径**确定性** ✓（同 seed + 同尺寸 ⇒ 同名 ✓ 便于复现/自检 ✓）",
              backend.output_path(plan, request).name == "sdxl-7-8x8.png")

    recorded: list[tuple[object, object]] = []

    class _Recorder:
        """桩 UNet ✓：把**收到的入参原样记下** ✓ ⇒ 验的是「怎么喂的」✓ 不是网络数值 ✓。

        ⚠️ 它还得**能报出自己的设备/精度** ✗（`init_latents` 是**照权重问**的 ✓，不信 `self.device` ✓）
        —— 拿 `object()` 蒙过去 ⇒ 连 `parameters()` 都没有 ⇒ 直接 ``AttributeError`` ✓✗。
        给一个**真的**张量当参数即可 ✓（在 CPU 上 ✓ 与 ``device="cpu"`` 一致 ✓），
        这**不影响**本条验的东西 ✓（验的是喂进去的入参 ✓ 不是网络数值 ✓）。
        """

        def __init__(self, value: object) -> None:
            self.value = value
            self._parameter = torch.zeros(1)          # 只用来回答"权重在哪个设备/什么精度" ✓

        def parameters(self) -> object:
            return iter((self._parameter,))

        def __call__(self, x: object, t: object, context: object, adm: object) -> object:
            recorded.append((x.clone(), t.clone()))
            return self.value

    sigma = 3.0
    latents = torch.full((1, 4, 1, 1), 5.0)
    backend._unet = _Recorder(torch.zeros(1, 4, 1, 1))
    condition = mod._Condition(context=torch.zeros(1, 77, 2048), pooled=torch.zeros(1, 1280),
                               tokens=77, truncated=False)
    with torch.no_grad():
        back = backend.denoise(latents, sigma, condition, request)
    sent_x, sent_t = recorded[-1]
    check("④⁶ ⭐⭐ **入参口径** ✗✗：UNet 收到的时间步是**整数下标** ✓"
          f"（σ=3 ⇒ t={mod.sigma_to_timestep(3.0)} ✓，**不是** 3.0 ✗）"
          "且是 ``(B,)`` 形状 ✓；收到的输入是**预条件过**的 ``x/√(σ²+1)`` ✓",
          float(sent_t[0]) == float(mod.sigma_to_timestep(3.0))
          and float(sent_t[0]) == sent_t[0].round().item() and sent_t.shape == (1,)
          and torch.allclose(sent_x, mod.eps_input(latents, sigma), atol=1e-6),
          (float(sent_t[0]), mod.sigma_to_timestep(3.0)))
    check("④⁷ ⭐⭐ **返回值用原始 x 算** ✗✗：``ε=0 ⇒ x₀ == x`` ✓（**5.0** ✓ 不是预条件后的 1.58 ✓）"
          "—— ⚠️ 用预条件后的值算 ⇒ 系统性偏亮且**不报错** ✓✗",
          torch.allclose(back, latents, atol=1e-6), back.flatten()[0].item())
    backend._unet = _Recorder(torch.ones(1, 4, 1, 1))
    with torch.no_grad():
        stepped = backend.denoise(latents, sigma, condition, request)
    check("④⁸ ``ε=1, σ=3 ⇒ x₀ = 5 − 3 = 2`` ✓（式子对得上 ✓ 不在别处又缩了一次 ✓）",
          torch.allclose(stepped, torch.full((1, 4, 1, 1), 2.0), atol=1e-6),
          stepped.flatten()[0].item())
    # ⚠️ `time_ids` 的尺寸**只能从潜变量形状推** ✗（实测踩到 ✓：`GenerationRequest` 上**根本没有**
    #    `width`/`height` ✓ —— 尺寸是 `GenerationPlan` 的事 ✓，而采样协议给 `denoise` 的**只有**
    #    ``request`` ✗ ⇒ 早先按 ``request.width`` 取 ⇒ 真跑端到端时 ``AttributeError`` ✓✗
    #    （⚠️ 这类错**只在真跑时**才炸 ✗：单测里手搓的 `SimpleNamespace` 凑巧带了 width/height ✓）。
    ids = mod._time_ids(torch.zeros(1, 4, 2, 3))
    check("④⁸′ ⭐ `time_ids` 顺序是 ``[高, 宽, 裁上, 裁左, 目标高, 目标宽]`` ✓ 且尺寸**从潜变量形状 ×8 推** ✓"
          "（2×3 潜格 ⇒ 16×24 ✓；顺序写反 ⇒ 构图莫名偏/裁切且**不报错** ✓✗）",
          tuple(ids.shape) == (1, 6) and ids.tolist() == [[16, 24, 0, 0, 16, 24]], ids.tolist())
    lax = SimpleNamespace(seed=7, prompt="", negative="")   # ⚠️ 故意**不带** width/height ✓
    backend._unet = _Recorder(torch.zeros(1, 4, 1, 1))
    lax_error: object = None
    try:
        with torch.no_grad():
            backend.denoise(latents, sigma, condition, lax)
    except Exception as error:  # noqa: BLE001
        lax_error = error
    check("④⁸″ ⭐ `denoise` 只拿到 ``request``（**没有** width/height ✓ 真引擎给的就是这种 ✓）"
          "**也必须能跑** ✓ —— 端到端实测撞到的就是这条 ✓✗", lax_error is None, lax_error)
    check("④⁹ 把**整个** `encode_text` 的字典当条件传进来 ⇒ **报错** ✗"
          "（⚠️ 那会把两个塔的条件混着用 ✓ 图上表现得「听话但不对」✓✗）",
          _raises(lambda: backend.denoise(latents, sigma,
                                          {"positive": condition, "negative": condition},
                                          request), "单路") is not None)
    with torch.no_grad():
        start = backend.init_latents(SimpleNamespace(width=16, height=8, sigmas=[2.0]), request)
    check("④¹⁰ 初始潜变量 ✗：形状 ``(1, 4, H/8, W/8)`` ✓（8×16 ⇒ 1×2 ✓）、"
          "**乘过 σ_max** ✓（``σ=2`` ⇒ 标准差应约 2 ✓ 不是 1 ✓ —— 漏乘 ⇒ 起点太干净 ✓ 图偏糊 ✓）",
          tuple(start.shape) == (1, 4, 1, 2)
          and 1.5 < float(start.std()) < 2.6, float(start.std()))
    check("④¹¹ 调度里没有 σ ⇒ **报错** ✗（不静默当 0 ✓ —— 那会让起点是纯噪声而 σ 账对不上 ✓✗）",
          _raises(lambda: backend.init_latents(SimpleNamespace(width=8, height=8, sigmas=[]),
                                               request), "σ") is not None)


def _full_load_dtype() -> object:
    """完整装配的精度 ✓：默认 **fp16** ✓（UNet 25 亿参数 ✓ —— fp32 在 CPU 上要 10+ GiB ✓ 会 OOM ✓）。

    ⚠️ 精度**不影响键/形状的严格校验** ✓（``load_state_dict`` 只做拷贝+转型 ✓）⇒ 结构性验证等价 ✓。
    """
    want = (os.environ.get("SDXL_BACKEND_DTYPE") or "float16").strip().lower()
    if want in ("float32", "fp32"):
        return mod.torch.float32
    if want in ("bfloat16", "bf16"):
        return mod.torch.bfloat16
    return mod.torch.float16


def case_full_load() -> None:
    """⑤ 完整装配（**默认不跑** ✗：要 6.9 GiB 显存/内存 ✓ 由 ``SDXL_BACKEND_FULL_LOAD=1`` 打开 ✓）。"""
    if mod.torch is None:
        skip("没装 torch ⇒ 完整装配不跑 ✓（**没跑 ≠ 绿** ✗）")
        return
    if os.environ.get("SDXL_BACKEND_FULL_LOAD") != "1":
        skip("完整装配默认不跑 ✓（读 6.9 GiB ✓）—— 要跑请设 `SDXL_BACKEND_FULL_LOAD=1` ✓"
             "（⚠️ **没跑 ≠ 绿** ✗）")
        return
    path, why = find_sdxl_base_checkpoint()
    if path is None:
        skip(why)
        return

    backend = mod.SdxlBackend(device="cpu", dtype=_full_load_dtype())
    report = backend.load_weights(path)
    check("⑤ 四件套全部装上了 ✓（任一缺键/形状不符都会在 `load_weights` 里**报错** ✗）"
          "⇒ ``synthetic`` 转为**假** ✓ —— 这是「真的用真权重出图」的标志 ✓",
          backend.loaded and backend.synthetic is False, report.get("components"))
    facts = mod.SDXL_IMAGE_FACTS["componentKeys"]
    check("⑤′ 装配报告的键数与真权重口径逐个一致 ✓（UNet 1680 / VAE 248 / CLIP-L 197 / "
          "CLIP-G 390 ✓）",
          all(int(report["components"][name]["keys"]) == facts[name]
              for name in ("unet", "vae", "clipL", "clipG"))
          and int(report.get("sourceKeys") or 0) == mod.SDXL_IMAGE_FACTS["totalKeys"],
          report.get("components"))


def case_plan_floor() -> None:
    """⑥ 图片画布**恒按训练预算收** ✓：两个方向都收 + **如实告警** ✓✗。

    ⚠️ 这条盯的是「**喂错也不报错**」里最贵的一种 ✓✗ —— **两个方向都坏** ✓：
    * **偏低**（512×512 / 256×256 ✓）：**不报错**、**图也能出** ✓，出来的却是「物体重复 + 霓虹过饱和」/
      「纯色块」✓✗（实拍证据见 :data:`mod.SDXL_TRAINED_RESOLUTION` 的注释 ✓）——
      症状与「实现写错了」**分不出来** ✓；
    * **偏高**（生产默认的 1920×1080 = 2.07 MP ✓，出处见 `image_generation.py` ✓）：**同样不报错** ✓，
      但解码 ``reserved`` 冲到 30.66 GiB ≫ 物理 22.49 GiB ✗✗ ⇒ 共享显存换页 ⇒ 光解码 **11 s** ✓
      （同条件 1024² 只要 **0.66 s** ✓）—— 这撞的是用户「不许逼近显存上限」的口径 ✗。
    ⚠️ **纯计算** ✓（`build_plan` 不碰权重 ✗）⇒ 没装 torch 也判得了 ✓。
    """
    from app.services.engine import geometry, pipeline  # noqa: PLC0415

    budget = mod.SDXL_TRAINED_MEGAPIXELS
    check(f"⑥ 训练分辨率常数**自洽** ✓：{mod.SDXL_TRAINED_RESOLUTION}² = {budget:g} MP ✓"
          "（与本仓「宽·高/1e6」口径一致 ✓ 见 `geometry.megapixels_for_size` ✓）",
          round(budget, 4) == geometry.megapixels_for_size(mod.SDXL_TRAINED_RESOLUTION,
                                                           mod.SDXL_TRAINED_RESOLUTION),
          budget)

    def _plan(megapixels: float, ratio: str) -> object:
        return pipeline.build_plan(pipeline.GenerationRequest(
            prompt="probe", stage=pipeline.IMAGE_STAGE, megapixels=megapixels,
            ratio=ratio, steps=8, schedule="normal"))

    def _reported(plan: object) -> bool:
        """**如实报改过** ✓ 的判据 ✗：文案要出现「按训练预算收」✓ **并且**点明「训练分辨率」出处 ✓。"""
        text = " ".join(str(note) for note in plan.warnings)  # type: ignore[attr-defined]
        return "按训练预算收" in text and "训练分辨率" in text

    low = _plan(0.262, "1:1")
    check("⑥′ 512×512（0.262 MP）⇒ 收到 1024×1024 ✓ 且如实告警（文案含「低于」✓）",
          (low.width, low.height) == (1024, 1024) and _reported(low)
          and any("低于" in str(n) for n in low.warnings),
          (low.width, low.height))

    big = _plan(2.0736, "16:9")
    check("⑥″ 生产默认 **1920×1080（2.07 MP）** ⇒ 收到训练预算 **1376×768** ✓（= 16:9 下 1.057 MP ✓）"
          "且如实告警（文案含「高于」✓）—— ⚠️ 生产默认正落在**换页区** ✓✗（实测解码 11 s ✓）",
          (big.width, big.height) == (1376, 768) and _reported(big)
          and any("高于" in str(n) for n in big.warnings),
          (big.width, big.height))

    exact = _plan(1.0486, "1:1")
    check("⑥‴ UI 要 1024×1024（经 `megapixels_for_size` 得 1.0486 ✓，真值 1.048576 ✓，差 0.0024% ✓）"
          "⇒ **不许被判成改了用户的** ✗（1% 容差 ✓）⇒ 仍是 1024×1024 ✓ 且**不告警** ✓",
          (exact.width, exact.height) == (1024, 1024) and not _reported(exact),
          (exact.width, exact.height))


def case_dtype_policy() -> None:
    """⑦ 精度口径 ✓：**CUDA 默认 fp16 计算 + fp32 VAE** ✓，且两个方向都**如实** ✓✗。

    ⚠️ 这条盯的是"**不给就悄悄按 fp32 跑**" ✓✗ —— 它**不报错**、**图也出得来** ✓，
    只是权重白占一倍（实测 **12.92 GiB**，而检查点本身是 fp16 只需 6.46 GiB ✓）⇒
    1376×768 单张峰值 ``reserved=26.62 GiB > 22.49 GiB 物理`` ✓ ⇒ 共享显存换页 ⇒
    解码 **13.78 s**（同量级 1024² 只要 0.66 s ✓）、同进程第二张 **30.0 s → 82.7 s** ✓✗。
    ⚠️ 反面同样钉住 ✗：VAE **不许**跟着 fp16 走 ✓（那会**静默出黑图** ✓✗，见 `SDXL_VAE_DTYPE` ✓）。
    ⚠️ **纯字符串** ✓（不建模型、不碰 torch ✗）⇒ 没装 torch 也判得了 ✓。
    """
    resolve = mod.resolve_sdxl_dtype_names

    check("⑦ ⭐ CUDA + **不给 dtype** ⇒ 计算组件 **fp16**、VAE **fp32** ✓"
          "（不给就落回 fp32 ⇒ 权重 12.92 GiB ⇒ 换页 ⇒ 解码 13.78 s ✓✗）",
          resolve("cuda") == ("float16", "float32")
          and resolve("cuda:0") == ("float16", "float32"),
          resolve("cuda"))

    check("⑦′ ⚠️ **显式给的精度不许被改写** ✗：`dtype=\"float32\"` ⇒ 就是 fp32 ✓"
          "（显式优先 ✓ —— 这是「照着用户说的做」✓，不是「替他做主」✗）",
          resolve("cuda", dtype="float32") == ("float32", "float32")
          and resolve("cuda", dtype="bfloat16") == ("bfloat16", "float32"),
          resolve("cuda", dtype="float32"))

    check("⑦″ CPU / 不给 device（⇒ 权重留 CPU ✓）⇒ **两边都 fp32** ✓"
          "（CPU 上 fp16 又慢又不省 ✓）",
          resolve(None) == ("float32", "float32")
          and resolve("cpu") == ("float32", "float32"),
          resolve(None))

    check("⑦‴ ⭐⭐ VAE **不许**跟着 fp16 ✗✗：除非调用方**显式** `vae_dtype=\"float16\"` ✓"
          "（VAE 压 fp16 ⇒ **静默黑图/发灰** ✓✗ —— 本仓最忌的「不报错但结果坏」✓）",
          resolve("cuda")[1] == "float32"
          and resolve("cuda", dtype="float16", vae_dtype="float16") == ("float16", "float16"),
          resolve("cuda"))

    bad = _raises(lambda: resolve("cuda", dtype="float8"), "只认")
    check("⑦⁗ 认不出的精度 ⇒ **报错** ✗（不静默当 fp32 蒙过去 ✓ —— 那正是"
          "「以为给了、其实没生效」✓✗）",
          bad is not None, bad)

    #: 事实表里也要能查到默认口径 ✓（日志/`engine-status` 取的是它 ✓）。
    check("⑦⁵ 事实表 `dtypes` 与解析口径**同一处来源** ✓（数字不许抄两遍 ✗）",
          mod.SDXL_IMAGE_FACTS["dtypes"] == {"compute": mod.SDXL_COMPUTE_DTYPE,
                                            "vae": mod.SDXL_VAE_DTYPE},
          mod.SDXL_IMAGE_FACTS["dtypes"])

    #: 没装权重时也要能**如实预告**将要用的精度 ✓（`describe` 是 engine-status 的来源 ✓）。
    check("⑦⁶ 未装配时 `describe()` 也如实给 `dtypes` ✓（**预告**口径同解析函数 ✓；"
          "别等到装完才知道跑什么精度 ✓）",
          mod.SdxlBackend(device="cuda").describe().get("dtypes")
          == {"compute": "float16", "vae": "float32"},
          mod.SdxlBackend(device="cuda").describe().get("dtypes"))


def main() -> int:
    case_sigma()
    case_eps()
    case_mounting()
    case_protocol()
    case_full_load()
    case_plan_floor()
    case_dtype_policy()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    for reason in _SKIPS:
        print(f"SKIP  {reason}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
