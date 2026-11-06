"""Qwen-Image **Edit 2511** 参考图通路自检 ✓（2026-11-01 接上 ✓）。

**为什么单独一份** ✗：这条通路横跨三处 ✓——视觉塔（语义 ✓）、VAE（像素 ✓）、DiT（``ref_latents`` ✓）——
任何一处漏接都**不报错**✗，只让"编辑"退化成"按提示词重画" ✓✗。本节把三处的**接线点**逐条钉住 ✓，
真权重那一条（⑧）在盘上有文件时才跑 ✓。

判据口径（**别写松** ✗✗）：

- **面积 ≠ 边长** ✓✗：``1024²`` 对 16:9 是 1368×768 ✓，不是 1024×576 ✓（照边长写 ⇒ 构图被压扁、
  一条错都不报 ✓；1368 这个数**别手算** ✓ —— 第一版手写 1344 就是算错的那次 ✓）；
- 视觉塔 **384² 面积** ✓ 与 VAE **1024² 面积** ✓ 是**两个数** ✗✗（混用只让整图偏色 ✓）；
- ``ref_latents`` 是 ``Sequence[Tensor]`` ✓✗ ⇒ 传单个张量会**按 batch 维遍历** ✓（少包一层列表 ✓）；
- Edit 权重**没给参考图** ⇒ 必须**当场报错** ✗✗（退回纯 t2i 是「能力凭空消失但 0 报错」 ✓）；
- **多图**（2026-11-01 接 ✓）：N 张 ⇒ 模板 N 个槽 ✓ ＋ 嵌入 N 块 ✓ ＋ ``ref_latents`` N 条 ✓✗，
  三处**任何一处**还按单图走都**不报错** ✓，只让第 2..N 张图的指令落空 ✓；
- 多图的**顺序即语义** ✗✗：``Picture i`` 与 ``embeds[i]`` / ``ref_latents[i]`` 必须同序 ✓（乱序不报错 ✓）。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 第一条打印就崩，判据一条都跑不到 ✗
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PROXY_TO_NODE", "0")

from app.services.engine import qwen25vl_vision as vl_mod  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_FAILS: list[str] = []
_SKIPS: list[str] = []
_TOTAL = 0

_TEXT_WEIGHTS = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/text_encoders"
                     "/qwen_2.5_vl_7b_fp8_scaled.safetensors")
_IMAGE_VAE = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/qwen_image_vae.safetensors")


def check(label: str, ok: bool, detail: object = "") -> None:
    global _TOTAL
    _TOTAL += 1
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _skip(label: str, why: str) -> None:
    _SKIPS.append(label)
    print(f"SKIP  {label}  ->  {why}")


#: 一台**小同构**视觉塔的形状 ✓（数值不对拍 ✓，只验"接线与形状" ✓ —— 真塔在 ⑧ 与
#: `engine_qwen25vl_vision_test` 里 ✓）。``head_dim = 32/2 = 16`` 能被 4 整除 ✓。
_SMALL_VIS = {
    "depth": 2, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64,
    "in_channels": 3, "temporal_patch_size": 2, "patch_size": 14,
    "spatial_merge_size": 2, "window_size": 112, "out_hidden_size": 16,
    "fullatt_block_indexes": (1,),
}

#: 视觉塔那一路的真实口径（`qwen25vl_vision.VISION_25VL` 的 out_hidden ✓ = 3584 ✓）。
_VIS_OUT_HIDDEN = int(vl_mod.VISION_25VL["out_hidden_size"])


def _png(width: int, height: int, folder: Path, name: str) -> Path:
    """造一张真 PNG ✓（走真 PIL/读盘那一段 ✓，不拿假张量冒充"参考图 ✓"）。"""
    from PIL import Image  # noqa: PLC0415

    path = folder / name
    Image.new("RGB", (width, height), (40, 90, 160)).save(path)
    return path


# ---------------------------------------------------------------- ① 面积口径


def case_area_size() -> None:
    backend = tb.TorchBackend(device="cpu")
    area = 1024 * 1024

    w, h = backend.area_size(1920, 1080, area, multiple=8)
    #: ⚠️ 期望值**别手算** ✗：``sqrt(1024²/(1920·1080)) = 0.71113`` ⇒ 1365.4/768.0
    #: 各按 8 取整 ⇒ **1368×768** ✓ —— 第一版手写 1344×768 就是算错的数 ✓✗。
    check("① 1024² **面积**对 16:9 ⇒ 1368×768 ✓✗（**不是** 1024×576 那种边长口径 ✓）",
          (w, h) == (1368, 768) and (w, h) != (1024, 576), (w, h))
    check("①′ 面积与目标差不到一格 ✓（±2% ✓）",
          abs(w * h - area) / area < 0.02, w * h / area)
    check("①″ 边长是 ``multiple`` 的倍数 ✓（VAE 要 8 ✓）",
          w % 8 == 0 and h % 8 == 0, (w, h))

    w, h = backend.area_size(1024, 1024, area, multiple=8)
    check("①‴ 正方形 ⇒ 1024×1024 ✓（面积口径在正方形上退化成边长 ✓）",
          (w, h) == (1024, 1024), (w, h))

    w, h = backend.area_size(1, 20000, 384 * 384, multiple=1)
    check("①⁴ 极端长条**不许**算出 0 边 ✓（下限一个 multiple ✓）",
          w >= 1 and h >= 1, (w, h))

    w, h = backend.area_size(320, 180, 384 * 384, multiple=1)
    ratio = w / h
    check("② 视觉口径 384² 对 16:9 仍保持 ≈16:9 ✓（等比 ✓，不是拉成方图 ✓）",
          abs(ratio - 320 / 180) < 0.12, (w, h, round(ratio, 3)))


# ---------------------------------------------------------------- ③ 真读盘


def case_pixels_of_area() -> None:
    backend = tb.TorchBackend(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        image = _png(320, 180, folder, "ref.png")

        pixels, info = backend._pixels_of_area(image, 1024 * 1024, multiple=8)
        check("③ 参考图读成 ``(1, 3, 1, H, W)`` ✓（VAE 的口 ✓）",
              tuple(pixels.shape)[:3] == (1, 3, 1) and pixels.ndim == 5,
              tuple(pixels.shape))
        check("③′ 目标宽高按**面积**算 ✓ 且回执写清原生尺寸 ✓✗（否则构图被压扁也看不出 ✓）",
              info["native"] == [320, 180] and info["scaled"][0] % 8 == 0
              and abs(info["scaled"][0] / info["scaled"][1] - 16 / 9) < 0.12, info)
        check("③″ ``value_range='0..1'`` ⇒ 值域在 [0, 1] ✓",
              float(pixels.min()) >= -1e-6 and float(pixels.max()) <= 1 + 1e-6,
              (float(pixels.min()), float(pixels.max())))

        signed, _ = backend._pixels_of_area(image, 384 * 384, multiple=1, value_range="-1..1")
        check("③‴ ``value_range='-1..1'`` 是**另一个档** ✓（视觉塔与 VAE 各要一档 ✓，混用偏色 ✓）",
              float(signed.min()) < -0.5, float(signed.min()))

        try:
            backend._pixels_of_area(folder / "absent.png", 1024 * 1024)
            check("③⁴ 图不存在 ⇒ 报错 ✓", False, "没报错")
        except tb.TorchBackendUnavailable as err:
            check("③⁴ 图不存在 ⇒ 报错 ✓（不静默拿零图顶上 ✓）", "不存在" in str(err), str(err))


# ---------------------------------------------------------------- ④ Edit 判定


def _backend_with(config: object) -> tb.TorchBackend:
    backend = tb.TorchBackend(device="cpu")
    backend._config = config
    return backend


def case_edit_reference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        image = _png(320, 180, folder, "ref.png")
        request = SimpleNamespace(prompt="把背景换成雪夜", negative="", reference_frames=[str(image)])

        backend = _backend_with(SimpleNamespace(ref_timestep_zero=False))
        backend.prepare_edit_reference = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("非 Edit 族**不该**碰参考图 ✓✗"))
        check("④ 2512 t2i（无 `ref_timestep_zero`）⇒ 返回 None ✓ 且**不读图** ✓",
              backend._edit_reference(request) is None)

        backend = _backend_with(SimpleNamespace(ref_timestep_zero=True))
        bare = SimpleNamespace(prompt="x", negative="", reference_frames=[])
        try:
            backend._edit_reference(bare)
            check("④′ Edit 权重**没给参考图** ⇒ 报错 ✓", False, "没报错")
        except tb.TorchBackendUnavailable as err:
            check("④′ Edit 权重没给参考图 ⇒ 报错 ✓（退回纯 t2i 是 0 报错的能力消失 ✓✗）",
                  "referenceFrames" in str(err), str(err))

        backend.prepare_edit_reference = lambda path, **_k: {
            "embeds": None, "latents": None,
            "report": {"image": path, "vision": {"tokens": 3},
                       "vae": {"channels": 16, "shape": [1, 16, 8, 8]}}}
        got = backend._edit_reference(request)
        check("④‴ Edit 权重 + 一张图 ⇒ 走 `prepare_edit_reference` ✓ 且 ``count == 1`` ✓",
              got is not None and got["report"]["images"] == 1 and got["count"] == 1, got)

        #: ⭐ 多图（2026-11-01 ✓）：**不再**报「只有一个槽」✓ —— 两张图就该有两份 ✓✗
        #: （拒跑会让 Plus 那套口径永远用不上 ✓；改成静默只用第一张则更糟 ✓✗）。
        hits: list[str] = []

        def _prepare(path: str, **_k: object) -> dict[str, object]:
            hits.append(path)
            #: 这一组只查**份数与顺序** ✓，所以嵌入/潜变量拿标记串占位即可 ✓（形状在 ⑤⑥ 与 ⑨ 查 ✓）
            return {"embeds": f"emb-{len(hits)}", "latents": f"lat-{len(hits)}",
                    "report": {"image": path, "vision": {"tokens": len(hits)},
                               "vae": {"channels": 16, "shape": [1, 16, 8, 8]}}}

        backend.prepare_edit_reference = _prepare
        two = SimpleNamespace(prompt="换背景", negative="",
                              reference_frames=[str(image), str(folder / "b.png")])
        got = backend._edit_reference(two)
        check("④⁴ 两张参考图 ⇒ **两份** ✓✗（旧版在这里报错 ✓；不接多图 = Plus 口径永远用不上 ✓）",
              got["count"] == 2 and len(got["embeds"]) == 2 and len(got["latents"]) == 2, got["count"])
        check("④⁵ 顺序就是语义 ✓✗：``paths`` 与 ``embeds`` 逐位对应 ✓（乱序不报错、只让编辑作用错图 ✓）",
              got["report"]["paths"] == [str(image), str(folder / "b.png")]
              and hits == got["report"]["paths"], (got["report"]["paths"], hits))
        check("④⁶ 回执把逐张的 token 数**加起来** ✓、并留每张一份 ✓✗（只看总数查不出哪张吃超了 ✓）",
              got["report"]["vision"]["tokens"] == 1 + 2
              and len(got["report"]["vision"]["perImage"]) == 2
              and got["report"]["vae"]["shapes"] == [[1, 16, 8, 8], [1, 16, 8, 8]],
              got["report"])

        try:
            backend._edit_reference(SimpleNamespace(prompt="x", negative="", reference_frames=[]))
            check("④⁷ 空参考图列表 ⇒ 报错 ✓", False, "没报错")
        except tb.TorchBackendUnavailable as err:
            check("④⁷ 空参考图列表 ⇒ 报错 ✓（`prepare_edit_references` 自己也要拦 ✓，不只靠上层 ✓）",
                  "至少一张" in str(err), str(err))


# ---------------------------------------------------------------- ⑤ 视觉塔那一路


def case_reference_embeds() -> None:
    backend = tb.TorchBackend(device="cpu")
    backend._visionTower = vl_mod.build_vision_tower(_SMALL_VIS, dtype=None, device="cpu")
    backend._visionShape = dict(_SMALL_VIS)
    with tempfile.TemporaryDirectory() as tmp:
        image = _png(320, 180, Path(tmp), "ref.png")
        got = backend.reference_embeds(image)
        embeds, report = got["embeds"], got["report"]
        merge = int(_SMALL_VIS["spatial_merge_size"])
        expected = (report["grid"][0] // merge) * (report["grid"][1] // merge)
        check("⑤ 视觉嵌入是 ``(合并后 token, out_hidden)`` ✓",
              tuple(embeds.shape) == (report["tokens"], 16), tuple(embeds.shape))
        check("⑤′ token 数 = 网格 ÷ merge² ✓（不是面积 ÷ 28² 手算 ✓）",
              report["tokens"] == expected, (report["tokens"], expected, report["grid"]))
        check("⑤″ 嵌入不是整片同值 ✓（塔真跑了 ✓，不是把像素塞回来 ✓）",
              float(embeds.std()) > 0, float(embeds.std()))
        check("⑤‴ 回执写清原生/缩放/网格三格 ✓✗（偏色时只有这几格能查 ✓）",
              report["native"] == [320, 180] and report["area"] == 384 * 384
              and len(report["scaled"]) == 2, report)

        backend._visionTower = None
        try:
            backend.reference_embeds(image)
            check("⑤⁴ 没挂视觉塔 ⇒ 报错 ✓", False, "没报错")
        except tb.TorchBackendUnavailable as err:
            check("⑤⁴ 没挂视觉塔 ⇒ 报错 ✓（不静默把参考图当没看见 ✓✗）", "视觉塔" in str(err), str(err))


# ---------------------------------------------------------------- ⑥ VAE 那一路


def case_reference_latents() -> None:
    backend = tb.TorchBackend(device="cpu")
    torch = backend._torch()
    backend._vae = SimpleNamespace()          # 只为过 ``_vae is not None`` 那道闸 ✓

    def _five_dim(pixels: object) -> object:
        #: Wan 族编码出 **5 维** ✓（``(1, 16, 1, 8, 8)`` ✓）；参考图是单帧 ⇒ 必须挤成 4 维 ✓✗
        return torch.zeros((1, 16, 1, 8, 8), dtype=torch.float32)

    backend._encode_pixels = _five_dim
    with tempfile.TemporaryDirectory() as tmp:
        image = _png(320, 180, Path(tmp), "ref.png")
        got = backend.reference_latents(image)
        latent, report = got["latents"], got["report"]
        check("⑥ 单帧 5 维 ⇒ 挤成 **4 维** ✓✗（5 维进 ``ref_latents`` 会在 patchify 上炸 ✓）",
              tuple(latent.shape) == (1, 16, 8, 8) and latent.ndim == 4, tuple(latent.shape))
        check("⑥′ 回执写清通道与网格 ✓（``16`` ch 是 2512/Edit 族的事实 ✓）",
              report["channels"] == 16 and report["grid"] == [8, 8], report)
        check("⑥″ VAE 那一路的面积是 **1024²** ✓✗（不是视觉塔的 384² ✓）",
              report["area"] == 1024 * 1024 and report["scaled"][0] % 8 == 0, report)

        backend._encode_pixels = lambda _p: torch.zeros((1, 16, 3, 8, 8), dtype=torch.float32)
        try:
            backend.reference_latents(image)
            check("⑥‴ 编出多个潜帧 ⇒ 报错 ✓", False, "没报错")
        except tb.TorchBackendUnavailable as err:
            check("⑥‴ 编出多个潜帧 ⇒ 报错 ✓（参考图是单帧 ✓）", "潜帧" in str(err), str(err))

    backend._vae = None
    try:
        backend.reference_latents("whatever.png")
        check("⑥⁴ 没挂 VAE ⇒ 报错 ✓", False, "没报错")
    except tb.TorchBackendUnavailable as err:
        check("⑥⁴ 没挂 VAE ⇒ 报错 ✓", "VAE" in str(err), str(err))


# ---------------------------------------------------------------- ⑦ 文本塔注入


class _Recorder:
    """**替身**文本塔 ✓：只记录"有没有收到 ``image_embeds``" ✓（真前向在
    `engine_qwen25_te_test` ⑧ 系列 ✓）—— 这一组要钉的是**接线**，不是数值 ✓。"""

    def __init__(self, tokens: int = 8, hidden: int = 3584) -> None:
        self.calls: list[dict[str, object]] = []
        self.tokens = int(tokens)
        self.hidden = int(hidden)

    def __call__(self, ids: object, *, image_embeds: object = None) -> object:
        import torch  # noqa: PLC0415

        #: ⚠️ 多图时 ``image_embeds`` 是**一列** ✓ ⇒ 记录时按块展开 ✓（单图 = 长度 1 的列表 ✓）；
        #: 拿 ``.shape`` 直接取会 ``AttributeError`` ✓（至少是报错 ✓，不会静默 ✗）。
        if image_embeds is None:
            shapes = None
        elif isinstance(image_embeds, (list, tuple)):
            shapes = [tuple(int(v) for v in item.shape) for item in image_embeds]
        else:
            shapes = [tuple(int(v) for v in image_embeds.shape)]
        self.calls.append({"rows": int(ids.shape[0]), "seq": int(ids.shape[1]),
                           "embeds": shapes})
        return torch.zeros((int(ids.shape[0]), int(ids.shape[1]), 4), dtype=torch.float32)


class _Tokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.limits: list[int] = []

    def encode(self, text: str) -> list[int]:
        self.texts.append(str(text))
        return [7] * max(1, len(str(text)))


class _Config:
    """替身配置 ✓：两个模板都留 ✓（``prompt_edit`` 缺席 ⇒ 注入时才报错 ✓）。"""

    max_length = 512

    def prompt(self, text: str) -> str:
        return f"<sys>{text}"

    def prompt_edit(self, text: str, image_count: int = 1) -> str:
        if int(image_count) == 1:
            return f"<sys><|vision_start|><|image_pad|><|vision_end|><user>{text}"
        prefix = "".join(
            f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(1, int(image_count) + 1))
        return f"<sys>{prefix}<user>{text}"


def case_encode_text_wiring() -> None:
    backend = tb.TorchBackend(device="cpu")
    torch = backend._torch()
    encoder = _Recorder(tokens=5, hidden=3584)
    tokenizer = _Tokenizer()
    backend._textEncoder = encoder
    backend._textConfig = _Config()
    backend._tokenizer = tokenizer
    backend._textKind = "qwen25"

    fake_embeds = torch.zeros((5, 3584), dtype=torch.float32)
    fake_latents = torch.zeros((1, 16, 8, 8), dtype=torch.float32)
    seen: list[str] = []

    def _prepare(path: str, **_kw: object) -> dict[str, object]:
        seen.append(path)
        return {"embeds": fake_embeds, "latents": fake_latents,
                "report": {"image": path, "vision": {"tokens": 5},
                           "vae": {"channels": 16, "shape": [1, 16, 8, 8]}}}

    backend.prepare_edit_reference = _prepare
    backend._config = SimpleNamespace(ref_timestep_zero=True)

    request = SimpleNamespace(prompt="换成雪夜", negative="blurry", reference_frames=["ref.png"])
    payload = backend.encode_text(request)

    check("⑦ 正向收到视觉嵌入 ✓、负向**不**收 ✓✗（负向那句是「不要什么」，与参考图无关 ✓）",
          encoder.calls[0]["embeds"] == [(5, 3584)] and encoder.calls[1]["embeds"] is None,
          encoder.calls)
    check("⑦′ 正向走 **edit 模板** ✓（`<|vision_start|>` 在 user 段、正文之前 ✓）",
          "<|image_pad|>" in tokenizer.texts[0] and "<|image_pad|>" not in tokenizer.texts[1],
          tokenizer.texts)
    check("⑦″ 参考图潜变量**存进后端** ✓✗（只放回执 ⇒ 只有第一步拿得到 ✓），且统一成**列表** ✓",
          isinstance(backend._refLatents, list) and len(backend._refLatents) == 1
          and tuple(backend._refLatents[0].shape) == (1, 16, 8, 8), backend._refLatents)
    check("⑦‴ 回执写清两路口径 + 逐分支 token 事实 ✓",
          payload["edit"]["vision"]["tokens"] == 5
          and payload["byBranch"]["positive"]["template"] == "edit"
          and payload["byBranch"]["negative"].get("template") is None, payload.get("edit"))
    check("⑦⁴ 回执的 `ids`/`tokens` 是**真 id 列表**的长度 ✓（不是字典键数 ✓）",
          payload["tokens"] == len(payload["ids"]) == encoder.calls[0]["seq"], payload["tokens"])

    #: ⭐ **两张图** ⇒ Plus 口径 ✓✗：模板要两个视觉块 ✓、嵌入要两块 ✓、`_refLatents` 要两条 ✓。
    #: 三处只要有一处还按单图走 ⇒ **不报错** ✓，只让第二张图的指令落空 ✓✗。
    tokenizer.texts.clear()
    encoder.calls.clear()
    two = SimpleNamespace(prompt="把第一张的人放进第二张的场景", negative="blurry",
                          reference_frames=["a.png", "b.png"])
    payload = backend.encode_text(two)
    check("⑦⁵ 两张图 ⇒ 模板拿到 ``image_count=2`` ✓✗（只传文本 ⇒ 永远走单图模板 ✓）",
          tokenizer.texts[0].count("<|image_pad|>") == 2
          and "Picture 2:" in tokenizer.texts[0], tokenizer.texts[0][:120])
    check("⑦⁶ 两张图 ⇒ 文本塔收到**两块**嵌入 ✓✗（合成一块 ⇒ 第二张图的条件与第一张混在同一个槽 ✓）",
          encoder.calls[0]["embeds"] == [(5, 3584), (5, 3584)], encoder.calls[0])
    check("⑦⁷ 两张图 ⇒ ``_refLatents`` 两条 ✓、回执标 ``edit-plus`` ✓ 且图数写对 ✓",
          len(backend._refLatents) == 2
          and payload["byBranch"]["positive"]["template"] == "edit-plus"
          and payload["byBranch"]["positive"]["images"] == 2, payload["byBranch"])
    check("⑦⁸ 长度预算 = 文本预算 **+ 图的 token 数** ✓✗（直接用 512 ⇒ 三张图会把尾部指令截掉 ✓）",
          payload["byBranch"]["positive"]["maxLength"] == 512 + 10, payload["byBranch"])

    #: ⚠️ 换回 t2i 请求 ⇒ 参考图**必须**清掉 ✓✗（不清 ⇒ 上一张参考图偷偷回到这一步采样里 ✓，
    #: 图会莫名"像上一张"，且**一条报错都没有** ✓）。
    backend._config = SimpleNamespace(ref_timestep_zero=False)
    backend.encode_text(SimpleNamespace(prompt="纯文生图", negative="", reference_frames=[]))
    check("⑦⁹ 非 Edit 请求 ⇒ `_refLatents` 被清空 ✓✗（残留会让下一张图偷偷带上一张参考图 ✓）",
          backend._refLatents is None, backend._refLatents)


# ---------------------------------------------------------------- ⑧ 采样带 ref


def case_denoise_ref() -> None:
    backend = tb.TorchBackend(device="cpu")
    torch = backend._torch()
    backend._ditFamily = "qwen_image"

    class _DiT:
        def __init__(self, ref_tzero: bool) -> None:
            self.shape = SimpleNamespace(ref_timestep_zero=ref_tzero)
            self.calls: list[dict[str, object]] = []

        def __call__(self, latents: object, sigma: object, condition: object,
                     ref_latents: object = None) -> object:
            self.calls.append({
                "refs": None if ref_latents is None else len(list(ref_latents)),
                #: ⚠️ 逐条记形状 ✓（只记 [0] ⇒ 多图时第 2 条被换掉也查不出来 ✓✗）
                "dims": ([[int(v) for v in item.shape] for item in ref_latents]
                         if ref_latents else None),
                "sigmaDim": int(getattr(sigma, "ndim", -1)),
            })
            return torch.zeros_like(latents)

    latents = torch.zeros((1, 16, 8, 8), dtype=torch.float32)
    condition = torch.zeros((1, 4, 4), dtype=torch.float32)
    request = SimpleNamespace()

    model = _DiT(True)
    backend._model = model
    backend._refLatents = [torch.zeros((1, 16, 8, 8), dtype=torch.float32)]
    backend.denoise(latents, 0.5, condition, request)
    backend.denoise(latents, 0.2, condition, request)
    check("⑧ Edit 权重 ⇒ **每一步**都带 `ref_latents` ✓✗（只有第一步给 = 中途忘了参考图 ✓）",
          [call["refs"] for call in model.calls] == [1, 1], model.calls)
    check("⑧′ 传进去的是**一条** 4 维参考潜变量 ✓（`Sequence[Tensor]` ✓，少包一层列表会按 batch 遍历 ✓✗）",
          model.calls[0]["dims"] == [[1, 16, 8, 8]], model.calls[0])
    check("⑧″ σ 走**逐 batch** 张量 ✓（`qwen_image` 契约是 `batch` ✓，喂标量不报错只是各条共用一个 σ ✓✗）",
          model.calls[0]["sigmaDim"] == 1, model.calls[0])

    #: ⭐ **两张参考图** ⇒ 采样每步带**两条** ✓✗：只搬第 0 条 ⇒ 第二张图的条件凭空消失 ✓，
    #: 且**一条报错都没有** ✓（症状：编辑只作用在第一张上 ✓）。
    multi = _DiT(True)
    backend._model = multi
    backend._refLatents = [torch.zeros((1, 16, 8, 8), dtype=torch.float32),
                           torch.zeros((1, 16, 8, 4), dtype=torch.float32)]
    backend.denoise(latents, 0.5, condition, request)
    check("⑧⁵ 两条参考潜变量 ⇒ 每步都带**两条** ✓✗（只搬 [0] 不报错，只让第二张图失效 ✓）",
          multi.calls[0]["refs"] == 2, multi.calls[0])
    check("⑧⁶ 各条**各自**搬设备/精度 ✓、尺寸不必相同 ✓✗（按第一条的形状闷头切片会越界 ✓）",
          multi.calls[0]["dims"] == [[1, 16, 8, 8], [1, 16, 8, 4]], multi.calls[0])

    backend._refLatents = None
    try:
        backend.denoise(latents, 0.5, condition, request)
        check("⑧‴ Edit 权重但 `ref_latents` 不在 ⇒ 报错 ✓", False, "没报错")
    except tb.TorchBackendUnavailable as err:
        check("⑧‴ Edit 权重但 `ref_latents` 不在 ⇒ 报错 ✓（拒跑 > 静默按 t2i 出图 ✓）",
              "_refLatents" in str(err) or "ref_latents" in str(err), str(err))

    #: 空列表也是「不在」 ✓✗：`[]` 判 `is None` 会漏过去 ⇒ 空 `ref_latents` 送进 DiT ⇒
    #: 参考图那一路整段消失 ✓，与「没给图」同一种病 ✓。
    backend._refLatents = []
    try:
        backend.denoise(latents, 0.5, condition, request)
        check("⑧⁴ 空列表也当「不在」 ⇒ 报错 ✓", False, "没报错")
    except tb.TorchBackendUnavailable as err:
        check("⑧⁴ 空列表也当「不在」 ⇒ 报错 ✓✗（`[] is not None` ⇒ 空参考图静默进 DiT ✓）",
              "ref_latents" in str(err) or "_refLatents" in str(err), str(err))

    t2i = _DiT(False)
    backend._model = t2i
    backend._refLatents = [torch.zeros((1, 16, 8, 8), dtype=torch.float32)]
    backend.denoise(latents, 0.5, condition, request)
    check("⑧⁷ 2512 t2i（无哨兵）⇒ **不**传 `ref_latents` ✓✗（它的 forward 没这个参数 ⇒ TypeError ✓）",
          t2i.calls[0]["refs"] is None, t2i.calls[0])


# ---------------------------------------------------------------- ⑨ 真权重


def case_real_weights() -> None:
    if not (_TEXT_WEIGHTS.is_file() and _IMAGE_VAE.is_file()):
        for label in ("⑨ 真权重：视觉塔装的是同文件的 `visual.*` ✓",
                      "⑨′ 真权重：视觉 3584 宽 / VAE 16 通道 ✓",
                      "⑨″ 真权重：两路口径不同源 ✓",
                      "⑨³ 真权重：两张不同长宽的图 ⇒ token 数各不相同 ✓"):
            _skip(label, "盘上没有 qwen_2.5_vl_7b / qwen_image_vae")
        return
    backend = tb.TorchBackend(device="cpu")
    tower = backend.attach_qwen25vl_vision(str(_TEXT_WEIGHTS))
    backend.attach_wan_vae(str(_IMAGE_VAE), family="qwen_image")
    check("⑨ 真权重：视觉塔装的是同文件的 `visual.*` ✓ 且 32 层 / 1280 宽 ✓",
          tower["layers"] == 32 and tower["hidden"] == 1280
          and tower["outHidden"] == _VIS_OUT_HIDDEN, tower)

    with tempfile.TemporaryDirectory() as tmp:
        image = _png(512, 288, Path(tmp), "ref.png")
        got = backend.prepare_edit_reference(str(image))
        vision, vae = got["report"]["vision"], got["report"]["vae"]
        check("⑨′ 真权重：视觉 3584 宽 / VAE 16 通道 4 维 ✓",
              tuple(got["embeds"].shape)[1] == 3584
              and vae["channels"] == 16 and len(vae["shape"]) == 4,
              (tuple(got["embeds"].shape), vae["shape"]))
        check("⑨″ 两路口径不同源 ✓：视觉面积 384² ≠ VAE 面积 1024² ✓、token 数≠像素数 ✓",
              vision["area"] == 384 * 384 and vae["area"] == 1024 * 1024
              and vision["tokens"] != vae["shape"][2] * vae["shape"][3],
              (vision["area"], vae["area"], vision["tokens"], vae["shape"]))

        #: ⭐ 多图真权重 ✓：两张**长宽不同**的图 ⇒ 各自 384² 面积算出的 token 数**不同** ✓✗
        #: （拿第一张的 token 数去套第二张 ⇒ 注入后序列错位 ✓，而且不报错 ✓）。
        #: ⚠️ NOT 用「同一张转置」当第二张 ✗：16:9 与 9:16 的合并网格只是行列互换 ⇒ token 数**相同** ✓，
        #: 那样这条判据永远绿、查不出"两张共用一份 token 数" ✓✗ ⇒ 第二张取**方图** ✓（180 vs 196 ✓）。
        wide = _png(512, 288, Path(tmp), "wide.png")
        square = _png(320, 320, Path(tmp), "square.png")
        many = backend.prepare_edit_references([str(wide), str(square)])
        tokens = [item["tokens"] for item in many["report"]["vision"]["perImage"]]
        check("⑨³ 两张不同长宽的图 ⇒ token 数各不相同 ✓ 且 `embeds` 逐张一块 ✓✗",
              many["count"] == 2 and len(many["embeds"]) == 2
              and tokens[0] != tokens[1] and many["report"]["vision"]["tokens"] == sum(tokens),
              (tokens, many["report"]["vision"]["tokens"]))


def main() -> int:
    print("=" * 78)
    print("Qwen-Image Edit 2511 参考图通路自检 ✓")
    print("=" * 78)
    case_area_size()
    case_pixels_of_area()
    case_edit_reference()
    case_reference_embeds()
    case_reference_latents()
    case_encode_text_wiring()
    case_denoise_ref()
    case_real_weights()
    print("-" * 78)
    print(f"跳过 {len(_SKIPS)}")
    for item in _FAILS:
        print(f"  ✗ {item}")
    for item in _SKIPS:
        print(f"  · {item}")
    print(f"SUMMARY: {_TOTAL - len(_FAILS)}/{_TOTAL} passed")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
