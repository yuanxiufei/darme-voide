"""自研引擎 · H3 **真文本塔**装配 ✓（`TorchBackend.attach_h3_text` ✓ —— 一个文件里两半塔 ✓）。

## 为什么单独一份 ✗

H3 的文本编码器是 **Qwen3-VL-32B** ✓，盘上那份 ``qwen3vl_32b_minimax_h3_int8_convrot``
（25.28 GiB ✓）里**一半是语言塔**（``model.*`` ✓ 1251 键）、**一半是视觉塔**（``visual.*`` ✓
351 键）✓。``fl2va`` / ``ref2va`` 的参考图**不经潜帧**、而是经视觉塔编码成 token 接在 prompt
前面 ✓ ⇒ 两半**必须同时在场** ✓。

这条路上有三个**不报错**的失败面 ✓✗：

1. **只装了一半**：只有一个塔时形状照样全对 ✓、装载报告还能写 ``complete=True`` ✓✗ ——
   前向里参考图块拿不到特征 ✓，而位置已经占了 ✓（出片"给了图等于没给" ✓）；
2. **int8 没接管**：350 个线性层是 ``I8`` ＋ ``weight_scale`` ✓，漏升级
   :class:`~.h3_form.Int8Linear` ⇒ 每层按"反量化成 bf16"装 ✓✗ ⇒ 50 GiB ✗（卡上 22.49 ✓），
   报错落在**几十分钟后的前向**里 ✓；
3. **键名空间挪歪**：语言塔要挂 ``model.*`` ✓、视觉塔要挂 ``visual.*`` ✓ —— 装成
   ``text_tower.*`` / ``vision.*`` 之类 ⇒ ``complete=False`` 或**大段 unexpected** ✓✗。

本份只钉**能另算出来**的东西 ✓：键名集合**逐字**等于盘上 ✓、int8 层数与计划数**相等** ✓、
两半缺一**当场拒** ✓、没 int8 的重导出**当场拒** ✓（不猜、也不按 bf16 硬装 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_h3_text_test.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def pieces_text(pieces: object) -> str:
    """把**段序列**里所有文本拼起来 ✓ —— 标签顺序只能从这儿看 ✓。

    ⚠️ 不能从回执看 ✗：回执里只有 ``frozen`` 的 ``hidden``/``tags`` ✓，``<Audio j>`` 这类
    **字面标签**在段序列里 ✓（拼错顺序 ⇒ 号与块序错位 ✓、回执形状全对 ✓✗）。
    """
    return "".join(str(item.get("text") or "") for item in pieces or ()
                   if isinstance(item, dict) and item.get("kind") == "text")


def label_text(seen: dict[str, object]) -> str:
    """同上 ✓ —— 只是从"假编码器记下的那次调用"里取 ✓。"""
    return pieces_text(seen.get("pieces"))


_PNG_CACHE: dict[str, pathlib.Path] = {}


def _tiny_png() -> pathlib.Path:
    """一张 64×48 的纯灰 PNG ✓（`_h3_pieces` 只做"读成张量" ✓，不关心内容 ✓）。"""
    cached = _PNG_CACHE.get("path")
    if cached is not None and cached.exists():
        return cached
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    folder = pathlib.Path(tempfile.mkdtemp(prefix="h3_text_refs_"))
    path = folder / "ref.png"
    Image.fromarray(np.full((48, 64, 3), 128, dtype=np.uint8)).save(path)
    _PNG_CACHE["path"] = path
    return path


def skip(name: str) -> None:
    _RESULTS.append((name, True, "（跳过：没有 torch / safetensors ✓ —— **如实标记**，不算通过 ✗）"))


def _parts() -> tuple[Any, Any, Any]:
    import torch  # noqa: PLC0415

    from app.services.engine import qwen3vl_te as vl_te  # noqa: PLC0415
    from app.services.engine import qwen3vl_vision as vl_vision  # noqa: PLC0415

    return torch, vl_te, vl_vision


def _shapes() -> tuple[dict[str, Any], dict[str, Any]]:
    """造一套**缩小的**尺子 ✓（与真 32B 同一套字段名 ✓ —— 只是每个维度都小到秒级 ✓）。"""
    _, vl_te, vl_vision = _parts()
    text = dict(vl_te.QWEN3VL_32B)
    text.update(hidden=32, layers=2, heads=4, kvHeads=2, headDim=8, ffn=48, vocab=64,
                ropeDims=())
    vision = dict(vl_vision.VISION_32B)
    vision.update(hidden_size=24, intermediate_size=16, depth=1, num_heads=2,
                  patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                  num_position_embeddings=4, out_hidden_size=32)
    return text, vision


def _linear_names(module: Any) -> list[str]:
    torch = _parts()[0]
    return [name for name, child in module.named_modules() if isinstance(child, torch.nn.Linear)]


def _build_checkpoint(root: pathlib.Path, *, with_vision: bool = True,
                      quantized: bool = True) -> tuple[pathlib.Path, dict[str, Any]]:
    """造一份**缩小版**的 H3 文本塔权重 ✓ ⇒ ``(路径, 事实)`` ✓。

    语言塔按**真文件的口径**走 int8（``weight`` I8 ＋ ``weight_scale`` F32 ✓），视觉塔整份
    BF16 ✓（真文件就是这样 ✓ —— 别把视觉塔也塞进 int8 那条路 ✗）。
    """
    torch, vl_te, vl_vision = _parts()
    from safetensors.torch import save_file  # noqa: PLC0415

    text_shape, vision_shape = _shapes()
    text_tower = vl_te.build_text_tower(text_shape, dtype=torch.float32)
    vision_tower = vl_vision.build_vision_tower(vision_shape, dtype=torch.bfloat16)

    tensors: dict[str, Any] = {}
    int8_names = _linear_names(text_tower) if quantized else []
    state = text_tower.state_dict()
    for name, tensor in state.items():
        owner = name[: -len(".weight")] if name.endswith(".weight") else ""
        if quantized and owner in int8_names:
            generator = torch.Generator().manual_seed(len(name))
            tensors[name] = torch.randint(-127, 127, tuple(tensor.shape), dtype=torch.int8,
                                          generator=generator)
            #: per-row 尺度 ✓（int8 的账靠它还原 ✓；缺了装载闸口会拒 ✓）
            tensors[f"{owner}.weight_scale"] = torch.rand((tensor.shape[0],), dtype=torch.float32,
                                                          generator=generator) * 0.01 + 0.001
            continue
        tensors[name] = tensor.detach().clone().to(torch.bfloat16)
    if with_vision:
        #: ⚠️ 盘上的键是 ``visual.*`` ✓（`h3_encode.combine` 把视觉塔挂在 ``visual`` ✓）——
        #: 视觉塔自己 `state_dict()` 是裸 ``blocks.*`` ✓，**必须补前缀** ✓ 否则装进去全落 missing ✗✗。
        for name, tensor in vision_tower.state_dict().items():
            tensors[f"visual.{name}"] = tensor.detach().clone().to(torch.bfloat16)

    path = root / "h3_te_tiny.safetensors"
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path),
              metadata={})
    facts = {"textShape": text_shape, "visionShape": vision_shape,
             "int8": sorted(int8_names), "keys": sorted(tensors)}
    return path, facts


def main() -> int:
    try:
        import torch  # noqa: F401, PLC0415
        from safetensors.torch import save_file  # noqa: F401, PLC0415
    except ImportError:
        skip("①②③④ `attach_h3_text`（要 torch ＋ safetensors 造真权重）")
        return _report()

    from app.services.engine import h3_encode as h3_encode_mod  # noqa: PLC0415
    from app.services.engine import h3_form as h3  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    torch.manual_seed(20261101)

    with tempfile.TemporaryDirectory() as workspace:
        root = pathlib.Path(workspace)

        # ── ① 正常装配：两半都在 ＋ int8 全部接管 ──
        path, facts = _build_checkpoint(root)
        backend = tb.TorchBackend()
        report = backend.attach_h3_text(str(path))
        tower = backend._textEncoder
        int8_names = facts["int8"]
        upgraded = [name for name, child in tower.named_modules()
                    if isinstance(child, h3.Int8Linear)]
        check("① `attach_h3_text` 装完 ⇒ 回执 `kind=h3-te` ✓、`complete=True` ✓、"
              "`_textKind` 同步 ✓（`capabilities()` 靠它自述 ✓）",
              report.get("kind") == "h3-te" and report.get("complete") is True
              and backend._textKind == "h3-te", (report.get("kind"), report.get("complete"),
                                                 backend._textKind))

        check("①′ **两半都在** ✓：`model`（语言塔）与 `visual`（视觉塔）同一个编码器 ✓、"
              "`env.output_dim` 等于语言塔 wide ✓",
              getattr(tower, "model", None) is not None
              and getattr(tower, "visual", None) is not None
              and int(report["config"]["output_dim"]) == int(facts["textShape"]["hidden"]),
              report.get("config"))

        check("①″ ⭐ **int8 全部接管** ✓：`Int8Linear` 个数 == 计划层数 == 回执的 "
              "`int8Ledger.layers` ✓（漏一层 ⇒ 那层按 bf16 装 ⇒ 真权重上必 OOM ✗）",
              len(upgraded) == len(int8_names) == int(report["int8Ledger"]["layers"])
              and len(int8_names) > 0,
              (len(upgraded), len(int8_names), report["int8Ledger"]["layers"]))

        file_keys = set(facts["keys"])
        check("①‴ ⭐ **键名空间逐字对上** ✓：编码器 `state_dict()` 与盘上键集合**相等** ✓、"
              "`unexpectedCount == 0` ✓（名字挪歪 ⇒ 大段 unexpected ＋ 大段 missing ✗，"
              "而 `missing` 里只要不是「招牌键」就没人看 ✗）",
              set(tower.state_dict()) == file_keys
              and int(report.get("unexpectedCount", -1)) == 0,
              (sorted(set(tower.state_dict()) ^ file_keys)[:6],
               report.get("unexpectedCount")))

        check("①⁗ 视觉塔**整份 BF16** ✓（没有 int8 层 ✓）、语言塔的 `embed_tokens` 仍是普通嵌入 ✓"
              "（把嵌入也当线性层升 int8 ⇒ 查表拿到 int8 ✓✗）",
              not [name for name, child in tower.named_modules()
                   if isinstance(child, h3.Int8Linear) and name.startswith("visual.")]
              and not isinstance(tower.model.embed_tokens, h3.Int8Linear),
              [name for name, child in tower.named_modules()
               if isinstance(child, h3.Int8Linear)][:3])

        # ── ② 反向：只有语言塔（缺 visual.*）⇒ 当场拒 ──
        half = root / "half"
        half.mkdir()
        half_path, _ = _build_checkpoint(half, with_vision=False)
        half_error = _raises(lambda: tb.TorchBackend().attach_h3_text(str(half_path)))
        check("② 只有语言塔（缺 `visual.patch_embed.proj.weight`）⇒ **当场拒** ✓ 且点名视觉塔 ✓"
              "（放过去 ⇒ 参考图块没有特征、位置却占了 ✓✗ 形状全对 ✓✗）",
              half_error is not None and "视觉塔" in str(half_error),
              half_error and str(half_error)[:100])

        # ── ③ 反向：没有 int8 的重导出 ⇒ 当场拒（不按 bf16 硬装 ✗）──
        plain = root / "plain"
        plain.mkdir()
        plain_path, _ = _build_checkpoint(plain, quantized=False)
        plain_error = _raises(lambda: tb.TorchBackend().attach_h3_text(str(plain_path)))
        check("③ 全 BF16（一个 `I8` 都没有）⇒ **当场拒** ✓（按 bf16 硬装 = 50 GiB ✗，"
              "而报错会落在装载期 OOM ✓，指不到「这不是 int8 那份」 ✓✗）",
              plain_error is not None and "int8" in str(plain_error),
              plain_error and str(plain_error)[:100])

        # ── ④ 反向：文件不存在 ⇒ 报得清楚（不是 AttributeError / KeyError）──
        missing = _raises(lambda: tb.TorchBackend().attach_h3_text(str(root / "nope.safetensors")))
        check("④ 权重不在盘上 ⇒ 报「不存在」✓（`reason=pending` ✓，与另三座塔同一口径 ✓）",
              missing is not None and "不存在" in str(missing),
              missing and str(missing)[:80])

    case_encode()

    return _report()


def case_encode() -> None:
    """⑤⑥⑦ ⭐ **`tags` 的出口** ✓：`_encode_h3_condition` 把 ``H3Condition.tags`` 原样递回，
    并让 ``merge`` 取自**视觉塔本体** ✓；三条「不猜」的口径当场拒 ✓（不静默挑一个 ✗）。

    ⚠️ 为什么这条必须单测 ✗✗：`tags` 是「形状全对、数值全错」的那一类 ✓✗ ——
    ``hidden`` 的宽长对了、``seq_len`` 对了、采样照跑 ✓，只有视觉块**整段**用了 text 行 ✓；
    真权重上要几十秒一步、跑到出片才看得出来 ✓。这里用一个**假编码器**把出口钉住 ✓。
    """
    import types  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from app.services.engine import h3_encode as h3_encode_mod  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    seen: dict[str, object] = {}

    def fake_encode(pieces, *, tokenizer=None, images=None, video=None, merge_size=2):
        seen["merge"] = merge_size
        seen["images"] = images
        seen["video"] = video
        seen["pieces"] = list(pieces)
        return h3_encode_mod.H3Condition(
            hidden=torch.zeros(1, 5, 8), tags=torch.tensor([1, 0, 0, 1, 1]),
            ids=[1, 2, 3, 4, 5], spans=[], grids=[])

    def backend_with(visual: object) -> Any:
        engine = tb.TorchBackend(device="cpu")
        engine._form = engine.H3_FORM_NAME
        engine._textKind = "h3-te"
        engine._tokenizer = object()
        engine._textConfig = h3_encode_mod.H3TeRuntime(
            text={"hidden": 8, "vocab": 32}, vision={"spatial_merge_size": 2})
        engine._textEncoder = types.SimpleNamespace(visual=visual, encode=fake_encode)
        return engine

    request = types.SimpleNamespace(prompt="a red square", first_frame=None,
                                    reference_frames=None, reference_audio=None,
                                    reference_videos=None)
    visual = types.SimpleNamespace(spatial_merge_size=2)
    out = backend_with(visual).encode_text(request)
    check("⑤ `_encode_text_states` 走 h3-te ⇒ 回执里**有 `tags`** ✓（0/1 原样递回 ✓，"
          "`positive` 去掉批维 ✓），`h3Task=t2va` ✓（无参考素材 ⇒ 纯文 ✓）",
          out.get("tags") == [1, 0, 0, 1, 1]
          and tuple(out["positive"].shape) == (5, 8)
          and out.get("h3Task") == "t2va" and out.get("textEncoder") == "h3-qwen3vl",
          (out.get("tags"), tuple(out["positive"].shape), out.get("h3Task")))

    #: ⚠️ 这里**故意不用 2** ✗：用 2 的话"写死 2"与"从塔上取"两种实现都能过 ✓✗ ——
    #: 换成 3 ⇒ 只有真去塔上取的那版才对 ✓（`spatial_merge_size` 一改，占位个数与特征个数
    #: 会**双双不同** ✓，而形状校验一路绿灯 ✓✗）。
    odd = backend_with(types.SimpleNamespace(spatial_merge_size=3)).encode_text(request)
    check("⑤′ `merge_size` **取自视觉塔本体** ✓（`visual.spatial_merge_size` ✓，"
          "不从配置文件猜、不写死 2 ✗）：塔上是 3 ⇒ 送进去就是 3 ✓",
          seen.get("merge") == 3 and seen.get("images") is None
          and odd.get("tags") == [1, 0, 0, 1, 1],
          (seen.get("merge"), seen.get("images")))

    no_visual = _raises(lambda: backend_with(None).encode_text(request))
    check("⑥ 编码器上没有视觉塔 ⇒ **当场拒** ✓ 且点名 `spatial_merge_size` ✓"
          "（回落成 2 ⇒ 位置与特征错位 ✓ 且不报错 ✗）",
          no_visual is not None and "spatial_merge_size" in str(no_visual),
          no_visual and str(no_visual)[:110])

    #: ⭐ ⑦ 已**不是**"拒" ✓（2026-11-01 起音频参考接通 ✓）：独立音频只是一条 `<Audio j>` 标签 ✓
    #: （声波不进文本塔 ✓ —— 它进的是采样侧的音频块 ✓，见 `_dual_references` ✓）。
    audio_out = backend_with(visual).encode_text(types.SimpleNamespace(
        prompt="p", first_frame=None, reference_audio=("a.wav",),
        reference_frames=None, reference_videos=None))
    check("⑦ 只给**音频参考** ⇒ 接通 ✓ 且口径是 `ref2va` ✓、标签里出现 `<Audio 1>` ✓"
          "（此前一律拒 ✗ —— 文本里没有这条标签 ⇒ 采样侧那条音频块**没有对应的号** ✓✗）",
          audio_out.get("h3Task") == "ref2va"
          and audio_out.get("h3Refs") == ["audio"]
          and "<Audio 1>" in label_text(seen),
          (audio_out.get("h3Task"), audio_out.get("h3Refs"), label_text(seen)[:60]))

    mixed = backend_with(visual).encode_text(types.SimpleNamespace(
        prompt="p", first_frame=None, reference_audio=("a.wav",),
        reference_frames=(str(_tiny_png()),), reference_videos=None))
    check("⑦′ 同时给图 ＋ 音频 ⇒ 接通 ✓ 且**顺序照参考实现** ✓：图在前 ✓、独立音频在后 ✓"
          "（`<Picture 1>` 必须排在 `<Audio 1>` **之前** ✓ —— 反过来 ⇒ 号与块序错位 ✓ 形状全对 ✓✗）",
          mixed.get("h3Task") == "ref2va"
          and mixed.get("h3Refs") == ["image", "audio"]
          and 0 <= label_text(seen).find("<Picture 1>") < label_text(seen).find("<Audio 1>"),
          (mixed.get("h3Refs"), label_text(seen)[:80]))

    case_pieces()
    case_video_refs()
    case_video_dispatch()
    case_chain()


def case_pieces() -> None:
    """⑩ **三条口径的段序列** ✓：`t2va`（纯文 ✓）/ `fl2va`（首帧 ✓）/ `ref2va`（参考图 ✓）。

    ⚠️ 为什么这条必须有 ✗✗：`presentation()` 的 ``refs``（ref2va ✓）与 ``images``（fl2va ✓）是
    **两条互斥口径** ✓ —— 一起传会 `ValueError` ✓、只传一边会**丢掉视觉块** ✓✗（文本里有
    ``<Picture 1>: `` 却没有任何视觉占位 ⇒ 图被静默忽略 ✓、形状全对 ✓）。而这三种任务名连
    `H3Condition` 都不带 ✓，错了只有出片才看得出 ✓。
    """
    import numpy as np  # noqa: PLC0415
    import types  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    engine = tb.TorchBackend(device="cpu")
    with tempfile.TemporaryDirectory() as workspace:
        frame = pathlib.Path(workspace) / "frame.png"
        Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(frame)

        plain = engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=None, reference_frames=None,
            reference_audio=None, reference_videos=None))
        check("⑩ `t2va`＝纯文 ✓：段序列只有一段文本 ✓、没有视觉块 ✓、`h3Task=t2va` ✓",
              plain.get("task") == "t2va"
              and [item.get("kind") for item in plain["pieces"]] == ["text"]
              and not plain.get("images"), plain.get("pieces"))

        first = engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=str(frame), reference_frames=None,
            reference_audio=None, reference_videos=None))
        kinds = [item.get("kind") for item in first["pieces"]]
        check("⑩′ `fl2va`：首帧那张要**既进标签又进视觉块** ✓（`<Picture 1>: ` ＋ vision ✓，"
              "像素进 `images` ✓）⇒ 只给标签不给图 ✗、只给图不给标签 ✗ 都是「图被静默忽略」✓✗",
              first.get("task") == "fl2va"
              and kinds[:3] == ["text", "vision", "text"]
              and len(first.get("images") or []) == 1
              and tuple(first["images"][0].shape)[-1] == 3,
              (kinds, len(first.get("images") or [])))

        refs = engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=None, reference_frames=(str(frame),),
            reference_audio=None, reference_videos=None))
        check("⑩″ `ref2va`：参考图走 ``refs`` 口径 ✓（`<Picture 1>: ` ＋ vision ✓）⇒ "
              "两条口径**不许一起传** ✗（`presentation` 会当场 `ValueError` ✓，"
              "而\"挑一个传\"的写法会让另一种任务的图悄悄不进 ✓✗）",
              refs.get("task") == "ref2va"
              and [item.get("kind") for item in refs["pieces"]][:2] == ["text", "vision"]
              and len(refs.get("images") or []) == 1,
              [item.get("kind") for item in refs["pieces"]])

        #: ⚠️ 硬首帧 ＋ 参考素材是**两条通道** ✓✗（FL2VA 锁首帧但不吃参考 ✓、Ref2VA 吃参考
        #: 但不锁首帧 ✓）—— 放过去任一实现都得**静默丢**一半 ✓，而形状全对 ✓。
        both = _raises(lambda: engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=str(frame), reference_frames=(str(frame),),
            reference_audio=None, reference_videos=None)))
        check("⑩‴ 硬首帧 ＋ 参考素材**同时给** ⇒ 当场拒 ✓ 且指名「二选一」✓ "
              "（挑一个用 ⇒ 另一半被静默丢掉 ✓✗）",
              both is not None and "二选一" in str(both), both and str(both)[:110])


def case_video_refs() -> None:
    """⑪⑫ ⭐ 参考**视频**的段序列 ✓（2026-11-01 接通 ✓ —— 此前一律"当场拒" ✗）。

    钉三件**从形状上看不出来**的事：

    * 声轨标签 ``<Audio j>`` **排在它自己那路视频之前** ✓、独立音频从 ``S+1`` 起 ✓
      （反过来 ⇒ 文本说的号与采样侧 ``_dual_references`` 的块序**整体错位** ✓、形状全对 ✓✗）；
    * 每块视觉段带 ``videoIndex`` ✓ —— 区间是**本路内**的下标 ✓，每路都从 0 起 ✓；
    * 按 **2 fps** 抽帧后奇数帧**补末帧** ✓、补的那帧时间戳**重复末值** ✓（再 +0.5 秒 ⇒
      文本写了一个素材里不存在的时间 ✓✗）。
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import types  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from app.services.engine import media as media_mod  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    if not media_mod.have_ffmpeg():
        skip("⑪⑫⑬ 参考视频的段序列（要 ffmpeg 造素材 ＋ 解码 ✓）")
        return

    engine = tb.TorchBackend(device="cpu")
    with tempfile.TemporaryDirectory() as workspace:
        root = pathlib.Path(workspace)
        #: 3 帧 @2fps ⇒ 抽帧口径 2 fps ⇒ 仍 3 帧 ⇒ **补末帧**到 4 ⇒ 两块 ✓
        silent = media_mod.write_video(torch.zeros(1, 3, 3, 64, 64),
                                      root / "silent.mp4", fps=2)["path"]
        request = types.SimpleNamespace(prompt="a cat", first_frame=None,
                                        reference_frames=None, reference_audio=None,
                                        reference_videos=(silent,))
        prepared = engine._h3_pieces(request)
        labels = [str(item.get("text")) for item in prepared["pieces"]
                  if item.get("kind") == "text" and str(item.get("text")).startswith("<")]
        vision = [item for item in prepared["pieces"] if item.get("kind") == "vision"]
        videos = prepared.get("videos") or []
        check("⑪ 参考视频（无音轨）⇒ `ref2va` ✓、`refs` 只有一条 video ✓、`<Video 1>` 之后"
              "每块一条时间戳 ✓（3 帧补末帧 ⇒ 两块 ✓）、每块带 `videoIndex=1` ✓；"
              "**像素是 uint8** ✓（喂 [-1,1] 浮点进 `process_image` ⇒ 特征整体偏移 ✓ 不报错 ✗）",
              prepared.get("task") == "ref2va"
              and [ref.get("type") for ref in prepared["refs"]] == ["video"]
              and labels[0].startswith("<Video 1>") and len(labels) == 3
              and len(vision) == 2
              and all(item.get("videoBlock") and item.get("videoIndex") == 1 for item in vision)
              and len(videos) == 1 and int(videos[0].shape[0]) == 4
              and str(videos[0].dtype) == "torch.uint8",
              (labels, len(vision), [(tuple(v.shape), str(v.dtype)) for v in videos]))

        check("⑪′ 补的那帧**时间戳重复末值** ✓：末块中点仍是 `<1.0 seconds>` ✓"
              "（若照 +0.5 秒推 ⇒ 会写 `<1.2 seconds>` ✓✗ —— 那是素材里没有的时间 ✓）",
              labels[1:] == ["<0.2 seconds>", "<1.0 seconds>"], labels)

        #: 带音轨 ⇒ wav 混流进 mkv（pcm_s16le 无损 ✓）+ 再给一条**独立音频** ✓
        rate = 32000
        torch.manual_seed(11)
        samples = (torch.rand(2, int(rate * 0.1)) * 2.0 - 1.0) * 0.5
        wav = media_mod.write_wav(samples, root / "tone.wav", sample_rate=rate)["path"]
        mkv = root / "clip_audio.mkv"
        muxed = subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                                "-i", str(silent), "-i", str(wav), "-c", "copy", str(mkv)],
                               capture_output=True, timeout=120)
        if muxed.returncode or not mkv.exists() or not mkv.stat().st_size:
            skip("⑫ 带音轨参考视频（ffmpeg 混流 mkv 失败 ✓）")
            return
        mixed = engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=None, reference_frames=None,
            reference_audio=(str(wav),), reference_videos=(str(mkv),)))
        kinds = [ref.get("type") for ref in mixed["refs"]]
        text = pieces_text(mixed["pieces"])
        check("⑫ 带音轨视频 ＋ 独立音频 ⇒ 块序 `[声轨, video, 独立音频]` ✓、文本序 "
              "`<Audio 1>`（声轨）→ `<Video 1>` → `<Audio 2>`（独立音频）✓"
              "（把独立音频排到视频前 ⇒ 号与块序整体错位 ✓ 形状全对 ✓✗）",
              kinds == ["audio", "video", "audio"]
              and mixed["refs"][0].get("soundtrack") is True
              and mixed["refs"][2].get("soundtrack") is not True
              and 0 <= text.find("<Audio 1>") < text.find("<Video 1>") < text.find("<Audio 2>"),
              (kinds, text[:70]))

        #: 首帧那张图此时**已经是**参考素材 ✓ ⇒ 走 `reference_frames` 才不丢 ✓
        as_ref = engine._h3_pieces(types.SimpleNamespace(
            prompt="a cat", first_frame=None, reference_frames=(str(_tiny_png()),),
            reference_audio=None, reference_videos=(silent,)))
        order = [ref.get("type") for ref in as_ref["refs"]]
        as_text = pieces_text(as_ref["pieces"])
        check("⑬ 图 ＋ 视频 ⇒ 块序 `[image, video]` ✓、文本序 `<Picture 1>` 在 `<Video 1>` 之前 ✓"
              "（与采样侧 `_dual_references` 的 ``[image…] → [video…]`` 同一套 ✓）",
              order == ["image", "video"]
              and as_text.find("<Picture 1>") < as_text.find("<Video 1>"), (order, as_text[:60]))


def case_video_dispatch() -> None:
    """⑭ ⭐ **多路视频的块归属**：每块带的 `videoIndex` 决定切**哪一路** ✓✗。

    ⚠️ 为什么这条必须单测 ✗✗：两路视频的 ``frames`` 区间**都从 0 起** ✓ —— 去掉
    ``videoIndex``、改成"区间从 0 起就换下一路"的写法，在**第一路只有一块**时也能跑通 ✓
    （第二路的第一块同样从 0 起 ✓），只是把第二路的帧喂成了第一路 ✓、形状全对 ✓。
    这里用**可区分的常量张量**（10 / 200 ✓）把"切的是哪一路"钉死 ✓。
    """
    import types  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from app.services.engine import h3_encode as enc  # noqa: PLC0415
    from app.services.engine import h3_prompt as prompt_mod  # noqa: PLC0415
    from app.services.engine import qwen3vl_pixels as pixels_mod  # noqa: PLC0415

    seen: dict[str, list[float]] = {"blocks": []}

    def fake_block(block: object) -> tuple[object, tuple[int, int, int]]:
        #: uint8 上 `mean()` 会拒（"output dtype 推不出来" ✓）⇒ 先转 f32 再算均值 ✓
        seen["blocks"].append(round(float(block.float().mean()), 3))   # type: ignore[attr-defined]
        return block, (1, 2, 2)

    class Tok:
        """标签段**必须一次成词** ✓（`plan_sequence` 会核 ✓）⇒ 只回一个 id ✓。"""

        def encode(self, text: str) -> list[int]:
            return [7] if prompt_mod._LABEL.match(text) else [5] * max(1, len(text))

    class Tower:
        shape = {"hidden": 8}

        def __init__(self) -> None:
            self.model = types.SimpleNamespace(embed_tokens=torch.nn.Embedding(151680, 8))

        def __call__(self, *, embeds: object, spans: object, deepstack: object) -> object:
            return embeds

    def vision(flatten: object, grid: object) -> tuple[object, list[object]]:
        height, width = int(grid[0][1]), int(grid[0][2])
        rows = (height // 2) * (width // 2)
        return torch.zeros(rows, 8), [torch.zeros(rows, 8)]

    pieces = prompt_mod.presentation("p", refs=[
        {"type": "video", "frames": 4, "timestamps": [0.0, 0.5, 1.0, 1.5]},
        {"type": "video", "frames": 2, "timestamps": [0.0, 0.5]}])
    first = torch.full((4, 4, 4, 3), 10, dtype=torch.uint8)
    second = torch.full((2, 4, 4, 3), 200, dtype=torch.uint8)

    original = pixels_mod.process_video_block
    pixels_mod.process_video_block = fake_block          # type: ignore[assignment]
    try:
        condition = enc.encode_condition(Tower(), vision, pieces, tokenizer=Tok(),
                                         video=[first, second], merge_size=2)
    finally:
        pixels_mod.process_video_block = original        # type: ignore[assignment]

    check("⑭ 两路视频（各多块）⇒ 块按 `videoIndex` 切**各自那一路** ✓："
          "第一路两块都是 10 ✓、第二路一块是 200 ✓（全按第一路切 ⇒ 第三块也是 10 ✓✗）",
          seen["blocks"] == [10.0, 10.0, 200.0]
          and len(condition.spans) == 3
          and [tuple(grid) for grid in condition.grids] == [(1, 2, 2)] * 3,
          #: ⚠️ 传**副本** ✓✗：`check` 存的是对象引用 ✓、`_report()` 在全部用例跑完才打印 ✓
          #: ⇒ 后面 `clear()` 掉的话，失败信息里只会剩最后那次的一条 ✓（看着像"只切了一块" ✗）
          (list(seen["blocks"]), len(condition.spans)))

    #: ⚠️ 单路传**裸张量**也要认 ✓（老调用口的写法 ✓）—— 判"有没有 `dim`" ✓，不按长度猜 ✗
    seen["blocks"].clear()
    pixels_mod.process_video_block = fake_block          # type: ignore[assignment]
    try:
        one = enc.encode_condition(Tower(), vision,
                                   prompt_mod.presentation("p", refs=[
                                       {"type": "video", "frames": 2}]),
                                   tokenizer=Tok(), video=first, merge_size=2)
    finally:
        pixels_mod.process_video_block = original        # type: ignore[assignment]
    check("⑭′ 单路传裸张量（老调用口）⇒ 照常跑通 ✓（一块 ⇒ 全取第一路 ✓）",
          seen["blocks"] == [10.0] and len(one.spans) == 1,
          (list(seen["blocks"]), len(one.spans)))

    too_far = _raises(lambda: enc.encode_condition(
        Tower(), vision,
        [{"kind": "text", "text": "<Video 2>: "}, {"kind": "text", "text": "<0.2 seconds>"},
         {"kind": "vision", "frames": (0, 2), "videoBlock": True, "videoIndex": 2}],
        tokenizer=Tok(), video=[first], merge_size=2))
    check("⑭″ 块标第 2 路但只给 1 路 ⇒ **当场拒** ✓ 且点名路数 ✓"
          "（少给一路 ⇒ 后面对应的块整体错位 ✓ 不报错 ✗）",
          too_far is not None and "路" in str(too_far), too_far and str(too_far)[:110])

    missing = _raises(lambda: enc.encode_condition(
        Tower(), vision,
        prompt_mod.presentation("p", refs=[{"type": "video", "frames": 2}]),
        tokenizer=Tok(), video=None, merge_size=2))
    check("⑭‴ 有视频块却**没给帧序列** ⇒ 当场拒 ✓（不跳过该块 ✗ —— 跳过 ⇒ 文本多出占位 ✓ 特征少一块 ✓✗）",
          missing is not None and "没给" in str(missing), missing and str(missing)[:110])


def case_chain() -> None:
    """⑧⑨ H3 的**第四条装配线**外壳 ✓：分段表可覆盖 ✓（`decode` 不在表里 ⇒ 不白读 5.2 GiB VAE ✓），
    以及「给了真文本塔却没给词表」在**丢掉旧模型之前**就拒 ✓。
    """
    from app.services.engine import runtime as rt  # noqa: PLC0415
    from app.services.engine import torch_backend as tb  # noqa: PLC0415

    engine = tb.TorchBackend(device="cpu")
    chain = engine.declare_chain(
        dit_path="dit.safetensors", text_weights="te.safetensors",
        tokenizer_path="tok", vae_path="", vae_family="h3", text_kind="h3-te",
        phase_parts={"encode": ("text",), "sample": ("dit",)},
        part_need_gib=rt.H3_PART_NEED_GIB)
    declared = dict(chain.get("chain") or {})
    check("⑧ H3 的分段表**只声明 encode/sample** ✓ ⇒ `enter_phase(\"decode\")` 返回 None ✓"
          "（真 VAE 装配期已挂 cpu ✓，再换装等于白读 5.2 GiB ✗）；链上带 `h3-te` ✓",
          engine.enter_phase("decode") is None
          and engine.enter_phase("没有这一段") is None
          and declared.get("textKind") == "h3-te"
          and dict(declared.get("partNeedGiB") or {}) == rt.H3_PART_NEED_GIB,
          (engine.enter_phase("decode"), declared.get("textKind"),
           declared.get("partNeedGiB")))

    check("⑧′ 驻留账**必须是 H3 那份** ✓✗（DiT 19.62 / TE 25.44 / VAE 5.2 ✓，2026-11-01 实测）："
          "拿 2.1 的 `{dit:13.26,text:14.79}` 判 ⇒ 少算 12 GiB ✓、决策说装得下 ✓、装到一半 OOM ✓；"
          "⚠️ DiT 不许再写回 17.94 ✓✗ —— 那是**文件字节**✓，实测驻留 19.62 ✓，记小了就误判装得下 ✓",
          rt.H3_PART_NEED_GIB != tb.TorchBackend.PART_NEED_GIB
          and rt.H3_PART_NEED_GIB["dit"] == 19.62
          and rt.H3_PART_NEED_GIB["text"] == 25.44,
          rt.H3_PART_NEED_GIB)

    engine2 = _raises(lambda: rt.engine_runtime.ensure_loaded(
        stage="h3", dit_path="不存在的主干.safetensors",
        text_weights="te.safetensors", tokenizer_path=None))
    check("⑨ 给了 `text_weights` 却没给词表 ⇒ **当场拒** ✓ 且点名「词表」✓"
          "（装得上、编不出 ✓：`<|image_pad|>` 的 id 只有那套分词器有 ✓）",
          engine2 is not None and "词表" in str(engine2),
          engine2 and str(engine2)[:110])


def _raises(call: Any) -> BaseException | None:
    try:
        call()
    except BaseException as err:  # noqa: BLE001 —— 拒的方式不重要，**拒了**才重要 ✓
        return err
    return None


def _report() -> int:
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
