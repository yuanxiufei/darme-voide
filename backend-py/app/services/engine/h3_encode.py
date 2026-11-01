"""H3 的**文本条件编码**串联 ✓：呈现格式 → 分词 → 视觉塔 → 拼 embeds → 语言塔 → hidden + tags。

## 一条链上的四个「静默错」高发点（本模块逐条拦住）

1. **每段各自分词** ✓（参考 `minimax.py::MiniMaxH3Tokenizer.tokenize_with_weights` 的
   ``add_text`` 就是逐段调一次 ✓）。把整条串起来分一次 ✗ ⇒ 标签段与正文粘连 ✓
   BPE 边界变了 ✓ 不报错 ✗。
2. **标签段必须是 1 个 token** ✓（``<Picture 1>: `` 靠词表里登记的特殊 token 一次成词 ✓）。
   切碎了 ⇒ 后面的视觉块位置整体偏移 ✓ 但形状全对 ✓ 不报错 ✗。
3. **视觉块展开成 N 个 ``<|image_pad|>``** ✓，N = ``(h/merge)·(w/merge)`` ✓ ——
   与视觉塔 merger 之后的 token 数**同一个数** ✓。少给/多给都**不报错** ✗（少给 ⇒ 特征塞不下 ✓
   多给 ⇒ 尾部正文被顶走 ✓）。
4. **视觉段之前的 token 逐字不许动** ✓（``token_tags`` 只把 ``[index-1, index+size+1)`` 标 0 ✓，
   参考 ``minimax.py::token_tags_from_embeds_info`` 逐字 ✓）。

## mrope 的 ``grid`` 是 **patch 网格**（未 merge）✗✗

``spans[i]["grid"] = (1, gh, gw)`` 里 ``gh/gw`` 是 **patch 数** ✓（``h_bar/16`` ✓）——
参考 ``minimax.py::process_video_block`` 返回的 ``grid_thw`` 就是这个 ✓，而 mrope 的
``len_max = grid.max() // 2`` 也正是「patch 网格除 merge」✓。
⚠️ 若误把 merge 后的网格塞进去 ⇒ 位置跳一半 ✓ 且不报错 ✗。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from . import h3_prompt as prompt_mod

__all__ = ["H3Condition", "H3TeRuntime", "combine", "encode_condition", "plan_sequence"]


class _Tokenizer(Protocol):
    """只要能 ``encode`` ✓（``text_encoder.Tokenizer`` 与本仓自研 BPE 都满足 ✓）。"""

    def encode(self, text: str) -> list[int]:  # pragma: no cover - 协议
        ...


def plan_sequence(pieces: Sequence[Mapping[str, Any]], tokenizer: _Tokenizer,
                  grids: Sequence[Sequence[int]], *, merge_size: int = 2) -> dict[str, Any]:
    """纯 Python：段序列 ⇒ ``(ids, spans, tags)`` ✓ —— 不需要 torch ✓。

    ``grids`` 按视觉段出现顺序给 **patch 网格** ``(1, gh, gw)`` ✓（``gh*w`` 应是 16 的倍数 ✓）。
    """
    issues = prompt_mod.check_pieces(pieces)
    if issues:
        raise ValueError("呈现格式有问题 ✗：" + "；".join(issues))
    vision_total = sum(1 for piece in pieces if piece.get("kind") == "vision")
    if vision_total != len(grids):
        raise ValueError(
            f"视觉段 {vision_total} 个 ✗、网格给了 {len(grids)} 个 ✓ —— 数量不一致 ⇒ "
            f"位置与特征**对不上** ✓ 且不报错 ✗")
    if int(merge_size) < 1:
        raise ValueError(f"merge_size 必须 ≥ 1（收到 {merge_size} ✗）")

    ids: list[int] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for piece in pieces:
        if piece.get("kind") == "text":
            text = str(piece["text"])
            chunk = [int(value) for value in tokenizer.encode(text)]
            if not chunk:
                raise ValueError(f"文本段 {text!r} 分不出任何 token ✗（空段也会多切一刀 ✗）")
            if prompt_mod._LABEL.match(text) and len(chunk) != 1:  # noqa: SLF001
                raise ValueError(
                    f"标签段 {text!r} 被切成 {len(chunk)} 个 token ✗ —— "
                    f"``<Picture i>``/``<Audio j>``/``<Video k>`` 必须**一次成词** ✓"
                    f"（分词器的 ``additional_special_tokens`` 没登记 ✗）")
            ids.extend(chunk)
            continue

        grid = tuple(int(value) for value in grids[cursor])
        cursor += 1
        if len(grid) != 3:
            raise ValueError(f"网格必须是 ``(t, h, w)`` 三段 ✗（收到 {grid!r}）")
        if grid[0] != 1:
            raise ValueError(
                f"网格的 ``t`` 必须是 **1** ✓（收到 {grid[0]} ✗ —— H3 的每个视觉块都是"
                f"「2 帧进 temporal_patch」✓ ``grid_t`` 恒 1 ✓）")
        height, width = grid[1], grid[2]
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"网格 {height}×{width} 不能被 merge {merge_size} 整除 ✗"
                f"（预处理那次 resize 就该切齐 ✓）")
        count = (height // merge_size) * (width // merge_size)
        ids.append(prompt_mod.VISION_START)
        index = len(ids)
        ids.extend([prompt_mod.IMAGE_PAD] * count)
        ids.append(prompt_mod.VISION_END)
        spans.append({"index": index, "size": count, "grid": grid})

    if not ids:
        # ⚠️ 参考兜底：``if len(entries) == 0: entries.append((151643, 1.0))`` ✓（空提示词 ✓）
        ids.append(prompt_mod.PAD_TOKEN_ID)
    tags = prompt_mod.token_tags(len(ids), [(span["index"], span["size"]) for span in spans])
    return {"ids": ids, "spans": spans, "tags": tags}


@dataclass(frozen=True)
class H3Condition:
    """编好的文本条件 ✓（``hidden`` 直接喂 DiT ✓、``tags`` 走 adaLN ✓）。"""

    hidden: Any
    tags: Any
    ids: list[int]
    spans: list[dict[str, Any]]
    grids: list[tuple[int, int, int]]

    @property
    def seq_len(self) -> int:
        return len(self.ids)


def _torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class H3TeRuntime:
    """H3 文本塔（语言塔 ＋ 视觉塔）的**结构事实** ✓ —— ``attach_h3_text`` 装完挂到 ``_textConfig`` ✓。

    ⚠️ 为什么不直接把 :func:`~.qwen3vl_te.infer_shape` 那份 dict 挂上去 ✗✗：下游三处要的是
    **属性** ✓ —— ``_te_tokenizer`` 读 ``shape.vocab`` ✓、`_check_text_encoder` 读 ``output_dim`` ✓、
    `_te_report` 调 ``to_dict()`` ✓。dict 挂上去 ⇒ ``getattr(config, "output_dim", 0)`` 拿 **0**
    ⇒ 跨来源校验（TE 输出宽 ↔ 主干 ``text_dim``）**静默跳过** ✓✗（不报错、也从不核对）；
    ``_te_tokenizer`` 那处则直接 ``AttributeError`` ✓。两种都不是"报得清楚"的失败方式 ✓。
    """

    text: dict[str, Any]
    vision: dict[str, Any]
    max_length: int = 512

    @property
    def output_dim(self) -> int:
        """语言塔 hidden ✓（= H3 主干的 ``text_dim`` ✓）。"""
        return int(self.text.get("hidden") or self.text.get("textDim") or 0)

    @property
    def vocab(self) -> int:
        return int(self.text.get("vocab") or 0)

    @property
    def shape(self) -> dict[str, Any]:
        return dict(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {"family": "h3-qwen3vl", "text": dict(self.text),
                "vision": dict(self.vision), "max_length": int(self.max_length),
                "output_dim": self.output_dim, "vocab": self.vocab}


def combine(text_tower: Any, vision_tower: Any) -> Any:
    """把语言塔与视觉塔拼成**一个** ``H3TextEncoder`` ✓ ⇒ ``state_dict()`` 键正好是
    ``model.*`` ＋ ``visual.*`` ✓ 与盘上**一一对应** ✓（一个文件一次装载 ✓）。

    ⚠️ 语言塔那半**必须**挂成 ``self.model`` ✗✗ —— 盘上的键就是 ``model.layers.N...`` ✓
    （:func:`~.qwen3vl_te.build_text_tower` 建出来本来就叫 ``model`` ✓）；
    视觉塔那半挂成 ``self.visual`` ✓ 对上 ``visual.*`` ✓。
    """
    from torch import nn

    class H3TextEncoder(nn.Module):
        """语言塔 + 视觉塔 ✓（键名与盘上一一对应 ✓）。"""

        def __init__(self) -> None:
            super().__init__()
            # ⚠️ 只挂**内部那两层** ✓：``text_tower`` 自己是个壳 ⇒ 挂进来会变成
            #    ``text.model.*`` ✗ 与盘上的 ``model.*`` 对不上 ✓✗
            self.model = text_tower.model
            self.visual = vision_tower
            # ⚠️ 原壳用 ``object.__setattr__`` 存 ⇒ **不进** Module 树 ✓（否则同一批参数
            #    注册两次 ✓ 键名翻倍 ✓ 装载时报一堆「多出来的键」✗）
            object.__setattr__(self, "_text", text_tower)
            self.shape = dict(text_tower.shape)
            self.visionShape = dict(getattr(vision_tower, "shape", {}) or {})

        def forward(self, **kwargs: Any) -> Any:
            return object.__getattribute__(self, "_text")(**kwargs)

        def encode(self, pieces: Sequence[Mapping[str, Any]], *,
                   tokenizer: _Tokenizer, images: Sequence[Any] | None = None,
                   video: Any = None, merge_size: int = 2) -> H3Condition:
            """高层入口 ✓ —— :func:`encode_condition` 的同一条路 ✓（视觉塔走 ``self.visual`` ✓）。

            ⚠️⚠️ 这里**必须**写 ``self.visual`` ✗✗ —— 写成 ``object.__getattribute__(self, "visual")``
            取不到 ✓✗：``nn.Module`` 把子模块放在 ``self._modules`` 里 ✓、靠 ``__getattr__`` 兜着 ✓，
            而 ``object.__getattribute__`` **绕过**了那层兜底 ✓ ⇒ 一调就
            ``AttributeError: 'H3TextEncoder' object has no attribute 'visual'`` ✓✗
            （2026-11-01 真权重端到端实测就是这个错 ✓ —— 装配、装载、切分全绿 ✓，
            死在第一句文本编码 ✓）。
            ⚠️ 上面 ``_text`` 那格**相反** ✓：它是 ``object.__setattr__`` 存进 ``__dict__`` 的 ✓
            （有意不进 Module 树 ✓，否则同一批参数注册两次 ✓ 键名翻倍 ✗）⇒ 取它**就得**用
            ``object.__getattribute__`` ✓。两种写法看着对称 ✓ 实际语义相反 ✗✗。
            """
            return encode_condition(self, self.visual, pieces,
                                    tokenizer=tokenizer, images=images, video=video,
                                    merge_size=merge_size)

    return H3TextEncoder()


def encode_condition(tower: Any, vision: Any, pieces: Sequence[Mapping[str, Any]], *,
                     tokenizer: _Tokenizer, images: Sequence[Any] | None = None,
                     video: Any = None, merge_size: int = 2) -> H3Condition:
    """串起「呈现格式 → 视觉塔 → 语言塔」✓ ⇒ :class:`H3Condition` ✓。

    ``images``：``(H, W, C)`` 的序列 ✓（按 ``<Picture i>`` 的顺序 ✓）
    ``video``：``(帧, H, W, C)`` 的帧序列 ✓（块按 pieces 里的 ``frames`` 区间切 ✓）——
    单路传**一个**张量 ✓、多路传**列表**（按 ``<Video k>`` 的顺序 ✓，一路一个 ✓）。

    ⚠️ 每个视觉块**单独**过视觉塔 ✓（块间不许互相看 ✓ —— 一次把多块拼起来喂 ⇒
    注意力会跨块 ✓ 但形状全对 ✓ 不报错 ✗）。
    """
    from . import qwen3vl_pixels as pixels_mod

    torch = _torch()
    image_list = list(images or ())
    picture_cursor = 0
    #: ⚠️ ``video`` 两种形态都收 ✓：单张张量（一路 ✓ —— 老调用口的写法 ✓）或**列表**（多路 ✓）。
    #: 分不清就按"有没有 ``dim``"判 ✓（``torch.Tensor`` 有 ✓，list/tuple 没有 ✓），
    #: 不按"长度像不像帧数"猜 ✗（一路 2 帧、两路各 2 帧在长度上是同一个 2 ✓✗）。
    if video is None:
        video_list: list[Any] = []
    elif hasattr(video, "dim"):
        video_list = [video]
    else:
        video_list = list(video)
    #: ⚠️⚠️ **视觉塔的输入也要跟着权重走** ✓✗（2026-11-01 真权重端到端实测 ✓）：
    #: 这里同样是**直接调** ``vision(...)`` ✓ —— 不经过 `H3TextEncoder.forward` ✓，
    #: 所以 `residency` 挂在父模块上的入口 pre-hook 帮不上忙 ✗。像素是预处理在 CPU 上造的 ✓、
    #: 视觉塔在卡上 ✓ ⇒ 卷积当场报「Input type (torch.FloatTensor) and weight type
    #: (torch.cuda.FloatTensor) should be the same」✓✗。
    #: 取**第一个参数**的设备 ✓（= `patch_embed` 所在那台 ✓，`parameters()` 按声明顺序 ✓）——
    #: 协同驻留下视觉塔也可能被切 ✓，那时 patch_embed 留在哪台就喂哪台 ✓。
    #: ⚠️ ``vision`` **不一定是 ``nn.Module``** ✓✗：这里收的是任意 callable ✓，
    #: 测试替身就是一只普通函数 ✓（没有 ``parameters`` ✗）。硬调 ``vision.parameters()``
    #: ⇒ ``AttributeError: 'function' object has no attribute 'parameters'`` ✓✗
    #: （2026-11-01 真权重端到端修完这片后，`engine_h3_text_test` 当场被这条打崩 ✓）。
    #: 取不到参数 ⇒ ``vision_device`` 留 ``None`` ⇒ **不搬张量** ✓：非 Module 的替身
    #: 自身没有设备概念 ✓，喂进来的像素在哪台它就在哪台算 ✓；真模型一定有参数 ✓，
    #: 所以 ``None`` 这条分支只在替身/无参占位上走得到 ✓（不是"静默当 CPU"✗）。
    vision_device = None
    _parameters = getattr(vision, "parameters", None)
    if callable(_parameters):
        for parameter in _parameters():
            vision_device = parameter.device
            break
    per_block: list[tuple[Any, tuple[int, int, int], Any]] = []
    for piece in pieces:
        if piece.get("kind") != "vision":
            continue
        if piece.get("videoBlock"):
            #: ⚠️ 挑**哪一路**只看 ``videoIndex`` ✓✗：``frames`` 的区间是**本路内**的下标 ✓，
            #: 每路都从 0 起 ✓ —— 靠"区间从 0 起就换路"去推 ⇒ 第一路只有一块（2 帧）时，
            #: 第二路的第一块也是 ``(0, 2)`` ⇒ 把第二路的帧喂成第一路 ✓、形状全对 ✓✗。
            index = int(piece.get("videoIndex") or 1)
            if not video_list:
                raise ValueError("有视频块 ✗ 但没给 ``video`` ✓（帧序列 ✓）")
            if not 1 <= index <= len(video_list):
                raise ValueError(
                    f"视频块标的是第 {index} 路 ✗，但只给了 {len(video_list)} 路帧序列 ✓"
                    f"（少给 ⇒ 后面整体错位 ✓ 不报错 ✗）")
            source = video_list[index - 1]
            start, stop = (int(value) for value in piece["frames"])
            block = source[start:stop]
            if int(block.shape[0]) != prompt_mod.FRAMES_PER_BLOCK:
                raise ValueError(
                    f"视频块 [{start}, {stop}) 切出 {int(block.shape[0])} 帧 ✗ ≠ "
                    f"{prompt_mod.FRAMES_PER_BLOCK} ✓（先过 ``pad_frames_even`` ✓）")
            flatten, grid = pixels_mod.process_video_block(block)
        else:
            if picture_cursor >= len(image_list):
                raise ValueError(
                    f"有 {picture_cursor + 1} 个单图视觉块 ✗ 但只给了 {len(image_list)} 张图 ✓"
                    f"（少给 ⇒ 后面整体错位 ✓ 不报错 ✗）")
            flatten, grid = pixels_mod.process_image(image_list[picture_cursor])
            picture_cursor += 1
        if vision_device is not None and flatten.device != vision_device:
            flatten = flatten.to(vision_device)
        grid_tensor = torch.tensor([grid], dtype=torch.long, device=flatten.device)
        if vision_device is not None and grid_tensor.device != vision_device:
            grid_tensor = grid_tensor.to(vision_device)
        merged, deepstack = vision(flatten, grid_tensor)
        per_block.append((grid, merged, deepstack))
    if picture_cursor != len(image_list):
        raise ValueError(
            f"给了 {len(image_list)} 张图 ✗ 但只有 {picture_cursor} 个单图视觉块 ✓"
            f"（多给的图会被**静默丢弃** ✓ 创作者以为放了两张 ✗）")

    grids = [grid for grid, _merged, _stack in per_block]
    plan = plan_sequence(pieces, tokenizer, grids, merge_size=merge_size)

    hidden_size = int(tower.shape["hidden"])
    if per_block:
        merged_all = torch.cat([merged for _g, merged, _s in per_block], dim=0)
        depth = len(per_block[0][2])
        deepstack_all = [torch.cat([stack[index] for _g, _m, stack in per_block], dim=0)
                         for index in range(depth)]
        if int(merged_all.shape[1]) != hidden_size:
            raise ValueError(
                f"视觉塔出 {int(merged_all.shape[1])} 宽 ✗ ≠ 语言塔 hidden {hidden_size} ✓"
                f"（merger 的 ``out_hidden_size`` 必须等于语言塔 hidden ✓）")
        total = sum(int(span["size"]) for span in plan["spans"])
        if int(merged_all.shape[0]) != total:
            raise ValueError(
                f"视觉特征 {int(merged_all.shape[0])} 个 ✗ ≠ imgpad 位置 {total} 个 ✓"
                f"（对不上 ⇒ 注入会错位 ✓ 不报错 ✗）")
    else:
        merged_all, deepstack_all = None, None

    torch_mod = _torch()
    #: ⚠️⚠️ **词表权重在哪台设备，`ids` 与视觉特征就得跟到哪** ✗✗（2026-11-01 真权重端到端实测 ✓）：
    #: 这里是**直接调** ``tower.model.embed_tokens(...)`` ✓ —— 不经过顶层 `forward` ✓，
    #: 所以 :mod:`residency` 那个「入口兜底」pre-hook（挂在 `H3TextEncoder` 上 ✓）**不会执行** ✗。
    #: GPU/CPU 协同驻留下 `embed_tokens` 在卡上 ✓、而 `ids` 是这里在 CPU 上造的 ✓ ⇒
    #: `F.embedding` 内部一句 `index_select` 就炸 ✓✗：
    #: ``Expected all tensors to be on the same device, but got index is on cpu,
    #: different from other tensors on cuda:0`` ✓ —— 报错点在 embedding ✓，
    #: 读起来像"词表坏了" ✓，其实是**两台设备各拿一半**的正常形态 ✓。
    embed_weight = tower.model.embed_tokens.weight
    ids_tensor = torch_mod.tensor([plan["ids"]], dtype=torch_mod.long,
                                  device=embed_weight.device)
    embeds = tower.model.embed_tokens(ids_tensor)
    if merged_all is not None:
        #: ⚠️ 视觉塔与语言塔**可能不在同一台设备**（各自切分 ✓）⇒ 视觉特征要跟到 embeds 那边 ✓，
        #: 只转 dtype 不转 device ⇒ 下面这句切片赋值报「device 不一致」✓✗。
        cursor = 0
        for span in plan["spans"]:
            start, size = int(span["index"]), int(span["size"])
            embeds[:, start:start + size, :] = merged_all[cursor:cursor + size].to(
                device=embeds.device, dtype=embeds.dtype)
            cursor += size
    hidden = tower(embeds=embeds, spans=plan["spans"], deepstack=deepstack_all)
    tags = torch_mod.tensor(plan["tags"], dtype=torch_mod.long)
    return H3Condition(hidden=hidden, tags=tags, ids=plan["ids"],
                       spans=plan["spans"], grids=grids)
