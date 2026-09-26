"""S7 自检：**SDXL UNet**（自研的「图」扩散主干 ✓ 2026-09-26）。

判据分两类，都是**可判真假的事实** ✓（本仓纪律：不把没验的说成验过 ✗）：

* **口径类**（零依赖的小张量 ✓）：时间嵌入 **cos 在前** ✗✗（拿 `t=0` 一眼看穿：前半必是 1 ✓、
  后半必是 0 ✓）、ADM 的 **6 个 id** 与拼接顺序 ✓、`GroupNorm` eps 两档（1e-5 / 1e-6 ✓）、
  `to_q` 无偏置 ✓、GEGLU 出两倍宽 ✓、**`out.2` 零初始化 ⇒ 新模型输出恒 0** ✗✗（最硬的一条 ✓）；
* **结构类**：缩小版走**同一条前向** ✓（含**奇数边长** 13 ⇒ 那条 `output_shape` 的路 ✓），
  再拿**本机真权重头**（`sd_xl_base_1.0.safetensors` ✓ 只读元数据、**不载张量** ✗）逐键逐形状核 ——
  **1680 键全等 + 参数量 2,567,463,684 全等** ✗✗ ⇒ 布局读错（少一个下采样 / 上行不是 `cat` ✗）**必红** ✓。

⚠️ 那份检查点是**扫描器找出来的** ✓（`find_sdxl_base_checkpoint` ✓ 走 `local_model_scan` ✓，
本文件**没有机器路径** ✗ ✗）；没装 torch / 扫不到 ⇒ 显式 SKIP ✓（**没跑 ≠ 绿** ✗、不是通过 ✗）。
⚠️ 参考实现（`reference/ComfyUI`）**导不进来**（本机 `comfy_kitchen` 版本对不上 ✗）⇒ 不做数值对拍 ✗；
本套改成「**真权重头逐键对齐** + 口径逐条见出处」✓ —— 结论按实测说 ✓，不夸大 ✗。

运行::

    ./.venv/Scripts/python.exe tests/engine_sdxl_test.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import sdxl as mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


#: SDXL **主权重**的文件名口径 ✓：先**扫描**出来 ✓ 再按名字筛 ✓（⚠️ 本文件**不写死机器路径** ✗）
SDXL_BASE_NAME = re.compile(r"sd_?xl[_-]?base", re.IGNORECASE)
#: 真权重实测：`model.diffusion_model.*` 的键数与参数量 ✓（见模块头 ✓）
REAL_KEYS = 1680
REAL_PARAMS = 2567463684


def find_sdxl_base_checkpoint() -> tuple[Path | None, str]:
    """找本机那份 SDXL base 主权重 —— **走本仓扫描器** ✓（`local_model_scan` ✓），不写死路径 ✗。

    扫描根**只有一个来源** ✓：`local_model_scan.get_default_roots` ✓（`MODELS_DIR` / `COMFYUI_PATH` /
    `configs/model-paths.json` / HF 缓存 / ModelScope / LM Studio / ComfyUI 类目录 … ✓）——
    本文件**一个新路径都不加** ✗（⚠️ 2026-09-26 用户口径 ✓：不许写死路径 ✗）。

    ⚠️ 走过的弯路（留档 ✓）：早先 `detect_comfyui()` 只取**第一个**命中 ⇒ 本机挑中
    `ComfyUI-Installs/…/ComfyUI` 之后 **`ComfyUI-Shared/models` 被遮住** ✗（那份 SDXL 恰在那儿 ✓，
    实测只走默认根**扫不到** ✗ ）—— 当时本套临时「并上候选表」绕过 ✓；**2026-09-26 已从根上修** ✓
    （`detect_comfyui_roots` 收**全部**命中 ✓、源码里也没有 `D:/…` 字面量了 ✓）⇒ 这里**回到只走默认根** ✓。
    ⚠️ 只认 `sd_xl_base` / `sdxl_base` ✗：**refiner 不算** ✓（它 `adm_in_channels` = 2560 ⇒
    键集与参数量都不同 ✓，拿它来对 1680 键会红得莫名其妙 ✗ —— 真要测 refiner 得另立配置 ✓）。

    返回 `(路径, 说明)`：路径为 `None` 时，说明里写清「为什么找不到」供 SKIP 如实报 ✓（**不许当通过** ✗）。
    """
    try:
        from app.services import local_model_scan as scan  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001
        return None, f"导不进本仓扫描器（{error}）⇒ ⑦ 真权重对齐不跑 ✓"
    roots = list(scan.get_default_roots())
    try:
        scanned = scan.scan_local_models({"roots": roots, "kinds": ["image"], "maxFiles": 20000})
    except Exception as error:  # noqa: BLE001
        return None, f"扫描器报错（{error}）⇒ ⑦ 真权重对齐不跑 ✓"
    hits = sorted(
        (
            item for item in scanned["models"]
            if item.get("role") == "standalone"
            and SDXL_BASE_NAME.search(str(item.get("filename") or ""))
        ),
        key=lambda item: str(item.get("path") or ""),
    )
    if not hits:
        cut = ("（⚠️ 扫描被 `maxFiles` 截断了 ⇒「扫不到」也可能只是**没扫完** ✓ 不许当通过 ✗）"
               if scanned["truncated"] else "")
        return None, (f"扫描器扫了 {len(roots)} 个根 / {scanned['total']} 个图类权重，"
                      f"没有 `sd_xl_base*` ⇒ ⑦ 真权重对齐不跑 ✓{cut}")
    return Path(str(hits[0]["path"])), f"扫描器命中 {len(hits)} 份"

#: 缩小版：**同一条前向** ✓、CPU 秒级 ✓（通道仍满足 `norm_num_groups` / `num_head_channels` 的整除 ✓）。
#: `adm_in_channels = 16 + 6×256` ⇒ 可以直接喂 `build_adm()` 的输出 ✓（ADM 那条路一并验到 ✓）。
TOY = dict(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    channel_mult=(1, 2),
    num_res_blocks=1,
    context_dim=16,
    adm_in_channels=16 + mod.ADM_TIME_IDS * mod.ADM_TIME_DIM,
    num_head_channels=32,
    norm_num_groups=8,
    transformer_depth=(0, 2),
    transformer_depth_middle=2,
    transformer_depth_output=(2, 2, 0, 0),
    zero_init_out=False,
)


def case_config() -> None:
    """① 配置口径 ✓：默认值就是官方 SDXL base ✓、派生量与校验都是**算出来**的 ✓。"""
    default = mod.SdxlUnetConfig()
    check("① 常量：`ADM_TIME_IDS=6` ✓ / `ADM_TIME_DIM=256` ✓ ⇒ 6×256+1280 = **2816** ✓",
          mod.ADM_TIME_IDS == 6 and mod.ADM_TIME_DIM == 256
          and mod.ADM_TIME_IDS * mod.ADM_TIME_DIM + 1280 == 2816)
    check("①′ 默认 = 官方 SDXL base ✓（320 / (1,2,4) / ctx 2048 / adm 2816 / 头 64 / 深度 (0,0,2,2,10,10) ✓）",
          (default.model_channels, default.channel_mult, default.context_dim,
           default.adm_in_channels, default.num_head_channels, default.transformer_depth)
          == (320, (1, 2, 4), 2048, 2816, 64, (0, 0, 2, 2, 10, 10)),
          default.describe())
    check("①″ 派生量：`time_embed_dim=320×4=1280` ✓、上行深度 9 项 ✓、"
          "`attends(0)=False`（320 层没注意力 ✓）/ `attends(2)=True` ✓、`heads_at(1280)=20` ✓",
          default.time_embed_dim == 1280 and len(default.transformer_depth_output) == 9
          and default.attends(0) is False and default.attends(2) is True
          and default.heads_at(1280) == 20 and default.num_levels == 3)
    check("①‴ ⚠️ 上行深度**按上行顺序**排 ✓（1280 层在前 ✓）—— 与 ComfyUI 配置里那份"
          "**反着写**的 `[0,0,0,2,2,2,10,10,10]` 是同一条事实的两种写法 ✓"
          "（它那边从尾部 `pop()` ✓）；两处**同时反**才对 ✓",
          default.transformer_depth_output == (10, 10, 10, 2, 2, 2, 0, 0, 0)
          and tuple(reversed(default.transformer_depth_output)) == (0, 0, 0, 2, 2, 2, 10, 10, 10))
    check("①⁴ 深度表长度写错 / proj 不是 Linear 的排布 / 通道除不尽分组 ⇒ **各自报错** ✗（不猜 ✗）",
          _raises(lambda: mod.SdxlUnetConfig(transformer_depth=(0, 0))) is not None
          and _raises(lambda: mod.SdxlUnetConfig(transformer_depth_output=(1,))) is not None
          and _raises(lambda: mod.SdxlUnetConfig(use_linear_in_transformer=False), "Linear") is not None
          and _raises(lambda: mod.SdxlUnetConfig(model_channels=30, channel_mult=(1,),
                                                 transformer_depth=(0, 0),
                                                 transformer_depth_output=(0, 0, 0)),
                      "整除") is not None)


def case_timestep() -> None:
    """② 时间嵌入 ✓：**cos 在前、sin 在后** ✗✗（`t=0` 一眼看穿 ✓）。"""
    import torch  # noqa: PLC0415

    emb = mod.timestep_embedding(torch.zeros(1), 8)
    check("② `t=0` ⇒ **前半 cos(0)=1** ✓、**后半 sin(0)=0** ✓ —— ⚠️ 反了就是 diffusers 的另一种口径 ✗✗"
          "（写反不会报错 ⇒ 只会让出图变味 ✓✗）",
          bool(torch.equal(emb[:, :4], torch.ones(1, 4)))
          and bool(torch.equal(emb[:, 4:], torch.zeros(1, 4))), emb.tolist())
    value = 1.0
    quarter = mod.timestep_embedding(torch.tensor([value]), 4)
    expect = torch.tensor([[torch.cos(torch.tensor(value)), torch.cos(torch.tensor(value / 100)),
                            torch.sin(torch.tensor(value)), torch.sin(torch.tensor(value / 100))]])
    check("②′ 频率按 `exp(-ln(10000)·i/(dim/2))` 铺 ✓（dim=4 ⇒ ω=1 与 ω=1/100 ✓）"
          "⇒ `cos(1), cos(0.01), sin(1), sin(0.01)` ✓",
          bool(torch.allclose(quarter, expect, atol=1e-6)), quarter.tolist())
    odd = mod.timestep_embedding(torch.zeros(1), 5)
    check("②″ 奇数维 ⇒ 末位补 0 ✓（同参考实现 ✓）、形状 `(B, dim)` ✓",
          tuple(odd.shape) == (1, 5) and float(odd[0, -1]) == 0.0)
    check("②‴ `dim<=0` ⇒ 报错 ✗（不许返回空张量糊过去 ✗）",
          _raises(lambda: mod.timestep_embedding(torch.zeros(1), 0)) is not None)


def case_adm() -> None:
    """③ ADM 拼接 ✓：`(B,1280) ++ 6×(B,256)` ⇒ `(B,2816)` ✓、**顺序**钉死 ✓。"""
    import torch  # noqa: PLC0415

    text = torch.randn(2, 1280)
    ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
                        [768.0, 1344.0, 10.0, 20.0, 512.0, 512.0]])
    adm = mod.build_adm(text, ids)
    check("③ 形状 `(B, 2816)` ✓（1280 文本 + 6×256 ✓）、前 1280 是**原样**的池化文本 ✓",
          tuple(adm.shape) == (2, 2816) and bool(torch.equal(adm[:, :1280], text)), tuple(adm.shape))
    width = mod.ADM_TIME_DIM
    segments = [adm[:, 1280 + index * width:1280 + (index + 1) * width] for index in range(6)]
    check("③′ ⭐ 第 k 段只由第 k 个 id 决定 ✓（**顺序**：height / width / crop_top / crop_left / "
          "target_h / target_w ✓）—— 改一个 id 只该动它自己那一段 ✓",
          all(bool(torch.equal(segments[index], mod.timestep_embedding(ids[:, index], width)))
              for index in range(6)))
    check("③″ ⭐ 6 个 id 是**各算各的** ✓（同尺寸出图时前后也可以不同 ✓）—— 不是「同一个值重复 6 遍」✗",
          not bool(torch.equal(segments[0], segments[5])))
    check("③‴ id 个数不是 6 / 文本不是二维 ⇒ **报错** ✗（绝不静默补零 ✗）",
          _raises(lambda: mod.build_adm(text, ids[:, :5])) is not None
          and _raises(lambda: mod.build_adm(text[0], ids)) is not None)


def case_structure() -> None:
    """④ 结构 ✓：键模式 + 通道账 + eps 两档 + 偏置 + 头数 ✓（缩小版 ✓）。"""
    import torch  # noqa: PLC0415

    model = mod.build_sdxl_unet(mod.SdxlUnetConfig(**TOY))
    state = model.state_dict()

    def has(key: str) -> bool:
        return key in state

    check("④ 键模式与真权重**同形** ✓：`input_blocks.0.0.weight`（conv_in ✓）、"
          "`input_blocks.2.0.op.weight`（下采样是**独立一层** ✓）、"
          "`output_blocks.1.2.conv.weight`（上采样在**那一层的最后** ✓）",
          has("input_blocks.0.0.weight") and has("input_blocks.2.0.op.weight")
          and has("output_blocks.1.2.conv.weight"))
    check("④′ ⚠️ 通道相等 ⇒ **没有** `skip_connection` 键 ✗（参考实现给 `Identity` ✓ ⇒ 真权重里也没那些键 ✓）："
          "32→32 无 ✓ / 通道变了才有 ✓",
          not has("input_blocks.1.0.skip_connection.weight")
          and has("input_blocks.3.0.skip_connection.weight"),
          [k for k in state if "skip_connection" in k][:4])
    check("④″ ⭐⭐ 上行 ResBlock 吃 **`cat([h, skip])`** ✗✗ ⇒ 输入维 = 当前通道 + skip 通道 ✓"
          "（真权重实测 `output_blocks.0.0.in_layers.2` = (1280, **2560**) ✓）；"
          "缩小版最上一层是 64 通道、skip 也是 64 ⇒ `in_layers.2` = (64, **128**, 3, 3) ✓"
          "（只吃 skip 的话会是 (64, 64, 3, 3) ✗）",
          tuple(state["output_blocks.0.0.in_layers.2.weight"].shape) == (64, 128, 3, 3),
          tuple(state["output_blocks.0.0.in_layers.2.weight"].shape))
    check("④‴ eps **两档** ✓：ResBlock 的 GroupNorm = **1e-5** ✓、SpatialTransformer 的 `norm` = **1e-6** ✗✗"
          "（抄成同一个数不会报错 ⇒ 只让出图悄悄变味 ✓✗）",
          model.input_blocks[1][0].in_layers[0].eps == 1e-5
          and model.middle_block[1].norm.eps == 1e-6
          and model.out[0].eps == 1e-5,
          (model.input_blocks[1][0].in_layers[0].eps, model.middle_block[1].norm.eps))
    attention = model.middle_block[1].transformer_blocks[0]
    check("④⁴ 注意力口径 ✓：`to_q`/`to_k` **无偏置** ✓、`to_out.0` **有偏置** ✓、"
          "前馈 **GEGLU**（`net.0.proj` 出两倍宽 → `net.2` 收窄 ✓）、三层全是 **LayerNorm** ✓"
          "（用 BatchNorm/GroupNorm 换上去会在这里露馅 ✗）",
          attention.attn1.to_q.bias is None and attention.attn1.to_out[0].bias is not None
          and attention.attn2.to_k.bias is None
          and tuple(attention.ff.net[0].proj.weight.shape) == (512, 64)
          and tuple(attention.ff.net[2].weight.shape) == (64, 256)
          and all(isinstance(getattr(attention, name), torch.nn.LayerNorm)
                  for name in ("norm1", "norm2", "norm3")),
          (attention.attn1.to_q.bias, tuple(attention.ff.net[0].proj.weight.shape)))
    check("④⁵ 头数 = 通道 / `num_head_channels` ✓（缩小版 64/32 = **2 头** ✓、官方 1280/64 = **20 头** ✓）"
          "⇒ 是**算出来**的 ✓（真权重逐键对齐也逼出这一条 ✓）",
          attention.attn1.heads == 2 and attention.attn1.dim_head == 32
          and attention.attn1.to_q.out_features == 64
          and mod.SdxlUnetConfig().heads_at(1280) == 20)
    check("④⁶ 缩小版的块数账 ✓：下行 `1 + 层数×(res+下采样)` = **4** 项 ✓、上行 = `(res+1)×层数` = **4** 项 ✓、"
          "且**只有高通道那层**的上行末项带上采样 ✓（`output_blocks.1.2.conv` ✓ 而有 `output_blocks.3.2` ✗ 无）",
          len(model.input_blocks) == 4 and len(model.output_blocks) == 4
          and has("output_blocks.1.2.conv.weight") and has("output_blocks.3.2.conv.weight") is False)
    check("④⁷ 真权重的**顶层杂项**逐个对得上 ✓：`time_embed.0/2` ✓、`label_emb.0.0`/`label_emb.0.2`"
          "（外层再套一层 Sequential ✓）、`out.0`（GroupNorm ✓）/`out.2`（3x3 ✓）",
          has("label_emb.0.0.weight") and has("label_emb.0.2.weight")
          and has("time_embed.0.weight") and has("time_embed.2.weight")
          and has("out.0.weight") and has("out.2.bias")
          and tuple(model.out[2].weight.shape) == (4, 32, 3, 3))
    rows = model.structure()
    check("④⁸ `structure()` 逐**层**列全实际排布 ✓（键索引就是行里的两个数字 ✓）："
          "缩小版共 **15** 行（下行 5 + 中间 3 + 上行 7 ✓）、首行是 conv_in ✓、"
          "中间块三层的行号正是 `middle_block.0.0/.1/.2` ✓",
          len(rows) == 15 and rows[0].startswith("input_blocks.0.0 Conv2d")
          and [row.split()[0] for row in rows if row.startswith("middle_block")]
          == ["middle_block.0.0", "middle_block.0.1", "middle_block.0.2"],
          rows)


def case_forward() -> None:
    """⑤ 前向 ✓：**零初始化 ⇒ 恒零** ✗✗、条件真接上了 ✓、奇数边长 ✓、报错路径 ✓。"""
    import torch  # noqa: PLC0415

    torch.manual_seed(7)
    zero_model = mod.build_sdxl_unet(mod.SdxlUnetConfig(**dict(TOY, zero_init_out=True))).eval()
    context = torch.randn(2, 5, TOY["context_dim"])
    timesteps = torch.full((2,), 500.0)
    adm = mod.build_adm(torch.randn(2, 16), torch.full((2, 6), 1024.0))
    with torch.no_grad():
        out = zero_model(torch.randn(2, 4, 8, 8), timesteps, context, adm=adm)
    check("⑤ ⭐⭐ `out.2` **零初始化** ⇒ 未训练的新模型输出**恒等于 0** ✗✗（LDM 的 `zero_module` ✓）"
          "—— 同时证明**维数/设备全对** ✓（跑得到最后那层才会出 0 ✓）",
          tuple(out.shape) == (2, 4, 8, 8) and float(out.abs().max()) == 0.0, float(out.abs().max()))

    torch.manual_seed(11)
    live = mod.build_sdxl_unet(mod.SdxlUnetConfig(**TOY)).eval()
    torch.nn.init.normal_(live.out[2].weight, std=0.05)
    torch.nn.init.normal_(live.out[2].bias, std=0.05)
    image = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        base = live(image, timesteps, context, adm=adm)
        again = live(image, timesteps, context, adm=adm)
        other_context = live(image, timesteps, torch.randn(2, 5, TOY["context_dim"]), adm=adm)
        other_adm = live(image, timesteps, context,
                         adm=mod.build_adm(torch.randn(2, 16), torch.full((2, 6), 256.0)))
        no_adm = live(image, timesteps, context)
        other_time = live(image, torch.full((2,), 10.0), context, adm=adm)
    check("⑤′ 解零后输出**非零** ✓（⇒ 上一条的 0 确实来自零初始化 ✓，不是「整条链断了」✗）、"
          "形状与输入**逐维相等** ✓（这是 UNet 不是上采样器 ✗）、"
          "同一输入两次**逐位相同** ✓（无残留状态/无随机 ✓）",
          float(base.abs().max()) > 0.0 and tuple(base.shape) == (2, 4, 8, 8)
          and bool(torch.equal(base, again)), float(base.abs().max()))
    check("⑤″ ⭐ **交叉注意力真的接上了** ✓：换 context 输出就变 ✓（把 context 丢掉/丢弃的实现会在这里露馅 ✗）",
          not bool(torch.equal(base, other_context)), float((base - other_context).abs().max()))
    check("⑤‴ ⭐ **ADM 真的接上了** ✓：换 `adm` 输出就变 ✓；**不给 adm 也能跑** ✓ 但结果与给了的不同 ✓"
          "（真权重有 `label_emb` ⇒ 出图路径必须给 ✓）",
          not bool(torch.equal(base, other_adm)) and not bool(torch.equal(base, no_adm)),
          (float((base - other_adm).abs().max()), float((base - no_adm).abs().max())))
    check("⑤⁴ ⭐ 时间步真的接上了 ✓：`t` 从 500 换到 10 输出就变 ✓（抄漏 `emb` 会在这里露馅 ✗）",
          not bool(torch.equal(base, other_time)), float((base - other_time).abs().max()))
    check("⑤⁵ 奇数边长 13 ⇒ 下到 7、再上回 13 ✓（`output_shape` 那条路 ✓："
          "无脑 ×2 得 14 ⇒ 与 skip **拼不上** ✗✗）",
          tuple(live(torch.randn(1, 4, 13, 13), timesteps[:1], context[:1], adm=adm[:1]).shape)
          == (1, 4, 13, 13))
    check("⑤⁶ 非方形 ✓（`H≠W` 时上下采样各自走 ✓）",
          tuple(live(torch.randn(1, 4, 7, 11), timesteps[:1], context[:1], adm=adm[:1]).shape)
          == (1, 4, 7, 11))
    check("⑤⁷ 输入维数/条件维数不对 ⇒ **各自报错** ✗（不静默广播 ✗）",
          _raises(lambda: live(torch.randn(2, 4, 8), timesteps, context), "(B, C, H, W)") is not None
          and _raises(lambda: live(image, timesteps, context[:, :, :8]), "context") is not None
          and _raises(lambda: live(image, timesteps, context, adm=torch.randn(2, TOY["context_dim"])),
                      "adm") is not None)


def case_load() -> None:
    """⑥ 装载守卫 ✓：剥前缀 / 跳过非 UNet 键 / 缺键多键**都要报错** ✗。"""
    import torch  # noqa: PLC0415

    model = mod.build_sdxl_unet(mod.SdxlUnetConfig(**TOY))
    unet = {mod.SDXL_UNET_KEY_PREFIX + key: value for key, value in model.state_dict().items()}
    mixed = dict(unet)
    mixed["conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"] = torch.zeros(1)
    mixed["first_stage_model.encoder.conv_in.weight"] = torch.zeros(1)
    others = mod.load_sdxl_unet_state_dict(model, mixed)
    check("⑥ 带前缀的**自动剥掉** ✓（官方检查点就是 `model.diffusion_model.*` ✓）、"
          "非 UNet 的键**如实返回** ✓（调用方可核「文本编码器/VAE 各多少键」✓）",
          len(others) == 2 and all(not key.startswith(mod.SDXL_UNET_KEY_PREFIX) for key in others),
          others)
    check("⑥′ 一个前缀键都没有 ⇒ **报错** ✗（说明这根本不是 SDXL 检查点 ✓，不许当成通过 ✗）",
          _raises(lambda: mod.load_sdxl_unet_state_dict(model, {"a": torch.zeros(1)}),
                  "不是 SDXL 检查点") is not None
          and _raises(lambda: mod.load_sdxl_unet_state_dict(model, {})) is not None)
    broken = dict(unet)
    broken.pop(mod.SDXL_UNET_KEY_PREFIX + "middle_block.0.emb_layers.1.weight")
    check("⑥″ ⚠️ **少一个键就报错** ✗✗（`strict=False` 会留下随机初始化的层 ⇒ 出图默默变形 ✗）",
          _raises(lambda: mod.load_sdxl_unet_state_dict(model, broken)) is not None)
    extra = dict(unet)
    extra[mod.SDXL_UNET_KEY_PREFIX + "middle_block.9.weight"] = torch.zeros(1)
    check("⑥‴ **多一个键也报错** ✗（说明配置与检查点版本对不上 ✓）",
          _raises(lambda: mod.load_sdxl_unet_state_dict(model, extra)) is not None)
    wrong = dict(unet)
    wrong[mod.SDXL_UNET_KEY_PREFIX + "input_blocks.0.0.weight"] = torch.zeros(4, 4, 3, 3)
    check("⑥⁴ **形状不对也报错** ✗（通道算错最直接的探测器 ✓）",
          _raises(lambda: mod.load_sdxl_unet_state_dict(model, wrong)) is not None)


def case_real_weights() -> None:
    """⑦ 真权重头逐键核对 ✓✓：**只读元数据、不载张量** ✗（秒级 ✓）。

    ⚠️ 那份检查点**由扫描器找** ✓（`find_sdxl_base_checkpoint` ✓ —— 本文件里没有任何机器路径 ✗）；
    没扫到 ⇒ 如实 SKIP ✓（**不是通过** ✗），SKIP 文案里带上「扫了几个根 / 几个权重」✓。
    """
    if not mod.has_torch():
        skip("没装 torch ⇒ ⑦ 真权重对齐不跑 ✓")
        return
    try:
        from safetensors import safe_open  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        skip("没装 safetensors ⇒ ⑦ 真权重对齐不跑 ✓")
        return
    ckpt, note = find_sdxl_base_checkpoint()
    if ckpt is None:
        skip(f"{note}（**不是通过** ✗）")
        return

    with safe_open(str(ckpt), framework="pt") as handle:
        real = {
            key[len(mod.SDXL_UNET_KEY_PREFIX):]: tuple(handle.get_slice(key).get_shape())
            for key in handle.keys()
            if key.startswith(mod.SDXL_UNET_KEY_PREFIX)
        }
    model = mod.build_sdxl_unet()
    mine = model.state_dict()
    check(f"⑦ ⭐⭐ 官方 `{ckpt.name}`（{note} ✓）的 Unet **{len(real)} 键**（实测 {REAL_KEYS} ✓）"
          f"与自研实现**键集全等** ✗✗（多一个缺一个都算红 ✓）",
          len(real) == REAL_KEYS and set(real) == set(mine),
          (sorted(set(real) - set(mine))[:5], sorted(set(mine) - set(real))[:5]))
    mismatched = [
        f"{key}: 我{tuple(mine[key].shape)}≠真{real[key]}"
        for key in sorted(set(real) & set(mine))
        if tuple(mine[key].shape) != real[key]
    ]
    check(f"⑦′ ⭐⭐ **{len(real)} 键逐键形状全等** ✗✗（通道账/头数/深度/上采样的位置全在这一条里 ✓）",
          not mismatched, mismatched[:5])
    total = sum(value.numel() for value in mine.values())
    check(f"⑦″ ⭐⭐ 参数量 **{total:,}** == 真权重 {REAL_PARAMS:,} ✗✗"
          f"（少一个块/多个块必然对不上 ✓ ⇒ 这是「布局读对没有」的总判据 ✓）",
          total == REAL_PARAMS, total - REAL_PARAMS)
    check("⑦‴ 逐块布局如实：下行 9 项 ✓、中间 3 项 ✓、上行 9 项 ✓"
          "（真权重的 `input_blocks.3.0.op` 是下采样 ✓、`output_blocks.2.2.conv` 是上采样 ✓）",
          len(model.input_blocks) == 9 and len(model.middle_block) == 3
          and len(model.output_blocks) == 9
          and "input_blocks.3.0.op.weight" in real and "output_blocks.2.2.conv.weight" in real)


def main() -> int:
    if not mod.has_torch():
        skip("没装 torch ⇒ ①~⑥ 全不跑 ✓（**没跑 ≠ 绿** ✗）")
    else:
        for case in (case_config, case_timestep, case_adm, case_structure, case_forward, case_load):
            case()
    case_real_weights()

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
