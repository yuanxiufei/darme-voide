"""族 ⇒ 骨架的**唯一装配表**（2026-11-01）。

## 为什么要有这一个文件

``torch_backend`` 里原先按族名 ``if/elif`` 分四个构建函数（``zimage`` /
``qwen_image21`` / ``qwen_image`` / ``krea2``），每个函数都**独立抄了一遍同一套样板**：

1. 从 ``info.tensors`` 抽形状表；
2. ``infer_shape(...)``，把 ``ShapeError`` 包成「推不出可装的结构」；
3. 默认精度（有卡 bf16 / 无卡 fp32）；
4. ``meta`` 建壳 ⇒ 降 dtype ⇒ 物化；
5. 把「权重里写着、但后端协议按**别的名字**问」的字段搬进 config；
6. 写 ``configSources``（哪些是权重事实、哪些是同族事实、哪些是默认值）。

样板抄四遍的代价不是行数 ✓ —— 是**改一处要记得改四处** ✗。少改一处**不报错** ✗：
``_configured_text_dim`` 会静默回落默认值、``in_channels`` 取不到就造一维占位潜变量 ✓✗
（本仓 2026-09-20 / 2026-10-31 各踩过一次）。

⇒ 现在：一个族 = 本表里**一行声明**，装配流程只有这一份实现。
新增一族要动的地方从「``torch_backend`` 里再抄一个 ``_build_*``」降为
「``dit_family._RULES`` 加一条判据 + 本表加一行」，有测试守着这条口径
（``tests/engine_dit_build_test.py``：表项完备 + 覆盖所有可建族 + ``torch_backend``
不再按族名分叉）。

## 边界（别把不该塞的塞进来）

* **判族**在 :mod:`dit_family` ✓（纯函数、只读头部键名、毫秒级、不 import torch）。
  本模块**不判族**，只接受判好的 ``key``。
* **读盘口径**在 :mod:`loader` / :mod:`quant` ✓（int8 计划、convrot 分组）。
  本模块只做「拿什么去建、建完搬哪些字段」。
* ``int8_scale_plan`` 由调用方注入 ✓ —— 它同时被 H3 那条路用、留在 ``torch_backend``；
  这里绝不复制一份（复制出来的那份迟早与原件不同步 ✗）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: ⭐ **静态 import，不许改成 ``importlib.import_module("dit_lumina")``** ✗✗（2026-11-01 修 ✓）：
#: 按名字字符串动态取模块 ⇒ ① 拼错只在**装到那一族时**才炸 ✗（`module="dit_lumin"` 能过编译、
#: 过自检、过全量测试 ✓）；② 结构守卫看不到这条线 ✓✗（它按 `import` 语句扫调用者 ⇒
#: 报「`dit_lumina` 生产代码 0 处 import」✓，于是真接线被判成死模块 ✓）。
#: 静态 import 之后：模块名拼错 ⇒ **ImportError** ✓；守卫能看见 ✓。
#:
#: ⚠️⚠️ 但**不许放在模块顶层** ✗✗（2026-11-01 二次修 ✓）：六个骨架模块顶层必然 `import torch` ✓，
#: 而 `capabilities()`／`readiness()`／`/engine/plan` 要读的 `BUILDERS` 是**纯数据表** ✓ ——
#: 顶层 import 会让「没装 torch 的环境算不出能力」✓✗（症状＝把 ``torch`` 打成 ``None`` 后
#: `import dit_build` 直接 ImportError ✓，读起来像"引擎本来就离不开 torch" ✓）。
#: 现在骨架走 :func:`skeletons` 惰性加载（**函数体内**的 `from . import` ✓ 静态语句 ✓ 守卫扫得到 ✓）。
from . import latent_formats
from .safetensors import SafetensorsInfo

__all__ = [
    "BUILDERS", "BuildError", "FamilyBuilder", "ForwardContract", "GENERIC_LATENT_FORMAT",
    "KREA2_PREFIXES", "SKELETON_MODULES", "SKELETONS", "build", "buildable_keys", "contract_of",
    "image_only_keys", "krea2_specs", "krea2_strip", "latent_format_of", "latent_format_problem",
    "materialize_meta", "readout", "sigma_form_of", "skeletons",
]

#: 骨架模块的**名字表** ⇒ 不 import 任何骨架、不拉 torch ✓（表里写错名字由 :func:`skeletons` 报 ✓）。
SKELETON_MODULES: tuple[str, ...] = (
    "dit_flux", "dit_generic", "dit_lumina", "dit_music3", "dit_qwen21", "dit_qwen_image",
    "dit_seedvr2", "dit_wan22")

#: 已加载的骨架模块（首次**真构建**时才填 ✓）—— 键集恒等于 :data:`SKELETON_MODULES` ✓。
SKELETONS: dict[str, Any] = {}

_SKELETONS_LOADED = False


def skeletons() -> dict[str, Any]:
    """**首次真构建时**才 import 全部骨架模块 ⇒ 返回 :data:`SKELETONS`（名字 → 模块）✓。

    ⚠️ 一次 import **全部** ✗✗（不是按名字挑一个 ✓）：按名字取会让「拼错的模块名」只在
    装到**那一族**时才炸 ✗，与旧 ``importlib`` 那版同一个坑 ✓；一次全 import ⇒ 任何一次构建
    都会立刻暴露拼错 ✓。
    """
    global _SKELETONS_LOADED
    if not _SKELETONS_LOADED:
        #: 静态 import（不是 ``importlib``）⇒ 拼错当场 ImportError ✓、守卫的 AST 扫得到 ✓
        from . import (  # noqa: PLC0415
            dit_flux, dit_generic, dit_lumina, dit_music3, dit_qwen21, dit_qwen_image,
            dit_seedvr2, dit_wan22)
        SKELETONS.update({mod.__name__.rpartition(".")[2]: mod
                          for mod in (dit_flux, dit_generic, dit_lumina, dit_music3,
                                      dit_qwen21, dit_qwen_image, dit_seedvr2, dit_wan22)})
        _SKELETONS_LOADED = True
    return SKELETONS

#: 同一份 Krea2 权重会被不同工具包成不同前缀（实测盘上三份分别写成
#: ``model.diffusion_model.*`` / ``diffusion_model.*`` / ``model.*``）
KREA2_PREFIXES: tuple[str, ...] = ("model.diffusion_model.", "diffusion_model.", "model.")


class BuildError(RuntimeError):
    """族认得、但权重推不出可装的结构 ⇒ **不建**（绝不拿出厂形状硬装）。"""

    def __init__(self, message: str, *, reason: str = "pending") -> None:
        super().__init__(message)
        self.reason = reason


def materialize_meta(module: Any, *, torch_mod: Any, device: Any = "cpu") -> int:
    """把还停在 **meta** 上的参数/缓冲区换成同形状同 dtype 的**未初始化**实存 ⇒ 返回换了几个。

    ## 为什么不能用 ``module.to_empty()`` ✗✗

    32B 的 H3 文本塔在 meta 上建壳之后，``to_empty`` 会把**整份 62 GiB bf16** 一次性分配出来
    —— 而那 350 个 ``nn.Linear`` 紧接着就被 ``h3_form.upgrade_linear_to_int8`` 换成 int8 常驻层
    （22.7 GiB）。**换掉不等于还给系统** ✗：实测进程 WS 停在 74 GiB、系统物理内存只剩 6 GiB
    ⇒ 后面的装载全在页错误里爬。

    ⇒ 这里只碰 ``device.type == "meta"`` 的那些：换过壳的 int8 层**已经不在 meta 上**，
    一根 byte 都不会被重复分配。调用点必须排在 ``upgrade_linear_to_int8`` **之后**。

    ⚠️ 分配出来是**未初始化**的（``torch.empty``）—— 这是**故意**的：值一律由
    ``weights.load_module_weights`` 逐张量写入，而它报 ``complete=False`` 时装载方当场拒
    （少写一个键 ⇒ ``missing`` 非空），所以"没填的壳"不可能被当成模型跑。
    """
    moved = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is None or param.device.type != "meta":
                continue
            sub._parameters[name] = torch_mod.nn.Parameter(
                torch_mod.empty(param.shape, dtype=param.dtype, device=device),
                requires_grad=bool(param.requires_grad))
            moved += 1
        for name, buffer in list(sub._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            sub._buffers[name] = torch_mod.empty(buffer.shape, dtype=buffer.dtype, device=device)
            moved += 1
    return moved


#: ``explicit`` 里同一件事的不同叫法（后端协议 vs 权重习惯）
_ALIASES: dict[str, tuple[str, ...]] = {
    "heads": ("heads", "num_heads"),
    "patch": ("patch", "patch_size"),
}

#: 没给就必须有个值的档（**不是**从权重读出来的 ⇒ 调用方要能看见这件事，见 configSources）
_DEFAULTS: dict[str, Any] = {"patch": 2}


def _arg(name: str, explicit: Mapping[str, Any]) -> Any:
    """从 ``explicit`` 取一个构建参数（按别名找 + 按类型归一）；没有 ⇒ ``None``。"""
    for alias in _ALIASES.get(name, (name,)):
        value = explicit.get(alias)
        if not value:                      # None / "" / 0 都当"没给" —— 0 头数没有意义
            continue
        if name in ("heads", "patch"):
            return int(value)
        if name == "axes_dim":
            return tuple(int(x) for x in value)
        return str(value) if name == "ref_method" else value
    return _DEFAULTS.get(name)


def _default_dtype() -> Any:
    """没有卡 ⇒ **fp32**（CPU 上 bf16 缺算子且更慢）；有卡 ⇒ bf16。

    ⚠️ 真装载时**不该**走这里：``dtype`` 必须由调用方按 ``self._compute_dtype()`` 给 ✓✗
    —— 有卡但 ``device="cpu"`` 时按 ``cuda.is_available()`` 自己判会建成 bf16。
    这里是给「自检 / 只推形状」那条路的兜底。
    """
    import torch  # noqa: PLC0415

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


#: 认不出的族（老路 :func:`dit.build_dit`）⇒ **口径未知** ✓ —— 用这个哨兵显式表态 ✓，
#: 不许拿某个具体族名顶上去 ✗✗（那等于替它认领一套反缩放 ✓；解出来图偏色还不报错 ✓✗）。
GENERIC_LATENT_FORMAT = "generic"


@dataclass(frozen=True)
class ForwardContract:
    """一族的**前向契约** ✓ —— 采样器只该认这三个事实，不该认族名 ✗✗。

    ## 为什么必须写成声明，而不是散在三个方法里

    2026-10-31 接 Krea2 之前，同一件事（"这族只出图、潜变量 4 维、σ 要 ``(b,)`` 张量 ✓"）
    以**族名元组**的形式被三处各读一遍：``latent_shape`` ✓、``_latent_mode`` ✓、``denoise`` ✓。
    手抄两份还能靠人记 ✓；第三份就是「加了新族只改了两处」✓✗ —— 而漏掉的那处**不会报错** ✗：
    形状还是对的 ✓、喂进去的数全是噪声 ✓✗（本仓最恨的那类 ✓）。

    ⇒ 现在这三处**只读本对象** ✓：加族声明一次，三处自动跟上 ✓。

    * ``latent_dims``：``5`` = ``(b, c, t, h, w)`` 视频口径 ✓；``4`` = ``(b, c, h, w)``
      只出图的 4 维口径 ✓（沿用 5 维那条会让 ``patchify`` 多一维 ⇒ 报「expected 4D」✗，
      指不到"这是图模型"✓✗）；``3`` = ``(b, c, t)`` **音频**口径 ✓（Music3 ✓）——
      没有空间维 ✓、``t`` 是**潜帧**（86.13 Hz ✓）不是像素 ✓✗，
      所以 ``vae_scale`` 对它没有意义 ✓（那一格恒 1 ✓，别拿它当"空间下采样 1×"去算 H/W ✗）；
    * ``sigma_form``：``"scalar"`` ⇒ 前向收标量 σ ✓；``"batch"`` ⇒ 收 ``(b,)`` 的
      **逐 batch 张量** ✓（见 :func:`torch_backend._timesteps_of` ✓）—— 喂错形态不会报错 ✗，
      只会把 batch 里各条当成同一个 σ ✓✗；
    * ``latent_format``：潜空间口径名 ✓ —— **必须是** :mod:`latent_formats` 里的规范名 ✓
      （``wan21`` / ``flux`` / ``qwen_image`` / ``qwen_image21`` ✓），本对象构造时当场
      :func:`latent_formats.normalize_latent_format` 归一化 ✓✗。
      ⚠️ 2026-11-01 之前这一格是**摆设** ✗：写着 ``flux8`` / ``qwen21`` 这种**表里没有的名字** ✓✗
      （谁也校验不了 ✓），而且**没有任何调用方读它** ✓✗ —— 于是 ``qwen_image`` 那一格写成
      ``wan21`` 也没人发现 ✓（真装配线传的是 ``qwen_image`` ✓，两边对不上 ✓）。现在它被
      :func:`latent_format_problem` 拿去和**已挂 VAE 的反缩放口径**对账 ✓（解码前判 ✓）。
    """

    latent_dims: int
    sigma_form: str
    latent_format: str

    def __post_init__(self) -> None:
        #: ⚠️ 拼错的枚举当场拒 ✗✗：写成 ``"Batch"`` / ``"tensor"`` 不会报错，只会让
        #: `denoise` 走 ``else`` 分支 ⇒ 拿标量喂张量口 ✓✗（症状是「σ 没生效」✓）。
        #: ``3`` 是 2026-11-02 接 Music3 时加的 ✓（音频 ⇒ ``(b, c, t)`` ✓）。
        #: ⚠️ 这一格是**枚举不是自由值** ✗：``6`` / ``2`` 这种"看起来也行"的值会被
        #: `torch_backend.latent_shape` 静默当成 5 维那条算 ✓✗ ⇒ 这里挡死 ✓。
        if self.latent_dims not in (3, 4, 5):
            raise ValueError(f"latent_dims 只能是 3（音频）/ 4（出图）/ 5（出片），"
                             f"拿到 {self.latent_dims!r}")
        if self.sigma_form not in ("scalar", "batch"):
            raise ValueError(f"sigma_form 只能是 scalar / batch，拿到 {self.sigma_form!r}")
        if not str(self.latent_format or "").strip():
            raise ValueError("latent_format 不许为空（报告与文档都靠它说清 VAE 口径）")
        #: ⚠️ **认不出就当场报** ✓✗（:class:`latent_formats.LatentFormatError` 是 ``ValueError`` ✓）：
        #: 这一格以前只判非空 ⇒ ``flux8`` / ``qwen21`` 一路过 ✓，报告里写着不存在的口径 ✓，
        #: 拿它去 :func:`latent_formats.get_latent_format` 才炸 ✓（离写错的地方几千行 ✓）。
        #: ``"generic"`` 是**唯一**的例外 ✓：它明确表示"老路通用 DiT，口径未知" ✓（不参与对账 ✓）。
        name = str(self.latent_format).strip()
        if name != GENERIC_LATENT_FORMAT:
            name = latent_formats.normalize_latent_format(name)   # 大小写/别名 → 规范名 ✓
        object.__setattr__(self, "latent_format", name)

    def latent_spec(self) -> "latent_formats.LatentFormatSpec | None":
        """潜空间口径的**规格** ✓；``generic``（老路，未知）⇒ ``None`` ✓（不假装知道 ✗）。"""
        if self.latent_format == GENERIC_LATENT_FORMAT:
            return None
        return latent_formats.LATENT_FORMATS[self.latent_format]


#: 认不出的族（老路 :func:`dit.build_dit`）用的口径 ✓ —— 通用 DiT 收标量 σ、5 维潜变量 ✓。
_GENERIC_CONTRACT = ForwardContract(latent_dims=5, sigma_form="scalar",
                                    latent_format=GENERIC_LATENT_FORMAT)


@dataclass(frozen=True)
class FamilyBuilder:
    """一族的装配声明 —— **只写差异**，公共流程在 :func:`build` 里。"""

    key: str
    #: 报错文案里的族名（**逐字**保留原文案，别为了好看改动 ⇒ 测试与运维都在读它）
    label: str
    #: ⭐ **前向契约**（潜变量几维 / σ 什么形态 / 哪个潜空间）—— 必填、无默认值 ✓✗：
    #: 默认值会让新族「一行都不用想」就把 5 维视频口径搬过去 ✓✗（见 :class:`ForwardContract`）
    contract: "ForwardContract"
    #: 骨架模块名（同包内，延迟 import ⇒ 不拖 torch）
    module: str
    #: 推结构的入口（模块内函数名）
    shape_fn: str = "infer_shape"
    #: 建网的入口（模块内函数名；``meta=True`` 时在 ``torch.device("meta")`` 里调）
    build_fn: str = "build_from_shapes"
    #: ``shape_fn`` 要透传的 ``explicit`` 键（不写就不传）
    shape_args: tuple[str, ...] = ()
    #: ``build_fn`` 的位置参数按名取值：``"shape"``（形态对象）或 ``"shapes"``（形状字典）
    #: —— 默认 ``"shape"``：多数骨架 ``build(shape)``，而 ``build_from_shapes(shapes, *)``
    #: 要的仍是**字典**（它内部自己再 infer 一遍）✓✗
    build_args: tuple[str, ...] = ("shape",)
    #: ``build_fn`` 的**关键字**参数：``"patch"`` / ``"dtype"``（骨架把它声明成 kw-only ⇒
    #: 必须按名传 ✗ 别传位置：``build_from_shapes() takes 1 positional argument`` ✓✗）
    build_kwargs: tuple[str, ...] = ()
    #: 是否走「meta 建壳 ⇒ 降 dtype ⇒ 物化」（大族必开：fp32 建一遍 = 两倍内存）
    meta: bool = False
    #: 后端协议字段 ← 权重里的字段（**改名的单一登记处**）
    shortcuts: tuple[tuple[str, str], ...] = ()
    #: 常量字段（同族事实，权重里没有）
    facts: tuple[tuple[str, Any], ...] = ()
    #: ⭐ **派生格**：从 ``shape`` 上取值的字段名（dataclass 字段之外的**属性**也算 ✓）——
    #: ``config = dict(vars(shape))`` 只收 dataclass 字段 ✓，而 ``@property`` 不在 ``vars()`` 里 ✓✗
    #: ⇒ 上层（`runtime` / `torch_backend`）读 ``config.get("<派生格>")`` 会**永远读到 None** ✓✗
    #: （2026-11-01 真跑踩到：Edit 2511 的 ``ref_timestep_zero`` 是 property ✓ ⇒ 装配线判 t2i ✓、
    #: ``encode_text`` 整段跳过参考图通路 ✓、直到 `denoise` 才报「ref_latents 不在」✓ ——
    #: 报错方向还指向上游 ✓✗）。加一格声明即固化 ✓，不许上层自己去猜 ``ref_method`` ✗。
    derived: tuple[str, ...] = ()
    #: ``configSources`` 的默认主体（有 ``source_fn`` 时被它整体替换）
    sources: tuple[tuple[str, str], ...] = (("结构", "权重形状反推 ✓"),)
    #: ``configSources`` 的动态主体（需要读 explicit / shape 才知道文案）
    source_fn: str = ""
    #: 族特有钩子（前缀剥离 / F32 岛 / int8 换壳 / 派生字段）
    adapter: str = ""


#: 装配表 —— **一个可建族一行**。顺序无优先级语义 ✓（判族在 ``dit_family`` 里做）。
BUILDERS: dict[str, FamilyBuilder] = {
    #: ⚠️ ``latent_format="flux"`` ✓ —— Z-Image 用的是 **Flux 的 ae 现成 VAE** ✓（:mod:`vae_flux`
    #: 的仿射 ``x/0.3611+0.1159`` ✓，与 :class:`latent_formats.Flux` 的 shift/scale **同一族** ✓）。
    #: 以前这格写 ``"flux8"`` ✓✗：表里没这个名字 ✓、也没人读它 ✓ ⇒ 写错等于没写 ✓。
    "zimage": FamilyBuilder(
        key="zimage", label="Z-Image", module="dit_lumina",
        contract=ForwardContract(latent_dims=4, sigma_form="batch", latent_format="flux"),
        shape_args=("patch",), build_args=("shapes",), build_kwargs=("patch", "dtype"),
        shortcuts=(("text_dim", "cap_dim"), ("in_channels", "channels")),
        facts=(("vae_scale", 8),),
        sources=(),                        # 由 _zimage_sources 整体给（含 patch 的 explicit/default）
        source_fn="zimage",
    ),
    "qwen_image21": FamilyBuilder(
        key="qwen_image21", label="Qwen-Image 2.1", module="dit_qwen21",
        contract=ForwardContract(latent_dims=4, sigma_form="batch",
                                 latent_format="qwen_image21"),
        build_fn="build", build_args=("shape",), meta=True,
        shape_args=("heads", "axes_dim"),
        shortcuts=(("text_dim", "context_dim"), ("in_channels", "channels")),
        facts=(("vae_scale", 16),),
        sources=(),
        source_fn="qwen21",
    ),
    #: ⚠️ ``latent_format="qwen_image"`` ✓✗（**不是** ``wan21`` ✓）：2512 的 VAE 是
    #: ``qwen_image_vae`` ⇒ 反缩放取**恒等** ✓（``vae_wan.latents_stats("qwen_image")`` ✓）。
    #: 这格以前抄成 ``wan21`` ✓✗ —— 而 ``wan21`` 现在是**真 Wan 均值/方差** ✓ ⇒ 一旦有人拿契约
    #: 驱动反缩放，2512 的图会整体偏色且**不报错** ✓✗（正是 :mod:`vae_wan` 抬头警告的那个坑 ✓）。
    "qwen_image": FamilyBuilder(
        key="qwen_image", label="Qwen-Image", module="dit_qwen_image",
        contract=ForwardContract(latent_dims=4, sigma_form="batch",
                                 latent_format="qwen_image"),
        build_fn="build", build_args=("shape",), meta=True,
        shape_args=("heads", "axes_dim", "ref_method"),
        shortcuts=(("text_dim", "context_dim"), ("in_channels", "channels")),
        facts=(("vae_scale", 8),),
        #: ⭐ Edit 2511 与 2512 **配置逐格相同** ✓✗（只差这一格 ✓）⇒ 它是「要不要走参考图
        #: 通路」的**唯一开关** ✓：`runtime` 按它决定 encode 段要不要多挂 vision+vae ✓、
        #: `torch_backend` 按它决定 `denoise` 每步带不带 `ref_latents` ✓。两处读同一格 ✓。
        derived=("ref_timestep_zero",),
        sources=(),
        source_fn="qwen_image",
        #: ⭐ **int8 常驻**（2026-11-01 ✓）：盘上 Edit 2511 是 840 层 int8 ✓ ⇒ 走 `Int8Linear`
        #: 常驻后驻留 19.09 GiB（能整份上卡 ✓）；同一族的 2512 是 bf16 文件 ⇒ `int8_scale_plan`
        #: 返回空表 ⇒ 自动退回原来的 bf16 路 ✓（**一个开关管两种文件** ✓，不会把 bf16 那份搞坏 ✓）。
        adapter="int8",
    ),
    "krea2": FamilyBuilder(
        key="krea2", label="Krea2", module="dit_generic",
        contract=ForwardContract(latent_dims=4, sigma_form="batch", latent_format="wan21"),
        build_fn="Krea2DiT", build_args=("shape",), meta=True,
        shape_args=("patch",),
        shortcuts=(("in_channels", "channels"),),
        facts=(("vae_scale", 8),),
        adapter="krea2",
    ),
    #: ⭐ **Flux.1**（双流 19 + 单流 38 ✓ 见 :mod:`dit_flux` ✓）。
    #: 潜变量走 ``latent_formats.Flux``：16ch / 8× / **有**仿射反缩放 ✓（与 Krea2 的 Wan21 不同族 ✓）。
    #: ⚠️ ``sigma_form="batch"`` ✓ 且 σ **不翻转**（``1000·σ`` 由骨架内部的 ``timestep_embedding`` 乘 ✓）——
    #: Z-Image 那族的 ``1000·(1−σ)`` 是那一族的约定 ✗，两族不可互抄 ✓。
    #: ⚠️ ``text_dim`` 走 4096（T5-XXL）✓、另有 768 的 pooled 向量（``vec_dim`` ✓）——后者**不在**本表里，
    #: 因为它不是「文本宽」而是「另一根输管」✓（见 ``FluxDiT.forward`` 的 ``vector=`` ✓）。
    "flux": FamilyBuilder(
        key="flux", label="Flux.1", module="dit_flux",
        contract=ForwardContract(latent_dims=4, sigma_form="batch", latent_format="flux"),
        build_fn="build", build_args=("shape",), meta=True,
        shape_args=("heads", "axes_dim", "guidance_embed"),
        shortcuts=(("text_dim", "context_dim"), ("in_channels", "channels")),
        facts=(("vae_scale", 8),),
        sources=(),
        source_fn="flux",
    ),
    #: ⭐ **Wan2.2 I2V 14B**（单专家；见 :mod:`dit_wan22` ✓）—— 唯一一个**五维 + 视频**的真建族 ✓。
    #: ⚠️ ``latent_format="wan21"`` ✓：盘上同目录的 ``wan_2.1_vae`` 就是这一族 ✓（16ch / 8× /
    #: **meanstd 16 组**）⇒ 解码前对账查的就是这格 ✓。
    #: ⚠️ ``in_channels`` 取 ``out_dim``（**16**）而**不是** ``in_dim``（36）✓✗：前者是潜变量/
    #: 噪声的通道数 ✓（``latent_shape`` 造的就是它 ✓）；后者是**主干输入宽** = 噪声 16 + 条件图 16 +
    #: 掩码 4 ✓ —— 拼接发生在上层（参考实现的 ``model_base`` ✓），本仓**尚未接** ✓（见
    #: ``configSources["i2vConcat"]`` ✓）。两者混用 ⇒ 要么造出 36ch 假噪声 ✓✗、要么主干当场拒 ✓。
    #: ⚠️ ``sigma_form="batch"`` ✓ 但 ``×1000`` 由**骨架自己**做 ✓✗（参考实现把它放在采样器的
    #: ``ModelSamplingDiscreteFlow.timestep`` 里 ✓，与 Flux 的 ``time_factor`` 同值不同位置 ✓）。
    "wan": FamilyBuilder(
        key="wan", label="Wan2.2 I2V（单专家）", module="dit_wan22",
        contract=ForwardContract(latent_dims=5, sigma_form="batch", latent_format="wan21"),
        build_fn="build", build_args=("shape",), meta=True,
        shape_args=("heads",),
        shortcuts=(("in_channels", "out_dim"),),
        facts=(("vae_scale", 8),),
        sources=(),
        source_fn="wan",
    ),
    #: ⭐ **SEEDVR2 / NaDiT 3B**（视频修复主干，见 :mod:`dit_seedvr2` ✓）—— 2026-11-01 与参考
    #: ``comfy/ldm/seedvr/model.py`` 逐位对拍：**真实 3B fp16 权重**下 ``maxdiff == 0`` ✓、
    #: 89 个键逐名逐形状对得上 ✓（反推的形状全部来自权重，没有一项是出厂常量 ✓）。
    #: ⚠️ 与出图族四点不同 ✗✗：
    #:  ① **双输入** —— 除 16ch 潜变量还要 17ch ``conditioning``（现状视频 16 ＋ loss mask 1）；
    #:     ``in_channels``（33）是两者**拼接后**的宽 ✓ ⇒ 拼接在骨架里做 ✓
    #:     （``SeedVR2DiT.forward(..., conditioning=)`` 是**必填** ✓，缺了当场拒并写明 17 通道 ✓）。
    #:     后果：**通用出图管线不能直接调它** ✓✗（它没有"凭空生成"这一路）。
    #:  ② 潜变量 **5 维** ✓ 且 ``sigma_form="batch"`` ✓，σ∈[0,1]，``×1000`` 由骨架自己做 ✓
    #:     （:data:`dit_seedvr2.SEEDVR2_TIMESTEP_SCALE` ✓，不是采样器的活 ✓✗）。
    #:  ③ 主干内部 ``patch=(1,2,2)`` 与 ``patch_in/out`` 的通道拆分**从权重反推** ✓
    #:     （:func:`dit_seedvr2.infer_shape` ✓）⇒ 本表不传 ``patch`` ✓（传了也无处可用 ✓）。
    #:  ④ ``text_dim`` 走 ``txt_in_dim``（5120）✓ —— 骨架自吃 context（``txt_in`` ＋ 逐块双塔 ✓），
    #:     文本塔在族外 ✓。
    #: ⚠️ ``build_kwargs`` 必须显式收窄成 ``("dtype",)`` ✓✗：默认值里那个 ``patch`` 会喂给
    #: ``dit_seedvr2.build``（它不收），报的是 TypeError 而不是"这族不需要 patch" ✓。
    "seedvr2": FamilyBuilder(
        key="seedvr2", label="SEEDVR2（NaDiT 3B）", module="dit_seedvr2",
        contract=ForwardContract(latent_dims=5, sigma_form="batch", latent_format="seedvr2"),
        build_fn="build", build_args=("shape",), meta=True,
        build_kwargs=("dtype",),
        shape_args=(),
        shortcuts=(("text_dim", "txt_in_dim"),),
        #: ⚠️ **必须声明派生格** ✗✗（2026-11-01 真跑踩到 ✓）：``SeedVR2Shape.latent_channels``
        #: 是 ``@property``（``= out_channels`` ✓ 16 ✓）⇒ ``dict(vars(shape))`` **收不到** ✓✗，
        #: 而 ``in_channels`` 是 **33**（噪声 16 ＋ 条件 17 ✓）⇒ 上层照 ``in_channels`` 造潜变量
        #: 会造出 33 通道 ✓；更要命的是 `torch_backend._check_attached_vaes` 拿 33 去核
        #: 「VAE 自述 16」⇒ **正确的 VAE 被判成挂错族** ✓✗（报错还写着"大概率是挂错族的 VAE" ✓
        #: —— 方向指反 ✓✗）。声明进 ``derived`` 才进 ``config`` ✓。
        derived=("latent_channels",),
        facts=(("vae_scale", 8),),
        sources=(("结构", "参照 comfy/ldm/seedvr/model.py 逐位重写 ✓"),
                 ("数值", "真实 3B fp16 权重前向 maxdiff == 0 ✓")),
    ),
    #: ⭐ **MiniMax Music3 的音频 DiT**（见 :mod:`dit_music3` ✓）—— 2026-11-02 与参考
    #: ``comfy/ldm/minimax_music/dit.py`` 逐位对拍：**真实 2.46B fp16 权重**下
    #: ``maxdiff = 4.8e-06``（相对 1.2e-06 ✓ 只有 sdpa/线性代数求和顺序的差 ✓）、
    #: 121→全部键逐名逐形状对上 ✓。
    #: ⚠️ 与出图/出片族**四处不同** ✗✗：
    #:  ① 潜变量 **3 维** ``(b, 128, T)`` ✓（音频 ✓ 没有空间维 ✓）⇒ ``latent_dims=3`` ✓；
    #:  ② ``vae_scale`` 恒 **1** ✓ —— 音频没有"像素÷8"这回事 ✓✗，填 8 会让上游按边长算潜帧 ✓；
    #:     真正的缩放方向是**时间**：潜帧 ÷ 512 样本 ÷ 44100 Hz ✓（见 :data:`_music3_sources` ✓）；
    #:  ③ ``in_channels`` 取 ``latent_channels``（**128** ✓）而**不是** ``in_dim``（2304 ✓✗ ——
    #:     那是入口宽度 ＝ 潜 128 ＋ 占位 128 ＋ 条件 2048 ✓）；混用会造出 2304 通道假噪声 ✓✗；
    #:  ④ ``text_dim`` 取 ``context_dim``（**32768** ＝ 8 层 × 4096 ✓）—— 条件在**主干内部**
    #:     对齐到 2048 ✓（``latent_conditioners`` ＋ 层 softmax 加权 ✓），不走 ``txt_in`` ✓✗。
    #: ⚠️ ``build_kwargs`` 必须收窄成 ``("dtype",)`` ✓✗：``dit_music3.build`` 的 ``patch`` 是
    #: **占位参数**（音乐族没有 patchify ✓），传默认 ``patch=2`` 不会报错 ✓、只是白传 ✓ —— 收窄
    #: 是为了让"这族不需要 patch"在表上可读 ✓。
    "music3": FamilyBuilder(
        key="music3", label="MiniMax Music3（音频 DiT）", module="dit_music3",
        contract=ForwardContract(latent_dims=3, sigma_form="batch",
                                 latent_format="minimax_music3"),
        build_fn="build", build_args=("shape",), meta=True,
        build_kwargs=("dtype",),
        shape_args=(),
        shortcuts=(("in_channels", "latent_channels"), ("text_dim", "context_dim")),
        derived=("context_dim",),
        facts=(("vae_scale", 1),
               ("latentSampleRate", 44100),
               ("latentHop", 512)),
        sources=(),
        source_fn="music3",
    ),
}


def contract_of(key: str | None) -> ForwardContract:
    """族的**前向契约**（单一口径）✓ —— 认不出的族（含 ``None``）回退到通用 DiT 口径 ✓。

    ⚠️ 调用方**不要**写 ``contract_of(family) if family else …`` ✗：老路的通用 DiT 就是
    "认不出的族" ✓，两个分支本来就是同一件事 ✓✓（写成两种写法 ⇒ 又长出手抄第二份 ✗）。
    """
    spec = BUILDERS.get(key or "")
    return spec.contract if spec is not None else _GENERIC_CONTRACT


def image_only_keys() -> tuple[str, ...]:
    """只出图的族（潜变量 4 维）—— :data:`torch_backend.IMAGE_ONLY_FAMILIES` 的唯一来源 ✓。

    以前这个元组是在 ``torch_backend`` 里手写的 ✓✗：加族时改表、忘改元组 ⇒
    ``latent_shape`` 照 5 维算 ⇒ 报「expected 4D input」✓（指不到根因 ✓）。
    """
    return tuple(k for k, spec in BUILDERS.items() if spec.contract.latent_dims == 4)


def sigma_form_of(key: str | None) -> str:
    """``"scalar"`` / ``"batch"`` —— 采样器按它决定 σ 喂标量还是 ``(b,)`` 张量 ✓。"""
    return contract_of(key).sigma_form


#: 不走装配表、但口径**说得出**的族 ✓（2026-11-01 补）：它们的骨架不在 :data:`BUILDERS`
#: （H3 走 `h3_form` 双流、SDXL 走 `sdxl_backend`）⇒ :func:`contract_of` 只能回
#: :data:`GENERIC_LATENT_FORMAT` ⇒ 对账**整条跳过** ✓✗。跳过本身不算错 ✓，但"明明知道"却
#: 在同一张表里装作不知道 ✓，等于把两个族的口径检查永久关掉 ✓✗。
#: ⚠️ 这里**只补口径这一格** ✓，不补 ``latent_dims``／``sigma_form``（那两格要么走各自的
#: 前向入口、要么真未知 ✓）—— 别顺手把 H3 塞进 :data:`BUILDERS` ✗（它会去抢装配流程 ✓）。
_NON_TABLE_FORMATS: dict[str, str] = {
    "h3": "h3_video",     # 24ch / 16× / `scale` 1.0（:mod:`latent_formats` ✓）
}
#: ⚠️ SDXL **不登记** ✓：它在 :class:`~.sdxl_backend.SdxlBackend` 里是**另一个类** ✓、
#: 全仓只有一种 VAE（`sdxl_vae` 固定 0.13025 ✓）⇒ 没有"挂错族"这个自由度 ✓，
#: 登记了也永远没人查 ✓✗（死配置和判据不存在一样糟 ✓）。


def latent_format_of(key: str | None) -> str:
    """族 ⇒ 潜空间口径名 ✓（装配族读契约 ✓，非装配族读 :data:`_NON_TABLE_FORMATS` ✓）。

    ⚠️ 与 :func:`contract_of` 的差别只有一格：``contract_of("h3").latent_format`` 是
    ``generic`` ✓（因为它连 ``latent_dims`` 都不在本表里 ✓），而这一问的答案**是知道的** ✓。
    """
    name = str(key or "").strip()
    if name in _NON_TABLE_FORMATS:
        return _NON_TABLE_FORMATS[name]
    return contract_of(name).latent_format


def latent_format_problem(key: str | None, vae_format: str | None) -> str | None:
    """DiT 族的潜空间口径 ↔ **已挂 VAE 的反缩放口径** ⇒ 不一致就返回人话 ✓（一致/未知返 ``None`` ✓）。

    ## 为什么值得单独一个判据

    潜变量**只在解码那一步**才和 VAE 的反缩放对上 ✓；对错不会报错 ✗✗ —— 16ch 的 Wan21 与
    16ch 的 Qwen-Image 形状**完全相同** ✓、只差均值/方差 ✓ ⇒ 挂了别的族的 VAE ⇒ 图整体偏色/发灰 ✓、
    形状全对 ✓、一个错都不报 ✓（:mod:`vae_wan` 抬头记的就是这个坑 ✓）。

    ## 三档语义（**不猜** ✗）

    * 两边都认得出且**同族** ⇒ ``None`` ✓；
    * 两边都认得出但**不同族** ⇒ 返回问题描述 ✓（调用方当场拒 ✓）；
    * 任一边**未知**（族没用:data:`GENERIC_LATENT_FORMAT` 表态 / 老路 VAE 不报口径 / 名字不在表里）
      ⇒ ``None`` ✓ —— 但这是"**没检查**"而不是"检查通过"✓✗，别把它当绿灯 ✓。
    """
    want = latent_format_of(key)
    if want == GENERIC_LATENT_FORMAT or not vae_format:
        return None
    def _norm(name: str) -> str | None:
        try:
            return latent_formats.normalize_latent_format(name)
        except ValueError:                         # 认不出（表里没有）⇒ 当作"未知"✓，不拦 ✗
            return None
    mine, theirs = _norm(want), _norm(str(vae_format))
    if mine is None or theirs is None or mine == theirs:
        return None
    return (f"潜空间口径对不上 ✗：DiT 族 ``{key}`` 是 ``{mine}`` ✓，"
            f"已挂 VAE 的反缩放是 ``{theirs}`` ✓ ⇒ 解码出来会**整体偏色/发灰**而且**不报错** ✓✗"
            f"　⇒ 换同族的 VAE 重挂 ✓（别改这里的判据 ✓）")


def _strip_krea2_specs(specs: Any, prefix: str | None) -> Any:
    """把 convrot 分组表的键也剥掉同一层前缀（形态照抄入参：``dict`` 就回 ``dict``）。

    ⚠️ 不剥的后果与 ``int8_scale_plan`` 的 ``strip_prefix`` 一模一样：``quant.group_for``
    是**按 stem 精确查表** ⇒ 表里带前缀 ⇒ 每次查空 ⇒ int8 层照常换壳、权重也照常装上，
    只是**没做旋转**（画面糊/脏、不报错）。
    """
    if not prefix or not isinstance(specs, dict):
        return specs
    return {name[len(prefix):] if isinstance(name, str) and name.startswith(prefix) else name: value
            for name, value in specs.items()}


def _logical_shapes_of(raw: dict[str, tuple[int, ...]], info: SafetensorsInfo,
                       path: Any) -> dict[str, tuple[int, ...]]:
    """把 **4bit 打包**的权重形状还原成**逻辑**列数（×2）后再交给形状推断 ✓✗。

    ## 为什么非做不可（2026-11-07 实测 ✓）

    int4 文件把两个 4bit 塞进一个字节 ⇒ ``tmlp.0.weight`` 是 ``[6144, 128]`` ✓，
    而逻辑上是 ``[6144, 256]`` ✓✗。:mod:`dit_generic` 的拓扑**全靠形状反推**
    （``tdim=_shape_of(shapes, "tmlp.0.weight")[1]`` ✓ / ``txtdim=txtmlp.1.weight[1]`` ✓）
    ⇒ 不还原就按**一半**的维建骨架 ⇒ 物化时逐键 ``shapeMismatch`` ⇒
    报「权重推不出可装的结构」✓✗ —— **读起来像拿错了文件，其实是列数少算一半** ✓✗。

    :mod:`dit_family` 那条新路径自己会还原（``_logical_dims`` ✓），它读的是 ``info``
    不是这里的返回值 ⇒ 两边**不会**各还原一次 ✓（各还原一次＝×4 ✓✗）。

    ``path`` 为 ``None``（自检/无盘）或读不到说明 ⇒ 原样返回 ✓（不猜 ✗）。
    """
    if path is None:
        return raw
    from . import loader, quant  # noqa: PLC0415

    try:
        specs = loader.quant_specs_of(path, info)
    except Exception:  # noqa: BLE001 —— 补证据失败不该把装配搞崩 ✓
        return raw
    packed = quant.packed_weight_stems_of(specs)
    if not packed:
        return raw
    out = dict(raw)
    for stem in packed:
        #: ⚠️ ``specs`` 的键是**层名**（``name[:-len(".comfy_quant")]`` ✓）——
        #: 张量名要补 ``.weight`` ✓；直接拿层名查 ``raw`` 会一条都命中不了 ✓✗。
        key = f"{stem}.weight"
        if key in out:
            out[key] = quant.logical_weight_shape(out[key], packed=True)
    return out


def _shell_int8(module: Any, info: SafetensorsInfo, prefix: str | None, path: Any,
                int8_plan: Callable[..., dict[str, tuple[int, ...]]] | None,
                int4_plan_fn: Callable[..., dict[str, tuple[int, ...]]] | None = None,
                ) -> tuple[dict[str, tuple[int, ...]], int, dict[str, tuple[int, ...]]]:
    """壳建好、**物化之前**：F32 岛 ＋ int8 常驻换壳 ⇒ ``(plan, convrot 层数)`` ✓。

    ## 为什么抽成模块级（2026-11-01 ✓）

    这两步**与族无关** ✓：判据全在**文件侧**（键的 dtype 是 ``F32`` ✓ / 键的 dtype 是 ``I8``
    且有配套 scale ✓），一点不依赖 Krea2 的形状或前缀 ✓。原来它埋在 `_Krea2Adapter.shell`
    里 ⇒ 第二条要接 int8 常驻的族（Qwen-Image ✓，见 `_Int8Adapter` ✓）只能整段复制 ✓✗ ——
    复制出去的副本一改就分叉，而这里的分叉后果是「静默按 bf16 物化」✓✗（画面能出、显存翻倍 ✓）。

    ## 两段的坑（原样保留，别搬回去）

    ① **F32 岛** —— 文件里写明 ``F32`` 的键必须留在 fp32：``*.weight_scale``（int8 层的逐行
       尺度）与 norm/``qknorm`` 的 ``.scale`` ✓。前者**不进**模块树（此刻还是普通 ``nn.Linear``，
       换壳时由 ``Int8Linear`` 自己登记为 float32 ✓）；后者是**真张量** ⇒ 这里逐个按文件 dtype 摆 ✓。
       反过来把它们降成 bf16 不会报错，只会让 int8 反量化的尺度精度掉一截 ✓✗（画面细节糊 ✓）。
    ② **int8 常驻层** —— convrot 说明必须**逐层**读 ✗（全量版逐层分组不同：同一文件
       ``blocks.0.adaln_proj`` 是 64、``blocks.0.attn.qkv_proj`` 是 256 ⇒ 拿族级判会让部分层
       少一步反旋转 ✓）。``path`` 为 ``None`` ⇒ 只能按形状建、按 bf16 物化（自检那条路 ✓）。
    """
    import torch  # noqa: PLC0415

    from . import h3_form, loader, quant  # noqa: PLC0415

    holders = (dict(module.named_parameters()), dict(module.named_buffers()))
    for logical, tensor in info.tensors.items():
        if tensor.dtype != "F32":
            continue
        name = logical[len(prefix):] if prefix and logical.startswith(prefix) else logical
        for holder in holders:
            item = holder.get(name)
            if item is not None and item.is_floating_point():
                item.data = item.data.to(torch.float32)

    plan: dict[str, tuple[int, ...]] = {}
    convrot_layers = 0                       #: ⚠️ 必须无条件有值 —— config 无条件读它
    int4_plan: dict[str, tuple[int, ...]] = {}
    if path is None or int8_plan is None:
        return plan, convrot_layers, int4_plan
    raw_specs = loader.quant_specs_of(path, info)
    convrot, convrot_gap = quant.convrot_group_of(_strip_krea2_specs(raw_specs, prefix))
    if convrot_gap:
        raise BuildError(
            f"convrot 说明自相矛盾 ✗ ⇒ 不装载（不按猜的 group 算 ✗）：{convrot_gap}")
    #: ⚠️ **4bit 打包层必须排除** ✗✗（2026-11-02 ✓）：它存成 ``I8`` ✓、也带 per-row
    #: ``weight_scale`` ✓ ⇒ 与「per-row int8」**逐字同形** ✓✗ ⇒ 换壳器会把它当 int8 常驻 ✓，
    #: 反量化时按「每字节一个数」解 ⇒ **形状全对、数值全错、一声不响** ✓✗（2026-10-30 那个假绿 ✓）。
    #: 只有文件自述分得清 ✓ ⇒ 把 ``int4_stems`` 递给 ``int8_plan``（键**带前缀**，与 ``info`` 一致 ✓）。
    int4_stems = quant.int4_stems_of(raw_specs)
    plan = int8_plan(info, strip_prefix=prefix, int4_stems=int4_stems)
    #: ⭐ **4bit 常驻换壳**（2026-11-02 ✓）：同一个文件自述同时喂给两张计划表 ✓ ——
    #: int8 那边「跳过」、这边「才要」⇒ 没有一层会被两边同时点名 ✓✗。
    #: ⚠️ 顺序与 int8 无关 ✓（集合互斥 ✓），但这个计划**必须在物化之前**换上去 ✓
    #: （物化之后再换 ⇒ 新层全 0 ✓✗，出片全黑且不报错 ✓）。
    if int4_plan_fn is not None:
        int4_plan = int4_plan_fn(info, strip_prefix=prefix, int4_stems=int4_stems)
    #: ⚠️ ``convrot`` 有**三种形状**（``None`` / ``int`` 整文件同组 / ``dict`` 逐层）——
    #:   别拿 ``len()`` 直接数：整文件同组时它是 ``int`` ⇒ ``TypeError``，而且是在
    #:   **换壳之前**炸 ⇒ 看着像"int8 不支持"，其实只是回执那一行。
    convrot_layers = (len(convrot) if isinstance(convrot, dict)
                      else (len(plan) if convrot else 0)) if plan else 0
    if plan:
        h3_form.upgrade_linear_to_int8(module, plan, convrot=convrot)
    if int4_plan:
        h3_form.upgrade_linear_to_int4(module, int4_plan, convrot=convrot)
    return plan, convrot_layers, int4_plan


class _Krea2Adapter:
    """Krea2 的三件族特有的事：剥导出前缀 / 保住 F32 岛 / int8 常驻换壳。"""

    @staticmethod
    def prepare(shapes: dict[str, tuple[int, ...]]
               ) -> tuple[dict[str, tuple[int, ...]], dict[str, str] | None, str | None]:
        """剥掉**导出前缀** ⇒ ``(剥好的形状, key_map, 前缀)``；没前缀 ⇒ ``(原样, None, None)``。

        ⚠️ 这条**不能省**：盘上三份 Krea2 权重分别写成 ``model.diffusion_model.*`` /
        ``diffusion_model.*`` / ``model.*``；不剥 ⇒ :mod:`dit_generic` 认不出形状，
        并且 ``key_map=None`` 时装载会把「缺 649 个键、多 649 个键」一起报出来
        —— 读起来像"权重不全/拿错文件"，其实只是前缀。

        ⭐ 前缀要**一处判、三处用**：形状表、int8 计划表、convrot 分组表剥的是**同一层**。
        """
        for prefix in KREA2_PREFIXES:
            if not shapes or not all(name.startswith(prefix) for name in shapes):
                continue
            stripped = {name[len(prefix):]: value for name, value in shapes.items()}
            return stripped, {"^" + prefix.replace(".", "\\."): ""}, prefix
        return dict(shapes), None, None

    @staticmethod
    def shell(module: Any, info: SafetensorsInfo, prefix: str | None, path: Any,
              int8_plan: Callable[..., dict[str, tuple[int, ...]]] | None, shape: Any,
              int4_plan_fn: Callable[..., dict[str, tuple[int, ...]]] | None = None
              ) -> tuple[dict[str, Any], dict[str, str]]:
        """壳建好之后：F32 岛 → int8 换壳 ⇒ ``(config 追加字段, 追加口径说明)``。

        ⚠️ 顺序不能反：``materialize_meta`` 必须排在换壳**之后**（见其文档）。
        """
        plan, convrot_layers, int4 = _shell_int8(module, info, prefix, path, int8_plan,
                                                 int4_plan_fn)

        extra = {
            #: 派生字段：``text_dim = txtlayers × txtdim``（12 层 × 每层宽度，权重里没有单值 ✗）
            "text_dim": int(shape.txtlayers) * int(shape.txtdim),
            #: ⭐ int8 常驻层的**入册**（装完拿它对账）：少换一层 ⇒ 那层按 bf16 落盘 ⇒
            #:   显存 2 倍 ⇒ 采样第 1 步 OOM（报错指不到"哪层没接管"）。
            "int8Plan": {"layers": len(plan), "convrotLayers": convrot_layers},
            #: ⭐ 4bit 打包常驻层的**入册**（2026-11-02 ✓）：它**不算**在 ``int8Plan`` 里 ✓
            #: （那边的判据是"文件自称 int8"✓），少换一层 ⇒ 整份按 bf16 反量化 ⇒ 5.98 → 23.89 GiB ✓✗。
            "int4Plan": {"layers": len(int4)},
        }
        sources = {
            "text_dim": f"权重事实（`txtlayers` {shape.txtlayers} × `txtdim` {shape.txtdim} ✓ = 12 层栈 ✓）",
            "in_channels": "权重事实（`first` 输入宽 ÷ patch² ✓）",
            "vae_scale": "同族事实（`latent_formats.Wan21`：16ch / 8× ✓；权重里没有 ✗）",
            "sigmaToTimestep": "σ 原样 ✓（`ModelSamplingFlux.timestep = sigma` ✓，×1000 在 `timestep_embedding` 里 ✓）",
            "weightPrefix": ("已剥 " + prefix) if prefix else "无前缀 ✓",
            "int8Residency": (f"{len(plan)} 层走 Int8Linear 常驻 ✓（文件侧 12.155B 个 int8 参数 ✓）"
                              if plan else "未走常驻（没给权重路径 ⇒ 自检口径 ✓）"),
        }
        return extra, sources


def krea2_strip(shapes: dict[str, tuple[int, ...]]
               ) -> tuple[dict[str, tuple[int, ...]], dict[str, str] | None, str | None]:
    """过渡入口：剥 Krea2 导出前缀 ⇒ ``(形状, key_map, prefix)``（实现见 ``_Krea2Adapter``）。"""
    return _Krea2Adapter.prepare(shapes)


def krea2_specs(specs: Any, prefix: str | None) -> Any:
    """过渡入口：把 convrot 分组表剥掉同一层前缀。"""
    return _strip_krea2_specs(specs, prefix)


class _Int8Adapter:
    """只做两件事的 adapter：F32 岛 ＋ int8 常驻换壳 ✓（前缀 / 派生字段一概不管 ✓）。

    ## 为什么需要它（2026-11-01 ✓ —— 用户问「为什么绝大部分计算由 CPU 承担」的直接答案之一）

    Qwen-Image 那族盘上**同时有** bf16 与 int8 两种文件 ✓：

    * ``qwen_image_2512_bf16.safetensors`` ＝ **38.05 GiB** ✓ ⇒ 22.49 GiB 的卡装不下 ⇒ 只能 split；
    * ``qwen_image_edit_2511_int8_convrot.safetensors`` ＝ **19.09 GiB / 840 层 int8** ✓ ⇒
      本可以**整份上卡** ✓✗，但这条装配线走的是老路（反量化成 bf16 再常驻 ✓）⇒ 驻留仍报
      **38.05 GiB** ✓✗ ⇒ 实测 ``gpu 15.9 / cpu 22.15 GiB`` ⇒ **58% 的权重在内存里逐层爬** ✓✗。

    同一台机器上 Krea2 走 `Int8Linear` 常驻 ⇒ 12.82B 只占 12.57 GiB、整份上卡 ✓。
    ⇒ 差的**不是调度判据** ✓（判据按 38.05 算得完全正确 ✓），是**这一族没接常驻** ✓。
    接上之后 Edit 的驻留降到 19.09 GiB ⇒ `decide()` 走 `free ≥ need + 1 GiB` ⇒ 整份上卡 ✓。

    ## 为什么能直接复用 Krea2 那一套

    ``int8_scale_plan`` 的判据只有「键以 ``.weight`` 结尾 ＋ dtype 是 ``I8`` ＋ 有配套 scale」✓，
    而 Edit 的 840 个 int8 层名与骨架模块名 **840/840 逐字对得上** ✓（实测：`transformer_blocks.0.attn.to_q`
    ⇄ `_Attention.to_q` ✓，**没有** Krea2 那种 ``model.diffusion_model.`` 前缀 ✓）⇒ 不需要剥前缀。
    另 6 个 ``nn.Linear``（``img_in`` / ``txt_in`` / ``proj_out`` 一类）本来就没量化 ✓ ⇒ 保持 bf16 ✗。
    """

    @staticmethod
    def prepare(shapes: dict[str, tuple[int, ...]]
               ) -> tuple[dict[str, tuple[int, ...]], dict[str, str] | None, str | None]:
        """不剥前缀 ✓：Qwen-Image 的键**就是**模块名 ✓（实测 840/840 ✓）。"""
        return dict(shapes), None, None

    @staticmethod
    def shell(module: Any, info: SafetensorsInfo, prefix: str | None, path: Any,
              int8_plan: Callable[..., dict[str, tuple[int, ...]]] | None, shape: Any,
              int4_plan_fn: Callable[..., dict[str, tuple[int, ...]]] | None = None
              ) -> tuple[dict[str, Any], dict[str, str]]:
        """F32 岛 ＋ int8 换壳 ⇒ ``(config 追加字段, 追加口径说明)`` ✓（顺序见 :func:`_shell_int8`）。"""
        plan, convrot_layers, int4 = _shell_int8(module, info, prefix, path, int8_plan,
                                                 int4_plan_fn)
        extra = {"int8Plan": {"layers": len(plan), "convrotLayers": convrot_layers},
                 "int4Plan": {"layers": len(int4)}}
        sources = {
            "int8Residency": (f"{len(plan)} 层走 Int8Linear 常驻 ✓（文件侧 int8 ✓，不再反量化成 bf16 ✓）"
                              if plan else
                              "未走常驻 ✓（文件本身是 bf16 ⇒ 反量化这步不适用 ✓；或没给权重路径 ⇒ 自检口径 ✓）"),
        }
        return extra, sources


_ADAPTERS: dict[str, Any] = {"krea2": _Krea2Adapter(), "int8": _Int8Adapter()}


def _zimage_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                    ) -> dict[str, str]:
    """Z-Image 的口径：``patch`` 是**默认值**还是调用方给的，必须写在报告里。

    ⚠️ 判据要读 ``explicit`` **本身** ✓✗，不能读 :func:`_arg` 的结果 ✓：``_arg`` 取不到时会
    回落 :data:`_DEFAULTS` ✓ ⇒ 它的返回值**永远非 None** ✓ ⇒ 写出来的永远是 "explicit" ✓✗
    （报告把"猜的"说成"你给的" ✓，与 :func:`readout` 的存在意义正相反 ✓）。
    """
    given = next((explicit.get(alias) for alias in _ALIASES.get("patch", ("patch",))
                  if explicit.get(alias)), None)
    return {
        "patch": "explicit" if given is not None else "default",
        "结构": "权重形状反推 ✓",
        "vae_scale": "同族事实（Flux VAE，8× ✓）",
    }


def _qwen21_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                    ) -> dict[str, str]:
    heads = _arg("heads", explicit)
    return {
        "结构": "权重形状反推 ✓",
        "heads": "权重算出（to_q 输出宽 ÷ norm_q 宽 ✓）" if heads is None else "explicit",
        "text_dim": "权重事实（`txt_in.in_layer` 输入宽 ✓）",
        "vae_scale": "同族事实（`latent_formats.QwenImage21`：64ch / 16× ✓）",
    }


def _qwen_image_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                        ) -> dict[str, str]:
    from . import dit_qwen_image as qimg_mod  # noqa: PLC0415 —— 只为一句话里的哨兵常量

    heads = _arg("heads", explicit)
    return {
        "结构": "权重形状反推 ✓",
        "heads": "权重算出（`attn.norm_q` 宽 = head_dim ✓）" if heads is None else "explicit",
        "text_dim": "权重事实（`txt_norm` 宽 / `txt_in` 出宽 ✓ = Qwen2.5-VL-7B 的 3584 ✓）",
        "vae_scale": "同族事实（`latent_formats.Wan21`：16ch / 8× ✓；权重里没有 ✗）",
        "ref_method": f"{shape.ref_method} ✓（Edit 哨兵 = `{qimg_mod.EDIT_SENTINEL}` ✓）",
        #: 2026-11-01 起参考图通路**已接**（视觉塔 384² ⇒ `image_pad` 槽、VAE 1024² ⇒ `ref_latents` 一列，
        #: `runtime` 按本格挂 encode 段的 vision+vae、`torch_backend.encode_text` 按本格预编码）✓。
        "editRefWired": "True —— 参考图通路已接 ✓（视觉塔 384² + VAE 1024²；单图/多图都走 `prepare_edit_references` ✓）"
                        if shape.ref_timestep_zero else "n/a（t2i 权重 ✓）",
    }


def _flux_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                 ) -> dict[str, str]:
    heads = _arg("heads", explicit)
    return {
        "结构": "权重形状反推 ✓",
        "heads": "权重算出（`img_attn.norm.query_norm.scale` 宽 = head_dim ✓）"
                 if heads is None else "explicit",
        "guidance_embed": "权重事实（有 `guidance_in` ⇔ dev ✓；schnell 没有 ✓）",
        "text_dim": "权重事实（`txt_in` 输入宽 = 4096 ✓ = T5-XXL ✓）",
        "vec_dim": "权重事实（`vector_in.in_layer` 输入宽 = 768 ✓ = CLIP-L pooled ✓）",
        "vae_scale": "同族事实（`latent_formats.Flux`：16ch / 8× / 有仿射 ✓）",
    }


def _wan_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                 ) -> dict[str, str]:
    """Wan2.2 的口径 —— **三格必须写清**：heads 靠查表 ✓、in_channels 是 16 不是 36 ✓、
    σ 的 ×1000 在骨架里（参考实现放在采样器里）✓、I2V 的条件拼接**未接** ✓。"""
    heads = _arg("heads", explicit)
    return {
        "结构": "权重形状反推 ✓（`patch_embedding` 是 Conv3d ⇒ patch 也从权重读 ✓，不用默认值 ✓）",
        "heads": "explicit" if heads is not None else
                 (f"同族事实查表（`dit_wan22.HEADS_BY_GEOMETRY` 的 "
                  f"({shape.dim}, {shape.ffn_dim}, {shape.layers}) ⇒ {shape.heads} ✓；"
                  f"权重里没有这一格 ✗）"),
        "text_dim": "权重事实（`text_embedding.0` 输入宽 = 4096 ✓ = umt5-XXL ✓）",
        "in_channels": "权重事实（`head.head` 出宽 ÷ patch³ = 16 ✓ ⇒ 噪声/潜变量 16 通道 ✓；"
                       "主干吃 36 = 16 + 16 + 4 ✓，那是输入宽不是潜宽 ✓）",
        "vae_scale": "同族事实（`latent_formats.Wan21`：16ch / 8× / meanstd ✓；权重里没有 ✗）",
        "sigmaToTimestep": "×1000 ✓（参考实现的这次乘法在**采样器** `ModelSamplingDiscreteFlow` 里 ✓，"
                           "本仓骨架收裸 σ ⇒ `dit_wan22.TIMESTEP_SCALE` 补 ✓）",
        "i2vConcat": "未接 ✓【未接事实】（骨架只吃**拼好的** 36 通道：噪声 16 + 条件图 16 + 掩码 4 ✓；"
                     "拼接在参考实现的 `model_base` 里 ✓，本仓 `denoise` 只喂 16ch 噪声 ⇒ "
                     "真出片前还要接这一段 ✓ 见 `dit_wan22.split_i2v` ✓）",
        "residency": "bf16 常驻 ≈26.62 GiB（14 288 901 184 参数 × 2 字节 = 26.62 GiB ≈ 28.59 GB ✓；"
                     "别把 GB 当 GiB 报）⇒ 22.49 GiB 的卡**装不下** ✓✗"
                     "（fp8_scaled 只省盘不省卡 ✓，反量化到 bf16 才是驻留量 ✓）",
    }


def _music3_sources(explicit: Mapping[str, Any], shape: Any, spec: FamilyBuilder
                    ) -> dict[str, str]:
    """Music3 的口径 —— **把"这族能建、但管线没接"写在报告里** ✓✗（别让人以为能直接出音频 ✓）。"""
    return {
        "结构": "权重形状反推 ✓（层数/维度/头宽/旋转维/时间步嵌宽全部逐键算出 ✓，"
                "无一项是出厂常量 ✓）",
        "heads": f"权重算出（``{shape.dim} // 64`` = {shape.heads} ✓ —— 头宽 64 是同族事实"
                 f"（``dit_music3.DEFAULT_DIM_HEADS`` ✓），权重里没有这一格 ✗）",
        "latent_channels": "权重事实（`postprocess_conv` 出宽 = 128 ✓ ⇒ 3 维潜变量 "
                           "`(b, 128, T)` ✓）",
        "vae_scale": "**恒 1** ✓【同族事实】（音频没有空间下采样 ✗✗；时间缩放是 "
                     "512 样本/潜帧 ÷ 44100 Hz ✓，见 `latentHop` / `latentSampleRate` ✓）",
        "text_dim": f"权重事实（`latent_conditioners` 入宽 = {shape.context_dim} ✓ "
                    f"= {shape.cond_layers} 层 × {shape.cond_hidden} ✓；主干内部对齐到 "
                    f"{shape.cond_proj} ✓ 不是入口宽 ✗）",
        "sigmaDirection": "⭐ **σ 要翻转成 t = 1 − σ** ✓✗（参考实现放在 `model_base.process_"
                          "timestep` 里 ✓，本仓骨架收裸 σ ⇒ `dit_music3` 自己做 ✓）；"
                          "且前向**输出取负** ✓（参考 `return -out` ⇒ `OUTPUT_SIGN = -1` ✓）"
                          "　—— 这两条**都不报错** ✓，只让歌听起来「糊」「噪」✓✗",
        "audioPipeline": "**已接** ✓【2026-11-06 ✓】DiT 吃**已编好的**条件 "
                         "`(b, frames, 32768)` ✓，条件由 **AR 语义塔**逐帧生成（25 Hz ✓ "
                         "`te_music3.generate` ✓、内含词表 ✓）、解码由 **DAV**（44.1 kHz 立体声 ✓ "
                         "`vae_music3.decode_latents` ✓）⇒ 全链在 "
                         "`TorchBackend._encode_music3_condition` ✓ / `_decode_music3` ✓ / "
                         "`runtime._ensure_music3_loaded` ✓ 三处 ✓，产物 `audio_seed*.wav` ✓。"
                         "⚠️ 仍**不做**的两件（如实记 ✗，不是「已接」）：① DiT 级 CFG 的负向条件 "
                         "（`negative=None` ✓ —— 那要**再跑一遍**整段自回归 ✓）；"
                         "② 参考音频/续写（`initAudio` 之类的条件注入 ✓ 未接 ✗）",
    }


_SOURCE_FNS: dict[str, Callable[[Mapping[str, Any], Any, FamilyBuilder], dict[str, str]]] = {
    "zimage": _zimage_sources,
    "qwen21": _qwen21_sources,
    "qwen_image": _qwen_image_sources,
    "flux": _flux_sources,
    "wan": _wan_sources,
    "music3": _music3_sources,
}


def _module(name: str) -> Any:
    """同包内骨架模块 ✓ —— 只认 :func:`skeletons` 里那几份 ✓（表里写错名字 ⇒ 当场 :class:`BuildError` ✓）。"""
    mod = skeletons().get(name)
    if mod is None:
        raise BuildError(
            f"装配表里的 module={name!r} 不在 SKELETONS（现有：{'/'.join(SKELETON_MODULES)}）✗"
            " —— 加族的**第一步**是把骨架模块 import 进来登记 ✓，不是只改表 ✓",
            reason="pending")
    return mod


def _build_call(spec: FamilyBuilder, shape: Any, shapes: Mapping[str, tuple[int, ...]],
                kwargs: Mapping[str, Any], dtype: Any
                ) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """``build_fn`` 的实参 ⇒ ``(位置参数, 关键字参数)``。

    ⚠️ ``"shapes"`` 与 ``"shape"`` 是两件事 ✗✗：``build_from_shapes(shapes, *, patch, dtype)``
    要的是**形状字典**（它内部自己再 infer 一遍），把它换成形态对象会
    ``TypeError: 'LuminaShape' object is not iterable``；而 ``build(shape)`` / ``Krea2DiT(shape)``
    要的正是形态对象 ✓。
    ⚠️ ``patch`` / ``dtype`` 在骨架里是 **kw-only** ⇒ 混着传会
    ``takes 1 positional argument but 2 were given`` ✓✗。
    """
    args: list[Any] = []
    for name in spec.build_args:
        if name == "shapes":
            args.append(shapes)
        elif name == "shape":
            args.append(shape)
    call_kw: dict[str, Any] = {}
    for name in spec.build_kwargs:
        call_kw[name] = dtype if name == "dtype" else kwargs.get(name, _DEFAULTS.get(name))
    return tuple(args), call_kw


def buildable_keys() -> tuple[str, ...]:
    """有骨架可建的族（**只判族不建**的族不在里面：h3 / sdxl / music3）。"""
    return tuple(BUILDERS)


def readout(key: str | None) -> str:
    """一句话：这个族由本表怎么建（给 CLI / 报告用）。"""
    spec = BUILDERS.get(key or "")
    if spec is None:
        return f"族：{key or '认不出'} ⇒ **无装配声明**（走老路 `dit.build_dit` 或不建）"
    mode = "meta 建壳再物化" if spec.meta else "按形状直接建"
    return (f"族：{spec.label} ⇒ 骨架 `{spec.module}.{spec.build_fn}`（{mode}）"
            f"　改名={len(spec.shortcuts)}　同族常量={len(spec.facts)}\n"
            f"　前向：潜变量 {spec.contract.latent_dims} 维 · σ 形态 `{spec.contract.sigma_form}` · "
            f"{_describe_latent_format(spec.contract)}")


def _describe_latent_format(contract: ForwardContract) -> str:
    """契约那一格的人话 ✓ —— 报告里必须**看得出**解码会发生什么 ✓✗。

    ⚠️ 只写族名不够 ✗：``wan21`` 与 ``qwen_image`` 都是 16ch/8× ✓、只差 mean/std ✓，
    名字相同与否**要连换算方式一起写**才读得出后果（``meanstd`` ⇒ 要 16 组数 ✓；
    ``affine`` ⇒ 只有 shift/scale 两格 ✓；其余 ⇒ 恒等/纯倍数 ✓）。
    """
    name = contract.latent_format
    spec = contract.latent_spec()
    if spec is None:
        return f"口径 `{name}`（**未知** ⇒ 解码前不参与对账 ✓，别当成已核 ✓✗）"
    if int(spec.latent_dimensions) == 1:
        #: ⭐ **音频族** ✓（2026-11-02 Music3 ✓）：`spatial_downscale_ratio` 恒 1 ✓ 是
        #: 「**没有空间维**」✓ 而**不是**「空间不缩放」✓✗ —— 照图族那句写成 `1×`
        #: ⇒ 读的人会以为"音频也能按边长算潜帧" ✓✗。真正的缩放方向是**时间** ✓。
        return (f"口径 `{name}`（{spec.process} · scale={spec.scale_factor} · "
                f"{spec.latent_channels}ch · **1 维音频**：时间 ÷{spec.temporal_downscale_ratio}"
                f"（{spec.temporal_downscale_ratio} 样本/潜帧 ÷ 44100 Hz ⇒ "
                f"{44100.0 / float(spec.temporal_downscale_ratio):.4f} Hz 潜帧率）· 无空间维 ✗)")
    if str(spec.process) == "affine":
        return (f"口径 `{name}`（affine · shift={spec.shift_factor} · scale={spec.scale_factor} · "
                f"{spec.latent_channels}ch）")
    if str(spec.process) == "meanstd":
        return (f"口径 `{name}`（meanstd ⇒ 需 {spec.latent_channels} 组 mean/std · "
                f"{spec.latent_channels}ch · {spec.spatial_downscale_ratio}×）")
    return (f"口径 `{name}`（{spec.process} · scale={spec.scale_factor} · "
            f"{spec.latent_channels}ch · {spec.spatial_downscale_ratio}×）")


def build(key: str | None, info: SafetensorsInfo, explicit: Mapping[str, Any] | None = None,
          *, dtype: Any = None, path: Any = None,
          int8_plan: Callable[..., dict[str, tuple[int, ...]]] | None = None,
          int4_plan: Callable[..., dict[str, tuple[int, ...]]] | None = None
          ) -> tuple[Any, dict[str, Any]] | None:
    """按族建骨架 ⇒ ``(module, config)``；这族没有装配声明 ⇒ ``None``（调用方走老路）。

    ``config`` 的三段口径与原手写版**逐字一致**（有测试守着）：
    ``dict(shape.__dict__)``（权重事实）+ 改名/常量/派生（前端协议字段）+ ``configSources``。
    """
    spec = BUILDERS.get(key) if key else None
    if spec is None:
        return None

    explicit = dict(explicit or {})
    adapter = _ADAPTERS.get(spec.adapter)
    #: ⭐ 4bit 打包权重的**逻辑**列数先还原 ✓✗（不还原 ⇒ 拓扑按一半的维建骨架 ⇒
    #: 物化时逐键 `shapeMismatch`，报错指向"拿错文件"✓✗ 见 :func:`_logical_shapes_of` ✓）
    raw = _logical_shapes_of(
        {name: tuple(entry.shape) for name, entry in info.tensors.items()}, info, path)
    shapes, key_map, prefix = (raw, None, None)
    if adapter is not None:
        shapes, key_map, prefix = adapter.prepare(raw)

    mod = _module(spec.module)
    shape_error = getattr(mod, "ShapeError", BuildError)
    kwargs = {name: value for name in spec.shape_args
              if (value := _arg(name, explicit)) is not None}
    try:
        shape = getattr(mod, spec.shape_fn)(shapes, **kwargs)
    except shape_error as err:
        raise BuildError(
            f"{spec.label} 权重推不出可装的结构 ✗（**不拿出厂形状硬装** ✗ —— 那会"
            f"「名字对、形状错」✓✗）：{err}") from err

    dtype = dtype or _default_dtype()
    entry = getattr(mod, spec.build_fn)
    args, call_kw = _build_call(spec, shape, shapes, kwargs, dtype)
    extra: dict[str, Any] = {}
    extra_sources: dict[str, str] = {}
    if spec.meta:
        import torch  # noqa: PLC0415

        with torch.device("meta"):
            module = entry(*args, **call_kw)
        #: ② 降到计算精度（meta 上只改 dtype、不分配 ✓）
        module.to(dtype)
        #: ③ 族特有钩子（Krea2：F32 岛 + int8 换壳）—— 必须排在物化**之前** ✗✗
        if adapter is not None:
            extra, extra_sources = adapter.shell(module, info, prefix, path, int8_plan, shape,
                                                 int4_plan)
        #: ④ 到这里才真分配（按换完壳之后的最终精度 ✓）
        materialize_meta(module, torch_mod=torch, device="cpu")
    else:
        module = entry(*args, **call_kw)
        if adapter is not None:
            extra, extra_sources = adapter.shell(module, info, prefix, path, int8_plan, shape,
                                                 int4_plan)

    config: dict[str, Any] = dict(vars(shape))          # dataclass ⇒ 字段名即权重事实 ✓
    config["family"] = spec.key
    for target, source in spec.shortcuts:
        config[target] = int(getattr(shape, source))
    for name in spec.derived:
        #: ⚠️ 声明的派生格必须真在 shape 上 ✓（拼错 ⇒ 当场报 ✓ 不写 ``None`` 进去 ✗：
        #: 上层 ``config.get("ref_timestep_zero")`` 拿到 None 与「没有参考图通路」**长得一样** ✓✗）
        if not hasattr(shape, name):
            raise BuildError(f"{spec.label} 的 derived 声明了 {name!r} ✗，"
                             f"而 {type(shape).__name__} 上没有这一格 ✓")
        config[name] = getattr(shape, name)
    for name, value in spec.facts:
        config[name] = value
    config.update(extra)
    if key_map is not None:
        #: ⭐ 装载时要用的前缀剥离表 —— 由 :meth:`TorchBackend.load_weights` 取走（不是跑期参数）
        config["weightKeyMap"] = key_map
    sources: dict[str, str] = dict(spec.sources)
    if spec.source_fn:
        sources = dict(_SOURCE_FNS[spec.source_fn](explicit, shape, spec))
    sources.update(extra_sources)
    config["configSources"] = sources
    return module, config
