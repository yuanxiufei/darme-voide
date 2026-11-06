"""GPU + CPU 联合驻留：**权重分两处放、激活过界只搬一次**。

## 为什么要有这一层（2026-10-31 创作者口径）

「不能只走 CPU，也要 GPU 和 CPU 联合跑 —— 单靠 CPU 很容易超载」。本仓此前只有两种状态：
全在卡上（放不下就 ``CUDA out of memory`` ✗）或全在内存里（CPU 算 DiT，几十倍慢 ✗）。
没有中间档 ⇒ 一张 22 GiB 的卡遇上 17.94 GiB 的 int8 权重 + 十几 GiB 激活，就只能整份退回 CPU。

这里补的是**按显存预算切分**：能放的层放卡上、放不下的留在 CPU，两种设备**都参与计算**，
跨界处只搬**激活**（几十 MB ✓）而不是搬权重（每层上百 MB ✗✗）。

## 两种搬运口径（都由 hook 实现，模型代码一行不用改）

| 模式 | 谁算 | 每次过界搬什么 | 代价 |
|---|---|---|---|
| ``split``（默认） | 卡上算 GPU 层、CPU 算 CPU 层 | **激活**（很小） | CPU 那几层慢 50–100× ⇒ 层数要少 |
| ``park`` | 只在卡上算 | **权重**（每层上百 MB） | 每步多一趟 PCIe，但 CPU 不参与矩阵乘 |

⇒ ``split`` 适合「缺一点显存」：把末尾几层丢给 CPU，损失可控；
``park`` 适合「CPU 弱、卡只差一点点」：CPU 只当仓库。
**选哪个不该猜** ✓：:func:`plan` 会把两种模式的预估搬运量一起算出来，由调用方按实测定。

## 诚实边界

* 只支持**推理**（``no_grad``）✗：hook 里就地改 ``param.data`` 的设备，反向图会断。
* 出错时 post-hook 不执行（Python 的 hook 不包异常 ✓⇒ 模块可能停在「已搬一半」的状态 ✗）：
  :meth:`Handle.detach` 会按计划复位，装载方应在异常路径上调用它。
* 预算只管**权重**：激活/工作区靠 ``reserve`` 那部分余量兜（比例是估的 ✗，实测为准 ✓）。

## 预算不是「一个数」（2026-11-02 补 ✓）

预算 = 「可用 − 余量 − 钉死」✓（:func:`decompose` ✓），**三段分开报** ✓ ⇒ 装不下时能直接答
「是余量档吃掉的 ✓ / 还是你钉死的那块 ✓ / 还是这件确实比卡大 ✓」，不用挨个试 ✓✗。

「切分救不了」（连非层参数都放不下 ✓）与「没有显存」（``budget <= 0`` ✓）**是两个布尔** ✓✗：
两者都表现为"整份在内存" ✓，可前者该换档位/别装 ✗、后者该去插卡或等卡空出来 ✗。
``on_refuse="raise"`` 可让前者中止装载 ✓（默认 ``"cpu"`` ⇒ 老调用方零变化 ✓）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

from . import foresight as foresee_mod
from . import host_cache as host_mod

#: 给激活/工作区留的显存比例 ✓（72×128 那种长序列的激活能到十几 GiB ✗ ⇒ 不能只留 5%）
DEFAULT_RESERVE = 0.25

#: ⭐ **按件类型**的激活余量 ✓✗（2026-11-01 加 ✓ —— 25% 一刀切对**文本塔/VAE**是纯浪费 ✓✗）。
#:
#: 为什么可以分开 ✗✗：余量留的是**激活** ✓，而激活量与件类型差几个数量级 ✓：
#:
#: | 件 | 激活从哪来 | 量级 | 依据 |
#: |---|---|---|---|
#: | ``dit`` | 每个采样步、卡上那几层的完整激活图 ✓ | **几～十几 GiB** ✓ | Qwen-Image 2512 单层 token 数 × hidden 已到 GiB 级 ✓ |
#: | ``te`` | 一次前向、序列≤512 ✓、无时间维 ✓ | **0.1～0.4 GiB** ✓ | umt5-XXL 4096h×512t×24L×2B ≈ 0.10 GiB ✓；Qwen3-VL-32B 5120h×512t×64L×2B ≈ 0.34 GiB ✓ |
#: | ``vae`` | 一次解码、逐块 ✓ | < 1 GiB ✓ | 0.33～0.63 GiB 的整件本身就在卡上，激活同量级 ✓ |
#:
#: ⚠️ 25% 摊在 22.49 GiB 卡上是 **5.2 GiB** ✓✗ —— 对文本塔等于**白扔 4.8 GiB** ✓：
#: 25.44 GiB 的 H3 文本塔只被允许上卡 61% ✓，剩下 39% 去 CPU 上逐层爬 ✓✗；
#: 按 5% 算能上卡 **77%** ✓（余量仍有 1.0 GiB，是估算激活的 **3 倍** ✓）。
#: ⇒ 这一格只按「这件真正需要多少工作区」给 ✓，**不是**把 DiT 的那把尺子借给小件 ✗。
RESERVE_BY_KIND: dict[str, float] = {"dit": DEFAULT_RESERVE, "te": 0.05, "vae": 0.05}


def reserve_for(kind: str | None) -> float:
    """``kind``（``"dit"`` / ``"te"`` / ``"vae"``）该留多少余量 ✓；不认识 ⇒ :data:`DEFAULT_RESERVE` ✓。

    ⚠️ 未知件**必须**退回 25% ✗✗（保守那一侧 ✓）：认错件类型时多留余量只是慢一点 ✓，
    少留余量是 OOM ✓✗ —— 这一格宁可认不出、也不许猜小 ✗。
    """
    return RESERVE_BY_KIND.get(str(kind or "").strip().lower(), DEFAULT_RESERVE)


#: 卡上可用空间低于这个数就不值得切 ✗（连几个 embedding 都放不下 ⇒ 别假装联合）
MIN_GPU_BYTES = 256 * 1024 * 1024

#: 余量**下限** ✓✗（2026-11-02 加 ✓，口径取自 `strataflow` 的 `planner.cpp`：
#: ``safety_margin = max(256 MiB, usable/100*6)``）。
#:
#: ⚠️ 为什么要下限 ✗：比例在小数上会缩没 ✓ —— ``te`` 那档是 5% ✓，卡上若有 600 MiB 可用，
#: 5% 只有 30 MiB ✓✗，而**一次前向的临时张量**（哪怕小塔）也到几十 MiB ✓
#: ⇒ 余量名义上"留了"、实际等于没留 ✓✗。
#:
#: ⚠️ 只在 ``reserve > 0`` 时生效 ✗✗：``reserve=0.0`` 是"我**明确**不要余量" ✓
#: （A/B 测速与判据都用这个口径 ✓），这时硬塞下限会**静默改掉**调用方的决定 ✗。
MIN_MARGIN_BYTES = 256 * 1024 * 1024

#: 真干活的两种模式 ✓（``off`` 不在这里 —— 它不是"另一种切法"，而是"不切" ✓）
WORK_MODES: tuple[str, ...] = ("split", "park")
#: 对外可配的全部取值 ✓（**单一真源** ✓ —— `TorchBackend` / `bridge` / HTTP 白名单都读它，
#: 不许在别处手抄一份清单 ✗：``int8_offload`` 就是这么漏过一次的 ✓）。
MODES: tuple[str, ...] = ("off", *WORK_MODES)

#: **层**的命名 ✓ —— 判据只认「编排容器里的第 N 个」✓，别认成"任何以数字结尾的名字"✗。
#:
#: ⚠️ 2026-10-31 **实测修正** ✗：初版只认 ``layers.N`` ✓，而本仓**两条真主干都叫 ``blocks``** ✓✗
#: （`dit.py` 的 ``self.blocks`` 与 `h3_form.H3FormTrunk.blocks` ✓）⇒ 真模型上**一层都识别不出来** ✓✗，
#: 于是「非层参数」= 整摞 blocks ⇒ 一律判 `fits=False`、整份退回 CPU ✓✗（联合驻留静默变成"纯 CPU"，
#: 正是这次要消灭的那一档 ✓）。`_Tiny` 那种测试模型用 ``layers`` ⇒ **测试全绿也发现不了** ✓✗。
#: 多认几个族名不会有坏处 ✓：它们只在对应形态里出现 ✓；而 ``final_modulation.1`` 这种
#: 「父名不是容器」的索引**不会**被认成层 ✓（正则锚定了父名）。
_LAYER_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in (
        r"(?:[^.]+\.)*layers\.\d+$",              # 通用 / 测试模型（也认 `model.layers.7` ✓）
        r"(?:[^.]+\.)*blocks\.\d+$",              # ⭐ 本仓 DiT 与 H3 主干都是它 ✓
        r"(?:[^.]+\.)*transformer_blocks\.\d+$",  # Flux / SD3 式命名
        r"(?:[^.]+\.)*double_blocks\.\d+$",       # Flux 双流
        r"(?:[^.]+\.)*single_blocks\.\d+$",
        r"(?:[^.]+\.)*layers_\d+\.\d+$",          # Wan2.2 的 layers_0.N
    ))


def is_layer(name: str) -> bool:
    """``name`` 是不是**层**（可串行、可切片、可只把前缀放卡上 ✓）。"""
    return any(pattern.match(name) for pattern in _LAYER_RES)


class ReloadSlot:
    """**延迟绑定**的重读入口 ✓✗（2026-11-02 加 ✓）。

    ## 为什么需要"延迟"（这是顺序问题，不是设计癖好 ✗）

    ``place()`` 必须在 ``load_module_weights()`` **之前**调 ✓（要先把参数摆好、把 hook 挂好 ✓）；
    可重读入口（:func:`weights.make_reloader` ✓）要等**装载时**才造得出来 ✓
    （它握着文件 reader 与逐张量还原上下文 ✓）。两者顺序**正好相反** ✓✗。

    所以 ``place(reload=slot)`` 先给一个"空壳" ✓，装载成功后由 +weights+ 把它
    ``bind`` 上 ✓：``weights.load_module_weights`` 只会写 ``module._residency_reload`` ✓，
    **13 处装载调用点一个都不用改** ✓（逐个加参数必漏 ✓，漏一处就是"卸了层再也读不回来"✗）。

    ⚠️ 没绑上就调用 ⇒ **当场抛** ✓✗：返回空字典等于告诉调用方"这层没有参数" ✓，
    前向会拿一堆 0 元素算出静默错误 ✓✗（本仓最不能接受那种 ✓）。
    """

    __slots__ = ("_fn", "bound")

    def __init__(self) -> None:
        self._fn: Any = None
        self.bound = False

    def bind(self, fn: Any) -> "ReloadSlot":
        self._fn = fn
        self.bound = fn is not None
        return self

    def __call__(self, group: str) -> dict[str, Any]:
        if not self.bound:
            raise RuntimeError(
                f"重读入口没接上 ✗：`{group}` 被内存这一级卸到了盘上 ✓，但装载时没登记还原器 ✓"
                f"（`report.restorer is None` ⇒ 这次装载走的是整份路径 / 预检就中止了 ✗）"
                f" ⇒ 关掉 host_cache，或让装载走流式路径 ✓")
        return self._fn(group)

    def close(self) -> None:
        """把重读口还回去 ✓（**只往里转手** ✓：还原器自己决定开没开、关不关 ✓）。

        ⚠️ 必须有人关 ✗✗：重读是**懒开**的一把新句柄 ✓（装载那把已经随 ``with`` 关了 ✓），
        一个 DIᵀ 加一个 TE 就是两把 ✓，每个阶段换来换去不还 ⇒ fd 与无缓冲读口一路堆到进程退出 ✓✗。
        """
        closer = getattr(self._fn, "close", None)
        if closer is None:  # pragma: no cover - `make_reloader` 走的是裸函数 ✓
            return
        try:
            closer()
        except Exception:  # noqa: BLE001 - 退件不许反过来搞坏退件 ✓✗
            pass

    def __repr__(self) -> str:  # pragma: no cover - 只在调试里看得见 ✓
        return f"ReloadSlot(bound={self.bound})"


@dataclass(frozen=True)
class Group:
    """一个**驻留单位** —— 通常是一个直接子模块（``layers.7`` / ``x_embedder`` / …）。

    ⚠️ 粒度是「直接子模块」而不是「单个张量」✗：一层里的 qkv/norm/mlp 必须**同设备**，
    否则一次前向内部就要来回搬几十次 ✓✗（那是把 offload 做成了 PCIe 压测 ✗）。
    """

    name: str
    nbytes: int
    layers: int = 0

    @property
    def is_layer(self) -> bool:
        return self.layers > 0


@dataclass(frozen=True)
class Plan:
    """切分结果（纯数据 ✓ ⇒ 不用 torch、不用卡也能单测 ✓）。"""

    budget: int
    on_gpu: tuple[str, ...]
    on_cpu: tuple[str, ...]
    gpu_bytes: int
    cpu_bytes: int
    fits: bool
    reason: str
    #: 所有**层**的字节合计 ✓ —— ``park`` 每步搬的就是这些 ✓（不是 ``cpu_bytes`` ✗：
    #: 全在卡上时 ``cpu_bytes`` 是 0，可 ``park`` 明明每层都要搬一趟 ✓）。
    layer_bytes: int = 0
    #: ⭐ **切分救不了** ✓✗（2026-11-02 加 ✓）—— 与「没有显存」是**两件不同的事** ✗✗：
    #:
    #: * ``budget <= 0``（没卡 / 卡被占满 ✓）⇒ 本来就该全在内存 ✓，不是拒绝 ✓；
    #: * ``refused=True`` ⇒ 有预算，但**连非层参数都装不下** ✓ ⇒ 切分这一招**在这件上无效** ✓✗。
    #:
    #: 两者的共同结果都是"全在 CPU" ✓，可要做的动作完全相反 ✗（前者去插卡 / 等卡空出来 ✓，
    #: 后者只能整份联合、换更小的档位、或干脆别装 ✓）⇒ 必须分成两个布尔 ✓，不许合成一个 ✗。
    refused: bool = False
    #: 预算分解 ✓（:meth:`Budget.to_dict` ✓）—— ``None`` ⇒ 调用方直接给了 ``budget`` 一个数、
    #: **没有分解可看** ✓（这时不编一份假的 ✓，报告里就是 ``null`` ✓）。
    budgetInfo: dict[str, Any] | None = None
    #: ⭐ 被**内存这一级**卸到盘上的组 ✓✗（2026-11-02 加 ✓）—— ``load_module_weights`` 要跳过它们 ✓：
    #: 参数已被换成 0 元素占位 ✓，照着 ``copy_`` 进去会当场形状不符 ✗✗。
    spilled: tuple[str, ...] = ()
    #: 卸掉的字节数 ✓（``cpu_bytes`` 已扣掉它 ✓ ⇒ ``guard`` 核内存时核的是**实际要占的**✓）。
    spillBytes: int = 0

    def __post_init__(self) -> None:
        """``refused`` 与 ``fits`` **互斥** ✓✗（2026-11-02 加 ✓）。

        这条不是洁癖 ✗✗：「切不动」与「切成了」同时为真，等于报告里既说"没联合"又说"联合了" ✓✗
        ⇒ 读报告的人会去查错地方 ✓。手写 :class:`Plan`（测试替身 / 将来的新调用方）最可能踩 ✓。
        """
        if self.refused and self.fits:
            raise ValueError(
                "Plan 自相矛盾 ✗：refused=True（切分救不了）与 fits=True（切分成了）不能同时成立 ✓✗")

    @property
    def total_bytes(self) -> int:
        return self.gpu_bytes + self.cpu_bytes

    @property
    def layers_on_gpu(self) -> int:
        return sum(1 for name in self.on_gpu if is_layer(name))

    @property
    def layers_on_cpu(self) -> int:
        return sum(1 for name in self.on_cpu if is_layer(name))

    @property
    def needs_split(self) -> bool:
        """需不需要 hook —— 全在一处时**一根 hook 都不挂** ✓（老路零开销 ✓）。"""
        return bool(self.on_cpu) and bool(self.on_gpu)

    def transfer_bytes(self, mode: str = "split") -> int | None:
        """每次前向的**过界搬运量**（单程；双向就 ×2 ✓）。

        ``split`` ⇒ 只搬激活 ⇒ 这一层不知道激活多大 ⇒ 返回 ``None`` ✗（不猜 ✓）；
        ``park`` ⇒ 搬的是**全部层的权重** ⇒ 能精确算 ✓。
        """
        if mode == "park":
            return self.layer_bytes
        return None


def _bytes_of(module: Any) -> int:
    total = 0
    for tensor in [*module.parameters(recurse=True), *module.buffers(recurse=True)]:
        if tensor is None:  # pragma: no cover
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def _own_bytes(module: Any) -> int:
    """模块**自己的**参数/缓冲字节数 ✓（不递归 ✓ —— 与 :func:`_bytes_of` 是两回事 ✗）。"""
    total = 0
    for tensor in [*module.parameters(recurse=False), *module.buffers(recurse=False)]:
        if tensor is None:  # pragma: no cover
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def _layer_list_of(child: Any) -> list[Any]:
    """``child`` 下面有多少个**编排容器**（``ModuleList``/``ModuleDict``/``Sequential``）✓。"""
    return [node for _sub, node in child.named_children()
            if type(node).__name__ in ("ModuleList", "ModuleDict", "Sequential")]


def _expand(name: str, child: Any) -> list[tuple[str, Any]]:
    """把 ``ModuleList``/``Sequential`` 展开一层 ✓ —— 否则整摞层会合成**一个**不可切分的组 ✗。

    ⚠️ 不 import torch ⇒ 按类型名认 ✓（``ModuleList``/``Sequential`` 的名字是公开约定 ✓）。

    ⭐ **穿透裸容器** ✓✗（2026-10-31 加 ✓）：HF 式写法会多包一层 ——
    ``self.model = nn.Module()`` 里面才是 ``embed_tokens`` / ``layers`` / ``norm`` ✓
    （`qwen35_te.Qwen35TeBackbone` 就是这样 ✓，Qwen3-4B 那条同理 ✓）。
    不穿透的话，``survey`` 看到的是**一个**名叫 ``model`` 的子模块 ✓ ⇒ 整份 18.5 GiB 被当成
    「非层参数」✗ ⇒ 要么全上卡、要么全 CPU ✓✗ —— 联合驻留**静默变成纯 CPU** ✓✗，
    正是 50–53 行记的同一个病（那次是 ``blocks`` 认不出 ✓）。

    ⚠️ 穿透要有**硬条件** ✗✗（宁可认不出、也不猜 ✗）：
      ① 类型名是 ``Module``（裸容器 ✓，不是 ``Linear``/``Conv2d`` 那种实参模块 ✓）；
      ② **自己名下没有**直接参数/缓冲 ✓（有 ⇒ 它本身是个真模块，切开是错的 ✗）；
      ③ 下面**恰好一个**编排容器 ✓（多个 ⇒ 分不清哪个是主干层，整块当非层 ✓）。
    """
    if type(child).__name__ in ("ModuleList", "ModuleDict", "Sequential"):
        return [(f"{name}.{sub}", node) for sub, node in child.named_children()]
    if (type(child).__name__ == "Module" and not _own_bytes(child)
            and len(_layer_list_of(child)) == 1):
        out: list[tuple[str, Any]] = []
        for sub, node in child.named_children():
            out.extend(_expand(f"{name}.{sub}", node))
        return out
    return [(name, child)]


def survey(module: Any) -> list[Group]:
    """按直接子模块盘点字节数（顺序＝**前向顺序**，规划靠它保持「连续前缀在卡上」✓）。

    ⚠️ 顶层裸参数（``x_pad_token`` 之类）也算一组 ✓：它们每步都用、又极小 ✓，
    规划时归进「必须上卡」那一档 ✓（留在 CPU 反而每步搬一次 ✗）。
    """
    groups: list[Group] = []
    loose = 0
    for _name, param in module.named_parameters(recurse=False):
        loose += (param.numel() * param.element_size()) if param is not None else 0
    for _name, tensor in module.named_buffers(recurse=False):
        loose += (tensor.numel() * tensor.element_size()) if tensor is not None else 0
    if loose:
        groups.append(Group(name="‹root›", nbytes=loose))
    for name, child in module.named_children():
        for full, node in _expand(name, child):
            groups.append(Group(name=full, nbytes=_bytes_of(node),
                                layers=1 if is_layer(full) else 0))
    return groups


def plan(groups: Iterable[Group], budget: int, *, reason: str = "",
         budget_info: dict[str, Any] | None = None) -> Plan:
    """按预算切分：**非层参数先上卡**，然后层按前向顺序依次上卡，装不下就停。

    ⚠️ 为什么「依次上卡」而不是「挑小的塞」✗：层是**串行**用的 ✓ —— 卡上若是 0,1,2,5,6 这种
    跳号 ✗，激活每步都要过界好几次 ✓✗；保持前缀连续 ⇒ 整个前向**只过一次界** ✓。

    ``budget_info``（2026-11-02 加 ✓）＝:meth:`Budget.to_dict` ✓，**只进报告、不参与切分** ✓：
    切分只认 ``budget`` 这一个数 ✓（两处都用它 ⇒ 口径不会分叉 ✓）。
    """
    items = list(groups)
    layer_bytes = sum(g.nbytes for g in items if g.is_layer)
    total_bytes = sum(g.nbytes for g in items)
    if budget <= 0:
        return Plan(budget=budget, on_gpu=(), on_cpu=tuple(g.name for g in items),
                    gpu_bytes=0, cpu_bytes=total_bytes, fits=False,
                    reason="预算 ≤ 0：没有可用的显存 ⇒ 只能整份放 CPU ✗",
                    layer_bytes=layer_bytes, budgetInfo=budget_info)
    non_layers = [g for g in items if not g.is_layer]
    layers = [g for g in items if g.is_layer]
    used = 0
    on_gpu: list[str] = []
    for group in non_layers:
        used += group.nbytes
        on_gpu.append(group.name)
    if used > budget:
        return Plan(budget=budget, on_gpu=(), on_cpu=tuple(g.name for g in items),
                    gpu_bytes=0, cpu_bytes=total_bytes, fits=False,
                    reason=(f"连非层参数（{used / 2**30:.2f} GiB）都超预算"
                            f"（{budget / 2**30:.2f} GiB）⇒ 切分救不了 ✗"),
                    layer_bytes=layer_bytes, refused=True, budgetInfo=budget_info)
    for group in layers:
        if used + group.nbytes > budget:
            break
        used += group.nbytes
        on_gpu.append(group.name)
    on_cpu = [g.name for g in items if g.name not in set(on_gpu)]
    gpu_bytes = sum(g.nbytes for g in items if g.name in set(on_gpu))
    cpu_bytes = sum(g.nbytes for g in items if g.name in set(on_cpu))
    text = reason or (f"卡上 {gpu_bytes / 2**30:.2f} GiB（{len([n for n in on_gpu if is_layer(n)])} 层）"
                      f"／CPU {cpu_bytes / 2**30:.2f} GiB（{len([n for n in on_cpu if is_layer(n)])} 层）")
    return Plan(budget=budget, on_gpu=tuple(on_gpu), on_cpu=tuple(on_cpu),
                gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes, fits=True, reason=text,
                layer_bytes=layer_bytes, budgetInfo=budget_info)


class BudgetRefused(ValueError):
    """预算**连非层参数都放不下** ⇒ 这件切不出联合 ✓✗（:func:`place` 的 ``on_refuse="raise"``）。

    ⚠️ 与 :class:`scheduler.InsufficientResources` 分工 ✗✗：那个管**内存**放不下 ✓
    （整份退 CPU 之前核 ✓），这个管**显存**切不动 ✓（切完才知道 ✓）—— 两件事，
    报错也要能分开 ✓（"换更大的卡"与"关掉别的进程"是不同的动作 ✓）。
    """


@dataclass(frozen=True)
class Budget:
    """卡上那笔预算**拆开**了给人看 ✓✗（2026-11-02 加 ✓，口径取自 `strataflow` 的 `planner.cpp`：
    ``planned_peak = trunk + kv + compute + process_floor + safety_margin`` ⇒ ``fits`` 由它判）。

    ## 为什么"一个数"不够 ✗✗

    ``budget`` 原先只有一个 ``int`` ✓ —— 装不下时报告只能说「预算 16.9 GiB ✗」，
    说不出**这笔预算被谁花掉了** ✓✗。同一句"装不下"背后的动作完全不同：

    * 余量占大头 ⇒ 该换**件类型档**（``dit`` 25% → ``te``/``vae`` 5% ✓）；
    * ``fixed`` 占大头 ⇒ 该**挪走**预留给别人的那部分 ✓；
    * 两者都不大 ⇒ 这件的权重**真的**比卡大 ✓ ⇒ 只能联合驻留或换更小的档 ✓。

    一个数把三种情况压成一句话，用户只能挨个试 ✓✗。

    | 字段 | 是什么 | 谁定的 |
    |---|---|---|
    | ``usable`` | 本次分配**真能拿到**的显存（free ＋ 可复用缓存 ✓） | 读数 |
    | ``margin`` | 留给激活/工作区的 ✓ ``= max(下限, usable × ratio)`` | :data:`RESERVE_BY_KIND` |
    | ``fixed`` | 调用方**声明**先扣的（别的件 / 已占用 ✓） | 调用方 |
    | ``effective`` | ``usable − margin − fixed`` ⇒ **真正给权重的那个数** | 算出来的 |
    """

    usable: int
    margin: int
    fixed: int
    effective: int
    ratio: float
    kind: str | None = None
    source: str = "cuda"

    @property
    def ok(self) -> bool:
        return self.effective > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "usableGiB": round(self.usable / 2**30, 2),
            "marginGiB": round(self.margin / 2**30, 2),
            "fixedGiB": round(self.fixed / 2**30, 2),
            "effectiveGiB": round(self.effective / 2**30, 2),
            "reserve": self.ratio,
            "reserveKind": self.kind,
            "source": self.source,
        }


def decompose(usable: int, *, reserve: float | None = None, kind: str | None = None,
              fixed_bytes: int = 0, source: str = "cuda") -> Budget:
    """把一笔可用显存拆成「余量 / 钉死 / 真给权重」三份 ✓✓（纯算术 ✓ ⇒ 没卡也能单测 ✓）。

    ``reserve=None`` ⇒ 按 ``kind`` 查 :data:`RESERVE_BY_KIND` ✓；显式 ``reserve`` **优先** ✓。

    ⚠️ ``reserve=0.0`` ⇒ 余量**就是 0** ✗✗（不补 :data:`MIN_MARGIN_BYTES`）：
    这个值在 A/B 测速与判据里是「我明确不要余量」✓，静默补一个下限会改掉调用方的结论 ✓✗。
    """
    total = max(0, int(usable))
    ratio = reserve_for(kind) if reserve is None else max(0.0, min(1.0, float(reserve)))
    fixed = max(0, int(fixed_bytes))
    margin = 0 if ratio <= 0 else max(int(MIN_MARGIN_BYTES), int(total * ratio))
    return Budget(usable=total, margin=margin, fixed=fixed,
                  effective=max(0, total - margin - fixed),
                  ratio=ratio, kind=kind, source=source)


def budget_of_cuda(torch_mod: Any, *, reserve: float | None = None,
                   index: int | None = None, fixed_bytes: int = 0,
                   kind: str | None = None) -> Budget | None:
    """读一次卡况 ⇒ **带分解**的预算 ✓；没卡/取不到 ⇒ ``None`` ✓（不猜 ✗）。

    ⚠️ 「可用」≠ ``mem_get_info`` 的 free ✓✗（2026-11-01 实测 ✓，与 `device_policy.probe` 同口径 ✓）：
    本进程 allocator 里**留着没还的缓存块**在驱动眼里是"已占" ✓，可它们随时能复用 ✓
    ⇒ 必须加进来 ✓；不加的结果是预算被凭空砍掉十几 GiB ✓（实测：退掉一件之后 `reserved` 比
    ``allocated`` 多 14.3 GiB ✓ ⇒ 25.28 GiB 的塔只被允许上卡 3.5 GiB ✓✗）。
    """
    if not getattr(torch_mod.cuda, "is_available", lambda: False)():
        return None
    try:
        free, _total = torch_mod.cuda.mem_get_info(index)
        reusable = max(0, int(torch_mod.cuda.memory_reserved(index))
                       - int(torch_mod.cuda.memory_allocated(index)))
    except Exception:  # pragma: no cover - 老驱动 / 不支持的设备
        return None
    return decompose(int(free) + reusable, reserve=reserve, kind=kind,
                     fixed_bytes=fixed_bytes)


def budget_from_cuda(torch_mod: Any, *, reserve: float = DEFAULT_RESERVE,
                     index: int | None = None) -> int:
    """按「当前可用显存 × (1 − 余量)」算预算 ✓；没卡/取不到 ⇒ ``0`` ✓（不猜 ✗）。

    ⭐ 这是 :func:`budget_of_cuda` 的**只要那个数**的薄壳 ✓（2026-11-02 收口 ✓）：
    两份读数口径只能有一处 ✗✗（分开写迟早一处补了缓存、另一处没补 ✓）。
    想要「余量吃掉了多少」⇒ 用 :func:`budget_of_cuda` 拿 :class:`Budget` ✓。
    """
    detail = budget_of_cuda(torch_mod, reserve=reserve, index=index)
    return 0 if detail is None else int(detail.effective)


# ── hook 调度（真正让两台设备一起干活的那部分）──────────────────────────────────────


def _move(value: Any, device: Any, torch_mod: Any) -> Any:
    """递归搬张量 ✓；非张量原样返回 ✓（``None``/数字/字符串 在引擎里到处都是 ✓）。"""
    if torch_mod.is_tensor(value):
        if value.device == device:
            return value
        pinned = value.is_pinned() if value.device.type == "cpu" else False
        return value.to(device, non_blocking=bool(pinned))
    if isinstance(value, tuple):
        return tuple(_move(item, device, torch_mod) for item in value)
    if isinstance(value, list):
        return [_move(item, device, torch_mod) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device, torch_mod) for key, item in value.items()}
    return value


def _pin(tensor: Any) -> bool:
    """锁页（异步 H2D 的前提 ✓）。锁不上不算错 ✓（只是退化成同步 ✓）。"""
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return True
    try:
        tensor.data = tensor.data.pin_memory()
        return True
    except RuntimeError:  # pragma: no cover - 无 CUDA / 锁页配额满
        return False


def _tensors_of(module: Any) -> list[Any]:
    return [t for t in [*module.parameters(recurse=True), *module.buffers(recurse=True)]
            if t is not None]


def _node(module: Any, name: str) -> Any:
    """按 ``layers.7`` 这种点路径取子模块 ✓（``‹root›`` ⇒ 模块自己 ✓）。"""
    if name == "‹root›":
        return module
    node = module
    for part in name.split("."):
        node = getattr(node, part)
    return node


class Handle:
    """一套已挂上的 hook ✓ —— ``detach()`` 摘掉它们并把参数复位 ✓。

    ``transfers`` 是**真的搬过几次**（测试与报告都读它 ✓）：跨设备时它必须涨 ✓，
    同一设备时必须**一动不动** ✓（这是「老路零开销」那条的机器判据 ✓）。
    """

    def __init__(self, module: Any, plan_: Plan, *, mode: str, main: Any, cpu: Any) -> None:
        self.module = module
        self.plan = plan_
        self.mode = mode
        self.main = main
        self.cpu = cpu
        self.transfers = 0
        self.hooks: list[Any] = []
        self.pinned = True
        self.unmoved: list[str] = []
        self.has_entry_hook = False
        #: 宿主层缓存 ✓（``None`` ⇒ 没启用「内存也有上限」这一级 ✓，行为与改动前逐字一致 ✓）
        self.hostCache: Any = None
        #: 这一趟**真的**被卸到盘上、等着被读回来的组 ✓（前向到它时 :func:`_ensure` 先补 ✓）
        self.spilled: set[str] = set()
        #: 内存这一级真搬过几次 ✓（``0`` ⇒ 配额定得够、一层都没卸 ✓ —— 报告要能区分
        #: "没启用" 与 "启用了但没触发" ✓✗，两者都不是"卸干净了" ✓）
        self.restores = 0
        self.spillBytes = 0
        #: 每个组**装载时**的形状/dtype 快照 ✓（重读回来要拿它核 ✓ —— 0 元素占位已经没形状了 ✗）
        self.hostMeta: dict[str, dict[str, Any]] = {}
        #: 重读入口 ✓（退件时要把它手里那把**懒开**的读口还回去 ✓，见 `ReloadSlot.close` ✓）
        self.reload: Any = None
        #: 层预取上卡 ✓（``None`` ⇒ 这一级**没启用** ✓，与改动前逐字一致 ✓；判据见 :mod:`foresight`）
        self.foresight: Any = None
        #: 槽的显存那一半 ✓（:class:`_SlotBook` ✓；``None`` ⇒ 没启用 ✓）
        self.slots: Any = None
        #: 这一趟**真在卡上算过**的层数 ✓（``0`` ⇒ 一层都没等到落地 ✓ —— 与"没启用"是两回事 ✓）
        self.usedSlots = 0
        #: 预取真搬上去的权重字节 ✓（这是 ``split`` 模式唯一"搬权重"的开销 ✓，必须报 ✓）
        self.prefetchBytes = 0
        #: 层名 → 换进槽里的那几对（``after`` 拿它换回来 ✓；正常路径下恒为空 ✓）
        self.slotPairs: dict[str, list[tuple[Any, Any]]] = {}

    # -- 报告 --------------------------------------------------------------
    def placement(self) -> dict[str, str]:
        """每个驻留组**现在**在哪台设备上 ✓（装了 hook 之后权重不动 ⇒ split 下恒为原样 ✓）。"""
        found: dict[str, str] = {}
        for group in survey(self.module):
            devices = {str(t.device) for t in _tensors_of(_node(self.module, group.name))}
            found[group.name] = ", ".join(sorted(devices)) or "∅"
        return found

    def report(self) -> dict[str, Any]:
        raw = self.plan.transfer_bytes(self.mode)
        return {
            "mode": self.mode,
            "onGpu": len(self.plan.on_gpu), "onCpu": len(self.plan.on_cpu),
            "layersOnGpu": self.plan.layers_on_gpu, "layersOnCpu": self.plan.layers_on_cpu,
            "gpuGiB": round(self.plan.gpu_bytes / 2**30, 2),
            "cpuGiB": round(self.plan.cpu_bytes / 2**30, 2),
            "hooks": len(self.hooks), "entryHook": self.has_entry_hook,
            "transfers": self.transfers,
            "pinned": self.pinned, "unmoved": list(self.unmoved),
            "perStepTransferBytes": None if raw is None else raw * 2,
            "perStepTransferMiB": None if raw is None else round(raw * 2 / 2**20, 2),
            #: ⭐ 「切分救不了」✓✗ —— 与 ``budget <= 0``（没卡）**分开报** ✓：
            #: 两者都表现为"整份在内存" ✓，可一个是"去插卡/等卡"、一个是"换档位/别装" ✓。
            "refused": self.plan.refused,
            #: 切分**成不成**（``refused`` 只有跟它一起看才完整 ✓：``fits=True`` 是"切了" ✓，
            #: ``fits=False & refused=False`` 是"没卡" ✓，``fits=False & refused=True`` 是"切不动" ✓）
            "fits": self.plan.fits,
            #: 预算分解 ✓（``None`` ⇒ 调用方直接给了一个数、没有分解可看 ✓ 不编 ✗）
            "budget": self.plan.budgetInfo,
            #: ⭐ 内存这一级 ✓（``None`` ＝ **没启用** ✓，不是"装得下" ✗✗ —— 两者都不报卸层 ✓，
            #: 可一个是有上限不触发 ✓、一个是根本没上限 ✓；混在一起读就会误以为内存受控 ✓）
            "hostCache": None if self.hostCache is None else {
                **self.hostCache.snapshot(),
                "spilledNow": sorted(self.spilled),
                "restores": self.restores,
                "spillGiB": round(self.spillBytes / 2**30, 2),
            },
            #: ⭐ 层预取上卡 ✓（``None`` ＝ **没启用** ✓✗ —— 与"启用了但一次没等到落地"分开 ✓：
            #: 后者 ``hits=0`` 但整个字段在 ✓，看一眼就知道是"拷不过来"还是"根本没开" ✓）
            "foresight": None if self.foresight is None else {
                **self.foresight.snapshot(),
                "usedLayers": self.usedSlots,
                "prefetched": 0 if self.slots is None else self.slots.prefetched,
                "prefetchGiB": round(self.prefetchBytes / 2**30, 2),
                "slotGiB": round(0 if self.slots is None else self.slots.bytes / 2**30, 2),
            },
        }

    # -- 拆装 --------------------------------------------------------------
    def detach(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        #: ⭐ 顺手把**重读口**还回去 ✓✗：那是懒开的一把独立句柄 ✓（装载那把已随 `with` 关了 ✓），
        #: 一个 DiT 加一个 TE 就两把 ✓，逐阶段换着装 → 不还就一路堆到进程退出 ✓✗。
        closer = getattr(self.reload, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:  # noqa: BLE001 - 退件不许反过来搞坏退件 ✓✗
                pass
        #: ⭐ 槽里的显存也要还 ✓✗（那是"计划之外"多占的那块地 ✓：不还，退件之后
        #: 报告里说内存已退、卡上却还挂着几层 ✓ ⇒ 下一次装配 OOM 找不到主 ✓）。
        if self.slots is not None:
            try:
                self.slots.close()
            except Exception:  # noqa: BLE001
                pass
        self.slotPairs.clear()

    def restore(self) -> None:
        """按计划复位设备（异常路径用 ✓：hook 在抛异常时**不会**执行 ✗）。"""
        for name in self.plan.on_gpu:
            try:
                node = _node(self.module, name)
            except AttributeError:  # pragma: no cover
                continue
            for tensor in _tensors_of(node):
                if tensor.device != self.main:
                    tensor.data = tensor.data.to(self.main)
        for name in self.plan.on_cpu:
            try:
                node = _node(self.module, name)
            except AttributeError:  # pragma: no cover
                continue
            for tensor in _tensors_of(node):
                if tensor.device != self.cpu:
                    tensor.data = tensor.data.to(self.cpu)
                    self.pinned = _pin(tensor) and self.pinned


def _slots_of(node: Any) -> dict[str, Any]:
    """组内**按名字索引**的张量表 ✓（参数优先于 buffer ✓：同名时参数才是权重 ✓）。

    ⚠️ 用**相对名**（``to_q.weight`` 而不是 ``blocks.7.to_q.weight`` ✓）✗：重读回执由
    :mod:`weights` 按组拼前缀 ✓，两边都用全名会让"前缀剥了几层"变成第二个真源 ✓✗。
    """
    found: dict[str, Any] = {}
    for name, tensor in node.named_buffers(recurse=True):
        found[name] = tensor
    for name, tensor in node.named_parameters(recurse=True):
        found[name] = tensor
    return found


def _release_group(node: Any, torch_mod: Any) -> int:
    """把这一组的参数**真的从内存里放掉** ✓ ⇒ 释放的字节数 ✓。

    ⚠️ 换成 0 元素张量，而不是 ``del`` / ``setattr(None)`` ✗✗：

    * ``del`` 会连**名字**一起没了 ✓ ⇒ 模块结构变了 ✓、``named_parameters()`` 少一项 ✓、
      :func:`_refill_group` 也没地方写回去 ✗；
    * ``setattr(module, name, None)`` 会让前向在**用到它的那一刻**报 ``NoneType is not callable``
      ✓✗（报错指向前向内部，看不出是内存这一级干的 ✓）；
    * 0 元素张量保留名字与结构 ✓，占 0 字节 ✓，且**如果 hook 漏了**（真出错时）会得到一个
      「形状不对」的明确错误 ✓，而不是一段随机数 ✗✗ —— 静默算错是本仓最不能接受的那种错 ✓。
    """
    freed = 0
    for tensor in _slots_of(node).values():
        if tensor.numel() == 0:
            continue
        freed += tensor.numel() * tensor.element_size()
        tensor.data = torch_mod.empty(0, dtype=tensor.dtype, device=tensor.device)
    return freed


def _refill_group(node: Any, values: dict[str, Any], meta: dict[str, Any],
                  torch_mod: Any) -> int:
    """把 :mod:`weights` 重读回来的值**写回**这一组 ✓ ⇒ 写回的张量数 ✓。

    ⚠️ 形状/dtype 必须**先用 attach 时的快照核一遍** ✗✗：0 元素占位已经丢了形状 ✓
    （``tensor.shape == (0,)`` ✓）⇒ 拿它当基准校验等于**用被破坏的东西验破坏** ✓✗。
    """
    restored = 0
    slots = _slots_of(node)
    for name, tensor in slots.items():
        value = values.get(name)
        if value is None:
            continue
        want = meta.get(name)
        if want is not None:
            shape, dtype = want
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"重读回来的 `{name}` 形状 {tuple(value.shape)} ≠ 装载时 {tuple(shape)} ✗"
                    f"（读错了文件 / 键被别名改写 ⇒ 继续算会得到静默错误的结果 ✗）")
            if str(value.dtype) != str(dtype):
                raise ValueError(
                    f"重读回来的 `{name}` dtype {value.dtype} ≠ 装载时 {dtype} ✗"
                    f"（反量化 / convrot 那一步没重放 ⇒ 数值会偏但**不报错** ✗✗）")
        tensor.data = value.to(dtype=tensor.dtype, device=tensor.device)
        restored += 1
    if restored != len(values):
        missing = sorted(set(values) - set(slots))
        raise ValueError(f"重读回执里有 {len(missing)} 个名字在这组里找不到：{missing[:4]} ✗")
    return restored


def _host_prepare(module: Any, plan_: Plan, host_cache: Any,
                  torch_mod: Any) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], int]:
    """把落在 CPU 侧的组过一遍**内存配额** ✓ ⇒ ``(形状快照, 被卸的组, 卸掉的字节)`` ✓。

    ⚠️ 必须在 ``guard``（核内存那一道闸）**之前**调 ✗✗：``guard`` 认的是 ``plan_.cpu_bytes`` ✓，
    可它核的是"整份 CPU 部分放不放得下" ✓ ⇒ 卸完之前就核，等于**拿没卸的数去拒** ✓✗
    （那正是这一级要解决的事 ✓）。
    """
    groups = {group.name: group for group in survey(module)}
    #: ⚠️ **先全部快照、再卸任何一组** ✗✗：卸完只剩 0 元素占位 ✓，那时再 snapshot 得到的是 (0,)
    #: ⇒ 重读回来无从校验 ✓（等于没有校验 ✗）。
    meta: dict[str, dict[str, Any]] = {}
    for name in plan_.on_cpu:
        try:
            node = _node(module, name)
        except AttributeError:  # pragma: no cover
            continue
        meta[name] = {slot: (tuple(tensor.shape), tensor.dtype)
                      for slot, tensor in _slots_of(node).items()}

    spilled: list[str] = []
    freed = 0
    for name in plan_.on_cpu:
        group = groups.get(name)
        if group is None:  # pragma: no cover
            continue
        verdict = host_cache.admit(name, group.nbytes, layers=group.layers)
        victims = list(verdict.victims)
        if verdict.status == host_mod.SPILLED:
            victims.append(name)
        for victim in victims:
            if victim in spilled:
                continue
            try:
                node_v = _node(module, victim)
            except AttributeError:  # pragma: no cover
                continue
            freed += _release_group(node_v, torch_mod)
            spilled.append(victim)
    return meta, tuple(spilled), freed


def _torch() -> Any:
    import torch  # noqa: PLC0415 —— 本仓纪律：模块级不 import torch ✓

    return torch


def attach(module: Any, plan_: Plan, *, main_device: Any, cpu_device: Any = "cpu",
           mode: str = "split", host_cache: Any = None,
           reload: Callable[[str], dict[str, Any]] | None = None,
           host_meta: dict[str, dict[str, Any]] | None = None,
           host_spilled: Iterable[str] = (), host_spill_bytes: int = 0,
           foresight: Any = None) -> Handle:
    """按计划把参数摆好，再挂 hook ✓ —— 返回 :class:`Handle` ✓。

    ``mode="split"``：权重**各自常驻** ✓，过界搬**激活** ✓（CPU 那几层由 CPU 算）；
    ``mode="park"``：权重全留 CPU 锁页 ✓，每层前向前搬上卡、算完搬回 ✓（CPU 不算矩阵乘）。

    ⭐ ``host_cache`` / ``reload``（2026-11-02 加 ✓）＝**内存这一级的上限** ✓✗：

    * 不给（默认 ✓）⇒ 行为与改动前**逐字一致** ✓：整份常驻内存 ✓，不建表、不 snapshot ✗；
    * 给 :class:`host_cache.HostCache` ＋ ``reload`` ⇒ 落在 CPU 侧的组要过配额 ✓，
      装不下的层就地**卸到盘上**（参数换 0 元素占位 ✓），前向进这一层之前先读回来 ✓。

    ⚠️ ``reload(group_name) -> {相对名: 张量}`` 必须返回**最终值**（反量化 / convrot / nvfp4 全做完 ✓）
    ✗✗ —— 把半成品交回来，前向会算出**不报错的错数** ✓（那是本仓最难查的一类 ✓）。
    由 :func:`weights.make_reloader` 提供 ✓（与装配共用同一段还原代码 ✓，不另写一份 ✗）。

    ⭐ ``foresight``（2026-11-02 加 ✓）＝**层预取上卡** ✓✗：落在 CPU 侧的层按**前向顺序**串成一条链 ✓，
    算第 N 层时（CPU 上 ✓）同时在**独立拷流**上把第 N+1 层的权重拷进槽里 ✓；进 N+1 层时 copy
    已落地 ⇒ 那层在**卡上**算 ✓，没落地 ⇒ 照原计划在 CPU 上算 ✓（绝不等 ✗）。只认 ``split`` ＋
    卡上主设备 ✓（``park`` 本来就搬权重 ✓、``off`` 没有卡 ✓，两档都不该再占一份槽的显存 ✓✗）。
    """
    if mode not in WORK_MODES:
        raise ValueError(f"mode 只认 {' / '.join(WORK_MODES)}，拿到 {mode!r}")
    if host_cache is not None and reload is None:
        raise ValueError(
            "给了 host_cache 就必须同时给 reload ✓✗：只记账不重读等于把参数换成 0 元素占位之后"
            "再也不填回来（前向会报形状错 ✗，甚至算错 ✗）")
    torch_mod = _torch()
    main = torch_mod.device(main_device) if not isinstance(main_device, torch_mod.device) \
        else main_device
    cpu = torch_mod.device(cpu_device) if not isinstance(cpu_device, torch_mod.device) \
        else cpu_device
    handle = Handle(module, plan_, mode=mode, main=main, cpu=cpu)

    #: ① 摆参数：split ⇒ GPU 组上卡、CPU 组留内存锁页；park ⇒ **层**留内存锁页。
    #: ⚠️ ``park`` 下唯独 ``‹root›``（裸参数，比如 ``x_pad_token``）**不上 CPU** ✓：它们没有
    #: 自己的 ``forward`` ✓ ⇒ 挂不上搬运 hook ✗，搬下去就是前向报「设备不一致」✗。
    for name in plan_.on_gpu:
        node = _node(module, name)
        parked = mode == "park" and name != "‹root›"
        for tensor in _tensors_of(node):
            want = cpu if parked else main
            if tensor.device != want:
                tensor.data = tensor.data.to(want)
            if tensor.device.type == "cpu":
                handle.pinned = _pin(tensor) and handle.pinned
    for name in plan_.on_cpu:
        try:
            node = _node(module, name)
        except AttributeError:  # pragma: no cover
            continue
        for tensor in _tensors_of(node):
            if tensor.device != cpu:
                tensor.data = tensor.data.to(cpu)
                handle.pinned = _pin(tensor) and handle.pinned

    #: ③ 内存这一级 ✓（``host_cache`` 给了才做 ✓）：落在 CPU 侧的组过一遍配额 ✓，
    #: 装不下的就地卸到盘上 ✓，前向进这一层之前先读回来 ✓（③ 与 ① 的区别：
    #: ① 决定"上不上卡" ✓、③ 决定"内存里留不留" ✓ —— 两级两本账 ✓，不互相顶 ✗）。
    ensure = None
    if host_cache is not None:
        handle.hostCache = host_cache
        if host_meta is None:
            host_meta, host_spilled, spill_bytes = _host_prepare(
                module, plan_, host_cache, torch_mod)
            handle.spillBytes += spill_bytes
        else:
            #: `place()` 已经卸过了 ✓：字节数由它带进来 ✓✗ —— 不带就会得到
            #: 「报告里 spilled 列了两层、spillGiB 却是 0.0」✓（假账 ✓）。
            handle.spillBytes += int(host_spill_bytes)
        handle.hostMeta = dict(host_meta or {})
        handle.spilled.update(host_spilled or ())
        #: ⭐ 三处约定都写在**模块上** ✓✗（2026-11-02 ✓）：``load_module_weights`` 从这里读
        #: "哪些组先别装"（``_residency_spilled`` ✓）与"装完把还原器接哪"（``_residency_reload`` ✓）。
        #: 理由见 :class:`ReloadSlot`：13 处装载调用点，逐个加参数必漏 ✓。
        module.__dict__["_residency_reload"] = reload
        module.__dict__["_residency_spilled"] = tuple(sorted(handle.spilled))
        handle.reload = reload

        def _spill(names: Iterable[str]) -> None:
            for victim in names:
                if victim in handle.spilled:
                    continue
                try:
                    node_v = _node(module, victim)
                except AttributeError:  # pragma: no cover
                    continue
                freed = _release_group(node_v, torch_mod)
                handle.spilled.add(victim)
                handle.spillBytes += freed
            #: ⭐ 真源回填 ✓：`load_module_weights` 读的是**模块上这一个元组** ✓✗
            #: （13 处装载调用点一个都不用改 ✓）—— 卸了哪几层，装载就跳过哪几层 ✓。
            module.__dict__["_residency_spilled"] = tuple(sorted(handle.spilled))

        def ensure(name: str) -> None:
            """这一层要用了 ✓：若它被卸到了盘上，先读回来 ✓（会计可能连带卸别人 ✓）。"""
            if name not in handle.spilled:
                return
            verdict = host_cache.restore(name)
            #: 先按账腾地方 ✓，再真填数据 ✓（顺序反了会瞬时超配额 ✓，且"腾不出来"要早报 ✓）
            _spill(verdict.victims)
            if verdict.status == host_mod.SPILLED:
                raise RuntimeError(
                    f"内存配额太小 ✗：`{name}` 需要的空间卸别的层也腾不出来"
                    f"（配额 {host_cache.quota / 2**30:.2f} GiB、当前 {host_cache.used_bytes / 2**30:.2f} GiB）"
                    f" ⇒ 调大 host_quota，或换更小的档位 ✓")
            values = reload(name)
            _refill_group(_node(module, name), values, handle.hostMeta.get(name, {}), torch_mod)
            handle.spilled.discard(name)
            handle.restores += 1

    #: ⭐ 层预取上卡 ✓（2026-11-02 ✓）：只认「``split`` ＋ 卡上主设备 ＋ 判据说启用」✓✗ ——
    #: 其余情形（``park`` / 没有卡 / ``slots=0``）一律**不建槽** ✓：一块"计划之外"的地都不占 ✓。
    book: Any = None
    nxt_of: dict[str, str | None] = {}
    handle.foresight = foresight
    if foresight is not None and getattr(foresight, "enabled", False):
        if mode != "split":
            foresight.slots = 0
            foresight.disabledReason = f"mode={mode} ⇒ 层预取不启用 ✓（park 本来就逐层搬权重 ✓）"
        elif main.type != "cuda":
            foresight.slots = 0
            foresight.disabledReason = f"主设备 {main} 不是卡 ⇒ 没有「上卡」这回事 ✓"
        else:
            order = [n for n in plan_.on_cpu if is_layer(n) and n not in handle.spilled]
            nxt_of = {a: b for a, b in zip(order, order[1:])}
            if len(order) < 2:
                foresight.slots = 0
                foresight.disabledReason = "CPU 侧的层不到两层 ⇒ 没有「下一层」可预取 ✓"
            else:
                book = _SlotBook(foresight, main, torch_mod)
                handle.slots = book

    #: ``split`` 要两边都有活干才需要 hook ✓；``park`` 只要**有层**就都要搬 ✓（它们的权重
    #: 全被留在内存里了 ⇒ 不挂搬运动 hook 就是「CPU 权重 × 卡上激活」当场报错 ✗）。
    needs_hooks = plan_.needs_split if mode == "split" else bool(plan_.on_gpu)
    #: ⚠️ 卸到盘上的层**必须**有 hook 才能读回来 ✗✗：``off`` 那一档（全在 CPU ⇒ ``needs_split``
    #: 为假 ✓）本来一根 hook 都不用挂 ✓，可一旦内存这一级卸了层 ✓ 就得挂上 ✓ ——
    #: 否则前向到那一层拿到 0 元素占位 ✓、报一个"形状不对"的错 ✓，看着像模型问题 ✗。
    if handle.spilled:
        needs_hooks = True
    if not needs_hooks:
        return handle            # 全在一处 ⇒ 一根 hook 都不挂 ✓（老路一字未变 ✓）

    #: ② 入口兜底：调用方给的是 CPU 张量（最自然的写法 ✓）也要能跑 —— 顶层 pre-hook 把
    #: 输入搬到主设备 ✓。没有它就会在**第一层**报「mat1 is on cpu」（父模块 hook 先于子模块 ✓）。
    #: ⚠️ ``with_kwargs=True`` 不是可选项 ✗✗：H3 主干是 ``trunk(video, audio, text, …,
    #: audio_cond_t=…)`` ✓ —— **关键字张量也是张量** ✓，只搬位置参数会让它在 CPU 层上"卡上的激活 ×
    #: 内存里的权重"当场报设备不一致 ✓✗（而且报错指向层内部，看不出是这个 hook 漏了 ✓✗）。
    def _entry(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
               ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        #: ⭐ 一趟前向的起点 ✓：「本窗口发了几次」从这里清零 ✓✗（层数固定 ⇒ 不清等于只允许第一趟 ✓）
        if handle.foresight is not None:
            handle.foresight.new_window()
        moved = tuple(_move(arg, main, torch_mod) for arg in args)
        return moved, {key: _move(value, main, torch_mod) for key, value in kwargs.items()}

    handle.hooks.append(module.register_forward_pre_hook(_entry, with_kwargs=True))
    handle.has_entry_hook = True

    targets = plan_.on_cpu if mode == "split" else tuple(plan_.on_gpu) + tuple(plan_.on_cpu)
    if handle.spilled:
        #: 被卸的层可能不在上面那份 targets 里（``off`` 档 / ``split`` 下它本来全在 CPU ✓），
        #: 补进来时**不许重复挂** ✓（同一模块两根 before 会搬两次 ✓ ⇒ ``transfers`` 虚高一倍 ✓）。
        targets = tuple(dict.fromkeys(tuple(targets) + tuple(sorted(handle.spilled))))
    for name in targets:
        if name == "‹root›":
            continue             # ⚠️ 裸参数没有 forward ✓ ⇒ 只能靠它所在的模块整体搬 ✓
        try:
            node = _node(module, name)
        except AttributeError:  # pragma: no cover
            handle.unmoved.append(name)
            continue
        if mode == "split":
            handle.hooks.extend(_split_hooks(
                node, main, cpu, torch_mod, handle, name, ensure,
                None if book is None else (book, nxt_of.get(name))))
        else:
            handle.hooks.extend(_park_hooks(node, main, cpu, torch_mod, handle, name, ensure))
    return handle


class _SlotBook:
    """预取槽的**显存**那一半 ✓ —— 判据在 :class:`foresight.Foresight` ✓，这里只管"哪块地、飞没飞完" ✓。

    为什么拆成两个对象 ✗✗：判据是纯逻辑（不 import torch 也能自检 ✓），显存必须碰 torch ✓ ——
    合成一个，``foresight`` 就再也不能在无卡机器上验了 ✓✗（本仓所有判据都要求在无卡环境能跑 ✓）。

    ⚠️ 三条硬约束 ✗✗：

    1. **拷贝没飞完那块显存不许复用** ✓：``Event`` 没 ``query()`` 成真之前写第二批 ⇒ 两批交错落地
       ⇒ 槽里是**不报错的错数** ✓；
    2. **"没落地"绝不等** ✓：只 ``query()``（问一句 ✓），不 ``synchronize()`` / 不 ``wait_stream()`` ✗✗
       （等就是把线程挂在 PCIe 上，比原计划在 CPU 上算还慢 ✓）；
    3. **用过的槽立刻还** ✓：不还，一趟前向里后面每层都抢不到槽 ✓（``busy`` 涨满 ✓）。
    """

    def __init__(self, foresight: Any, main: Any, torch_mod: Any) -> None:
        self.fs = foresight
        self.main = main
        self.torch = torch_mod
        self.stream: Any = None
        try:
            self.stream = torch_mod.cuda.Stream(device=main)
        except Exception as error:  # pragma: no cover - 有卡那一路进不来 ✓
            foresight.slots = 0
            foresight.disabledReason = f"独立拷流建不起来（{error}）⇒ 层预取不启用 ✓"
        #: 组名 → {相对名: 卡上缓冲} ✓（用完就放 ✓：分配器把地留在缓存里给下一层 ✓，不做池化）
        self.tensors: dict[str, dict[str, Any]] = {}
        #: 组名 → (流水号, 落地事件) ✓（**在飞** ✓ —— 没确认完之前不许动那块显存 ✓）
        self.flying: dict[str, tuple[int, Any]] = {}
        self.prefetched = 0
        self.bytes = 0

    def prefetch(self, name: str, node: Any) -> bool:
        """给 ``name`` 发一次拷贝 ✓ ⇒ 真发了没有 ✓（判据不过 ⇒ ``False`` ✓，一组字节都不动 ✓）。"""
        if self.stream is None or name in self.flying or name in self.tensors:
            return False                      # 没有拷流 / 已经装着 / 正在飞 ✓ 都不重复发 ✓
        slots = _slots_of(node)
        if not slots:
            return False
        want = sum(int(t.numel()) * int(t.element_size()) for t in slots.values())
        token = self.fs.request(name, want)
        if token is None:
            return False
        bag: dict[str, Any] = {}
        try:
            #: ⚠️ 拷贝**只**在拷流上发 ✓✗：放计算流上会和这一层的激活抢同一条流 ✓
            #: ⇒ "并行预取"退化成"排队" ✓（白搭一层槽的显存 ✓）。
            with self.torch.cuda.stream(self.stream):
                for rel, tensor in slots.items():
                    buf = self.torch.empty_like(tensor, device=self.main)
                    #: ``non_blocking`` 只在**锁页**内存上真异步 ✓（``split`` 下权重是锁页的 ✓）；
                    #: 非锁页时 torch 退化成同步拷 ✓ —— 慢，但**结果一样** ✓（不因此拒发 ✓）。
                    buf.copy_(tensor, non_blocking=bool(tensor.is_pinned()))
                    bag[rel] = buf
            event = self.torch.cuda.Event()
            event.record(self.stream)
        except Exception as error:  # pragma: no cover - 显存真不够时那一路
            self.fs.landed(token, ok=False)
            self.fs.disabledReason = f"预取拷不上去（{error}）⇒ 这一级不启用 ✓"
            return False
        self.tensors[name] = bag
        self.flying[name] = (token, event)
        self.prefetched += 1
        self.bytes += want
        return True

    def landed(self, name: str) -> bool | None:
        """``name`` 那份拷完了没有 ✓ ⇒ ``True`` 落地 ✓／``False`` 还在飞 ✓／``None`` 没发过 ✓。

        ⚠️ **只问不等** ✓✗（类头第 2 条 ✓）：返回 ``False`` 时调用方照原路在 CPU 上算 ✓，
        槽里那份**留着**（下一趟前向再问一次，那时一般已落地 ✓）—— 丢掉反而白拷 ✓。
        """
        flying = self.flying.get(name)
        if flying is None:
            return None
        token, event = flying
        if not event.query():
            return False
        self.flying.pop(name, None)
        self.fs.landed(token, ok=True)
        return True

    def swap_in(self, name: str, node: Any) -> list[tuple[Any, Any]] | None:
        """把这一层权重**就地换成**槽里那份 ✓ ⇒ ``[(张量, 原 CPU 那份)]``（换回来用 ✓）／``None`` ✓。

        ⚠️ 先整份核一遍再动手 ✗✗：核到一半返回 ``None`` 时前几项**已经换过了** ✓ ⇒ 这层权重一半在卡
        一半在内存 ✓ ⇒ 报一个指向前向内部的设备不一致 ✓（看不出是这个 hook 干的 ✓）。
        """
        bag = self.tensors.get(name)
        if not bag:
            return None
        slots = _slots_of(node)
        if any(rel not in bag for rel in slots):
            return None
        pairs: list[tuple[Any, Any]] = []
        for rel, tensor in slots.items():
            pairs.append((tensor, tensor.data))
            tensor.data = bag[rel]
        return pairs

    def swap_out(self, pairs: list[tuple[Any, Any]]) -> None:
        for tensor, original in pairs:
            tensor.data = original

    def release(self, name: str) -> None:
        """用完／作废 ⇒ 槽腾出来 ✓（显存与判据**两本账一起还** ✓，只还一本就会"判据说空着、
        其实那块地还占着" ✓）。"""
        self.tensors.pop(name, None)
        self.flying.pop(name, None)
        self.fs.release(name)
        self.fs.discard(name)
        self.bytes = sum(
            int(t.numel()) * int(t.element_size())
            for group in self.tensors.values() for t in group.values())

    def close(self) -> None:
        #: ⚠️ 还在飞的拷贝必须**先等它落地**再放显存 ✗✗：拷流还在往那块地上写 ✓，
        #: 这时 free 掉 ⇒ 写进别人的地 ✓（不报错、下一个人在用的显存被神秘改写 ✓）。
        #: 退件路径等一等没关系 ✓（热路径上绝不等 ✓✗ —— 见类头第 2 条 ✓）。
        for _token, event in list(self.flying.values()):
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001 - 退件不许反过来搞坏退件 ✓
                pass
        self.tensors.clear()
        self.flying.clear()
        self.bytes = 0
        self.fs.clear()


def _split_hooks(node: Any, main: Any, cpu: Any, torch_mod: Any,
                 handle: Handle, name: str = "",
                 ensure: Callable[[str], None] | None = None,
                 foresight: tuple[Any, str | None] | None = None) -> list[Any]:
    """``split``：**激活过界** ✓ —— 进 CPU 层前把输入搬到 CPU，算完把输出搬回卡上 ✓。

    ⭐ ``ensure``（2026-11-02 加 ✓）：这一层若被**内存这一级**卸到了盘上，先在搬激活之前读回来 ✓
    ✗✗ 顺序不能反 —— 反了会先拿到一堆 0 元素占位 ✓、再覆盖 ✓，白读一次盘 ✓（且错得看不出来 ✓）。

    ⭐ ``foresight=(槽表, 下一层组名)``（2026-11-02 加 ✓）＝**层预取上卡** ✓✗：
    进来先问"我这层的权重拷上来了没" ✓ —— 落地了就在**卡上**算这一层 ✓（权重换成槽里那份 ✓、
    输入**不往下搬** ✓）；没落地就照原计划搬到 CPU 上算 ✓。参考层那一层永远走第二条 ✓。
    """
    book = foresight[0] if foresight is not None else None
    nxt = foresight[1] if foresight is not None else None

    def before(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
               ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if ensure is not None:
            ensure(name)
            handle.hostCache.touch(name)
        used: list[tuple[Any, Any]] = []
        if book is not None:
            #: ⚠️ 先「确认我这层落地没」再「给下一层发拷贝」✗✗：反了会把**刚腾出来的那条**
            #: 判据通道先用掉 ✓（``take`` 一读就涨 ``pending`` ✓，报告里全是假等待 ✓）。
            book.landed(name)
            if book.fs.take(name) is not None:
                swapped = book.swap_in(name, _node(handle.module, name))
                if swapped is not None:
                    used = swapped
                    handle.prefetchBytes += sum(
                        int(t.numel()) * int(t.element_size()) for t in _tensors_of(node))
                else:
                    #: 判据说"能用"但槽里少一项（组变了名 / 后加的成员 ✓）⇒ 立刻还掉 ✗✗：
                    #: 不还就是每趟都 ``hits++`` 而**一次也用不上** ✓（报告里像个"特别好用"的假象 ✓）
                    book.release(name)
        if nxt is not None:
            try:
                book.prefetch(nxt, _node(handle.module, nxt))
            except Exception as error:  # noqa: BLE001 - 预取是**优化** ✓，它坏不许带崩前向 ✗✗
                #: 但**不许静默** ✗：当场停用并留一句理由 ✓（报告里 ``enabled=False`` ＋ note 就是它 ✓）
                book.fs.slots = 0
                book.fs.disabledReason = (
                    f"给 {nxt} 发预取时出错（{error}）⇒ 这一级当场停用 ✓（前向照原路继续 ✓）")
        if used:
            #: 这层在卡上算 ✓ ⇒ 输入**保持原样** ✓（上一层 ``after`` 已把它搬到 ``main`` ✓）；
            #: 换回来的那一刻由 ``after`` 负责 ✓。
            if _first_on_main(torch_mod, args, kwargs):
                handle.usedSlots += 1
                handle.slotPairs[name] = used
                return args, kwargs
            #: ⚠️ 输不在卡上 ⇒ 权重已经换成卡上那份了 **必须立刻换回来** ✗✗：
            #: 接着往下走会把输入搬到 CPU ✓ ⇒ "卡上权重 × 内存激活"当场报设备不一致 ✓
            #: （报错指向层内部，看不出是这里 ✓）。槽里那份算白拷 ✓（``prefetched`` 照涨 ✓ 不瞒账 ✓）。
            if book is not None:
                book.swap_out(used)
                book.release(name)
            used = []
        moved = tuple(_move(arg, cpu, torch_mod) for arg in args)
        moved_keys = {key: _move(value, cpu, torch_mod) for key, value in kwargs.items()}
        if _crossed(torch_mod, args, moved) or _crossed(torch_mod, kwargs, moved_keys):
            handle.transfers += 1
        return moved, moved_keys

    def after(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        pairs = handle.slotPairs.pop(name, None)
        if pairs:
            if book is not None:
                book.swap_out(pairs)
                book.release(name)
        return _move(output, main, torch_mod)

    return [node.register_forward_pre_hook(before, with_kwargs=True),
            node.register_forward_hook(after)]


def _first_on_main(torch_mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """这一层的输入**是不是已经在卡上** ✓（``split`` 里"在卡上算"的前提 ✓）。

    不在卡上（比如跨阶段换了装 ✓）⇒ 那条捷径作废 ✓、照原路搬到 CPU 算 ✓ ——
    宁可慢一趟，也不拿"卡上权重 × 内存激活"当场报设备不一致 ✓（报错还指向层内部 ✗）。
    """
    for value in tuple(args) + tuple(kwargs.values()):
        if torch_mod.is_tensor(value):
            if value.device.type != "cuda":
                return False
        elif isinstance(value, (tuple, list)):
            for item in value:
                if torch_mod.is_tensor(item) and item.device.type != "cuda":
                    return False
    return True


def _crossed(torch_mod: Any, before: Any, after: Any) -> bool:
    """这一趟**有没有真搬家** ✓（同设备 ⇒ False ✓，报告里 ``transfers=0`` 才说明"老路没动"✓）。"""
    if torch_mod.is_tensor(before):
        return torch_mod.is_tensor(after) and before.device != after.device
    if isinstance(before, (tuple, list)):
        return any(_crossed(torch_mod, old, new) for old, new in zip(before, after))
    if isinstance(before, dict):
        return any(_crossed(torch_mod, before[key], after.get(key)) for key in before)
    return False


def _park_hooks(node: Any, main: Any, cpu: Any, torch_mod: Any,
                handle: Handle, name: str = "",
                ensure: Callable[[str], None] | None = None) -> list[Any]:
    """``park``：**权重过界** ✓ —— 算这一层时权重在卡上，算完立刻搬回内存 ✓（峰值＝单层 ✓）。

    ⚠️ ``tensors`` 在**挂 hook 那一刻**取一次 ✗：内存这一级卸/填用的是 ``tensor.data = …``
    （**就地换数据** ✓，张量对象本身不变 ✓）⇒ 列表里的对象永远有效 ✓。
    改成 ``setattr(module, …)`` 才会让这份列表失效 ✗（:func:`_release_group` 特意没那么做 ✓）。
    """
    tensors = _tensors_of(node)

    def before(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if ensure is not None:
            ensure(name)
            handle.hostCache.touch(name)
        for tensor in tensors:
            if tensor.device != main:
                tensor.data = tensor.data.to(main, non_blocking=tensor.data.is_pinned())
                handle.transfers += 1
        return args

    def after(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        for tensor in tensors:
            if tensor.device != cpu:
                tensor.data = tensor.data.to(cpu, non_blocking=True)
                handle.pinned = _pin(tensor) and handle.pinned
        return output

    return [node.register_forward_pre_hook(before), node.register_forward_hook(after)]


#: 切分**救不了**时怎么办 ✓✗（2026-11-02 加 ✓）
#:
#: * ``"cpu"``（默认 ✓）⇒ 老行为：整份留内存、照样装载 ✓（**报告里必须看得见** ``refused`` ✓）；
#: * ``"raise"`` ⇒ 抛 :class:`BudgetRefused` 中止装载 ✓（"装得上但慢 50 倍"比"装不上"更难查 ✓）。
#:
#: ⚠️ 默认必须是 ``"cpu"`` ✗✗（老调用方一个都没改过 ✓）：切成"拒绝"会让本仓所有
#: 「卡比模型小」的族当场装不起来 ✓✗，而它们现在**能出图**（只是慢 ✓）。
REFUSE_MODES: tuple[str, ...] = ("cpu", "raise")


def place(module: Any, *, main_device: Any, budget: int | None = None,
          cpu_device: Any = "cpu", mode: str = "split", reserve: float | None = None,
          kind: str | None = None, guard: Any = None, fixed_bytes: int = 0,
          on_refuse: str = "cpu", host_cache: Any = None,
          reload: Callable[[str], dict[str, Any]] | None = None,
          foresight: Any = None) -> Handle:
    """一步到位：盘预算（不给就按当前空闲显存算 ✓）⇒ 切分 ⇒ 摆参数 ⇒ 挂 hook ✓。

    返回的 :class:`Handle` 自己带报告 ✓（``handle.report()`` ✓）—— 装载方照抄进
    ``describe()`` / 装载报告即可，**不许**在这里静默降级 ✗。

    ⚠️ ``guard``（2026-11-01 加 ✓）＝**切完、摆之前**的最后一道闸 ✓：收到 :class:`Plan` 一个
    参数 ✓，抛异常即中止装载 ✓✗。本模块**只管切**（显存那一侧 ✓），"切出来的那部分放内存放不放
    得下"由调用方（:mod:`scheduler` 的宿主内存判据 ✓）决定 ✓ —— 判据不在这里再造一套 ✗。

    ⚠️ ``kind`` / ``reserve``（2026-11-01 加 ✓）＝**这件**该留多少余量 ✓✗：给 ``kind``
    （``"dit"`` / ``"te"`` / ``"vae"`` ✓）就按 :data:`RESERVE_BY_KIND` 取 ✓、显式 ``reserve``
    优先 ✓、都不给 ⇒ :data:`DEFAULT_RESERVE` ✓（与改动前逐字一致 ✓）。不传 ``kind`` 的调用方
    **行为零变化** ✓，绝不会因为这次改动多吃或少吃显存 ✗。

    ⭐ ``fixed_bytes`` / ``on_refuse``（2026-11-02 加 ✓）＝把「装不下」拆成**可查的两件事** ✓✗：

    * ``fixed_bytes``：调用方声明"这块地先别用" ✓（例：同一张卡上还要塞 VAE / 解码缓冲 ✓）。
      它**先**从可用显存里扣 ✓，再算余量 ✓ ⇒ 报告里的 ``budget.marginGiB`` 与
      ``budget.fixedGiB`` 能分开看 ✓（"装不下"到底是余量吃掉的、还是你钉死的 ✓）；
    * ``on_refuse``：切分**救不了**时（预算连非层参数都放不下 ✓）⇒ ``"cpu"`` 降级 / ``"raise"``
      中止 ✓，取值见 :data:`REFUSE_MODES` ✓。默认 ``"cpu"`` ⇒ **老调用方零变化** ✓。

    ⚠️ 显式给了 ``budget`` 时 ``fixed_bytes`` **照扣** ✓（不静默忽略 ✗）：那时的
    ``usable`` 就是这个数 ✓，报告里 ``source="explicit"`` ✓ ⇒ 扣在哪一步看得见 ✓。

    ⭐ ``host_cache`` / ``reload``（2026-11-02 加 ✓）＝**内存这一级的配额** ✓✗（细节见
    :func:`attach` ✓）。不给 ⇒ 整份常驻内存 ✓、与改动前逐字一致 ✓。
    给了也不保证"一定会卸" ✓：配额够大时一层都不卸 ✓、``report()["hostCache"]`` 里
    ``evicted=0`` ✓（**这不是"没启用"** ✗✗ —— 没启用时整个字段是 ``None`` ✓）。

    ⭐ ``foresight``（2026-11-02 加 ✓）＝**层预取上卡** ✓✗（细节见 :func:`attach` ✓）。收
    ``None``／``0``（关 ✓，默认）、``int``（槽数 ✓）、``Foresight`` ✓。
    ⚠️ 槽是**计划之外**多占的显存 ✓ ⇒ 这里先从预算里**扣掉**再切分 ✗✗（不扣 ⇒ 计划按整张卡排满 ✓、
    再额外多要一层 ✓ ⇒ 装配不报错、第一次预取 OOM ✓，且报错指向 ``empty_like`` 看不出主 ✓）。
    扣完比预算一半还大 ⇒ **不启用**并写明理由 ✓（为预取把一半以上的层赶去 CPU 是赔本 ✓）。
    """
    if on_refuse not in REFUSE_MODES:
        raise ValueError(f"on_refuse 只认 {' / '.join(REFUSE_MODES)}，拿到 {on_refuse!r}")
    groups = survey(module)
    detail: Budget | None = None
    if budget is None:
        detail = budget_of_cuda(_torch(), reserve=reserve, kind=kind,
                                fixed_bytes=fixed_bytes)
        budget = 0 if detail is None else int(detail.effective)
    elif int(fixed_bytes) > 0:
        #: ⚠️ 显式预算也要能报分解 ✓：这时 ``usable`` ＝调用方给的那个数 ✓、余量是 0 ✓
        #: （余量**已经含在**他的数里 ✓ —— 再扣一遍就是替调用方做决定 ✗）。
        detail = decompose(int(budget), reserve=0.0, kind=kind,
                           fixed_bytes=fixed_bytes, source="explicit")
        budget = int(detail.effective)
    #: ⭐ 层预取的槽要**从预算里先扣** ✓✗（2026-11-02 ✓，理由见 docstring ✓）：扣不动 / 扣完太亏
    #: ⇒ 直接把 ``slots`` 置 0 并留一句理由 ✓（报告里 ``foresight.enabled=False`` ＋ note 就是它 ✓）。
    fs = foresee_mod.of_spec(foresight)
    if fs.enabled:
        layer_max = max((g.nbytes for g in groups if g.is_layer), default=0)
        need = int(fs.slots) * int(layer_max)
        if layer_max <= 0:
            fs.slots = 0
            fs.disabledReason = "盘不出「层」⇒ 没有可预取的单位 ✓"
        elif mode != "split":
            fs.slots = 0
            fs.disabledReason = f"mode={mode} ⇒ 层预取不启用 ✓（park 本来就逐层搬权重 ✓）"
        elif need * 2 > budget:
            fs.slots = 0
            fs.disabledReason = (
                f"{fs.slots} 个槽要 {need / 2**30:.2f} GiB，而预算只有 {budget / 2**30:.2f} GiB ⇒ "
                "为预取把一半以上的层赶去 CPU 是赔本买卖 ✓✗ 不启用 ✓")
        else:
            budget -= need
            fs.reserveBytes = need
    if not fs.enabled and foresight is None:
        #: 调用方**没提**这一级 ⇒ 报告里也必须是 ``None`` ✓✗（"没启用"与"启用了但一次没等到落地"
        #: 是两件事 ✓：前者整个字段为 ``None`` ✓、后者字段在而 ``hits=0`` ✓）
        fs = None
    if budget <= 0:  # 没卡 ⇒ 就是 CPU 单机，如实全留内存 ✓（不假装联合 ✗）
        chosen = Plan(budget=0, on_gpu=(), on_cpu=tuple(g.name for g in groups),
                      gpu_bytes=0, cpu_bytes=sum(g.nbytes for g in groups), fits=True,
                      reason="没有可用显存 ⇒ 整份在内存里（CPU 单机路径 ✓）",
                      layer_bytes=sum(g.nbytes for g in groups if g.is_layer),
                      budgetInfo=None if detail is None else detail.to_dict())
    else:
        chosen = plan(groups, budget, budget_info=None if detail is None else detail.to_dict())
    if chosen.refused and on_refuse == "raise":
        raise BudgetRefused(
            f"切分**救不了**这件 ✓✗：{chosen.reason}"
            f"（预算 {chosen.budget / 2**30:.2f} GiB"
            f"{'' if detail is None else f'，其中余量 {detail.margin / 2**30:.2f} GiB'}"
            "）⇒ 按 `on_refuse=\"raise\"` 中止装载 ✓"
            "（要它跑 ⇒ 换 `on_refuse=\"cpu\"` 整份留内存 ✓，或换更小的档位 ✓）")
    #: ⭐ 内存这一级 ✓（2026-11-02 ✓）：**必须在 guard 之前** ✗✗ ——
    #: guard 核的是 ``plan_.cpu_bytes``（整份 CPU 部分 ✓），先核再卸就等于**拿没卸的数去拒** ✓✗，
    #: 而"卸到盘上换出空间"正是这一级存在的理由 ✓。
    host_meta: dict[str, dict[str, Any]] | None = None
    host_spilled: tuple[str, ...] = ()
    spill_bytes = 0
    if host_cache is not None:
        if reload is None:
            raise ValueError(
                "给了 host_cache 就必须同时给 reload ✓✗（只记账不重读 ⇒ 卸了层的参数是 0 元素占位，"
                "前向会报形状错 ✗ 甚至算错 ✗）")
        #: ⚠️ ``chosen.refused``（切分救不了 ⇒ 整份留内存 ✓）**不跳过**这一级 ✓✗ ——
        #: 那种情形恰恰是内存最紧的时候 ✓（卡上一放不下、内存再放不下就只剩中止 ✓）。
        host_meta, spilled_now, spill_bytes = _host_prepare(module, chosen, host_cache, _torch())
        if spilled_now:
            host_spilled = spilled_now
            chosen = replace(
                chosen, cpu_bytes=max(0, int(chosen.cpu_bytes) - spill_bytes),
                spilled=tuple(spilled_now), spillBytes=int(spill_bytes),
                reason=(chosen.reason +
                        f"；内存这一级卸了 {len(spilled_now)} 组（{spill_bytes / 2**30:.2f} GiB）到盘上 ✓"))
    if guard is not None:
        guard(chosen)
    return attach(module, chosen, main_device=main_device, cpu_device=cpu_device, mode=mode,
                  host_cache=host_cache, reload=reload, host_meta=host_meta,
                  host_spilled=host_spilled, host_spill_bytes=spill_bytes, foresight=fs)
