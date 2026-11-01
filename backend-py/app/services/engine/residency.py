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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

#: 给激活/工作区留的显存比例 ✓（72×128 那种长序列的激活能到十几 GiB ✗ ⇒ 不能只留 5%）
DEFAULT_RESERVE = 0.25
#: 卡上可用空间低于这个数就不值得切 ✗（连几个 embedding 都放不下 ⇒ 别假装联合）
MIN_GPU_BYTES = 256 * 1024 * 1024

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


def plan(groups: Iterable[Group], budget: int, *, reason: str = "") -> Plan:
    """按预算切分：**非层参数先上卡**，然后层按前向顺序依次上卡，装不下就停。

    ⚠️ 为什么「依次上卡」而不是「挑小的塞」✗：层是**串行**用的 ✓ —— 卡上若是 0,1,2,5,6 这种
    跳号 ✗，激活每步都要过界好几次 ✓✗；保持前缀连续 ⇒ 整个前向**只过一次界** ✓。
    """
    items = list(groups)
    layer_bytes = sum(g.nbytes for g in items if g.is_layer)
    if budget <= 0:
        return Plan(budget=budget, on_gpu=(), on_cpu=tuple(g.name for g in items),
                    gpu_bytes=0, cpu_bytes=sum(g.nbytes for g in items), fits=False,
                    reason="预算 ≤ 0：没有可用的显存 ⇒ 只能整份放 CPU ✗",
                    layer_bytes=layer_bytes)
    non_layers = [g for g in items if not g.is_layer]
    layers = [g for g in items if g.is_layer]
    used = 0
    on_gpu: list[str] = []
    for group in non_layers:
        used += group.nbytes
        on_gpu.append(group.name)
    if used > budget:
        return Plan(budget=budget, on_gpu=(), on_cpu=tuple(g.name for g in items),
                    gpu_bytes=0, cpu_bytes=sum(g.nbytes for g in items), fits=False,
                    reason=(f"连非层参数（{used / 2**30:.2f} GiB）都超预算"
                            f"（{budget / 2**30:.2f} GiB）⇒ 切分救不了 ✗"),
                    layer_bytes=layer_bytes)
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
                layer_bytes=layer_bytes)


def budget_from_cuda(torch_mod: Any, *, reserve: float = DEFAULT_RESERVE,
                     index: int | None = None) -> int:
    """按「当前可用显存 × (1 − 余量)」算预算 ✓；没卡/取不到 ⇒ ``0`` ✓（不猜 ✗）。

    ⚠️ 「可用」≠ ``mem_get_info`` 的 free ✓✗（2026-11-01 实测 ✓，与 `device_policy.probe` 同口径 ✓）：
    本进程 allocator 里**留着没还的缓存块**在驱动眼里是"已占" ✓，可它们随时能复用 ✓
    ⇒ 必须加进来 ✓；不加的结果是预算被凭空砍掉十几 GiB ✓（实测：退掉一件之后 `reserved` 比
    ``allocated`` 多 14.3 GiB ✓ ⇒ 25.28 GiB 的塔只被允许上卡 3.5 GiB ✓✗）。
    """
    if not getattr(torch_mod.cuda, "is_available", lambda: False)():
        return 0
    try:
        free, _total = torch_mod.cuda.mem_get_info(index)
        reusable = max(0, int(torch_mod.cuda.memory_reserved(index))
                       - int(torch_mod.cuda.memory_allocated(index)))
    except Exception:  # pragma: no cover - 老驱动 / 不支持的设备
        return 0
    return max(0, int((int(free) + reusable) * max(0.0, 1.0 - float(reserve))))


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
        }

    # -- 拆装 --------------------------------------------------------------
    def detach(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

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


def _torch() -> Any:
    import torch  # noqa: PLC0415 —— 本仓纪律：模块级不 import torch ✓

    return torch


def attach(module: Any, plan_: Plan, *, main_device: Any, cpu_device: Any = "cpu",
           mode: str = "split") -> Handle:
    """按计划把参数摆好，再挂 hook ✓ —— 返回 :class:`Handle` ✓。

    ``mode="split"``：权重**各自常驻** ✓，过界搬**激活** ✓（CPU 那几层由 CPU 算）；
    ``mode="park"``：权重全留 CPU 锁页 ✓，每层前向前搬上卡、算完搬回 ✓（CPU 不算矩阵乘）。
    """
    if mode not in WORK_MODES:
        raise ValueError(f"mode 只认 {' / '.join(WORK_MODES)}，拿到 {mode!r}")
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

    #: ``split`` 要两边都有活干才需要 hook ✓；``park`` 只要**有层**就都要搬 ✓（它们的权重
    #: 全被留在内存里了 ⇒ 不挂搬运动 hook 就是「CPU 权重 × 卡上激活」当场报错 ✗）。
    needs_hooks = plan_.needs_split if mode == "split" else bool(plan_.on_gpu)
    if not needs_hooks:
        return handle            # 全在一处 ⇒ 一根 hook 都不挂 ✓（老路一字未变 ✓）

    #: ② 入口兜底：调用方给的是 CPU 张量（最自然的写法 ✓）也要能跑 —— 顶层 pre-hook 把
    #: 输入搬到主设备 ✓。没有它就会在**第一层**报「mat1 is on cpu」（父模块 hook 先于子模块 ✓）。
    #: ⚠️ ``with_kwargs=True`` 不是可选项 ✗✗：H3 主干是 ``trunk(video, audio, text, …,
    #: audio_cond_t=…)`` ✓ —— **关键字张量也是张量** ✓，只搬位置参数会让它在 CPU 层上"卡上的激活 ×
    #: 内存里的权重"当场报设备不一致 ✓✗（而且报错指向层内部，看不出是这个 hook 漏了 ✓✗）。
    def _entry(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
               ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        moved = tuple(_move(arg, main, torch_mod) for arg in args)
        return moved, {key: _move(value, main, torch_mod) for key, value in kwargs.items()}

    handle.hooks.append(module.register_forward_pre_hook(_entry, with_kwargs=True))
    handle.has_entry_hook = True

    targets = plan_.on_cpu if mode == "split" else tuple(plan_.on_gpu) + tuple(plan_.on_cpu)
    for name in targets:
        if name == "‹root›":
            continue             # ⚠️ 裸参数没有 forward ✓ ⇒ 只能靠它所在的模块整体搬 ✓
        try:
            node = _node(module, name)
        except AttributeError:  # pragma: no cover
            handle.unmoved.append(name)
            continue
        if mode == "split":
            handle.hooks.extend(_split_hooks(node, main, cpu, torch_mod, handle))
        else:
            handle.hooks.extend(_park_hooks(node, main, cpu, torch_mod, handle))
    return handle


def _split_hooks(node: Any, main: Any, cpu: Any, torch_mod: Any,
                 handle: Handle) -> list[Any]:
    """``split``：**激活过界** ✓ —— 进 CPU 层前把输入搬到 CPU，算完把输出搬回卡上 ✓。"""
    def before(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
               ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        moved = tuple(_move(arg, cpu, torch_mod) for arg in args)
        moved_keys = {key: _move(value, cpu, torch_mod) for key, value in kwargs.items()}
        if _crossed(torch_mod, args, moved) or _crossed(torch_mod, kwargs, moved_keys):
            handle.transfers += 1
        return moved, moved_keys

    def after(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        return _move(output, main, torch_mod)

    return [node.register_forward_pre_hook(before, with_kwargs=True),
            node.register_forward_hook(after)]


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
                handle: Handle) -> list[Any]:
    """``park``：**权重过界** ✓ —— 算这一层时权重在卡上，算完立刻搬回内存 ✓（峰值＝单层 ✓）。"""
    tensors = _tensors_of(node)

    def before(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
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


def place(module: Any, *, main_device: Any, budget: int | None = None,
          cpu_device: Any = "cpu", mode: str = "split", reserve: float = DEFAULT_RESERVE,
          guard: Any = None) -> Handle:
    """一步到位：盘预算（不给就按当前空闲显存算 ✓）⇒ 切分 ⇒ 摆参数 ⇒ 挂 hook ✓。

    返回的 :class:`Handle` 自己带报告 ✓（``handle.report()`` ✓）—— 装载方照抄进
    ``describe()`` / 装载报告即可，**不许**在这里静默降级 ✗。

    ⚠️ ``guard``（2026-11-01 加 ✓）＝**切完、摆之前**的最后一道闸 ✓：收到 :class:`Plan` 一个
    参数 ✓，抛异常即中止装载 ✓✗。本模块**只管切**（显存那一侧 ✓），"切出来的那部分放内存放不放
    得下"由调用方（:mod:`scheduler` 的宿主内存判据 ✓）决定 ✓ —— 判据不在这里再造一套 ✗。
    """
    groups = survey(module)
    if budget is None:
        budget = budget_from_cuda(_torch(), reserve=reserve)
    if budget <= 0:  # 没卡 ⇒ 就是 CPU 单机，如实全留内存 ✓（不假装联合 ✗）
        chosen = Plan(budget=0, on_gpu=(), on_cpu=tuple(g.name for g in groups),
                      gpu_bytes=0, cpu_bytes=sum(g.nbytes for g in groups), fits=True,
                      reason="没有可用显存 ⇒ 整份在内存里（CPU 单机路径 ✓）",
                      layer_bytes=sum(g.nbytes for g in groups if g.is_layer))
    else:
        chosen = plan(groups, budget)
    if guard is not None:
        guard(chosen)
    return attach(module, chosen, main_device=main_device, cpu_device=cpu_device, mode=mode)
