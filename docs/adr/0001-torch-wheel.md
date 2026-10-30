# ADR 0001 —— CUDA 轮子（E0.3）

- 状态：**待创作者裁定**（事实前提已被实测推翻 ⇒ 请只裁 §3 的两个选项）
- 日期：2026-10-30
- 裁定人：创作者
- 相关：`docs/env/runtime-versions.json`、`docs/compatibility-matrix.md`、`app/services/engine/facade.py`（`limits` / `knownGaps`）、`tests/engine_env_docs_test.py`

## 1. 这份 ADR 要判什么

WBS **E0.3** 的原文是「**torch 轮子决策（阻塞）**」，退出条件写的是「二选一均算过：

1. 换轮子后 `torch.cuda.is_available() = True`；
2. ADR 写明「本阶段接受 CPU 轮子」，并把 `capabilities.limits` 补一条」。

这条的前提是「装进来的是 **CPU 轮子**」—— 那个前提**已经不成立** ✓✗。所以本 ADR 不再论证
「换不换轮子」，改为把 **今天的实测事实** 落盘，并把**真正还需要人裁的那一点**（§3）摊开。

## 2. 实测事实（本机，2026-10-30）

| 项 | 实测值 | 怎么来的 |
|---|---|---|
| Python | `3.12.10` | `platform.python_version()` |
| torch | `2.14.1+cu130` | `torch.__version__`（**带 `+cu130` ⇒ 不是 CPU 轮子** ✓） |
| CUDA 运行时 | `13.0` | `torch.version.cuda` |
| cuDNN | `92400`（9.24.0） | `torch.backends.cudnn.version()` |
| `torch.cuda.is_available()` | **`True`** | 直接调用（E0.3 选项①的退出条件**已满足** ✓） |
| GPU | `NVIDIA RTX A5000`，算力 **8.6**（sm_86），`23028 MiB` 可用，驱动 `610.88` | `torch.cuda.get_device_properties(0)` |
| `pip check` | `No broken requirements found.` | `pip check` |

落盘位置：`docs/env/runtime-versions.json`（版本段）＋ `docs/env/hardware-report.json`（能力段，＝"`facade.capabilities()` 现场重生成"）。
两份都由 `tests/engine_env_docs_test.py`（**18/18**）逐项与**现场**对账 —— 改代码忘了刷文件即红 ✓。

## 3. 真正待裁的两点（都不是「CPU 轮子」）

### 3.1 sm_86 没有 fp8 硬件加速

`sm_86 < sm_89` ⇒ fp8 只能是 **emulated**：fp8 权重**装得下**（这是 H3 主 DiT 的必需 ✓），但**不加速** ✓✗。
`runtime-versions.json` 的 `knownGaps`
（`id: fp8-no-hw`、`severity: info`）已按这个口径落盘，`fix` 写的是「属硬件边界，不改」。

### 3.2 24 GiB 装不下 H3（这一条与轮子无关，但同属 E0 的边界）

`facade.readiness("h3").vram` 的实测估算（权重精确 ✓、激活由调用方给）：

```
weightsGiB 39.55   estimatedGiB 49.44   capacityGiB 24.0   fits false   headroomGiB -25.44
```

即：**换了轮子也装不进去** ✗。这与 E0.3 的「换轮子」是两件事，别混着记 ✓
（见 §5「未落」与方案 §C）。

## 4. 两个选项与各自代价

**选项 A（保持现状）**：以 `torch 2.14.1+cu130` ＋ sm_86 ＋ 24 GiB 为基线，把「fp8 emulated」与
「H3 单卡装不下」写进 `capabilities().limits` / `knownGaps`（**现状已经是这样** ✓）。
代价：所有性能数字都只能标注「A5000 / emulated fp8 / 24 GiB」，不能外推到 sm_89+ 或大显存机器 ✗。
产出：无代码改动；E0.3 关闭，`R-E01`（CPU 轮子长期不换）可**解除** ✓。

**选项 B（换硬件或换更匹配的轮子）**：目标是 sm_89+（fp8 硬件加速）或更大显存（≥ 51 GiB 才容得下
§3.2 的估算 ＋ 安全系数）。
代价：不是本仓能自决的成本（整机采购 / 租用）✗；且**轮子本身已经是对的** ✓ ⇒ 换轮子并不解决任何
当前问题，只有换硬件才解决 ✗。

> 两个选项的差别只在「是否为本阶段的能力边界签字」；**代码与事实文件目前都按 A 写着** ✓
> —— 裁 B 时改动面是 `docs/env/*` 与 `capabilities.limits`，不涉及引擎逻辑 ✓。

## 5. 与代码的接口（裁定后要动哪儿）

| 位置 | 现在写的是什么 | 裁 B 时要改的 |
|---|---|---|
| `facade.capabilities()["limits"]` | 「只做单机单卡」「显存配额是预算控制」「量化按族判…」 | 追加/替换硬件相关的一句 |
| `_known_gaps(status)` | 空数组（＝这台机器当前无既知**前提**缺口） | 若换机器，`fp8-no-hw` 那条应消失 |
| `docs/env/runtime-versions.json` | `knownGaps`：`fp8-no-hw` / `mxfp4` / `no-baseline` | 换硬件后重采 |
| `docs/env/hardware-report.json` | ＝`capabilities()` 现场重生成 | 重生成 |

## 6. 仍未落（别当成已完成）

1. **H3 端到端未跑通**：24 GiB 装不下（§3.2），且本仓 32B Qwen3-VL 文本编码器**尚无模型骨架**
   ⇒ `nvfp4` 虽已实现并能逐层还原出正确权重（ADR 0002），仍没有挂载点 ✗。
2. `generate` 无性能基线、无 60 分钟稳定性（E3.5 / E5.2）—— `unverified` 两栏就写着这两条 ✓。
3. 本 ADR 记的是**这台机器**的事实；换机后 `tests/engine_env_docs_test.py` 会当场变红 ✓
   （这是故意的：文档没有判据就一定烂 ✓）。
