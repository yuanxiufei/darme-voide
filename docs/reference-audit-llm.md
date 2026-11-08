# LLM 低显存 / 低内存推理 · 参考审计（LLM 侧）

> 落盘日 2026-11-08。七仓在盘上位置：`D:\code\voide\_audit-lowvram\`（不在本仓 `reference/` 下）。
> 审计那一轮按创作者要求**未下载任何模型权重**，因此 §1–§6 一切「占用」均为源码预算逻辑或仓内 bench
> 记录，**不是**本机运行实测。凡属作者声明一律标注「声明」，不写成实测结论。
> ⚠️ **§7.x 例外**：那一节是落地时在 A5000 上**真权重跑出来的**（`§7.1.1`/`§7.2.1`/`§7.3.1`，
> 探针在 `data/experiments/`），每条都注明模型文件与日期，与上文「不是实测」不矛盾 —— 说的是两批事。

## 0. 为什么另起一份

`docs/reference-audit.md` 做了三轮审计，但落点定在 **DiT 扩散**（本仓主力），因此几个 LLM/MoE 引擎被判
「领域错位、无可搬」。那个判决对 DiT 成立，对本仓的 **自研 LLM 推理**（`engine/llm.py` +
`gguf_to_llm.py`，用于顶替 ollama 的本地文本生成）不成立——落点一换，同一批仓从「错位」变成「正对口」。

本文件只判 **LLM 侧**适用性，不推翻 `reference-audit.md` 的任何 DiT 结论。

## 1. 核实表

| 仓 | 真实存在 | HEAD | 许可 | 平台（README/构建文件） | 宣传数字（作者口径） |
|---|---|---|---|---|---|
| Strata | ✓ remote 一致 | `54828c7` 2026-11-06 | **MIT** | Windows + Linux，CUDA / SYCL / Metal | 1.75 bpw 盘占 239–245 GB（DeepSeek 档） |
| airllm | ✓ | `0f824d9` 2026-11-06 | **Apache-2.0** | 全平台（纯 Python + torch） | 70B 跑在单张 **4 GB** 卡 |
| Edge0 | ✓ | `8a52b68` 2026-11-04 | Apache-2.0 | Windows=C++ **Vulkan**；macOS=MLX | `edge0-35b` 峰值 **2.9 GiB**（Mac mini M4 Pro 24 GB） |
| ds4 | ✓ | `b3f5018` 2026-11-03 | MIT | Metal 主；CUDA（**sm89+ / DGX Spark**）；ROCm | Mac 64 / 96 / 128 GB；双 128 GB RDMA |
| pulsar | ✓ | `18f9900` 2026-09-01 | **CUSTOM（非 OSI）** | Linux + CUDA（Rust） | RTX 5060 Ti 16 GB ×2 + 30 GB RAM |
| expert-stream | ✓ | `89275f1` 2026-08-16 | MIT | **Windows-only** | 284B 跑在 16 GB RAM 笔记本（UD-IQ1_S 82.5 GB） |
| moe-stream | ✓ | `b3024b3` 2026-02-23 | Apache-2.0 | **macOS Metal / Linux CPU-only；CUDA planned** | 80B 跑在 24 GB Apple Silicon |

两条许可红线：`pulsar` 是**定制许可**（不可当 MIT 用，要引必须逐条读）；`moe-stream` 只有 `LICENSE-APACHE`。

## 2. 七项研究重点 × 七仓

「✓」= 有实现且能指到源码；「—」= 没有。

| 研究重点 | Strata | airllm | Edge0 | ds4 | pulsar | expert-stream | moe-stream |
|---|---|---|---|---|---|---|---|
| 1 SSD Streaming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓（最强） | ✓ |
| 2 MoE Expert Streaming | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 Layer-wise Loading | ✓ | ✓（方法论原型） | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 CPU/GPU Offloading | ✓ | ✓ | ✓ | ✓ | ✓ | —（纯 CPU） | —（统一内存） |
| 5 Predictive Prefetch | ✓ | ✓（确定性，非预测） | ✓ | ✓ | ✓ | 部分（离线仿真） | ✓ |
| 6 **KV Cache Quantization** | ✓ | — | — | ✓ | ✓（移植 ds4） | — | — |
| 7 **Speculative Decoding** | ✓ | — | —（`spec.py` 是路由规格，不是投机） | ✓（最完整） | — | — | — |

**第 6、7 两项是前三轮审计完全没覆盖的**，也是本仓 LLM 侧的真空白。

## 3. 逐项：证据与本仓可学点

### 3.1 Layer-wise / CPU-GPU offload

- **airllm** 是这个方法论的原始原型：`utils.py::split_and_save_layers()` 把 HF checkpoint 按
  `embed / layers.N / norm / lm_head` 切成独立分片落盘，`load_layer()` 一次只读一层。
  **但本仓不需要它的落盘切分**——GGUF 本身就是随机访问的，`gguf_to_llm.py` 已经是逐张量
  反量化 → `copy_` 进参数。它的价值在别处：`airllm_base.py::load_layer_to_cpu` 里的
  `max_pinned_layer_bytes` 口径——**锁页内存不可换出，所以只在层足够小时才 pin**
  （注释原文：frontier MoE 单层 ~17 GB，预取两层会锁死 ~34 GB RAM）。
  本仓 `residency` 目前对锁页是「一律」，没有按层量级分档。
- **Strata `plan/plan.hpp`**：放置是**算术结果**，不是可配置选择；算不平当场 REFUSE，不许静默超额。
  更强的一条纪律：手抄的期望值只用于**校验**，与几何推导不一致时**几何赢并报差异**。
- **ds4 `ds4_layer_pack.c`**：层放置 = 单调连续前缀 + 溢出到 CPU。

本仓落点：`llm.resolve_device()` 目前只有 `cuda`/`cpu` 二选一 ⇒ 换成「按显存预算放连续前缀」。

### 3.2 KV cache 量化

- **Strata `kernels/kv_q8.hpp`**：INT8 KV，grouped-64（每 64 元素一个 fp16 scale）。
- **Strata `kernels/kv_q4.hpp`**：Q4_0 + **256 点 Walsh-Hadamard 旋转**。要点全在文件头注释里：
  4 bit 直接量化会炸在离群通道；先用正交 H 旋转把离群摊到 256 维才可用；query 同样旋转 ⇒
  `<Hq,Hk> = <q,k>` 分数不变；注意力输出（旋转值混合）再乘 H 转回（H 自逆）。
  体积：**576 B/cell vs INT8 的 1056 B**。落差只来自 4 bit 舍入本身，且**仓内有实测记录**
  （`bench/results/2026-09-27-kv-q4`）。
- **Strata `kernels/kv_stream.hpp`**：codes 放 host 副本、按块驻留槽位 ⇒ 等于「KV 换出到 CPU」。
- **ds4 `dsv4_fp8_kv_quantize_row_inplace_cpu`**（被 pulsar `dsv4_kernels.inc:361` 逐段移植）：
  每行前 `head_dim-n_rot` 维走 e4m3fn，64 宽块，scale 取 `ceil(log2(amax/448))` 的**2 的幂**，
  钳 ±448。2 的幂 scale 是为了让反量化变成纯位移/乘法，不吃额外乘法精度。

本仓落点：`llm.py` 的 KV 现在是每步 `torch.cat` 拼接，不量化、不换出、无上限。
**先做 q8 grouped-64**（改动局限在 cache 读写两处），q4+旋转作为第二步。

### 3.3 投机解码

- **ds4 最完整**：`ds4_session_eval_speculative()`、`mtp_draft_tokens`、`dspark_confidence_threshold`。
  最关键的一行口径：默认只对**贪心**验证 draft token，而 `dspark_exact_sampling` 走**精确 p/q 接受**
  ——**温度 >0 时贪心验证的分布与目标模型不等价**，这是投机解码最常见的正确性分叉。
  TP 下还有 DRAFT/MAINTAIN 的 lockstep 协调。
- **Strata `spec/spec_start.hpp`** 记了一个「静默失效」教训：`0` 同时被当作「不做投机」的哨兵和
  合法起始位置 ⇒ 单 token prompt **从不**触发投机，改成 `-1` 才修好。这类 bug 不报错、只是慢。

本仓现状：`llm.py` 无 draft 路径。**但本仓没有可用的 draft model**（LLM 只有主模型），
所以这一项属「算法可学、真通路暂无收益」。

### 3.4 预测预取

只有 **airllm 之外的仓**才需要「预测」：LLM 是逐层顺序的，下一层是什么**是确定的**，
`airllm` 的 prefetch 就是这个确定性预读。Strata 的 `foresight_swap`、Edge0 的 `prerouter`
（`PrerouterSpec`：训练的跨 token 路由预测器）针对的是 MoE **专家**不确定性。
本仓 LLM 是 dense（Qwen3/Llama 架构）⇒ 预取按「下一层」确定性做即可，不需要预测器。

### 3.5 跨调用 KV 复用（本仓另一处空白）

- **Strata `core/conversation_memory.hpp` + `core/session_save_reclaim.hpp`**：
  会话快照落盘前先过**预算准入** `conversation_memory_admit(available, allocation, floor)`，
  不够就先回收 cache，仍不够则不保存。`ple_residency.hpp` 复用同一个准入函数。

本仓 LLM 每次调用都从零 prefill，对话与提示词扩写场景有大量重复前缀可省。

## 4. 对 A5000 24 GB + 96 GB RAM + NVMe 的适配

| 仓 | Windows 11 | CUDA | A5000 判定 |
|---|---|---|---|
| Strata | ✓（README 列 RTX 30/40/50 Windows+Linux） | ✓ | **可用**：A5000 未在它档位表（写的是 3060 12G / 5070 12G），属上档但**未实测** |
| airllm | ✓ | ✓ | 可用但慢（逐层流式 0.1–0.4 tok/s 级），只当方法论参考 |
| Edge0 | ✓ 但走 **Vulkan** | ✗（Windows 侧不吃 CUDA） | 可跑，性能未知；它的 2.9 GiB 是 **MLX allocator 峰值**，不能搬到 CUDA 上指望 |
| ds4 | ✓（C 单文件） | ✓ 但目标 **sm89+** | ⚠️ **A5000 是 sm86，不在其 CUDA 支持列表**；算法可读，跑通无据 |
| pulsar | 需 WSL2（Rust+CUDA） | ✓ | 参考机是 2×16 GB，单卡 24 GB 未验证；**许可为定制，引用需先读** |
| expert-stream | ✓ **原生** | 不需要（CPU-only） | 能跑但**完全不用 A5000**；IQ1_S 量化质量代价极大 |
| moe-stream | ✗（Metal / Linux CPU-only） | ✗ planned | **Windows 不可用**，排除 |

## 5. 宣传 vs 可复核实测

| 仓 | 仓内 bench 原始数据 | 判定 |
|---|---|---|
| Strata | **1523 份**（`bench/results/`，含 2026-09-24-final 的 matrix/estimates） | 最强的可复核性：有原始矩阵，可复算 |
| expert-stream | 68 份 CSV（`bench/results/`） | 有原始数据，但模型是 UD-IQ1_S（1 bit 级，质量不可用于生产） |
| ds4 | 有 `bench/` 目录，无 results 文件 | 只有方法，无留档数据 |
| airllm / Edge0 / pulsar / moe-stream | 无 | **只有 README 声明，不得当实测引用** |

本机实测：**零**。本轮按约束未下载权重，所有速度/占用数字都是上述来源，不是 A5000 上的观测。

## 6. 分级

- **可用（源码与平台都对得上）**：`Strata`（MIT、Windows、1523 份实测、七项里占六项）。
- **实验性（能读能跑，收益不确定）**：`ds4`（投机解码与 fp8 KV 的实现最值得读，但 CUDA 目标是 sm89+）、
  `pulsar`（KV 量化移植自 ds4，但许可与单卡验证都缺）。
- **仅研究价值（平台或质量不匹配）**：`airllm`（Tok/s 太低，只取 `max_pinned_layer_bytes` 口径）、
  `Edge0`（Windows 侧 Vulkan 且实测在 Mac）、`expert-stream`（CPU-only、1 bit 量化）、
  `moe-stream`（Windows 不可用）。

## 7. 落到本仓 LLM 的计划（按「真通路收益」排序）

1. **KV cache 量化（q8 grouped-64）**——本仓 KV 现在 `torch.cat` 拼接、无上限。改动局限在
   `llm.py` cache 的写入/读取两处，长上下文显存线性下降。参考 `Strata/kernels/kv_q8.hpp`。
   **✅ 2026-11-08 已完成**（详见 §7.2）。
2. **分层放置接进 `build_llm`**——`resolve_device` 从二选一变「按显存预算放连续前缀 + 溢出 CPU」，
   让 14B+ 能上卡。`gguf_to_llm.py` 已是逐张量，接口成本低。参考 `Strata/plan/plan.hpp` 的
   「算术放置 + 算不平即 REFUSE」。**✅ 2026-11-08 已完成**（详见 §7.1）。
3. **跨调用 KV 复用（前缀缓存）**——对话/扩写有重复前缀。参考 `conversation_memory.hpp` 的
   预算准入写法：**装不下就不装，不许静默超额**。
   **✅ 2026-11-08 已完成**（详见 §7.3）。
4. **投机解码**——算法口径（温度 >0 必须精确 p/q，不能贪心验证）记下，本仓无 draft model。
   **✅ 2026-11-08 已落地为「默认关的可选开关」**（不是改判前面的收益账 ✗ —— 账没变：窗口成本比
   `ratio(T=8)=1.01×`、回本门槛 0.01 个 token/轮、真实场景接受率 0.01～0.02 个/轮 ⇒ **收益 1.02~1.03×**）。
   改判的是**代价**：上一稿把「加一条 KV 回滚路径」算成门槛，而 `truncate_kv` 早在 ③ 就有了 ⇒
   真实代价只是把「一拍一 token」改成「一拍 1+T」（纯结构改动）。落点 `llm.LookupDrafter` ＋
   `generate_llm(spec_lookup=...)`、开关 `specLookup/specWindow/specMinMatch`、回执 `describe()["spec"]`；
   只在 `temperature=0` 启用，温度 >0 **如实退回**（回执写原因，不假装开了）。详见 §7.4.3。
5. **不做**：SSD 流式专项（GGUF 已逐张量读盘、NVMe 顺序读够用）、`airllm` 式分片落盘
   （本仓无需预先切分 checkpoint）。

### 7.1 落地实况：分层放置（②）

落点与口径（全部有判据守着，`tests/engine_llm_device_test.py` ⑦/⑧/⑨ 三组）：

| 层 | 落点 | 口径 |
|---|---|---|
| 算术 | `llm.plan_layer_devices` | 每层字节＝`layer_weights_bytes(config, dtype_bytes)`（＝`_block_param_count × dtype_bytes`，**算**不抄表）× **连续前缀**；一层都放不上 ⇒ `gpuLayers=0` + `why` 写原因；`want_gpu_layers` 是**下限**、不够时报 `shortfall`（不悄悄减配） |
| 执行 | `llm.apply_layer_plan` | 逐层 `.to()` 后**从参数读回**实际设备（`"cuda"` 与 `"cuda:0"` 不是同一个 `torch.device`，照抄请求值会让 `forward` 里比较恒真 ⇒ 每层白搬）；`embed`/`norm`/`lm_head` **钉末层**（tie 时钉两处会把同一个 `Parameter` 拆成两份） |
| 前向 | `LlmModel.forward` | 只在**层边界**把激活 `.to(target)`（每步一份 MB 级激活 ⇒ 这才是 split 与 `park` 档搬全部层权重的区别）；RoPE 按设备各缓存一份，只搬一次；KV 各留**自己那层**的设备 |
| 装载 | `gguf_to_llm.load_llm_from_gguf(..., layer_devices=)` | 流式逐张量 `copy_` 跨设备一次 ⇒ 不产生「先整份装一处再搬」的第二份权重 |
| 配置 | `settings.gpuLayers`（`llm_backend._layer_placement`，`text_generation.engine_gpu_layers` 单一真源） | `off`/空＝不分层（老路径逐字节相同）；`0`＝全 CPU（回执写明**显式要求**）；`N`＝前 N 层上卡（超过层数、没有卡、放不下 ⇒ **具名拒**，不截到能装为止）；`auto`＝`probe().freeBytes − HEADROOM_BYTES` 铺满 |

三条踩过的真坑（都是「形状/字段全对、跑起来才发现」）：

1. **`build_llm` 的 `device` 默认值 `"cpu"` 不能当哨兵**——`device="cpu"` 与 `layer_devices` 必须互斥，
   而 `"cpu"` 恰好等于老默认值 ⇒ 拿默认值比对分不出「没传」。改 `None` 更糟：`resolve_device(None)`
   是 `auto`，会把老路径的默认悄悄改成「有卡用卡」。用 `_UNSET_DEVICE` 哨兵（`device_llm_test` ⑧）。
2. **`plan_layer_devices` 的 `want_gpu_layers` 是下限不是上限**——显式写 `gpuLayers=1` 时它按预算
   给 2 层（该族只有 2 层）⇒ 后端必须**截到 1**，否则「配了 1 层、实际全上卡」（⑨⁶/⑨⁴）。
3. **分层时 `model_device()` 返回的是首张参数**——`embed` 被钉在末层（CPU）⇒ 回执会写「要了卡却在
   cpu」。分层口径只认**首层**在哪（⑨″）。

**未做**（有意）：`foresight` 式预取上卡（本仓 DiT 侧已有 `foresight.py`，
LLM 侧层数少、收益不明）。`kv_q8.hpp` 的 INT8 grouped-64 已落地（①，见 §7.2）；
跨调用 KV 前缀复用已落地（③，见 §7.3）；投机解码已落地为**默认关的可选开关**（④，见 §7.4.3）。

#### 7.1.1 A5000 真机实测（2026-11-08，不是作者宣传数据）

`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`（2.33 GiB，Q4_K_M，**36 层**，层权重 201 861 120 B/层
＝192.6 MiB，整份 6.75 GiB）＋ 本机 22.49 GiB 卡，探针落点
`data/experiments/layer_split_probe/probe.py`（实验产物按现行布局进 `<dataRoot>/experiments/<实验名>/`）：

| 口径 | 装载 | 生成 12 token | 真在 CUDA 的层 | `lm_head` | greedy 输出 |
|---|---|---|---|---|---|
| 全 CPU（`gpuLayers=off`） | 20.4 s | 2.4 s | — | cpu | `显存分层放置是指将显存中的数据按照` |
| 前 4 层上卡（`gpuLayers=4`） | 19.8 s | 2.9 s（**1.16–1.21×**） | `[0,1,2,3]` | cpu | 同上，**逐字相同** |

三点结论：① 回执与**实际参数设备**一致（`blocks[0..3]` 读回来 `.device.type == "cuda"`，
`lm_head` 仍在 cpu ✓）—— 这条比回执自己说的更硬；② 分层**不改变数值口径**（同精度 bf16 下 greedy
逐字相同，跨界的只是激活）；③ 代价如实：2.4 s → 2.9 s，即每 token 多一次跨界搬激活
（≈1 MB 级），**与 `park` 档「每步搬全部层权重」不是一个量级**（后者 2512 档实测 3 秒+/步）。
本机 4B 整份 6.75 GiB < 21.7 GiB 可用 ⇒ `gpuLayers=auto` 会把 36 层全放上卡；分层真正的用武之地是
「权重中文化后装不下」的场景（如 14B bf16 ≈ 28 GiB），届时 `auto` 会自动把尾巴溢出到 CPU。

⚠️ 落点踩坑（本轮真事）：探针一开始写在 `backend-py/output/_bench/` 下，判据 ⑦ 立刻红
（`backend-py/output/` 已整体退役，产物只许进 `<dataRoot>/`）。实验/探针一律进
`<dataRoot>/experiments/<实验名>/`，`backend-py/` 里**不留**任何产物目录。

### 7.2 落地实况：KV cache q8 分组量化（①）

落点与口径（`tests/engine_llm_test.py` ⑧ 系列 13 条 ＋ `tests/engine_llm_backend_test.py` ④ 系列守着）：

| 层 | 落点 | 口径 |
|---|---|---|
| 量化 | `llm.quantize_kv` / `dequantize_kv` | **沿 `head_dim` 每 64 元素共用一个 fp32 scale**、零点恒 0（口径同 `Strata/kernels/kv_q8.hpp` 的 grouped-64，只是 scale 用 fp32 而非 fp16：一组 64 个 int8 才 64 B，省 32 B 的 scale 换 1 位精度不值）。误差界 = `scale/2`（判据实测均值 ≈ `scale/4`） |
| 缓存 | `llm.Q8Kv` / `update_kv_cache` | 值、scale、组宽、**原始 `head_dim`** 绑在一个壳里（裸 int8 张量也能给 `.shape`，但没有 scale 的 int8 是垃圾）；`shape` 与量化前**逐字相同**；`append` 只 `cat` 旧段、**绝不重新量化** |
| 注意力 | `llm.GQAAttention.forward` | q8 分支先 `update_kv_cache(..., "q8")` 再 `materialize_kv` 回来算注意力 —— 读的必须是**量化后再反量化**的值，否则 prefill 用原 k、decode 读 `dequant(k)` ⇒ 同一段历史两套值 |
| 配置 | `settings.kvMode`（`text_generation.engine_kv_mode` 单一真源，三处调用口共用） | `off`/`fp`/`fp16`/`bf16` = 原样浮点（**老路径逐字节相同**）；`q8`/`int8`/`auto` = 分组量化；拼错 ⇒ `load()` **第一行**具名拒（装完 8 GiB 才报「kvMode 拼错」＝白等一轮） |
| 回执 | `LlmBackend.describe()["kv"]` | `mode/group/kBytes/vBytes/totalBytes/fpTotalBytes/ratio/contextBytes/fpContextBytes/dtypeBytes/assumedDtype`；**每 token 字节**而不是百分比（4B 档 144.0 → 76.5 KiB，ratio 0.531）；装载前按**请求值**归一（配了 `q8` 却报 fp 的字节，人会以为配置没吃）并标「装载后生效」 |

三条踩过的真坑：

1. **分组方向必须沿 `head_dim`** —— 沿序列分组是「语义无关的 token 共用一把尺子」，误差界看着一样、实际会把一整段上下文压成同一个量级（同一头内相邻 token 的 k 范数能差 10³ 倍，⑧⁴ 那条判据就是拿量级差 1000× 的两段验的）。
2. **全零组 `scale=1`** —— `amax=0` 时算出的 scale 是 0 ⇒ 反量化 `0/0` ⇒ NaN ⇒ **softmax 整行崩**，而且只在「某头某 token 全零」时触发（不报错，输出直接变乱码）。⑧⁵ 专判这一格。
3. **旧段 int8 原样 `cat`，绝不重复量化** —— 「反量化→拼新段→整段再量化」写法更短、单步误差也一样小，但每步都对同一批历史值再量化一次 ⇒ 32K 上下文等于 3 万次累积；这是 `Q8Kv.append` 存在的唯一理由。

#### 7.2.1 A5000 真机实测（2026-11-08，`data/experiments/kv_q8_probe/probe.py`）

`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`（36 层、32 头/8 KV 头、`head_dim=128`）＋ 22.49 GiB 卡，**2200 token** 提示词：

| 口径 | 每 token | 实测 cache（2200） | 回执算 | 满窗 262144 | prefill | 生成 16 token |
|---|---|---|---|---|---|---|
| `fp` | 144.0 KiB | 309.4 MiB | 309.4 MiB（差 +0 KiB） | 36.00 GiB | 0.8 s | 1.2 s |
| `q8` | 76.5 KiB | 164.4 MiB | 164.4 MiB（差 +0 KiB） | 19.12 GiB | 0.5 s | 1.3 s |

- **压缩比实测 0.531**，与「int8 数据减半 ＋ 每 64 个一组一个 fp32 scale」算出来的 `(128+4)/256` 逐位吻合；
- **回执与实测差 0 KiB** —— `describe()["kv"]` 那一栏不是估的，是 `kv_bytes_per_token` 按 config 算出来的同一个数；
- 速度代价 **1.14×**（16 token）/ **1.27×**（96 token）：每步多一次量化 + 一次反量化，都是逐元素 kernel；
- **数值口径如实**：16 token greedy 逐字相同；96 token 在第 ~50 个 token 分叉（`去重并清晰表达` vs `清晰表达`），两边的续写都通顺、只是措辞不同 ⇒ **q8 是有损的**，要逐字复现（判据基准、缓存命中比对）就用 `fp`。

⚠️ 量法踩坑（第一版探针就是错的）：**不能量「峰值显存差」** ✗ —— q8 每次前向会把整段 cache 反量化再算注意力，prefill 那一拍的瞬时峰值里含一份 fp 大小的临时副本 ⇒ 2200 token 下量出 207 MiB、理论 319 MiB，判据直接报「对不上」还看不出是量法问题。**cache 自己的字节**才是这个优化的真账（`q.numel()/scale.numel()` 逐个累加，见探针的 `cache_bytes`）。

### 7.3 落地实况：跨调用 KV 前缀复用（③）

落点与口径（`tests/engine_llm_test.py` ⑨ 系列 13 条 ＋ `tests/engine_llm_backend_test.py` ⑩ 系列守着）：

| 层 | 落点 | 口径 |
|---|---|---|
| 缓存 | `llm.KvPrefixCache` | **单槽**（只留最近一段）＋ 字节预算 ＋ 命中回执。`lookup(ids)` 返回 `(cache 段, 可复用 token 数)`；`store(ids, past)` 存**整段**（提示词 ＋ 续写 —— agent 循环下一轮带回的历史里就含这段续写，连上一轮输出也不必重算）；`describe()` 给 `enabled/mode/group/budgetBytes/storedTokens/bytes/hits/misses/skipped/reusedTokens/stored/note` |
| 裁剪 | `llm.truncate_kv` / `_slice_kv` | 把常驻段裁到 `reuse` 再喂前向；**`q` 与 `scale` 一起裁**（q8 只裁 q ＝ 长度对、数值错位），裁完 `contiguous()`（切片视图会把整段长张量吊住不放） |
| 生成 | `llm.generate_llm(..., kv_cache=)` | 命中时 `offset = reuse`（位置由 `past_len` 自然接上），只 prefill 尾部 `len(ids) − reuse`；**默认 `None` ⇒ 老路径逐位不变** |
| 预算 | `llm.resolve_kv_cache_bytes` | `off`/`0` ＝关（**与「不写」不是同一格**：不写 ⇒ 默认 512 MiB）；`512MiB`/`1GiB`/纯字节数都认；拼错 ⇒ `load()` **第一行**具名拒（又是「装完 8 GiB 才报配置拼错」那一类） |
| 记账 | `llm.kv_cache_bytes(past)` | 逐张量累加（`Q8Kv` 数 `q` ＋ `scale`）—— 与 §7.2.1 同一条纪律：**cache 自己的字节**才是真账，不量峰值显存 |
| 配置 | `settings.kvReuse` ＋ `settings.kvCacheBytes`（`text_generation.engine_kv_cache_bytes` 单一真源，三处调用口共用） | `kvReuse=off/false/0` ⇒ 预算 0；`kvCacheBytes` 缺省 ⇒ `None` ⇒ 512 MiB。缓存键**新增第 7 格** `cache_slot`（`None` ≡ 默认 ≠ `0`＝关 ≠ 拼错）——不归一就会「改了预算仍拿到老后端、回执一动不动」 |
| 回执 | `LlmBackend.describe()["kvCache"]` | 装载前／装载后都出（不抛）；`clear_kv_cache()` 换权重/换模板/腾显存时清 |

四条踩过的真坑（每条都是「不做就静默出错」）：

1. **按 token id 比前缀，不按文本比** —— 文本相等不保证 id 相等（模板空白、特殊符、编解码往返都能让同一句话变成另一串 id）；而且公共前缀必须 `≤ 序列长 − 1`：复用整段就没有查询可算、拿不到 logits，采样无从谈起。
2. **一点共同前缀都没有 ⇒ 当场释放** —— 「留着万一」会让一段永不命中的 cache 一路占着显存，卡满时报在别的件上（vae/text 塔），错误信息指不到这里。
3. **口径必须同一套** —— `fp` 的 cache 喂给 `q8` 跑＝长度对、数值两套（反向亦然）；`generate_llm` 里 `getattr(kv_cache, "mode", mode) == mode` 一票否决，宁可不命中。
4. **装不下就不装** —— `store()` 超预算直接不存并把原因写进 `note`（`N token 的 cache 要 X B > 预算 Y B ⇒ 不常驻`），不许「先存着试试」：那一拍正是别的件要显存的时候。这是 `conversation_memory.hpp` 的预算准入写法。

⚠️ 为什么复用在 q8 下也**数值无损**：量化沿 `head_dim` 分组 ⇒ 每个 token 的 scale 只由它自己那 64 个元素决定、与前后 token 无关 ⇒ 「先算好的段」与「重算一遍」逐位相同。这是当初把分组放在 `head_dim` 而不是序列维的第二个理由。

#### 7.3.1 A5000 真机实测（2026-11-08，`data/experiments/kv_prefix_reuse_probe/probe.py`）

`Qwen3-4B-Instruct-2507-Q4_K_M.gguf` ＋ 22.49 GiB 卡，**1650 token** 提示词（同一段 30 遍）、
`temperature=0`、16 token、`kv_mode=fp`、预算 `512MiB`：

| 口径 | 耗时 | 命中 | 复用 token | 常驻 |
|---|---|---|---|---|
| 冷 ②（复用关，同一串第二遍） | 0.9 s | — | — | — |
| 热 ②（复用开，同一串第二遍） | 0.6 s（**1.60×**） | 2 | 3344 | 1691 token / 237.8 MiB |
| 冷 ③（增量轮：提示词＋上轮输出＋新问题） | 0.9 s | — | — | — |
| 热 ③（增量轮） | 0.5 s（**1.69×**） | 同上 | 同上 | 同上 |

- **增量轮（agent 工具循环的真实形状）输出与冷跑逐字相同** ✓ —— 这条最要紧：循环每轮只 prefill 新问的那几十个词元；
- **末位 logits 差 max 0.145（相对 0.56%），argmax 相同** —— 复用的 KV 段与重算**逐位相同**，差别只来自「末位那一次前向的查询块形状从 `M×N` 变成 `1×N`」⇒ bf16 累加顺序差；
- ⇒ **同串整段复用只能到「舍入级一致」**：贪心在近似并列处会翻（实测：多/少一个空格 token）。判据基准、缓存命中比对这类要**逐字复现**的场景就关复用（`kvReuse=off`）；
- 回执与实测一致：`hits=2 / reusedTokens=3344 / storedTokens=1691 / bytes=237.8 MiB`、`misses=1`（第一次冷跑）。
- 预算账：4B 档 fp 每 token 144.0 KiB ⇒ **512 MiB 约装 3600 token**；真实长对话（>3600 token）会走 `skipped` 分支如实不常驻 —— 要么抬 `kvCacheBytes`，要么叠 ① 的 `kvMode=q8`（76.5 KiB/token，同样的预算装两倍）。

⚠️ 探针第一版判「四次生成逐字相同」⇒ 直接报红。不是接错位置（接错是乱码级别），而是上面那条 bf16 累加顺序差：同串整段复用的末位查询块只有 1 行、冷跑是 1650 行。判据改成相对阈值（`diff ≤ 5%·scale`），并把「要逐字复现就关复用」写进 `generate_llm` docstring —— **这条比判据本身重要**：复用省的是时间，不该被写成「无损」。

### 7.4 投机解码（④）：参考侧代码地图 ＋ 本仓实测 ＋ **已落地（默认关）**

#### 7.4.1 参考侧到底哪些代码在实现它

**先说清"参考库有几个"——这件事比代码地图更重要，因为查错库就等于没查。**

`_audit-lowvram/` 七个库里，**只有两个**与投机解码有关，且**分工不重叠**：

| 库 | 有什么 | 没有什么 |
|---|---|---|
| **Strata**（C++，三后端复制） | lookup/后缀草稿（`spec/suffix_drafter`）、决策层（`draft_policy`/`controller`/`window_rule`）、概率修正（`core/spec_prob.hpp`） | 草稿头本体是 `dflash2`/`mtp_*`，可读性差 |
| **ds4**（C 单文件 + 头文件注入） | **DSpark** 全套：`ds4_deepseek41_dspark.h`（`DS41_DRAFT_ROWS 5`／`verify_residual`／`confidence` 门控／`old_positions` 快照）、**回滚三件套**、`dspark_acceptance_fixture.sh` 的语义契约（`C_ADD_MIN_ACCEPTED`／`REQUIRE_DIRECT_COMMIT`／`REQUIRE_PROPOSAL_QUALITY`） | **没有 lookup 草稿** |

⚠️ 两个反面教训（都是我这轮踩的）：
1. 按 Strata 的命名（`prompt_lookup`/`suffix_drafter`/`n_gram`）去搜全仓 ⇒ 只命中 Strata，
   会误判「只有单一来源」。ds4 用的是 `dspark`，一次没沾上这些词。
2. 看到 ds4 的 `ds4_engram.c`（108 处 `ngram`）就以为是 n-gram 草稿 —— 读原文才确认它是
   **Engram 记忆模块**（哈希 n-gram → 查 `per_layer_token_embd` 取 embedding 行、`DS4_ENGRAM_DIM=256`），
   属模型结构，与草稿无关。**名字里带 gram 不等于草稿。**
⇒ 结论：**lookup 草稿层是本仓唯一「没有第二家可对拍」的层**（不像 Q5_K 有 Strata+ds4 双轨互核）；
草稿头、回滚、概率修正三层都有两家，可以对拍。

Strata 的投机解码不是单个文件，是**七层**，每层各有自己的失败模式。按数据流列（行号取自
`c62d5e3a15b4`）：

| 层 | 落点 | 干什么 / 失败模式 |
|---|---|---|
| **① 提案** | `spec/suffix_drafter.{hpp,cpp}` | `SuffixDrafter::observe(token)` / `propose(k, buf)` / `last_match()`：三词元后缀 → 历史中最近的几种出现位置 → 取该位置之后的 `k` 个词元当草稿。**零权重**，纯查表 —— 这是本仓唯一能用的形态。实现细节：开放寻址表（容量 2 的幂、负载 ≤0.5、`key|1` 作非空哨兵）、`key_at` 走 `mix()` 混淆、每槽存 `WAYS=3` 个位置且**最近优先**（故等长匹配保留更近的）、**允许草稿与当前后缀重叠**（周期性文本需要）；`propose_after(extra,...)` 把外部草稿临时接到尾部但不建索引，是多源串接的接缝 |
| | `spec/draft_source.{hpp,cpp}` | `DraftSource` 抽象 ＋ `PromptLookupSource`；`propose_from_sources(own, tail, n_tail, n_pending, max_k, min_match)`：**多源串联**（MTP 抽几个、后面再接 lookup 候选），`chain_tail` 就是串上的接缝 |
| | `core/dflash2.hpp`、MTP 那套 `core/mtp_*` | 真·草稿头：`Dflash2Drafter::draft/truncate(keep)`。**本仓没有，也不打算为它下载权重** |
| **② 词表对齐** | `spec/draft_vocab.hpp` | `df2_vocab_resolve(...)`：草稿头词表 ≠ target 词表时做映射。**映射错 = 草稿全是别的词，但一步都不报错**，只表现为接受率变成 0 |
| **③ 决策** | `spec/draft_policy.{hpp,cpp}` | `DraftPolicy::choose(t_mtp, lookup_k, match) -> Pick{lookup, t}`、`chain(...)`、`apply_costs(p)`：拿**实测**的「每轮期望词元 / 每轮实测毫秒」比，不是拿理论 FLOPs 比 |
| | `spec/controller.{hpp,cpp}` | `CostModel` ＋ `Controller::choose(lookup_available, lookup_match, mtp_ready)` |
| | `spec/window_rule.hpp`（纯函数，header-only，CLI 与 serve 共用） | `choose_window(WindowRuleIn)` 按**本轮活跃源**定窗宽；`cost_window_applies`（成本规则只算 MTP 自己的候选）、`mtp_q_applies`（概率修正**只对 MTP 抽的候选**生效 —— lookup 候选不是从 q 抽的，没有 q 可修正）、`skip_draft_for_lookup`（长匹配直接投，不必先跑草稿头） |
| | `spec/spec_start.hpp` | `spec_loop_runs(spec_pos, produced, max_new, ended)`：还剩几个词元、是否已 EOS —— 不够开一轮窗口就别开 |
| **④ 拼窗** | `program/generate.cpp:18600-18677` | `window[0] = x`（上一轮最后那个词元，**必须是 target 自己产生的**，不能是草稿，否则窗口里没有可信的查询起点）＋ `window[1..T-1]` = 草稿/提示词尾；`merge_T` 是首窗把提示词最后几个词元并进来一起验证（预热）；`graph_fits(T)` 是因为窗宽被 CUDA graph 的固定尺寸钉死 |
| **⑤ 验证** | `core/verify.hpp:215` ＋ `core/verify.cpp:3852` | `run(T, window, ...)` 一次前向算 T 行 logits；`commit(n_keep, err)` **不是截断，是按位置写前 `n_keep` 行**：`h_commit_ = [n_keep, n_keep-1, pos0+t (t<n_keep) 否则 -1]`，`-1` 表示该行不落账，下一轮从 `pos0+n_keep` 继续写自然覆盖。两个特例：`last_t_==1 && one_token_self_commit()` 时窗口已自行推进、**不能再跑 commit graph**；`ple_stage()` 要逐行推进 `ple_prev`。这一步错了不报错，只会让后续位置整体错位 |
| **⑥ 接受** | `program/generate.cpp:18879` | 贪心：`int a = merged ? T-1 : 0; while (a < T-1 && window[a+1] == outv[a]) ++a;` 然后 `ver.commit(a+1)`。**每轮推进 a+1 个词元** |
| **⑦ 概率修正** | `core/spec_prob.hpp`、`core/spec_proposal.hpp`、`verify.set_spec_q/set_spec_proposal` | 温度 >0 时的拒绝采样：以 `min(1, p/q)` 接受，拒绝则从残差 `max(0, p−q)` 重采样 ⇒ 分布与纯 target 采样**逐位同分布**。`set_spec_q` 把 q 传给 sampler kernel，**消费点在 GPU kernel 里、不在 host 侧** —— 本仓没有这套 kernel，照抄 host 侧代码会得到「贪心验证 + 采样温度」的错配 |
| **⑧ 统计** | `generate.cpp` 的 `sfx_windows/sfx_drafts/sfx_ok`、`chain_*`、`dp_tested/dp_ok`（按草稿序号统计）、`t2_rej/t2_hit` | 这些计数**就是 ③ 决策层的输入**（自我校准的成本模型）。没有它们，policy 只能靠常数猜 |

一句话：**①③④⑤⑥ 是可搬的算法骨架；②⑦ 依赖 draft head 与 CUDA kernel，本仓无对应物；⑧ 是让前五层不靠猜的前提。**

#### 7.4.2 本仓实测（2026-11-08，`data/experiments/spec_lookup_probe/probe.py`）

`Qwen3-4B-Instruct-2507-Q4_K_M` / bf16 / 22.49 GiB，past 固定 512 词元、7 次取中位：

| T（一次喂几行） | 墙钟 | `ratio(T)` | 每轮回本所需接受数 |
|---|---|---|---|
| 1 | 31.7 ms | 1.00× | — |
| 2 | 33.4 ms | 1.05× | 0.05 |
| 4 | 36.3 ms | 1.14× | 0.14 |
| 6 | 30.9 ms | 0.97× | −0.03 |
| 8 | 32.0 ms | **1.01×** | **0.01** |

**`ratio(8) ≈ 1.0` 是本轮最重要的一个数**：本仓 decode 每步被「2.33 GiB 权重读取 ＋ 逐层小算子 ＋ Python」
占满，多算 7 行的 FLOPs 落在噪声里 ⇒ 窗口**几乎白送**，回本门槛只剩「每轮接受 > 0 个词元」。
两次独立测量分别得 1.01× 与 0.99×，全部落在 1.0 附近（T=4 那次 1.14× 是唯一明显偏高的点，
说明这个口径本身有 ±10% 的抖动，判据要看趋势不能抠单点）。

门槛既然这么低，剩下的问题就只有一个：**草稿命中率**。用三词元后缀索引在真实产物上离线模拟
（先跑一遍逐 token greedy 当真值，再判断「如果当时投机会接受几个」）。模拟**与
`suffix_drafter.cpp:64-86` 逐行同构**：槽内最近优先、等长保留更近的、**允许草稿与当前后缀重叠**
（参考实现注释明写 periodic text 需要它）：

| 场景 | 生成长度 | 草稿/接受（命中率） | 每轮平均接受 | 同尺度加速 = (1+a)/ratio(8) |
|---|---|---|---|---|
| A · 原创中文台词续写 | 96 词元 | 133 / 1（0.8%）· 95 轮 | 0.01 | **1.02×** |
| B · 结构化 JSON 输出（同字段反复填） | 96 词元 | 238 / 2（0.8%）· 94 轮 | 0.02 | **1.03×** |
| C · 逐字复述（对照组） | 96 词元 | **644 / 0（0%）** · 96 轮 | 0.00 | 1.00× |

⇒ 收益就是 **1～3%，落在测量噪声里**（这个数字没有变，落地后也没有被"优化"成更大）。

⚠️ 但**代价被高估过，这里更正**：上一稿写「代价是加一条 KV 回滚路径」，那是错的 —— 本仓
`llm.py:74` 的 `truncate_kv` 早在 ③（跨调用前缀复用，同一天落地）就已经存在，且注释正写着
「多留一个 token 会让位置与 cache 错位 ⇒ 注意力看见一个幽灵位置」，正是回滚要处理的那个坑；
`GQAAttention.forward` 的因果掩码按 `past_len + i` 算，**天然支持一次喂 T 行**（prefill 就是按 1024 chunk 喂的）。
所以本仓缺的**不是**多行前向、也不是回滚，而是把 `generate_llm` 的「一拍一 token」循环改成「一拍 1+T」——
纯结构改动。**门槛被高估这件事本身要记下来**：判断「不做」时给的代价理由必须能落到一个不存在的原语上，
否则就是拿「听起来难」当依据。

⚠️⚠️ **场景 C 是最值得记的一条**：644 个草稿、0 个接受，其中匹配长度 ≥16 的有 66 轮 ——
**匹配越准越不接受**。原因是参考实现取草稿用的是**最长匹配**的结束位置
（`suffix_drafter.cpp:84` 的 `hist[best_end + 1 ...]`）：逐字复述时匹配会一直扩到当前尾部，
`cur - best_end` 恰好等于「已复述的篇幅」，于是草稿从**原文第 4 个词**开始把已复述内容再讲一遍，
整体错位 k 位。匹配越长错位越确定。
⇒ 这不是本仓的问题，是**参考实现「最长匹配优先」这个取舍的盲区**：它押短局部重复（代码/JSON 补全），
一旦文本是长复述就系统性失效。要修得改成「偏好 `best_end + 1 + max_k <= cur` 的匹配」
或一次投多个候选位置 —— 那已经超出「照搬」的范围，且本仓场景 A/B 也用不着。

⚠️ 与 Strata 实测（lookup +33%～+52%）不矛盾：那边报的是**代码补全**类短局部重复任务的数，
且用 native MMVQ 内核的成本模型。中文创作型台词里可预测的重复只有「角色名＋冒号」这种
3～4 词元的格式壳，长度不足以撑起一轮窗口。

**若将来条件变了**（接入 draft head，或出现真正的复述型任务），门槛已经量好了：
`ratio(T)≈1.0` ⇒ 每轮多接受 0.01 个词元就回本，收益上限 = 窗口宽 8×。**这个门槛比结论更值得留**。

#### 7.4.3 落地实况（2026-11-08）：做成「默认关的可选开关」

为什么不因为收益小就不做：**收益小 ≠ 代价大**。上一稿把「加 KV 回滚」当成代价，而 `truncate_kv` 早已存在，
于是真实工作量落在「把一拍一 token 改成 1+T 行」上 ⇒ 既然只有结构改动、且默认关不影响任何人，
就按「能力先具备、默认不开」落地（与 ①②③ 同一个口径：能力齐、开关在手、回执如实）。

落点（四处，一处不少）：

| 层 | 落点 | 口径 |
|---|---|---|
| 草稿器 | `llm.LookupDrafter`（`observe`/`append`/`propose`） | 三词元后缀 → 槽内**最近优先**、`WAYS=3`、**等长保留更近**（`>` 写成 `>=` 结果不同但不报错）、**允许与当前后缀重叠**、`min_match` 不够返回**空列表**（不是抛错） |
| 主循环 | `generate_llm(..., spec_lookup/spec_window/spec_min_match/spec_report)` | `window[0]` ＝ target **刚产出但还没喂过前向**的那个 token ⇒ 一拍吃 `1+T` 行；接受判据是「第 `i` 行 argmax == `window[i+1]`」（抄成 `window[i]` 全拒且不报错）；回滚 `keep = min(past_len + 1 + a, total + len(generated))`（上限压到已产出总数，stop 打断时 cache 不许比 ids 长）；`a == len(draft)` 时第 `a` 行就是 bonus token，被拒时它是修正值 ⇒ 两条路同一个 `_sample_token(win_logits[:, a, :])` |
| 后端 | `LlmBackend(spec_lookup/spec_window/spec_min_match)` ＋ `describe()["spec"]` | **开关**与**上一次真的投机了没**分开报（`enabled`/`rounds`/`drafted`/`accepted`/`acceptedRate`/`reason`/`note`）；拼错的旋钮在 `load()` **装权重之前**具名拒（照 `kvCacheBytes` 的老纪律：`describe` 不许因拼错而抛） |
| 服务口 | `text_generation.engine_spec_kwargs(settings)` ＋ `_engine_backend_key` 三格 | **三个调用口共用一把**（普通生成／流式／agent），缓存键含投机开关与两个数；**关着时那两格为空**（否则写个 `specWindow=4` 就白装一份权重）；改 `specLookup` 必须换后端，否则配置写了等于没写 |

`temperature > 0` 的处置：**如实退回**，不做假实现。理由：随机采样要「按 `min(1, p/q)` 概率接受 ＋ 把拒绝处的
采样落在与不开时**同一条随机序列**上」（Philox 计数空间隔离），本仓采样是 `torch.multinomial` 全局 RNG，
塞进去必然改变输出 ⇒ 那就不是「同一个生成、只是更快」了。回执写 `enabled=False, reason="temperature>0"`，
输出与不开**逐位相同**（测试 ⑪¹⁰ 判）。

判据（不靠"看起来对了"）：

* `engine_llm_test` ⑩ 系列 12 条：真模型下开关**逐 token 相同**、q8 下相同、全接受 `accepted == drafted > 0`、
  回滚差值区间（**部分拒绝时必须严格小于窗口行数** —— 全接受时「不裁」的差值恰好等于行数 ⇒ 只看区间会假绿 ✗✗，
  为此专门加了 `stuck_at` 的部分拒绝跑）、输出**依赖 `past_len`** 的模型下仍等价（这条才真正抓住"没裁"）、
  温度退回、三条草稿器口径、额度边界、参数校验。
* `engine_llm_backend_test` ⑪ 系列 10 条：回执分得开、两个旋钮拼错/越界在 `load()` 开头拒、缓存键三格
  （关着不进格、开着进格）、`engine_spec_*` 两种键名、普通口真把三格递下去、**AST 判三个调用口都展开
  `engine_spec_kwargs`**、**AST 判后端里 3 处 `generate_llm` 都带 `spec_report`**、真跑 `drafted > 0` 且与不开
  逐 token 相同、温度 > 0 退回。
* 反向验证过两种坏法：把 `truncate_kv(...)` 换成「原样留 `win_kv`」⇒ ⑩⁴／⑩⁵ 当场红；把
  `ln > best_len` 改成 `>=` ⇒ ⑩⁹ 当场红。复原后 60/60。
