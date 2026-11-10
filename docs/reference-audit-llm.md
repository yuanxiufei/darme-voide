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

### 7.5 落地实况：融合核快路径（`fastPath`，**默认关**）

§7.4 的 profile 已经把病根说尽了：decode 单步墙钟 35.7 ms，CUDA 侧合计只有 16 ms 级，而
`wall ≈ CPU 侧 40.2 ms` ⇒ **卡在等指令，不在算**。一步 2506 次 `cudaLaunchKernel` 自身就烧掉
17.8 ms。所以第一轮只做「**减 launch**」，不碰算法：换两个融合核，再把 RoPE 折成一次成型；
第二轮补第三档 `kvbuf`（KV 定长缓冲 ＋ 原位写，省的是**带宽**不是 launch —— 见 §7.5.6）。

| 档 | 换掉了什么 | 换来什么（一次前向） |
|---|---|---|
| `norm` | `RMSNorm` 的 `pow/mean/add/rsqrt/mul/mul` 五连 → `F.rms_norm` | 一次调用 **16 → 2** 个 kernel；一步 **2320 → 290** |
| `attn` | 手写三式 ＋ `repeat_kv` 物化 → `F.scaled_dot_product_attention(enable_gqa=True)` | 每层 **6 → 1** 个 kernel；一步 **−216**，且不再写那 1.15 GiB/步的复制 |
| `kvbuf` | 每步 `cat([past, fresh])` 整段重拷 → 定长缓冲 `KvBuf` 原位写新来的那几行 | 一步 72 次整段 `aten::cat`（2048 词元＝288 MiB/步）→ 72 次 `copy_` 各写 **1 行**（288 KiB）；**launch 数不变**、墙钟 −5.8% |
| （无开关） | RoPE 折叠：`cos/sin` 从 `(seq, d/2)` 折成 `(seq, d)` | 每次 **7 → 4** 个 kernel；一步 **−216**（且省掉每层 2 次 dtype cast） |

真机实测（A5000 22.49 GiB、`Qwen3-4B-Instruct-2507-Q4_K_M`、2048 词元上下文、greedy、同一个已装载模型
只改 `fastPath` 一个变量；探针 `data/experiments/llm_attn_cost/fast_path.py`）：

| 档 | 每步墙钟 | token/s | kernel launch | CUDA 合计 | logits max\|差\| |
|---|---|---|---|---|---|
| `off` | 33.67 ms | 29.7 | 4449 | 50.86 ms | 0（基准） |
| `norm` | 27.95 ms | 35.8 | 2419 | 47.90 ms | 3.6e-1 |
| `attn` | 30.80 ms | 32.5 | 4233 | 46.91 ms | 3.1e-1 |
| `kvbuf` | 31.71 ms | 31.5 | 4449 | 47.23 ms | **0**（逐位相同） |
| `all` | 23.73 ms | 42.1 | 2203 | 47.10 ms | 3.4e-1 |

`all` ⇒ **−29.5% 墙钟 / −50.5% launch**；整条生成循环（含采样、惩罚、回文字）**31.2 → 22.9 ms/步**、
**32.1 → 43.7 token/s**。`norm` ＋ `attn` 合计砍掉 2246 次 launch（`kvbuf` 一次也不减 ✓ 见 §7.5.6），
其中 `norm` 那 2030 次**逐次对得上账**：
本模型带 QK-norm ⇒ 一步 145 次 `RMSNorm` 调用（36 层 × 4 ＋ 末层 1），老路 16 核/次 ⇒ 2320，
融合 2 核/次 ⇒ 290，差 **2030** ＝ 4449 − 2419 ✓。⚠️ 这一步是**量出来才发现的**：纸面算「5 个 kernel ×
73 次」只得 365，与实测差 5.6 倍 —— 真机上 bf16→fp32 往返与两段 reduction 把老路放大到 16 核/次。

#### 7.5.1 一个真错：`fold_rope` 折错了配对

折叠 RoPE 的第一版按「**相邻交错**」写（`stack((cos, cos), -1).flatten(-2)`），而 `apply_rope` 与全部
Qwen/Llama 权重都是「**前后半配对** `(x[i], x[i+d/2])`」。折错的症状是：形状一模一样、不报错、能出
token，只是**模型变哑巴** —— 而当时它已经无条件接进了主路（折叠本身不设开关，因为宣称与老写法逐位
相同）。抓到它的是 ⑫⁴ 那条「`fold_rope` ＋ `apply_rope_wide` 与 `apply_rope` **逐位相等**」的判据：
`torch.equal` 直接 False。改法：`cat((cos, cos), -1)` ／ `cat((-sin, sin), -1)`。

**教训**：凡是「宣称逐位相同、所以不需要开关」的改写，判据必须是 `torch.equal` 而不是 `allclose`
（`allclose` 对这一版也是绿的：错的是配对，输出量级不变）。

#### 7.5.2 为什么默认关、关了怎么保证没动老路

* 数值会变：`F.rms_norm` 与老写法差 **1 个 bf16 ulp**（实测 logits max\|差\| 3.1e-1，但 4 条路首 token
  相同）；SDPA 与手写三式差 1e-7 级。真正的风险是**长程**：贪心在第 40、200 个 token 处可能岔开。
  所以与 `kvMode`／`specLookup` 同一套纪律：默认关、**关闭时逐位不变**、拼错**当场具名拒**。
* 「关着逐位不变」靠 `fast_kw = {"fast_path": fuse} if fuse else {}` 保证 —— 空集时**根本不传**那个
  关键字 ⇒ 调用签名与老路逐字相同（本仓一堆只认 `(ids, past_kv, kv_mode)` 的桩模型测试靠这条活着，
  判据 ⑫¹³／⑫¹⁴ 正反两面都判）。
* 「配了 ≠ 生效」三个老坑各配一条判据：① `fastPath` 必须进 `_engine_backend_key`（不进格 ⇒ 开关一改
  还是拿到**关着的旧后端**，而回执照报「已开」）；② `norm` 在缺 `F.rms_norm` 的机器上**静默退回**老写法
  ⇒ 回执单列 `fusedNormAvailable` 并明写「一步也没省」；③ 三处生成口（普通／流式／agent 工具循环）都
  必须展开 `engine_fast_kwargs` —— AST 判，漏一处＝那条路永远是老写法。

#### 7.5.3 顺手对齐 §7.2.1 那笔「75 ms/步」

§7.2.1 记的是「生成 16 token 1.2 s ⇒ 75 ms/步」，§7.4 与这次量到的都是 31–33 ms/步，差 2.4 倍。
实测给答案：**prefill 2048 词元 480 ms**、decode 31 ms/步 ⇒ `480 + 16×31 = 976 ms`，与 1.2 s 同量级
⇒ 那笔账把 **prefill 摊进了每 token 的分母**，两个数都没错，只是口径不同。今后凡报「ms/步」必须写明
**含不含 prefill**，否则同一台机器能差 2.4 倍。

#### 7.5.4 判据

* `engine_llm_test` ⑫ 系列 14 组共 19 条：`resolve_fast_path` 归一表（大小写／`_`／`+`／列表／布尔）、返回
  `frozenset`（`"all"` 不含 `"norm"` 子串，用字符串包含判断会把 `all` 读成「只开 attn」）、拼错拒、
  **折叠 RoPE 与 `apply_rope` 逐位相同**（fp32/bf16 两档）＋ 反证（前半减号写反必须看得出差）、
  `fold_rope(dtype=)` 与「先 `.to` 再折」逐位相同、宽形／窄形两条支路**逐位相同**、SDPA 与手写三式
  数值一致（prefill／带 past／decode 三种）＋ 反证（掩码对角偏 1 必须远超容差）、**spy 数
  `F.rms_norm`／`F.scaled_dot_product_attention` 的调用次数**（`off` ⇒ 0／0，`norm` ⇒ 5/0，`attn` ⇒ 0/2，
  `all` ⇒ 5/2 —— 只验「形状对、不崩」在「开着却静默走老路」时也是绿的）、`fused_norm_available()` 不抛、
  默认关不传关键字 ＋ 反证（开着必 `TypeError`）。
* `engine_llm_backend_test` ⑫ 系列 12 条：回执 `norm`/`attn` 分开报 ＋ 关着也说清「逐位不变」、
  `fusedNormAvailable` 单列且缺原语时 note 必含「退回」、拼错在 `load()` **开头**拒 ＋ `describe()` 不抛、
  缓存键含 `fastPath`（关＝不进格、`all`＝`norm,attn`＝`True` 归一成同一格、拼错原样留格）、
  `engine_fast_path`／`engine_fast_kwargs` 两种键名、普通口真把值递下去、**AST 判三个调用口都展开
  `engine_fast_kwargs`**、**AST 判后端里 3 处 `generate_llm` 都取 `fast_kwargs`**、`_fast_call()` 空集返回
  空字典、真跑 spy（`fastPath=all` 一路走到 `generate_llm` 形参上是 `frozenset({"norm","attn"})`）＋ 反证
  （关着时 spy 收到「没传」）。
* 反向验证过三种坏法（都真跑过、都当场红、都复原）：`fold_rope` 折回相邻交错 ⇒ ⑫⁴／⑫⁵／⑫⁷ 红
  （这不是"造出来的坏法"，是**真发生过**的那次）；`_fast_call()` 改成空集也传 `{"fast_path": frozenset()}`
  ⇒ ⑫⁹／⑫¹¹ 红（55/57）；`fast_slot` 恒 `""`（不进缓存键）⇒ ⑫‴／⑫⁴ 红（55/57）。复原后 79/79 ＋ 57/57。

**`kvbuf`（第二轮）另加 `engine_llm_test` ⑬ 系列 12 条**，因为这一档的判据性质与上面两档相反 ——
它宣称**逐位相同**，所以判据只能是 `torch.equal`，而且必须**同时数 `aten::cat`**（只判值对，在
「其实还在 cat」时照样绿）：

* ⑬¹ fp 逐位相同 ＋ `shape[2]` 报的是**已用长度**（报 `cap` ＝ 位置/掩码整排错位、形状还自洽）；
  ⑬² q8 连**量化字节**（`q` 与 `scale` 各对一遍）；⑬³ 150 行 ⇒ `length=150`、`cap` 64→128→256，
  且 `kv_cache_bytes()` 对缓冲报**已分配**（翻倍留的尾巴也在卡上，会 OOM）；
* ⑬⁴ **spy `torch.cat`**：老路 3 步 2 次、缓冲路 **0** 次（q8 档也 0）；⑬⁵ 整模型 prefill ＋ 3 步 decode
  的 logits 逐位相同；⑬⁶ 一 step 的 cat 数：老路 `fixed + 2×depth`、缓冲路只剩 `fixed`
  （`fixed` = 折叠 RoPE 2 ＋ 每层 q/k 各 1，实测 4 次/depth＝2 —— 这条数**量出来**的，
  `apply_rope_wide` 每层也 cat 两次，纸面「只剩 2 次」是错的）；
* ⑬⁷ ⭐ `truncate_kv` 对缓冲必须是**独立副本**（就地改长度会改坏 `KvPrefixCache` 那份常驻 cache ⇒
  下次命中更长前缀读到脏行、不报错）；⑬⁸ 反证 `rollback_kv` **就地**回拨（同一对象、零拷贝）；
* ⑬⁹ 混用（开→关→开）**字节原样**（`Q8Kv`↔`KvBuf` 走 `as_q8()`／`_take()`，中途 dequant 再 quantize
  会多损一次）；⑬¹⁰ 批/头数与 `kvMode` 口径不符**当场拒**；⑬¹¹ 反证：把新段一律写到位置 0 ⇒ ⑬¹ 立刻红；
* ⑬¹² `generate_llm` 整条：开缓冲与老路逐 token 相同 ＋ 带 `KvPrefixCache` 第二次命中且结果仍相同。

`engine_llm_backend_test` ⑫ 系列同步把 `kvbuf` 单列进回执（`norm`/`attn`/`kvbuf` 三项分开报、
`items` 三元排序、缓存键 `all` ＝ `norm,attn,kvbuf` ＝ `True` 同一格）—— 仍 57/57。
`engine_llm_test` ⑫ 的归一表也补了 `kvbuf` 的两个别名组与「`all` ＝ `FAST_PATH_ITEMS` 全集」
（**这条会随档位增加自动跟着变**，抄死字面量 `{"norm","attn"}` 的写法在加档当天就红）。合计 **91/91**。

另做了两次**真改代码**的反向验证（都真跑过、都复原）：

* `_slice_kv` 的 `KvBuf` 分支改回 `x.rollback(length)`（就地）⇒ **90/91**，只 ⑬⁷ 红，报 `(104, 104)` ——
  正是「原份被改小」；
* `KvBuf.write` 里偷偷多 `torch.cat` 一次（**值一个字节不变**）⇒ **89/91**，只 ⑬⁴／⑬⁶ 红
  （`(2, 3, 3)`／`(10, 10, 6)`）—— 这条才是这一档的关键：**只判「值对」的判据在这里会全绿**，
  所以 `aten::cat` 计数是必判项，不是锦上添花。

#### 7.5.5 第二轮：KV 定长缓冲（`kvbuf`）

老路的病：`update_kv_cache` 每拍 `torch.cat([past, fresh], dim=2)` —— 每层每步把**整段历史**重新
分配并拷一遍。4B 档 2048 词元：一步 **72 次 `aten::cat`**（36 层 × k/v；q8 连 scale 一起 ⇒ 144 次）
＋ 每步 **288 MiB** 的分配/拷贝。

改法：`KvBuf`（`store`／`scale` ＋ `length`／`cap`）—— 沿序列维先留 `KV_BUFFER_MIN_ROWS=64` 行，
新段**原位写**进 `[length, length+s)`，容量不够就**翻倍**（翻倍只搬已用的那几行，摊还 O(1)）。
口径四条，每条都是「不做就静默错」：

1. `shape[2]` **仍然是已用长度** —— 全仓十几处拿 `past_kv[0][0].shape[2]` 当序列长（位置、掩码偏移、
   前缀复用、投机回滚）。暴露 `cap` ＝ 位置直接错位，而且错得**形状自洽**（不报错）。
2. q8 段的「值 ＋ scale」必须**一起**预分配、一起原位写（只搬一半 ＝ 长度对、数值错位）。
3. 只写新算出来的那一段，历史一个字节都不碰（老路 `cat` 也不重量化 —— 重量化会随步累积误差）；
   矩阵乘前必须过 `materialize_kv` 拿 `[:length]` 视图（直接喂 `KvBuf` 会把 `cap` 行未初始化值算进
   注意力：数值全烂、不报错）。
4. `truncate_kv`（前缀复用处）与 `rollback_kv`（投机窗口处）**不许互换**：前者必须返**独立副本**
   （`past` 十有八九是 `KvPrefixCache` 里那份常驻 cache，就地改长度会覆盖别人的历史），后者必须
   **就地**零拷贝（那块存储是本轮自己刚算的，被拒的行下次原位覆盖 ＝ 参考实现的「从未写入」）。

实测（同一次运行，探针第九节）：

| 档 | `aten::cat` | `aten::copy_` | kernel | CUDA | 常驻 cache | logits max\|差\| |
|---|---|---|---|---|---|---|
| `off` | 146 | 366 | 4449 | 50.86 ms | 288.7 MiB | 0（基准） |
| `kvbuf` | **74** | 438 | **4449** | 47.23 ms | 576.3 MiB | **0.000e+00** |

三条读数各自打掉一个想当然：

* **`aten::cat` 146 → 74，正好少 72**（36 层 × k/v），证明「每层每步那两次整段重拷」确实没了；
  剩下的 74 次与 KV 无关（折叠 RoPE 2 ＋ 每层 `apply_rope_wide` 2）。
* ⚠️ **`kernel` 数 4449 → 4449 一个不少**，`copy_` 反而 +72 —— 72 次 `cat` 换成了 72 次 `copy_`，**一比一**。
  所以这一档**不是省 launch**（与 `norm` 那 2030 次不是一回事），它省的是**带宽**：每步拷贝量
  288 MiB → 288 KiB，CUDA 侧 50.86 → 47.23 ms、墙钟 −5.8%。**别在回执里写「少掉 216 次 launch」** ——
  第一版就写错了，是实测数（`cat` 只少 72、`kernel` 一个不少）才对出来的。
* 常驻 cache 288.7 → **576.3 MiB**：2048 行那一步正好翻到 4096（`cap` 按翻倍涨）。这是这一档的代价，
  而且**会随长度阶梯跳**（不是稳态 2×）；报数一律走 `kv_cache_bytes()` 的「已分配」而不是「已用」。
  ⚠️ 2026-11-08 更正：自 `generate_llm` 按 `kv_rows=(room, room)` 预分配后，生成路径**一次就分配到
  精确需要**、不再翻倍（`ceiling == need` ⇒ `cap` 不会 overshoot）⇒ 这条读数属**老跑法**，见 §7.5.6 ②；
  「报已分配」这条口径不变（前缀缓存那份 `frozen()` 仍按实际长度精确分配）。

精度：4B 真权重（bf16）logits 最大差 **0.000e+00** —— 与宣称一致。但**仍默认关**：矩阵乘读到的是
**带跨步的视图**（`[:len]` 而非 `contiguous()` 的整块），换成别的卡/别的内核选择可能改变累加顺序，
所以与 `norm`/`attn` 同一套纪律，只用 ⑬ 系列的逐位对拍盯着。顺带修了一处真会炸的：`dequantize_kv`
里 `view` 改 `reshape` —— 跨步切片过 `view` 当场 `RuntimeError`，而 `.to(float32)` 不保证洗掉跨步。

#### 7.5.6 收口（2026-11-08/09）：图化已落地，两笔尾账的真相

`kvbuf` 就是 CUDA Graph 的前置：图里的指针必须每步相同，而老路每步都换一块新存储 ⇒ 录不进图。
指针固定之后，2203 次 launch 已**一次收掉**：

**① 图化（P1-A，`cudaGraph` 档、默认关、`bitExact=False`）** —— Qwen3-4B-Q4_K_M／A5000／prompt 256：
单张图重放 28.7 → 20.2 ms/步，产品路径**每请求都重新捕获**（捕获 ≈100 ms ≈3 次前向）后实测
8 词元 ×0.91、16 ×1.09、32 ×1.17、64 ×1.24、**128/256/512 ×1.24–1.27**（35.4 → 44.2–45.4 tok/s）
⇒ 本仓真实负载（写剧本/对白 128–512 词元）稳吃 ×1.24 以上。代价不是那 100 ms，是**输出不再逐字
可复现**：同一档 `padRows=10` 下 8/16/32 词元 0 分叉，但 64 词元 11/64、128 词元 72/128、
**512 词元 443/512** 与老路不同（`6.48` 那节的 1 ulp 随步累积）⇒ 长生成要逐字复现就别开。
回执：`describe()["cudaGraph"]` 三态 `requested`/`enabled`/`last`，`last` 交 `padRows`/`captureMs`；
**关着的那一档也把实测收益与代价写出来**（否则没打开的人永远看不到打开值不值 ＝ 自锁）。

**①′ `cudaGraph=auto`（2026-11-09 补 —— 把 ① 的长度全表变成行为）**：此前 `auto` 与 `on` **同义**
（都落在 `resolve_cuda_graph` 的接受表里）⇒ 人写 `auto` 期望「该开则开」，实际**无条件捕获**，
连 ① 里那个 ×0.91 的 8 词元档也照捕 100 ms 的图，而回执照旧只报 `on`（安静地做亏本的事，与
「安静地没开」同一族禁令）。现在 `resolve_cuda_graph_policy` 出 `off/on/auto` 三档，`auto` 在
`max_new_tokens < CUDA_GRAPH_AUTO_MIN_TOKENS(16)` 时**主动不捕**并写 `reason="autoTooShort(n<16)"`；
16 取自 ① 的 8→×0.91／16→×1.09，是**不亏线** ✗ 不是甜点（×1.24+ 要到 128–512）。`auto` 跳过时
**连 kvbuf 都不替你加**（跳过＝完全老路径）。回执 `cudaGraph` 多一格 `policy`（`enabled=True` 分不出
`on` 与 `auto`），三态 note 各自成句、且**不混**「让路/前置不满足」那句（一个该改 maxOutputTokens、
一个该查驱动）；缓存键第 14 格改用三档原文（`on` 与 `auto` 背后是两个后端实例，压成 bool 会切档
静默无效）。判据：⑮¹′（三档归一）、⑮⁵′（**阈值两侧**：15 ⇒ `autoTooShort(15<16)` 且不替你加 kvbuf／
16 ⇒ 落到 `device=cpu` ＝它试着捕了）、⑬⁷（递下去的材料必须是三档原文）、⑬⁷‴（自动档回执三态）、
⑰⁵（`on` 与 `auto` 各占一格）。三处均做过反向验证（改成 `auto_short=False` ⇒ ⑮⁵′ 精确红；
`graph_slot` 压回 bool ⇒ ⑰⁵ 精确红；`_graph_call` 递 `True` ⇒ ⑬⁷ 精确红）。

**② 尾账一「容量翻倍那一下的分配尖峰」—— 已被预分配收掉，不用改代码、改的是老口径**：
`generate_llm` 现按 `kv_rows=(room, room)` 传下去（`room` ＝ 本轮总 token ＋ `max_new_tokens` ＋
投机余量 ＋ 2），`KvBuf._reserve` 里 `cap = min(max(need, cap*2, 64), ceiling)` ⇒ `ceiling == need`
时**一次就分配到精确需要**，既不翻倍、也没有搬旧行的动作（那一刻 `length=0`）。所以 §7.5.5 结尾那句
「2048 行那一步正好翻到 4096、常驻 288.7 → 576.3 MiB」是**没给 `ceiling` 的老跑法**的读数，
生成路径已不再出现；仍会阶梯跳的只剩「前缀缓存冻结的那份 `past`」（`frozen()` 按实际长度精确分配）。
`ceiling` 顺带把「超预算」变成**具名拒**（要 N 行、上限 M 行，并给出缩上下文/开 q8/调小输出三条出路），
不再让分配器去撞 OOM、把报错甩到几十步之后的算子上。

**②′ 同一个 `room` 在滑窗档上算错过一次（2026-11-09 修）** —— 滑窗的压缩只发生在**两个生成词元之间**
（`_compact`），而 prefill 的提示词段是**整段一次性写进缓冲**的 ⇒ 缓冲峰值是 **prefill 峰值与生成期峰值的
大者**，不是窗口。旧算法在滑窗档只留 `sink + window`（`kv_window=3, kv_sink=2` ⇒ 上限 15 行）⇒ 20 行提示词
在**第 2 个 prefill 块**就报「本次生成要 20 行，而允许的上限是 15 行」，而回执那三条出路（缩上下文／
`kvMode=q8`／调小 `maxOutputTokens`）**一条都不是真因** —— 又一例「假补救：理由指向错误的处理者」，
且同一请求在 `kv_rows` 预分配接上之前（老路边跑边 cat）是**能跑**的。修法 `room = max(total, sink+window)`
（非滑窗档仍是 `total + maxNewTokens`），生成期照样压到 `sink+window = 5` 行 ⇒ 修的是峰值、不是「干脆不压了」。
判据 ⑭¹⁰′（20 行提示词必须能跑 ＋ forward pre-hook 数行数证明生成期被压到 5 行、不滑窗同一条路涨到 19 行），
做过反向验证（去掉 `max` ⇒ 精确报「要 20 行、上限 15 行」）。另有 ⑮⁷′ 钉住 `kv_reserve_rows` **只抬不压**：
`room = max(int(kv_reserve_rows), room)`，写成赋值 ⇒ 第 2 个词元就超预算（而回执仍说「调小 maxOutputTokens」）；
给 1 行与给 64 行都必须产满同样 6 个词元（放大只是白占，不得改变结果）。

**②″ `ceiling` 不是预算闸门（2026-11-09 定论，关掉「边界未量」这条）**：② 里那个「具名拒」的上限**由
`room` 自己派生**（`kv_rows = (room, room)`）⇒ 它挡的是**写端算漏**（投机一拍多写、以后改循环），
**不挡 `room` 本身太大**：直调 `generate_llm` 塞个 1e9 的 `max_new_tokens` 就是真去分配、撞 OOM。
真正的预算闸门是 `LlmBackend._guard_context` 的显存/权重两本账（`kv_room_tokens` ＋ `effectiveTokens`,
报错还分 `by="vram"`／`by="weight"`）—— 生产路径先过那道，直调路径没有 ⇒ 别把这里的 `ceiling`
读成「有预算保护」。这条不需要再量：`generate_llm` 没有暴露 `kv_rows` 参数、内部只有一个赋值点，
「算法漏算」在结构上已不可能，剩下的只有「输入本身越界」这一种，归上层闸门。

**③ 尾账二「q8 原位写仍每次现建临时张量」—— 判定不做，因为账不在那里**：`KvBuf.write()` 的 decode
单步只量化**新算的 1 行**（1 KiB 级 int8 ＋ scale，几微秒，占 22 ms/步 的 0.0x%）。q8 真正的每步成本是
`materialize_kv` 把**整段历史**反量化回来算注意力 —— 探针 ⑦ 第一次把它量出来：**q8 比 fp 每步
×1.29–1.38（+8–10 ms）**，且 256 行与 1024 行**两个点几乎一样** ⇒ 更接近**固定成本**（多出来的那批
kernel），**别按「上下文越长越亏」想当然外推**。⇒ 不做这个微优化，改为把账**成对**写进
`describe()["kv"].note`：省 47% KV 字节 ↔ 每步 +8–10 ms，`q8` 与 `fp` **两个档都写**（省显存不省时间，
显存够就别开）。
