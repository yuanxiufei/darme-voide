# 参考仓审计（`reference/` 各批：引擎侧七仓 ＋ 知识库侧七仓）

> 采集日：§零~§五 2026-11-01（`reference/` 七仓）、§六 2026-11-02（三级内存）、§七 2026-11-04（RAG／知识库七仓）。
> **不采信 README 的宣传语** ✗ —— 每条结论都必须能指到源码行号或盘上文件 ✓。
> 判据只有一条：**这东西在本仓是真通路上的收益，还是只有测试能跑** ✓✗。
> 本仓引擎 = `backend-py/app/services/engine/`（PyTorch，跑 DiT 扩散 ＋ 少量小 LLM，
> Windows 11 ＋ RTX A5000 24G / sm86 / **无 fp8 硬件加速**）。
>
> ⚠️ **盘上现状（2026-11-04 清理）**：`reference/` 整棵是 `.gitignore` 的"学习用拷贝"，**已全部删除**
> （46 个仓、8.5 GiB，其中 `comfy-org/` 独占 7.1 GiB），并且**本仓不再依赖其中任何一个** ✓：
> `app/services/lora_train/paths.py` 的工具根只认环境变量（`MUSUBI_TUNER_ROOT` / `SD_SCRIPTS_ROOT`）
> 与 `configs/model-paths.json`（两处都没配就当场报错、不再往 `reference/` 里兜底 ✓），
> `tests/engine_quant_nvfp4_test.py` 的对拍真源同样从 `MUSUBI_TUNER_ROOT` 取、没配就**如实 skip** ✓。
> 要重跑那两层：把 musubi-tuner 克隆到**仓外任意位置**，再设 `MUSUBI_TUNER_ROOT` ✓。
>
> ⇒ **本文（以及全仓注释、`run_all.py` 台账）里所有 `reference/<仓>/…` 一律是"出处标注"**：
> 指的是**上游项目**的同名路径（谁能拿到自己那份副本就能核），**不代表本仓盘上还有这份拷贝**。
> 要重新核，按下面表格里的 `HEAD` 短 sha 克隆到别处（别放回 `reference/`）。

## 零、三轮查法与结论

| 轮次 | 查什么 | 抓到了什么 |
|---|---|---|
| 第一遍 | 提交／许可／文件量／目录语言构成，README 只当**线索** | 七仓里 5 个是 **LLM/MoE 推理引擎**，与本仓（DiT 扩散）领域错位；先天的可搬面很窄 |
| 第二遍 | 逐仓扫核心源码，列「纯逻辑候选」并标依赖 | 12 条候选，逐条带文件行号；其中 4 条一开始很像能搬 |
| 第三遍 | **回本仓逐一对照已有实现**，能当场否掉的否掉 | **否掉 3 条冗余**（见 §二），**只留 1 条真缺口** ⇒ 本轮实际搬运 = Q5_K 反量化 |

三轮的价值全在第三遍：第二遍给出的 `BLOCK_GEOMETRY`／`expected_bytes`、token 采样、
「模型字节 vs 系统 RAM」三条，本仓**都已经有、而且更严** ✓✗ —— 直接搬会变成第二真源 ✓✗。

## 一、七个仓的实证事实

| 仓 | HEAD | 最后提交 | 文件数 | 许可（原文首行／文件名） | 它到底是什么 |
|---|---|---|---|---|---|
| `Strata` | `c62d5e3a15b4` | 2026-10-31 | 5319 | `MIT License` | 本地跑大模型的**桌面应用**（C++ 引擎 ＋ Python 控制面 ＋ Web 前端） |
| `airllm` | `75a6bb3c41c6` | 2026-10-31 | 107 | `Apache License`（全文） | 逐层流式推理的**方法论仓**（0.1–0.4 tok/s 那种） |
| `Edge0` | `d4a2fa02fcf0` | 2026-10-29 | 1165 | `LICENSE` 11358 字节（Apache-2.0 全文） | SSD 专家卸载 ＋ Recover-LoRA ＋ **prerouter 路由预测** 的流式 MoE |
| `ds4` | `c97964b8aa77` | 2026-10-30 | 3488 | `MIT License`（另有 ggml / DeepSeek 版权行） | **DeepSeek V4 专用**原生引擎（`ds4.c` 不链 ggml，但复用其量化表） |
| `pulsar` | `18f990094e10` | 2026-09-01 | 114 | `MIT License` | Rust＋CUDA 的巨型 MoE（专家按 token 从 NVMe 流式读） |
| `expert-stream` | `89275f1c50d2` | 2026-08-16 | 194 | `MIT License` | Windows 原生 MoE 流式（`engine/expert-stream.c`） |
| `moe-stream` | `b3024b332505` | 2026-02-23 | 53 | `LICENSE-APACHE` ＋ `LICENSE-MIT`（双许可，二选一） | macOS Metal 的 SSD 流式 MoE ＋ OpenAI 兼容外壳 |

平台面一句话：**这七仓没有一个是为 DiT 扩散写的** ✓。它们的收益全部来自「每 token 只用
一小撮专家」这个前提 —— 而本仓每步前向都要**全部** DiT 权重 ✓✗，热度的概念在这里不存在 ✓。

## 二、第三遍否掉的三条冗余（看着能搬，其实本仓更严）

| 候选（出处） | 本仓已有 | 为什么不能再搬 |
|---|---|---|
| `Strata/tools/gguf_reader.py:21-84` 的 `BLOCK_GEOMETRY` ＋ `expected_bytes` | `gguf.py::TENSOR_TYPES`（类型/块元素/块字节）＋ `tensor_nbytes()` ＋ 截断与重叠校验 ＋ `dtype_counts` | 本仓已经拿它做了**三件上游没做的事**：截断检测、块边界重叠检测、按类型计数。搬上游那份等于再造一个类型表 ✓✗ |
| `ds4.c::sample_build_probabilities` 的 top-k/top-p/min-p 与「证明尾巴被截掉」快路径 | `llm.py::generate_llm` 已有 temperature / top-k / top-p ＋ fp32 稳定 softmax | 动机不成立：上游是 C 单线程在 15 万词表上排序才需要「有界堆 ＋ 可证截断」；本仓是 torch 张量，排序成本可忽略 ✓。**搬了只有维护成本** ✓✗ |
| `moe-stream/src/config.rs:376-443` 的「模型字节 vs 系统 RAM」双资源判定 | `scheduler.py::ensure_host_room`（宿主内存闸），**已接在真装配线的四个点上**（`torch_backend.py` 的 CPU 决策 / split 的 CPU 一半 / int8 钉卡） | 本仓的判定比上游更细：显存侧按**两本账**（`PART_NEED_GIB` 分时 vs `CHAIN_NEED_GIB` 整链同驻）算，宿主侧另有 `InsufficientResources` 具名拒绝 ✓ |

## 三、逐仓结论（按审计顺序）

### 1. `Strata` —— **搬了 1 处（Q5_K 公式）**

* **搬走**：`include/strata/artifact/dequant.hpp:212-234`（`dequantize_q5_K`）⇒ 本仓
  `gguf_dequant.py::_dequant_q5_k`。它是一份 llama.cpp `dequantize_row_q5_K` 的逐行转写（MIT），
  第三遍与 `ds4.c` 的标量版**两份独立转录互核**后才算数 ✓。
* **为什么非补不可**：盘上 `diffusion_models/Krea2/krea2_turbo-Q4_K_M.gguf` = 432 张量，
  其中 **63 张是 Q5_K**（F32 167 / F16 1 / Q5_K 63 / Q4_K 201），而 `general.file_type` 明写
  `Q4_K_M` ⇒ 「M 后缀只含 Q4_K＋Q6_K」是错的口径 ✓✗；不补 Q5_K，这个文件**装不进** ✓。
* **没搬**：`plan/plan.hpp`（预算与「算不平就拒绝」）、`program/prefill_plan.hpp`（chunk/ring 成本模型）、
  `prefill/kva_floor.hpp`（LRU＋最长公共前缀地板）、`core/expert_cache.cpp` / `core/expert_nibble.hpp`
  （MoE 专属）。理由：前三件的前提在本仓不成立（预算闸已有、本仓按张量读没有 chunk、
  本仓 LLM 无跨调用 KV 复用），第四件是 MoE 专属 ✓。

### 2. `airllm` —— **无第二块**

上一轮已搬 `_prefetch_stream` / `_materialize` / `_evict` 三处语义（`prefetch.py`，
由 `weights.load_module_weights(read_ahead=2)` 调用）。本轮复查：其余是逐层 `state_dict`
挂载、`ModelPersister`、bitsandbytes 压缩 —— 全是「按 token 逐层重读整模型」范式的零件，
而那正是它 0.1–0.4 tok/s 的来源 ✓✗。**不搬**。

### 3. `Edge0` —— **无一处可搬**

预取、缓存、回落三件都以「**下一层要哪些专家**」为前提（`python/edge0/moe/routing.py` 输出路由），
本仓 DiT 每步用全部权重 ⇒ 预测无处可用 ✓。`streaming/cache.py` 的 `SharedExpertCache` 是单预算
LRU，本仓 `scheduler.Ledger`（显存账本）＋ `reclaim_for`（按相位回收）已经更强 ✓；
`streaming/layer.py` 的五级回落链里「预测失败 ⇒ **填零**」这一手尤其不能学 ✗✗（会静默算错还不出声）。

### 4. `ds4` —— **只贡献第二份转录（互核用）**

* 供了 `dequantize_row_q5_K` 的标量版（与 Strata 那份独立）⇒ 用于逐位对拍 ✓。
* **没搬**：KV cache 的 fp8/E2M1 量化 ＋ `hadamard128` 旋转（本仓 LLM 没有 KV 量化通路，
  且 9B TE 的 prompt 很短，收益≈0）、`ds4_ssd.c:108-140` 的缓存预算（↔ `device_policy`）、
  `ds4_layer_pack.c:14-45` 的层放置（↔ `residency`，同为「单调连续前缀 ＋ 溢出到 CPU」）、
  `ds4_kvstore.c` 的磁盘 KV 淘汰打分（本仓无磁盘 KV）。

### 5. `pulsar` —— **无一处可搬**

它的放置算法（`census` 系列）依赖**跨请求的专家热度分布**；本仓每步用全部权重 ⇒ 热度恒等，
按热度排序退化成任意序 ✓✗。`hybrid_miss` 的 `q*`（带宽比）以「同一个 token 二选一：H2D 还是
CPU lane」为前提，本仓是「按层整段切分常驻/流式」，不是逐 token 选择 ✓。
本地 `residency.py` 的贪心连续前缀在这个前提下**已经是最优形态**（层同构 ⇒ 谁上卡只由字节数决定）✓。

### 6. `expert-stream` —— **无第二块**

已搬 `engine/expert-stream.c` 的 Win32 三件（`NO_BUFFERING|OVERLAPPED`、一次提交多笔 ＋ 每槽
手动复位事件、`VirtualAlloc` 扇区对齐槽区 ＋ `slop`）⇒ `win_io.py`，由 `safetensors.py::TensorReader`
调用。本轮第二块候选是 `tools/analyze_reuse.py` / `analyze_predictors.py`（LRU / Belady /
EWMA 优先级的淘汰仿真）：**不搬** —— 它的输入是 MoE 路由 trace（盘上没有，本仓也不产生）；
而本仓相位是固定的 encode→sample→decode，最优常驻集手算即得（22.49 GiB 卡上
DiT 13.26 ＋ TE 14.79 = 28.05 装不下 ⇒ 只能「DiT 常驻、TE 与 VAE 各换一次」），
仿真不会产生新结论 ✓✗。

### 7. `moe-stream` —— **无一处可搬**

三模式自动选择 ↔ `device_policy.decide`（本仓多一层两本账 ＋ 四档策略 ＋ 可解释 `reason`）；
`experts_per_layer` 元数据解析 ↔ 本仓不跑 MoE；「Q4 matmul 跳过反量化」必须手写 kernel，
本仓是 torch 表达 ⇒ 无动机 ✓。双资源判定见 §二第三行 ✓。

## 四、本轮搬运的验收

| 判据 | 位置 | 结果 |
|---|---|---|
| 手造块精确值（`sc=mn=63`：第 5 bit 全 0 ⇒ `31.5`、全 1 ⇒ `1039.5`，列级对照） | `tests/engine_gguf_dequant_test.py` ⑤ 系列 | 7 条全过 |
| 与标量参考版 **4097 块随机字节逐位对拍**（两版写法刻意不同） | 同上，`_q5_k_reference` | 偏差 0 |
| 盘上真文件：432 张量里 63 张 Q5_K，白名单覆盖率 **100%**、真张量有限非零 | 同上，`case_real_file_q5_k` | 通过 |
| 反向验证：抽掉第 5 bit ⇒ 5 条判据同时报；位号写成常量 0 ⇒ ⑤‴ 与对拍报 | 手工反验后复原 | 判据有真咬合 |
| 未实现类型仍具名拒绝（Q2_K 10 / Q3_K 11） | 同上 ④ 系列 | 通过 |

收尾时连跑的相关回归（都绿）：

| 测试 | 结果 |
|---|---|
| `engine_gguf_dequant_test` | 35/35 |
| `engine_gguf_test` | 23/23 |
| `engine_gguf_to_llm_test` | 30/30 |
| `engine_readahead_test`（上一轮从 airllm 搬的预读） | 10/10 |
| `engine_win_io_test`（上一轮从 expert-stream 搬的 `FILE_FLAG_NO_BUFFERING`） | 0 failed |
| `engine_licenses_test` | 13/13 |
| `engine_env_docs_test` | 31/31 |
| `structure_test`（305 模块全点名） | 33/33 |

两个**顺手补掉**的既有缺口（都不是 Q5_K 引入的，是跑回归时撞出来的）：

1. `engine_licenses_test` ①（权重表 == 清单条目）报 `dit_krea2_turbo` / `te_krea2_qwen3vl4b` / `vae_krea2`
   三条漏登 —— Krea2 三件套一直只在 `configs/models.json` 里，`THIRD_PARTY_NOTICES.md` §一没跟上（26→29）。
   补的是**登记**不是许可结论：三件仍写「未声明」✓。这条判据的价值就在这里 ——
   本表是下游判断再分发条款的唯一入口，漏一条等于那件权重在合规面上不存在 ✓✗。
2. `engine_win_io_test` **直接跑会崩在 print 上**：首行 `✓`（U+2713）在 cp936 控制台解不出来 ⇒
   `UnicodeEncodeError`，一条判据都没跑就退 1。仓内其它测试都有
   `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` 兜底，只有这个文件漏了 ⇒ 已补。
   ⚠️ 注意它与 `run_all.py` 下是绿的：`run_all` 自己的 stdout 已重配过，所以这个坑**只在单独跑时露头** ✓✗。

## 五、剩下的账（明确不搬，不是待办）

`Q2_K` / `Q3_K` 未实现 ✓、且**不打算补**：盘上 7 个 GGUF 一件都没有这两种块
（实测 `dtype_counts` 只有 F32/F16/Q8_0/Q4_K/Q5_K/Q6_K）⇒ 补了就是死料 ✓✗。
真需要时的入手点已写清：`ds4.c` 的 `dequantize_row_q2_K` 与 Strata 的 `Q3_K` 分支，
两者都只需复用现成的 `_q4_k_scale_min` 变体（Q3_K 的 6-bit 是**有符号**且带 −32 偏置，
这是它不能照抄 Q4_K 的唯一理由 ✓）。

## 六、第二轮：三级内存（GPU／CPU／SSD，2026-11-02）

提问是「`reference/` 里哪些能参考 GPU＋CPU＋SSD 三层内存」。结论：**只有两个仓有真实现**，
同一批被问到的 `CogVideo` / `Open-Sora` **一行都没有**。

### 1. 逐仓实证（不是按 README 猜）

| 仓 | 三级分层的实现 | 证据（绝对路径在 `reference/<仓>/` 下） |
|---|---|---|
| `strataflow` | **完整且成体系**：放置规划器 ＋ 三层缓存 ＋ 无缓冲异步 IO ＋ KV 分页 | `src/plan/planner.h`（234 行）／`src/tws/weight_store.h`／`src/tws/async_io.h` |
| `Strata` | **运行时机制最细**：前瞻换显存槽 ＋ 直接读 ＋ 专家缓存／淘汰／锁页 | `core/foresight_swap.hpp`／`platform/direct_file.hpp` |
| `Edge0` | 有，且**只在 Windows 这一侧**：工作集硬帽 ＋ 私有页热池 ＋ 头驱动预取 | `windows/serve/mem_budget.h`／`windows/serve/prefetch.h` |
| `CogVideo` | **无**：一行 ``pipe.enable_sequential_cpu_offload()`` | `inference/cli_demo.py:150` |
| `Open-Sora` | **无**：全仓 offload 零实现 | `opensora/utils/inference.py` 搜不到 |

⇒ `CogVideo` / `Open-Sora` 属于「调 diffusers 现成 API」，代码里没有分层逻辑。
它们唯一值得借的不是内存分层，而是 ``vae.enable_slicing()/enable_tiling()``（**激活分块**）——
「VAE 落 CPU 慢」的正解可能是分块上卡，而不是把整份 VAE 留在内存。
⚠️ 这条**已经实测否掉**（只对 Qwen-Image 那族）✓：见本节末尾「6.『分块上卡』的实测结论」✓。

### 2. 本仓其实已经是三级，缺的是第二级的**缓存**

| 层 | 本仓 | 差什么 |
|---|---|---|
| GPU 显存 | `residency.place()` ＋ `split/off/park` 三档 ＋ **层预取上卡**（`foresight.py` ＋ `residency._SlotBook`，2026-11-02 补） | 「热池深度可调」＝ `DARME_FORESIGHT_SLOTS`（默认 0 槽＝这一级不存在） |
| CPU 内存 | 只放「常驻的那部分层 ＋ 锁页」 ＋ **带 LRU/按层配额的 `host_cache.py`**（2026-11-02 补） | 已补 |
| SSD | `win_io.py` 的 `FILE_FLAG_NO_BUFFERING` ＋ `OVERLAPPED` 多笔并发（从 `expert-stream` 搬，实测乱序 +23%） | 与 `Strata` / `strataflow` 的 `direct_file` / `async_io` **是同一件事** ⇒ 不需要再搬 |

`weights.py` 抬头那句「19.53 GiB 走页缓存等于同时占 19.53 GiB 内存」正是这套 IO 的设计前提：
**绕开页缓存 ⇒ 读盘不占内存**。所以「内存占用低」是设计结果、不是异常。

### 3. 本轮搬了 4 处（搬**口径**，没有逐行转写；③④ 即 §5 那条过期结论的落地）

1. `strataflow/src/plan/planner.cpp` 的**预算分解 ＋ 拒绝语义** ⇒ `residency.decompose()` / `Budget`
   * ``planned_peak = trunk + kv + compute + process_floor + safety_margin`` ⇒ 本仓
     ``effective = usable − margin − fixed``；
   * `safety_margin = max(256 MiB, usable/100*6)` ⇒ `MIN_MARGIN_BYTES`，但**只在 `reserve > 0` 时生效**：
     `reserve=0` 在本仓的 A/B 测速与判据里是「我明确不要余量」，补下限等于改调用方的决定；
   * 「算不平就拒绝」⇒ `Plan.refused` ＋ `place(on_refuse="raise")`。
   * ⚠️ 与 §三.1「没搬 `plan/plan.hpp`」的关系：那时判的是「预算闸已有」——本仓确实有**闸**
     （`device_policy.decide` ＋ `scheduler.ensure_host_room`），但**没有分解**：
     装不下时只能报一个总数，说不出这笔预算是被余量吃掉的、还是被 `fixed` 钉死的。这是本轮补的差。
2. `Edge0/windows/serve/mem_budget.h` 的 `--mem-budget-mb` ⇒ `scheduler.limit_working_set()`
   * `SetProcessWorkingSetSize` 把已在工作集里的页挤出去 ⇒ 下次访问**真回盘**；
   * 用途只有一个：让「降级到 SSD」在测试里**走得到**。不压帽，工作集/页缓存把整份权重兜住
     ⇒ 测出来的「三级分层」其实还是内存速度（假绿）；
   * 纪律判据：`case_working_set` 用 AST 扫 `torch_backend` / `runtime` / `pipeline` / `weights` /
     `loader` **一个都不许调**它（调了就是每步回盘）。
   3. `strataflow/src/tws/weight_store.h` 的**按层配额淘汰** ⇒ `engine/host_cache.py`
    （**逐层配额 ＋ LRU**，不是全局 LRU；由 `torch_backend._host_cache_of(what)` 按件接上）。
    判据全在 `engine_host_cache_test`（37 条），含「中间层扫过去不许把前面层的热数据全踢掉」。
   4. `Strata/core/foresight_swap.hpp` 的**前瞻换槽** ⇒ `engine/foresight.py` ＋ `residency._SlotBook`
    （独立 stream 拷、`Event.query()` 没落地就不使用、槽显存在预算外所以先从预算扣、
    扣完太亏就不启用）。判据全在 `engine_foresight_test`（26 条）。

### 4. 验收

| 判据 | 位置 | 结果 |
|---|---|---|
| 分解三段之和 == `usable`（不多不少一字节） | `engine_residency_test` ⑦ 系列 | 55/55 |
| `reserve=0` ⇒ 余量 0（下限不偷偷生效） | 同上 | 通过 |
| 小卡 × 文本塔 5% ⇒ 30 MiB 补到 256 MiB 下限 | 同上 | 通过 |
| `fixed` 只扣 `effective`、不动 `margin` | 同上 | 通过 |
| `refused` 与 `budget<=0` 是**两个**布尔 | 同上 | 通过 |
| `on_refuse="raise"` 真抛 `BudgetRefused` | 同上 | 通过 |
| 压帽后工作集读数**真的**变小 | `engine_scheduler_test` ⑦ 系列 | 52/52 |
| 压帽不进正常装载路径（AST 扫 5 个文件） | 同上 | 通过 |
| 逐层配额不会被中间层把前面层的热数据踢光（＝上游存在的理由） | `engine_host_cache_test` | 37/37 |
| 预取「没落地就不用」：替身 Event 永不 landing ⇒ 一次不 swap_in、`pending>0`、数值不变 | `engine_foresight_test` | 26/26 |
| 槽显存在**预算外** ⇒ 先扣、扣完太亏不启用（防首次预取 OOM） | 同上 | 通过 |
| 预取抛错 ⇒ 当场 `slots=0` 且**前向照原路跑完** | 同上 | 通过 |

反向验证：把 `reserve=0` 也补下限 ⇒ `⑦ reserve=0 ⇒ 余量 0` 报；
把 `refused` 与 `budget<=0` 合成一个布尔 ⇒ `⑦ 没有显存 ⇒ refused=False` 报。

### 5. 当初「明确没搬」的两条，现已全部落地（2026-11-02 晚更新）

⚠️ 这一节原来写着「两件都别做」，**已经过期** ✓ —— 判据变了（清单要求做三级内存的完整形态），
两件都按要求补上了，而且都按参考实现的**模型**搬、没有自造：

* `Strata::core/foresight_swap.hpp`（前瞻换显存槽）⇒ 落地 `engine/foresight.py` ＋
  `residency._SlotBook`：每层额外的**槽** ✓、独立 `torch.cuda.Stream` 拷 ✓、`Event.query()`
  **没落地就不用**（退回 CPU 算）✓、用完 `swap_out` 还槽 ✓。原来这条写的是「本仓 `attach` 是
  原地换设备、没有槽位概念 ⇒ 搬它等于重写 `attach`」—— 现在就是**照这句话**扩的 `attach`：
  槽与「原地换设备」并存，`place(..., foresight=...)` 不给 ⇒ 老路径逐位不变 ✓。
  与上游同纪律：默认 **0 槽＝这一级不存在**（`STRATA_FS_SLOTS > 0` 的对应），
  `DARME_FORESIGHT_SLOTS` 才开。
* `strataflow::tws/weight_store.h` 的**按层配额淘汰** ⇒ 落地 `engine/host_cache.py`：
  每层配额 ＋ LRU，**不是**全局 LRU ✓（上游那句「逐层扫的访问模式下全局 LRU 会被中间层把前面层
  的热数据全踢掉」正是它存在的理由，照搬了这条**判据**）。第二轮担心的「本仓还没有第二级缓存
  ⇒ 建了是死料」不再成立：缓存已由 `torch_backend._host_cache_of(what)` 按件接上 ✓。

仍然**不搬**的只剩两条：

* `Edge0` / `Strata` 的**代码形态**（依赖 llama.cpp patch / ggml）：只能借规则，不能借代码。
  `Edge0/windows/serve/prefetch.h` 那份「头驱动预取」尤其不能借 ✓：它的前提是
  「路由头预测下一层要哪些**专家**」，本仓 DiT 每步用全部权重 ⇒ 无「预测」可言 ✓✗
  （已在 §三.3 判过；本仓的预取是**确定性**的下一层，不需要预测器）。
* 本轮没有逐行转写的代码 ⇒ `THIRD_PARTY_NOTICES.md` 不新增条目（§五 的登记仍只有
 `Strata` `c62d5e3a15b4` 与 `ds4` `c97964b8aa77` 两条）。

### 6. 上面第 1 节那句「分块上卡」的实测结论（2026-11-02 加：**潜变量整段切**已否 ✓，up 尾段分块已做 ⇒ §7 ✓）

原话是「本仓『VAE 落 CPU 慢』的正解可能是分块上卡」——动手实测后**否掉** ✓：

* `vae_tiling.py` 本身没问题 ✓（真跑 24 条，含「解码器改了通道数就必须显式给 `out_channels`」✓）；
* 但**不能拿去切 Qwen-Image 那一族** ✗：`vae_wan` 的解码器 `decoder.middle` 是**全图单头
  自注意力** ✓（`AttentionBlock`）⇒ 潜变量一切块，注意力就只看得到自己那块，拼回来**整张**都变 ✓✗。
  真 2.1 权重（238 键）实测：10×10 潜变量按 6×6 切 ⇒ 相对差 **1.35**；按 16×16（＝整张、
  等于没切）⇒ **0**。小形状同结论，`engine_vae_wan_test` ⑨ 现在**反向钉住**这条。
* ComfyUI 的 `WanVAE` 同样**没有** `decode_tiled`（`comfy/ldm/wan/vae.py` 只有 `decode`）⇒
  不是本仓偷懒 ✓。
* ~~顺带核过「到底缺不缺显存」：现存最大画布 2048×1536 ⇒ 整张解峰值几百 MB，**用不着**分块~~
  —— **这句是错的** ✗✗，2026-11-02 晚按真权重、真画布（bf16、单帧、卡 22.49 GiB）重测：
  2512 的 8× VAE 在 2048×1536 就要 **20.96 GiB**（93% 卡）、3072×2304 直接 **OOM**；
  2.1 的 16× 在 2048×1536 是 8.69 GiB、3072×2304 是 18.75 GiB。原数是从**小画布**外推的 ⇒
  差了两个数量级（见 §六.7 的实测表）。
* 真要省显存只能切「注意力**之后**的那几级 up 块」✓（后面纯局部卷积 ＋ 逐像素 RMSNorm）——
  这条已经**做了** ✓，但**不是**「带 halo 就逐位」✗（见 §六.7）。

### 7. VAE 解码省显存的两个旋钮（2026-11-02 晚落地）

**为什么做**：§六.6 里「用不着分块」的实测是错的（小画布外推）。真值按真权重、真画布重测
（bf16、单帧、卡 22.49 GiB、`_tile_probe*` 量 allocator 峰值）：

| VAE | 画布 | 整张 | `attn_chunk=1024` | `＋tile=512` |
|---|---|---|---|---|
| 2.1（16×，64ch） | 2048×1536 | 8.69 GiB | 8.69（**没用** ✗） | **3.21 GiB** |
| 2.1（16×，64ch） | 3072×2304 | 18.75 GiB | 18.75 | **3.28 GiB** |
| 2512（8×，16ch） | 2048×1536 | **20.96 GiB** | **3.94 GiB** | **1.03 GiB** |
| 2512（8×，16ch） | 3072×2304 | **OOM** | — | — |

⇒ **两个旋钮各治一半、缺一不可**：8× 那族的峰值几乎全在**全图注意力**（只切 up 链一点不降 ✗），
2.1 那族几乎全在 **up 链**（注意力只占 ~0.6 GiB）。

**落地**（`engine/vae_wan.py` ＋ `engine/torch_backend.py`）：

* `Decoder3d.forward(x, *, tile, halo, attn_chunk)`：`conv1 + middle` **整张**跑（那之前有全图
  注意力，切了就是重排），之后按 `tile` 切 up 链、带 `halo` 内圈裁剪；`pad = max(halo,
  receptive_latent())` —— **不信调用方**给的 halo（给 0 也抬到感受野下限）。
* `AttentionBlock.forward(x, chunk)`：按 **query** 分块（softmax 每行独立 ⇒ 数学上无近似）。
* `WanVAE.decode(z, *, tile, halo, attn_chunk)`：`tile` 是**像素**口径（与 ComfyUI
  `VAEDecodeTiled` 同），内部换算成 latent 并按 `patch_size` 对齐。
* `TorchBackend.set_vae_tiling(tile=, halo=, attn_chunk=, temporal_chunk=, fallback=)` /
  `reset_vae_tiling()`；`decode()` 的回执多一格 `vaeTiling`（`None` ⇒ 全关、逐位；非空 ⇒ 写明档位与
  `reason`）。`temporal_chunk` 是**时间维**档、只对 SEEDVR2 那族有效（§六.8），**与整段逐位相等** ✓。
* **外部入口也已开**（同一轮，2026-11-02）：`POST /engine/load` 收 `vaeTiling`（也认
  `vae_tiling`），三种写法一个意思（`{"tile":512,"attnChunk":1024}` / 快照形 / `null`）。
  归一只有一处 `torch_backend.normalize_vae_tiling`，装配线、路由白名单、回执三个消费点都调它
  ⇒ 不会出现「HTTP 认一套、Python 层认另一套」。非法值（拼错的键、`"big"` 这种非数字）
  ⇒ **400**（`StageError`），不许落到 500；字符串数字（`"768"`）照收。
  全关 ⇒ 该键**不进装载键**（老请求的复用行为零变化）；一旦给了值 ⇒ 进键 ⇒ 改旋钮要重装一次
  （回执 `reused=false` 会如实说，不装成"只是换了个开关"）。
* **装配回执**里写 `report["vaeTiling"]`（挂在 `_commit_load` —— 所有装配线共用的唯一人口）；
  SDXL 那条（`_ensure_image_loaded`）没有这个旋钮 ⇒ 配了会明确写 `reason="not-applied"` 并说明，
  不静默吞掉。

**两条都「不是逐位」** ✗（所以默认关 ✓）：

* `tile` 实测相对差 **5.8e-03**；**加 halo 消不掉**（16→24 仍是 5e-2 量级）——
  接缝算术是精确的，变的是 **cuDNN 对不同窗口尺寸选了不同卷积算法** ⇒ 归约顺序变；
* `attn_chunk` 只切 query、数学上等价，但 **SDPA 内部自己也按 query 分块** ⇒ 只有 chunk 与它
  对齐时才逐位（2.1 上 4096 逐位、1024 差 8.55e-03）。

**默认策略**（这条是本轮定的纪律）：两个**改数值**的旋钮默认关 ✓；`fallback`（OOM 时才降档重试
一次，回执 `reason="oom"`）默认**开** ✓ —— 它只在「本来就要崩、出不了图」那条路上生效，装得下时
逐位不变，所以对「同 seed 同文本逐像素复现」零影响。降完仍 OOM ⇒ 报成一句人话
（点名两个档位 ＋ 指路降分辨率），不吞成原始 aten 串。时间档 `temporalChunk` **不动数值**，但
**也默认关** ✓ —— 理由不是画质而是纪律：关＝走与整段完全同一条代码路径，连"分块拼回来"这一步都
不进，复现性零风险；要省显存时显式开，实测与整段相等（§六.8）。

**验收**：`tests/engine_vae_wan_test.py` ⑩（旋钮本体 10 条：默认逐位、halo 不许信调用方、
SDPA 真切成 4 批…）＋ ⑪（后端接线 11 条：默认不开、OOM 才降档、**非 OOM 的错必须原样抛**、
`fallback=False` 就得真报错）⇒ `SUMMARY: 96/96 passed`；`tests/engine_router_contract_test.py` ⑪¹³/⑪¹⁴/⑪¹⁵
（HTTP 透传、两种键写法、`0`/负数当关、全关不进键、拼错键与非数字 ⇒ `StageError`；⑪¹⁵ ＝ 时间档
`temporalChunk` 两种写法 ＋ `0` 即关）⇒ 36/36；`tests/engine_runtime_test.py::case_vae_tiling_wiring`
（**静态 AST** 5 条：七条装配线**都收**参数且**都传给** `TorchBackend(...)`、`_ensure_loaded_internal`
七处分派**每一处**都带、两处 `_load_key` 都带、`_commit_load` 回吐实值、`decode` 的 SEEDVR2 分支
**真读** `_vaeTiling["temporalChunk"]` 并**真传给** `decode_latents` —— 这类「某档位配了没生效」
运行期测试**撞不到**，`int8_offload` 2026-10-29 就是这么丢的）⇒ `SUMMARY: 83/83 passed`；时间档本体
在 `tests/engine_seedvr2_vae_test.py` ⑫⑬ ⇒ `SUMMARY: 105/105 passed`（详见 §六.8）。反向验证过：抽掉
Z-Image 装配那处的传参 ⇒ 第一条判据点名 `_ensure_zimage_loaded`；抽掉 SDXL 分派那处的 `vae_tiling`
⇒ 第二条判据报红；抽掉 `decode_latents(temporal_chunk=…)` 那个关键字 ⇒ 第五条判据报红；把「只认真
OOM」那道闸门拿掉 ⇒ ⑬ 第二条报红；复原后全绿。★这条判据自己误红过一次：它原只认
`self._vaeTiling.get("temporalChunk")`，而后端为了「一次快照、逐项读稳」改成了
`_knobs = dict(self._vaeTiling); _knobs["temporalChunk"]` ⇒ 判据把**更稳**的写法判成「没读旋钮」✗
（症状就是红 + 详情里 `读旋钮: 0`）。已改成三种等价写法都认 ✓ —— **判据盯语义不盯写法** ✓✗。

**仍然没有的**：① 没有抄 `vae_tiling.py` 那套通用于其它 VAE 的切块逻辑（那是给**无全局注意力**
的解码器用的，这一族用不上）；② `vaeTiling` 的空间档（`tile`/`halo`/`attnChunk`）只作用于
`WanVAE` 这一族（2.1 / 2512 / Wan2.2 / Krea2 共用），SDXL / H3 有声明的「不生效」回执，尚未接入；
③ 时间档 `temporalChunk` 只对 `SEEDVR2` 那族有效（按族分派，给错族就是闲着，不报错）。

### 8. SEEDVR2 VAE 的**时间维**分块：**已抄进来**，与整段逐位相等（2026-11-02 晚）

动机是真账（§10.1g 那个 720×480 / 49 帧的单跑）：整段解码 **336 秒**、分配器峰值 **47.60 GiB**
（卡 22.49）⇒ 全程在 OOM 重试里打滚。空间维那套（`_forward_tiled` ＋ halo）在这族**用不上** ✗——
它的峰值来自**时间 ×2 上采样之后**的重排张量，与空间分辨率无关 ✓，所以只能动时间维。

**先纠正一条旧结论** ✓✗：`vae_seedvr2.py` 原文档与本档 §10.1g 都写着「`_remove_head` 的『首帧
特例』让每块第一个潜帧被当成整段首帧 ⇒ 相位少 3 个像素帧」——**这条是猜的，实测不成立** ✗。

按「补足上文 ＋ 正确丢帧」先试了四个变体，**全部与整段解码差 1.8~2.8** ✗（留作反面教材）：

| 变体 | 结果 |
|---|---|
| 带 1 帧上文、丢那段上文的输出（旧版） | 最大差 1.8~2.8（旧账，差的数量级原样重现 ✓） |
| 按因果相位把丢帧数改成 `4k-3`（k＝上文帧数，k=1 时只丢 1 帧），k 试到 6 | 仍是 1.8~2.8 ✗ |
| 非首块整个跳过 `_remove_head`（改丢 `4k`） | 仍失败 ✗ |
| 对输出做 **±8 帧平移扫描**找逐位对齐 | **一帧都找不到** ✗ |

**真根因**（用因果性判定法实测：只扰动某一输入帧，看输出哪些帧变）：

* 只改**末帧** ⇒ 输出只变**最后 4 帧** ⇒ 链是**严格因果**的 ✓；
* 只改第 **8** 帧 ⇒ 输出从第 `4·8-3 = 29` 帧起**一直变到末帧**（29..60 全变）✓。

⇒ 问题不在丢帧数，在**块首帧的左 padding 是伪造的** ✓✗：`_InflatedCausalConv3d` 用
`_extend_head` 把**张量最左端那一帧复制 2 份**当左 padding —— 整段跑时它正好是真首帧 ✓ 正确；
块跑时它变成**块首帧** ✗ ⇒ **块首帧自己的输出就与整段不同**，而因果链把它送到块内**全部后续帧**
✓。所以「块首错一点点」＝「整块全错」，**补多少上文都救不回来** ✗（补进来的上文帧自己也被当成
块首 ✓）；「只丢几帧就够」这条路从原理上就不成立 ✓。

**抄的就是 comfy 的这条正解** ✓（`comfy/ldm/seedvr/vae.py` 的 `MemoryState` ＋ 逐层
`memory_cache`，见 `extract_head`/`extend_head(memory=...)`/`slicing_decode`）：把**块外真实帧**
接到块首当左 padding（长度仍是 `|stride − kernel|` ⇒ 输出帧数不变 ✓），并且**逐层**带 —— 每一层
都有自己的 `_extend_head`，只在输入层补不够 ✗。落点与对外口按本仓口径重写（只搬机制 ✓）：

| 落点 | 内容 |
|---|---|
| `_MemoryState(DISABLED/INITIALIZING/ACTIVE)` | 整段老路 / 第一块（复制首帧但**要记**缓存）/ 后续块（左 padding 取缓存） |
| `_InflatedCausalConv3d.forward(…, memory_state, memory_cache)` | 记的是**本层输入**的最后 `stride−kernel` 帧（记输出就错 ✗） |
| `_Upsample3d.forward` | `ACTIVE` 时**不丢**第 2 帧（那块首帧不是真首帧），只有非首块时间 ×2 才有修正 |
| `_ResnetBlock3d`/`_Downsample3d`/`_MidBlock3d`/`_DownBlock3d`/`_UpBlock3d`/编解码器 | 逐层显式下传（`nn.ModuleList` 写 `up(x)` 就丢状态 ✗） |
| `SeedVR2Vae.decode(temporal_chunk=…)` / `decode_latents` | 首帧单拎 ＋ 其余按潜帧切块拼回；`None`/`0` ⇒ 老路 |
| `TorchBackend.set_vae_tiling(temporal_chunk=…)` → `vaeTiling.temporalChunk` | 唯一外部旋钮，**默认关** ✓（= 同一条代码路径 ⇒ 复现性零风险） |

**验收（`tests/engine_seedvr2_vae_test.py` ⑫⑬，105/105 全绿 ✓）**：两种时间倍数（×2 / ×4）× 四个块大小
（1/2/4/6 潜帧）都「帧数 `4T-3` 相同 ✓ ＋ `max|Δ| ≤ 1e-5` ✓」；**真权重（250 键）**也在内 ✓
（`≤1e-4`）。残下的 ~1e-6 是 fp32 卷积在不同 `T` 上选的累加顺序不同 ✓，与"块首伪造"那种
`1.8~2.8` 差六个数量级 ✓。另有一条**反向**判据：块切法照旧、只把 `memory_cache` 换成空的 ⇒
数值必须崩 ✓ —— 这条要是也过，就说明缓存根本没接上 ✗。

**OOM 兜底也落在它身上** ✓（⑬ 十三条）：`fallback` 在**这一族**降的是 `temporalChunk`
（阶梯 `chunk_ladder()` ＝ `4→2→1` ✓），**不是** `tile` ✗ —— 这一族的峰值来自时间 ×2 上采样后的重排，
空间切块救不了；而且这条兜底**不换画质** ✓，回执多一格 `"exact": true`（Wan 那族降档是换字节 ✗，
两边读回执的方式必须分开 ✗✗）。判据只接**真 OOM**（`_is_oom`）：非 OOM 的错原样抛、一次都不重试、
`fallback=false` 时 OOM 也真报错、**整条阶梯试遍**仍 OOM 才给人话「降帧数/分辨率，别降精度」
（写到 1 档就是「一块只 1 像素帧」还不行 ⇒ 真的是帧数/分辨率超了 ✓）。反向验过：把那道
`_is_oom` 闸门拿掉 ⇒ ⑬ 第二条当场报红，复原后 105/105。

#### 8.1 真跑（2026-11-06，720×480、13 潜帧＝49 帧、A5000 22.49 GiB、单开进程各量一档）

| 档位 | 耗时 | 峰值 allocated | 峰值 reserved |
|---|---|---|---|
| 整段（默认） | 496.2 s（另一次 339.3 s） | — | **95.74 GiB** |
| `temporalChunk=4` | 231.1 s | 19.51 GiB | 86.67 GiB |
| `temporalChunk=2` | 154.3 s | 11.94 GiB | 38.65 GiB |
| `temporalChunk=1` | **8.3 s** | 8.15 GiB | 17.47 GiB |

三条结论都是「不真跑就写不出来」的：

1. **这档不报 OOM，它溢出** ✗✗：`allocated` 只有 19.51 GiB（看着装得下 ✓），`reserved` 却是 86.67 GiB
   —— Windows 驱动把超出的部分落到了**主机内存** ⇒ 完全不抛 OOM，只慢 **28 倍**（8.3 s → 231 s）✓✗。
   ⇒ 只等 `_is_oom` 的兜底**永远等不到触发** ✓，必须另立判据。
2. **自建缓存原先钉住了整块激活** ✗✗：`memory_cache[self] = x[:, :, mem_size:].detach()` 里那个
   `.detach()` **不复制存储** ✓ ⇒ 缓存是**整块激活的视图** ⇒ 每层都把这一块的整份激活吊住不放 ✓，
   症状是「分块比整段还费」（7 潜帧 `chunk=2` 峰值 22.62 GiB > 整段 4 潜帧 13.48 GiB ✓）。
   改 `.detach().clone()`（只拷 ≤2 帧）后同一测点降到 **11.90 GiB** ✓，且分块终于低于整段 ✓。
   判据直接量**存储字节数**：缓存那块 `untyped_storage().nbytes()` 只该装 `mem_size` 帧
   （旧写法会量出整块帧数 ⇒ 当场报红）✓。
3. **一档一档慢慢疼没意义，按峰值直接跳** ✓：`shrink_for_peak(chunk, peak, total)` 纯函数，
   峰值 ≈ 与每块帧数成正比 ⇒ 该降到 `chunk × 总显存 ÷ 实测 reserved`，再向下取阶梯档。
   真数据 `4 × 22.49 / 86.67 = 1.03` ⇒ **一次跳到 1** ✓（不用先疼一次 4、再疼一次 2）。
   用 `reserved` 而非 `allocated` 是**故意偏保守** ✓（切小只会更省 ✓，切大要再慢一遍 ✗）。

落点：`SUGGESTED_TEMPORAL_CHUNK` 仍是 4 但**只是起点** ✓；`TorchBackend.decode` 的 SEEDVR2 分支
解码后量 `max_memory_reserved()`，`> 物理显存` ⇒ spill ✓ ⇒ 记 `self._vaeTiling["temporalChunk"]`
并把该值记进 `self._vaeAutoChunk`（**自动降的不算创作者点名要的** ✓，下一轮还能继续降 ✓），
回执给 `spilled`/`peakReservedGiB`/`nextTemporalChunk`/`applied`；显式配置的档位**只回报不擅改**
（`applied: false`）✓。**溢出后下一轮开跑前先 `_trim_cuda()`** ✓✗：`_vaeSpilled` 那条路不清缓存，
新的一轮会在自己造成的碎片里再颠簸一次（实测已降到 1 档却仍花 **200.3 秒** ✓，清了之后 **13.3 秒** ✓）。
真通路两轮实证：第 1 轮整段 496.2 s ⇒ 回执 `spilled=True, nextTemporalChunk=1, applied=True`
⇒ 第 2 轮 **13.3 s**、`reason="auto"`、`exact=true` ✓。

**还没做** ✗：① 默认仍关 ✓（要开得显式给 `vaeTiling.temporalChunk`）—— 也正因为默认关，**第一轮
必然整段**、必然慢一次 ✓（496 s 那笔躲不掉 ✓，除非显式先配 1 档）；② 自动降档只会**往下**，
换到小分辨率后不会自动升回来 ✓（宁可慢一点也不再溢出一次），要升得显式 `set_vae_tiling`。

**为什么不抄别处** ✓：`CogVideo` / `Open-Sora` 的「VAE 分块」全是 diffusers 的
`vae.enable_tiling()/enable_slicing()` 一句调用（`inference/cli_demo.py`、`cli_vae_demo.py`），
仓内**没有实现**（§六.1 已核）；diffusers 那份对 3D 因果 VAE 是**重叠 + 线性混合**，产出**不逐位**
（时间维上混合会糊掉运动），拿它换 336 秒等于用画质换速度 ✗。`minimax-desgin-plugin` 的
`ldm/seedvr/vae.py::memory_limit_conv` 是**空间**切块且带 padding，本仓 `_forward_tiled`
（tile + halo + 裁内圈）反而更严 ✓，也不值得搬。

## 七、第四轮：七个 RAG／知识库仓（2026-11-04）

触发：创作者把一张 RAG 框架对照表摆出来，要求「有用的直接拉取然后进行抄写转化为现有的技术框架」。
**先把边界说清** ✓✗：表里七个仓全是「文档 → 解析 → 切块 → 向量/图 → 检索重排」的**完整栈**，
而本仓的知识库是**只读第三方 fts5 库 ＋ 进程内 md 小节词频表**（`services/script_library.py`），
没有向量库、没有 embedding 依赖、没有实体图谱 —— 抄栈＝引一堆用不上的依赖并造第二真源 ✗。
所以这一轮照 §二／§三的老规矩走：**只抄纯逻辑口径，代码本仓重写**。

### 7.1 实证事实

| 仓 | HEAD | 最后提交 | 文件数 | 许可 | 它到底是什么／判定 |
|---|---|---|---|---|---|
| `RAGFlow` | `c2808b14fee4` | 2026-11-04 | 5841 | `Apache License` | Python＋React 的完整 RAG 平台（跑起来要 ES／MySQL／Redis／MinIO）。**只有 `token.go` 的拼窗契约可搬** ✓ |
| `anything-llm` | `eda0eb5e25a4` | 2026-11-03 | 1582 | `MIT License` | TS 桌面＋服务端产品形态、向量库适配层。**产品形态不搬、适配层无落点** ✗ |
| `txtai` | `e37e5a66c91c` | 2026-11-03 | 658 | `Apache License` | embed＋索引＋检索的最小编排。**只有 `hybrid.py::rrf` 的融合公式可搬** ✓ |
| `LightRAG` | `aaebfe1bb8d5` | 2026-10-14 | 1331 | `MIT License` | 纯 Py 的图＋向量双层检索、增量更新。**无图谱 ⇒ 整条不成立** ✗ |
| `nano-graphrag` | `acb35c065614` | 2026-01-27 | 71 | `MIT License` | 极小 GraphRAG 原理实现。**读懂了，不搬** ✗ |
| `llama_index` | `444fdf068470` | 2026-11-03 | 9798 | `MIT License` | 文档／Node／Index 抽象层。**当文档读** ✗（会反向绑架数据模型） |
| `haystack` | `82dce4f7aed8` | 2026-11-03 | 12850 | `Apache License` | Pipeline／Component 抽象层。同上 ✗ |

### 7.2 搬了两处（都是「口径」，不是代码）

| 出处 | 落点 | 抄的契约 |
|---|---|---|
| `RAGFlow/internal/ingestion/component/chunker/token.go::mergeUnits`（L1093-1205）＋ `computeOverlapPrefix`（L436）／`splitOversizedText`（L1293）／`overlapFitPrefix`（L1609）／`sentenceDelimiter`（L414） | `script_library.py::_split_oversized` / `_overlap_tail` / `_fit_prefix` / `_windows` | 三步：① 超宽**单位**先按句界展开、单句仍超宽才硬切（硬切是兜底）② 拼窗阈值按重叠缩小 `width*(100-pct)/100`，给重叠**预留位置** ③ 开新窗**无条件前置**前窗尾巴，装不下从左逐字裁（宁可少重叠也不超宽） |
| `txtai/src/python/txtai/embeddings/search/hybrid.py::rrf`（L85-103，累加式是 L103 的 `uids[uid] += (1.0/(r+1))*weights[v]`） | `script_library.py::_fuse` ＋ 回执 `fusion` | `Σ weight/(k+rank)`，**只依赖秩** ⇒ 单源调用是单调变换、顺序不变 ⇒ 可无条件调用不必分支 |

两处都按本仓口径做了取舍，不是照抄 ✓✗：

* **句界分隔符**取 `\n!?;。；！？`，与 RAGFlow 的生产口径 `DEFAULT_DELIMITER` 逐字一致 ——
  **有意不含英文句点**：本库正文以中文为主，`Mr.`／`e.g.` 被切碎的概率远高于真句末 ✓。
  （凭据：`token.go` L404-407 的注释逐字引了 `DEFAULT_DELIMITER` 与 `\n!?;。；！？`；本仓 clone 到的
  这一版已无 Python 侧 `rag/app/naive.py` —— 上一版是 Py＋Go 双份，`internal/` 是重写后的形态。
  所以常量原文这一条**只有二手凭据**，如实记在此 ✓✗。）
* **`k=1`**（同 txtai），**不用**标准文献的 60 ✓：两源分数量纲本就不可比（bm25 负值／词频计数／
  LIKE 根本没分数），k=60 会把两源的差异压到小数点后三位 ⇒ 等于没融合。
* **末窗过短并入前窗**本仓默认开 ✓：Haystack `document_splitter.py::_concatenate_units`
  （L419-424）做的是**同一件事**（`len(current_units) < split_threshold` 就并进 `text_splits[-1]`，
  **不是丢弃** —— 一度凭印象写成「只能丢弃」，回源码核后已改 ✓✗），但它的 `split_threshold`
  **按 units 计、默认 0（＝关闭）**；本仓按字数的 `<20% 窗宽` 且默认生效（这里的窗宽是硬上限，
  并入只会让末窗略微超宽，比丢一段不可检索的文字划算）。
* `SOURCE_WEIGHTS` 一律等权 ✓：**没有评测集就调权重等于拍脑袋**；留这张表只为让偏置有地方可调、
  且一眼看见「当前是等权」。
* 回执里 `score` 仍是**各源原分数**、另给 `fusion` 承载排序依据，并明写「量纲不可比」✓ ——
  融合值不许冒充「相似度」回给 Agent。单源时 `fusion=None`（不伪造一个「融合过」的交代）。

### 7.3 否掉的（不是待办）

| 候选 | 否掉的理由 |
|---|---|
| `anything-llm` 的向量库适配层（LanceDB／pgvector／Chroma…） | 本仓**没有向量库**、也没有 embedding 依赖；适配层是给「要支持多家向量库」的产品用的，这里搬进来就是一套永不触发的抽象 ✗ |
| `LightRAG` 的图＋向量双层检索与增量更新 | 本仓无实体图谱；它的增量更新建立在「KV 存储 ＋ 图存储双写」上，而本仓索引是**只读**打开的第三方 fts5（`knowledge.py` 故意没有写入端）⇒ 结构上无一寸能落 ✗ |
| `nano-graphrag` 的 GraphRAG 原理（实体/关系抽取 → 社区摘要） | 原理读懂了，但代价是**每次入库都调 LLM 抽取**；本仓知识库是静态 md，收益全在「能增量维护大语料」这个前提上 ✓✗ |
| `llama_index` / `haystack` 的 Document／Node／Store／Component 抽象 | 创作者原话「抽象会反向绑架你的数据模型」；本仓已有 `script_catalog` 的国家·类型·奖项·权限字段契约与 `script_library` 的回执形状，套一层抽象只会多一层翻译 ✓✗ |
| `RAGFlow` 的文档解析链（OCR／版面／表格）与「带引用的检索」 | 解析链要 ES／MySQL／Redis／MinIO 四件；引用回溯本仓 `open_document` 已有**页号＋块号**，不缺 ✓ |
| `RAGFlow` 按文档类型的切片模板（法律／论文／手册／表格…） | 本仓只有 md 小节这一种形态，模板分派无对象；只取 `token.go` 里那份**与文档类型无关**的拼窗契约 ✓ |

### 7.4 验收

| 判据 | 结果 |
|---|---|
| `tests/script_catalog_test.py` | **52/52**（新增 §4c 十条：句界切片片尾落在 `。`、`"".join(cuts)` 逐字无损、仅单句硬切、相邻窗重叠、末窗 ≥20% 窗宽、两源第 N 名交替、`fusion` 值、单源顺序不变、`all` 真交错、回执交代） |
| `tests/script_library_test.py` | 34/34 |
| `tests/structure_test.py` ／ `tests/agent_script_corpus_test.py` | 36/36 ／ 28/28 |
| 真库切窗实况 | 2900 小节里 **98 节多窗**、最多 **7 窗**；测试里的重叠判据是 `windows[1][:30] in windows[0]`（末窗另有 `<20%` 窗宽并入前窗的判据） |
| `all` 模式排序 | 实测序列 `methods, playbook, methods, playbook…`、`fusion = [0.5, 0.5, 0.333333, 0.333333, …]`（原先 methods 全排前、playbook 沉底） |

**许可处理**：两处都是**按契约重写**、无逐行转写 ⇒ 按本仓既有纪律 `THIRD_PARTY_NOTICES.md` **不新增条目**；
出处与行号登记在本节，供日后溯源。若哪天改成逐行照搬，必须先回来登记 ✓。

**留着的账**：本轮没有跑这七个仓里的任何一个（RAGFlow 要四件中间件、LightRAG／nano-graphrag 要 LLM
抽取）—— 判定全部基于**读源码 ＋ 回本仓对照**，不是基于跑通 ✓。这条与本档 §零 的判据一致：
「真通路上的收益」而不是「只有测试能跑」。

### 7.5 顺带量出来的一个洞：**3139 份扫描件根本没有文字层**

对照 `curated-screenplay-methods.sqlite` 的账面：

| 口径 | 实测 |
|---|---|
| `documents` 登记 | **5403** 份 |
| 真有正文块（`chunks` 里有 `document_id`） | **2262** 份（41.9%） |
| 登记了但零正文块 | **3139** 份 —— 按扩展名全是 `.pdf` |
| 其中 `08-编剧方法书籍` | 2795 份（含《世界电影》36 年期刊合订 2495 份，约 4.9 GB） |
| 其中 `05-中日韩美高热类型片` | 332 份（爆款短剧／剧本扫描件） |

**不是抽取失败，是没文字层**：PyMuPDF 抽 5 份样本（《序列编剧法》《电影剪辑》《折腰》
《世界电影 2014-09》《外国当代剧作选（4）》）的前 5 页，`get_text()` 全部 `chars=0`、
`page.get_images()` 有图 ⇒ 纯扫描件。供应商那 2262 份也只覆盖了有文字层的那批。

处理方式（不当成「以后再说」）：本仓建 **sidecar**（`data/ocr/library-ocr.sqlite`），
`app/scripts/corpus/ocr_index.py` 逐页 OCR 写入，检索层两库按秩合并、`from` 标出处。
真机实测：RapidOCR 150 dpi 约 **0.9–3.5 s/页**（首份含模型加载），中文识别干净
（「毫无防备」「泥淖」「序列是经常被忽略的一个重要利器」全部正确）；
17.5 万页全量约 **44 小时**，去掉期刊那 10 万页后剩约 **19 小时**。
**判定**：这件事必须在本仓做，七个参考仓里没有一个能替（它们都假设输入已有文字层）✓。
