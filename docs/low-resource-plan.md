# 低显存 / 低内存 / SSD 环境下的运行与开发方案

> 状态：方案（未实施）。前置：`docs/reference-audit.md` §六（三级内存）、`docs/feature-map.md` §10（族表）。
> 本文只写**缺口与落法**，已完成的机制只标路径、不重做。

---

## 1. 目标环境（写死数字，否则所有取舍都无从判）

**当前开发机（基线真源）**：A5000 sm86，显存可用 **22.49 GiB**（23028 MiB，ECC 开）；内存 96 GB；NVMe。
已实测基线：Z-Image 整链 40.5 s；Krea2 装配 9.2 s / 8 步 51.4 s；SEEDVR2 720×480×49 帧 8.3 s（`temporalChunk=1`）。

**验收要对齐的低配档**：

| 档 | 显存 | 内存 | 磁盘 | 典型机器 | 目标能力 |
|---|---|---|---|---|---|
| **L1** | 8 GB | 32 GB | SATA/NVMe | 3060 Ti / 4060 | Krea2·Qwen-Image 单图（int8 分片）、Z-Image |
| **L2** | 6 GB | 16 GB | SATA | 1660 / 笔记本 3060 | Z-Image 小画幅、SEEDVR2 短片 |
| CPU | — | ≥32 GB | 任意 | 无卡 | 仅 `dryrun`（不出真图） |

SSD 决定策略的三个事实：① 顺序大块读 ≫ 随机小读 ⇒ 权重必须按张量顺序读（`weights._load_streaming` 已是）；
② SATA ≈500 MB/s ⇒ **19 GiB 冷读 ≈ 38 s**，NVMe ≈6 s ⇒ 「重复读盘／重复反量化」是低配头号成本；
③ 页缓存跨进程可复用 ⇒ 第二次装载其实就是内存读，**不需要自己造 mmap 层**。

---

## 2. 参考项目对齐（只取设计，不取代码）

| 参考 | 取什么（证据） | 不取什么 |
|---|---|---|
| **llmfit**（`reference/llmfit`） | 四档 run mode（GPU/MoE/CPU+GPU/CPU）、四档 fit level（Perfect/Good/Marginal/Too Tight）、**动态量化阶梯**、**装不下先降 workload（half context）再降档**、速度估算 `带宽÷大小×0.55`；`llmfit-core/docs/how-it-works.md:21,36-40,58-71`、`src/plan.rs` | token/s 带宽公式（DiT 瓶颈不是带宽）、MoE 专家卸载、复合评分数 |
| **tabby**（`reference/tabby`） | `LocalModelConfig{parallelism, num_gpu_layers, context_size}`，**默认 `num_gpu_layers=9999`＝尽可能全上卡，靠下调一个整数适配**；`crates/tabby-common/src/config.rs:341-366` | llama.cpp 后端、embedding、并发调度 |
| **Strata / airllm / Edge0 / ds4** | 已移植完毕：`foresight.py`、`prefetch.py`、`win_io.py`、`gguf_dequant.py`（Q5_K）；三轮审计结论见 `reference-audit.md` | 不再搬新东西，只做接线 |
| **VAE tiling 类** | 只要「空间切块＋时间分块」的思想（已自研并实测过「分块不一定省」） | 其实现 |

**结论**：参考侧还能给的只剩「对外怎么表达」（verdict/advice 与整数旋钮），搬字节那层我们更完整。

---

## 3. 现状 vs 缺口

**已有**：`residency`（off/split/park、层前缀、hook 搬激活）、`device_policy`（纯函数决策、`probe`、0.8/0.92 阈值）、`host_cache`（内存 LRU 卸盘）、`win_io`＋`weights`＋`prefetch`（顺序读）、`gguf_dequant`＋`dit_build._Int8Adapter`（int8/int4/fp8/Q5_K）、`scheduler`（Slot/Ledger/`limit_working_set`）、`vae_seedvr2.chunk_ladder`＋`shrink_for_peak`、`dryrun`（零依赖编排）。

**四个真缺口**（四条均已于 2026-11-07 补齐，落点见 §9）：

* **G1 没有「机器档」这一层**：旋钮散在 6 处（`policy`／`reserveFor`／`host_cache` 配额／`foresight` 槽／`vaeTiling`／`temporalChunk`），没有一处能回答「8 GB 卡全按最低配跑」。✅ 已补 `profile.py`。
* **G2 没有任务级可行性预检**：`engine_readiness` 只查权重在不在盘，不回答「1024² 跑不跑得动、不行改什么」，只能等 `/load` 失败。✅ 已补 `feasibility.py`。
* **G3 反量化产物不落盘**：GGUF 文本塔每次装载重算一遍反量化，多付 CPU 与**峰值内存**；`hybrid_load`＋`cache_guard` 的「指纹＋原子写＋余量预检」范式只服务了 H3，没服务量化产物。✅ 已补 `materialize.py`（**只覆盖 GGUF**：本仓 int8 走 `Int8Linear` **常驻壳**、逐层在前向里反量化，没有「整份 bf16」这个中间产物可落盘 ⇒ 不硬造一层）。
* **G4 没有低配模拟夹具**：`probe()`／`host_memory()` 无注入点，8 GB 行为只能上真机。✅ 已补 `DARME_FAKE_GPU_TOTAL_GIB`／`DARME_FAKE_HOST_GIB`。

---

## 4. 运行策略

**4.1 分片加载**：已有 `place()`，补对外旋钮 `gpuLayers`（`null`＝自动，`9999`／≥层数＝全上卡，语义沿用 tabby）。
判据：`gpuLayers=0` 必须等于 park、给定整数必须真按整数切，不许「给了数还走 auto」。

✅ **已完成（2026-11-07）**：单一真源 `residency.normalize_gpu_layers`（`None`/空/`"auto"` ＝没提、`0` ＝整份留内存且 `refused=False`、`n` ＝前 n 层上卡）；拒绝 llama.cpp 的 `-1` 口径（同一命令两种含义必错）、`true`/`3.5`/`"abc"` 当场 `ValueError` ⇒ 路由转 400。
预算只当**上限**（富余不多切）：`granted<asked` ⇒ `capped=True` 但 **`fits` 仍为 True** —— 部分层在卡上成立，置 False 会被 `place` 整份退回 CPU（静默反转）。
非层参数（embeddings/head/norm）先上卡、再按前缀连续取层；只给 `gpuLayers` 而 `residency=off` ⇒ 后端**自动顶开 `split`** 并回执 `residencyOpenedBy="gpuLayers"`（`off` 路根本不调 `place`，不顶开就是静默吞旋钮）。
进装载键：`runtime` 两处 `_load_key` 都带 `gpu_layers`，且 `ensure_loaded` 里**先归一、后拼键**（否则 `"3"` 与 `3` 是两个键、白装两遍）。
8 条 `TorchBackend` 装配线全接线（`engine_runtime_test` 用 ast 静态核，唯一例外 `_ensure_image_loaded`＝`SdxlBackend` 没有可切的层，回执 `granted=None` **不是** 0）。
判据：`engine_residency_test` ⑧ 系列 20 条 ＋ `engine_runtime_test::case_gpu_layers_wiring` ＋ `engine_router_contract_test` ⑪¹⁶–⑪¹⁸。

**4.2 量化**：bf16 2 B/param → fp8·int8 1 B → int4 0.5 B（19.5 GiB 对应 9.75／9.75／4.9）。
补**落盘物化**：按「源指纹＋目标 dtype」命名写盘，复用 `cache_guard.file_fingerprint` 与 `hybrid_load` 四条纪律（键集不变／余量不足即拒／原子写／sidecar 失败不阻塞）；命中写回执 `materialized{hit,gib}`。只加速装载、不改数值。

✅ **已完成（2026-11-07，G3）**：`materialize.py` ＋ `gguf_to_llm` 接线，详见 §G3。**只服务 GGUF 反量化**——本仓 int8 是 `Int8Linear` 常驻壳（权重一直以 int8 待着、逐层在前向里现反量化），没有「整份 bf16」这个中间产物可落盘 ⇒ 不硬造一层。

**4.3 mmap 与 IO**：**明确不做 mmap** —— `safetensors.py:273` 已裁定顺序读与 mmap 等价，而 mmap 页由 OS 回收 ⇒ 峰值不可控，与低配要的「可预测峰值」直接冲突。第 0 级缓存就是页缓存。补：预取并发度按盘速自适应（NVMe 2–4 / SATA 1–2），初值由 §4.5 档位给，不猜。

✅ **已完成（2026-11-07）**，但**「初值由 §4.5 档位给」这一半改了**：§4.5 是降级顺序表（`temporalChunk`／`policy`／量化／分辨率），里面**没有盘速这一项**；而预取深度是**盘**的属性（和卡多大、机器档是 l1 还是 gpu 无关），硬塞进机器档就是把两件事绑一起 ⇒ 归 `profile.NOT_WIRED` 的 `readAhead` 一格并写明理由（单件读前置有 256 MiB 上限 ⇒ 2 槽只多占十几 MiB，「低配档压到 1」换不来内存、只丢掉「读」与「反量化」的叠合）。
真源改为**按盘速分档**：`prefetch.DISK_CLASSES` ＝ `nvme (2, 上限 4)`／`sata (2, 上限 2)`／`hdd (1, 1)`／`unknown (2, 2)`；`probe_disk()` 真读 **32 MiB 顺序读**（1 MiB 一块 ⇒ 探测自身峰值 2 MiB，**不许**用 4 MiB 块：`read()` 是先申请新的再释放旧的 ⇒ 峰值 2×块大小，实测 8 MiB 会顶红「关预取峰值 < 单件」那条判据）＋**每块盘只量一次**。
优先级 `resolve_depth(path, explicit=)`：**显式参数 > `DARME_READ_AHEAD` > 盘档默认**；显式值也要截到该盘档上限并如实报 `capped`（在机械盘上要 4 条并行读是有害的）。`unknown` 一律退回 **2** —— 与本轮之前写死的值同 ⇒ 量不出盘速时**行为零变化**（不是猜一个新数）。
`weights.load_module_weights(read_ahead=None)` ＝ 自适应；`weights._READ_AHEAD = 2` 降级为「`unknown` 回落值／老调用方锚点」，不再是谁的装载深度。
观测面：`weights.last_read_ahead()` ＋ `/load` 回执 `readAhead{value,source,requested,capped,default,hi,disk,diskClass,mbps,how,why}`（挂在 `_commit_load` 这唯一人口）；关掉预取（`0`）时也写（`disabled=True`）——「关掉了」与「这一趟没跑过」是两件事。
判据：`tests/engine_readahead_test.py` **37 条**（新增 ⑦–⑭：表完备／夹具三档／显式截断／env 赢过默认／五种坏值当场抛／小文件不测速／默认值必须是 `None`／观测面读得出凭据／关掉也如实／自适应路权重逐张量相同）。

**4.4 三级缓存**：显存 `residency` → 内存 `host_cache`（LRU）→ SSD（原文件＋§4.2 物化缓存）。新增 `diskCacheLimitGiB`（`null`＝不限）与命中账；配额不足 ⇒ 拒写并回退重算（慢但不错），严禁「三级全满」死循环。

✅ **已完成（2026-11-07）**：配额与「拒写即重算」由 G3 落地（`DARME_MATERIALIZE_LIMIT_GIB`＋5% 余量，超了 `MaterializeSkip` 一个字节都不写）；**命中账**在 `materialize.stats()`／`store_usage()`／`cache_report()`，对外一格 `GET /api/v1/engine/runtime` → `diskCache{dir,enabled,files,sidecars,diskGiB,limitGiB,freeGiB,hits,misses,written,refused,failed,hitGiB,writeGiB,lastReason}`。
三个易读错的默认值已钉住：`enabled=false` ＝ **总开关关着**（≠「没命中」）、`limitGiB=null` ＝ **不限**（≠「限了 0」）、`hits` 是**本进程**的（重启清零，累计看 `diskGiB`）。
⚠️ `refused`（磁盘不够，一个字节没写）**单独一栏**，不许并进 `misses`：前者要清盘／调配额，后者要重算，合成一个命中率会把两件事读成一件。
⚠️ 体积读到 **1 KB**（6 位小数），不是 1 MB：3 位小数会让几 MB 的缓存一律显示 `0.0`（「有产物但读成 0」比不显示更糟）。

**4.5 降级顺序（红线）**：

| 序 | 动作 | 换画质 | 谁决定 |
|---|---|---|---|
| 1 | `temporalChunk` 4→2→1、`tile` | 不换 | 引擎自动（已有） |
| 2 | `policy` off→split→park | 不换 | 引擎自动（须标 `degraded`） |
| 3 | 量化 bf16→int8→int4 | **换** | **只建议** |
| 4 | 分辨率／帧数 | **换＝交付物** | **只建议，绝不自动** ✗✗ |

理由：修复族输出分辨率就是交付物，自动降等于静默降质。

---

## 5. 交互设计

**5.1 `POST /api/v1/engine/plan`（零副作用预检）**

```json
请求 {"stage":"krea2","width":1040,"height":1040,"frames":1,"quant":"auto","profile":"auto"}
回执 {"verdict":"marginal",
      "need":{"gpuGiB":12.57,"hostGiB":7.5,"diskGiB":0},
      "have":{"gpuGiB":7.8,"hostGiB":28.4,"diskGiB":120.0},
      "chosen":{"policy":"split","gpuLayers":14,"quant":"int8"},
      "advice":[{"action":"lowerQuant","to":"int4","savesGiB":6.3},
                {"action":"reduceResolution","to":"832x832","why":"交付物，需创作者点头"}],
      "estimate":{"loadSeconds":18,"runSeconds":95,"basis":"table"}}
```
口径：`verdict` 逐字用 llmfit 四档；`estimate.basis ∈ {table, measured}`（`measured` 从 `tuning.py` 台账回填），**估算不许写成承诺**；纯函数、不装权重、不碰 torch。

✅ **已完成（2026-11-07）**，两处与原计划不同，都是**故意的**：
① **名字叫 `feasibility` 不叫 `plan`** —— `plan` 早就被「装载计划」（件齐不齐、退出码 0/1）占了；同名同命令两套退出码语义 ⇒ 既有脚本会把「缺件」读成「卡不够」。新面是 `GET /api/v1/engine/feasibility`、`facade.feasibility()`、CLI `feasibility` **三处同一个词**（`fromLedger=true` 才交给服务层 `engine_readiness.task_feasibility` 取跑次样本 —— 内核只**收**样本、绝不读业务库）。
② **`estimate.basis` 只有 `{unknown, measured}`，没有 `table`** —— 本仓没有逐族秒数表；只有 ``sample ≥ ESTIMATE_MIN_SAMPLES(3)`` 才报 `measured`（p50/p95），否则 `basis="unknown"` 且 `runSeconds=None`。⚠️ 绝不给 `0`（会被读成「立刻完成」）；`loadSeconds` **恒 `None`**（跑次台账只记整任务耗时，拿它冒充装载秒数会把 40 秒说成 300 秒）。
回执另有 `size{asked,snapped,step,megapixels,problem,note}`：目标尺寸**吸附口径委托画布**（各族步长不同，不在这里再写一遍），`problem/note` 如实说明吸附或不动的原因；`activation{basis,gib,whatItMeans}` 一律 `basis="unknown"`、`gib=None`（方案 A：没有激活峰值表就**不编**，代价是 `perfect` 也降一档报 `good` 并写明为什么）。
判据：`tests/engine_feasibility_test.py`（含「激活不许编数」「同驻/换装两种口径必须真算出不同的数」「样本不够不给秒数」）。

**5.2 `/load` 增 `profile`**（`auto|l1|l2|cpu|gpu`）＝G1 唯一入口，内部展开成 §4 旋钮组；与 `devicePolicy` 不是两套词（`profile`＝机器档，`policy`＝本件驻留方式），冲突以 `policy` 为准并标 `profileIgnored`。`/engine/runtime` 增 `budgets{...,source}` 与 `diskCache{gib,limitGiB,hits,misses}`。

✅ **已完成（2026-11-07）**：`budgets{...,source}`（G1）与 `diskCache{...}`（§4.4）两格都在 `/engine/runtime` 回执里；`profile` 由 G1 接在 `/load` 上。`diskCache` 的键比原计划多（见 §4.4 那张单子）：原计划只要 `{gib,limitGiB,hits,misses}`，实际还得有 `refused`／`failed`／`enabled`／`dir`／`freeGiB`——否则「慢了」查不到是磁盘不够、还是开关关着。

**5.3 CLI**：`python -m app.services.engine plan --stage krea2 --size 1040x1040 --json`；退出码 0 能跑／1 跑不了／2 参数错（与 `readiness` 同口径）。

✅ **已完成（2026-11-07）**，实际命令是 `python -m app.services.engine feasibility --stage krea2 --size 1040x1040 [--policy auto|gpu|cpu|balanced] [--capacity 8] [--measure 95,102,88] [--json]`（同理改名的原因见 §5.1 ✓ ①）。
退出码 0 能跑／1 跑不了（`too_tight`）／2 参数错；`--strict` 时只把 `perfect|good` 判 0（`marginal` 也算跑不了 —— 要的是「排期敢不敢按它跑」，不是「理论能不能装」）。
`--measure` 是本机实测秒数的入口（逗号分隔）；**有意不提供 `--from-ledger`** ✗✗：跑次台账在业务库，而引擎内核是纯函数零 IO —— 要从台账取样本走服务层（HTTP `?fromLedger=true`）。

---

## 6. 开发与调试流程

| 目的 | 做法 |
|---|---|
| 22 GiB 机上验证 8 GiB 行为 | 夹具：`DARME_FAKE_GPU_TOTAL_GIB`／`DARME_FAKE_HOST_GIB`，**只接在 `device_policy.probe()` 与 `scheduler.host_memory()` 两个出口** |
| 编排级 | `dryrun` 补低配档用例（零依赖） |
| 决策级 | 假预算跑 `decide()`／`place()`／`host_cache`，断定档位与卸层序列 |
| 真权重级 | `_bench` 补 L1/L2 的「降到几层、慢多少」实测表；`tuning.py` 采样回填 `estimate` |

---

## 7. 验收标准（可判读）

* **结构**：新模块必须登记三处（`engine/__init__.GROUP_*`、`feature-map.md`、`run_all.TESTS`），`structure_test` 绿。
* **行为**：8 GiB 模拟下看 Krea2 1040² 必须返回 `marginal/tooTight`＋准确 `advice`（不是崩）；`off` 请求在 8 GiB 下必须降为 split 且 `degraded=true`；`gpuLayers=0` 必须等价 park；物化缓存命中与否必须出现在回执。
* **性能**：Krea2 装配 ≤12 s（现 9.2）；Z-Image 整链 ≤45 s（现 40.5）；SEEDVR2 49 帧 ≤15 s（现 8.3）；SATA 冷装载 19 GiB ≤45 s。
* **估算一致性**：`/plan` 预估 vs 真跑误差 ≤25%；超了判「表错」并修表，**不许放宽判据**。

---

## 8. 关键取舍与风险

| 取舍 | 代价 | 风险与对策 |
|---|---|---|
| 不 mmap | 冷启多一次拷贝 | SATA 上首装 38 s ⇒ 常驻预热＋物化缓存；风险低 |
| 物化落盘 | 磁盘占用（int8 9.75 GiB） | 盘满：先算后写、不足即拒；配额 LRU |
| 预检估算 | 表值与真实激活有偏差 | 用 `tuning.py` 实测回填收敛；判据 ≤25% 误差，避免「预检过、真跑 OOM」 |
| 降级只到「不换画质」 | 8 GB 卡可能直接判 `tooTight` | 宁可拒绝并给建议，也不静默降分辨率 |
| `gpuLayers` 整数 | 层间无法按量细分 | 沿用现有「连续前缀」约束；层内不均等靠 hook 搬激活兜 |

---

## 9. 实施进度

### G1 机器档（profile）—— 已完成（2026-11-07）

**一句话**：把「8 GB 卡按最低配跑」这句话，展开成**一组旋钮的唯一出处** —— `POST /api/v1/engine/load`
的 `profile`、`GET /api/v1/engine/runtime?profile=`、`python -m app.services.engine status --profile`，
三处认的是**同一张表**（`app/services/engine/profile.py`，纯函数、不 import torch、不读盘）。

**五档**（`auto` / `gpu` 是「不预设」，只有后三档动旋钮）：

| 档 | 假设 | policy | 内存配额比例 | 视频解码分时 | 动的是什么 |
|---|---|---|---|---|---|
| `auto`（默认） | — | 不设 | 默认 `HOST_USE_RATIO`(0.75) | 不设 | **什么都不动**（`applied={}`） |
| `gpu` | — | 不设 | 默认 | 不设 | **什么都不动**（大卡不必压） |
| `l1` | 8 GiB 卡 / 32 GB 内存 | `auto`（**不是** `cpu`） | 0.60 | 1 潜帧 | 上面两格 |
| `l2` | 6 GiB 卡 / 16 GB 内存 | `auto` | 0.50 | 1 潜帧 | 上面两格 |
| `cpu` | — | `cpu` | 默认 | 1 潜帧 | policy ＋ 分时 |

三条纪律（都有判据）：

1. **表只引用、不抄数**：默认值一律引 `host_cache.HOST_USE_RATIO` / `device_policy.DEFAULT_POLICY`，
   `profile.py` 里出现那个数字的字面量即判红（抄一遍就是第二本账，那边改了这边不跟）。
2. **`applied` 只放真会改的项**：`auto`/`gpu` 是 `{}` 且 `source="default"` —— 把「等于默认值」
   报成「已改」会让用户以为换了档就有区别，还会让装载键多一格。
3. **假设值绝不当实测**：`Profile.gpu_gib/host_gib` 只用于 `budgets.assumed` 与 `hint()`，
   **不覆盖** `device_policy.probe()` 的真读数。

**冲突裁决**：机器档是「这台机器大概什么样」，`policy` / `vaeTiling` 是「这一次这一件怎么放」。
两个都给 ⇒ **显式的那一方赢**，被压掉的项从 `applied` 里剔除并写进 `ignored`、`profileIgnored=true`
（`profile=cpu` ＋ `policy=gpu` ⇒ policy 进 `ignored`；`profile=l1` ＋ 显式 `policy` ⇒ 不算冲突，
因为 l1 本来就不压 policy）。落法在 `runtime.ensure_loaded`：先 `resolve()`，再按 `applied` 改
`policy` 与 `vae_tiling` —— 只有这一处展开。

**明确不接的四项**（`notWired`，随回执带出，不让人以为改了）：`reserveFor` / `gpuLayers` /
`readAhead` / `foresight`。`foresight` 尤其要说清：引擎默认预取槽就是 `0`（关），低配档**压不动**也
**不必压**，而显式 `DARME_FORESIGHT_SLOTS` 必须赢过档位 —— 所以档位表里干脆没有这一格。

**`budgets`（一处回答「这一档下还剩多少」）**：`{profile, source, gpuGiB, gpuFreeGiB, hostGiB,
hostFreeGiB, hostRatio, hostQuotaGiB, assumed, hint}`。`source` 四值如实说是哪几栏**真读数**
（`cuda+host` / `cuda` / `host` / `none`），测不到写 `null` **不写 0**（0 会被读成「一点都没有」）。
两处读数只走既有入口（`device_policy.probe()` 已含 allocator 可复用块那条修正、`scheduler.host_memory()`
已接低配模拟），不另立第二套；⚠️ `host_memory()` 返回 `(可用, 总量)` 而 `budgets` 要 `(总量, 可用)`，
两个 `tuple[int, int]` 形状全同、顺序反了不报错只会把配额算虚高 ⇒ 装配层传带名字的
`profile.HostMemory`。`hint` 只提示不拦：卡比假设大一倍以上（22.49 vs 8 ⇒ 「低配档会白慢」）、
或小得真装不下、或内存比假设少 10% 以上，才出一句人话。

**装载键**：`profile` 进 `_load_key` 两处（`ensure_loaded` 与 `_ensure_loaded_internal`，逐字同键）——
不进键就会「换了档却复用上一套布局」，且一声不响。顺带把 `policy` 的归一写成
`_policy_key(...)` 放进建键那一行：`None` 与 `"auto"` 必须是同一个键，否则按签名默认值手拼出来的键
是 `policy='None'`，同键被算成「换了装配」⇒ 真去装一遍 19 GiB。

**接线**（一档到底，中途不换）：`runtime.ensure_loaded(profile=)` → `resolve()` → 六条 DiT 线 ＋
H3 的 `TorchBackend(profile_name=, profile_applied=)`（构造期定死，不提供运行时改档 —— 配额账本
按第一件定死，改档必须重装）→ `_host_cache_of()` 按 `profileApplied["hostRatio"]` 取配额，
账本事件里带 `ratio` 与 `ratioSource`。出图线（SDXL 四件套）走 `SdxlBackend`，**没有 `host_cache`
那本账** ⇒ 档位的 `hostRatio` 在它上面必然无效，回执 `profileNote.ignored` 明写。
`/load` 白名单收三写法同族里的 `profile`，归一失败转 `StageError`（400，不是 500），`auto` 视为没配
（从 `values` 里摘掉 ⇒ 换回默认档不白装一遍）；`/engine/runtime` 与 CLI `status` 的 `--profile` 只影响
`budgets`，**不装载**；拼错一律当场报（路由 400 / CLI `E_BAD_CONFIG` 退出 2）。

**判据**：`tests/engine_profile_test.py` **71 条**（已登记 `run_all`）—— 五档取值域与假设值 +
默认值不许抄字面量 + 别名一张表 + 不认的抛（含非字符串 `8` 不许猜成 `8g`）+ `applied` 三档真值 +
冲突三例（含「不算冲突」那一例）+ `describe()` 与 `resolve()` 逐项相同 + `budgets` 六例
（含 `HostMemory`/元组顺序的**反验**）+ `hint` 五例（含「阈值不是 3 倍」）+ AST 反验
（不 import torch、不读盘、`_load_key` 两处都带 `profile`、8 条 `_ensure_*_loaded` 都收档位、
8 处装配用 `TorchBackend(...)` 都下传）+ `/load` 与 `/engine/runtime` 与 CLI 三处接线 + 真调
（`status(profile=l1)` 的配额 0.60、`ensure_loaded(profile=low1)` 在装载前就拒、换档装载键必然不同）。
反向验证过：摘掉装载键里的 `profile` ⇒ ⑦c 红；把 `l1` 配额改回 0.75 ⇒ ①h/①i/③c/⑤b/⑧d 五条红。
回归：runtime 83、pipeline 163、qwen_image、zimage、krea2 35、device_policy、feasibility 61、
structure 41 全绿。

### G2 任务级可行性预检 —— 已完成（2026-11-07）

**一句话**：**不装载**就答「这台机器跑不跑得动这条链」——`GET /api/v1/engine/feasibility?stage=&capacityGiB=`，
结论四档 `perfect` / `good` / `marginal` / `too_tight`，每档都带**人话理由**。

**与两个邻居的分工**（别混）：

| 问题 | 谁答 | 口径 |
|---|---|---|
| 件**齐不齐**（盘上有没有） | `readiness()` / `GET /readiness` | 文件名 / 头部字段 |
| 装齐之后**怎么排班** | `plan()` / `GET /load-plan` | **文件字节**（"读进来多少"） |
| 装完**占多少**、够不够 | `feasibility.check()` / `GET /feasibility` | **驻留 GiB**（"卡上占多少"） |

**单一真源**：逐件需求只走 `torch_backend.stage_part_need()`（装配线同一个出口），表值只走
`TorchBackend.STAGE_NEED_TABLES` —— 值是那几本 `*_PART_NEED_GIB` 的**同一批 dict 对象**
（`is` 相等），所以改账预检立刻跟着变，库里没有第二份数字。预检自己写 `if stage == ...` 就是第二本账，
本仓已在 Krea2（按 int8 量）与 Qwen-Image 2511（按 38.05 判）上各踩过一次。

**「峰值」逐档登记**（`feasibility.STAGE_PEAK`），口径必须等于装配线**真怎么装**：

| 口径 | 档 | 峰值 |
|---|---|---|
| 整链同驻求和 `sum` | zimage / music3 / seedvr2 | 19.28 / 13.35 / 7.27 |
| 分时换装取最大 `max` | h3 / qwen_image21 / krea2 / wan22 / qwen_image | 25.44 / 14.79 / 12.57 / 26.62 / 38.05 |

写错口径的症状是「算得挺像、判得全反」：2.1 三件之和 28.68 > 卡 22.49，按 sum 判会给每天都在跑的档
报「装不下」；music3 按 max 判会少算 8 GiB（实测三件真同驻 13.34）；wan22 双专家不并存，把 `dit2`
加上就是虚报。

**卡与内存两栏都判**：`split`（联合驻留）也要内存放溢出的那 `peak − free` 那部分，所以
「8 GiB 卡 + 32 GB 内存」跑 12.57 GiB 的链如实报 `too_tight`，不是 `marginal`。

**激活如实标未知（方案 A）**：`activation = {"basis": "unknown", "gib": null}`，不编数；
因此 `perfect` 一律降一档报 `good` 并带 `downgraded=true` + 理由。回执里 `need.gpuGiB` 与
`activation` 分两栏，**不合成一个模糊数**。

**`capacityGiB`**（"如果我买 16G 卡跑得动吗"）：丢开真读数按**空卡**算，回执 `have.assumed=true`；
不填就用当前真读数（低配模拟 `DARME_FAKE_GPU_TOTAL_GIB` 也在这条路上生效）。

**判据**：`tests/engine_feasibility_test.py` **61 条**（已登记 `run_all`）——表完备 + `is` 相等（防第二本账）+
逐档峰值数值 + 四档真能出现 + split 也要内存 + 缺件不等于显存不够 + 激活不许编数 + AST 对账（源码里
真有 `decide_for(("dit","text","vae"))` / `("dit","vae")` / `("dit",)`、`_zimage_device_decision`、
`_backend_phase` 三边界）+ 门面唯一入口 + 端点存在。
回归：qwen_image 66、device_policy 0 failed、runtime 83、h3_text 28、structure 41 全绿。

### G3 反量化产物落盘缓存 —— 已完成（2026-11-07）

**一句话**：把「GGUF 反量化 ＋ cast 算出来的**目标 dtype 字节**」写成一份正常的 safetensors ⇒
**第二次装载只读盘，一次反量化都不做**。省的是 CPU 与**峰值内存**（不再需要 fp32 中间量）。

**键**（`Materializer.key()`）：`sha256(schema, converter, dtype, 源大小, 源头 64KB 哈希)` 前 12 位。

| 进键 | 为什么不进 |
|---|---|
| `MATERIALIZE_SCHEMA`（布局/元数据键改了 +1） | — |
| `CONVERTER = "gguf-dequant-v1"`（反量化公式改必 +1） | — |
| 目标 dtype（bf16 与 fp32 各一份产物，并存不互覆） | — |
| 源 `size` ＋ 头 64KB `sha256` | **mtime 刻意不进** ✗✗：进了就「`touch` 一下换一个文件名 ⇒ 多留一份 8 GiB 产物」 |

mtime 仍在 **sidecar** 里，由 `cache_guard.is_cache_valid` 比三元组 ⇒ 「只动了时间戳」的源
**判失效 ⇒ 重算并覆盖同一个文件**（既不误用，也不增生）。

**四条失效缺一不可**（`:meth:`Materializer.valid``）：产物不在 / 头不自证（截断、偏移与 dtype·形状不符）/
sidecar 指纹不符或缺失 / 元数据**自述** 的 schema·converter·dtype 与本次不一致。
少最后一条时，「同名的旧产物（换过 converter 但键算法没变）」会被当成有效 —— 那是「数值不对、一路绿灯」的形状。

**数值不变是构造出来的**：写盘字节取自**真装进模型的那张 torch 张量**的原始位模式
（`gguf_to_llm._tensor_bytes` = `.view(torch.int16).numpy().tobytes()`），命中侧
`torch.frombuffer(...).view(dtype).reshape(...).copy_()` ⇒ 「逐位相同」由构造保证，
不靠两条舍入公式对拍。自己写「fp32 取高 16 位」是**向零**舍入（torch 是 RNE）⇒ 差 1 ulp ⇒
命中的模型与不命中的模型不是同一个模型。

**写侧纪律**：`.tmp` → `os.replace` 原子替换（别人看不到半成品）；键集与每张字节数逐个核 ⇒
少写一个张量即 `MaterializeError` 且**不留残缺产物**（缺张量的产物装得上、跑得动、输出不对）；
装载中途抛异常 ⇒ `with` 退出即删 `.tmp`。

**失败一律退化成「慢」**：超配额（`DARME_MATERIALIZE_LIMIT_GIB`）／磁盘余量不足（`DISK_HEADROOM=1.05`）／
没权限 ⇒ `MaterializeSkip`，**一个字节都不写**，本次改为重算，装载照常成功。
sidecar 写失败不阻塞（没有指纹 = 下次重算）。开关 `DARME_MATERIALIZE=0/off`（**默认开**）、
目录 `DARME_MATERIALIZE_DIR`；默认目录 `get_data_root()/materialized`，**不放权重目录旁边**
（ComfyUI 会递归扫 `models/**`，一份 8 GiB 产物会被当模型列出来）。

**命中账（§4.4）**：`materialize.stats()` 五个计数各指一件事，刻意不合成「命中率」——
`hits`（直接复用几次）/ `misses`（判了不能用）/ `written`（真写成几次）/ `refused`（余量·配额不足，
**一个字节都没写**）/ `failed`（写到一半失败）。`store_usage()` 报盘上压了多少（产物个数与字节 +
sidecar 个数 + 配额 + 余量），`cache_report()` 把两者并成一格给 `/engine/runtime` 的 `diskCache`。
理由：只有次数 ⇒ 用户不知道盘上多了 8 GiB；只有体积 ⇒ 看不出这层在不在起作用；
而把 `refused` 并进 `misses` 会把「该清盘」读成「该重算」。

**可见面**：`LlmBackend.describe()["materialize"]` 原样抬出 `model.materialized`
（`None` = **还没装载**，≠「没命中」；装载失败则清回 `None`，不留上一次的结论）。
没有这一格，「这次加载快了 30 秒」是缓存生效还是权重变小，看回执的人分不出来。

**边界**：**只服务 GGUF 反量化**（`gguf_to_llm`）—— 本仓 int8 是 `Int8Linear` **常驻壳**
（权重一直以 int8 待着、逐层在前向里现反量化），没有「整份 bf16」这个中间产物 ⇒ 不为了对齐
一句话去硬造一层（造出来的东西没人用，还多占 8 GiB 盘）。不做 LRU 淘汰（超配额只拒新写，不替人删权重类文件）；不验张量数值（与 `cache_guard`
同一条边界：头自洽 ≠ 数值对，数值由「命中/不命中逐位相等」在真装载上保证）；
源张量**全浮点**（没有量化）⇒ 整份不做并写明原因（抄一份 16 GiB 换 0 秒）。

**判据**：`tests/engine_materialize_test.py` **45 条**（已登记 `run_all`）——
① 模块自身（键稳定性、`touch` 后路径不变但判失效、sidecar 缺失/不符、产物截断、自述 converter 不符、
写入顺序/字节数/缺张量/中途异常不留半成品、配额与余量回退、三条环境变量、**AST 反验不 import torch/numpy**）；
② 真装载接线（首次写 / 二次命中且**反量化计数为 0** / **命中与现算逐位相等** / 权重不是全零常量 /
源变更换键 / 坏产物重算且装载照常 / `materialize=False` / 全浮点不物化 / 超配额装载仍成功）；
③ 可见面（`describe()["materialize"]`：装载前 `None` / 装载后**逐字**等于装载结论 / 命中标志没被拷贝吃掉）。

⚠️ 本轮**只用假 GGUF（替身 `inspect` ＋ `_read_tensor`）验**，真权重那份（398 张量 4B 塔）尚未跑
—— 省多少秒没有实测数，不写。

### G4 低配模拟 —— 已完成（2026-11-07）

**两个开关**（只在调试/验收时设，缺省不生效）：

```
DARME_FAKE_GPU_TOTAL_GIB=8      # 假装卡只有 8 GiB
DARME_FAKE_HOST_GIB=32          # 假装内存只有 32 GiB
```

**口径：只改总量、不改占用。** 假装 8 GiB ⇒ 已用字节原样保留、可用 = 8 − 已用（夹到 0）。
不采用「把可用直接写成 N」的做法 —— 那会让 `usedRatio` 恒为 0，`decide()` 里按占用率判的那一支
永远走不到，判据变成假绿。

**三处读数口必须同时接**（否则只模拟了半台机器）：

| 出口 | 作用 | 落点 |
|---|---|---|
| `device_policy.cuda_memory()` | 唯一的 `(total, free, faked)` 读数 | `probe()` 与 `residency.budget_of_cuda()` 都走它 |
| `device_policy.probe()` | 决策（档位/降级） | `Pressure.faked` 标记，进 `to_dict()` 与回执 |
| `scheduler.host_memory()` | 内存余量（`ensure_host_room` / `assign`） | `_fake_host()` |

**如实标注**：`Pressure.faked=True` 会出现在 `decide()` 的 `pressure` 回执里 —— 模拟值不许冒充真读数。
**写错的变量当场抛**（`abc` / `0` / `-2` / `8GB`），不做「静默当成没模拟」——那会让人以为在 8 GiB 上验过。

**实测踩到的一个口径**：`probe()` 的可用 = 驱动空闲 **+ allocator 缓存块**（`reserved − allocated`）。
所以 22.49 GiB 卡上「驱动空闲 20、已分配 18、已预留 20」时可用是 **22 GiB 而不是 20 GiB**；
模拟 8 GiB 后可用是 8 − 驱动已用 2.49 + 缓存 2 = **7.51 GiB**。判据必须按这个算，否则会
把「缓存块也被模拟吃掉」误判成回归。

**判据**：`tests/engine_fake_resources_test.py` **37 条**（已登记 `run_all`）——
解析四坏法、纯函数三档、`probe` 十一项（含"占用不变"与"faked 可见"）、
`budget_of_cuda` 四档、`host_memory` 六项（含 `host_room_bytes` 联动）、
端到端决策四档（真卡 off / 8 GiB split+degraded / 2 GiB CPU）、**反向验证**
（不给变量时结论必须不同，否则判据是空转）。
回归：device_policy、residency 56、scheduler 52、runtime 83、structure 41 全绿。

### G5 §5.1 可行性预检 + §4.1 层数旋钮 + §4.3 预取分档 —— 已完成（2026-11-07）

三件同轮，共同点是**都不许静默**：给了数必须真按数走、给了不合法就当场拒、报了数必须有凭据。

**① §5.1 可行性预检**（`GET /api/v1/engine/feasibility` ＝ `facade.feasibility()` ＝ CLI `feasibility`）：
纯函数、不装权重、不碰 torch、不读业务库。回执 `verdict`（llmfit 四档）＋ `need/have/decision/
advice`/`size`（尺寸吸附委托画布）/`estimate`/`activation`。三处「不知道就说不知道」：`activation`
恒 `basis=unknown`（没有逐族激活峰值表 ⇒ `perfect` 降一档报 `good`）、`estimate.basis` 只有
`measured|unknown`（样本 < 3 ⇒ `runSeconds=None`，绝不给 `0`）、`loadSeconds` 恒 `None`。

**② §4.1 对外层数旋钮 `gpuLayers`**：`residency.normalize_gpu_layers` 单一真源；
`null/空/auto` ＝没提、`0` ＝整份留内存、`n` ＝前 n 层上卡；拒 `-1`（llama.cpp 口径）、拒
`true/3.5/"abc"`（400）。预算只当上限 ⇒ `capped=True` 但 `fits` 仍 True；非层参数先上卡；
只给层数而 `residency=off` ⇒ 自动顶开 `split` 并回执 `residencyOpenedBy`。8 条装配线全接线
（ast 静态核，例外 `_ensure_image_loaded` 报 `granted=None`）。进装载键且**先归一后拼键**。

**③ §4.3 预取深度按盘速自适应**：`prefetch.DISK_CLASSES`（nvme 2/上限 4、sata 2/2、hdd 1/1、
unknown 2/2）＋ `probe_disk()` 真读 32 MiB 量顺序读吞吐（**1 MiB 块**：`read()` 先申请新的再释放
旧的 ⇒ 探测峰值 ＝ 2×块大小，4 MiB 块会占 8 MiB 并把「关预取峰值 < 单件」那条判据顶红）＋每块盘
只量一次；`resolve_depth()` 优先级 显式 > `DARME_READ_AHEAD` > 盘档默认，显式值截到该盘档上限并
报 `capped`。`weights.load_module_weights(read_ahead=None)` ＝ 自适应，`_READ_AHEAD=2` 降为
「unknown 回落值」；量不出盘速 ⇒ 2 ⇒ 与本轮之前**行为零变化**。观测面：`weights.last_read_ahead()`
＋ `/load` 回执 `readAhead{value,source,requested,capped,disk,mbps,how,why}`。

**判据**：`engine_feasibility_test`、`engine_residency_test` ⑧ 系列、`engine_runtime_test::
case_gpu_layers_wiring`、`engine_router_contract_test` ⑪¹⁶–⑪¹⁸、`engine_readahead_test` **37 条**
（新增 ⑦–⑭）。
回归：residency、runtime、profile、weights_streaming 16、prefetch、readahead 37、env_docs、structure 全绿。

### G6 真机 8 GiB 验证：int8 常驻账目的分母写错了（§4.1／§4.2 交界）—— 已完成（2026-11-07）

**怎么发现的**：拿假预算（`DARME_FAKE_GPU_TOTAL_GIB=8`、`DARME_FAKE_HOST_GIB=32`）真装 Krea2
（`residency="split"`）⇒ **整份被拒装**，报

> int8 常驻层没接管齐 ✗ … 文件里只有 **88** 个 int8 权重被原样装进常驻层，而计划换掉的是 **224** 层

对照：**同一台机器、同一条文件**、`residency="off"` 只花 5.7 s 就装进去了 ⇒ 判据错，不是权重错。

**根因（两段各自正确、交界处对不上）**：§4.2 的「内存这一级」按 `skip_prefixes` 把用不上的键
**故意不写进模块**（留盘／内存，靠 `report.restorer` 重读 ✓，`report.skipped` 计数 ✓）；
而 `torch_backend._int8_residency_problems` 的分母只取 `int8Resident`（文件侧真写进去的层数）⇒
**卸得越多、`int8Resident` 越小、越被判成「差的层会按 bf16 落盘」** ⇒ 内存越紧张越装不上 ✗✗
（实测 224 层里 88 层写进去、136 层卸下去）。也就是说：**省内存这条策略本身会把自己判死**。

**修法（三处，一条不能少）**：

1. `weights.LoadReport.int8Skipped`（新）＋ 在 `skip_prefixes` 剔除块里记账 —— 只数「卸下去的这几层
   **本来是不是 int8 常驻**」（`logical in resident`），不是数全部 `skipped`；
2. 分母改成 `int8Resident + int8Skipped`，且报错串里**两个数都写出来** + 差的层数（`layers - accounted`）
   —— 只说「差 136 层」而不说「其中 136 层是策略卸的」等于把账做成了谜语；
3. **卸了就必须填得回来** ✗✗：`int8Skipped > 0` 而 `report.restorer is None` ⇒ **照样拒装**，并说明
   「那几层填不回来、前向到那层就是空壳」—— 放宽分母**不等于**放过真洞（`restorer=None` 意味着这次
   装载没走到逐张量那一步，盘上的权重没人会去重读）。

观测面：`int8Skipped` 同时进 `_int8_residency_problems` 的账目 dict 与 `LoadReport.to_dict()`
（后者连带 `skipped` 一起补 —— 回执里只看得到 `loadedKeys` 变小、读不出「是策略卸的还是真缺」）。

**实测（修后，同假预算）**：`int8Resident=96 / int8Skipped=128 / 已交代 224 ✓`，装载通过；
回执 `residencyTotal.gpuGiB=4.88 / cpuGiB=1.21 / gpuShare=0.8013`、`ditNeed{12.57,'table','int8'}`、
`lastDecision.residency='split'` 且 `degraded=True`（兜底要如实说降级 ✓）。

**同轮顺手清掉的两笔对外契约债**（都是「自述落后于代码」）：

- `SCHEMA_VERSION` **6→7** ＋ `docs/engine-api.schema.json` 补 `status().profile/budgets/diskCache`
  （顶层加键＝破坏性变更 ⇒ 必须升版本；三条来自 §4.4／§5.2 那两轮，`engine_router_contract_test` ⑨ 一直红）；
  顺带把这三格的**关键嵌套**也登记进去（`budgets` 10 格、`profile` 4 格、`diskCache` 16 格）。
- `docs/env/hardware-report.json`：`schemaVersion` 6→7；并修两处**没人对账**的陈旧 —— `quant.scaleSuffixes`
  少写 `.scale_weight`、`quant.supportedFormats` 把 `convrot_w4a4` / 两种 fp8 漏在外面（读的人会按旧表
  判某份权重「不支持」✗）。`engine_env_docs_test` 的 quant 对账栏补这两项，防再漂。

**判据**：`engine_int8_resident_test` **25 条**（⑧ 系列新增 ⑧⁗「卸下去的那层也算交代清了」、
⑧⁵「卸了但没还原器 ⇒ 照样拒装」、⑧⁶「既没写也没卸 ⇒ 仍报错」）＋ `engine_weights_streaming_test` ①′
（`to_dict` 键集合含 `skipped`/`int8Skipped`，仍守「只增不改」）＋ `engine_router_contract_test` 39/39 ＋
`engine_env_docs_test` 32/32。
回归：int8_resident 25、weights_streaming 16、host_cache 37、materialize 56、loader 33、quant 60、
runtime 92、residency 80、profile 71、hybrid_load 32、structure 41 全绿。

### G7 假 8 GiB 卡跑完整张图：换装两处「省内存反把自己判死」—— 已完成（2026-11-07）

G6 让 Krea2 **装得上**了 ✓，接着往下跑到 `encode → init → sample → decode` ✓，又撞两处 ✓✗。
两处的共同形状与 §4.1／§4.2 那条一样：**两段各自正确，交界处口径不一致** ✓。

#### ① 退件说退了，卡上一字节没退（`release_part` 认不出「摆的是本体、存的是外壳」）

`attach_krea2_te` 摆位置时传的是 **`module.tower`**（要让驻留看得到 `model.layers.N` ✓），
而 `self._textEncoder` 存的是**外壳** `module`（参数全在 tower ✓）。
`release_part` 原先只判 `holder.module is self._textEncoder` ⇒ **恒 False** ⇒
`residency.Handle` 不 `detach()` ⇒ 钩子环攥着每一层 ⇒ `gc.collect()` 收不掉 ⇒
**卡上那 6.41 GiB 一字节不退** ✓✗（实测：`release_part("text")` 前后
`torch.cuda.memory_allocated` 都是 6.41 GiB，而回执写 `released: true` ✓ 这就是静默的那种错 ✓）。

连带第二笔：账本按 `what` 记（`"te"` ✓），`release_part` 按 `part` 退（`"text"` ✓）
⇒ `release("text")` 找不到键 ⇒ **只涨不退**，`residencyTotal` 里那件永远还在 ✓✗。

**修法**：立一张 `_WHAT_TO_PART`（`dit/h3 → dit`、`te/h3-te → text` ✓），
摆放时把用的名字记在 `_residencyOwnerWhat` 上 ✓；`release_part` 改成**两条判据任一成立即摘**
（同一性 ✓ ／ 名字归一后等于本件 ✓），账本则把该件的**所有别名**（`part` 与 `_WHAT_TO_PART`
里指向它的名字，各带 `#host`）一起退 ✓。
反方向也钉住：退 `text` **不许**掀掉 `dit` 的把手 ✓✗（那会把 DiT 的钩子拆了 ⇒ 下次前向层不再搬 ✓）。

修后实测（同假预算）：`release_part("text")` ⇒ `alloc 6.41 → 0.00 GiB`、`free 0.38 → 6.79 GiB` ✓。

#### ② 换装那一拍把「已在场的件」又算了一遍需求（`enter_phase`）

`decide_for` 的 `need` 是「这几件加起来多大」，而 `free` 是**此刻**卡上余量 ——
已在场那件**本身就占着** `free` 里被扣掉的那几 GiB ⇒ 按整份 `wanted` 算＝**数了两遍** ✓✗
⇒ 需求虚高 ⇒ 判 `cpu/off` ⇒ `ensure_host_room` 要「整份进内存」⇒ 宿主不够 ⇒ **整条链拒装**。

实测：`init` 那拍 `text` 已在场（6.41 GiB）、只差 **0.24 GiB** 的 `vae`，却按「text+vae = 7.74 GiB」
判 ⇒ 卡上只剩 0.38 GiB ⇒ 转 CPU ⇒ 要 7.74 GiB 内存（宿主只剩 3.59 GiB）⇒
报「要整份放进内存」✓✗ —— **看着像内存不够，其实只差一个 0.24 GiB 的 VAE**，
而同机 `residency="off"` 装得进 ✓。

**修法**：`enter_phase` 只把 **`missing = [p for p in wanted if not present[p]]`** 交给决策 ✓
（全部在场时早就在 `reused` 分支返回了 ✓，所以 `missing` 非空 ✓）。

**修后实测（假 8 GiB 卡 / 假 32 GiB 宿主 / Krea2）**：三拍全过 ✓ ——
`encode` ⇒ `cuda/split`（text 6.41 GiB 上卡 ✓）；退 text ⇒ `free 6.79` ✓；
`init` ⇒ `loaded: [text, vae]` ✓；`sample` ⇒ `released: [text, vae]`、`loaded: [dit]`、
`alloc 4.88 GiB`／`free 1.91 GiB` ✓。

#### 判据

`engine_device_policy_test` 新增 ⑨⁗（摆本体存外壳 ⇒ 把手照样摘得掉 ＋ 账本 `te` 也退 ✓）、
⑨⁘（退 text 不许掀 dit 的把手 ✓）、⑮（`enter_phase` 只把还缺的那件交给决策 ✓，
替身跑、不读权重 ✓）；`engine_int8_resident_test` 25／`engine_residency_test` 80 仍全绿。

#### ③ 全链真跑（假 8 GiB 卡 / 假 32 GiB 宿主 / Krea2 / `residency="split"`）

| 档 | 装配 | encode | sample | decode | 整链 | 产物 |
|---|---|---|---|---|---|---|
| 320×320 · 1 步 | 3.7 s | 21.4 s | 13.5 s | 1.3 s | **36.2 s** | PNG 114 KB ✓ |
| 1056×1056 · 1 步 | 3.7 s | 20.3 s | 51.3 s | 1.1 s | **72.8 s** | — |
| 1056×1056 · 3 步 | — | 19.6 s | 150.4 s | 1.3 s | **175.2 s** | — |
| **1024×1024 · 8 步** | 3.7 s | 19.4 s | 375.5 s | 1.2 s | **400.1 s** | **PNG 1 087 063 B · 1024×1024 · `blank=false` ✓** |

对照：**同一张真卡（22.49 GiB）走 `off` 整份上卡是 51.4 s**（1040×1040 / 8 步）⇒
8 GiB 档的 `sample` 慢 **7.3 倍** —— 这就是"约一半层落在 CPU 上算"的代价 ✓（如实记，不当缺陷 ✗）。
换装三拍全过、`released` 真退（`alloc` 6.41 → 0.00 GiB ✓）、`degraded=True` 如实自报 ✓。

⚠️ **一处必须写下来的误判**（免得后人当 bug 追 ✓✗）：验证过程中有两次跑到 `sample` 就
**整进程无声消失**（无 traceback、无回执 ✓，`faulthandler(all_threads=True)` 也没留帧 ✓）。
真因不是代码：那两次我**同时**在跑几套引擎测试（`runtime`／`materialize`／`profile` ✓），
几份进程一起抢同一张真卡的显存与宿主内存 ⇒ 被系统/CUDA 原生层终结 ✓✗。
**单跑 400.1 s 正常收尾 `status=done`** ✓。教训：低配验证**不许与其他跑图/测试并发** ✓，
否则测出来的是"资源抢不到"，不是"低配跑不了" ✓。

⚠️ 这两处都**只在"卡真的不够"时才露头**：真 24 GiB 卡上 Krea2 走 `off`（整份上卡 ✓）⇒
`release_part` 退的是普通模块（同一性判据成立 ✓）、换装只有一拍在装 ✓ ⇒ 永远不会触发 ✓✗。
低配这条路必须先有 `split` 才打得开 —— 这也是 §4.1「先有降级路径、再谈降级策略」的实证 ✓。

### G8 "400 秒 vs ComfyUI 42 秒"的账：换装税与两个不同档 —— 已完成（2026-11-07）

创作者拿 ComfyUI（同机、同 Krea2、1024×1024）**42 秒**来问 G7 那张 **400.1 秒**。对账结论：
**两笔不是同一档**，而真卡上我们本来就在 **45 秒级**（≈ 42 秒），400 秒只出现在"卡只有 8 GiB"
这个被伪造出来的低配档里。但这一轮也确实抓到一笔**每张图都要付**的真税 ✓✗（见 ②）。

#### ① 口径先对齐（串行单跑，不与任何测试并发 ✓）

| 档 | 权重 | 步数 | 尺寸 | 驻留 | 装配 | encode | sample | decode | 图 |
|---|---|---|---|---|---|---|---|---|---|
| 真卡 22.49 GiB | int8 12.57 | 4 | 1056² | `off`（分时） | 64.1 s（冷读盘） | **27.4 s** | 16.8 s | 1.2 s | **45.6 s** |
| 真卡 22.49 GiB | **int4 5.98** | 4 | 1024² | `off`（**全驻留**） | 5.1 s | 21.2 s（首张装三件） | 42.2 s | 0.4 s | **63.9 s**；第 2 张起 **43 s** |
| 假 8 GiB / 假 32 GiB | int8 12.57 | 8 | 1024² | `split` | 3.7 s | 19.4 s | **375.5 s** | 1.2 s | **400.1 s** |
| 假 8 GiB | int8 12.57 | 4 | 1024² | **`park`** | 6.2 s | 28.3 s | **261.7 s**（65.4 s/步） | 1.7 s | **291.9 s** |

三条读数：
1. **真卡 int8 是 45.6 s**（不算冷装配）⇒ 与 ComfyUI 的 42 s 同一档 ✓；**400 s 只属于 8 GiB 档** ✓。
2. `split` 的代价全在 `sample`：47 s/步 vs 整份上卡 4.2 s/步（1056² int8）✓ —— 约 14% 的层在 CPU 上
   用 bf16 现算 ✓（不是搬运慢，是 CPU 算得慢 ✓）。
3. **`park` 比 `split` 更慢**（65 s/步）✓✗ —— 与仓库里那条"auto 不选 park"的结论**同结论、不同理由**：
   原理由写的是"PCIe ~25 GB/s ⇒ 3 秒+/步"✓，实测每步搬 12.57 GiB 却要 **约 60 s** ⇒
   真实搬运速率只有 **约 200 MB/s** ✓✗（裸 `copy_` 实测 17–25 GiB/s ⇒ 差 100 倍，**不是 PCIe 的账** ✗）。
   已登记为待查（见 ④），在那之前 park 仍不该进 `auto` ✓。

#### ② 每张图都付一次的"换装税"：装得下就别分时（本轮已修 ✓）

int8 那 45.6 s 里 **27.4 s（60%）在 encode 换装** ✓✗：`text` 被 `sample` 那一拍退掉，下一张图
再**从盘上重读一遍文本塔** ✓ —— 与提示词内容无关 ✓。三件之和：int8 **20.31 GiB**（分时 ✓）、
int4 **13.72 GiB**（装得下 ✓）。

修法（`runtime._ensure_krea2_loaded`）：装配**前**按**整份** `part_need` 问一次设备策略 ✓，
判成 `off` 就 `declare_chain(phase_parts=KREA2_ALL_RESIDENT_PHASES)`（四段都点名三件 ⇒
`enter_phase` 一件都不退 ✓），否则照老路给 `None`（互斥分时 ✓，零变化 ✓）。回执加 `allResident`。

实测（真卡 int4，两张连跑）：第 1 张 `换装 encode :: 20692 ms 装 [dit,text,vae]`、
第 2 张 `换装 encode :: 0 ms 复用 [dit,text,vae]` ✓；`sample`/`decode` 两拍恒 0 ms ✓。

⚠️ 修的过程里踩到一个**只有真跑才露头**的坑 ✓✗：门闩写成 `verdict.get("residency")` 而
`decide_for` 返的是 **`Decision` dataclass** ✓ ⇒ `AttributeError` 被 `except` 兜住 ⇒
**静默变成"永远分时"** ✓✗（回执一切正常、只是全驻留永不生效 ✓）。判据 `engine_krea2_test` ⑦⁗
用桩件**同时喂 `Decision` 与 `dict`** 两种形状 ＋ 抛异常退回分时，三种都要对 ✓。

#### ②′ int8 这种**装不下三件**的档：退 ≠ 重读（`keepParts` ✓ 2026-11-07 同日第二轮）

②只救 int4（整份装得下 ✓）。int8 **20.31 GiB** 挤不进 22.49（还要留 1 GiB 余量 ＋ 1.21 GiB
torch 上下文 ✓）⇒ 必须分时 ✓，而分时的老路是「`sample` 把文本塔**退掉**（`release_part` ✓）
⇒ 下一张 `encode` **从盘上重读一遍 7.5 GiB**」✓✗。真卡实测（int8 / 1024² / 4 步 / 两张连跑）：

| 版本 | 第 1 张 | 第 2 张 | 第 2 张 `encode` 明细 |
|---|---|---|---|
| 老路（退即重读） | 37.8 s | **35.1 s** | `换装 encode 17515 ms 装 ['text']` |
| `keepParts = (dit, text)` | 40.0 s | **16.1 s** | `换装 encode 1211 ms 退 ['vae'] 复用 ['text']` |
| `keepParts = (dit, text, vae)` | 37.6 s | **15.1 s** | `换装 encode 1035 ms 退 [] 复用 ['text']` |

关键判据是**换装回执里的 `loaded`**：老路第 2 张 `装 ['text']`（＝读盘 7.5 GiB ✓），
`keepParts` 之后三拍**全是 `loaded: []`** ✓（`encode` 复用 `text`、`sample` 复用 `dit`、
`decode` 复用 `vae` ✓）—— 剩下的 1.0–1.5 s 是**纯设备拷贝**（7.5 GiB / 1.0 s ≈ 7.5 GiB/s ✓）。

修法：`runtime._krea2_keep_parts` 逐件问 `_krea2_decision`，**只收判成 `off` 的件** ✓✗
（`split` 的件混进 `keepParts` ⇒ `_relocate_keep_parts` 把它**整份**搬去 cpu 再整份搬回卡上 ✓
⇒ 分层计划当场作废、搬回时直接 OOM ✓，8 GiB 档正是这种 ✓）；`declare_chain(keep_parts=…)` 接住 ✓。
`torch_backend.enter_phase` 的顺序是**先腾卡（`to_host_only=True`）再决策再装** ✓✗ —— 反了就是
「往满卡里塞」⇒ OOM ✓；两条分支（复用全命中 / 部分命中）都必须先腾 ✓（早退那条同样会被占着的件堵住 ✓）。
`_pinned_gib` 按 `_WHAT_TO_PART` 的**别名全试一遍** ✓✗（文本塔摆放时叫 `te`、查询叫 `text` ✓，
按单名查账本会读到 **0 字节** ⇒ 后续余量判断失真 ✓）。

⭐ **质量红线（创作者："按你说的来但是不能质量掉"）**：搬的是**同一份权重字节** ✓ ——
`nn.Module.to()` 同 dtype 只做设备拷贝 ✓、不转精度 ✗、步数 / 采样参数 / 量化档一律不动 ✓。
出图实证：同一 seed 三批（老路 / `(dit,text)` / `(dit,text,vae)`）落盘 PNG **sha256 逐字节相同** ✓：

```
seed7  1110732 B  BF6FA7620021B6E0…   ×3 批全同
seed8  1129987 B  24641AF1C655CDB0…   ×3 批全同
```

判据 `engine_krea2_test` ⑦⁷：含 int8 参数 ＋ fp32 / bf16 / int64 buffer 的 `nn.Module`
走 `to("cuda").to("cpu")` 一圈，逐位 `torch.equal` ＋ dtype 不变 ✓（无 CUDA 则跳过，是设备事实不是放宽 ✓）。
回执新增 `keepParts` / `keptOnHost{dit,text,vae}`，`note` 按"停内存 / 逐件都 split 只能重读"两支如实写 ✓。

#### ③ int4 每步比 int8 慢 2.5 倍（**未解，已定界**）

同卡同尺寸：int8 **4.2 s/步**、int4 **10.5 s/步**（1024²，都是整份驻留）。已排除的：
`unpack_int4` —— 单层 6144×3072 打包张量在卡上 **1.3 ms**（改成位运算 `<<4 >>4` ＋ 交错写
后 **0.6 ms**，逐位与旧实现相同 ✓），224 层 × 4 步也只有 1.2 s ⇒ **不足以解释 6 s/步** ✓；
"反量化＋矩阵乘"的合成基准 int8 5.12 ms/层 vs int4 5.51 ms/层 ✓ ⇒ 也不是它。
⇒ 差异在**模型级**（同一族两份权重的 convrot 分组 / 层形状 / 门控实现可能有别），**未定界** ✓✗。
工程结论：**能装 int8 就用 int8**（每步快 2.5 倍）；int4 的价值只在"装得下"（5.98 GiB ⇒ 全驻留免换装税）。

#### ④ 待查清单（都带可复现档位，别再从头猜 ✓）

1. **逐张量材质化只有约 250 MB/s**：`装配 64.1 s / 12.57 GiB`、`换装 encode 27.4 s / 7.5 GiB`、
   `park 每步约 60 s / 12.57 GiB` 三条**同一量级** ✓；而裸 `copy_` 是 17–25 GiB/s ✓、
   224 块 × 919 MB 分块搬也未见退化 ✓ ⇒ 嫌疑在**逐张量 `cudaMalloc`（Windows/WDDM 上单次 5–10 ms ×
   数千张量）**或每层现反量化，未验证 ✓。
   ⭐ 新证据（②′）：`keepParts` 用 `module.to()` 搬 7.5 GiB 只要 **约 1.0 s（≈7.5 GiB/s ✓）** ⇒
   `to()` 本身**不是**瓶颈 ✓；那 27.4 s 里大头是「读盘 ＋ 建壳 ＋ 逐张量装」（`loaded` 那一路 ✓）。
   这条把范围收窄了一半：**只查"装"这一路** ✓。
2. **`foresight`（层预取）在假 8 GiB ＋ `slots=2` 下卡死在 `enter_phase("sample")`**
   （`换装 sample` 一行都没打出来；进程 5 分钟只烧 25 s CPU、卡上 883 MiB、GPU util 14% ✓）——
   复现：`DARME_FAKE_GPU_TOTAL_GIB=8 DARME_FORESIGHT_SLOTS=2 RES=auto`。
   这条**不能当"已修"记**：它是低配档唯一可能把 CPU 侧的层搬回卡上算的路径 ✓，卡死等于不可用 ✓✗。
3. int4 每步 2.5 倍（见 ③）。

#### 判据

`engine_krea2_test` **43 条**（⑦ 系列：表四段都点名三件 ✓、`declare_chain` 真收到
`phase_parts=` 且按门闩给 ✓、门闩走 `decide_for` 而非自算阈值 ✓、`keep_parts=` 真给到 ✓、
⑦⁶ 只收 `off` 件 ✓、⑦⁗ 两种返回形状 ✓、⑦⁷ `cpu→cuda→cpu` 逐位不变 ✓、
int4 判 `off` / int8 不判 `off` ✓）。回归：krea2 43/43 ✓。

