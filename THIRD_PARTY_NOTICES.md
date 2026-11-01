# 第三方许可审计（初稿）

> WBS E0.6。**本文件只登记事实，不替上游补许可** ✗：清单里没有 `license` 就写「未声明」，
> 绝不按印象填 `Apache-2.0` / `MIT` 之类 ✓✗（判据见 §四，写错就红）。
>
> 采集环境：Python 3.12.10 · torch 2.14.1+cu130 · 采集日 2026-10-30
> 采集方式：权重行取自 `configs/models.json`；依赖许可取自**已安装包的元数据**
> （`importlib.metadata` 的 `License-Expression` / `License` / `License ::` 分类器），不靠记忆 ✓。

## 一、模型权重（26 条）

清单 `configs/models.json` 每条都有一个 `license` 位，**目前 26 条全为空** ⇒ 下表许可列一律「未声明」。
其中 6 条连落点（`filename`）都还没有，`facade.manifest_of()` 对它们无从谈起（该接口按**文件**出清单）。

> 补登说明（2026-10-31）：此前本节停在 16 条，而 catalog 已长到 26 条 —— 新增的 10 条（Z-Image 三件套、
> Qwen-Image 2.1/2512/Edit 2511 与它们的 TE·VAE）**一直在清单里、却不在本表**，`engine_licenses_test`
> 的「权重表条目 == 清单条目」正是盯这件事。缺的后果不是"页面难看"：本表是**下游判断再分发条款的
> 唯一入口**，漏一条就等于那件权重在合规面上不存在 ✓✗。

| key | filename | category | 许可 | 依据 |
|---|---|---|---|---|
| dit_fl2va_int8 | minimax_h3_fl2va_pruned_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_ref2va_int8 | minimax_h3_ref2va_pruned_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
| te_nvfp4 | qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | video | 未声明 | 清单无 `license` 字段 |
| te_qwen3_4b | qwen_3_4b.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_video_fp16 | minimax_h3_video_vae_fp16.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_audio_fp32 | minimax_h3_audio_vae_fp32.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_video_fp8mix | minimax_h3_video_vae_fp8mix.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_audio_bf16 | minimax_h3_audio_vae_bf16.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_fl2va_gguf_q4km | minimax_h3_fl2va_pruned-Q4_K_M.gguf | video | 未声明 | 清单无 `license` 字段 |
| sdxl_base | sd_xl_base_1.0.safetensors | image | 未声明 | 清单无 `license` 字段 |
| dit_zimage_turbo | z_image_turbo_bf16.safetensors | image | 未声明 | 清单无 `license` 字段 |
| te_zimage_qwen3 | qwen_3_4b.safetensors | image | 未声明 | 清单无 `license` 字段 |
| vae_flux_ae | ae.safetensors | image | 未声明 | 清单无 `license` 字段 |
| dit_qwen_image21 | qwen_image_2.1_int8_convrot.safetensors | image | 未声明 | 清单无 `license` 字段 |
| te_qwen35_image21 | qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors | image | 未声明 | 清单无 `license` 字段 |
| vae_qwen_image21 | qwen_image_2.1_vae_bf16.safetensors | image | 未声明 | 清单无 `license` 字段 |
| dit_qwen_image_2512 | qwen_image_2512_bf16.safetensors | image | 未声明 | 清单无 `license` 字段 |
| dit_qwen_image_edit_2511 | qwen_image_edit_2511_int8_convrot.safetensors | image | 未声明 | 清单无 `license` 字段 |
| te_qwen25_vl_7b | qwen_2.5_vl_7b_fp8_scaled.safetensors | image | 未声明 | 清单无 `license` 字段 |
| vae_qwen_image | qwen_image_vae.safetensors | image | 未声明 | 清单无 `license` 字段 |
| qwen3_14b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| qwen3_32b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| glm4_9b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| deepseek_r1_14b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| cosyvoice2 | —（无落点） | tts | 未声明 | 清单行 `filename` 为空 |
| wan26_i2v_14b_gguf | —（无落点） | video | 未声明 | 清单行 `filename` 为空 |

**为什么这一列空着**：这些权重来自多个上游（H3 主线各仓、Comfy-Org 权重镜像、fp8/GGUF 再分发仓、
SDXL 官方仓），再分发条款各不相同，有「需随附许可原文」的、也有「仅限研究与个人使用」的 ⇒
必须**逐件核上游 LICENSE 文件**后再填 ✓。按印象填一个通用的 `Apache-2.0` 会让下游误以为可商用 ✗✗，
比留空危害大得多 ✓。

## 二、Python 依赖

许可取自包元数据（`License-Expression` 优先，其次 `License`，再次 `License ::` 分类器）。
版本是**本机实测**，不是 requirements 的下限。

| 包 | 版本 | 许可 | 来源字段 |
|---|---|---|---|
| fastapi | 0.162.0 | MIT | `License` |
| uvicorn | 0.54.0 | BSD-3-Clause | `License` |
| SQLAlchemy | 2.1.6 | MIT | `License` |
| PyYAML | 6.0.3 | MIT | `License` |
| httpx | 0.28.1 | BSD-3-Clause | `License` |
| python-multipart | 0.0.32 | Apache-2.0 | `License` |
| torch | 2.14.1+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | `License`（PEP 639 表达式，未截断） |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `License`（PEP 639 表达式） |
| safetensors | 0.8.0 | Apache-2.0 | `License ::` 分类器（该包 `License` 字段为空） |
| pillow | 13.0.0 | MIT-CMU | `License` |
| transformers | 5.21.0 | Apache 2.0 License | `License`（上游写法如此） |

⚠️ 三者口径注意：

* `torch` / `numpy` 是 **PEP 639 表达式**（多许可并列），照抄原值即可 —— 不要"简化"成单一 `Apache-2.0`：
  它们各自捆绑了 LLVM 例外、BSD、CC0 等子部件，简化即失真 ✗。
* `safetensors` 的 `License` 字段是**空串**，许可只写在分类器里 ⇒ 读元数据必须**三级回落**，
  否则会把一个 Apache-2.0 的包记成「未声明」✗。
* `transformers` 上游写的就是 `Apache 2.0 License`（不是 SPDX 的 `Apache-2.0`）——照抄，不校正。

## 三、缺口清单（待创作者/上游确认）

1. **权重 16/16 条许可证为空** ⇒ 商用边界目前**不可判定**。补法：逐件在清单里加
   `"license": "<SPDX 或上游原文>"`，然后 `facade.manifest_of(path)` 会自动把它带出来
   （该接口已有 `license` / `licenseNote` 两个位，见 `app/services/engine/facade.py`）。
2. **6 条无落点的条目**（`qwen3_14b` / `qwen3_32b` / `glm4_9b` / `deepseek_r1_14b` / `cosyvoice2` /
   `wan26_i2v_14b_gguf`）连文件名都没有 ⇒ 既无清单可出，也无许可可核。
3. **`reference/` 下的第三方源仓**（Comfy-Org 若干、diffusers、EveryDream2trainer、lora-pilot 等）
   是**阅读用的参考副本**、不进任何产物链路；它们的 LICENSE 随各自目录留在原地，本文件不重述。
   ⚠️ **例外只有两条**：`expert-stream` 与 `airllm` 各有一小段被移植进引擎
   （`app/services/engine/win_io.py` / `prefetch.py`）⇒ 已登记在下面第五节，
   许可原文随代码入仓 `docs/third-party/`；除这两条外，`reference/` 仍不进产物链路。

## 四、机器判据

`tests/engine_licenses_test.py` 钉住本文件，四条：

1. **条目不多不少**：权重表的 key 集合 == `configs/models.json` 的 key 集合（双向，少一条＝漏登、
   多一条＝文档里有盘上不存在的模型）。
2. **许可列与清单逐条一致**：清单某条有 `license` ⇒ 该行必须逐字等于它；清单没有 ⇒ 该行必须是「未声明」
   （这条就是「不猜许可」的机器形态：谁手写一个 `Apache-2.0` 进去，立刻红）。
3. **许可列不许出现任何具体许可名**（`Apache` / `MIT` / `BSD` / `GPL` / `OpenRAIL` 等词表）——
   与第 2 条互为冗余：即便清单将来加了一个许可，也只能从清单来，不能从人脑来。
4. **依赖版本与实测一致**：§二 每一行的版本 == 本机 `importlib.metadata.version(<包>)`；
   包没装 ⇒ 该行不许写版本（写「未安装」）。依赖升级后本文件会红 —— 这是**故意的**，
   提醒刷新审计，而不是巧合。

## 五、参考实现源码（移植）

**只有两条**（`reference/` 下其余仓一件代码都没进来，理由见各仓审计结论）：

| 上游仓 | 许可（原文入仓） | 被移植的提交 | 落在本仓 | 取了什么 | **没取什么** |
|---|---|---|---|---|---|
| `expert-stream`（Hemant） | MIT ⇒ `docs/third-party/expert-stream-LICENSE.txt` | `89275f1c50d2`（2026-08-16） | `backend-py/app/services/engine/win_io.py`（并由 `safetensors.py::TensorReader` 调用） | `engine/expert-stream.c` 的 Win32 三件：`CreateFileW` 的 `NO_BUFFERING\|OVERLAPPED` 标志、一次提交多笔 `ReadFile` ＋ 每槽手动复位事件、`VirtualAlloc` 槽区按扇区对齐并用 `slop` 吃掉非对齐偏移 | ggml 侧任何符号、n-gram/路由预取、`manifest` 专家表（那些绑 284B MoE ✓，与本仓 DiT 无关 ✗） |
| `airllm`（lyogavin） | Apache-2.0 ⇒ `docs/third-party/airllm-LICENSE.txt` | `75a6bb3c41c6`（2026-10-31） | `backend-py/app/services/engine/prefetch.py`（并由 `weights.py::load_module_weights(read_ahead=)` 调用） | `airllm_base.py` 的 `_prefetch_stream` / `_materialize` / `_evict` 三处调度语义：后台读下一件、取走即作废、取不到就同步读 | 逐层 `state_dict` 挂载、`ModelPersister`、bitsandbytes 压缩路径、按 token 逐层重读整模型的范式（那是 0.1–0.4 tok/s 的来源 ✗✗） |

**移植时必须一起改掉的三处上游习惯**（否则进不了本仓）：

1. **错误不许静默**：上游「读失败 ⇒ 标 -1 让上层回落 mmap」在本仓不可接受（会装出半件模型还装得上）
   ⇒ 移植件一律抛 `OSError` / `PrefetchError`；
2. **状态必须可查**：上游靠 `print` ⇒ 移植件给 `describe()` / `snapshot()`，直接喂 `GET /engine/runtime`；
3. **不许新增第二真源**：`win_io.py` 只读字节（偏移由 `safetensors.py` 算）、`prefetch.py` 只管调度
   （装什么由装配线给）⇒ 两件都不许自己判断「该装哪件权重」。

**怎么核**：`tests/engine_win_io_test.py`（9 组，含「三笔同时在飞 ⇒ 无事件就返回 0 字节」这条真事故回归）
与 `tests/engine_prefetch_test.py`（9 组，含「同一件不许取两次」）。两份都是纯逻辑、不碰权重、不 import torch。
