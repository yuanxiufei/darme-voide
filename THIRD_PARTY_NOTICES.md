# 第三方许可审计（初稿）

> WBS E0.6。**本文件只登记事实，不替上游补许可** ✗：清单里没有 `license` 就写「未声明」，
> 绝不按印象填 `Apache-2.0` / `MIT` 之类 ✓✗（判据见 §四，写错就红）。
>
> 采集环境：Python 3.12.10 · torch 2.14.1+cu130 · 采集日 2026-10-30
> 采集方式：权重行取自 `configs/models.json`；依赖许可取自**已安装包的元数据**
> （`importlib.metadata` 的 `License-Expression` / `License` / `License ::` 分类器），不靠记忆 ✓。

## 一、模型权重（16 条）

清单 `configs/models.json` 每条都有一个 `license` 位，**目前 16 条全为空** ⇒ 下表许可列一律「未声明」。
其中 6 条连落点（`filename`）都还没有，`facade.manifest_of()` 对它们无从谈起（该接口按**文件**出清单）。

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
