# 第三方许可审计（初稿）

> WBS E0.6。**本文件只登记事实，不替上游补许可** ✗：清单里没有 `license` 就写「未声明」，
> 绝不按印象填 `Apache-2.0` / `MIT` 之类 ✓✗（判据见 §四，写错就红）。
>
> 采集环境：Python 3.12.10 · torch 2.14.1+cu130 · 采集日 2026-10-30
> 采集方式：权重行取自 `configs/models.json`；依赖许可取自**已安装包的元数据**
> （`importlib.metadata` 的 `License-Expression` / `License` / `License ::` 分类器），不靠记忆 ✓。

## 一、模型权重（36 条）

清单 `configs/models.json` 每条都有一个 `license` 位，**目前 36 条全为空** ⇒ 下表许可列一律「未声明」。
其中 6 条连落点（`filename`）都还没有，`facade.manifest_of()` 对它们无从谈起（该接口按**文件**出清单）。

> 补登说明（2026-10-31）：此前本节停在 16 条，而 catalog 已长到 26 条 —— 新增的 10 条（Z-Image 三件套、
> Qwen-Image 2.1/2512/Edit 2511 与它们的 TE·VAE）**一直在清单里、却不在本表**，`engine_licenses_test`
> 的「权重表条目 == 清单条目」正是盯这件事。缺的后果不是"页面难看"：本表是**下游判断再分发条款的
> 唯一入口**，漏一条就等于那件权重在合规面上不存在 ✓✗。
>
> 补登说明（2026-11-01）：catalog 再长到 29 条，Krea2 三件套（`dit_krea2_turbo` / `te_krea2_qwen3vl4b` /
> `vae_krea2`）同样一直只在清单里、不在本表 ⇒ 同一条判据再次报出 ✓。补的是登记，不是许可结论：
> 三件仍「未声明」。
>
> 补登说明（2026-11-02）：Wan 2.2 I2V 接线，catalog 长到 33 条 —— 四件（高噪/低噪双专家、UMT5-XXL
> 文本塔、Wan 2.1 VAE）入表 ✓。同前：**只登记、不结论**，四件仍「未声明」（上游 Wan 系再分发条款
> 需逐件核原仓 LICENSE ✗ 不许按印象填 Apache-2.0 ✓✗）。
>
> 补登说明（2026-11-04）：H3 走自研引擎后新增 `te_h3_int8_convrot`（Qwen3-VL-32B int8_convrot，
> 25.28 GiB，`attach_h3_text` 唯一认的那份），catalog 长到 36 条、本节补其一行 —— 同上，
> **只登记、不结论**，仍「未声明」。
>
> 补登说明（2026-11-06）：Music3 音频链接线（`stage="music3"` 三件同驻），catalog 长到 **39 条** ——
> 三件（音频 DiT fp16 / 语义塔 int8_convrot（**词表焊在权重里**）/ DAV 音频解码器）入表 ✓。
> 同上：**只登记、不结论**，三件仍「未声明」—— Music3 上游条款要逐件核原仓 LICENSE ✗，
> 尤其「词表内含」这一点**不影响**许可判定，别拿"只是编码器"去套通用条款 ✓✗。

| key | filename | category | 许可 | 依据 |
|---|---|---|---|---|
| dit_fl2va_int8 | minimax_h3_fl2va_pruned_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_ref2va_int8 | minimax_h3_ref2va_pruned_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
| te_nvfp4 | qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | video | 未声明 | 清单无 `license` 字段 |
| te_h3_int8_convrot | qwen3vl_32b_minimax_h3_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
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
| dit_krea2_turbo | Krea2/krea2_turbo_int8_convrot.safetensors | image | 未声明 | 清单无 `license` 字段 |
| te_krea2_qwen3vl4b | Qwen3_VL_4B/Krea2-Engineer-V1-int8-convrot.safetensors | image | 未声明 | 清单无 `license` 字段 |
| vae_krea2 | qwen_image_vae.safetensors | image | 未声明 | 清单无 `license` 字段 |
| dit_seedvr2_3b | SEEDVR2/seedvr2_3b_int8_convrot.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_seedvr2 | SEEDVR2/ema_vae_fp16.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_wan22_i2v_high | wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_wan22_i2v_low | wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors | video | 未声明 | 清单无 `license` 字段 |
| te_wan22_umt5_xxl | umt5_xxl_fp8_e4m3fn_scaled.safetensors | video | 未声明 | 清单无 `license` 字段 |
| vae_wan22 | wan_2.1_vae.safetensors | video | 未声明 | 清单无 `license` 字段 |
| dit_music3 | minimax_music3_dit_fp16.safetensors | audio | 未声明 | 清单无 `license` 字段 |
| te_music3 | minimax_music3_text_encoder_pruned_int8_convrot.safetensors | audio | 未声明 | 清单无 `license` 字段 |
| vae_music3 | minimax_music3_dav.safetensors | audio | 未声明 | 清单无 `license` 字段 |
| qwen3_14b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| qwen3_32b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| glm4_9b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| deepseek_r1_14b | —（无落点） | text | 未声明 | 清单行 `filename` 为空 |
| cosyvoice2 | —（无落点） | tts | 未声明 | 清单行 `filename` 为空 |
| wan26_i2v_14b_gguf | —（无落点） | video | 未声明 | 清单行 `filename` 为空 |

**为什么这一列空着**：这些权重来自多个上游（H3 主线各仓、Comfy-Org 权重镜像、fp8/GGUF 再分发仓、
SDXL 官方仓、SEEDVR2 视频修复（ByteDance SeedVR 系）），再分发条款各不相同，有「需随附许可原文」的、也有「仅限研究与个人使用」的 ⇒
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
| pypdfium2 | 5.15.0 | BSD-3-Clause, Apache-2.0, dependency licenses | `License`（上游把 PDFium 的 BSD-3 与自身 Apache-2.0 并列，照抄） |

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
3. **`reference/` 下的第三方源仓**（Comfy-Org 若干、diffusers、EveryDream2trainer、lora-pilot、
   LoRAMaster、musubi-tuner 等）是**阅读用的参考副本**、不进任何产物链路；它们的 LICENSE 随各自
   目录留在原地，本文件不重述。
   ⚠️ **2026-11-04 清理**：这些拷贝已从盘上删除（`.gitignore` 的学习用副本，共 46 个仓、8.5 GiB），
   并且**本仓不再依赖其中任何一个** ✓ —— `app/services/lora_train/paths.py` 的工具根只认
   `MUSUBI_TUNER_ROOT` / `SD_SCRIPTS_ROOT` 或 `configs/model-paths.json`，
   `tests/engine_quant_nvfp4_test.py` 的对拍真源同源取得（没配就 skip）。因此本文与各模块抬头里
   出现的 `reference/<仓>/…` 一律是**出处标注**（指上游项目同名路径），不再代表本仓有此副本。
   ⚠️ **例外只有三条**：`expert-stream` 与 `airllm` 各有一小段被移植进引擎
   （`app/services/engine/win_io.py` / `prefetch.py`），`Strata`＋`ds4` 只贡献**一个公式**
   （`gguf_dequant.py::_dequant_q5_k`）⇒ 已登记在下面第五节，
   许可原文随代码入仓 `docs/third-party/`；除这三条外，`reference/` 仍不进产物链路。

4. ~~**`ocr_index.py`（扫描件 OCR 线）按需 `import fitz`，即 PyMuPDF —— AGPL-3.0**~~
   ✅ **2026-11-07 已摘掉**：那条线现在与主链路共用 `pypdfium2`（BSD-3/Apache-2.0）——
   取文字层 `get_textpage().get_text_range()`、渲染 `page.render(scale=dpi/72, rev_byteorder=True)`、
   数页数 `len(doc)` 三件事 PDFium 都做得了，`page.get_pixmap()` 换成 `PdfPage.render()` 即可，
   没理由为省二十行代码背 AGPL。同批改动：`requirements-ocr.txt` 的 `pymupdf>=1.24.0` →
   `pypdfium2>=4.0`；`tests/ocr_index_test.py` 造 PDF 样本从 fitz 改成**手写最小 PDF**
   （生成器与被测实现同库，测的就是那个库自己）；`tests/ocr_engine_smoke.py` 改走
   `ocr_index._pdf_open/_pdf_page_image`，不再自带一份渲染。
   ⇒ **本仓产品代码与测试代码的 AGPL 使用已清零**。仅剩 `backend-py/_bench/` 下 **14 个**
   一次性体检脚本（`scan_all.py`／`scan_textlayer.py`／`scan_risk.py`／`probe_accuracy.py`
   及 `probe_moji*` 等）仍 `import fitz` —— 它们 `git ls-files` **零登记**、不随安装包分发、
   不被任何测试调用，成果是 `_bench/*.txt` 里那批语料质量数字（文档引用时可回溯），不构成分发面。
    ✅ **2026-11-07 同日也一并迁走**：新增 `_bench/pdfium_compat.py`（`pypdfium2` 之上的极薄
    同名适配层，只实现这 14 份真用到的 `open/page_count/doc[i]/get_text("text")/get_pixmap/
    is_encrypted`），脚本侧只把 `import fitz` 换成 `import pdfium_compat as fitz`，调用处一字未改
    ⇒ **连本地体检脚本也不再需要 PyMuPDF，全仓 `import fitz` 归零**。两处口径差写在适配层模块头：
    PDFium 没有「是否加密」旗标（要密码的文件打不开 ⇒ 能打开即 `False`）、`get_text_range()`
    不按块补换行（数字数／找乱码等价，别拿它跟库内 `chunks` 逐行 diff）。

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

**只有四条**（`reference/` 下其余仓一件代码都没进来，理由见 `docs/reference-audit.md`）：

| 上游仓 | 许可（原文入仓） | 被移植的提交 | 落在本仓 | 取了什么 | **没取什么** |
|---|---|---|---|---|---|
| `expert-stream`（Hemant） | MIT ⇒ `docs/third-party/expert-stream-LICENSE.txt` | `89275f1c50d2`（2026-08-16） | `backend-py/app/services/engine/win_io.py`（并由 `safetensors.py::TensorReader` 调用） | `engine/expert-stream.c` 的 Win32 三件：`CreateFileW` 的 `NO_BUFFERING\|OVERLAPPED` 标志、一次提交多笔 `ReadFile` ＋ 每槽手动复位事件、`VirtualAlloc` 槽区按扇区对齐并用 `slop` 吃掉非对齐偏移 | ggml 侧任何符号、n-gram/路由预取、`manifest` 专家表（那些绑 284B MoE ✓，与本仓 DiT 无关 ✗） |
| `airllm`（lyogavin） | Apache-2.0 ⇒ `docs/third-party/airllm-LICENSE.txt` | `75a6bb3c41c6`（2026-10-31） | `backend-py/app/services/engine/prefetch.py`（并由 `weights.py::load_module_weights(read_ahead=)` 调用） | `airllm_base.py` 的 `_prefetch_stream` / `_materialize` / `_evict` 三处调度语义：后台读下一件、取走即作废、取不到就同步读 | 逐层 `state_dict` 挂载、`ModelPersister`、bitsandbytes 压缩路径、按 token 逐层重读整模型的范式（那是 0.1–0.4 tok/s 的来源 ✗✗） |
| `Strata`（Niko1221）＋ `ds4`（ds4.c authors，含 ggml/DeepSeek 版权行） | MIT ⇒ `docs/third-party/Strata-LICENSE.txt`、`docs/third-party/ds4-LICENSE.txt` | Strata `c62d5e3a15b4`（2026-10-31）、ds4 `c97964b8aa77`（2026-10-30） | `backend-py/app/services/engine/gguf_dequant.py::_dequant_q5_k` | 只有**一个函数的公式**：`dequantize_row_q5_K`（llama.cpp 血统）。两份上游各是一份独立转录（Strata `include/strata/artifact/dequant.hpp:212-234` 的逐行转写、ds4 `ds4.c` 的标量版），本仓向量化版与这两份**逐位对拍**后才算数 ✓ | 两个仓的引擎主体：KV cache 分页/压缩、专家 popularity census、SSD 预算 planner、token 采样器、Metal/CUDA kernel、HTTP/MCP 外壳 —— 一条都没取（理由见 `docs/reference-audit.md`） |
| `ComfyUI-Local-Infinite-Canvas`（1314521gjy） | MIT ⇒ `docs/third-party/ComfyUI-Local-Infinite-Canvas-LICENSE.txt` | `0d7cfc102bae`（2026-11-01） | `backend-py/app/services/comfyui_catalog.py` / `comfyui_bindings.py` / `comfyui_slots.py` / `comfyui_match.py` / `comfyui_ui_workflow.py` / `comfyui_speaker_tags.py` / `comfyui_lora_chain.py` / `comfyui_progress.py` / `media_size.py` / `video_interpolate.py`（＋ 工作流与资源资产 `configs/comfyui/workflows/**`、`configs/comfyui/resources/index.json`、提示词模板 `configs/comfyui/prompts/**`、提示词库 `app/skills/h3-canvas/`、前端画布 `frontend/app/{composables/useCanvas.ts,composables/useComfyRun.ts,utils/canvasGeometry.ts,utils/canvasNodeSize.ts,utils/canvasImageOps.ts,utils/canvasReferences.ts,utils/videoFrame.ts,utils/mediaFile.ts,utils/nodeSettingsMemory.ts,utils/keyboardEvent.ts,utils/canvasMask.ts,utils/canvasOutpaint.ts,components/canvas/**,pages/canvas.vue}`） | 十一块**语义**：① 工作流目录（API/UI 格式判定、变体展开、多阶段步数估算）；② `bindings` 参数注入（9 种 transform、首尾帧 / 多参考 / 连线三种挂法、Qwen 2.1 的 sigma 插值）；③ 模型槽位表（18 条 `(class_type, input)` ＋ 从 `/object_info` 动态发现）与可用性判定；④ 模型名相似匹配（七条红线 + Dice·Jaccard + 歧义不自动选）；⑤ UI 格式 → API 格式转换（Reroute/Primitive 穿透、旁路直连、widget 顺序表、游离节点剪枝）；⑥ H3 说话人标签与画外音标注。**代码是自研重写** ✓，没有拷源文件 | 网关主体（`server.js` 的任务队列 / 上传 / 更新与补丁校验）、17 个随包 ComfyUI 自定义节点、`prompt-optimizer` / `resources` 下载器、React 画布的大型节点面板（`canvas-node.tsx` 等，未接生成链路前搬过来是死代码）、Windows launcher 启动器 |

**移植时必须一起改掉的三处上游习惯**（否则进不了本仓）：

1. **错误不许静默**：上游「读失败 ⇒ 标 -1 让上层回落 mmap」在本仓不可接受（会装出半件模型还装得上）
   ⇒ 移植件一律抛 `OSError` / `PrefetchError`；
2. **状态必须可查**：上游靠 `print` ⇒ 移植件给 `describe()` / `snapshot()`，直接喂 `GET /engine/runtime`；
3. **不许新增第二真源**：`win_io.py` 只读字节（偏移由 `safetensors.py` 算）、`prefetch.py` 只管调度
   （装什么由装配线给）⇒ 两件都不许自己判断「该装哪件权重」。

**怎么核**：`tests/engine_win_io_test.py`（10 组，含「三笔同时在飞 ⇒ 无事件就返回 0 字节」这条真事故回归
与 ⑩「4 线程 × 24 段并发读必须与串行结果逐字节相同」）与 `tests/engine_prefetch_test.py`（9 组，含「同一件
不许取两次」）、`tests/engine_readahead_test.py`（10 组，含「预取路径与同步路径权重逐张量相同」）。
三份都是纯逻辑、不碰权重、不 import torch。
Q5_K 那条另有判据：`tests/engine_gguf_dequant_test.py` 的 ⑤ 系列（手造块精确值 ＋ 与标量参考版
4097 块随机字节**逐位对拍** ＋ 盘上 `krea2_turbo-Q4_K_M.gguf` 的 63 张真 Q5_K 张量）。

### 五·补：词表资产（`models/tokenizers/`，**本地素材、不入库**）

`models/tokenizers/{qwen3_4b,qwen3vl_4b,qwen35_9b}/`（2026-11-02 落）＝ Qwen 官方 HF 仓库的
分词器三件（`tokenizer_config.json`＋`vocab.json`＋`merges.txt`；**Apache-2.0**；经 hf-mirror.com
的 `resolve/main` 直链下载 —— 新版 `huggingface_hub` 的 CAS 通道对镜像 401 ✗，curl 直链可以 ✓）。
第四档（`qwen_image`，Qwen2.5-VL-7B）用 HF 缓存里 `Qwen/Qwen2.5-7B-Instruct` 的**同一份词表**
（Qwen2.5 家族统一 BBPE ✓），不重复落仓。
引擎按档解析的表与双判据在 `tokenizer_hub.STAGE_TOKENIZERS` / `find_tokenizer_for_stage`；
这些是**几 MB 的文本表、不是权重** ✗ ⇒ 不进 `configs/models.json` 清单 ✗、不占 `readiness` 的 GiB 账 ✓。

## 六、本地参考资料（`剧本知识库/` 不入仓；方法语料 2026-11-04 起入仓）

| 目录 | 是什么 | 来源 / 许可 | 本仓怎么用 |
|---|---|---|---|
| `剧本知识库/` | 供应商预建的影视剧本资料库：约 29 GB 原文（PDF/剧本/写作书）＋一份 SQLite `fts5 trigram` 索引（5403 条文档 / 128166 片段） | **来源与许可均不明**（随盘到货，无 LICENSE、无 README 出处） | **只读**：不起副本、不重建索引、不改文件。检索层 `app/services/script_library.py` 以 `mode=ro` 打开那份索引；Agent 工具 `search_script_library` 与 CLI `app/scripts/corpus/library.py` 共用它。仅作**写法与结构**参考，输出不许搬运原文句子 |
| `corpus/playbook/` | 137 份方法类 Markdown（招式库、造型库、特效库、分镜规范、技法资料库等**原料**），无自带索引、无代码；`film-expert-bridge/` 下另有 combat 77 ＋ ultimate 11 份专项素材（2026-11-04 扁平化：原 `packs/film-expert-bridge/references/library/`，`packs`／`references`／`library` 三层壳已删） | **来源与许可均不明**（课程包随盘到货，无 LICENSE） | **2026-11-04 起入仓分发**（1.48 MiB，创作者裁定「有用的资料必须进系统（前后端 / data / 知识库任一）」）：`search_playbook()` 在进程内按小节切块打分（2-gram 加权），不落中间产物；Agent 走 `search_script_library(source="playbook")`，知识库页面走 `/api/v1/knowledge/*`（同源）。skill 侧另装**归纳出的判据**八份：`video-prompt-library/references/` 下 `shot-directing.md`／`performance-directing.md`／`review-and-repair.md`／`human-texture.md`、`audio-prompt-library/references/mix-layers.md`、`wuxia-action/combat-choreography/references/` 下 `hit-reaction-and-moves.md`／`shot-spec-and-break.md`／`action-vocabulary.md`，不依赖本目录也能出稿。同日清理：先删空壳模板／安装脚本／宣传图／导出 docx（4.6 → 1.9 MiB），再清 94 份课程机制件（交互菜单／专家路由与项目模板／飞书下载入口／他人机器路径／已抄完的源文件／课程文档），267 → 137 文件 |

只有 `剧本知识库/` 还在 `.gitignore` 里（29 GB 二进制、版权不明 ⇒ 只读参考，绝不入仓）。
方法语料 137 份 md 已入库：它只有 1.48 MiB，而入库前的实际后果是**新克隆的仓库里知识库页面与
`search_script_library(source="playbook")` 一起是空的**（`available=False`）—— 资料只存在于作者机器上，
「系统里没有」就等于没有。许可风险照旧记在本节：**只作写法与结构参考，输出不许搬运原文句子**。
目录不在盘上时检索层如实回 `available=False` + `corpus=None`（**不编 0**），Agent 与 CLI 照常可用。
`scripts/` 里没有任何代码读 `剧本知识库/` —— 删掉它只影响「这一项参考能力」，不影响运行与回归。
