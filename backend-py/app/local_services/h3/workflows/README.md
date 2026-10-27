# H3 ComfyUI 工作流模板（本目录）

本目录的 JSON 是 **UI 格式**（ComfyUI 界面可直接打开、可看连线 ✓），**不能**直接 `POST /prompt` ✗ ——
提交前必须先经 `../workflow.py::ui_to_api()` 转成 **API 格式** ✓（这是 ComfyUI 的规矩，不是我们的选择 ✗）。

## 来源与署名

两批来源，**别混着看** ✗：

### A 批：自有/第三方实测模板（`minimax-h3-comfyui`）

| 文件 | 来源 | 说明 |
|---|---|---|
| `MiniMax_H3_Fast_T2V.json` | `minimax-h3-comfyui/workflows/`（上游仓库 *MiniMax H3 Fast ComfyUI Workflows*） | 文生视频（T2VA）：FL2VA INT8 + Qwen3-VL 32B NVFP4 + 10 步 + 内存友好 SageAttention ✓ |
| `MiniMax_H3_Fast_I2V.json` | 同上 | 图生视频（I2VA）：默认只用首帧；**可选尾帧通道默认关闭** ✓（要用需显式把那段节点的 `mode` 改回 0 并接上连线 ✓） |

⚠️ 这批上游**未在仓库内声明许可证**（`minimax-h3-comfyui/` 只有 README + workflows + .gitignore ✗）。

### B 批：Comfy 官方模板（`comfy-org/workflow_templates`，**MIT** ✓）

来源：`comfy-org/workflow_templates` 的 `templates/` 目录（该仓库 `LICENSE` = MIT，
*Copyright (c) 2023-present Comfy Org* ✓）；这里**原样复制、未改动内容** ✓。

| 文件 | 上游路径 | 说明 |
|---|---|---|
| `video_minimax_h3_t2v.json` | `templates/video_minimax_h3_t2v.json` | 本地跑（T2VA）：`MiniMaxH3ImageToVideo` 核心节点链，**主节点包在子图里** ✓ |
| `video_minimax_h3_i2v.json` | `templates/video_minimax_h3_i2v.json` | 本地跑（I2VA）：首帧 + `ImageScaleToTotalPixels` / `GetImageSize` |
| `video_minimax_h3_i2v_continuation.json` | `templates/video_minimax_h3_i2v_continuation.json` | 续写：拿上一段末帧当首帧 |
| `video_minimax_h3_r2v.json` | `templates/video_minimax_h3_r2v.json` | 参考素材（`MiniMaxH3ReferenceToVideo`） |
| `api_minimax_h3_t2v.json` | `templates/api_minimax_h3_t2v.json` | 走 **API 节点**（云端 `MinimaxHailuo03*` ✓，**不是**本地权重 ✗） |
| `api_minimax_h3_flf2v.json` | `templates/api_minimax_h3_flf2v.json` | 同上，首尾帧 |
| `api_minimax_h3_r2v.json` | `templates/api_minimax_h3_r2v.json` | 同上，参考素材 |

⚠️ **`api_*` 那三个不是本地推理路径** ✗：它们调的是 Comfy 的**云端 API 节点** ⇒ 与 `app/services/engine`
的「本地跑」口径**不是一回事** ✓✗（放这里只为对齐官方模板、便于比对参数与选项表 ✓）。

⚠️ 两批的**许可证不同** ✓✗：A 批无声明、B 批 MIT ⇒ 落盘时**分开记来源**（上表）✓，别把 MIT 的署名罩到 A 批上 ✗。

⚠️ 开发期的上游副本放在脚手架目录 ``reference/`` 下，而**项目完善后整目录删除** ✗（用户 2026-09-26 口径 ✓）⇒
① **本目录里的这份副本才是长期事实** ✓、② 别处（含注释/文档）**不许**再把 ``reference/…`` 当落点 ✗、
③ **删掉 ``reference/`` 之后本文件读起来必须仍然是对的** ✓（因此正文不写那边的具体路径 ✗）。
若将来上游补许可或要求移除，请以那边为准 ✓。

上游 README 的实测耗时（用于设置超时）：冷启动 T2V 3 秒片 **53s**、热启 **23s**、I2V 热启 **27s**、
标准 5 秒片 **85s** ✓ ⇒ `comfyui_client.DEFAULT_TIMEOUT_SECONDS` 默认 1800s 是**故意给足**的 ✓。

## 这份工作流用到什么（2026-09-16 在本版本 ComfyUI 源码里逐个核对过 ✓）

`ComfyUI` 这一版里：**16 个节点是核心自带** ✓（含 `MiniMaxH3ImageToVideo` —— 它在
`comfy_extras/nodes_minimax_h3.py` ✓，**不是**第三方节点 ✓）；另有 3 个需要第三方节点包、
1 个是界面便签（**不执行** ✗）：

| 节点类 | 归属 | 对应清单条目（`configs/models.json > nodes[]`） |
|---|---|---|
`MiniMaxH3ImageToVideo` / `MiniMaxH3*` | 核心 ✓（本快照） | — |
`MiniMaxH3MemoryEfficientSageAttentionPatch` | 第三方 | `ComfyUI-MiniMaxH3-Easy`（或 H3 节点包） |
`ModelPreviewOverrideKJ` | 第三方 | `ComfyUI-KJNodes` |
`Power Lora Loader (rgthree)` | 第三方 | 未列入清单 ⚠️（工作流里 LoRA 槽全关 ⇒ 可实测确认是否必需） |
`MarkdownNote` | **界面便签、不执行** | 转换时按「无输出被消费」自动丢弃 ✓（见 `workflow.py`） |

⇒ 这三条第三方依赖由 `app/scripts/model_manager.py install-nodes` 负责安装 ✓；缺哪个，
`comfyui_client.missing_nodes()` 会在提交**之前**直接点名 ✓（而不是等 ComfyUI 抛 400 ✗）。

### B 批模板用到什么（**展平后**数出来的 ✓）

B 批是**新版官方模板** ✓：主节点链被包进 **子图** ✓（顶层 `definitions.subgraphs[...]`，
顶层那个节点的 `type` 其实是**子图 uuid** ✗ 不是节点类 ✗）⇒ 必须先
`workflow.py::flatten_subgraphs()` **摊平** ✓ 才谈得上转 API 格式 ✗（`ui_to_api` 已自动摊 ✓）。

实测（本目录文件、展平前后顶层节点数 ✓）：

| 文件 | 顶层 → 展平 | 展平后节点类（去掉便签 `MarkdownNote` 前的全量 ✓） |
|---|---|---|
| `video_minimax_h3_t2v.json` | 6 → 26 | `MiniMaxH3ImageToVideo` + `CLIPLoader` `UNETLoader` `VAELoader` `ResolutionSelector` `PrimitiveBoolean/Float/Int` `ComfySwitchNode` `ComfyMathExpression` `LoraLoaderModelOnly` `SamplerCustomAdvanced` `BasicGuider` `BasicScheduler` `KSamplerSelect` `RandomNoise` `VAEDecode*` `CreateVideo` `SaveVideo` |
| `video_minimax_h3_i2v.json` | 9 → 29 | 同上，另加 `LoadImage` `ImageScaleToTotalPixels` `GetImageSize`（首帧先对齐像素预算 ✓） |
| `video_minimax_h3_i2v_continuation.json` | 6 → 26 | 同上（I2V 链去掉图像对齐两步） |
| `video_minimax_h3_r2v.json` | 29 → 29（**没有子图** ✓） | `MiniMaxH3ReferenceToVideo` + `PrimitiveStringMultiline`（参考素材描述 ✓） |
| `api_minimax_h3_*.json` | 2–4 → 2–4 | 云端 API 节点 `MinimaxHailuo03{TextToVideo,FirstLastFrame,Reference}Node` + `SaveVideo`（+ `LoadImage`） |

⚠️ B 批**一个第三方节点包都不需要** ✓（全在核心 ✓）：`LoraLoaderModelOnly` 在 `nodes.py` ✓、
`ResolutionSelector` 在 `comfy_extras/nodes_resolution.py` ✓、`ComfySwitchNode` 在
`comfy_extras/nodes_logic.py` ✓、`Primitive*` 在 `comfy_extras/nodes_primitive.py` ✓ ——
与 A 批要装三个包 ✓✗ 正相反 ✓。

### B 批暴露的三个「抄错就静默错」的点 ✓✗

1. **画幅尺寸要按 `ResolutionSelector` 的式子算** ✗：模板里 `MiniMaxH3ImageToVideo` 的
   `width`/`height` 是**连着 `ResolutionSelector` 的** ✗ ⇒ 节点自带的 `widgets_values`
   （旧值 `1344×768` ✗）**不是**真值 ✓，真值由节点现场算（0.65MP @16:9 @`multiple=32` ⇒ **1088×608** ✓）。
   权威公式 = ``comfy_extras/nodes_resolution.py`` 的 `ResolutionSelector.execute` ✓
   （`megapixels × 1024²` + `round` 到 `multiple` ✓），本仓照抄在
   `workflow.py::resolution_selector_size` ✓，对账入口 `workflow.py::expected_resolution` ✓。
   ⚠️ **别**和 `app/services/engine/geometry.py::size_for_megapixels`（本仓早先按实测反推 ✓）混用 ✗ ——
   两者只在两个实测锚点上重合（`0.65MP ⇒ 1088×608`、`0.98MP ⇒ 1344×768` ✓），一般情形**会发散** ✓✗。
2. **COMBO 是白名单** ✗：官方站点侧配置（`site/src/lib/demos/mmh3/config.ts`）把画幅写成
   `"21:9 (Cinemascope)"` ✗ 而节点真枚举是 `"21:9 (Ultrawide)"` ✓
   （见 `workflow.py::ASPECT_RATIO_OPTIONS` ✓）⇒ 表外的串提交必被 400 拒 ✓。
   `workflow.py::check_combo_choices()` 会在提交前把所有「值不在选项表里」的输入点名 ✓（**只报不改** ✗）。

### 第 3 个点：`video_minimax_h3_r2v.json` 的**参考素材**（2026-09-27 逐条核过 ✓）

主节点 `MiniMaxH3ReferenceToVideo`（节点 **136**，本文件是**平铺**的、没有子图 ✓）的真 schema：
`prompt` / `width` / `height` / `length` / `ref_image_size`（**第 5 个 widget** ✓，
`io.Combo.Input("ref_image_size", options=["match", "max"], default="match")` ✓）——
⚠️ 少声明一个 widget ⇒ `workflow.ui_to_api` 会判「多余的非 dict 值 = **字段错位**」**当场拒转** ✗✗
（本仓自检实测踩过 ✓：`values=5 但 object_info 只声明 4 个` ✓）。

⚠️ 它**没有** `first_frame` / `last_frame` 端口 ✗（那两个只在 `MiniMaxH3ImageToVideo` 上 ✓）
⇒ 「既要参考图又要首尾帧」这种请求，换到本模板就会**丢首尾帧** ✗（服务端只按图上有的端口注入 ✓✗）。

参考素材的**四类**端口（节点 schema 的能力 ✓，出处同上的 `define_schema` ✓）：
`ref_images`（`ref_image_0..8`，max=9 ✓）/ `ref_videos`（≤3 ✓）/ `ref_video_audios`（≤3 ✓）/
`ref_audios`（≤3 ✓）—— ⚠️ 但**本模板只接了图** ✗（音频/视频一格都没接 ✓）
⇒ 「几路能落地」只能**看模板**、不能看节点类 ✓✗（`workflow.py::reference_slot_names` 数的是图上真有的端口 ✓）。

三处「不图上读就会静默错」的地方 ✓✗：

| 事实（从图上量出来 ✓） | 抄错的后果 ✓✗ |
|---|---|
| 主节点的 `prompt` 是**被拉着线的端口**（`inputs[prompt].link = 279` ✓，上游是 `PrimitiveStringMultiline` ✓） | 往主节点 `prompt` 塞字面量会把那根线**覆盖掉** ✗ ⇒ 提示词落空的写法却看起来「设置了 ✓」 |
| 参考槽位名是 `ref_images.ref_image_0` / `_1` / `_2` ✓（`ref_images` 是**结构**，不是节点 ✓） | 按「节点标题」找槽 ⇒ 一个都找不到 ✗（主节点上根本没有 `LoadImage` ✓）；按「节点 schema 的 max=9」算槽 ⇒ 接不到的空槽也去接 ✗ |
| 模板里**只有** `_0` / `_1` 接着 `LoadImage`（各 1 个 ✓），`_2` 是**空插座** ✓ | 空插座**别去接** ✓；反过来，给**少于**槽位数的参考图时，多余槽位**连同它上游的 `LoadImage` 一起摘掉** ✗ —— 那几个 `LoadImage` 指着模板自带的示例图 ✓，在**我们的** `input` 目录里不存在 ✓，而 `image` 是 **COMBO** ✓ ⇒ 留着它 ComfyUI 的提交前校验就 400 ✓✗（哪怕那节点不参与出图 ✓） |

⇒ 这三条都是**图判据**的范围 ✓（`workflow.py::video_plan` 一次算清 ✓），**不写死节点 id、不认标题** ✗✗。

## 参数注入点（`MiniMaxH3ImageToVideo` 的真实 schema ✓）

```python
# comfy_extras/nodes_minimax_h3.py :: MiniMaxH3ImageToVideo.define_schema()
io.Clip.Input("clip")        # 连线
io.Vae.Input("vae")          # 连线
io.String.Input("prompt")    # ← 提示词
io.Int.Input("width", default=1344, step=32)
io.Int.Input("height", default=768, step=32)
io.Int.Input("length", default=124, step=17)   # 24fps 的帧数；124 ≈ 5s，模型训练区间 ≈ 124–362
io.Image.Input("first_frame", optional=True)   # ← 首帧（FL2VA 的几何锚点）
io.Image.Input("last_frame", optional=True)    # ← 尾帧（按比例 cover-crop）
```

⇒ 时长要走 **`length`（帧数）**，不是秒 ✗：`length ≈ round(seconds × 24)`（并按 17k+5 网格向上取 ✓）。
参考工作流里那个 `ComfyMathExpression` + `PrimitiveFloat «DURATION (seconds)»` 组合干的就是这件事 ✓
（换算公式：`max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17` 一类 ✓）。
