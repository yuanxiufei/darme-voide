# H3 ComfyUI 工作流模板（本目录）

本目录的 JSON 是 **UI 格式**（ComfyUI 界面可直接打开、可看连线 ✓），**不能**直接 `POST /prompt` ✗ ——
提交前必须先经 `../workflow.py::ui_to_api()` 转成 **API 格式** ✓（这是 ComfyUI 的规矩，不是我们的选择 ✗）。

## 来源与署名

| 文件 | 来源 | 说明 |
|---|---|---|
| `MiniMax_H3_Fast_T2V.json` | `reference/minimax-h3-comfyui/workflows/`（上游仓库 *MiniMax H3 Fast ComfyUI Workflows*，随 `reference/` 一并收录） | 文生视频（T2VA）：FL2VA INT8 + Qwen3-VL 32B NVFP4 + 10 步 + 内存友好 SageAttention ✓ |
| `MiniMax_H3_Fast_I2V.json` | 同上 | 图生视频（I2VA）：默认只用首帧；**可选尾帧通道默认关闭** ✓（要用需显式把那段节点的 `mode` 改回 0 并接上连线 ✓） |

⚠️ 上游未在仓库内声明许可证（`reference/minimax-h3-comfyui/` 只有 README + workflows + .gitignore ✗）。
这里**原样复制**仅为让本机服务自包含（`reference/` 是开发期资料、不进镜像 ✗）。
若将来上游补许可或要求移除，请以那边为准 ✓。

上游 README 的实测耗时（用于设置超时）：冷启动 T2V 3 秒片 **53s**、热启 **23s**、I2V 热启 **27s**、
标准 5 秒片 **85s** ✓ ⇒ `comfyui_client.DEFAULT_TIMEOUT_SECONDS` 默认 1800s 是**故意给足**的 ✓。

## 这份工作流用到什么（2026-09-16 在本版本 ComfyUI 源码里逐个核对过 ✓）

`reference/ComfyUI` 这一版里：**16 个节点是核心自带** ✓（含 `MiniMaxH3ImageToVideo` —— 它在
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
