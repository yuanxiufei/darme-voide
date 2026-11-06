# 源项目迁移审计（`ComfyUI-Local-Infinite-Canvas` → 本仓）

> 审计对象：`D:\code\voide\anli\ComfyUI-Local-Infinite-Canvas`（MIT）
> 落点：`D:\code\voide\darme-voide`
> 口径：**只搬能独立成事的语义**，代码一律自研重写（未拷源文件）；许可登记见 `THIRD_PARTY_NOTICES.md` §五。

## 一、结论摘要

| 分类 | 数量 | 含义 |
|---|---|---|
| **保留（已搬）** | 12 块 | 原样语义搬过来，落点已接线或有守卫测试 |
| **适配改造** | 18 块 | 结构/栈不同，改造后才成立（下面逐条写"改了什么、为什么不能照搬"；2026-11-03 追加蒙版重绘 / 扩图两块，见 §4.5） |
| **舍弃** | 其余 | 分三类：结构层已有替代 / 与本仓领域无关 / 未接生成链路前是死代码 |

## 二、保留（已搬，逐项落点）

| 源 | 本仓落点 | 搬的是什么 |
|---|---|---|
| `gateway/prompts/*.json`（4 份） | `configs/comfyui/prompts/` | 提示词模板**数据**（原文保留） |
| `src/bindings.js` | `app/services/comfyui_bindings.py` | 参数注入：9 种 transform、首尾帧/多参考/连线三种挂法、Qwen 2.1 sigma 插值 |
| `src/step-plan.js` | `app/services/comfyui_catalog.estimate_step_plan` | 多阶段步数估算（进度条归一） |
| `src/workflow-availability.js` + `model-loaders.js` | `app/services/comfyui_slots.py` | 18 条 `(class_type, input)` 规格 + `dirs` + 动态发现；**逐条比对过，源表已全覆盖，无缺项** |
| `src/model-match.js` | `app/services/comfyui_match.py` | 七条红线 + Dice·Jaccard + 歧义不自动选 |
| `src/ui-workflow.js` | `app/services/comfyui_ui_workflow.py` | UI→API：Reroute/Primitive 穿透、旁路直连、widget 顺序表、游离节点剪枝 |
| `src/speaker-tags.js` | `app/services/comfyui_speaker_tags.py` | H3 `(S1)` 说话人 id、`off-screen` 与"只加一次" |
| `src/lora-chain.js` | `app/services/comfyui_lora_chain.py` | `loras` → `LoraLoaderModelOnly` 串联；已接进 `comfyui._build_upstream_body` |
| `web/src/lib/media-size.ts` | `app/services/media_size.py` | 档位×比例→像素表、比例/档位反推、视频「短边=分辨率」+ 偶数化；已接进 `bindings.build_context` |
| `web/src/lib/canvas/canvas-node-geometry.ts` | `frontend/app/utils/canvasGeometry.ts` | 分组收纳（上留白 52、按中心点判归属、整体平移）、连线三不许、聚焦视口 |
| `web/src/lib/canvas/{canvas-node-size,canvas-node-factory}.ts` + `constant/canvas.ts` | `frontend/app/utils/canvasNodeSize.ts` | `NODE_DEFAULT_SIZE`（340×240 / 420×236 / 760×480）、`fitNodeSize`（≤640 只缩不放）、`nodeSizeFromRatio`（0.25–4 之外回默认） |
| `web/src/lib/keyboard-event.ts` | `frontend/app/utils/keyboardEvent.ts` | 输入法守卫（`isComposing` / `keyCode 229`）；8 个回车提交点已接 |

## 三、适配改造（改了什么、为什么不能照搬）

| 源 | 本仓 | 改造点 |
|---|---|---|
| `web/src/lib/canvas/canvas-import.ts`（ZIP + IndexedDB blob） | `useCanvas.normalizeDoc` | **只取"先整份校验、再写"的口径**。本仓画布存 URL 不存 blob、不引 jszip ⇒ 不搬归档格式；保留两种策略：导入严格（整份拒、现画布不动）、本地读档宽松（丢坏留好） |
| `canvas-export.ts`（项目包版本） | `CANVAS_DOC_VERSION` + `exportDoc` | 只搬"导出带版本号"这一条；不做多项目打包 |
| `node-settings-memory.ts`（24 个 metadata 字段） | `utils/nodeSettingsMemory.ts` | 源记的是生成参数（`model/size/seed/fps…`），本仓节点**没有 metadata** ⇒ 照搬是一张永远空的表（死代码）。改成记**尺寸**（所有类型）+ **配置卡正文**（仅 `config`）；文本卡不记（会覆盖用户刚写的稿子） |
| `canvas-node-factory.ts` 的 `metadata` 构造 | `utils/canvasNodeSize.ts` + `useCanvas.addNode` | 只搬尺寸与"落点居中"；`buildImageGenerationMetadata` 依赖生成链路，**未接前搬过来是死代码**，不做 |
| `workflow-visibility.js`（工作流隐藏开关） | `comfyui_catalog.entries(include_hidden=False)` | 源的"隐藏"是**用户偏好存储**；本仓的 `hidden` 是**注册表数据**（旧入口不该出现在下拉里），语义更窄但已够用 ⇒ 不新增偏好存储 |
| `canvas-image-data.ts`（裁剪/切分/旋转/放大） | `utils/canvasImageOps.ts` | 输出从 **dataURL 改 Blob**（本仓画布存 URL：dataURL 进 localStorage 两三张就爆 ✗，且刷新后才报错）；`loadImage` 补 `onerror`（源只 resolve ⇒ 坏图**永久挂起** ✗）；PNG >12MB 回退 JPEG；源的伪 3D 摆角要 4 个叙事参数且本仓无面板宿主 ⇒ **不搬**，改为正交 90° 旋转 |
| `canvas-video-frame.ts`（抽帧） | `utils/videoFrame.ts` | 加**超时**（源无超时，跨域/编码不支持会一直等 ⇒ "点了没反应"✗）；保留"取完清 `src` + `load()` 释放解码器"✓ |
| `canvas-resource-references.ts`（节点引用装配） | `utils/canvasReferences.ts` | 只保留**纯数据层**（连线 + 同组 + 分组展开 + 分类独立编号）；源里与 React 面板/生成链路耦合的部分不搬 |
| `video-interpolation.js`（RIFE 2×/3×） | `services/video_interpolate.py` + `routers/video_ops.py` | 实现从 **ComfyUI RIFE 工作流换成 ffmpeg `minterpolate`**（不再要求用户装节点 + 放 `_internal/video-rife.json`）；保真"倍率只接 2×/3×""输出帧率跟随源"；**新增**超时 kill 进程与尺寸偶数化 |
| `preview-hub.js`（ComfyUI WS 预览 + `progress_state`） | `services/comfyui_progress.py` + `routers/comfyui_events.py` | 拆成三块搬：① **加权进度**（按节点 `value/max` 求和，不是按节点平均）② **收尾三态**（done/error/cancelled）+ 预览只留最新帧 + 10 分钟 TTL ③ **可选 WS**（`websockets` 用 `find_spec` 探测，缺失则只推轮询状态并如实 `progressAvailable=false`）；总线**按 runId** 另建，不动 `sse_hub` 的 drama 频道 |
| `image-utils.ts`（`formatBytes` / `readImageMeta` / `dataUrlToFile`） | `utils/mediaFile.ts`（`formatBytes`） | 只搬 `formatBytes`，**精度规则刻意不同**：源「≥10 就取整」会把 12.5MB 显示成 13MB（丢精度），沿用本仓「≥100 取整」；`readImageMeta` / `dataUrlToFile` 本仓无宿主（画布节点存 URL 不存 dataURL）⇒ 不搬。顺带把 `models.vue` 的局部 `formatSize` 收敛到同一处 |
| `image-thumbnail.ts`（渲染缩略图 / 按缩放选源） | `utils/mediaFile.ts`（`compressImageForUpload`） | 源的用途是画布渲染降采样，本仓 `fitNodeSize` 已把显示尺寸压到 ≤640 ⇒ 不需要缩略图；**改造成"超 20MB 上传前压缩"**（后端上限照抄原 TS 不许改，不压＝用户那张 4K 图永远传不上来 ✗），输出 **WebP 保透明**、压缩失败原样返回让后端如实报错 |

## 四、舍弃（三类理由）

### 4.1 结构层已有替代（Node 网关运行时 → 本仓 FastAPI + DB）

`server.js` / `tasks.js` / `runner.js` / `comfy-client.js` / `sessions.js` / `uploads.js` / `config.js` /
`directories.js` / `folder-picker.js` / `model-paths.js` / `model-events.js` / `base-models.js` /
`main-models.js` / `admin.html` / `setup.js` / `update.js` / `assistant.js` / `util.js` / `mock/**` /
`scripts/**` / `workflow-baseline.json`

理由：这些是"给 ComfyUI 套一层本地网关"的**运行时设施**（队列、上传、安装更新、目录选择、后台管理页）。
本仓已有对应物且更贴领域：`comfyui_runs` 表 + 状态机 + 崩溃续询、`gpu_manager` 租约、产物下载进自有数据根、
`local_model_scan`/`model_folders`/`model_fingerprint` 的盘上扫描与指纹、`sse_hub` 的 SSE 通道。
搬过来只会出现两套队列/两套目录扫描（**冗余重复**，正是本次要避免的）。

### 4.2 与本仓领域无关

| 源 | 理由 |
|---|---|
| `resources.js`（135 KB 资源下载器） | 本仓以 `configs/comfyui/resources/index.json` + `resource_library` 管理资产，不做外网下载器 |
| `prompt-optimizer.js`（32.7 KB） | 本仓 `llm.py` / `prompt_utils.py` / `prompt_polish.py` 已有同类能力（且走自有 provider 体系） |
| `system-stats.js` | 本仓 `gpu_manager` / `device_policy.probe()` / `runtime_doctor` 已覆盖 |
| `skin-unlocks.js` / `progress-ring-skins.ts` / `canvas-theme.ts` / `app-theme.ts` / `use-canvas-theme.ts` | 皮肤/解锁体系，本仓无对应宿主与运营玩法 |
| `audio-generation.ts` | 本仓音频走 `tts_generation` / `voice_clone`，生成范式不同 |
| `layout-prompt.js`（MOSS 区域标注 → 版式提示词） | 绑定 `MOSS` 专用布局协议（`object/character/text` 区域），本仓分镜提示词是另一套结构 |
| `free-workflows.js`（免绑定工作流 store + `applyFreeInputs`） | `applyFreeInputs` 与 `comfyui_bindings` 同类（已有）；store 与 `comfyui_catalog` 同类（已有）⇒ 双份 |
| `qwen-edit.js` / `canvas-generation-helpers.ts`（13 KB） / `canvas-node-prompt-panel*` 等 React 面板 | React 面板部分依赖其组件树（本仓 Vue3 重写）；`qwen-edit.js` 的"目标图轮到最后 + 提示词编号重排 + 剥离标签前言"**已于 2026-11-03 复审并否决**（前言在本仓不存在；轮换方向与盘上节点相反）⇒ 见 §4.8 |
| `infinite-canvas` 其余 `lib/*`（analytics / local-session / localforage / release / zip / app-theme） | 浏览器端会话/存储/发版设施，本仓用 localStorage + 自有 API 层 |

### 4.3 无（本轮清空）

原先挂在这里的五项 —— `preview-hub` / `canvas-image-data` / `canvas-video-frame` /
`canvas-resource-references` / `video-interpolation` —— **已全部搬完**，改造点见第三节 ✓。

`preview-hub` 尤其值得记一笔：它原本被判「必须等生成链路」，
实际拆开看**只有 WS 订阅那半依赖外部条件**，加权进度与收尾三态是纯计算 ⇒
本轮按「纯函数 + 按 runId 总线 + 可选 WS 降级」拆开搬入，
进度通道先于生成链路可用（轮询状态即数据源 ✓），WS 预览作为加分项 ✓。

### 4.4 全量清点（2026-11-02）：搬完 / 不搬，各有理由

源项目比本仓大得多（gateway 约 1.5 万行 + 前端数万行）。这一轮把它的**每一块**都过了一遍，
判定标准只有两条：**本仓有没有真实缺口**、**搬进来有没有调用方**（没有调用方的能力不算能力 ✗）。

**本轮新增搬入（四块，都已接到真实路径）**

| 源 | 落点 | 为什么它算缺口 |
|---|---|---|
| `workflow-availability.js` + 上游 `/v1/workflows/validate` | `comfyui.validate_workflow_body` + `POST /comfyui/validate` | 原先后端**从不预检**：用户点生成要等队列跑完，才在 ComfyUI 日志里看到 `Prompt outputs failed validation`，缺哪个节点包全靠猜 ✗ |
| 无（源里是 `canvas-image-generation`） | `comfyui.build_registry_graph` + `build_submit_body` | `POST /runs` 只收**整张图**或**盘上路径** ⇒ 前端要跑「image-krea2-turbo」得自己下载 JSON、自己找 `#5.text`、自己吸尺寸 = 让前端复刻一遍 bindings ✗ |
| `canvas-resource-references.ts` 的**消费面** | `collectReferenceInputs` + `comfyui.stage_reference_uploads` | 画布连线素材原来只用来显示摘要，**从没进过生成** ⇒ angle / four-view / upscale / ref2va 这些吃参考图的工作流等于跑不了 ✗ |
| `canvas-node-crop-dialog.tsx` | `CanvasCropDialog.vue` | 只有「一键居中裁方形」，裁不出「左下角那个人」✗ |

**明确不搬（附理由，不是遗漏）**

| 源 | 理由 |
|---|---|
| `launcher/`、`update.js`、`setup.js`、`patches/`、`tools/` | 安装/更新/分发层。本仓是开发仓，模型与引擎自己管（`engine/` 自研），搬来没有宿主 |
| `assistant/`（`server.mjs` / `repair-tools.mjs`） | 另起一个常驻助手进程做诊断修复；本仓已有 `runtime_doctor` / `doctor` 覆盖同一职责 |
| `resources.js`（2692 行） | 一键装模型资源。本仓 `local_models` + `resource_library` 已覆盖下载/校验，且它强绑自身 `resources/index.json` 的落点与 ComfyUI 目录布局 |
| `prompt-optimizer.js`（637 行） | 本仓 `prompt_polish` + `configs/comfyui/prompts/*.json` + `prompt_utils` 已有一条优化链；再搬一套 = 两套提示词口径互相打架 ✗ |
| `model-paths.js` / `model-match.js` / `directories.js` / `folder-picker.js` | 本仓 `model_folders` / `local_model_scan` / `model_ecosystems` / `comfyui_match` 已覆盖且更细（含指纹与生态判定） |
| `skin-unlocks.js` / `sessions.js` | 皮肤解锁与单会话守卫，本仓没有产品化/多用户场景 |
| `comfy-client.js` 的端口自动发现 | 本仓 base 由 `COMFYUI_SERVICE_URL` 显式给；扫端口属分发层行为，会把"配置错"变成"偶尔能连上"✗ |
| `camera-orbit-widget.tsx`（680 行） | 伪 3D 预览要四个叙事参数（horizontalAngle/pitch/cameraDistance/wideAngle）+ 专属面板，画布没有宿主 |
| `canvas-node-progress-ring.tsx`（307 行） | 语义已由配置卡进度条实现（读同一份 SSE 数据），再搬是对同一件事的第二套实现 |
| `canvas-node-hover-toolbar.tsx`（314 行） | 右键菜单已覆盖同一批动作，悬浮条是体验优化而非能力 |
| `agent-*` 面板（约 4000 行） | 本仓已有 `agents.vue` + `AgentOutputDialog`，且那是"短剧智能体"域，与画布的 agent 面板语义不同 |

### 4.5 2026-11-03 补搬：蒙版局部重绘 + 扩图（并翻掉一条旧判断）

原先这两块被记成「盘上没有 inpaint / outpaint 工作流 ⇒ 搬来没有后端落点」，**这个判断是错的**：
`configs/comfyui/workflows/image-qwen21-inpaint.json` 与 `image-qwen21-outpaint.json` 一直在盘上，
注册表 `image-qwen21` 的 `variants.inpaint` / `variants.outpaint` 也齐全。真正缺的是**注入侧**（见下面第 3 条），
所以搬过来不是"点了没反应"，而是"点了**出图**、但图是错的且一行都不报错" —— 那才是更该修的。

| 源 | 落点 | 改造点 |
|---|---|---|
| `canvas-node-mask-edit-dialog.tsx` | `frontend/app/utils/canvasMask.ts` + `components/canvas/CanvasMaskDialog.vue` + `pages/canvas.vue`（`img-inpaint` 菜单项 / `confirmInpaint`） | ① 笔迹按**原图像素坐标**记录、导出时重放（`replayStrokes`），不存截图快照 —— 快照会吃死画布缩放与 DPR，改一次窗口大小蒙版就错位；② 双 canvas（底图 + 蒙版）增量绘、撤销走全量重放；③ 三道**静默失败**前置拦截：提示词必填（空提示词 = 原地重画）、至少一笔、蒙版覆盖率 < 0.2% 拒（几乎空的蒙版会出一张"几乎没改"的图，用户只会以为模型不行） |
| `outpaint-dialog.tsx` | `frontend/app/utils/canvasOutpaint.ts` + `components/canvas/CanvasOutpaintDialog.vue` + `pages/canvas.vue`（`img-outpaint` 菜单项 / `confirmOutpaint`） | ① 撑大后的画布尺寸必须与 `#18` 目标尺寸**成对**算（`outpaintPlan` 一处出数），只改一半 = 整张被硬缩回不相干比例、整体拉伸；② 比例预设 + 四边手填 + 羽化/像素预算；③ 预览占比靠**恒在 DOM** 的隐藏 `<img class="out-probe">` 量原图尺寸 —— 挂在"依赖尺寸才渲染"的元素里会死锁（不出现 ⇒ 图不加载 ⇒ 尺寸永远量不到） |

**两条口径**（别在后续改动里破掉）：

1. 对话框**只回答"怎么画蒙版 / 补多少"**，落成的是一张**真会跑的配置卡**（`runMode: 'comfyui'` + `workflowId` + `variant`），
   提交仍走卡片那唯一一条通道 —— 另开一条提交路径必然与卡片漂移；
2. 蒙版走 `uploadImageBlob` → `uploadedImages`，靠**连线顺序**表达 `imageIndex 0/1`（源图在前、蒙版在后），
   顺序反了就是把蒙版当底图。

**顺带修掉的服务层单点缺陷**（`comfyui.build_registry_graph`）：
图是 `catalog.load_graph(entry, variant)` 读的（用了变体），注入却是 `bindings.apply_bindings(graph, entry, …)`（**传的主 entry**）⇒
变体自己声明的 `bindings` / `parameters` 一条都不生效。症状全是静默的：
选了「局部重绘」提示词不落 `#7.prompt`、`steps` 不进 `#12` sigma 串、只给一张图不报"缺第 2 张"（蒙版被拿源图顶替）、
扩图四边恒 `0` 而 `#18` 恒 1024×1024（等于参数全丢）。修法：`resolved = catalog.resolve_variant(entry, variant)` 后交给 `apply_bindings`。

守卫：`backend-py/tests/comfyui_inpaint_test.py`（26 条，已进 `run_all`）——
钉参考图顺序、`ImageToMask(red)` 通道、两段尺寸成对、`int` 变换、缺第 2 张当场拒、
sigma 串**随 steps 变**（只钉"等于官方串"等于没测：工作流默认值就是官方串）。

### 4.6 2026-11-03 复审：`assistant-runtime` / `assistant` / `launcher` / `tools` / `infinite-canvas` + 源 `docs`

上一轮 4.4 的清点是按 `comfyui-gateway` 与 `infinite-canvas/web` 做的，**这三块当时没进视野**：
`assistant-runtime/`（整目录漏审）、`canvas-agent/`、`plugins/`。这次逐目录过了一遍，结论如下。

| 源 | 规模 | 判定 | 理由 |
|---|---|---|---|
| `assistant-runtime/` | 257 文件（**本体只有 4 个 mjs**，其余是 `workspace/infinite-canvas/web` 的前端旧副本） | **迁内核** ✓ | 常驻修复 Agent：L0 只读诊断 → L1 读/搜源码 → L2 提案 → L3 应用+回滚 → L4 更新判断。本仓 `runtime_doctor` 只覆盖 L0 ✗，用户拿到 blockers 之后**没法自助修**。`workspace/` 那份前端与 `infinite-canvas/web` 重复且更旧 ⇒ 不迁 |
| `assistant/` | 7 文件 | 不迁 | 同一套助手的**旧版**：文件结构与 `assistant-runtime` 一一对应，但缺 `skill.mjs`（权能边界提示词）与提案/线程落盘 ⇒ 取新者 |
| `launcher/`（`start.mjs` 76KB） | 8 文件 | 不迁 | 探测/接管 ComfyUI、托管 dist、看门狗、依赖自愈 —— 全是**分发层**。本仓 ComfyUI 用显式 `COMFYUI_SERVICE_URL`，没有"接管别人机器上的 ComfyUI"这个场景 |
| `tools/` | 7 文件 | 不迁 | `build-patch` / `build-workflow-baseline` / `build-assistant-runtime` / `update-bridge`：源项目的打包与差量补丁工具链 |
| `infinite-canvas/canvas-agent/` | ~30 TS 文件（含高覆盖测试） | **已重写并接通** ✓（原判"后置"，同日补齐两端） | 画布 Agent 服务：MCP 服务端 + `codex app-server --stdio` + 34 个画布工具 → op 数组。op 转换层（`buildCanvasToolRequest`）**必须有"agent 服务 + 前端 op 应用方"才有意义**，只迁这一层等于死代码 ✗ ⇒ 本仓把**两端都补上**：`canvas_agent.py`（模型→op）+ 前端 `utils/canvasOps.ts`（op→画布），故不再是死代码；`codex` 子进程层不迁，详见 §4.8 |
| `infinite-canvas/plugins/` | SDK + 7 个内置插件 | **已重写** ✓（契约 + 5 个 `.vue`） | 源 SDK 供 **React JSX runtime** 与 React external ⇒ SDK 本身不迁（本仓是 Vue）；6 个插件的**行为**重写成 5 个 `.vue`，节点类型放开成「内置五类 ∪ `<pluginId>:<name>`」，自定义键落 `node.values` |
| `infinite-canvas/canvas-proxy/` | 1 文件 5KB | 不迁 | 开发期反代 |
| 源 `docs/ARCHITECTURE.md` | 129 行 | 已吸收 | 启动器/网关/画布/资源安装器四层划分与"刻意一致"的行为口径，已并入 `feature-map.md` 与本文第五节 |

**本轮已迁（有调用方、可独立测）**：

| 源 | 落点 | 说明 |
|---|---|---|
| `assistant-runtime/repair-tools.mjs` | `backend-py/app/services/repair_tools.py` | 护栏内核：路径逃逸、拒绝段/拒绝文件、**每处替换唯一命中**、应用前全量备份、**方案过期整单拒**、回滚缺备份不静默 |
| `assistant-runtime/server.mjs` 的 proposals 段 | `backend-py/app/routers/repair.py` | 7 个端点；提案落盘、只回 `proposalId`（**校验基准不出网**） |
| — | `backend-py/tests/repair_tools_test.py` | 30 条，已进 `run_all` |

**三处刻意差异**（改了要重新想清楚）：

1. 源把可改范围硬编码成 `infinite-canvas/web/`；本仓改成 `allowed_prefixes`（默认只给 `frontend/app/`，最小权能）。
2. 源 `applyProposalFiles` **没有写中途失败回滚**（只靠"写前已全量校验"）；本仓补了：第 N 个文件写失败就把前 N-1 个**就地回滚** —— 校验过 ≠ 写不失败（磁盘满 / 文件占用）。
3. ⚠️ **换行符**：源是 JS 字符串原样读写；Python 默认 `open(newline=None)` 会把 `\n` 转成 `os.linesep`，在 Windows 上等于**把整份 CRLF 文件改写成 LF** ✗ —— 于是 `oldContent == 盘上内容` 的过期校验从此永远为假、备份与原文件字节不一致。读写两边都强制 `newline=""`，测试里用 CRLF 文件跑完整「提案→应用→回滚」逐字节比对钉住它。

**资源侧**：四目录里 `resources/` 的 73 条资源清单已与「本仓实际会引用哪些文件名」比对过 ——
`taeh3.safetensors` 直接迁入共享池 ✓，13 件需下载（清单写随包但源里没有的那件也在内），
其余 57 条（自定义节点 / LLM / VDN / ONNX / TensorRT）本仓用不上。详见 [pending-downloads.md](pending-downloads.md)。

### 4.7 2026-11-03 复审：`comfyui-gateway` 剩余面（两条行为搬入、五处不搬、一处顺带修）

4.1 里 `runner.js`（队列/单飞/超时/产物回收）是**整份**判成"结构层已有替代"的。这条判断
对**运行时骨架**成立（本仓 `comfyui.py` 自己有队列、轮询、超时、落地），但它把两条**行为**一起略掉了：

| 源 | 落点 | 为什么它算缺口（也是这轮翻掉的旧判断） |
|---|---|---|
| `src/runner.js` 的 `isOOM` + `POST /free` + 重提交一次 | `comfyui._process_run`（`_submit_and_poll` + `oom_retried` 分支） | 上游第一句失败常常就是 `allocation on device … out of memory`，**换一次显存后往往就跑得过**；原先我们直接判失败 ✗ ⇒ 用户看到"生成失败"却不知是显存、更不知本可以自动过。本仓 8765 门面早就有 `/v1/free`（`comfyui.free_upstream` 也一直在），**只是没人接在失败路径上**。口径：**只重试一次**（每次重提交都是一次真 GPU 占用 ✗），重试成功也要落 `warnings` 并把用量 `retryCount` 记成 1（否则"这次为什么慢一倍"台账查不到 ✗） |
| `src/runner.js` 的 `failureHint` | `comfyui._FAILURE_HINTS` / `failure_hints()` / `describe_failure()` | 上游原文是英文节点堆栈（`Value not in list: unet_name: '…' not in […]`），对我们没用 ✗。现口径：**原文一字不改** ✓ + 追加一条能直接动手的建议（缺模型 / 缺节点包 / 缺 SageAttention / VDN 缺 Stage-B / 显存不足 / 上游不可达）；**认不出就只留原文**（不硬编原因 ✗） |

**五处不搬（这轮逐个验过，不是遗漏）**

| 源 | 理由 |
|---|---|
| `src/qwen-edit.js`（目标图轮到最后 + 提示词编号重排 + 剥离"参考图片编号"前言） | ① 前言是**源前端自己加的**（`infinite-canvas/web` 的 `imageReferences.promptPrefix`），本仓画布从不加（`frontend/` 里 `参考图片编号` 零命中）⇒ `stripImageLabelPreamble` 没有对象；② 轮换方向与盘上节点**相反**：`comfy-studio/ComfyUI/comfy_extras/nodes_qwen.py::TextEncodeQwenImage21.execute` 的空 latent 取的是**第一张**参考图尺寸（原注释："Empty latent on the first reference image's size, to match with sampling as any other size shifts the edit"）⇒ 把目标图轮到最后 = 输出尺寸跟着**参考图**走，静默跑偏且一行不报 ✗。本仓口径：顺序**照连线原样**（`imageIndex` 0/1/2…），谁当底图由连线顺序决定；`image-qwen21` 的 `parameters.resolution` 已写明"输出尺寸跟随第一张参考图" |
| `mock/comfyui-mock.mjs` + `test/fixtures/*` | 本仓测试直接起 Starlette stub（`tests/comfyui_runs_test.py` 现起 8 个端点，含刚加的 `/v1/free`）⇒ 不需要第二个 mock 服务 |
| `scripts/convert-ui-workflow.mjs` | UI→API 转换已内建在提交路径（`comfyui_ui_workflow.to_api`，`POST /runs` 收 UI 格式自动转）⇒ 独立 CLI 没有额外能力 |
| `src/free-workflows.js` | "免绑定工作流"＝本仓 `POST /runs` 直接收整张图；再建一套库等于两条提交通道 ✗ |
| `configs/comfyui/prompts/*.json`（4 份，已随数据迁入）+ `prompt-optimizer.js` | 数据留档可用（能随 `POST /runs` 的 `workflow` 直接提交），但**不建** `/prompts/optimize|caption` 端点：这 4 份依赖上游 `TE_*` / `llama_cpp` / `Florence2` 节点包，建端点等于把"上游装了什么插件"变成我们的可用性前提；文本增强本仓自己有 `text_generation` 链 |

**顺带修掉一个真缺陷**（这轮跑守卫时露头，与本轮改动无因果关系）：
`comfyui.prepare_reference_media` 原先对**所有 kind** 都做"把远端参考图下载再上传"，
而 H3 走的是另一条协议（`/v1/video_generation` 收 `referenceImageUrls` 自己取）⇒
一条本来能跑的 H3 参考图请求会被前置处理**自己打掉**（远端取不到就抛，`tests/comfyui_runs_test.py`
的 ⑫′ 用 `https://x/a.png` 正好撞上）。现按 kind 分流（h3 直接返回 ✓）。

守卫：`backend-py/tests/comfyui_runs_test.py` 追加 ⑬–㉒（显存重试成功/重试仍失败/只重试一次/记账、
缺模型与缺节点包的建议、`describe_failure` 认不出不编原因、SageAttention 与 VDN 的退路提示、`is_out_of_memory` 判据），
并把后面原本重复的 ⑬–⑰ 顺延为 ㉓–㉗（编号跟执行顺序一致）。

### 4.8 语言栈铁律：源项目的 Node/React/TS **一律重写，不搬文件**

创作者 2026-11-03 明确：**不符合本仓语言的，都要重写**。所以「语言不同」**不是**不迁的理由，
而是**换语言重写**的理由。

| 位置 | 只准有 | 判据 |
|---|---|---|
| `backend-py/` | `*.py` | 后端不出现 `*.js/*.mjs/*.ts/*.tsx` |
| `frontend/` | `*.vue` + `*.ts` | 前端不出现 `*.tsx/*.jsx`（React 语法） |
| `desktop/` | Rust + `*.mjs` 构建脚本 | Tauri 工具链本身是 Node，属必需 |

源项目是 **Node(Express) + React**，本仓是 **Python(FastAPI) + Vue3/Nuxt** ⇒ 逐件重写的对应关系：

| 源（不搬） | 本仓语言重写 | 状态 |
|---|---|---|
| `assistant-runtime/repair-tools.mjs` | `app/services/repair_tools.py` | **已重写** ✓（Python 原生，非转译） |
| `assistant-runtime/diagnostics.mjs` | `app/services/repair_diagnose.py` + `GET /api/v1/repair/diagnose` | **已重写** ✓ |
| `assistant-runtime/server.mjs` 的 agent 循环 | 复用本仓 `app/routers/agent.py`（已有工具调用环） | 复用，不新增 |
| `assistant-runtime/skill.mjs`（L0–L4 权能边界） | `app/services/repair_skill.py` + `GET /api/v1/repair/skill` | **已重写** ✓（只留纪律：清单从 `repair_tools`/`repair_diagnose` **取值**；剔除源项目运维手册，见下） |
| `infinite-canvas/canvas-agent/src/canvas/*.ts`（34 画布工具 + op 转换） | `app/services/canvas_ops.py` + `canvas_agent.py` + `app/routers/canvas.py` + `frontend/app/utils/{canvasOps,canvasAgent}.ts` + 画布页 AI 输入框 | **已重写并端到端接通** ✓（22 工具迁移，12 个本仓无宿主并逐条写理由；**不含** `codex-client` / `http` / `session`，见下） |
| `infinite-canvas/plugins/canvas/*`（React 插件 SDK + 6 插件） | `app/services/canvas_plugins.py`（契约 + 校验 + 5 插件清单）+ `frontend/app/utils/canvasPlugins.ts` + `components/canvas/plugins/*.vue`（5 个渲染组件） | **契约 + 渲染槽都已接** ✓（`hosted=true`，见 4.8 节末） |
| `launcher/start.mjs`（1910 行：探活/接管/托管/看门狗） | — | **不重写**：逐项都有本仓等价物（见下），重写＝再造一套已在 `desktop/` 实测过的进程管理 ✗ |
| `tools/*.mjs`（打包/补丁链） | — | **不重写**：那套是源项目的发版工具，本仓无对应场景 |

**`skill.mjs` 为什么必须「重写」而不是「翻译」**：源正文 13742 字节里，真正可用的只有纪律那部分
（一工具一问句、先取证后动手、最小改动、每处替换唯一命中、不越界、不谎称已改）；
其余是**源项目的运维手册** —— `v0.22.7/8/11/12` 的版本行为差异、`/api/models/events` 的 SSE、
`LOCAL_ONLY` 渠道、`QwenAnglePromptConcat` 节点、魔搭/GitHub 的 raw 地址、`http://127.0.0.1:8291`、
`restart_comfy` / `replace_model` 两个我们**没有**的工具。照抄进来 = 一份让模型去查不存在东西的说明书 ✗。

现口径（`repair_skill.py`）：正文用 f-string 从 `repair_tools.DENY_SEGMENTS` / `DENY_FILES` /
`DEFAULT_ALLOWED_PREFIXES` / 各 `MAX_*` 与 `repair_diagnose.MAX_PROBLEMS` / `MAX_MISSING_ROWS`
**取值**，所以「说明书与实现打架」这一族缺陷结构上不可能发生。
守卫 `tests/repair_skill_test.py`（27 条）双向对拍清单、并用 monkeypatch 改常量反证「正文真的跟着变」，
另有一条硬断言：上述源项目词汇**一个都不许出现**（出现即判「照抄而非重写」✗）。

**`canvas-agent` 的重写口径（34 → 22）**：源 `src/canvas/` 约 26 KB，是本仓**唯一真缺的能力**
（让 LLM 用自然语言摆画布），所以按本仓模型重写；但它**不是整体搬家**，有三处必须分开说清：

1. **重写了 `schemas.ts` + `operations.ts` + `tools.ts`** ⇒ `canvas_ops.py`（op 编译）
   + `canvas.py`（HTTP）+ `canvasOps.ts`（执行器）。关键差异全部按本仓写：
   `position:{x,y}`→`x/y`、`width/height`→`w/h`、`viewport.k`→`scale`（夹 0.2–3）、
   `metadata.content`→`text`、`metadata.*`→`values`（**一律字符串**）、
   **未知生成模式编译期就报错**（源会静默归成 image，那会让「跑音乐」变成「跑图」且不报错 ✗）。
   ⭐ `audio` 这一格**原先确实不迁**（本仓画布没有音频卡 ⇒ 迁了也无处落 ✗）；2026-11-06 **补回来了**
   ✓：自研引擎 Music3 档真出 wav 之后画布才有了宿主 ⇒ `NODE_TYPES`/`GENERATION_MODES`/
   `DEFAULT_NODE_SIZE` 三处加 `audio`、`canvas_generate_audio` 从「未迁」挪进「已迁」、
   `CanvasNodeCard` 真渲染 `<audio controls>`。**驱动这条改动的是能力，不是"源里有所以要有"** ✓
   （判断标准：产物真出得来、卡上真播得响；两条缺一就仍不该迁 ✗）。
2. **一处语义修正**：源把提示词同时写进「文本节点」与「配置卡的 metadata」——同一句话存两份，
   改一处另一处不动 ✗。本仓配置卡正文是 `@[node:<文本 id>]` 的**真引用**，改文本节点配置卡跟着变 ✓。
3. **12 个工具不迁**（`site_navigate`、`canvas_list_projects`、`workbench_*`、`assets_*`、
   `prompts_search`、`generation_get_status`…）—— 源头是「多画布项目 + 生图工作台 + 素材库」
   这一整套页面，本仓画布是**单例 localStorage**、生成参数就是配置卡的 `values`、
   资产走 libraries 域 ⇒ 这些工具在本仓**没有宿主**。逐条理由写在 `canvas_ops.py::DROPPED_TOOLS`，
   由 `canvas_ops_test.py` 断言「22 + 12 = 34 且每条都有理由」，防止以后被当成漏迁又补回来 ✗。

**`canvas-agent` 的 Agent 环怎么落的（比源短得多）**：源是「MCP 服务端 + `codex app-server --stdio` 子进程 + 34 工具」；
本仓 `app/services/canvas_agent.py`（约 170 行）复用已有的文本适配器与工具环（`default_generate` + `get_text_config`），
不启子进程、不连 MCP。三条设计判据：

1. **工具不产生副作用** —— 画布状态在前端（`useCanvas` 的 localStorage），服务端看不见也改不了 ⇒
   写工具只往 `sink` 里攒 op，回给模型的必须是「已登记 N 条」这种人话，**不能**假装已执行 ✗（此刻画布没动）。
2. **只读工具必须真给数据** —— 没快照时如实回 `error`；返回 `{}` 会让模型以为画布是空的，
   然后凭空造节点 id ✗。有快照时给**压缩后**的（正文截断、未知字段不外泄）。
3. **参数 schema 单一真源** —— `/canvas/tools` 与 Agent 的工具注册都走 `canvas_ops.param_schema`。
   特别地 `ops` / `items` / `ids` / `connections` 必须是 `array`：写成 `string` 时模型会真的把数组
   **序列化成字符串**再传，于是校验报「不是数组」，看起来像模型蠢其实是 schema 错 ✗。

**本轮实证抓到的一个坑（只有接了真实画布才露头）**：`validate_ops` 原先只把「快照里的 id」当存在，
于是 `canvas_generate_image` **带快照**时会把自己同批刚建的文本/配置节点判成「指向不存在的节点」而整条报错；
**不带快照时一切正常** ⇒ 单元测试全绿、一接真实画布就崩。修法：本批 `add_node` 的 id 按 op 顺序推进进
`known`，并把「要不要做存在性校验」显式化为 `strict = snapshot is not None`。守卫加在
`canvas_ops_test.py::编译: 带快照时本批自建的连线不算幽灵`。

**实测（2026-11-03，本机 ollama `qwen3:14b`）**：`建三个文本节点，内容是：开头、高潮、结尾，竖着排开`
⇒ 一次工具调用 `canvas_create_text_nodes`、产出 3 条 `add_node`、`errors=[]`、回话「已创建三个文本节点…并竖着排开」。
前端入口在画布页工具栏（`pages/canvas.vue` 的「AI」输入框），落盘走 `utils/canvasAgent.ts` → `utils/canvasOps.ts`。

**`canvas-agent` 里这三块不重写，理由不是语言**：

* `src/agent/codex-client.ts`（51 KB）+ `codex-history.ts`（42 KB）：是**把 Codex CLI 当大脑**的
  子进程协议适配层（spawn、JSONL 事件流、会话恢复）。本仓的 agent 走自己的 HTTP 工具环
  （`app/routers/agent.py`）⇒ 重写它等于在本仓再造一个 Codex 客户端，**不是迁移是新增技术债** ✗。
* `src/server/http.ts`（36 KB）+ `session.ts`（29 KB）：源那边**自己起一个 HTTP+WS 服务**；
  本仓 FastAPI 已经是服务，画布状态在前端 `useCanvas` ⇒ 服务骨架与"画布连接状态"在本仓没有位置。
* `src/skills/store.ts`（22 KB）：源那边的技能存储；本仓技能在 `app/skills/` + `/api/v1/skills` ✓ 已有。

**`plugins/canvas` 的重写口径（SDK + 6 插件 → 契约层 + 渲染槽都已接）**：

* **契约层**：插件清单的**形状与校验** ⇒ `app/services/canvas_plugins.py` + `GET /api/v1/canvas/plugins`
  + `POST /api/v1/canvas/plugins/validate`。校验盯 8 件事：id 必须 kebab-case、id/type 不许撞内置节点类型、
  type 必须 `<pluginId>:<name>`（否则两个插件能抢同一个 type 而后者静默覆盖 ✗）、尺寸 120–2000、
  `defaultMetadata` 不许占节点**平铺真字段**（写了也读不到，还会与节点状态打架）、
  `hidePanel` 与 `builtinPanel` 不许并存、内置面板 mode ∈ `text|image|video|audio`（判据**读** `GENERATION_MODES`，不手抄四字）、
  `contentComponent` 必填。5 个插件清单登记在册，`template` **不迁**（源里就是给插件作者的起步样板，注释原文「改我」）。
* **渲染层**：5 个 `Content` 重写成 `frontend/app/components/canvas/plugins/*.vue`
  （`pluginOf` 查清单 → `PluginNodeBody` 按 `contentComponent` 取组件 ⇒ HTML / Markdown / 全景 / 便利贴 / SVG）。
  三处安全口径：HTML 用 `sandbox="allow-scripts"`（**不带** `allow-same-origin`）+ 默认 `pointer-events:none`；
  SVG 走 `<img src="data:image/svg+xml">` 而不是 `v-html`；Markdown 自写渲染器先 `escapeHtml` 再拼标签、
  链接只放行 `http(s)/mailto/#//`。
* **为接宿主动的三处画布核心**（按此顺序）：① `CanvasNodeType` 从封闭联合放开成
  「内置五类 ∪ 模板字面量 `<pluginId>:<name>`」（**不是**裸 `string` —— 那样 `NODE_TYPE_LABEL[type]`
  会静默返回 `undefined` ✗），并加 `isBuiltinNodeType` / `isKnownNodeType` / `nodeTypeLabel` 三个判据；
  ② `CanvasNodeCard.vue` 加插件槽 `PluginNodeBody.vue`，三态（正常 / 登记了但缺组件 / 未安装）**都必须画**
  —— 未登记的类型保留节点数据并显示「未安装的插件」，导入别家画布时不会凭空吞节点 ✓；
  ③ 插件自定义键统一落 `node.values`（不造 `metadata` 层），由 `RESERVED_METADATA_KEYS` 挡「插件键占用真字段」。
* **可建性**：`canvas.vue` 工具栏多了 5 个插件按钮；`canvas_ops.node_type_problem()` 让 Agent 也能建
  **已登记**的插件节点（形态对但没登记 ⇒ 当场报错并列出可用清单；插件节点**不带** `w/h`，
  默认尺寸由前端清单填 —— 后端猜一个数值会把插件自己的默认尺寸顶掉 ✗）。`hosted=true` 如实标记已接。
* **三处口径必须改**：`{width,height}` → `{w,h}`；`icon: ReactNode` → **字符串**（Vue 侧吃不下 React 节点 ✗）；
  运行时注入的 `React`/`jsx`/`Fragment` **一律不注**（本仓是 Vue，注 React 会装两份框架 ✓✗）。

**`launcher/start.mjs` 为什么不重写**（1910 行，逐项对账，不是"没场景"式的空话）：

============================  ======================================================
`start.mjs` 的职责            本仓等价物
============================  ======================================================
探活 ComfyUI(8188) / 网关(8190)  `GET /api/v1/comfyui/system`（上游体检 + 本地运行统计）
拉起缺失服务（comfy→gateway→canvas）  不需要：本仓后端就是网关本体；模型由自研 `engine/` 自己装配
静态托管 `web/dist` + SPA fallback     Nuxt 构建产物由后端托管，无独立静态服务器
关页面连带停后端（`stopAdoptedComfyOnCanvasClose`）  Tauri 壳做进程回收（`desktop/`，已实测送 WM_CLOSE 后 python 子进程全灭、5790 释放）
`bootstrap-nodes.mjs` 装 custom_nodes  不需要：`engine/` 直读 safetensors / GGUF，不走 ComfyUI 节点
============================  ======================================================

**结论**：`launcher` 是"源项目必须由 Node 一键起全栈"的产物；本仓已有**更完整**的等价链
（Tauri Rust 壳 + FastAPI + 自研引擎 + ComfyUI 域端点），而且**连 ComfyUI 进程都不是必需的**
（`engine/` 八条装配线已真跑出片）⇒ 重写它等于在本仓再实现一遍进程探活/回收，与 `desktop/`
那份已实测的实现并存，是**新增技术债而非迁移** ✗。

**存量核查（全仓扫过，2026-11-03）**：后端里只有两处非 Python ——
`app/skills/production-tools/{multi-shot,n-storyboard}/scripts/index.js`（11285 + 17788 行）。

⚠️ **这两处不是本仓逻辑，不能按"重写"处理，登记为唯一例外**：

* 它们是 **MiniMax 官方技能包自带的 GUI 面板打包产物**（`meta.yaml` 写 `author: MiniMax Design`；
  `index.js` 首行注入 Tailwind CSS、第 2 行起是 `rolldown/runtime.js` ⇒ 编译产物，不是手写源码）。
* 用途是**远端**的：`SKILL.md` 的 `allowed-tools: hub_open_remote_tool_gui, hub_submit_dag`，
  `STEP 1` 让模型调 `hub_open_remote_tool_gui` 并传 `entry: "scripts/index.js"` ——
  面板在 **MiniMax Hub 侧**运行，与本仓技术栈无关。
* 本仓 `backend-py/app` 与 `frontend/app` 对它**引用数均为 0**（`gui.json` 也只有 `guiWidth/guiHeight` 两个尺寸字段）。
* ⇒ 改写或删除会**打断外部平台的 DAG 调用契约**（`hub_submit_dag` 找不到入口），属能力削减 ✗。
  保留原样并在本节登记；若某天要脱离 Hub 自持这两个技能，正确做法是**用 Vue 重写参数面板**，
  而不是把 React bundle 塞进 Nuxt。

### 4.9 2026-11-03 提示词模板：从「数据留档」补成「可用通路」

4.7 里四份 `configs/comfyui/prompts/*.json` 的结论是「数据留档可用，但不建端点」。这条判断的
**前半句其实不成立**：四份模板的落点全是中文输入名，其中两格**不是常量、要按上下文算** ——
H3 的 `#3.输入模式`（有参考图=图片）与生图增强的 `#1.任务模式`（有参考图=图生图），
让调用方自己拼，这两格必然停在上一次的值上，而**错了不报错**（参考图白传 / 只当风格参考 ✗✗）。
所以「能当 `workflow` 直接提交」只在"有人手工改对这两格"的前提下成立。

**补的是编译层，不是端点式的聊天通路**：

| 落点 | 内容 |
|---|---|
| `app/services/comfyui_prompts.py` | 四份模板的**参数落点声明**（`Param(node, input, kind, default, choices, 上下限)`）+ 一条公共填图流程 + 每份一个 `post` 钩子（只放"按上下文算"的那几格）。产出**可提交的 API 图**，**不提交、不连上游、不碰显卡** ⇒ 离线可用 ✓ |
| `GET /api/v1/comfyui/catalog/prompts` | 目录：参数表 + 行为键 + **所需节点类**（拿去和 `GET /catalog/nodes` 比，缺哪个插件当场看得见）+ 模板漂移体检（声明的落点不在模板里 ⇒ 该行报 `error`，不给人一张填了也不生效的假表单 ✗） |
| `POST /api/v1/comfyui/catalog/prompts/{id}/compile` | 填参数 ⇒ `{workflow, requiredNodes, images, language, warnings}`；结果原样塞进 `POST /runs` 的 `workflow` 就跑（DB/记账/产物/失败归因都在那条路上 ✓） |
| `configs/comfyui/prompts/skills/h3-prompt-writing/` | H3 提示词写作 skill 的三份文本（`SKILL.md` 2663 B、`references/base-en.txt` 15995 B、`references/ref-en.txt` 23894 B）随 MIT `LICENSE` + `NOTICE.md` **逐字节**拷入（sha256 对过 ✓）。**只作文本资产留存**：它服务的是源项目的「云端增强」分支，本仓没有云端 chat 客户端 ⇒ 不搬 `buildCloudH3System`/`cloudChat` ✗（搬了就是没有调用口的死代码 ✗），目录接口里 `h3Skill.consumedBy` 如实写明 |

**刻意与源不同的四处**（都是"源会静默、我们当场说"）：

1. **未知键直接 400** ✗（源用 JS 解构，拼错的键被静默忽略 ⇒ `{"secods": 10}` 会按默认 5 秒跑，人以为改了 ✗）；
2. **超长提示词报错、不截断** ✓（源也是 `prompt_too_long`；截断会让人以为整段都送进去了 ✗）；
3. **夹紧留 warning** ✓（时长 1..15 与源同口径 `clamp`，但夹了要留痕 ✓）；
4. **`engine` 开关拆成两个 template id**（源靠 `engine: "florence"` 切图，口径上仍是一份"模板"；本仓目录里点哪份就是哪份 ⇒ 不存在"传了 florence 却仍在 qwen35 图上跑"的错位 ✗）。

**未搬且不打算搬**：`prompt-optimizer.js` 的 HTTP 服务层、`/prompts/optimize|caption` 端点、
云端 chat 拼装 —— 前者是 Node 运行时骨架（4.1 已判），后两者要求上游装着 `TE_*`/`llama_cpp`/`Florence2`
并把"云端密钥"变成我们的可用性前提（与 4.7 同一理由）。文本增强本仓继续走 `text_generation` 链 ✓。

守卫：`backend-py/tests/comfyui_prompts_test.py`（60 条：目录四份齐备 / 两格按上下文算 / 参考图接成
`LoadImage(9101+i)` / **反复编译系统提示词不叠加** / 夹紧留 warning / 拼错的键·非法任务类型·超长·超 8 张一律拒 /
两份反推各自的 system prompt 与 `instruction` 覆盖 / 模板被改坏时报错且不拖垮整个目录 / skill 三份资产非空）。

### 4.10 2026-11-03 参数落点推断：未登记工作流也能填参数（`registry.js` 的 `inferBindings`）

4.1 把 `runner.js` 判成"结构层已有替代"时，`registry.js` 的 `inferType` / `inferBindings`
被一起略过了。它的真实价值不是"再写一份注册表"，而是**兜住没登记过的图**：本仓 21 条工作流的参数
落点全是手写（`configs/comfyui/workflows/registry.json`），用户从 ComfyUI 导出一张自己拼的图
交上来 ⇒ `POST /runs` 能跑，但**一个参数都改不了**（提示词/种子/尺寸全图里烘死）✗。

| 落点 | 内容 |
|---|---|
| `app/services/comfyui_infer.py` | API 图 ⇒ `type / bindings / parameters / requiredNodes / nodeCount / found / missing / warnings / suggested`，**与 registry 同形状** ⇒ 前端复用同一套渲染，人过目后写进注册表即可当正式工作流用（推断只起草，**不自动登记** ✗）。纯函数：不写盘、不连上游、不改原图 |
| `POST /api/v1/comfyui/workflows/infer` | body `{workflow, name?, type?}`；两种**必须当场拒**的：① UI 格式（带 `nodes` 键，那不是能提交的图）② 没有输出节点（跑一遍什么都不落盘）。错误消息带 `[ui_format]` / `[no_output]` / `[bad_hint]` 前缀给脚本分流 |

**判据优先级（这是整件事的关键）**：① 顺采样器 `positive` / `negative` 的**连线**反查真正吃文本的那一格
（支持往下**两跳**：`CFGGuider.positive → TextEncodeQwenImage21.prompt(连线) → QwenAnglePromptConcat.base_prompt`）；
② 才退回标题/顺序兜底，且标题含"负向"的节点**不许**当正向。

**刻意与源不同的五处**（源会静默，本仓当场说）：

1. **图上本来是连线的格一律不绑** ✓（源不区分）：绑上再填一个值 = **拆掉别人接好的线**，
   图照跑、产物全变样、**不报错** ✗ ⇒ 只进 `warnings`（"这一格是上游算出来的"）；
2. **正负向是两套键** ✓：源的单一候选表会把 `TextEncodeQwenImage21` 的 `negative_prompt` 读成正向
   （同节点两格反查时又会被"同节点去重"吞掉负向）⇒ 提示词全灌进负向格、图照出、内容全反 ✗；
3. **负向保留图上原值作默认** ✓（源写死 `default: ""`）：抹掉别人写好的负向 = 静默降质 ✗；
4. **采样器/调度器不绑** ✓（源推 `samplerName`）：可选值是**节点定义里的枚举**（服务在线才拿得到），
   绑成自由文本 = 让人填一个不存在的名字，ComfyUI 悄悄回落到默认 ✗；
5. **不动调用方的 `name`** ✓（源 `ensureTypePrefix` 强制改名成 `image-xxx`）：只在 `suggested` 里给建议。

**并额外推了本仓才有的落点**：`unet` / `ckpt` / `lora` + `lora_strength`（换 LoRA 不给强度 ⇒
一直用图里烘死的那个强度 ✗）/ `upscale_model` / `image` · `image2` · `video`（参考素材）/
**参考图 array**（`ref_image_N` 与 Qwen-2.1 的 `images.image_N` 都认，`indexBase` 按图给）/

**推不出来就如实进** `missing` **+ warning**，三类：① 未连线的**可选数组槽**在 API 图里根本不出现
（`TextEncodeQwenImage21.images.image_1…4`、H3 的 `ref_images.*`）；② 族专用旋钮
（`rho` / `w_max` / `lowres_scale` / `transition_step` / `blocks_to_swap` / `task_type` / 各种 cache 开关）——
值域只有该族自己知道；③ 本来就是连线的格。

守卫：`tests/comfyui_infer_test.py`（71 条：口子/类型/文本三路/尺寸与帧族/采样/参考/模型/旋钮/纯函数 +
**与 21 条已登记工作流逐格对拍**，并反验"真图里没有一条绑到连线格"）+ `tests/comfyui_infer_route_test.py`（12 条）。

### 4.11 2026-11-03 运行记录保留清理：过期任务与产物一起走

`comfyui_runs` 每跑一次长一条，附带产物（`static/comfyui/` 下的 PNG/MP4）几百 KB 到几十 MB。
源项目没有这块（它把记录交给浏览器本地）；本仓是持久化 + 产物落盘，不清就一直涨，清错就**不可逆** ✗。

| 落点 | 内容 |
|---|---|
| `app/services/comfyui.py::purge_runs` | 过期运行记录与其本地产物一起清，返回的是**账**不是布尔（`considered/runs/files/bytes/missingFiles/keptShared/keptOutside/keptRunning/failed`） |
| `app/scripts/comfyui_retention.py` | CLI 薄壳：**默认干跑**、`--apply` 才真删、`--json` 给自动化；退出码 `1` = 有产物删不掉（需要人看占用） |
| `GET /api/v1/comfyui/runs/retention` | **只读预演**：`?days=` 不传就按 `COMFYUI_RETENTION_DAYS`／缺省 30；恒 `dry_run=True`，能真删的人口只留 CLI 一个（删是不可逆的，不该做成地址栏里能点的 GET）；必须声明在 `/runs/{run_id}` **之前**，否则 `retention` 被当 id ⇒ 400「run_id 非法」 |

四条守卫（都属于"删错了不会报错"的口子）：

1. **在跑的绝不碰** ✗：`queued` / `processing` 过期也不清 —— 删了记录，后台还在写产物 ⇒ 孤儿文件再没人知道该删；
2. **行与产物同生共死** ✓：产物被占用删不掉（Windows 独占打开）⇒ **这条记录也留着**（先删行会让"该删哪个文件"的线索永久丢失）；
   反过来"文件早就不在了"不算错（`missingFiles` 计数、记录照删）；
3. **共享产物不删** ✗：同一路径还被别的记录引用（另一次运行 adopted 了同一份）⇒ 留着并报 `keptShared`；
4. **不认识的路径整行留着** ✗：只动 `static/comfyui/` 下的东西，其余路径宁可不清也不误删。

守卫：`tests/comfyui_retention_test.py`（40 条，含"删不掉 ⇒ 记录也留着 ⇒ 放开占用后下一次能清掉"的完整闭环，
`COMFYUI_RETENTION_DAYS` 作默认值、显式 `days` 压过它、环境变量写坏退回默认，
以及 ⑫ 系列盯预演端点：恒干跑、路由顺序没被 `/runs/{run_id}` 抢走、跑完文件与记录都还在、`days=0` 一条不清）。

### 4.12 2026-11-04 对账：画布 `lib/` 余量与数据资产（结论：无新增可迁）

**一、数据资产已 100% 对齐**

| 项 | 源 | 本仓 | 结论 |
|---|---|---|---|
| 工作流模板 | `comfyui-gateway/workflows/` 递归共 22 个 JSON（21 个顶层 + `_internal/video-rife.json`） | `configs/comfyui/workflows/` 递归也是 22 个，同结构 | **逐个 sha256 全同**（21 个顶层 + `_internal/video-rife.json`），零差异 ✓。该 RIFE 工作流在本仓只作留档：实际插帧走 ffmpeg `minterpolate`（§三，`video_interpolate.py` 抬头写明为什么不用它），没有任何代码加载它 |
| 登记条目 | 源同目录 `registry.json`（84 754 B） | `configs/comfyui/workflows/registry.json` | **sha256 相同** ✓；21 条 = 17 个顶层 `file` + 3 个子工作流（`image-qwen21-inpaint/outpaint/t2i`），递归 20 个文件引用全在盘上，也无「盘上有未登记」 |
| 提示词模板 | `prompts/` 4 个 JSON（另有 LICENSE / NOTICE / SKILL / base-en / ref-en） | `configs/comfyui/prompts/` 同 4 个，**逐个 sha256 相同** ✓；另加本仓自有的 `h3-prompt-writing/`（源没有） | 零差异 + 一处增量 |

> 两处会误报的对比方式（本轮都踩过、都已逐条核实）：① 只读条目的顶层 `file` 字段 ⇒ `inpaint/outpaint/t2i` 三个子工作流被算成「未登记」，其实它们由条目内嵌引用（`registry.json:1717/1774/1879`）；② `Get-ChildItem workflows` 不带 `-Recurse` ⇒ 漏掉 `_internal/video-rife.json`，得出「本仓少一个」的错觉。用 sha256 逐文件对账才是准的。

**二、画布 `lib/` 剩余件（逐个验过，全部不迁）**

| 源 | 为什么不迁 |
|---|---|
| `canvas/node-registry.ts`（动态注册表 + `getNodePluginId`） | 只有「运行时从远程加载插件」才需要；本仓 `canvasPlugins.ts` 是**编译期静态清单**，5 个插件固定 ⇒ 动态表在没有加载器时是一张空 Map |
| `canvas/plugin-loader.ts`（170 行） | 远程插件下载 + 执行，需要沙箱与版本市场；本仓插件是仓库内 `.vue`，不开这个口子（安全面得不偿失） |
| `canvas/plugin-registry.ts`（官方市场 + semver） | 无市场端点、无 `PLUGIN_REGISTRY_URL`；`hasUpgrade` 在本仓走 `desktop/` 自己的更新通道 |
| `canvas/plugin-runtime.ts`（向插件注入 React / jsx / Fragment） | React 专用，按 §4.8 铁律不成立 |
| `canvas/plugin-node-context.ts`（`updateNode` / `getUpstream` / `applyOps` / `emit` / `on` / `storage`） | 本仓 `PluginNodeBody.vue` 只给三样：`node`（含 `values`）、上游 `text` / `images`（props 注入）、`patch` 事件——**读上游已经通了**，缺的是「按 id 取任意节点 / 应用 op / 跨节点事件 / 插件私有存储」。5 个内置插件（便签 / 全景 / Markdown / HTML / SVG）都是展示型，这四样无人调用 ⇒ 先接就是空接口 ✗。将来有插件需要它们时按这张清单补，每一项都已核对能在 `useCanvas` 找到落点 |
| `canvas/canvas-event-bus.ts`（事件总线 + 按 pluginId 隔离的 localforage） | 同上：当前无订阅方。插件私有存储还牵出一个口径问题（插件状态进不进画布文档）——本仓口径是落 `node.values` 随文档走 ✓，不另起一层 |
| `canvas/video-model-mode.ts`（4 个 `video-h3-*` → `videoMode`） | 本仓用**数据**表达同一件事：`registry.json` 每条自带 `inputs` / `minMedia`（ref2va 支持 1–6 图、frames 要 1–2 图…）。再写一份硬编码 id 表＝第二真相源 ✗ |
| `canvas/prompt-optimize-jobs.ts` / `canvas/caption-jobs.ts`（按 nodeId 的 AbortController 注册表：register / release / has / abort） | 它们服务的是**画布节点上的长任务**，而本仓画布没有「优化提示词」「打标」入口。本仓的提示词优化在 `VideoEditor.vue`：按 storyboard id 调 `/optimize-prompt`、`optimizing` 防重入、**不提供取消** ⇒ 没有 nodeId 这个维度，注册表无处落；§4.7 也已判定不建 `/prompts/optimize|caption` 端点 |
| `image-reference-prompt.ts`（14 行：给提示词**自动**加「参考图片1、2：」前言） | 同族的**编号**本仓已有：`canvasReferences.ts` 的 `label`（`参考图 1` / `参考视频 1`，按类别各自从 1 起）＋ `describeReferences()`（`参考图 1 · 标题` 行），够人和模型自己写进提示词。**不迁的是「自动拼前缀」那一步**：源拼完又在提交前 `stripImageLabelPreamble` 擦掉（§4.7），本仓没有这个来回，用户写什么就是什么 ✓ |
| `canvas/camera-orbit-widget.ts`（795 行） | three.js 相机轨道手柄（方位角 / 俯仰 / 距离），与本仓全景节点的 CSS 圆柱是两件事；为一个节点的手柄引入 three.js ✗ |
| `agent/agent-site-tools.ts`（331 行）+ `agent/agent-url-bootstrap.ts` | 源的 MCP 站点工具宿主；本仓 `canvas_agent.py` 只走自有 `default_generate`，§4.8 已定不启子进程、不连 MCP |
| `local-gateway-sync.ts`（420）+ `services/app-sync.ts`（422） | 与源 Node 网关做配置同步；本仓是单一 FastAPI，不存在「两处配置要同步」 |

**三、源代码测试不逐条搬**：源 `test/` 50 个 `.test.js` 是 mocha + 自带 mock 服务；本仓按域重写（`comfyui_*_test.py` 9 个 + 画布 3 个），口径是**守同一批行为**，不是一对一翻译文件。

**本轮结论：无新增可迁项，代码零改动。** 若某个判定要翻，只有一种情形——出现「有插件需要宿主能力」或「要接远程插件」的新需求，届时按 `plugin-node-context.ts` 那一行的清单补，而不是先搬空接口。

### 4.13 覆盖率对账：源侧文件「有没有归属」（结论：零盲区）

**方法**：反向查一遍——把源侧全部文件名（`comfyui-gateway/src` + 画布 `web/src/lib`，**77 个** `.js/.ts` 去重 basename）拿去比审计文档全文，**一个字都没出现过的**就是可疑盲区。跑出 9 个，逐个打开看过：

| 源文件 | 行数 | 结论 |
|---|---|---|
| `canvas/canvas-agent-ops.ts` | 89 | **已迁，只是改名** ⇒ 本仓 `utils/canvasOps.ts`，8 种 op 一一对应（`add_node` / `update_node` / `delete_nodes` / `delete_edges` / `connect_nodes` / `select_nodes` / `set_viewport` / `run_generation`；源叫 `delete_node`、`delete_connections`，本仓复数化） |
| `canvas/canvas-node-size.ts` | 16 | **已迁，只是改名** ⇒ 本仓 `utils/canvasNodeSize.ts`：`fitNodeSize`（最长边 ≤640、只缩不放）、`nodeSizeFromRatio`（比例 0.25–4 之外回默认）、`DEFAULT_NODE_SIZE`（text/image/config 340×240、video 420×236、group 760×480）三处逐条同口径 |
| `zip.ts` | 21 | 不迁：`createZip` / `readZip` 在全源仓**零调用方**（只有定义），本仓也没有 zstd/fflate 类依赖 ⇒ 死代码 |
| `localforage-storage.ts` | 34 | 不迁：zustand 的 `StateStorage` 适配器；本仓不用 zustand（画布状态在 `useCanvas`，文档落服务端） |
| `local-session.ts` | 58 | 不迁：local 模式 5 秒心跳 + 「最后一个页面关掉后回收后台」；本仓是 Tauri 壳管进程（已实测 WM_CLOSE ⇒ 子进程全灭、5790 释放） |
| `analytics.ts` | 86 | 不迁：官网 GA4 / 百度统计埋点（官方站用环境变量给 ID，fork 默认不注入）；本仓桌面端不该向第三方发请求 |
| `release.ts` | 24 | 不迁：解析 CHANGELOG（`## 版本 - 日期` + `+ [类型] 文案`）；本仓没有更新日志页，桌面更新走 `desktop/` 自己的通道 |
| `canvas/skin-unlocks.ts` | 60 | 不迁：卡片皮肤密码解锁（`fish` 皮肤存 PBKDF2 校验、不存明文）；本仓无皮肤系统（调色板里的 `skinTone` 是肤色参数，不是这套） |
| `comfy-bridge.js` | 115 | 不迁：把 `open-workflow.js` 装进 ComfyUI 的 `extensions/local-canvas/`，还得靠猜 `python_embeded/Lib/site-packages/comfyui_frontend_package/static` 定位前端目录；本仓直接打 API，不需要往 ComfyUI UI 里装扩展 |

**两条口径都要留话**（否则这套脚本会反复误报）：① 文件名没出现 **≠** 没审——上表前两行就是「改名已迁」，`canvasOps.ts` / `canvasNodeSize.ts` 在 §4.7、本文件抬头都写着出处；② 文件名出现 **≠** 审过——可能是被别处顺口提到。所以脚本只用来**找候选**，逐条必须打开看实现。

**结论：零盲区，代码零改动。** 下次源仓升级后复跑同一脚本，命中的先判「是不是改名已迁」，再判「是不是真漏」。

## 五、刻意保持一致的行为口径（回归时别改）

1. `snap16` **上取整**、`snap32` **四舍五入**（混用会让 832 变 816 / 1040 变 1024）；
2. 随机种子**右开区间** `[1, 2147483647)`；
3. H3 帧数 **17k+5**（不是 4n+1）；
4. 说话人标签只动描述段、画外音说明**只加一次**、认不出主体**不硬编 id**；
5. LoRA 链**必须串联**、模型源按**被引用次数**选、id 从 **9000** 起、无 lora 时**返回原对象**；
6. 媒体尺寸 `auto` 档返回**比例串而非像素**（调用方要能接住"还不是像素"）；
7. 视频尺寸**短边=分辨率**且**偶数化**；图片偶数化属 `snap16/snap32`，不在这里做；
8. 画布导入**先整份校验再写**，本地读档**丢坏留好**；
9. 新节点以**落点为中心**；卡片按素材比例适配时**只缩不放**、差 2px 内不动。
