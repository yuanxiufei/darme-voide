# 后端功能分类地图（feature map）

> **这份文档回答什么**：这个后端**已经实现**了哪些能力、每个能力落在哪个文件、彼此怎么分层。
> **不回答什么**：某个端点的字段/错误码逐条契约（那是 [`api-contract.md`](./api-contract.md)）、
> 某个模型格式能不能装（那是 [`compatibility-matrix.md`](./compatibility-matrix.md)）。
> 三者分工：**功能面**（本文）× **接口契约**（api-contract）× **格式兼容**（compatibility-matrix）。
>
> 归类时间：2026-10-30。归类方式：逐个读模块头 docstring + `app/services/engine/__init__.py::MODULE_GROUPS`，
> 由模型逐条判断落位，**不用脚本猜**。落点路径由 `tests/structure_test.py` 的两条判据守着：
> ㊄（点到的必须在盘上）/ ㊆（盘上的必须被点到）—— 任一侧不符即 FAIL，所以本文不可能悄悄落后于代码。

## 0. 分类原则（三条，先看这里再看表）

1. **一级 = 运行时职责层**，不是业务题材。理由：仓库本身就是这么分层的，且 `tests/layering_test.py`
   把依赖方向钉死了 —— 分类跟着这条线走，才不会出现「文档一套、代码一套」。
   单向链：`top → routers → agent → services → core`（`mcp` 只挂 `core`，`agent` 可调 `mcp`）。
2. **二级 = 层内的领域**。同一层内按「一次改动会同时碰到哪些文件」来切 —— 改剧本要素不会碰生成编排，
   改生成编排不会碰平台层。这样每个二级域就是一个**可独立交付的最小面**。
3. **一个落点只进一个格子**。互斥性由人工核对（本轮 285 个模块逐个归位，表格落点列里
   每个路径只出现在一个二级域 —— 2026-10-30 实测跨域重复 0）；覆盖性由不变量保证：
   `app/` 下每个 `.py`（除 `__init__.py`）都能在本文找到唯一一行，由 ㊆ 判据守着。
   找不到 ⇒ 说明分类体系缺一格，**先补体系再加功能**，不许塞进「其他」。

⚠️ 一条容易踩的边界：`app/services/` 顶层 `.py` 归 **L3**，它下面的**子包**（`engine/` `adapters/`
`ollama_native/` `lora_train/`）各自成层（L4/L5）。这样 L3 与 L4/L5 的落点永不重叠。

## 1. 一页总览

| # | 一级层 | 目录 | 模块数 | 二级域数 | 一句话职责 |
|---|--------|------|-------|---------|-----------|
| L1 | 接口层 | `app/routers/` 42 + `app/` 顶层 3 | 45 | 9 | HTTP 契约：入口挂载、路由、错误码形状 |
| L2 | 智能体层 | `app/agent/` `app/mcp/` | 23 | 5 | 协议、工具、循环、多智能体、评测闭环 |
| L3 | 业务服务层 | `app/services/*.py` | 75 | 12 | 领域逻辑：剧本→资产→提示词→生成→合成→质检 |
| L4 | 推理引擎层 | `app/services/engine/` | 64 | 6 | 自研 torch 推理：格式/量化/骨架/编排 |
| L5 | 本地执行与厂商适配 | `services/{adapters,ollama_native,lora_train}` `app/local_services/` `app/vendor/` | 48 | 4 | 把请求落到具体后端、进程、显卡 |
| L6 | 平台层 | `app/core/` | 6 | 3 | 配置、DB、模型表、请求/响应/CPU 预算 |
| L7 | 开发期工具 | `app/scripts/` `tests/` | 21 | 4 | 体检、基准、结构守卫（不进运行时） |

> 模块数口径：**该目录下 `.py` 文件数，排除 `__init__.py`，含子目录**。七层相加 **282**；
> 另有 `app/skills/**/scripts/` 的 3 个技能脚本（见 L2-5：不进应用分层，但也要点名）
> ⇒ ㊆ 实际要求全文点名 **285** 个 `.py`。
> 数字是 2026-10-30 实测值；增删模块必须同时改正本表与各节标题 —— 漏改会被
> `tests/structure_test.py` 的 ㊆ 抓出来（新模块没在本文点名即 FAIL）。
> L7 的「166 套测试」是套件数，会随开发增长 —— 以 `tests/run_all.py::TESTS` 为准，本文不追着改
> （但「盘上每份 `tests/*_test.py` 必须进那个清单」由 `structure_test` 的 ㊇ 守着 ✓）。

## 2. L1 接口层（45 模块：`app/routers/` 42 + `app/` 顶层 3，9 个域）

前 8 个域都住在 `app/routers/`，且**都是薄壳**：解析请求 → 调 L2/L3 → 按 `app/core/response.py` 的
`success/bad_request/internal_error` 出统一形状。业务逻辑一律不写在这里（写了就是分层违规，
`layering_test` 不拦这条，靠 code review 拦）。第 9 个域（L1-0）是**请求进门之前**的那三个文件，
它们不属于任何业务，但决定了「哪些请求走 Python、哪些转发给 Node」。

### L1-0 应用入口与横切（3，`app/` 顶层）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 应用挂载与路由注册顺序 | `app/main.py` | 顺序是**语义的一部分**：健康检查 → 已迁业务 → `/static/*` → `/api/v1/{path}` 兜底 → `/webhooks/{path}` → SPA fallback |
| 未迁移端点的透传接缝 | `app/passthrough.py` | 反代 Node 或返回 501；含「被参数路由遮蔽」的显式委派写法，`tests/route_parity_test.py` 靠它找遮蔽 |
| 请求日志中间件 | `app/http_logger.py` | 移植 `middleware/logger.ts`；`HTTP_LOG=0` 是给自检降噪的开关，默认开 |

### L1-1 项目与剧集（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 剧目 CRUD | `app/routers/dramas.py` | 与 Node 侧 `dramas.ts` 逐端点对齐 |
| 分集 CRUD | `app/routers/episodes.py` | 整域 10/10 已迁 |
| 全自动管线 | `app/routers/auto_pipeline.py` | 4 端点；编排逻辑在 `app/agent/auto_pipeline.py` |
| 生成历史 | `app/routers/generations.py` | 聚合视图，1 端点 |

### L1-2 剧本要素（7）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 角色库 | `app/routers/characters.py` | 12 端点全迁 |
| 场景库 | `app/routers/scenes.py` | — |
| 物品库 | `app/routers/props.py` | — |
| 分镜 | `app/routers/storyboards.py` | — |
| 连续性表体检 | `app/routers/continuity.py` | 契约 §6-§11 |
| 提示词生产工具 | `app/routers/prompt_tools.py` | 占位符解析 + 五段质感层 |
| 视觉关系图 | `app/routers/visual_graph.py` | 角色/场景/道具的图结构 |

### L1-3 生成与后期（8）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 图片生成 | `app/routers/images.py` | — |
| 视频生成 | `app/routers/videos.py` | Node 侧 290 行 |
| ComfyUI 能力 | `app/routers/comfyui.py` | 本项目自有入口（2026-09-17） |
| 宫格图 | `app/routers/grid.py` | 347 行；宫格提示词生成在 `app/agent/tools/grid_prompt_tools.py` |
| 合成 | `app/routers/compose.py` | 多轨拼接 |
| 合并 | `app/routers/merge.py` | 视频合并 |
| 开跑前体检 | `app/routers/preflight.py` | 四块能力串成一次调用 |
| 资产验收门 | `app/routers/asset_gate.py` | `asset-manifest.json` 形状 + 缺件阻断 |

### L1-4 模型与运行时（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 引擎体检 | `app/routers/engine.py` | 自研引擎的健康报告 |
| 本地模型 | `app/routers/local_models.py` | 11 端点 + 2 自研端点，Node 侧 688 行 |
| LoRA 训练 | `app/routers/lora_train.py` | LoRAMaster 移植 |
| AI 供应商配置 | `app/routers/ai_configs.py` | 含 `/ai-providers` |
| AI 音色 | `app/routers/ai_voices.py` | Node 侧 437 行 |

### L1-5 智能体与技能（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 会话 | `app/routers/agent.py` | 对齐 `agent.ts`（57 行） |
| Agent 配置 | `app/routers/agent_configs.py` | 含 skills 绑定 |
| 技能管理 | `app/routers/skills.py` | 6 端点；目录域非数据库 |
| MCP 客户端 | `app/routers/mcp.py` | 3 端点 |
| 评测 | `app/routers/evaluation.py` | 5 端点 |

### L1-6 资产与存储（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 资产版本 | `app/routers/asset_versions.py` | — |
| 上传 | `app/routers/upload.py` | 3 端点 |
| 存储根切换 | `app/routers/storage.py` | 2/2 |
| 资源库规格 | `app/routers/libraries.py` | 角色/场景/兵器/服装四类声明 |
| 导出 | `app/routers/export.py` | 7/7 |

### L1-7 预设与风格（3）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 预设 CRUD | `app/routers/presets.py` | — |
| 预设框架 | `app/routers/preset_framework.py` | 247 行 |
| 风格档案 | `app/routers/style_profiles.py` | — |

### L1-8 运维与观测（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 运行时体检 | `app/routers/doctor.py` | 把「服务活着没有」摆到开跑之前（2026-10-30） |
| 追踪 | `app/routers/traces.py` | 3 端点 |
| 用量成本 | `app/routers/usage.py` | — |
| Webhook | `app/routers/webhooks.py` | 123 行 |
| 应用设置 | `app/routers/app_settings.py` | — |

## 3. L2 智能体层（`app/agent/` 22 + `app/mcp/` 1，5 个域）

这一层是「谁能自己决定下一步」的地方：上接 HTTP，下用 L3 的领域能力，自己不管数据形状。

### L2-1 运行时内核（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 主循环 | `app/agent/runtime.py` | 移植 `agents/index.ts`（745 行），S5 ③ |
| 工具基座 | `app/agent/tool.py` | 工具的注册、参数校验、结果封装 |
| 上下文预算与压缩 | `app/agent/context_budget.py` | 零依赖；超预算时的裁剪策略 |
| 输出协议 | `app/agent/protocol.py` | 移植 `agents/protocol.ts`（79 行）——**唯一权威**（services 侧那份分叉副本已于 2026-10-30 删除） |

### L2-2 工具集（6，`app/agent/tools/`）

工具是「Agent 能做的动作」，全部是纯函数式封装，不持有状态。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 语料检索 | `app/agent/tools/corpus_tools.py` | 找相似片段/成法引用 |
| 角色·场景·物品提取 | `app/agent/tools/extract_tools.py` | 445 行，六大工具之一 |
| 剧本改写 | `app/agent/tools/script_tools.py` | 83 行 |
| 分镜拆解 | `app/agent/tools/storyboard_tools.py` | 517 行，体量最大 |
| 宫格图提示词 | `app/agent/tools/grid_prompt_tools.py` | 92 行 |
| 角色音色分配 | `app/agent/tools/voice_tools.py` | 148 行 |

### L2-3 上层编排（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 实例创建 | `app/agent/creator.py` | 按 agent_type 装配工具与 skill |
| 技能注入 | `app/agent/skills.py` | 默认绑定由 SKILL.md frontmatter 自述；`list_skill_ids` 等为全仓唯一实现 |
| 子 Agent 调度 | `app/agent/subagent.py` | 整域关闭（Node 侧已停用） |
| 全自动管线 | `app/agent/auto_pipeline.py` | 移植 `services/auto-pipeline.ts`（814 行） |
| 审片重跑闭环 | `app/agent/qc_retry.py` | 移植 `services/qc-retry.ts`（175 行） |

### L2-4 评测闭环（7，`app/agent/evaluation/`）

一条独立的「测 → 评 → 改」小流水线，与运行时解耦。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 数据契约 | `app/agent/evaluation/types.py` | 151 行 |
| 基准 case 目录 | `app/agent/evaluation/catalog.py` | 单一来源 |
| 执行器 | `app/agent/evaluation/evaluator.py` | 210 行 |
| 确定性评分 | `app/agent/evaluation/scorer.py` | 327 行，**纯函数，不依赖 LLM/DB** |
| 提示词优化器 | `app/agent/evaluation/optimizer.py` | 216 行，状态机闭环 |
| 无人值守调度 | `app/agent/evaluation_scheduler.py` | 移植 `evaluation-scheduler.ts`（126 行） |
| 命令行入口 | `app/agent/evaluation/cli.py` | 64 行 |

### L2-5 外接协议与技能库（1 + 10 个 skill 目录）

| 功能点 | 落点 | 说明 |
|---|---|---|
| MCP 客户端 | `app/mcp/client.py` | 远端工具包成本地 `Tool`；只依赖 `core` |
| 技能内容（自有） | `app/skills/` | `extractor` / `script_rewriter` / `storyboard_breaker` / `voice_assigner` / `grid_prompt_generator` / `production-tools` 等，靠 `SKILL.md` 自述绑定 |
| 技能内容（外部库） | `app/skills/*/library.yaml` | 库由声明文件识别，库内 skill **不默认注入** |

`app/skills/` 里的 `.py` **不属于应用分层**：它们是技能自带的命令行脚本，由 `SKILL.md` 指明何时调用，
后端进程不 import 它们（改它们不需要跑 `layering_test`）。逐份点名（㊆ 要求覆盖）：

| 脚本 | 落点 | 说明 |
|---|---|---|
| 节拍检测 | `app/skills/production-tools/beat-sync-editor/scripts/beat_detect.py` | librosa 打拍点 |
| 能量/情绪峰值 | `app/skills/production-tools/beat-sync-editor/scripts/energy_analyze.py` | 找高张力片段与裁剪窗口 |
| 字幕校验与纠错 | `app/skills/production-tools/subtitle-correction/scripts/subtitle_tool.py` | 对照原稿分词级差异 |

## 4. L3 业务服务层（`app/services/*.py` 75 模块，12 个域）

L3 是本仓库的主体：**领域知识全在这里**。判据是「换个 HTTP 框架、换个 Agent 循环，这段逻辑还成立吗」——
成立就属 L3。

### L3-1 剧本要素与连续性（8）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 剧本指纹门禁 | `app/services/script_fingerprint.py` | 改剧本即失效下游签名 |
| 台词说话人匹配 | `app/services/character_match.py` | 移植 `shared/character-match.ts` |
| 场景地点/时段匹配 | `app/services/scene_match.py` | 提取去重与陈旧关联清理共用 |
| 连续性表 | `app/services/continuity.py` | 移植 `production-plan-contract` |
| 连续性写入 | `app/services/continuity_store.py` | 校验 + 幂等替换 |
| 分镜 → 连续性适配 | `app/services/storyboard_continuity.py` | 零依赖 |
| 分镜关联同步 | `app/services/storyboard_helpers.py` | 含台词解析与 TTS 匹配校验 |
| 视觉图谱 | `app/services/visual_graph.py` | 角色/场景/道具的图结构 |

### L3-2 提示词与镜头语言（7）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 可复用提示词块 | `app/services/prompt_blocks.py` | 107 行 |
| 五段式质感层 | `app/services/prompt_polish.py` | Mx-Shell 五段式，零依赖 |
| 提示词工具 | `app/services/prompt_utils.py` | 分块移植 `shared/prompt-utils.ts` |
| 占位符解析 | `app/services/shot_placeholders.py` | `<location>L1</location>` → 具体内容 |
| 运镜指导 | `app/services/camera_movement_guides.py` | 运镜 → 首/尾帧构图 |
| 多集节奏相位 | `app/services/rhythm_phase.py` | **整域关闭**（Node 侧停用，实现保留） |
| take 预算 | `app/services/take_budget.py` | per-shot 重试预算 |

### L3-3 设定集与风格（3）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Bible 六键 ID | `app/services/bible_ids.py` | 移植 `services/bible-ids.ts` |
| 时代背景 | `app/services/era_background.py` | 13 行级小服务 |
| 风格档案 | `app/services/style_profiles.py` | CRUD + 提炼 |

### L3-4 生成链路（10）

四条生成线（图/视频/文/音）＋ 三个入口前处理。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 图片生成 | `app/services/image_generation.py` | 75 行级编排 |
| 视频生成 | `app/services/video_generation.py` | 495 行，最大的一条 |
| 文本生成 | `app/services/text_generation.py` | 556 行，可走 Ollama 或自研引擎 |
| TTS 合成 | `app/services/tts_generation.py` | 140 行级 |
| 音色快速复刻 | `app/services/voice_clone.py` | 135 行 |
| 配音情绪/语速契约 | `app/services/voice_contract.py` | 口径来自 IndexTTS-2.5 节点面 |
| 段级音频合成 | `app/services/segment_audio.py` | 模型声 + 配音 → 一条音轨 |
| 逐镜路由 | `app/services/shot_router.py` | 决定每镜走哪条线 |
| 宫格图切分 | `app/services/grid_split.py` | 45 行 |
| 超长输入保护 | `app/services/text_slice.py` | 39 行 |

### L3-5 后期与交付（6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 单镜头合成 | `app/services/ffmpeg_compose.py` | FFmpeg 单镜 |
| 多镜头拼接 | `app/services/ffmpeg_merge.py` | FFmpeg 多镜 |
| 尾帧提取 | `app/services/frame_extractor.py` | 连续性状态机 v3 的衔接链路 |
| 校色 | `app/services/color_grade.py` | **整域关闭** |
| 剪映草稿导出 | `app/services/jianying_draft.py` | 339 行 |
| 导出服务 | `app/services/export_service.py` | 192 行级 |

### L3-6 质检（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 图像连续性 QC | `app/services/consistency_qc.py` | 移植 `consistency-qc.ts` |
| 技术维度 QC | `app/services/technical_qc.py` | 281 行 |
| 镜头级打分 | `app/services/qc_scoring.py` | 272 行 |
| QC 报告交付物 | `app/services/qc_report.py` | 515 行 |

### L3-7 模型与设备治理（9）

| 功能点 | 落点 | 说明 |
|---|---|---|
| GPU 显存管理 | `app/services/gpu_manager.py` | 473 行，云端/本地双模 |
| 引擎就绪体检 | `app/services/engine_readiness.py` | 四块串成一次调用 |
| 本地模型扫描 | `app/services/local_model_scan.py` | 709 行 |
| 模型生态落点 | `app/services/model_ecosystems.py` | 各族模型在本机落在哪 |
| 权重指纹 | `app/services/model_fingerprint.py` | **读文件头判架构**，不加载权重、不需要 torch |
| 模型目录类别 | `app/services/model_folders.py` | ComfyUI 口径 + 任意目录声明解析 |
| 运行时体检 | `app/services/runtime_doctor.py` | 「这条流水线现在能不能真跑」（2026-10-30） |
| 厂商连通性探测 | `app/services/provider_probe.py` | URL 拼接 + 探测描述 |
| 视频时长探测 | `app/services/video_probe.py` | 15 行 |

### L3-8 开跑前体检（2）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 体检取数（DB 侧） | `app/services/preflight_source.py` | 按 episodeId 组装输入 |
| 体检纯逻辑 | `app/services/production_preflight.py` | 零依赖，四块串一条 |

### L3-9 外部执行通道（10）

「把活派出去」的三种通道：ComfyUI、Ollama、云端供应商。

| 功能点 | 落点 | 说明 |
|---|---|---|
| ComfyUI 工作流运行 | `app/services/comfyui.py` | 项目自己的一等公民能力 |
| ComfyUI 渲染目标 | `app/services/comfyui_targets.py` | 图片记录 → target 的单一真源 |
| Ollama 辅助 | `app/services/ollama.py` | 小工具集合 |
| 模型清单合成 | `app/services/ollama_catalog.py` | 盘上清单 × 服务端能力 |
| Ollama 客户端 | `app/services/ollama_client.py` | 有类型的 `/api/*` 接口 |
| Ollama 离线库读取 | `app/services/ollama_store.py` | 不依赖 `ollama serve` |
| 视频识别 | `app/services/ollama_video.py` | 抽帧 + 视觉模型逐帧读 |
| 云端供应商解析 | `app/services/ai_configs.py` | 纯 DB 部分 |
| 供应商路由 | `app/services/ai_providers.py` | 移植 `services/ai.ts` 整文件 |
| 厂商错误映射 | `app/services/vendor_errors.py` | 统一成中文可读 |

### L3-10 成本与台账（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 成本单价目录 | `app/services/cost_catalog.py` | 移植 `cost-catalog.ts` |
| 生成前预估 | `app/services/estimate_service.py` | 开跑前给价 |
| 用量看板 | `app/services/usage_tracking.py` | 实际花费统计 |
| 工程账本 | `app/services/project_ledger.py` | 纯 DB |

### L3-11 存储与资产（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 数据存储 | `app/services/data_storage.py` | **整域关闭** |
| 文件存储 | `app/services/file_storage.py` | 移植 `utils/storage.ts` |
| 资源库共享实现 | `app/services/resource_library.py` | 角色/场景/兵器/服装模板 |
| 资产清单 + 验收门 | `app/services/asset_manifest.py` | 缺件即阻断 |
| 资产版本 | `app/services/asset_versions.py` | 历史与回滚 |

### L3-12 可配置内容与执行机制（7）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 出厂提示词 | `app/services/agent_prompts.py` | 单一真源，runtime/creator/optimizer/routers 共用 |
| Agent 成员登记 | `app/services/agent_registry.py` | Agent 层唯一成员来源 |
| 预设框架 | `app/services/preset_framework.py` | 473 行 |
| SKILL.md 解析 | `app/services/skill_parser.py` | **唯一权威实现** |
| SSE 事件总线 | `app/services/sse_hub.py` | 57 行级 |
| 任务日志 | `app/services/task_logger.py` | 可追踪执行 |
| Trace 持久化 | `app/services/trace_store.py` | 只移植回放侧 |

> **「整域关闭」读法**：Node 侧已停用、Python 侧实现完整保留并仍被测试钉着。
> 它们不是死代码 —— 删掉会让「Node 停用」变成「能力消失」，将来重启时无从对照。见第 9 节。

## 5. L4 推理引擎层（`app/services/engine/` 63 模块 + 1 个 CLI 入口，6 组）

**这一层的存在理由**：产品要的是生成能力本身（加载 → 采样 → 解码 → 出片），不是去调别人的推理服务。
两条硬约束决定了实现方式：① **ComfyUI 是 GPL-3.0** ⇒ 一行源码都不抄，数学自己实现、脚手架只用
Apache/MIT（torch/diffusers/safetensors）；② torch 已装且有 CUDA，但**不许写死设备**，一律实测。

目录**是平的**（63 个模块不分箱）—— 这里是「按能力切」，任何一次搬家都要同改 300+ 处调用点与
6 套静态守卫。分工靠 `app/services/engine/__init__.py::MODULE_GROUPS` 登记，盘上多一个没登记的模块即 FAIL。

### L4-1 `numeric` 零依赖数学与规划（11）

唯一允许「不装 torch 也能测」的一组；不 import torch、不读文件。

| 功能点 | 落点 |
|---|---|
| 噪声调度（sigma 序列） | `app/services/engine/schedules.py` |
| 几何数学（分辨率/帧数/潜空间形状） | `app/services/engine/geometry.py` |
| CFG 引导 + 重标定 + 步区间 | `app/services/engine/guidance.py` |
| 采样器（扩散/流匹配去噪循环） | `app/services/engine/sampler.py` |
| 长视频分段计划 | `app/services/engine/segments.py` |
| 首/尾帧下标规划 | `app/services/engine/keyframes.py` |
| 条件掩码（首帧/参考图那一半） | `app/services/engine/conditioning.py` |
| 段级音频合成（纯数学） | `app/services/engine/audio_mix.py` |
| 参考素材指纹（哪段可跳过） | `app/services/engine/cache_key.py` |
| 加速链（串成配置而非代码） | `app/services/engine/accel_chain.py` |
| **VAE 分块缩放/解码**（`tile`/`overlap` 走**像素**单位＝`VAEDecodeTiled` 同义；权重累积＋除法归一 ⇒ 重叠带精确平均；`overlap<tile`、不足 1 潜像素、`tile` 过小都**报错**；只切空间、不切时间维） | `app/services/engine/vae_tiling.py` |

### L4-2 `format` 容器与像素（9）

读得进、写得出。不做模型数学。

| 功能点 | 落点 |
|---|---|
| safetensors 读取器（纯 Python） | `app/services/engine/safetensors.py` |
| GGUF 读取器（纯 Python） | `app/services/engine/gguf.py` |
| 权重内嵌元数据契约 | `app/services/engine/checkpoint_meta.py` |
| 潜空间口径表 | `app/services/engine/latent_formats.py` |
| 联合音视频潜变量互操作 | `app/services/engine/latent_container.py` |
| 落盘（帧→mp4 / 音→wav） | `app/services/engine/media.py` |
| 图像算子集（纯 torch，不引 kornia/scipy） | `app/services/engine/image_ops.py` |
| **Qwen3-VL 像素预处理**（H3 图片/视频块 ⇒ `(t,h,w)` 网格：resize 到 **32** 倍数、`(v−0.5)/0.5` 归一化、**merge 块优先**的行序、2 帧进时间维；只做像素不做模型数学 ⇒ 不挂 `model` 组） | `app/services/engine/qwen3vl_pixels.py` |
| 缓存/权重文件完整性守卫 | `app/services/engine/cache_guard.py` |

### L4-3 `weight` 量化与装载（11）

「文件里的字节 ⇒ 能算的张量」。不决定跑哪一步。

| 功能点 | 落点 |
|---|---|
| 反量化（fp8/int8/int4） | `app/services/engine/quant.py` |
| NVFP4（E2M1 + e4m3 块 scale） | `app/services/engine/quant_nvfp4.py` |
| 权重装载进 `nn.Module` | `app/services/engine/weights.py` |
| 跨族张量名映射 | `app/services/engine/mappings.py` |
| 混合加载（fl2va 基底 + ref2va adaLN） | `app/services/engine/hybrid_load.py` |
| 混合加载合并 | `app/services/engine/hybrid_merge.py` |
| 档位表（步数/分辨率/加速件/显存建议） | `app/services/engine/tiers.py` |
| LoRA 装配 | `app/services/engine/lora.py` |
| LoRA 内容身份 + 引用计数 | `app/services/engine/lora_cache.py` |
| **无缓冲重叠读**（Windows：`FILE_FLAG_NO_BUFFERING\|OVERLAPPED` ＋ 每槽手动复位事件 ＋ 扇区对齐槽区用 `slop` 吃掉非对齐偏移；非 Windows/`force_plain` 退回普通读，降级原因写进 `describe()`；越界当场拒。**已接进 `safetensors.TensorReader`**：Windows 默认走它、建不起来退回 seek+read 并报 `ioNote`；真权重 11.46 GiB/453 张量实测 2.88 ⇒ 3.43 GiB/s，逐字节相同） | `app/services/engine/win_io.py` |
| **预取单槽**（提交永不抛、取走即作废、槽满当场拒 `busy`、`snapshot()` 报在飞/就绪/命中/错误。**已接进装载**：`weights.load_module_weights(read_ahead=2)` 边建当前张量边让后台读后面两件，作用域由 `_prefetch_scope` 管、出函数 `close(wait=True)`；单件超 `_READ_AHEAD_MAX_BYTES`(256 MiB) 一律就地读；2 GiB/64 件实测 2.7 ⇒ 3.9 GiB/s） | `app/services/engine/prefetch.py` |
| **任务分配与压力调度**（零 torch 纯函数：**GPU 优先** ⇒ 装不下就 `gpuBytes+cpuBytes` **正好等于**需求的协同切分 ⇒ 两边都不够就 `InsufficientResources` 拒跑（不靠交换文件硬跑）；协同后**第二台设备也要核**（`host_memory`/`host_room_bytes`/`ensure_host_room`，无 `psutil` 依赖）；`Ledger` 记「谁在占、占多少」（卡/内存两栏，**不是判据**）；`LoadWatch` 三档滞回（88%/95%、连续 2 次才认 `hard`）只管**回收**不搬权重；`reclaim_for` 是纯函数（阶段表没登记 ⇒ 一个都不退）） | `app/services/engine/scheduler.py` |
| **GPU + CPU 联合驻留**（按显存预算切连续前缀层、`split` 搬激活 / `park` 搬权重、无卡/null 如实说） | `app/services/engine/residency.py` |
| **设备策略**（默认 GPU 优先 ✓；每次装载前现测显存 ⇒ 够就上卡 / 差一点改联合 / 差太多转 CPU；`auto`·`gpu`·`cpu`·`balanced` 四档，判据只有显存占用率、`utilPercent` 留 nvml 注入位） | `app/services/engine/device_policy.py` |
| **四条装配线的装配前决策**（2.1 按分时账 13.26 GiB / Z-Image 按整链同驻账 19.28 GiB / **Qwen-Image 2512 按本族分时账 38.05 GiB** / SDXL 按盘上文件真字节；落不了 `split` 的线按「装不下」整份转 CPU，但「装得下只是没摊开」不算降级） | `runtime.py::_chain_device_decision` |
| **策略的可见面**（`capabilities().devicePolicy` 报四档＋判定规则＋**三本账**（`part` 2.1 / `chain` Z-Image / `qwenImagePart` 2512 —— 族不同但键名相同的账**不能并**）、只报静态事实不读卡况；**生效账**另有 `GET /engine/runtime` 的 `devicePolicy.partNeedGiB`（声明链后就是本族那本，同时留 `defaultPartNeedGiB`）；`POST /load` 收 `policy` 三写法、拼错当场 400、回执两格 `devicePolicy`(现在状态) 与 `chainDevicePolicy`(这次为何在 CPU)；`SCHEMA_VERSION` 4） | `facade.py::capabilities` / `torch_backend.py::describe` / `routers/engine.py::load_runtime` |

### L4-4 `token` 文本入口（7）

prompt ⇒ token ⇒ 向量。不做图像数学。

| 功能点 | 落点 |
|---|---|
| 自研 BPE | `app/services/engine/tokenizer_bpe.py` |
| Unigram / WordPiece 扩充 | `app/services/engine/tokenizer_own.py` |
| GGUF 自带词表 → 分词器 | `app/services/engine/tokenizer_gguf.py` |
| 分词器总入口（自研优先/回退/缓存） | `app/services/engine/tokenizer_hub.py` |
| 对 transformers 的运行时改造 | `app/services/engine/tokenizers_tuning.py` |
| chat 模板（从权重元数据取） | `app/services/engine/chat_template.py` |
| **H3 呈现格式 + `token_tags`**（`fl2va`/`ref2va` 的 `<Picture i>:` / `<Video k>` / `<T.T seconds>` 交错布局；视觉块前占 N 个 `IMAGE_PAD`；视频块**每块带 `videoIndex`** ⇒ 区间是**本路内**下标（多路都从 0 起，不带就分不出是第几路）；标签段必须**逐段单独分词** ⇒ 输出段序列；`token_tags` 只把 `[index-1, index+size+1)` 标 0；纯 Python 零依赖） | `app/services/engine/h3_prompt.py` |

### L4-5 `model` 网络本体（31）

一族模型一个骨架。不编排阶段、不管显存回收。

| 功能点 | 落点 |
|---|---|
| DiT 主干 | `app/services/engine/dit.py` |
| Z-Image / Lumina 骨架（`NextDiT` 同构，已与 ComfyUI 参考实现逐层对拍；**已接进装载分支**） | `app/services/engine/dit_lumina.py` |
| Qwen-Image 2.1 骨架（单流 + SwiGLU + 三轴 RoPE，两行调制；键集合与盘上 649 张量对账、与独立参照实现双精度一致；**已接进装载/前向分支**，TE 与 VAE 未做） | `app/services/engine/dit_qwen21.py` |
| **Qwen-Image 2512 / Edit 2511 骨架**（双流 MMDiT：文本与图像各 6 组调制＋各一套 MLP；时间口径 `1000·σ` **不翻转**；`heads` 由 `attn.norm_q` 宽反推；Edit 侧参考图 token 串联＋`__index_timestep_zero__` 哨兵喂 σ=0 行调制；2512 与 Edit 2511 同一份，差别只在 `ref_latents`；**已接进装载分支**，Edit 的参考图通路未接 ⇒ `denoise` 明确拒跑） | `app/services/engine/dit_qwen_image.py` |
| 按权重键名判族 + 读拓扑 | `app/services/engine/dit_family.py` |
| **通用单流 DiT 前向**（维度从权重形状反推、键集合双向对账） | `app/services/engine/dit_generic.py` |
| H3 DiT 积木（含 **`token_tags` → adaLN 行号**：文本段按标签切成 text/video run，视觉占位 token 走 video 行；不给标签 ⇒ 整段 text 行，纯文本提示词等价；装配侧 **meta 建壳 → 降精度 → 扳 fp32 岛 → 换 int8 常驻层 → 才物化**，峰值 ≈ 19.5 GiB 而非 fp32 主干的 78 GiB） | `app/services/engine/h3_form.py` |
| H3 检查点键名核对（torch-free） | `app/services/engine/h3_keys.py` |
| H3 续拍 / 重拍 | `app/services/engine/h3_edit.py` |
| SDXL UNet | `app/services/engine/sdxl.py` |
| SDXL 出图后端闭环 | `app/services/engine/sdxl_backend.py` |
| SDXL VAE | `app/services/engine/sdxl_vae.py` |
| **Flux 族 VAE**（16 通道 / 无 `quant_conv` / 244 键；`(x−shift)/scale` 反缩放内置） | `app/services/engine/vae_flux.py` |
| **Wan 系 VAE**（先接 Qwen-Image 2.1 的 64 通道：Wan 2.2 布局 / 238 键 / 单帧解码 / 出 RGBA 要丢第 4 通道） | `app/services/engine/vae_wan.py` |
| 视频 VAE | `app/services/engine/vae.py` |
| H3 真视频 VAE（3D 因果 CNN + ViT3D 解码） | `app/services/engine/vae_h3.py` |
| H3 音频 VAE（32 kHz 立体声） | `app/services/engine/audio_vae.py` |
| 超清规划层 | `app/services/engine/upscale.py` |
| 超清网络本体 | `app/services/engine/upscale_net.py` |
| 文本编码器 | `app/services/engine/text_encoder.py` |
| CLIP 文本塔（SDXL 那一半） | `app/services/engine/clip_text.py` |
| **Qwen3 文本塔**（Z-Image 的提示词入口：36 层 / `layer_idx=-2` / ChatML 模板） | `app/services/engine/qwen3_te.py` |
| **Qwen3.5 文本塔**（Qwen-Image 2.1 的提示词入口：24 层 Gated DeltaNet ＋ 8 层 full attention 混合 / `layer_idx=-2`；已与 comfy 参考块逐位对拍 `maxDiff=0`） | `app/services/engine/qwen35_te.py` |
| **Qwen2.5-VL-7B 文本塔**（Qwen-Image 2512 / Edit 2511 的提示词入口：28 层 / 3584 / 28×128 头 / kh=4；**q/k/v 带 bias**、无 QK-norm、取**最后一层**且**不过** final norm；`system`＋`user` 两段模板按第 2 个 `<|im_start|>` 截掉系统段；只挂语言塔，同文件里的 `visual.*` 与 `lm_head` 有意不装；**已接进 `attach_qwen25_te`**） | `app/services/engine/qwen25_te.py` |
| decoder-only LLM 推理 | `app/services/engine/llm.py` |
| **Qwen3-VL-32B 语言塔**（H3 文本口的本体：interleaved mrope `ropeDims=(2,1,1)` / `len_max = grid.max()//2` / **有 QK-norm、无 final norm** / DeepStack 三份 merger 在**层之后**注入；与盘上 551 语言键逐名逐形对账） | `app/services/engine/qwen3vl_te.py` |
| **Qwen3-VL ViT 视觉塔**（27 层 / `patch_embed.proj` 2 帧时间核 / `merger` 在 **pre-shuffle**、`deepstack_merger_list` 在 **post-shuffle**；与 H3 的 TE 权重同文件、键名逐字一致） | `app/services/engine/qwen3vl_vision.py` |
| 进程内文本后端 | `app/services/engine/llm_backend.py` |
| GGUF 反量化 | `app/services/engine/gguf_dequant.py` |
| GGUF → LlmModel 装载 | `app/services/engine/gguf_to_llm.py` |
| torch 后端（设备实测，不写死） | `app/services/engine/torch_backend.py` |

### L4-6 `orchestration` 编排与可见面（11 + CLI 入口）

「谁来调度、缺什么、退到哪」。不写模型数学。

| 功能点 | 落点 |
|---|---|
| 生成管线（可观测/可取消/可归因） | `app/services/engine/pipeline.py` |
| 多段成片（分段→拼接→规范化） | `app/services/engine/chain.py` |
| 进程内运行时（装载/排队/卸载） | `app/services/engine/runtime.py` |
| 独立门面 SDK（不经 HTTP/路由） | `app/services/engine/facade.py` |
| 加载计划 | `app/services/engine/loader.py` |
| stage 注册表 + 装载前兼容闸门 | `app/services/engine/registry.py` |
| 权重体检 / 就绪报告 | `app/services/engine/inventory.py` |
| 干跑后端（无 torch 也能跑整条管线） | `app/services/engine/dryrun.py` |
| 业务链路 → 进程内引擎桥 | `app/services/engine/bridge.py` |
| **H3 文本条件编码串联**（呈现格式 → 逐段分词 → 视觉塔 → 写 embeds → 语言塔 ⇒ `H3Condition`：hidden / tags / ids / spans；`combine()` 把两座塔并成一个编码器且**不重复注册** `model.*` 键 ⇒ 与盘上权重全名空间逐字对得上；不写模型数学 ⇒ 归编排不归 `model` 组）；**装配口** ＝ `TorchBackend.attach_h3_text`（两半塔**同文件一次装**、int8 常驻、切分驻留先切再装）⇒ `_encode_h3_condition` 回执里同时给出 `hidden` 与 **`tags`**（0 = video 行 / 1 = text 行 ⇒ 采样侧 `mod_segments_for` 靠它切 run）；参考**视频**按 `videoIndex` 选路（单张量 = 单路；列表 = 一路一个，越界/缺路当场拒），每块按 2 fps 抽帧后奇数帧补末帧且**时间戳重复末值** | `app/services/engine/h3_encode.py` |
| 导演稿 → 生成段（纯规则） | `app/services/engine/script_parse.py` |
| 独立命令行（`doctor`/`plan`/`readiness`/`inspect`/`manifest`/`status`） | `app/services/engine/__main__.py` |

> `__main__.py` 是**入口脚本**，不在 `MODULE_GROUPS` 的 63 个模块里（㊀ 的登记表只管模块）。



### L4 的两处已知边界

- **未接线的 3 个模块**（已实现、只有测试在钉口径）：`image_ops`、`mappings`、`latent_formats`
  —— 它们该接到谁身上，登记在 `tests/structure_test.py::UNWIRED`，接了线就该删那条登记。
  （`dit_lumina` 2026-10-31 已接线 ⇒ 登记已删，见下一条。）
- **有骨架的族有 7 个**：H3（`h3_form` `vae_h3` `audio_vae`）、SDXL（`sdxl` `sdxl_vae`）、
  GGUF LLM（`llm` `gguf_to_llm`）、DiT 判族（`dit` `dit_family`）、**通用单流 DiT**
  （`dit_generic`：Krea2 一族已能建结构并前向，维度全部从权重形状反推）、**Z-Image**
  （`dit_lumina`：判族命中 `zimage` ⇒ `torch_backend` 按权重形状建 `LuminaDiT` 并接进前向，
  与盘上 453 张量逐键对账；**TE/VAE 与采样也已接通**，`stage=zimage` 从提示词一路到 PNG，
  并已进 `runtime.ensure_loaded` 装配口，见 10.1b）、**Qwen-Image 2512 / Edit 2511**
  （`dit_qwen_image` 双流 MMDiT ＋ `qwen25_te`：结构与前向都写了、装配线也接了
  `stage=qwen_image`，与盘上 1933 张量逐键对账；**但还差一次真权重实跑出图**，Edit 2511 的
  参考图通路明确拒跑，见 10.1c）。Flux.1-dev / Wan2.2 / SEEDVR2 目前**只有格式与量化支持，
  没有前向**。见第 9 节。

## 6. L5 本地执行与厂商适配（48 模块，4 个域）

判据：**这段代码换个模型就作废吗**？作废 ⇒ 属 L5（协议/厂商特化）；不作废 ⇒ 属 L3/L4。
L5 是唯一允许「知道某家厂商的怪癖」的地方。

### L5-1 云端厂商适配（8，`app/services/adapters/`）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 适配器契约 | `app/services/adapters/types.py` | 四个能力面的接口定义 |
| 适配器注册 | `app/services/adapters/registry.py` | 按 provider 取实现 |
| 图像适配器 | `app/services/adapters/image_adapters.py` | — |
| 视频适配器 | `app/services/adapters/video_adapters.py` | — |
| 文本适配器 | `app/services/adapters/text_adapters.py` | — |
| TTS 适配器 | `app/services/adapters/tts_adapters.py` | — |
| JS 语义兼容 | `app/services/adapters/jscompat.py` | `undefined` vs `null`、真值判定与 Node 对齐 |
| 厂商 URL 拼接 | `app/services/adapters/url.py` | 各家的 base 路径差异 |

### L5-2 Ollama 原生协议（17 + vendored 客户端 3）

不调 Ollama 的 HTTP，而是**自己实现它的协议与文件格式**（这样离线也能读、断网也能装）。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 版本协商 | `app/services/ollama_native/version.py` | — |
| 鉴权 | `app/services/ollama_native/auth.py` | — |
| 环境配置 | `app/services/ollama_native/envconfig.py` | — |
| 能力面探测 | `app/services/ollama_native/capabilities.py` | — |
| 模型清单 | `app/services/ollama_native/manifest.py` | — |
| 命名规则 | `app/services/ollama_native/names.py` | — |
| Modelfile 解析 | `app/services/ollama_native/modelfile.py` | — |
| 参数映射 | `app/services/ollama_native/params.py` | — |
| Go 模板渲染 | `app/services/ollama_native/gotemplate.py` | Ollama 用 Go template |
| 模板库 | `app/services/ollama_native/template.py` `templates/` | — |
| Harmony 格式 | `app/services/ollama_native/harmony.py` | GPT-OSS 风格控制 token |
| 思考流解析 | `app/services/ollama_native/thinking.py` | `thinking` 字段 |
| 工具调用解析 | `app/services/ollama_native/toolparse.py` | — |
| 格式化输出 | `app/services/ollama_native/format.py` | JSON schema 约束 |
| Anthropic 兼容 | `app/services/ollama_native/anthropic.py` | Messages API 形状 |
| OpenAI 兼容 | `app/services/ollama_native/openai.py` | Chat Completions 形状 |
| 下载进度 | `app/services/ollama_native/progress.py` | — |

`app/vendor/ollama/` 是**原样搬进来的官方 Python 客户端**（只读、不改造，改它等于改上游）：

| 功能点 | 落点 | 说明 |
|---|---|---|
| HTTP 客户端 | `app/vendor/ollama/_client.py` | 1430 行；`Client`/`AsyncClient`/`BaseClient`，含流式与工具调用 |
| 请求/响应类型 | `app/vendor/ollama/_types.py` | pydantic 模型：`ChatResponse`/`Message`/`Options`/`ProgressResponse` 等 |
| 工具函数 | `app/vendor/ollama/_utils.py` | 90 行；docstring 解析、Python 函数 → 工具 schema |

### L5-3 LoRA 训练执行（9 + `dataset/` 5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 训练命令拼装 | `app/services/lora_train/commands.py` | 调外部训练器 |
| 训练配置 | `app/services/lora_train/config.py` | — |
| 参数选项 | `app/services/lora_train/options.py` | — |
| 路径布局 | `app/services/lora_train/paths.py` | — |
| 执行器 | `app/services/lora_train/runner.py` | 进程生命周期 |
| 运行时探测 | `app/services/lora_train/runtime.py` | — |
| 进度上报 | `app/services/lora_train/progress.py` | — |
| 优雅退出 | `app/services/lora_train/shutdown.py` | — |
| 错误分类 | `app/services/lora_train/errors.py` | — |

`dataset/` 是**素材侧独立队列**（改名/转格式/打标都走它，与训练队列分开：训练吃 GPU 几小时，
素材多是 CPU/IO 几十秒）：

| 功能点 | 落点 | 说明 |
|---|---|---|
| 打标 | `app/services/lora_train/dataset/caption.py` | 生成/写回 caption |
| 字段定义 | `app/services/lora_train/dataset/fields.py` | 素材任务的数据形状 |
| 文件操作 | `app/services/lora_train/dataset/files.py` | 改名/复制/转格式 |
| 提示词模板 | `app/services/lora_train/dataset/prompts.py` | 打标用的 prompt 与选项映射 |
| 素材运行时 | `app/services/lora_train/dataset/runtime.py` | 串行排队、日志可增量取、可取消 |

### L5-4 本地原生服务（6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| H3 薄封装服务 | `app/local_services/h3/server.py` | 端口 8765；`/v1/video_generation` 任务形状 + ComfyUI 能力门面 |
| ComfyUI HTTP 客户端 | `app/local_services/h3/comfyui_client.py` | 零第三方依赖（只用 `urllib`）；`/prompt` `/history` `/view` `/free` |
| UI 图 → API 工作流 | `app/local_services/h3/workflow.py` | 两种格式互不通用，这里做转换与参数注入（纯函数） |
| H3 prompt 契约 | `app/local_services/h3/prompt.py` | 全局提示词拼接 + `@图1`/`【图2】` 标签化 + 提交前核编号 |
| 加速链接线 | `app/local_services/h3/accel.py` | 用 ComfyUI 真实注册表逐项校验；连不上判 `unchecked`，不判错 |
| TTS 薄封装服务 | `app/local_services/tts/server.py` | 端口 9880；对外只认 `POST /tts` 一套协议，对内用 `TTS_ENGINE` 换引擎（`edge` 等） |

## 7. L6 平台层（`app/core/` 6 模块，3 个域）

最底层：谁都能依赖它，它不依赖任何业务。`core` 只允许 import `core`。

### L6-1 配置与路径（1）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 全部路径与环境变量真源 | `app/core/config.py` | `skills_dir()` / 数据目录 / 锁文件位置等，**别处不许自己拼** |

### L6-2 数据与模型表（2）

| 功能点 | 落点 | 说明 |
|---|---|---|
| SQLite 连接与迁移 | `app/core/db.py` | 所有表结构的唯一来源 |
| ORM/表模型 | `app/core/models.py` | 与 Node 侧 `schema.ts` 对齐 |

### L6-3 请求、响应与资源（3）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 统一响应形状 | `app/core/response.py` | `success/bad_request/internal_error` + `js_truthy` |
| 请求工具 | `app/core/request_utils.py` | 分页/过滤/幂等键 |
| CPU 预算 | `app/core/cpu_budget.py` | 并发度与线程数控制 |

## 8. L7 开发期工具（`app/scripts/` 21 = 顶层 15 + `corpus/` 6，4 个域；另有 `tests/` 166 套）

**不进运行时**。判据：删了它，线上功能一条都不少；但删了它，下次事故就没人拦。

### L7-1 体检与守卫（6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 一键全检 | `app/scripts/check_all.py` | 4 道自检串成一条 |
| CLI 编码守卫 | `app/scripts/check_cli_encoding.py` | 裸跑 GBK 控制台会崩的套件 |
| 内存巡检 | `app/scripts/check_memory.py` | — |
| 技能引用守卫 | `app/scripts/check_skill_refs.py` | 刻意保留独立实现（脚本不 import `app.*`） |
| 测试自检 | `app/scripts/test_guards.py` | 守卫本身的反验 |
| 资源观察 | `app/scripts/resource_watch.py` | — |

### L7-2 引擎与模型（8）

| 功能点 | 落点 | 说明 |
|---|---|---|
| H3 就绪体检 | `app/scripts/h3_readiness.py` | — |
| H3 安装 | `app/scripts/h3_install.py` | — |
| DiT 判族清单 | `app/scripts/dit_families.py` | 读权重键名给出族表 |
| 分词器基准 | `app/scripts/tokenizer_bench.py` | 吞吐/一致性对比 |
| 模型管理 | `app/scripts/model_manager.py` | 清单同步 |
| 模型迁移 | `app/scripts/migrate_models.py` | 目录搬家 |
| SD-H3 兼容探针 | `app/scripts/sd_h3_compat_probe.py` | — |
| SD-H3 管线 | `app/scripts/sd_h3_pipeline.py` | — |

### L7-3 数据库与语料（1 + 语料 6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 库升级 | `app/scripts/db_upgrade.py` | 迁移执行器 |

`app/scripts/corpus/` 是**外部 prompt 语料的四步流水线**（采集 → 归一 → 检索 → 统计，各步可单独重跑）：

| 功能点 | 落点 | 说明 |
|---|---|---|
| 采集 | `app/scripts/corpus/fetch_raw.py` | 只下载原始字节并记账，不解析（网络最易波动，必须能重跑） |
| 归一化 | `app/scripts/corpus/normalize.py` | 各源 → 统一 JSONL；只抽可检索的最小字段集 |
| 检索 | `app/scripts/corpus/search.py` | 「找相似镜头」最小闭环；全量读内存打分，暂不上向量库 |
| 统计（主） | `app/scripts/corpus/analyze.py` | 读 `metadata.jsonl` 出分布 |
| 统计（i18n/spec 深挖） | `app/scripts/corpus/analyze2.py` | 保留原始对象，含首条样本 |
| 统计（第三视角） | `app/scripts/corpus/analyze3.py` | 同一语料的另一层统计 |

### L7-4 测试与文档（`tests/` 166 套 + `docs/`）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 全量回归 / 权威套件清单 | `tests/run_all.py` | 新套件必须登记进 `TESTS`（漏登记由 ㊇ 拦下） |
| 结构守卫（8 类事故） | `tests/structure_test.py` | 登记表/`__all__`/零依赖/模块头/空壳/路径腐烂/**接线状态**/**地图覆盖**/**清单完整** |
| 分层守卫 | `tests/layering_test.py` | `top → routers → agent → services → core` 单向 |
| 引擎数值核对 | `tests/engine_*.py` | 每族一个数值基线 |
| 功能分类地图 | `docs/feature-map.md` | 本文 |

## 9. 制作主线（把七层按「一部剧怎么出来的」串一遍）

分层是**实现视角**；这一节是**使用视角**。同一步在不同层都有落点，运行时它们串成一条链。

| # | 制作步骤 | 主责层 | 关键落点 |
|---|---|---|---|
| 1 | 建项目 / 导入原文 | L1 | `app/routers/dramas.py`、`app/routers/episodes.py` |
| 2 | 拆要素（角色/场景/物品/世界观/情节/关系/对话） | L2 → L3 | `app/agent/tools/extract_tools.py` → `app/services/character_match.py`、`app/services/scene_match.py` |
| 3 | 写/改剧本 | L2 | `app/agent/creator.py`、`app/agent/tools/script_tools.py`、`app/services/script_fingerprint.py` |
| 4 | 资产设定（定妆板/场景卡/道具板） | L3 | `app/services/asset_manifest.py` → `app/services/image_generation.py` |
| 5 | 拆分镜 | L2 → L3 | `app/agent/tools/storyboard_tools.py` → `app/services/storyboard_helpers.py` |
| 6 | 连续性对表 | L3 | `app/services/continuity.py`、`app/services/storyboard_continuity.py` |
| 7 | 提示词（含占位符/质感层/运镜） | L3 | `app/services/shot_placeholders.py`、`app/services/prompt_polish.py`、`app/services/camera_movement_guides.py` |
| 8 | 逐镜路由（选线/选模型） | L3 | `app/services/shot_router.py` |
| 9 | 出图 / 出片 | L3 → L5 → L4 | `app/services/image_generation.py`、`app/services/video_generation.py` → `app/services/comfyui.py` / `app/services/ollama_client.py` → `app/services/engine/runtime.py` |
| 10 | 声音（复刻/契约/段级合成） | L3 | `app/services/voice_clone.py`、`app/services/voice_contract.py`、`app/services/segment_audio.py` |
| 11 | 质检 + 重跑 | L3 → L2 | `app/services/qc_scoring.py`、`app/services/technical_qc.py` → `app/agent/qc_retry.py` |
| 12 | 合成 / 拼接 / 导出 | L3 | `app/services/ffmpeg_compose.py`、`app/services/ffmpeg_merge.py`、`app/services/jianying_draft.py`、`app/services/export_service.py` |
| 13 | 成本与台账 | L3 | `app/services/estimate_service.py`、`app/services/usage_tracking.py`、`app/services/project_ledger.py` |
| 14 | 全程可观测 | L1 → L3 | `app/services/sse_hub.py`、`app/services/task_logger.py`、`app/services/trace_store.py` |

## 10. 已知边界（三类，别把它们的落点当「已完成」）

### 10.1 已实现但未接线（3）

`image_ops` / `mappings` / `latent_formats` —— 实现齐、口径被测试钉死，但生产代码 0 处 import。
危险不在占地方，在**看起来像已经能用**。登记在 `tests/structure_test.py::UNWIRED`，接了线必须删登记
（㊅ 判据双向检查）。

### 10.1b Z-Image 端到端**已闭合**（2026-10-31 实测，`stage=zimage`）

「提示词 → PNG」整条链在自研引擎里跑通，**不再依赖 ComfyUI**：

| 环节 | 落点 | 实测 |
|---|---|---|
| 提示词 | `qwen3_te.py` + `torch_backend.attach_qwen3_te()` | `qwen_3_4b`（7.49 GiB / 398 键）⇒ `(1,23,2560)` bf16，0.68 秒 |
| 主干 | `dit_lumina.py` + `_build_zimage_module()` | `z_image_turbo_bf16`（11.46 GiB / 453 键）⇒ 8 步 7.1 秒（≈0.89 秒/步） |
| 解码 | `vae_flux.py` + `attach_flux_vae()` | `vae/ae.safetensors`（244 键）⇒ 1024×1024，4.9 秒 |
| 出图 | `media.write_image()` 4 维分支 | `image_seed42.png` 1.0 MB，**同 seed 逐字节可复现**（sha256 相同） |

**整图 12.9 秒**（1024×1024 / 8 步 / cfg=1 无 CFG ⇒ `guidanceSteps=0`）。

采样口径四条，缺一条就是「形状全对、不报错、图不对」（判据分散在
`tests/engine_zimage_test.py` 与 `tests/engine_dit_lumina_test.py`）：

1. σ 必须是 flow 那套（**恒在 [0,1]**，shift=3 ⇒ `σ=3t/(1+2t)`）；拿 karras（σ 到 14.6）顶上
   ⇒ `1−14.6` 是个大负数 ⇒ 图是全噪。
2. 喂给主干的 `timestep` **就是 σ**（`multiplier=1`，**不乘 1000**）；主干内部才做
   `t=(1−σ)×time_scale(1000)`（`comfy/ldm/lumina/model.py` 的 `t = 1.0 - timesteps`）。
3. **caption 不填位**（`pad_tokens=0`）：参考是 `NextDiT(pad_tokens_multiple=None)`，而
   `cap_mask = None` ⇒ 填进去的 token **不遮罩、直接污染注意力**。那两个 `*_pad_token` 参数
   权重里**有**（必须建，否则 453 键对不上）、但**从不使用**——「建」与「用」是两件事，
   混成一格就是「多填 31 个 token，形状对、结果偏」（`LuminaShape.has_pad_tokens` vs `pad_tokens`）。
4. 解码前把**模型空间**换回 **VAE 空间**：`x / 0.3611 + 0.1159`（= `Flux.process_out`，
   与 `samplers.py::inner_sample` 的 `process_latent_out(samples)` 配对）。起点噪声则是**标准正态、
   不缩放**（参考 `inner_sample` 只在有图输入时才 `process_latent_in`，注释写着
   "Don't shift the empty latent image."）。

⚠️ 第 4 条曾写成 `(x − 0.1159) / 0.3611`：在 **SDXL 上恒等价**（shift=0）⇒ 在 SDXL 上永远试不出来，
换到 Flux 族就是常数偏移 ≈0.437 —— 不报错、只是整体偏色。修正后同 seed 出图
`mean 0.525 / std 0.267 / 无死黑死白`（修前 `mean 0.627`，且被 clamp 掩盖）。

**踩到的两个 dtype 坑**（都是「报错指向 `nn.Linear`/`conv`，读起来像权重坏了」那类）：
潜变量有意保持 f32（更准），进 `x_embedder` 前要在模型内 cast 到权重精度（`dit_lumina.forward`）；
时间向量的正弦嵌入也要在 f32 里算完再 cast（`TimestepEmbedder.forward`），
否则 bf16 算 `cos/sin` 会把相邻时间步糊成同一个数；
VAE 解码同理，反缩放留在原精度、到 `decode` 门口才对齐（`sdxl_vae.decode_latents`）。

**已接进产品入口**（2026-10-31 收尾）：`runtime.ensure_loaded(stage="zimage", …)` 现在**真装**
（`_ensure_zimage_loaded` 单独一条装配线），四格全要：

| 口 | 要什么 | 缺了怎样 |
|---|---|---|
| `ditPath` | `unet/z_image_turbo_bf16.safetensors`（453 键） | 拒，并说明它不是 Z-Image 时判成了哪族 |
| `textWeights` | `text_encoders/qwen_3_4b.safetensors`（398 键，**不是** `qwen_3_vl_32b`） | 拒，点名这一格 |
| `imageVaeWeights` | `vae/ae.safetensors`（244 键，**是 Flux 的 ae**） | 拒，点名这一格 |
| `tokenizerPath` | `tokenizers/qwen3vl_32b/`（目录，含 `tokenizer.json`） | 拒，点名这一格 |

三处「不说清就会白折腾」的判据：① 缺格拒的**首行只列真缺的那几格**（第二行起才是四格说明）；
② `load_weights` 那条**显式核族**（判出来不是 `zimage` ⇒ 拒）——`dit_generic` 会把 Krea2 的权重
也「装得进去」，配 Qwen3 TE + Flux VAE 出来是噪声；③ `textWeights`/`imageVaeWeights` **进装载键**，
换 TE 或换 VAE 即换模型，不复用上一套。

实测（同 prompt / seed 42）：装配 **40.5 秒**（三份真权重，`reserved 19.39 GiB` / 卡上 22.49
⇒ 余 1.77 GiB，比 1024² 那档刚够）、出图 **13.1 秒**、PNG **1096471 字节**——与先前「手动
`load_weights`+`attach_*`」那条路的产物**逐字节相同**（两条装配路径等价，不是各出一版）；
二次同参调用 `reused=True`。四格传法同时认驼峰与下划线（`textWeights`/`text_weights`），
`bridge.assembly_from_config` 也转手（**只转手、不补默认**：盘上同名的 `qwen_image_vae` 不是它）。

**清单与就绪档也接上了**（同批收尾）：`configs/models.json` 补 `dit_zimage_turbo` /
`te_zimage_qwen3` / `vae_flux_ae` 三条（`tags` 含 `zimage`，`required: false` —— 理由见下），
`STAGE_FILTERS` 加 `"zimage": {"category": "image", "tags": ["zimage"], "requireAll": True}`，
`registry.STAGES` 同步注册（那张表与筛选器**双向核**，只加一边必红）。于是：

- `readiness("zimage")` = DiT 11.46 + TE 7.49 + VAE 0.31 = **19.27 GiB**，三件全必需 ——
  与实测 `reserved 19.39 GiB` 对得上，这是「这条链装不装得下」的唯一真数；
- `readiness("image")`（SDXL）的必需权重仍是 **6.46 GiB**，**不含**那 19.27 —— ⚠️ 这是本轮最容易
  做错的地方：若三条直接标 `required: true`，两套**互斥备选**会被加成 25.73 GiB，24 GB 卡上
  凭空报「装不下」，而没有任何一条链用得上这个数。所以必需性归 `zimage` 档的 `requireAll`
  （只改筛选出的**副本**，不动清单、不串档）；
- `capabilities().stages` 多一个 `zimage`（`docs/env/*.json` 两份落盘事实同步：`stages` 与
  `catalog.models` 16→19）。

两条 Z-Image 件**不写 `repo`/`file_path`**：`model_manager` 会拿这两个字段拼下载链接，而这两份在
共享池里没有已核实的官方直链 ⇒ 写了就是造一个 404（它俩在盘上，安装器走 `find_elsewhere` 直接跳过）。

⚠️ 同批还抓出并修了一个**真缺陷**：条目筛选原本在 `readiness` 与 `loader.plan_stage` 里**各写了一遍**
`category == rule["category"]` —— 加 `zimage` 档（靠 `tags` 与 SDXL 区分）当场让它露馅：

```
plan_stage("zimage")  → 列出 sdxl_base + 三条 Z-Image 件（不认 tags）
                      → 而三条的 required 是 false
                      → ready = all([]) = True     ← 三份权重一份都没有，它说"就绪"
```

比之前修过的「必需组件全缺却 `fits: true`」更坏（那个至少 `fits` 是错的、`ready` 还是对的）。修法不是让
`plan_stage` 再抄一遍 `tags` 判断（那只是把两份漂移变成三份），而是把口径抽成单一真源
`inventory.stage_entries(stage)`（`readiness` 与 `plan_stage` 都调它），同时把 `required` 一路透到
`plan.components[].required`（`docs/engine-api.schema.json` 的快照同步加这一栏）——不然消费者看不出
哪些件是必需的，而「必需件在不在」正是 `ready` 的全部依据。

反验（两条都成立）：把 `requireAll` 去掉 ⇒ 三件不再算必需，`⑦″` 红；把三条直接标 `required: true`
⇒ `readiness("image")` 变 25.73 GiB，`⑦‴` 红；把 `stage_entries` 的 filename 换成不存在的 ⇒
`plan_stage("zimage").ready` 变 `False`、`residency.missing` 列出三件。

### 10.1c Qwen-Image **2512 / Edit 2511**：骨架、前向与装配线已接，**还差一次真权重实跑**

`stage="qwen_image"` 是第四张出图档（与 `image` / `zimage` / `qwen_image21` 并列）。它和 2.1
**族名只差一个 `21`**（`qwen_image` vs `qwen_image21`），但五格事实件件不同 —— 双流 MMDiT vs 单流、
`in_channels=16` vs 64、时间口径 `1000·σ` **不翻转** vs lumina 式翻转、flow shift **3.1** vs 0.69、
文本塔 Qwen2.5-VL-7B（3584 / 取最后一层）vs Qwen3.5-9B（4096 / 取 −2）。**所以族名比较一律 `==`，
`startswith` 会让 2.1 被 2512 的 plan 接走**（shift 3.1 套在 649 键单流 DiT 上：形状全对、日志全对、
图僵）。

装配口要四格，缺一格当场拒、且首行只列真缺的那几格：

| 口 | 要什么 | 缺了怎样 |
|---|---|---|
| `ditPath` | `qwen_image_2512_bf16`（1933 键 / 38.05 GiB） | 无输出 |
| `textWeights` | `qwen_2.5_vl_7b_fp8_scaled`（语言塔 1108 键；`visual.*` 714 与 `lm_head` 单列不装） | 无提示词 |
| `tokenizerPath` | Qwen2.5 分词器目录 | ChatML 拼不出来 |
| `imageVaeWeights` | `qwen_image_vae`（16 通道） | 无解码 |

出图口径（`_fill_qwen_image_plan`）：默认 0.65 MP 的**视频口径换成官方 1.76 MP（1328²）并如实告警**、
尺寸吸附 **16**（8× 之后潜边长还要能被 2×2 拍平整除 ⇒ 只吸 8 会得到奇数潜边，`img_in` 拍平当场形状错）、
`frames=1` / `fps=0`、`timesteps` **就是 σ**、cfg=1.0 时只报不改（官方档 4.0 / 50 步）。

**三个按族参数**是本档与 2.1 共用 `enter_phase` 的解法（写成全局常量必错一族）：

- `declare_chain(text_kind="qwen25")` ⇒ 文本阶段接 `attach_qwen25_te`（缺省 `qwen35` 是 2.1 的，
  接错报的是「缺 `linear_attn.*`」，读起来像权重下错了）；
- `declare_chain(vae_family="wan21")` ⇒ 解码阶段按 16 通道装 VAE；反缩放是**恒等**
  （`latent_formats.Wan21` 没覆写 `scale_factor`/`shift_factor`），已按族显式登记在
  `vae_wan.latents_stats` 的 `wan21` 一行 —— 这 16 个 0/1 是**族事实**，不是「取不到就回退成恒等」；
- `part_need_gib=TorchBackend.QWEN_IMAGE_PART_NEED_GIB`（38.05 / 16.1 / 0.24）。**账不许混**：
  拿 2.1 的 13.26 判 38.05 ⇒ 结论"装得下"、`residency` 留 `off`、装到 20 GiB 才 OOM。以本机
  22.49 GiB 卡为例：2.1 判「整件上卡不降级」，2512 必须判 `split`（余量连 0.5 GiB 都没有时才整份转 CPU）。
  这本账在**三个可见面**上都得是生效值：`capabilities().devicePolicy.needGiB.qwenImagePart`（静态，
  单开一格，不能并进 `part` —— 两族键名相同会互相覆盖）、`describe().devicePolicy.partNeedGiB`
  （声明链后就是 38.05，类缺省另存 `defaultPartNeedGiB`）、以及 `declare_chain` 回执的 `partNeedGiB`/`peakGiB`。

清单与就绪档：`configs/models.json` 三条（`dit_qwen_image_2512` / `te_qwen25_vl_7b` / `vae_qwen_image`，
`tags` 含 `qwen2512`、`required: false`），`STAGE_FILTERS["qwen_image"]` 用 `requireAll` 定必需性，
`registry.STAGES` 同步注册（两处双向核）。于是 `readiness("qwen_image")` = 38.05 + 8.74 + 0.24 =
**47.03 GiB（盘上字节）**，三件全必需，与驻留账 54.4 GiB 是**两个数**（fp8/int8 装进来要还原成 bf16）。
Edit 2511 **故意不打 `qwen2512` tag**：它与 t2i 互斥，进 tag 会让 `requireAll` 把两代都算必需、就绪档假红。

**Edit 2511 = 能装不能跑（显式拒跑，不静默）**：那份权重靠 `__index_timestep_zero__` 哨兵把参考图
token 与目标 token 拼在一条序列里、并给参考图行喂 σ=0 的调制行。`QwenImageDiT.forward` 已经收
`ref_latents`，但**参考图通路没接**（Qwen2.5-VL 视觉塔 + 参考图 VAE 编码），`denoise` 也没有这个口
⇒ 照装照跑会画出"像提示词"的图、参考图那一路整段缺席、0 报错。所以 `TorchBackend.denoise` 在
**前向之前**就抛（`reason="pending"`），并给两条出路：先接视觉塔与 `ref_latents`，或换 2512 的 t2i 权重。

⚠️ **当前状态别读成"已闭合"**：结构与前向写了、键集与盘上 1933 张量逐键对账过、装配线也接了，
但**还没有一次真权重实跑出图**（无 PNG 证据）。验收口径是 `engine_qwen_image_test.py`（48 项：
σ 的 3.1、吸附 16 与偶数潜边、四格点名、三个按族参数、三本账分隔与三个可见面、Edit 拒跑位置），
它**不装真权重**；
真装真出图是下一步（DiT 38.05 GiB 单件超卡 ⇒ 必然 `split`/`cpu`，跨界前向慢是必付成本）。

### 10.2 有格式支持、无前向骨架的模型族

格式与量化层（L4-2 / L4-3）对所有盘上权重都通用：safetensors、GGUF、fp8、int8、int4 打包、convrot、
NVFP4 都能读能还原。但**网络本体**（L4-5）只有七个族有骨架：H3、SDXL、GGUF LLM、DiT 通用（Krea2）、
Z-Image（`dit_lumina`）、Qwen-Image 2.1（`dit_qwen21` + `qwen35_te`）、Qwen-Image 2512/Edit
（`dit_qwen_image` + `qwen25_te`）——后两族都由 `torch_backend` 按权重形状建起来。
Flux.1-dev / Wan2.2 / SEEDVR2 / Music3 / F5-TTS **暂无前向**。
这是当前最大的功能缺口，也与 L4 的既定目标（跑通共享池里的全部族）直接相关。

Z-Image 那条 2026-10-31 的实测证据（`unet/z_image_turbo_bf16.safetensors`，11.46 GiB / 453 张量）：
判族 ⇒ `zimage`、装载 20 秒（`residency=off`，全量上卡）、`describe()` 自述 `latentMode=real-shape`、
bf16 前向 `(1,16,64,64)` 耗时 0.41 秒。**踩到的坑**：`rope_freqs` 返回 fp32，`apply_rope` 直接把
bf16 的 q/k 升成 fp32 ⇒ SDPA 报 `query float32 / value bf16`；小配置 fp32 自检全绿，只有真权重
才暴露 ⇒ 判据必须单独钉「bf16 进 bf16 出」（`engine_dit_generic_test` 与
`engine_dit_lumina_test::case_precision` 各一处）。与 ComfyUI `NextDiT` 的**逐层中间张量对拍**
脚本留在 `backend-py/_crosscheck_lumina.py`（必须在 ComfyUI 的 venv 里跑，因为要 `import comfy`；
小配置就够暴露时间轴取反、调制顺序、RoPE 轴与 theta、qk-norm 位置、补位 token 位置、首尾
patchify 顺序这些"错一个数就对不上"的点）—— 它不是一次性残留，是那条「相对差 0」结论的**唯一
复现路径**，别顺手删。

### 10.3 整域关闭（实现保留，调用链已断）

`app/services/color_grade.py`、`app/services/data_storage.py`、`app/services/rhythm_phase.py`，
以及 `app/agent/subagent.py`。Node 侧已停用、Python 侧实现完整并被测试覆盖。
**不许当死代码删** —— 删掉等于把「暂时停用」变成「能力消失」，将来重启时无从对照。

## 11. 维护规则（改功能时必须同步哪份文档）

| 你改了什么 | 必须同步 |
|---|---|
| 新增/删除一个后端模块 | 本文对应二级域 + 该层数字（总览表与节标题）+ `tests/structure_test.py::MODULE_GROUPS`（若在 engine） |
| 新增/删除一个 engine 模块 | `app/services/engine/__init__.py::MODULE_GROUPS` + 本文 L4 |
| 新增一个功能域（二级域装不下） | **先补分类体系**，再落功能（不许塞进「其他」） |
| 新增/改端点 | `docs/api-contract.md`，本文不用改（本文不写端点细节） |
| 新增测试套件 | `tests/run_all.py::TESTS` |
| 判断某格式能不能跑 | `docs/compatibility-matrix.md` |

**怎么核覆盖完整**（一条命令列出没被本文点到的 `app/**.py`）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import pathlib; doc=pathlib.Path('../docs/feature-map.md').read_text(encoding='utf-8'); miss=[p.as_posix() for p in sorted(pathlib.Path('app').rglob('*.py')) if '__pycache__' not in p.parts and p.name!='__init__.py' and p.as_posix() not in doc]; print('未登记:', len(miss), miss[:20])"
```

**落在本文上的两条机械判据**（`tests/structure_test.py`，本文与代码同生共死，不靠人记）：

- ㊄：本文点名的每个 `app/…py` 都要在盘上存在 —— **路径腐烂即 FAIL**。
- ㊆：`app/` 下每个 `.py` 都要在本文被点名 —— **新模块没归类即 FAIL**（`__init__.py` 除外）。
- ㊇：盘上每份 `tests/*_test.py` 都要登记进 `run_all.py::TESTS` —— **漏登记即 FAIL**
  （只是「规则表上写了要登记」拦不住：2026-10-31 实测 `lora_train_test.py` 85 项漏了一个多月）。
