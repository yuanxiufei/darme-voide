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
| L1 | 接口层 | `app/routers/` 43 + `app/` 顶层 3 | 46 | 9 | HTTP 契约：入口挂载、路由、错误码形状 |
| L2 | 智能体层 | `app/agent/` `app/mcp/` | 24 | 5 | 协议、工具、循环、多智能体、评测闭环 |
| L3 | 业务服务层 | `app/services/*.py` | 78 | 12 | 领域逻辑：剧本→资产→提示词→生成→合成→质检 |
| L4 | 推理引擎层 | `app/services/engine/` | 64 | 6 | 自研 torch 推理：格式/量化/骨架/编排 |
| L5 | 本地执行与厂商适配 | `services/{adapters,ollama_native,lora_train}` `app/local_services/` `app/vendor/` | 48 | 4 | 把请求落到具体后端、进程、显卡 |
| L6 | 平台层 | `app/core/` | 6 | 3 | 配置、DB、模型表、请求/响应/CPU 预算 |
| L7 | 开发期工具 | `app/scripts/` `tests/` | 22 | 4 | 体检、基准、结构守卫（不进运行时） |

> 模块数口径：**该目录下 `.py` 文件数，排除 `__init__.py`，含子目录**。七层相加 **288**；
> 另有 `app/skills/**/scripts/` 的 3 个技能脚本（见 L2-5：不进应用分层，但也要点名）
> ⇒ ㊆ 实际要求全文点名 **291** 个 `.py`。
> ⚠️ 表内数字是 **2026-10-30 快照**，此后不断新增、**已明显偏小**（实测 `app/` 现有 382 个
> `.py`，`app/services/` 连子包 254 个）。它只用于看相对规模；㊀/㊆ 判的是**点名覆盖**，
> 不是「数字相等」。新增/删除模块仍请顺手 ±1 并改正各节标题 —— 漏点名必被
> `tests/structure_test.py` 的 ㊆ 抓出来（FAIL），数字漂移则不拦。
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
| 分集 CRUD | `app/routers/episodes.py` | 12 端点：整域 10/10 已迁 ＋ 场次 2（`GET/POST /{id}/script-scenes`，读走 `get_conn`、解析走 `get_tx`，`?force=` 重算）|
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
| 引擎体检 | `app/routers/engine.py` | 自研引擎的健康报告；另含 `GET /engine/capabilities`（**本轮补上 HTTP 口**：以前只有 `facade.capabilities()` 函数、路由落到 `not_migrated` 501 ⇒ 前端想知道"认哪些采样器/调度器/画布档位"只能读代码猜，猜错就是静默按默认跑）、`POST /engine/canvas-plan`、`POST /engine/plan` 收 `stage`/`target`/`canvas`/`denoise`/`initImage` |
| 本地模型 | `app/routers/local_models.py` | 11 端点 + 2 自研端点，Node 侧 688 行 |
| LoRA 训练 | `app/routers/lora_train.py` | LoRAMaster 移植 |
| AI 供应商配置 | `app/routers/ai_configs.py` | 含 `/ai-providers` |
| AI 音色 | `app/routers/ai_voices.py` | Node 侧 437 行 |

### L1-5 智能体与技能（6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 会话 | `app/routers/agent.py` | 对齐 `agent.ts`（57 行） |
| Agent 配置 | `app/routers/agent_configs.py` | 含 skills 绑定 |
| 技能管理 | `app/routers/skills.py` | 7 端点（`GET/POST /`、`GET /meta`、`GET /usage`、`GET/PUT/DELETE /{id}`）；目录域非数据库 |
| MCP 客户端 | `app/routers/mcp.py` | 3 端点 |
| 评测 | `app/routers/evaluation.py` | 5 端点 |
| 文本流式 / 结构化 | `app/routers/text.py` | 2 端点（SSE 流式 + schema 校验+回喂）；Node 侧无对应域 ⇒ 只在 route_parity 的 EXTRA 表 |

### L1-6 资产与存储（6）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 资产版本 | `app/routers/asset_versions.py` | — |
| 上传 | `app/routers/upload.py` | 3 端点 |
| 存储根切换 | `app/routers/storage.py` | 2/2 |
| 资源库规格 | `app/routers/libraries.py` | 角色/场景/兵器/服装四类声明 |
| 导出 | `app/routers/export.py` | 7/7 |
| 知识库检索与写能力 | `app/routers/knowledge.py` | 7 条 GET（report/tree/documents/search/facets/document/scenes）＋ 6 条写端（`GET /document/state`、`POST|DELETE /document/chunk`、`DELETE /document`、`POST /document/restore`、`GET /overlay`、`POST /upload`）；目录维度过滤交给 `services/script_catalog`、`skill=` 交给 playbook，`/facets` 一并给 `playbookSkills`；`/tree`＋`/documents` 是前端「浏览态」的数据源（目录树带 `documents/withText/ocr/ocrPartial/pending` 五计数：`ocr` **只认 sidecar 的 `status='ocr'`**，`ocrPartial` 是「有记录但只跑了几页」且**计入 `pending`** 的子集；文档清单按 `only=all|withText|ocr|partial|pending|extra|edited|buried` 筛，`q=` 只按路径模糊匹配（与 `/search` 的正文 bm25 是两条路）；`/search` 另有 `includePartial` 才返回半成品，并逐条回 `coverage={pages,chunkPages}`）；**写操作一律落第四库覆盖层 `library-overlay.sqlite`**，三只读库（第三方预建＋本仓 OCR sidecar＋本仓补录）仍只读打开；**正文只走 `POST /document/chunk` 的 JSON body，没有 PUT**（800 字正文走 query 会被 URL 上限截断，症状是「保存成功但后半段丢了」）；`/overlay` 同时下发上传口径（`upload.accept/label/maxBytes/maxFiles/dir`），前端不抄第二份；`/scenes?path=` 读 A 侧场次派生库，三种回执分得很清：`available=false`（还没建索引）／`scenes=[]`（这一份还没解析过，属按需生成、**不是**漏做）／`document.status=no-heading`（解析过但没认出边界）；**不在端点里触发解析**（一份要读全本几百页，放端点里必超时）|

### L1-7 预设与风格（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 预设 CRUD | `app/routers/presets.py` | — |
| 预设框架 | `app/routers/preset_framework.py` | 247 行；6 端点此前**零前端调用**（整域孤儿），2026-11-04 起由 `PresetPipeline.vue` 全接 |
| 预设框架入口 | `frontend/app/components/PresetPipeline.vue` + `composables/useApi.ts::presetFrameworkAPI` | 首页「框架开局」：取卡（可换一张，`excludeFamily` 挡重）→ 一键（`full-pipeline`）或只建结构（`create`）→ 面板按 `status` 逐镜显示，缺首帧时给「生成首帧」，再「生成视频」。⭐ 语义坑：`storyboards.status` 是**入队态**（`generate_image()` 返回 id 时就写 `image_ready`，异步成败不回写），完成态只看 `first_frame_image`/`video_url` 两列 —— 反过来读会得到「首帧就绪但首帧为空、0/5」；卡里的 `THEME_FAMILY_A` 是后端占位数据池，不是乱码（2026-11-04） |
| 风格档案 | `app/routers/style_profiles.py` | — |

### L1-8 运维与观测（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 运行时体检 | `app/routers/doctor.py` | 把「服务活着没有」摆到开跑之前（2026-10-30） |
| 追踪 | `app/routers/traces.py` | 3 端点 |
| 用量成本 | `app/routers/usage.py` | — |
| Webhook | `app/routers/webhooks.py` | 123 行 |
| 应用设置 | `app/routers/app_settings.py` | — |
| 知识闭环 | `app/routers/knowledge_loop.py` | `/api/v1/knowledge-loop`：跑次归一 `/runs`、台账 `/entries`、蒸馏 `/distill`、投喂 `/feed`、数据集 `/datasets`＋`/datasets/resolve`（= 训练 `datasetRef` 的可选值与同一判据）、参数覆盖 `/tuning`、训练登记 `/training`（**统计坏了返回空且报 `problems`，不 500**）、质检人工复核 `/qc/hit-rate`＋`/qc/pending`＋`/qc/reviews`（`POST` 覆盖式改判、`DELETE /qc/reviews/{id}` 撤销）；**管理面**（2026-11-07 补齐，契约与验收真源 = `docs/knowledge-base-contract.md`，判据 = `tests/knowledge_management_test.py` 75 条）：`PATCH /entries/{id}`（改错，非写入）、`DELETE /entries/{id}?hard=`、`POST /entries/{id}/restore`、`GET /entries/{id}/versions`＋`/{version}`、`POST /entries/{id}/revert`、`GET /tags/facets`、`GET /taxonomy`、`GET /audit`。五条不许含糊的口径：① **叠加 ≠ 编辑** —— `POST /entries` 是同 (kind, scope, 标题) 的「又一份证据」（`support+1`、覆盖正文），改错只能走 `PATCH`（`version+1`、`support` 不动），所以**没有 PUT**（整条替换漏传 body 就清空正文）；② 版本是**乐观锁**：`expectedVersion` 落后回 409 且正文未被覆盖，`entry_versions` 存「被替换掉的旧值」＋`editor/reason`，`revert` 是**普通编辑**（`version+1`，后续版本不删）；③ 删除三态：默认软删（`deleted_ns/deleted_by` 落值，列表/检索/详情全不可见，详情口不是后门）、恢复是人的动作、**硬删必须先软删**（否则 409 `not_deleted`，角色 `owner`），硬删清内容但 `entry_audit` 保留；已软删条目**不可编辑**（409），蒸馏撞上它一律拒收（409 `deleted`）而不复活；④ 错误码按**机器码**分派（`_write_error` 映射 `not_found→404`/`conflict、deleted、not_deleted→409`/`evidence→422`/`unavailable→500`/其余 400），400 只给「请求形状不成立」、422 给「格式对但内容站不住」，两者前端动作不同（改请求 vs 补证据）；⑤ `role` 是**声明式**（本仓无认证中间件、无登录态，`viewer<editor<owner`，空值不判以兼容旧前端与 Agent 工具）—— 只保证「审计里的角色与动作不自相矛盾」，**不防冒充**，见契约 §4 的未做清单；标签是逗号串落库（写入口归一全角/顿号、单标签 ≤24 字、≤12 个），匹配走 `',' || tags || ','` 词边界（点「节奏」不会命中「节奏慢」），facet 与列表同口径，多标签是 **AND**；⭐ 2026-11-08 补**关联读侧**：`/entries/{id}/related`（邻居：`shared` 共用同一 `stage`/`shot` 是**事实**、`similar` 是正文 trigram 启发式，两档分开给不混分数）＋ `/context`（按 `stage`/`shot` 反查，两个都不给 ⇒ `total=0`+`hint`，**不猜**）＋ `/context/refs`（台账里真挂过的 ref 清单）；契约 `docs/knowledge-base-contract.md` §3.2（总表 40→42 口），Agent 侧工具 `context_knowledge`，见 `docs/knowledge-loop.md` §11。⭐ 同日补**投喂的作用域键域闭合**：`compile_for_prompt(scopes=…)`＋`scopes_for(episode_id, drama_id)`（字面形对齐写入口 `EP12`/`DRAMA3`），`assemble_full_instructions` 用手上真值算 ⇒ 「同一集刚采纳的经验」次次能注入（此前只对 `global` 生效，回执还说"还没积累"）；回执增 `scopes`/`blocked`，把「作用域不匹配」与「确实没积累」分成两种空；`/feed/preview` 增 `episodeId`/`dramaId`/`scopes`。`stage:`/`shot:` 有意不自动加（跑次手上没有真值，编了就是乱猜），走 Agent 工具。⭐ 同日再补**定向优先占预算**：`_scope_rank` 两段式排序（`global` < 未知/`agent:*` < `DRAMA<id>` < `EP<id>` < `stage:*` < `shot:*`，档内才看 weight），回执增 `targeted`（这批几条属于本剧/本集/本阶段）—— 准入通了而排序只看 weight 时，本集经验仍会被全局高置信条目挤出 `limit`，口子只开一半。见 `docs/knowledge-loop.md` §12.5 |

## 3. L2 智能体层（`app/agent/` 23 + `app/mcp/` 1，5 个域）

这一层是「谁能自己决定下一步」的地方：上接 HTTP，下用 L3 的领域能力，自己不管数据形状。

### L2-1 运行时内核（4）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 主循环 | `app/agent/runtime.py` | 移植 `agents/index.ts`（745 行），S5 ③；**四条驱动**：OpenAI 兼容 / Ollama / Gemini / **自研引擎（`provider="engine"`，2026-11-07 补）** —— 引擎那条不发 HTTP（进程内 `to_thread` 跑前向），工具调用按 `<tool_call>{json}</tool_call>` 协议从**模型原文**里解（`parse_engine_tool_calls`：块坏就抛、只截断不算调用），`assistant.tool_calls` 与 `role=tool` 的 `name` **原样回灌**；缺 `settings.ggufPath` 或缺 chat 模板当场拒。**小模型漏标签的退路**（`_bare_tool_payloads`，极窄）：整段正文就是一串裸 JSON 对象（对象间只许空白、可套 ```` ```json ```` 围栏）、键 ⊆ `{name,tool,arguments,parameters}`、名字**全都**真注册过、个数 ≤ 32 ⇒ 认；混着编造的名字**整串不认**（不许"认得几个跑几个"）；夹在散文里的对象一律不认（那是举例/回显，认了就是凭空造副作用）。⚠️ 必须认「一串」：extractor 真跑模型一口气吐了 4 个读工具，只认单对象的写法会把它整串丢光 ⇒ 一个读工具都不执行 ⇒ 模型永远走不到 `save_*` ⇒ `/tune` 四跑恒 0 分（看起来像"模型不行"，其实是解析层吃掉了合法调用） |
| 重复调用守卫 | `app/agent/repeat_guard.py` | 同一工具同一参数连着调到 `REPEAT_LIMIT` 就**不再执行**（第 limit 次即拦，不是第 limit+1 次），回灌一句"你已经调过 N 次、结果在上下文里"给模型；同批（一条 `assistant` 消息里的多调用）超 `BURST_LIMIT` 只跑前几条。守卫靠 `ContextVar` 包住 `driver(...)`（三条驱动签名一个字不改，`runtime._ToolCallGuard` 是唯一主人），`fallback` 重试开的是**新账**（上一轮撞墙不该把重试也堵死）；`receipt()` 记账（拦了几次、各是哪条工具、省下多少次执行） |
| 两段式人工介入 | `app/agent/intervention.py` | 政策（`INTERVENTION_MODES`）＋规则表＋安全黑名单＋记忆：命中后**先记账后批准** —— 本轮只入 `pending_approvals` 并把"等你点头"回给模型，调用方再以 `options["approvals"]` 回传才真执行（`"*"` 是兜底通配）。⚠️ `pending_approvals` 非空**不等于失败**（没跑是设计，不是崩了）；`match_pattern` 除 `前缀:*`／`*` 外还认 `前缀 + 空格`（参考实现的注释例子 `git add:*` 对真实命令 `git add foo.md` 永不命中，照抄等于规则表全是哑的） |
| 工具基座 | `app/agent/tool.py` | 工具的注册、参数校验、结果封装 |
| 上下文预算与压缩 | `app/agent/context_budget.py` | 零依赖；超预算时的裁剪策略 |
| 输出协议 | `app/agent/protocol.py` | 移植 `agents/protocol.ts`（79 行）——**唯一权威**（services 侧那份分叉副本已于 2026-10-30 删除） |

### L2-2 工具集（7，`app/agent/tools/`）

工具是「Agent 能做的动作」，全部是纯函数式封装，不持有状态。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 语料检索 | `app/agent/tools/corpus_tools.py` | 找相似片段/成法引用 |
| 剧本/编剧知识库检索 | `app/agent/tools/library_tools.py` | 挂编剧/分镜 Agent，透传 `services/script_library` 的结果（不另造字段） |
| 技能附件按需读取 | `app/agent/tools/skill_ref_tools.py` | `list_skill_refs`/`read_skill_ref` 两个工具，挂编剧/分镜 Agent；**只转发**，路径校验与上限全在 `services/skill_refs.py` |
| 角色·场景·物品提取 | `app/agent/tools/extract_tools.py` | 445 行，六大工具之一 |
| 剧本改写 | `app/agent/tools/script_tools.py` | 83 行 |
| 分镜拆解 | `app/agent/tools/storyboard_tools.py` | 517 行，体量最大 |
| 宫格图提示词 | `app/agent/tools/grid_prompt_tools.py` | 92 行 |
| 角色音色分配 | `app/agent/tools/voice_tools.py` | 148 行 |
| 既往经验检索与提案 | `app/agent/tools/knowledge_tools.py` | `search_knowledge`（只读，透出 `ranking=bm25/like/recent` 三态，不伪造分数）+ `propose_knowledge`（**只能写 `proposed`**，证据核不过整条拒） |

### L2-3 上层编排（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| Agent 实例创建 | `app/agent/creator.py` | 按 agent_type 装配工具与 skill |
| 技能注入 | `app/agent/skills.py` | 默认绑定由 SKILL.md frontmatter 自述；`list_skill_ids` 等为全仓唯一实现。**超预算裁剪**（缺口 10）：装得下就按 `priority` 原序全量注入（一个字节不变），装不下才按 `priority + 降级惩罚` 排（惩罚来自 `services/skill_usage.penalty`，只在「样本 ≥ 10 且成功率 < 0.5」时非零）；`sizes`/`dropped` 是**出参**，本函数不落库（同一跑次会被调用两次，写库就记成两笔） |
| Skill 注入台账 | `app/services/skill_usage.py` | 派生统计库 `data/knowledge/skill-usage.sqlite`（`SKILL_USAGE_DB` 可换）：记「注入了哪些 / 各多少字符 / 哪些被裁 / 该跑次成败」，`run_id` 用 `trace:Agent/<id>`、同 run 覆盖幂等。⚠️ **没有**「命中率」——本仓没有「模型读没读它」的观测通路，``okRate`` 只能读成「注入它的跑次成功率」（相关，不是因果），样本 < `MIN_SAMPLES`(10) 一律 `judged=false` 不说话。`GET /api/v1/skills/usage` 是唯一读口 |
| 子 Agent 调度 | `app/agent/subagent.py` | 整域关闭（Node 侧已停用） |
| 全自动管线 | `app/agent/auto_pipeline.py` | 移植 `services/auto-pipeline.ts`（814 行） |
| 审片重跑闭环 | `app/agent/qc_retry.py` | 移植 `services/qc-retry.ts`（175 行） |

### L2-4 评测闭环（10，`app/agent/evaluation/`）

一条独立的「测 → 评 → 改」小流水线，与运行时解耦。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 数据契约 | `app/agent/evaluation/types.py` | 五类 kind（`storyboard`/`extractor`/`script_rewriter`/`voice_assigner`/**`grid_prompt`**）；`grid_prompt` 是 2026-11-07 补的第 5 类，卡在「产出层」与「成片层」之间：提示词写砸了，账最后会记在成片维度上，而那时视频生成的钱已经花掉了 |
| 基准 case 目录 | `app/agent/evaluation/catalog.py` | 单一来源 |
| 执行器 | `app/agent/evaluation/evaluator.py` | 五个 kind 一个入口；`grid_prompt` 分支要把基准镜**真落成 `storyboards` 行**（宫格 Agent 的工具 `read_shots_for_grid` 只按 id 查库），产出解析走 `services.grid_payload`，与线上同源。另含**单步对照** `step_trace(tool_calls, tool_results)`：把逐步账与 `stats(total/ok/failed/unjudged/unknownTool/tools)` 记进报告，五个分支的 return 全经 `_report_steps()` 收口（不收口 = 某类 case 悄悄没有 steps，而测试正好只测了另一类）；`evaluate_case(..., skills=, tools=)` 的按次覆盖只走 `options["skillsRaw"]/["toolsRaw"]`，不碰库 |
| 确定性评分 | `app/agent/evaluation/scorer.py` | 五个评分器，**纯函数，不依赖 LLM/DB** |
| 语义判分器 | `app/agent/evaluation/judge.py` | 规则评分器判不了的（"这段改写保住了人物动机吗"）交给 LLM 判官：五段 prompt（`[Criteria]/[Input]/[Output]/[Evidence]/[Expected]` 顺序固定）＋ `JUDGE_SCHEMA`（`score` 0..1、`reason` 必填）。三条口径：**`score=0` 是合法判决**（不是"没判"，别把它当翻车重试）、**只在判官翻车时重试**（解析不出／越界／NaN ⇒ `inconclusive`，默认 3 次、温度固定 0）、**`inconclusive` ≠ 0 分**（没判成不许折成最差）。开关是 case 定义里的 `judge` 字段 —— 现存 case 的 `criteria` 是规则词表，误拿它当开关会让所有 case 真跑 LLM |
| 成片评测 | `app/agent/evaluation/film_check.py` | 缺口 8 的成片半边：只读 `storyboards`/`video_generations`/`video_quality_checks`/`qc_reviews`，**不重跑 ffmpeg、不调模型**。四维等权（覆盖/规格/缺陷/复核），**判不了的维度不进总分**（`judged=false` 绝不折成 0 或满分），`missing` 点名；`review` 分母只含 `hit+false_alarm`，漏报单列进 notes；跑次 `run_id=film:<集>` 幂等覆盖（点五次评测不等于五条样本），`kind="eval"` |
| 提示词优化器 | `app/agent/evaluation/optimizer.py` | 状态机闭环：Reference → 候选 → 再评 → 严格更高才落库。三条口径 2026-11-09 补齐：① **v0 基线＝现网**（`effective_prompt()`：`agent_configs.system_prompt` 优先，无行/空串才回退出厂默认，回执 `baselineSource`），否则重跑会把已落库的更优版本静默覆盖；② **`version` 跨运行连续**（共用 `param_optimizer.next_version`，`run` 是本轮第几次、`version` 是轮次序号，两个数不许混用），否则 `mark_persisted` 一次标中多行；③ **单轮跑崩只作废那一轮**（`generate_candidate_prompt` 的「输出太短」是设计内失败 ⇒ 记 `eval_failed`、占版本号与分母、继续下一轮；`TuneCancelled` 原样抛出；失败轮按 `stage` 决定留不留候选 —— 生成失败没候选、评测失败留住已生成的那份全文），`benchmarks/history/<agent>/<case>.json` **逐轮落盘**且每条带 `reason`，落库结论（`persisted`）之后再补落一次（逐轮 flush 在落库之前，不补就永远读不到「这版生效没有」）。④ **抖动带可选**（2026-11-10）：`repeat`＝**Reference 复测次数**，默认 1 ⇒ 抖动带 0、零容差、与历史逐字一致（回执 `jumpKind="none"` 并印「没测抖动带」，不印「抖动带 0」）；传 2–5 取样本极差 `max−min` 当抖动带（`within`），采纳判据变 `candidate > best + jump`。`iterations` / `repeat` 有上限（`MAX_ITERATIONS=20` / `MAX_PROMPT_REPEAT=5`）且**越界即拒、不夹逼**（`normalize_prompt_options` 是入队口与执行口共用的唯一校验点，`/tune-prompt` 入队前就 400）；进度分母是 `repeat + iterations`（`repeat` 全在 Reference 上，候选每轮只评一次）
| 参数（温度）调优 | `app/agent/evaluation/param_optimizer.py` | 缺口 4 第一格：同一份提示词只换 `temperature`，候选是**一批平级**（`TEMPERATURE_GRID` 只往低扫）而非父子演化链 ⇒ 单开模块，混进 `optimizer.py` 会让 version/parent 语义失真。三条硬约束：`repeat>=2`（单次评测定不了输赢）、`delta > 抖动带`（基线各轮 max−min，涨了但没超过抖动带记 `below_noise`）、换模型记 `runtime_mismatch`。落库走 `creator.persist_agent_temperature`（只 update 那一列，无配置行返 0 不插空壳）。见 `docs/knowledge-loop.md` §1 缺口 4 |
| 技能绑定调优 | `app/agent/evaluation/skill_optimizer.py` | 缺口 4 第二格（2026-11-08 开）：同一份提示词、同一个温度，只换**注入哪几个 skill**（候选是 leave-one-out）。⚠️ 判据与温度**方向相反**——温度是「涨过抖动带才采纳」，这一格是「**摘掉后跌幅不超过抖动带 ⇒ 没有证据说明它有用 ⇒ 采纳这一摘**」（`delta >= -jump` 即采纳，`pruned` / `skill_needed` 两个理由码，所以这类行 Δ 为 0 或负数也可能是「采纳」，不是账记错）。四条硬约束：**一次只摘一个**（摘掉后基线就变了，同轮连摘是拿旧基线判新绑定）、**不许摘成空**（剩 1 个时有 0 候选，不是「摘掉最后一个」）、**覆盖走 `options["skillsRaw"]` 且不写 `skill_usage`**（假设性绑定若计入使用统计，会污染下一轮的裁剪顺序，把真在用的 skill 排在可疑位）、**`autoPersist` 默认 False**（温度最坏是更死板，删 skill 最坏是「该知道的规范不知道了」⇒ 只出结论、由人点头）。落库走 `creator.persist_agent_skills`，无配置行返 0 不插空壳。绑定形状必须是 `[{"id","enabled","priority"}]` —— 纯字符串数组会被 `parse_skills_config` **静默跳过**并回退默认绑定（症状：页面写已采纳、线上还是旧绑定，两边都不报错）。见 `docs/knowledge-loop.md` §1 缺口 4 |
| 工具集调优 | `app/agent/evaluation/tool_optimizer.py` | 缺口 4 第三格（2026-11-08 开）：同一份提示词、同一个温度、同一套 skill，只换**给模型看哪些工具**（候选是 leave-one-out）。这一格的前置是 `evaluator.step_trace` 的**单步对照**——整轮一个分数颗粒太粗（摘掉 `search_script_library` 后模型可能改调别的工具、总分几乎不动，也可能整轮崩掉却只掉 2 分），所以判据是**两把尺子**：整轮分 + `unknownTool` 计数（模型照着被摘掉的工具名去调 = 它真需要这个工具）。四个理由码：`pruned`（分数没掉 **且** 不再调它）、`tool_wanted`（分数没掉但模型仍在调 ⇒ 不摘，技能格没有这一档）、`tool_unjudged`（分数没掉但**逐步账没明细**，`stats.unavailable` 非空 ⇒ `unknownTool` 恒 0 是读数缺失、不是"没再调" ⇒ 同样不摘，与 `tool_wanted` 必须分成两码；**部分**缺账 `stats.unjudged>0`「有调用没回灌结果」同属这码，因为 `unknownTool` 正是靠结果里的 error 正文认的）、`tool_needed`（跌幅超抖动带）。基线那一跑就缺明细时**在基线处收工**（第二把尺子整格哑掉，候选跑再多也只能得 `tool_unjudged`），且理由如实写「没有明细」而**不是**「空工具集」（缺明细时暴露集同样是空的，照空分支走等于把"没记账"说成"模型没调过工具"）。端到端也钉在 HTTP 面（`evaluation_route_test` 两条）：`POST /evaluate/{case_id}` 返回的报告带 `steps`/`stepStats`，成对 calls/results 记 1 步 `ok`（`unjudged=0`）、结果含 `Unknown tool:` 记 `unknownTool=1, failed=1`；且 `evaluator._UNKNOWN_TOOL_PREFIX` 与 `runtime._run_tool` 抛出的字面量必须一致（改任何一半 ⇒ `unknownTool` 恒 0 而 `failed` 照常计数 ⇒ 判决链错摘）。三条硬约束：**覆盖走 `options["toolsRaw"]` 且顺序跟注册表**（`ToolRegistry.only()` 按键或按 `tool.id` 两套口径匹配，`missing` 如实上报，不静默全开）、**候选源是基线那一跑的「内存逐步账」**（`stepStats.tools`，不是盘上轨迹文件——`meta.toolCalls` 实测只存了个数，没有逐步明细；模型没用过的工具摘掉本来就测不出东西）、**恒 `autoPersist=false` 且 `persistSkip` 明写理由**——工具集真源是代码里的装配语句，`agent_configs` 没有 tools 列，所以这格**只落优化史结论**（`target="tools"`）、不写库。见 `docs/knowledge-loop.md` §1 缺口 4 |
| 调优的后台任务表 | `app/agent/evaluation/tune_runtime.py` | 把 `POST /api/v1/evaluation/tune/{case_id}` 从「一次 7 分钟的同步请求」改成「入队立刻返 `taskId` + `GET /tune/{taskId}` 查进度」（真机实测 `repeat=2` 单跑才 85–165 秒，四跑 423.9 秒，挂在同步返回链上必然是**网关先超时、后台还在烧卡**）。**四个目标共用这一张表**（`target=params` 温度 / `target=skills` 技能绑定 / `target=tools` 工具集 / `target=system_prompt` 提示词改写 —— 任务形状一样，runner 按 target 选；复制一张表会让「取消只在两跑之间」「工作线程自己开连接」这些口径迟早一边改一边没改），去重键是 **(agent, case, target)**（漏掉 target 会把刚提交的技能调优当成温度任务的重复而静默丢弃，还回一个别人的 taskId）。三条口径都不可选：**单工作线程 + 队列**（只有一张卡，并发调优只互相抢显存 ⇒ 串行是唯一诚实的并发度，排队就明说 `queuePosition`）、**取消只在两次评测之间生效**（一次评测内部那 85–165 秒的采样循环掐不断 ⇒ `cancel` 是「不再开下一跑」，不假装秒停）、**工作线程自己 `engine.begin()`**（`get_conn` 的连接生命周期跟着 HTTP 请求走，拿它跑后台任务症状是 `connection is closed`；⚠️ 且必须是 `begin()` 不是 `connect()` —— SQLAlchemy 2.0 下 `connect()` 里的写入在 `close()` 时**一律回滚**，用 `connect()` 等于回执写 `persisted=true`、库里一个字没有，全程无报错。2026-11-08 实测改掉）。任务表是**进程内存** ⇒ 重启后 `GET /tune/{taskId}` 给 404 属刻意设计（判决与落库已在 `agent_optimizations`/`agent_configs`）。七个入口：`POST /tune/{caseId}`（温度，默认入队；`wait=true` 才等回执，超时回快照标 `timedOut` 而**任务不停**）、`POST /tune-skills/{caseId}`（技能绑定，同上，多回一份 `preview.current`/`preview.candidates`）、`POST /tune-tools/{caseId}`（工具集，同上，`preview.candidateSource="trace"` 并带 `persistSkip`）、`POST /tune-prompt/{caseId}`（提示词改写入队版，2026-11-09 补：`autoPersist` 默认 **true**（四格里唯一真写 `agent_configs.system_prompt`）、**没有 `repeat`** ⇒ 分母 `1 + iterations`、`jumpKind="none"`；同步那条 `POST /optimize/{caseId}` **一字未动**，它与入队口共用同一个 `optimize_agent_prompt`，区别只在投递方式）、`GET /tune/{taskId}`（`status`/`stage`/`doneRuns`/`totalRuns`/`receipt`；`target` 标明这次动的是哪一格）、`GET /tune`（列表）、`POST /tune/{taskId}/cancel` |
| 无人值守调度 | `app/agent/evaluation_scheduler.py` | 移植 `evaluation-scheduler.ts`（126 行）；`iso_in`/`next_run_delay` 是**唯一**的「下一轮时间」算法（知识复盘定时器共用，别抄第二份） |
| 命令行入口 | `app/agent/evaluation/cli.py` | 64 行 |

### L2-5 外接协议与技能库（1 + 10 个 skill 目录）

| 功能点 | 落点 | 说明 |
|---|---|---|
| MCP 客户端 | `app/mcp/client.py` | 远端工具包成本地 `Tool`；只依赖 `core` |
| 技能内容（自有） | `app/skills/` | `extractor` / `script_rewriter` / `storyboard_breaker` / `voice_assigner` / `grid_prompt_generator` / `production-tools` 等，靠 `SKILL.md` 自述绑定 |
| 技能内容（外部库） | `app/skills/*/library.yaml` | 库由声明文件识别，库内 skill **不默认注入** |
| 技能附件读取 | `app/services/skill_refs.py` | 加载器**只读 SKILL.md**，`references/` 下正文本来没有任何通路会去读（＝死料）。本模块给模型一只只读的手：根只能在 `skills_dir()`、只允许进 `references/`、后缀白名单＋512 KB／400 行上限，路径判据是**解析后**仍在 `references/` 内（不是字符串前缀，后者会被 `../` 绕过）；`available=false`（没有附件目录）与 `files=[]`（有但空）**分得开** |

`app/skills/` 里的 `.py` **不属于应用分层**：它们是技能自带的命令行脚本，由 `SKILL.md` 指明何时调用，
后端进程不 import 它们（改它们不需要跑 `layering_test`）。逐份点名（㊆ 要求覆盖）：

| 脚本 | 落点 | 说明 |
|---|---|---|
| 节拍检测 | `app/skills/production-tools/beat-sync-editor/scripts/beat_detect.py` | librosa 打拍点 |
| 能量/情绪峰值 | `app/skills/production-tools/beat-sync-editor/scripts/energy_analyze.py` | 找高张力片段与裁剪窗口 |
| 字幕校验与纠错 | `app/skills/production-tools/subtitle-correction/scripts/subtitle_tool.py` | 对照原稿分词级差异 |

## 4. L3 业务服务层（`app/services/*.py` 82 模块，13 个域）

L3 是本仓库的主体：**领域知识全在这里**。判据是「换个 HTTP 框架、换个 Agent 循环，这段逻辑还成立吗」——
成立就属 L3。

### L3-1 剧本要素与连续性（13）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 剧本/编剧知识库检索 | `app/services/script_library.py` | 只读第三方 fts5 库（bm25）+ 切 md 小节；旧盘路径重映射、隔离目录剔除（**path 过滤一律用 `instr`，不用 `LIKE`**：这几处 SQL 打在 fts5 虚拟表 `chunks` 上、`path` 是 `UNINDEXED` 列，fts5 的 LIKE 优化对**长度恰为 3 的纯 ASCII** pattern 会静默返回 0 行——实测 `path LIKE '%doc%'` = 0 条而 `instr(path,'doc')>0` = 30923 条，同一库上 `'%Chinatown%'` 又是 184 条正常，症状是筛「BBC」「S02」「doc」永远筛出空、中文关键字却照常，看不出是过滤写错）；1–2 字回落 LIKE 并如实标 `ranking`。**扫描件有第二条命**：供应商库登记 5403 份、只有 2262 份有正文块（3139 份是扫描件，`get_text()` 全空），本仓用 `app/scripts/corpus/ocr_index.py` 逐页 OCR 进 sidecar（`indexes/library-ocr.sqlite`，`SCRIPT_LIBRARY_OCR_DB` 可换路径）⇒ `search` 两库各取窗口后**按秩融合**、每条带 `from: curated|ocr` 且 `parts` 分列两库秩与命中数，`open_document` 主库 0 块时自动落到 sidecar 并回 `origin`（两库分数不可比，只比秩）；**未跑完的 OCR 文档默认不进检索**——`partial`（只扫了前几页，块却已写进 fts5）与 `quarantine` 用 `NOT IN (SELECT id FROM documents WHERE status IN (...))` 排除（实测 sidecar 80 块里 49 块是这种，其中一份 920 页的书只扫了 1 页），要看得显式给 `include_partial=True` 并逐条回 `coverage={pages,chunkPages}`；参考侧的 playbook 数据在 `corpus/playbook/`（`README.md` 记来源/版本/许可）、**随仓库分发**（2026-11-04 起；此前在 gitignore 的 `方法语料/` 里 ⇒ 新克隆的仓库该源为空），只索引 `PLAYBOOK_DIRS` 登记的来源目录（`film-expert-bridge/`、`film-cinedance/` —— **目录名即 `skill=` 取值**；2026-11-04 扁平化删掉 `packs/`／`references/`／`library/` 三层零信息壳，最深路径从 6 层目录降到 3 层，旧写法 `skill="packs/xxx"`／`"skills/xxx"` 仍由 `_skill_arg()` 归一；根规约不进、未索引数照报），`skill=` 只筛 playbook，>1500 字小节切窗并回 `part/parts`；**切窗按句界对齐 ＋ 15% 重叠**（口径抄 RAGFlow `chunker/token.go::mergeUnits` 的三步契约：超宽单位先按句界展开、阈值按 overlap 缩小 `width*(100-pct)/100`、新窗无条件前置前窗尾巴）——不重叠时跨窗那半句只剩一半，模型会误读残句；**`source=all` 按 RRF 重排**（口径抄 txtai `embeddings/search/hybrid.py::rrf`，`k=1`、两源等权）：两源分数量纲不可比（bm25 负值／词频／LIKE 无分），排序依据落在回执 `fusion`，`score` 仍是各源原分数且明写「量纲不可比」，单源时 `fusion=None` |
| 知识库覆盖层（编辑/删除/上传台账） | `app/services/library_overlay.py` | 第四只库 `data/db/library-overlay.sqlite`（**跟着数据根走**，与 `drama.db` 同级；三只派生索引仍住 `indexes/`。`SCRIPT_LIBRARY_OVERLAY_DB` 可换路径）：`edits`（UNIQUE(source,path,chunk_no)）／`tombstones`／`uploads`／`edits_fts`（trigram）／`meta`；**三只读库是派生物（重跑会覆盖）⇒ 就地改会让 `onDisk=true` 变假话**，故一切写操作只落这一层，`open_document` 读时叠加、`_scan_documents`/`search_methods` 读时剔除墓碑（fts5 的 `path` 是 UNINDEXED ⇒ 窗口后内存过滤并如实报 `hidden.count`，不假装「本来就没有」）；块号对不上（原库重建过）报 `staleEdits` 而**不**静默丢弃；`hard` 物理删只对「本仓上传且文件在盘上」成立，其余只能进回收站；`_stamp()` 进程内缓存戳保证改完即时生效 |
| 本仓补录上传 | `app/services/library_ingest.py` | 落 `00-本仓补录/`，`MAX_FILE_SIZE=64MB`／`MAX_FILES=20`／扩展名白名单（2026-11-07 起含 `.pdf`／`.html`／`.htm`／`.pptx`，前端 `accept` 由 `limits()` 下发、改这一处两端同时生效）；解析复用 `lib_index._build/_save`（与脚本同一口径，不另写一套），同名不覆盖改 `-2`；三档回执不许合并成「失败」：`ingested`（已入库可搜）／`quarantined`（**文件已在盘上**、只是抽不出正文 ⇒ 转 OCR；PDF 走 `ocr_index._extra_quarantined_docx`，`SUPPORTED_SUFFIXES` 里的 `.pdf` 是它唯一补救路）／`rejected`（未落盘，理由说清）；`limits()` 是前端校验口径的唯一真源 |

| 知识库目录层 | `app/services/script_catalog.py` | 两份 CSV（主目录 **UTF-8 BOM** / 电影节标签 **GBK**）→ 内存索引；国家·类型·奖项·权限·资料类型过滤，三态分明（没给条件 / 目录不在 / 没人满足），**不建 sidecar DB** |
| 增量变更集 | `app/services/markdown_patch.py` | 模型改剧本别重写整篇：`apply_edits` 每条 `oldText` 必须**恰好命中一次**（0 次＝抄错或已改过、≥2 次＝定位不到），**整批原子**（任一条不中 ⇒ 一条都不落、返回原文），失败句三段式（事实／原因／怎么改 ＋ `failedIndex`）让模型能照着重送；`coerce_edits` 同时吃 `oldText/newText` 与 `old_text/new_text`；接的是 `edit_script` 工具（与 `save_script` 并存，两条保存路径各自刷新指纹、互不降级） |
| 剧本指纹门禁 | `app/services/script_fingerprint.py` | 改剧本即失效下游签名 |
| 台词说话人匹配 | `app/services/character_match.py` | 移植 `shared/character-match.ts` |
| 场景地点/时段匹配 | `app/services/scene_match.py` | 提取去重与陈旧关联清理共用 |
| 连续性表 | `app/services/continuity.py` | 移植 `production-plan-contract` |
| 连续性写入 | `app/services/continuity_store.py` | 校验 + 幂等替换 |
| 分镜 → 连续性适配 | `app/services/storyboard_continuity.py` | 零依赖 |
| 分镜关联同步 | `app/services/storyboard_helpers.py` | 含台词解析与 TTS 匹配校验 |
| 视觉图谱 | `app/services/visual_graph.py` | 角色/场景/道具的图结构 |
| 场次解析（纯函数） | `app/services/scene_parser.py` | 剧本正文 → 场次，**零 IO／零模型**（services 层不依赖 LLM）；同一行内按下列顺序先命中先用：`md-heading` 0.95 ＞ `cn-numbered` 0.9／裸数字 0.85 ＞ **`ep-scene` 0.85**（短剧「集-场号」`1-1`、`1-1a日外焉州-康郡`、带「场」前缀的 `场1-1 金銮殿 日 内`：`scene_no` **只取场段**，集号每集重置、schema 无 episode 列，故集号只留在 `heading` 原串里）＞ **`hash-scene` 0.85**（韩语／英语剧本的 `S#1. 치킨 집 앞 (밤)`：`S` 是 scene 首字母、`#` 只是分隔符；单独起头的 `#1` 不认 —— markdown 标题、脚注、页码都是 `#`）＞ `slug` 0.9（`INT./EXT.` 起头）＞ **`place-time` 0.6**（无场号、无 INT/EXT 字面，只有「地点＋时段+内外」黏合成一行的 `纯黑环境 夜内`／`二层客厅 日内`；判据最弱 ⇒ 整行要短、无句读与台词冒号、无 `接N-N`／`插入镜头`／`字幕` 这类注记，且**全篇 ≥3 条才采纳**，单条一律丢弃）＞ `none`（整篇一场、如实标 missing）。内外景有三条来源：`INT./EXT./内景/外景` 字面、`夜内`/`日外` 这类「时段+内外」粘合词、以及**挂在地点尾字的单字 `内`/`外`**（`民政局内 日`、`（回忆）客厅内`）；最后这条 `郊外`/`城内` 这类地名**不截字**（截完只剩一个「郊」，更难读），只补内外景。跨行还有一次仲裁：OCR 把正文按页压平后标记多埋在行中间，行内头数 > 行首头数 2 倍时改走 `inline` 重切（`cast`/`opening` 改按句读当行看）。`SCENE_STRATEGIES` 是**在册枚举**，`parse_scenes` 真会返回 `inline` ⇒ 它必须在册，否则"返回值不在枚举里"这类漂移永远查不出来。**纯空行不切**——中文剧本空行到处都有，按空行切会得到几十个假场次，假结构比没结构更坏；判不出来的如实标 `no_source`/`confidence`/`strategy`，`core_event`/`scene_tail` 一律 None（那是语义判断，不拿首句动作糊弄）。**围栏代码块整体不算**（`_fenced_lines` 返回行号集＋字符区间两份：走行结构的按行号挡、走 `inline` 的按区间挡，同一件事两处都要挡），`extract_cast`/`_opening_line` 也只吃 `_content_lines` 过滤后的行；首个场次头之前 ≥200 字单列为 `extractor="preamble"` 的一场（`scene_no=null`、不假装第 0 场、不拉低整篇置信度） |
| 第三库：主库没登记的格式 | `app/scripts/corpus/lib_index.py` | 「盘上有、搜不到」的那批补进 `indexes/library-extra.sqlite`（`SCRIPT_LIBRARY_EXTRA_DB` 可换路径），**供应商库一字不写**。切块口径与 OCR sidecar 逐字对齐（800 字/块、1500 字/页、可打印率 <0.85 判 `quarantine` 且一块不写——坏文本进了 fts5 就再也分不清"书里就这么写"和"抽错了"）。**扩展名没有信用**：实测 873 份 `.doc` 里 50 份其实是 RTF、77 份是 WPS 写的 Word（antiword 一律回 `is not a Word Document`），所以按**内容签名**分派——`{\rtf` 走内置 RTF 解、`d0cf11e0` 先 antiword 再退内置 FIB 分片表（`WordDocument` 流头取 `fcClx/lcbClx`、`1Table/0Table` 里读 CLX 的 Pcdt，`fc` 的 bit30 表示 8 位压缩还是 UTF-16LE）。两处码表都是**必需**而非保险：RTF 的 `\'hh` 先按当前字体的 `\fcharset`（本库 50 份中文 RTF 里混着韩国剧本，文档头声明 `\ansicpg936`、正文用 `\f1`(129) 排韩文，只按文档级码表解就是整屏 GBK 乱码且**每个字都可打印**）；Word 的压缩分片按 FIB 的 `lid` 定码表（cp936 与 cp949 字节区间几乎重叠，两边都能"成功"解出字，只有一边是真字）。OLE 头解析一律带**按文件体积的扇区上限＋环检测**（DIFAT 链条与 FAT 链都可能指回自己，实测某份 116KB 的 Word 让纯 Python 解析器原地转死）。**可打印率里"空白算可打印"是判据本身的修正**：`str.isprintable()` 对 `U+3000`（全角空格）与 `U+00A0` 返回 `False`，而中文剧本整段靠全角空格缩进——实测《生死场》33884 字里 5833 个是这类空白，按 `isprintable` 算得 0.83 就被判隔离、一块不写，正文其实一个字没坏（一轮全量里 37 份完好剧本因此漏收）；现在空白一律算可打印，扣分项显式点名控制符与解码替换符 `U+FFFD`（后者 `isprintable` 反而是 `True`），隔离行同时报「抽出 N 字」以便区分「抽出来是垃圾」与「正文是图片/嵌入对象，该走 OCR」。入库前统一过 `_tidy_text`（`\r\n`/`\r`→`\n`、`U+00A0`→空格、删零宽空格与控制符、三行以上空行压成一行）。`--force` 重算本库已收录的（主库/OCR 覆盖的仍跳过），`--path-substr=`／`--limit=` 圈定范围，跳过原因逐条合计报出（`.csv` 是登记表不是语料，`.ppt` 这类 OLE2 旧格式在 `HANDLED_ELSEWHERE` 里如实报「不收」）。**2026-11-07 起 PDF/HTML/PPTX 归本脚本**（此前 PDF 被直接拒，只有扫描件 OCR 那条线碰得到它）：PDF 走 `pypdfium2`（Apache-2.0/BSD-3；12 份真实剧本 12/12 成、0.08 s/份 —— `pdfminer.six` 9/12 且把每行套成 `| … |` 表格会切碎正文，`PyMuPDF` 12/12 但 AGPL-3.0 进分发主链路即传染）⇒ 抽出的正文过 `_cjk_deglue`（中文 PDF 两端对齐会把「证人席」写成「证 人 席」，实测两份分别 269/2070 处，不去胶着搜「证人席」命中 0；**只作用于 PDF 通道**，不动共用的 `_tidy_text`，否则 873 份 `.doc` 的旧摘要口径集体失配）；HTML 用 `HTMLParser` 剥标签（`script/style` 整段丢、块级标签换行、**行内标签两侧的空格属于词**：strip 掉会让 `INT. HOTEL <i>— DAY</i>` 变成 `HOTEL— DAY`，英文剧本整篇词粘连）；PPTX 按 `slideN` 顺序取 `<a:t>`（属性与实体容错）。**上传白名单与扫库白名单是同一个真源** `SUPPORTED_EXTS`（`library_ingest.ALLOWED_SUFFIXES` 直接读它）：实测抄成两份的结果是 `_extract` 能抽、`_candidates` 却把同一批文件记成「未登记抽文本通道」静默跳过，只有界面上传那条路能用。真实语料实测（库根里主库/OCR 都没覆盖的 25 份）：6 份 HTML 全部抽出 1.6–12 万字真实剧本正文、0 处词粘连；19 份 PDF 一份都抽不出，逐份查过不是解析器的锅 —— 3 份是 `book\0mark` 私有格式改名、6 份 0 字节、1 份 53 MB 全零、9 份 `%PDF-1.x` 头但没有 `startxref`/`%%EOF`/`endobj` 的**截断**文件 ⇒ 全部如实 `quarantine` 并转 OCR 线（PDF 三通道对比、为什么不接通用 RAG／向量库、逐档理由见 `docs/knowledge-stack-audit.md`）；`--prune [--apply]` 清掉「口径已经变了」的旧记录（早期按文本收的 6 份 `.csv` 目录清单会被检索命中，`--all` 只增不删所以留着就是自相矛盾；默认只报清单，`--apply` 才删 `documents`+`chunks`），清完本库 880 份／31157 块与真实候选数逐一对上 |
| 场次索引（A 侧派生） | `app/services/scene_index.py` | 库里 5403 份参考资料 → 两张表 `scene_docs`（每份一行：sha256/strategy/confidence/status/hint）＋ `scenes`（每场一行），落在 `indexes/library-scenes.sqlite`，**不进 `darme.db`**：A 侧纯派生（`DELETE` 后重跑即可），进项目库会被算进备份体积、被 `db_upgrade.py` 当项目结构去补列、换数据根时莫名跟着跑丢。对齐键是 `path`（相对库根）**不是** `document_id`——两库各自编号，只有 path 是共同锚。拼正文时**必须**查 `chunk_no` 连续：OCR 门禁把整页判 reject 时一块都不写，跳号处往往正是场边界，不记就会把两场粘成一场还不自知。跳过判据**两条都要对**：正文 `sha256` 与 **解析器指纹** `parser_sha`（`scene_parser.py` 源码字节的 12 位 sha1，`PARSER_SHA` 模块级算一次）——只比 sha256 的话，「解析器改了、正文没改」的旧派生行会被当成最新，`--all` 整批 `skipped` 却看起来"跑完了"；换指纹就自动重算，`--all`（不带 `--force`）因而是**可断点续跑**的，`--stats` 会单独报 `staleDocuments` |
| 场次结构（B 侧事实） | `app/services/script_scenes.py` | 本仓剧本 → `script_scenes`（每场一行）＋ `script_scene_docs`（每集一行：strategy/confidence/notes/`source_script_hash`），进 `darme.db`、剧本一改标 stale（`stale` 用**当前正文现算**的指纹判，不读 `episodes.script_hash` 那列——列是缓存，直接改库时它是陈的，最该报 stale 的时候反而不报）；与 A 侧**共用解析器但不共用表**——混在一张表里，"重跑解析"会意外清掉项目数据。偏移契约：`char_start`/`char_end` 相对**原文未 strip**，而指纹按 strip 后内容算（既有口径不能改）⇒ 偏移以原文为准、哈希以 strip 为准，回执 `sourceField` 说明该切 `scriptContent` 还是回退 `content` |

### L3-2 提示词与镜头语言（7）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 可复用提示词块 | `app/services/prompt_blocks.py` | 107 行 |
| 五段式质感层 | `app/services/prompt_polish.py` | Mx-Shell 五段式，零依赖 |
| 提示词工具 | `app/services/prompt_utils.py` | 分块移植 `shared/prompt-utils.ts` |
| 宫格载荷解析（单一真源） | `app/services/grid_payload.py` | 从 `routers/grid.py` 上收的三个纯函数：`extract_json_candidate`（先 `json.loads`，再抠 ```json 围栏、再退首个 `{…}`）、`normalize_grid_payload`（camelCase 兼容、空 prompt 格子丢弃）、`find_grid_payload`（递归下降）。生产（`routers/grid.py` 保留同名别名）与评测（`agent/evaluation/evaluator.py`）**必须走同一份**——评测自己认一套格式，线上换写法时就会照旧打高分 |
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
| 文本生成 | `app/services/text_generation.py` | 556 行，可走 Ollama 或自研引擎；含 `generate_json`（schema 校验 + 回喂重试） |
| 文本流式编排 | `app/services/text_stream.py` | 归一事件 `meta/thinking/content/usage/done/error`；三条路（自研引擎 / Ollama ndjson / OpenAI SSE） |
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
| 视频插帧 | `app/services/video_interpolate.py`、`app/routers/video_ops.py` | FFmpeg `minterpolate` 2×/3×（输出帧率跟随源、超时杀进程、尺寸偶数化） |
| 校色 | `app/services/color_grade.py` | **整域关闭** |
| 剪映草稿导出 | `app/services/jianying_draft.py` | 339 行 |
| 导出服务 | `app/services/export_service.py` | 192 行级 |

### L3-6 质检（5）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 图像连续性 QC | `app/services/consistency_qc.py` | 移植 `consistency-qc.ts` |
| 技术维度 QC | `app/services/technical_qc.py` | 471 行；九个阈值默认值仍是 `TECH_QC_THRESHOLDS`（TS 镜像常量，`route_parity_test` 逐值核），读取统一走 `qc_thresholds()` 过 `tuning` 的 `qc` 伪档（见下方「参数覆盖层」）；`ISSUE_THRESHOLDS` 把 issue `code` 映射到该伪档的主字段（`relax` 说明放宽方向，负值区间乘法方向相反；`spec.unreadable`/`spec.probe_failed` **有意不登记**） |
| 质检人工复核（阈值命中率的分母） | `app/services/qc_reviews.py` | 表 `qc_reviews` **单开一张**（漏报没有 issue 行可挂／一次质检同 code 可报两段／复核是事实层不该被 QC 重跑覆盖），`(check_id, code)` **覆盖式改判**；三值 `hit/false_alarm/missed`；命中率分母只含 `hit+false_alarm`，漏报率分母＝全部复核，没判过 ⇒ `None`（不是 0）；`MIN_REVIEWS=10` 不足只摆数字不出候选。只统计、不写台账，消费方是 `engine/tuning.suggest()` 的第三条规则与知识闭环页 |
| 优化史（这次改动提升了多少） | `app/services/optimization_log.py` | 表 `agent_optimizations` 一版一行、**追加不覆盖**（`agent_configs` 是覆盖式的 ⇒ 上一次生效的提示词与分数当场就没了，答不了"提升了多少"）；`accepted`（分数严格更高）与 `persisted`（真写成默认值）**分两个布尔**（合成一个就会出现"页面说已采纳、默认还是旧的"假绿）；`delta` 由服务端算，`score_before` 为空时给 `None` 不写 0；只存候选提示词的 `sha1` 指纹（全文仍在 `benchmarks/history/*.json`）；`target` 必填，现有四格 `system_prompt`（提示词优化器）、`params`（温度调优）、`skills`（技能绑定调优）与 `tools`（工具集调优），其余仍未开（留列不留假数据）。`mark_persisted` 的定位**必须带 `target`**（四格的 `version` 各自从 1 数 ⇒ 只按 agent+version+case 会把「提示词落库」标到温度那行，**不抛错、只标错人**），三处调用方都显式传且 `optimization_log_test` ④′（真跑跨格）/⑨′（全仓 AST 扫）双管。提示词格的 v0 基线是**现网在用的那份**（`agent_configs.system_prompt` 优先、无配置行才回退出厂默认），不是出厂默认——否则重跑会把更优的现网版本静默覆盖成更差的；`version` 是 (agent, case) 上的**轮次序号**、四格共用 `next_version` 递增（每次运行从 v1 数会让重跑的行撞号，`mark_persisted` 一次标中多行）。写口 `record`/`mark_persisted` **失败只 warn 不抛**（长跑任务不许被日志拖废），读口反过来必须报 `available=false`；写点只有四处：`agent/evaluation/optimizer.py::optimize_agent_prompt`（`target=system_prompt`）、`agent/evaluation/param_optimizer.py::optimize_agent_temperature`（`target=params`，HTTP 口 `POST /api/v1/evaluation/tune/{case_id}`）、`agent/evaluation/skill_optimizer.py::optimize_agent_skills`（`target=skills`，HTTP 口 `POST /api/v1/evaluation/tune-skills/{case_id}`）与 `agent/evaluation/tool_optimizer.py::optimize_agent_tools`（`target=tools`，HTTP 口 `POST /api/v1/evaluation/tune-tools/{case_id}`，**恒 `persisted=false`**——工具集真源是代码装配语句，这格只留结论）；四处分数都由评测器算，HTTP 不接受手写"提升了多少"。读口在 `routers/knowledge_loop.py` 的 `GET /optimizations`（本表本身**没有**读路由，`/api/v1/evaluation/optimizations` 是 501 占位）。见 `docs/knowledge-loop.md` §1 缺口 3 |
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
| 修复内核 | `app/services/repair_tools.py` | 提案/应用/回滚三件套：每处替换**唯一命中**、应用前全量备份、方案过期整单拒（源 `assistant-runtime/repair-tools.mjs`，2026-11-03） |
| 修复诊断 | `app/services/repair_diagnose.py` | 「现在缺什么、缺了堵住哪条工作流」：从工作流 loader 反推依赖 + 与本机盘面比对（源 `diagnostics.mjs` **重写**，只取摘要式设计，2026-11-03） |
| 修复闭环 | `app/services/repair_closure.py` | 「应用之后那个问题还在不在」：诊断快照只留缺口**键**（`kind:小写文件名`）＋`resolved/still/stale/unverified` 四档，明细被截断时一律 `unverified`（缺席可能只是没列出来）；结论按 `kind="repair"` 写进 `outcome_index`（知识闭环缺口 9） |
| 修复技能 | `app/services/repair_skill.py` | L0–L4 权能边界提示词，清单**从实现取值**（拒绝段/拒绝文件/可改前缀/各上限）不手抄（源 `assistant-runtime/skill.mjs` **重写**：只留纪律，剔除源项目运维手册，2026-11-03） |
| 修复域 | `app/routers/repair.py` | `/api/v1/repair/{workspace,source,search,propose,proposals,apply,rollback,diagnose,diagnose/history,verify,skill}`；提案落盘只回 `proposalId`（校验基准不出网）。闭环三处口位：`diagnose` 默认顺带落快照（`persist=false` 才不留痕）、`apply` 写盘成功后**当场复诊**并把结论挂进回执、`verify` 给「改动加了但东西要人手动放」的场合事后补判、`rollback` 把判定覆盖成终态 `rolled_back`（2026-11-03；闭环 2026-11-07） |
| 画布工具层 | `app/services/canvas_ops.py` | LLM 工具调用 → 画布 op 的编译器（**22 工具**，源 34 个里 12 个本仓无宿主、逐条写理由）。⭐ 常量与前端 `canvasNodeSize.ts`/`useCanvas.ts` **逐项同源**，由 `canvas_ops_test.py` 交叉断言（源 `canvas-agent/src/canvas/*.ts` **重写**，2026-11-03；2026-11-06 补 `audio`：节点类型/生成模式/尺寸表三处 + `canvas_generate_audio` 从「未迁」挪进「已迁」，因为 Music3 真出 wav 之后画布才有音频卡的宿主） |
| 画布工具域 | `app/routers/canvas.py` | `/api/v1/canvas/{tools,tool,ops/validate,ops/snapshot,next-x,agent,plugins,plugins/validate}`；**只编译 op 不碰画布**，执行在前端 `utils/canvasOps.ts`（2026-11-03） |
| 画布 Agent | `app/services/canvas_agent.py` | 一句话 → op（`POST /api/v1/canvas/agent`）。**工具无副作用**：只往 `sink` 攒 op，回模型的是「已登记 N 条」；只读工具没快照时如实报错（防模型凭空编 id）。复用本仓 `default_generate` + `get_text_config`，不启子进程、不连 MCP（源 `canvas-agent` 的 codex 子进程层**不迁**，2026-11-03） |
| 画布插件契约 | `app/services/canvas_plugins.py` | 5 个内置插件清单（源 6 个，`template` 起步样板不迁）+ 校验（id/type 命名空间/尺寸界/`RESERVED_METADATA_KEYS`/内置面板互斥）+ `/api/v1/canvas/plugins{,/validate}`。⭐ `hosted=true`：渲染槽已接（前端 `utils/canvasPlugins.ts` 清单 + `components/canvas/plugins/*` 五件，源 `plugins/canvas/*` **重写**，2026-11-03） |
| 画布插件渲染 | `frontend/app/utils/canvasPlugins.ts` + `components/canvas/plugins/*` | 清单（type/title/默认尺寸/默认键/`contentComponent`）与后端 `BUILTIN_PLUGINS` **逐项对拍**（`canvas_plugins_test.py` ③b，含 `.vue` 在盘与分发器登记）；`PluginNodeBody` 三态（正常 / 缺组件 / 未安装）；HTML `sandbox="allow-scripts"`＋默认 `pointer-events:none`、SVG 走 `<img>`、Markdown 自写渲染器（先转义再拼标签、链接白名单）；插件键落 `node.values`（2026-11-03） |
| 画布计划编译 | `app/services/canvas_plan.py` | 「意图 → 计划 → op」中间那道**人工确认**里的编译器（纯函数、可单测）：模型只答「放哪几个节点、谁接谁」（严格 schema），本模块确定性算坐标、生成稳定 id、做交叉校验、出 preflight 计数。四条硬口径：`goal` 逐字不动（不许把用户说的「雨夜客栈」润色成「夜晚的客栈」）／缺参**一次报全**（不给默认值糊过去，否则模型 8 步全耗在试探上）／坐标由编译器算（item 不给 `x/y` 就按 `options.layout` 排，让模型自由发挥的后果是节点叠成一坨）／**确定性 id**（同一份计划编两次得到同一批 id ⇒ 重放不长出两套节点）。`plan_digest` 是计划内容指纹、`planId` 就是它；守卫 `canvas_plan_test.py`（2026-11-08） |
| 画布计划存档 | `app/services/canvas_plan_store.py` | 计划先落盘成 draft（`ops` 存着但**不下发**）→ 用户看 preflight → 确认才重新编译下发。**确认时重编译**不是浪费：草案期间用户可能动了画布，用当下快照重编才能把「那条线要连的节点被删了」挡在确认这一步（落地后才发现就只剩一根断线）。幂等：`id = plan_digest`，同一份计划重复提交覆盖同一份草案、不长出第二条记录；重复确认返回上一次缓存的 ops 并标 `replayed: true`（前端据此不会把同一批节点建两遍）。计划档落 `<dataRoot>/canvas/plans/<id>.json`（`indent=1` 给人打开看草案的**证据**文件，已登记进 `route_parity_test` 的 `json.dumps` 豁免表）；`data/canvas/` 在 `.gitignore` 里（机器本地数据，`git add -A` 不许带进去） |
| 知识库页面 | `frontend/app/pages/knowledge.vue` + `composables/useApi.ts::knowledgeAPI` | 三栏：左目录树（`/tree` 计数，OCR/待补直接标在目录上）／中「浏览·检索」双态（浏览走 `/documents` 的 `q=` 文件名筛选 ＋ **真分页** `PAGE_SIZE=30`，检索走 `/search` 并按 `from` 标 `OCR`/`抽取`）／右抽屉翻 `/document` 分块原文（`hasMore` 续读）。**写能力**（2026-11-06 起，此前「只读」口径作废）：页签多 `已修改`/`回收站` 两档（覆盖层不在任何一份 sqlite 里，没这两档用户改完/删完就再也找不到）；抽屉里按 `GET /document/state` 决定按钮（`hardDeletable` 才画「彻底删除」，回收站里的件只留「还原」——画满按钮＝点了才报错）；块级编辑走 `POST /document/chunk`、撤销走 `DELETE`，`staleEdits` 当场明说「块号对不上、改动仍在覆盖层」；上传弹层拖拽＋点选同一件事（触屏没拖拽），本地校验口径由 `/overlay` 的 `upload` 下发，进度靠 XHR `upload.onprogress`（fetch 报不出进度）到 100% 后改说「服务器抽正文中」，逐文件三档结果分开摆（`待 OCR` 不说成失败）并注明「文件已在盘上」。检索结果为空时把 `sources.methods.parts` 的两库命中数摆出来（`0 条` 与「两库都没命中」是两件事） | 
| 知识闭环页面 | `frontend/app/pages/knowledge-loop.vue` + `composables/useApi.ts::knowledgeLoopAPI` | `/knowledge-loop`（导航「闭环」，桌面端逐字节同构）：左分区导航 5 项 + 分区懒加载（首屏只打 `/report`+`/scheduler`）——概览（事实层阶段表／台账分布／只读定时复盘／蒸馏候选，候选没有文字归因就直说「只有统计信号」并列出 run_id 凭据）、经验台账（三筛＋真分页＋**由 4 条白名单边生成**的迁移按钮、`lift:null` 显示「样本不足」、新建是**叠加**语义故不给编辑）、检索测试（左 form／右 result，`ranking=bm25\|like\|recent` 必须显示）、投喂与数据集（L1 预览原文＋`truncated` 警示／L3 导出集表格带 `itemsExists`、「核一遍」与训练 `datasetRef` 同一判据）、调参与训练（覆盖层只对 `source!=='default'` 给清除、只读候选 `canApply` 恒 false 只给「按这条写覆盖」、产物不自动生效）、质检命中率与人工复核（按 code 的触发/复核/判对/误报/漏报两率一表，`—` 表示还没判过而**不是 0**；待复核列表把最近 QC 记录直接摆出来，每条 code 四个按钮＝判对/误报/漏报/撤销，`pending` 未判完的行比已判完的该被看）。结构复用 `reference/dify` 的分区 shell 与 `reference/ragflow` 的数据集列／检索双栏／抽屉详情，差异与约束见 `docs/knowledge-loop.md` §10（2026-11-07）。⭐ 接口层 **2026-11-08 才补齐**：此前 `useApi.ts` 从不导出 `knowledgeLoopAPI`（`git log -S` 为空）与 12 个类型，页面 `import` 拿到 `undefined` ⇒ **一进来就 `TypeError` 白屏**，而静态守卫全绿、只有 `typecheck`（TS2724）会说。现 36 个口逐一对应后端路由（`api.patch` 是这轮新加的动词，后端只提供 `PATCH` 不给 `PUT`），守卫＝`frontend_api_coverage_test.py`（14 判据／262 调用点／37 个带 query 键的调用点）＋ 36 个口 `TestClient` 各实打一次。⭐ 同日第二件：`entry_links` 自蒸馏上线起**只写不读**（`detail` 也不回邻居）⇒ 补 `knowledge_ledger.related/for_context/context_refs` 三个读侧函数＋两条只读端点＋Agent 工具 `context_knowledge`＋详情面板「挂载／相关知识」（`opened.links` 此前也是只传不显示），两档 `via` 的证据强度在类型与界面里都分开声明；`npm run typecheck` 两端零错、`check:parity` 绿，详见 `docs/knowledge-loop.md` §11 |
| 厂商连通性探测 | `app/services/provider_probe.py` | URL 拼接 + 探测描述 |
| 视频时长探测 | `app/services/video_probe.py` | 15 行 |

### L3-8 开跑前体检（2）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 体检取数（DB 侧） | `app/services/preflight_source.py` | 按 episodeId 组装输入 |
| 体检纯逻辑 | `app/services/production_preflight.py` | 零依赖，四块串一条 |

### L3-9 外部执行通道（20）

「把活派出去」的三种通道：ComfyUI、Ollama、云端供应商。

| 功能点 | 落点 | 说明 |
|---|---|---|
| ComfyUI 工作流运行 | `app/services/comfyui.py` | 项目自己的一等公民能力 |
| ComfyUI 渲染目标 | `app/services/comfyui_targets.py` | 图片记录 → target 的单一真源 |
| ComfyUI 工作流目录 | `app/services/comfyui_catalog.py` | `configs/comfyui/workflows` 读取口：注册表 / 变体展开 / API 格式判定 / 步数估算（2026-11-02 移植） |
| ComfyUI 提示词模板 | `app/services/comfyui_prompts.py` | `configs/comfyui/prompts` 四份（H3 增强 / 生图增强 / 两档图片反推）：参数落点声明 + **按上下文算的两格**（`输入模式` 有图=图片、`任务模式` 有图=图生图）+ 参考图接成 `LoadImage(9101+i)`；只**编译**成可提交 API 图，不提交不连上游，故离线可用；拼错的键 / 非法任务类型 / 超长提示词 / 超 8 张参考图**一律当场拒**（静默忽略=按默认值跑 ✗）；模板每次重读 ⇒ H3 的 `系统提示词` 追加语言指令与输出锁不会越叠越长；守卫 `tests/comfyui_prompts_test.py`（2026-11-03 自 `comfyui-gateway/src/prompt-optimizer.js` 迁入） |
| ComfyUI 参数注入 | `app/services/comfyui_bindings.py` | `bindings` 语义：9 种 transform + 参考素材的三种挂法（文件名 / 连线 / 多参考） |
| ComfyUI 参数落点推断 | `app/services/comfyui_infer.py` | **未登记**的 API 图 ⇒ `type/bindings/parameters/warnings`（与 registry 同形状，人工过目后登记）；判据优先级：顺 `positive/negative` 连线反查真文本格（含第二跳拼装节点）> 标题/顺序兜底；**图上本来是连线的格一律不绑**（拆线=产物变样且不报错）、正负向两套键（`TextEncodeQwenImage21` 是同节点两格）、负向保留原值作默认；采样器/调度器与族专用旋钮**不绑**（值域在节点定义里/只有该族知道）；端点 `POST /api/v1/comfyui/workflows/infer`（UI 格式 / 无输出节点 / `type` 别名当场 400，前缀 `[ui_format]` 等可机判）；守卫 `tests/comfyui_infer_test.py` + `tests/comfyui_infer_route_test.py`（2026-11-03 自 `comfyui-gateway/src/registry.js` 迁入，与 21 条已登记工作流逐格对拍） |
| ComfyUI 运行保留清理 | `app/services/comfyui.py::purge_runs` + `app/scripts/comfyui_retention.py` | 过期运行记录与它们的本地产物一起清：**在跑的（`queued/processing`）一律不动**、**行与产物同生共死**（产物删不掉 ⇒ 记录留 ⇒ 线索不丢）、共享产物（同路径还被别的记录引用）不删、`static/comfyui/` 之外的路径整行留着；默认干跑只算账（记录数/文件数/字节/`missingFiles`），`--apply` 才真删；只读预演端点 `GET /api/v1/comfyui/runs/retention?days=` **恒干跑**（能真删的口子只留 CLI 一个，删是不可逆的，不做成能在地址栏里点的 GET），且该路由必须声明在 `/runs/{run_id}` **之前**（排后面会被当 id ⇒ 400「run_id 非法」）；守卫 `tests/comfyui_retention_test.py`（含 ⑫ 系列：恒干跑 / 路由顺序 / 跑完文件与记录都在 / `days=0` 一条不清） |
| ComfyUI 模型槽位 | `app/services/comfyui_slots.py` | 节点输入 → 模型类别（18 条）+ 从 `/object_info` 动态发现 + 可用性判定 |
| ComfyUI 模型名匹配 | `app/services/comfyui_match.py` | 七条红线（格式/角色/限定词/代次/规模/训练规格/家族）+ Dice·Jaccard 相似度；歧义不自动选 |
| ComfyUI UI 图转换 | `app/services/comfyui_ui_workflow.py` | UI 格式（nodes+links）→ API 格式；Reroute/Primitive 穿透、旁路直连、widget 顺序表、游离节点剪枝 |
| H3 说话人标签 | `app/services/comfyui_speaker_tags.py` | 描述段内补 `(S1)` 说话人 id（同主体复用）、画外音标 `off-screen` 且说明只加一次 |
| 媒体尺寸换算 | `app/services/media_size.py` | `2k`/`16:9`/`auto` → 像素（预设表）、比例与档位反推、视频「短边=分辨率」+ 偶数化；已接进 `bindings.build_context` |
| ComfyUI LoRA 链 | `app/services/comfyui_lora_chain.py` | `loras` 清单 → `LoraLoaderModelOnly` 串联（模型源按引用数选、下游全量重接、id 从 9000 起）；已接进 `comfyui._build_upstream_body` |
| ComfyUI 进度 / 预览桥 | `app/services/comfyui_progress.py`、`app/routers/comfyui_events.py` | `progress_state` 按节点 value/max **加权**百分比 + 采样预览帧（只留最新 + 10 分钟 TTL）；`GET /api/v1/comfyui/runs/{id}/progress` 快照、`/events` SSE 增量；`websockets` 缺失时**降级**为轮询状态并如实 `progressAvailable=false`；已接进 `comfyui._process_run` 的轮询循环与取消路径 |
| ComfyUI 注册表生成链路 | `app/services/comfyui.py`（`build_registry_graph` / `stage_reference_uploads` / `validate_workflow_body`）、`frontend/app/composables/useComfyRun.ts` | 前端只传 `workflowId`+`values`，服务层按注册表 `bindings` 生成图并注入（不让前端复刻拼图 ✗）；画布连线/同组素材先上传到 ComfyUI input 再落 `LoadImage`；提交前调上游 `/v1/workflows/validate`（缺节点包/悬空连线当场报，不占 GPU 空等）；SSE 收尾事件名 `error`→`failed`（避开 `EventSource` 连接错误同名词）；画布配置卡 = 工作流面板 + 进度条 + 采样预览 + 产物自动回卡。⚠️ **图与注入必须认同一个变体**：`load_graph(entry, variant)` 读的是变体那份，`apply_bindings` 就必须收 `resolve_variant(entry, variant)` 的结果 —— 传主 entry 会让变体自己声明的 `bindings`/`parameters` 一条都不生效，症状是「选了 inpaint 却按 t2i 跑、只给一张图也不报缺蒙版」，且**一行错都不报** ✗✗（蒙版重绘 / 扩图两个变体就是这么坏的，2026-11-03 修，守卫 `tests/comfyui_inpaint_test.py`） |
| ComfyUI 失败归因 / 显存重试 | `app/services/comfyui.py`（`failure_hints` / `describe_failure` / `is_out_of_memory` / `_submit_and_poll`） | 上游原文**一字不改** + 追加一条能直接动手的建议（缺模型 / 缺节点包 / 缺 SageAttention / VDN 缺 stage-B / 显存不足 / 上游不可达；**认不出就只留原文**，不硬编原因 ✗）；上游报显存不足 ⇒ `POST /v1/free` 卸载后**重提交一次**，成功也落 `warnings` 并把用量 `retryCount` 如实记 1（**只重试一次**：每次重提交都是一次真 GPU 占用 ✗；`COMFYUI_OOM_RETRY_WAIT` 默认 1.5s）。另：参考素材预处理只服务 `kind=workflow`，H3 走自己的 `referenceImageUrls` 协议（否则远端取不到会在提交前自己打掉一条本来能跑的请求 ✗）（2026-11-03 自 `comfyui-gateway/src/runner.js` 迁入） |
| ComfyUI 模型名匹配 | `app/services/comfyui_match.py` | `model-match` 语义（七条红线 + 相似度排序）：工作流里的作者文件名 ↔ 盘上另一档，`safetensors`/`gguf` 不互顶、角色与语义限定词必须同集（2026-11-02 移植） |
| ComfyUI UI→API 转换 | `app/services/comfyui_ui_workflow.py` | `ui-workflow` 语义（自研重写）：界面导出的 `nodes[]`/`links[]`/`widgets_values` → `POST /prompt` 吃的 API 格式；`Note`/组开关/Reroute/`PrimitiveNode`/`mode`（2 静音、4 旁路）六处「转得出来但图是错的」保真点 |
| H3 说话人标签 | `app/services/comfyui_speaker_tags.py` | `speaker-tags` 语义（自研重写）：H3 音视频同生成的 `(S1)` 主体标签校验与补齐 —— 只动描述段、不动音响/配乐段，同主体复用同一 id，`off-screen` 只加一次 |
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
| 数据存储 | `app/services/data_storage.py` | 数据根信息与切换（`/api/v1/storage/{info,change}`；**已接线**，见 §10.3） |
| 文件存储 | `app/services/file_storage.py` | 移植 `utils/storage.ts` |
| 资源库共享实现 | `app/services/resource_library.py` | 角色/场景/兵器/服装模板 |
| 资产清单 + 验收门 | `app/services/asset_manifest.py` | 缺件即阻断 |
| 验收报告合同 | `app/services/acceptance_report.py` | 「这一件产物凭什么算过」：`plan` 先声明 + `cases` 挂证据介质 + **覆盖率门禁改判** + 没验到至多 `partial`（`POST /api/v1/assets/acceptance/validate`；纯校验不碰 DB） |
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

### L3-13 知识闭环（5）

「跑过的活 → 可查的事实 → 能执行的判断 → 真的改行为」这条链，四段都是**派生表 + 一条回执纪律**：
每段都能只靠盘上文件重建（A 侧），不往 `darme.db` 里塞（那是 B 侧，进备份、进 `db_upgrade`）。

| 功能点 | 落点 | 说明 |
|---|---|---|
| 跑次归一（事实层） | `app/services/outcome_index.py` | `outcome-index.sqlite` 的 `runs` 表：把 traces / `video_quality_checks` / `comfyui_runs` / `api_usage` 归成同一形状（`run_id/kind/scope/status/score/cost`）；`index_state` 存 `(mtime,size)` 指纹增量重建；**没写 `status` 的记录直接拒收**（塞 `partial` 会稀释失败率）；生产写入点只有两个 —— `engine/runtime.py::_index_run`（引擎任务封口，锁外）与 `technical_qc.py::_index_qc_run`（质检写回后，`run_id` 确定性覆盖） |
| 经验台账（知识层） | `app/services/knowledge_ledger.py` | `entries/entry_links/entry_usage` + FTS5 trigram；`upsert` 证据核不过**整条拒**；`search` 三态 `ranking=bm25(<3 字退 like)/recent`，命中数 vs 返回数分开报 |
| 确定性蒸馏 | `app/services/knowledge_distill.py` | 四条信号（失败聚集 / 分数高分离 / 返工热点 / 成本离群）纯规则算，LLM 只负责「归纳成文」；无 LLM 时降级成模板条目且 `confidence` 封顶 0.5 |
| 复盘定时器 | `app/services/knowledge_scheduler.py` | 每日 `HH:MM` 自动跑一次蒸馏（`knowledge.distill.enabled`，**默认关**）：串行 + 防重入（`skipped=running` 且不覆盖 `lastResult`）+ 失败隔离（蒸馏与 `lift` 各算一步）+ `running` 在 **`finally`** 里复位（`evaluation_scheduler` 缺这段、抛错即永久停摆）；**只写 `proposed` 且不调模型**（无人值守不花钱、不自动改结论）；`lift` 另有独立开关（它会把旧条目降级）；状态经 `GET /api/v1/knowledge-loop/scheduler` 如实透出（关着也答 `enabled=false`） |
| 投喂与数据集 | `app/services/knowledge_feed.py` | L1 `compile_for_prompt`（1200 字预算，只给标题+首句）/ `record_injection`（`lift` 的唯一数据源）/ L3 `export_*_dataset`（负样本只进 eval 集）/ `resolve_dataset`＋`list_datasets`（`datasetRef` 的**唯一判据**：条数·角色·权利与 manifest 逐项对账，**再按 `items.jsonl` 现算 `sha256` 与 `contentSha256` 比字节** —— `fingerprint` 只对 `source` 取值，改正文它看不出来；老 manifest 无该字段时退到重算指纹并在 `integrity` 里如实标低一档），缺 `source`/`rights`、空集、被手改一律拒 |
| 训练登记表 | `app/services/training_registry.py` | `training-registry.sqlite`：任务与产物**重启后仍可查**（运行时任务表是内存 dict）；产物一律 `candidate`，`promote` 才成默认且要 `basis`+证据，同类只留一个默认档；`problems()` 点名「默认档文件被删」「永远 running 的僵尸」「数据集样本文件不在盘上（血统断了）」；`dataset_json` 两表都存（`CREATE TABLE IF NOT EXISTS` 不改老库，靠 `_ensure_columns` 补列） |

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
| 噪声调度（sigma 序列；`SCHEDULES` = `flow`（默认，t 等分＋shift）／`karras`／`normal`／`exponential`／`sgm_uniform`／`ddim_uniform` ＋ **`simple`**（1000 格模型梯子上等距取点，`steps` 整除 1000 时与 `flow` 逐位相同）／**`beta`**（`beta_ppf(1−i/steps)` 取点，零依赖 `_betacf`＋二分反演；`rint` 撞车合并 ⇒ 长度可 < steps+1）；两个新档都吃 `shift` ⇒ 走 flow 域梯子，不给 `shift` 才是几何梯子（σ 到 14.6，EDM 域）） | `app/services/engine/schedules.py` |
| 几何数学（分辨率/帧数/潜空间形状；`latent_frames` 对 `frames < 1` / 非整数**当场拒**——原来 `max(1, int(frames))` 会把「没算出来帧数」和「1 帧视频」压成同一个值，下游只看到"怎么只有一拍"） | `app/services/engine/geometry.py` |
| CFG 引导 + 重标定 + 步区间 | `app/services/engine/guidance.py` |
| 采样器（扩散/流匹配去噪循环；`SAMPLERS` = `euler`／`heun`／`dpmpp_2m`／`dpmpp_2s_ancestral`／`dpmpp_sde`／`lms`／`euler_ancestral` ＋ **`res_multistep`**（φ₁/φ₂ 指数积分，`multistep_step` 是**唯一**符号真源，双流也调它）／**`er_sde`**（`scaler(v)=v(e^{v^0.3}+10)` 200 点积分＋三级 stage；`logsnr` 默认 `flow`）／**`lcm`**（每步直接取 denoised 再按 σ_next 回噪，不是 ODE）；`STOCHASTIC` 决定要不要 `noise_like`） | `app/services/engine/sampler.py` |
| 长视频分段计划 | `app/services/engine/segments.py` |
| 首/尾帧下标规划 | `app/services/engine/keyframes.py` |
| 条件掩码（首帧/参考图那一半；**数目类参数一律走 `_as_int` 严格整数**：整值浮点 `2.0` 照收、`2.5` / `bool` 拒——截断会把「2.5 路参考」静默当成「2 路」，且 `max(0, …)` 式夹取会让紧随其后的 `if x < 0` 判据永远进不去） | `app/services/engine/conditioning.py` |
| 段级音频合成（纯数学） | `app/services/engine/audio_mix.py` |
| 参考素材指纹（哪段可跳过） | `app/services/engine/cache_key.py` |
| 加速链（串成配置而非代码） | `app/services/engine/accel_chain.py` |
| **VAE 分块缩放/解码**（`tile`/`overlap` 走**像素**单位＝`VAEDecodeTiled` 同义；权重累积＋除法归一 ⇒ 重叠带精确平均；`overlap<tile`、不足 1 潜像素、`tile` 过小都**报错**；只切空间、不切时间维；`decode_latents_tiled` 的 `out_channels` **必须由解码器给**——VAE 是「4 潜通道 → 3/4 像素通道」，不传就按输入通道核每块形状、当场报。「Qwen-Image 2.1 那族为什么不接分块」有实测结论：`decoder.middle` 是**全图单头注意力**，潜变量切块 ⇒ 整张都变（真权重相对差 1.35），ComfyUI 的 `WanVAE` 也没有 `decode_tiled`；要省显存只能切注意力**之后**的 up 块 —— 那一条**已做**在 `vae_wan`：`WanVAE.decode(tile=, halo=, attn_chunk=)` 两个旋钮各治一半（8× 那族峰值几乎全在全图注意力、2.1 那族几乎全在 up 链），实测 2048×1536：8× 20.96→3.94(attn)→1.03 GiB(tile)、2.1 8.69→3.21 GiB；**两条都不是逐位**（tile 5.8e-03，halo 消不掉；attn 只在 chunk 与 SDPA 内部分块对齐时才逐位）⇒ 默认全关、`TorchBackend.set_vae_tiling` 才开、OOM 兜底才自动降档，回执带 `vaeTiling`；**外部入口已开**：`POST /engine/load` 收 `vaeTiling`，归一只有 `normalize_vae_tiling` 一处，非法值 400，给了值才进装载键） | `app/services/engine/vae_tiling.py` |

### L4-2 `format` 容器与像素（9）

读得进、写得出。不做模型数学。

| 功能点 | 落点 |
|---|---|
| safetensors 读取器（纯 Python） | `app/services/engine/safetensors.py` |
| GGUF 读取器（纯 Python） | `app/services/engine/gguf.py` |
| 权重内嵌元数据契约 | `app/services/engine/checkpoint_meta.py` |
| 潜空间口径表 | `app/services/engine/latent_formats.py` |
| 联合音视频潜变量互操作 | `app/services/engine/latent_container.py` |
| 落盘（帧→mp4 / 音→wav；**双流音轨混进同一个 mp4**（`apad`＋`-t` 以画面收尾，wav 仍原样另存）、**PNG 写参数 iTXt**、**纯色图判 degraded**） | `app/services/engine/media.py` |
| **产物取证与落点**（每个产物目录一页 `manifest.json`：request/plan/装配摘要/ffprobe 事实/sha256；任务没给落点 ⇒ `<static>/engine/<kind>-<taskId>/`，**干跑** ⇒ `<数据根>/engine-dryrun/<kind>-<taskId>/`（不进静态 ⇒ 永不发 `primaryUrl`）；`kind_of_path` 是"什么算媒体产物"的唯一判据） | `app/services/engine/torch_backend.py` · `app/services/engine/runtime.py` · `app/services/engine/bridge.py` |
| 图像算子集（纯 torch，不引 kornia/scipy） | `app/services/engine/image_ops.py` |
| **Qwen3-VL 像素预处理**（H3 图片/视频块 ⇒ `(t,h,w)` 网格：resize 到 **32** 倍数、`(v−0.5)/0.5` 归一化、**merge 块优先**的行序、2 帧进时间维；只做像素不做模型数学 ⇒ 不挂 `model` 组） | `app/services/engine/qwen3vl_pixels.py` |
| 缓存/权重文件完整性守卫 | `app/services/engine/cache_guard.py` |

### L4-3 `weight` 量化与装载（11）

「文件里的字节 ⇒ 能算的张量」。不决定跑哪一步。

| 功能点 | 落点 |
|---|---|
| 反量化（fp8/int8/int4） | `app/services/engine/quant.py` |
| NVFP4（E2M1 + e4m3 块 scale；**另有常驻壳** `NvFp4Linear` ＝ 打包字节原样留卡、反量化挪到**前向逐层**、AWQ `pre_quant_scale` 折进权重列 ⇒ 14.61 GiB 的 32B 文本塔装得下；`attach_h3_text` 按**文件自述**换壳，档位判 `nvfp4` ⇒ 驻留 **15.33 GiB** 而不是 int8 那本的 25.44。**逐层反量化的真账与边界**（单层 5120×12800 微基准 **1.42 ms** ＝ 两字节查表 0.73 ＋ 块 scale 还原 0.07 ＋ 乘 0.62；**真机 350 层逐层实测 631 ms/次 encode**、吞吐 **99–115 GB/s 且与张量大小无关**（>64M 元素 150 层 99、16–64M 100 层 98、4–16M 100 层 85）⇒ 病根**不是**小 kernel，而是**四个 kernel 串联的 HBM 往返**（`to(int32)` 98 MB ＋ `embedding` 197 MB ＋ 块 scale 8 MB ＋ 乘 266 MB ≈ 569 MB，而理论最少只要 164 MB）。A−B 实测（29/128/512 token）：反量化 **571/577/625 ms**，占 encode **76%/74%/59%** —— 与 token 数近乎无关（按权重搬运付费 ✓✗）。三处纯性能选择各配一条判据：索引 **int32 不是 int64**（−0.4 ms/层）、**`uint16` 零扩展不是 `int16`+`& 0xFFFF`**（−0.24）、**两字节一次查表不是逐字节**（−0.24 —— 后两条数值全相同 ⇒ 只能源码判据盯 ✓✗）；**纯 torch 已到天花板**：要突破必须把「查表→乘 scale」融成单 kernel ⇒ 本机无 triton（`torch.compile` 用不了）且无 nvcc/cl（写不了 CUDA 扩展）⇒ 如实停在这里；且 H3 真跑一次 318 秒，此处满打满算也只省 0.37 秒 ⇒ 不为它引编译链；实测量法的脚本在 `data/experiments/nvfp4_dequant_bench/`：`bench.py` 单层分项、`bench_index.py` 索引路径对照、`bench_fused.py` 融合可行性、`bench_encode.py` 真权重 A−B、`bench_layers.py` 逐层吞吐分布） | `app/services/engine/quant_nvfp4.py` · `app/services/engine/h3_form.py` |
| 权重装载进 `nn.Module` | `app/services/engine/weights.py` |
| 跨族张量名映射 | `app/services/engine/mappings.py` |
| 混合加载（fl2va 基底 + ref2va adaLN） | `app/services/engine/hybrid_load.py` |
| 混合加载合并 | `app/services/engine/hybrid_merge.py` |
| **反量化产物落盘缓存的命中账（§4.4）**（`stats()` 五个计数各指一件事：`hits` 复用／`misses` 判了不能用／`written` 真写成／`refused` 余量·配额不足**一字节没写**／`failed` 写到一半失败 —— **不许合成「命中率」**：把 `refused` 并进 `misses` 会把「该清盘」读成「该重算」；`store_usage()` 报盘上体积（产物个数/字节＋sidecar 个数＋配额＋余量，精度 **1 KB** 不是 1 MB——3 位小数会让几 MB 的缓存显示 `0.0`）；`cache_report()` 两者并成一格给 `/engine/runtime` 的 `diskCache`。`enabled=false` ＝**总开关关着**≠「没命中」；`limitGiB=null` ＝**不限**≠「限了 0」；`hits` 是**本进程**的，重启清零） | `app/services/engine/materialize.py` |
| **反量化产物落盘缓存（G3）**（把「反量化＋cast 算出来的**目标 dtype 字节**」写成一份正常 safetensors ⇒ 第二次装载只是读盘、**一次反量化都不做** ✓。键 = `sha256(schema, converter, dtype, 源大小, 头 64KB 哈希)` 前 12 位 ✓ —— **刻意不含 mtime** ✗✗（含了 `touch` 一下就换文件名 ⇒ 越用越涨）；mtime 只在 sidecar 里由 `cache_guard.is_cache_valid` 比 ✓。四条失效缺一不可（产物不在 / 头不自证 / sidecar 指纹不符 / 元数据自述 converter·dtype 不符）；写侧 `.tmp` → `os.replace` 原子替换、键集与字节数逐个核 ⇒ 少写一个张量即 `MaterializeError` 且**不留半成品**；配额（`DARME_MATERIALIZE_LIMIT_GIB`）与 5% 磁盘余量不足 ⇒ `MaterializeSkip` **退化成「慢」不是「错」**（`DARME_MATERIALIZE=0/off` 关、默认开、目录 `DARME_MATERIALIZE_DIR`）。⚠️ 数值不变是**构造**出来的：写盘字节取自真装进模型那张张量的原始字节（`.view(torch.int16)`）⇒ 命中与现算逐位相等 ✓（自己写「fp32 取高 16 位」是向零舍入 ⇒ 差 1 ulp ⇒ 两个模型 ✗✗）） | `app/services/engine/materialize.py` |
| 档位表（步数/分辨率/加速件/显存建议） | `app/services/engine/tiers.py` |
| LoRA 装配 | `app/services/engine/lora.py` |
| LoRA 内容身份 + 引用计数 | `app/services/engine/lora_cache.py` |
| **无缓冲重叠读**（Windows：`FILE_FLAG_NO_BUFFERING\|OVERLAPPED` ＋ 每槽手动复位事件 ＋ 扇区对齐槽区用 `slop` 吃掉非对齐偏移；非 Windows/`force_plain` 退回普通读，降级原因写进 `describe()`；越界当场拒。**已接进 `safetensors.TensorReader`**：Windows 默认走它、建不起来退回 seek+read 并报 `ioNote`；真权重 11.46 GiB/453 张量实测 2.88 ⇒ 3.43 GiB/s，逐字节相同） | `app/services/engine/win_io.py` |
| **预取单槽**（提交永不抛、取走即作废、槽满当场拒 `busy`、`snapshot()` 报在飞/就绪/命中/错误。**已接进装载**：`weights.load_module_weights(read_ahead=None)` 边建当前张量边让后台读后面几件，作用域由 `_prefetch_scope` 管、出函数 `close(wait=True)`；单件超 `_READ_AHEAD_MAX_BYTES`(256 MiB) 一律就地读；2 GiB/64 件实测 2.7 ⇒ 3.9 GiB/s。**深度不再写死**（§4.3）：`resolve_depth()` 优先级 显式 > `DARME_READ_AHEAD` > **盘速档**（`DISK_CLASSES`：nvme 2/上限 4、sata 2/2、hdd 1/1、unknown 2/2），`probe_disk()` 真读 32 MiB 顺序读（**1 MiB 块**：`read()` 先申请新的再释放旧的 ⇒ 探测峰值 ＝ 2×块大小）＋每块盘只量一次；显式值超该盘档上限 ⇒ 截断并报 `capped`；量不出盘速 ⇒ `unknown` ⇒ 2 ⇒ **老行为零变化**，不是猜新数） | `app/services/engine/prefetch.py` |
| **任务分配与压力调度**（零 torch 纯函数：**GPU 优先** ⇒ 装不下就 `gpuBytes+cpuBytes` **正好等于**需求的协同切分 ⇒ 两边都不够就 `InsufficientResources` 拒跑（不靠交换文件硬跑）；协同后**第二台设备也要核**（`host_memory`/`host_room_bytes`/`ensure_host_room`，无 `psutil` 依赖）；`Ledger` 记「谁在占、占多少」（卡/内存两栏，**不是判据**）；`LoadWatch` 三档滞回（88%/95%、连续 2 次才认 `hard`）只管**回收**不搬权重；`reclaim_for` 是纯函数（阶段表没登记 ⇒ 一个都不退）） | `app/services/engine/scheduler.py` |
| **GPU + CPU 联合驻留**（按显存预算切连续前缀层、`split` 搬激活 / `park` 搬权重、无卡/null 如实说；**对外层数旋钮 `gpuLayers`**（§4.1）：`normalize_gpu_layers` 单一真源 —— `null`/空/`auto`＝没提（按预算自算）、`0`＝整份留内存（`refused=False`）、`n`＝前 n 层上卡；预算只当**上限** ⇒ `granted<asked` 时 `capped=True` 但 **`fits` 仍 True**（置 False 会被 `place` 整份退回 CPU）；非层参数先上卡；拒 llama.cpp 的 `-1` 与 `true`/`3.5`/`"abc"`（400）；**余量按件分档** `RESERVE_BY_KIND = {dit 0.25, te 0.05, vae 0.05}` —— 原先 `split` 一律留 25%、`off` 只留 1 GiB，同一张卡两套口径（空载 20.71 GiB 时 `off` 可用 19.71、`split` 只 15.53，差 4.2 GiB），老调用方不传 `kind` 仍 25% ⇒ 行为零变化） | `app/services/engine/residency.py` |
| **内存这一级（三级内存的中段）**（`HostCache`：**按件一本**（`dit` 与 `te` 的组名都叫 `blocks.0`，合成一本会卸掉对方的层）；配额 = `min(quota_of(总内存×75%), host_room_bytes())`，读数取不到 ⇒ `None` ⇒ 这一级**不启用**（不拿假数卸层）；超配额按 **LRU 只卸层**（钉住的非层组永不动）、卸不够就**一件都不卸**；参数换 **0 元素占位**（不 `del`/不 `setattr(None)`，Parameter 对象自始至终是同一个）；`residency._host_prepare` **快照先于卸载**（否则重读拿 `(0,)` 当基准）并在 `guard` **之前**执行（guard 核的是实际要占的 `cpu_bytes`）；`weights._load_streaming` 读 `_residency_spilled` 跳过卸过的键（**单独记 `skipped`，不混进 `missing`；也不许判成 `shapeMismatch`**），装完把还原器经 `ReloadSlot` 接回 `_residency_reload`（13 处装载调用点一行不改）；⚠️ 卸下去的键里**本来是 int8 常驻层**的那几层另记 `LoadReport.int8Skipped`（2026-11-07）：`_int8_residency_problems` 的分母是「写进去的 ＋ 卸下去的」，只取 `int8Resident` 会**卸得越多越判成装不上**（假 8 GiB 卡装 Krea2 实测 88/224 整份被拒，而 `residency="off"` 同机同文件 5.7 s 装得进）—— 但 `int8Skipped>0` 且 `restorer is None` ⇒ 那几层是**填不回去的洞**，照样拒装；前向进层前 `ensure` 重读 → 形状/dtype 对快照 → 写回 ⇒ 数值与整份常驻**逐元素相同**；重读用**第二把**句柄（装载那把随 `with` 关了 ⇒ 拿它读只会得到「读取已关闭」），`Handle.detach` 回收） | `app/services/engine/host_cache.py` |
| **层预取上卡（`foresight_swap` ✓，默认 0 槽＝不存在这一级）**（算落在 CPU 上的第 N 层时，同时在**独立拷流**`torch.cuda.Stream` 上把第 N+1 层的权重拷进**额外槽**（`empty_like`，与常驻层分开记账）；进 N+1 层**先问落地没**（`Event.query()`）⇒ 落地了 `swap_in` 把 `param.data` 就地换成槽里那份、那层在**卡上**算（输入不往下搬）、`after` 换回 CPU 那份并 `release` 腾槽；**没落地就照原计划在 CPU 上算**——绝不等（不 `synchronize`/`wait_stream`）、绝不拿半份数据算（两批交错落地＝不报错的错数）。⚠️ 槽是**计划之外**的显存 ⇒ `place()` **先从预算里扣**（21 GiB 卡上不扣 ⇒ 计划排满＋多要一层 ⇒ 装配正常、首次预取 OOM 且报错指向 `empty_like`），扣完比预算一半还大就**不启用**并写明理由；`slots<2` 也能跑但退化成「先拷完再算」（`busy` 涨满＝没有重叠的证据）；判据（该不该发／落地没落地／能不能用／`busy`·`overBudget`·`pending`·`missed` 分账）纯逻辑不碰 torch，分配与拷贝在 `residency._SlotBook`；`mode≠split`／主设备不是卡／CPU 侧不到两层 一律不建槽；`report()["foresight"]=None`＝**没开**，字段在而 `usedLayers=0`＝开了没等到，两者必须分得开） | `app/services/engine/foresight.py` |
| **设备策略**（默认 GPU 优先 ✓；每次装载前现测显存 ⇒ 够就上卡 / 差一点改联合 / 差太多转 CPU；`auto`·`gpu`·`cpu`·`balanced` 四档，判据只有显存占用率、`utilPercent` 留 nvml 注入位。⚠️ **`auto` 只给 `off`/`split` 两档**（`AUTO_RESIDENCIES`），**不选 `park`** ✗：park 每步单向搬 `plan.layer_bytes`＝**全部层权重**（2512 那族 ≈38 GiB、双向 76，PCIe 4.0 ~25 GB/s ⇒ 3 秒+/步），split 同一步搬的是一条**跨界激活**（MB 级），而 `Plan.transfer_bytes("split")` 明写返回 `None` —— 不知道激活多大、**不猜**。⇒ park 留手工档 `--residency park`，选 `split` 必须如实标 `degraded=True`（兜底不是优化）） | `app/services/engine/device_policy.py` |
| **任务级可行性预检（G2，`GET /api/v1/engine/feasibility`）**（**不装载**就答「这台机器跑不跑得动这条链」⇒ `verdict` 四档 `perfect`/`good`/`marginal`/`too_tight`。逐件需求**只走** `torch_backend.stage_part_need`（装配线同一个出口），表值**只走** `TorchBackend.STAGE_NEED_TABLES`（**引用**那几本 `*_PART_NEED_GIB`、`is` 相等 ⇒ 没有第二本账）；「峰值」逐档登记 `STAGE_PEAK`：整链同驻求和 / 分时换装取最大 —— 2.1 峰值 14.79（三件之和 28.68 不能当门槛）、music3 真同驻 13.35、wan22 双专家不叠加。⚠️ 卡与**内存**两栏都判（split 也要内存放溢出的那部分 ⇒ 8 GiB 卡 + 32 GB 内存如实 `too_tight`）；⚠️ 激活 `basis="unknown"` **不编数** ⇒ `perfect` 降一档报 `good` 并标 `downgraded`（`docs/low-resource-plan.md` 方案 A）；`capacityGiB` 假装小卡时按空卡算并标 `assumed`，低配模拟 `DARME_FAKE_GPU_TOTAL_GIB` 同一条路） | `app/services/engine/feasibility.py` |
| **机器档（G1，`POST /load` 的 `profile` / `GET /engine/runtime?profile=`）**（一句话「8 GiB 卡按最低配跑」⇒ 一组旋钮的**唯一出处**：五档 `auto`/`gpu`/`l1`(8 GiB 卡·32 GB 内存)/`l2`(6 GiB·16 GB)/`cpu`，别名（`8g`/`6g`/`nocuda`）只有一张表、**不认的当场抛**（不静默回落 `auto` ✗ —— 那会让「我选了 L1」变成假话）；`resolve()` 只把**相对默认真会改**的项放进 `applied`（`auto`/`gpu` 是 `{}`），显式 `policy` / `vaeTiling.temporalChunk` 与档位冲突 ⇒ 剔除该项并标 `profileIgnored=true`；四项**明确不接**（`reserveFor`/`gpuLayers`/`readAhead`/`foresight`，随回执 `notWired` 带出）；`gpuGIB`/`hostGIB` 只是**假设**，绝不覆盖 `device_policy.probe()`；`budgets()` 一处回答「这一档下卡/内存各多少能用」并如实标 `source`（`cuda+host`/`cuda`/`host`/`none`，测不到 `null` **不写 0**）；`profile` 进**装载键**（换了档必须重装，否则复用旧布局且不报错）；`hostRatio` 只对 `l1`(0.60)/`l2`(0.50) 生效、SDXL 那条线走 `SdxlBackend` 没有这本账 ⇒ 回执 `profileNote.ignored` 明写） | `app/services/engine/profile.py` |
| **五条装配线的装配前决策**（2.1 按分时账 13.26 GiB / Z-Image 按整链同驻账 19.28 GiB / **Qwen-Image 2512 按本族分时账（`dit` 那一格**分文件**取：bf16 的 2512 是 38.05 GiB，Edit 2511 的 int8 文件走 `resident_need_gib()` ⇒ 数据区字节 19.09 GiB ⇒ `off` 整份上卡）** / **Krea2 按本族分时账 12.57 GiB** / SDXL 按盘上文件真字节；落不了 `split` 的线按「装不下」整份转 CPU，但「装得下只是没摊开」不算降级。⚠️ 前三条走 `_chain_device_decision` ✓，**2.1 与 Krea2 是另一种形状**：先构造 `TorchBackend`（此时 need 未知），装配 DiT 前再按**实测分时账** `decide_for(("dit",), need_gib=本族账)` 判一次 —— 两条都只装 DiT、其余靠 `declare_chain` 换装，所以「整链 28.68（2.1）/ 20.31（Krea2，12.57＋7.50＋0.24）GiB」都不能拿来当门槛） | `runtime.py::_chain_device_decision` |
| **一个数回答「卡上真占多少」**（`runtime.residency_total(report)` ⇒ `POST /load` 回执的 `residencyTotal`：`gpuGiB`/`cpuGiB`/`gpuShare`/`parts`/`judgedOn:"bytes"`；挂在 `_commit_load` —— **四条装配线共用的那一个人口**，各线自己加必漏一条。⚠️ 递归找「同时带 `gpuGiB` 与 `cpuGiB`」的字典并按 `id` **去重**（同一份账可能既在 `weights.residency` 又在 `components[x].residency`，加两遍凭空翻倍）；一件都没报 ⇒ **不写这一格**（不编 0：`None`＝"这次没有联合驻留的账"、`0`＝"算了，卡上什么都没"）；分母为 0 ⇒ `gpuShare` 给 `None`（编 1.0 会被读成「整份在卡上」）；`judgedOn:"bytes"` 明写**不是速度占比**） | `runtime.py::residency_total` |
| **策略的可见面**（`capabilities().devicePolicy` 报四档＋判定规则＋**四本账**（`part` 2.1 分时 / `chain` Z-Image 整链 / `qwenImagePart` 2512 分时 / `krea2Part` Krea2 分时 —— 族不同而键名相同的账**不能并**，并进去只会留最后写的那本；`qwenImagePart.dit` 是 **bf16 那份文件**的账，int8 文件的实际需求只在 `POST /load` 回执的 `ditNeed` 里，`capabilities()` 不读盘 ⇒ 静态值不许改成 19.09 ✗）、只报静态事实不读卡况；**生效账**另有 `GET /engine/runtime` 的 `devicePolicy.partNeedGiB`（声明链后就是本族那本，同时留 `defaultPartNeedGiB`）；`POST /load` 收 `policy` 三写法、拼错当场 400、回执两格 `devicePolicy`(现在状态) 与 `chainDevicePolicy`(这次为何在 CPU)；`SCHEMA_VERSION` **7**（`needGiB` 内部加键属 nested、不动版本；6→7 是 `status()` 顶层新增 `profile`/`budgets`/`diskCache` 三格 —— 顶层加键＝破坏性变更，`docs/engine-api.schema.json` 连 `SCHEMA_VERSION` 一起改，`engine_router_contract_test` ⑨ 逼着同步）） | `facade.py::capabilities` / `torch_backend.py::describe` / `routers/engine.py::load_runtime` |

### L4-4 `token` 文本入口（7）

prompt ⇒ token ⇒ 向量。不做图像数学。

| 功能点 | 落点 |
|---|---|
| 自研 BPE | `app/services/engine/tokenizer_bpe.py` |
| Unigram / WordPiece 扩充 | `app/services/engine/tokenizer_own.py` |
| **UMT5 sentencepiece unigram**（`nmt_nfkc` 归一 ＋ `add_dummy_prefix`、词表 256384；与 `tokenizer_bpe` 是两套 ✓✗） | `app/services/engine/umt5_tokenizer.py`（⚠️ 唯一出口是 `umt5_te`，塔没接线前无人调用） |
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
| **注意力核选择**（网络本体每次 SDPA 的唯一出口：默认 dispatch 在本机选 `EFFICIENT`，**cuDNN fused** 同形状快一倍 —— 但它只在**别的都关掉**时才被选中（优先级 flash → efficient → cudnn → math）⇒ 「能不能用」必须自己先判、不能靠回退；真块实测 26.6 → 23.4 ms（−12.1%）、峰值分配不变 1.062 GiB，且屏蔽 padding 位后输出差 1.1e-1 ⇒ 反证掩码真被应用，不是「更快所以更不准」；**十个本体**（`dit`/`dit_flux`/`dit_generic`/`dit_lumina`/`dit_music3`/`dit_qwen21`/`dit_qwen_image`/`dit_seedvr2`/`dit_wan22`/`h3_form`）全走这里，SEEDVR2 的 **256 token 变长窗口段也仍赢**（0.047 → 0.041 ms，短序列没被固定开销反噬）；`can_use_cudnn_attention` 在 24–44 个形状上与「强制 CUDNN 真能不能跑」**0 条不符**（head_dim 256 判 False 且真跑抛）⇒ 先判后用，判不过走原路、绝不抛；**非逐位**（真块 img max|Δ| 3.1e-2 / 均值 6.7e-5）⇒ `DARME_SDPA=off` 可逐位复现旧结果；**文本塔有意不走这里**（`qwen35_te`/`qwen3vl_te`/`umt5_te`/`te_music3` 与 comfy 有 `maxDiff=0` 钉子） | `app/services/engine/sdpa.py` |
| Z-Image / Lumina 骨架（`NextDiT` 同构，已与 ComfyUI 参考实现逐层对拍；**已接进装载分支**） | `app/services/engine/dit_lumina.py` |
| Qwen-Image 2.1 骨架（单流 + SwiGLU + 三轴 RoPE，两行调制；键集合与盘上 649 张量对账、与独立参照实现双精度一致；**已接进装载/前向分支**，TE 与 VAE 未做） | `app/services/engine/dit_qwen21.py` |
| **Qwen-Image 2512 / Edit 2511 骨架**（双流 MMDiT：文本与图像各 6 组调制＋各一套 MLP；时间口径 `1000·σ` **不翻转**；`heads` 由 `attn.norm_q` 宽反推；Edit 侧参考图 token 串联＋`__index_timestep_zero__` 哨兵喂 σ=0 行调制；2512 与 Edit 2511 同一份，差别只在 `ref_latents`；**已接进装载分支**，且这一族的 `qwen_image` 行挂了 `adapter="int8"` ⇒ 盘上的 840 层 int8 文件**原样常驻**（19.09 GiB，不再反量化成 bf16 38.05；bf16 文件自动得到空 plan ⇒ 走老路） | `app/services/engine/dit_qwen_image.py` |
| **Edit 2511 参考图通路**（2026-11-01 接上，**含多图**）：参考图 → 视觉塔（**384² 面积**，`<|image_pad|>` 整段替换）→ 文本塔 `image_embeds`；参考图 → VAE（**1024² 面积**、8 的倍数、Wan 族 5 维挤成 4 维）→ `ref_latents`，`denoise` **每一步**带进 DiT（`Sequence[Tensor]`，**一条一张、必须都是列表元素**）。N 张图 ⇒ 模板 `Picture i: <视觉块>` 逐图一个槽（一张走 `…ImageEdit` 口径、没有编号）、`image_embeds` 一列块、`ref_latents` 一列，**顺序即语义**；`_refLatents` 统一存列表（单图长度 1）。视觉塔**只在 encode 段在场**（`declare_chain(phase_parts=…)` ⇒ 峰值 17.69 GiB，仍判 `full`，与图数无关）；编辑口径下 `max_length` 叠加图的 token 数；Edit 权重缺 `ref_latents`（含空列表）/ Edit 请求没给 `referenceFrames` 一律当场报错 | `app/services/engine/torch_backend.py`（`encode_text` / `_edit_reference` / `prepare_edit_references` / `denoise`）、`qwen25_te.py`（`prompt_of` / `edit_prefix` / `_with_image_blocks`）、`runtime.py`（`is_edit` 分支） |
| 按权重键名判族 + 读拓扑 | `app/services/engine/dit_family.py` |
| **族 ⇒ 骨架的唯一装配表**（一族一行 `FamilyBuilder`：模块 / 入口 / 形状参数 / 建法 / 改名 / 同族常量 / 前缀适配器 / **前向契约**；公共流程一份：抽形状 → 剥前缀 → 推结构 → dtype → meta 建壳 → 物化 → 拼 config；加族只动本表 + `dit_family._RULES` 两处） | `app/services/engine/dit_build.py` |
| **潜空间口径对账**（契约的 `latent_format` 必须与 `latent_formats` 表里的规范名同族；解码前比「已挂 VAE 的反缩放口径」，不同族当场拒——16ch 的 `wan21` 与 `qwen_image` 形状全同，挂错只会整体偏色而不报错；三条装配线的 `vae_family` 一律从契约读，不写族名字面量；`readout()` 连换算方式一起写出来；H3 这类非装配族由 `latent_format_of()` 单独答口径，不因不在表里就整条跳过；`meanstd` 口径的族必须在 `vae_wan.latents_stats` 里有 16 组数） | `app/services/engine/dit_build.py`、`latent_formats.py` |
| **通用单流 DiT 前向**（维度从权重形状反推、键集合双向对账） | `app/services/engine/dit_generic.py` |
| **Flux.1 双流＋单流 MMDiT 前向**（19 双流 + 38 单流、拼接顺序 `(txt, img)`、调制 6/3 组顺序 `(shift, scale, gate)`、时间口径 `1000·σ` **不翻转**、一根 vec 由 time+guidance+vector 三路相加；`guidance` 走默认值而非权重事实） | `app/services/engine/dit_flux.py` |
| **SEEDVR2 / NaDiT 视频修复主干**（32 块；前 10 块双塔 / 后 22 块共享；`in=33`；四处「形状全对、只有数值错」：`ada.emb` 切片组号 / `mlp` 的 SwiGLU 门控顺序 / `attn_norm` 有没有 affine / 末块 `vid_only` 的文本分支。**已接进装配表与 `stage="seedvr2"` 装配线** ✓ 见 §10.1g） | `app/services/engine/dit_seedvr2.py` |
| **Wan2.2 I2V 14B 单专家骨架**（40 层 / `dim` 5120 / `ffn_dim` 13824 / `in_dim` 36 / 文本口 umt5 `text_dim` 4096 / 五维潜变量 + 视频；`heads` 是**唯一查表项**、按几何取 `HEADS_BY_GEOMETRY`；时间口径由 `TIMESTEP_SCALE` 补 —— 本仓骨架收裸 σ；`split_i2v` 是出片前还欠的那段。**已接进 `dit_build` 的 `wan` 行**，能建能装能前向，不能从提示词出片） | `app/services/engine/dit_wan22.py` |
| **Music3 音频/歌曲主干**（第十族骨架，族内三段链 `文本+歌词 → context → 本模块 → 速度 (b,128,L) → DAV → 44.1k 立体声` 的中间那段；`latents (b,128,L)` ＋ `sigma` ＋ `context (b,F,8×4096)` ⇒ `(b,128,L)`。四处「形状全对、只有歌不对」的口径：①时间步要**翻**（`1−σ`）②输出要**取负**（与①是一对，只改一边立刻崩且不报错）③latent 帧数 = 条件帧数 × **3.4453125**（25 Hz → 86.13 Hz；同帧率假设会被 pad/crop 静默吃掉）④结构上**无 ada 调制**，时间步作**第 0 个 token** 拼进序列；GLU 是 `value * silu(gate)`、RoPE 只转 head_dim 前半 32 维且 split-half **正号**、`v` 不转）。`velocity()` 与 `forward()` **分开**（σ 翻转与取负各一处，合写就分不清谁漏了）；长音频按 689 / 344 潜帧滑窗，累加后**除命中次数**、最后一次取负。**能建**（`dit_build.BUILDERS["music3"]`，`latent_dims=3`）、**接线已闭合** ✓ 2026-11-06（`stage="music3"` 装配线 ＋ 音频 plan ＋ `encode_text` 条件支 ＋ `decode` 的 `music3-real` 支 ＋ `write` 纯音频支；滑窗在 `velocity()` 内、`pipeline` 只诊断不重切）。判据 `tests/engine_music3_test.py`（46 项）＋ `tests/engine_music3_wiring_test.py`（59 项接线对账）） | `app/services/engine/dit_music3.py` |
| **Music3 语义塔**（第十族的 **context 生产段** ✓ 2026-11-02：Qwen3-8B 形态 36 层 / GQA 32+8 头 / **QK per-head RMSNorm 在 RoPE 前** / 合并 qkv+gate_up / **剪枝版**词嵌入与输出头（`embed_tokens_prefill` 151675 ＋ `embed_tokens_audio` 16384 ＋ `lm_head_pruned` 16385，全量那份不存在 ✗）＋ 4 层 RVQ 深度解码器（无 QK-norm 无 RoPE、位置嵌入 16 帧、键名 `layers.N.self_attn/mlp` **两层嵌套** ✓）；**词表内嵌在 TE 文件里**（`tokenizer_json` 键 11.4 MB ✓）不用外置目录 ✗；160 层 int8_tensorwise+convrot(256) 常驻 ✓（装载六步序：头部核结构 → meta 建壳 → 换 Int8Linear → 物化 → 排布 → 填权重 ✓）。四处「形状全对、只有歌不对」：①CFG 非条件份只换 `[:, 1:-2]`（首尾各留两真 token ✗）②c0 引导采样的 top-k 阈值取自 **conditioned** 那份 ✗✗ ③深码本查表要加 `(index−1)×1024` 段偏移（7 个码本叠在同一张 7168 行表 ✓）④帧隐状态只收**条件份** `cat(last_hidden[:1], depth_hidden)` ✓。判据 `tests/engine_music3_keys_test.py`（10 项：键对账 327/327 ✓、int8 计划 160/160 ✓、词表特殊 token、H3 双流路由闸门 ✓）＋ `tests/engine_music3_test.py`（46 项口径）＋ `tests/engine_music3_wiring_test.py`（A6/A7/A8：两把尺子各自独立、窗口数逐值一致、潜帧换算 1..9000 与主干逐帧对拍）） | `app/services/engine/te_music3.py` |
| H3 DiT 积木（含 **`token_tags` → adaLN 行号**：文本段按标签切成 text/video run，视觉占位 token 走 video 行；不给标签 ⇒ 整段 text 行，纯文本提示词等价；装配侧 **meta 建壳 → 降精度 → 扳 fp32 岛 → 换 int8 常驻层 → 才物化**，峰值 ≈ 19.5 GiB 而非 fp32 主干的 78 GiB） | `app/services/engine/h3_form.py` |
| H3 检查点键名核对（torch-free） | `app/services/engine/h3_keys.py` |
| H3 续拍 / 重拍 | `app/services/engine/h3_edit.py` |
| SDXL UNet | `app/services/engine/sdxl.py` |
| SDXL 出图后端闭环 | `app/services/engine/sdxl_backend.py` |
| SDXL VAE | `app/services/engine/sdxl_vae.py` |
| **Flux 族 VAE**（16 通道 / 无 `quant_conv` / 244 键；`(x−shift)/scale` 反缩放内置） | `app/services/engine/vae_flux.py` |
| **Wan 系 VAE**（先接 Qwen-Image 2.1 的 64 通道：Wan 2.2 布局 / 238 键 / 单帧解码 / 出 RGBA 要丢第 4 通道；**编码侧 `encode_latents` 已补**：`Encoder3d.forward` 原本**不存在** ⇒ `WanVAE.encode` 一直没有，取 `head` 输出的前一半（mean，不采样）再过 `process_in` 的 16 组 mean/std） | `app/services/engine/vae_wan.py` |
| 视频 VAE | `app/services/engine/vae.py` |
| H3 真视频 VAE（3D 因果 CNN + ViT3D 解码） | `app/services/engine/vae_h3.py` |
| **SEEDVR2 VAE**（ByteDance `VideoAutoencoderKL` 的逐位重写：4 组因果下采样 / 4 组上采样、通道 `(128,256,512,512)`、**没有** `quant_conv`/`post_quant_conv`（编码器直出 32 通道，取前 16 均值即 latent）、空间 8× / 时间 4×、帧数铁律 `T = 4n + 1`、注意力只在 mid block 且**逐帧**、潜变量缩放 0.9152 交给 `latent_formats` 的 `seedvr2` 口径。**两半都已接线** ✓：`attach_seedvr2_vae` ＋ `decode` 的 `seedvr2-real` 支 ＋ 像素→潜变量 `encode_pixels`，见 §10.1g） | `app/services/engine/vae_seedvr2.py` |
| H3 音频 VAE（32 kHz 立体声） | `app/services/engine/audio_vae.py` |
| **Music3 DAV 音频 VAE**（潜变量 ⇄ **44.1 kHz** 立体声；与 H3 `audio_vae` 是**两个族**，数字全不同：采样率 44100 vs 32000、hop 512 vs 800、潜帧率 86.1328 vs 40 Hz、潜通道 **128 vs 32**；⚠️ 128 = 2 声道 × 64，`decode` 里 `reshape(b*2, 64, T)` 是**行主序折半** ⇒ 通道 0..63 左声道、64..127 右声道，把 128 当 128 个特征通道直接卷积＝「形状全对、声音是两声道混在一起的糊」。121 键、检查点 `weight_g/weight_v` ↔ `state_dict` 的 `parametrizations.weight.original*` **两向**互归（`_canonical_key` 与 `as_legacy_name` 各一向，少一向 ⇒ 「自证零问题」变「缺 60 / 多 60」），`shape_problems` 两种写法都收；折半的**行为判据**只许判「没动的那半**逐位不变**」✗ 不许判「清零那半恒为常数」（转置卷积在 `T` 小时整段都是边界，零输入照样解出时变波形 ⇒ 会误报）；`Snake1d.alpha` 初值必须**确定**（`torch.empty` 是未初始化内存，本机实测抽到 3.5e7 ⇒ 不装权重的壳解出假声音）） | `app/services/engine/vae_music3.py` |
| 画布计划层（"这次出多大"的**唯一**计算处，与超清同构：**计划层定、管线只执行**。吸附步长按族给 —— H3 32 / SDXL 8 / 图族 16，全局一个数是错的；档位命中时**像素优先**，因为 `1344×768`(1.75) 与 `16:9`(1.778) 不闭合，拿 MP 重算会落到 `1360×768`；没点名用途 ⇒ 旧路径**逐位不变**） | `app/services/engine/canvas.py` |
| 用途档位表（用途 ⇒ 画布 / 步数 / cfg / 采样器 / 调度器 / denoise，每档写**出处工作流文件名**；数值取自生产线实测而非模板文档；表里没定画布的档（H3 导演台出画布）⇒ 明确报错，**不猜**一个"看着合理"的） | `app/services/engine/profiles.py` |
| **图生图（`denoise<1` + `initImage`）两半齐**：① σ 梯子按名义步数 `int(steps/denoise)` 算**再取尾巴** `[-(steps+1):]`（`_denoise_split`/`_denoise_slice` 是全仓唯一做这条算术处，五个图族走同一出口 `_image_flow_sigmas`、SDXL 单接）；② 起点 latent = `initImage` 像素 → `_encode_pixels` → 按 `σ0·noise + (1−σ0)·latent` 混（`noise_scaling` 逐字）。**缺任一半都当场拒** —— 只砍 σ 不换起点 = 与参考图无关的低对比图且**不报错**（本仓最恨的那类坑）。回执如实报四格 `denoise`/`totalSteps`/`startSigma`/`initFrom`；视频侧不接（接着上一段跑走 `pin`/`lastFrame`） | `app/services/engine/pipeline.py`（`_denoise_split`/`_denoise_slice`/`_finish_plan`）/ `torch_backend.py`（`init_image_latents`/`_mix_init_image`） |
| **`normal` 调度器在 flow 域是真排程**（不是 EDM 回落）：参考实现 `normal_scheduler` 的语义是「**timestep 等距**」，而 flow 族 `multiplier=1` ⇒ `timestep` 就是 σ ⇒ 落到 σ 上是**σ 等距**（≠ flow 的 t 等距）。端点必须取模型自己那对 `flow_sigma_bounds(shift)`（`σ(1)=1` / `σ(1/1000)`），拿 EDM 的 `0.0292/14.6146` 会得 `t=1−σ<0` ⇒ 不报错、出糊图 | `app/services/engine/schedules.py`（`normal_sigmas(shift=)`/`flow_sigma_bounds`）/ `pipeline.py`（`_image_flow_sigmas`/`_video_sigmas`） |
| 超清规划层（契约**读 + 结构核一步到位** `load_upscaler_contract`：只读不核时缺一个字段 ⇒ 计划层静默绿灯、状态接口 `KeyError`、张量层按 `H3LatentUpscalerV2` 默认值建网络 ⇒ 三处调用点一律走它） | `app/services/engine/upscale.py` |
| 超清网络本体 | `app/services/engine/upscale_net.py` |
| 文本编码器 | `app/services/engine/text_encoder.py` |
| CLIP 文本塔（SDXL 那一半） | `app/services/engine/clip_text.py` |
| **Qwen3 文本塔**（Z-Image 的提示词入口：36 层 / `layer_idx=-2` / ChatML 模板） | `app/services/engine/qwen3_te.py` |
| **Qwen3.5 文本塔**（Qwen-Image 2.1 的提示词入口：24 层 Gated DeltaNet ＋ 8 层 full attention 混合 / `layer_idx=-2`；已与 comfy 参考块逐位对拍 `maxDiff=0`） | `app/services/engine/qwen35_te.py` |
| **Qwen2.5-VL-7B 文本塔**（Qwen-Image 2512 / Edit 2511 的提示词入口：28 层 / 3584 / 28×128 头 / kh=4；**q/k/v 带 bias**、无 QK-norm、取**最后一层**且**不过** final norm；`system`＋`user` 两段模板按第 2 个 `<|im_start|>` 截掉系统段；只挂语言塔，同文件里的 `visual.*` 与 `lm_head` 有意不装；**已接进 `attach_qwen25_te`**） | `app/services/engine/qwen25_te.py` |
| **UMT5-XXL 文本塔**（Wan 系提示词入口：24 层 / 4096 宽 / **双向**注意力 ＋ 每层各一份相对位置偏置 ＋ FFN 门控；与本仓其余几座塔口径**相反** ⇒ 别照抄 `qwen3_te` 的 mask 写法。**只有塔本体，装配线未接** ✗ 见 §10.1g） | `app/services/engine/umt5_te.py` |
| decoder-only LLM 推理 | `app/services/engine/llm.py`（**`generate_llm` 整段跑在 `torch.inference_mode()` 里**：不关梯度 ⇒ 参数默认 `requires_grad=True` ⇒ 36 层中间量（注意力分数 `(heads, L, L)` 最大）被 backward 节点吊住，4B / 3205 词元实测 PyTorch 已分配 **94.62 GiB**（卡 22.49 GiB，WDDM 甩进共享内存硬撑 60 秒才 OOM）；**长提示词必须分块预填充** `prefill_chunk`（默认 1024 词元，峰值从 `O(L²)` 降到 `O(chunk × L)`，带 KV cache 数学等价、逐 token 结果相同）；解码步 `s == 1` 不造因果掩码（全零，白占显存） |
| **Qwen3-VL-32B 语言塔**（H3 文本口的本体：interleaved mrope `ropeDims=(2,1,1)` / `len_max = grid.max()//2` / **有 QK-norm、无 final norm** / DeepStack 三份 merger 在**层之后**注入；与盘上 551 语言键逐名逐形对账） | `app/services/engine/qwen3vl_te.py` |
| **Qwen3-VL ViT 视觉塔**（27 层 / `patch_embed.proj` 2 帧时间核 / `merger` 在 **pre-shuffle**、`deepstack_merger_list` 在 **post-shuffle**；与 H3 的 TE 权重同文件、键名逐字一致） | `app/services/engine/qwen3vl_vision.py` |
| **Qwen2.5-VL ViT 视觉塔**（Edit 2511 参考图的**语义**通路：32 层 / patch **14** / CLIP 均值方差 / 窗口注意力 112 ⇒ 合并域 4 格（块 7·15·23·31 走全图）/ 旋转位置表与 hidden **同一套窗口置换** / 融合后 `argsort(window_index)` 还原；与 `qwen3vl_vision` **并列不合并**（尺子差 patch 16 vs 14、均值 0.5 vs CLIP）；像素侧走 `qwen3vl_pixels.process_image_with` 传本族口径） | `app/services/engine/qwen25vl_vision.py` |
| **LLM 分层放置（卡上前 N 层 ＋ CPU 尾巴，2026-11-08）**（`llm.plan_layer_devices` 只算**算术**：每层字节＝`_block_param_count × dtype_bytes`（`layer_weights_bytes` 同一出口，不抄配额表）× 连续前缀 ⇒ 一层都放不上就给 `gpuLayers=0` 并写 `why`；`apply_layer_plan` 逐层 `.to()` 后**从参数读回**实际设备，`embed`/`norm`/`lm_head` **钉末层**（`tie_embeddings` 下 `.to()` 到两处会把它拆成两份）；`LlmModel.forward` 只在**层边界**搬激活、RoPE 按设备各缓存一份、KV 各留层设备。对外旋钮 `settings.gpuLayers`：`off`＝不分层（老路径逐字节相同）／`0`＝全 CPU（回执写明"显式要求"）／`N`＝前 N 层上卡（**超过层数**或**没有卡**都具名拒，放不下也拒——不悄悄减配）／`auto`＝按 `device_policy.probe().freeBytes − HEADROOM_BYTES` 铺满；`plan_layer_devices` 的 `want_gpu_layers` 是**下限**，显式 `N` 由后端**截到 N**。回执 `describe()["layers"]` 带 `gpuLayers/cpuLayers/layerBytes/usableBytes/why`，分层时 `device` 报**首层**在哪（报 embed 所在会永远是 cpu） | `app/services/engine/llm.py` / `llm_backend.py` / `gguf_to_llm.py`（`layer_devices=` 转给 `build_llm`，流式装配逐张量 `copy_` 跨设备一次） / `app/services/text_generation.py`（`engine_gpu_layers` 单一真源，三个调用口共用；缓存键含 `gpu_layers`） |
| **KV cache INT8 分组量化（2026-11-08）**（`llm.quantize_kv` / `dequantize_kv` / `update_kv_cache`：**沿 `head_dim` 每 64 元素共用一个 fp32 scale、零点恒 0**（口径同 Strata `kernels/kv_q8.hpp`）——沿序列分组是错的（语义无关的 token 共用一把尺子）；全零组 `scale=1`（否则 0/0 ⇒ NaN ⇒ softmax 全崩）；旧段 int8 **原样 cat 不重复量化**（32K 上下文反复量化会累积误差）；`head_dim` 除不尽组宽 ⇒ 尾部 pad 到整组、反量化按 `head_dim` 裁回；注意力读的是**量化后再反量化**的值（prefill 用原 k、decode 读 dequant(k) 会是两套历史）。`Q8Kv` 壳把「值＋scale＋组宽＋原始 headDim」绑在一起、`shape` 与量化前逐字相同。对外旋钮 `settings.kvMode`（`off`/`fp`＝原样浮点老路径逐字节相同／`q8`/`int8`/`auto`＝量化，拼错**装载前**具名拒）：`kv_bytes_per_token` 给的就是**每 token 字节**（4B 档 144 KiB ⇒ 76.5 KiB，32K 上下文 4608 MiB ⇒ 2448 MiB），`describe()["kv"]` 带 `mode/group/kBytes/vBytes/totalBytes/fpTotalBytes/ratio/contextBytes`，装载前按**请求值**报数并标 `assumedDtype`（配了 q8 却报 fp 的字节会让人以为配置没吃）。★2026-11-09 补**时间代价**（此前只报「省 47% 字节」＝半个账）：探针 ⑦ 实测 q8 比 fp **每步 ×1.29–1.38（+8–10 ms）**，且 256 行与 1024 行**两个点几乎一样** ⇒ 更接近**固定成本**（多出来的那批 kernel），**别按「上下文越长越亏」想当然外推**；口径＝**q8 省显存不省时间**，显存够就别开。`describe()["kv"].note` 现按档成对写：`q8` 写「每步 +8–10 ms」，`fp` 写「`q8` 可省约 47% 字节、代价每步 +8–10 ms」——**关着的档也写**（不打开的人否则永远看不到值不值）。 | `app/services/engine/llm.py` / `llm_backend.py`（`kvModeRequested` ＋ `kv` 回执） / `app/services/text_generation.py`（`engine_kv_mode` 单一真源；缓存键归一 `None`＝`off`、`auto`＝`q8`，否则同义写法会各装一份 8 GiB） |
| **跨调用 KV 前缀复用（2026-11-08）**（`llm.KvPrefixCache` 单槽 ＋ 字节预算 ＋ 命中回执：**按 token id 比公共前缀**（文本相等不保证 id 相等；模板空白/特殊符/编解码往返都会换一串 id）且**必须 ≤ 序列长 − 1**（复用整段就没有查询可算 ⇒ 拿不到 logits）；`truncate_kv` 裁段时 **`q` 与 `scale` 一起裁**（只裁 q ＝ 长度对数值错位）并 `contiguous()`（切片视图会吊住整段长张量）；`store` 存**整段**（提示词＋续写 ⇒ agent 循环下一轮连上轮输出也不必重算）；**一点共同前缀都没有 ⇒ 当场释放**（留着「万一」＝一段永不命中的 cache 一路占显存，卡满时报在别的件上）；**超预算不存**并写明原因（`conversation_memory.hpp` 的预算准入写法）；口径必须同一套（`fp` 的 cache 喂 `q8` 跑＝长度对数值两套）。⚠️ 「无损」只到 bf16 舍入级：复用的 KV 段与重算**逐位相同**（q8 分组沿 `head_dim` ⇒ 每 token 的 scale 与前后无关），但末位查询块从 `M×N` 变 `1×N` ⇒ 4B 真权重末位 logits 差 0.145（0.56%），**同串整段复用在近似并列处会翻 token**（实测差一个空格）⇒ 要逐字复现就关复用；增量轮（agent 循环真实形状）实测**逐字相同**。对外旋钮 `settings.kvReuse`（`off`/`false`/`0` ＝关）＋ `settings.kvCacheBytes`（`512MiB`/`1GiB`/字节数，不写＝512 MiB，`0`＝关；拼错**装载前**具名拒）；`describe()["kvCache"]` 带 `enabled/mode/group/budgetBytes/storedTokens/bytes/hits/misses/skipped/reusedTokens/stored/note`（**命中数必须真报**：不报 ⇒ 「配了复用」与「一次没命中」分不清） | `app/services/engine/llm.py` / `llm_backend.py`（`kvCache` 回执 ＋ `clear_kv_cache()`） / `app/services/text_generation.py`（`engine_kv_cache_bytes` 单一真源，三个调用口共用；缓存键新增第 7 格 `cache_slot`，`None` ≡ 默认 ≠ `0`＝关，否则改了预算仍拿到老后端） |
| **投机解码：后缀查表草稿（2026-11-08）**（`llm.LookupDrafter` ＋ `generate_llm(spec_lookup=True)`：三词元后缀 → 历史里出现过的后继（零权重、不训草稿头，与 Strata `spec::SuffixDrafter` 同构）。**只在 `temperature=0` 启用**——随机采样要「概率接受 ＋ Philox 计数空间隔离」才能落在同一条随机序列上，本仓没有那套 ⇒ `spec_report` 如实报 `enabled=False, reason="temperature>0"`，同种子输出与不开**逐位相同**（不假装开了）。主循环口径四条：① `window[0]` 是 **target 自己刚产出但还没喂过前向**的那个 token（参考实现的 `window[0]` 口径），所以「一拍吃 1+T 行」比老路径**少一拍**；② 接受判据是「第 `i` 行 argmax == `window[i+1]`」（第 `i` 行预测的是窗口**之后**那个 token），抄成 `window[i]` 全拒且不报错；③ 回滚 `keep = min(past_len + 1 + a, total + len(generated))`——上限压到「已产出总数」是因为 stop 打断时 cache 不能比 ids 长（否则 `KvPrefixCache.store` 存错配）；④ `a == len(draft)` 时第 `a` 行就是 bonus token，被拒时它就是修正后的 token ⇒ 两条路同一个 `_sample_token(win_logits[:, a, :])`。草稿器口径：**允许与当前后缀重叠**（周期文本唯一能拿到草稿的路径）、等长匹配**保留更近**（`>` 写成 `>=` 结果不同但不报错）、`min_match` 不够就返回**空列表**（不是错）。收益实测 1.02~1.03×（逐字复述时最长匹配贴着当前末尾 ⇒ 匹配越准越不接受，这是 lookup 的结构性盲区，照参考实现保留、不擅自加错位修正）。旋钮 `spec_lookup`/`spec_window`(8)/`spec_min_match`(3)/`spec_report`，`< 1` 具名拒。⚠️ 默认关闭 ⇒ 老调用方逐字节相同 | `app/services/engine/llm.py`（`LookupDrafter` ＋ `spec_*` 参数） / `tests/engine_llm_test.py`（⑩ 系列 12 条：真模型等价、q8 等价、`accepted == drafted > 0`、回滚差值区间（**部分拒绝时严格小于窗口行数**——全接受时不裁也「差 == 行数」⇒ 只看区间会假绿）、输出依赖 `past_len` 的模型下仍等价、温度退回、三条草稿器口径、额度边界、参数校验） |
| **融合核快路径（`fastPath`，2026-11-08，默认关）**（两个**减 launch** 的开关 ＋ 一个**减带宽**的开关 ＋ 一处**无开关**的等价改写：`norm`＝`RMSNorm` 走 `F.rms_norm`（老五连在真机上是**一次 16 个 kernel**，不是纸面的 5 个 ⇒ 4B 档一步 145 次调用、2320→290）、`attn`＝`F.scaled_dot_product_attention(enable_gqa=True)`（免掉 `repeat_kv` 物化 1.15 GiB/步、每层 6→1）、`kvbuf`＝KV 改**定长缓冲 ＋ 原位写**（`KvBuf`：每层每步那 2 次 `cat` 把整段历史重拷一遍，2048 词元＝288 MiB/步 ⇒ 换成往 `[length, length+s)` 写新来的 1 行＝288 KiB；`shape[2]` **仍报已用长度**，暴露 `cap`＝位置/掩码整排错位且形状自洽）、RoPE 折叠（`fold_rope`/`apply_rope_wide`：`cat((cos,cos),-1)` ／ `cat((-sin,sin),-1)`，**前后半配对**不是相邻交错，与 `apply_rope` **逐位相同**所以不设开关）。`resolve_fast_path` 收 `off/norm/attn/kvbuf/all/组合/布尔/列表`，返回 **`frozenset`**（`"all"` 里没有 `"norm"` 子串 ⇒ 字符串包含判断会把 all 读成只开 attn），拼错**当场具名拒**。关着时 `fast_kw = {}` ⇒ **不传关键字**、调用签名逐字不变（本仓桩模型测试靠这条）。真机实测（A5000／4B-Q4_K_M／2048 上下文）：`all` 每步 33.7→23.7 ms、launch 4449→2203、整条生成循环 **32.1→43.7 token/s**；`norm`/`attn` 的收益全在 launch 不在算（CUDA 侧 50.9→47.1 ms），所以**默认关**（融合核差 1 个 bf16 ulp ⇒ 长程贪心可能岔口）。⚠️ `kvbuf` 是另一种账：**kernel 数 4449→4449 一个不少**（72 次 `aten::cat` 一比一换成 72 次 `copy_`），省的是**带宽**（每步 288 MiB→288 KiB、CUDA 50.19→47.32 ms、墙钟 −5.8%），**logits 最大差 0.000e+00**（真权重逐位相同，所以这一档不改数值）；代价是**已分配**容量按翻倍涨（2048 行那步 288.7→576.3 MiB，`kv_cache_bytes` 报 cap）。`truncate_kv`（前缀复用处）必须返**独立副本**、`rollback_kv`（投机窗口处）必须**就地零拷贝**，两者**不许互换**。回执 `describe()["fastPath"]` 报 `requested/norm/attn/kvbuf/items/enabled/fusedNormAvailable/note`，缺 `F.rms_norm` 时如实写「退回老写法、一步也没省」；`settings.fastPath` 经 `text_generation.engine_fast_kwargs` 三处生成口共用，缓存键含 `fastSlot` | `app/services/engine/llm.py`（`FAST_PATH_ITEMS`／`resolve_fast_path`／`KvBuf`／`update_kv_cache(buffer=)`／`rollback_kv`／`fold_rope`／`apply_rope_wide`／`sdpa_attention`／`fused_norm_available`） / `llm_backend.py`（`_fast_report`／`_fast_call`） / `app/services/text_generation.py`（`engine_fast_path`／`engine_fast_kwargs`） / `tests/engine_llm_test.py`（⑫ 系列 19 条：逐位对拍＋反证、SDPA 三种 past_len＋掩码偏 1 的反证、**spy 数融合核调用次数**、默认关不传关键字＋反证；⑬ 系列 12 条：`kvbuf` 的 fp/q8 逐位对拍、`shape[2]` 语义、**spy `aten::cat` 归零**、容量翻倍、`truncate_kv` 独立副本＋`rollback_kv` 就地反证、开→关→开字节原样、写错位置的反证、`generate_llm` 逐 token 相同） / `tests/engine_llm_backend_test.py`（⑫ 系列 12 条：回执三态、`load()` 开头拒、缓存键、**AST 判三个调用口＋后端 3 处 `generate_llm`**、真跑 spy） / `data/experiments/llm_attn_cost/fast_path.py` / `docs/reference-audit-llm.md` §7.5） |
| **CUDA Graph 图化 decode 单步（`cudaGraph`，2026-11-08，默认关）**（省的是**每步几千次 launch 的提交开销**，不是算力；探针 `data/experiments/llm_graph_probe/probe.py` 量「纯提交时间 vs 重放墙钟」，重置 KV 必须用**零拷贝** `rollback_kv` —— 用 `truncate_kv` 会给 KvBuf 返独立副本、把 256 行×36 层的复制成本算进 plain，图看起来虚快）。⚠️ 图化要改**三处结构**，漏一处就「捕获成功、重放出来是错的」且**不报错**（形状全对、只是从第 2 步起算错，症状＝越往后越胡言乱语）：① 历史长度不能是 Python int（老路每步读 `past_kv[0][0].shape[2]`，图里被烘成常量）⇒ `KvSpan`「读满桶 `cap` ＋ 加性 `-inf` 掩码 + `used` 是**设备张量**、`advance()` 走 `add_`”；② 位置不能是 Python int（RoPE 的 `pos0`）⇒ `RopeTable` 一次算好 `(cap, head_dim)` 折宽 cos/sin、图里只留 `index_select(0, span.used)`；③ KV 缓冲不许换指针（老路 `torch.cat` 每步重分配）⇒ 必须 `fastPath=kvbuf` ＋ `KvBuf.write_at`（`index_copy_(2, at, …)`，**不改 `length`、不 `_reserve`、只收 `fp`**）⇒ 本档**隐含 kvbuf**（回执 `impliedKvbuf` 明写是不是替你加的）。收尾必须补 `length`（图里只写**行**）否则前缀缓存只存 prefill 那一段、下次命中接错位置且不报错。九条前置任一不满足就**退回 plain 并在回执写理由**（`device=cpu` / `kvMode=q8` / `kvWindow=n` / `specLookup` / `kvStorage=cpu-cat` / `layerSplit` / `capTooSmall(base+max>cap)` / `captureFailed`），捕获前把 KV 尾部**清零一次**（`torch.empty` 残留若是 NaN ⇒ `q@kᵀ` 出 NaN ⇒ 加 `-inf` 还是 NaN ⇒ 整条 softmax 全 NaN 而不报错；`exp(-inf)=0` 只救有限值）。⚠️ **默认关**不是忘了开：掩码把 softmax 归约长度从 `used+1` 改成 `cap` ⇒ 与老路**不逐位相同**（`exp(-inf)=0` 精确、但浮点加法不结合），回执如实写 `bitExact=False`（＝不承诺逐位相同；「必然不同」是过宽结论）。⭐ 2026-11-08 真权重把差异**量**出来了：分歧**全来自多读的 padding 行** —— 同一份 KV／同一步／同入口，`cap = used+1` 时与截断读**逐位相同**（|Δlogits| = 0）、`cap` 多 96 行时 bf16 上 |Δlogits| **1.9e-1**、32 词元 **25 个**分叉（成因＝bf16 下 SDPA 归约分块随列数变，**不是掩码算错**）；默认预留 `padRows = 2 + specWindow = 10` 实测 32 词元 **0 分叉** ⇒ 别为省事把 `kv_reserve_rows` 放大（那会把输出静默改掉）。回执直接交 `padRows = cap − pastLen − maxNewTokens`（`None` ＝没捕获成、**不是 0**）＋ `captureMs`（热身 2 次＋捕获 1 次＋硬同步，实测 **≈100 ms**，**计在第一次生成里**），前端只在 `padRows > 16` 时标 `·pad N`。★★ 探针 ⑥「长度扫描」把两句好听话同时推翻，这才是开不开的真正判据：① **32 词元是探针挑的长度** —— 捕获 100 ms 而重放每词元只省 ~12 ms ⇒ **8 词元 ×0.91（净亏 ~28 ms、白干）**、16 词元起才到 ×1.1 档，短问答／agent 的一两句回答正落在这几档 ⇒ **那些请求开图是净亏**；② **「紧桶 ⇒ 逐 token 一致」只在短生成成立** —— 同一档 `padRows=10` 下 8/16/32 词元 0 分叉，**64 词元 11/64、128 词元 72/128、512 词元 443/512 照样分叉**（每步残留 1 ulp 随步数累积）⇒ padding 是**主因不是唯一原因**，长生成仍须与老路比对；★★ 长度全表（同一次跑，每档都新捕获）：8→×0.91、16→×1.09、32→×1.17、64→×1.24、**128/256/512→×1.24–1.27**（35.4→44.2–45.4 tok/s）⇒ **本仓真实负载（写剧本/对白 128–512 词元）稳定 ×1.24 以上**，捕获那 ~100 ms 摊进这段只剩零头；⚠️ 口径更正：短生成那十几毫秒净亏**不足以否掉整档**，但也别拿 32 词元的 ×1.17 当默认收益 —— 收益随长度从 ×0.91 走到 ×1.27，真正的代价是 `bitExact=False`（512 词元里 443 个词元已与老路不同）而不是那十几毫秒；③ 每次请求都**重新捕获**（无缓存、用完即弃不漏），跨请求复用同一张图需 KV 缓冲跨请求常驻＝改内存模型，但长度全表已把捕获摊销算进去 ⇒ **收益只剩 1–2%，不值得为它动内存模型**，明确不做。收益实测（A5000／4B-Q4_K_M／prompt 256／32 词元，**更早一批读数**，与长度全表 32 词元 ×1.17 同量级）：② 单张图重放 28.7→20.2 ms/步（×1.42，乐观值）、③ **产品路径 35.1→40.2 tok/s（×1.14，过 1.1 门槛）**、纯提交时间 915→578 ms；⭐ 2026-11-09 补：**`auto` 从「`on` 的别名」升级成真档**（此前它落在 `resolve_cuda_graph` 的接受表里 ⇒ 无条件捕获、连 8 词元那张净亏的图也照捕，回执还只报 `on` ⇒ 说的和做的不一致）：`resolve_cuda_graph_policy` 出**三档** `off/on/auto`（`resolve_cuda_graph` 仍只答「**可能**开」），`generate_llm` 内 `max_new_tokens < CUDA_GRAPH_AUTO_MIN_TOKENS(16)` ⇒ **主动不捕**并写 `reason="autoTooShort(n<16)"`（16 是 8→×0.91／16→×1.09 量出来的**不亏线** ✗ 不是甜点），auto 跳过时**连 kvbuf 都不替你加**（跳过＝完全老路径）；回执 `describe()["cudaGraph"]` 多一格 `policy`（`enabled=True` 分不出 `on` 与 `auto`，而这两档在用户眼里含义不同），auto 的三态 note 各自成句（还没跑过＝先说阈值／因短跳过＝「**不是故障**、要强制捕请改 `on`」／真捕到＝「本次 ≥16 才捕」），且**不混**「让路/前置不满足」那句（一个该改长度、一个该查驱动）；缓存键第 14 格改用**三档原文**（`on` 与 `auto` 背后是**两个后端实例** ⇒ 压成 bool 会让切档静默无效）。`resolve_cuda_graph` 拼错（`grpah`）**当场拒**，`load()` 在装权重之前具名拒，缓存键含第 14 格 `graph_slot`（不进格 ⇒ 把 off 改成 on 仍拿老后端、回执照报关） | `app/services/engine/llm.py`（`KvSpan`／`RopeTable`／`span_attention`／`sdpa_mask_gqa_ok`／`KvBuf.write_at`／`resolve_cuda_graph`／`generate_llm(cuda_graph=,cuda_graph_report=)`） / `llm_backend.py`（`_graph_call`／`_graph_report`／`describe()["cudaGraph"]` 三态 requested/enabled/last） / `app/services/text_generation.py`（`engine_cuda_graph`／`engine_graph_kwargs` 三处生成口共用；`_engine_backend_key` 第 14 格；`engine_llm_status` 的 `requested.cudaGraph` ＋ **第三格** `matches` 判据） / `models.vue`（`graphLabel`：off／on·未跑(理由)／on·重放 N 步／⭐ auto 档前缀 + `auto·未捕(短)` ＋ `auto·重放`，两端逐字同源） / `tests/engine_llm_test.py`（⑮¹–⑮¹²：归一表、拼错拒、**定长读窗口 ≡ 截断读**（3 步 logits 差 ≤1e-3 ＋ argmax 全同）、反证掩码写反差量级、闸门逐条理由、非 cuda 逐位回退、隐含 kvbuf、真捕获重放词元流一致、长度收尾、`padRows` 真值＝`2+specWindow` ＋ 放大 `kv_reserve_rows` 后必须跟着变、⭐ ⑮⁷′ `kv_reserve_rows` **只抬不压**（给 1 行／给 64 行都产满 6 词元；写成赋值 ⇒ 第 2 个词元就「超预算」，而回执说的是「调小 maxOutputTokens」＝假补救）；⭐ ⑮¹′ 三档归一＋`resolve_cuda_graph('auto')` 仍答「可能开」、⑮⁵′ **阈值两侧**（15 ⇒ `autoTooShort(15<16)` 且不替你加 kvbuf／16 ⇒ 落到 `device=cpu` 说明它试着捕了）） / `tests/engine_llm_backend_test.py`（⑬ 系列 9 条：回执三态、`load()` 开头拒、缓存键、AST 判三调用口＋后端 3 处 `generate_llm`、`_graph_call` 关＝空字典、CPU 真跑 `last.reason=device=cpu`、⭐ ⑬⁷ 递下去必须是**三档原文**不是 `True`、⑬⁷‴ auto 回执三态（阈值／不是故障／本次为啥捕）分得开） / `tests/text_generation_test.py`（⑰¹–⑰⁵：`requested.cudaGraph` 缺省报 `None`、双向 mismatch、拼错只进 `errors` 不进 `mismatch`、⭐ ⑰⁵ **`on` 与 `auto` 各占一格**（压成 bool ⇒ 切档静默无效）） |
| **KV 显存预算（两本账，2026-11-08）**（`llm.kv_room_tokens` 纯算术：`free − 512 MiB 激活余量` ÷ 每 token 字节 ⇒ 还能放多少行 ⇒ **显存侧**窗口；与权重自带的 `*.context_length`（**权重侧**窗口）是**两本账**：低显存下先撞的一定是前者，而它**不写在文件里**、只能现探）。`LlmBackend.refresh_kv_budget()` 在 `load()` 成功后立刻快照（`torch.cuda.mem_get_info` ⇒ 可用字节），并算 `effectiveTokens = min(weightContext, vramTokens)` ＋ `limitedBy`；`_guard_context` 按两本账的 `min` 判，**报错区分是哪一本**（`by="vram"` ⇒「显存放不下这么长的上下文 ✗…（缩上下文／`kvMode=q8`／退卡后 `refresh_kv_budget()`）」，`by="weight"` ⇒ 原来的「上下文超窗 ✗」）—— 混成一句「超窗」会让人去改 `context_length`（改不动显存）。三态铁律：`free_bytes=None` ⇒ `maxTokens=None, ok=False`＝**探不到**（≠ `0`＝一个 token 都放不下）。回执 `describe()["kvBudget"]` 带 `mode/perToken/maxTokens/ok/freeBytes/reserveBytes/roomBytes/weightContext/effectiveTokens/limitedBy/note`；装载失败时置 `None`（不许拿上一份的旧快照） | `app/services/engine/llm.py`（`kv_room_tokens`／`KV_ACTIVATION_RESERVE_BYTES`） / `llm_backend.py`（`refresh_kv_budget`／`_guard_context` 双账） / `tests/engine_llm_test.py`（⑭¹–⑭³） / `tests/engine_llm_backend_test.py`（④ 系列补双账比对） |
| **KV 滑窗压缩（`kvWindow`，2026-11-08，默认关，有损）**（`llm.compact_kv` 保留**开头 `kvSink` 行（attention sink）＋最近 `kvWindow` 行**，中间整段丢弃 ⇒ cache 行数**恒定** `sink+window`，长对话不再线性涨 KV（StreamingLLM 的做法；`Q8Kv` 的 `q`/`scale` 两段一起压、`KvBuf` 返**独立副本**不就地改）。⭐ 与它配套的硬约束：压缩后**`cache 行数 ≠ 历史长度`** ⇒ RoPE 必须按**绝对**位置走（`generate_llm` 每拍传 `position_offset=abs_pos`，`LlmModel.forward` 的 `pos0` 用它）—— 位置跟着行数退回 ＝ **每滚一轮整体相位跳一次、且不报错**（只有生成质量悄悄烂）。⚠️⭐ **峰值口径：窗口只压生成期，管不了 prefill**（2026-11-09 修）—— 压缩只在**两个生成词元之间**跑（`_compact`），而 prefill 的提示词段是**整段一次性写进缓冲** ⇒ 缓冲上限必须按 `max(prompt + maxNewTokens, sink + window)` 留：只按窗口留 ⇒ 越长的提示词越在第 2 个 prefill 块就撞「KV 缓冲超预算」，而回执给的补救（缩上下文／`kvMode=q8`／调小 `maxOutputTokens`）**三条都不是真因**（实测 `kv_window=3, kv_sink=2` ＋ 20 行提示词 ⇒ 报「要 20 行、上限 15 行」，同一请求在预分配接上之前是能跑的；修后生成期照样压到 5 行 ⇒ 修的是峰值、不是「干脆不压了」）。闸门三条：与 `KvPrefixCache` **结构互斥**（缓存存的是完整历史、滑窗后那段已不存在 ⇒ 复用回来会让模型看见「本不该记得」的行；后端选择**滑窗优先**并在 `kvCache` 回执写 `suspendedBy="kvWindow"`，不许静默停掉）、`kvSink < kvWindow`（相等＝没有可滚的窗口）、`kv_window` 只接受正整数行数（`"3k"`／`True`／负数**当场具名拒** —— 静默当 0 的后果是「人以为开了滑窗、显存照涨」）。对外旋钮 `settings.kvWindow`（`off`/`0`/不写 ⇒ 0＝不滑窗、老路径逐字不变）＋ `settings.kvSink`（不写＝4；`0` **合法**，与 `specWindow` 那条「< 1 即拒」口径**不同**）；`text_generation.engine_kv_window_kwargs` 三处生成口共用，缓存键第 12/13 格 `win_slot`／`sink_slot`（关＝空串、**窗口关着时 sink 不进格**，否则写个 `kvSink=8` 就多装一份权重）；回执 `describe()["kvWindow"]` 带 `enabled/window/sink/rows/note` | `app/services/engine/llm.py`（`compact_kv`／`resolve_kv_window`／`resolve_kv_sink`／`KV_SINK_DEFAULT`／`generate_llm(kv_window=,kv_sink=)`） / `llm_backend.py`（`_kv_window_report`／`_kv_window_call`／`_kv_cache_for_call`） / `app/services/text_generation.py`（`engine_kv_window`／`engine_kv_sink`／`engine_kv_window_kwargs`） / `tests/engine_llm_test.py`（⑭⁴–⑭¹³：三存法逐位、独立副本、预分配钳制、**pre-hook 数行数＋位置序列**、两条闸门、归一函数边界；⭐ ⑭¹⁰′ **滑窗 ＋ 长提示词必须能跑**（20 行 > `sink+window+2+spec`＝15 ⇒ 峰值按 `max(total, sink+window)` 留，且生成期照样压到 5 行）） / `tests/engine_llm_backend_test.py`（⑮ 系列 7 条：回执、`load()` 开头拒、`kvSink>=kvWindow` 拒、缓存键、`kvCache` 暂停、AST 判三调用口＋后端 3 处 `generate_llm`） |
| **prefill 分块按显存预算分档（`prefill`，2026-11-08）**（块大小只解决**峰值**：`generate_llm` 里分数矩阵是 `O(chunk × 总长)`，与 KV 容量无关 ⇒ 块大小归**后端**，因为「显存有多紧」只有刚 `refresh_kv_budget()` 过的后端知道；调用方拿不到这个数只能写死 1024，而 2 GiB 空闲的卡上 prefill 3240 词元当场 OOM）。`LlmBackend._prefill_chunk()` 按 `kvBudget.freeBytes` 取档 `PREFILL_CHUNK_TIERS`：`≥4 GiB ⇒ 1024`（默认档 ⇒ `_prefill_call()` 返**空字典** ⇒ 调用签名与老路径逐字相同）、`≥2 ⇒ 512`、`≥1 ⇒ 256`、否则 128；**只缩不涨** ✗（8 GiB 也给 1024 —— 涨没有实测收益却把峰值翻倍）、`freeBytes=None`（没探到）⇒ 默认 1024（**不等于「很紧」**）。三处 `generate_llm` 统一展开 `**self._prefill_call()`；回执 `describe()["prefill"]{chunk,default,freeBytes,note}` | `app/services/engine/llm_backend.py` / `tests/engine_llm_backend_test.py`（⑯⁰–⑯²：分档表逐档、常见环境空字典、回执说「为什么」、AST 判三处都展开） |
| **LLM 后端档位回执口（`GET /api/v1/engine/llm`，2026-11-08）**（此前自研文本后端**一条 HTTP 口都没有** ⇒ `kvMode`/`kvWindow`/`gpuLayers`/`fastPath` 只活在文本服务的 settings JSON 里，而「配了没生效」看不出来：缓存键按档位分格 ⇒ 改完配置，已装后端仍是旧档位，显存/速度一动不动）。`text_generation.engine_llm_status(settings)` 把两侧并排报：`requested`（当前 settings 的档位，拼错进 `errors`）／`backends`（每份已装后端的 `device/dtype/gpuLayers/kvMode/kvWindow/kvBudget/kvCache/fastPath/spec/layerBytes`）／`matches`＋`mismatch`（只判 `kvMode`、`kvWindow`、`cudaGraph` 三个**纯函数归一**的格；`gpuLayers` 的实际层数依赖装载那一刻的显存 ⇒ 只报不判）／`duplicates`（**同权重多档位各占一份显存**：4B Q4_K_M 物化 bf16 ≈ 8 GiB，切一次档就多一份，只报不 release —— 还有调用在用旧后端）。⚠️ **GET 不装载**：探活把 8 GiB 权重推进卡是 GET 能造成的最坏副作用；路由只读 `_engine_backends` 现值 ＋ 库里的请求档位，`provider` 不是 `engine` 时如实说「不判请求 vs 生效」 | `app/services/text_generation.py`（`engine_llm_status`／`_norm_weight_path`） / `app/routers/engine.py`（`GET /llm`） / `tests/text_generation_test.py`（⑯¹–⑯⁷：空缓存不装、一致/漂移/未装载三态、拼错不假判、duplicates、只读不装、坏后端不砸整表） |
| **前端「自研文本后端档位回执」卡片（2026-11-08）**（`models.vue` 的「本地服务状态」分栏底部，**只读**，数据来自 `GET /api/v1/engine/llm`）。一屏回答三件事：① 已装的每一份后端 `device · dtype · kvMode · gpuLayers/(gpu+cpu) · kvWindow · prefill 分块 · KV 有效词元`；② **当前设置**（`requested` 全栏）与它的差 —— `mismatch` 点名具体键并明说「下次生成会按新档位另装一份」（改配置后正在跑的后端不会自己变）；③ `duplicates`「同一份权重占了多份显存 ⇒ 切档不会自己退旧后端，重启进程才回收」。⭐ 卡片上写明「装载只由生成触发，这张卡片不会把权重推进显存」—— 探活型 GET 最坏能造成 8 GiB 被推进卡，UI 必须说清。⚠️ **只读不写**：档位仍走文本服务的 settings（`kvMode`/`kvWindow`/`kvSink`/`gpuLayers`/`fastPath`），前端暂无编辑口；先有回执口是因为没有它「改了不知道生效没」，编辑口再快也没用 | `frontend/app/pages/models.vue`（`refreshLlmStatus`／`llmStateText`／`llmDuplicates`／`shortPath` ＋ `.llm-*` 样式；两端逐字同源） / `frontend/app/composables/useApi.ts`（`aiConfigAPI.engineLlmStatus`） |
| 进程内文本后端 | `app/services/engine/llm_backend.py`（`generate_messages` 走权重自带的 chat 模板；超窗在**进采样前**按 `*.context_length` 拒 ⇒ 不半路越界） |
| GGUF 反量化 | `app/services/engine/gguf_dequant.py` |
| GGUF → LlmModel 装载 | `app/services/engine/gguf_to_llm.py`（**tie embeddings 以张量表为准**：文件里没有 `output.weight` ⇒ 只能 tie ⇒ `tie_embeddings_of`，2026-11-07 真机 `Qwen3-4B-Instruct-2507` 两个来源都缺、旧实现按 `False` 判 ⇒ 完好文件被报「缺 `lm_head.weight`」；对账放行的凭据是 `lm_head.weight is embed.weight` 这件**事实**，不是配置位） |
| torch 后端（设备实测，不写死；**像素→潜变量只有一个口子 `_encode_pixels`**：H3 自带归一化 / Wan 走 16 组 meanstd / Flux 走 `(x−0.1159)*0.3611` 并补 `T=1` / 参考未训 VAE 走 `normalize_latents` —— 四族 `encode` 的返回口径（张量 / 二元组 / 未归一化）本来各不相同，裸调只会静默歪掉首帧条件） | `app/services/engine/torch_backend.py` |

### L4-6 `orchestration` 编排与可见面（12 + CLI 入口）

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
| **可跑档位与自动装配**（只报 `stage` 就能跑 ✓） | `inventory.assembly_for_stage()` + `GET /api/v1/engine/stages` | 清单条目按 `kind` 落槽（DiT / TE / VAE，`COMPONENT_SLOTS` 单一真源）⇒ 前端不必知道权重路径；Wan2.2 双专家按文件名 high/low 分 `dit_path`/`dit_path2`；`ready`（件齐）与 `fits`（装得下）**分开报**（`qwen_image` 件齐但 47 GiB 装不下 24 GB 卡）；词表**按档**解析（`tokenizer_hub.find_tokenizer_for_stage`：目录名/仓库名认族 + `tokenizer_class` 排错族双判据；落点 `tokenizers/<subdir>` ⇒ 仓内 `models/tokenizers/<subdir>` ⇒ HF 缓存快照；四档词表分别是 Qwen3-4B / Qwen3.5-9B / Qwen2.5-VL-7B / Qwen3-VL-4B，**绝不拿别族顶**——喂错族「装得上、编得出、id 全错且不报错」） |
| **画布引擎通道**（本机引擎 = 默认生成方式 ✓） | `frontend/app/composables/useEngineRun.ts` + `CanvasNodeCard.vue` 双模式 + `engine._local_media` / `_with_urls` | 画布配置卡默认走 `/engine/generate`（不依赖 8765/ComfyUI ✓），ComfyUI 工作流降级为可选通道；进度只读**事件**（`{stage,step,total}` ⇒ 按阶段区间插值，8 步采样不当全程 100% ✗）；回执补 `primaryUrl`（`primaryPath` 是本机绝对路径 ✗，数据根外的产物**不硬转** URL ✗）；画布素材 URL → 本机路径（`/uploads|static/…` 严格解析、裸文件名放行、远程 URL 当场拒）；I2V 档（`wan22`/`seedvr2`）取连线第一张当首帧（`h3` 是文生视频，**不逼**首帧 ✗）；`taskId` 落 `localStorage` ⇒ 刷新可接回（恢复时**已终态只回状态文本**、不重复建卡），任务号失效（引擎重启）⇒ 忘掉并回 idle（不转假圈）；`E_ENGINE_BUSY` **不标失败**（是另一张卡在跑，人话提示稍等）；**时长留空 = 5 秒**这条默认在视频档是**重成本**（Wan2.2 16fps ⇒ 81 帧、H3 24fps ⇒ 121 帧；24 GB 卡装不下 26.62 GiB 的单份专家 ⇒ 走 split、注意力随帧数平方涨 ⇒ 实测 81 帧 @608×352 跑到 14 分钟才第 2 步，而 17 帧 @576×320 历史 155 秒落盘）⇒ 界面按时长实时算帧数并提示，`/engine/stages` 的 `fits=false` 时明说「先 1 秒验通路」；**引擎的 `plan.warnings` 必须显示**（尺寸被吸附成多少 / 参考图被忽略 / 24fps 换成 16 这类「不报错但与你以为的不同」全在那一栏，后端刻意不当 error ⇒ 界面不显示就永远没人看见）；连线图**三条去向各不同**且界面写明（I2V 首帧 / H3 全当参考图 `ref2va` / 图片档「起点图 denoise<1」→ `initImage`；都不满足 ⇒ 红字「本档不吃参考图 ⇒ 会被忽略」＋后端 `_warn_unused_references` 兜底同一句——八条线里**只有 H3 真读** `reference_frames`，其余七条一条字节都不用，静默忽略正是本仓最恨的一类）；**音频档（`music3`）自 2026-11-06 起在线** ✓：产物是 wav ⇒ `generated` 的 kind **先按后缀判**（`mp3/wav/flac/m4a/ogg`）再按档位特征 ⇒ 画布建的是**音频卡**（`<audio controls preload="metadata">` ＋ 固定 54px 高）而不是一张裂图（⚠️ 三处 `:src` 一律过 `mediaUrl()` —— `node.url` 存的正是根相对的
`primaryUrl`，**裸绑**在桌面壳里会打到 `tauri.localhost` 整片 404，而 dev 被 Vite 的
`/static` proxy 掩盖；守卫 `check-isolation` 的 F 段已把 `node.url` 纳入字段名并加了自检）；配置卡多一格**歌词**（`values.lyrics`，段落标签用方括号、` ^ ` 当换行）、时长那格改用 `needsSeconds`（视频与音频都要——音频同样有「留空＝5 秒」这个默认，留空只够一句）；`lyrics`/`seconds` 允许写在工具调用**顶层**（`canvas_ops._merge_top_values` 并进 `values`，显式 `values` 优先，否则模型按 `text` 直觉写在顶层时歌词会**静默消失**）；⚠️ 同轮还查到一起**静默丢档**：`useCanvas.normalizeDoc()` 重建节点是**白名单**，`engineStage` 漏在外面 ⇒ 用户选了 Music3、刷新一次档位就回落成清单第一个可用档（Z-Image），点生成跑出来的是一张图，而 `persist()` 会把清洗后的对象写回盘 ⇒ **丢是永久的**；已补 `runMode`/`engineStage`，判据 `canvas_ops_test.py`「normalizeDoc 白名单覆盖 CanvasNode 全部字段」（解析接口字段 vs 搬运字面量，实测删掉那行即红）；媒体口径 2026-11-06 已**两端统一**：`mediaUrl()` 是两端逐字同构的唯一出口（`pages/drama/*`、`pages/library/*`、6 个编辑/媒体组件、画布页与节点卡片全部改齐），`openImageViewer` 在**收口处**过格式化（模板是裸 `<img :src="imageViewer.src">`，20 余处调用方送根相对路径），成片下载改 `fetch→blob`（跨源 `<a download>` 被浏览器忽略，会把整个 SPA 顶成视频）；`check:parity` 白名单相应从 21 条收到 14 条 |
| **产物落点兜底（唯一真源）**（没给 `outputs_dir` ⇒ `<storageRoot>/engine/<kind>/`；干跑 ⇒ `<dataRoot>/engine-dryrun/<stage>/`；`DARME_ENGINE_OUTPUTS` 整棵改根） | `app/services/engine/output_paths.py` | 四个落盘入口（`torch_backend.write` / `dryrun.write` / `chain.run_chain` / `sdxl_backend.output_path`）此前各自兜底：三处 `tempfile.mkdtemp(prefix="engine_*")` ＋ 一处 `Path.cwd()/"outputs"` ⇒ **每跑一次多一个新目录**、图散在 `%TEMP%` 且系统一清就没了、想留还得人工搬；现在同一处真源；`bridge.outputs_dir_for` 的**根**也改从这里取（业务通路 `<kind>-<id>` 与兜底必须是同一棵树，否则盘上长出两个 `engine/`）；**只兜底**（调用方给了就一字不改）、**不编时间戳**（同 seed 覆盖同名文件 —— 文件名是回执的一部分，编时间戳会让"同一请求"不可复现）；干跑不进 `static`（报告不是产物，落进去会被当"成品链接"点开）；守卫 `engine_output_paths_test.py`（AST 判四处函数体零 `mkdtemp`/`tempfile`/`cwd`、全目录零 `mkdtemp`、`tempfile` 只许自动清理的 `TemporaryDirectory`） |
| **引擎 → 服务层的单向接线口**（引擎不 import 服务层 ✓） | `app/services/engine/hooks.py` | 引擎有四处确实要读业务数据：跑次索引（`outcome_index`）/ 知识台账（`knowledge_ledger`）/ 质检复核（`qc_reviews`）—— 那三块都在**服务层**。原先的做法是在函数里 late import，形式上躲开「顶层依赖」，方向却仍是引擎 → 服务层 ⇒ 引擎被单独拉起（CLI / 纯引擎进程）时隐式拖上 sqlite 与业务表，且 `tuning.suggest()` 会表现为「ImportError 或空数据」（**不报错地给 0 条建议 = 假绿** ✗）。现在反过来：服务层在 `app/main.py` 启动时 `hooks.provide(run_index=…, ledger=…, qc_reviews=…)` 登记**模块对象**，引擎只查 `hooks.SOURCES` 这张表 ⇒ 登记了走真数据（Web 通路行为一字不变）、没登记各调用点**如实说「没接入」**（`suggest()` 不给候选并写明原因；`check_evidence()` 返回 `ok=False` 且带 `unavailable=True`，**不是静默放行** ✗）；名字写错当场 `ValueError`（不静默丢），**读侧**一样拦（`get()`/`provided()` 传未知名字也是 `ValueError` —— 静默回 `None` 会把「我拼错了」伪装成「服务层没登记」，排查方向直接错 ✗），`provide(x=None)` 同样拒（等价于没登记却让 `provided()` 说「登记了」，两口自相矛盾 ✗）；判据 `engine_independence_test.py` ②（引擎顶层只许 `app.core.*` 与自身，`ast.walk` ⇒ 函数内 late import 也算越界；★**相对写法同样折算**：`from .. import outcome_index` = `app.services.outcome_index` —— 此前只收 `level == 0` ⇒ 整类相对越界被放过 ✗；残留的**一处**服务层依赖（`bridge.py`：`task_logger`＝任务日志面板，数据源只在服务侧）改成 `_SEAMS` 逐条登记「为什么现在非碰不可／怎么撤销」—— 2026-11-08 销掉两条：`outcome_index` 改走 `hooks.get("run_index")`（原先在 `_remember_record` 里 late import ⇒ 方向仍是引擎→服务层，纯引擎进程还会拖起 sqlite；没登记现在**如实说没接入**而不是静默跳过），`file_storage` 的两个纯函数搬进 `app/core/media.py`（服务层改为再导出**同一个对象**，②b 用 `is` 判「搬家 ≠ 抄一份」），`local_model_scan` 的 ComfyUI **安装根探测**搬进 `app/core/comfyui_roots.py`（`bridge.resolve_tokenizer_path` 直接取 core ⇒ `bridge.py` 那条随之销掉；探测层在服务层只剩再导出 ＋ 一层薄委托，且 `path=` / `detect=` **两个**隔离口都得显式传下去 —— 漏 `detect=` 会让本套 patch 的假根静默失效），**同日再把「默认扫描根」整体搬进 `app/core/model_roots.py`**（连带 `app/core/model_folders.py`／`model_ecosystems.py`／`ollama_roots.py` 三个落点模块 ⇒ `inventory.candidate_roots()` 不再 import 服务层、`inventory.py` 那条也销掉；扫描层 `get_default_roots()` 成薄委托，`path=` / `comfyui=` 两个隔离口同样必须显式传，判据是 `local_models_test` 的两条 AST 守卫 ＋ 八处再导出的 `is` 判「搬家 ≠ 抄一份」），③ 双向对账 ⇒ 修好的必须删表、偷加的一条立刻红，⑤ 另判**模块加载期**不许碰服务层）+ `engine_hooks_test.py`（33 条：表与读者**双向对账**「表里有没人读＝死源／有人读表里没有＝读空」、`app/main.py` 登记键与 `SOURCES` 逐字相同、未登记时 `_scan()` 回 `available=False` 而不是 `True`、**谁调 `tuning.suggest/set_field` 或 `qc_feedback.*` 的套件必须自己 `provide(...)`** —— 少写一行的症状是业务断言红、报错不指向接线 ✗；★另加**可见面**：`hooks.status()` 与 `python -m app.services.engine doctor` 的「接线」行逐条点名未登记的源并写明是「没接入」**不是**没有数据，`doctor --json` 的 `wiring` 与 `status()` 逐字段同源；`capabilities()` 有意**不带** wiring（静态表要跟 `docs/env/*.json` 逐字段对，塞进程态会让它随进程漂 ✗）） |
| 干跑后端（无 torch 也能跑整条管线） | `app/services/engine/dryrun.py` |
| 业务链路 → 进程内引擎桥 | `app/services/engine/bridge.py` |
| **H3 文本条件编码串联**（呈现格式 → 逐段分词 → 视觉塔 → 写 embeds → 语言塔 ⇒ `H3Condition`：hidden / tags / ids / spans；`combine()` 把两座塔并成一个编码器且**不重复注册** `model.*` 键 ⇒ 与盘上权重全名空间逐字对得上；不写模型数学 ⇒ 归编排不归 `model` 组）；**装配口** ＝ `TorchBackend.attach_h3_text`（两半塔**同文件一次装**、int8 常驻、切分驻留先切再装）⇒ `_encode_h3_condition` 回执里同时给出 `hidden` 与 **`tags`**（0 = video 行 / 1 = text 行 ⇒ 采样侧 `mod_segments_for` 靠它切 run）；参考**视频**按 `videoIndex` 选路（单张量 = 单路；列表 = 一路一个，越界/缺路当场拒），每块按 2 fps 抽帧后奇数帧补末帧且**时间戳重复末值** | `app/services/engine/h3_encode.py` |
| 导演稿 → 生成段（纯规则） | `app/services/engine/script_parse.py` |
| **参数覆盖层**（调优结论 → 真改行为，可回滚） | `app/services/engine/tuning.py` | 外部 `configs/engine/tuning.json` + 白名单 `TUNABLES`（逐族登记 `steps` 与区间，**不通配**）；只覆盖「调用方没显式表态」的值（`request.steps == default` 才换），没有文件就**行为逐字不变**（所以不需要开关）；每次写先备份到 `configs/engine/_backups/`；`set_field` 必须给 `basis` + 可核 `evidence`，否则拒写；生效时 `plan.warnings` 写明「参数来源: tuning:<stage>.<field>（30 → 24，依据 …）」。**同一个白名单还挂了第 5 个"伪档" `qc`**：九个质检阈值（`blackMinDuration`/`blackPixThreshold`/`freezeMinDuration`/`freezeNoiseThreshold`/`truePeakDb`/`loudnessTarget`/`loudnessTolerance`/`fpsMin`/`durationToleranceSec`）不是引擎档、`GenerationRequest` 上也没有这些字段 ⇒ `apply_to_request("qc", …)` 永远换不进请求，只有 `technical_qc.qc_thresholds()` 读它；好处是**证据核验、备份、history、`report()`、`clear_field` 回滚**一套白拿，不用为质检再造一份写入口；默认值必须与 `TECH_QC_THRESHOLDS` **逐值相等**（`qc_thresholds_test.py` 的 AST + 逐值比对判据守住，常量本身不许被覆盖写回）。**读向是 `suggest()`**（`GET /api/v1/knowledge-loop/tuning/suggest`）：只读 `outcome_index` 的跑次，出「失败聚集」（某族 `failed` 过样本闸门且最大 `error_code` 占比 ≥50%）与「步数平台期」（`meta.steps` 分桶，最高档与次高档均分差 <0.01）三类候选（2026-11-07 补第三条「质检命中率」：读 `qc_reviews.hit_rate()`，`reviewed ≥ MIN_REVIEWS` 才出，`false_alarm_heavy` ⇒ 放宽、`miss_heavy` ⇒ 收紧，幅度是 ×1.25/×0.8 起手值并在 `why` 里如实写明 **方向有据、幅度不是结论**）；它**一行都不写**（AST 判「除四个写口外无人调用它们」），`canApply` **恒 false**（影子验证不在本模块），`notEvaluated` 如实列出算不了的维度（sampler/schedule 不在白名单）；**两个输入侧写点也已接**（此前只有 agent trace 进索引 ⇒ 夹具一撤就"没有聚集"，`outcome_wiring_test.py` 28 条）：`runtime.py::_index_run` 在任务封口后落账（`kind` 按产物后缀、`error_code` 只取类别、meta 带 `steps/seed/cfg` 等分桶原料、写盘在 `with self._wake` 之外），`technical_qc.py::_index_qc_run` 在质检写回后落账（确定性 `run_id` 覆盖同一条、分数归一到 0–1、规则维度失败标 `cause=rule` 不编伪码），超过 500 条扫描量标 `sampled=true`，候选 `evidence` 的 run_id 要过 `check_evidence` |
| **质检反哺引擎参数（知识闭环缺口 6 的另一半）**（问题不是"少一条规则"而是**归因断了**：`technical_qc` 写的是伪档 `stage="video_spec"`（"这是规格判据"，不是"哪族出的"），引擎跑次写的是真档 `h3`/`wan22`/`krea2`，两边**没有共同键** ⇒ 「这一族最近质量掉了」永远成不了一条关于这一族的结论。归因键＝`outcome_index.records` 表：`bridge.run_job` 在**刚拿到 `taskId` 的那一行**登记 `(video|image, 记录 id) ⇒ (引擎档, 引擎跑次)`（过了这行 `stage` 与 `video_generation_id` 就再也碰不到一起了），`technical_qc._index_qc_run` 写回时查它、把 `producerStage`/`engineRunId` 放进 QC 行的 `meta`；**查不到就不带这两格**（不是填 `null` —— 云 provider 的片没有"引擎档"，硬填就等着被算进某族）。⚠️ 为什么要这张表：`video_generations.task_id` 是给云 provider 存轮询号的，引擎这条路本来就不写它 ⇒ 两个方向都 join 不上。读向三件：`stage_stats()`（按档报条数/通过率/失败 code 分布，`attributed` 与 `unattributed` **分开报**）、`pass_by_steps()`（同族按 `meta.steps` 分桶算 **QC 通过率**，拿不到步数的单独计 `missingSteps` —— 那是"这次比不了"不是"没通过"）、`gate(high,low)` 三档 `insufficient`/`pass`/`blocked`（每桶 ≥ `MIN_BUCKET=5` 条才敢判，**样本不足≠通过**；低档通过率掉出 `TOLERANCE=0.1` 就拦）。接线：`tuning._steps_candidates` 出「降步数」候选**之前**问闸门 —— `blocked` 当场不出并留 note（不是静默丢弃），放行则把 `qcGate` 结论随候选带出去；顺带补上一个真缺陷：引擎跑次**根本不记 `score`**（`runtime._index_run` 无此字段）⇒ 那条"步数平台期"规则在真机上一向空转，质检通过率才是真能判的那个量。库坏/归因表空 ⇒ 一律答"判不了"并退回原行为（主链路不因统计不可用而卡）。⚠️ 索引库**由服务层登记**（`engine/hooks.py`，引擎不 import 服务层 ⇒ 相对写法也不行），没登记时四个口统一答 `_ABSENT`「跑次索引没接入」**而不是**「没有质检数据」（后者会把排查方向带向"再攒点 QC"）；判据 `qc_feedback_test.py` ⑧（clear 后逐口验真因、再 provide 验残态已消） | `app/services/engine/qc_feedback.py` |
| 独立命令行（`doctor`/`plan`/`readiness`/`inspect`/`manifest`/`status`） | `app/services/engine/__main__.py` |

> `__main__.py` 是**入口脚本**，不在 `MODULE_GROUPS` 的 63 个模块里（㊀ 的登记表只管模块）。



### L4 的两处已知边界

- **未接线的 1 个模块**（已实现、只有测试在钉口径）：`mappings`
  —— 它该接到谁身上，登记在 `tests/structure_test.py::UNWIRED`，接了线就该删那条登记。
  （`dit_lumina` 2026-10-31 已接线 ⇒ 登记已删，见下一条；`latent_formats` 2026-11 已被
  `dit_build` / `vae_wan` / `pipeline` 用上 ⇒ 登记同样已删；`image_ops` 2026-11-08 接在
  超清二采的「缩到计划倍率」那一步 ⇒ 登记已删，见 §10.1。）
- **有骨架的族有 10 个**（判族另算：`dit` `dit_family`）：H3（`h3_form` `vae_h3` `audio_vae`）、
  SDXL（`sdxl` `sdxl_vae`）、
  GGUF LLM（`llm` `gguf_to_llm`）、**通用单流 DiT**
  （`dit_generic`：Krea2 一族已能建结构并前向，维度全部从权重形状反推；**TE/VAE 与采样也已接通**，
  `stage=krea2` 从提示词一路到 PNG，见 10.1d）、**Z-Image**
  （`dit_lumina`：判族命中 `zimage` ⇒ `torch_backend` 按权重形状建 `LuminaDiT` 并接进前向，
  与盘上 453 张量逐键对账；**TE/VAE 与采样也已接通**，`stage=zimage` 从提示词一路到 PNG，
  并已进 `runtime.ensure_loaded` 装配口，见 10.1b）、**Qwen-Image 2.1**
  （`dit_qwen21` ＋ `qwen35_te`：族 `qwen_image21`、盘上 649 键、与 comfy 逐位对拍；
  `stage=qwen_image21` 是第四条装配线，见 10.1c）、**Qwen-Image 2512 / Edit 2511**
  （`dit_qwen_image` 双流 MMDiT ＋ `qwen25_te`：结构与前向都写了、装配线也接了
  `stage=qwen_image`，与盘上 1933 张量逐键对账；Edit 2511 的参考图通路 2026-11-01 已接
  （含**多张**参考图），**但还差一次真权重实跑出图**，见 10.1c）、**Flux.1-dev**
  （`dit_flux` 双流 19 + 单流 38：判族命中 `flux` ⇒ 按 `dit_build` 表建网并接进装载/前向，
  与盘上 780 张量逐键对账、与独立双精度参照一致；**条件靠外部注入**（盘上无 T5-XXL/CLIP-L），
  见 10.1e）、**Wan2.2 I2V 14B**
  （`dit_wan22`：40 块 / 5120 / 40 头、真头 1095 键与 14 288 901 184 参数双向对账、与独立 fp64 参照
  同一前向；**I2V 36 通道拼接未接、26.62 GiB 装不进卡** —— ⚠️ 原写的「文本塔（umt5）无实现」已于
  2026-11-01 **更新**：塔本体 `umt5_te` 与分词器 `umt5_tokenizer` 都已写出、与 comfy 逐位对拍已过，
  **但装配线还没接** ✗，见 10.1f / 10.1g）、**SEEDVR2**
  （`dit_seedvr2` ＋ `vae_seedvr2`：骨架形状反推 ＋ 前向口径 ＋ 真实 3B 权重头对账，
  2026-11-01 **已接进 `dit_build` 与 `stage="seedvr2"` 装配线并真权重出片跑通**
  （720×480 / 8fps / 49 帧 ⇒ 1.27 MB mp4、715 秒、`ok=True`），唯一短板是**解码慢**
  （单段 336 秒、峰值 47.6 GiB，潜帧分块已被实测否掉）见 10.1g）。
  Music3 **已不再是「只有格式」** ✓（2026-11-02）：`dit_music3` 骨架 ＋ 前向口径（σ 翻转取负、
  RoPE split-half 正号 / 只转 q/k 前 `rotary` 维、潜帧率 86.1328 Hz、689/344 滑窗）与
  `vae_music3` 的 DAV 解码（121 键两套写法两向互归、128 = 2 × 64 行主序折半）都写完并自检过
  （`engine_music3_test` 46 项）。**2026-11-06 接线已闭合** ✓：`stage="music3"` 装配线
  （`runtime._ensure_music3_loaded` 三格同驻 13.34 GiB、不分时）＋ 音频 plan（`_fill_music3_plan`：
  秒 →×25 条件帧 →×3.4453125 潜帧、`width/height` 恒 0）＋ 歌词-文本塔（`encode_text` 的
  `music3-ar` 支）＋ 解码（`decode` 的 `music3-real` 支）＋ 出盘（`write` 的纯音频支 → 44.1k 立体声 wav），
  长音频滑窗仍在 `velocity()` 里（`pipeline` 只诊断窗数、绝不重切 ✓）。逐处对账 56 项
  （`engine_music3_wiring_test`）全绿，**2026-11-06 真权重已出歌** ✓（5 秒档 ⇒ 4.99 s / 880 684 B / 2 声道 / 44.1 kHz / 峰值 −6.6 dBFS、全链 73.6 s；20 秒档 ⇒ 19.99 s / 3 526 700 B / 881 664 样本 / RMS −20.5 dBFS、编码 249.2 s ＋ 采样 9.3 s ＋ 解码 0.35 s、全链 268.9 s；谱形不是噪声（谱平坦度 0.14、主峰 258/86/433 Hz），逐秒 RMS 有起伏）。首跑连抓**四处**「前面几环全绿、最后一步才炸」的接线坑：DAV 键名两套写法（`shape_problems` 归一了而装载器没归）、条件留在 CPU（塔有意腾卡）、解码入口 dtype 未对齐、解码分支误引不存在的 `vae_music3.SAMPLE_RATE` —— 四处都补了成对判据（A9–A16）。接线对 ≠ 出歌对。
  **对参考实现逐处对账又抓到一处真差异** ✓✗（同日第二轮）：σ 日程。本族是 `ModelType.FLOW` ⇒
  `ModelSamplingDiscreteFlow` 取 `shift=1.0`（`sampling_settings` 只写了 `multiplier: 1.0`）＋
  `KSampler` 默认 `scheduler="normal"` ⇒ 落到 σ 上是**σ 等距**、端点 `(1.0, 0.001)`、末尾 0；
  原先走的是 `_image_flow_sigmas`（**t** 等分、端点 `(1.0, 0.0)`）—— 30 步时末段是
  `0.0333 → 0` 对 `0.001 → 0`，**噪声跳幅差 33 倍**，症状只是「尾巴那点残噪没清」，不报错、
  不看波形发现不了。改 `schedules.normal_sigmas(steps, shift=MUSIC3_FLOW_SHIFT)`，判据 A17a/b/c
  （逐值对拍 ＋ 端点 ＋ 源码不许走回 `_image_flow_sigmas`），本套 59 项。
  **2026-11-06 又走通了 HTTP 通道**（`_bench/music3_api.py`，`POST /api/v1/engine/generate` ＋ `wait`）：
  5 秒档 HTTP 200、74.0 s、`primaryUrl='/static/engine/audio_seed777.wav'` ⇒ 前端音频卡拿得到可播放地址。
  ⚠️ 任务成功态是 **`done`**（不是 `succeeded`）—— 拿错名字会让跑成功的任务在收尾那一刻被判失败。
  画布侧同轮补齐：`canvas_ops.NODE_TYPES`/`GENERATION_MODES`/`DEFAULT_NODE_SIZE` 三处加 `audio`、
  `canvas_generate_audio` 从「未迁」挪进「已迁」（22＋12=34），`CanvasNodeCard` 真渲染 `<audio>`。
  F5-TTS 仍是**只有格式与量化支持，没有前向** ✗ —— 注意这**不是「缺权重」**：
 `D:/Comfy-Desktop/ComfyUI-Shared/models/checkpoints/F5-TTS/model_1250000.safetensors`（约 1.29 GB）
 **在盘上** ✓、`inventory` 也认得出它；缺的是**网络本体**（flow-matching DiT ＋ 声学解码）与
 vocoder 接线 ⇒ 目前**装配不出**（如实报 pending，不静默降级 ✗）。要补的是代码，不是下载。
  见第 9 节。

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
| 文件操作 | `app/services/lora_train/dataset/files.py` | 改名/复制/转格式；备份目录在**素材目录同级**（在里层会自己抄自己）；转格式回执里的 `backup` 就是**这次真用的那个路径**（旧写法收尾再调一次 `backup_directory()`，跨秒会报出不存在的目录） |
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

## 7. L6 平台层（`app/core/` 13 模块，3 个域）

最底层：谁都能依赖它，它不依赖任何业务。`core` 只允许 import `core`。

### L6-1 配置、路径、媒体与模型落点（8）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 全部路径与环境变量真源 | `app/core/config.py` | `skills_dir()` / 数据目录 / 锁文件位置等，**别处不许自己拼** |
| **模型根在哪（配置面）** | `app/core/model_paths.py` | `models_dir()` 是**唯一**解析口：`MODELS_DIR` > `configs/model-paths.json` 的 `models_dir` > `<data_root>/models`，三项顺序不许改（与 `local_model_scan.get_default_roots()` 第一个根必须同源，否则「体检说这里有、加载说那里没有」）；`read/write_paths_config` 的 `path=` 是测试隔离口，读、写、`models_dir()` 三处**必须共用**同一个位置（只让读走那个口 ⇒ 一次自检就把真配置冲掉）。只管配置面；ComfyUI 探测／生态落点／目录声明／Ollama 库根另有四个 `core` 模块（下一行起），2026-11-08 起「默认扫描根」的拼装也收进 `app/core/model_roots.py`，`local_model_scan.get_default_roots()` 只剩薄委托。唯一登记在案的**副本**是 `app/scripts/model_manager.py::resolve_models_dir`（`scripts/` 只吃标准库、不 import `app.*`）—— 登记≠放行：`tests/local_models_test.py` 拿三档情形与本体逐档对拍，且 `local_model_scan` 里每处调用都必须传 `path=`（AST 守卫） |
| **ComfyUI 类目录在哪（扫描面 ＋ 安装面）** | `app/core/comfyui_roots.py` | 回答两个**不同**的问题：①「扫模型」的根 ⇒ `<根>/models`（`local_model_scan.get_default_roots()` 的第一类）；②「安装根」⇒ `<根>/comfy/...` 这类**安装内部**落点（`engine/bridge.resolve_tokenizer_path` 找 `sd1_tokenizer` 词表就靠它 —— SDXL 的 CLIP-BPE 词表**不是权重**，按「有没有 `.safetensors`」永远找不着）。它住在 `core` 是为了让引擎不必 `import app.services.local_model_scan` 才能知道自己去哪找词表，那正是 `engine_independence_test._SEAMS` 要逐条解释的越界。只读配置与盘（`configs/model-paths.json`／`COMFYUI_PATH`／盘符枚举／目录名），一个业务数据都不碰 ⇒ 与 `model_paths` 同一类。分层**只搬到这里**：`get_default_roots()` 的另外几类根（`extra_roots`／`model_ecosystems` 生态落点／`model_folders` 声明表）要读生态约定与用户声明，属「模型库功能」，仍留在 `local_model_scan`（`_SEAMS` 里 `inventory.py` 那条登记的就是这半）。⚠️ **2026-11-08 当日这半也搬齐了**（下面四行）：`_SEAMS` 里 `inventory.py` 那条已删、`bridge.py` 那条同日先销（本模块搬来时）。服务层那份现在是**再导出同一批对象**（`SKIP_DIRS`／`list_drives`／`detect_comfyui_roots` …）＋ `get_comfyui_roots()` 一层薄委托，且委托必须同时传 **`path=`**（可 patch 的 `_paths_config_file()`）与 **`detect=`**（本模块那个可 patch 的 `detect_comfyui_roots`）—— 跨模块 monkeypatch 不生效，漏传一个就是「换了文件还在读真配置」或「注入的假根不生效」的假绿（`local_models_test` 两条 AST 守卫 ＋ `engine_bridge_test` 的替身打在使用点 `bridge.comfyui_roots`） |
| **媒体引用 → 本机路径**（`static/...` 怎么解成绝对路径、data URL 怎么拆包） | `app/core/media.py` | `get_absolute_path()`（**防路径穿越**，全仓唯一实现）＋ `parse_data_url()`。两者都是从 `app.services.file_storage` **搬**来的纯函数（一个只用 stdlib、一个只用 `config.get_storage_root()`）：引擎读参考图必经这两步（`bridge.materialize_reference`），而引擎不许 import 服务层 ⇒ 留在服务层就是一条无谓的越界（`_SEAMS` 里那条已删）。`file_storage` 现在只**再导出同一批对象**（`file_storage.get_absolute_path is app.core.media.get_absolute_path`），业务侧十来个调用点一字不改，也不存在第二份实现 |
| **ComfyUI 目录类别表 ＋ 任意目录声明** | `app/core/model_folders.py` | 两件事：①按**父目录名**（ComfyUI 口径）再按文件名认「这是哪一类权重」＋下载/安装时挑类别子目录、校验扩展名；②读用户**已经声明**的任意目录（ComfyUI 的 `extra_model_paths.yaml`／`model-paths.json` 的 `extra_paths`／环境变量 `EXTRA_MODEL_PATHS`）⇒「放哪个盘都能用，声明一下就行」。原生就零服务层依赖（只用 `json`／`os`，读的是约定与用户声明，不碰业务数据），2026-11-08 整份搬进 `core` 只为让引擎 `inventory.candidate_roots()` 能看见声明过的目录（引擎顶层只许 `app.core.*`）。口径来源是 ComfyUI 的 `folder_paths.py`／`utils/extra_config.py`，但那份参考副本是 **GPL-3.0** ⇒ 只搬语义、代码自写。服务层 `services/model_folders.py` 现在纯再导出（`FOLDER_SPECS is` 同一对象，`local_models_test` 判 `is`） |
| **Ollama 库根候选** | `app/core/ollama_roots.py` | 只答「盘上的 Ollama 库可能在哪」，顺序是 `OLLAMA_MODELS` > Ollama **桌面端** `db.sqlite` 里声明的库根 > `%LOCALAPPDATA%\Ollama\models` > `~/.ollama/models`（保序、`normcase` 去重）。⭐ 为什么非得读桌面端那个库：桌面端把库装到自定义目录时，`OLLAMA_MODELS` 只注入给 `ollama serve` 子进程 ⇒ 后端自己看不见（真机复现过）。读法是 `mode=ro` ＋ 0.5 秒超时（桌面端此刻也在写那个库，绝不能建日志/改文件），并按 `mtime + 大小` 缓存（请求链路上会被反复调用）。**只搬「候选」那半**：`models_root()`（挑一个真能读的）与 `is_available()`（看 `manifests/`）留 `services/ollama_store.py` —— 那两个回答的是「用哪个、能不能读」，属 store 自己的口径；`ollama_store` 现在对这批名字纯再导出（判 `is`） |
| **各生态落点表** | `app/core/model_ecosystems.py` | 11 个生态（HuggingFace / ModelScope / LM Studio / GPT4All / Jan / llama.cpp / text-generation-webui / SD WebUI / Stability Matrix / InvokeAI / torch hub）的本地落点 ＋ 命中文件归属哪个生态 ＋ 「这个权重能不能直接调、怎么才真能用」（默认 `callable=False`＋`note`：缓存 ≠ 能推理）。原生零业务数据；唯一的服务层牵连是 Ollama 那一行的 `candidates`，现已改取 `app.core.ollama_roots` ⇒ 整份能住 `core`。⚠️ 排在 ComfyUI **之前**：ComfyUI 的 `models/` 动辄上万文件，后置会被扫描上限吃光而「扫不到也不报错」 |
| **默认扫描根的唯一拼装** | `app/core/model_roots.py` | `default_roots()` 是「该去哪些目录找模型」的**唯一一份**：`models_dir` → `local_services` → `extra_roots`（显式意图优先）→ 声明过的任意目录 → 各生态落点 → **每个** ComfyUI 类目录的 `models/` → 显式配的那个根整体（只加显式那个：探测到的其它安装整包动辄上万文件，会把扫描上限吃光）。扫描层 `local_model_scan.get_default_roots()` 与引擎 `inventory.candidate_roots()` **同调本函数**（根表只此一份：各写一份必然漂成「扫描能看见、加载看不见」）。⚠️ 五个隔离口全部可注入（`path=`／`comfyui=`／`extra_roots=`／`declared=`／`ecosystems=`）：跨模块 monkeypatch 不生效，服务层那层薄委托**必须**同时传 `path=` 与 `comfyui=`，少传一个就是「换了临时配置还在读真配置」或「注入的假安装根不生效」的假绿（`local_models_test` AST 守卫 ＋ 反验） |
| **产物落点登记表（全仓一张表）**（`static/` 16 个子目录 ＋ 数据根 9 个子目录，其中实验落 `experiments/<实验名>/`） | `app/core/config.py`（docstring 那张表）＋ `tests/output_layout_test.py`（机读白名单 `STATIC_DIRS`/`DATA_DIRS`，13 条判据）＋ `tests/export_service_test.py`（中转件 3 条） | 三路合并对账、**只判"多出未登记"**：`get_storage_root() / "<子目录>"` 字面量、`get_absolute_path("<子目录>")`、回执 URL 前缀 `static/<子目录>/…` —— 三处合起来就是"盘上真会长什么"，出现表外目录即红（写死名单而不是扫盘，因为目录名多半是参数化的，扫盘认不出）。**铁律：`get_storage_root()` 返回的**已经**是 `<dataRoot>/static`**，后面只许直接跟子目录名；再拼一次 `"static"` 会落成 `static/static/xxx`，而回执 URL 指 `static/xxx` ⇒ 文件在、链接 404（`video_interpolate` 插帧就是这么裂的，`ffmpeg_compose._MIX_SUBDIR` 同病写成 `"static/mixed"`）；新增落点必须**同时**改那两处（表 + 白名单），否则测试红。四类不同待遇：**产物**进 `static/`（要发 URL）、**中转**（抽音轨/切片/`ffmpeg_merge` 的 concat 清单）用 `tempfile.TemporaryDirectory` 自动清理且不进 `static`（`static/temp` 已从白名单撤下 —— 那是产物区，进程一被杀就永久留垃圾）、**实验**一律进 `<dataRoot>/experiments/<实验名>/`（`backend-py/output/` 已退役、存在即红；探针脚本不许往代码目录里扔；**不许进 `static/`** —— 那会被 `/static` 公开发布并当成品回执）、**现场证据**进 `backend-py/tmp/`（只许日志与脚本，媒体产物一律不收，判据 ⑨）。**一次请求的中转件**（目前只有导出包）单独一条：落 `<temp>/darme-export/export-*.zip`（`export_service.export_tmp_dir()`），**不许落 `%TEMP%` 根**，且必须三道删除 —— `BackgroundTask` 正常删 ＋ 端点 `except` 里自己删 ＋ `build_export_zip` 建包前 `sweep_stale_exports()` 清超龄件；**再加启动兜底**：`main.py` 的 `lifespan` 启动时也清一次（建包前清扫依赖"下次还建包"，用户几个月不导出就永远等不到，那包可能几十 GB）。H3 本地原生服务的默认落点是 `<storageRoot>/h3`（`_default_output_dir()` 先问 `app.core.config`，ImportError 才退回 `<repoRoot>/data/static/h3`），**不再**落源码目录 `local_services/h3/outputs` —— 那里重装/切分支就丢，主后端也找不到 |

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

## 8. L7 开发期工具（`app/scripts/` 29 = 顶层 16 + `corpus/` 13，4 个域；另有 `tests/` 231 套）

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

### L7-3 数据库与语料（1 + 语料 7）

| 功能点 | 落点 | 说明 |
|---|---|---|
| 库升级 | `app/scripts/db_upgrade.py` | 迁移执行器 |

`app/scripts/corpus/` 是**外部 prompt 语料的四步流水线**（采集 → 归一 → 检索 → 统计，各步可单独重跑），外加两条支线：**扫描件 OCR**（`ocr_*` 四件＋`text_clean`，只服务 `script_library` 的第二条命）与 **场次索引**（`scene_index`，把库里的剧本切成场次写进独立 sidecar，只服务 A 侧 `kb_scenes`）：

| 功能点 | 落点 | 说明 |
|---|---|---|
| 知识库检索（CLI） | `app/scripts/corpus/library.py` | 复用 `services/script_library`，与 Agent 工具同一打分口径；`--q-file` 按 `utf-8-sig` 读防 PS 写 BOM |
| 场次索引（CLI） | `app/scripts/corpus/scene_index.py` | A 侧建库：一份要读全本正文（几百页 OCR 文本），5403 份是小时级 ⇒ **做成批处理而不是 HTTP 端点**（藏在端点后面既超时又没法续跑），可 Ctrl-C、`--limit` 试跑、`--force` 重算，与 `ocr_index.py` 的断点纪律一致。**不重复实现解析**：切场／角色提取／置信度在 `services/scene_parser`、落库与读回在 `services/scene_index`，HTTP 端点与 Agent 工具读的是同一份派生库 ⇒ 三者不会给出三种答案。参数 `--stats`／`--path=P`／`--path-file=F`（一行一个，绕开 PowerShell 传中文路径乱码）／`--all`／`--limit=N`／`--force`／`--json` |
| 扫描件 OCR 入库（CLI） | `app/scripts/corpus/ocr_index.py` | 引擎两档（`OCR_ENGINE=auto` 优先新档 `rapidocr` 3.x／PP-OCRv6，回退 `rapidocr-onnxruntime` 1.4.x／PP-OCRv4），新档 0.32 s/页 vs 老档 1.75 s/页（同页同机实测，识别字数 417 vs 410）；`--stats`／`--plan`／`--run`；**`--only=partial` 只补「跑了没跑完」的那批**（它们的块已进 fts5、默认被检索层挡着，混在 3135 份待办里按体积排序根本轮不到；拼错的值当场退 2，不许静默跑全量）；只做供应商库**登记了但没正文**的四种形态——PDF 扫描件（pymupdf 按 dpi 渲染）、散图（多帧 TIFF 按帧算页）、**纯文本 `txt`/`md`**（`_ocr_text`：`utf-8-sig→gb18030→big5→cp1252` 按可疑度选址解码、按 1500 字切伪页、`plain=True` 跳掉「跨页页眉剔除/整页重排」并把判废口径收成「只有解码失败才判废」，因为这两步与「符号段判废」都是为 OCR 碎行/表格准备的，用在作者排好的对白与目录上就是删真内容；主库那 80 份纯文本件本来就有正文块，这条路是「万一某份的块丢了」的出口，不是补一个不存在的坑）、**整页图贴进 Word 的 docx**（docx 就是 zip，按 `word/document.xml` 里 `a:blip` 的顺序取 `word/media/*` 再 OCR，不需要 python-docx；实测《撕夜》43 页图/开端 49 页图，原先当「坏文件」跳过 ⇒ 永远补不上）；默认剔除 `90-隔离与待审`、期刊排最后，`signature=size:mtime` 幂等可续跑，`--pages=N` 试跑标 `partial` 不当完成；★速度三处口径（都量过，见函数文档串）：**先读 PDF 自己的文字层**（约 1 ms/页，有字就不 OCR）、`rec_batch_num=16`（默认 6 时 7.68 s/页 → 2.02 s/页，识别字数 2816→2811 不变）、`intra_op_num_threads=预算÷进程数`（预算改由 `cpu_budget()` 给：先减 `OCR_CPU_RESERVE`（默认留 2）得硬件余量，再按本机**外部占用**收缩，见行末「资源调度」；CPU 档单进程封顶 16——32 线程反而 3.29 s/页，GPU 档只给 2 条因为 CPU 只剩解码/后处理；`OCR_THREAD_CAP` 只能收紧，表头逐次打印「`6 进程 ×（2 识别 + 3 渲染）＝ 30（预算 30 / 硬件 32 逻辑线程）`」便于当场核对不超订）；★**渲染与识别流水**（2026-11-04 第二处修正，起因是创作者指出「只用 GPU、CPU 没充分使用」）：原先一页一页「渲染 → 识别 → 渲染」，实测 4 进程 × 2 线程只占 32 条逻辑线程的 **~20%**，而 GPU 利用率在 25–82% 之间抖（GPU 每页都在等 PyMuPDF 渲染）。现改法两条：① **渲染线程也算进线程配额**——`worker_threads()` 把每进程配额切成 `(ORT 线程, 渲染线程)`，`workers ×（ort+render）≤ cpu_room()`，渲染不再算「白送」；GPU 档 ORT 只留 2 条、其余全给渲染（3–5 条），CPU 档 ORT 拿 `share−1`；② `_ocr_one` 用 `ThreadPoolExecutor` 按 `2×渲染线程` 的窗口预取后续页（连「先读 PDF 文字层」也搬到渲染线程），主线程连续喂 GPU。渲染任务**每页自己 open/close PDF**：PyMuPDF 的 `Document` 不保证跨线程并发安全，线程本地缓存在长批次里又会攒下几百个句柄（主线程无法安全回收别的线程的对象），open 是惰性 mmap、相对一页渲染是 1% 量级开销。进程数由显存封顶 `GPU_MAX_WORKERS=4`（每个 CUDA 会话驻留 ≈4.2 GiB；第 5 个会话把 `memory.used` 顶到 22489 MiB、每页反从 0.44 s 劣化到 0.88 s ⇒ 会话之间抢显存比抢算力更致命），**加进程提吞吐有物理上限**；`OCR_RENDER_THREADS` 可手钉做对照。★**页级切分**（2026-11-04 第三处修正）：进程池按**文件**分发，真跑到最后只剩一两份大书时（实测「只剩一份 920 页的《纪录电影文献》」）4 个进程里 3 个空转 —— GPU 掉到 18%、CPU 只占 ~22%，等于把并发白扔。现由 `plan_segments()` 在「份数 < 进程数」时把大文件切成 `(lo,hi)` 页区间并行（门槛 `MIN_SPLIT_PAGES=120`／单段 `MIN_SEG_PAGES=40`，只给「当前单段页数最多」的文件加段，切到补齐进程数为止；`--pages` 试跑不切；docx/图片数不出页不切，页数一律现场 `fitz.open` 数而**不信主库 `pages`**——报小了会静默丢页）：各段只做「渲染＋识别」并回传原始页文本，`_merge_segments()` 在主进程按页号排序后**统一清洗一次**（跨页版头剔除、碎行合并必须看全文档，这是切分不伤质量的前提），`seconds` 取最大段而非相加（段并行，相加＝把同一段时间数 N 遍）。实测 920 页切 1–230／231–460／461–690／691–920 四段：GPU 100%、显存 12.2 GiB、**180.0 s 跑完 = 5.11 页/秒（0.20 s/页）**，同一份书单进程不切段实测 0.44 s/页（60 页 26.3 s，推算 920 页要 403 s）⇒ **2.25×**。⚠️ 加速比不是 4× 的原因：4 段并行时每段掉到 0.78 s/页，GPU 已 100% 饱和，多出来的进程在排队而不是白算 ⇒ 想再快只能减少每页的 GPU 计算量（det 输入尺寸／rec batch），不是加线程。**CPU 占用低的正确解读**：这是扫描件 PDF，CPU 只做 150 dpi 渲染＋前后处理（≈0.15 s/页当量），GPU 满速出 5 页/秒时 CPU 需求本来就小；4 进程 28 条线程实测只用到 ~3 核 ≠ 没用上 CPU，而是这个任务的 CPU 占比就这么低；★GPU 档（2026-11-04）：`OCR_DEVICE=auto|cuda|cpu`，`pip install onnxruntime-gpu` 后仍会**静默回落 CPU** —— ORT 的 provider DLL 在**建会话那一刻**才 LoadLibrary，缺 CUDA 13/cuDNN 9 的 `cublasLt64_13.dll`／`cudnn64_9.dll` 只往 stderr 吐一行；本仓 torch cu130 轮子自带这两件，`prepare_cuda_dlls()` 把它们加进 `os.add_dll_directory` 并前置 `PATH`（每个子进程各自挂一遍，spawn 不继承 DLL 目录）。真正的门禁是**预检**：主进程真建一次引擎、跑一张带字 224×72 小图（rapidocr 3.x 的 det/rec 模型**懒加载**，不跑一次就没有 session 可问，`device_report` 只会如实回「无法确认」）、再遍历引擎对象取 `get_providers()`；显式要 cuda 却拿不到 `CUDAExecutionProvider` 就**退 3 中止**，绝不整批跑在 CPU 上还报成功；节流/`--stats` 的均速只能读**主进程墙钟账** `runSeconds÷runPages`（逐份秒数相加在并发下把同一段时间数了 N 遍，实测把 5 s/页报成 42.3 s/页、剩余 2058 小时）；★**口径重烧入口 `--only=done`**（2026-11-04 新开）：切块/清洗/门禁规则改了口径、要按新规则重烧存量时，只给 `--redo` **不够用** —— 它会把 3000+ 份待办一起排进 plan，`--limit` 默认 20，真正要重跑的老份可能一份都轮不上（实测就因此白烧了三份大书的前 1000 页）。`--only=done` ＝「只留 sidecar 里有记录的份」并**隐含 redo**（`ocr`／`partial`／`quarantine` 都算跑过），与 `--only=partial`（只补没跑完的）互补。★**参数形式守卫**：取值型参数（`limit`／`dpi`／`pages`／`path-substr`／`only`／`workers`／`verdict`／`vlm-*`）**只认 `--name=值`** —— 写成 `--name 值` 时 `_arg()` 取不到值会**静默**退回默认，对 `--path-substr` 而言「退回默认」就是「不筛、按优先级跑前 N 份」（实测一条本该只重跑 1 份的 `--redo --path-substr 某书名` 变成前 20 份全量重烧，白跑十几分钟）；现由 `_split_value_args()` 在 `main()` 最前把这种写法**当场退 2** 并给改写示例，`VALUE_ARGS` 是与各 `_arg` 调用点一一对应的清单（新增取值参数必须同步登记，`tests/ocr_index_test.py` ⑪ 六条锁住：空格写法／等号写法／值缺失／`--only` 同型／纯开关不受影响）。★**块页口径＝起始页**（2026-11-04 第五处修正）：`chunks.page` 原记**块收尾页**（`_chunks_of_page` 返回的 `tail_page`），而同一个 `_page_chunks` 里最后那条尾巴缓冲落的却是 `buffer_page`（**起始页**）—— 同一份书两套口径，症状是「按页跳」整体后移约一页、第一页永远没有块指着自己（实测 920 页那份 `page` 最小值是 4，1/2/3 页无块，全篇 25 页无块）。现统一记**块起始页**：检索命中后用户要做的是「翻原文」，从块首页往下读必然读到命中处，从块尾页打开往往要往回翻一页，而 `chunks` 表只有一列、只能选一边。守卫：`page` 在 `chunk_no` 顺序下**必然非降**（掉下去即首尾记反），`tests/ocr_index_test.py` ⑩ 四条用例锁住（跨页块记 1 不记 2、`_chunks_of_page` 两个页号必须分开给、短尾巴块记自己那页）。⚠️ **已入库的 19 份／2893 页是旧口径**，改口径只能 `--redo` 重跑（块文本里不含页号，无法用 SQL 回填）；`coverage.chunkPages`（`count(DISTINCT page)`）本来就是**近似**（跨页块只贡献一个页号），两套口径都低估真实覆盖页数，别拿它当「扫了多少页」。★**并发识别 `page_threads`**（2026-11-04 第六处修正，起因是创作者看「CPU 只用了 20% 多」）—— 真相是**瓶颈不在线程数**：200 DPI 一页渲染只 0.034 s（3234×4981 的大页），CPU 侧几乎不花时间，剩下的全是 GPU 上的 det/rec，而它们是一串**浅请求**、一次只有一个在飞 ⇒ 4 进程跑批时实测 GPU 只忙 53%、CPU 20%、内存 8 GiB/96 GiB。进程数又被显存封在 4（5 会话实测顶到 22489 MiB 且每页 0.5→0.88 s），于是唯一还能加吞吐的地方是**同一引擎实例被多线程重入**（ORT 的 `session.run` 线程安全、预处理只读）：`_ocr_one` 把识别也投进线程池，队列深度＝页线程×2，与渲染预取窗口并存。实测同进程、同一份前 40 页（当时后台 4 进程仍占着卡）：串行 20.2 s＝0.504 s/页、2 线程 11.6 s＝0.291、**4 线程 7.4 s＝0.184（2.74×）**，三次块数都是 34 ⇒ 结果等价。默认 `GPU_PAGE_THREADS=2`（4 进程×2＝8 个请求在飞已够填深；`OCR_PAGE_THREADS=4` 能再快，但会与渲染/后处理挤核，自己权衡），CPU 档默认 1（那边 ORT 已吃 16 条线程、实测 0.870 s/页，本就不是瓶颈端），非法值退回默认。★**并发必然乱序** ⇒ 新增 `_digest_pages()` 显式按页号收（顺序一乱，页眉剔除就会把正文首行当页眉剔掉），短页计入 `blank` 的口径与串行逐字一致；VLM 升级器不是线程安全的，`_page_text` 里的 `consider` 用 `_ESCALATE_LOCK` 串起来（只在可疑页触发，占比很小）。守卫：`tests/ocr_index_test.py` ⑫ 七条（默认 2／显式覆盖／给 0 夹到 1／非法值退回／CPU 档 1／乱序按页序收／短页与留痕不串页），全套 **153 条**。★**资源调度**（2026-11-04 第四处修正，起因是创作者要求「CPU 要吃满、GPU 按**实时空闲显存**分、别人占多了要提示或自动调」）—— 一整批开几个进程、每进程几条线程**不再由硬件常量决定**，改由跑批前一次 :func:`resource_snapshot` 决定、:func:`apply_snapshot` 落进 `OCR_CPU_BUDGET`／`OCR_DEVICE` 环境变量（`spawn` 子进程唯一能继承的口径；必须由主进程算一次而不是各进程自采，否则各进程预算不同、渲染线程数会漂）：① **CPU 预算按外部占用收缩** —— `psutil.cpu_percent(interval=0.25)` 采样（用 `interval=None` 首次恒返 0.0，会把「机器空着」写进预算，必须带区间），`cpuBudget=min(硬件余量, 逻辑线程×(1−busy))`；外部占用 ≥`CPU_BUSY_HINT=50%` 即告警「本批只按剩余算线程数，要独占先停掉它们或用 `--workers=1`」；② **GPU 会话数只按实时 free 显存算** —— `nvidia-smi --query-gpu=memory.total,memory.free`（`CREATE_NO_WINDOW`＋5 s timeout；拿不到就当不知道、按 `GPU_MAX_WORKERS` 不猜），`sessions=min(4,(free−1024)//4200)`；free 连「一个会话 4200 + 给人留 1024」都不够时 `gpu_session_cap` 返 0；③ **开不起会话就降级或告警，绝不硬上到 OOM** —— 自动档（`OCR_DEVICE` 未显式指定）改 `OCR_DEVICE=cpu` 并告警「空闲显存不够开一个 OCR 会话 ⇒ 自动回落 CPU 档（慢 3–5 倍），等占卡任务结束再跑 GPU 版」（改环境变量是唯一能让 `engine_device()`／`worker_threads()`／子进程**同时**按 CPU 档配线程的做法）；用户显式 pin `cuda` 则只告警不降级；`0<sessions<4` 另告警「显存被占 N MiB，只能开 X 会话，吞吐低于满配」；④ 进程数 `default_workers(snap)`：GPU 档 `min(sessions, 4, 预算//7)`、开不起会话或 CPU 档则 `min(8, 预算//4)`；⑤ **跑批前那一行抬头就是决策依据**（实测）：「`资源：CPU 预算 30/32 条线程（全机占用 4% ⇒ 可用 ≈31/32，硬件余量上限 30）｜GPU 空闲显存 21138/23028 MiB ⇒ 4 会话`」；⑥ **批内 5 s 采样** `_ResourceSampler`（daemon 线程，只读、不参与决策）：`report()` 按份触发，一份 900 页跑三分钟才报一次、批内峰谷全被抹平 ⇒ 收尾另打「`资源实测：CPU 平均 11%（峰值 11%）｜GPU 平均 34%（峰值 34%）｜显存最低空闲 18.4 GiB`」＋ :func:`verdict` 一句结论（GPU≥85% ⇒ 瓶颈在算力，扫描件 OCR 的 CPU 占比本来就低、数字上不去不是线程没配上；CPU≥85% ⇒ 加进程只会互抢；两边<40% ⇒ 卡在磁盘 IO/单线程段，再加线程也不会更快；CPU<60% ⇒ 可加 `--workers`，注意别超过显存能开的会话数），免得白调参；采样期显存跌破 `2×GPU_HEADROOM_MIB` 只告警一次（会话不能中途缩，只能让人知道谁在抢卡）。⚠️ 采样线程的 Event **绝不能叫 `_stop`** —— 那是 `threading.Thread` 的私有方法，同名属性覆盖后 `join()` 内 `self._stop()` 调到 Event ⇒ `TypeError`，且**只在真跑完 join 时**炸（纯逻辑单测发现不了，已配真 start/join 回归用例）。实测：`tests/ocr_index_test.py` **134 条全绿**；20 页真跑抬头如上、`0.4 s/页`、`引擎实报：CUDAExecutionProvider,CPUExecutionProvider`。依赖见 `requirements-ocr.txt`（**不进服务端镜像**）。★**长批抗锁**（2026-11-04 真实事故）：整批跑到 2739/2988 时被一次**只读**探测查询（整库备份／大范围 `LIKE`）顶掉，`_save` 抛 `database is locked` 直接冲出 `main()`，9 小时的活儿当场结束（已入库的没丢，剩下 249 份要重跑）。根因是「长任务没给余地」：`sqlite3.connect` 默认只等 5 秒，而 `_save` 是全批唯一没有兜底的调用点。三道防线：① `_open_sidecar` 一律 `timeout=SIDECAR_BUSY_MS/1000`＋`PRAGMA busy_timeout=60000`（`timeout=` 只管第一条语句，整条连接要靠 pragma）；② `_save_atomic` 在**写锁**上退让重试（`LOCK_RETRIES=4`、第 n 等 n×5 s；**非锁**的 `OperationalError` 立刻抛，不把真 bug 埋进等待里）；③ `report` 再兜一层——单份存不进去只记一条失败并继续，该份在 sidecar 里没登记 ⇒ 下一轮自动重来，既不丢也不假绿。纪律：批次在跑时别对它做整库备份／全表扫描；要核对就「先取有正文的 id 集合、再在内存里做差集」，别写 `NOT EXISTS (SELECT 1 FROM chunks …)`（`chunks.document_id` 是 fts5 UNINDEXED 列，那是每份扫一遍全表）。`tests/ocr_index_test.py` 170→**174 条**（busy_timeout／锁错误重试 3 次后成功／重试用尽必抛／非锁错误只试 1 次）。★**整份保底与「隐形空书」**（同日终局对账抓到的真缺陷）：终局对账查出 1 份 `status=ocr` 却 0 块 —— `8 波琳_凯尔_美国影评家_.pdf` 只有 1 页、574 字中文清清楚楚，判废理由是「符号连续段占 20%（3 处）＋有效字符占比偏低 79%」（期刊装饰线被识别成一排制表符，**字本身认得对**），于是整页丢弃、0 块，而 `status` 仍是 `ocr`：`--stats` 报「已 OCR」、检索页页 0 命中、`--audit` 只多一行 reject，没有任何一处能把它指出来。修法＝`_salvage_all_rejected`：**判据是「有没有一页带着正文活下来」，不是「是不是每页都标了 reject」**（空白页判定是 `ok`+0 字，按后者会漏掉「封面空白＋正文判废」的件），触发后把**软性**判废（符号段密集／表格页／短行）降为存疑留正文，**硬伤**（乱码／`maxRun` 超线＝整块被认成符号）照旧丢。★软性判据刻意用「**实打实的有效字符够多**」（`validChars ≥ 120` 或 `validRatio ≥ 0.6`）而**不是**「有效字符占比够高」：占比是比例判据，而期刊「总目录」那种篇名＋点线的页占比天生只有 45–58%、去重比 7–9%（实测 `26 本刊一九八一年总目录.pdf` 有效字 1291 个），字一个没错——按占比会把整本判废，而那本书就再也搜不到了。同理也**不**用「去重比／不同字符数」拦保底：重复页也是书里真实的一页（页眉被刷几十遍、清版页），在「整本否则全丢」的处境下留成 suspect 比彻底消失划算；另在 `_status_of` 加一条「0 块且 `rejectedChars ≥ MIN_PAGE_CHARS` ⇒ 隔离」，让真丢光的件至少出现在隔离清单里而不是消失。隔离语义随之收窄：只管「读出来的东西不能用」（硬伤），「排版被认成符号但字是对的」一律留内容——与「假阳性丢内容比留一点噪声严重得多」同口径。★**`--reaudit`**（只重判「有存疑页」的件，隐含 `--redo`）：门禁判据改了（如「去重比」加中文门槛）之后，旧判据留下的存疑行会一直挂在 `--audit` 里当噪声，而这些页一个字都没丢（`suspect` 照常写块），不值得全书重烧 CPU；`_suspect_paths()` 取件集合、`_keep_suspects()` 收缩候选并**如实打收缩比例**（`--limit` 默认 20，收缩后可能一份不剩，不打出来就像"跑了 0 份"）。`tests/ocr_index_test.py` 174→**192 条**（⑭ 整份保底：软废留正文／空白页+软废也触发／**点线目录页（有效字符占比 47%、有效字 1291 个）必须保底放行**／硬伤不放行（乱码、`maxRun` 超线）／有效字太少（<120 且占比 <60%，插图页口径）不放行／0 块有字被丢判隔离／全空白页不判隔离／成片软废不整份隔离／`--reaudit` 只认 suspect 件／`--run --reaudit` 自动带 redo／`--limit=0` 不被夹成 1）。★`--limit` 专用 `_limit_arg`：`_count` 会把 0 夹成 1（对 `--dpi`／`--workers` 正确，对 `--limit` 错）—— 实测 `--run --reaudit --limit=0` 本意"368 份全做完"，实际只做 1 份而日志像正常收尾，差 367 份一声不响，故 `--limit` 改走 `_limit_arg`（0＝全部，与 `_run` 里 `plan[:limit] if limit else plan` 的本意对齐）。★**第三库的「抽不出字」件在这里接一手**（`_extra_quarantined_docx()`）：主库是供应商清单、**不包含** `05-…/短剧待精筛/` 这批盘面，而 `lib_index.py` 收到图画型 docx 只能记成 `quarantine` 一块不写 —— 两条通道各自都"没做错"，合起来那本书就消失了（实测 1 份：`…/188-短剧剧本+微短剧项目写作基础/短剧编剧和剧本资源/基础知识.docx`，3.95 MB 里 `word/document.xml` 只有 7.9 KB、其余全是 6 张 jpeg）。只接 `status='quarantine'` 且扩展名在 `SUPPORTED_SUFFIXES` 的件：正常收下的（`extra`）已有正文块，排进 OCR 队列是白烧算力；`.pdf`／散图本来就由本库自己扫主库，不从第三库进。`tests/ocr_index_test.py` 192→**194 条**（图画型 docx 进待办／第三库正常件不进队列）。★**渲染器 2026-11-07 换成 `pypdfium2`**（原先按需 `import fitz` ＝ PyMuPDF/AGPL-3.0，整条摘掉，见 `THIRD_PARTY_NOTICES.md` §三 第 4 条）：取字 `get_textpage().get_text_range()`、渲染 `page.render(scale=dpi/72, rev_byteorder=True)`（PDFium 原生字节序是 **BGR**，不翻就红蓝对调且不报错）、数页 `len(doc)`；真机复测一份真扫描件的目录页（1980 年代《世界电影》期刊）150 dpi 出 1187×815、识别 0.64 s，9 条目录条目与页内文字逐条对上。`tests/ocr_index_test.py` 194→**199 条**（页数按真文件数／出图 `(H,W,3)`／红页还是红的＝字节序／96 dpi 794×1123＝dpi 换算／源码零 `import fitz` 的 ast 守卫），造 PDF 样本也从 fitz 改手写最小 PDF（生成器与被测实现同库＝测它自己） |
| OCR 校对机制 | `app/scripts/corpus/text_clean.py` | 入库前的清洗＋质量门禁（2026-11-04）。原先唯一门槛是 `MIN_PAGE_CHARS=12`，引擎把整页认成一排符号或把一行刷几十遍时字数轻松过 12 ⇒ 脏文本进 fts5 且**事后看不出来**。三层：①**清洗**——控制字符/零宽、孤字与纯标点碎行并回上一行（实测 79% 有文字层的页都有这种碎行；两字以上的行一律不动，合并「雨水」「他说」会把两句并成一句）、跨页重复的页眉页脚（同一归一形在 ≥3 页且 ≥50% 页的页首/页尾出现才删，**且只删 ≤24 字、不带句读**的行——初版用 40 字上限，实测把 34 字的剧本收尾句当页脚删了）；②**门禁** `judge_page` → `ok|suspect|reject`，只盯结构上不可能正常的信号（替换符/私用区/控制字符、同字符连续 8 次、有效字符占比 < 60%、整页去重字符 < 4 且 ≥60 字）；**CJK 占比不参与判定**（英文原版剧本 CJK 恒 0，拿它判会误杀整批）；③**隔离** reject 占比 ≥50% 且判定页 ≥3 ⇒ `documents.status='quarantine'`，整份不写块但**留审计行**（含理由与原文预览），且算"已处理"不会被下次批处理重烧 CPU，重试须显式 `--redo`。`--audit [--verdict=reject|suspect] [--limit=N]` 出复核清单；`tests/ocr_clean_test.py` 77 条。★同日真跑抓出三处只有接真实语料才露头的错，都已修并配反证：ⓐ **三步顺序不能换**——`clean_document` 必须「规整+碎行合并 → 跨页剔版头 → **最后**才整页重排」。原先重排在前，页尾水印被当正文续接拼到上一行（碎行合并把 `)` 粘在句号后、行尾不再是句号，重排继续吞下整条水印）⇒ 水印行从 45 字变 46/76/136/189 字**每页不同**，且粘连行含句号被"正文保护"跳过 ⇒ 一页都数不到、`剔除=0` 而水印照旧进 fts5（实测《世界电影》1984-02 八页，改成新顺序后 `版头=14`、水印清零）；水印常印成「花纹+商号」两行，OCR 把花纹认成 `· ((&)`/`一((+/` ⇒ 剔版头时连带删同侧 `_is_decoration`（≤8 字、真字符 ≤1、含符号、**不含句末标点**，最后一条防误删 `一。` 这种正文）。ⓑ **`judged` 没透传** ⇒ `_status_of` 读 `int(None or 0) >= 3` 恒假 ⇒ **整份隔离在真跑路径永不触发**（单测直接喂 `outcome` 所以看不出来）；现已两条扫描路都透传，并加 AST 守卫扫回执字典字面量必须含 `judged`/`droppedHeads`。ⓒ **符号连续段「1 处即判废」是假阳性丢内容**：764 字的正常评论页因一处表格线整页不写块；改按占比分档（`JUNK_RUN_SHARE_REJECT=0.15`、`JUNK_RUN_MAX_REJECT=200`），零星几处只记 suspect 并写进 `--audit` 预览；顺带修 `_JUNK_RUN` 的捕获组（带括号量化时 `findall` 只回单字符，长度永远量不出来）。ⓓ **正文保护的判据要看"行的实质"而非"整行"**：水印框被 OCR 认成两个句号 ⇒ 整行成「。。实惠图书之家…」⇒「含句读就是正文」把十页里唯一这一页放行（其余九页都删了，症状是"删得挺干净、就漏一页"）；改为 `_body_of()` 先剥掉**行首**装饰符号再判 `_PROSE_PUNCT`（正文不会以句号开头；句中断句仍是正文证据）。短文档（<3 页）做不了跨页统计 ⇒ 水印洗不掉，`--run` 当场打「⚠ 页数不足，未做跨页版头剔除」、`--audit` 单列 `shortDocs`，不假装已清理。★**「去重比」只对中文页成立**（2026-11-04 批次实证）：1500 字的英文页去重字符只有 30–60 个（26 个字母天然反复），去重比恒在 2–6% ⇒ 每本英文书**页页判存疑**，《Monsters, Inc.》改编本 170 页全进 `page_audit`，把真正要看的坏页埋掉。现要求本页 `cjkRatio ≥ DISTINCT_CJK_MIN=0.3` 这条判据才算数；另一头「整页只有 <4 个不同字符」**不设门槛**（拉丁文刷屏照样拦），`"他把信放在桌上…"*30` 反证中文重复页仍判存疑。既有 `page_audit` 里的英文噪声行是旧判据产物（终局对账实测量化：**8458 页／368 份**，最集中的 12 份全是英文剧本 `Once Upon A Time In America`／`Magnolia`／`Butch Cassidy…`；这些件总页数 48576 ≈ 51 分钟 @16 页/秒）——用 `--run --reaudit` 按新判据重判（页文本没单独存，重判只能重跑；`suspect` 页本来照常入库，重判只为把 `--audit` 的噪声清掉，不是补内容） |
| 版面分栏重排 | `app/scripts/corpus/ocr_layout.py` | 双栏页按「栏序 → 栏内 y」重排（纯几何、零模型、零依赖）。单栏页是**恒等变换**（`ocr_index` 默认开，`--no-layout` 可关）：不重排时双栏页的左右两栏会被逐行交叉，句子读起来半句左半句右——字全认得，就是读不通，且这种「脏」骗得过字数门槛 |
| VLM 升级通道 | `app/scripts/corpus/ocr_vlm.py` | 只把本地判定为 `suspect`／`reject` 的页交给 VLM 重跑（`--vlm` 开，默认关）。local＝进程内 transformers，http＝环回服务；判据是「升级后仍要过同一套门禁」——VLM 也可能认错，不能因为换了模型就免检 |
| VLM 权重取用 | `app/scripts/corpus/ocr_vlm_fetch.py` | 一次性把 PaddleOCR-VL 权重拉到 `models/ocr/PaddleOCR-VL-1.6`（先跑 `--fetch` 才有 local 档） |
| VLM 进程内装载 | `app/scripts/corpus/ocr_vlm_local.py` | local 后端：transformers 直接吃本地权重，不起外部服务（远端正跑着的 ComfyUI 机器不受影响） |
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
| 结构守卫（10 类事故） | `tests/structure_test.py` | 登记表/`__all__`/零依赖/模块头/空壳/路径腐烂/**接线状态**/**地图覆盖**/**清单完整**/**收尾口径**/**生成物不入库** |
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

## 10. 已知边界（四类，别把它们的落点当「已完成」）

四类分别是：**10.1** 实现齐但生产代码零引用、**10.1b–10.1h** 各推理族的真实闭合程度、
**10.2** 仍缺前向的族、**10.3 / 10.4** 调用链断在哪一层（停用 / 有接口没界面）。

### 10.1 已实现但未接线（1）

`mappings` —— 实现齐、口径被测试钉死，但生产代码 0 处 import。
危险不在占地方，在**看起来像已经能用**。登记在 `tests/structure_test.py::UNWIRED`，接了线必须删登记
（㊅ 判据双向检查：白名单只增不减也会报，见「㊅ 欠账表里的模块已经接上线 ⇒ 必须报该删条目」）。

**总数只认 `UNWIRED`，别照抄本文**：2026-11 起 `latent_formats` 已经接上生产线
（`dit_build.py` 拿它取 `LatentFormatSpec`、`vae_wan.py` 走 `process_in/process_out`、
`pipeline.py` 读 `h3_video` 的下采样比），因此从 3 条减到 2 条；2026-11-08 `image_ops`
也接上了 —— 落在超清二采的「缩到计划倍率」那一步（`torch_backend._scale_video_latents`
走 `image_ops.common_upscale`，它是少见的**吃 5 维 `(B,C,T,H,W)` 并逐帧缩**的实现 ⇒ 时间维不动），
起因是一处真错：`ai-2x-resize` 的第二步「高质量缩放 2× → 1.5×」此前**一步都没执行**
⇒ 计划要 1.5× 实际出 2×、而 `result.refine.scale` 照写 1.5（尺寸与回执对不上，回执自己却"自洽"）。
守它的是 `tests/engine_refine_test.py` ⑦ 系列（真装载缩小版放大器真跑，含时间维不变与拒绝 4×）。

⚠️ 那批判据此前**全是方形潜变量**（2×2 → 3×3），照不出「宽高传反」——`common_upscale(x, width, height, ...)`
的参数是**显式长名**，传反了形状依然合法、任何一步都不报错，只有非方形才把它照成各向异性拉伸。
⑦¹³/⑦¹⁴ 因此补了非方形（2×4 → 应得 **H=3、W=6**，传反则是 4×3），主视频与关键帧各一条。

⚠️ 「实测值」这一面原先只靠人自觉，三处一起补（2026-11-08）：

- `result.refine` 现在**并排**给计划值与实测值——`scale`（计划倍率，计划层口径不变）+ `scaleActual`
  （由后端自述的尺寸算出的实测空间倍率）；两者差超过 1e-6 时另写 `scaleMismatch{planned, actual}`，
  以 `scaleActual` 为准。此前只有 `scale`，画面其实是几倍**没有任何一格能证明**。
- 后端跑完二采**必须自述实况**（`refineDetails` 非空），否则 refine 阶段直接失败——与此前
  「不自述 `locksAudio` 就不认超清」同一纪律。只写计划倍率就当超清成功，属于「回执自洽、与画面无关」。
- `refine_latents` **入口先清空 `refineDetails`**：它是实例属性，上一次的值会活到这一次；本次若在设值
  之前失败，回执会写**上一次**的倍率与尺寸而这一跑一张画面都没有（实测证据：失败后残留
  `spatialScale=1.5`、`videoShapeScaled=[24,2,3,3]`）。
- 实测倍率对**每一次**二采都写（`ai-2x` 那条没有第二步缩放，此前整天缺席 ⇒ 管线只能拿计划值顶），
  并另存 `spatialScaleW`（非方形下宽高可能因整数取整不等，不等时进 `notes`）。

守它的是 `engine_refine_test` ⑨/⑨′/⑨″ 与 `engine_pipeline_test` ⑤²³/⑤¹⁴′；三条都用
「短路实现 ⇒ 判据变红」反验过（⑨ 短路后 `spatialScale` 为 None、⑨″ 短路后残留上一次的 1.5×）。

⚠️ 同族两处「实况写进去了、却读不出来」（2026-11-08 续）：

- **`notes` 被覆盖**：`_second_pass()` 返回的 `denoise_report` 自己也带 `notes`，`details.update(...)`
  把上面攒的缩放假话（「放大器 2× → 计划 1.5×」）整段冲掉——而**只要二采真跑了就必然冲掉**
  （成功/失败两条路径都带 `notes`）。症状是回执只剩「二采：尾部 N 步」，读的人不知道画面被缩过；
  倍率/尺寸两格还在，所以不报错。改为 `pop("notes")` 后 `extend` 并进去（⑦¹⁵）。
- **二采跳过只在 `details` 里**：张量层对「二采失败」的口径是**回退到上采样结果**（不炸），于是
  `applied=true` 照样成立、`secondPass:false` 静静躺在 `details`；而计划层 `upscale.secondPass`
  写的是 `true`——拿「applied + 计划」来读会以为二采跑了。现在顶层单列 `secondPass`
  （实测真值）＋ `secondPassSkipped{planned, why, note}`（不判失败，回退是上游口径）；
  反向的「计划没要、后端真跑了」写 `secondPassUnplanned`（⑤²⁴/⑤²⁵/⑤²⁶）。

**同一轮的另外两处「计划层有真源、执行入口不照它核」**（2026-11-08）：

- `_upscale_from`（`app/routers/engine.py`，HTTP dict → 计划对象的**重建口**）漏搬派生量
  `refine_tail_steps` ⇒ 重建出来的计划 `refineTailSteps` 恒为 0，于是回执 `plan.upscale`
  写着「尾部 0 步」而张量层真跑了 10 步（它自己拿公式重算）—— **同一条回执里
  `secondPass: true` 与 `refineTailSteps: 0` 并排**。现在三档：给了但与公式不符 ⇒ **400**
  （两处口径必有一处错）；没给 ⇒ 用计划层**同一个** `upscale.refine_tail_steps()` 现算；
  都不给 ⇒ 0。顺带把二采坏值（`refineSteps` 有、`refineDenoise` 无）**拦在 plan 阶段**
  —— 此前要等**采样 + 放大器全跑完**进到二采才 `ValueError`。判据 `engine_pipeline_test`
  ⑤¹⁸/⑤¹⁹/⑤²⁰。
- `ai-2x-resize` 的**倍率区间**没人核 ⇒ `scale=1.0` 是「先放大 2× 再缩回原尺寸」
  （尺寸与普通模式**一模一样**、耗时 ≈6×）、`scale≥2` 是「2× 之后靠 bicubic 插值冒充分块放大」
  （不会长出新细节）。两者都交付一张**自洽得没有破绽的假超清**（`applied=true`、`scale` 照写、
  `costFactor` 照算）。计划层只在 `1 < scale < 2` 才产这个模式 ⇒ 现在 `_upscale_from` 与
  `torch_backend._plan_target_size` **同一判据双保险**（前者早一步 400、后者兜住非 HTTP 调用方，
  边界必须一致：别一处 `>2` 一处 `≥2`）。判据 `engine_pipeline_test` ⑤²¹/⑤²² 与
  `engine_refine_test` ⑦¹⁰–⑦¹²（含 ⑦¹²「1.5× 不许误拦」）。

⚠️ 同一模式再犯一次「一条口播抄到所有分支」（2026-11-08 续）：`_finish_plan` 原先对**所有**
`usesUpscaler` 的模式一律印「耗时按逆向口径 ≈6×」，而 ≈6× 只是 `ai-2x`（放大器**原样 2×**）
那一条的逆向读数（它本来就写在 `upscale.plan_upscale` 给该模式的 `notes` 里）。
`ai-2x-resize`（先 2× 再缩到 1.5×）与 `tiled-ai-2x`（分块）**都没有**这个读数，
1.5× 的用户于是被告知一个没有出处、却读起来像实测的倍率。现在只**逐字引用**计划层那句
（`notes` 里含「耗时」的那条），找不到就明说「本模式没有逆向耗时读数，不猜」——
单一真源留在 `upscale.plan_upscale`，`_finish_plan` 不再自己写数。
判据 `engine_pipeline_test` ⑤²⁷（2× 那句原样引用）/ ⑤²⁸（1.5× 不许出现 6×，且要说明没有读数）；
反验：改回硬写 ≈6× ⇒ 两条同时变红，失败回执里 1.5× 那条真实印出「耗时按逆向口径 ≈6×」。

⚠️ 紧接着同一个病根的第三种形状——**一个数只覆盖一半、却被并排摆成整体**（2026-11-08 续）：
`costFactor` 从来只算**采样**（步数 × 采样器 × 引导），**不含**超清放大器与二采（那两段没有可精确算的
系数，≈6× 是逆向读数）。可它和 `upscale` 并排躺在同一份 `plan` 里，开 2× 超清时读者会把
「相对耗时约 ×1.00」当成整跑代价——而同一份 `warnings` 的下一条正写着「超清 ≈6×」。
现在：`plan` 增两格 `costCovers:["sample"]` / `costExcludes:["upscale"]`（不开超清时为空数组，
不编一个「总耗时」），且引导那条 `×N` 在开超清时补「这个系数只算采样，不含下面那段超清」。
判据 ⑤²⁹/⑤²⁹′/⑤³⁰；反验（把 `costExcludes` 写死为空、去掉引导那句补语）⇒ 两条同时变红，
失败回执里可见「相对耗时约 ×2.00」与「超清 ≈6×」直接并排。

⚠️ 同一病根落在**产物尺寸**上（2026-11-08 续，SDXL 装配线）：`sdxl_backend.write` 的
`width`/`height` 原先直接抄 `plan.width/height`，而 `media.write_image` 手里握着**帧张量实测**
的宽高却没人用 ⇒ 回执报的是「计划值冒充产物尺寸」。缩小版 VAE（2 倍、潜 1×1）那条用例实测
产物是 **2×2**、`plan` 是 8×8，而旧断言明文写着「报告的尺寸与 plan 一致 ✓」——**测试本身在给
缺陷背书**。现在改报 `written` 的实测值，与计划不同时另给
`plannedWidth/plannedHeight` ＋ `sizeMismatch{planned, actual, why}`（写明以 `width`/`height` 为准）。
同一份回执的另一个后端（`torch_backend.write` 的图分支）本来就报 `report['width']`（实测），
两条装配线口径至此一致。判据 `engine_sdxl_backend_test` ④⁗（改为断言实测 2×2）/ ④⁵′（点名
两侧数）；反验：改回抄 `plan.width` ⇒ 两条同时变红。

⚠️ 同一病根落在**耗时**上，而且两个数就躺在**同一份**回执里（2026-11-08 续）：`task.elapsedMs`
是真墙钟（含装载），`result.totalMs` 只是 `stageMs` 那几个键之和——**不含装载**（装载在
`runtime._execute` 里、`run_sync` 之外）、不含阶段间隙、不含 `gc`/`empty_cache` 收尾。实测 H3
那一跑墙钟 94.7 s（装载 20 ＋ encode 22.2 ＋ sample 39.6 ＋ decode 11.5）而 `totalMs` 只有 73.3 s，
差额 21 s 没有任何一格解释——而少掉的那段恰恰是「要不要常驻」最该看的数。另外事件里的
`elapsedMs` 原先也等于 `totalMs`（同名不同义）。
现在：`PipelineResult` 由 `run_sync` 薄壳记真墙钟（**返回后**统一赋值，不在十几处早退前各补一行——
漏的那条必然是失败路径）；`to_dict()` 增 `totalCovers`（逐字 `list(stageMs)`，动态取键而非手抄）
／`wallMs`／`unaccountedMs`／差额 >0 时附 `timingNote`；事件 `elapsedMs` 改真墙钟（`elapsed_ms()`，
起点未知则退回阶段之和，不编数）；`EngineTask.loadMs` 由 `_execute` 的 `finally` 计（成功/失败都记），
`to_summary()` 增 `timing{elapsedMs, loadMs, pipelineMs, pipelineWallMs, unaccountedMs, note}`。
删掉 `run_sync` 的分阶段早退假设：原函数体改名为 `_run_stages`，公开名只做计时与封口。
判据 `engine_pipeline_test` ⑤³¹（`totalCovers` 等于键且非空含 `plan`）、`engine_runtime_test`
⓪¹/⓪²（5 s ＝ 装载 2 ＋ 管线墙钟 3，未入账 0；干跑 `loadMs=None` 不编 0）；反验：删 `totalCovers`
⇒ ⑤³¹ 红，`unaccounted` 漏掉管线墙钟 ⇒ ⓪¹ 红（实测印出 3000 ≠ 0）。

⚠️ 同一病根的最后一处落在**步数**上，而且是「读数缺失被当成读数说没有」：`result.sampleSteps`
原先只有**采样器正常返回**那一处赋值，取消 / 中途 OOM 一律停在 **0** —— 而同一份回执里
`stageMs.sample` 是真实秒数、`error.step` 写着「停在第 3 步」（`on_step` 在**每步算完之后**才回调，
所以步号＝**已跑完**的步数）。三个数互相矛盾且无人解释；消费方照 `sampleSteps` 走：`chain.py`
逐段累加（被取消那段记 0）、`bridge.log_task_progress` 把 0 写进日志 ⇒「跑了 3 步被取消」与
「一步没开始」在留下的记录里完全同形。
现在：新增 `pipeline._absorb_steps(result, stage, step)`（**只在 `stage == "sample"` 时认这个步号**，
取 `max` 不赋值——双流那条路已先写过 `sampled["steps"]`，真跑完再抛不该把已入账的步数改小），
`_fail` 的两条路径（取消 / 失败）都调它；`to_dict()` 在「采样中途停住且步数不满」时附 `stepsNote`
（「真跑了 N / M 步就中断了」），并**防御性地**取 `max(sampleSteps, error.step)` 作句子里的 N——
反验时真见过「真跑了 0 步…（取消在第 3 步）」这种自相矛盾的句子，比不说更糟。
`chain.py` 的同形漏洞在上层：`_stopped` / `_segment_failed` 都**新建** `PipelineResult` 早退，
`stageMs` 有人逐项累加着交过去而 `sampleSteps` 没人传 ⇒ 默认 0（跑完 3 段×12 步真花掉几分钟，
回执写 0）；三处调用口（循环顶上 / 段内取消 / 段失败）现在都带 `sample_steps=steps,
guidance_steps=guided`。
判据：`engine_pipeline_test` ㉙¹（取消 ⇒ `sampleSteps == 3` 且与 `error.step` 同一个数、`stepsNote`
写 3/20）/ ㉙²（一开始就取消 ⇒ 仍 0 且**没有** `stepsNote`，不伪造）/ ㉜¹（第 5 步失败 ⇒ 5，写 5/10）；
`engine_chain_test` ②¹/②²/①¹（三个早退口分别 12 / 12 / **24** —— 24 是因为**失败那段自己也跑了
12 步**，第一版按 12 写被这条判据当场纠正）。反验：`_absorb_steps` 首行 `return` ⇒ ㉙¹/㉜¹ 精确红
（且 `stepsNote` 变成「0 步…（取消在第 3 步）」）；chain 三个调用口改 `sample_steps=0` ⇒ ②¹/②²/①¹
精确红。

⚠️ 最重的一处不在耗时、而在**内容**上，且旧判据在给缺陷背书（2026-11-08 续）：`media.write_video`
混音用 `apad`＋`-t <画面秒数>`，`-t` 是**单向**的——音轨比画面**短**是补静音（无损），比画面**长**
就是**截掉**（内容真丢）。而 `probe()` 只能读**混完之后**那条音轨（已被收尾 ⇒ 天然等于画面时长），
于是「音轨被剪掉 0.5 秒」与「音轨本来 0.5 秒」在回执里**完全同形**（`audioMuxed: True` ＋
`audioDurationSeconds: 0.5`）。`engine_artifact_test` 的 ⑩/⑪ 只断言「成片 0.5 秒 ✓」——1.0 秒
音轨丢了 0.5 秒这件事，**测试正是那个担保人**。另有一行
`report["audioDurationSeconds"] = report.get("audioDurationSeconds")`（自己赋给自己，读起来像在
核对、实际什么都没干）。
现在：`media._wav_seconds()` 用标准库 `wave` 读**源** wav（读不出返 `None` 不猜 0），`write_video`
按「源音轨 vs 画面（`count/fps`）」两侧对账，出 `videoSeconds` / `audioSourceSeconds` /
`audioTrimSeconds`（被截，内容丢失）/ `audioPadSeconds`（被补，无损）；真截过（阈值＝一帧，
小于它的是帧对齐噪声不喊狼）再给 `audioTruncated` ＋ `audioTruncatedNote`。`torch_backend.write`
把它转成 `result.audioTruncated{videoSeconds, sourceSeconds, trimmedSeconds, wavPath, why}`
并在 `note` 里加一句人话（指明 mp4 里没后半段、全长音频在 `audioPath`）。
判据：`engine_artifact_test` ⑤′（短 ⇒ 补 0.2、零截断）/ ⑩′（长 ⇒ 截 0.5、`audioTruncated` 真）/
⑩″（mp4 里那条确实只剩画面长度）/ ㉟′–㉟‴（整条装配线：回执单列一格、`note` 点名、产物卡两个数
能并排读）。两级分开反验过：`media` 里 `if False` ⇒ ⑩′/㉟′/㉟″ 三条同时红（回执里那句仍写着
「音轨补静音到画面长度、以画面收尾」，正是被截时最误导的一句）；只把 `torch_backend` 那段关掉
⇒ ⑩′ 绿而 ㉟′/㉟″ 红（证明「读数对了」与「读数交出去了」是两道独立闸门）。

⚠️ 同一病根的下一处落在**帧数**上（2026-11-08 收口）：`nb_frames` 只在封装**写了 index** 时才有
（分片 mp4 / 某些编码器 / 重封装中间态全都没有），`probe()` 如实给 `None`；而 `torch_backend.write`
的闸门写的是 `if expected_frames and encoded_frames is not None:` ⇒ **读不到就整条跳过**，`note`
却照旧写着「真 mp4 ✓（ffprobe 可复核 ✓）」⇒「帧数逐帧核过了」与「根本没核」在回执里**同形**
（`pipeline.seedvr2_target` 早就立过对的一半规矩：`nbFrames` 缺失 ⇒ 按 `时长 × fps` 估并进 `warnings`）。
现在三层表态，且**判定只有一处**：`media.frame_verdict(facts, fallback_fps, tolerance=1)` 返
`framesWritten` / `encodedFrames` / `framesVerified`（`"nb_frames"` 直读 / `"duration×fps"` **估的** /
`None` **没核过**）/ `framesMatch`（`True` / `False` / **`None` ＝没有可比读数**，谁把它读成「通过」
就是同一个病根）/ `framesTolerance`（直读 0、估 ±1）；`write_video` 内 `report.update(...)`，两条落盘路
都只读字段不重算。`torch_backend.write` 保留 `_discard_written()`（不符即删掉这条废片再抛，㉕「失败路径
一个文件都没写」）与 `note` 两句人话（估的写明「按时长×帧率**估**得 N 帧」、没核过点名「⚠️ 落盘帧数
**没核对过** ✗」）；`pipeline.seedvr2_target` 改为调 `media.frames_from_duration`（非法/缺失回 `None`，
**不猜 0**），`None` 才抛「帧数与时长都读不出来」。
`chain.run_chain` 是同族漏网的那一半：`stitch_frames` 只核**张量层**守恒（== 各段 keep 之和 == 总帧数），
**管不到编码器**（丢/补帧、index、重封装），成片此前**一次都没核过** ⇒ 现读同一份 `framesMatch` /
`framesVerified` / `encodedFrames` 交进回执，`framesMatch is False` 与单段路同口径（删成片 + `ChainError`，
段 mp4 都是有效的、留着），`None` 则在 `note` 里点名没核过。
判据：`engine_artifact_test` ㉔′（`nbFrames` 缺失 ⇒ `framesVerified == "duration×fps"` 且 `encodedFrames`
一致）/ ㉔″（`note` 含「估」与 `nbFrames`）/ ㉔‴（两格都缺 ⇒ `framesVerified is None` + `note` 含「没核对过」+
文件仍在）/ ㉔⁗（时长写着少 10 帧 ⇒ 抛 `MediaError`）/ ㉕′（判失败后渲染目录为空）；
`engine_chain_test` ⑤¹（直读三格齐）/ ⑤²（估的 + `note` 有「估」）/ ⑤³（`None` 两格 + 点名 + 成片仍落盘）/
⑤⁴（估出来差 10 帧 ⇒ `ChainError` 且废片不留在盘上）；`engine_io_test` ⑭¹（`write_video` 自带判定）/
⑭²（直读容差 0 ⇒ 差一帧就 False）/ ⑭³（估 ±1，差 12 帧仍 False）/ ⑭⁴（两格都缺 ⇒ `None`；时长 0 ⇒
`frames_from_duration` 回 `None` 不猜 0）。反验三处各自精确红：`torch_backend` 闸门退回旧口径
（`nbFrames is not None and ...`）⇒ ㉔⁗/㉕′ 红；`chain` 闸门 `if False` ⇒ ⑤⁴ 红（废片留在盘上）；
`media` 里 `return None` 改 `return 0` ⇒ ⑭⁴ 红（印出 `encodedFrames: 0` + `framesMatch: False`，
正是「猜 0 变成真读数」）。

紧邻的**同一病根第二例**落在**音轨长度**上（2026-11-08）：`media.write_video` 里
`if audio_seconds is not None:` 之后**没有 else** ⇒ 源音轨时长读不出就整段跳过，回执里既没有
`audioTruncated`（既不是「没截」也没说没量），而 `torch_backend` 那句人话照旧写着「音轨补静音到
画面长度 ✓」——**肯定句盖住了「这次没核」**。修法两面，缺一不可：
① **先修通路，再谈「不知道」**：标准库 `wave` 只认 PCM，`pcm_f32le` / `WAVE_FORMAT_EXTENSIBLE`
这些**完全合法**的 wav 它会 `wave.Error: unknown format`，而 ffmpeg 照混不误 ⇒ 新增
`media._audio_length(path) -> (秒数, 来源)`：`wave` 直读 → **`ffprobe` 兜底**（`_probe_audio_seconds`
只认**音频流**那条，不拿容器时长顶替）→ 两路都没有才 `(None, None)`；
② **两路都没有时必须明说**：`audioLengthVerified` 三档 `"wave"` / `"ffprobe"` / `None`，
`None` 时 `audioSourceSeconds`/`audioTrimSeconds`/`audioPadSeconds` **一格都不写**（不编数）×，
另给 `audioLengthNote` 一句人话；`torch_backend` 在 `audioLengthVerified is None` 时 **换掉**那句
「补静音到画面长度」（写成「按 `-t` 收尾到画面长度，但源音轨多长**读不出来** ⇒ 有没有被截
**不知道**」）＋ 单独一格 `video.audioTrimUnverified{videoSeconds,wavPath,why,note}`，并回填
`result.audioLengthVerified`。
判据 `engine_artifact_test` ㊱¹（前提：`pcm_f32le` 的 wav 是合法 wav 而 `wave` 读不了它、ffprobe
读得到）/ ㊱²（`wave` 读不了 ⇒ 走兜底，`audioLengthVerified == "ffprobe"`、源时长 ≈1.0 秒、截断照旧
对账出 0.5 秒）/ ㊱³（两路都缺 ⇒ `None`、三格全不出现、`audioLengthNote` 点名「不知道」）/
㊱⁴（整条装配线：`audioTrimUnverified` 出现 ＋ 人话含「不知道」＋ **不含**「补静音到画面长度」）。
反验三处各自精确红：`_audio_length` 去掉 ffprobe 兜底 ⇒ 仅 ㊱² 红（印出 `audioLengthVerified: None`、
三个秒数全 `None`，正是「只认 `wave` 就把能读到的说成读不出来」）；删掉 `audioLengthNote` 那段 ⇒
仅 ㊱³ 红；`torch_backend` 退回无条件那句 ⇒ 仅 ㊱⁴ 红，且失败详情里那句人与人话**同时**出现
「补静音到画面长度」与「不知道」（自相矛盾一眼可见）。★另记一处判据自身的假绿：`_hiding_probe`
最初按文件名 `tone_f32.wav` 比，而整条装配线那份源 wav 叫 `audio_seed12.wav`（`write_wav` 起名）
⇒ **一次都没涂掉**，判据看着绿其实空转；改成按 `.wav` **后缀**判才真的生效 —— 模拟型判据要盯
「它到底命中了没有」，别只看结论绿不绿。

**同族第三处：容器时长顶替流时长**（2026-11-09）。`media.probe()` 原写 `audio_duration =
(audio.get("duration") or duration) if audio else None`：**有**音频流、流的 `duration` 却缺时，
它**静默**回落到容器时长。容器时长是「最长那条流 ＋ 封装余量」，实测 1.0 秒的 mp3 原样封进
**mkv** ⇒ `format.duration=1.045`、封进 **webm** ⇒ `1.008`，两种封装都**不给**流的 `duration`
也**不给** `duration_ts`（只有 `tags.DURATION`，同样带余量）；回执里 `audioDurationSeconds`
于是与「音轨真就那个长度」完全同形。而 `_probe_audio_seconds` 正是拿它当**源音轨长度**去和画面
做截断对账 ⇒ 端到端会报一个**纯属封装余量**的假截断（反验实测：0.5 秒画面 ＋ mkv 源 ⇒
`audioTrimSeconds: 0.535`，其中 0.035 秒是假的）。
修法：`probe()` 新增 `audioDurationSource` 三档（`"stream"` 流自己给的 / `"container"` 容器顶替 /
`None` 没有音频流），回落本身**保留**（对混流后的 mp4 无害且必要，`-t` 已让两条流同长）；
`_probe_audio_seconds` **只认 `"stream"`** ⇒ 回落值一律按「没读到」处理，交给上一段的
`audioLengthVerified: None` ＋ `audioLengthNote` 明说不知道。**宁可说不知道，也不拿容器数当音轨数。**
判据：`engine_io_test` ⑭⁵（纯 wav ⇒ `stream` / 封进 mkv ⇒ `container` / 无音轨 mp4 ⇒ `None`）/
⑭⁶（`_probe_audio_seconds(mkv) is None` 而 `_probe_audio_seconds(wav)` 有值）；
`engine_artifact_test` ㊱⁵。

**同族第四处（同日补完整）：mkv/webm 不该停在「不知道」**。第三处修完只剩「mkv 只能说不知道」，
但那不是容量问题而是**没试**。实测同一批 1.0 秒内容：

| 源 | 容器声明 | 包级 span | **全解码** |
|---|---|---|---|
| mp3 → mkv | 1.045 ✗ | 1.045 ✗ | 1.025 ✓ |
| mp3 → webm | 1.008 ✗ | 1.008 ✗ | **1.000** ✓ |
| s16 wav | 1.0 | 1.0 | 1.000 ✓ |
| mp3 | 1.0449 ✗ | 1.0449 ✗ | **1.000** ✓ |

⇒ **包级 span 就是容器声明**（那不是内容长度，是编码 padding），"数包"这条路也是错的；只有**解码**
口径对得上内容。新增 `_decode_audio_seconds()`（第四级）：`ffmpeg -f null -` 只量不存，`-progress
pipe:1` 取 `out_time_us`（微秒，比解析人类可读的 `time=` 稳）；成本闸门 `_MAX_DECODE_SECONDS=600`
按容器时长预估，超了就**不解码**、如实落到「不知道」档（不静默降级）。`audioLengthVerified` 因此四档：
`"wave"` / `"ffprobe"` / `"decoded"` / `None`。
判据：`engine_io_test` ⑭⁷（mkv ⇒ `how == "decoded"`、≈1.0 秒、取解码值不取声明值）＋ ㊱⁵ 改成端到端
（`audioLengthVerified == "decoded"`、`audioSourceSeconds` 等于解码值、`audioTrimSeconds ≈0.5`）。
反验两条：`_audio_length` 去掉第四级 ⇒ ⑭⁷ 与 ㊱⁵ 红（`decoded: 1.0` 与 `length: None` 并排）；
让 `_decode_audio_seconds` 偷懒返回容器声明 ⇒ 两条红，且 ㊱⁵ 印出 `decoded: 1.035` ＝ `declared: 1.035`、
`audioTrimSeconds: 0.535`（正是那个带封装余量的假截断）。
★踩坑记一笔：两条判据最初用 `|declared − decoded|` 当「是否需要区分两者」的前提，那是**自洽陷阱** ——
解码那支偷懒返回 `declared` 时这个差算成 0、条件判直接放行，**判据与被测对象共享同一个错误源**，
反验二首跑仍然全绿。改成锚在**独立事实**（源内容是造出来的 1.0 秒）后才抓住。
**同族第五处（同日补）：连「不知道」本身也得分两句说**。第四处把 mkv 从「不知道」里救出来了，但落回
`None` 还剩两种**性质完全不同**的原因：一是**源太长、闸门不让解**（`_MAX_DECODE_SECONDS=600` 命中，
这次**没试**），二是三路全失败（文件坏了 / 根本没有音轨，**试过读不出来**）。此前两者都塌成
`(None, None)`，回执只能说「源音轨时长读不出来」——于是「你丢进来一条 2 小时的播客」在界面上长得像
「这个文件编码有问题」，用户照着去换封装、重编码，白查一场。
现在 `_audio_length()` 返三元组 `(seconds, how, miss)`，`miss ∈ {None, "too_long", "unreadable"}`
（`"too_long"` 由 `_decode_audio_seconds(..., missed=[...])` 在闸门命中时交回，不是在上层重算一遍预估），
`write_video` 照它分两句：`"too_long"` ⇒「**没有量到底**（不是读不出来）」＋ 点名上限秒数；
其余 ⇒「三路都没给出可用时长」。两种都不写 `audioSourceSeconds` / `audioTrimSeconds` /
`audioPadSeconds`（不编数）。
判据：`engine_io_test` ⑭⁸（`_audio_length(mkv, max_decode_seconds=0.001)` ⇒ `(None, None, "too_long")`，
同一文件用默认上限仍是 `"decoded"`——差别只来自闸门）＋ `engine_artifact_test` ㊱⁶（端到端：只涂掉流的
时长、保留容器时长让闸门有预估，再把 `_MAX_DECODE_SECONDS` 压到 0.001 ⇒ `audioLengthNote` 说「没有量到底」
且**不许**出现另一档的开头，三格假数一个不出现）。
反验两条：把两档人话合成一档 ⇒ ㊱⁶ 精确红（note 变回「源音轨时长**读不出来**」）；把
`_decode_audio_seconds` 的上限写成字面量 `600.0`（等价于「默认值在 `def` 时绑死」）⇒ ㊱⁶ 红在
`verified: 'decoded'`（patch 模块常量对它毫无作用、真解码照样成功）。
★又踩一个坑记一笔：判据最初写「`"读不出来" not in note`」，而**正确**人话里本来就有一句
「（不是读不出来 ✓）」⇒ 首跑把对的判红。禁的是**另一档的开头字面形**（`源音轨时长**读不出来**`），
不是那个词；同理 `max_seconds` 的默认值必须是 `None` 再在函数体内读模块常量，写成
`= _MAX_DECODE_SECONDS` 会让整个闸门在判据里静默失效（这正是反验二守的那条）。

**同族第五处·续：两档还得一路走到回执**。上面只在 `media.write_video` 当场分好了，往下走一层就散：
`torch_backend` 那句人话是**写死**的「但源音轨到底多长**读不出来**」，`audioTrimUnverified.why` 也写死
「**两路**都没读到」，而且那两句今天有**两处**已经过时（早就有第四级全解码，不是两路；源过长那次是
**没量**不是读不出来）——于是 media 刚分好的两档在装配线被**拍平**，前端看到的还是同一句。这就是这条线
的老病根换个层犯：**上游分了、下游吃掉 = 白分**。
落法：`media.write_video` 的 report 增 `audioLengthMiss ∈ {"too_long","unreadable"}`（机器码，别让上游去
匹配中文长句）；`torch_backend` 按它分三处 —— `audio_tail`、`audioTrimUnverified.reason`（新增，机器可读）、
`audioTrimUnverified.why`（两句分别写明「超过上限⇒没量」与「三路都没读到」），并把 `audioLengthMiss`
原样抬到回执顶层。判据 `engine_artifact_test` ㊱⁷ 同时钉两档：源过长那次回执级 `audioLengthMiss` 与
`reason` 都是 `"too_long"`、人话含「没有量到底」；㊱⁴ 那次两者都是 `"unreadable"`。
★反验踩坑记一笔：第一版 ㊱⁷ 只钉 `audioTrimUnverified` 那半（note 里含「没有量到底」），把 `audio_tail`
的 too_long 分支关掉后判据照样 PASS——因为 note 是**两段人话拼起来的**，另一段仍写着「没有量到底」。
所以判据对**两处人话各钉一句自己的字面形**（`audio_tail` 回退成写死的「读不出来」⇒ note 里出现
「源音轨到底多长**读不出来**」），并把原因抬成机器码而不是靠中文分。两条反验：关 `audio_tail` 分支 ⇒ 红
在那一句字面形；把 `reason` 写死回 `"unreadable"` ⇒ 红在 `tooLongReason: 'unreadable'`（62/63）。

**同族第六处（同日再补）：音轨「在」不等于音轨「有声音」**。前面每一格——`videoHasAudio`、`audioLengthVerified`、
`audioTruncated`、`audioLengthMiss`——都只回答「这条流**存在**吗」，没有一格回答「这段波形**有没有振幅**」。
于是**全零波形**与一段正常音频在回执里完全同形：wav 照写（一个字节都不差）、混进 mp4 也照成、回执照说
「真 wav ✓ 已混进 mp4 ✓」，用户点开是一段**无声片**，而回执里**一个提示都没有**。这正是「没量过振幅
＝宣布音频没问题」——而 `write_image` 那边早就有 `blank_std` 抓纯色图了，音频侧一直没有对应物。
落法：`write_wav` 增 `peak` / `rms` / `peakDbfs` / `silent`，**从写进文件的 int16 量**而不是从入参张量量
（要答的是「这条 wav 里有没有声音」，`1e-6` 的 float 在 16-bit 里就是 0）；`silent` 判据用**峰值**而非 rms
（一声咳嗽 rms 很低但不是静音），门限 `_PEAK_SILENT=1e-4`（≈ −80 dBFS，比 16-bit 量化底噪 3.05e-5 高
约 3 dB）；`peakDbfs` 在 `peak == 0` 时给 `None` —— **不写 `-inf`**，`-Infinity` 是非法 JSON，前端
`JSON.parse` 直接炸。报而不拦，与 `blank_std` 同口径。`torch_backend` 补一句人话，且必须把两件事**分开**：
说的不是「音频没混进去」（那会让人去查 ffmpeg 参数），而是「音轨在、内容是静的」（该查的是波形生成那一段）。
判据：`engine_io_test` ㉑′（全零 ⇒ `silent`＋`peakDbfs is None`；**0 帧**边界不崩；正常 0.8 正弦不许误判）
＋ `engine_artifact_test` ㊱⁸（端到端全零波形 ⇒ `audio.silent` 与那句人话都在；对照组直接用 ㊱⁷ 那次真 tone，
同一段代码路径，差别只在波形有没有振幅）。
反验两条：`silent` 恒 `False` ⇒ ㉑′ 与 ㊱⁸ **同时**红；只关掉 `torch_backend` 那句人话 ⇒ ㊱⁸ 红而 ㉑′ 仍绿
（media 层的读数还在，只是没人念出来——两层各被一条判据守着）。
★踩坑：㊱⁸ 首跑红在字面形上——人话原文是「**不是**「音频没混进去」」，判据写成「不是「音频没混进去」」
**恒假**。判中文人话时必须连加粗标记一起对（与第五处「只判『不含读不出来』会把对的判红」是同一个坑的两面）。

**同族第六处·续：说清楚了还得让人看见**。上面每一句都写进 `result.note`——而前端**不读** `note`：
`useEngineRun.ts` 一个 `note` 字段都不取，`CanvasNodeCard.vue` 里那行 `engineState.note` 显示的是**提交时**的
`autoNote`（装配说明），不是跑完的结论。于是「音轨几乎是静音 ✗」「mp4 里音轨被截掉 0.6 秒 ✗」「落盘帧数
没核对过 ✗」「源音轨没有量到底 ✗」这四句在界面上**一句都看不到**，只有直接读回执 JSON 的人才看得见。
这是这条线的老病根换了个方向：之前是「读到了却不说」，这次是「说了却没人看见」——两种都等于没说。
落法：`torch_backend.write` 里加 `_alert(sentence)`——把结论 append 进 `result["alerts"]` 并**返回原句**，
`note` 那行写成 `+ _alert("；⚠️ …")` ⇒ 两处是同一个字符串，不会哪天只改一处；纯音频（Music3）分支同办，
且那里更需要（没有画面可佐证，用户只能听，会先怀疑「我哪设错了」）。前端 `useEngineRun.ts` 的 `syncDetail`
把 `result.plan.warnings`（提交前的计划警告）与 `result.alerts`（跑完的结论）**合成一个 `warnings` 列表**——
复用现成渲染（⚠️ 前缀、只显示前两条、悬停看全量），不动 UI 结构。Web 与桌面两端同改（`useEngineRun.ts`
是 parity 硬约束的同名文件）。
★两条判据都钉「同源」：㊱⁸ 断言 `alerts` 里每句都**逐字出现在 `note` 里**（`any(str(s) in note_z …)`），
别处不许各写一遍；对照组（㊱⁷ 那次真 tone）可以有**别的** alerts（它本来就有「没有量到底」那条），只判
「静音那条不出现」——第一版把整个列表判空，被这条对照拦下来了。
㊱⁹ 单独守**纯音频**那一支（无帧 ⇒ 落 `.wav`）：`audio.silent` 与 `alerts` 都在、那句与 `note` 逐字同源、
`primaryPath` 以 `.wav` 结尾（这条分支没有 mp4，不许把 wav 当视频用）。反验：删掉 `result["alerts"] = [_sentence]`
⇒ ㊱⁹ 精确红在 `alerts: None`（64/65），而 5 维那支仍绿 —— 两条分支各被一条判据守着。

**同族第七处：图那一支漏了同一件事**。上一轮只给音轨补了 `alerts`，`write()` 的 **4 维图像分支**还留在原样：
`write_image` 早有 `blank_std` ⇒ 纯色/全黑/全白**不抛异常**，`result["degraded"]` 与 `note` 里那句「但画面
**几乎是纯色**」都写了，可前端只显示 `warnings`（`note` 与 `degraded` 它一个都不读）⇒ 用户拿到一张灰片、
界面上零提示。同一个病根在**同一个函数**里换个分支再犯一次——正是「一句话手抄多份，修它那一轮再犯」那条
教训的镜像版：**一个机制补到一半**。落法：图像分支同样 `result["alerts"] = ["；⚠️ 但画面**几乎是纯色** ✗
（见 `degradedReason` ✓）"]`，与 `note` 那句**逐字相同**（`_alert` 闭包绑的是 5 维那支的 `result`，这里不能
复用，但也不许各写一版）。判据 `engine_artifact_test` ㊱¹⁰：纯色 4 维 ⇒ `degraded is True` ＋ `alerts` 那句
逐字出现在 `note` 里；对照组用随机噪声图（同一段代码路径，差别只在像素有没有内容）。
★踩坑：对照组**不能判 `degraded is False`** —— 正常图那一支**根本不写** `degraded` 这个键（只在真标了才写
`True`），读到的是 `None`，判 `is False` 会**恒假**（首跑就红在 `controlDegraded: None`，65/66）。
这条也说明「三态键」与「布尔键」是两种约定：`degraded` 是**只在异常时才存在**的键，缺键即正常。

**同族第八处：帧率读回来了，却没人跟请求对账**。`probe()` 早把盘上的 `r_frame_rate` 读成 `report["fps"]`，
`write_video(fps=…)` 手里也有请求值——**两格从来没有比过一次**。于是「按 24 fps 封好」与「它按 12 fps
封了（画面慢一倍）」在回执里完全同形，而且**每一格都有值、都 ffprobe 可复核**，看不出缺了什么。
更麻烦的是它不止影响一格：`videoSeconds`、`-t`、音轨 `apad` 的截断与补长**全按请求 fps 算**，而
`durationSeconds` 是**盘上真实**的——帧率一漂，音画就不同步，可这两格都"有值"。

落法：`media.compare_fps(requested, on_disk)` 返回 `(verdict, note)`，三档 `"exact"` / `"drift"` /
`None`（**盘上没读到 ⇒ 不知道**，不许塌成 exact）；`write_video` 补 `fpsRequested` / `fpsVerified` /
`fpsDrift` / `fpsNote`，`torch_backend` 把这三格抬到回执顶层，`drift` 时 `_alert` 同一句（音画不同步）；
`fpsVerified is None`（**没读到**）也要 `_alert` ——「没核」不许安静放过，回执里 `fps` 是 `None` 而
`durationSeconds`/`videoSeconds` 都有值，读起来像一切正常。`chain.run_chain` 那条成片路同样交出
`fpsVerified`/`fpsRequested`/`fpsOnDisk`（此前只有 `framesVerified` ⇒ 帧率被改在成片路上一声不响）。
容差 `_FPS_TOLERANCE = 0.01`（`r_frame_rate` 可能是 `2997/100`）。
判据：`engine_io_test` ⑭⁹（端到端 exact ＋ 请求值与盘上读数分开）/ ⑭¹⁰（三档分得开，`drift` 人话必须提
「音画」）；`engine_artifact_test` ㊱¹¹（顶层三档 ＋ **一致时不许喊帧率**——老喊狼，用户第三次就学会无视它）；
`engine_chain_test` ⑤⁵/⑤⁶/⑤⁷（**成片路**三档：`exact` 时顶层**不出这个键** / `drift` 顶层那句话与
`outputs["alerts"]` **逐字一致** / 读不到 ⇒ `fpsVerified is None` 且顶层写「没读到」—— 单段路补了、成片路
不补，就是这一族的老形状）。
★反验：把 `compare_fps` 的 drift 分支塌成恒 `exact` ⇒ ⑭¹⁰ 精确红在
`{'same': 'exact', 'drift': 'exact', 'blind': None}`（69/70）；成片路三条各自反验过（见第九处的记录）。
★踩坑：`compare_fps` 里 `blind`（盘上读不到）只能给 `None`——若给 `"exact"`，那正是这一族的老毛病
（读数缺失被当成读数说没问题）换个位置再犯一次。

**同族第九处：`alerts` 从来没到过用户眼前（通道断在最后一段）**。这是本轮最重的一处，而且是**上一轮我自己
留下的假绿**：各后端把跑完才知道的结论写进 `backend.write()` 的**返回**里（`result["alerts"]`），而
`pipeline.run_sync` 把它原样收进 `PipelineResult.outputs`，`task.result = result.to_dict()` —— 于是 `alerts`
落在 `result.outputs.alerts`；可前端 `useEngineRun.ts::syncDetail` 读的是 `task.result.alerts`（**顶层**）。
⇒ 2026-11-09 起那一整批「音轨几乎是静音 / mp4 里音轨被截 / 落盘帧数没核对过」**一条都没显示过**，
后端回执字斟句酌地写着，人以为界面上也写着。**后端说了、通道断了 ＝ 白说，比不说更糟**（它让人以为已经提示了）。
落法：唯一真源仍留在 `outputs`（后端只管往里写），由 `PipelineResult.to_dict()` **统一抬一层**：
`outputs["alerts"]` 非空才写顶层 `"alerts"`（**缺键** ＝ 一条都没有，不用 `[]` 混淆「没说」）；
`outputs=None` 不许崩。`chain.run_chain` 那条成片路同样把要人查的结论写进 `outputs["alerts"]`（它此前连
`fpsVerified` 都没有——单段路补了、成片路不补，正是 2026-11-08 帧数那次的同一个形状）。
判据：`engine_pipeline_test` ㊱¹²（抬层逐字一致 ＋ 无 alerts 时顶层**没有这个键** ＋ `outputs=None` 不崩）、
`engine_artifact_test` ㊱¹⁵（用**真** result 走 `to_dict()`，判的就是用户拿到的那份对象）、
`engine_chain_test` ⑤⁵/⑤⁶/⑤⁷（成片路：清白的成片**顶层无** `alerts` 键；帧率被改 / 读不到时顶层那句必须
与 `outputs["alerts"]` 逐字一致）。
★反验（四处，逐处精确红）：① `to_dict()` 里那句 `payload["alerts"] = …` 关掉 ⇒ ㊱¹² 精确红在 `{'loud': None}`
（183/184）；② 成片路 `if outputs["fpsVerified"] == "drift"` 关掉 ⇒ 只有 ⑤⁶ 红在 `{'fpsVerified': 'drift',
'payloadAlerts': None}`（54/55）；③ `elif … is None` 关掉 ⇒ 只有 ⑤⁷ 红（同上形状）；④ 成片路
`outputs["alerts"] = alerts` 关掉 ⇒ ⑤⁶＋⑤⁷ 同时红（53/55，证明抬层是唯一通路）。
★教训：**"我把结论写进回执了" ≠ "用户看到了"** —— 断言 `result["alerts"]` 有值只能证明前半句，
必须是「`to_dict()` 之后仍读得到」才算到人眼前；而**成片路要单独判**，因为链层的输出根本不经过
`torch_backend.write`（一处改了另一处不改就是这么来的）。
★★同时收掉第二层隐患（同一格上的**写法**）：`alerts` 生产侧有**四处**写入点（纯色图 / 元数据丢失 / 静音 /
削顶），此前是两种写法混用 —— `result["alerts"] = [...]` **覆盖** 与 `list(result.get("alerts") or []) + [...]`
**追加**。今天恰好覆盖写法都排在追加之前（所以没错），可只要有人以后往上面再加一句，前面那条结论就**静默消失**
（不报错、行为判据也看不出来）。⇒ 四处统一成 `result.setdefault("alerts", []).append(...)`（单一写法），
并立**写法判据** `engine_artifact_test` ㊱¹⁶（`app/services/engine/*.py` 里除 `to_dict()` 那一处抬层口
`payload["alerts"] = [...]` 之外，不许出现任何 `X["alerts"] = [...]` / `= list(…)`）＋ ㊱¹⁶′（抬层口**只有一处**）。
★反验（续）：⑤ 把静音分支改回覆盖写法 ⇒ ㊱¹⁶ 精确点名 `['torch_backend.py:8625']`（71/72）；
⑥ 在 `to_dict()` 里多插一处抬层 ⇒ ㊱¹⁶′ 红在 `['pipeline.py:2056', 'pipeline.py:2058']`（71/72）。

**同族第十处：图像元数据只核了「打算写」**。`write_image` 交的 `metadataKeys` 是**送出去的键名**，
而 `_png_info` 会**跳过** `value is None` 的项、键名 strip 后为空也跳过 ⇒ 「打算写 3 个」被当成
「写成了 3 个」，甚至交出一个**根本不在 PNG 里**的键名。mp4 那侧 2026-11-07 就有 `metadataDropped` 对账，
图这侧一直缺。落法：写完 `Image.open(target).info` **读回来**逐键核 ⇒ `metadataKeys` 只列真读到的、
`metadataMissing` 列送了却没进去的、`metadataReadback` 三档（`True` 核过 / `None` **本次没送**）；
读不回来（文件都打不开）⇒ **一个都不算数**。`torch_backend` 4 维分支把 `metadataMissing` 写进
`note` ＋ `alerts`（图离开作业单后元数据就是它唯一的出处），并与 mp4 的 `metadataDropped` **同口径**
（那句此前只进 `note`，本轮一并补进 `alerts`）。
判据：`engine_io_test` ⑭¹¹、`engine_artifact_test` ㊱¹³。
★反验：把「读回来核」退化成「全收」⇒ ⑭¹¹ 精确红，证据 `bad.metadataKeys == ['', 'dropped', 'prompt']`
且 `metadataMissing == []`（70/71）—— 正是"没写进去被说成写进去了"。

**同族第十一处：削顶测了但没人说**。`write_wav` 早把 `clipped`（越界样本数）算出来交出去了，可它只躺在
`audio` 子字典里，`note`/`alerts` 一个字不提。削顶**不抛异常**、也**不会**被判成 `silent`（峰值恒 1.0，
甚至更"响"）⇒ 整段爆音与"增益刚好"在回执里同形，是本族最容易被当成"正常"的一格。落法：唯一句源
`TorchBackend._clip_alert(report)`（5 维那支与纯音频那支都调它），`clipped > 0` 才喊，人话**点名
爆音/失真**、给占比，并**先排除**解码/混流（锅在波形生成的增益/归一化）。
判据：`engine_artifact_test` ㊱¹⁴（三档：0 ⇒ `None`、>0 ⇒ 含「削顶/爆音/占比」、键缺失 ⇒ `None`）。
★反验：让 `_clip_alert` 恒返回 `None` ⇒ ㊱¹⁴ 精确红在 `{'some': None}`（69/70）。
★规矩：**"测量到了" 与 "有人知道了" 之间隔着 `alerts` 这一格** —— 一个数字躺在子字典里，等于没测。

至此「读数缺失被当成读数说没有」这条线在**帧数**（三层表态 ＋ 单点判定）、**源音轨长度**（`wave` →
ffprobe 流时长 → **全解码** → 明说不知道，四档来源；`None` 再分「没量到底」/「读不出来」两档）、
**音频时长的来源**（stream / container / None）三处都收口了，加上「波形有没有振幅」这一格，
共计数缺失类四格全部如实表态。

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
二次同参调用 `reused=True`。⚠️ **`force=true` 是唯一的例外**（2026-11-09 ✓）：同参也**真重装一次**（`_ensure_loaded_internal` 的同键短路由 `force_reload` 跳过 ⇒ 旧的真丢、新的真装 ✓），回执这一次带 `forced=true` ／ `reused=false` 且**不写进缓存报告**（否则下一次复用也带着它 ⇒ 把「重装过」读成「正在重装」✗）；不传 force 的默认路径**逐字不变**。四格传法同时认驼峰与下划线（`textWeights`/`text_weights`），
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

### 10.1c Qwen-Image **2512 / Edit 2511**：骨架、前向、装配线**与真权重实跑**都已闭合

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
- `part_need_gib=` **按文件精度校正过的**本族账（`QWEN_IMAGE_PART_NEED_GIB` 的 `dit` 那一格当 fallback，
  过一遍 `resident_need_gib()`：bf16 的 2512 ⇒ 38.05 / Edit 2511 的 int8 ⇒ 数据区 19.09；TE 16.1 / VAE 0.24 不动）。
  **账不许混**：拿 2.1 的 13.26 判 38.05 ⇒ 结论"装得下"、`residency` 留 `off`、装到 20 GiB 才 OOM。
  以本机 22.49 GiB 卡为例：2.1 判「整件上卡不降级」，2512 bf16 必须判 `split`（余量连 0.5 GiB 都没有时才
  整份转 CPU），Edit 2511 int8 则 `off`（19.09 < 19.71 可用）。这本账在**三个可见面**上都得是生效值：
  `capabilities().devicePolicy.needGiB.qwenImagePart`（静态，单开一格，不能并进 `part` —— 两族键名相同会
  互相覆盖）、`describe().devicePolicy.partNeedGiB`（声明链后就是本次生效值，类缺省另存
  `defaultPartNeedGiB`）、以及 `declare_chain` 回执的 `partNeedGiB`/`peakGiB`；`POST /load` 的回执另加
  `ditNeed`（这次按几 GiB 判、是 int8 常驻还是 bf16 1:1）。

清单与就绪档：`configs/models.json` 三条（`dit_qwen_image_2512` / `te_qwen25_vl_7b` / `vae_qwen_image`，
`tags` 含 `qwen2512`、`required: false`），`STAGE_FILTERS["qwen_image"]` 用 `requireAll` 定必需性，
`registry.STAGES` 同步注册（两处双向核）。于是 `readiness("qwen_image")` = 38.05 + 8.74 + 0.24 =
**47.03 GiB（盘上字节）**，三件全必需，与驻留账 54.4 GiB 是**两个数**（fp8/int8 装进来要还原成 bf16）。
Edit 2511 **故意不打 `qwen2512` tag**：它与 t2i 互斥，进 tag 会让 `requireAll` 把两代都算必需、就绪档假红。

**Edit 2511 = 能装也能跑（2026-11-01 接上参考图通路）**：那份权重靠 `__index_timestep_zero__` 哨兵把
参考图 token 与目标 token 拼在一条序列里、并给参考图行喂 σ=0 的调制行。参考图走两路：
视觉塔（**384² 面积**）⇒ `<|image_pad|>` 整段替换进文本序列；VAE（**1024² 面积**、8 的倍数）⇒
`ref_latents`，`denoise` **每一步**带进 DiT（`Sequence[Tensor]`，单图也要包一层列表）。
多张参考图同样成立：模板按图数二选一 —— 一张走 `TextEncodeQwenImageEdit` 口径（模板自带一个视觉块），
多张走 `…ImageEditPlus` 口径（正文前 `Picture i: <视觉块>` 逐图一个槽），`ref_latents` 一列、
**顺序即语义**（`Picture 1` ↔ `embeds[0]` ↔ `ref_latents[0]`）。两处「能力凭空消失」都当场报错：
Edit 权重没给 `referenceFrames`、Edit 权重缺 `ref_latents`（含空列表）。编辑口径下 `max_length`
会把图的 token 数**加到文本预算之上**（一张 384² 图约 196 token，三张就 588 > 512，直接用 512
截断掉的正好是尾部那句指令）。视觉塔只在 encode 段在场 ⇒ 峰值 17.69 GiB 仍判 `full`。

**2026-11-01 真权重实跑已闭合（两档都出了图）**。2512 t2i：`qwen_image_2512_bf16` ＋
`qwen_2.5_vl_7b_fp8_scaled` ＋ `qwen_image_vae` ＋ Qwen 分词器，1328×1328 / 4 步 / cfg 1.0（先用小步数
验链路）—— 装配 **20.8 s**（`residency=split`：卡上 25 层 15.9 GiB ＋ 内存 35 层 22.15 GiB，71 个跨界
hook）、`init` 37.3 s（换装：退 TE、DiT 上卡切层）、4 步 **290 s**（**72.5 s/步**，跨界搬激活的代价）、
`decode` ≈19 s、PNG **2 780 264 字节**。判图不吃"看起来像"：低频方差保留 **92.6%**（4×4 下采样后
方差 / 原方差）、相邻像素差 std **9.2**、**36 979** 种颜色、8×8 亮度格上亮下暗分层 ⇒ 是照片结构、
不是高频噪声也不是灰块（噪声图的下采样方差保留会掉到 10% 以下）。

**三个"真权重才暴露"的缺陷**，全部已修并配判据（这三条的共同点见下方小结）：

1. **前向入口缺精度对齐**（`dit_qwen_image.DiT.forward`）：潜变量有意保 f32（`randn` 默认、逐步累积更准），
   权重是 bf16/fp16 ⇒ `img_in` 当场 `mat1 and mat2 must have the same dtype, but got Float and BFloat16`。
   报错落在 `nn.Linear` 上、读起来像"权重坏了"，其实是精度没对齐（与 `dit_lumina.forward` 同一条口径：
   **在模型内部 cast 输入**，不是把采样侧的潜变量降精度——后者会丢掉累积精度）。现按
   `self.img_in.weight.dtype` 统一 cast `latents` / `context` / 每个 `ref_latents`。
2. **0 元素张量装不进**（`weights._tensor_from_bytes`）：Edit 2511 的 `__index_timestep_zero__` 在
   `int8_convrot` 那份里是 `shape=[0]`，`torch.frombuffer(b"")` 直接抛
   `both buffer length (0) and count (-1) must not be 0`——字节数本来就该是 0，它的意义只在"键存在"。
   现 numel 为 0 时返回同形状空张量（`engine_weights_streaming_test` ⑤/⑤′ 守：能装、且不算缺键）。
3. **`config` 收不到 `@property`**（`dit_build`）：`config = dict(vars(shape))` 只收 dataclass 字段，
   而 `ref_timestep_zero` 是 property ⇒ 装配线 `config.get("ref_timestep_zero")` 永远 `None` ⇒
   **Edit 被当 t2i 装**（encode 段不挂 vision/vae、`encode_text` 整段跳过参考图），一路无报错，
   直到 `denoise` 才报"Edit 权重缺 `ref_latents`"——**报错落点与病因差了三层**。修法是给
   `FamilyBuilder` 加 `derived` 栏（声明"这一格是从 shape 的属性取值的"），拼错当场 `BuildError`、
   不许把 `None` 写进 config（`engine_dit_build_test` ⑧ 三条守：登记、t2i 为 False / Edit 为 True、拼错即报）。

★小结：三条都是**"装得进、跑得动、没有任何报错，但功能整块消失或图不对"**的同一族 —— 真权重是唯一
能照出来的镜子。`engine_qwen_image_test.py`（66 项）与 `engine_qwen_image_edit_test.py`（50 项）里
⑨ 系列组装真权重的判据仍保留；上面三条新判据在 `engine_dit_build_test`（48 项）与
`engine_weights_streaming_test`（16 项）。

#### int8 常驻 + 按文件取需求（2026-11-01，回答「显存那么大为什么还是 CPU 在算」）

原来这一族**只有 bf16 一条路**：`dit_build` 的 `qwen_image` 行没有 `adapter`，1933 键一律物化成 bf16
⇒ 2512 占 38.05 GiB、Edit 2511 占 ≈35.5 GiB，两个都超 22.49 GiB 卡 ⇒ `auto` 只能判 `split`
（实测卡上 25 层 15.9 GiB ＋ 内存 35 层 22.15 GiB、71 个跨界 hook、**72.5 s/步**）。三处改动：

1. **`_Int8Adapter`（`dit_build._ADAPTERS["int8"]`）**：`qwen_image` 行加 `adapter="int8"` ⇒
   `prepare()` 不剥前缀（键名 840/840 逐字就是模块名）、`shell()` 走与 Krea2 **同一个** `_shell_int8`
   （F32 岛 ＋ int8 换壳）。bf16 文件自动得到**空 plan** ⇒ `_shell_int8` 空 plan 不换壳 ⇒ **老路零变化**
   （所以同一行同时服务两种文件，不用按文件名分支）。实测 Edit 2511：`int8Plan.layers=840`、
   驻留 **19.09 GiB**（≈20.4B 参数 × 1 字节 ＋ scale，与盘上数据区一致）、`kinds` 里出现 `Int8Linear`。
2. **需求数按文件取 `resident_need_gib()`**（`torch_backend`）：文件侧 `.weight` 是 `I8` 且有配套 scale
   ⇒ 这一层原样常驻 ⇒ 该件驻留＝**数据区字节**；否则退回表值（**不在这里编倍率**：`fp8_scaled` 是 2×、
   bf16 是 1×，两者都叫"低精度文件"）。⚠️ 只改 `decide_for` 一处不够：`declare_chain(part_need_gib=)`
   也读它，否则装配报"整份上卡"、一进 `sample` 段又按 38.05 重判成 `split`（症状：回执与跑起来的驻留不一致）。
3. **余量按件分档**（`residency.RESERVE_BY_KIND = {dit 0.25, te 0.05, vae 0.05}`）：原先是 `split` 一律留
   25%、`off` 只留 1 GiB ⇒ 同一张卡两套口径（空载 20.71 GiB 时 `off` 可用 19.71、`split` 只 15.53，差 4.2 GiB）。
   DiT 留 25%（显存碎片与 1328² 激活），文本塔/VAE 留 5%（25.44 GiB 的 H3 文本塔从"只放 61%"变成 ≥75%）。
   老调用方不传 `kind`/`reserve` ⇒ 仍 25% ⇒ 行为零变化。

判据：`engine_dit_build_test` ⑫（只有 `qwen_image` 的 adapter 是 `int8`、`prepare` 不剥前缀、
空 plan 不换壳、与 Krea2 共用 `_shell_int8`）、`engine_qwen_image_test` ⑩ 系列（读不到文件退回表值且
`int8=None`、假头部 int8 取数据区 19.09、无 int8 层照旧用表、**同一张卡上 19.09⇒`off` / 38.05⇒`split`**、
两条调用口都吃校正后的表、只动 `dit` 一格、回执有 `ditNeed`）、`engine_residency_test` / 
`engine_device_policy_test` 的按件余量与 `reserve`/`reserveKind`/`gpuShare` 三格。

**实测（Edit 2511 int8 / auto / 1328²·3 步·cfg 4）**：装配 **14.2 s**，`ditNeed={"gib":19.09,
"table":38.05,"int8":true,"layers":840}`、`weights.int8Resident=840`、`dequant.count=0`、
`devicePolicy.lastDecision={"device":"cuda","residency":"off","degraded":false,"needGiB":19.09,
"reason":"装得下 ✓：余 21.28 GiB ≥ 需 19.09 GiB ＋ 1 GiB 余量 ⇒ 上卡"}`、`cache.reserved=19.56`
—— 改前同一份文件 `int8Resident=0 / dequant.count=840 / partNeedGiB.dit=38.05 / residency=split`
（卡上 15.9 + 内存 22.15、71 个跨界 hook、**72.5 s/步**）。

### 10.1d Krea2 端到端**已闭合**（2026-11-01 实测，`stage=krea2`）

第五张出图档（与 `image` / `zimage` / `qwen_image21` / `qwen_image` 并列），「提示词 → PNG」在自研引擎里
跑通、不依赖 ComfyUI。真跑实测：装配 9.2 s（`int8Ledger = 224/224` 常驻、`int8OnCpu = 0`）、
1040×1040 / 8 步全部 **51.4 s**（encode 23.9 + sample 26.3 + decode 1.1 + write 0.1）、PNG 1 104 711 字节。

| 环节 | 本族口径 | 别族的数（套过来不报错、图不对） |
|---|---|---|
| DiT | 单流 12.82B（I8 12.155B + BF16 0.665B），**224 层 `Int8Linear` 常驻** | bf16 物化 23.9 GiB ⇒ 装不进 22.49 GiB 卡 |
| 文本 | Qwen3-VL-4B **12 层隐状态栈**（`taps=range(2,36,3)`）⇒ 宽 **30720** | 2.1 的 4096 末层 / 2512 的 3584 |
| σ | flow，**`1.15` 是 μ** ⇒ 底数 `e^1.15≈3.158` | 把 1.15 当底数 ⇒ 曲线几乎不弯 |
| 潜变量/VAE | `Wan21` **16 通道 / 8×**，文件却是 `qwen_image_vae.safetensors` | 2.1 那份同文件走 64 通道 `QwenImage21` |
| 官方步 / 引导 | **8 / 1.0**（turbo 蒸馏） | cfg=1 在 2.1/2512 上要告警，这里**是正解** |

装配口四格（`ditPath` / `textWeights` / `imageVaeWeights` / `tokenizerPath`），缺一格当场拒、首行只列真缺的
那几格、说明段点名 `12.82B` 与「反缩放取 **Wan21**」。清单三件（`dit_krea2_turbo` / `te_krea2_qwen3vl4b` /
`vae_krea2`，`tags:["krea2"]` + `required:false`，必需性归 `STAGE_FILTERS["krea2"].requireAll`）⇒
`readiness("krea2")` = **17.88 GiB** 三件全必需，且**不并进** `image` 档（互斥备选相加会凭空报装不下）。
驻留有本族账 `TorchBackend.KREA2_PART_NEED_GIB`（12.57 / 7.5 / 0.24），与 2.1 的 13.26 / 14.79 是两个数。

⭐ **装得下就三件同驻（2026-11-07）**：分时链每张图都要在 `encode` 那一拍把文本塔**从盘上重读一遍**
（真卡实测 27.4 s，占 1056²/4 步整链 45.6 s 的六成 ⇒ 换装税与提示词无关）。现在装配**前**按**整份**
`part_need` 问一次设备策略，判 `off` 就 `declare_chain(phase_parts=KREA2_ALL_RESIDENT_PHASES)`
（四段都点名三件 ⇒ `enter_phase` 一件都不退），否则给 `None` 走老的分时（零变化）。回执多一格
`allResident`，`components.textEncoder/vae` 的 `phased`/`phase` 随之如实（`resident` / 具体阶段）。
int4 档（5.98＋7.5＋0.24＝13.72 GiB）判 `off` ⇒ 第 2 张起 `换装 0 ms`；int8 档（20.31 GiB）仍判
`split` ⇒ 老实分时（余量不够 1024² 的激活，硬凑会 OOM 在采样第 1 步）。⚠️ 门闩必须同时认
`Decision`（dataclass）与 `dict` —— 只认 `dict` 会让 `AttributeError` 被 `except` 兜成"永远分时"，
回执一切正常、优化静默失效。账与待查清单见 `docs/low-resource-plan.md` §G8。

⭐ **分时档里的"退"不许等于"重读"（`keepParts`，2026-11-07 同日第二轮）**：②只救 int4；
int8 挤不进去必须分时，而老路是「`sample` 把文本塔退掉 ⇒ 下一张 `encode` 从盘上重读 7.5 GiB」
（真卡 19.4–27.4 s/张）。现在 `runtime._krea2_keep_parts` 逐件问同一个门闩、**只收判成 `off` 的件**
（`split` 的件进 `keepParts` ⇒ 整份搬来搬去 ⇒ 分层计划作废 ＋ 搬回时 OOM），交给
`declare_chain(keep_parts=…)`，`enter_phase` 便只在**卡↔内存**之间搬（**字节不重读**）。
`torch_backend.enter_phase` 顺序是**先腾卡再决策再装**（反了就是往满卡里塞 ⇒ OOM），`_pinned_gib`
按 `_WHAT_TO_PART` 别名全试（文本塔摆放叫 `te`、查询叫 `text`，单名查会读到 0 字节）。实测 int8
1024²/4 步两张连跑：第 2 张 **35.1 s ⇒ 15.1 s**，三拍回执 `loaded: []`（剩下 1.0–1.5 s 是纯设备拷贝）。
**质量不动**（搬的是同一份字节、`to()` 同 dtype 不转精度）⇒ 同 seed 三批落盘 PNG **sha256 逐字节相同**；
判据 `engine_krea2_test` ⑦⁵/⑦⁶/⑦⁷。回执多一格 `keepParts` / `keptOnHost`。

⚠️ **本轮抓到的四处「形状自洽、键名全对、只有数值错」**（都会出图、都不报错，靠对拍才抓得住）：

1. **注意力闸门位置**：参考是 `wo(out·sigmoid(gate))`，写成 `wo(out)·g` 形状一样、结果不同
   （`wo` 无 bias ⇒ 两式分别是「逐输入通道加权后混合」与「混合后按输出通道加权」）。症状：换一条完全
   不同的提示词，输出相关 **0.9998**（图是一团纹理）。
2. **文本融合的序列轴**：`layerwise_blocks` 的序列维是**层**（`(b·token, layers, dim)`），
   写成「批 = 层、序列 = token」形状完全自洽，但逐层混合变成逐 token 混合 ⇒ 同样是「提示词几乎不生效」。
3. **μ 当底数**（见上表 σ 行）：σ 曲线几乎不弯，先几步没把噪声压下去。
4. **VAE 单帧 + `temporal_kernel=3`**：`T==1` 时 comfy 走 `autopad="causal_zero"`，把时间核**截到最后一拍**；
   旧写法「T==1 就不补零」直接抛 `kernel(3) > input(1)`，读起来像尺寸算错。

判据：`engine_krea2_test.py`（31 项：五名互斥、μ ⇒ `e^μ`、尺寸吸 16、1040≠1024 的回执、四格点名、
换 TE/VAE 即换装载键、就绪档三件、`run_sync` 三处换装顺序）、`engine_dit_generic_test.py`（闸门与融合两处
各配「参考公式对拍 + 反验有区分力」）、`engine_vae_wan_test.py`（单帧截核 ≡ 左补 2 帧零、`temporal_kernel=3`
的 VAE 能解单帧）。这三份**都不装真权重**，真装真出图是人为验收那一步（上面那组数是实测的）。

⭐ **业务层已接五档**（2026-11-01 收尾）：`bridge.assembly_from_config` 现在认 **五条**出图 stage
（`sdxl` 四件套 ＋ `zimage` / `qwen_image21` / `qwen_image` / `krea2` 四条三件套链）＋ `h3`，
业务侧出图链路（`image_generation.py`）的 `engineStage` 白名单与它**同一份**。三条纪律：

1. **`engineStage` 显式给了就听它的，绝不降级** —— 要的那档装不上就当场报，不偷偷换成"能跑的那一档"；
   没给才走老路（装得上 SDXL 走 SDXL、装不上 H3 兜底，并把原因记进 `fallbackReason`）。
2. **`assembly_from_config` 不认识就当场拒**，报错文案**列全五档**（原先这里是 `else:` ⇒ 任何没认过的
   stage 都被静默改写成 `h3` ＋ 一份探测出来的**视频** DiT，装得进去、跑得出「视频首帧」而调用方以为在出图）。
3. **桥的 stage 清单与 `runtime.ensure_loaded` 的装配线由测试对账**（AST 扒 `runtime` 的分派分支，
   逐条比对桥的 `_chain_stages() ∪ {sdxl}`、业务层 `allowed` 元组、报错文案三处）——
   漏一档的症状是「用户选了它、报错说"不认识"、而 runtime 明明装得下」，属于"支持但没接线"，
   比"不支持"难查得多。本轮就是这么抓出 `qwen_image`(2512) 漏接的。

### 10.1e Flux.1 DiT 骨架已建（2026-11-01；条件靠外部注入）

「按书序」补的第八族骨架：`dit_flux.py`。**能建、能装、能前向**，但**不能从提示词出图** ——
卡住的不是骨架，是盘上根本没有它的文本塔（见下面「如实标的三条」）。

| 环节 | 落点 | 实测／口径 |
|---|---|---|
| 判族 | `dit_family._RULES` | `double_blocks` / `single_blocks` / `img_in` / `txt_in` / `vector_in` ⇒ `flux` |
| 装配 | `dit_build.BUILDERS["flux"]` | `ForwardContract(latent_dims=4, sigma_form="batch", latent_format="flux")`；`vae_scale=8` |
| 建网 | `torch_backend.load_weights` → `_build_via_table` | meta 建壳 → `to_empty` → 物化，峰值只一份 |
| 条件 | `torch_backend.attach_flux_conditioning(vector=…, guidance=…)` | 挂 `default_vector` / `default_guidance`，**不进 `denoise` 的位置参数** |
| 潜变量 | `latent_formats.Flux` | 16ch / 8× / 仿射 `x/0.3611 + 0.1159` |

**结构事实全部读自真权重**（`unet/flux1-dev.safetensors`，780 张量、**11.90 G 参数**、bf16 ≈ 22.17 GiB）：
19 双流 + 38 单流、`heads=24` / `head_dim=128` / `hidden=3072` / `context=4096`（T5）/ `vec=768`（CLIP-L 池化）/
`mlp=12288`。六条口径缺一条就是「形状全对、图是噪声」：

1. 联合注意力拼接顺序是 **(txt, img)**，单流出口按 `[:, txt_len:]` 切回 —— 拼反了不报错，只是每步都错；
2. 三轴 RoPE `axes_dim=(16,56,56)`、`t` 位置号恒 **0**、**h/w 不居中**（0…H−1）—— 与 Qwen-Image 族的居中口径**相反**；
3. 时间嵌入吃 `1000·σ`，**σ 不翻转**（与 Z-Image 族的 `1000·(1−σ)` **相反**）；
4. 调制三元组顺序 **(shift, scale, gate)**：双流出 6 组、单流出 3 组；
5. 块内 pre-norm 是 **LayerNorm**（减均值 / 总体方差），只有 `query_norm`/`key_norm` 是 **RMSNorm**（权重即 scale、不 `+1`，与 `dit_generic` 的偏移口径**不同**）；
6. 一根 `vec` = time 嵌入 + guidance 嵌入 + pooled 向量**三路相加**，decoder 前的 `final_layer` 出 `patch²·16` 通道。

**自检 32 条全绿**（`tests/engine_dit_flux_test.py`）：① 真权重逐键对账（780 键**完全相等**）；
② 形状 → 模块 → `strict=True` 装载；③ 与**刻意换写法**的独立双精度参照对拍；④ 掩码只落前 `n_txt` 列；
⑤ 三路 vec 各自可辨；⑥ `1000·σ`、不居中位置号、**`t_scale` 是活字段**（详见下面第 8 条）；
⑦ bf16 进 bf16 出 + 换序反验（1.1e-1）；
⑨ `default_axes` 对任意偶数 `head_dim` 都给三个**非零偶数**且和为 `head_dim`（详见下面第 7 条）；
⑩ `attach_flux_conditioning` 的回执报**真在生效**的值（详见下面第 9 条）。

**真权重整份装载也已实证**（⑧，默认跳过：吃 22 GiB 内存 ⇒ 不能进 `run_all`；
`ENGINE_FLUX_REAL=1` 开 ⇒ 37/37）：`dit_build` 真建 0.1 秒、**780/780 装载**
（missing 0 / unexpected 0 / shapeMismatch 0）、**驻留 22.17 GiB**、CPU bf16 真尺寸前向
1×16×32×32 一次 1.5 秒、guidance 3.5→7.0 输出变化 **4.480**（`generator(20261101)` 固定才可复现；
早先记的 2.25 是裸 `torch.randn` 的一次观测 ✗）。★① 只比键名形状、② 只在小配置闭环 ——
**只有真文件那一条能证明装配表那一行接对了**，三条都要有。

⚠️ **③ 的阈值是 1e-7 不是 1e-9，别当成"放松判据"**：实现里有**两处刻意的 fp32** —— `rope_freqs` 出的
旋转矩阵按 ComfyUI 口径降 fp32 ✓、时间/guidance 的正弦角度在 fp32 里算 ✓（guidance 3.5 ⇒ 3500 rad，
fp64 算角度会与实现差到 1e-1）。纯 fp64 参照因此有 **≈2.1e-9 的下限**（把 `_RMSNorm` 的 fp32 工作精度也
去掉后仍是 2.1e-9 ⇒ 残差**全部**来自那两处）。1e-7 仍比 bf16 分辨率（2⁻⁸ ≈ 4e-3）紧四个数量级。

**按需激活已通**：`residency._LAYER_RES` 本来就含 `double_blocks.N` / `single_blocks.N` ⇒ 22.17 GiB 可以
按层切上卡；`dit_build` 一行声明让 `load_weights` / `latent_shape` / `_latent_mode` / `denoise` 四处自动跟上。

⚠️ **如实标的三条**（都不加修饰地写在这里，避免下次误读成"Flux 全通"）：

1. **文本塔盘上没有**：`text_encoders/` 只有 `umt5_xxl`（Wan 用的）、`clip/` 只有 Qwen3-VL-4B 系 ——
   **T5-XXL (4096) 与 CLIP-L (768) 都不存在** ⇒ 条件只能由 `attach_flux_conditioning()` 外部注入；
   不挂则走**零向量 + 默认 guidance（3.5）**，那是**结构验证、不是语义**（出图有意义必须等文本塔）。
   也正因为此，**故意不给 Flux 单开 `STAGE_FILTERS`/`registry.STAGES` 档** —— 空开一档只会让
   `readiness("flux")` 报出一个假就绪。
2. **VAE 有本体、未接产品入口**：`vae_flux.py`（16ch / 244 键）在，Z-Image 那条线正用它；
   Flux 自己的 `stage` 装配线没有（同上：缺文本塔就不该有装配线）。
3. **两处"变体/尺寸边界"（诚实拒绝，不静默降级）**：① 有些 Flux 衍生权重在 `txt_in` 前多一层
   `txt_norm.scale`（ComfyUI 参数表里的开关），`flux1-dev` 没有这层、本骨架也不建 ⇒ 换上去会把该键
   报成 `unexpected`；② latent 边长必须是 patch(2) 的整数倍，参考实现会 `pad_to_patch_size` 先补边再切回，
   本仓**不补**（补边＝悄悄改构图），奇数边长直接抛 `ShapeError` —— 1024² 档恒为 128×128，正常路径碰不到。

**本轮顺手修的三处判据自伤**（都是「实现没错、测试自己错」那类，记下来别再犯）：

- `_ref_ln`：参照里把块内 pre-norm 也写成 RMS（不减均值）⇒ 相对误差 0.35，看着像骨架坏了；
- `axes_dim` 不能硬套 `(16,56,56)`：小配置 `head_dim=8` 会算成 1/3.5/3.5（奇数轴 ⇒ RoPE 配对直接抛）
  ⇒ 新增 `default_axes()`，128 ⇒ `(16,56,56)` ✓、8 ⇒ `(2,4,2)` ✓；
- `_RMSNorm` 原来**无条件** `x.float()`，把 fp64 自检悄悄降成 fp32（6.1e-9 → 2.1e-9 后才看清残差来源）。

**第 7 条（2026-11-01 复查新抓，实现侧真缺陷，不是判据）**：`default_axes()` 的分支判据错了两轮 ——
第一版按 `head_dim % 16 == 0` 走 1:3.5:3.5，`16 ⇒ (2, 7, 7)`、`48 ⇒ (6, 21, 21)`；改成 `% 4` 走 ¼½¼ 后
`12 ⇒ (3, 6, 3)`、`20 ⇒ (5, 10, 5)` 仍出奇数轴。两次的症状都是**建得成、跑不动**（`infer_shape` 通过、
`rope_freqs` 抛「轴 1 的维度 7 必须是偶数」），而**报错点落在 axes 上**，读起来像调用方传错轴 ✗。
现在改成三级回落（32 的倍数走真比例 / 8 的倍数走 ¼½¼ / 其余偶数保底 `t∈{2,4}`），并新增 ⑨ 三条判据：
任意偶数 `head_dim` 都给三个非零偶数且和为 `head_dim`、关键档位与文档一致、`head_dim=16` 能真建出 RoPE。

**第 8 条（2026-11-01 复查第二轮，实现侧真缺陷）**：`FluxShape.t_scale` 是**没人读的写死字段** ——
前向写的是 `timestep_embedding(sigma, s.t_freq)`，而 `timestep_embedding` 的 `time_factor`
**默认值恰好也是 1000** ⇒ 行为与正确实现一模一样（自检全绿），但改 `t_scale` 不生效、只有改默认值才生效
✗。这类「配置项是摆设」比算错更难查：它不报错、也不影响默认路径。修法＝两处显式传
`time_factor=s.t_scale`（时间与 guidance 各一处，同族 `dit_qwen_image` 一直这么写），并在 ⑥ 补一条判据
「`t_scale` 1000→1 必须改输出」（拿同一份 `state_dict`、只换 `t_scale` 跑两次）。

**第 9 条（2026-11-01 复查第三轮，回执会说谎）**：`attach_flux_conditioning` 的挂载是**写进模型属性**
（`default_vector` / `default_guidance`），但回执的 else 分支一律报 `guidance_default`、vector 一律报
「零向量」⇒ **第二次只传 vector 时，回执说 3.5、模型里还挂着上一次的 7.0** ✗。这类缺陷不影响出图，
只影响「人看着回执判断挂了什么」—— 而本仓全部报告都是拿回执当事实的。修法＝回执读**活属性**
（`getattr(model, "default_guidance", None)`），并另起一条说明「沿用上次、本次没给不回落默认」；
判据 ⑩ 三连用 `__new__` 轻量替身验（不动显存、不装 22 GiB）。

**第 10 条（2026-11-01 复查第四轮，两处"看着在管、其实没管"）**：
① `FluxShape.__post_init__` 里对单流 `linear1`/`linear2` 宽度的两条校验是**恒真**的 —— 期望宽本来就是
`3·features+mlp_hidden` / `features+mlp_hidden`，`if x != 同一个表达式` 永不成立 ✗，却带着「错误归因留给这里」
的注释，让人以为这里管着（真校验在 `infer_shape` 对着权重形状做）。已删掉，并留注释说明为什么不能在这里校验。
② 键数分解式写成 `19×24 + 38×8 + 14`：14 加起来只有 **774**、与同句的「780 键」自相矛盾 ✗。对着盘上
`flux1-dev.safetensors` 头数出来是 **19×24 + 38×8 + 顶层 16 + guidance 4 = 456+304+20 = 780** ✓，
`expected_keys()` 的两处说明一并改正（schnell 系按此推定 776，盘上无该文件 ⇒ 标注未验）。

### 10.1f Wan2.2 I2V 14B **端到端已接线**（2026-11-02；骨架 2026-11-01）

第九族：`dit_wan22.py`。**判得出、建得出、装得进、前向跑得动**，2026-11-02 起
**plan → 首帧条件 → 双专家接力 → 解码**四段也接上了 ✓（这条线的三处「不报错但错」写在
`tests/engine_wan22_test.py` 的文件头 ✓）。

#### 真跑出片（2026-11-02 首次 ✓）

`_run_qwen_image.py --stage wan22`，真权重四件（high/low 两份 fp8_scaled ＋ UMT5-XXL ＋
`wan_2.1_vae`），首帧用 §10.1d 那张 `output/engine_run/image_seed20261101.png`：

| 项 | 实测 |
|---|---|
| 装配 | 32.0 s；`mode=split`（`onGpu=33 / onCpu=17`、`gpuGiB=15.49 / cpuGiB=11.13`、`gpuShare=0.5819`、`fits=true`） |
| plan | **576×320** / **17 帧** / 16 fps / 4 步 / `wanHighSteps=2` / flow `shift=5.0` / `latent_frames=5` |
| 装配后到落盘 | **155 s**（encode 段换 UMT5＋VAE ⇒ 采样 4 步含两次专家**原地重装** ⇒ decode 段换 VAE） |
| 产出 | `output/wan22_i2v/video_seed20261102.mp4`，**79 292 字节**、576×320、17 帧、16 fps、h264 |
| 画面核验 | 首帧与条件图 L1 **0.0399**（VAE 往返量级 ⇒ 首帧确实被吃进去了 ✓）；相邻帧 L1 0.017~0.04（有运动且不跳）；高频残差占比 **0.08**（噪声会是 0.4+ ⇒ 不是噪点片 ✓） |

**这一跑连抓八处「形状/字段全对、跑起来才炸」**（①~④ 上一轮修，判据在
`engine_wan22_test.case_wiring_attributes`；⑤~⑧ 收尾这轮修，判据在
`engine_vae_wan_test.case_temporal` 与 `engine_wan22_test.case_real_run_truth`）：

1. `TorchBackend.WAN22_PART_NEED_GIB` 只写在文档里 ⇒ 装配第一步 `AttributeError`；
2. 入口通道读 `config["in_channels"]` ⇒ 它按契约短路到**潜宽 16** ⇒ 盘上 36 通道的真文件照拒
   （修：优先 `in_dim`，两者都无则当场拒）；
3. `text_kind="umt5"` 落到 `qwen35` 兜底 ⇒ 报「缺 `model.embed_tokens.weight`」指不到真因；
4. `self._device()` 把**字符串属性**当方法调 ⇒ `TypeError` 报在「文本编码失败」里；
5. ⭐ **Wan 2.1 VAE 从未跑 `time_conv`** ⇒ 时间维一动不动 ⇒ `encode` 报「参考 latent 时间维 17 ✗
   ≠ 潜变量 5 ✓」，解码侧则是「5 潜帧出 5 帧而不是 17 帧」且**不报错**。修在
   `vae_wan.Resample`（两条 3D 分支真跑 `time_conv`，首帧按参考的 `'Rep'` 分支不过卷积 ⇒
   `4k+1` 对 `4(T−1)+1`）；判据 `engine_vae_wan_test.case_temporal`（10 条）＋
   `engine_wan22_test.case_real_run_truth` 的 4× 不变量；
6. ⭐ 首帧条件的潜帧数**只**问请求的 `temporalCompression` ⇒ wan22 请求不带它（那是族常量 4）
   ⇒ 报「首帧条件需要 temporalCompression」，读起来像用户少填字段。修：`pipeline.run_sync`
   先取 `plan.latent_frames`、没有才按请求压缩比现算；
7. ⭐ `dit_wan22.WanDiT.forward` 旧写法 `conv(latents.to(pw.dtype)).to(latents.dtype)` ⇒ conv 走
   bf16、紧接着的 `Linear` 吃回 fp32 ⇒ 采样第 1 步 `mat1 Float / mat2 BFloat16`。修：计算精度
   **跟权重走**且整条前向只留一种（与 `dit_qwen21` 同口径）；
8. `UpResidualBlock.forward` 一把 `self.upsamples(x)` ⇒ `nn.Sequential` 只传位置参数 ⇒ 里面的
   `Resample` 收不到 `first_chunk`，按 `2T` 吐帧而 `DupUp3D` 按 `2T−1` 吐 ⇒ `out + avg_shortcut`
   当场尺寸不符。修：逐层传 `first_chunk`。

**教训**：这条线当时的 52 条静态判据（源码结构 / 报错文本 / plan 口径）一条也没拦住上面 8 处 ——
它们全是「两段独立正确的东西在交界处对不上」（类属性 vs 文档、契约短路 vs 权重事实、
浮点精度 vs 权重精度、「形状刚好对上」vs 真做了时间压缩、开关只传到一半）。
交界处的判据只能靠**真跑**（现在 58 条里的 ⑤ 那 6 条就是照真跑倒推出来的回归钉 ✓），
静态判据只能防回归。

接线落点（一张表看全，避免下次又把「已接」读成「没接」✓✗）：

| 段 | 落点 | 关键判据 |
|---|---|---|
| plan | `pipeline._fill_wan22_plan` | **必带 `firstFrame`** ✓；尺寸吸 **16** ✓；像素帧吸 **4k+1** ✓；`fps` 24 ⇒ **16** ✓；`steps` 30 ⇒ **4** ✓；`wanHighSteps = round(steps×0.5)` ✓；σ 走 flow **shift=5.0** ✓（`WAN22_FLOW_SHIFT` ✓） |
| `initImage` | `pipeline.build_plan` 的 `elif init_image` 分支 | 与 `firstFrame` **互斥** ⇒ **当场拒** ✓✗（不是「静默忽略」✓ —— 静默丢掉会让用户以为起点是他那张图 ✓） |
| 文本塔 | `torch_backend._encode_wan22_text` | UMT5-XXL **4096** 宽 / 序列 512 ✓；**`padding` 位置的条件清零** ✓（不清零会让补位词进 cross-attn ✓ 不报错 ✓） |
| 首帧条件 | `torch_backend._wan_i2v_condition` | 像素过 VAE ⇒ 校验 **16 通道** ＋ 时间维 == `t` ✓；**只存两段** ✓（拼接交给 `i2v_concat`，单一真源 ✓） |
| 36 通道拼接 | `torch_backend._wan_concat_latents` → `dit_wan22.i2v_concat` | 没编过首帧条件就 `denoise` ⇒ **拒跑** ✓✗（零填充凑 36 会出「与首帧无关」的片 ✓、不报错 ✓） |
| 双专家换装 | `register_wan_experts` ＋ `_wan_expert_tick` | 判据是 **σ 阈值** `sigmas[wanHighSteps]` ✓、边界 **`<=`** ✓（写 `<` ⇒ 4 步退化成 3/1 ✓）；`bind_plan` 每轮**复位** high ✓ |
| 装配 | `runtime._ensure_wan22_loaded` | 三格必给 ✓；装完核 **36 通道** ✓（同族 T2V 是 16 ✓）；`ditPath2` **进装载键** ✓；`declare_chain` 三件**分时** ✓ |

⚠️ 2026-11-01 的旧结论已作废 ✓：`umt5_te` ＋ `umt5_tokenizer` 当时「有实现、没接线」✗，
现在装配线收 `textWeights` ✓、`_encode_wan22_text` 直接调它 ✓（见 10.1g 的 UMT5 段 ✓）。

| 环节 | 落点 | 实测／口径 |
|---|---|---|
| 判族 | `dit_family._RULES` | `blocks.N.self_attn` + `cross_attn` + `modulation`（6 路）⇒ `wan` |
| 装配 | `dit_build.BUILDERS["wan"]` | `ForwardContract(latent_dims=5, sigma_form="batch", latent_format="wan21")`；`in_channels=16`、`vae_scale=8` |
| 建网 | `torch_backend.load_weights` → `_build_via_table` | meta 建壳 → `to_empty` → 物化；`patch` 从 `patch_embedding`（Conv3d）权重读，不用默认值 |
| VAE | `torch_backend.attach_wan_vae` | 已有（Krea2 线在用它）；`wan21` 口径 = 16ch / 8× / meanstd |
| 文本塔 | `umt5_te` ＋ `umt5_tokenizer` **已接线** ✓（`_encode_wan22_text`） | 盘上有 `umt5_xxl_fp8_e4m3fn_scaled`；塔与分词器都过了 `engine_umt5_test`，装配线收 `textWeights` ✓ |
| I2V 拼接 | **已接线** ✓（`_wan_i2v_condition` ＋ `i2v_concat`） | 骨架吃**拼好的** 36 = 16 + 16 + 4 ✓；`dit_wan22.split_i2v()` 只切不造 ✓（反查用 ✓） |

**结构事实全部读自真权重**（`diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`，
头里 1908 条 = 真权重 **1095** + fp8 伴生 **813**；F8 406 / F16 689；**14 288 901 184 参数**
⇒ bf16 常驻 **≈26.62 GiB（≈28.59 GB）**，大于 22.49 GiB 的卡 ⇒ **本机装不下**）：
40 块 / `dim=5120` / `ffn_dim=13824` / `heads=40`（`head_dim=128`）/ `in_dim=36` / `out_dim=16` /
`text_dim=4096` / `freq_dim=256` / `patch=(1,2,2)`。六条口径缺一条就是「形状全对、图是噪声」：

1. **六路调制的位置与形态**：`time_projection` 出 `6·dim` → `unflatten(2, (6, dim))` ⇒ `[B,1,6,dim]`，
   块内 `(modulation.unsqueeze(0) + e).unbind(2)` 切出 6 路 `[B,1,dim]` —— **不是** Flux 的
   `chunk(6, dim=1)`（那样切出来还是 `[B,1,6,dim]`，形状一直到 `addcmul` 才炸，报错点看着像 e 没算对）；
2. 自注意力用三路 `0·1·2`、FFN 用 `3·4·5`；**交叉注意力不受门控**且前置 `norm3` 是**带权带偏**的 LayerNorm；
3. 门控残差三连都是 `addcmul(e, norm(x), 1 + e')` / `addcmul(x, y, e'')` —— 那个 **`1 +`** 缺了不报错，
   只是整支近似乘以 0（自检④专盯这一项，反向参照把 `1+e` 写成 `e` 会差 **1.86e-1** 被抓住）；
4. **时间口径 `1000·σ` 在骨架里**：参考实现把这次乘法放在采样器 `ModelSamplingDiscreteFlow.multiplier`，
   本仓收裸 σ ⇒ `dit_wan22.TIMESTEP_SCALE` 补（自检⑧用 spy 抓 `time_embedding` 的输入，漏乘差 1.72）；
5. **3D RoPE 三轴宽度不等分**：`axes_dim = [d−4(d//6), 2(d//6), 2(d//6)]`（128 ⇒ **44/42/42**），
   位置号**从 0 起、不居中**，`theta=10000`，只转 q/k（v 不转）；写成等分 `(128/3…)` 或居中都会整条偏；
6. `heads` **读不出来**（q/k/v 是单条 `[dim,dim]`、`norm_q` 宽 = `dim`）⇒ 只能查
   `dit_wan22.HEADS_BY_GEOMETRY[(dim, ffn_dim, layers)]`（盘上只有 `(5120,13824,40) ⇒ 40` 一条），
   表外几何必须显式给 `heads`，**不猜 `dim//128`**。

**自检 46 条全绿**（`tests/engine_dit_wan22_test.py`）：① 真权重逐键对账（1095 缺 0 / 813 余项**全部**是
fp8 伴生 / 参数量与头里 numel 之和互核）；② 形状反推 → 建 → `strict=True` 装载无缺无余；③ 与**刻意换写法**的
独立 fp64 参照前向差 **5.3e-16**；④ 门控 `1+` 项 + 反向反验；⑤ RoPE（1×1×1 恒等 ⇒ 不居中；三轴宽度；
保范数；只转 q/k）；⑥ `heads` 只许查表；⑦ 五条具名拒绝（4 维 latent / 通道 16 / context 宽错 /
非 patch 倍数 **不补边** / sigma 个数）+ bf16 进 bf16 出；⑧ `1000·σ`；⑨ 36 通道布局只切不造；⑩ `param_count`
公式（每加一层的增量必须等于该层实测）；⑪ 装配表一行接对（契约、`in_channels=16` 不是 36、`configSources` 如实）。

⚠️ **③ 的残差只剩 5.3e-16，靠的是两处"精度交接点"照实现镜像**（`rope_freqs` 出的旋转矩阵按 ComfyUI 口径
降 fp32、时刻正弦按 `sinusoidal_embedding_1d` 降回入参精度）。不镜像这两步，纯 fp64 参照恒定停在
**≈6.5e-9** —— 那个量级不是逻辑错（真缺陷在这轮实测都是 1.9e-3 ～ 1.9e-1），只是 fp32 落地差，
但足够把「< 1e-9」这条判据钉死成假失败。⚠️ 别为了让判据过就把阈值放宽到 1e-6：
6.5e-9 与 1.9e-3 之间差五个数量级，一旦真错，放宽的阈值照样能抓 —— 但镜像之后 5e-16 能把
「多路调制顺序、unpatchify 通道序、交叉注意力不带门控」这三件事一起钉住。

**如实标的一条边界**（**不是**代码缺口 ✓，别写回 `codeGaps` 里报「没实现」✗）：

1. **单份 bf16 26.62 GiB > 卡 22.49 GiB**：`fp8_scaled` 只省盘不省卡（反量化到 bf16 才是驻留量）⇒ 必须
   `split`/`cpu`，并**按件分时**（`declare_chain` 的 `encode`/`sample`/`decode` ⇒ 峰值只有一份 DiT）；
   两份专家**不并存** ⇒ 采样中途 `load_module_weights` 原地重装（约 20~40 秒 ✓，算进耗时 ✓）。
   本机 sm86 无 fp8 硬件加速 ⇒ 「装得下」不等于「跑得快」✓。就绪档 `readiness("wan22")` 如实报
   **33.14 GiB / `fits=false`** ✓（四件全是必需 ✓；`tags` 用 `wan22` 而**不是** `wan` ✓，
   否则会把 `wan26_i2v_14b_gguf` 也卷进来 ✓✗）。

**本轮顺手修的三处判据自伤**（都是「实现没错、测试自己错」，记下来别再犯）：

- 参照的 `_ln` **漏了 bias**：`norm3` 是带权**带偏**的 LayerNorm，只乘权重差 **1.9e-3**，
  症状看着像「RoPE 没对齐」（实测 `norm3.bias` 那一项单独就贡献 0.5947）；
- 参照 `_rope` 的 `even/odd` 是 `out` 的**视图**：第一行赋值后就被就地改写，第二行读的是脏值
  （跨轴差 `0.84147` vs `0.4546`，读起来像「角度算错」）⇒ 必须 `clone()`；
- ④ 的 `zero_gate` 把**合并后的 slot**（含时间 `e`）整支清零，语义变成「连时间条件也不要了」
  ⇒ 只能清 `modulation` 这个**参数**、`e` 照留（与实现里把参数置零是同一件事）。

### 10.1g SEEDVR2 修复线（**端到端已接线**）＋ UMT5 文本塔（**已接线**，2026-11-02 起）（2026-11-01）

**SEEDVR2 这条线从「能装」走到了「能修」** ✓（2026-11-01 当天三步一起接完 ✓）：

| 格 | 落点 |
|---|---|
| 主干 | `dit_build.BUILDERS["seedvr2"]` ✓ ＋ 契约（5 维潜变量 / σ 批形态 / 口径 `seedvr2`）✓ ＋ **真实 3B fp16 权重与参考实现逐位对拍 `maxdiff == 0`** ✓（89 键逐名逐形状 ✓，反推的形状没有一项是出厂常量 ✓） |
| VAE | `torch_backend.attach_seedvr2_vae` ✓／`decode` 的 `seedvr2-real` 支 ✓／**像素→潜变量** `encode_pixels` ✓（编码器直出 32 通道取前 16 均值 ＋ `×0.9152` ✓）；`_check_attached_vaes` 对「**33 = 16 ＋ 17**」单独放行 ✓（拿一个全局「主干通道 == VAE 通道」的判据去卡，会把这条唯一合法的加法路径卡死 ✗） |
| 装配 | `runtime.ensure_loaded(stage="seedvr2", ditPath=…, imageVaeWeights=…)` ✓ —— **只两格**：这一族**没有文本塔** ✓ ⇒ 要 `textWeights` 就是把图片/视频那几族的清单照抄过来 ✗（多一个必填格 = 这条线永远装不上 ✓） |
| 修复支路 | `pipeline._fill_seedvr2_plan`（尺寸/帧数**只认源视频** ✓、步数 30→**1** ✓、`cfg>1` 只报不拒 ✓）＋ `_encode_seedvr2_conditioning`（`cat(vae_latent 16, mask 1)` ⇒ **17** ✓）＋ `denoise` 的条件位 ✓ ＋ `pipeline` 的 `bind_plan` 钩子 ✓ |

**`bind_plan` 为什么必须存在** ✗✗：`encode_text(request)` 的契约里**没有 plan** ✓（出图族确实不需要 ✓：
提示词编码与画布/帧数无关 ✓✗）；**修复族不成立** ✗ —— 它的"条件"**就是**源视频编码出的 16 通道 ✓，
而「编多少帧、编成多大」正是 plan 说了算 ✓ ⇒ 两边必须看**同一份** ✓。两处各算一遍的症状：plan 说 94 帧、
条件编出 121 帧 ✓ ⇒ 采样第 1 步 33 通道拼不上 ✓，而报错指不到「算了两遍」✓✗。

**三处「不报错、只是结果不对」的口径**（都已入判据 ✓）：

- ① **`positive` 必须回 17 通道张量** ✓，不能塞文本状态位 ✗，也不能回 `None` 让 `_branches` 把整个 dict
  当条件 ✓（后者会让 `denoise` 的 `context` 位收到一个 dict ⇒ `_resolve_context` 报「要 (B,L,D)」，
  读起来像"提示词坏了"✓✗）；`negative=None` ⇒ **单次前向** ✓（本族官方 cfg=1 ✓）；
- ② **`mask` 通道数只能是 1** ✓：写成 3 或 16 会得 19 / 32 通道 ✓ ⇒ 主干 `vid_in` 的 Conv3d 当场抛
  「期望 33、实际 N」✓（还算清楚 ✓）；但**拼接在主干里做** ✓ —— 这里若先拼成 33 ✗，
  主干再拼一次就是「16 ＋ 17 ＋ 17 = 50」✓✗；
- ③ **落盘回执不许把真 VAE 的成品叫"噪声"** ✓✗：`write` 的 5 维分支曾把 `synthetic=True` ＋
  「未训练 VAE」写死 ✓（接这条线**当场踩到**的假证 ✓）⇒ 现由 `outputs["synthetic"]` 说了算 ✓。

| 模块 | 是什么 | 断在哪 |
|---|---|---|
| `dit_seedvr2` | SEEDVR2 / NaDiT 视频修复主干（32 块；前 10 块双塔、后 22 块共享；`in=33`） | **已闭合** ✓ —— 装配表 ＋ 契约 ＋ `stage="seedvr2"` 装配线（只两格）＋ 修复支路全通 ✓ ＋ **真 3B 权重出片跑通** ✓（720×480 / 8fps / 49 帧 ⇒ 1.27 MB mp4，715 秒，`ok=True`）；唯一短板是**解码慢**（整段解码单独量 336 秒、分配器峰值 47.6 GiB，见本节末） |
| `umt5_te` | UMT5-XXL 文本塔（Wan2.1/2.2 的提示词入口：24 层 / 4096 / **双向**注意力 ＋ 每层各一份相对位置偏置 ＋ FFN 门控） | `ensure_loaded(stage="wan22")` 还不收 `textWeights` ✗ ⇒ 生产 0 处 import。⚠️ 它与本仓其余几座塔（因果 ＋ `layer_idx=-2`）**口径相反** ✗，照抄 `qwen3_te` 的 mask 写法会「装得上、算得出、出图与提示词几乎无关」🤐 |
| `umt5_tokenizer` | UMT5 的 sentencepiece **unigram** 分词器（`nmt_nfkc` ＋ `add_dummy_prefix`、词表 256384） | 唯一出口是 `umt5_te` ⇒ 塔不接线它就没有调用口；接线顺序是**先塔后分词器** |

**判据**：`tests/engine_seedvr2_test.py`（82 项：小号合成几何钉形状反推 ＋ 前向守门 ＋ 装配族表 ＋
真实 3B 权重**只读头**核几何与参数量 ＋ ⑮ 修复 plan 口径（`seedvr2_target` 的吸附算术；帧率**只抄源视频**
——盘上那份真 mp4 是 **8fps**，写死 24 会让 6.125 秒的片子变成 2.04 秒且不报错；`nbFrames` 缺失才按时长估、
超 121 帧如实截断、不足 5 帧当场拒；`build_plan` 无 `referenceVideos` 拒、给两路拒、路径不存在拒；
plan 的 `width/height/frames/latent_frames/fps` 五个字段全部来自源视频；默认 30 步换 1 步并如实报；
显式 `steps=4` 照跑只报不改；无文本塔 ⇒ `negative` 与 `cfg>1` 各有告警）＋ ⑯ 条件通路（没挂
`attach_seedvr2_vae` 拒且点名、缺 `referenceVideos` 拒、无条件不许造噪声、噪声形状**跟着条件走**
`(1,16,3,2,2)` 而不是按 plan 猜、plan 被**真解码**帧数校正且留告警、同 seed 逐位复现、
`latent_channels` 缺则拒**不猜 16**）、`tests/engine_seedvr2_vae_test.py`（105 项：250 真键反推
`(128,256,512,512)`／时间 ÷4／空间 ÷8／16 通道，四种坏权重各一条人话判据，帧数铁律 `4n+1`，
**因果性双侧**——扰动末帧不许改首潜帧／首像素帧，三处口径的反向验证（时间 padding 换零填、
`remove_head` 恒等、组归一化跨帧 ⇒ 判据必须红），与 ComfyUI `comfy/ldm/seedvr/vae.py` 对拍得到的
**黄金数值**，以及⑪ `TorchBackend` 接线 13 条：`encode_pixels` 必须回**模型空间**潜变量、
Wan/Flux 的 VAE 顶替要当场拒 ✓；⑫ 时间分块 22 条（×2/×4 × 块 1/2/4/6 ＝ 帧数 ＋ 逐位，
真权重 250 键，外加「掐掉缓存 ⇒ 数值必崩」的反向判据）；⑬ OOM 兜底 8 条（降到
`SUGGESTED_TEMPORAL_CHUNK`／标 `exact`／非 OOM 原样抛／`fallback=false` 真报错／降完仍 OOM 给人话／
显式配置照办））、`tests/engine_seedvr2_pipeline_test.py`（21 项修复支路：
11 帧真 mp4 ⇒ **修 9 帧**、`(1,17,3,4,6)` 条件、mask 全 1、主干入维 **33**、`condition=None` 回落、
`write` 的 `synthetic` 说真话）、`tests/engine_umt5_test.py`（五条坏法各一条判据：键名错一个字 /
相对位置偏置**只写第 0 层** / 掩码方向反 / FFN 写成单支 / RMSNorm 写成 LayerNorm）。

SEEDVR2 这三份里，主干与 VAE 的**逐位对拍已经入册**（VAE 走黄金数值 ✓）、修复通路的**接线**也已入册 ✓；
**真 3B 权重出片已于 2026-11-01 跑通** ✓（`seedvr2_3b_int8_convrot` ＋ `ema_vae_fp16`、`horse.mp4`
720×480 / **8fps** / 49 帧 ⇒ 修完仍是 720×480 / 8fps / **6.125 秒**、mp4 **1 270 000 字节左右**，
`ok=True`；全程 715 秒 ＝ 装载 11 ＋ 编码 95 ＋ 采样 34 ＋ **解码 571** ＋ 写盘 ✓）。

⚠️ 真卡首跑逼出**三个只在真卡/真权重下才露头**的坑 ✓，都已修并各自配了判据：
① `SeedVR2DiT.forward` 原来拿 `latents.dtype` 当 `dtype` ✓ —— 采样侧潜变量是 **f32**（故意 ✓，
逐步累积更准 ✓）、权重是 bf16 ⇒ 第一个 `nn.Linear`（`emb_in`）当场
`mat1 and mat2 must have the same dtype, but got Float and BFloat16` ✓，报错指向 Linear、
读起来像"权重装错档"✗；现在改成**在模块入口对齐到权重精度** ✓，口径与 `dit_qwen_image` 逐字相同
（判据 ④′，含"条件给 float64 也要被对齐 ⇒ `cat` 不炸"✓）。
② `_flatten` 把逐样本 `vid_shape` 建在 **CPU** 上 ✓ —— 它下游是当**索引张量**用的
（`_window_idx` → `index_select`）⇒ CUDA 上 `Expected all tensors to be on the same device,
but got index is on cpu` ✓；CPU 小尺寸测试**永远试不出来** ✗；现在跟随数据设备 ✓。
③ `_repeat_concat_idx` 里 `txt_len * txt_repeat` 跨设备 ✓（`_window_count_tensor` 建在 CPU）⇒
同款报错；现在入口把 `txt_repeat` 搬到 `vid_len` 的设备 ✓。
**解码是唯一的大瓶颈** ✓：720p/49 帧整段解码 **336 秒**（这是**单独量**解码那一次的账 ✓；端到端那次
落在解码段的是 **571 秒**，两次差值的归因**尚未做** ✗ 别把两个数混起来引 ✓）、分配器峰值 **47.60 GiB**（卡 22.49 ✓），
单笔最大 ≈17.3 GiB，来自"时间 ×2 上采样之后"的 512/1024 通道重排张量 ⇒ 全程在 OOM 重试里打滚
（片子仍是对的 ✓）。**潜帧分块（时间维）现已接通且与整段相等** ✓：真根因**不是**丢帧数
（「`_remove_head` 首帧特例让块内相位少 3 帧」那条是猜的、2026-11-02 已否 ✗）——是
`_InflatedCausalConv3d` 的 `_extend_head` 把**块首帧**复制 2 份当左 padding ⇒ 块首帧自己就错了，
因果链把它送到块内**全部后续帧**，补多少上文都救不回来（四个变体 `4k-3`/`4k`/跳过 `_remove_head`/
±8 帧平移扫描全差 1.8~2.8）✓✗。正解抄自 comfy 的 `MemoryState` ＋逐层 `memory_cache`：块首接
**块外真实帧**、`ACTIVE` 时时间 ×2 不丢第 2 帧、逐层显式下传；外部旋钮 `vaeTiling.temporalChunk`
（**潜帧**单位，**默认关** ✓）；**OOM 兜底在同一族也降它**（阶梯 `chunk_ladder()` ＝ `4→2→1`，回执多
一格 `"exact": true` —— 这条兜底**不换画质**，与 Wan 那族降 tile 换字节不是一回事，读回执别混）✓。
验收：`engine_seedvr2_vae_test` ⑫⑬ 105/105 —— ×2/×4 × 块 1/2/4/6 帧都「帧数 `4T-3` 相同 ＋
`max|Δ| ≤ 1e-5`」，**真权重 250 键也在内** ✓，另有「掐掉缓存 ⇒ 数值必崩」的反向判据 ✓，兜底侧
「非 OOM 不许降档／`fallback=false` 必须真报错／**整条阶梯试遍**仍 OOM 才给人话／自动降的档如实标
`reason="auto"`」也各一条 ✓。组归一化改写成**不拷贝**的 5D 版本（数值与黄金值一致 ✓）也没能降峰值
⇒ 别把它当性能成果汇报 ✗。

**真跑实测（720×480 / 49 帧 / A5000 22.49 GiB，2026-11-06，每档单开进程）**：整段 496.2 s；
`chunk=4` 231.1 s / 一块 19.51 GiB；`chunk=2` 154.3 s / 11.94 GiB；`chunk=1` **8.3 s / 8.15 GiB**。
三条只有真跑才露头的事实：① 这档**不报 OOM 而是溢出到主机内存**（`reserved` 86.67 GiB ≫ 卡，
`allocated` 只有 19.51）⇒ 只慢 28 倍、不崩，只认 `_is_oom` 的兜底**永远等不到** ✗✗；
② 自建缓存原用 `.detach()` 的**视图** ⇒ 钉住整块激活 ⇒ 分块比整段还费（22.62 vs 13.48 GiB），
改 `.clone()` 后 11.90 GiB，判据改量**存储字节数**；③ `shrink_for_peak()` 按「峰值÷卡容量」
一次跳到该降的档（4 → 1），别一档一档疼；溢出后下一轮开跑前 `_trim_cuda()`（不清就在碎片里再颠簸：
已降到 1 档仍 200.3 s，清了 13.3 s）。真通路两轮实证：整段 496.2 s ⇒ `spilled/nextTemporalChunk=1`
⇒ 第二轮 13.3 s、`reason="auto"`。详见 `docs/reference-audit.md` §六.8 与 §六.8.1。

### 10.1h 图生图（`initImage` ＋ `denoise`）真跑实证 · 2026-11-02

**这是「公式对、单测绿、真跑斜」的标本** ✓ —— 修之前 `init_latents` 在单测里拿到的 VAE 是
**常驻替身**，而真装配线是分时的（2512：`encode`→text / `sample`→dit / `decode`→vae）⇒
「起点由参考图编码而来」这条路**从来没被真走过** ✗。

真跑两档（`qwen_image` 2512 bf16 38.05 GiB、卡 22.49 ⇒ `split`：25 层上卡 15.9 GiB ＋ 35 层留
CPU 22.15 GiB；`_run_qwen_image.py --stage qwen_image --target storyboard-last --init <1344×768>`）：

| denoise | 起点 σ | 实跑步数 | 产物字节 | 与参考图 MAE | PSNR | 64×64 相关 |
|---|---|---|---|---|---|---|
| 0.55 | 0.7949 | 20 | 1 612 329 | 0.0506 | 23.49 dB | **0.9133** |
| 0.05 | 0.1403 | 6 | 1 593 819 | 0.0267 | 29.24 dB | **0.9982** |

相关 **0.9982** ⇒ 起点 latent 就是参考图本身 ✓（若起点仍是纯噪声，两档都会 ≈0 ✗）；
0.55 档退到 0.9133 ⇒「保结构、允许改」的量级正确 ✓，且与 `denoise` 单调 ✓。

**踩到并已修的真缺陷**：`TorchBackend.PHASE_PARTS` 原本只有三段 ⇒ `init` 这一拍手上只有 TE、
VAE 要到 decode 才装 ⇒ 图生图必报「没挂 VAE ✗ ⇒ 没法把图片编码成潜变量」（文生图永远走不到）。
修法：新增 `"init": ("text", "vae")` —— **text 必须一起写进去** ✗（`enter_phase` 会退掉不在
`wanted` 里的件，只写 `vae` 就把刚 encode 完的 16.1 GiB TE 退掉、`sample` 又得从盘重读一遍）；
两者同驻 16.34 GiB < 22.49 ✓ 装得下。`pipeline` 侧只在 `if request.init_image:` 时申请
（纯文生图不多跑一趟换装）。判据：`engine_pipeline_test` ⑰''''（表里有这段且含 `text`）与
⑰'''''（AST 认定那句调用被 `if request.init_image:` 包住）；反向验证过两种坏法（删段 / 段名写错）。

**成本**（同卡同权重、`cfg=1`）：装载 19.9 秒 ⇒ encode 段（换装 TE ＋ 编码）37.7 秒 ⇒ `init` 段
VAE 编码本身 **88 毫秒**（换装另算）⇒ 换回 DiT 约 70 秒 ⇒ 每步 18 秒（`cfg=4` 时每步 55 秒：
两遍前向 ＋ split 跨界）⇒ **分时驻留族做图生图，代价在两趟换装、不在 VAE 编码**。

**⚠️ 2026-11-07 低配真机补的两处（都只在"卡真的不够"时才露头 ✓）**：
① `release_part` 原先只判 `holder.module is self._textEncoder` ⇒ 对**摆本体、存外壳**的文本塔
（`attach_krea2_te` 摆 `module.tower`、存 `module`）恒 False ⇒ 把手不摘 ⇒ 钩子环吊着每层 ⇒
**卡上一字节不退**（实测退 text 前后 `alloc` 都是 6.41 GiB，而回执写 `released: true`）；
修法＝`_WHAT_TO_PART`（`dit/h3→dit`、`te/h3-te→text`）＋ 摆放时记 `_residencyOwnerWhat`，
`release_part` 按**同一性或名字**任一成立即摘，账本把所有**别名**（含 `#host`）一起退；
② `enter_phase` 原先按**整份** `wanted` 判，已在场那件**本身就占着** `free` ⇒ 数了两遍 ⇒
需求虚高 ⇒ 判 `cpu/off` ⇒ 报「要整份放进内存」；修法＝只把 `missing` 交给 `decide_for`
（实测 `init` 只差 0.24 GiB 的 `vae` 却被按 7.74 GiB 判 ⇒ 拒装 ✓✗）。
判据 `engine_device_policy_test` ⑨⁗／⑨⁘／⑮。

两条会让人误判的真事实：① `split` 装载要求「DiT 的 CPU 那一半整份进内存」＝ 22.79 GiB ⇒
宿主只剩 7.02 GiB 时**直接拒装**（那次被误读成「权重有问题」，实际是上一次会话残留的探针进程
吃了 50 GiB 内存）；② `_run_qwen_image.py` 曾把 `write()` 返回里的 `note`（一句中文说明）当路径
去 `exists()` ⇒ 每次真跑都以「声明了但盘上没有」收尾、把真异常埋掉，已按**键名**跳过。

### 10.2 前向骨架的覆盖进度与剩下几族的缺口

格式与量化层（L4-2 / L4-3）对所有盘上权重都通用：safetensors、GGUF、fp8、int8、int4 打包、convrot、
NVFP4 都能读能还原。但**网络本体**（L4-5）只有十一个族有骨架：H3、SDXL、GGUF LLM、DiT 通用（Krea2）、
Z-Image（`dit_lumina`）、Qwen-Image 2.1（`dit_qwen21` + `qwen35_te`）、Qwen-Image 2512/Edit
（`dit_qwen_image` + `qwen25_te`）、Flux.1（`dit_flux`）、Wan2.2 I2V（`dit_wan22`，见 §10.1f）、
SEEDVR2（`dit_seedvr2`，**已接线** 见 §10.1g）、Music3 音频主干（`dit_music3` ＋ `te_music3` ＋ `vae_music3`，
**已接线并真权重出歌** ✓ 2026-11-06、`stage="music3"` 全链见 §10.1 末段）——
除 H3 / SDXL 外的几族都由 `torch_backend` 按权重形状建起来。F5-TTS **暂无前向** ✗
（权重**在盘上**：`checkpoints/F5-TTS/model_1250000.safetensors`，约 1.29 GB ✓ —— 缺的是网络本体与
vocoder 接线，**不需要再下载** ✓✗）。
文本塔侧补了 **UMT5-XXL**（`umt5_te` ＋ `umt5_tokenizer`）—— 2026-11-02 起**已接线** ✓：
Wan2.2 I2V 装配线收 `textWeights`、`_encode_wan22_text` 直接调它（见 §10.1f / §10.1g）。
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

### 10.3 曾记为「整域关闭」的四件，2026-11-08 核实**全部已接线**

本节早先写的是「实现保留、调用链已断」——那描述的是 Node→Python 绞杀期的中间态，**现已全部作废**。
四个文件今天都在真调用链上（`agent/runtime.py` 文件头自己就写着「曾经的『依赖未迁』项现已全部
落地接线」，与旧文本直接矛盾；这条矛盾是 2026-11-08 全面检查时抓到的）：

| 文件 | 当前调用方（按名字核，不写行号） |
|---|---|
| `app/services/color_grade.py` | `services/image_generation.py` 出图后调 `apply_color_grade_to_file`；`routers/presets.py` 与 `routers/characters.py` 调 `normalize_color_grade` |
| `app/services/data_storage.py` | `routers/storage.py` 的 `GET /api/v1/storage/info` 与 `POST /api/v1/storage/change`（`main.py` 挂 `storage_router`） |
| `app/services/rhythm_phase.py` | `agent/tools/storyboard_tools.py` 的 `assign_rhythm_phases`；`agent/runtime.py` 的 `rhythm_guidance_for_episode`；`routers/dramas.py` 的 `get_drama_rhythm` |
| `app/agent/subagent.py` | `agent/runtime.py` 的 `_tool_factory("subagent")` 装 `run_subagent` / `list_available_agents` |

**仍然不许当死代码删**，但理由换了：不是「留着将来重启」，而是「**删了当场断链**」——
`data_storage` 一断，前端设置页切数据根立刻 501（它正是「删 `backend/`」的最后一块阻塞，见
`routers/storage.py` 模块头）；`subagent` 一断，orchestrator 退化成不能委托领域专家。
判据：`route_parity_test`（224 条端点、0 条未迁移）＋ `storage_change_test` 43 条 ＋ `agent` 各套。

**切根到底搬什么**（`data_storage.change_data_root`）：库三件（`drama.db` ＋ `-wal` ＋ `-shm`）、
`static/`、`library-overlay.sqlite`，加上 `MIGRATED_TREE_SUBDIRS` 里整目录搬的四个
（`traces`／`lora-train`／`canvas`／`knowledge`）。收录判据只有一条——**「丢了算不算得回来」**：能重下/重算/重跑的
一律不进（那些进 `app/core/config.py` 的数据根登记表并标 `✗`），画布草案、训练产物与知识台账没有源可再生，故必须搬
（`canvas/` 与 `knowledge/` 都是 2026-11-08 补的：两者落点都现算 `get_data_root()`，切根后直接指向新根的
空目录 ⇒ 症状都是「换个目录，XX 全空了」。`knowledge/` 那格是**改判**不是新认知 —— 它原先整目录被归进
「能重算」，而里面四个库里 `knowledge-ledger`（条目／版本／审计）与 `training-registry`（promote 依据与凭据）
是真源，只有 `outcome-index`／`skill-usage` 能重放 ⇒ 教训是**别按目录名判、要按目录里的东西判**；
`storage_change_test` ⑨′ 因此改成现算 `db_path()` 的父目录名，将来真源挪去白名单外会自己亮）。
迁移是**复制**、旧根保留作备份，所以起手先做磁盘预检
（`MIGRATION_HEADROOM_BYTES` 固定余量，`shutil.disk_usage` 取不到可用空间时**不拦**——宁可不预检也不误拒
合法目录），预检放在 `close_engine()` **之前**，保证拒绝路径「状态一点没动」。参考侧
（`reference/anything-llm`／`reference/ragflow`）的数据根都只是配置项 + 重启生效、没有运行时迁移，无可搬实现。

### 10.4 接口有、界面没有（4 个域，2026-11-08 实测两端）

迁移把**端点**搬齐了（`route_parity_test.py`：224 条、0 条未迁移），但「有端点」≠「界面上摸得到」。
下面 4 个域在 **Web `frontend/app/` 与桌面 `desktop/frontend/src/` 里都 0 次调用**（按域关键词 grep 核过，
不是按路径模式匹配猜的），而它们的定性**各不相同**——别一律当成「已完成」：

| 域 | 端点数 | 现在谁能到 | 定性 |
|---|---|---|---|
| `lora-train` | 16 | 只有 HTTP | **接口齐、入口缺**：参数表 / `settings.toml` / `preview`（只拼 argv 不跑）/ `train` 入队 / `tasks` / `cancel` / `dataset` 三工具全在，训练产物还有下游读（`services/training_registry.py`、`services/knowledge_feed.py`）⇒ 缺的纯粹是界面，能力一点不缺 |
| `repair` | 11 | 只有 HTTP | **底座等上层接**：`services/repair_*` 除 `routers/repair.py` 外**零引用**，Agent 侧也没有对应工具 ⇒ 连「谁产生提案」这一方都还没有，接了也没有用户 |
| `mcp` | 3 | 只有 HTTP | **运维口**：`/status` `/refresh` `/test` 是诊断外部 MCP server 用的，`agents.vue` 只在注释里提「MCP 工具运行时动态发现」⇒ 不适合做成日常界面，可接受 |
| `continuity` | 2 | HTTP **空着**，但能力另有入口 | **承诺未兑现**：`POST /continuity/check` 无前端调用，而 `app/services/continuity.py` 自己**有别的使用方**（`agent/tools/storyboard_tools.py`、`routers/preflight.py`）⇒ 能力没丢；但该模块头写的「可在分镜编辑时实时查」目前没有界面实现，**别读成已接好** |

为什么容易误判：拿「前端路径模式 vs `app.openapi()["paths"]`」做覆盖统计会**大量漏报前端调用**
（实测把 `evaluation` / `export` / `visual-graph` / `usage` 四个真有调用的域误报成零调用——
前端那边是 `` `/evaluation/film/${episodeId}?record=…` `` 这种带插值与 query 的模板串，路径匹配器一遇到
`${…}` 与 `?` 就整条不认）。判断某个域有没有界面入口，只认**按域关键词在两个前端根里 grep + 逐条读调用点**：

```powershell
Get-ChildItem -Recurse frontend/app,desktop/frontend/src -Include *.ts,*.vue |
  Select-String -Pattern 'lora-train','repair/','mcp/status','continuity/check'
```

## 11. 维护规则（改功能时必须同步哪份文档）

| 你改了什么 | 必须同步 |
|---|---|
| 新增/删除一个后端模块 | 本文对应二级域 + 该层数字（总览表与节标题）+ `tests/structure_test.py::MODULE_GROUPS`（若在 engine） |
| 新增/删除一个 engine 模块 | `app/services/engine/__init__.py::MODULE_GROUPS` + 本文 L4 |
| 新增一个功能域（二级域装不下） | **先补分类体系**，再落功能（不许塞进「其他」） |
| 新增/改端点 | `docs/api-contract.md`，本文不用改（本文不写端点细节） |
| 新增/迁移一个端点**域** | 核一遍两个前端根有没有调用点；没有就登记进 §10.4（否则下一个人会以为「有 API 就有界面」） |
| 新增测试套件 | `tests/run_all.py::TESTS` |
| 新增测试要写临时文件/探针 | 落 `tempfile.mkdtemp()`（**系统临时目录**），别写进 `tests/`、也别写进仓库数据根：本仓常在 CodeBuddy 沙箱下跑，它把 `Path.unlink` 换成「移到回收站」且**只对系统临时目录旁路成原生删除** ⇒ 长跑偶发 `OSError: SHFileOperationW 失败: 0x2` / `PermissionError: WinError 32`，断言全过却整套显示红（2026-11-08 实测 4 套：`engine_audio_vae`／`engine_h3_form`／`comfyui_workflows`／`ocr_index`） |
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
