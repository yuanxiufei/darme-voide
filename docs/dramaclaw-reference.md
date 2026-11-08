# DramaClaw 功能参考说明（`reference/dramaclaw`）

> 采集 2026-11-08 ｜ 上游 `https://github.com/dramaclaw/dramaclaw`（产品名 **DramaClaw / SuperTale**，CE 版）
> HEAD `c7d9f0e`（2026-11-03，`fix: preserve FastAPI validation diagnostics in agent tools (#919)`），浅克隆 `--depth 50`
> 每条结论标注「实测」或「读码」，实测口径见 §8。

---

## 0. 一页结论

**它是什么**：本地优先的「AI 短剧工作台」，一个 FastAPI 后端 + 一个 React 前端，两条产品线共用同一份数据：

1. **Xia Director（剧集流水线）**：导入小说 → 角色/场景/道具资产 → 分集 → 剧本 → **节拍（beat）** → 身份/场景/道具方案 → 草图 → 视觉闸门 → 视频提示词 → I2V 出片 → 评分 → 合成导出。
2. **Freezone（无限画布）**：21 种节点、40+ 节点操作（重绘/扩图/打光/多视图/360 场景/首尾帧/视频续写/扩音轨/3DGS），**由智能体把一句话意图编译成一张完整节点图**，人审后一次性应用并批量跑。

**与本仓定位差异**

| 维度 | DramaClaw | darme-voide 现状 |
| --- | --- | --- |
| 推理执行 | 全走云端中转网关（NewAPI 兼容），本地不算力 | 自研 torch 引擎，本地 DiT/VAE/TE 常驻与换装 |
| 画布 | 一等公民：21 节点 + 意图编译器 + 运行编排 + 审批事务 | `canvas_ops.py` 21 工具 + `CanvasNodeCard.vue`，**无规划器、无运行编排、无审批事务** |
| 智能体 | 双后端（codex SDK / Hermes 插件）+ MCP 双服务器 + 技能包 | `canvas_agent.py` 单轮工具调用，无技能/配方层 |
| 质量 | 节拍驱动 + 草图闸门 + 失败族库回灌 | 剧本为时长真源 + `提示词库.py` 负向族 + `制作线.py` 校验器 |
| 数据 | 单机 SQLite + 项目目录 | SQLite + `data/static` 媒体根 |

**值得借鉴的 6 条（§5 展开）**

1. 意图 → 计划 → 命令三段式，编译器**确定性、不改用户原话、严格校验不猜**（实测一句话出 18 节点/35 边）。
2. 画布保存是 **CAS + 幂等**（`base_revision` + `client_save_id`），冲突实测返回 `409` 并带 `current_revision`。
3. **审批权留在宿主**：核心库「不做模型调用、不写画布、不批审批」，HTTP 与 MCP 共用同一份变换。
4. 智能体工具面**收敛**：home 作用域只给 4 个通用 REST 工具，画布作用域才放开 54 个细粒度工具（**但白名单与插件族同名绑定，错配会静默归零**，见 §7.6）。
5. 生成调度分 **4 条 lane**（default/video/world/ffmpeg）各带并发+排队上限，另有 per-project / per-user 配额。
6. 失败族是**一等数据**：`sketch_failure_modes` 表把「现象/检测/预防规则/修正模板/负向词条/fixture」入库并计命中数。

**不要抄的 4 条**

1. `routes/freezone.py` 单文件 700 KB、`routes/generation.py` 238 KB 的聚合写法。
2. 生成能力全靠外部网关 + 远程模型别名（`.env` 里 `DC-*-LLM`），本仓是自持引擎，成本与失败语义完全不同。
3. 「功能覆盖面」不可照搬（后端 23 万行、前端 42 万行），只取契约层。
4. 许可为 **Elastic License 2.0**（§7），代码不能整段搬进本仓再对外提供服务。

---

## 1. 运行环境、依赖与启动（除注明外均实测）

### 1.1 规模

| 项 | 数值 |
| --- | --- |
| 后端 Python | 968 文件 / 231,121 行（`src/`） |
| 前端 TS/TSX | 1,366 文件 / 417,118 行（`frontend/src`） |
| 测试 | 468 个 `test_*.py` |
| API | 27 个路由文件，OpenAPI 共 **374** 端点（画布 120、剧集 93） |
| 依赖锁 | `uv.lock` + `frontend/pnpm-lock.yaml` 双锁均已提交（复现的前提） |
| 许可 | `LICENSES/Elastic-2.0.txt` |

### 1.2 依赖

- Python `>=3.11,<3.13`（实测 3.12）：FastAPI + pydantic v2 + SQLAlchemy + pydantic-ai（`[anthropic,google,openai,openrouter]`）、cognee（剧本图谱）、litellm、`openai-codex`（git 直依赖）、ffmpeg 封装、sqlite。
- 前端：React 19 + Vite + TypeScript + **React Flow**（画布）+ Zustand + TanStack Query + Tailwind 4 + Three.js（3DGS）。
- 外部服务：**一个 OpenAI 兼容网关**（`NEWAPI_BASE_URL`，默认 `http://127.0.0.1:3000`；`docker-compose.yml` 的 `newapi` 服务来自另一仓 `dramaclaw-gateway`）。`.env.example` 把 `DC-codex-agent-LLM` / `DC-image-*` / `DC-video-*` 这类**别名**映射到具体模型——没有这层，出图/出片不可用（画布与编排仍可用）。

### 1.3 安装（Windows 实测）

```powershell
cd D:\code\voide\darme-voide\reference
git clone --depth 50 https://github.com/dramaclaw/dramaclaw.git dramaclaw
cd dramaclaw
uv python install 3.12
uv sync --frozen --group dev          # ★ 必须 --frozen，见下
cd frontend; pnpm install --frozen-lockfile; pnpm build:ce
```

**必须记的坑（实测）**：裸 `uv sync` 会重新解析依赖并失败——

```
× No solution found ... openai-codex-cli-bin==0.149.0 ... not found in registry
```

原因是 git 依赖 `openai-codex` 自己声明了上游已撤下的 `openai-codex-cli-bin==0.149.0`。`--frozen` 按仓库自带 `uv.lock` 安装可完全绕过解析，装完 `import openai_codex` 正常。

耗时参考：`uv sync` 4–6 分钟；`pnpm install` **49.5 s**（617 MB）；`pnpm build:ce` **16.67 s**。

### 1.4 三种启动方式

| 方式 | 命令 | 组件 | 端口 |
| --- | --- | --- | --- |
| 本地开发 | `uv run --no-sync novelvideo api` + `pnpm dev` | 后端 + Vite | 8780 / 5173 |
| 单进程（推荐自测） | 设 `DRAMACLAW_FRONTEND_DIST=frontend/dist` 后照常起 API | 后端托管 SPA | 8780 |
| Docker | `docker compose --profile ce up -d` | api / newapi / web | 8780 / 3000 / 8080 |

**单进程托管已实测通过**：设 `DRAMACLAW_FRONTEND_DIST` 后 `GET /` 返回 SPA；不设则 `/` 404、只有 `/api/v1/*`。一个进程、一个端口、一份 SQLite，是最省事的验收姿势。

最小环境变量（无模型 key 也能起；画布/编排可用，出图不可用）：

```powershell
$env:ST_EDITION='ce'
$env:NOVELVIDEO_DATA_ROOT='<repo>\ce-data'
$env:NOVELVIDEO_STATE_DIR="$env:NOVELVIDEO_DATA_ROOT\state"
$env:NOVELVIDEO_OUTPUT_DIR="$env:NOVELVIDEO_DATA_ROOT\output"
$env:NOVELVIDEO_RUNTIME_DIR="$env:NOVELVIDEO_DATA_ROOT\runtime"
$env:DRAMACLAW_CHAT_BACKEND='codex'
.\.venv\Scripts\python.exe -m uvicorn novelvideo.api.app:app --host 127.0.0.1 --port 8780
```

`GET /api/v1/config` 实测返回 `{"edition":"ce","auth_required":false,"mcp_direct_canvas_apply":false,...}`——最后一项即「MCP 不能直写画布、必须走审批」的开关，CE 默认关。

---

## 2. 架构总览

```
frontend/src/features/{canvas,chat,studio,...}/{domain,application,infrastructure,presentation}
        │  REST + SSE (/tasks/stream, /chat/*/stream, /chat/canvas-commands/stream)
        ▼
api/app.py  ── 27 route 模块 / 374 端点（薄：鉴权、schema、调度）
        ▼
services ── freezone（画布）｜chat（智能体）｜agents（流水线角色）｜
            verification（闸门/失败族）｜generators（模型适配）｜task_backend（任务）
        ▼
ports/ ── 19 个端口（project / asset / task / credit_quote / render_plan ...）
        ▼
adapters ── sqlite_store.py（单文件库）＋ 项目目录（文件系统）＋ NewAPI HTTP
```

四条可复述的原则（读码 + 文件头自述）：

1. **薄路由、厚服务**：路由只做参数校验与 202/SSE 转发。
2. **端口/适配器**：EE 与 CE 的差别收在端口实现里，同一份服务代码双形态部署。
3. **确定性核心 + 双宿主**：`workflow_transactions.py` 文件头原文「Host-neutral, deterministic … no model calls, canvas writes, or approval authority. HTTP and MCP adapters share the same transformations」。审批与落盘只在宿主（路由 / 前端）。
4. **重活全出进程**：请求只写队列，SSE 推状态；长任务在 worker，靠 lease + heartbeat 在重启后恢复。

---

## 3. 核心功能模块

### 3.1 Freezone 无限画布

| 事实 | 证据 |
| --- | --- |
| 21 种节点类型（文本/图像生成/图像编辑/视频/首尾帧/音频/合成/3DGS/HTML/分组…） | `frontend/src/features/canvas/domain/canvasNodes.ts::CANVAS_NODE_TYPES` |
| 画布域 120 个端点 | OpenAPI `/projects/{project}/freezone/*` |
| 40+ 节点操作（重绘/扩图/改光/多视图/360 场景/草图上色/视频续写/扩音轨…） | `.../freezone/nodes/{node_id}/{image,image-edit,outpaint,relight,multiview,scene-360,sketch}...` |
| 自动保存 = 全量文档 + 版本 + 历史 + 备份 | `freezone/canvas_store.py`（幂等键、`revision`、保留 100 版历史、写前备份） |
| 并发用**每画布互斥锁** + CAS | `freezone/canvas_lock.py`；`canvas_store.py` 的 `base_revision` 校验 |
| 批量应用是**事务**，失败回滚 | `freezone/workflow_transactions.py`（538 行） |
| 改上游只重算下游 | `freezone/stale.py`（32 行纯函数）＋ `mark_node.py`（节点级标记） |

要点：画布文档是**唯一真源**（节点里既有参数也有产物引用），所以「AI 改画布」与「人拖画布」走同一条保存与冲突通道，不需要第二套同步逻辑。

### 3.2 工作流意图编译器（本仓最该抄的一块）

三段式（权威契约在 `agent-kit/skills/dramaclaw-workflows/` 的 `SKILL.md` 与 `references/`）：

```
能力协商 → 意图(compact) → 计划(plan) → 命令(commands) → 人审 → 一次性批量应用 → run → 观察
workflow_capabilities  intent_compile   graph_compile    prepare_workflow_draft
                                                              ↓ confirm
                                                        apply（整批 CAS 写）
                                                              ↓
                                                     run_workflow → observe
```

- **严格校验、绝不猜**（`freezone/workflow_plan.py`）：节点/连线类型规则、端口方向、数据形状硬校验；推不出来就报错让人补。文件头原文：agent-authored graph 必须与契约逐字一致；编造节点、偷换 `link_type`、静默删连线一律拒。
- **编译器是纯函数**：同一 `(skill_version, plan)` → 同一 `(commands, preflight)`，可对拍、可回放。
- **preflight 先算账**：返回各类节点计数与可执行性，以及 `execution_policy`（`requires_user_confirmation: true`、`auto_create_nodes: false`）。
- **命令协议 `canvas_chat_commands.v1`**：`create_node / create_edge / layout / group / select` 等，带 `client_id` 做本地→服务端 id 对映；前端逐条执行，表现为「一次性长出一张图」。
- **审批 gate**：`prepare_workflow_draft` 产草稿（不落盘），`confirm` 才 apply；`workflow_transactions.py` 保证整批成功或整批不写。
- **运行观察**：`run_workflow` 返回 `run_id` + 观察句柄，`observe_workflow_run` 按 token 等下一次状态变化，并把失败**分类**（`invalid_request` / `auth` / `quota` / **`ambiguous` 传输超时**）；对 `ambiguous` 明确要求 `next_action = reconcile_before_retry`——先对账再重试，避免重复扣费（`workflow_observation.py`）。

### 3.3 智能体层（Xia Director）

| 事实 | 证据 |
| --- | --- |
| 双后端：`DRAMACLAW_CHAT_BACKEND=codex`（本地 codex SDK）或 `hermes`（仓内 `.hermes/plugins/{dramaclaw,freezone}`） | `chat/service.py::_chat_backend()` |
| 会话/线程持久化 + SSE 流式 + 中断 | `chat/` 32 文件；`/chat/.../stream`、`/chat/canvas-commands/stream` |
| 工具面按作用域收敛：home 只给 4 个通用 REST 工具，画布作用域给 54 个 `freezone_*` | `chat/dramaclaw_mcp.py::_agent_tools()`＋`HOME_TOOL_NAMES`（实测四组合：home+dramaclaw=4 / project+dramaclaw=45 / **home+freezone=0** / project+freezone=54，详见 §7.6） |
| MCP 双服务器：`tools`（通用 REST 代理）、`workflows`（只读＋编译，6 工具） | `agent-kit/scripts/launch_mcp.py::MODULES`；实测见 §4.3 |
| 外链 agent 走短时 token；`DRAMACLAW_LOCAL_AGENT_TRUST` 仅 loopback 允许，非 loopback 直接拒起 | `launch_mcp.py:73-86` |

要点：**tool 爆炸是智能体退化的主因**。它的做法是「一域一插件 + 作用域白名单 + 通用 REST 工具兜底」，细粒度工具只在画布域、只在画布会话可见。

### 3.4 技能（Skill）与配方（Recipe）目录

- 内置 **12 个技能**（`short-drama-quick` / `novel-adaptation` / `comic-drama-batch` / `story-video-expansion` / `remix-video` / `ecommerce-ad` / `japanese-anime-short` …）＋ **60 个配方**，纯 JSON，落 `src/novelvideo/freezone/agent_catalog/builtins/{skills,recipes}/`。
- 技能＝契约：`required_inputs`（缺了编译失败；实测缺 `inputs.visual_style` 被当场拒）、`expansion_rules`（默认节点数/时长/画幅）、`required_roles`、`version`。
- 配方＝单节点**结构化提示词模板**：`must_have_items`（缺项即判不完整）、`output_kind`、`action_keys`、`default_skill`。例 `drama-shot-planning.json` 把「机位/景别/主体/光位/时长」做成必填清单。
- 二者都**版本化**（`skill.version` 参与编译与幂等），并由 `workflow_skill_get` 原样吐给智能体——即「提示词资产」与「代码」分开演进。

### 3.5 系列流水线（beats 驱动）

`api/routes/episodes.py` 93 端点，暴露完整阶段机。核心事实：**原子单位是 beat（节拍）**：

```
导入(ingest) → 角色/场景/道具资产 → 分集(plan-episodes) → 剧本(script)
  → beat.adapted_content（改编文本）
  → identity_plan / scene_plan / prop_plan（谁出现、在哪、用什么；引用资产 id）
  → sketch（草图/首帧候选）→ sketch 视觉闸门（不过则带失败族重画）
  → video_prompt → video（I2V）→ score（评分挑选）
  → audio/voice → compose → export（srt / zip / 成片）
```

关键端点族（`/projects/{project}/episodes/...`）：`plan-episodes`、`generate-script`、`plan-identities|scenes|props`、`sketches`、`sketches/{id}/verify|failure-modes|candidates|regenerate`、`video-prompts`、`generate-video`、`score`、`compose`、`export`、`stitch/srt`、`episode-auto`。

要点：

- **文本先于画面**：beat 改编文本是唯一真源，身份/场景/道具方案只引用**资产库 id**，所以资产改一次全片生效。
- **草图闸门**：草图真值来自模型判图，不过就带失败族重画，不靠人肉挑。
- **资产↔剧集双向反查**（如 `characters/{id}/episodes`）：长剧一致性的基础能力。

### 3.6 生成适配层与任务后端

| 事实 | 数值/证据 |
| --- | --- |
| 任务 lane | 4 条：`default`(12/8)、`video`(4/2)、`world`(2/1)、`ffmpeg`(2/1)（并发/排队）——`task_backend/limits.py` |
| 配额 | per-project / per-user 两档 |
| 状态推送 | SSE：`/tasks/stream`、`/tasks/{type}/{episode}/stream`；事件 `task_updated / heartbeat / deleted / auth_revoked` |
| 任务信封 | HMAC 签名，密钥首启生成于 `STATE_DIR/task_envelope_keyring.json`（`task_backend/signing.py`、`envelope.py`）——防 SSE 通道里的任务事件被伪造/串号 |
| 恢复 | run 状态机 `running/completed/failed/cancelled/interrupted`，节点态 `pending/running/completed/failed/blocked/skipped`，lease + 心跳可续跑（`workflow_runs.py`） |
| 计费 | `ports/credit_quote.py`：先报价后提交，报价可审计 |

要点：**并发上限按"重活类型"分档**而不是全局一个数——视频 4 路、合成 2 路，长任务不会饿死短任务。

### 3.7 一致性 / 质量闸门

- `verification/` 41 文件：视觉闸门（VLM 判图）、一致性校验、版本哈希、训练库、回放抓取。
- **失败族是一张表**（`sqlite_store.py` 的 `sketch_failure_modes`），字段：`layer / detection / prevention_rule / correction_template / negative_prompt_clause / gate_enabled / fixture_path / hit_count`。命中即 `hit_count += 1`，闸门按 `gate_enabled` 决定是否阻断，负向词条直接进生成参数。
  → 与本仓 `片场/提示词库.py` 的负向族（`NEG_BOARD`/`NEG_PROP`/`CLASH_SETTLED`）同思路，但它多了「修正模板 + fixture + 命中计数」三样。
- **回灌**：判图不过 → 记失败族 → 按「修正模板」重写提示词 → 重画；成功样本进训练库。

### 3.8 交付与其他

- 合成走 ffmpeg（`ffmpeg` lane），产出 `srt` / `zip`（分镜脚本包）/ 成片 mp4，另有 `episode/export` 打包端点。
- 3DGS `world/`（高斯泼溅）与 `html-artifacts`（节点内跑 HTML/JS）与本仓无交集，本次未深挖。
- `mainline projection`：画布产物与「剧集主线」互投（回填资产/分镜）——思路值得单评（§5-B4）。

---

## 4. 关键接口与数据流转

### 4.1 画布自动保存契约（实测三次调用）

```
PUT /api/v1/projects/{project_id}/freezone/canvases/{canvas_id}
{ schema_version:2, canvas_id, project_id, canvas_scope:"default",
  client_save_id:"save-1", save_source:"manual_save", base_revision:0,
  nodes:[...], edges:[...] }
```

| 场景 | 实测结果 |
| --- | --- |
| 首次保存（`base_revision:0`） | `200`，返回 `revision:1` |
| 同 `client_save_id` 重放 | `200` 幂等，版本不涨 |
| 换 `client_save_id` 但仍 `base_revision:0` | **`409 canvas_revision_conflict`**，响应带 `current_revision` |
| `GET .../revision` | 当前版本号（多端协作/轮询锚点） |
| `GET .../canvases` | 列表带 `revision` 与更新时间 |

→ 三条可直接复用的规则：**全量文档上传、版本号做乐观锁、客户端生成幂等键**。

### 4.2 意图 → 命令（MCP `workflows` 实测，无任何模型 key）

输入（一句话 + 技能 + 风格）：

```json
{ "schema_version":"freezone_workflow_intent.v1", "skill_id":"short-drama-quick",
  "user_goal":"一名女侠在雨夜走进客栈，看到柜台后的老人抬头，气氛肃杀",
  "inputs":{"visual_style":"水墨写实古风"},
  "planner":{"mode":"standard","item_count":2},
  "include_compose":true, "include_audio":true }
```

输出：

```
status: draft      nodes: 18      edges: 35
preflight: 7 text / 5 image / 2 video / 3 audio / 1 compose
expansion_rules.item_count: 16
execution_policy: { requires_user_confirmation: true, auto_create_nodes: false }
```

即**一句话 → 18 节点 35 连线的可执行图，停在草稿态等人点头**。这正是本仓画布最缺的一环（现在是模型逐条调工具建节点）。

### 4.3 MCP 服务器（实测）

| 模式 | 模块 | 工具面 |
| --- | --- | --- |
| `workflows` | `novelvideo.chat.workflow_mcp` | **6 个**：`workflow_catalog_search` / `workflow_skill_get` / `workflow_graph_compile` / `workflow_intent_compile` / 草稿准备 / 确认（只读＋编译，无落盘权） |
| `tools` | `novelvideo.chat.dramaclaw_mcp` | 按作用域：home = 4 个通用（`dramaclaw_get/post/patch/delete`）；canvas = 54 个 `freezone_*`——**前提是 `DRAMACLAW_PROJECT_ID` 非空**，否则静默返回 0 个（§7.6） |

`workflow_graph_compile` 传缺 `inputs.visual_style` 的计划会**当场拒**并指明缺哪个必填输入——「严格编译器」最直观的证据。

`tools` 模式的最小可用 env（实测）：

```powershell
$env:DRAMACLAW_CE_DIR      = "D:\code\voide\darme-voide\reference\dramaclaw"
$env:DRAMACLAW_API_URL     = "http://127.0.0.1:8780"
$env:DRAMACLAW_PROJECT_ID  = "<project_id>"   # ← 不给则 tools/list = []
# 可选：$env:DRAMACLAW_CANVAS_ID = "<canvas_id>"（画布域动作要它）
.\.venv\Scripts\python.exe agent-kit\scripts\launch_mcp.py tools
```

### 4.4 端点热区（复用价值最高）

| 域 | 端点数 | 代表端点 |
| --- | --- | --- |
| 画布 `freezone` | 120 | `PUT canvases/{id}`、`.../workflow-drafts`、`.../workflow-intent`、`.../nodes/{id}/run-action`、`.../nodes/{id}/image-edit`、`.../runs/{id}/observe` |
| 剧集 `episodes` | 93 | `plan-episodes`、`generate-script`、`plan-identities`、`sketches`、`sketches/{id}/verify`、`video-prompts`、`generate-video`、`compose`、`export` |
| 任务 / 聊天 | 20+ | `/tasks/stream`、`/chat/.../stream`、`/chat/canvas-commands/stream` |
| 资产 `characters/scenes/props` | 40+ | 资产 CRUD + `/{asset}/episodes` 反查 + `generate-*` |

---

## 5. 可借鉴的设计要点（对齐本仓落点）

### A. 语义可直接照搬（不搬代码）

| # | 做法 | 本仓落点 | 收益 |
| --- | --- | --- | --- |
| A1 | 意图 → 计划 → 命令三段式 + 确定性编译器 + preflight 计数 | 新增 `backend-py/app/services/canvas_plan.py`，接在 `canvas_agent.py` 之后 | 智能体从「逐条工具调用」变成「一次出整图」，可渲染、可审、可回放 |
| A2 | 草稿 / 确认 / 一次性 apply + 事务回滚 | `routers/canvas.py` 加 `/ops/apply` 与 `expected_revision` | 消除「AI 改到一半」的中间态 |
| A3 | 画布 CAS + 幂等保存 | 前端 `useCanvas` 的 persist + 后端快照接口 | 多端/刷新/重试不丢稿（本仓已有刷新丢档事故） |
| A4 | 失败分类 `ambiguous → reconcile_before_retry` | `engine` 生成重试路径 | 超时≠失败，避免重复扣费/重复占卡 |
| A5 | 并发 lane 分档（default/video/world/ffmpeg） | `engine/scheduler` | 长任务不饿死短任务 |
| A6 | 失败族表带 `correction_template` + `hit_count` | `片场/提示词库.py` 负向族已有一半，缺「修正模板/命中计数」 | 哪条负向真在起作用一目了然 |

### B. 需要改造再引入

| # | 做法 | 改造点 |
| --- | --- | --- |
| B1 | 技能 + 配方 JSON 目录 | 本仓"配方"其实是 `提示词库.py` 模板 + `01/02/03` 工作流 JSON；可加一层 recipe（必填项清单 + 输出契约），但**真源仍在 `提示词库.py`**，不能二次分叉 |
| B2 | 运行编排（节点态 + lease + 恢复） | 本仓画布是"按节点单独生成"，无 run 概念；引入前先定节点依赖与失效传播（对应它的 `stale.py`） |
| B3 | beat 作为流水线原子 | 本仓是「剧本→分镜→每镜 5 秒→视频」，改原子会动剧本真源，**与「永不降剧本质量」铁律冲突**；只借"beat 文案派生资产引用"的思想 |
| B4 | 主线投影（画布 ↔ 剧集互投） | 需先有稳定的资产↔画布节点映射表，风险中等 |
| B5 | 视觉闸门 VLM 判图 + 回灌 | 本仓 `片场/看图.py` 已是模型亲自判图；可加"结论落库 + 命中计数"，但**判定权仍归模型**，不要外包给脚本 |
| B6 | 工具面按场景收敛（home 白名单在小集合、画布才放开全量） | 本仓 `canvas_ops.py` 的 34→21 工具可加场景白名单；**两条硬约束**：白名单键必须与被过滤集合同族命名、交集为空必须报错——否则重演 §7.6「静默返回 0 个工具」的坑 |

### C. 明确不抄

1. 单文件 700 KB 的路由聚合写法。
2. 生成侧完全依赖外部网关（`.env` 里那套 `DC-*-LLM` 别名无法映射到本仓引擎）。
3. 23 万行/42 万行的功能面——只取契约层三件（plan / commands / approval）。
4. Elastic-2.0 代码整段搬运后对外提供服务（§7）。

---

## 6. 本地复现与二次开发配置

```powershell
# 1) 后端（已实测可起）
cd D:\code\voide\darme-voide\reference\dramaclaw
uv sync --frozen --group dev
$env:ST_EDITION='ce'
$env:NOVELVIDEO_DATA_ROOT="$PWD\ce-data"

# 2) 单进程（自带前端）
$env:DRAMACLAW_FRONTEND_DIST="$PWD\frontend\dist"
.\.venv\Scripts\python.exe -m uvicorn novelvideo.api.app:app --host 127.0.0.1 --port 8780
# 浏览器打开 http://127.0.0.1:8780

# 3) 只给外部 agent 用（不需要前端）
.\.venv\Scripts\python.exe agent-kit\scripts\launch_mcp.py workflows   # stdio MCP
# 或让 Cursor / Claude Code 指向 .mcp.json
```

**数据落点**（`NOVELVIDEO_DATA_ROOT` 下）：`state/`（SQLite、信封密钥、会话）、`output/`（图/片/音）、`runtime/`（临时）。备份只需拷 `state/` + `output/`。

**要出图/出片**还须补：① 一个 OpenAI 兼容网关（`NEWAPI_BASE_URL` + key）；② `.env` 里把 `DC-image-*` / `DC-video-*` / `DC-codex-agent-LLM` 等别名绑到真实模型；③ `ffmpeg` 在 PATH（合成/抽帧）。这三样本次未配，故「生成」链路只做了编译层与保存层的验证。

**二次开发切入点**（按性价比排序）：

1. 只读 `agent-kit/skills/dramaclaw-workflows/`（SKILL.md + 2 份 reference）——约 300 行，把「意图/计划/命令/审批/观察」五件事讲透，可直接当本仓画布规划器的规格书。
2. 参考 `freezone/workflow_plan.py`（严格校验）与 `workflow_transactions.py`（批量事务）的实现方式，不要整文件搬。
3. 用 `agent-kit/scripts/launch_mcp.py workflows` 起 MCP，让外部 agent 直接调 `workflow_intent_compile` 看实际命令 JSON——比读代码快。

---

## 7. 注意事项

1. **许可**：`Elastic License 2.0`。可以阅读、学习、内部改造；**不得把本仓功能作为托管服务对外提供**，也不得移除许可与商标声明。若要复用代码片段，须登记到 `THIRD_PARTY_NOTICES.md` 并保留来源；纯设计借鉴不受此限。
2. **依赖漂移**：`openai-codex` 的 git 依赖链已在上游断供（`openai-codex-cli-bin==0.149.0`），**只能 `uv sync --frozen`**；哪天 `uv.lock` 重置就会装不上。
3. **无 key 也可用的部分**：配置、画布 CRUD、工作流编译/草稿、任务查询、MCP `workflows`。「出图/出片/剧本生成」必须要网关。
4. **仓库体积**：克隆后含 `.venv`(≈700 MB) + `node_modules`(617 MB) + `dist`，实测 **2.5 GB**；`reference/` 已被本仓 `.gitignore` 忽略，不会进版本库。
5. **单机假设**：CE 版数据全在本机 SQLite + 目录，**没有多租户隔离**（`auth_required=false`）。别把它当多用户服务暴露到公网。
6. **`tools` 模式 MCP 的静默归零（已查清，是配置陷阱不是代码 bug，实测坐实）**：
   - 现象：只设 `DRAMACLAW_CE_DIR` 按 README 最小写法起 `launch_mcp.py tools`，`initialize` 正常（`dramaclaw v0.1.0`），但 `tools/list` 返回 **0 个工具且无任何报错**。
   - 机理（两处叠加）：① `launch_mcp.py::_configure_tools_env()` 用 `setdefault` 把 `DRAMACLAW_CHAT_SURFACE=freezone` / `DRAMACLAW_TOOL_MODE=freezone_canvas` / `DRAMACLAW_AGENT_PROFILE=freezone:main` 全部兜底写上 ⇒ `_freezone_canvas_mode()` 恒真 ⇒ `_agent_plugin_name()` 恒为 `freezone`；② `_scope_kind()` 只看 `DRAMACLAW_PROJECT_ID` 是否为空，为空即 `home`，而 `_agent_tools()` 在 home 下取 `HOME_TOOL_NAMES`（`dramaclaw_get/post/patch/delete`）与插件工具取交集。这 4 个名字**只存在于 `dramaclaw` 插件**，`freezone` 插件全是 `freezone_*` ⇒ 交集为空。
   - 实测四组合（`_agent_tools()` 直接量）：`home+dramaclaw=4`、`project+dramaclaw=45`、**`home+freezone=0`**、`project+freezone=54`；stdio 握手同证：只给 `CE_DIR` → 0，给 `PROJECT_ID` → 54（`CANVAS_ID` 有无都 54，只影响画布域动作）。
   - 规避：`tools` 模式**必须**带 `DRAMACLAW_PROJECT_ID`（`render_config.py` 的 `--project-id` **默认空串**，这是官方模板自身的坑）；`workflows` 模式不受影响（无作用域过滤，实测恒 6 工具）。
   - 借鉴价值：这正是本仓 `canvas_ops` 工具面若做「按场景收敛」时必须防的坏法——**白名单的键空间必须与被过滤集合同一命名族，且交集为空要报错不能静默返回空表**。
7. **其余未验证项**（本次只到编译/保存/托管层）：真实出图与出片、`hermes` 后端、3DGS world、`episode-auto` 整集自动跑。
8. **前端契约同源**：`frontend/src/features/canvas/domain/canvasNodes.ts` 与服务端 `workflow_schema.py` 的节点/连线类型必须同步改，否则编译出来的命令前端不认——本仓改画布类型时同样要守住「一处定义、两处引用」。
9. **代码质量参差**：仓内自述 "vibe-coded"，巨型文件多、抽象层次不齐；**只取契约与状态机，不要照着它的文件布局抄**。

---

## 8. 验证矩阵（本次实际执行）

| # | 动作 | 结果 |
| --- | --- | --- |
| 1 | `git clone --depth 50` → `reference/dramaclaw` | ✓ HEAD `c7d9f0e` |
| 2 | `uv python install 3.12` + `uv sync --frozen --group dev` | ✓（裸 `uv sync` ✗：`openai-codex-cli-bin==0.149.0` 缺失） |
| 3 | `import fastapi / pydantic_ai / cognee / openai_codex / novelvideo` | ✓ |
| 4 | `pnpm install --frozen-lockfile` | ✓ 49.5 s |
| 5 | `pnpm build:ce` | ✓ 16.67 s，产出 `frontend/dist` |
| 6 | 起 API 8780 → `GET /api/v1/config` | ✓ `edition=ce`、`mcp_direct_canvas_apply=false` |
| 7 | `POST /api/v1/projects` 建 `ref_demo` | ✓ 返回 `id`（ULID）——注意后续端点要用 **id** 而非 name |
| 8 | `PUT .../freezone/canvases/canvas_demo`（2 节点 1 边） | ✓ `revision:1` |
| 9 | 同 `client_save_id` 重放 / 旧 `base_revision` 重存 | ✓ 幂等 200 / ✓ 409 `canvas_revision_conflict` |
| 10 | `GET .../revision`、`GET .../canvases` | ✓ |
| 11 | MCP `workflows` stdio：`initialize` + `tools/list` | ✓ 6 工具 |
| 12 | `workflow_graph_compile`（缺 `inputs.visual_style`） | ✓ 被拒并指明缺项 |
| 13 | `workflow_graph_compile`（补齐后） | ✓ 产出 `canvas_chat_commands.v1` 命令 + layout/group/select |
| 14 | `workflow_intent_compile`（1 句中文） | ✓ 18 节点 / 35 边 / preflight 7·5·2·3·1 / 草稿态待确认 |
| 15 | 设 `DRAMACLAW_FRONTEND_DIST` 起 8781 | ✓ `GET /` 返回 SPA |
| 16 | MCP `tools` stdio：只给 `DRAMACLAW_CE_DIR`（README 最小写法） | ⚠ `initialize` 正常但 `tools/list` = **0 个工具**，无报错 |
| 17 | 同上 + `DRAMACLAW_PROJECT_ID` | ✓ **54 个 `freezone_*`** |
| 18 | 同上 + `DRAMACLAW_PROJECT_ID` + `DRAMACLAW_CANVAS_ID` | ✓ 54（`CANVAS_ID` 不影响工具数） |
| 19 | `_agent_tools()` 四组合直量（scope × plugin） | ✓ `home+dramaclaw=4` / `project+dramaclaw=45` / `home+freezone=0` / `project+freezone=54`——根因坐实，见 §7.6 |

> 环境：Windows 11 + PowerShell ｜ Python 3.12（uv 管理）｜ Node 22 + pnpm 11.14 ｜ 未配置模型网关，故未做真实出图出片。

