# backend-py —— Python 后端（绞杀者迁移）

> **这是什么**：Node 后端（`backend/`，Hono + Mastra）的 Python 替代品，**按域逐个迁移**，
> 迁移期间两个后端并存、共用同一个 SQLite 文件与前端契约。
>
> **目标：把 `backend/` 全量迁完后删除它**。⇒ 这已不是「翻译」而是**重写**：剩余部分要求
> 重建 **Mastra Agent 循环、17 家厂商适配器、ffmpeg 媒体链**（即最初评估里 50–80 人日的部分）。
>
> **现状**：已迁移 **34 个域 / 173 个端点**（口径由 `tests/route_parity_test.py` 机械扫描得出，
> 不是手数）—— `dramas`(9)、`episodes`(8)、`characters`(4, 含 `merge` 的资产版本重编号)、
> `scenes`(4)、`props`(5)、`storyboards`(5, 含台词匹配与 TTS 过期检测)、
> 四个资源库 `character/scene/weapon/costume-library`(10+11+10+10，**规格驱动共享实现**)、
> `presets`(4)、`app-settings`(2)、`asset-versions`(2, 含回滚写回主表)、`traces`(3, **只读侧**)、
> `storage`(**2/2** 整域：info + change)、`usage`(**3/3**: 汇总 / 成本看板 / **生成前费用预估**)、
> `agent-configs`(**5/5** 整域，含 defaults/generate)、`style-profiles`(**7/7** 整域，含 LLM 提炼 distill)、
> `generations`(1, 双表聚合)、**`ai-configs`(**17/17**: 含 Ollama 启停/拉取/删除、模型列举、连通性探测、
> 本地运行时健康)** + `ai-providers`(1)、
> **`skills`(6, 整域迁移: 含 SKILL.md 解析 / 默认绑定 / 删除保护)**、`upload`(3)、
> `export`(**7/7** 整域: 工程账本 JSON/MD + 断点续作 stale + EDL/ZIP)。
> **自检 2473 项全绿**（冒烟 476 + 适配器 101 + 错误归因 58 + 文本生成 67 + 图片生成 64 + 视频生成 50 + TTS/音色复刻 34 + 分镜 prompt/图谱 38 + 宫格 prompt/运镜 48 + 逐镜路由/videos 43 + 单镜合成 35 + 整集拼接 29 + 宫格路由 42 + **图谱/图片/回调 37** + **AI 音色 43** + **前端调用覆盖 5** + **契约镜像 5**）
> **+ 路径守卫 0 遮蔽 + 镜像常量 0 漂移**（含 `prompt_utils` 词表、适配器注册表与文案、
> `text-generation` 的 9 个提示词常量与 8 张词表、**视觉图谱 41 节点逐条**、
> 全仓 `json.dumps` 紧凑性的机械比对）。
> ⚠️ 2026-09-15 起 **Node 后端已删除、未注册端点 0 条** ⇒ 不再有「反代兜底」这回事；
> 未实现的路径一律 **501 + 说明**（接缝保留只为把「没实现」说清楚）。

## 全量迁移路线（S1–S7，**已完成：`backend/` 已删**）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **S1 地基** | `cost_catalog`(定价) / `estimate_service`(费用预估) / `task_logger`+trace **写入侧** / `ai_providers`(厂商配置解析) / `protocol`(Agent 输出协议) | ✅ **完成** |
| S2 提示词资产 | `prompt-utils.ts`(**1629 行**: 词表/模板/解析)，**分 6 块** | ✅ **6/6 块完成**（全量迁完）：①画风层+负面词层 ②角色/装备/单品/表情/物品/场景 ③分镜静帧/视频 ④DB 上下文辅助 ⑤台词校验/标签剥离 ⑥宫格 prompt；另含 `camera-movement-guides.ts` |
| S3 厂商适配器 | `registry` + `types` + 16 个适配器(1611 行) | ✅ **完成**（17 家 / 纯函数，`adapters_test.py` 101 用例 + 守卫覆盖） |
| S4 媒体服务 | image/video/TTS 生成、`compose`(ffmpeg)、宫格、合并、视觉图 | ✅ **主链全通**：`vendor-errors` + text/image/video/TTS+voice-clone 四条链路 + 逐镜路由 + `videos`(6) + `compose`(3) + `merge`(2) + **`grid`(4)**；仅剩**像素处理(校色/参考图压缩)**与**镜头 QC 打分**（调用点均已占位） |
| S5 **Mastra 替换** | Agent 循环 + 工具调用 + 协议 + **运行时** + 6 组工具(~2900 行) + `agent`(2) | ✅ **完成**：`protocol` + `tool` + **6 组工具** + **`runtime`(运行时：失败分类/退避/模型 fallback/风格注入)** + `agent`(2 端点，**非流式**)；✅ **`DEFAULT_PROMPTS` 已搬**（`services/agent_prompts.py` + 逐字守卫，2026-09-12 决策变更）；⚠️ 未迁 `subagent`/`rhythm-phase`（`skills`/`mcp` 已迁）；⚠️ **Gemini 函数调用循环未支持**（显式报错） |
| S6 编排/长任务 | `auto-pipeline`、`local-model-scan`、`evaluation`、崩溃恢复 | ✅ **全部迁完（剩 0 条）**：MCP（`agents/mcp.ts` 285 行自写 JSON-RPC 客户端 + 3 端点）｜`auto-pipeline` 整域关闭（8 阶段编排 + SSE）｜`evaluation` 整域关闭（types/catalog/scorer/evaluator/optimizer/scheduler + **5/5 端点**）｜GPU 显存管理 + 租约接线｜崩溃恢复（项目台账 / 资产版本 / SSE 总线 / 提取尾帧）|
| S7 收尾 | 剩余 AI 端点 + 全量回归等价验证 + **删 `backend/`** | ✅ **全部完成 —— 含「删 `backend/`」本身（2026-09-15 真删并复核）**：删后守卫自动回退快照，结论与删前**逐字一致**（Node 224 / Python 227 / 未注册 0 / 0 漂移）｜benchmarks 4 个 case JSON 已搬（逐字节一致、`catalog`/`optimizer` 默认路径已切）｜评测 CLI 已迁｜GPU 显存租约已迁（`/ai-configs/gpu/*`）｜**未注册 0 条**（绞杀者已收口：`storage/change` 是最后一条，2026-09-15 迁完）｜`app/` 对 `backend/` 的**路径依赖为 0**｜对拍工具与差分比较器就绪（**2026-09-15 实测：一致 12 / 已知差异 0 / 不存在路径 4 / 新差异 0**，CASES 已补齐 S7 新增只读 GET）｜守卫快照已重冻为 **Python 模块快照**（74 条 / 625 KB，`tests/frozen_ts_source.py`），且「真源码 vs 冻结」结论一致 ⇒ 删库后九道守卫价值保留 |

顺序是按**依赖**排的：S1 是所有适配器/Agent 的入口，S2/S3 被 S4/S5 依赖，S5 被 S6 依赖。
✅ **`backend/` 已于 S7 删除（2026-09-15）** —— 删前已证明等价：Node 224 条路径全量对照 +
差分对拍 **0 新差异**；删后守卫自动回退快照，结论逐字一致。

## 为什么不一次性重写

原始体量实测：**133 个 TS 文件 / 27,389 行 / 226 个端点 / 29 张表**，且**零自动化测试**。
一次性重写的问题不是「写不完」，而是**中途没有任何可验证的中间态** ——
4~6 周后才能第一次真正跑通，期间风险全部堆积在最后一刻。

绞杀者模式下每一天都有可跑、可回退的产物：

```
浏览器 ──► FastAPI :5790 ────► 全部域：Python 直接服务（34 域 / 227 条路径）
                            └─► 未实现的路径：501 + 说明（⚠️ 不再是反代）
```

迁移期每迁完一个域，就在 `app/main.py` 里 `include_router(...)` 一行，再把该域从 Node 侧停用；
**现在已全部迁完**（Node 侧 224 条路径 100% 覆盖），Python 独占 `5790` 即可。

## 删 `backend/` 前的等价性对拍（S7 第 5 步）

> ⚠️ **本节已进入历史**：`backend/` 已于 2026-09-15 删除 ⇒ 这里的两侧并排对拍**不可能再跑**
> （`tests/parity_run.py` / `parity_diff.py` 保留为「当时如何证明等价」的证据）。
> 最后一次对拍（删除当天）结论：**一致 12 ｜ 已知差异 0 ｜ 不存在路径 4 ｜ 新差异 0** ✓。
> 日常回归由 `route_parity_test.py` 的**快照模式**继续保等价性，不需要 Node。

     两侧**并排起**（Node 5789 / Python 5790），**指向同一份数据**，然后跑差分对拍。
     ⚠️ **以下为历史命令**（Node 侧已删 ⇒ 现在跑不了，仅供回看当时怎么做）：

    ```bash
    cd backend; $env:DATA_ROOT='<空目录>'; npx tsx src/index.ts        # ① Node(5789)
    cd ../backend-py; $env:DATA_ROOT='<同一目录>'; .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790   # ② Python(5790)
    .venv\Scripts\python.exe tests/parity_diff.py --report ..\tmp\parity.json    # ③ 逐字段对拍（退出码 1 = 有新差异）

    **2026-09-15 实测：一致 12 ｜ 已知差异 0 ｜ 不存在路径 4 ｜ 新差异 0**（覆盖 16 条只读路径逐字段等价；CASES 已补齐 S7 新增的 `dramas/*/rhythm` 与 `export/dramas/*/qc-report?format=json`）。
    ⚠️ MISSING 类 = 「两边都不该有」的路径（Node 404 未匹配 / Python 501 兜底），**Node 回 200 才算 new**。
    ```

    ⚠️ **`<项目根>/.data-root` 标记文件优先级高于 `DATA_ROOT` 环境变量**（Node 侧）⇒ 有它就会读到别处的库、
    对拍结果无意义；跑之前先确认它不存在（本仓当前**不存在**）。⚠️ 只比对**只读端点**（表里全是 GET），
    因为两侧启动时都会写库。

## 删 `backend/` 之后：守卫靠**冻结快照**继续工作（S7 第 6 步）

九道漂移守卫原先是**读 TS 源码**来证明「Python 的路由表/常量/提示词没漂移」。删库前先冻结：

```bash
python tests/freeze_ts_snapshot.py          # 把**守卫与自检**读到的 76 个 .ts **逐字**收进 tests/frozen_ts_source.py（674 KB；清单 = 手写 ∪ 自动发现**两种形态**）。⚠️ 删 `backend/` 后**照样能跑**：取源顺序是「真源码 → 现有快照（逐字保留删库前现场）→ git 历史」，可用来扩快照
python tests/freeze_ts_snapshot.py --check  # 校验覆盖完整性（真源码还在时会逐个核对）
python tests/freeze_ts_snapshot.py --dump tmp/frozen_ts   # 物化成 .ts 文件树，仅供人读
python tests/route_parity_test.py           # 照常跑
PARITY_USE_FROZEN=1 python tests/route_parity_test.py   # 强制用快照（验证「删库后照样能跑」）
```

守卫里的源码根是 ``_SRC_ROOT``：**真源码优先，缺失自动回退快照**。已实测：真源码与快照两条路径
结论**完全一致**（2026-09-15 实测：Node 224 / Python 227 / 未注册 0 / 0 漂移；由 `tests/freeze_snapshot_test.py` 守着「快照覆盖每个守卫读文件」）⇒ 删库后守卫价值完整保留。

## 运行

```bash
cd backend-py

# 1) 建虚拟环境（首次）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux

# 2) 启动（默认 5790；**现在它是唯一后端**）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5790

# 3)（可选）把未实现的路径转发给**别的**上游 —— Node 已删除，只在你自建上游时才有意义。
#    ⚠️ 保持 0（默认）时未实现路径回 501 并说明；设 1 但没有上游 ⇒ 只会 502。
set PROXY_TO_NODE=1
.venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
```

自检（**唯一回归安全网，改完代码必跑**）：

```bash
.venv\Scripts\python.exe tests\smoke_test.py      # 退出码 0 = 全过
```

**进程级冒烟（可选，改动启动/静态站/兜底文案时值得跑一次）**：`smoke_test.py` 走的是
`TestClient`（不起真进程）⇒ 它验不到「真端口 + 真 HTTP + 真静态站」。手跑一次（**用临时数据根，别碰真库**）：

```bash
$env:DATA_ROOT=$env:TEMP\py-check; .venv\Scripts\python.exe -m uvicorn app.main:app --port 5799
# 另开一个终端：
curl http://127.0.0.1:5799/api/v1/health          # 200 + {"status","timestamp"}
curl -i http://127.0.0.1:5799/api/v1/episodes     # 501 + 「未实现（Node 后端已于 2026-09-15 删除）」
curl -i http://127.0.0.1:5799/static/nope.png     # 404
```

（2026-09-15 删库当天实测全通过：真进程启动 ✓ / health ✓ / dramas 信封 ✓ / 501 新文案 ✓ / 静态站 404 ✓ /
有前端产物时 `/` 直出 HTML ✓。）

它做三件事：把表/列定义与真实 `data/drama.db` 的 `PRAGMA table_info` 逐列比对（并与 **TS 快照**里的
`db/index.ts` 建表清单交叉印证 —— Node 已删 ⇒ 走 `frozen_ts_source.py`）、打真实接口核对响应信封与**错误文案逐字**、写操作全部落在
数据库副本上（真实库只读）。报告落在系统临时目录，路径在结尾打印。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PY_PORT` | `5790` | 本后端端口。**刻意不读 `config.yaml` 的 `server.port`** —— 那是 Node 时代的端口（5789），并存期读同一个必然抢占；Node 已删，保留这条是为了「端口来源确定」 |
| `PORT` | — | 兼容用，优先级低于 `PY_PORT` |
| `HOST` | `0.0.0.0` | |
| `CORS_ORIGINS` | 见 `app/config.py` | 逗号分隔 |
| `DATA_ROOT` | `configs/config.yaml` 的 `database.path` 所在目录 | 数据根目录（DB + static + traces） |
| `DB_PATH` / `STORAGE_PATH` | — | 仅在未显式指定 `DATA_ROOT` / `.data-root` 时生效（与 Node 同规则） |
| `CONFIG_PATH` | `configs/config.yaml` | |
| `PROXY_TO_NODE` | `0` | ⚠️ **遗留开关**（Node 已删 ⇒ 无对象）：`1` = 把未实现路径转发到 `NODE_BACKEND_URL`；`0` = 回 501 并说明。**建议保持 0** |
| `NODE_BACKEND_URL` | `http://127.0.0.1:5789` | 遗留反代目标（只有你自建上游时才用得上）|

## 目录结构

> **先分清楚两类东西**：`app/`（含 `tests/`、`scripts/`）是**代码**；`skills/`、`local_services/`、
> `configs/`、`data/` 是**内容资产 / 运行时状态**。
>
> 常被问的那一对 —— **`app/services/agents/`（Agent 运行时：协议 / 工具 / 循环，Python 代码）
> 与 `skills/`（SKILL.md 内容库）为什么分开放**：
>
> 1. **一个是被 `import` 的代码，一个是被**读写**的资产**：`skills/` 会被 `/api/v1/skills` 的
>    PUT/DELETE **真的改写**（前端「技能」页就是它的 UI，用户可自己加 skill、装外部技能库）；
> 2. **守卫与生命周期不同**：技能内容由 `scripts/check_skill_refs.py` + `.githooks/pre-commit`
>    按**资产**管；代码由 `tests/run_all.py` 按**行为契约**管，两者判据、跑法都不同；
> 3. **打包边界**：`.dockerignore` 写明「镜像里只需 `app/` 与 `skills/`」—— 两者都要进镜像，
>    但一个是**依赖树**、一个是**内容树**；
> 4. **路径只留一处权威**：`app/config.py::skills_dir()`（`SKILLS_DIR` 环境变量可覆盖）；
>    `services/skills.py` 与 `services/agents/skills.py` 现在都**转发**它
>    （2026-09-15 收口：此前三处各写一遍、靠注释互相提醒「必须一致」✗）。
>
> ⚠️ 反过来，`app/services/*.py` **刻意保持扁平**（59 个模块同层）：它是**迁移索引** ——
> 守卫按「TS 文件 → Python 模块」逐条比对常量（如 `services/technical-qc.ts` ↔
> `app.services.technical_qc`），重排目录会**成片打断这些映射** ⇒ 结构优化应落在
> 「文档 / 常量收口」上，**不要动这一层**。

```
backend-py/
├─ app/
│  ├─ config.py                  配置解析（与 backend/src/config.ts 逐项对齐）
│  ├─ db.py                      SQLite 连接（WAL + busy_timeout，对齐 connection.ts）
│  ├─ models.py                  29 张表的 SQLAlchemy Core 定义（对齐 db/schema.ts）
│  ├─ response.py                统一响应层 + 行→dict / 字段映射工具
│  ├─ request_utils.py           读请求体（对齐 Hono c.req.json() 的容错）
│  ├─ main.py                    FastAPI 入口：信封兜底 / 静态 Range / SPA / 反代接缝
│  ├─ routers/
│  │  ├─ dramas.py               ✅ 9 端点（返回 snake_case）
│  │  ├─ episodes.py             ✅ 8 端点（返回 snake_case）
│  │  ├─ characters.py           ✅ 4 端点（返回 camelCase；含 merge）
│  │  ├─ scenes.py               ✅ 4 端点（返回 camelCase）
│  │  ├─ props.py                ✅ 5 端点（返回 camelCase；customPrompt 例外）
│  │  ├─ storyboards.py          ✅ 5 端点（创建/更新/删/台词校验/QC 读取）
│  │  ├─ libraries.py            ✅ 四个资源库的**规格声明**（41 端点由此生成）
│  │  ├─ presets.py              ✅ 4 端点
│  │  ├─ app_settings.py         ✅ 2 端点（⚠️ 内含一个照抄的过期白名单，见下）
│  │  ├─ asset_versions.py       ✅ 2 端点（列表 / 回滚）
│  │  ├─ traces.py               ✅ 3 端点（**含写入侧**；Node 已删）
│  │  ├─ storage.py              ✅ 2 端点（info + change：切数据根，含关库重开）
│  │  ├─ usage.py                ✅ 2 端点（summary / board）
│  │  ├─ agent_configs.py        ✅ 5 端点（defaults / generate 均已落地）
│  │  ├─ style_profiles.py       ✅ 7 端点（含 distill）
│  │  ├─ generations.py          ✅ 1 端点（image+video 双表聚合）
│  │  ├─ ai_configs.py           ✅ **整域完成** + ai-providers(1)（含 /gpu/* 显存管理）
│  │  ├─ skills.py               ✅ 6 端点（**纯文件系统域**：SKILL.md 扫描/解析/增删改）
│  │  ├─ upload.py               ✅ 3 端点（multipart 上传：图片/音频/视频）
│  │  └─ export.py               ✅ **整域完成**（工程账本 JSON/MD + stale + EDL/ZIP…；**裸响应无信封**）
│  ├─ passthrough.py             绞杀者接缝共享实现（反代 / 501 / **显式委派**）
│  └─ services/
│     ├─ adapters/                ✅ **厂商适配器层（S3）**：17 家 / 纯函数；含 jscompat.py（JS 语义垫片）
│     ├─ era_background.py        时代背景（解析 + AI 提炼，整域）
│     ├─ bible_ids.py             六键 Bible：STYLE_ / COST_ / LOC_ 三键
│     ├─ script_fingerprint.py    剧本指纹门禁（整服务，纯逻辑）
│     ├─ character_match.py       台词说话人 → 角色别名归一匹配
│     ├─ storyboard_helpers.py    分镜关联同步 + 台词解析 + TTS 匹配校验
│     ├─ resource_library.py      资源库共享实现（规格驱动 + 原生 SQL 封装）
│     ├─ color_grade.py           校色（规整 + ffmpeg 像素管线，整域）
│     ├─ asset_versions.py        资产版本留档 / 回滚 / 写回主表
│     ├─ trace_store.py           trace 回放（只读侧）
│     ├─ data_storage.py          数据存储信息（只读侧）
│     ├─ usage_tracking.py        用量汇总 + 多集成本看板（只读侧）
│     ├─ style_profiles.py        风格 Profile CRUD（不含 LLM 提炼）
│     ├─ ai_configs.py            settings 合并逻辑 + is_local_config + 预设常量
│     ├─ skill_parser.py          SKILL.md frontmatter 解析 / 渲染 / 外来工具引用提取
│     ├─ skills.py                skill 目录扫描 + 默认绑定解析 + 体量预算常量
│     ├─ agent_registry.py        Agent 阶段名/显示名/宿主工具名（**TS 常量的镜像副本**）
│     ├─ file_storage.py          落盘上传文件 + 安全绝对路径（防穿越）+ base64 存取
│     ├─ project_ledger.py        工程账本 JSON / Markdown / stale 扫描（纯 DB）
│     ├─ provider_probe.py        厂商 URL 拼接 / 探测描述 / 模型列举解析 / key 遮蔽
│     ├─ ollama.py                Ollama 可执行探测 / 启停轮询 / 本地四大运行时探活
│     ├─ cost_catalog.py          成本单价目录 + settings.pricing 覆盖（S1）
│     ├─ estimate_service.py      生成前费用预估（S1）
│     ├─ task_logger.py           任务日志 + trace 句柄 + 落盘前脱敏（S1）
│     ├─ ai_providers.py          厂商配置解析层：所有适配器与 Agent 的入口（S1）
│     ├─ protocol.py              Agent 输出协议契约与解析（S1）
│     ├─ prompt_utils.py          prompt-utils.ts 移植（S2 第 1–2/6 块；漂移守卫覆盖）
│     ├─ vendor_errors.py         厂商错误归因（用户可读中文）+ 指数退避重试（S4 前置）
│     ├─ text_generation.py       文本链路：动作建议/拆镜头/续写/优化/拆分视觉/音色打标 + 本地规则拆分器
│     ├─ image_generation.py      图片生成链路：入队 + 多模型 fallback + 轮询 + 落盘 + 回写 + 崩溃恢复
│     ├─ video_generation.py      视频生成链路：同上 + prompt 洗白 / Vidu 免轮询 / ffprobe 补时长
│     ├─ tts_generation.py        TTS 合成（**同步**：无轮询，hex 直接落盘）+ 角色试听
│     ├─ voice_clone.py           音色快速复刻（MiniMax 上传+复刻 / CosyVoice 零样本）
│     ├─ visual_graph.py          视觉图谱（41 节点：景别/构图/运镜/灯光）+ 中文→英文术语翻译
│     ├─ shot_router.py           逐镜路由决策（T2V/I2V/FL2VA/R2V/keyframe/blocked）+ 回写 route
│     ├─ ffmpeg_compose.py        单镜合成（ffmpeg 子进程：视频+配音+烧字幕；自己管短事务）
│     ├─ ffmpeg_merge.py          整集拼接（concat + BGM 混音 + 响度归一化；QC 调用点占位）
│     ├─ grid_split.py            宫格切分（**ffmpeg crop 替 sharp**：不新增 Pillow 依赖）
│     ├─ camera_movement_guides.py 运镜 → 首尾帧构图指导（29 条中文表，顺序即匹配优先级）
│     ├─ video_probe.py           本地视频时长探测（ffprobe；失败即 0）
│     └─ take_budget.py           per-shot take 预算（生成次数收敛门禁）
└─ tests/
   ├─ smoke_test.py             冒烟测试（模式 + 契约，476 用例；删 `backend/` 后 TS 侧走快照）
   ├─ adapters_test.py          适配器层自检（JS 语义逐条对齐，101 用例）
   ├─ vendor_errors_test.py     错误归因 + 重试（MockTransport，零真实网络，58 用例）
   ├─ text_generation_test.py   文本生成纯逻辑（提示词 / 规则拆分器，67 用例）
   ├─ image_generation_test.py  图片生成链路（门禁 / 入队字段 / 回写优先级 / 崩溃恢复，64 用例）
   ├─ video_generation_test.py  视频生成链路（prompt 洗白 / 空数组差异 / 同步路径 / 恢复，50 用例）
   ├─ tts_generation_test.py    TTS + 音色复刻（MockTransport 捕获请求体，34 用例）
   ├─ prompt_storyboard_test.py 分镜 prompt 构建 + 视觉图谱 + DB 上下文辅助（38 用例）
   ├─ grid_prompt_test.py       宫格 prompt 三模式 + 运镜构图表 + 参考资产收集（48 用例）
   ├─ videos_route_test.py      逐镜路由优先级 + videos 6 端点（富化/帧统一/降级，43 用例）
   ├─ compose_test.py           单镜 ffmpeg 合成（假 ffmpeg，锁参数与状态机）+ compose 3 端点（35 用例）
   ├─ merge_test.py             整集拼接流水线（concat/BGM 混音/响度归一化，假 ffmpeg）+ merge 2 端点（29 用例）
   ├─ grid_route_test.py       宫格 prompt 归一/出图参数/切分回写 + 4 端点（42 用例）
   ├─ misc_routes_test.py      视觉图谱/图片/回调三域（裸 JSON、负面词二选一、webhook 三种码，37 用例）
   ├─ ai_voices_test.py        AI 音色：列表/试听/复刻/按角色批量/同步（43 用例）
   ├─ agent_protocol_test.py   Agent 输出协议 + 工具基座（25 用例）
   ├─ agent_tools_test.py      宫格工具集 + 工厂契约（15 用例）
   ├─ agent_voice_tools_test.py 音色工具：两条硬规则 + 兜底表（22 用例）
   ├─ agent_script_corpus_test.py 截断 + 剧本工具 + 语料检索（25 用例）
   ├─ agent_extract_tools_test.py 提取工具：去重合并 + 提示词兜底（23 用例）
   ├─ agent_storyboard_tools_test.py 分镜工具：整集重建 + 说话人绑定（32 用例）
   ├─ agent_runtime_test.py   Agent 运行时：失败分类/退避/模型 fallback/风格注入（49 用例）
   ├─ agent_route_test.py     Agent 聊天域 2 端点（文案差异/校验顺序/falsy，16 用例）
   ├─ mcp_test.py             MCP 接入层（假 stdio server 端到端）+ 3 端点（32 用例）
   ├─ agent_prompts_test.py   Agent 出厂提示词（完整性/插值/接上运行时，20 用例）
   ├─ evaluation_scorer_test.py 评测打分器（位精对齐 JS/契约校验真实基准文件，39 用例）
   ├─ skills_test.py          Skill 解析/加载（隔离目录 + 真实 backend-py/skills/ 集成，45 用例）
   ├─ evaluation_route_test.py 评测执行器（seed 提交/抽取口径/清理）+ 2 端点（28 用例）
   ├─ creator_test.py         Agent 创建器 + POST /agent-configs/generate（28 用例）
   ├─ optimizer_test.py       提示词优化器状态机 + POST /optimize/{id}（28 用例）
   ├─ evaluation_scheduler_test.py 评测调度器（防重入/失败隔离/延时）+ 2 端点（22 用例）
   ├─ sse_hub_frames_test.py  SSE 事件总线 + 尾帧提取（真实尾帧不碰设计尾帧，24 用例）
   ├─ auto_pipeline_test.py   全自动管线（8 阶段幂等/崩溃恢复/依赖补全/SSE，50 用例）
   ├─ local_models_test.py    本地模型：规则优先级 + 扫描上限 + 断点续传（89 用例）
   ├─ characters_generate_test.py 角色生成链路 8 端点（锚定/画风收口/错误码不统一，59 用例）
   ├─ props_scenes_generate_test.py 物品/场景出图（跨集参考图防串图，23 用例）
   ├─ storyboards_generate_test.py 分镜 TTS/出图/LLM 5 端点（两套回执/拆分顺延，39 用例）
   ├─ episodes_continue_script_test.py 剧集续写（软删 404 / mode 判定，8 用例）
   ├─ export_service_test.py  导出：EDL 时间码 + ZIP 打包（22 用例）
   ├─ jianying_draft_test.py  剪映草稿：UUID 引用 + 微秒计时（25 用例）
   ├─ qc_report_test.py      QC 报告 + 联系表（探测回退链 / 60 分边界，42 用例）
   ├─ era_style_distill_test.py 时代背景提炼 + 风格提炼（截断/围栏/异常包装，31 用例）
   ├─ eval_cli_test.py        基准资产搬迁 + 评测 CLI（22 用例）
   ├─ gpu_manager_test.py     GPU 显存调度（队列/驱逐/卸载策略，28 用例）
   ├─ gpu_lease_wiring_test.py GPU 租约接线（text/tts 即用即放 + image/video 长租约，20 用例）
   ├─ parity_diff_test.py     差分对拍**比较器**（归一化/三态判定/白名单越界/MISSING 类，21 用例）
   ├─ rhythm_phase_test.py    多集节奏相位（阈值/累计占比/加权 balance，22 用例）
   ├─ qc_scoring_test.py      镜头 QC 打分（加权总体分/三态 status/upsert/接线，29 用例）
   ├─ qc_retry_test.py        审片重跑闭环（软删产物/FL2VA/参考图开关，14 用例）
   ├─ set_frame_test.py       设置首尾帧 + 抽帧泛化（首帧不 seek/尾帧回退 0.2s，14 用例）
   ├─ regenerate_frame_test.py 重生成镜头帧（帧类型白名单/帧提示词/拼接顺序，17 用例）
   ├─ consistency_qc_test.py   图像连续性 QC（真实图 dHash：ok/info/warning 三档，28 用例）
   ├─ freeze_snapshot_test.py    TS 快照反漂移（Python 模块快照 + 物化逐字一致，11 用例；删库后 1 条真源码存在性检查按设计 [skip]）
   ├─ grid_agent_prompt_test.py 宫格 Agent 提示词 + 端点（12 用例）
   ├─ subagent_test.py        子 Agent 调度工具（16 用例）
   ├─ http_logger_test.py     请求日志中间件（格式/截断/开关，13 用例）
   ├─ compressed_data_url_test.py 参考图压缩（ffmpeg，16 用例）
   ├─ storage_change_test.py  数据根切换 + 存储 2 端点（复制/回滚/零副作用，38 用例）
   ├─ dockerfile_contract_test.py 生产镜像布局一致性（Dockerfile/compose 路径/端口 ↔ 代码常量，23 用例）
   ├─ frontend_api_coverage_test.py **前端调用点 ↔ 后端路由覆盖**（删库后唯一后端的安全网，5 用例）
   ├─ contract_mirror_test.py  **共享契约镜像**（contracts.ts ↔ 后端，5 用例）
   ├─ route_parity_test.py      路径 + 常量守卫（防「未迁移端点被参数路由吞掉」与镜像漂移）
   └─ run_all.py                一次跑完以上六十五项（**套件权威清单就在这个文件里**，本树只是摘录）
├─ skills/                      Agent 技能库（**2026-09-15 从仓库根 skills/ 并入**；自有 SKILL.md + 外部技能库）
│  ├─ README.md                 技能库权威约定（改 skill 前必读）
│  ├─ <name>/SKILL.md           自有 skill（frontmatter `agents:` 决定默认注入）
│  └─ <lib>/library.yaml        外部技能库声明（库由声明文件识别，零代码加库）
├─ scripts/                     工具与自检脚本（**全部 Python**，2026-09-15 从仓库根 scripts/ 搬来）
│  ├─ check_memory.py           记忆层守卫（8k 预算 + 锚点 + 落点路径）
│  ├─ check_skill_refs.py       skills 引用完整性守卫
│  ├─ test_guards.py            两套守卫的自检（13 例，含 ④b「步骤小节必须有锚点」）
│  ├─ check_all.py              一键跑全部三道自检
│  ├─ corpus/                   语料管线（fetch_raw → normalize → search + analyze{,2,3}）
│  └─ model_manager.py 等       AI/GPU 工具链（+ sd_h3_pipeline / compat_probe / h3_install / migrate_models）
│     ↑ 细节见 `scripts/README.md`（含「为什么不放仓库根」「pre-commit 怎么触发」）
└─ local_services/              **本地服务根**（2026-09-15 从仓库根 local_services/ 并入）
   └─ h3/{server.py,requirements.txt}  项目自带的 H3 薄封装（端口 8765，复用 minimax adapter）
   ↑ ⚠️ 其余子目录是 `model_manager.py --runtime git` **clone 来的第三方服务**（机器相关，已 gitignore）；
     默认值三处同步：本目录 / `app/services/local_model_scan.py` / TS 侧 `local-model-scan.ts`
```

## 厂商适配器层（S3）要点

**这一层全是纯函数** —— 这是移植时最重要的发现，也是它好迁的原因：

```python
adapter.build_generate_request(config, record)  # → {"url", "method", "headers", "body"}（不发请求）
adapter.parse_generate_response(result)         # → {"isAsync", "taskId"/"imageUrl"}（不解析 HTTP）
```

真正发 HTTP 的是 `image-generation` / `video-generation` / `text-generation` / `tts-generation`
那几个服务。所以适配器层**不需要网络、不需要密钥、不需要 mock**，可以完全离线逐条比对。

| 能力 | 厂商 |
|---|---|
图片 | `minimax` / `openai` / `gemini` / `volcengine` / `ali` / `chatfire`（复用 OpenAI）/ `local-sd` |
视频 | `minimax` / `volcengine` / `vidu` / `ali` |
TTS | `minimax` / `cosyvoice` |
文本 | `openai` / `openrouter` / `chatfire` / `ollama` / `volcengine` / `ali` / `minimax`（7 家**共享一个实例**，只差端点前缀）/ `gemini` |

几个「读代码才知道」的行为（都有用例锁住，别当成 bug 改）：

* MiniMax 图片的 `aspect_ratio` 只在**入参**给了 `size` 时才派生 —— `body.size` 有默认值
  `1920x1080`，但那个默认值**不参与**派生（原 TS 判的是 `record.size`）；
* Vidu 的 `Authorization` 是 **`Token x`** 而不是 Bearer，且**没有轮询接口**
  （返回 `vidu://no-polling-endpoint` 伪 URL，状态靠 Webhook 回调推进）；
* 火山视频的 `duration` 会被夹到 **`[4, 12]`**，非有限值回落 5；
* Ali 两家的 `seed` 用**随机数**（已抽成可替换的函数，测试里替换掉）；
* Gemini 图片的响应判定顺序与别家**不同**：URL → base64 → taskId → error，且
  `finishReason` 命中安全策略时抛的是**中文可行动提示**（不是英文 `finishMessage`）。

## 必跑的自检（一条命令）

```bash
cd backend-py
# 依赖：python-multipart（upload 域的 multipart 解析，Starlette 的 request.form() 必需）
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe tests/run_all.py     # 十六项一次跑完，退出码 0 = 全过
```

单项也可以单独跑（调试时更快）：

| 自检 | 覆盖 |
|---|---|
`tests/smoke_test.py` | 契约冒烟：模式 + 字段 + 错误码（**跑在数据库副本上**） |
`tests/adapters_test.py` | 适配器层：JS 语义逐条对齐（**纯函数，无需网络/密钥**） |
`tests/vendor_errors_test.py` | 厂商错误归因 + 指数退避重试（`MockTransport`，**零真实网络**） |
`tests/text_generation_test.py` | 文本链路纯逻辑：提示词拼装 / 本地规则拆分器 / 防幻觉校验 |
`tests/image_generation_test.py` | 图片生成链路：门禁 / 入队字段 / 回写优先级 / 崩溃恢复（**跑在临时数据根上**） |
`tests/video_generation_test.py` | 视频生成链路：prompt 洗白 / 空数组差异 / 同步路径不回写 / 恢复（**跑在临时数据根上**） |
`tests/tts_generation_test.py` | TTS + 音色复刻：hex 截断语义 / fallback / 零样本 / multipart（**`MockTransport`，零真实网络**） |
`tests/prompt_storyboard_test.py` | 分镜 prompt 构建 / 视觉图谱匹配规则 / 分镜上下文与参考图（**跑在临时数据根上**） |
`tests/videos_route_test.py` | 逐镜路由优先级 / `videos` 6 端点：富化、帧来源统一、referenceMode 降级（**跑在临时数据根上**） |
`tests/route_parity_test.py` | 路径守卫（0 遮蔽）+ 镜像常量漂移守卫（**跑在临时数据根上**） |

**都不碰真实库，也不需要任何 API Key。** 每迁完一块请先跑一次，它们专拦四类**静默失效**
（都不报错、只是结果悄悄不对）：

1. **参数路由吞掉未迁移端点** —— 返回一个看起来合理的 404，实则丢了 Node 的实现；
2. **镜像常量漂移** —— `agent_registry.py` / `prompt_utils.py` / 适配器注册表 /
   `text_generation.py` 的提示词与词表与 TS 源不一致时，`missingTools` 会误报、
   Agent 标签会显示错名、生图提示词会变味、**规则拆分器会拆分错**；
3. **JS 语义写歪** —— `Math.round` / `Number` / `??` / `[]` 真值 / `parseInt` /
   `Buffer.from(x,'base64')` / `JSON.stringify` 丢 `undefined` 这几处，
   写成 Python 的直觉版本都能跑通，但结果与 Node 不同；
4. **错误归因错档** —— 审核拦截被当成普通错误时，用户看到的是英文原文而不是
   「该换参考图还是该改提示词」的行动建议。

⚠️ **冒烟用例名只用 ASCII 可打印字符**：Windows 控制台默认 GBK，用例名里出现 `⊆` / `∪` / `∈`
这类数学符号会直接 `UnicodeEncodeError` 打断整个测试（已被咬过一次，且是在**所有断言都通过**的
情况下被"报告环节"打断，很容易误判成业务 bug）。

⚠️ **临时 `.ps1` 探针脚本也只用 ASCII**：PowerShell 5.1 按 **ANSI 代码页**读 `.ps1`，
而本仓库的文件是无 BOM 的 UTF-8 ⇒ 脚本里的中文会被打乱成乱码字节，**连引号都会被吃掉**，
报出一堆看不懂的 `Missing closing ')'`。这个坑在「控制台输出」「测试用例名」之后**第三次**出现，
形态各不相同 —— 统一原则：**凡是要给 Windows 工具链读的文本，只用 ASCII**。

## 迁移时必须守住的对齐约定

1. **响应信封与错误文案逐字对齐**。前端 `useApi.ts` 是原生 `fetch`，判据为
   `!resp.ok || (json.code && json.code >= 400)`，错误文案直接取 `json.message` 展示给用户。
   而 FastAPI 默认返回 `{"detail": ...}`，会让文案变成「请求失败 (422)」——
   故 `main.py` 把 `HTTPException` / `RequestValidationError` / 兜底异常**全部收口**成信封。
   * 成功：HTTP 200 + `{code:200, data, message:"success"}`
   * 创建：HTTP 201 + `{code:201, data, message:"created"}`
   * 失败：HTTP 4xx/5xx + `{code, message}` —— **没有 `data` 键**
   * 特例：`GET /api/v1/health` 是裸对象（Node 侧就没包信封）
2. **HTTP 层 snake_case、聚合视图 camelCase**。`GET /dramas/:id/prompts` 的聚合结果在原
   Node 版里就是 camelCase（`customPrompt` / `imageUrl`），照抄不要「顺手统一」。
3. **SQLAlchemy 的 `**kwargs` 不等于 JS 的对象展开**。JS 是「后者覆盖前者」，
   Python 遇到同名键直接 `TypeError: got multiple values for keyword argument`。
   凡是「先展开白名单字段、再注入主键/时间戳」的写法，必须写成
   `values = {**fields, "drama_id": ...}` 再覆盖。**此坑已在 `PUT /dramas/:id/episodes` 上真实踩到。**
4. **返回形状按域不同：有的 snake_case、有的 camelCase**。Node 里显式调过 `toSnakeCase` 的域
   （`dramas` / `episodes`）返回 **snake_case**；直接返回 drizzle 行的域
   （`characters` / `scenes` / `props`）返回 **camelCase** —— 因为 drizzle 返回的是 TS 声明的
   属性名。前端确实按这个差异写代码（episode 工作台是 `c.voice_style || c.voiceStyle`
   的防御式双读，而 `CharacterEditor.vue` 直接读 `c.voiceStyle`）⇒ **照抄，不要"顺手统一"**。
   Python 侧用 `row_to_camel(row, 表名)`。

   ⚠️ 同源的两个陷阱：
   * **`prop_templates` 是全库唯一「属性名 ≠ camelCase(列名)」的字段** ——
     TS 写的是 `customPrompt: text('image_prompt')`（列名是历史遗留的 `image_prompt`）。
     493 个字段实测**仅此 1 处**，靠 `_COLUMN_TO_JS_OVERRIDES` 处理。不做这层映射就会返回
     `imagePrompt`，而前端与 props 路由用的都是 `customPrompt`。
   * **入参方向也要转**：TS 里 `updates[key] = body[key]` 的 `key` 是 drizzle **属性名**，
     Python Core 只认 **DB 列名** ⇒ 双写兼容分支必须写成 `updates[snake] = body[camelKey]`。
     照抄原样会让 SQLAlchemy 报 `Unconsumed column names: voiceStyle`（**已在
     `PUT /characters/:id` 实测踩到**，且因为先测 snake 入参而被掩盖过一轮）。
5. **~~`success(c, undefined)` ⇒ `data` 键会被丢掉~~ —— 这条是错的，已修正**。
   原判断基于「`JSON.stringify` 会省略值为 `undefined` 的键」，据此做了个
   `success_without_data()` 并让 `PUT /storyboards/:id`、`DELETE /agent-configs/:id` 用它。
   但 **JS 的默认参数会先生效**：

   ```ts
   export function success(c: Context, data: any = null) { … }        // ← data = null
   return success(c, dialogueValidation ? { dialogue_validation } : undefined)
   ```

   显式传入的 `undefined` 被默认参数换成 `null` ⇒ 契约实际是 `{"code":200,"data":null,...}`，
   **data 键一直都在**。该 helper 已删除，两处路由改回 `success()`。
   ⚠️ 更值得记的是**成因**：当时我为这条「发现」写了测试断言 `"data" not in json`，
   于是**测试锁住的是我自己的 bug**——断言必须对着 Node 的源码/真实响应写，不能对着自己的实现写。
   真正的「无 data 键」只存在于**错误信封**（`bad_request` / `not_found` / `conflict`）。
6. **JS 默认参数会把显式 `undefined` 变成 `null`**（承接上一条）。凡是
   `fn(x = 默认值)` 被调用时传 `undefined`（含条件表达式产生 `undefined`），拿到的都是默认值。
   排查「键在不在」这类问题时，别只看 `JSON.stringify` 的行为。
7. **JS 正则与 Python 的三处语义差异**（台词解析踩过）：
   * JS `$`（无 `m` 标志）只匹配**字符串结尾**，而 Python `$` **还会匹配结尾换行符之前** ⇒
     一律改用 `\Z`；否则 `"音效\n"` 会被误判命中忽略列表。
   * JS `replace` 无 `g` 标志时**只替换第一处**，Python `re.sub` 默认全替换 ⇒ 显式 `count=1`。
   * JS 的 `while (re.exec(raw))` 与 Python `re.finditer(raw)` 语义一致（从左到右、不重叠），
     可直接对应；但**捕获组要自己加**（`\A(.+?)[:：]` 而不是 `\A.+?[:：]`）。
7. **JSON 对象的数字键**：JS 里 `obj[3]` 会先把键强制转成 `"3"` 所以能命中；Python 的
   `3 in {"3": ...}` **恒为 False** ⇒ 收到 `character_costumes` 这类以 id 为键的对象时，
   必须把字符串键转回 int，否则镜头服装变体**静默丢失**（已修复并有用例锁定）。
8. **资源库那四个域用的是另一套信封**（与项目其余部分不同，前端靠 `json.code` 才判得出来）：
   * 成功 `{"code": 0, "data": ...}`（**不是 200**）；列表/详情**没有 `message` 键**，
     只有写操作才有（`创建成功` / `更新成功` / `删除成功` / `成功删除 N 个X模板`）
   * 错误 `{"code": 400|404|500, "data": null, "message": ...}`，且 **HTTP 仍是 200**
   * 它们走**原生 SQL**（Node 侧是 better-sqlite3 的 `qAll/qGet/qRun`），所以列名是 DB 列名，
     返回体是 **snake_case + 少量 camelCase 别名**（`referenceImages` / `voiceConfig`），
     而 `tags` / `metadata` 是**原地替换**成解析后的对象
   * id 解析用 `parseInt` 语义（`/12abc` → 12），**不是** `parseParamId`
9. **`Math.round` 不是 Python 的 `round`**。JS 的 `Math.round` 是「.5 向 +∞」，而 Python 内置
   `round` 是**银行家舍入**（`round(2.5) == 2`）⇒ 成本金额一直按
   `Math.round(x * 10000) / 10000` 保留 4 位小数，用错会与 Node 差 1 个最小单位。
   已收口为 `response.js_round`（`math.floor(x + 0.5)`）。同理 `Number(x)` 收口为
   `response.js_number`（非有限值返回 `None`，**注意它不校验 `> 0`**，与 `parse_param_id` 不同）。
10. **JS 里 `[]` 和 `{}` 是「真值」，Python 里是「假值」** —— 这是最容易踩的一条。
   `input.preferences ? JSON.stringify(p) : null`、`body.storytelling || {}`、
   `if (body.skills)` 都依赖它；直接写 `if value:` 会把用户显式传入的空对象/空数组
   误判成「没传」而落成 `null`（契约上应是 `"{}"` / `"[]"`）。已收口为
   `response.js_truthy`；配套的 `response.js_nullish` 镜像 `a ?? b`（**只看 null**，
   与 `||` 语义不同 —— `model: body.model ?? existing.model` 传空串要保留空串）。
11. **绞杀者迁移的静默杀手：参数路由遮蔽未迁移端点**。注册 `GET /agent-configs/{id}` 后，
   Node 侧未迁移的 `GET /agent-configs/defaults` 会被它吞掉，返回一个看起来合理的
   404「Invalid agent config id」，**悄悄丢掉 Node 的实现**（前端表现为功能莫名失效）。
   对策：这类路径在各 router 里用 `passthrough.delegate` **显式声明、且注册在参数路由之前**；
   并用 `tests/route_parity_test.py` 机械扫描全部 Node 路径兜住（判据是行为：未注册路径
   必须落到兜底 501）。**别再靠肉眼排查这一类问题。**
13. **适配器层的 JS 语义比业务层更密**（S3 移植时集中收口在
   `app/services/adapters/jscompat.py`，101 条用例逐条锁住）：
   * `parseInt` **不是** `int()`/`float()`：取前导数字 `'1920abc'→1920`、向零截断、
     解析不出是 **NaN**（`None`）——注意 NaN 是**值**不是「不赋值」，所以键仍在、
     序列化成 `null`（`{width: NaN}` → `{"width":null}`）；
   * `Buffer.from(x,'base64')` 是**宽容**解码：忽略非法字符、**容忍缺失的 `=` 填充**，
     而 Python 的 `b64decode` 会直接抛错（CosyVoice 回音频时踩到）；
   * `a?.b?.[0]?.c` 链在缺字段时只得到 `undefined`，Python 直写 `a["b"][0]["c"]` 会抛
     `KeyError` ⇒ 收口 `jscompat.dig`；
   * `.length` 只有字符串/数组有，`` ({}).length === undefined `` ⇒ `jscompat.js_length`
     （**别用 `len()`**：`len({})` 是 0，与 `x.length === 0` 的结论相反）；
   * `String(16.0)` 是 `"16"` 而 Python 的 f-string 是 `"16.0"` ⇒ 画幅比例
     `gcd` 归约会输出 `"16.0:9.0"` ⇒ 收口 `jscompat.js_num_str`；
   * `slice(0, 200)` 前的 `JSON.stringify` 是**紧凑 + 不转义中文**的 ⇒ 收口
     `jscompat.js_json_stringify`（报错正文里塞响应体时才会发现）。
14. **`json.dumps` 默认不是 `JSON.stringify`**：Python 默认分隔符是 `(', ', ': ')`，会多出空格
   （`{"exposure": 0.5}` vs JS 的 `{"exposure":0.5}`）。这些字符串会写进**与 Node 共用**的列、
   作为**请求体**发给厂商、或**直接作为响应体**返回 —— 三种场景下多一个空格都是偏差。
   **这个偏差曾同时在 8 处存在**（`image-generation` / `asset-versions` / `dramas` / `presets` /
   `ai-configs` / `text-generation` / `resource-library` / `export`），只靠某一条用例
   （「colorGrade 有调整 -> 紧凑 JSON」）露了头。现已全部修正，并加了**机械守卫**：
   `tests/route_parity_test.py` 的 `_json_dumps_drift` 用 **AST** 扫全仓 `json.dumps`，
   缺 `separators` 就报错（白名单只留深拷贝与日志格式化两处）。
   > 守卫第一版是**文本扫描**，把「讲解这个坑的 docstring」和 `json.dumps(x, **_JSON)`
   > 都误报了 —— 改成 AST 后精确到「是不是一次 json.dumps 调用、关键字写在哪」。
15. **适配器层「显式 `undefined`」与「DB 的 `null`」要分开对待**。JS 里
   `parameters: {seed: hasRef ? undefined : Math.random()}` 的 `undefined` 会被
   `JSON.stringify` **丢掉整个键** ⇒ Python 侧应**不加这个键**（写 `None` 会变成 JSON
   `null` 发给厂商）；而来自 DB 行的可空字段本身就是 `null`，如实保留。两者在 Python 里
   都是 `None`，**只能靠上下文区分**。
15. **SQLAlchemy 原生 SQL 的两个坑**（资源库踩到，都用 `q_all/q_one/q_run` 收口）：
   * **`text()` 不支持 `?` 占位符**（只认 `:named`）⇒ 带 `?` 的 SQL 必须走
     `exec_driver_sql`（绕过编译、交给 sqlite3 原生处理）
   * 即使是 `exec_driver_sql`，**位置参数必须传 tuple** —— 传 list 会被当成 executemany 的
     「多组参数」。两者都会抛 `ArgumentError: List argument must consist only of dictionaries`，
     **症状相同、成因不同**，别只改一个就以为修好了。

## 已知差异与遗留物（均有实测证据）

| 项 | 说明 |
|---|---|
| ✅ **已补齐**：本地模型的 GPU 显存租约 | `gpu-manager.ts` 已整域移植（`services/gpu_manager.py`）+ `/gpu/*` 两端点已注册 + 四个调用点接线：`text`/`tts` 即用即放（每轮模型尝试各自申请/释放）、`image`/`video` **长租约**（提交时持有，重试/失败/完成三处释放）。`acquire` 失败只 warn 不中断任务（与原 TS 一致） |
| 🔧 `format_vendor_http_error` 比原实现**更稳**（有意差异） | 原 TS 在 `((apiErr?.code \|\| parsed?.topCode) \|\| '').toLowerCase()` 处，若厂商返回**数字** code（`{"error":{"code":404}}`）会抛 `TypeError`，把整个归因流程打断、把中文说明变成 500。这里统一 `str()` 归一 —— 只会得到一条中文错误。已用用例锁住（`vendor_errors_test.py` 的「健壮性」两条） |
| ✅ **参考图压缩已迁**（`read_image_as_compressed_data_url`，ffmpeg 实现） | 2026-09-15 落地：`scale=min(W\,iw):min(H\,ih):force_original_aspect_ratio=decrease:flags=lanczos`（等比、**不放大**，`<=0` 视为该维不约束）+ 有 alpha 走 `geq` **白底合成**（⚠️ 取 alpha 的函数名是 `alpha(X\,Y)`，写成 `a(X\,Y)` 会报 `Unknown function`）+ 末尾 `trunc(iw/2)*2` 保偶数（mjpeg 4:2:0 要求，与 sharp 最多差 1px）⇒ mjpeg。⚠️ **编码器不同 ⇒ 体积不相等**：实测 1024×768，sharp `q68+mozjpeg`=**17399 B** vs ffmpeg `-q:v 10`=**29039 B**（同视觉质量约大 1.4~1.7×，mozjpeg 本就是压缩率优化版）⇒ `quality→-q:v` 用**线性近似**（q68→10）优先保视觉质量，想要更小体积就调低 `quality` |
| ✅ **像素级校色已迁**（`apply_color_grade_to_file`，ffmpeg 实现） | 2026-09-15 落地：8 步链中 7 步是逐通道点式 ⇒ 合成 ``RGB乘法LUT → eq=saturation → 对比度/肤色 LUT``，**不装 Pillow**。⭐ 用真实 sharp 实测（`backend/probe-sharp.cjs`）才发现 **`.gamma()` 在无 resize 时是 no-op**（128→127、200,100,50→199,9?,4? 只差 ±1 取整）⇒ Python 侧**有意跳过** ⑥⑦ 两步；`modulate({brightness})` 是 **L 星感知乘法**（128→199，非 192）⇒ 用 RGB 乘法近似、约 3% 差异（已写进模块 docstring 与 `tests/color_grade_test.py`）。校准/对比度/白平衡/肤色与 Node **逐值一致**（192 / 236 / 154 / 136…） |
| ⚠️ **`ai_service_configs.model` 存的是 JSON 数组字符串** | 不是单模型名。读侧两边都是 `JSON.parse(row.model)` 再取 `models[0]`；接口入参则是**数组**（路由会 `json.dumps` 后落库）。写成裸字符串 `"dall-e-3"` 会让 `model` 解析成**空串**（Node 同样如此）—— 排查"模型没生效"时先看这里 |
| ✅ **镜头 QC 打分已迁**（`qc_scoring.py` + `technical_qc.py`） | 规则打分与技术维度均已接线（`_run_qc_after_video_complete` / webhook / 合并前置）。🔴 但技术维度里**冻帧 / 音频真峰 / 集成响度三项是两侧共同的继承缺陷**（正则与 ffmpeg 真实输出格式不符 ⇒ 永不生效；黑场/帧率/时长正常）—— 详见 `technical_qc.py` 的「继承缺陷」段与 `tests/technical_qc_test.py`，**要修必须两边一起修** |
| ⚠️ **`probe_video_duration` 需要系统 `ffprobe`** | 异步提供商不返回 `duration` 时用它补时长。**没装 ffprobe 不报错**，返回 0 ⇒ 分镜的 `duration` 键不写（保留旧值）——与原实现的 `resolve(0)` 一致 |
| ⚠️ **CosyVoice 的接口是「按猜想写的」** | `voice-clone.ts` 原文注释即写明：``/inference_zero_shot`` 约定参考 CosyVoice 官方 FastAPI 封装，**本地服务部署后需按实际接口核对**。因此 Python 侧也只做等价移植，未额外加固 —— 真机联调时以实际响应为准 |
| ⚠️ **参考音频的绝对 URL 默认指向 5789（Node 的端口）** | `to_public_media_url` 沿用原 TS 的默认基址 `http://localhost:5789`。**只跑 Python 后端时必须设 `PUBLIC_BASE_URL=http://localhost:5790`**，否则本地 H3 服务按 5789 拉参考音频会 404（绞杀期两边都在则无感） |
| ⚠️ **写自检/后台任务时别嵌套事务** | SQLite 只有一个写者：在 `with engine.begin()` 里再开一个 `engine.begin()`（例如调用了「内部自己开事务」的辅助函数）会直接 `database is locked`。生产代码里 `image/video_generation` 一律**每步一个短事务、不嵌套**；这条坑是写 `prompt_storyboard_test.py` 时踩到的 |
| 真实库比 Node 模型多 2 张表 | `assets`、`props` —— 旧版本残留，Node 侧 `db/index.ts` 不建、代码零引用，Python 同样不建模 |
| `image_generations` / `video_generations` 各有 `minio_url` 列 | 同上，旧 MinIO 存储遗留，Node 的 Drizzle 模型里也没有，全仓库零引用 |
| 列序与 DB 不同 | 若干表的列序与 DB `PRAGMA` 顺序不一致。**不影响正确性**（SQLAlchemy 全程按列名生成 SQL，不会位置化 INSERT/SELECT）；仅让 JSON 的键顺序不同，而 JSON 对象键序无语义 |
| 剧本指纹的**写入侧** `stamp_storyboards_script_hash` **两边都没接线** | 2026-09-15 实测：Node 侧 `stampStoryboardsScriptHash` 也**只有定义、零调用**（全仓 grep 只 1 处定义）⇒ Python 现状与 Node **一致**，不算缺口；哪天真要接线，两边一起改 |
| 媒体生成端点（`characters` 的 `generate-image`/`three-views`/`equip-image`/`expressions`/`batch-generate-images`/`generate-voice-sample`、`scenes` 与 `props` 的 `generate-image`、`storyboards` 的 `generate-tts`/`regenerate-image`/`regenerate-frame`/`set-frame`） | 依赖 `services/image-generation.ts` / TTS / ffmpeg，**不注册** → 走反代（或 501） |
| `DELETE /storyboards/:id` **只清 `storyboard_characters`，不清 `storyboard_props`** | **继承自 Node 的行为**（会留下 props 关联孤儿行）。照抄未改，但要知情 —— 若日后要清，需同时改两边 |
| `POST /characters/:id/generate-prompt` | 纯函数但依赖 `shared/prompt-utils.ts`（1456 行、含全部画风词表）⇒ 需作为**独立一次移植**，不宜夹在 CRUD 域里做 |
| ✅ 节奏相位 / 时代背景 AI 提炼 / 续写剧本 / 一致性 QC / 宫格 Agent 提示词 —— **均已迁**（2026-09-15） | 对应端点全部注册（**未注册 0 条**，绞杀者收口） |
| ⚠️ **`PUT /app-settings` 的画风白名单只有 6 种，而画风体系有 10 种** | **真缺陷（继承自 Node，有意照抄未修）**：`noir` / `ink-wash` / `cyberpunk` / `pixar3d` 会被 400 拒绝。权威列表在 `prompt-utils.ts` 的 `ART_STYLE_CATALOG` 与前端 `artStyles.ts`，`app-settings.ts` 那份是**过期的第三份副本**。**要修就两边一起修** —— 单边修会让切域那一刻行为静默改变 |
| 资源库的「兜底枚举」几乎不会生效 | 同样继承自 Node：兜底只在「该表一条非空值都没有」时触发，而创建时未传的字段会被写成 `''`，`IS NOT NULL` 对 `''` 成立 ⇒ 实际很少走到。已用直删数据的方式验证兜底分支本身是通的 |
| **`GET /traces/stats` 在真实数据上恒为 `runs=0`** | **预期行为，不是 bug**：真实 trace 里 token 值被**脱敏成字符串** `"***"`（实测 20 个 trace 全如此），而 `Number("***")` 是 `NaN`、`NaN‖NaN‖NaN` 为假 ⇒ 整条被跳过。Node 同样如此（已用直造 trace 的用例把这条行为锁住） |
| ✅ `POST /storage/change` **已迁**（2026-09-15，**最后一个未注册端点**） | 写项目级 `.data-root` 标记文件 + 关库重开（`db.reopen_engine()`）。⚠️ 绞杀期仍会连带挪动 Node 的数据根（共享标记），故**并存期不要随手调**；Node 下线后由单后端正常使用。自检锁住「复制不是移动 / 失败回滚 / `migrate !== false` 只认字面量 false / 探针不可写」等语义 |
| ✅ `GET /usage/estimate` **已迁** | `estimate-service` / `cost-catalog` 都已落地（2026-09-15 复核） |
| ✅ trace **写入侧**（`append_trace_event`）**已实现**（`trace_store.py`） | 与只读侧同文件、JSONL 追加；绞杀期两侧同写一份文件（纯文件、无锁竞争） |
| ✅ `GET /agent-configs/defaults` **已迁** | `DEFAULT_PROMPTS` 等价物在 `agent_prompts.py` + `agent_registry.py`（**只镜像静态度量，不镜像提示词正文** —— 历史上吃过「提示词多头维护」的亏）。⚠️ 它会被 `GET /agent-configs/{id}` 遮蔽 ⇒ **必须显式声明且注册在参数路由之前**（这条坑仍有效） |
| `agent_registry.py` 是 **TS 常量的第二份副本** | 只镜像**静态度量**（5 个阶段名 / 6 个 Agent 显示名 / 22 个宿主工具名），**不镜像任何提示词正文**。`tests/route_parity_test.py` 的**漂移守卫**直接从 `agents/index.ts` 与 `tools/*.ts` 抽取这些值比对，改名/加工具而未同步会立刻红（现为 0 漂移） |
| ✅ `POST /agent-configs/generate`、`POST /style-profiles/:id/distill` **均已迁** | creator 链路 + distill（LLM 走 `text_generation`、探测走 ffprobe 子进程）；`/apply` 仍在 |
| **布尔列曾返回 `1`/`0` 而非 `true`/`false`** | 真 bug，已修：`models._bool` 原用 `Integer`，而 drizzle 的 `integer({mode:'boolean'})` 在 JS 侧读出的是**真布尔**。改为 SQLAlchemy `Boolean`（存储层同为 SQLite INTEGER，兼容既有库）。前端只要用 `=== true` 严格比较就会挂 —— 这个偏差曾存在于**全部**已迁移域 |
| **成功信封曾错误地省略 `data` 键** | 真 bug，已修（详见下节第 5 条）：误判 `success(c, undefined)` 会让 `data` 键消失，实际 JS 默认参数会把它换成 `null`。`PUT /storyboards/:id` 与 `DELETE /agent-configs/:id` 已改回带 `data:null`；`success_without_data()` 已删除 |
| **`js_number` 曾把整数返回成 float** | 真 bug，已修：`Number("37")` 在 JS 里序列化成 `37`，而 Python 的 `json.dumps(37.0)` 是 `37.0`。**宽松比较看不出**（`37 == 37.0` 为真），但会出现在响应体字节里 —— 前端用 `===` 严格比较、或把值拼进字符串（`scope: "drama-37.0"`）就会暴露。现 `js_number` 在 `\|x\| ≤ 2^53` 内返回 `int`（更大 JS 用指数形式，转 int 反而更不像） |
| ~~`aiConfigs` 的 8 个外部依赖端点未迁移~~ | **已迁移 7 个**：`/ollama/*`(4)、`POST /models`、`POST /test`、`GET /runtime/health` —— 它们依赖的只是**子进程与 HTTP**（`where ollama` / `Popen` / nvidia-smi / 厂商探测），Python 完全等价。我一度误判成"不可迁"，实际是**能迁但当时没排上** |
| ✅ `/gpu/status` 与 `/gpu/release-all` **已迁**（2026-09-15） | `gpu_manager.py` 473 行 + 两端点 + 四个调用点接线：text/tts 即用即放、image/video **长租约**（提交持有，完成/失败/重试释放）。「绞杀期两侧各有租约视图」的顾虑随 Node 下线消失 |
| **`quick-preset` / `quick-local` 会覆盖既有配置** | 它们是 upsert（按 `service_type`+`provider`）。⇒ **冒烟测试跑在数据库副本上**没问题，但**不要对着真实库调这两个端点**（会覆盖用户配置）。真机验证时我只跑只读接口 |
| ✅ `export` **整域关闭 7/7** | `/edl`、ZIP 打包、`/jianying-draft`、`/qc-report`(JSON/MD/HTML)、`/contact-sheet` 全部已迁（2026-09-15） |
| ✅ `utils/storage.ts` **已全搬** | `downloadFile`（httpx）与 `readImageAsCompressedDataUrl`（ffmpeg）均已迁（2026-09-15）|
| `upload` 的类型判据是**客户端 Content-Type** | 与原 TS 一致（浏览器的 `file.type`），**不做文件头嗅探**。刻意不"顺手加固"：改成嗅探会让原本能上传的文件被拒，属契约变更 |
| `ai_service_providers`（服务商目录）**不由 Python seed** | Node 启动时 `seedServiceProviders()` 幂等写入；那张表**没有唯一约束**，两边同时 seed 有产生重复行的风险 ⇒ Python 只读，不重复 seed。将来 Node 下线、或需要支持全新空库时，再把这个调用搬到 Python 的 lifespan |
| `weapon-library` / `costume-library` **没有 `/apply`** | Node 侧就没实现（只有 `character` 与 `scene` 有）⇒ 调用会落到反代/501，属预期 |
| `GET /props/1.5` 与 `GET /scenes/1.5` 都是 404，但**文案不同** | 非 bug：`props` 用自定义 `parseId`（`Number.isInteger`，严格），`characters`/`scenes` 用 `parseParamId`（`Number.isFinite`，**放行小数**）——原 TS 就是两套，已各自照抄 |
| `GET /dramas/:id/prompts` 里 `episodes[].episodeNumber` 等 | 保持 camelCase，见上文第 2 条 |

## 下一个域的迁移 SOP

1. 读对应的 `backend/src/routes/<domain>.ts`（以及它依赖的 `services/*.ts`），
   逐个端点抄下来：方法、路径、查询参数、错误文案、响应形状。
2. 在 `app/routers/<domain>.py` 建 `APIRouter(prefix="/api/v1/<domain>")`。
   **路径顺序有语义**：静态子路径（如 `/stats`）必须声明在 `/{id}` 之前，否则会被吃掉。
3. 只注册已实现的路由；**没实现的不要注册** —— 留给反代接缝，这样比返回 501 更可用。
4. 在 `app/main.py` 的 `include_router` 处加一行。
5. 把该域的用例加进 `tests/smoke_test.py`，跑通（含错误路径的文案断言）。
6. 域内依赖、且输入输出明确的纯函数，先移植成 `app/services/<x>.py`；
   **依赖 LLM / 子进程 / 长任务的服务放到最后**，它们的行为需要真实调用才能验证。

## 删 `backend/` 清单（2026-09-15 **干跑实测**得出）

做法：把 `backend/` 临时改名成 `backend__dryrun_off` → 跑守卫与自检 → 立刻改回（**全程可回滚**）。
干跑结论：**删库后仓库自检全绿** —— `check_all.py` rc=0（引用 / 记忆 / 守卫自检 12/12）、
`freeze_snapshot_test.py` 11/11（真源码缺失时自动跳过存在性检查）、
`route_parity_test.py` rc=0（**自动回退 Python 快照**：224 / 227 / 未注册 0 / 0 漂移）、
`skills_test.py` 45/45。

### 真删当天（2026-09-15）又现形 1 处 —— 干跑没抓到，**必须记**

删掉 `backend/` 后跑全量：**契约冒烟 `smoke_test.py` 在导入期 FileNotFoundError** ✗（它直接读
`backend/src/db/index.ts` / `schema.ts` 来校验「models 表集/列集 == Node DDL」）。
干跑没抓到它，是因为**冻结脚本的自动发现只扫守卫**（`route_parity_test.py` / `parity_diff_test.py`），
`smoke_test.py` 不在清单里 ⇒ 那两处硬读**从来没进过快照**。

修法（三处，都已落地）：
1. `smoke_test.py` 改走 `_ts_path()`：**真源码优先、缺失回退快照**（与守卫同一策略）；
2. `freeze_ts_snapshot.py` 的 `FILES` 补 `db/index.ts` / `db/schema.ts`，
   并把 `smoke_test.py` **加进 `_GUARD_SOURCES`**（以后它再读新的 TS，自动发现能拦住）；
3. 取源新增 **git 历史**兜底 —— ⚠️ 删库当天现场常已被提交成「删除」⇒ `git show HEAD:…` 直接 fatal，
   得回溯到「最后一个还含该文件」的提交（本项目实测：`HEAD` 已无 `db/index.ts`，上一提交里还在）。
   「现有快照优先于 git」是刻意的：快照是**删库前现场**，git HEAD 可能落后（曾差点静默回退一次改动）。

结果：删库后全量 **63 套件 / 2463 项 / 0 失败** ✓（**当时**的实测值；此后新增
`frontend_api_coverage_test.py` 与 `contract_mirror_test.py` ⇒ 现为 **65 套件 / 2473 项**），
快照 **76 条 / 674 KB** ✓。

⚠️ 比删库前（2465）少的 **2 项**是**预先设计好的显式跳过**（各自会打印 `[skip]`，不是静默消失）：
`freeze_snapshot_test.py` 的「自动发现的路径在真源码下确实存在」、`eval_cli_test.py` 的
「benchmarks 与 Node 侧逐字节一致」—— 两条都只在真源码还在时才有意义。

干跑**暴露并已修掉**的三类问题（都不在「代码常量」里，所以只有真删一次才会现形）：

1. **技能文档里 13 处指向 `backend/src/…` 的引用** ⇒ 删库即断链。已全部改指 Python 侧对应实现
   （`app/services/agents/skills.py`、`app/routers/skills.py`、`app/services/prompt_utils.py`、
   `prompt_blocks.py`、`agents/tools/corpus_tools.py`、`agents/runtime.py`），
   并给其中 6 处**原本没加反引号**的写法补上反引号 —— 否则守卫（只采集反引号 token）看不见它们。
2. **守卫对 `backend/…` 的旧前缀不设防** ⇒ 已加入 `_LEGACY_HINTS`，写旧前缀一律判**致命**并给改法。
3. **冻结脚本的自动发现依赖真源码存在** ⇒ 删库后「表驱动普通字符串形态」会**静默扫不到**。
   已改成：真源码在 ⇒ 看文件是否存在；真源码已删 ⇒ **改看快照里有没有**（`--check` 在删库后依然有效）。

### 仍需在「删库当天」处理（**现在做会打断 Node，故留到最后一起做**）

| # | 项 | 说明 / 改法 |
|---|---|---|
✅ 1 | **前端代理已切到 Python**（2026-09-15 提前做） | `frontend/nuxt.config.ts` 的 `/api`、`/static` 由 **5789 → 5790**，并留了逃生门：`NUXT_API_TARGET=http://localhost:5789 npm run dev` 可临时对着 Node 调试。⚠️ 只影响 **dev 代理**（生产同源静态产物由部署侧决定） |
✅ 2 | **共享契约类型已搬到前端**（2026-09-15 提前做） | `git mv backend/src/shared/contracts.ts frontend/app/types/contracts.ts`（git 记为 `R`，79 行纯类型零 import）+ nuxt 别名两处 + 唯一 importer `app/composables/useApi.ts`；Node 侧 `era-background.ts` 同步改成跨项目 import（它随 `backend/` 一起消失）。**已用 `npm run generate` 真构建验证**（`✔ Server built` / `Prerendered 15 routes` / `✔ Generated public .output/public`） |
✅ 3 | **`Dockerfile` 已重写为 Python 镜像**（2026-09-15 提前做） | 运行时 `python:3.12-slim` + uvicorn（Node 只留前端构建阶段）、端口 **5790**、`COPY backend-py/skills/`、前端产物落 `frontend/dist`（与 `FRONTEND_DIST` 一致）、`.dockerignore` 补排除 `.venv` / `__pycache__` / `tests` / `scripts`。⚠️ **本机无 Docker ⇒ 未做真构建**；布局/端口/产物路径已由新增自检 `dockerfile_contract_test.py`（**23 用例**）钉在代码常量上，改任一侧立刻报错 |
✅ 4 | **`parity_run.py` / `parity_diff.py` 已写明生命周期**（2026-09-15） | 两者 docstring 顶部都加了「本工具与 `backend/` 绑定 ⇒ 删库后失效；留作历史证据」，并指明日常回归走 `run_all.py` + `route_parity_test.py` 的快照模式（**不需要 Node**） |
5 | **文档/记忆里的溯源引用** | 全仓还有 ~137 处提到 `backend/src/…`，**绝大多数是「移植自 X」的溯源注释**（应保留，正是它们的价值）。只需清理那些「把它当权威源去查」的指路语（`docs/api-contract.md` 等） |

> 第 1、2 项做完后又干跑了一次（`backend/` 临时改名）：`check_all.py` rc=0（引用 102 处 / 0 断链）、
> `route_parity_test.py` rc=0（224 / 227 / 未注册 0 / 0 漂移）⇒ **删库当天只剩「删目录 + Dockerfile 重写」**。

## 工期参考

完整等价替换的实测估算：**单人对 AI 当助手 ≈ 99–152 人日**；AI 当主力（人只审阅 + 验证）
≈ 50–80 人日。**其中不可压缩的部分是验证，不是编码** ——
226 个端点冒烟、5 个 Agent 行为回归、17 家厂商适配真机跑通、前端零改动走查。
最短「能替换」单人也需 3~5 周（降级验收，只冒烟）。
