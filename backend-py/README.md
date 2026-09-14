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
> `storage`(1, 只 info)、`usage`(**3/3**: 汇总 / 成本看板 / **生成前费用预估**)、
> `agent-configs`(5, 纯 DB 部分)、`style-profiles`(7, 不含 LLM 提炼)、
> `generations`(1, 双表聚合)、**`ai-configs`(15/17: 含 Ollama 启停/拉取/删除、模型列举、连通性探测、
> 本地运行时健康)** + `ai-providers`(1)、
> **`skills`(6, 整域迁移: 含 SKILL.md 解析 / 默认绑定 / 删除保护)**、`upload`(3)、
> `export`(**2/7**: 工程账本 JSON/MD + 断点续作 stale)。
> **自检 2318 项全绿**（冒烟 477 + 适配器 101 + 错误归因 58 + 文本生成 67 + 图片生成 64 + 视频生成 50 + TTS/音色复刻 34 + 分镜 prompt/图谱 38 + 宫格 prompt/运镜 48 + 逐镜路由/videos 43 + 单镜合成 35 + 整集拼接 29 + 宫格路由 42 + **图谱/图片/回调 37** + **AI 音色 43**）
> **+ 路径守卫 0 遮蔽 + 镜像常量 0 漂移**（含 `prompt_utils` 词表、适配器注册表与文案、
> `text-generation` 的 9 个提示词常量与 8 张词表、**视觉图谱 41 节点逐条**、
> 全仓 `json.dumps` 紧凑性的机械比对）。
> 其余端点由 `PROXY_TO_NODE=1` 反代到 Node，**系统始终可用**。

## 全量迁移路线（S1–S7，`backend/` 只在最后删）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **S1 地基** | `cost_catalog`(定价) / `estimate_service`(费用预估) / `task_logger`+trace **写入侧** / `ai_providers`(厂商配置解析) / `protocol`(Agent 输出协议) | ✅ **完成** |
| S2 提示词资产 | `prompt-utils.ts`(**1629 行**: 词表/模板/解析)，**分 6 块** | ✅ **6/6 块完成**（全量迁完）：①画风层+负面词层 ②角色/装备/单品/表情/物品/场景 ③分镜静帧/视频 ④DB 上下文辅助 ⑤台词校验/标签剥离 ⑥宫格 prompt；另含 `camera-movement-guides.ts` |
| S3 厂商适配器 | `registry` + `types` + 16 个适配器(1611 行) | ✅ **完成**（17 家 / 纯函数，`adapters_test.py` 101 用例 + 守卫覆盖） |
| S4 媒体服务 | image/video/TTS 生成、`compose`(ffmpeg)、宫格、合并、视觉图 | ✅ **主链全通**：`vendor-errors` + text/image/video/TTS+voice-clone 四条链路 + 逐镜路由 + `videos`(6) + `compose`(3) + `merge`(2) + **`grid`(4)**；仅剩**像素处理(校色/参考图压缩)**与**镜头 QC 打分**（调用点均已占位） |
| S5 **Mastra 替换** | Agent 循环 + 工具调用 + 协议 + **运行时** + 6 组工具(~2900 行) + `agent`(2) | ✅ **完成**：`protocol` + `tool` + **6 组工具** + **`runtime`(运行时：失败分类/退避/模型 fallback/风格注入)** + `agent`(2 端点，**非流式**)；✅ **`DEFAULT_PROMPTS` 已搬**（`services/agent_prompts.py` + 逐字守卫，2026-09-12 决策变更）；⚠️ 未迁 `subagent`/`rhythm-phase`（`skills`/`mcp` 已迁）；⚠️ **Gemini 函数调用循环未支持**（显式报错） |
| S6 编排/长任务 | `auto-pipeline`、`local-model-scan`、`evaluation`、崩溃恢复 | 🔄 **MCP 已迁**（`agents/mcp.ts` 285 行自写 JSON-RPC 客户端 + 3 端点）；**`auto-pipeline` 已整域关闭**（8 阶段编排 + SSE）；剩 **9 条**（storyboards 4 / gpu 2 / dramas 1 / episodes 1 / storage/change）、**`evaluation` 域已整域关闭**（types/catalog/scorer/evaluator/optimizer/scheduler + **5/5 端点**）、崩溃恢复 |
| S7 收尾 | 剩余 AI 端点 + 全量回归等价验证 + **删 `backend/`** | 🔄 **只剩「删 `backend/`」本身（等用户点头）**：benchmarks 4 个 case JSON 已搬（逐字节一致、`catalog`/`optimizer` 默认路径已切）｜评测 CLI 已迁｜GPU 显存租约已迁（`/ai-configs/gpu/*`）｜**未注册仅 1 条**（`storage/change`，有意延后）｜`app/` 对 `backend/` 的**路径依赖为 0**｜对拍工具与差分比较器就绪（2026-09-14 实测 0 新差异；CASES 仅 10 条只读 GET，**删库当天建议补齐 S7 新增 GET 再跑一次**）｜守卫快照已重冻 **74 文件 / 715 KB**，且「真源码 vs 冻结」结论一致 ⇒ 删库后九道守卫价值保留 |

顺序是按**依赖**排的：S1 是所有适配器/Agent 的入口，S2/S3 被 S4/S5 依赖，S5 被 S6 依赖。
**`backend/` 只能在 S7 删** —— 删之前必须先证明等价（129→224 端点的全量对拍）。

## 为什么不一次性重写

原始体量实测：**133 个 TS 文件 / 27,389 行 / 226 个端点 / 29 张表**，且**零自动化测试**。
一次性重写的问题不是「写不完」，而是**中途没有任何可验证的中间态** ——
4~6 周后才能第一次真正跑通，期间风险全部堆积在最后一刻。

绞杀者模式下每一天都有可跑、可回退的产物：

```
浏览器 ──► FastAPI :5790 ──┬─► 已迁移的域：Python 直接服务
                            └─► 未迁移的域：反代 ──► Node :5789
```

迁完一个域，就在 `app/main.py` 里 `include_router(...)` 一行，然后把该域从 Node 侧停用。
全部迁完后去掉反代，Python 独占端口即可（`PY_PORT=5789`）。

## 删 `backend/` 前的等价性对拍（S7 第 5 步）

     两侧**并排起**（Node 5789 / Python 5790），**指向同一份数据**，然后跑差分对拍：

    ```bash
    # 1) 先起 Node（它会初始化库/种服务商）
    cd backend; $env:DATA_ROOT='<空目录>'; npx tsx src/index.ts
    # 2) 再起 Python（同一个 DATA_ROOT）
    cd ../backend-py; $env:DATA_ROOT='<同一目录>'; .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
    # 3) 对拍（逐字段 diff；new 差异会以退出码 1 报出来）
    .venv\Scripts\python.exe tests/parity_diff.py --report ..\tmp\parity.json
    # 或一条命令编排（起两侧 + 对拍 + 收尾）：python tests/parity_run.py

    **2026-09-15 实测：一致 12 ｜ 已知差异 0 ｜ 不存在路径 4 ｜ 新差异 0**（覆盖 16 条只读路径逐字段等价；CASES 已补齐 S7 新增的 `dramas/*/rhythm` 与 `export/dramas/*/qc-report?format=json`）。
    ⚠️ MISSING 类 = 「两边都不该有」的路径（Node 404 未匹配 / Python 501 兜底），**Node 回 200 才算 new**。
    ```

    ⚠️ **`<项目根>/.data-root` 标记文件优先级高于 `DATA_ROOT` 环境变量**（Node 侧）⇒ 有它就会读到别处的库、
    对拍结果无意义；跑之前先确认它不存在（本仓当前**不存在**）。⚠️ 只比对**只读端点**（表里全是 GET），
    因为两侧启动时都会写库。

## 删 `backend/` 之后：守卫靠**冻结快照**继续工作（S7 第 6 步）

九道漂移守卫原先是**读 TS 源码**来证明「Python 的路由表/常量/提示词没漂移」。删库前先冻结：

```bash
python tests/freeze_ts_snapshot.py          # 把守卫读到的 74 个 .ts 复制到 tests/frozen_ts/（715 KB；清单 = 手写 + 从守卫源码自动发现）
python tests/freeze_ts_snapshot.py --check  # 校验完整性（真源码还在时会逐个核对）
python tests/route_parity_test.py           # 照常跑
PARITY_USE_FROZEN=1 python tests/route_parity_test.py   # 强制用快照（验证「删库后照样能跑」）
```

守卫里的源码根是 ``_SRC_ROOT``：**真源码优先，缺失自动回退快照**。已实测：真源码与快照两条路径
结论**完全一致**（2026-09-15 实测：Node 224 / Python 226 / 未注册 1 / 0 漂移；由 `tests/freeze_snapshot_test.py` 守着「快照覆盖每个守卫读文件」）⇒ 删库后守卫价值完整保留。

## 运行

```bash
cd backend-py

# 1) 建虚拟环境（首次）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux

# 2) 启动（默认 5790；与 Node 的 5789 并存）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5790

# 3) 想让它把未迁移的域转发给 Node（需要 Node 后端已在 5789 运行）
set PROXY_TO_NODE=1
.venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
```

自检（**唯一回归安全网，改完代码必跑**）：

```bash
.venv\Scripts\python.exe tests\smoke_test.py      # 退出码 0 = 全过
```

它做三件事：把表/列定义与真实 `data/drama.db` 的 `PRAGMA table_info` 逐列比对（并与 Node 的
`db/index.ts` 建表清单交叉印证）、打真实接口核对响应信封与**错误文案逐字**、写操作全部落在
数据库副本上（真实库只读）。报告落在系统临时目录，路径在结尾打印。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PY_PORT` | `5790` | 本后端端口。**刻意不读 `config.yaml` 的 `server.port`** —— 那是 Node 的端口，并存期读同一个必然抢占 |
| `PORT` | — | 兼容用，优先级低于 `PY_PORT` |
| `HOST` | `0.0.0.0` | |
| `CORS_ORIGINS` | 见 `app/config.py` | 逗号分隔 |
| `DATA_ROOT` | `configs/config.yaml` 的 `database.path` 所在目录 | 数据根目录（DB + static + traces） |
| `DB_PATH` / `STORAGE_PATH` | — | 仅在未显式指定 `DATA_ROOT` / `.data-root` 时生效（与 Node 同规则） |
| `CONFIG_PATH` | `configs/config.yaml` | |
| `PROXY_TO_NODE` | `0` | `1` = 未迁移的域反代到 Node；`0` = 返回 501 并说明未迁移 |
| `NODE_BACKEND_URL` | `http://127.0.0.1:5789` | 反代目标 |

## 目录结构

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
│  │  ├─ traces.py               ✅ 3 端点（只读侧；写入侧仍在 Node）
│  │  ├─ storage.py              ✅ 1 端点（只 info；change 见下）
│  │  ├─ usage.py                ✅ 2 端点（summary / board；estimate 见下）
│  │  ├─ agent_configs.py        ✅ 5 端点（defaults / generate 走委派）
│  │  ├─ style_profiles.py       ✅ 7 端点（distill 走委派）
│  │  ├─ generations.py          ✅ 1 端点（image+video 双表聚合）
│  │  ├─ ai_configs.py           ✅ 15/17 端点 + ai-providers(1)（仅 /gpu/* 走委派）
│  │  ├─ skills.py               ✅ 6 端点（**纯文件系统域**：SKILL.md 扫描/解析/增删改）
│  │  ├─ upload.py               ✅ 3 端点（multipart 上传：图片/音频/视频）
│  │  └─ export.py               ✅ 2/7 端点（工程账本 JSON/MD + stale；**裸响应无信封**）
│  ├─ passthrough.py             绞杀者接缝共享实现（反代 / 501 / **显式委派**）
│  └─ services/
│     ├─ adapters/                ✅ **厂商适配器层（S3）**：17 家 / 纯函数；含 jscompat.py（JS 语义垫片）
│     ├─ era_background.py        时代背景解析（纯函数；AI 提炼未迁）
│     ├─ bible_ids.py             六键 Bible：STYLE_ / COST_ / LOC_ 三键
│     ├─ script_fingerprint.py    剧本指纹门禁（整服务，纯逻辑）
│     ├─ character_match.py       台词说话人 → 角色别名归一匹配
│     ├─ storyboard_helpers.py    分镜关联同步 + 台词解析 + TTS 匹配校验
│     ├─ resource_library.py      资源库共享实现（规格驱动 + 原生 SQL 封装）
│     ├─ color_grade.py           校色参数规整（纯函数；像素处理未迁）
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
   ├─ smoke_test.py             冒烟测试（模式 + 契约，482 用例）
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
   ├─ skills_test.py          Skill 解析/加载（隔离目录 + 真实 skills/ 集成，45 用例）
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
   ├─ freeze_snapshot_test.py    TS 源码快照反漂移（守卫读到的文件必须在快照里，7 用例）
   ├─ route_parity_test.py      路径 + 常量守卫（防「未迁移端点被参数路由吞掉」与镜像漂移）
   └─ run_all.py                一次跑完以上五十六项
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
| ⚠️ **参考图压缩未迁**（`read_image_as_compressed_data_url`） | 原实现是 sharp 的「长边 ≤768 等比缩放（不放大）→ 有 alpha 则 flatten 白底 → JPEG q68」。Python 侧需要 Pillow ⇒ 当前**退化为原图 data URL**并告警一次：**内容与画质不变**，只是发给厂商的载荷更大（768/q68 本来是为控制体积与厂商限制）。装上 Pillow 后替换函数体即可（参数已写在 docstring 里） |
| ⚠️ **像素级校色未迁**（`apply_color_grade_to_file`） | 原实现是 sharp 的 RGB 增益 / gamma / 白平衡 / 曝光 / 饱和度 / 对比度 / 肤色 / 阴影高光（142 行）。Pillow 与 sharp 的重采样与 JPEG 编码**不会逐字节一致**，而校色结果是**持久化且用户可见**的资产 ⇒ 不在没有对照验证的情况下换实现。当前行为**与原实现的失败路径完全一致**：无参数原样返回；有参数抛错 → 调用方记 `color-grade-failed` 并**保留未校色图**（Node 校色失败时也是这个结果） |
| ⚠️ **`ai_service_configs.model` 存的是 JSON 数组字符串** | 不是单模型名。读侧两边都是 `JSON.parse(row.model)` 再取 `models[0]`；接口入参则是**数组**（路由会 `json.dumps` 后落库）。写成裸字符串 `"dall-e-3"` 会让 `model` 解析成**空串**（Node 同样如此）—— 排查"模型没生效"时先看这里 |
| ⚠️ **镜头 QC 打分未迁**（`qc-scoring.ts` + `technical-qc.ts`，共约 510 行） | 视频完成后的 fire-and-forget 增强（`execFile` 调 ffmpeg/ffprobe 做技术质检 + 写 `storyboards.qc_*`）。调用点已就位 `_run_qc_after_video_complete`，当前只记一条 `qc-skipped` 告警。与「媒体收尾域」一起做 |
| ⚠️ **`probe_video_duration` 需要系统 `ffprobe`** | 异步提供商不返回 `duration` 时用它补时长。**没装 ffprobe 不报错**，返回 0 ⇒ 分镜的 `duration` 键不写（保留旧值）——与原实现的 `resolve(0)` 一致 |
| ⚠️ **CosyVoice 的接口是「按猜想写的」** | `voice-clone.ts` 原文注释即写明：``/inference_zero_shot`` 约定参考 CosyVoice 官方 FastAPI 封装，**本地服务部署后需按实际接口核对**。因此 Python 侧也只做等价移植，未额外加固 —— 真机联调时以实际响应为准 |
| ⚠️ **参考音频的绝对 URL 默认指向 5789（Node 的端口）** | `to_public_media_url` 沿用原 TS 的默认基址 `http://localhost:5789`。**只跑 Python 后端时必须设 `PUBLIC_BASE_URL=http://localhost:5790`**，否则本地 H3 服务按 5789 拉参考音频会 404（绞杀期两边都在则无感） |
| ⚠️ **写自检/后台任务时别嵌套事务** | SQLite 只有一个写者：在 `with engine.begin()` 里再开一个 `engine.begin()`（例如调用了「内部自己开事务」的辅助函数）会直接 `database is locked`。生产代码里 `image/video_generation` 一律**每步一个短事务、不嵌套**；这条坑是写 `prompt_storyboard_test.py` 时踩到的 |
| 真实库比 Node 模型多 2 张表 | `assets`、`props` —— 旧版本残留，Node 侧 `db/index.ts` 不建、代码零引用，Python 同样不建模 |
| `image_generations` / `video_generations` 各有 `minio_url` 列 | 同上，旧 MinIO 存储遗留，Node 的 Drizzle 模型里也没有，全仓库零引用 |
| 列序与 DB 不同 | 若干表的列序与 DB `PRAGMA` 顺序不一致。**不影响正确性**（SQLAlchemy 全程按列名生成 SQL，不会位置化 INSERT/SELECT）；仅让 JSON 的键顺序不同，而 JSON 对象键序无语义 |
| 剧本指纹的**过期判定**已迁移，但**写入侧**未全 | `checkEpisodeFingerprint` / `refreshEpisodeScriptHash` / `checkStoryboardGate` / `stampStoryboardsScriptHash` 都已具备；`stampStoryboardsScriptHash` 要等 `storyboards` 域迁移时接上 |
| 媒体生成端点（`characters` 的 `generate-image`/`three-views`/`equip-image`/`expressions`/`batch-generate-images`/`generate-voice-sample`、`scenes` 与 `props` 的 `generate-image`、`storyboards` 的 `generate-tts`/`regenerate-image`/`regenerate-frame`/`set-frame`） | 依赖 `services/image-generation.ts` / TTS / ffmpeg，**不注册** → 走反代（或 501） |
| `DELETE /storyboards/:id` **只清 `storyboard_characters`，不清 `storyboard_props`** | **继承自 Node 的行为**（会留下 props 关联孤儿行）。照抄未改，但要知情 —— 若日后要清，需同时改两边 |
| `POST /characters/:id/generate-prompt` | 纯函数但依赖 `shared/prompt-utils.ts`（1456 行、含全部画风词表）⇒ 需作为**独立一次移植**，不宜夹在 CRUD 域里做 |
| 节奏相位 / 时代背景 AI 提炼 / 续写剧本 / 一致性 QC / 自动拆分视觉特征 | 尚未移植（依赖 LLM 或视觉模型），对应端点不注册 → 走反代（或 501） |
| ⚠️ **`PUT /app-settings` 的画风白名单只有 6 种，而画风体系有 10 种** | **真缺陷（继承自 Node，有意照抄未修）**：`noir` / `ink-wash` / `cyberpunk` / `pixar3d` 会被 400 拒绝。权威列表在 `prompt-utils.ts` 的 `ART_STYLE_CATALOG` 与前端 `artStyles.ts`，`app-settings.ts` 那份是**过期的第三份副本**。**要修就两边一起修** —— 单边修会让切域那一刻行为静默改变 |
| 资源库的「兜底枚举」几乎不会生效 | 同样继承自 Node：兜底只在「该表一条非空值都没有」时触发，而创建时未传的字段会被写成 `''`，`IS NOT NULL` 对 `''` 成立 ⇒ 实际很少走到。已用直删数据的方式验证兜底分支本身是通的 |
| **`GET /traces/stats` 在真实数据上恒为 `runs=0`** | **预期行为，不是 bug**：真实 trace 里 token 值被**脱敏成字符串** `"***"`（实测 20 个 trace 全如此），而 `Number("***")` 是 `NaN`、`NaN‖NaN‖NaN` 为假 ⇒ 整条被跳过。Node 同样如此（已用直造 trace 的用例把这条行为锁住） |
| `POST /storage/change` **未迁移** | 它写项目级 `.data-root` 标记文件，而 **Node 启动时也读该文件** ⇒ 从 Python 切目录会连带把 Node 的数据根一起挪走（跨进程副作用），且需关库重开。等 Node 下线再迁 |
| `GET /usage/estimate` **未迁移** | 依赖 `estimate-service.ts` + `cost-catalog.ts`（各厂商单价目录）⇒ 属媒体域前置，随那一批一起做 |
| trace **写入侧**（`appendTraceEvent`）未迁移 | 唯一调用方是 Agent 链路（尚未迁移）。Python 只读、Node 写，纯文件无锁竞争，可共存 |
| `GET /agent-configs/defaults` **仍未迁移**（`skills` 域迁完后依然如此） | SKILL.md 那半边（`agents:` 绑定）**已经可以算了**，卡点只剩 `DEFAULT_PROMPTS[].instructions` —— **5 个 Agent 数百行提示词正文**。我**刻意不搬**：Node 侧源码注释明确写过本项目吃过「提示词多头维护」的亏（前端曾自留一份副本并与后端漂移、还引用了已不存在的文件），把这份资产复制进 Python 等于**重建那个问题**。等 Node 下线（单一后端）再搬，那时它就是唯一来源。⚠️ 它会被 `GET /agent-configs/{id}` 遮蔽，所以**必须在 router 里显式声明委派且注册在参数路由之前** |
| `agent_registry.py` 是 **TS 常量的第二份副本** | 只镜像**静态度量**（5 个阶段名 / 6 个 Agent 显示名 / 22 个宿主工具名），**不镜像任何提示词正文**。`tests/route_parity_test.py` 的**漂移守卫**直接从 `agents/index.ts` 与 `tools/*.ts` 抽取这些值比对，改名/加工具而未同步会立刻红（现为 0 漂移） |
| `POST /agent-configs/generate`、`POST /style-profiles/:id/distill` 未迁移 | 前者调 LLM 生成配置；后者依赖 `@mastra` Agent + `@ai-sdk` + `ffprobe`。但 `/style-profiles/:id/apply` **已迁**，用户可把 Node 侧提炼好的结果贴回来落库，链路不阻塞 |
| **布尔列曾返回 `1`/`0` 而非 `true`/`false`** | 真 bug，已修：`models._bool` 原用 `Integer`，而 drizzle 的 `integer({mode:'boolean'})` 在 JS 侧读出的是**真布尔**。改为 SQLAlchemy `Boolean`（存储层同为 SQLite INTEGER，兼容既有库）。前端只要用 `=== true` 严格比较就会挂 —— 这个偏差曾存在于**全部**已迁移域 |
| **成功信封曾错误地省略 `data` 键** | 真 bug，已修（详见下节第 5 条）：误判 `success(c, undefined)` 会让 `data` 键消失，实际 JS 默认参数会把它换成 `null`。`PUT /storyboards/:id` 与 `DELETE /agent-configs/:id` 已改回带 `data:null`；`success_without_data()` 已删除 |
| **`js_number` 曾把整数返回成 float** | 真 bug，已修：`Number("37")` 在 JS 里序列化成 `37`，而 Python 的 `json.dumps(37.0)` 是 `37.0`。**宽松比较看不出**（`37 == 37.0` 为真），但会出现在响应体字节里 —— 前端用 `===` 严格比较、或把值拼进字符串（`scope: "drama-37.0"`）就会暴露。现 `js_number` 在 `\|x\| ≤ 2^53` 内返回 `int`（更大 JS 用指数形式，转 int 反而更不像） |
| ~~`aiConfigs` 的 8 个外部依赖端点未迁移~~ | **已迁移 7 个**：`/ollama/*`(4)、`POST /models`、`POST /test`、`GET /runtime/health` —— 它们依赖的只是**子进程与 HTTP**（`where ollama` / `Popen` / nvidia-smi / 厂商探测），Python 完全等价。我一度误判成"不可迁"，实际是**能迁但当时没排上** |
| **`/gpu/status` 与 `/gpu/release-all` 刻意不迁** | 它们依赖 423 行的 GPU 租约管理器，而租约是**进程内状态**：绞杀期两个后端会各有各的租约视图 ⇒ VRAM 协调失效。**真正的 GPU 工作（媒体生成）仍在 Node，租约就该跟它在一起**。等媒体域迁完再整体搬 |
| **`quick-preset` / `quick-local` 会覆盖既有配置** | 它们是 upsert（按 `service_type`+`provider`）。⇒ **冒烟测试跑在数据库副本上**没问题，但**不要对着真实库调这两个端点**（会覆盖用户配置）。真机验证时我只跑只读接口 |
| `export` 的 **5 个**端点未迁移 | `/edl`（需 ffprobe 探时长）、`/dramas/:id` 打包 ZIP（archiver）、`/jianying-draft`（339 行 + 时长）、`/qc-report`（ffprobe）、`/contact-sheet`（515 行 + 媒体探测）。已迁的是**纯 DB** 的工程账本两个端点 |
| `utils/storage.ts` 只搬了**纯存储**部分 | `readImageAsCompressedDataUrl`（缩放 + mozjpeg，需 Pillow）与 `downloadFile`（httpx 抓远程，调用方是媒体适配器）**未迁** ⇒ 后者会随媒体域一起做 |
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

## 工期参考

完整等价替换的实测估算：**单人对 AI 当助手 ≈ 99–152 人日**；AI 当主力（人只审阅 + 验证）
≈ 50–80 人日。**其中不可压缩的部分是验证，不是编码** ——
226 个端点冒烟、5 个 Agent 行为回归、17 家厂商适配真机跑通、前端零改动走查。
最短「能替换」单人也需 3~5 周（降级验收，只冒烟）。
