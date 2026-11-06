# 知识闭环：积累 → 自主学习 → 引擎自我完善 → 模型投喂与改造

> 目标：每多跑一集剧、多出一批图、多一本作品资料，系统就变聪明一点，并且**这个"变聪明"
> 是可查、可回滚、可证伪的**，而不是一句口号。
>
> 本文只写设计（落点、表、判据、阶段），不写「已经实现」。文末「已具备 / 缺口」按 2026-11-06
> 的代码实况列出，读的人能立刻分清哪些是现成的、哪些要新建。

---

## 0. 一条硬规矩先说在前面

**四个模块之间不互相调用，只共享三层数据。**

```
事实层（只追加，可重建）  →  知识层（有状态、有证据）  →  配置层（可回滚、生效即审计）
```

理由：任何"Agent 直接改引擎参数""引擎直接改模型权重"的直连，都会在下一次改动时变成
谁也说不清的死结；而三层数据是**单向流**的，每一层都能单独重放。本仓已有先例支持这个取向：
`script_library` 三只库是派生物（重跑会覆盖）⇒ 一切写操作只落覆盖层，读取时叠加；
`evaluation/optimizer.py` 的"候选 → 再评 → 赢了才落库"也是同一条思路。

---

## 1. 现状：哪些是真的，哪些是空的

### 已具备（可直接复用，不必重造）

| 能力 | 落点 | 可复用什么 |
|---|---|---|
| 运行轨迹落地 | `services/task_logger.py:187-219` → `<dataRoot>/traces/<scope>/<traceId>.jsonl` | `level/action/elapsedMs/meta`，meta 里已有 `agentType/episodeId/dramaId/toolCalls/protocolStatus/token` |
| 镜头级结果回执 | `video_quality_checks`（`core/models.py:477-493`） | `drama_id/episode_id/storyboard_id` + 4 个分数 + `issues/dimensions`，天然可聚合 |
| 成本与重试账本 | `api_usage`（`status/retry_count/cost_amount/is_local`）、`comfyui_runs`（`step/error_msg/warnings/elapsed_ms`） | 「成功率 / 返工率 / 单镜成本」三个指标现成 |
| 评测→优化状态机 | `agent/evaluation/optimizer.py:161-274` | evidence → candidate → 再评 → 接受/回滚，含防作弊（只回分数不回 detail） |
| 定时骨架 | `agent/evaluation_scheduler.py:40` | 每日 HH:MM、串行、防重入、失败隔离；**默认关** |
| 真训练执行链 | `services/lora_train/runtime.py:251` | 真起子进程，参数表/校验/增量日志/取消齐全 |
| Agent 工具注册单一来源 | `agent/runtime.py:326`（`_TOOL_FACTORIES`）+ `services/agent_registry.py` 的 `tool_set` | 加一个工具组＝一个装饰器 + 一个字面量 |
| 引擎参数外部化成纯数据表 | `engine/profiles.py:102-159`、`device_policy.POLICIES`、`samplers/schedules`、`technical_qc.TECH_QC_THRESHOLDS:69-79` | 已是"数据"，加一层 override 即可，不必改代码；**两处都已接线**（P3 覆盖层 + QC 伪档，见 §6.1） |

### 缺口（十个，按阻塞程度排）

1. **没有知识写入口** —— `agent/tools/` 8 组工具全只读（`library_tools.py` 只有一个
   `search_script_library`），全仓 grep `knowledge|write_note` = 0 命中。「知识积累」目前是空环。
2. **事实层不可查询** —— trace 只有 `scope`+`traceId` 两个维度，`/traces/stats` 只算 token；
   无按 drama/episode/shot、无按 `level=ERROR`、无按 `agentType` 的聚合。
3. **没有优化记录表** —— 优化史只落 `benchmarks/history/*.json`；`agent_configs`
   无 `version/score/parent_id`，回答不了「这次改动提升了多少」。
4. **优化只覆盖提示词** —— 参数、工具集、skill 绑定、QC 阈值、引擎参数都不在目标里。
5. **引擎无自我完善** —— 无人读 `comfyui_runs`/`video_quality_checks` 反推 `PROFILES` 或阈值；
   `pipeline.py:1262/1524` 的默认值是硬编码字面量。
   【2026-11-06 前半闭合】`tuning.suggest()` 已会读 `outcome_index` 的跑次出候选（失败聚集／步数平台期），
   **但只出候选、不自动应用**。同日补上**输入侧**：引擎任务封口（`runtime._worker_loop` →
   `_index_run`）与质检写回（`run_technical_qc` → `_index_qc_run`）都真往索引落
   `image`/`video`/`tts`/`qc` 跑次，技术 issue 也有了稳定 `code`（`technical_qc.ISSUE_CODES`）
   ⇒ 失败聚集不再是「真机上永远没有聚集」的空转。仍未做的只剩「按阈值统计命中率」——
   那要人工复核结论当分母（见 §4.3）。
6. **QC ↔ engine 单向断开** —— `engine/bridge.py` 对 `qc|quality` 零引用；QC 失败只触发
   `qc_retry.py` 重跑这一镜，不影响后续同模型的参数选择。
7. **训练链路无数据供给、无产物登记** —— 数据集靠人手填 `pathDir`；产物落
   `<dataRoot>/lora-train/tasks/<taskId>/`，**无 DB 登记、重启即丢**。
8. **评测面太小** —— 只有 4 份基准（`benchmarks/{extractor,script-rewriter,storyboard-breaker,voice-assigner}.json`），
   不评"成片好坏"。
9. **修复域不闭环** —— `repair` 提案只有 `pending/applied/rolled_back`，没有「应用后问题是否消失」；
   `diagnose` 结论不落盘。
10. **Skills 静态全量注入** —— 按 `priority` 排序 + 60000 字符预算，无触发词/命中率/历史效果选择。

---

## 2. 四模块的联动关系

### 2.1 数据流（单向，三层）

```
              ┌──────────────────────── 事实层（只追加，可重建）────────────────────────┐
              │  traces/*.jsonl    video_quality_checks    comfyui_runs    api_usage   │
              │  lora-train/tasks/*（要补登记）      repair/proposals/*.json             │
              └───────────────────────────────┬────────────────────────────────────────┘
                                              │ ① 归一（scripts/knowledge/rebuild_runs.py）
                                              ▼
                            data/knowledge/outcome-index.sqlite  ── runs 表
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │ ② 候选发现（确定性）      │ ② 候选发现（确定性）       │ ② 候选发现（确定性）
                    ▼                         ▼                         ▼
              高失败聚集                 高分离（好/坏各一批）        高频返工/超时
                    └─────────────────────────┼─────────────────────────┘
                                              │ ③ 归纳（LLM，只能引用给定 run_id）
                                              ▼
              ┌──────────── 知识层 knowledge-ledger.sqlite ─ entries ────────────┐
              │  kind: fact | recipe | lesson | failure | threshold | prompt_patch │
              │  status: proposed → accepted → deprecated/rejected                 │
              │  每条必须带 evidence(run_id 列表) + support/contra + lift           │
              └───┬──────────────────┬──────────────────────┬─────────────────────┘
                  │                  │                      │
        ④a 注入提示词/检索    ④b 引擎参数建议          ④c 编译微调数据集
                  │                  │                      │
                  ▼                  ▼                      ▼
            Agent 运行时      configs/engine/tuning.json   lora_train → model_registry
             （读+写提案）      （profiles 叠加 override）      （新权重登记）
                  │                  │                      │
                  └──────────────────┴──────────┬───────────┘
                                                │ ⑤ 下一次跑次
                                                ▼
                                       回到事实层（闭环）
```

### 2.2 职责与写权限（谁只能读、谁能写）

| 模块 | 读 | 写 | 触发 | 谁消费它的产出 |
|---|---|---|---|---|
| **知识积累** | 事实层全部 + 外部作品资料 | 知识层 entries（含 `fact`/`work` 类） | 跑次事件（完成/失败/QC 出分）+ 内容入库 | Agent 检索、引擎建议、数据集导出 |
| **Agent 自主学习** | 知识层（检索）、事实层（自查） | 知识层**只能写 `status=proposed`** | **每日批量复盘**（`knowledge_scheduler`，默认关）+ 检索命中后反馈 | 知识层（经审/判据转 accepted） |
| **引擎自我完善** | 事实层（runs 聚合）+ 知识层 `threshold` 类 | 配置层 `tuning.json`（带 backup）+ 提案 | 定时（每日）+ 样本量达标 | 引擎 `pipeline` 取参、QC 阈值 |
| **模型投喂与改造** | 知识层 + 高分成片/资产 | 配置层 `models`/`training_jobs` | 人工确认（默认关）或样本量达标 | Agent 选模型、引擎 `readiness` |

> **为什么不是"每次任务结束复盘一次"**（原 §3 草稿的写法）：单次跑次凑不出统计显著
> （`knowledge_distill.MIN_SAMPLE` 门槛下必然空转），真要写就只能靠"单例直觉"，
> 而那种条目一旦进 `accepted` 会污染后续**所有**提示词；反过来，失败/低分的原始事实已经
> 全量落在 `outcome-index`，日批一定看得到它 ⇒ **批量看、逐条留痕**比"当场随手记"更可信。
> 代价是"今天犯的错最快明天才进台账"—— 对"二十集之后不再犯同一个错"这个目标是够的。

**不能破的三条**：

1. Agent 不能直接写 `accepted`，也不能改 `tuning.json` —— 它只能提案；
2. 引擎不能自己改 `profiles.py` 源码 —— 只能写 `tuning.json`，`PROFILES` 保持"只描述"；
3. 训练产物必须登记（`models` 表）才能被 Agent/引擎选到 —— 没登记等于不存在。

---

## 3. 三个新库

### 3.1 `data/knowledge/outcome-index.sqlite`（事实索引，纯派生）

```sql
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,          -- trace_id / video_generation_id / comfyui run id，加前缀归一
  kind TEXT NOT NULL,               -- agent | image | video | tts | qc | train | repair
  scope TEXT,                       -- Agent / Video / Image …
  drama_id TEXT, episode_id TEXT, storyboard_id TEXT, shot_id TEXT,
  agent_type TEXT, model TEXT, stage TEXT,
  status TEXT NOT NULL,             -- ok | failed | partial
  score REAL,                       -- QC total 或自评
  cost_amount REAL, is_local INTEGER,
  elapsed_ms INTEGER,
  error_code TEXT,                  -- 归一后的错误类（不是原串，原串进 meta）
  trace_id TEXT, started_ns INTEGER, ended_ns INTEGER,
  meta TEXT
);
CREATE INDEX runs_by_episode ON runs(drama_id, episode_id, kind, started_ns);
CREATE INDEX runs_by_status  ON runs(kind, status, started_ns);
```

* **为什么独立**：它是纯派生（删掉重跑即可）。按 `scene_index.py` 同款理由，不进 `darme.db`
  —— 否则会被算进备份体积、被 `db_upgrade.py` 当项目结构补列、换数据根时莫名跑丢。
* **重建脚本**：`app/scripts/knowledge/rebuild_runs.py [--since=] [--dry-run]`，可断点续跑
  （以 `run_id` 去重，同一 run 后到的行覆盖先到的）。
* **双写不阻断生产**：`task_logger` 落盘后顺手插一条 runs；插入失败只 `warn`，绝不因为
  "索引没写成"让一次生成失败。

### 3.2 `data/knowledge/knowledge-ledger.sqlite`（知识真源，本仓可写）

```sql
CREATE TABLE entries (
  id TEXT PRIMARY KEY,              -- kind + scope + 归一标题 的 hash（同一件事只一条）
  kind TEXT NOT NULL,               -- fact | work | recipe | lesson | failure | threshold | prompt_patch
  scope TEXT NOT NULL,              -- global | drama:<id> | genre:悬疑 | stage:qwen_image | agent:script
  title TEXT NOT NULL,
  body TEXT NOT NULL,               -- 正文（recipe 用可执行步骤；failure 用"现象→根因→修法"）
  tags TEXT DEFAULT '',
  confidence REAL DEFAULT 0,        -- 0..1：由证据数、support/contra、lift 共同决定
  status TEXT DEFAULT 'proposed',   -- proposed | accepted | deprecated | rejected
  evidence TEXT DEFAULT '',         -- run_id / path 列表，必须逐条真实存在
  support INTEGER DEFAULT 0,        -- 被后续跑次证实
  contra  INTEGER DEFAULT 0,        -- 被证伪
  lift    REAL,                     -- 采纳前后主指标差（正=有用）
  source_kind TEXT,                 -- distill | agent | human | import
  source_ref TEXT,
  created_ns INTEGER NOT NULL, updated_ns INTEGER NOT NULL
);
CREATE TABLE entry_links (          -- 挂到剧/集/镜/阶段/模型上，支持"这部剧的经验"
  entry_id TEXT NOT NULL, ref_kind TEXT NOT NULL, ref_id TEXT NOT NULL,
  UNIQUE(entry_id, ref_kind, ref_id)
);
CREATE TABLE entry_usage (          -- 每条知识被注入过哪些 run，以及那次结果
  id INTEGER PRIMARY KEY, entry_id TEXT NOT NULL, run_id TEXT, used_ns INTEGER NOT NULL,
  outcome TEXT                      -- ok | failed | null（还没结论）
);
CREATE VIRTUAL TABLE entries_fts USING fts5(
  title, body, tags, content='entries', content_rowid='rowid', tokenize='trigram'
);
```

* **入口唯一**：`services/knowledge_ledger.py`（`upsert/search/detail/accept/deprecate/record_usage`）。
  路由、Agent 工具、蒸馏脚本都只调它，不各自写 SQL。
* **去重靠 id**：同一件事（同 `kind` + 同 `scope` + 归一标题）再次被提炼 ⇒ `support+1`，
  不新建。否则"经验"会以每天 10 条的速度膨胀成噪声。
* **`fact`/`work` 是外部知识**：上一轮设计的作品资料库（短剧/电影/电视剧的题材、主创、播出、
  评分、海报、关联）就是本表的 `kind='work'`。它与内部经验**同库不同 kind**，共用检索，
  这样「搜悬疑短剧的一般节奏」和「这部剧第 3 集为什么崩」能在一次检索里同时命中。
* **状态机**：`proposed`（Agent/蒸馏写的）→ `accepted`（人审，或满足自动判据）→
  `deprecated`（`contra>=2` 或 `lift<0` 或超过 180 天没被任何 run 用到）／`rejected`（人否决）。

### 3.3 `data/knowledge/` 下的配置层（不是库，是文件 + 一张登记表）

* `configs/engine/tuning.json` —— 引擎参数 override（见 §4.3），**每次写入前备份到
  `data/repair/backups/tuning-<ns>.json`**（复用 `repair` 已有的备份目录约定）。
* `models` 表（进 `darme.db`，因为它是项目资产的登记，不是派生物）：

```sql
CREATE TABLE models (
  id TEXT PRIMARY KEY, base_stage TEXT, kind TEXT,        -- lora | ckpt | gguf
  path TEXT NOT NULL, sha256 TEXT, dataset_ref TEXT,
  metrics TEXT,                                            -- 训练时基准分 / 抽样人工分
  status TEXT DEFAULT 'candidate',                         -- candidate | active | retired
  created_ns INTEGER NOT NULL
);
CREATE TABLE training_jobs (                               -- 补"重启即丢"的缺口
  id TEXT PRIMARY KEY, model_key TEXT, dataset_ref TEXT, params TEXT,
  status TEXT, output_path TEXT, sha256 TEXT,
  created_ns INTEGER, ended_ns INTEGER
);
```

---

## 4. 四个模块各自怎么落

### 4.1 知识积累

**落点**：`services/knowledge_ledger.py`（库）＋ `services/knowledge_distill.py`（提炼）
＋ `routers/knowledge.py` 增加 `/entries` 系列（列表/详情/审/驳回/统计）。

**候选发现全部是确定性的**（不花一次 LLM 调用，也就不会幻觉）：

| 模式 | 判据（跑在 `runs` 上） | 产出 kind |
|---|---|---|
| 失败聚集 | 同 `stage`（或同 `model`、同 `agent_type`）7 天内 `status=failed` ≥5 且占比 >40% | `failure` |
| 高分离 | 同 `stage` + 同 `agent_type`，`score` 前 10% 与后 10% 的差 >0.25（样本各 ≥10） | `recipe` |
| 返工热点 | 同 `shot_id` 的 `kind=video` 行数 ≥3（`take_budget` 之外的额外返工） | `lesson` |
| 成本离群 | 同 stage 的 `elapsed_ms` 或 `cost` P95 > 2×P50 且样本 ≥20 | `lesson` |
| 新内容入库 | 新剧/新作品资料齐全度不足（缺 logline/主创/评分来源） | `fact`（待补清单） |

**归纳一步才用 LLM，且只有一只"手"**：给模型的输入是**候选行 + 允许引用的 run_id 白名单**，
输出必须是 `{title, kind, body, evidence:[run_id]}`；写库前逐条校验 `run_id` 在 `runs` 里存在，
不在就**整条拒**（本仓「判不出来就拒绝」的既有纪律）。模型不能自己编 run_id，也就编不出"证据"。

**自动化分级**（默认最保守）：

| 档 | 行为 | 默认 |
|---|---|---|
| A | 只生成 `proposed`，人审后才 `accepted` | **开** |
| B | 满足阈值（evidence ≥3 且 support ≥2 且同类无 contra）自动 `accepted` | 关 |
| C | 直接进提示词/配置 | **永不**（没有这个档） |

### 4.2 Agent 自主学习

**补的第一块是"手"**：`app/agent/tools/knowledge_tools.py`

* `search_knowledge(query, kind?, scope?)` —— 读，走 `knowledge_ledger.search`（≥3 字 bm25 /
  1–2 字回落 LIKE 并如实回 `ranking`，与 `script_library` 同口径）；
* `propose_knowledge(title, kind, body, evidence)` —— **写，但只能写 `proposed`**，
  且 `evidence` 里每一个 run_id 都要在 `runs` 里查得到。

挂载：`agent_registry.AGENT_SPECS` 的 `tool_set` 不变，在 `runtime.py` 的
`_tool_factory("script")`/`("storyboard")`/`("voice")` 里各加一行 `create_knowledge_tools()`。

**第二块是"复盘"这一步**（真正的自主学习）：一次 Agent/生产任务结束、QC 出分之后，
用**同一条 trace 的 meta + QC issues** 拼一个极短的复盘任务（走本地 ollama 优先），要求输出
"这次哪一步起了作用/哪一步白做"，落成 `lesson`。约束：

* 只在 `status=failed` 或 `score` 低于该 stage 的 P25 时触发（成功案例不必复盘，否则一天几百条）；
* 单次预算：≤1 次 LLM 调用、≤1500 输出 token、超时 60s 直接放弃（不许拖住生产链路）；
* 复盘结果**不参与**本次任务的任何决策，只入库。

**第三块是"用后反馈"**：`entry_usage` 记「哪条知识被注入到哪个 run」。同一条知识注入后的
run 成功率 vs 未注入的基线 ⇒ `lift`。`lift < -0.05` 且样本 ≥10 ⇒ 自动 `deprecated`
（这条知识是负资产，留着有害）。

**Skills 的动态路由**（缺口 10）：不改 `SKILL.md` 的自述机制，只在 `load_agent_skills`
选择"注入哪些"时加排序依据 `priority × 历史命中率`，命中率从 `entry_usage` 同构的
`skill_usage` 读。默认仍全量注入 —— 排序只在超预算时生效（60000 字符预算一破就用它裁）。

### 4.3 引擎自我完善

**不改源码，只加一层 override。** `engine/profiles.PROFILES` 继续"只描述、不替创作者裁定"；
新增 `engine/tuning.py`：

```python
def override(stage: str, target: str, field: str, default: Any) -> tuple[Any, str]:
    """返回 (取值, 来源)。来源一定要回给调用方：override:<id> / profile / default。"""
```

`pipeline` 取参处一律走它，`plan` 回执里加 `source`，创作者一眼能看出"这次跑的是官方值
还是被调过的值"。**没有 `tuning.json` 时行为与今天逐字节相同**（这是回归判据）。

**建议怎么来**（`engine/tuning.py::suggest`）：

| 现象（来自 `runs`） | 建议 |
|---|---|
| 某 stage 的 `failed` 里 `error_code` 高度集中（如 OOM / 尺寸不整除） | 降 `mp`、改吸附步长 |
| 某 stage `score` 与 `steps` 正相关但 30 步以上增益 <0.01 | 降 `steps`（省时间不降质） |
| `TECH_QC_THRESHOLDS` 某条在 95% 的合格片上都被触发 | 该阈值与本机 ffmpeg 口径不匹配，建议改 |
| 某 sampler/schedule 组合的 `score` 显著更高（样本 ≥20） | 换默认组合 |

**三道闸门**（缺一条就会变成"自我损坏"）：

1. **样本量**：<30 只报不应用；
2. **影子验证**：候选值先在**同一批历史基准**上回放（引擎侧用 `_bench` 已有跑次；Agent 侧
   直接用 `evaluation/optimizer.py` 那套"候选 → 再评 → 赢了才落"），赢了才 `accepted`；
3. **可回滚**：`tuning.json` 每次写前备份，且 `applied` 记录带 `basis`/`evidence`；
   一条 `status=failed` 的后续跑次即可回滚（自动，且回执里明说回滚了哪条）。

**已落（2026-11-06）**：`tuning.suggest()` ＋ `GET /api/v1/knowledge-loop/tuning/suggest`
（`tests/tuning_suggest_test.py` 31 条）。四条现象规则里落了**前两条**，且都只报不应用：

- 失败聚集：某族 `failed` 过了样本闸门、且最大 `error_code` 占比 ≥50% ⇒ 出候选。
  50% 是刻意的下限 —— 三类各占 1/3 说明是三件事，并成一个结论就是编归因；
- 步数平台期：按 `runs.meta.steps` 分桶比均分，最高档与次高档差 <0.01 ⇒ 建议降回次高档
  （`meta` 里没 `steps` 就不算，**不拿 `elapsed` 反推步数**）。

**同日补的输入侧**（不做这一步，上面两条在真机上是瞎的：此前只有 agent trace 被
`ingest_trace` 灌进去，`image`/`video`/`qc` 跑次**没有任何生产代码写** ⇒ 夹具一撤就只会答
「没有哪一族的失败集中」——看着像"没问题"，其实是**没人喂它**；`tests/outcome_wiring_test.py` 28 条）：

- **引擎**：`runtime._worker_loop` 任务封口后调 `_index_run` ⇒ `run_id=engine:<taskId>`，
  `kind` 按**产物后缀**判（mp4/webm/mov/mkv⇒`video`、wav/mp3/flac/m4a/ogg⇒`tts`、其余⇒`image`），
  产物没落盘（失败/取消）时退回族表（只认确定是视频/音频的族，宁可粗一档也别编）；
  `meta` 带 `steps/width/height/frames/fps/seed/cfg/sampler/schedule/dryRun`（正是 `suggest`
  分桶的原料）；`error_code` **只取类别** —— `error.code|kind|type`，或
  `TorchBackendUnavailable: 人话` 这类标识符式前缀，整句话**宁可 `None` 也不当码**
  （"装不下 36.9 GiB > 22.49 GiB" 每次数字都不同 ⇒ 一失败一桶，规则永远"没有聚集"）。
- **质检**：`run_technical_qc` 写回记录之后调 `_index_qc_run` ⇒
  `run_id=qc:<storyboardId>:<videoGenerationId>`（**重跑覆盖同一条**，否则失败率被自己的
  重试顶高）、`score` 归一到 0–1（原值留 `meta.overallRaw`，因为 `suggest` 的 0.01 阈值是
  1 分制）、`error_code` 取第一条 `issue["code"]`；规则维度判失败没有码 ⇒ `meta.cause="rule"`
  如实标（**不编伪码**）。
- **两条硬纪律**：落账必须在 `with self._wake` **之外**（持锁写 sqlite 会把 `submit` 一起卡住，
  AST 判据盯行号）；整段 `try/except: pass`（P0 只加事实，出片与质检结论不受影响）。

**后两条有意没落**，`suggest()` 的 `notEvaluated` 逐条写明原因（别把它读成"没问题"）：
QC 命中率的 `code` 前提**已经补齐**（`ISSUE_CODES` 闭集 ⇒ `error_code` 可聚类），真正缺的是
「该判没判／判了是误报」的**人工复核结论**当分母（`video_quality_checks` 没有该字段）——
只有触发频次、没有误报率，算出来的"命中率"是假的；sampler/schedule 换默认等于换族语义，
`TUNABLES` 里有意没这两格。因此 `candidates[].canApply` **恒为 false**：影子验证（闸门②）
不在本模块，没做完就给"可应用"等于把闸门换成一个布尔值。

### 4.4 模型自动投喂与改造

分三层，**别混着叫"投喂"**：

| 层 | 做什么 | 落点 | 生效速度 | 风险 |
|---|---|---|---|---|
| L1 提示词注入 | 把 accepted 的 `recipe/lesson/prompt_patch` 挑 top-N 塞进 system prompt 的 `## Knowledge` 段 | `services/knowledge_feed.compile_for_prompt(agent_type, budget)` | 立即 | 低（可逐条关） |
| L2 检索注入 | Agent 主动 `search_knowledge` | `agent/tools/knowledge_tools.py` | 立即 | 低 |
| L3 权重注入 | 编译数据集 → LoRA 训练 → 登记 → 可被选用 | `knowledge_feed.export_dataset` → `lora_train` → `models` | 小时级 | **高**（必须人确认） |

**L1 的预算与排序**：只注入"标题 + 一句话"，全文留在库里按需检索；排序按
`lift × confidence` 取 top-N，`N` 由 `budget_chars` 决定（默认 1200 字）。**不许全量塞**
—— 60000 字符预算里塞满经验，模型会开始"背规则"，反而丢掉当前剧本的细节。

**L3 的数据集怎么来（这是缺口 7 的核心）**：

* 正样本：`video_quality_checks` 分数 ≥ P75 且人工未打回的成片 → 其
  （分镜提示词 / 角色设定 / 运行参数）作为 `(prompt, completion)` 或 caption；
* 负样本：`score` ≤ P25 且 `issues` 明确的 → 只作为**对照**进评测集，**不进训练集**
  （把废片当正样本教模型，是这类系统最常见的自毁方式）；
* 每条样本带 `source_ref`（哪一集哪一镜）与 `rights`（本仓自产/第三方，第三方默认不进训练）；
* 导出产物：`<dataRoot>/lora-train/datasets/<dataset-id>/{items.jsonl, manifest.json}`
  （manifest 记条数、来源分布、去重指纹）；
* 训练：`POST /api/v1/lora-train/train`（预览 `…/preview`）收 `datasetRef` ——
  指上面那个导出集，**核不过当场 400、任务不入队**（`knowledge_feed.resolve_dataset`）；
  ⚠️ 它**替代不了手填 `pathDir`**：训练器吃的是 `dataset_config` 那份 toml，导出集是 jsonl，
  两种格式。硬拼 argv 只会造出跑不起来的命令 ⇒ `datasetRef` 只回答「这次拿哪批数据训的」，
  不进命令行。可选值由 `GET /api/v1/knowledge-loop/datasets` 列出来（带 `itemsExists`），
  单条核验走 `GET /api/v1/knowledge-loop/datasets/resolve?ref=…`；
* 登记：任务与产物落 `training_jobs` / `trained_models`，两边都存 `datasetRef` 的**核验结果**
  （`fingerprint`/`contentSha256`/`integrity`/`itemsPath`/`count`/`roles`/`rights`）——
  产物也存一份是因为人从产物反查，
  不会先记得 `taskId`；样本文件被清掉时 `training_registry.problems()` 会点名（血统断了）；
  `status=candidate`；
  经一次基准对比（同批 case，提升 >2% 且无回退）才 `active`，可被 Agent 选模型时命中；
* **`models.status=active` 只影响"可选清单"，不自动替换默认模型** —— 默认模型仍由创作者配置，
  自动化最多做到"把它排到推荐位第一"。

---

## 5. 一次完整走位（把四模块串起来）

以「第 12 集镜头 27 的视频三次都被 QC 判糊」为例：

1. 三次 `video` 跑次落 `runs`（`storyboard_id=27`，`status` 依次 ok/ok/failed，`score` 0.51/0.49/0.42）；
2. `rebuild_runs` 增量刷入（也可实时双写）；`kind=video` 同 shot ≥3 行 ⇒ 命中「返工热点」；
3. 蒸馏器把 3 条的 `meta`（尺寸/步数/模型）与 QC `issues` 打包，LLM 产出
   `{kind: lesson, title: "夜景逆光镜头用 480p 出片易被判糊", evidence: [3 个 run_id]}`，
   校验 run_id 通过 ⇒ `entries(status=proposed)`；
4. 同一批数据里「`qwen_image/character-sheet` 28 步与 30 步 `score` 差 <0.01」⇒
   `tuning.suggest` 出 `steps: 30→28`，样本 42 条 ⇒ 影子回放通过 ⇒ 写 `tuning.json`（带备份）；
5. 下集同 stage 出图取参时走 `tuning.override` ⇒ 回执 `source=override:t-20261106-3`；
6. 该 `lesson` 被人审为 `accepted` ⇒ 下次 `storyboard` Agent 的 system prompt「## Knowledge」
   段里出现这一条（1200 字符预算内 top-N）；Agent 在写分镜时就避开"480p 夜景"；
7. 跑到 200 集时，`video` 高分成片累积到 300 条 ⇒ 导出数据集 ⇒ LoRA 训练 ⇒
   `models(status=candidate)` ⇒ 基准对比通过 ⇒ `active` ⇒ Agent 选模型时可见；
8. 若第 6 步之后同类镜头的失败率没降（`lift<0`）⇒ 该 `lesson` 自动 `deprecated`，
   并从提示词段里消失。**闭环的终点是"能被证伪并自动撤下"，不是"越积越多"。**

---

## 6. 分阶段落地（每阶段都能独立交付、独立验证）

| 阶段 | 内容 | 交付物 | 验收判据 |
|---|---|---|---|
| **P0 事实层** | `outcome-index.sqlite` + `rebuild_runs.py` + `/runs` 查询端点（按剧/集/镜/阶段/状态） | 3 文件 | 同一批 trace 重建两次结果一致；`/runs?episodeId=` 的条数与 QC 表对齐；索引插入失败不影响一次生成 |
| **P1 知识层 + 写口** | `knowledge-ledger.sqlite` + `knowledge_ledger.py` + `/entries` CRUD/审/统计 + `knowledge_tools.py`（读 + 提案） | 4 文件 | 提案写不进 `accepted`；不存在的 run_id 整条拒；同一经验二次提炼只 `support+1`；1–2 字中文能搜到并回 `ranking=like` |
| **P2 蒸馏 + 注入** | `knowledge_distill.py` + `knowledge_feed.compile_for_prompt` + 复盘步骤（默认关） | 2 文件 + 定时任务 | 关闭时提示词逐字不变；超预算时按 lift 裁且不超；`lift<0` 的条目会被自动 deprecated |
| **P3 引擎 override** | `engine/tuning.py` + `configs/engine/tuning.json` + `plan.source` | 2 文件 | 无 tuning.json 时 `PROFILES` 输出与今天逐字节相同；有 override 时回执 `source` 标明；回滚后恢复原值 |
| **P4 投喂与训练闭环** | `knowledge_feed.export_*` + `resolve_dataset` + `training_jobs`/`trained_models` 表 + `datasetRef` 参数 | 3 文件 + 迁移 | 导出集含来源与权利标记；负样本不进训练集；`datasetRef` 核不过当场 400（手改过、缺 rights/source、空集、地址不存在全拒；**条数/角色/权利都照旧、只换样本正文也算手改** —— 按 `items.jsonl` 的 sha256 判）；重启后任务清单**与数据集来路**都还在；`candidate` 不自动成为默认 |

**每阶段的默认开关都是关**（与 `evaluation.auto_optimize` 一致），由创作者逐项打开。
唯一默认开的是 P0 —— 它不改任何行为，只是把已有事实变得可查。

### 6.1 落地实况（2026-11-06，P0–P4 全部到位）

| 阶段 | 状态 | 落点 | 判据（主干 `tests/knowledge_loop_test.py` 131 条；各阶段另挂专测，见「落点」列） |
|---|---|---|---|
| P0 事实层 | ✅ | `app/services/outcome_index.py` + `app/routers/knowledge_loop.py` + **输入侧两处写点**（`engine/runtime.py::_index_run`、`technical_qc.py::_index_qc_run`，`tests/outcome_wiring_test.py` 28 条） | 同批 trace 重建两次总条数不变；没写 `status` 的记录**拒收**（不塞 `partial` 稀释失败率）；索引写失败只返回 `False`，不炸主链路。输入侧：引擎封口在 `with self._wake` **之外**落账（AST 判行号）、`kind` 按产物后缀、`error_code` 只取类别（整句/带数字的文案一律 `None`）；质检用确定性 `run_id` 覆盖同一条（重跑不堆行）、分数归一到 0–1、规则维度失败标 `cause=rule` 不编伪码；`record_run` 抛异常时两处写点均静默 |
| P1 知识层 | ✅ | `app/services/knowledge_ledger.py` + `app/agent/tools/knowledge_tools.py` | 假 run_id **整条拒**；`propose_knowledge` 只能写 `proposed`；2 字中文搜得到且如实回 `ranking=like`；`search_knowledge` 只转发不造分数 |
| P2 蒸馏 + 注入 | ✅（默认关） | `app/services/knowledge_distill.py` + `knowledge_feed.py` + `app/agent/runtime.py::append_knowledge_feed` + **定时任务** `app/services/knowledge_scheduler.py`（`tests/knowledge_scheduler_test.py` 23 条） | `KNOWLEDGE_FEED` 未设置时 system prompt **逐字不变**；注入挂在 `assemble_full_instructions` 这**唯一组装点**（AST 判据）；台账炸了只跳过；`record_injection` 在成功/失败两个出口都记。定时复盘：`knowledge.distill.enabled` 只能由显式 `true` 打开；关着时 `start()` 回「未启用」且**不挂空定时器**；一轮只写 `proposed`、**不调模型**、空索引如实回 0 条；上一轮未结束 ⇒ `skipped=running` 且不覆盖 `lastResult`；蒸馏炸了 `lift` 仍跑、错误进 `errors`、`ok=false`；`running` 必须在 **`finally`** 复位（结构化判据 + 把 `finally` 改成 `except` 的反验）；`lift` 独立开关（默认 off）；`arm` 的下一轮是未来时间且为 `…Z` 三位毫秒形、`stop` 后清空；`GET /scheduler` 关着也如实答 `enabled=false` |
| P3 引擎 override | ✅（无文件即不变；建议只读） | `app/services/engine/tuning.py` + `pipeline.build_plan` 调用口 + **QC 伪档**（`technical_qc.qc_thresholds()`，`tests/qc_thresholds_test.py` 28 条）＋ **只读候选**（`tuning.suggest()` / `GET /tuning/suggest`，`tests/tuning_suggest_test.py` 31 条） | 无 `tuning.json` ⇒ 返回**同一个对象**、零 notes；显式 `steps=8` 不被覆盖；`plan.sigmas` 与 `plan.steps` **同源**（不是事后改 `plan.steps`）；`report()` 标 `source=override:…`。QC 九个阈值走同一张 `TUNABLES`（族 `qc`）：无文件时**逐值等于** `TECH_QC_THRESHOLDS` 且返回副本（写不回常量）；覆盖真进 ffmpeg 命令行（`blackdetect=d=0.5`）与用户可见文案；越界/非数字/typo 键**退回默认**并由 `problems()` 点名；写入口缺 `evidence`、证据路径不存在、越界一律拒；`apply_to_request("qc", …)` 原样返回零 notes（请求上没有这些字段，伪档只能被质检读，不会被换进生成参数）。只读候选：连跑两次 `tuning.json` 逐字节不变、AST 判「除四个写口外无人调用它们」、<样本闸门只报、`canApply` 恒 false、候选带的 `run_id` 过 `check_evidence`、>500 条如实 `sampled=true` |
| P4 训练登记 | ✅（候选不自动升） | `app/services/training_registry.py` + `lora_train/runtime.py` 三处调用口 | 新实例（=重启）`list_tasks` 仍看得到历史且标 `restored`；扫盘产物一律 `candidate`；`promote` 缺 basis/缺证据/文件不在盘上**全拒**；同类只留一个默认档；重扫盘不许把人工结论打回 `candidate`；`datasetRef` 核不过当场拒（条数/角色/权利对不上、空集、地址不存在），核过的来路随任务**与产物**落库、重启后仍在；HTTP 面真发请求：`GET /datasets` 列得出可选 ref（带 `itemsExists`）、`GET /datasets/resolve` 核得过 200／核不过 **400 + why**、缺 `ref` 参数 400 且点名 `ref` |

四条纪律（对应上面的判据，改代码时别破坏）：

1. **每个写口都不许把失败传出去** —— 统计/登记坏了最多是「查不到」，不能拖垮一次真生成或真训练。
2. **回执必须带上「谁改的」** —— P2 注入写进 `plan.warnings` 式的分节标题、P3 写「参数来源: tuning:…」、P4 写 `restored: true`。没有回执的自动化等于悄悄改行为。
3. **候选与默认分开** —— P3 的覆盖要人给 `basis`+`evidence`，P4 的产物默认 `candidate`，两者都不许由代码自行"提升"。
4. **只认白名单** —— P3 的 `TUNABLES` 逐族登记（`steps` 在 seedvr2/wan22/krea2 语义各不相同），P4 的产物 `kind` 不猜。
5. **引用只记核得过的** —— `datasetRef` 不是备注字段：核不过就 400，**不许把没核的字符串写进登记表**。
   一段随便写的 ref 三个月后会变成「可能是这批、也可能不是」的假血统，比没有记录更坏。

---

## 7. 关键实现要点（踩过坑才写得出来的那些）

1. **trigram 只索引 3 字符以上**：2 字中文（「悬疑」「糊」）在 trigram 下**一条都命中不到且不报错**
   ⇒ 必须 ≤2 字回落 `LIKE` 并如实回 `ranking`。同一坑在 `script_library` 已经踩过。
2. **多值字段禁用 `LIKE '%x%'`**：题材「犯罪」会命中「犯罪心理」⇒ 一律
   `instr('|' || v || '|', '|x|') > 0`（与 `script_library` 的 `instr` 纪律同源）。
3. **`path`/`LIKE` 的长度 3 纯 ASCII 陷阱**：fts5 虚拟表的 `path` 是 `UNINDEXED`，
   `LIKE '%doc%'` 静默 0 条、`instr` 才是 30923 条。任何打在 fts5 表上的过滤都用 `instr`。
4. **证据必须可核**：`evidence` 里的每个 id 都要在 `runs` 里查到，查不到就拒写入 ——
   这比"提示词里写『不要编造』"有效一万倍。
5. **同一经验只一条**：`id = hash(kind, scope, 归一标题)`，命中即 `support+1`。
   去重键写进模块头，未来改判据要一起改。
6. **`lift` 要有基线**：注入组的成功率必须与**同期未注入**的同类 run 比（不是与"上月"比），
   否则季节/题材变化会被误读成知识有效。
7. **override 必须回显来源**：`plan.source` / `prompt.sources` 里不写清楚，创作者会以为
   模型"自己变差了"。这是本仓「回执纪律」的延伸：`available` / `onDisk` / `ranking` 三者
   分开报，这里就是 `source` 必须单列。
8. **负样本不进训练集**：只进评测集。训练集污染是不可逆的（权重里洗不掉）。
9. **定时任务串行且防重入**：复用 `evaluation_scheduler` 的形态（已有防重入、失败隔离），
   不要新写一个定时器。但**别连它的缺陷一起抄**：那一份把 `running` 复位写在正常路径上
   （没有 `try/finally`），循环之外抛错就永久卡在 running ⇒ 此后每一轮都被「防重入」
   静默吃掉，而回执依然写着 `enabled=true`。新写的定时器（`knowledge_scheduler.py`）
   一律 `try/finally` 复位，并有结构化判据 + 反验盯着这一行。
9b. **无人值守的自动化不许花钱、不许改结论**：定时复盘只写 `proposed`、**不调模型**
   （夜里自动烧模型调用＝账单事故；无人看的"经验"第二天只会被批量点通过）；
   `lift` 单独开关默认关 —— 蒸馏只**加一条待审**，lift 会**撤一条旧结论**，两种力度不绑一起。
10. **一切可回滚**：`tuning.json` 有备份、`models` 有 status、`entries` 有 status、
    `repair` 有 `rolled_back`。没有回滚路径的自动改动一律不做。
11. **`CREATE TABLE IF NOT EXISTS` 不会给老库加列**：登记表加了 `dataset_json` 之后，
    已经存过训练的库不会报错、只会永远查不到来路 —— 必须靠 `_ensure_columns()`
    每次连库 `PRAGMA table_info` 后补 `ALTER TABLE`（幂等）。「看起来正常的空字段」
    比缺列更难发现。
12. **`datasetRef` 不进 argv**：训练器吃 `dataset_config` 的 toml、导出集是 jsonl，
    硬拼只会造出跑不起来的命令；它只做来路登记，所以**必须当场核验**（`resolve_dataset`）。
    也别把它塞进 `values` —— 那里逐键过 `options.merge`，未知键当场报错。
13. **指纹不是内容哈希**：`fingerprint` 只对样本的 `source` 取值（那是「集合里有哪些条」），
    所以「条数、角色、权利、指纹全对上、只把某条的正文换了」它一路绿灯 —— 这正是最常见的
    一种手改（有人觉得某句提示词写得不好，就地改）⇒ 必须另存整份 `items.jsonl` 的
    `contentSha256` 并按字节核；老 manifest 没有该字段时**如实降到「只核了指纹」**
    （`integrity` 字段），不许假装核过。同理：转格式回执里的 `backup` 要存**当次真用的**
    路径，别在收尾处用时间戳再算一遍（跨秒就会报出一个不存在的目录，人照着找备份会扑空）。
14. **核验失败回 400，不回 `200 {ok:false}`**：`/datasets/resolve` 核不过是 **400 + `why`**，
    前端才敢用 status 分流（200 里塞 `ok:false` 会让所有调用方都得多写一层判断，总有人忘）。
    顺带一个写测试时的坑：本仓 `RequestValidationError` 被全局处理器统一成 **400 带 message**，
    所以「缺参数」的判据是 400 而不是 FastAPI 默认的 422 —— 照默认值断言会红，且红得莫名其妙。

---

## 8. 明确不做（避免过度设计）

* **不做**"全自动改源码"：`profiles.py`/`pipeline.py` 的默认值只通过 `tuning.json` 覆盖，
  不改文件；
* **不做**"训练无人值守"：L3 必须人点一次（成本与不可逆性都不允许全自动）；
* **不做**"知识全量注入"：只 top-N + 按需检索；
* **不做**"向量库"：现有 bm25 + trigram + 维度筛选在本仓的数据量级（万级条目）足够，
  引入 embedding 会多一套部署与一致性负担（`script_library` 已有的 RRF 融合就是为此克制过）；
* **不做**"跨项目共享经验"：`scope` 已按 `drama:` 隔离，跨项目默认不互见
  （不同题材/画风的经验互相污染比稀缺更糟）。

---

## 9. 与既有模块的接线点（改哪里，不动哪里）

| 现有模块 | 改动 | 不动 |
|---|---|---|
| `services/task_logger.py` | 落盘后加一次 runs 双写（失败只 warn） | JSONL 格式与字段 |
| `routers/traces.py` | 新增 `/runs` 聚合查询（转调 `outcome_index`） | 现有 `/traces*` 端点 |
| `agent/runtime.py` | `_tool_factory` 加一行 `create_knowledge_tools()` | 主循环与重试 |
| `agent/qc_retry.py` | 重跑结束追加一次"复盘提案"调用（异步、超时即弃） | 只重写失败镜的既有纪律 |
| `engine/profiles.py` | 不改；`pipeline` 取参处已改走 `tuning.resolve`／`apply_to_request` | `PROFILES` 表本身 |
| `services/engine/tuning.py` | **已接读向**：`suggest()`（失败聚集／步数平台期，三道闸门如实标）＋ `GET /api/v1/knowledge-loop/tuning/suggest`（`tests/tuning_suggest_test.py` 31 条） | 候选**永不自动应用**：`canApply` 恒 false，人核过再 `set_field` |
| `services/technical_qc.py` | **已接**：`qc_thresholds()` 逐键过 `tuning.resolve("qc", …)`，回执带 `thresholds-overridden` 警告（`tests/qc_thresholds_test.py` 28 条） | 阈值默认值（`TECH_QC_THRESHOLDS` 仍是唯一真源）与 ffmpeg 判据 |
| `routers/lora_train.py` | 加 `datasetRef` 参数与 `training_jobs` 登记 | 现有参数表与执行链 |
| `routers/knowledge.py` | 挂 `/entries` 子路由（新 router 文件，避免与语料端点混） | `/search` `/documents` `/upload` 等 |
