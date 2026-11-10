# 知识闭环：积累 → 自主学习 → 引擎自我完善 → 模型投喂与改造

> 目标：每多跑一集剧、多出一批图、多一本作品资料，系统就变聪明一点，并且**这个"变聪明"
> 是可查、可回滚、可证伪的**，而不是一句口号。
>
> 本文只写设计（落点、表、判据、阶段），不写「已经实现」。文末「已具备 / 缺口」按 2026-11-06
> 的代码实况列出，读的人能立刻分清哪些是现成的、哪些要新建。
>
> 台账**管理面**（条目的编辑/删除/恢复、版本与回滚、标签与分类、审计、权限声明）的
> 接口契约不在本文，见 `docs/knowledge-base-contract.md`（端点表、输入输出、边界、
> 错误码分工、验收标准与测试要点；该文档与 openapi 双向受测，改契约不同步会红）。

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

### 缺口（十二个，按阻塞程度排）

> **2026-11-08 复核口径**：十二格已全部闭合（每格下面的【…闭合】就是落点与守卫位置）。
> 唯一还空着的是缺口 4 里那半句「引擎参数」—— 它是**按设计**不进这张表的：QC 阈值与
> 采样档走 §4.3 那条「质检反哺引擎参数」的链（`engine/profiles` 覆盖层 + `technical_qc`
> 伪档），进 `agent_optimizations` 会把「跑次经验」与「档位参数」两套口径混进一张表。
> 所以往下读这批条目时请把它当**沿革**看，不要当成待办列表 —— 真要开新活，
> 看 §7 的「已知边界」与各节末的 ⚠️ 段。

1. **没有知识写入口** —— `agent/tools/` 8 组工具全只读（`library_tools.py` 只有一个
   `search_script_library`），全仓 grep `knowledge|write_note` = 0 命中。「知识积累」目前是空环。
   【2026-11-06 缺口闭合·写侧，2026-11-08 补读侧】`agent/tools/knowledge_tools.py`
   （`search_knowledge` / `context_knowledge` / `propose_knowledge`），在 `runtime` 三条
   工具装配线上真接（`_as_registry(create_knowledge_tools(...))`）。写侧**只能写 `proposed`**：
   状态机 `proposed → accepted → …` 由人审推进，Agent 只提议不拍板；`evidence` 逐条可核
   （`outcome_index` 的 run_id 或盘上真实路径），核不过整条拒收，不是"先存下来再标可疑"。
2. **事实层不可查询** —— trace 只有 `scope`+`traceId` 两个维度，`/traces/stats` 只算 token；
   无按 drama/episode/shot、无按 `level=ERROR`、无按 `agentType` 的聚合。
   【2026-11-06 缺口闭合·事实层，端点是**另一个**】这一条的真问题不是"没数据"而是
   "问错了地方"：`traces/**.jsonl` 是**原始事件流**（只读、只答 token 与回放），
   要按剧/集/镜/阶段/状态/agentType 聚合的是**派生索引** `outcome_index`
   （`data/knowledge/outcome-index.sqlite`，可由 traces 重建）——
   `GET /runs` 按七维筛（`drama_id`/`episode_id`/`storyboard_id`/`shot_id`/`stage`/`status`/`agent_type`，
   `total` 与 `items` 分开读）、`GET /runs/stats?groupBy=stage|status|kind|model|agent_type|drama_id…`
   给条数/失败率/平均分/耗时 P50·P95，`groupBy` 非法**明确 400 不静默回全量**。
   `/traces/stats` 保持只有 scope ＋ token **是有意的**（两个口径混在一张表里，
   日后没人分得清"这是原始观测还是归一结果"）。见 §2.1 与 §3.1。
3. **没有优化记录表** —— 优化史只落 `benchmarks/history/*.json`；`agent_configs`
   无 `version/score/parent_id`，回答不了「这次改动提升了多少」。
   【2026-11-07 缺口闭合】新表 `agent_optimizations` ＋ `services/optimization_log.py`
   ＋ 只读端点 `GET /api/v1/knowledge-loop/optimizations` ＋ 页面「优化史」分区 —— 见 §4.2 末。
4. **优化只覆盖提示词** —— 参数、工具集、skill 绑定、QC 阈值、引擎参数都不在目标里。
   【2026-11-07 第一格闭合】参数（`temperature`）已进目标：`agent/evaluation/param_optimizer.py`
   ＋ `POST /api/v1/evaluation/tune/{case_id}` ＋ 优化史表新开 `target="params"` —— 见 §4.2「参数调优」。
   【2026-11-08 第二格闭合】skill 绑定已进目标：`agent/evaluation/skill_optimizer.py`
   ＋ `POST /api/v1/evaluation/tune-skills/{case_id}` ＋ 优化史表新开 `target="skills"` —— 见 §4.2「技能绑定调优」。
   【2026-11-08 第三格闭合】工具集已进目标：`agent/evaluation/tool_optimizer.py`
   ＋ `POST /api/v1/evaluation/tune-tools/{case_id}` ＋ 优化史表新开 `target="tools"`
   —— 见 §4.2「工具集调优」。前置的「单步对照」评测形态也同日落地
   （`evaluator.step_trace()` ＋ 五个评测分支全部经 `_report_steps()` 收口），
   判据是**两把尺子**：整轮分 ＋ `unknownTool` 计数 —— 分数没掉但模型仍在调那个被摘掉的工具
   记 `tool_wanted`（不摘），只看分数就会把"真需要的工具"删掉。
   ⚠️ 这一格**只能留结论、不能落库**：工具集真源是代码里的装配语句，`agent_configs` 没有
   tools 列 ⇒ 恒 `autoPersist=false` ＋ `persisted=false` ＋ `persistSkip` 说明理由。
   仍缺：引擎参数（QC 阈值走 §4.3 那条链，不进这张表）。
5. **引擎无自我完善** —— 无人读 `comfyui_runs`/`video_quality_checks` 反推 `PROFILES` 或阈值；
   `pipeline.py:1262/1524` 的默认值是硬编码字面量。
   【2026-11-06 前半闭合】`tuning.suggest()` 已会读 `outcome_index` 的跑次出候选（失败聚集／步数平台期），
   **但只出候选、不自动应用**。同日补上**输入侧**：引擎任务封口（`runtime._worker_loop` →
   `_index_run`）与质检写回（`run_technical_qc` → `_index_qc_run`）都真往索引落
   `image`/`video`/`tts`/`qc` 跑次，技术 issue 也有了稳定 `code`（`technical_qc.ISSUE_CODES`）
   ⇒ 失败聚集不再是「真机上永远没有聚集」的空转。
   【2026-11-07 缺口闭合】「按阈值统计命中率」已接上**人工复核结论**当分母（新表 `qc_reviews` ＋
   `services/qc_reviews.py` ＋ `GET/POST /api/v1/knowledge-loop/qc/*`），`suggest()` 的第三条规则
   （质检命中率）随之落地 —— 见 §4.3。仍未做的只剩「换 sampler/schedule 默认」（那不是阈值、
   是族语义，`TUNABLES` 里有意没这两格）。
6. **QC ↔ engine 单向断开** —— `engine/bridge.py` 对 `qc|quality` 零引用；QC 失败只触发
   `qc_retry.py` 重跑这一镜，不影响后续同模型的参数选择。
   【2026-11-07 缺口闭合】根因不是"少一条规则"而是**归因断了**：QC 行落在伪档 `video_spec`、
   引擎行落在真档 `h3`/`wan22`/`krea2`，两边**没有共同键** ⇒ 「这一族最近质量掉了」从来
   成不了一条**关于这一族**的结论。四条落地：① `outcome_index.records` 表（`link_record`/
   `record_of`）—— `bridge.run_job` 在**刚拿到 `taskId` 的那一行**登记
   `(记录类型, 记录 id) ⇒ (引擎档, 引擎跑次)`；② `technical_qc._index_qc_run` 写回时查它，
   把 `producerStage`/`engineRunId` 放进 QC 行的 `meta`（**查不到就不带这两格**，不填 `null`）；
   ③ `app/services/engine/qc_feedback.py`：按档聚合通过率、按 `meta.steps` 分桶、
   给闸门 `gate(high, low)` 三档 `insufficient`/`pass`/`blocked`；
   ④ `tuning._steps_candidates` 出「降步数」候选**之前**问闸门：`blocked` 当场不出并留 note
   （不是静默丢弃），放行则把 `qcGate` 随候选带出去 —— **这才是"影响后续同模型的参数选择"**
   那一格。顺带查实一个真缺陷：引擎跑次根本不记 `score`（`runtime._index_run` 无此字段）⇒
   那条"步数平台期"规则在真机上一向空转，QC 通过率才是真能判的量。口实与坑见
   §4.3 末「质检反哺引擎参数」与 §7 第 17 条。
7. **训练链路无数据供给、无产物登记** —— 数据集靠人手填 `pathDir`；产物落
   `<dataRoot>/lora-train/tasks/<taskId>/`，**无 DB 登记、重启即丢**。
   【2026-11-06 缺口闭合】两半都补上了，落法见 §4.4「L3 的数据集怎么来」：
   ① **数据供给** `knowledge_feed.export_dataset`（导出集 `items.jsonl`+`manifest.json`，
   带 `source_ref`/`rights`/去重指纹，负样本**只进评测集不进训练集**）＋
   `resolve_dataset` 核验（核不过当场 400、任务不入队）＋ `GET /datasets` 列可选值；
   ⚠️ `datasetRef` **替代不了手填 `pathDir`**（训练器吃 toml、导出集是 jsonl，硬拼 argv 只会
   造出跑不起来的命令），它只回答"这次拿哪批数据训的"。
   ② **产物登记** `services/training_registry.py`（`training_jobs` / `trained_models` 两表，
   `record_job` / `scan_artifacts` / `promote` / `demote`），两表都存 `datasetRef` 的核验结果
   （血统断了 `problems()` 会点名）；产物状态默认 `candidate`，
   经一次同批 case 基准对比（提升 >2% 且无回退）才 `active`，
   且 `models.status=active` **只影响可选清单、不自动替换默认模型**。
8. **评测面太小** —— 只有 4 份基准（`benchmarks/{extractor,script-rewriter,storyboard-breaker,voice-assigner}.json`），
   不评"成片好坏"。
   【2026-11-07 缺口闭合】两半都补上了：
   ① **成片评测**：`app/agent/evaluation/film_check.py` ＋ `POST /api/v1/evaluation/film/{episode_id}`。
   它**读数不重量**——判据全取自已发生的事实（`storyboards` / `video_generations` /
   `video_quality_checks` / `qc_reviews`），**不重跑 ffmpeg、不调模型**（重跑会与质检页的口径打架，
   且每条片子几十秒）。四维等权（覆盖 / 规格 / 缺陷 / 复核），**判不了的维度不进总分**
   （`judged=false` 折成 0 是谎报"差"、折成满分是谎报"好"），`missing` 逐项点名、`verdict`
   降成 `partial`；`review` 的分母只含 `hit+false_alarm`（漏报单列，没有 issue 行可挂）；
   缺陷面的分母是**有 QC 的镜**而不是全部镜；跑次幂等 `run_id=film:<集>`、`kind="eval"`
   （混进 `agent` 会污染 agent 成功率）。
   ② **第 5 类基准 `grid_prompt`**（`benchmarks/grid-prompt.json`）：它卡在"产出层"与"成片层"
   之间——宫格提示词写砸了，账最后记在成片维度上，而那时视频生成的钱已经花掉。
   评分 `scorer.score_grid_prompts` 五维全硬判据（长度 / 格数吻合 / 中文占比 / 占位符残留 /
   格间不重复），产出解析走 `services/grid_payload.py`（从 `routers/grid.py` 上收的**单一真源**），
   执行时把基准镜头**真落成 `storyboards` 行**（宫格 Agent 的工具只按 id 查库）。
   口实与坑见 §7 第 18 条。
9. **修复域不闭环** —— `repair` 提案只有 `pending/applied/rolled_back`，没有「应用后问题是否消失」；
   `diagnose` 结论不落盘。
   【2026-11-07 缺口闭合】`services/repair_closure.py`（快照 / 缺口键 / 四档判定 / 跑次记录）
   ＋ `routers/repair.py` 五处接线：`GET /diagnose` 默认**顺带落快照**（`persist=false` 是
   「只查不记」的口）、`GET /diagnose/history` 回答「缺口在变少还是变多」（同 `fingerprint`
   即同一批问题）、`POST /propose` 收 `finding` 把「这条提案修的是哪条缺口」钉在提案上、
   `POST /apply` 写盘成功后**当场复诊**、`POST /verify` 给「改动加了但模型文件要人去下」的
   场合事后补判、`POST /rollback` 覆盖成终态。判定四档 `resolved / still / stale / unverified`
   （＋终态 `rolled_back`），结论按 `kind="repair"` 写进 `outcome_index`。口实与坑见 §7 第 15 条。
10. **知识只按"条目"存在，节点之间没有边** —— 蒸馏每天写下 `stage:video` / `shot:SHOT-012`
   的关联行，但**全仓没有一处读 `entry_links`**：一条知识挂在哪个阶段上、哪些条子是同族的、
   "这一镜踩过什么坑"三个问题都问不出来。
   【2026-11-08 缺口闭合】`knowledge_ledger.related()` / `for_context()` / `context_refs()`
   ＋ 两个只读端点 ＋ Agent 工具 `context_knowledge` ＋ 详情面板「挂载／相关知识」。口实与坑见 §11。
11. **Skills 静态全量注入** —— 按 `priority` 排序 + 60000 字符预算，无触发词/命中率/历史效果选择。
    【2026-11-07 缺口闭合·半格】`services/skill_usage.py`（派生统计库
    `data/knowledge/skill-usage.sqlite`，与 `entry_usage` 同构）＋ `agent/skills.py`
    超预算裁剪 ＋ `agent/runtime.py` 跑次收口回填 ＋ `GET /api/v1/skills/usage`。
    **触发词选择有意不做**（那要引入检索层，与「`SKILL.md` 自述」的机制冲突）；
    「历史效果」只落到**可观测的那一半**：注入了哪些 / 各多少字符 / 哪些被裁 / 该跑次成败。
    ⚠️ **没有命中率**——本仓没有「模型读没读它」的通路，`okRate` 只能读成「注入它的跑次成功率」。
    预算够时行为逐字不变，只有装不下才用 `priority + 降级惩罚` 改裁剪次序。口实与坑见 §7 第 16 条。
12. **投喂的作用域键域与台账对不上** —— 写入口给的是 `EP<id>` / `DRAMA<id>`（Agent 工具）
   与 `stage:<name>` / `shot:<id>`（蒸馏），而 `compile_for_prompt` 只认 `global` 与
   `agent:<type>` —— 后者**全仓没有一处写**。于是「同一集里刚被采纳的经验」在下一次同集跑次里
   依然注入不进去，且回执只说「还没有 accepted 的知识」，看起来像库空。
   【2026-11-08 缺口闭合·集/剧集两档】`compile_for_prompt(scopes=…)` ＋
   `scopes_for(episode_id, drama_id)` ＋ `assemble_full_instructions` 用手上真值算 ⇒
   生产侧自动带上 `EP<id>` / `DRAMA<id>`；回执新增 `scopes` / `blocked`，把
   「有知识但作用域不匹配」与「确实还没积累」分成两种空。
   `stage:` / `shot:` 有意**不自动加**：agent 跑次手上没有「我属于哪个引擎档 / 哪一镜」的真值，
   编一个只会让投喂看起来在定向、其实在乱猜 —— 那两档走 Agent 工具 `context_knowledge`（§11）。
   口实与坑见 §12。

---

## 2. 四模块的联动关系

### 2.1 数据流（单向，三层）

```
              ┌──────────────────────── 事实层（只追加，可重建）────────────────────────┐
              │  traces/*.jsonl    video_quality_checks    comfyui_runs    api_usage   │
              │  lora-train/tasks/*（要补登记）    repair/{proposals,diagnose}/*.json    │
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

skill 侧有同构但**不同库**的一张表 `skill_usage`（`data/knowledge/skill-usage.sqlite`，
`services/skill_usage.py`）：记注入与跑次成败，**不记"读没读"**（没有那条通路），
也**没有 `lift`** —— 不存在"没注入任何 skill 的对照组"，算不出干净基线，所以只给 `okRate`
并把它标成相关信号（缺口 10，见 §4.2 与 §7 第 16 条）。

**优化史表（缺口 3，2026-11-07 落地）**：`agent_optimizations`（`core/models.py`）
＋ `services/optimization_log.py` ＋ `GET /api/v1/knowledge-loop/optimizations`（只读）。
五条口径写成硬约束，都是为了让「提升了多少」这个数**不能被读错**：

* **一版一行、追加不覆盖**。`agent_configs` 是覆盖式落库（同一 agent 只有当前一行），
  `benchmarks/history/*.json` 又按 case 整段覆盖 —— 两者都答不了「v3 比 v2 强多少」。
  故每轮优化独立 insert，`version` / `parent_version` 显式成链。
* **`accepted` 与 `persisted` 是两个布尔**。「分数严格更高」（判决）与「真写成默认值」（生效）
  是两件事：前者只是评测结论，后者才改变后续出图行为。合成一个必然出现
  「页面说已采纳、而默认提示词还是旧的」。`mark_persisted()` 是唯一置真点，且**必须**排在
  `persist_agent_config()` 之后（AST 判据盯顺序）。
* **`mark_persisted()` 的 WHERE 必须带 `target`（2026-11-09 修）**。四格的 `version` 是**各自**
  从 1 数的（提示词格按 `history` 的 best 递增，另三格走 `param_optimizer._next_version`），
  同一 `agent_type` 上「提示词 v1」与「温度 v1」并存是常态。只按 `(agent_type, version, case_id)`
  定位就会把「提示词落库」标到温度那行头上 —— 页面显示「温度调优也生效了」，而真正被写进
  `agent_configs` 的那一版在优化史里查不到生效痕迹。**不抛异常、不改任何数值**，只把生效标错人。
  三处调用方（`optimizer.py` / `param_optimizer.py` / `skill_optimizer.py`）都**显式**传
  `target`，判据双管：`optimization_log_test.py` ④′ 真跑跨格（同 agent 同 version 两行只标一行、
  缺那格 ⇒ 0 行）＋ ⑨′ 全仓 AST 扫每处调用都显式给且与所在文件身份一致。
  注意 `_next_version` 跨目标取 max **不是**这条护栏：提示词那格的号不经过它，撞号躲不开。
* **`delta` 服务端算，算不了就 `None`**。`score_before` 取的是**当时 best 分数**（不是
  Reference 分数），否则连续晋升会把同一段涨幅重复计入。首版没有对手 ⇒ `score_before`
  为空 ⇒ `delta=None`，**不写 0**（写 0 页面就会显示「提升 0」，那是编数）。
* **只存指纹不存全文**：`prompt_sha` 是提示词 sha1 前 16 位，全文仍在
  `benchmarks/history/<agent>/<case>.json` —— 这张表是给人看版式的，不该变成第二份提示词仓库。
* **写口不抛、读口不许返空表**。`record()` 失败只 `log_task_warn`（记账失败绝不能回滚一次
  真晋升）；反过来读不出来就 `available=false` ＋ `reason`，页面原样透出那句病因，
  而不是拿一张空表说「没有优化」。

`target` 是闭集，现有四格：`system_prompt`（提示词优化器）、`params`（参数调优，见下）、
`skills`（技能绑定调优，见下文那一节）与 `tools`（工具集调优，见下文那一节），
其余仍未开（留列不留假数据）。

**四格判据对照（2026-11-09 横向核过一遍，别只看某一格）**。同一个问题——「分数变了，
是改动起作用还是模型抖了一下」——四格的答法**不是同一个**，这张表就是答案：

| 格 | 优化器 | 噪声口径 | 判决理由码 | 效力 | 调用面 |
| --- | --- | --- | --- | --- | --- |
| `system_prompt` | `optimizer.py` | **可选**：`repeat=1`（默认）零容差、与历史逐字一致；`repeat=2–5` 取 **Reference 重复跑**的极差 `max−min` 作抖动带（`jumpKind="within"`） | `accepted` / `no_gain` / `runtime_mismatch` / `eval_failed`（单轮跑崩只作废那一轮） | 真写 `agent_configs.system_prompt` | `POST /optimize/{case_id}` 同步 ＋ `/tune-prompt` 入队（2026-11-09 补），两条共用同一个 `optimize_agent_prompt` |
| `params` | `param_optimizer.py` | 基线重复跑的极差 `max−min` | `accepted` / `below_noise` / `eval_failed` / `runtime_mismatch` | 写 `temperature` | `/tune` 入队 |
| `skills` | `skill_optimizer.py` | 基线重复跑的极差 `max−min` | `pruned`（摘了没掉＝采纳摘除）/ `skill_needed`（摘了真掉＝保留）/ `eval_failed` | 写技能绑定 | `/tune-skills` 入队 |
| `tools` | `tool_optimizer.py` | **候选**内部 `repeat` 样本的极差 `max−min`（基线只跑 1 次，取不到基线抖动） | `tool_needed` / `tool_wanted` / `tool_unjudged` / `pruned` / `eval_failed` | 只进优化史，**不写配置** | `/tune-tools` 入队 |

三条读这张表时必须知道的事：

* **幅值口径已在 2026-11-09 统一**（都取 `max−min`、不折半；`tool_optimizer_test` ③-26 用
  AST 守着）。此前工具格折半，同一个词「抖动带」在三格里三种量纲。**来源**仍合法地不同：
  前三格的抖动取自**基线**重复跑，工具格基线只跑 1 次 ⇒ 取**候选**自己那 `repeat` 个样本。
* **`tools` 是唯一「判了也不改行为」的一格**：工具集的真源是代码（`ToolRegistry`），
  摘除要人去删注册行，所以 `accepted` 只落在优化史里。别把它读成「已经不管用了」。
* **`system_prompt` 的抖动带 2026-11-10 立成可选档，默认仍是零容差**。它 `accepted` 会
  **真覆盖提示词**，而真跑实测同一配置重复测量的极差是 **8 分**（`[80,88]`）⇒ 在真模型上
  「碰巧高 0.5 分」也会被判采纳。所以这一格给 `repeat`（**Reference 复测次数**，不是候选复测）：
  * `repeat=1`（默认）⇒ 与历史逐字一致：`candidate > best` 严格大于、回执 `jumpKind="none"`
    （「**没测**抖动带」，不是测出 0）。
  * `repeat=2–5` ⇒ Reference 连跑 `repeat` 次，抖动带 = 样本极差 `max−min`（`jumpKind="within"`），
    采纳判据变 `candidate > best + jump`、落库判据变 `best > reference + jump`。
  * 回执与 history JSON 都带 `repeat` / `samples` / `jump` / `jumpKind`，样本数列出来才看得出
    那 8 分是从哪两个点量出来的。
  **口径沿革**：源自 `optimizer.ts` 216 行，但**那个文件已随 Node 后端移出本仓**，现在没有可跑的
  对拍物 ⇒ 判据是本仓自守的，改它不必同步任何外部文件；别再把「破坏 parity」当成不许动的理由。
  **超时那一半 2026-11-09 已解**：新增入队口 `POST /api/v1/evaluation/tune-prompt/{case_id}`
  （`tune_runtime` 第四格 `_prompt_runner` → `optimize_agent_prompt(..., progress=, should_stop=)`），
  提交即返回、进度走 `GET /tune/{taskId}`，不再需要一次请求里同步跑完 3 轮；
  `POST /optimize/{case_id}` 只同步了同一处基线口径（仍留给 CLI/脚本，它是同步判据的入口）。
  **零容差那一半仍是默认**：不加 `repeat` 就照旧「碰巧高 0.5 分」被采纳并覆盖提示词，
  要收紧就显式传 `repeat≥2`，那是调用方每次的取舍、不是全局开关。
  ⚠️ 这一格的**分母是 `repeat + iterations`**（Reference 那 `repeat` 跑 ＋ 每轮 1 次评测的
  `iterations` 跑）。照抄姊妹格的 `repeat × (1 + 候选数)` 会让任务 done 了页面还停在半程；
  反之把这里的 `repeat` 当候选复测次数来算，`repeat=1` 时碰巧对，`repeat≥2` 就错了。
  分母由**调用方**算好传 `tune_runtime.submit(total_runs=...)`——`submit` 不按 `target` 分叉算术，
  它手里只有 `options`、看不出这格 `repeat` 是量谁的。
* **`iterations` 与 `repeat` 都有上限且越界即拒，不夹逼**：`MAX_ITERATIONS=20`、
  `MAX_PROMPT_REPEAT=5`（`optimizer.normalize_prompt_options` 唯一校验点，入队口与执行口共用）。
  `iterations=0` 合法（只跑 Reference、不生成候选；`repeat≥2` 时它才真的是「只量抖动带」，
  `repeat=1` 就只是「验通路跑一次」，两种都合法但别混着说）；`repeat` 必须是 int（拒 `bool`/`float`）。
  上限按实测算得出来：本格单跑 **47–165 秒** ⇒ 最坏档 `repeat=5 + iterations=20` = **25 跑 ≈
  20–70 分钟**（改写那几步另计）；后台任务表在进程内、没有超时回收（跑长了只能人工 cancel，
  进程一重启任务就没了），所以两条上限的依据是可算的时间成本，不是拍脑袋。
  拒而不是夹的道理：夹逼会让回执里的 `totalRuns` 与调用方要的东西不是一回事，而页面会照那个数
  显示进度。**校验必须在入队前做完**（`/tune-prompt` 先 `normalize_prompt_options`，400 直接退），
  否则「入队放行、排到才炸」的那一段，用户只能在后台任务里看到一条 `failed`。
* **v0 基线＝运行时在用的那份提示词**（`optimizer.effective_prompt()`：`agent_configs.
  system_prompt` 优先，没有配置行／是空串才回退出厂默认，回执 `baselineSource="config"|"default"`）。
  2026-11-09 之前一律取出厂默认，那是**唯一一格基线不是现网值**的（另三格早就是「DB 里那个温度／
  那份绑定／代码里那套工具」）⇒ 手工改过提示词、或上次优化已落库一版 88 分的，重跑时会拿默认那份
  的分数当靶子，候选只要高过默认就被采纳并**把 88 分顶成 82 分**，而优化史那行还写着
  `score_before=80`，读起来像「涨了 2 分」。改成现网基线后这条判据自动读作「要赢过正在跑的版本」。
* **`version` 是 (agent, case) 上的轮次序号，四格共用一个递增器**（`param_optimizer.
  next_version`）。本格原先每次运行都从 v1 数 ⇒ 同一 case 重跑后优化史里同 target 同 case
  会出现两个 v1，而 `mark_persisted` 按 `(agent, version, case_id, target)` 定位，**一次标中多行**
  ——上次并没落库的那几轮被标成「已生效」。序号连续后定位唯一（判据：`optimizer_test.py`
  「版本号跨运行连续」＋`mark_persisted` 只改 1 行）。
* **单轮跑崩不许废掉整轮**（2026-11-09 补齐，与另三格同口径）。本格原先
  `generate_candidate_prompt` / `evaluate_case` 是裸调用 ⇒ 抛错直接冒泡、任务 `failed`；
  可 `generate_candidate_prompt` 那句 `ValueError`（模型改写出来太短 / 剥完围栏是空串）是
  **设计内的正常失败**，`iterations=3` 时第 2 轮撞上就把第 1 轮已经花掉的评测钱全丢。
  现在失败那轮记 `eval_failed`（`score_after=None`，不编 0）、占一个 `version`、占分母，
  循环继续跑下一轮；`TuneCancelled` **原样抛出**（不许被 `except Exception` 降级成
  「这一轮评测失败」然后接着跑）。
* **`benchmarks/history/<agent>/<case>.json` 逐轮落盘**，不是只在循环之后写一次。这份 JSON 是
  **唯一存着候选提示词全文**的审计物（优化史表只存 `prompt_sha`），而取消只在两次评测之间生效
  ⇒ 取消那一刻已跑完的轮次必须已经在盘上，否则「取消即丢档」。每条 `iterations` 都带 `reason`
  （`accepted` / `no_gain` / `runtime_mismatch` / `eval_failed`，与优化史同一套枚举）。
  **落库结论也要再落一次盘**（`_flush_history()` 在 `persisted` 赋值之后还有一次）：逐轮 flush
  发生在落库之前，不补这一次，文件里就永远没有 `persisted` 字段 —— 内存里有、文件里没有，
  读档的人看不出这一版到底生效没有。
  **失败轮按阶段决定留不留候选**：`stage="generate"` 失败 ⇒ 压根没有候选（`prompt=None`，
  拿 best 顶上去会让人读成「这一轮试的就是 best」）；`stage="evaluate"` 失败 ⇒ 候选已生成、
  模型调用已花掉，`history` 存全文、优化史存指纹（另三格的 `eval_failed` 也都带非空
  `prompt=instructions`，本格原先一律写空串会把那次调用一起丢掉），`evidence` 里带 `stage`
  以便分辨「模型没产出」与「跑分时炸了」。
* **两份审计物要逐轮对账**（判据在 `optimizer_test.py`「审计对账」）。本格的审计物有两份：
  `benchmarks/history/<agent>/<case>.json`（存候选全文）与 `agent_optimizations` 表（存指纹与判决）；
  另三格只有后者。分别查「JSON 的 reason 列表」与「表里 `eval_failed` 行数」只能证明
  **两份各自对**，证不了它们**逐轮对得上** —— 某处少调一次 `record`，JSON 有 3 条、表里 2 条，
  那些判据全绿而页面上整整少一轮。所以还要断言：JSON 每一轮的 `(version, accepted, reason)`
  都能在表里找到**同版本、同判决**的一行。

**参数调优（缺口 4 第一格，2026-11-07 落地）**：`param_optimizer.optimize_agent_temperature`
同一份提示词只换 `temperature`，判决落 `agent_optimizations(target="params")`，页面上与提示词
那几版并排看。

为什么单开一个模块而不是塞进 `optimizer.py`；六条硬约束：

* **候选是「一批平级」不是演化链** —— `optimizer.py` 的 `version/parent_version` 是"v2 由 v1 改出
  来的"，而 0.5 与 0.3 之间没有父子关系。硬塞进链里，"上一版是谁"当场变成假话。
* **`repeat >= 2`**（`MAX_REPEAT=5`，默认 2）。单次评测的分数差里分不清「换温度了」与「这次恰好
  抖高了」；`repeat=1` 一律 400，也**不许**悄悄抬到 2 —— 那等于替调用方撒谎。
* **抖动带**：基线各轮的 `max-min` 就是这台机器/这个模型的噪声下限；候选的 Δ 必须**严格大于**它
  才算赢，否则记 `below_noise`（"涨了，但没超过抖动带"）。这和 `no_gain` 在页面上意思完全不同：
  前者"看不出差别"，后者"确实没变好"。
* **只往低扫**：`TEMPERATURE_GRID=(0.0,0.1,0.2,0.3,0.5,0.7,1.0)`，从当前值往下取最近两档。
  确定性判分下这一格方向单一（更低 ⇒ 格式漂移更少）；往上扫是拿"更有创意"赌判分，没有依据。
  另外两格（`max_tokens`/`max_iterations`）**有意不做**：调小会静默截断工具链，症状看起来像
  "模型变笨"，该由成本预算定，不该由分数定。
* **换模型不算**：候选那轮 `model` 与基线不同 ⇒ 记 `runtime_mismatch`、不参与比较（换了模型还
  比分数，等于把两件事的效果都算到温度头上）。
* **没有配置行就不写**：`persist_agent_temperature` 只在 `agent_configs` 已有该 agent 行时 update
  那一列；落 0 行则 `persisted=false` ＋ `persistSkip` 说明。插一行只带温度的空壳会把
  `system_prompt` 留空 —— 等于把"用默认提示词"改成"用空提示词"。

温度经 `runtime.merge_model_settings` 并进 `modelSettings`（键缺省即不覆盖；`0.0` 是正当值，
不能 `or` 被默认 0.7 顶掉），`evaluator.evaluate_case(..., settings=...)` 再把它透给四个评测分支。

⚠️ **落库必须复用调用方那条连接**：SQLite WAL 只允许一个写者，优化史那些 INSERT 已把写锁攥在
调用方手里，`persist_agent_temperature` / `persist_agent_config` 另开连接会在 `busy_timeout` 里
白等 30 秒再抛 `database is locked`（`tests/optimizer_test.py` 与 `tests/param_optimizer_test.py`
都实测撞过）。两处写口因此都加了可选 `conn` 形参 —— 缺省行为不变，优化器传自己的连接，
顺带让"判决"与"生效"落进同一个事务。

**验证到什么程度（参数调优那格，2026-11-07 如实记录）**：`tests/param_optimizer_test.py` 58 条，其中 ⑬ 是**真发
HTTP** 的 —— 起一个本地 stub，跑 `default_generate` 两次（`temperature=0.0` 与 `0.7`），把请求体
抓下来逐值核对。这条做过反向验证：把 ollama adapter 的 `js_nullish(...)` 换成 `... or 0.7`，判据
立刻报 `[0.7, 0.7]`；换回即绿。已确定的是「那一格旋钮真的拧到了」，而不是"只加了个形参"。

**模型从哪来：本仓自己下载 + 本仓自己推理**（同日补，别再拿「本机没模型」当借口）：下载走
`app/scripts/model_manager.py` / `POST /api/v1/local-models/hf/download`（HF、hf-mirror、魔搭三源，
断点续传 `.part`）⇒ 落到 `model-paths.json` 声明的 `models/llm/`；推理走 `services/engine/llm_backend.py`
（`LlmBackend`：chat 模板取自权重元数据、`generate_llm` 前向、Qwen2/Qwen3/Llama dense 架构）。
实测把 `unsloth/Qwen3-4B-Instruct-2507-GGUF` 的 `Q4_K_M`（2.33 GiB）拉到共享池 `models/llm/` ✓。

⚠️ **缺的那一跳已经补上（本轮）**：`text_generation.generate_text` 早就支持 `provider="engine"`，但那
只覆盖**一问一答**；agent 要的是「调工具 → 回灌结果 → 再想一步」。而 `default_generate` 里
`provider="engine"` 会先撞 `get_text_adapter('engine')` 的 `UnknownProviderError` ⇒ **本仓下载的模型
永远驱动不了 agent**（评测/优化全都打不上去，报错还长得像「没模型」）。现在 `default_generate` 在最
前面分派到 `_run_engine_tool_loop`：工具调用按 `<tool_call>{json}</tool_call>` 协议从模型原文解出
（块坏就抛、只截断不算调用），`assistant.tool_calls` 与 `role="tool"` 的 `name` 原样回灌
（`chat_template._normalize_messages` 不再把这两个字段压掉），超窗与「权重没有 chat 模板」两种
情况**当场拒**而不是静默降级。判据 `tests/engine_agent_loop_test.py` 26 条（含「分派必须在
`default_generate` 里」与「registry 里确实没有 engine」的反证）。

**真机已跑通（2026-11-07 补，取代此前「只差一次长跑」的记录）**：探针 `_bench/engine_llm_probe.py`
（装载 → 单轮生成 → `provider="engine"` 真调一次 `save_note`）末行 `PROBE_OK`：
`LOAD 62.4s`（带反量化产物缓存；首次 98.3s）／`DESC device=cuda:0 dtype=bfloat16`／
`ONE-SHOT 1.5s`／`AGENT 1.9s`／`TOOL_CALLS [save_note{text:"第一集：剑噬天下"}]`／
`SAVED` 与 `STEPS 2`（第一轮出调用、第二轮收尾）—— 即本仓下载的 GGUF **真能驱动 agent 工具循环**。

同日探针扩到五项（新增 ④ 长提示词峰值 ⑤ 流式），复跑一次全中：`LOAD 56.4s`／
`PREFILL tokens=4303 1.6s peak=9.68/22.49GiB OK`／`STREAM 0.4s deltas=9`（`1、2、3、4、5`）／
`AGENT 1.1s` ⇒ `PROBE_OK`。**这两项是旧探针的盲区**：它只看「`save_note` 有没有被调」⇒
上面那个 94.62 GiB 的预填充、以及流式半路不发字，都能照样报绿（`PROBE_OK` 现在要三票齐）。
peak 从 94.62 GiB 落到 **9.68 GiB**（4303 词元档，比 `/tune` 的 3205 还长）是 `inference_mode`
＋分块预填充的直接读数。

途中两处「文件和代码各自没错、交界处对不上」，都已修并配判据：

* **tie embeddings**（装不进去那句的真因）：`Qwen3-4B-Instruct-2507-GGUF` 的 Q4_K_M **既没有**
  `qwen3.tie_embeddings` 元数据键、**也没有** `output.weight` 张量（llama.cpp 对 tie 模型允许省略）。
  `infer_llm_config` 原先只看元数据、缺键按 `False` ⇒ 模型建出独立 `lm_head` ⇒ 键集对账要
  `lm_head.weight` ⇒ 一个**完好的文件**被报「GGUF 里缺 1 个张量」。现在 `tie_embeddings_of()` 以
  **张量表**为准（没有 `output.weight` ⇒ 只能 tie；声明写 `False` 而表里又没这张 ⇒ 具名拒绝），
  且 `_check_accounting` 认得**真共享的键**——放行凭据是 `lm_head.weight is embed.weight` 这件
  事实（`state_dict()` 会把同一张 Parameter 在 `embed.weight`／`lm_head.weight` 两个名字下各列
  一次，这是对账必然误报的机制），不是配置位；配置说 tie、实物没 tie 时**照旧报缺**。
* **漏标签的裸调用**：权重自带的 chat 模板明写 `<tool_call>...</tool_call>`，4B 那次却只吐
  `{"name": "save_note", "arguments": {...}}` ⇒ 旧判据算「这轮没调工具」，工具不执行、`PROBE_NO_TOOL`。
  现在 `parse_engine_tool_calls(text, tool_names=...)` 认一种**极窄**退路：**整段**就是一串调用
  对象（可带 ```json 围栏）、键只限于 `name/tool/arguments/parameters`、且工具名**全都真注册过**、
  个数 ≤ 32；缺任何一条就退回「没调工具」（夹在散文里的同形状对象不认、名字没注册当散文**不抛**、
  多出键不认、不给 `tool_names` 一律不认）。
  ⚠️ **必须是「一串」**：同日第二次真跑（`/tune`）才露头 —— extractor 那跑模型一口气吐了**四个**
  裸对象（`read_script_for_extraction` + 三个 `read_existing_*`），只认单对象的写法把整串丢光 ⇒
  读工具一个都没执行 ⇒ 模型永远走不到 `save_*` ⇒ `/tune` 四跑**恒 0 分**（看起来像"模型不行"，
  其实是解析层吃掉了合法调用）。混着编造的名字时**整串不认**（只跑认得的几个＝替模型裁剪它没说过的调用）。
* **预填充不关梯度**（`/tune` 真跑才露头的真 bug，`_bench/tune_probe.py` 里 OOM）：`generate_llm`
  没有 `torch.inference_mode()`，参数默认 `requires_grad=True` ⇒ 每步前向都在建 autograd 图，
  36 层中间量（注意力分数 `(heads, L, L)` 最大）被 backward 节点吊住。4B / 3205 词元的 extractor
  提示词实测 **PyTorch 已分配 94.62 GiB**（卡 22.49 GiB —— WDDM 甩进共享内存硬撑 60 秒才 OOM）。
  现在整函数跑在 `inference_mode` 里，且长提示词按 `prefill_chunk`（默认 1024 词元）分块预填充
  （峰值 `O(L²)` → `O(chunk × L)`，带 KV cache 数学等价、逐 token 结果相同），解码步 `s == 1`
  不再造全零因果掩码。判据 `engine_llm_test` 的 ⑦ 系列（钩子直接问「前向时梯度开着吗」「一次喂了多少词元」、
  分块与一次吃完逐 token 对齐、`prefill_chunk=0` 当场拒）—— 这类"结果全对、只是吃 4 倍显存"的病，
  ①②④ 那些数值判据**发现不了**。真机读数见上文探针 ④：4303 词元档 **1.6 秒、峰值 9.68 GiB**。

**`/tune` 整轮已真跑（2026-11-07，取代此前「仍待一次」）**：`_bench/tune_probe.py` 把真库整份复制成
沙箱（`DATA_ROOT` 指过去、`mklink /J` 复用 7.5 GB 反量化缓存）、往副本里插一条
`provider="engine" / priority=999 / settings.ggufPath` 的文本配置，再调
`param_optimizer.optimize_agent_temperature(conn, "extractor", ext-case-001, {repeat:2, maxCandidates:1})`
—— **不碰用户库**。末行 `TUNE_OK`，耗时 423.9 秒：

| 档 | 两次得分 | 均值 | 判据 |
|---|---|---|---|
| 基线 0.7 | 75.0 / 75.0 | 75.00 | 抖动带 0.00 |
| 候选 0.5 | 75.0 / 75.0 | 75.00 | Δ +0.00 ⇒ `no_gain` ⇒ `KEEP`（温度保持 0.7） |

即：分数是**真分数**（extractor 抽取评分 75/100，不是解析失败给的 0），而在这台 4B 上
**温度 0.5 与 0.7 无差别**（两档各自两跑都稳在 75.00、抖动带 0）。单跑 85–165 秒（出现
`missing-write-nudge` 的那跑翻倍）、四跑合计 424 秒。

**424 秒这件事逼出的接口改造（2026-11-07 同日）**：原来 `POST /evaluation/tune/{case_id}` 是
**同步长请求** —— 前端 fetch 不超时会一直等，**中间代理却会先截断**，结果是界面报失败、后台还在
烧卡（最难受的一种：日志显示跑完了，用户看到的是超时）。现在改成**入队 + 查进度**
（`app/agent/evaluation/tune_runtime.py`）：

| 入口 | 行为 |
|---|---|
| `POST /api/v1/evaluation/tune/{case_id}` | **立刻**返回 `{taskId, status, queuePosition, pollUrl}`；`wait=true` 才等回执（`timeoutSeconds`，默认 900 秒，超时回快照标 `timedOut`，**任务不停**） |
| `GET /api/v1/evaluation/tune/{taskId}` | `status` / `stage` / `doneRuns` / `totalRuns` / `progress`；跑完带 `receipt`，失败带 `error` |
| `GET /api/v1/evaluation/tune` | 最近任务列表（新的在前） |
| `POST /api/v1/evaluation/tune/{taskId}/cancel` | 取消：排队中立刻生效；**运行中只在两次评测之间生效**（一次真模型生成从外面掐不断，见下） |

三条口径写进代码注释也写进测试（`tests/tune_runtime_test.py`，40 条）：① 一张卡 ⇒ **单工作线程 +
队列**，第二个任务排队而不是并发抢显存；② 同一 (agent, case, **target**) 已在跑 ⇒ `deduped=true` 不重复入队（漏掉 target 会把刚提交的技能
  调优当成温度任务的重复而静默丢弃，还回一个别人的 taskId）；
③ 取消**不假装能秒停** —— `TuneCancelled` 在每跑开头抛，排队中被取消的一次都不跑。任务自己
开连接、且必须是 `engine.begin()`（请求那条连接响应返回就归还了，拿它跑后台的症状是跑到一半
`connection is closed`；而 `connect()` 里的写入在 `close()` 时**一律回滚** ⇒ 回执照写
`persisted=true`、库里一个字没有 —— 2026-11-08 实测改掉，见下文「技能绑定调优」末尾那条），
任务表是进程内存、重启即空（产物早已落 `agent_optimizations`/`agent_configs`）。

**改造后又整轮真跑了一次，这次走的是线上默认那条路（2026-11-07 晚）**：
`_bench/tune_probe.py --via-runtime` = 入队 → 工作线程跑 → 每 2 秒轮询。任务 `tune-4434da372aca`
**560.3 秒**、终态 `succeeded`、末行 `TUNE_OK`；进度时间线 `2.0s 0/4 → 172.0s 1/4 → 392.1s 2/4
→ 510.1s 3/4 → 560.1s 4/4`。回执这一轮与上面直连那轮不同（采样随机）：

| 档 | 两次得分 | 均值 | 判据 |
|---|---|---|---|
| 基线 0.7 | 100.0 / 75.0 | 87.50 | 抖动带 25.00（同一档两次差 25 分） |
| 候选 0.5 | 100.0 / 51.7 | 75.85 | Δ −11.65 ⇒ `no_gain` ⇒ `KEEP`，`accepted=null`、`persisted=null` |

这条路径不可替代：单测里工作线程跑的是假 runner、直连探针跑在主线程 ⇒「**工作线程 + 自建连接 +
真引擎 + 长耗时**」四者只有 `--via-runtime` 能同时覆盖。另附两个实测口径：
① 探针 stdout 必须 `-u` 直连日志，别指望工具会话（3 分钟就超时）—— 超时截断后日志里只剩
`faulthandler` 堆栈，看不到任何进度行；
② 判断"真在跑还是卡死"看 `Get-Process` 的 **CPU 时间是否还在涨**，别拿日志 mtime 判断
（每次 dump 都会刷新它）。

**技能绑定调优（缺口 4 第二格，2026-11-08 落地）**：
`skill_optimizer.optimize_agent_skills` —— **同一份提示词、同一个温度**，只换「注入哪几个
skill」，判决落 `agent_optimizations(target="skills")`，入口 `POST /api/v1/evaluation/tune-skills/{case_id}`。
和温度那格共用后台任务表（`tune_runtime` 的 `target` 就是这几格的公因子），进度/取消/列表仍是
`/tune/{taskId}` 那三条 —— 所以下面只写**不一样**的地方。

* **判据方向相反（最容易读错的一处）**：温度是「涨过抖动带才采纳」；这一格是
  **「摘掉后跌幅不超过抖动带 ⇒ 没有证据说明它有用 ⇒ 采纳这一摘」**（`delta >= -jump`）。
  动的是注入量，少注入本身就是收益 ⇒ 优化史上这类行 **Δ 为 0 甚至负数也会显示「采纳」**，
  那是账记对了。两个理由码：`pruned`（摘掉后没测出差别，已精简绑定）、`skill_needed`（摘掉后明显下降，保留）。
  页面注释同步写了这一条，否则看的人第一反应是「Δ 是负的怎么算采纳」。
* **候选是 leave-one-out，一遍只摘一个**。摘掉一个之后基线就变了（剩下的 skill 之间会互相补位），
  同一轮里接着摘第二个 = 拿旧基线去判新绑定。采纳即停，下一轮再看新的可疑对象。
* **不许摘成空**：`len(current) <= 1` 时**没有候选**（不是「摘掉最后一个」）—— 绑定清空等于
  让 agent 裸奔，那是「改需求」不是「调优」。
* **排序按可疑度**：`skill_usage.penalty`（失败/返工惩罚）降序，同分时绑定里**靠后的先摘**
  （后绑定的通常是补丁）。这一格是「读使用统计」而不是「猜哪些看起来像废话」。
* **覆盖走 `options["skillsRaw"]`，且覆盖时不写 `skill_usage`**。这是本格唯一必须动 runtime 的地方：
  假设性绑定若记进使用统计，会**污染下一轮的排序**（把真在用的 skill 排到可疑位，越调越偏）。
  `runtime` 里用 `skills_overridden` 一个开关同时管「注入用哪份」与「记不记账」，两件事必须同源。
* **`autoPersist` 默认 False**（温度那格默认 True）。温度调低最坏是「更死板」；删一个 skill 最坏是
  「该知道的规范不知道了」——这类删减默认只出结论、由人点头，`autoPersist=true` 才落库。
* **没有配置行就不写**：`persist_agent_skills` 同温度那格，只 update 已有行，返 0 则 `persisted=false`
  ＋ `persistSkip`。插一行只带 `skills` 的空壳会把 `system_prompt` 留空。

⚠️ **形状是这一格唯一会静默咬人的地方**：落库/覆盖体都必须是
`[{"id":…,"enabled":true,"priority":n}]`。写成纯字符串数组（`["a","b"]`）时
`skills.parse_skills_config` 会**整条跳过**、`load_agent_skills` 静默回退默认绑定 ——
症状是「页面写已采纳、线上还是旧绑定」，两边都不报错。所以 `creator._normalize_skills`
的输出形状与 `skill_optimizer._skills_json` 的形状都用测试①钉着，且假评测器按 `b["id"]`
读覆盖体（形状写错当场 `TypeError`，而不是悄悄变成默认绑定）。

**同日修掉一个真 bug（静默丢档，温度那格也中招）**：`tune_runtime._execute` 原本用
`db_module.engine.connect()` 跑优化器。SQLAlchemy 2.0 下 `connect()` 里的写入在 `close()` 时
**一律回滚** —— 实测 `with engine.connect() as c: INSERT …` 之后另开连接读回 `rows = 0`。
于是「优化史写入」与「采纳后的 `agent_configs` UPDATE」**一条都没落盘**，而回执照样写
`persisted=true`：页面说生效了、线上还是旧值、全程无报错。改成 `engine.begin()` 并加源码判据
（`tests/skill_optimizer_test.py` ⑧-9：`tune_runtime.py` 里不许再出现 `engine.connect()`）。
路由那层的 `get_conn` 不能挪用（它的生命周期跟着 HTTP 请求走）——所以「必须自己开连接」是对的，
错的只是「自己开的这条没有事务」。

**技能绑定的验证到什么程度（2026-11-08 如实记录）**：`tests/skill_optimizer_test.py` 79 条，
判据分六组 —— ① 绑定形状（`_skills_json` / `_normalize_skills` 两边逐字段钉着）、
② 候选生成（留一个时零候选、penalty 同分靠后先摘）、③ 判据方向（`delta >= -jump` 即
`pruned`，跌幅超带即 `skill_needed`，两条都做反向验证）、④ 覆盖体不污染 `skill_usage`、
⑤ 落库（无配置行返 0、`autoPersist=false` 时不写库）、⑥ 路由与任务表（`/tune-skills` 真发
HTTP、case 不存在 404、`target` 参与去重）。⑧-9 是源码判据那条：`tune_runtime.py` 里出现
`engine.connect()` 即红。

**工具集调优（缺口 4 第三格，2026-11-08 落地）**：`tool_optimizer.optimize_agent_tools`
—— **同一份提示词、同一个温度、同一套 skill**，只换「给模型看哪些工具」（候选是
leave-one-out，候选源是**基线那一跑的逐步账**，见下第三条：没被调到过的工具摘掉本来就测不出东西）。
判决落 `agent_optimizations(target="tools")`，入口 `POST /api/v1/evaluation/tune-tools/{case_id}`。
这一格的前置不是代码量，是**尺子**：原来一个 case 就是一整轮对话打一个分（见 §1 缺口 4
末尾那段"得先有单步对照"），颗粒太粗 —— 摘掉 `search_script_library` 后模型可能改调别的工具、
总分几乎不动；也可能整轮崩掉却只掉 2 分。所以先落 `evaluator.step_trace()`：

* **单步对照**：把 `tool_calls` / `tool_results` 按序配成逐步账，每步三态 ——
  `ok=True`（结果里没有错误标记）/ `ok=False`（工具报错或 `Unknown tool`）/
  `ok=None`（**这一跑没有对应结果**，驱动没回灌 ⇒ 不判成失败，没证据 ≠ 失败）；
  `stats` 给 `total/ok/failed/unjudged/unknownTool/tools/unavailable`，五条评测分支
  （整轮对话、剧本、宫格、成片、提取…）的 return 全经 `_report_steps()` 收口 ——
  不收口就会出现「某一类 case 悄悄没有 steps，而测试正好只测了另一类」。
* **两把尺子判据**：整轮分 ＋ `unknownTool` 计数。模型照着被摘掉的工具名去调，
  等于它**真需要**这个工具（`_UNKNOWN_TOOL_PREFIX = "Unknown tool"` 认这一档。
  ⚠️ 它让该步记成 **`failed`**（`ok=False`：有结果、结果里有 error），**不是** `unjudged`
  —— `unjudged` 专指「这一拍根本没有结果」（`ok=None`），两者别混）。四个理由码：
  `pruned`（跌幅没超抖动带 **且** 不再调它）、`tool_wanted`（分数没掉但模型仍在调
  ⇒ **不摘**）、`tool_unjudged`（分数没掉，但**逐步账本身没明细** ⇒ **不摘**）、
  `tool_needed`（跌幅超过抖动带）。后两档都是这一格独有的 —— 技能格既没有
  「模型仍在用但分数没变」，也没有「第二把尺子哑了」这种情形。
  ⚠️ `tool_unjudged` 与 `tool_wanted` 必须分成两码：前者是**读数缺失**（没有明细时
  `unknownTool` 恒 0，与"模型真没再伸手"在数字上一模一样），后者是**读数充分且说
  模型仍在调**。混成一码 ⇒ 页面会写「模型仍在调它」而事实是"没记账"。
  落法：`_step_stats()` 原样透传 `unjudged`/`unavailable`（**不许在消费点丢掉**，
  丢了就是这个 bug），候选样本任一缺明细 ⇒ `stepsUnavailable` 非空 ⇒ 不进 `pruned` 分支。
  ⚠️ **缺账有两档，只挡一档等于没挡**：整格缺（`unavailable` 非空，整个数组不是列表）之外还有
  **部分缺**（`unjudged > 0`：有调用、没回灌结果）。`unknownTool` 是靠**结果的 error 正文**认出来的
  ⇒ 那几步没有结果时，模型真在其中调了被摘的工具也读不到，与整格缺同罪。旧逻辑只看
  `unavailable`，实测反向验证：造一份 5 步、末步 `ok=None` 的候选报告 ⇒ `reason="pruned"`、
  `accepted=True`，即**直接错摘并记成「已采纳」**。两档合并挡进 `tool_unjudged`，回执里
  `stepsUnavailable`/`stepsUnjudged` **分开写**（能分辨是哪一种），理由码不拆（同一件事：判不了）。
  这条判据的成立前提跟着钉住了三条（`tool_optimizer_test` ③-23/③-24/③-25）：驱动侧
  `tool_calls_out.append`/`tool_results_out.append` **只在 `_run_tool` 里各出现一次、相邻无条件**
  （三条驱动循环全走它 ⇒ 真实跑里 `unjudged` 恒 0，那条拦截平时静默）；`AgentRunResult` 必须把
  `generate()` 的 `tool_results` 带出来（漏一行 ⇒ 每跑都成「有调用没结果」）；以及**触发器的跨模块
  两半必须一致** —— 产出只有 `runtime._run_tool` 的 `{"error": f"Unknown tool: {name}"}` 一处，
  识别只有 `evaluator._UNKNOWN_TOOL_PREFIX` 一处，改任何一半都**不报错**，只让 `unknownTool` 恒 0
  而 `failed` 照常计数（实测：把常量改成 `No such tool` ⇒ 端到端那条立刻红，实得
  `failed=1, unknownTool=0`，两把尺子当场分家 ⇒ 判决链读成「模型没再调它」⇒ 错摘）。
  端到端口径也钉在 HTTP 面上（`evaluation_route_test` 两条）：`POST /evaluate/{case_id}` 返回的
  报告带 `steps`/`stepStats`，成对的 calls/results 记 1 步 `ok`（`unjudged=0`），结果里带
  `Unknown tool:` ⇒ `unknownTool=1, failed=1` —— 此前这两格只在单测里验过，那些 `report` 是测试
  自己造的，等于「第二把尺子只在假数据上验过」。
  ⭐ **真模型实测（2026-11-09，ollama `qwen3:14b` / A5000）**：跑 `sb-case-001` 基线一次，
  实际过程是 —— `read-context` 工具 **41 秒完成**（日志 `[StoryboardTool] DONE read-context
  | characters=2 scenes=2 scriptLength=244`）⇒ **真实 provider 下工具确实被调用**，候选源
  `stepStats.tools` 不会是空的；但**紧接着那次模型响应 180 秒 ReadTimeout**，重试 3 次共
  **775 秒**后 `all-models-failed`，`evaluate_case` 抛异常。两条结论：
  ① **安全**：`optimize_agent_tools` 的基线那行 `baseline_report = await run_eval(...)` **没有
  try** ⇒ 异常一路冒到 `tune_runtime._execute` 的 `except Exception` ⇒ 任务标 `failed` 并带
  真实错误文本 —— **缺账/超时的最坏结果是「任务失败」，不是「错摘工具」**（这是本格最该守住
  的那条线）；
  ② **超时的真因是当时 GPU 被别的负载占着，不是本格配置**：同一份「只回复 OK」的极小请求
  （`think:false`、`num_ctx` 32768、模型已 100% GPU 常驻）在超时那段时间 **180 秒不回**，
  排空负载后**同一请求 3.35 秒返回**（`load_duration` 3.17 s 冷加载 ＋ `prompt_eval_count`
  19 词元 0.11 s ＋ `eval_count` 2 词元 0.05 s）。占卡的是并行那条工作流的
  `data/experiments/llm_graph_probe/probe.py`（进程内 `llm_backend` 跑 Qwen3-4B，7.49 GiB
  全 GPU，**不经 ollama 但也吃显存**）；同时 ollama 服务日志里另有一条 **19455 词元** 的 prompt
  按 180 s 的节奏反复重试（`n_keep=4`、`cached n_tokens=19451`），而 14b 要 14 GB、卡只有
  22.49 GiB ⇒ 争用下 `tg_3s` 掉到 **0.01 tok/s**（正常是 8–40 tok/s）。
  ⚠️ 旧记录写的「`OllamaTextAdapter` 只下发 `options.num_ctx`、**没有** `think` 这一格」
  **是错的**：`think` 一直在下发（`services/adapters/text_adapters.py` 的 `think: bool = False`
  与 body 里的 `"think": self.think`），这本来就是 ollama 必须走原生 `/api/chat`、
  不走 OpenAI 兼容口的原因。所以「跑不通」的正确表述是 **「当时卡被别人占着」**：
  要真跑的前置条件是**排空 GPU 上的其他负载**（`nvidia-smi` 看 `memory.free`，
  14b 要 14 GB、8b 要 10 GB），而不是换模型或调大超时；只有「卡已空闲仍超时」才该怀疑
  `_transport_retry` 的本地 180 s。8b 对照实测 16.7 秒返回（含 11.8 秒冷加载）。
  ⇒ **「只差一次真跑」到此撤掉（同日排空后兑现）**：前置条件就是上面那句「先把卡让出来」。
  排空后同一 case 单跑基线 **47.1 秒**、得分 **78.3**，`stepStats = {"total": 2, "ok": 2,
  "failed": 0, "unjudged": 0, "unknownTool": 0, "unavailable": null, "tools":
  ["read_storyboard_context", "save_storyboards"]}` —— 即**真实 provider 下逐步账是完整的**
  （`unavailable=None`、`unjudged=0`）、`stepStats.tools` 非空 ⇒ 本轮前面钉的三条前提
  （驱动 `unjudged` 恒 0、结果真被回灌、候选源非空）在真数据上成立，不再只是「假数据上验过」。
  完整判决链（`optimize_agent_tools`，`repeat=2 / maxCandidates=2`）随后跑通 **182 秒 5 跑**：
  基线 **95.00**（3 个工具：`read_storyboard_context` / `save_storyboards` /
  `save_continuity_states`）、候选 `save_continuity_states` ⇒ `[80, 88]`、
  候选 `read_storyboard_context` ⇒ `[0, 0]`。另有一次 14 秒的跑（基线只暴露 1 个工具 ⇒
  走「候选不足」分支，`reason="no_candidate"` 如实收工）—— 模型行为不稳定（同一 case 有时
  只读不存），但「有候选」与「没候选」两条分支都验到了。
* ⭐ **抖动带不许拿「候选之间的极差」充噪声**（同日真跑抓到的真缺陷，已修）：
  上面那次真跑里 `save_continuity_states` 摘掉后 **95 → 84（Δ −11）** 却被判 `pruned` 并采纳
  —— 旧口径 `jump = (max(候选均值) − min(候选均值)) / 2` 被崩到 0 的那个候选撑成 **42**，
  −11 落在带内就被读成「没测出差别」。**候选之间的差距本来是信号**（摘关键工具 vs 摘边缘工具
  就该差很多），噪声只能是**同一配置重复测量**之间的差。改成「各候选自己那 `repeat` 个样本的
  **极差（`max − min`）**，取最大者」：`[80, 88]` ⇒ 8、`[0, 0]` ⇒ 0 ⇒ `jump = 8` ⇒
  −11 < −8 ⇒ 正确判 `tool_needed`。**幅值不折半**，与参数格/技能格逐字同口径（它们都写
  `jump = max(baseline_scores) - min(baseline_scores)`）—— 旧版我曾取半极差，那会让同一个词
  「抖动带」在三格里有三种量纲，已改齐。⚠️ **来源如实不同、别在文档里说成同源**：参数格与
  技能格的抖动带取自**基线**重复跑，本格基线只跑 1 次（这一跑只为拿暴露集与基线分）⇒ 只能取自
  **候选**自己那 `repeat` 个样本；两者都是「同一配置重复测量」，但不是同一批样本。
  `repeat=1` 时估不出抖动 ⇒ 按 0 计（与参数格「抖动带为 0 时 Δ>0 即算提升」同一口径：
  没有抖动，就不存在「掉的那点可能是噪声」）。回执 `jumpKind` 如实写 `within` / `none`
  （旧值 `proxy` 撤销，它本来就在承认「这不是真抖动带」）。
  `pruned` 的页面标签同步改成「摘掉后**没测出差别**（跌幅在抖动带内），已精简绑定」——
  原来写「分数没掉」，而这条实测里它明明掉了 11 分，标签会骗人。
  判据三条：`tool_optimizer_test` ③-9b/③-9c/③-9d 直接用上面那组真实分数做反向验证（把 `jump`
  改回跨候选极差 ⇒ 精确报 `accepted=True / reason='pruned'`，`jump=42`、`delta=-11`）；
  ③-26 守**三格口径一致**（AST 判「都取 `max−min`、都不含除法」，反向验证把本格改回 `/ 2`
  ⇒ 精确报 `tool=False`；★该判据第一版只查了 `jump` 赋值、没查 `spreads`，折半藏在后者里
  ⇒ 反向验证时它**没红**，是补上 `spreads` 的除法检查后才盯住的）。该套 86 → 90 条。
* **进度分母别照抄技能那格**：本格基线**只跑 1 次**（它不吃 `repeat`，这一跑只为拿暴露集与
  基线分），候选数又要等基线跑完才知道（`tool_candidates` 依赖轨迹）⇒ 真值是
  `1 + repeat × 真候选数`，先按上限 `1 + repeat × maxCandidates` 估、`baseline` 事件起校正。
  抄成 `repeat × (1 + n)` 是**两处都错**（基线多算 `repeat−1` 跑 ＋ 候选按上限算），
  症状只是「任务已 done、页面永远停在 5/6」—— 不崩、不报错，因此最容易长期没人修。
* **基线就没明细 ⇒ 在基线处收工，别烧候选**：`stepStats.unavailable` 在**基线那一跑**就非空
  ⇒ 第二把尺子整格哑掉，候选跑多少次都只能得 `tool_unjudged`（证据缺失恒不采纳）⇒
  `repeat × 候选数` 跑（8 分钟量级）纯浪费。同时**理由不许写成「空工具集」**：缺明细时
  `current_tools` 也是空的，照空分支走会说「只出现了 0 个工具」—— 那是把「没记账」说成
  「模型没调过任何工具」，两件事实正相反（同一天同一个病根犯了两次：判决链一次、理由文案一次）。
  落法：`baseline_unavailable` 早退 ＋ `_stage_text()` 的 tools 基线分支读 `steps.unavailable`。
* **候选源是基线那一跑的「内存结果」，不是盘上的轨迹文件**：候选从基线报告的
  `stepStats.tools` 取（老报告没有这一格就退回扫 `steps[].tool`），只摘**真被调到过**的工具
  —— 模型一次没用的工具，摘掉它本来就测不出任何东西，把它算成候选等于拿噪声当证据。
  ⚠️ 2026-11-08 实测一条硬事实：`<dataRoot>/traces/*.jsonl` 里 `meta.toolCalls` 是**整数计数**
  （68 个文件、61 条带它的记录，**全部**是 `int`），逐步明细根本没落盘 —— 所以
  "从盘上轨迹重建候选"这条路**不存在**，别去写它。`step_trace` 也顺势加固：传进来一个计数
  不炸，`stats["unavailable"]="toolCallsNotList"` 如实说明「没有逐步明细」而不是静默给
  一个 `total=0` 的空账（空账看起来像"这一跑没调工具"，与"账没落"是两件事）。
* **覆盖体走 `options["toolsRaw"]`**：`runtime` 在 MCP 合并**之后**裁剪，
  `ToolRegistry.only(wanted)` 按键或按 `tool.id` 两套口径匹配（模型侧函数名是 `tool.id`，
  注册表键是另一套 —— 只认一套就会静默漏掉或全开）、顺序跟注册表、`missing` 如实上报；
  `_parse_tool_names` 收 JSON 字符串或 list，非法当场 `ValueError`，**不回落成"全开"**
  （回落会把"这次只给 3 个工具"静默变成"这次全都给"，而分数看起来还挺好）。
* **恒 `autoPersist=false` ＋ `persistSkip`**：工具集的真源是代码里的装配语句
  （`runtime` 里 `_as_registry(create_*_tools(...))` 那一串），`agent_configs` **没有 tools 列**
  ⇒ 这一格**只落优化史结论**、不写库，回执里 `persisted=false` 并写明为什么。
  路由侧 `normalize_tool_options` 把 `autoPersist` 钉死成 `False`，不接受外部打开。

**提示词优化的入队口（缺口 4 第四格的第二半，2026-11-09 落地）**：`system_prompt` 这一格
原来只有一条同步路 —— `POST /api/v1/evaluation/tune-prompt/{case_id}`
（`tune_runtime.TARGETS` 第四格 → `_prompt_runner` → `optimize_agent_prompt`）。
`POST /optimize/{case_id}` 只同步了一处口径（同一天把基线改成「DB 里正在生效的那份」，
否则重跑会把更优的现网版本覆盖成更差的）：它是同步判据的对照入口（`optimizer_test.py` 守着
「严格高于」那几条），入队口改的是**投递方式**、不是判决，两条路共用同一个
`optimize_agent_prompt`；同步这条留给 CLI/脚本。入队这条存在的唯一理由是**超时**：
一轮 =「一次 LLM 改写 + 一次 85–165 秒评测」× `iterations`，十几分钟起步，走请求面必被代理掐断。

三处**与另三格不同**的口径（照抄任一条都会得到错的读数）：

* **`repeat` 是 Reference 的复测次数，不是候选复测**（2026-11-10 加）：`iterations` 是「改写几轮」，
  每轮候选只评一次 ⇒ 分母 **`repeat + iterations`**（那 `repeat` 跑全在 Reference 上）。
  ⚠️ 姊妹三格的分母各不相同（`params`/`skills` = `repeat × (1+候选数)`、`tools` 基线只跑
  1 次、本格 `repeat + iterations`）—— 四格四种算法，只能逐格读，别写一条通式。分母由调用方
  算好传 `submit(total_runs=...)`；`submit` 只按 `options["repeat"] × (1 + expected)` 兜底，
  它看不出这格 `repeat` 量的是谁。
* **是四格里唯一真写配置的一格**：`auto_persist` 默认 **True**（与 `/optimize` 一致），
  best 高于 Reference **＋抖动带** 才写进 `agent_configs.system_prompt`；传 `false` 只出结论。
* **抖动带可选，默认还是零容差**：`repeat=1` ⇒ `jumpKind="none"`，回执的 `baseline` 事件因此
  **不印「抖动带 None」**（那读起来像漏算），而写「本格没测抖动带」；`repeat≥2` ⇒
  `jumpKind="within"`，印实测极差与样本数。

`iterations` 的校验：合法区间 `0..MAX_ITERATIONS`（`iterations=0` ＝只评 Reference 就收工，
用来验通路）；**越界即拒**（`/tune-prompt` 入队前 400、执行口抛 `ValueError`，两处共用
`normalize_prompt_options`）。负数仍夹到 0 —— 那不是「越界要拒」的场景，而是「只评 Reference」
的老口径写法。`repeat` 必须是 int 且在 1–`MAX_PROMPT_REPEAT`，同样越界即拒。
`wait:true` 仍可用（`timeoutSeconds` 默认 900），但那是给测试/脚本用的，不是给页面用的。

判据：`tests/tune_runtime_test.py` `case_prompt_target()` 十项 —— 四目标闭集同名、
`_runner_for` 指向 `_prompt_runner`、别名 `prompt` 当场拒、`start` 文案写「共 N 跑…共 M 轮」
（不写「每档」）、`run` 事件区分 Reference 与「第 N 轮候选」（不印 `1/1`）、`baseline` 不印
「抖动带 None」、`candidate` 印 `v{version}：{why}`、入队分母 = 4、`options` 白名单含
`iterations`。另加：`repeat≥2` 时 `start` 文案点名「Reference 复测 N 次取抖动带」、
`baseline` 按 `jumpKind` 分两支（`within` 印极差、`none` 明说「没测抖动带」）、
显式 `total_runs=5` 时入队分母就是 5（`repeat=1 + iterations=4`）而不是兜底算出的 8。
`optimizer_test.py` 另守：`normalize_prompt_options` 的六种越界（`iterations>20`、负数
`repeat`、`repeat=6`、`repeat` 是 `bool`/`float`/`str`）都抛 `ValueError`、`iterations=0` 合法、
以及 `/tune-prompt` 入队前就 400（stub 掉 `submit` 判「一次都没提交」）。
⚠️ `tool_optimizer_test` ⑥-10 用源码切块核「`/tune-tools` 那端点不带
`auto_persist`」，切块上界原本取 `"/tune")` —— 新端点正好插在两者之间会被切进来误判，
现改取 `@router.post("/tune-prompt/{case_id}")` 作上界。

**Skills 的动态路由**（缺口 10，2026-11-07 已落地半格）：不改 `SKILL.md` 的自述机制，
只在 `load_agent_skills` 选择"注入哪些"时加排序依据。落地形状与设计稿的差异写在这里，
免得后人以为是漏实现：

| 设计稿 | 实际落法 | 为什么 |
|---|---|---|
| 触发词选择 | **不做** | 要引入检索层，与「`SKILL.md` 自述绑定」的机制冲突；且 skill 之间不是同义检索关系而是互补规则 |
| `priority × 历史命中率` | `priority + 降级惩罚`（`PRIORITY_PENALTY=50`） | `priority` 是序数量纲，乘 0..1 的率会把 1 与 2 压成没差别；加法语义是「排到明显更好的候选之后」，不是"降权" |
| 命中率（从 `skill_usage` 读） | **`okRate`＝注入该 skill 的跑次成功率** | 「模型读没读它」**没有观测通路**，硬算命中率就是编。`okRate` 是相关不是因果，看板不许改叫「命中率／贡献度」 |
| 命中率低就少注入 | 只在**超预算**时改**裁剪次序** | 预算够时注入顺序与内容必须逐字不变（否则每次统计抖动都会改提示词，diff 无从判读） |

`skill_usage` 记的是可观测事实：注入了哪些、各多少字符、哪些被裁、该跑次成败
（`run_id` 用 `trace:Agent/<id>` 与知识投喂同一个句柄、同 run 覆盖幂等）。
`MIN_SAMPLES=10` 以下一律 `judged=false` —— 没有基线不说话。读口只有
`GET /api/v1/skills/usage`，且**永不影响主流程**（统计库坏了退化成纯 `priority` 排序）。

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
| `TECH_QC_THRESHOLDS` 某条触发多、人工复核里**误报占多数**（或反过来漏报多） | 该阈值与本机 ffmpeg 口径不匹配：误报多就放宽、漏报多就收紧 |
| 某 sampler/schedule 组合的 `score` 显著更高（样本 ≥20） | 换默认组合 |

**三道闸门**（缺一条就会变成"自我损坏"）：

1. **样本量**：<30 只报不应用；
2. **影子验证**：候选值先在**同一批历史基准**上回放（引擎侧用 `_bench` 已有跑次；Agent 侧
   直接用 `evaluation/optimizer.py` 那套"候选 → 再评 → 赢了才落"），赢了才 `accepted`；
3. **可回滚**：`tuning.json` 每次写前备份，且 `applied` 记录带 `basis`/`evidence`；
   一条 `status=failed` 的后续跑次即可回滚（自动，且回执里明说回滚了哪条）。

**已落（2026-11-06 前两条 / 2026-11-07 第三条）**：`tuning.suggest()` ＋
`GET /api/v1/knowledge-loop/tuning/suggest`（`tests/tuning_suggest_test.py` 31 条 ＋
`tests/qc_reviews_test.py` 53 条）。四条现象规则里落了**前三条**，且都只报不应用：

- 失败聚集：某族 `failed` 过了样本闸门、且最大 `error_code` 占比 ≥50% ⇒ 出候选。
  50% 是刻意的下限 —— 三类各占 1/3 说明是三件事，并成一个结论就是编归因；
- 步数平台期：按 `runs.meta.steps` 分桶比均分，最高档与次高档差 <0.01 ⇒ 建议降回次高档
  （`meta` 里没 `steps` 就不算，**不拿 `elapsed` 反推步数**）；
- 质检命中率（2026-11-07）：见下方「人工复核」小节 —— 分母不再是"触发次数"，而是人判过的
  结论，方向（放宽/收紧）由误报与漏报谁多决定，幅度只是起手值。

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

**人工复核与阈值命中率（2026-11-07 落地，原 `notEvaluated` 里的 `qc-threshold-hit-rate`）**

缺口不在统计代码，而在**分母**：`video_quality_checks` 只记"报了哪几条缺陷"，不记"人看过之后
判它对还是错"，所以 `black` 报了 12 次到底是 12 次真黑场、还是 9 次片头淡入的误杀，机器答不出。
补的是这张表 ＋ 一条统计 ＋ 一条规则：

- **表** `qc_reviews`（`core/models.py`，**单开一张**而不是给 `video_quality_checks` 加列）：
  理由三条 —— 漏报（该报没报）**没有 issue 行可挂**；一次质检里同一 `code` 可能报两段，
  复核粒度是 `(check_id, code)` 而不是 issue 行；复核是**事实层**，不该被 QC 重跑覆盖掉。
  故**不建 UNIQUE 索引**（旧库不必手工补索引，改判在服务层做覆盖式 upsert）。
- **三值闭集** `services/qc_reviews.py::VERDICTS` = `hit` / `false_alarm` / `missed`；
  `record_review()` 对 `(checkId, code)` **覆盖式改判**（先判误报、看放大帧后改判判对，那是
  同一件事的结论变了，追加会把命中率按人的犹豫稀释）。`code` 必须落在
  `technical_qc.ISSUE_CODES` 闭集里 —— 放任意字符串进来会长出"没人认得、也没有阈值可调"的行。
- **口径**（直接影响数字，写死在 `qc_reviews` 模块头）：命中率 `hitRate = hit / (hit + false_alarm)`
  （分母只含**触发过且判过对错**的），漏报率 `missRate = missed / reviewed`（分母＝该 code 全部
  复核）—— 两者**不共分母**：混在一起会让"漏报多"看起来像"命中率下降"，而这俩要调的阈值方向
  正相反。没判过 ⇒ `hitRate = None` **不是 0**（0 会被读成"从没误报过"，那是最危险的假绿）。
- **门槛** `MIN_REVIEWS = 10`（复核是人工成本，30 条要好几天；同一条规则的错法又很集中）。
  不足只摆数字、**不出候选**，并在 `notes` 里写"还差几条"。
- **阈值映射** `technical_qc.ISSUE_THRESHOLDS`：每个 code 指到 `qc` 伪档里的主字段
  （如 `black → blackminduration`、`fps.low → fpsmin`、`loud.peak → truepeakdb`），
  `relax` 说明放宽是"调高"还是"调低"（负值区间的乘法方向是反的，`truePeakDb=-1` 放宽要 ×0.8）；
  配合字段（黑场的 `blackpixthreshold`、冻帧的噪声阈值）只作 `aux` 提示、**不同时动**
  （两格一起动就说不清哪一格起效）。`spec.unreadable` / `spec.probe_failed` **有意不登记** ——
  它们的"误报"是 ffprobe 环境问题，改阈值等于把探测失败藏起来。
- **候选怎么出**：`false_alarm_heavy`（命中率 <0.5）⇒ 放宽；`miss_heavy`（漏报率 ≥0.3）⇒ 收紧；
  幅度是 `_next_threshold()` 的 **×1.25 / ×0.8 起手值**，`why` 里如实写明"方向有据、幅度是建议
  起手值，不是从数据里算出来的"。
- **入口**：`GET /qc/hit-rate`（按 code 的统计＋红线 `notes`）、`GET /qc/pending`（最近 QC 记录
  与还没判的 code，直接摆到人面前 —— 表建好了没人填，命中率永远是 `None`）、
  `GET/POST /qc/reviews`、`DELETE /qc/reviews/{id}`（撤销＝打错码）；前端在知识闭环页
  「调参候选」与「写一条覆盖」之间，命中率表 ＋ 待复核列表 ＋ 判对/误报/漏报/撤销四个按钮。
- 守卫：`tests/qc_reviews_test.py` 53 条（改判不新增、未知 verdict/code 拒、`hitRate=0.5`、
  `missRate=0.3333`、`None ≠ 0`、映射不漂、误报→放宽、漏报→收紧、负值区间方向、门槛不足
  不出候选、候选恒 `canApply=False`），其中 7 条走 **HTTP 层**（`TestClient`）：信封里少一层 `data`
  或 `checkId` 漂成 `check_id` 服务层都看不出来，非法 `verdict`/闭集外 `code` 要 400、
  `DELETE` 第二次要 404（撤销不静默成功）。

**质检反哺引擎参数（2026-11-07 落地，缺口 6）**

原来这条路上**两边各自都是通的**：QC 会写 `kind="qc"` 跑次、`suggest()` 也会读跑次，
但它们**从来没有对上过** —— QC 落在伪档 `video_spec`（说的是"这是规格判据"，不是"哪族出的"），
引擎落在真档 `h3`/`wan22`/`krea2`，没有共同键。「这一族最近出的片质量掉了」因此永远
成不了一条**关于这一族**的结论，只能重跑这一镜。

- **归因键** `outcome_index.records`（新表 `(record_type, record_id)` 主键 ＋ `stage`/`run_id`）：
  由 `bridge.run_job` 在**刚拿到 `taskId` 的那一行**登记 —— 那是 `stage` 与
  `video_generation_id` **唯一同时出现在手上**的时刻，过了这行就再也对不上。
  ⚠️ 不要再打 `video_generations.task_id` 的主意：那一列是给**云 provider** 存轮询号的，
  引擎这条路本来就不写它（`recover_video_tasks_on_startup` 里明文这么讲），拿它当引擎
  taskId 用会让"续跑轮询"与"引擎任务号"两种语义混在一格。
- **质检侧**：`_index_qc_run` 查归因表，把 `producerStage`/`engineRunId` 写进 QC 行的 `meta`。
  **查不到就不带这两格**（不是填 `null`）—— 云 provider 的片没有"引擎档"，
  填一格空值就等着被"最近一次引擎档"之类的猜测填上，然后拿别人的片去降这一族的步数。
- **读向** `engine/qc_feedback.py`：`stage_stats()`（按档报条数/通过率/失败 `code` 分布，
  `attributed` 与 `unattributed` **分开报**，只报"20 条 QC"会让人以为这 20 条都能判族）、
  `pass_by_steps()`（同族按 `meta.steps` 分桶算 **QC 通过率**；拿不到步数的单独计
  `missingSteps` —— 那是"这次比不了"，不是"没通过"）、`gate(high, low)` 三档。
- **闸门三档必须分得开**：`insufficient`（每桶不足 `MIN_BUCKET=5` 条 ⇒ **判不了**）、
  `pass`（低档通过率没明显下滑）、`blocked`（掉出 `TOLERANCE=0.1` ⇒ 别降）。
  把 `insufficient` 当 `pass` 就是"拿空样本降质"，这是这条链上最危险的假绿。
- **接线**：`tuning._steps_candidates` 出「降步数」候选**之前**问闸门；`blocked` 当场不出并
  把理由写进 `notes`（**不许静默丢弃** —— 静默丢弃和"没有候选"在回执上长得一模一样），
  放行则把 `qcGate`（两桶条数/通过率/落差）随候选带出去，人能看到凭据。
- **顺带查实的真缺陷**：引擎跑次**根本不记 `score`**（`runtime._index_run` 没有这个字段）⇒
  那条"步数平台期"规则在真机上一直靠"没有分数 ⇒ 没有候选"**空转**。所以本轮闸门用
  **通过率**（离散、有人/有判据看过）而不是分数（连续、且根本没数）。
- **降级**：索引库坏 / 归因表空 ⇒ 一律答"判不了"并退回原行为，主链路不因统计不可用而卡；
  `bridge` 那边登记失败只记一条 `engine-record-link` warn（少一条归因只是少一条统计）。
- 守卫：`tests/qc_feedback_test.py` 38 条（归因折得回/覆盖不并列/未知类型拒、
  QC 行真带出 `producerStage`、云 provider 不带这两格、按档聚合与未归因分开报、
  分桶与 `missingSteps`、闸门三档、**`blocked` 真的让候选消失且留下理由**、
  四条 AST 接线判据、坏库降级），并新增两条设计稿差异：`stage` 语义**不动**
  （QC 行仍落 `video_spec`，族信息只进 `meta` —— 改 `stage` 会让 `_failure_candidates`
  把 QC 失败并进引擎族的失败率里）、`insufficient` 时**候选照出**（只是把"闸门没数据"带出去）。

**最后一条仍有意没落**：换 sampler/schedule 默认等于换族语义，`TUNABLES` 里有意没这两格
（`notEvaluated` 里照旧逐条写明原因，别读成"没问题"）。`candidates[].canApply` **恒为 false**：
影子验证（闸门②）不在本模块，没做完就给"可应用"等于把闸门换成一个布尔值 —— 命中率补上了也不改
这条。

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

| 阶段 | 状态 | 落点 | 判据（主干 `tests/knowledge_loop_test.py` 本节落地时 131 条、现 **178**（见 §12.5）；各阶段另挂专测，见「落点」列） |
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
15. **修复闭环的四条判定口径**（缺口 9，`services/repair_closure.py`）：
   * **快照只存缺口键，不存全盘清单**：键是 `kind:小写文件名`（`missing:xxx.safetensors`），
     文件路径只留目录以下的小写文件名 —— 同一份工作流换个盘符不该变成"另一个问题"；
     明细不存是因为它三个月后一定过期，而"当时有多少条、哪些键"是能被后来的诊断对上的。
   * **按键集合判定，不按条数**：应用后新冒出一条无关缺口，不算这次修复失败（`resolved`）；
     只看"当时那个键还在不在"。反过来，键的缺席要配 `truncated` 才能下结论 ——
     `diagnose` 有条数上限，被截断时缺席**可能只是没列出来**，一律判 `unverified`。
   * **`stale` ≠ 修复失败**：落单时该键本来就不在（人按旧快照提的案、或已被别的改动带走）。
     所以提案上存的是 `findingBefore.present` 的三态：`true`/`false`/**`null`**。没有快照时
     必须存 `null` 而不是 `false` —— 存 `false` 会让每次应用都判 `stale`，把"没基准"
     说成"你抄错了"。
   * **回滚不重新诊断**：改动都撤了，问题回来是必然的；重跑诊断只会把回滚记成又一次失败。
     回滚直接判终态 `rolled_back`、按 `partial` **覆盖**应用期那条跑次 —— 留着应用时的
     `ok` 会让修复成功率虚高。
   **跑次**：`kind="repair"`、`run_id=repair-<proposalId>`（幂等覆盖），映射
   `resolved→ok`／`still→failed`／`stale→partial`／`rolled_back→partial`；**`unverified` 不写** ——
   分母里塞"没测过"会把成功率稀释成一个假的数字。守卫 `tests/repair_closure_test.py` 67 条
   （含 6 条 AST 接线判据：`apply` 必须调 `_verify_proposal`、`diagnose` 必须调 `_persist_snapshot`）。
16. **Skill 注入台账的五个坑**（缺口 10，`services/skill_usage.py`）：
   * **`with sqlite3.connect(...)` 不关连接**：那个上下文只管**事务**（提交/回滚），
     连接一直开着 ⇒ Windows 上删库 `PermissionError: WinError 32`、进程长期跑是句柄泄漏。
     必须 `try/finally: conn.close()`（本模块 `_session()`）。测试里删库还要连
     `-journal`/`-wal`/`-shm` 一起清，否则偶发红。
   * **明细走"出参"不走落库**：`load_agent_skills` 在同一跑次被调用两次
     （`build_agent_config` 建配置一次、`default_generate` 拼提示词一次），在函数内写库会把
     **一次注入记成两笔样本**。所以它只回传 `sizes`/`dropped`，由跑次收口写。同 `run_id`
     覆盖是第二道保险（跑次重试也不该翻倍）。
   * **排序只许发生在超预算分支**：`total > budget else list(bindings)`。一旦在预算充足时也排，
     提示词顺序就会随统计抖动而变 —— 每次微调 skill 都产生一大段无法判读的 diff。
   * **裁剪理由的文案不许改**：改的是"谁被裁"，不是"为什么裁"，所以通知仍是
     `（超预算）`。既有 `skills_test` 逐字断言那句文本 = 免费的回归守卫。
   * **`priority` 缺省 0 要用序号兜住**：DB 配置里不写 priority 的条目全是 0，不兜就集体
     抢到最前面（比任何 `priority: 3` 都靠前）；而测试造景里 `alpha` 必须**显著长于** `beta`、
     预算取 `alpha_len + 5` —— 取 `beta_len + 5` 会让 alpha 一开始就装不下，
     「按 priority 裁 beta」与「低成功率让位裁 alpha」两个用例都退化成同一个结果，
     看着像实现错了。
17. **质检反哺的四个坑**（缺口 6，`engine/qc_feedback.py`）：
   * **"没有共同键"才是根因**：一开始容易当成"再加一条规则"，但 QC 行与引擎行之间
     **压根没有一列能 join**（一个只有 `video_generation_id`，一个只有 `taskId`）——
     规则写出来也只能拿 `video_spec` 这个伪档当钥匙，结论永远落不到某一族头上。
     真正的动作是**先造归因键**（`outcome_index.records`），规则才有地方站。
   * **归因只能在提交那一刻做**：`bridge.run_job` 里"刚拿到 `taskId`"那一行是唯一窗口，
     那时 `stage` 与 `record_id` 同时在手上；所以登记调用点必须紧跟 `engine_runtime.submit`，
     挪到 `await` 之后、挪到视频落盘之后都不行（那时 `record_id` 早就不在作用域里）。
   * **查不到就不填**：`_producer_of` 返回空字典（不是 `{"producerStage": None}`）。填了空值，
     下游一个 `or` 兜底就会把"未知"回升成"最近一次引擎档"，然后拿别人的片降这一族的步数 ——
     这类错误的特征是**回执全绿、数字全错**。
   * **判不了 ≠ 通过**：`insufficient` 这一档存在的全部意义是"样本不够，别说结论"。
     实现上很容易写成 `if verdict == "blocked": skip` 然后其他一律放行 —— 那样库一坏、
     归因表一空，闸门就自动变成"全部通过"。所以守卫里有一条专门盯着源码里
     不许出现 `insufficient` 直通的写法。
18. **评测面两次"看着很专业"的翻车**（缺口 8，`film_check.py` ＋ `scorer.score_grid_prompts`）：
   * **判不了 ≠ 通过，也 ≠ 0 分**：成片四维里任一维没有样本（没有 QC 记录 / 没有人复核过）
     就不进总分，而不是"当 0 分算"。折 0 会让一集没质检的好片被判 `blocked`，
     折满分会让一集没质检的烂片直接 `ready` —— 两种都是**把"不知道"谎报成结论**。
     实现上总分只能按**有值的维度**等权平均，`missing` 逐项点名，`verdict` 降成 `partial`。
   * **分母选错，指标越算越好看**：缺陷面如果用"全部镜头"当分母，一集里大量没出片的镜
     会白送干净；所以分母是**有 QC 的镜**。人工复核的命中率如果用"全部复核"当分母，
     `missed`（漏报）就会稀释命中率 —— 漏报没有 issue 行可挂，只能单列进 notes，
     分母只含 `hit + false_alarm`。
   * **点五次评测 ≠ 五条样本**：评测跑次必须幂等（`run_id=film:<集>`），否则同一条片子
     点几次就变成几条独立样本，喂给 `param_optimizer` 的 `repeat>=2` 直接变成自欺。
     另开 `kind="eval"`：混进 `"agent"` 会把"评测跑了几次"算进 agent 成功率。
   * **"含中文就扣分"会把合规产出判死**：宫格 Agent 的系统提示词**自己规定**要把位置写成
     「格1」、参考图写成「图片1」、骨架里还有「首帧/尾帧」。按"有没有中文"判，完全合规的
     产出会被判不合格；判据只能是**中文占比**（标签是个位数百分比，真写成中文散文在 40% 以上）。
   * **空产出会白送分**：占位符维第一版是"文本里搜不到 `[` 就给满分"，
     `payload=None` 时文本是空串 —— 当然搜不到，于是白送 15 分。分母必须是**全部待判文本（含空的）**，
     分子只数"非空且干净"。凡是"找不出坏东西就给满分"的判据，都要先问一句"根本没有东西"算不算坏。
   * **评测与生产必须是同一套解析**：宫格载荷的三段解析从 `routers/grid.py` 上收成
     `services/grid_payload.py`，评测不自己认一套格式 —— 否则线上换一种写法时，
     评测照旧打高分，那是最典型的假绿。

---

## 8. 明确不做（避免过度设计）

* **不做**"全自动改源码"：`profiles.py`/`pipeline.py` 的默认值只通过 `tuning.json` 覆盖，
  不改文件；
* **不做**"质检一失败就自动改参数"：闸门只**拦候选**，改参仍要人核（`set_field`）——
  自动改会把一次误判连锁污染到后续全部出片，而"质量掉了"恰好是最难自己发现的那类错；
* **不做**"知识全量注入"：只 top-N + 按需检索；
* **不做**"向量库"：现有 bm25 + trigram + 维度筛选在本仓的数据量级（万级条目）足够，
  引入 embedding 会多一套部署与一致性负担（`script_library` 已有的 RRF 融合就是为此克制过）；
* **不做**"跨项目共享经验"：`scope` 已按 `drama:` 隔离，跨项目默认不互见
  （不同题材/画风的经验互相污染比稀缺更糟）；
* **不做**"工具集落库"：`agent_configs` **不开 tools 列** —— 工具集的真源是代码里的装配语句
  （`_TOOL_FACTORIES` 那几个 `create_*_tools`），再存一份进库就是两份真源：改了代码没改库时
  "调优结论"与"线上实际"会各说各话，而这种不一致**不报错**。所以工具集调优只落
  优化史结论（`target="tools"`，`persisted=false` ＋ `persistSkip`），人看完改代码；
* **不做**"引擎参数进 `agent_optimizations`"：QC 阈值与采样档走 §4.3 那条「质检反哺」链
  （`profiles` 覆盖层 + `technical_qc` 伪档）。塞进这张表会让"跑次经验"和"档位参数"
  共用一套 version/parent 语义，而档位参数的回滚粒度是**整个 profile**，不是一次实验。

---

## 9. 与既有模块的接线点（改哪里，不动哪里）

| 现有模块 | 改动 | 不动 |
|---|---|---|
| `services/task_logger.py` | 落盘后加一次 runs 双写（失败只 warn） | JSONL 格式与字段 |
| `routers/traces.py` | 新增 `/runs` 聚合查询（转调 `outcome_index`） | 现有 `/traces*` 端点 |
| `agent/runtime.py` | `_tool_factory` 加一行 `create_knowledge_tools()`；**已接**：跑次收口写 `skill_usage`（`tests/skill_usage_test.py` 37 条）；**已接**：`options["toolsRaw"]` 在 MCP 合并**之后**裁剪（`merged.only(...)`，非法名当场 `ValueError`，`missing` 进 trace） | 主循环与重试 |
| `agent/skills.py` | **已接**：`load_agent_skills` 超预算裁剪（`priority + 降级惩罚`），`sizes`/`dropped` 只做出参 | 预算够时的注入顺序、`priority` 语义、`SKILL.md` 自述机制 |
| `agent/qc_retry.py` | 重跑结束追加一次"复盘提案"调用（异步、超时即弃） | 只重写失败镜的既有纪律 |
| `engine/profiles.py` | 不改；`pipeline` 取参处已改走 `tuning.resolve`／`apply_to_request` | `PROFILES` 表本身 |
| `services/engine/tuning.py` | **已接读向**：`suggest()`（失败聚集／步数平台期，三道闸门如实标）＋ `GET /api/v1/knowledge-loop/tuning/suggest`（`tests/tuning_suggest_test.py` 31 条） | 候选**永不自动应用**：`canApply` 恒 false，人核过再 `set_field` |
| `services/technical_qc.py` | **已接**：`qc_thresholds()` 逐键过 `tuning.resolve("qc", …)`，回执带 `thresholds-overridden` 警告（`tests/qc_thresholds_test.py` 28 条） | 阈值默认值（`TECH_QC_THRESHOLDS` 仍是唯一真源）与 ffmpeg 判据 |
| `services/outcome_index.py` | **已接**：`records` 表（`link_record`/`record_of`）＋ 按主键取单行的 `get(run_id)` —— 缺口 6 的**唯一归因键** | `runs` 表结构、既有 `record_run` 的幂等覆盖语义、`query`/`samples` 的排序与白名单 |
| `services/engine/bridge.py` | **已接**：`run_job` 在 `engine_runtime.submit` 之后立刻 `_remember_record` 登记"这条记录是谁出的"（失败只记 `engine-record-link` warn） | 请求构建 `request_from_*`、装配校验、回执字段 |
| `services/engine/qc_feedback.py` | **已接**：质检结论 → 按档通过率／按步数分桶／降档闸门三档（`tests/qc_feedback_test.py` 38 条） | 不改任何族数学、不写参数（改参数仍只走 `tuning.set_field`，那是人的裁决） |
| `routers/lora_train.py` | 加 `datasetRef` 参数与 `training_jobs` 登记 | 现有参数表与执行链 |
| `routers/knowledge.py` | 挂 `/entries` 子路由（新 router 文件，避免与语料端点混） | `/search` `/documents` `/upload` 等 |
| `routers/repair.py` | 接 `repair_closure`：`diagnose` 落快照、`propose` 收 `finding`、`apply`/`verify` 复诊、`rollback` 记终态（`tests/repair_closure_test.py` 67 条） | 提案落盘格式、`proposalId` 单参数口、可改前缀白名单、应用前全量备份 |
| `agent/evaluation/film_check.py` | **已接**：成片四维评测（`tests/film_check_test.py` 37 条）；只读四张事实表、`kind="eval"` 幂等跑次 | 不改 QC 判据、不重跑 ffmpeg/不调模型（评测只读数） |
| `routers/evaluation.py` | **已接**：`POST /film/{episode_id}`（`record` 默认 true）；坏 id 走 400、无成片走 200 且 `verdict=empty`；**已接**：`POST /tune-skills/{case_id}`、`POST /tune-tools/{case_id}`、`POST /tune-prompt/{case_id}`（`tests/tool_optimizer_test.py` 90 条；`tests/tune_runtime_test.py` 76 条含提示词格十项＋HTTP 八项＋**真链路端到端十二项**；`tests/optimizer_test.py` 64 条含基线取现网值＋版本号跨运行连续＋抖动带双侧＋两条上限越界即拒＋`iterations=0` 真跑收尾；`tests/evaluation_route_test.py` 33 条，含两条端到端逐步账） | 既有 case 评测四个端点、`GET /cases` 无 try（坏 JSON 直接 500）、`/tune/{case_id}` 的入队/等待口径、`POST /optimize/{case_id}` 的同步判据（`optimizer.ts` 已移出本仓，现由 `optimizer_test.py` 自守） |
| `agent/tool.py` | **已接**：`ToolRegistry.only(wanted)`（**按注册表键或按 `tool.id` 两套口径**匹配、顺序跟注册表、`missing` 如实上报）＋ `names()`（模型侧函数名，与 `ids()` 的注册表键分开叫） | 注册/查找/执行本体，`ids()` 的既有语义 |
| `agent/evaluation/evaluator.py` | **已接**：`step_trace()`（单步对照，`stats.unavailable` 区分"没明细"与"没调工具"）＋ `_report_steps()`（五条分支统一收口）＋ `evaluate_case(..., skills=, tools=)` 的按次覆盖 | 各 kind 的评分函数本身、报告既有字段的语义 |
| `services/optimization_log.py` | **已接**：`TARGETS` 四格（＋`tools`）、`REASONS` 十值（＋`tool_needed`/`tool_wanted`/`tool_unjudged`） | 写口"失败只 warn"、读口"必须报 available=false"、`mark_persisted` 的语义 |
| `services/grid_payload.py` | **已接**：宫格载荷解析上收为单一真源，`routers/grid.py` 保留同名别名（调用点零改动），评测同源 | 解析口径本身（camelCase 兼容、空 prompt 丢弃、递归下降、先 `json.loads` 再抠围栏） |

## 10. 前端呈现（2026-11-07 落地）

页面 `frontend/app/pages/knowledge-loop.vue`（桌面端 `desktop/frontend/src/pages/knowledge-loop.vue`，**逐字节同构**；两端 `composables/useApi.ts` 里的 `knowledgeLoopAPI` 也同一份，接口层实况与守卫见 §10.6），导航入口「闭环」，路由 `/knowledge-loop`。

**为什么单开一页，不并进 `/knowledge`**：`/knowledge` 是**仓外剧本资料**的读写口（5403 份 PDF 的索引 + 覆盖层），本页是**本仓自己跑出来的经验**（跑次 → 蒸馏 → 人审 → 投喂/调参）。两者备份策略、重建代价、误删后果都不同；并页会让「删语料」与「删经验」这两个动作长得一样。

### 10.1 页面结构（左导航分区 + 分区懒加载）

| 分区 | 读的接口 | 写/触发的接口 | 图上能看见的诚实口径 |
|---|---|---|---|
| 概览 | `GET /report?includeCandidates`、`GET /scheduler` | `POST /runs/rebuild`、`POST /distill`、`POST /lift/refresh` | 事实层按阶段列 条数/失败率/均分/P50/P95；`available=false` 与「0 条」分开摆；定时复盘区**只读**（默认关就写「未启用」）；蒸馏候选没给文字归因就直说「只有统计信号」，凭据那排 run_id 原样列出 |
| 经验台账 | `GET /entries`、`GET /entries/{id}` | `POST /entries/{id}/status`、`POST /entries` | 状态/类型/范围三筛 + 真分页；迁移按钮**由 4 条白名单边生成**（`proposed→accepted\|rejected`、`accepted→deprecated\|rejected`、`deprecated→accepted\|rejected`、`rejected→proposed`），前端不另造边；`lift: null` 显示「样本不足」而不是 `0`；新建条目明示是**叠加**（同标题 `support+1`），所以页面**不提供编辑按钮** |
| 检索测试 | `GET /search` | — | 左 form／右 result 双栏；`ranking`（bm25 / like / recent）必须显示，三种排序不可比；空查询会被拒而不是回落成全表 |
| 投喂与数据集 | `GET /feed/preview`、`GET /datasets`、`GET /datasets/resolve` | `POST /feed/export` | L1 编译预览原文进 `<pre>`，`truncated` 单独警示；L3 数据集表格带 `itemsExists` 徽章，逐条「核一遍」走的判据与训练 `datasetRef` 是同一个函数 |
| 调参与训练 | `GET /tuning`、`GET /tuning/suggest`、`GET /training` | `POST /tuning`、`DELETE /tuning`、`POST /training/scan`、`POST /training/models/{id}/promote\|demote` | 覆盖层 `source !== 'default'` 才给「清除」；只读候选的 `canApply` 恒 false，按钮不画「应用」，只画「按这条写覆盖」（prefill 到表单，仍要人写 basis + evidence）；承诺默认档必须带凭据、撤销要写原因；训练产物**不会自动生效** |
| 质检命中率与人工复核（同属"调参与训练"分区） | `GET /qc/hit-rate`、`GET /qc/pending`、`GET /qc/reviews` | `POST /qc/reviews`、`DELETE /qc/reviews/{id}` | 一表一行按 issue `code` 列 触发/复核/判对/误报/漏报/hitRate/missRate，**没判过显示 `—` 不是 0**（0 会被读成"从没误报过"）；`reviewed < 10` 行标「样本不足」且该 code 不出候选；待复核列表把最近 QC 的未判 `code` 直接摆出来逐条四个按钮（判对/误报/漏报/撤销），撤销是用来改错码的 —— 改判本身是"再点另一个按钮"，不是"删了重录" |
| 成片评测（2026-11-07 补的第六个分区） | `GET /dramas`、`GET /dramas/{id}`（只为一件事：选出那一集） | `POST /evaluation/film/{episodeId}?record=true` | 四维等权（出片覆盖 / 技术规格 / 缺陷面 / 人工复核）：判不了的维度显示「判不了」并且**不参与总分**，总分一维都没有时显示「判不了」而不是 `0`；`verdict` 四档里 `partial` 用中性色（它的意思是"有几维没数据"，既不是通过也不是不通过）；复核行把**漏报单独列出来**并注明"没有 issue 行可挂、不进分母"；`recorded=false` 时明说这条没落库 |

首屏只打 `/report` + `/scheduler` 两次请求，其余分区靠 `watch(view)` 懒加载（`loaded` 标记防重入）。

「算不了的维度」一栏**逐条带出 `why`**（2026-11-07 补）：只摆 `rule` id 时，`qc-threshold-hit-rate`
会被读成"这个维度有问题"，而它其实是"复核条数还没到 10 条门槛"——原因不显示，人就会以为缺功能。

### 10.2 与参考项目复用了什么、改了什么

| 参考项目 | 复用的结构 | 本仓的适配差异 |
|---|---|---|
| `reference/dify`（`web/features/new-rag/new-knowledge-list.tsx`、`knowledge-space-shell.tsx`） | 左侧分区导航 + 右侧列表的**信息架构**，以及「列表页一行就是一个可核查对象」的组织方式 | 不做 dify 的 `/datasets` 卡墙与创建向导：本仓的「数据集」是导出产物（L3），不是用户先建后灌的容器；分区导航是 5 个视图而不是多个路由 |
| `reference/ragflow`（`dataset/dataset/use-dataset-table-columns.tsx`） | 数据集表格的列口径（名称 / 条数 / 角色 / 完整性 / 时间），以及状态用徽章而不是文字颜色表达 | 列名与判据换成 `manifest` 真有的字段（`count`/`roles`/`rights`/`itemsExists`）；不做多选批量操作（导出集是单份核验） |
| `reference/ragflow`（`dataset/testing/testing-form.tsx`、`testing-result.tsx`） | 检索测试的**双栏 form/result** 与「先调参再跑一次」的交互 | 结果卡片多一行 `ranking` 标签（三种排序不可比是硬约束，不藏起来）；`available=false` 用独立警告条而不是空列表 |
| 两者共有的抽屉详情 | 点击行 → 右侧抽屉，列表保持原位 | 抽屉内容按本仓字段重写：`links`（挂哪条跑次/镜头）、`usage`（被注入过几次、成功失败各几次）——这是 dify/ragflow 没有的「结果反查」 |

### 10.3 实现约束（改这页之前先看）

* 样式：两个 `kb-*` 页面是**两个组件**，scoped 样式不跨组件，所以 `knowledge-loop.vue` 里带一份同名 `kb-*` 副本 + 本页专有的 `kl-*`；断点沿用 `1023px` / `767px` 两档，由 `frontend/scripts/check-responsive.mjs` 硬判。
* 类型：闭环检索回执叫 `KnowledgeLoopSearch`，**不能**叫 `KnowledgeSearch` —— `/knowledge` 那个回执已占用该名，TypeScript 同名 `interface` 会**合并**而不是覆盖，两侧字段都得齐，任一侧初始化对象都缺字段（2026-11-07 实测报错）。
* 候选 `KnowledgeCandidate` 是**并集**：蒸馏候选给 `pattern/key/kind/score/stats`，调参候选给 `rule/stage/field/current/proposed/why/samples/canApply`。页面取 `stats` 兜底显示「作用面/样本量」，**不替后端编归因**。
* 导航高亮：`/knowledge` 的 active 判据改成 `path === '/knowledge' || startsWith('/knowledge/')`，否则 `startsWith('/knowledge')` 会把 `/knowledge-loop` 也点亮。
* 守卫：`node scripts/check-parity.mjs`（两端逐字一致）、`node scripts/check-responsive.mjs`、两端 `vue-tsc`。`composables/useApi.ts` 与 `pages/knowledge-loop.vue` 都是**两端同名同内容**，改一侧必须同步另一侧（`check:parity` 会红）。
* ⚠️ 曾经别在 `useApi.ts` 的**注释**里写"带 API 前缀的完整调用形状"：`frontend_api_coverage_test.py` 是纯文本扫描，注释里的一条路径会被当真调用点去反查后端，而闭环**没有裸根端点**，当场红（2026-11-07 实测，写了两次才注意到）。**2026-11-08 已从「纪律」改成「机制」**：该自检现在先把注释遮蔽成等长空格再扫描（`mask_comments` 认引号，`.vue` 只遮 `<script>` 区内的 JS 注释），注释里记老路径不再红 —— 但**注释里写错的真调用照样会被抓**，别把这条当成放宽。

### 10.4 质检命中率卡片（2026-11-07 补的第三块）

放在「调参候选」下面、「写一条覆盖」上面 —— 顺序是刻意的：先看候选（机器认为哪条该动）→ 再看命中率（机器的依据够不够）→ 最后写覆盖（人的结论落盘）。三条实现约束：

* **`—` 与 `0` 必须分开**：`hitRate === null` 渲染成 `—` + 「未复核」，`0` 才是「判过但一次没对」。把两者都画成 0，页面就会在"还没人看"的时候显示"阈值从没判对过"，人照着改就改错了方向。
* **改判不是新增**：四个按钮里「判对/误报/漏报」都是对同一 `(checkId, code)` 的覆盖（同一个人看放大帧后改主意是常事），只有「撤销」才是删除；所以按钮上不写"删除"，写"撤销"，并在按钮组旁边说明「撤销只用于判错编码」。
* **复核要能被找到**：待复核列表按 QC 记录倒序列出**还没判**的 `code`，已判的排在后面 —— 表建好了没人填，命中率永远是 `None`，所以"把人叫过来"这件事必须在同一屏里做完，不能让人自己去别处翻质检报告。

### 10.5 成片评测分区（2026-11-07 补的第六块，缺口 8 的下半）

前五个分区看的都是**跑次/经验/提示词**这一层，成片质量从来没人量过 —— 而花钱、决定交付质量的恰恰是成片。这一块把 `POST /evaluation/film/{集}` 摆到页面上，四条实现约束：

* **两级选择，不做"只有一部剧就默认选中"**：本仓可以有多部剧，默认选中会让创作者在自己没看清评的是哪一集时就把结论记进闭环。所以先选剧组、再选集，两个 `select` 都是显式的。
* **`判不了` 是一个独立状态，不是 0 也不是满分**：`dimensions[].judged=false` 的行显示「判不了」，总分一维都没有时显示「判不了（一维都判不了）」；`verdict=partial` 用中性色徽章，文案是「有维度判不了」—— `ready`/`blocked` 才是通过/不通过。
* **漏报单独列、并且写明它不进分母**：`review.missed` 没有 issue 行可挂，如果混进分母，命中率会被"没被发现的错"稀释——那是把系统看不见的东西算成系统的功劳。
* **`empty` 是"这集还没分镜"**：不是 0 分。页面在没出片时给出的是这句话 + 一条 `info` toast，而不是一条绿色的"评测完成"。

⚠️ 页面这一块**只读**接口，不触发 ffmpeg/模型：重跑技术质检不但贵，还会与质检页的口径打架（两处显示不同的黑场数，人只会信错的那边）。真要重跑，那是质检页的按钮，不是这里。

### 10.6 接口层（2026-11-08 补齐：此前这一页一进来就崩）

**发现的真问题不是"某个方法参数写错"，而是整个 `knowledgeLoopAPI` 不存在**：页面顶部 `import { dramaAPI, knowledgeLoopAPI, type FilmEvalReport, … } from '~/composables/useApi'` 里的 `knowledgeLoopAPI` 与 12 个类型，`useApi.ts` **一个都没有导出**（`git log -S knowledgeLoopAPI -- useApi.ts` 为空 ⇒ 从来没写过，不是被谁删了）。后果是 `import` 拿到 `undefined`，页面初始化第一句 `knowledgeLoopAPI.report(...)` 直接 `TypeError` —— 不是"部分功能不可用"，是**整页白屏**；而 `npm run typecheck` 早在报 `TS2724: has no exported member named 'knowledgeLoopAPI'`，只是没人跑。

补齐的 36 个口（页面实际用到的，**不为将来预留**）：

* **读**：`report`／`scheduler`／`taxonomy`／`entries`／`entry`／`entryVersions`／`entryVersion`／`audit`／`search`／`optimizations`／`feedPreview`／`datasets`／`datasetResolve`／`tuning`／`tuningSuggest`／`qcHitRate`／`qcPending`／`training`。
* **写**：`createEntry`／`updateEntry`／`deleteEntry`／`restoreEntry`／`revertEntry`／`setEntryStatus`／`rebuildRuns`／`distill`／`refreshLift`／`feedExport`／`setTuning`／`clearTuning`／`qcReview`／`deleteQcReview`／`trainingScan`／`promoteModel`／`demoteModel`。
* **评测**：`filmEval` → `POST /evaluation/film/{集}?record=`（与 §10.5 同一口）。

四条口径：

1. **`api.patch` 是新加的动词**。此前 `api` 只有 `get/post/put/del`，所以"前端全仓零 `PATCH`"不是没人用，是**根本没这个口**。后端 `PATCH /entries/{id}` 有意不提供 `PUT`（忘记带 `body` 的 `PUT` 会把几百字正文清空），前端因此也不得不用 `PATCH`。
2. **`changed` 有两种形态，别当一种读**：`PATCH` 回执给的是**改动字段名数组**（`[]` ＝ 请求合法但内容没变），而 `delete`/`restore`/`revert` 给的是 **bool**。`KnowledgeWriteResult.changed` 如实声明成 `boolean | string[]`；页面里 `res.changed === false` 只出现在 `restore`，是对的，但**照抄到 `PATCH` 上就会把"没改动"显示成"已更新"**。
3. **可选参数一律"不传"而不是"传空"**：`Query(50, ge=1)` 这类整型参数收到 `limit=` 会 422；字符串参数虽然后端用真值判断（`if kind:`）能吃空串，但两类都不该靠后端兜。所以整型一律带默认值送出，字符串类走 `+ (p.kind ? … : '')`。
4. **`role` 缺省＝放行**（`_role_gate` 空值直接过）、但 `evidence`/`basis` 缺省＝拒：写口的"凭据核不过"返回的是 **400 + 人话**（"知识台账没接入 ⇒ 凭据核不了，按『核不过』处理"），**不是** 422 形状错。看错误码别只看数字。

**怎么守住**（两层，都跑过）：

* 静态：`tests/frontend_api_coverage_test.py`（14 条判据）把前端每个 `api.get/post/put/patch/del` 的字面路径按 `@方法+路径` 反匹配 FastAPI 路由模板，并把路径里写死的 query 键与后端声明对账。本轮现状：**261 个调用点**（补之前 215）／36 个带 query 键的调用点，全对；`PATCH` 由"靠合成样本自证"变成**真有调用点**。
* 实打：36 个方法用 `TestClient` 各调一次（`with TestClient(app)` 起 lifespan，否则调参口会因为"台账没接入"报 400 —— 那是**缺启动事件**，不是接口坏）。验收看的是"路径与字段名被读懂"：写链路 `create → PATCH → versions → version/1 → status → revert → audit → search → delete → restore` 全程 200，`entry` 详情的 `tagList`/`version`/`deleted`/`usageSummary`、`entryVersions` 的 `current`/`deleted`、`audit.items[]` 的 `atNs`、`filmEval` 的 `dimensions[].judged` 与 `review.{hit,falseAlarm,missed,reviewed,hitRate}` 与 TS 声明逐字段一致。

⚠️ 页面/文档都按"已完成"写、而接口层从来没落地，是这套代码最贵的一类错：**静态守卫（同构、隔离、响应式）全绿、页面看着像做完了**，只有 `typecheck` 与真跑会说话。新增页面时先跑一次 `npm run typecheck` 再谈样式。

---

## 11. 知识关联与上下文注入（2026-11-08：`entry_links` 从只写变可读）

**发现的真问题**：`entry_links` 自蒸馏上线起一直在写（按 `stage`/`shot` 挂），但**全仓没有一处读它** —— `detail()` 只回显本条挂的 ref，既没有反向也没有邻居。于是"这条经验挂在哪个阶段上"这个**事实**存了等于没存：蒸馏每天产出 `scope=stage:video` 的条目，人问「这一镜/这一阶段踩过什么坑」时没有任何入口，只能退回去关键词检索 —— 而「SHOT-012」这种 id 在正文里根本不出现，检索永远命中不了。

这一层正是参考实现（graphiti 的双向边、cognee 的图补全、mem0 的相近记忆召回、auto-ontology 的图谱）把"一堆条目"变成"一张网"的地方。**但它的前提不是向量**：边已经在库里，先把边读出来。因此 `docs/knowledge-stack-audit.md` 对向量库／RAG 框架的结论（不采用）**不变** —— 这一层是确定性的，零新依赖，也不引分词器（中文用 3 字滑窗）。

### 11.1 两档 `via` 的证据强度不同，回执里分开、不混成一条分数

| `via` | 依据 | 强度 | 界面显示 |
| --- | --- | --- | --- |
| `shared` | 共用同一 `(refKind, refId)`（同阶段/同镜头） | **事实**（蒸馏真的挂过），`score=null` | `共用 stage:video` |
| `similar` | 标题或正文的 trigram Jaccard ≥ `SIMILAR_RATIO=0.6` | **启发式**，`score` 0–1 | `相似 0.72` |

混成一档的后果是真实的：`similar` 只说明「字面像」，而同题材、**结论相反**的两条在这本台账里字面恰恰很像（"夜景低步数会过曝" vs "夜景低步数反而稳"）。把相似度当合并依据，最先抹掉的就是这些分歧 —— 那才是台账最值钱的部分。所以：`similar` 只作提示，**自动合并明确不做**（§10 未做清单同款理由）。

### 11.2 三个函数、两条端点、一个工具

* `knowledge_ledger.related(entry, limit, include_deleted)` → `GET /entries/{entry_id}/related?limit=&includeDeleted=`
  回 `{found, deleted, available, links, items, counts:{shared,similar}, hint}`。三档如实分开：**不存在** ⇒ `found=false`（不抛 404，也不假装成功）；**已软删** ⇒ `found=true, deleted=true` 且 `items=[]`（否则就是绕软删看邻居的后门）。
* `knowledge_ledger.for_context(stage=, shot=, status=, ...)` → `GET /context?stage=&shot=`
  按 ref 反查，顺序 `confidence DESC, updated_ns DESC`。**两者都不给 ⇒ `total=0` + 人话**，不返回全量：「没点名上下文」被当成「全库都是上下文命中」，调用方会以为自己拿到了针对性经验。
* `knowledge_ledger.context_refs(ref_kind=, entry=, limit=)` → `GET /context/refs?refKind=&entry=`
  台账里**真被挂过**的 ref 清单＋条目数。没有它只能猜阶段名（`video`? `h3`? `krea2`? 三个都是本仓真实存在的字面值），猜错的表现是「查了没结果」，与「确实没经验」永远分不开。
* Agent 工具 `context_knowledge`（`app/agent/tools/knowledge_tools.py`，第三个知识工具）：`stage`/`shot` → 上下文反查；`entryId` → 邻居；**三个都不给 → 回 ref 清单并提示"给 stage 或 shot 才能取到针对性经验"**，即让模型别猜。

### 11.3 接线与守卫

* 端点 3 个，已登记 `docs/knowledge-base-contract.md` §3.2（总表 40→42）与 §9；`tests/knowledge_management_test.py` ⑧ 组会**双向核**（openapi 里的端点必须在契约文档出现，反之亦然）—— 本轮补完文档后 75/75。
* 前端只加 `relatedEntries`（详情面板「相关知识」；顺带把 `opened.links` 显示的「挂载」补上 —— 那也是只传不显示）。`/context`、`/context/refs` **刻意不给前端方法**：页面上「这一阶段有哪些经验」还没有稳定的信息层级，不为将来预留。
* `tests/knowledge_loop_test.py` 本节落地时 **159 条**（现 **178**，见 §12.5），本轮新增 20 条盯真行为：`shared` 档不带分数、跨 `kind` 不比相似（`fact` 与 `lesson` 字面再像也不是一件事）、已软删邻居默认不出现／`includeDeleted=true` 才回来、空文本相似度是 0 而**不是** 1、不给参数 `total=0` 且 `hint` 非空、驼峰 `includeDeleted` 不 422、工具侧"已删条目不回邻居"。
* `tests/frontend_api_coverage_test.py`：262 个调用点／37 个带 query 键的调用点全对；两端 `useApi.ts`、`knowledge-loop.vue` 逐字同构（`npm run check:parity` 绿），两端 `npm run typecheck` 零错。

⚠️ 踩到的坑（下次别再踩）：新端点**只加后端不写契约文档**，`knowledge_management_test` ⑧ 组会当场红；而它红的时候若控制台是 cp936，`✗` 打印会抛 `UnicodeEncodeError`，看起来像"脚本崩了"而不是"有一条判据没过"（`PYTHONIOENCODING=utf-8` 才看得到真话）。

---

## 12. 投喂的作用域：写读键域闭合（2026-11-08）

### 12.1 缺口长什么样

`compile_for_prompt` 原来是这么选条目的：

```python
scopes = {"global"}
if agent_type:
    scopes.add(f"agent:{agent_type}")
```

而全仓写 `scope` 的地方只有三类，**没有一处写 `agent:<type>`**：

| 写入口 | 写下的字面形 |
|---|---|
| `agent/tools/knowledge_tools.py::propose_knowledge` | `EP<id>` / `DRAMA<id>` / `global` |
| `services/knowledge_distill.py` | `stage:<key>` / `shot:<key>`（并同时写 `links`） |
| `POST /entries`（管理接口） | 调用方给，实际就是上面两种 |

所以自动投喂**实际只对 `scope="global"` 生效**：人工把一条 `EP12` 的经验采纳（`accepted`）之后，
EP12 的下一次跑次依然注入不进去，而回执说的是「台账里还没有 accepted 的知识」——
库里明明有、且已经审过，读起来却像还没积累。这是「写入口与读入口的键域不一致」，
和 §11 那个 `entry_links` 只写不读是同一种形状：**两边都不报错，只是白写**。

### 12.2 落法

* `knowledge_feed.compile_for_prompt(agent_type=None, *, scopes=None, …)`：`scopes` 是
  **本次跑次真实拥有**的作用域，并进允许集合；**默认 `None` ⇒ 老的 `{global, agent:<type>}` 逐字不变**，
  不传的调用方行为零变化。
* `knowledge_feed.scopes_for(*, episode_id=None, drama_id=None)`：字面形对齐**写入口**
  （`EP12` / `DRAMA3`），非数字的不编、直接不产出。
* `runtime.append_knowledge_feed(..., scopes=[…])` 透传；
  `assemble_full_instructions` 从自己**已经拿到**的 `episode_id` / `drama_id` 算 ⇒
  两个入口不再各自为政，也不需要各调用方记得多传一个参数。
* `GET /feed/preview` 增 `episodeId` / `dramaId` / `scopes`（逗号分隔）——
  否则预览出来的是「只剩全局成法」的另一种结果，恰好看不见被漏掉的那批。

### 12.3 三种「空」必须分开报

回执新增 `scopes`（本次允许集合）与 `blocked`（kind 够格但作用域不匹配的条数）：

| 情形 | 回执 | 该去做什么 |
|---|---|---|
| 台账坏了 | `available=False` | 修库 |
| 有知识但进不来 | `text=""`, `blocked>0`, hint 点名作用域并列出允许集合 | 传 `scopes` / 查 `episodeId` 是否正确 |
| 确实还没积累 | `text=""`, `blocked=0` | 去审条目 |

混成一句的代价是：人一直去审已经审过的条目，而真正的毛病在作用域。

### 12.4 为什么不自动带上 `stage:` / `shot:`

`stage:` / `shot:` 才是真正「定向」的那一档（一条关于 `SHOT-012` 夜景过曝的经验，
只该出现在做 SHOT-012 的跑次里）。但 agent 跑次手上**没有**「我属于哪个引擎档 / 哪一镜」的真值，
硬拼一个只会让投喂**看起来在定向、其实在乱猜** —— 正是 §11 里刚吃过的亏（猜错阶段名的表现是
「查了没结果」，与「确实没经验」分不开）。所以这两档留给 Agent 主动调 `context_knowledge`（§11），
等到跑次能如实报出 stage/shot 的那天再接进来。

### 12.5 定向条目要先占预算（同日补，同一次检查里发现的）

12.2 只解决了**准入**（作用域匹配才进池子），但池子的排序当时仍是 `pool.sort(key=_weight)`，
而 `_weight = confidence × (1+lift)`。于是出现一个半开的口子：一条 `global` 的高置信通用成法，
完全可以把**本集**的具体经验挤出 `limit=8` 之外 —— 准入打开了，位子却被别人占着，
定向的价值只兑现了一半。

落法是**两段式排序**（`_scope_rank`，只用于排次序、绝不用于准入）：

| 档 | 作用域 | 说明 |
|---|---|---|
| 0 | `global` | 放之四海 |
| 1 | 未知写法 / `agent:<type>` | 不认识的不当定向，也不压到底 |
| 2 | `DRAMA<id>` | 本剧 |
| 3 | `EP<id>` | 本集 |
| 4 | `stage:<name>` | 本阶段/本引擎档 |
| 5 | `shot:<id>` | 本镜 |

先按档降序、档内再按 `_weight` 降序。回执新增 `targeted`（这批里有几条属于本剧/本集/本阶段）——
**定向到底有没有生效，看这个数，不要靠读文本猜**。

代价与取向：一条 `EP12` 低置信条目会排在 `global` 高置信条目前面。这是有意的
（相关性先于强度，mem0/cognee 的 recall 同一取向）；若真发现某条定向条目在拉低质量，
处理办法是**人工降它的 confidence 或 deprecate**，而不是让排序去猜"这条到底算不算好"。

预览页跟着把两个读数写出来（两端同源）：卡头是 `其中 N 条定向`，正文第一行是
`作用域：…`，`blocked>0` 时同一行还会写「另有 N 条知识存在、但作用域不符没进这一段」——
否则用户看见空段只会以为"没积累"，实际是**有知识、只是作用域不对**（三种空里最难自查的一种）。

### 12.6 守卫

`tests/knowledge_loop_test.py` 159 → **178**，新增 19 条：

* 键域（8 条）：传 `scopes=["EP1"]` 才带出 EP1 条目・不传时同一条仍被排除且 `blocked` 如实多算・
  `scopes_for` 字面形与写入口一致・「有知识但作用域不匹配」时 `text` 空而 `blocked>0` 且 hint 点名作用域・
  `append_knowledge_feed` 的 `scopes` 真透传进编译・组装点 AST 断言其 `scopes=` 来自
  `knowledge_feed.scopes_for(episode_id=…, drama_id=…)`・`/feed/preview` 带 `episodeId` 与生产同口径・
  不带 `episodeId` 时 `blocked>0`。
* 定向优先（11 条）：9 条造数逐条核 `ok is True`（**upsert 拒收是静默的**，不核就会被"数据没写进去"
  骗到）・9 条 `global` 0.9 对 2 条 `EP1` 0.1/0.05、`limit=8` 时定向两条必须进预算且 `targeted==2`・
  档内仍按 weight 排（0.1 在 0.05 前，不是按插入顺序）・不给 `scopes` 时两条都在预算外且
  `targeted==0`、`blocked==2`。

造数用 `kind="work"`（不在 `DEFAULT_KINDS` 里）以免污染其它判据的默认投喂 —— 第一版用了 `lesson`，
当场把「注入: 打开后追加分节」那条判据挤红，是真实的连锁污染。

反向验证过三条：① 把 `allowed.add(text)` 改成 `pass` ⇒ 4 条红（含预览）；② 把组装点的 `scopes=`
传参删掉 ⇒ AST 判据当场红并打印残留关键字；③ 把两段式排序退回 `key=_weight`
⇒ 定向判据当场红，回执里 8 条全是 `[作品] 通用成法*`、`targeted=0`。复原后 178/178。

⚠️ 顺手取到的一条纪律：`upsert(require_evidence=False)` 只免除**空证据**的拒绝，
**带了证据仍无条件过 `check_evidence`**（`if items:` 那一段不看这个开关）。
所以测试造数别用假 `run_id` 再关核验 —— 会静默拒收；要挂真实存在的 run。
