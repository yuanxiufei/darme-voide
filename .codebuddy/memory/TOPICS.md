# TOPICS — 专题细节（MEMORY.md 的溢出层）

> MEMORY.md 受注入长度限制，故把**低频但体量大的专题**拆到这里。
> 需要时显式读取本文件；`MEMORY.md` 中每节只留一行指针。

## 本地模型下载 + H3 视频推理
- **下载闭环**：`routes/localModels.ts` + `aiConfigs.ts`/`settings.vue`/`scripts/model_manager.py` + `configs/models.json`；三源 HF / hf-mirror / ModelScope + 断点续传（`.part` + Range）。**本机直连 HF 全超时 → 必须 `hf-mirror.com`**；大文件复用此机制，勿另写。
- **H3**：ComfyUI(**8188**) 驻留 DiT/VAE/text_encoder；**8765** 薄封装（`POST /v1/video_generation`、`GET .../task/:id`）；卸载 GPU 走 ComfyUI `POST /free`。权重扫描 → `runtime='h3'`、`baseUrl='http://localhost:8765'`。FL2VA / Ref2VA 双路线；六键 Bible `CHAR_ID/SPEAKER_ID/VOICE_ID/LOCATION_ID/COSTUME_ID/STYLE_ID` 跨集锁定。方法论 `docs/local-h3-video-system.md`。
- ⚠️ 此链路 provider 名 `minimax` 是 **AI 服务商标识**，与 `skills/` 下外部技能库**无关**。
- **GPU**：RTX A5000 22 GiB；CUDA UMD 13.3；PyTorch 2.13.0+cu130。**无 nvcc** → 编不了 CUDA 版 sd.cpp；H3 Turbo 唯一可用路线 = Turbo LoRA/int8 + ComfyUI（1–3min）。

## 视频提示词语料
- **合规红线**：他人提示词正文有版权，**不得搬运进仓库**；参考图用自有三视图；凭证绝不入库。PromptMart 免费区（500 中 23 免费）→ **只用免费区，不付费**。
- **落盘约定**：第三方语料放 `data/prompt-corpus/<源>/`（gitignored）；脚本 `scripts/corpus/`；结论 `docs/`。三者分离，勿混。
- **Seedance2 语料**：8755 条 → `data/prompt-corpus/seedance2/metadata.jsonl`；脚本 `scripts/corpus/analyze{,2,3}.py`（`SEEDANCE2_CORPUS` 可覆盖）。**提示词在 `i18n.zh.p`**（英文是平台扩写版）；`spec.duration` 有脏值 → 过滤 `0 < d <= 300`。报告 `docs/seedance2-corpus-analysis.md`，源清单 `docs/video-prompt-data-sources.md`。
- **两类写法并存**：PromptMart（短剧/漫剧）**时间码分段** → 范式 6.1–6.9；Seedance2（平台通用短视频）**散文式多段**（时间码仅 14%）→ 6.10。
- **三条实测修正**：① 「一镜到底」仅 4%，高频是 旋转/手持/环绕/推进/定格/跟随；② 一致性写法 = **部位清单式锚定**（面部比例/眼型/下颌线/发型轮廓/皮肤质感）；③ 有**反 AI 感词族**（不完美自然构图 / 自动对焦不完美 / 环境瑕疵 / 轻微手持不稳），比堆 `8k, ultra detailed` 更压塑料感。
- **语料源筛选**：核仓库看 `size`(KB) 与真实文件树，**不要看 star**；**许可证是硬约束**（TIP-I2V 禁商用仅内部统计、Seedance2 CC BY 4.0 需署名、Semonxue 无 LICENSE 不得搬运）。

## Skill 体系细节（自 MEMORY.md §Skill 下移，红线仍留在 MEMORY.md）
- **注入可见性 UI**：`/skills/meta` 给 `charBudget`+各 Agent `charCount`；`/skills` 每项给 `charCount`/`referenceCount`/`protected`/`category`/`source`/`sourceLabel`（库展示名；vendor 分属 2 库须可辨）。消费：`skills.vue`（已用/预算、超预算预警、受保护、侧栏分组、选中值 `'all'|agentType|'lib:<库名>'`）与 `agents.vue` 绑定面板（单个体量+已启用合计/预算+超预算预警+「外部库」/「已失效」）。
- **宿主工具兼容性（完整判据）**：外部库按**另一套宿主平台**编写 → 依赖的工具（`hub_*`/`question`/`task`/`read`）本项目**从未注册**。**依赖 = frontmatter `allowed-tools`（声明）∪ 正文 `hub_*` 反引号引用（`ParsedSkill.foreignToolRefs`）**，二者缺一即漏报（实测 9→15/36，漏的 6 个只在正文）。`allowed-tools` 归一须兼容 `[a,b]`/`a,b`/换行三种写法；正文提取**只认 `hub_` 前缀**，不可做「未知 snake_case 即工具」的宽泛猜测（`shot_type`/`bgm_url` 等字段名会污染）。路由比对直出 `missingTools` → UI 标「缺 N 个工具」（`agents.vue` 绑定+候选、`skills.vue`）。**工具集必须取 `tool.id`，不能用工厂返回记录键**（工厂给 camelCase `readScriptForExtraction`，发给模型的是 `read_script_for_extraction`），取错会把 21 个工具全误判为缺失。MCP 工具运行时异步发现 ⇒ 只作预警、不硬拦截。
- **`meta.yaml` 为何是「死数据」**：`production-tools` 20 个各带 1 个（`genre-templates` 9 个全无），**全仓库零引用、零读取**（`routes/skills.ts` 只读顶层 `library.yaml`，加载器只读 `SKILL.md`，无 `*.yaml` 泛化 glob）⇒ 它不像 `library.yaml` 那样是配置。但它是 `version`/`display-name-zh`/`author-*`/`source` 的**唯一载体**（SKILL.md frontmatter 无这些字段）⇒ 删 = 丢上游溯源。已记入 `skills/README.md` §一。
- **改名后核对 DB 绑定的步骤**：用 `better-sqlite3` **`{ readonly: true }`** 直开 `data/drama.db`（绕开 `agents/index.ts` 的清洗副作用，坑⑦）。**2026-09-12 于 36 文件改名后复测**：`agent_configs` 5 行 `skills` **全为 `NULL`**、旧 `minimax/*` 命中 0、绑定 id 不在磁盘 0 ⇒ **改名对 DB 零影响**。DB 为空 ⇒ **默认绑定（frontmatter `agents:`）是唯一生效来源**，故改名风险集中在路径引用（由守卫覆盖）而非 DB。
- **守卫的采集范围与启用状态**：`check-skill-refs.mjs` 的 `collectTokens` **只采集反引号 code span 与 markdown 链接目标** ⇒ **裸路径文本不在检测范围**（要受保护必须写成 `` `references/x.md` ``）；这不是 bug 而是有意取舍（裸路径多为叙述）。识别为候选后还需首段属 `references`/`scripts`/`agents` 且文档在某个含 SKILL.md 的 skill 内，否则计入「基准不明」跳过。钩子 `.githooks/pre-commit` 三分支已实测（触发校验 / 空暂存早退 / 真断链红灯），但 **`core.hooksPath` 默认未设置 ⇒ 守卫当前不生效**，需手动 `git config core.hooksPath .githooks` 启用。
- **无执行入口的 skill 不要默认注入**：`style-reference-reverse`（参考图反推画风）于 2026-09-12 由 `agents: [grid_prompt_generator]` 改为 **`agents: []`**。判据是「**三查皆空**」：后端 0 处引用其 protocol 字段（`suggested_style_key`/`reusable_anchors`）、前端 0 处「反推」文案、无专属 Agent 承接。它此前占 `grid_prompt_generator` 注入的 3,068 字符。**文件保留**（UI 仍可手动绑），将来接入该能力时改回 `agents:` 即可。
- **注入体量的两次治理（2026-09-12）**：① 第 6 节视频范式从 `prompt-style-library` 整体拆为 `video-prompt-library`（仅绑 `storyboard_breaker`）；② 停掉 `style-reference-reverse` 的默认注入。累计 **37,263 → 27,842 字符（−25.3%）**，其中 `grid_prompt_generator` **18,898 → 8,485（−55%）**。**复现命令**（无需起后端、不碰 DB、无 `agents/index.ts` 清洗副作用）：`loadAgentSkills(agent, null)` 遍历 5 个 Agent 累加 `.length` —— 脚本放 `tmp/`（gitignored），跑 `npx tsx tmp/xxx.ts`。
- **词库为什么必须按介质分家（拆因）**：`prompt-style-library` 原有 14,241 字符、第 6 节「中文视频提示词范式」独占该文件 **55%（270/486 行）**，而它的消费方 `grid_prompt_generator` 是**纯出图** Agent（其 SKILL 明写"只负责宫格"）⇒ 每次出图白吃整套视频范式。拆出 `video-prompt-library`（`agents: [storyboard_breaker]`、`priority: 30`）后：grid 18,898 → 11,818、psl 27,099 B → 11,176 B。**拆分手法**是整章机械搬家（按行范围切分，保留 CRLF），原库位置留占位说明 ⇒ **§7/§8 编号不变、零交叉引用破坏**。
- **本工具怪癖**：`search_content` 的 **`glob` 不生效**（`**/SKILL.md` 恒 0 命中）→ 改用 `path` 收窄。
- **守卫「示意引用」判据**：标记（`e.g.`/`such as`/`例如`）**必须紧邻**路径之前（标记后只允许非字母数字非汉字字符，含 token 前的开启反引号）；放宽成「同行出现过 e.g.」会**连真断链一起吞掉**（已用负向测试证实）。基线约定见 `scripts/check-skill-refs.mjs` 脚本头。

## 画风体系细节（自 MEMORY.md §画风体系 下移）
- **视频与静帧必须分开**：静帧收口词 `cinematic illustration style` 会把动漫/水墨拉回写实 ⇒ 视频用中性 `VIDEO_STYLE_TAIL` + `VIDEO_MOTION_BASE`，且**命中画风时不追加** `VISUAL_STYLE_MASTER`。
- **反 AI 感白名单**：`IMPERFECTION_ANCHORS` **仅按 `PHOTOREAL_ART_STYLES`（写实系 4 种）注入** —— 注入到动漫/水墨会破坏风格。
- **画风词一律后端收口**：`auto-pipeline` 直接拼 `buildVideoArtStyleSuffix(artStyle)`，**现算不落库**。⚠️ **若今后放开 agent 自写画风英文词，这里的幂等校验必须同步改**（否则同一剧内画风漂移）。

## 前端验证与测试工具（自 MEMORY.md §前端约定 下移）
- **工作台元素计数**（点击测试定位用）：`nav button` = 12 主步骤；`aside button` = 18（12 + 5 `sidebar-jump-dot` + 1 `.refresh-btn`）。**工作台改版后须重新核对**。
- **验证 SFC 编译**：`node -e "require('@vue/compiler-sfc')"` 跑 `compileScript` + `compileTemplate`，**无需启 dev server**；纯 TS（如 `useApi.ts`）用 `ts.transpileModule`。两者都能在改完立刻抓语法/模板错误。

## `backend-py/`（Python 后端 / 绞杀者迁移，2026-09-12 起）
- **为什么不是一次性重写**：原始体量 **133 TS / 27,389 行 / 226 端点 / 29 表**且**零自动化测试** ⇒ 重写期没有任何可验证中间态。故 FastAPI 作新入口（**5790**），**未迁移的域反代到 Node（5789）**，每迁一域就 `include_router` 一行并把该域从 Node 停用；全迁完后 `PY_PORT=5789` 切单端口。
- **端口铁律**：`app/config.py` **刻意不读 `config.yaml` 的 `server.port`**（那是 Node 的端口，并存期读同一个必然抢占）。优先级 `PY_PORT > PORT > 5790`。
- **三条契约对齐约定**：① 信封 `{code,data,message}`、**错误响应无 `data` 键**（前端 `useApi.ts` 用原生 fetch，判据 `!resp.ok || json.code >= 400`，文案取 `json.message`）⇒ FastAPI 默认的 `{"detail"}` 会让文案变「请求失败 (422)」，故三类异常全部收口；特例 `/api/v1/health` 是**裸对象**。② `GET /dramas/:id/prompts` 聚合视图**本来就是 camelCase**（`customPrompt`/`imageUrl`），照抄勿「顺手统一」。③ **SQLAlchemy `**kwargs` ≠ JS 对象展开**：`values(**fields, title=...)` 同名键直接 `TypeError: got multiple values for keyword argument`，必须 `values = {**fields, ...}` 再覆盖 —— **已在 `PUT /dramas/:id/episodes` 真实踩到**。
- **为什么用 SQLAlchemy Core 而不是 ORM**：Core 的行**本身就是 snake_case = HTTP 契约形状** ⇒ 原 Node 的 `toSnakeCase()` 转换层在 Python 侧**不存在**，少一层字段漂移源，也不必维护「Python 属性名 ↔ DB 列名」映射。
- **模式核对用真实库、别只对 `schema.ts`**：`backend-py/tests/smoke_test.py` 拿 `data/drama.db` 的 `PRAGMA table_info` 逐列比对，实测**真实库比 Node 模型多 2 表（`assets` / `props`）与 2 列（`image_generations.minio_url` / `video_generations.minio_url`）** —— 全是旧版本 MinIO 遗留（Node `db/index.ts` 只建 **29** 表、全仓库零引用）⇒ 两边都不建模。**列序差异不影响正确性**（SQLAlchemy 全程按列名生成 SQL，不做位置化 INSERT/SELECT）。
- **回归唯一入口**：`backend-py/tests/smoke_test.py`（**64 用例**：模式 + 信封 + **错误文案逐字** + 软删 + 反代 501）。写操作**全部落在 `data/drama.db` 的副本**上（`DATA_ROOT` 指向临时目录），**真实库只读**。改 Python 侧代码后必跑。
- **已迁移域**：`dramas`（9 端点，含 `ensureStyleId`/`ensureCostumeId`）+ `episodes`（8 端点，含十步 `pipeline-status` 与 `script-fingerprint` 门禁）；依赖 LLM / 视觉模型 / 节奏服务的端点**刻意不注册**（`dramas/:id/rhythm`、`dramas/:id/era-background/extract`、`episodes/:id/continue-script`、`episodes/:id/consistency-qc`）走反代 —— 比返回 501 更可用。
- **分支约定**：Node/TS 侧修复提交到 **`main`**；Python 迁移工作在 **`feat/python-backend`**。`backend/probe-*.ts` 是临时探针（头部自述「用完即删」）⇒ **不入提交**。
- **JS 语义坑（第二次踩，务必对照）**：① `??`（空值合并）**不等于** `||` —— `ep.scriptContent ?? ep.content` 在 `script_content=''` 时**不回退**，写成 Python `a or b` 会让门禁/取值与 Node 分叉；同一份代码里两种都有（`era-background` 用 `||`、`script-fingerprint` 用 `??`），**必须逐个确认**。② `GET /episodes/:id/pipeline-status` 的「有图」判据是 `composed_image` **单字段**，而 `GET /dramas` 的 `progress` 是 `composed_image || first_frame_image` —— 原 TS 就是两套，**照抄勿合并**（已各写各的）。③ 数值字段的 `Number(x)` 语义：`Number(null)=0`、`Number('')=0`、`Number('abc')=NaN` ⇒ Python 侧实现 `_js_number`，仅 `NaN` 分支返回 `None`（写 NULL 而非 NaN，属有意偏差且该列本可空）。
- **跨域复用**：读请求体统一走 `app/request_utils.py` 的 `read_json`（对齐 Hono `c.req.json()` 的容错：坏 JSON → 空 dict → 业务校验给 400，而非 500）。
- **下一个域的 SOP**：见 `backend-py/README.md`。要点：静态子路径必须声明在 `/{id}` 之前；**只注册已实现的**；依赖 LLM/子进程/长任务的服务**放到最后**（其行为只能真调验证）。
- **工具坑**：`tempfile.mkdtemp()` 在 Windows 返回 **8.3 短路径**，与 `resolve()` 后的长路径做子串比较会**假红** ⇒ 比较 resolve 后路径；`sqlalchemy` 顶层导出的是 **`REAL` 而非 `Real`**；PowerShell 组合命令含 `&` / `@{...}` 会被路由到 cmd（报「不是内部或外部命令」）⇒ **写 `.ps1` 落盘再执行**；`Start-Process` 的相对 `-WorkingDirectory` + 相对 exe 路径会报「系统找不到指定的路径」⇒ 用字面绝对路径。
