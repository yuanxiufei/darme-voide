# TOPICS — 专题细节（MEMORY.md 的溢出层）

> MEMORY.md 受注入长度限制，故把**低频但体量大的专题**拆到这里。
> 需要时显式读取本文件；`MEMORY.md` 中每节只留一行指针。

## 本地模型下载 + H3 视频推理
- **下载闭环**：`routes/localModels.ts` + `aiConfigs.ts`/`settings.vue`/`backend-py/app/scripts/model_manager.py` + `configs/models.json`；三源 HF / hf-mirror / ModelScope + 断点续传（`.part` + Range）。**本机直连 HF 全超时 → 必须 `hf-mirror.com`**；大文件复用此机制，勿另写。
- **⭐ 模型目录自主化（2026-09-25 ✓）**：``models_dir`` 默认从 ComfyUI 目录改为 **``<data_root>/models``** ✓（``local_model_scan.default_models_dir()`` ✓；``configs/model-paths.json`` 的 ``models_dir``/``comfyui_root`` 已清空走默认 ✓）。⚠️ 但「检测/扫描」**不排斥第三方** ✗：``get_default_roots`` 仍覆盖电脑内模型（本仓目录 + 本地服务 + ComfyUI 目录探测 + extra_roots ✓）—— 自主的是「从哪下载、存到哪」，**不是**「不许看见别人已装的模型」✓（先误删了 ComfyUI 扫描、用户一句话纠正 ✓）。
- **H3**：ComfyUI(**8188**) 驻留 DiT/VAE/text_encoder；**8765** 薄封装（`POST /v1/video_generation`、`GET .../task/:id`）；卸载 GPU 走 ComfyUI `POST /free`。权重扫描 → `runtime='h3'`、`baseUrl='http://localhost:8765'`。FL2VA / Ref2VA 双路线；六键 Bible `CHAR_ID/SPEAKER_ID/VOICE_ID/LOCATION_ID/COSTUME_ID/STYLE_ID` 跨集锁定。方法论 `docs/local-h3-video-system.md`。
- ⚠️ 此链路 provider 名 `minimax` 是 **AI 服务商标识**，与 `backend-py/app/skills/` 下外部技能库**无关**。
- **GPU**：RTX A5000 22 GiB；CUDA UMD 13.3；PyTorch 2.13.0+cu130。**无 nvcc** → 编不了 CUDA 版 sd.cpp；H3 Turbo 唯一可用路线 = Turbo LoRA/int8 + ComfyUI（1–3min）。

## 视频提示词语料
- **合规红线**：他人提示词正文有版权，**不得搬运进仓库**；参考图用自有三视图；凭证绝不入库。PromptMart 免费区（500 中 23 免费）→ **只用免费区，不付费**。
- **落盘约定**：第三方语料放 `data/prompt-corpus/<源>/`（gitignored）；脚本 `backend-py/app/scripts/corpus/`；结论 `docs/`。三者分离，勿混。
- **Seedance2 语料**：8755 条 → `data/prompt-corpus/seedance2/metadata.jsonl`；脚本 `backend-py/app/scripts/corpus/analyze{,2,3}.py`（`SEEDANCE2_CORPUS` 可覆盖）。**提示词在 `i18n.zh.p`**（英文是平台扩写版）；`spec.duration` 有脏值 → 过滤 `0 < d <= 300`。报告 `docs/seedance2-corpus-analysis.md`，源清单 `docs/video-prompt-data-sources.md`。
- **两类写法并存**：PromptMart（短剧/漫剧）**时间码分段** → 范式 6.1–6.9；Seedance2（平台通用短视频）**散文式多段**（时间码仅 14%）→ 6.10。
- **三条实测修正**：① 「一镜到底」仅 4%，高频是 旋转/手持/环绕/推进/定格/跟随；② 一致性写法 = **部位清单式锚定**（面部比例/眼型/下颌线/发型轮廓/皮肤质感）；③ 有**反 AI 感词族**（不完美自然构图 / 自动对焦不完美 / 环境瑕疵 / 轻微手持不稳），比堆 `8k, ultra detailed` 更压塑料感。
- **语料源筛选**：核仓库看 `size`(KB) 与真实文件树，**不要看 star**；**许可证是硬约束**（TIP-I2V 禁商用仅内部统计、Seedance2 CC BY 4.0 需署名、Semonxue 无 LICENSE 不得搬运）。
- **检索管线已就绪（自 MEMORY.md 下移 2026-09-15）**：`fetch-raw` → `normalize` → `search`（**8987 条 / 3 源**）⇒ **别另起一套**；语料 gitignored ⇒ 消费方**须在缺失时优雅降级**（不可硬依赖语料存在）。
- **已排除源勿引回（判据见 `backend-py/app/scripts/README.md`）**：TIP-I2V｜Semonxue｜`HitPaw` / `geekjourneyx` / `fantasylights`。

## Skill 体系细节（自 MEMORY.md §Skill 下移，红线仍留在 MEMORY.md）
- **注入可见性 UI**：`/skills/meta` 给 `charBudget`+各 Agent `charCount`；`/skills` 每项给 `charCount`/`referenceCount`/`protected`/`category`/`source`/`sourceLabel`（库展示名；vendor 分属 2 库须可辨）。消费：`skills.vue`（已用/预算、超预算预警、受保护、侧栏分组、选中值 `'all'|agentType|'lib:<库名>'`）与 `agents.vue` 绑定面板（单个体量+已启用合计/预算+超预算预警+「外部库」/「已失效」）。
- **宿主工具兼容性（完整判据）**：外部库按**另一套宿主平台**编写 → 依赖的工具（`hub_*`/`question`/`task`/`read`）本项目**从未注册**。**依赖 = frontmatter `allowed-tools`（声明）∪ 正文 `hub_*` 反引号引用（`ParsedSkill.foreignToolRefs`）**，二者缺一即漏报（实测 9→15/36，漏的 6 个只在正文）。`allowed-tools` 归一须兼容 `[a,b]`/`a,b`/换行三种写法；正文提取**只认 `hub_` 前缀**，不可做「未知 snake_case 即工具」的宽泛猜测（`shot_type`/`bgm_url` 等字段名会污染）。路由比对直出 `missingTools` → UI 标「缺 N 个工具」（`agents.vue` 绑定+候选、`skills.vue`）。**工具集必须取 `tool.id`，不能用工厂返回记录键**（工厂给 camelCase `readScriptForExtraction`，发给模型的是 `read_script_for_extraction`），取错会把 21 个工具全误判为缺失。MCP 工具运行时异步发现 ⇒ 只作预警、不硬拦截。
- **`meta.yaml` 为何是「死数据」**：`production-tools` 20 个各带 1 个（`genre-templates` 9 个全无），**全仓库零引用、零读取**（`routes/skills.ts` 只读顶层 `library.yaml`，加载器只读 `SKILL.md`，无 `*.yaml` 泛化 glob）⇒ 它不像 `library.yaml` 那样是配置。但它是 `version`/`display-name-zh`/`author-*`/`source` 的**唯一载体**（SKILL.md frontmatter 无这些字段）⇒ 删 = 丢上游溯源。已记入 `backend-py/app/skills/README.md` §一。
- **改名后核对 DB 绑定的步骤**：用 `better-sqlite3` **`{ readonly: true }`** 直开 `data/drama.db`（绕开 `agents/index.ts` 的清洗副作用，坑⑦）。**2026-09-12 于 36 文件改名后复测**：`agent_configs` 5 行 `skills` **全为 `NULL`**、旧 `minimax/*` 命中 0、绑定 id 不在磁盘 0 ⇒ **改名对 DB 零影响**。DB 为空 ⇒ **默认绑定（frontmatter `agents:`）是唯一生效来源**，故改名风险集中在路径引用（由守卫覆盖）而非 DB。
- **守卫的采集范围与启用状态**：`check_skill_refs.py` 的 `collectTokens` **只采集反引号 code span 与 markdown 链接目标** ⇒ **裸路径文本不在检测范围**（要受保护必须写成 `` `references/x.md` ``）；这不是 bug 而是有意取舍（裸路径多为叙述）。识别为候选后还需首段属 `references`/`scripts`/`agents` 且文档在某个含 SKILL.md 的 skill 内，否则计入「基准不明」跳过。钩子 `.githooks/pre-commit` 三分支已实测（触发校验 / 空暂存早退 / 真断链红灯），但 **`core.hooksPath` 默认未设置 ⇒ 守卫当前不生效**，需手动 `git config core.hooksPath .githooks` 启用。
- **无执行入口的 skill 不要默认注入**：`style-reference-reverse`（参考图反推画风）于 2026-09-12 由 `agents: [grid_prompt_generator]` 改为 **`agents: []`**。判据是「**三查皆空**」：后端 0 处引用其 protocol 字段（`suggested_style_key`/`reusable_anchors`）、前端 0 处「反推」文案、无专属 Agent 承接。它此前占 `grid_prompt_generator` 注入的 3,068 字符。**文件保留**（UI 仍可手动绑），将来接入该能力时改回 `agents:` 即可。
- **注入体量的两次治理（2026-09-12）**：① 第 6 节视频范式从 `prompt-style-library` 整体拆为 `video-prompt-library`（仅绑 `storyboard_breaker`）；② 停掉 `style-reference-reverse` 的默认注入。累计 **37,263 → 27,842 字符（−25.3%）**，其中 `grid_prompt_generator` **18,898 → 8,485（−55%）**。**复现命令**（无需起后端、不碰 DB、无 `agents/index.ts` 清洗副作用）：`loadAgentSkills(agent, null)` 遍历 5 个 Agent 累加 `.length` —— 脚本放 `tmp/`（gitignored），跑 `npx tsx tmp/xxx.ts`。
- **词库为什么必须按介质分家（拆因）**：`prompt-style-library` 原有 14,241 字符、第 6 节「中文视频提示词范式」独占该文件 **55%（270/486 行）**，而它的消费方 `grid_prompt_generator` 是**纯出图** Agent（其 SKILL 明写"只负责宫格"）⇒ 每次出图白吃整套视频范式。拆出 `video-prompt-library`（`agents: [storyboard_breaker]`、`priority: 30`）后：grid 18,898 → 11,818、psl 27,099 B → 11,176 B。**拆分手法**是整章机械搬家（按行范围切分，保留 CRLF），原库位置留占位说明 ⇒ **§7/§8 编号不变、零交叉引用破坏**。
- **本工具怪癖**：`search_content` 的 **`glob` 不生效**（`**/SKILL.md` 恒 0 命中）→ 改用 `path` 收窄。
- **守卫「示意引用」判据**：标记（`e.g.`/`such as`/`例如`）**必须紧邻**路径之前（标记后只允许非字母数字非汉字字符，含 token 前的开启反引号）；放宽成「同行出现过 e.g.」会**连真断链一起吞掉**（已用负向测试证实）。基线约定见 `backend-py/app/scripts/check_skill_refs.py` 脚本头。

## 画风体系细节（自 MEMORY.md §画风体系 下移）
- **视频与静帧必须分开**：静帧收口词 `cinematic illustration style` 会把动漫/水墨拉回写实 ⇒ 视频用中性 `VIDEO_STYLE_TAIL` + `VIDEO_MOTION_BASE`，且**命中画风时不追加** `VISUAL_STYLE_MASTER`。
- **反 AI 感白名单**：`IMPERFECTION_ANCHORS` **仅按 `PHOTOREAL_ART_STYLES`（写实系 4 种）注入** —— 注入到动漫/水墨会破坏风格。
- **画风词一律后端收口**：`auto-pipeline` 直接拼 `buildVideoArtStyleSuffix(artStyle)`，**现算不落库**。⚠️ **若今后放开 agent 自写画风英文词，这里的幂等校验必须同步改**（否则同一剧内画风漂移）。

## 前端验证与测试工具（自 MEMORY.md §前端约定 下移）
- **工作台元素计数**（点击测试定位用）：`nav button` = 12 主步骤；`aside button` = 18（12 + 5 `sidebar-jump-dot` + 1 `.refresh-btn`）。**工作台改版后须重新核对**。
- **验证 SFC 编译**：`node -e "require('@vue/compiler-sfc')"` 跑 `compileScript` + `compileTemplate`，**无需启 dev server**；纯 TS（如 `useApi.ts`）用 `ts.transpileModule`。两者都能在改完立刻抓语法/模板错误。

## 自 MEMORY.md 下移（2026-09-20 第三次腾 8k 预算）

**本机环境 / 跑批细节**（原文自 MEMORY.md §本机环境 下移 ✓；第三条是 2026-09-20 新踩的 ✓）：
- **PyPI 在本机下不动** ✗（两次卡在同一文件、无报错 ✓）⇒ 一律 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
  （20–58 MB/s ✓）；装 torch：`--index-url https://download.pytorch.org/whl/cpu --extra-index-url <镜像> torch==<ver>+cpu` ✓。
- 长命令常被判「在后台运行」✗ ⇒ **先查产物再决定重跑** ✓（它可能真跑完了 ✓）；批次日志**标签不可复用** ✓
  （复用会读到上次会话的陈旧结果 ✓）。
- ⚠️⚠️ **跑批工具坑**（2026-09-20 实测 ✓，与代码无关但会**骗人** ✓✗）：PowerShell `*>>` 写的是 **UTF-16** ✗，
  再叠一个 `Add-Content -Encoding UTF8` ⇒ **混编码文件** ⇒ `Select-String` / `grep` **0 命中** ✓✗
  （"看起来一个 FAIL 都没有" ✓ 实际是读不出来 ✓）；子进程 stdout 默认 **GBK** ✗ ⇒ 套件里的 `✓` 直接
  `UnicodeEncodeError` ✓✗（**看起来"失败"、其实只是打印炸** ✓）⇒ 跑批时给子进程
  **`PYTHONIOENCODING=utf-8`** ✓，或干脆在**脚本里**做 UTF-8 落盘 ✓（别经 PowerShell 重定向 ✗）。

## 自 MEMORY.md 下移（2026-09-18，第二次腾 8k 预算）

**`backend-py/` 迁移明细**（原文自 MEMORY.md §项目与运行 下移；红线仍在 MEMORY.md）：
- **全部域已迁完**（未注册 0 条，Node 侧 224 条路径已 100% 覆盖）；**端口 5790**（刻意不读 `config.yaml` 的 `server.port`）；
- 回归跑 `backend-py/tests/run_all.py`；动手前读 `backend-py/README.md`；
- **`backend/` 已于 2026-09-15 删除**：TS 原文现只存于 `backend-py/tests/frozen_ts_source.py`（76 条 / 674 KB），**守卫与自检读 TS 一律走「真源码优先 → 快照」**；`PROXY_TO_NODE` 已无对象（接缝保留只为兜底 501）。

**Skill 体系明细**（原文自 MEMORY.md §Skill 体系 下移；**绑定/命名/删除保护等红线仍在 MEMORY.md**）：
- **注入闸**：`SKILL_CHAR_BUDGET`（默认 6 万，`AGENT_SKILL_BUDGET` 覆盖，0=关）→ 超预算按 priority 跳过 + 末尾「未注入：…」诊断。**口径 = `renderSkill(parseSkill(...)).length`**（≠ 字节数）。外部库单体最大 ~2.7 万字符 ⇒ 只勾两个即逼近上限。
- **DB 配置优先铁律**：`parseSkillsConfig(raw)` 解析出配置项即「用户已做过选择」→ **全关也不回退默认**（仅 `null`/空串/空数组/解析失败才回退）；`enabled` 缺省 = **启用**（`!== false`），`priority` 缺省 `0`。默认绑定**仅当** DB `agent_configs.skills` 为空时生效。
- **宿主工具兼容性**：外部库依赖的 `hub_*` 等工具本项目**从未注册**；依赖 = `allowed-tools` ∪ 正文 `hub_*` 引用（**只认 `hub_` 前缀**，宽泛猜会被字段名污染）；**工具集须取 `tool.id`**（取错把 21 个工具全误判为缺失）。
- **改名/挪库后必核对 DB 绑定**：`agent_configs.skills` 存 skill id ⇒ 旧绑定失效；核对用 `better-sqlite3` **`{ readonly: true }`** 直开（绕开清洗副作用，坑⑦）。**实测 5 行全 `NULL` ⇒ 改名零影响**。
- **兜底库**：顶层目录无 `library.yaml` → `/meta.sources` 补合成条目（`declared:false`，label = 目录名）→ 保证「core + Σ各库 = 总数」自洽且侧栏可达。

**新引擎与"事前省钱链"（2026-09-18 建，细节见当日日志）**：`backend-py/app/services/engine/`（零依赖算法层：schedules/geometry/sampler/guidance/conditioning/dit/vae/text/media/segments/mappings/safetensors/inventory/loader/pipeline/dryrun ✓）+ `app/agent/context_budget.py` ✓ + 生产链四模块（`shot_placeholders`/`prompt_polish`/`asset_manifest`/`continuity` ✓ + 粘合 `production_preflight` ✓ + 取数 `preflight_source` ✓ + 适配 `storyboard_continuity` ✓）+ 前端 `components/PreflightPanel.vue` ✓。**判据只有一份** ✓：粘合层**只调用**、不重写判据 ✓（自检钉"与直接调用逐字段一致"✓）。

## ⭐ 可照抄项目清单（用户 2026-09-20 点名；原话「这几个项目**功能可以直接抄**，**不要忘记**」）

> ⚠️ 用户当时补了一句「不要忘记」⇒ 这是**长期挂账的待办源** ✓：能力缺口优先从这五个项目里找现成实现 ✓。

| 参考项目（`reference/` 下） | 它提供什么 | 该抄进本仓哪里 |
|---|---|---|
`ComfyUI` | **执行引擎骨架**：节点图 → 拓扑执行、缓存/复用、队列与取消、类型校验、进度事件 | `app/services/engine/`（把"图执行/校验"这套机制搬成自研执行器 ✓）；⚠️ **别整包搬** ✗（十万行级第三方应用 ✓ 只取机制 ✓） |
`minimax-h3-comfyui` | **H3 的 ComfyUI 侧实现**：节点定义、T2V/I2V 工作流模板、采样参数与条件注入 | 对照 `engine/` 的 `dit`/`conditioning`/`sampler`/`pipeline` 补齐 ✓；工作流模板 → 本仓 `local_services/h3/workflows/` ✓ |
`ollama`（Go 服务端） | **本地模型服务**：模型清单/加载/keep-alive、流式响应、Modelfile、模板与参数默认值 | LLM 侧自研服务的形态参考 ✓（把"本地起服务 + 流式 + 模型管理"做成自家的 ✓）；⚠️ 当前 LLM 走的是本机 ollama 进程 ⚠️ |
`ollama-python` | **客户端与结构化输出**：函数→工具 schema、结构化输出约束、（增量）流式解析 | 已在第 85 步判过：本项目 agent 工具集**已有**等价件 ✓、流式工具调用**暂无调用方** ⇒ 等真接流式再抄 ✓ |
`minimax-desgin-plugin` | **ComfyUI 插件形态**：节点注册、参数校验、设计稿/资产对接 | 与 `ComfyUI` 那条合并看 ✓：**节点注册 + 参数校验**这套机制 |
（早先已抄 ✓）`short-drama-agent` / `Mini-Agent` / `Open-AI-Micro-Drama-Generator` | 生产契约（占位符/质感层/资产门/连续性 ✓）、上下文预算 ✓ | 已落：`shot_placeholders` / `prompt_polish` / `asset_manifest` / `continuity` / `agent/context_budget` ✓ |

**抄的姿势（延续本仓判据 ✓）**：① 只抄**能接线**的 ✓（搬来没人调用 = 没抄 ✓ 见第 106 步教训 ✓）；
② 抄**机制**不抄**体量** ✗（尤其 `ComfyUI` ✓）；③ 抄完**写清出处** ✓（`docs/` 或模块 docstring ✓，
与「他人提示词正文不得搬运」的红线不冲突 ✓ —— 那是**语料**✗，这是**代码/机制** ✓）。

## 外部调用审计（2026-09-20 实测；用户要求「不要调用外部的」）

**真库 `data/drama.db` 的 `ai_service_configs`（只读查 ✓ 8 条 active）**：

| service_type | provider | base_url | 外部? | priority |
|---|---|---|---|---|
audio | minimax | https://api.minimax.chat | **外部 ✗** | 200 |
image | volcengine | https://ark.cn-beijing.volces.com | **外部 ✗** | 200 |
video | volcengine | https://ark.cn-beijing.volces.com | **外部 ✗** | 200 |
text | minimax | https://api.minimax.chat | **外部 ✗** | 300 |
text | openai | http://localhost:11434 | 本地 ✓（其实是 ollama ✓） | 85 |
image | local-sd | http://localhost:7860 | 本地 ✓ | 84 |
video | minimax | http://localhost:8765 | 本地 ✓（H3 门面） | 83 |
audio | cosyvoice | http://localhost:9880 | 本地 ✓ | 82 |

**取用口径（决定性）**：`services/ai_providers.py:87` = `rows.sort(key=lambda r: r.priority or 0, reverse=True)`
⇒ **priority 越大越优先** ✓（`estimate_service.py` 另有一套：`is_default` 先、再 priority 降序 ✓ 别混 ✓）。
⇒ **四类当前都会选中外部行** ✗（本地 82–85 输给外部 200/300 ✓✗）。

**为什么会长成这样**：`ai_configs.LOCAL_PRESET_SERVICES` 的本地预设 priority 定在 **85/84/83/82** ✓，
而 `PRESET_SERVICES`（厂商一键配置）是 **100/99/98/97** ✓ ⇒ **本地预设天生排在厂商之后** ✗
（"先一键配置、再切本地" ⇒ 厂商行仍在且更高 ✓✗）。

**要「不调用外部」的动作（按代价排序）**：
1. **抬高本地行 priority**（如 200+ ✓）或**把外部行 `is_active` 置 0 / 直接删** ✓；
2. 把 `LOCAL_PRESET_SERVICES` 的 priority 改到**高于** `PRESET_SERVICES` ✓（否则每次一键配置又颠倒 ✓）；
3. 长期：图片/视频/TTS 从**本地第三方运行时**（SD-WebUI 7860 ✓ / ComfyUI→8765 ✓ / CosyVoice 9880 ✓）
   换成**自研引擎** ✓（缺真权重 ✗ ⇒ 见 §自研引擎现状 ✓）。

**三层依赖阶梯（汇报时要说清 ✓）**：① **外部厂商 API** ✗✗（要去的 ← 上表 4 行）→
② **本地第三方运行时** ⚠️（ollama ✓ SD-WebUI ✓ ComfyUI ✓ CosyVoice ✓ —— 本地但非自研 ✓）→
③ **完全自研** ✓（`services/engine/` ✓ —— ⚠️ **别再往这里写模块数** ✗：唯一权威是
`app/services/engine/__init__.py` 的 `__all__` ✓）。

## 自研引擎现状（2026-09-22 改写 —— ⚠️ 只留「去哪儿看」✗，不留会腐烂的数字 ✓）

⚠️⚠️ 这一节原来罗列 **19 个模块 + 9 套 343 用例** ✗ —— 那是 2026-09-20 的快照 ✓，
之后模块与用例一直在长（`quant` ✓ / `h3_form` ✓ / `h3_keys` ✓ / 词表三件套 ✓ / `gguf` ✓ …）⇒
**罗列必腐烂** ✗（本仓规矩：规模数字**要么指向唯一权威、要么带实测日期** ✓）。现在改成指针 ✓：

* **模块清单** ⇒ `app/services/engine/__init__.py` 的 `__all__` ✓（**唯一权威** ✓）；
* **自检规模与最近一次实测** ⇒ `tests/run_all.py` 的表头 ✓（**唯一权威** ✓，每次实测后同步 ✓）；
* **还缺什么** ⇒ `torch_backend.PENDING_PARTS` ✓（**只减不骗** ✓ —— 关掉一条移进
  `dit.H3_GAPS_CLOSED` / `h3_form.H3_FORM_TODO` ✓）；**H3 形态的权威实现**是 `h3_form.H3FormTrunk` ✓
  （⚠️ `dit.py` 那 7 条结构差异**不要**去补 ✗ —— 那是「通用 DiT」的设计 ✓，补成半套 H3 反而装不上真权重 ✓✗）。

**已能自主完成的**（能力面 ✓ 与上条独立 —— 能力有没有比"有几个模块"稳 ✓）：
σ 调度 ✓ / 采样循环 + CFG 引导 ✓ / 首帧条件（图生视频 ✓ 真 VAE 编码 + 掩码混合 ✓）/
长视频分段（保留帧数守恒 ✓）/ DiT 真前向 ✓ / **自研词表三件套** ✓（BPE ✓ / Unigram ✓ / WordPiece ✓ +
11 种 normalizer + 10 种预分词器 ✓ —— 规则逐例对齐参考 ✓）/ H3 行级主干 + 双流 ✓（真 mp4 + 真 wav ✓）/
**低精度权重反量化** ✓（fp8/int8 ✓ 四种布局 ✓ 判不出来就拒绝 ✓）/ VAE 解码 ✓ / 帧→真 mp4（ffprobe 复核 ✓）/
音频→真 wav（标准库 `wave` ✓）/ 权重体检 + 加载计划 ✓（含**反量化计划** ✓）/ 管线编排（进度 / 取消 /
错误归因 ✓）/ 干跑后端 ✓。

**唯一硬缺口 = 真权重 + 真配置** ✗（见 `torch_backend.PENDING_PARTS` ✓：H3 主 DiT 19.53 GiB 未下载 ✓
⇒ 上机前先跑 `python app/scripts/h3_readiness.py` ✓）⇒ `canGenerate=False` ✓。

## 自主化进度 + 下一步待办（2026-09-25 ✓ 用户「完全自主不依赖第三方」—— **下次接手看这里** ✓）

> 目标：模型下载/存储/管理 + 四类生成（文本/图片/视频/音频）**全自研**，不依赖第三方运行时
> （ollama 11434 / SD-WebUI 7860 / CosyVoice 9880 / ComfyUI 8188+8765 ✓）。参考 `reference/ollama`、
> `reference/ollama-python`、`reference/ComfyUI` 只是为了**抄模型管理/推理机制**，不是要调它们 ✓。

**已完成 ✓（2026-09-25）：**
1. **模型目录自主化** ✓ —— `models_dir` 默认 `<data_root>/models` ✓（`local_model_scan.default_models_dir()` ✓；
   `configs/model-paths.json` 已清空 ComfyUI 硬编码 ✓）；⚠️ 检测/扫描**不排斥第三方**（`get_default_roots` 仍扫电脑内模型 ✓
   —— 自主的是「从哪下载、存到哪」，不是「不许看见别人装的」✓）。
2. **文本 LLM 自研闭环** ✓（不依赖 ollama）—— 四块拼图 + 接线，全在 `engine/` 下：
   - `llm.py` ✓（decoder-only：RMSNorm / GQA / RoPE / SwiGLU / 因果 mask / KV cache / 采样 greedy+top-p+top-k）
   - `gguf_dequant.py` ✓（F32/F16/BF16/Q8_0/Q4_K 反量化；公式照 llama.cpp ggml-quants.c ✓ MIT ✓）
   - `gguf_to_llm.py` ✓（ggml 命名映射 + 线性层转置 + tie embeddings + `infer_llm_config` 从 GGUF 元数据读架构）
   - `llm_backend.py` ✓（`describe` 如实报缺 / `load` / `generate` encode→生成→decode）
   - 接线 ✓（`text_generation.generate_text` 的 engine 分支：provider=engine ⇒ 走自研后端，`asyncio.to_thread` 包同步推理 + 后端缓存）
   全量 **129 套 / 4141 项 / 0 失败** ✓（记账见 `INDEX.md §㉓~㉘` ✓）。

**下一步待办 ✗（「完全自主」的剩余，按优先级）：**
1. **图片自研**：SDXL 那条路 ✓（替代 SD-WebUI 7860）——
   ✅ **UNet 已落地** ✓（2026-09-26）：`engine/sdxl.py` ✓（`SdxlUnetConfig` 全显式 ✓ + `build_sdxl_unet` / `load_sdxl_unet_state_dict` ✓），
   逐块布局**照本机真权重头**量出 ✓ ⇒ 与官方 `sd_xl_base_1.0` **1680 键逐键形状全等 + 参数量 2,567,463,684 全等** ✓（`tests/engine_sdxl_test.py` **39/39** ✓）；
   ⚠️ 口径：上行 ResBlock 吃 **`cat([h, skip])`** ✗✗、eps **两档**（GroupNorm 1e-5 / SpatialTransformer 1e-6 ✓）、时间嵌入 **cos 在前** ✓、
   ADM **6 个 id**（`build_adm` ✓）、`out.2` 零初始化 ✓；⚠️ 参考实现本机**导不进来**（`comfy_kitchen` 版本对不上 ✓）⇒ **没做数值对拍** ✓。
   ✅ **潜空间口径表 + 图像算子集已落地** ✓（2026-09-26）：`engine/latent_formats.py` ✓（一个族一条口径：
   scale/shift/通道/维数/下采样 ✓ + `process_in`/`process_out` **五种**语义 ✓；⚠️ `meanstd` 的 mean/std 属**权重** ⇒
   不假装有 ✓、`rearrange` 型**具名拒绝** ✗、认不出**报错** ✗✗ **不回退默认 4 通道**；⭐ H3 的 **24** 通道**引用**
   `latent_container` ✓、音频 **32** 与 `h3_form` 同源 ✓ ⇒ 测试**逐值比对** ✓ 不另发明 ✓）+
   `engine/image_ops.py` ✓（**纯 torch** ✓：六核重采样含自研 `bislerp`/`lanczos` ✓、缩放族/裁剪旋转翻转/拼接/
   Porter-Duff **18 模式**/掩罩族/形态学/BT.601 色彩/量化/Canny ✓；⚠️ 自检当场抓出两条「看着对的错值」已修：
   Canny 平坦图**伪边缘** ⇒ 加 `CANNY_NOISE_FLOOR` ✓、`quantize` 借灰阶调色板 ⇒ Pillow **忽略 `colors`** 且压成灰阶 ⇒
   改**中位切分** ✓）；规模与最近实测见 `tests/run_all.py` 表头 ✓（**唯一权威** ✓，本处**不罗列** ✗）。
   ✗ **仍缺**：**接线**（`image_generation.generate_image` 的 engine 分支：CLIP-L + OpenCLIP-bigG 双文本编码 → `build_adm` → 采样循环 → `vae` 解码落盘 ✓）
   + **真权重**（TE ≈2.5 GiB ×2 + VAE 334 MB ✓，`sd_xl_base_1.0.safetensors` 本身含 US 版 TE/V AE ✓）。
   ⚠️ **接线的前置**（2026-09-26 实测 ✓）：**扫描器看不见本机那份 SDXL / VAE / TE** ✓✗ ——
   `detect_comfyui()` **只取第一个命中** ⇒ 挑中 `ComfyUI-Installs/…/ComfyUI`、把
   **`ComfyUI-Shared/models` 遮住** ✗（默认根扫**扫不到** `…/ComfyUI-Shared/models/checkpoints/sd_xl_base_1.0.safetensors` ✓；
   补上 `COMFYUI_CANDIDATES` 后 22 ms 命中 ✓）。两条路选一 ✓：**修扫描语义**（`get_default_roots` 收**所有**存在的
   ComfyUI 模型根 ✓ —— 但该模块有 TS 镜像与多处守卫 ⇒ 得连 `model_manager.py` / `local_models_test` 一起改 ✓）
   或**显式配**（`COMFYUI_PATH` / `configs/model-paths.json` 的 `comfyui_root` / `extra_roots` ✓）—— **等定** ✗。
2. **TTS 自研**：声学模型（替代 CosyVoice 9880）—— 引擎已有 `audio_vae`（声码器那半 ✓），缺的是
   **文本→声学特征**（音素化/时长/声学模型）✓；参考 `reference/ollama` 之外的 TTS 方案（IndexTTS 逆向已收口：本仓走 CosyVoice，
   情绪 8 维已落地 `voice_contract.EMOTION_ORDER` ✓）。这块最复杂，放最后 ✓。
3. **真权重下载**：H3 主 DiT 19.53 GiB + Qwen3 GGUF ≈9 GiB + 词表 ✓ —— 文本代码已闭环但**没权重跑不动** ✗；
   下载入口 `model_manager.py download --category text/video` ✓（三源 HF/hf-mirror/ModelScope ✓）。
4. **小项收尾**：chat 模板（Qwen3 `<|im_start|>` 对话骨架，现为裸拼接 ✗）；其余 k-quant 反量化（Q2_K/Q3_K/Q5_K/Q6_K，现具名拒绝 ✗）。

**下次写代码要遵守（本仓红线，摘要在 `MEMORY.md` ✓）：**
半角引号（中文文案嵌半角双引号 ⇒ SyntaxError ✓ 已踩 4 次 ✓）/ 懒导入 torch（模块级不 import ✓）/ `__all__` 惰性导出
（先查名单再构造 ✓）/ 架构参数全显式（不猜 ✗）/ 缩小版模型走同一条前向验证（不下载真权重就能验 ✓）/ 判据一次写齐
再跑全量、跑完再报数（`tests/run_all.py` 是规模唯一权威 ✓）/ 记账 + `check_memory.py` ✓。

## 自 MEMORY.md 下移（2026-09-20，第三次腾 8k 预算）

**Skill 体系完整表述**（红线仍留在 `MEMORY.md` §Skill 体系）：绑定解析入口唯一 = `resolveDefaultSkills(agentType)`（**只扫自有**），消费 `loadAgentSkills`/`getAgentDefaults`/`routes/skills.ts`；**改绑定 = 改 md**（`AGENT_SKILL_MAP` 已删）。注入闸：`SKILL_CHAR_BUDGET`（默认 6 万，`AGENT_SKILL_BUDGET` 覆盖，0=关）→ 超预算按 priority 跳过并给诊断（口径 = `renderSkill(parseSkill(...)).length`，不等于字节数）；`agents.vue` 绑定面板是外部库 Skill **唯一 UI 挂载入口**，**拖拽真实生效**（列表顺序 = 注入顺序 = 超预算跳过顺序），合计**只算 `enabled=true`**。同一规则只留一处：`shared/prompt-blocks.ts` 的 `SCREENPLAY_FORMAT_RULES`、`IMAGE_PROMPT_TEMPLATE_CHARACTER/SCENE/SHOT`。兜底库：顶层目录无 `library.yaml` ⇒ `/meta.sources` 补合成条目（`declared:false`，label = 目录名）⇒ 「core + Σ各库 = 总数」自洽且侧栏可达。`meta.yaml` 是**死数据**（改它不生效 ✓ 全仓零读取）但 `version`/`author-*`/`source` **只此一处** ⇒ 勿擅自删 ✗（丢溯源）。改名/挪库后必核对 DB 绑定：只读直开 `{ readonly: true }`（绕开清洗副作用）；实测 5 行全 `NULL` ⇒ 当时改名零影响。

**画风体系完整表述**（红线仍留在 `MEMORY.md` §画风体系）：`prompt-utils` **不拆**；找副本顺序 = 后端常量 → skill 正文 → 工具 instruction → 前端硬编码。解析链 `characters.style` → `dramas.style` → `app_settings.art_style` → `realistic`，**唯一入口** `resolveEffectiveArtStyle()`（脏值跳过不透传），各路由不得存副本。**正负成对收口**：场景 / 分镜静帧+宫格 / 视频 / 角色·装备·道具·表情各有 `buildXxxArtStyleSuffix` + `buildXxxNegativePrompt`；skill **不得输出画风英文词**（一律后端 suffix 收口，auto-pipeline 现算不落库）。

**代码约定坑清单**（2026-09-20 第四次腾预算时从 `MEMORY.md` 下移 ✓；红线仍在那边 ✓）：

* **模块级状态**：凡「为跨项比较」引入的模块级状态 ⇒ **入口必须清零** ✗（单次调用**测不出来** ✓，连调两次才现形 ✓）。
* **`compile()` 过了 ≠ 名字在** ✗ —— 改了模块级引用还要**真的 `import` 一次并跑到那条路径** ✓（语法与名字是两道门 ✓）；本仓已应验 **3 次** ✓（最近一次：编辑时顺手把 `_mod_row` 的定义圈走 ✗ ⇒ `compile()` 照样过 ✓，一跑到 forward 才 `NameError` ✓）。
* **汇总必须打分母 + 点名异常项** ✗（只报总数 ⇒「**没读到**」与「**真的是 0**」在输出里长得一样 ✓）。
* **文档里的规模数字要么指向唯一权威、要么带实测日期** ✗（逐项罗列会随增删**腐烂** ✓ —— `run_all.py` 的 docstring、`README.md` 都被咬过 ✓）。
* ⭐ **占位符不许留在模块体里** ✗ —— 同名空壳会把**真实现静默遮住** ✓✗（导入拿到空壳 ✓，到调用才炸 ✓）；要"可选依赖"就用**取名字时才构造**（`__getattr__` ✓），且**先查名单再构造** ✓（否则问一个不存在的名字也会去 `import torch` ✗）。
* ⭐ **判据要"响亮"不要"静默"** ✗ —— 惰性导出的模块一定配一份**名字名单**（`__all__` ✓）并**先校验**：漏加名单时立刻 `AttributeError` ✓（本仓实测：新加的 `packed_rows` 忘了进名单 ⇒ 一跑就**点名**报出来 ✓✓），而不是给一个空壳/None 让错误漂到下游 ✓。
* ⭐ **写断言时，凡"顺序 / 布局 / 形状 / dtype"都要当场算一遍** ✗（实测：一天里 7 处红全是**我的期望**错，代码都是对的 ✓ —— 典型如「笛卡尔积两列逐位相等」✗（应为**值集合相同** ✓）、「token 逐个递增 t」✗（实为**一帧内共用同一个 t** ✓）、`nn.Linear(in,out).weight` = **(out,in)** ✓、`float64 × float32` 直接 `RuntimeError` ✓）。

## Skill 体系坑清单（2026-09-15 自 `MEMORY.md` 下移，腾 8k 预算 —— 本文件逼近上限时**尾部区块最先被截断**）

- ① `renderSkill()` **不含 frontmatter name** → 验证注入要用正文特征串；
- ② `prompt-utils.ts` 有顶层副作用 ⇒ 前端只能 `import type`；
- ③ **`/skills/meta` 与 `/agent-configs/defaults` 必须注册在通配路由之前**（否则被当 id 吃掉）；
- ④ 本工具 `search_content` 的 `glob` 不生效（按 path 收窄代替）；
- ⑤ `PUT /skills/<id>` 保存后回读校验 frontmatter：缺 `---` 头或 `agents` 为空 ⇒ 返 `{ warning }`（不阻断）→ 前端 `toast.warning`（防「改正文 → 默认注入静默消失」）；
- ⑥ 前端回显 DB `skills` **必须规范化**（`enabled !== false`、`priority` 缺省 `0`、过滤无 `id` 项、非数组兜 `[]`），否则**显示与实际相反**；
- ⑦ 导入 `agents/index.ts` 有 DB 清洗副作用（`[db] sanitized…`）⇒ 验证脚本会改数据；
- ⑧ **改 skill 名 / 挪库 / 引 `docs/` 后必跑 `python backend-py/app/scripts/check_skill_refs.py`**（1 = 断链；基线 **0 致命 / 0 非致命**（96 处）⇒ **非零即真回归**）。路径须写成 `` `references/x.md` `` 才受采集；⚠️ 守卫**默认未启用**（需 `git config core.hooksPath .githooks`）；
- ⑨ 改记忆必跑 `check_memory.py`、改守卫自身再跑 `test_guards.py`、体检 `check_all.py` —— 判据与基线详见 `backend-py/app/scripts/README.md`。

## `backend-py/`（Python 后端 / 绞杀者迁移，2026-09-12 起）
- **为什么不是一次性重写**：原始体量 **133 TS / 27,389 行 / 226 端点 / 29 表**且**零自动化测试** ⇒ 重写期没有任何可验证中间态。故 FastAPI 作新入口（**5790**），**未迁移的域反代到 Node（5789）**，每迁一域就 `include_router` 一行并把该域从 Node 停用；全迁完后 `PY_PORT=5789` 切单端口。
- **端口铁律**：`app/config.py` **刻意不读 `config.yaml` 的 `server.port`**（那是 Node 的端口，并存期读同一个必然抢占）。优先级 `PY_PORT > PORT > 5790`。
- **三条契约对齐约定**：① 信封 `{code,data,message}`、**错误响应无 `data` 键**（前端 `useApi.ts` 用原生 fetch，判据 `!resp.ok || json.code >= 400`，文案取 `json.message`）⇒ FastAPI 默认的 `{"detail"}` 会让文案变「请求失败 (422)」，故三类异常全部收口；特例 `/api/v1/health` 是**裸对象**。② `GET /dramas/:id/prompts` 聚合视图**本来就是 camelCase**（`customPrompt`/`imageUrl`），照抄勿「顺手统一」。③ **SQLAlchemy `**kwargs` ≠ JS 对象展开**：`values(**fields, title=...)` 同名键直接 `TypeError: got multiple values for keyword argument`，必须 `values = {**fields, ...}` 再覆盖 —— **已在 `PUT /dramas/:id/episodes` 真实踩到**。
- **为什么用 SQLAlchemy Core 而不是 ORM**：Core 的行**本身就是 snake_case = HTTP 契约形状** ⇒ 原 Node 的 `toSnakeCase()` 转换层在 Python 侧**不存在**，少一层字段漂移源，也不必维护「Python 属性名 ↔ DB 列名」映射。
- **模式核对用真实库、别只对 `schema.ts`**：`backend-py/tests/smoke_test.py` 拿 `data/drama.db` 的 `PRAGMA table_info` 逐列比对，实测**真实库比 Node 模型多 2 表（`assets` / `props`）与 2 列（`image_generations.minio_url` / `video_generations.minio_url`）** —— 全是旧版本 MinIO 遗留（Node `db/index.ts` 只建 **29** 表、全仓库零引用）⇒ 两边都不建模。**列序差异不影响正确性**（SQLAlchemy 全程按列名生成 SQL，不做位置化 INSERT/SELECT）。
- **回归唯一入口**：`backend-py/tests/smoke_test.py`（**64 用例**：模式 + 信封 + **错误文案逐字** + 软删 + 反代 501）。写操作**全部落在 `data/drama.db` 的副本**上（`DATA_ROOT` 指向临时目录），**真实库只读**。改 Python 侧代码后必跑。
- **🎯 当前总目标：把 `backend/` 全量迁完后删除它**（用户指令）。⇒ 不是翻译而是**重写**：剩 ~90 端点 + ~12k 行服务代码 + Mastra 替换。**7 阶段路线（S1–S7）已写进 `backend-py/README.md`**：S1 地基 ✅ → S2 `prompt-utils`(1456) → S3 适配器(1611) → S4 媒体服务 → S5 **Mastra 替换**(~2900) → S6 编排/长任务 → S7 收尾+删 `backend/`。**`backend/` 只在 S7 删**（删前须证明 224 端点等价）。
- **JS 数值序列化坑（第三形态）**：`js_number` 曾把整数返成 float ⇒ 响应体出现 `1.0` 而 JS 是 `1`。**被字符串断言揪出**（`drama-37.0`），数值比较完全掩盖 ⇒ **断言数字要看字节，别只用 `==`**。现 `js_number` 在 `|x|≤2^53` 内返 `int`。
- **已迁移域（24 域 / 130 端点，口径由 `tests/route_parity_test.py` 机械扫描得出）**：`dramas`(9)、`episodes`(8)、`characters`(4)、`scenes`(4)、`props`(5)、`storyboards`(5)、四个资源库(10+11+10+10)、`presets`(4)、`app-settings`(2)、`asset-versions`(2)、`traces`(3 只读)、`storage`(1 只 info)、`usage`(2)、`agent-configs`(5)、`style-profiles`(7)、`generations`(1)、`ai-configs`(15/17) + `ai-providers`(1)、`skills`(6，纯文件系统域)、`upload`(3)、`export`(2/7，工程账本)。依赖 LLM / 视觉模型 / 媒体生成 / ffmpeg / 节奏服务的端点**刻意不注册**走反代 —— 比返回 501 更可用。四个资源库用**规格驱动单一实现**（`LibrarySpec` × 4），因为 Node 侧那 4 个文件是复制体。
- **⚠️ 别把「依赖外部」当成「不可迁移」**：`/ollama/*`（子进程 + HTTP）、`POST /models`、`POST /test`、`runtime/health` 我都迁了 —— Python 做子进程与 HTTP 完全等价。**真正不可迁的是「需要重建的框架」**（Mastra Agent 循环、17 家厂商适配器、ffmpeg 链路）。
- **⭐ 不该迁的明确信号：跨进程共享的「内存态」**。`/gpu/status` 与 `/gpu/release-all` 依赖进程内的 GPU 租约管理器 —— 两个后端会各有各自的一份视图 ⇒ VRAM 协调失效。租约必须和真正用 GPU 的代码（媒体域）待在一起。**能迁 ≠ 该迁。**
- **🧪 「假上游」测试手法**：对外探测类端点用 `http.server` + `threading`（端口 0）起假服务即可确定性测试，**别依赖环境里装没装**（Ollama 就是这么测的：标签省略匹配、reachable 白名单、key 遮蔽全覆盖）。
- **⭐ 词表/提示词资产必须配「逐字漂移守卫」**：`prompt_utils.py` 的词表**就是生成结果本身**，改一个字就改变出图/出片，两边不一致时**不报错**、只表现为「同一剧在两套后端下风格不同」。守卫从 TS 抽常量逐字比对（含 `${BASE}` 模板与**跨行拼接**），并比对画风 key 集合。现 **0 漂移**。⇒ 抽取器的两个要点：能续行吃 `+`/未闭合引号、**遇下一条声明即停**（否则吞掉后续声明 ⇒ 静默漏比对）；**"抽不到"要报问题，不能静默跳过**（否则守卫假绿）。
- **S2 进度**：`prompt-utils.ts` 实为 **1629 行**，切成 6 块；**第 1–6 块已全部完成（S2 收口）** —— 另含 `camera-movement-guides.ts`（`camera_movement_guides.py`，29 条中文运镜表）。宫格链路的反直觉点：换行拼接 `filter(Boolean)` **丢空串**（与 `, ` 拼接相反）｜画风回退是字面量 `cinematic illustration style` 而非 MASTER｜三分支标点各不相同｜参考资产**上限 6 张、顺序决定编号**｜`_pos_label` 的 rows 参数没用｜`first_frame` 格子数=分镜数而另两模式=rows×cols。（①画风层+负面词层 ②预设/角色/装备三视图/单品/表情/物品/场景构建器）。三条硬性规则：命中画风 `art+TAIL`｜未命中才回退 `VISUAL_STYLE_MASTER`（**二选一，禁叠加**）｜视频链只用 `VIDEO_*` 家族。第 3 块起有外部依赖（`resolveVisualTerm` 在 L938+、`getCameraMovementComposition` 在 L1537+）。
- **⚠️ prompt 构建器里五处「看起来一样、语义不同」的地方（已单独断言，别合并）**：`build_character_appearance_text`（无 `core features:` 前缀 / `accessories:` / **`': '`** 拼接）vs `build_character_visuals_clause`（有前缀 / `wearing accessories:` / `', '`）｜装备三视图负面词**不排除** `side-by-side`，单品图负面词**必须排除**｜表情图服装链**无** `costume` 字段｜预设图片/视频两个 prompt 字段顺序不同｜物品图分类用**全角括号**。
- **保真细节**：JS `Array.join` 会为空槽留下分隔符（`[undefined,'x'].join(', ')` → `", x"`）⇒ Python 侧**不过滤空值**，否则词序与逗号位置变化即改变 prompt。
- **S3 厂商适配器层已迁完**（`app/services/adapters/` 9 文件 ← `backend/src/services/adapters/` 16 文件 1611 行）：**17 家**（图片 minimax/openai/gemini/volcengine/ali/chatfire/local-sd｜视频 minimax/volcengine/vidu/ali｜TTS minimax/cosyvoice｜文本 openai/openrouter/chatfire/ollama/volcengine/ali/minimax/gemini，后 8 家里 7 家 OpenAI 兼容**共享一个实例**）。`tests/adapters_test.py` **101 用例**。
- **⭐ 关键认知：适配器层是纯函数** —— 只 `buildGenerateRequest → {url,method,headers,body}` 和 `parseGenerateResponse`，**自己不发起 HTTP**（I/O 在 `image-generation` 那层）⇒ 整层无需网络/密钥/mock，可离线逐字比对。这条把工时评估里"必须用真实密钥复验"的一块降级为纯单测。
- **适配器 JS 语义垫片在 `adapters/jscompat.py`（6 条，别在业务层重写）**：`parseInt`（取前导数字；**NaN 是值不是"不赋值"** ⇒ 键仍在、序列化成 null）｜`Buffer.from(x,'base64')` **宽容解码**（容忍缺 `=`，Python 会抛错）｜`a?.b?.[0]?.c` ⇒ `dig`｜**`.length` 只有字符串/数组有**（`len({})`=0 vs `({}).length`=undefined，结论相反）⇒ `js_length`｜`String(16.0)==='16'` ⇒ `js_num_str`｜`JSON.stringify` 紧凑+不转义中文 ⇒ `js_json_stringify`。
- **「显式 undefined」与「DB null」在 Python 都是 None 但处理相反**：JS 的 `{seed: cond ? undefined : rand()}` 会被 `JSON.stringify` **丢键** ⇒ Python **不加键**；DB 可空字段是 `null` ⇒ 如实保留 None。
- **适配器层反直觉行为（有用例锁，别当 bug 改）**：MiniMax 图片 `aspect_ratio` 只在**入参**给 `size` 时派生（默认 size 不参与）｜Vidu 用 `Token x` 认证 + **无轮询接口**（伪 URL + 靠 Webhook）｜火山视频 duration 夹 `[4,12]`｜Ali 两家 seed 随机（抽成可替换函数）｜Gemini 图片判定顺序 URL→base64→taskId→error，安全拦截抛**中文**提示。
- **守卫第一次跑就抓到我的 fixture 写错**：TS 的 URL 是 `join(baseUrl,'/v1','/image_generation')` **两个独立字面量**，拿拼起来的整串去比对必然误报。**教训：漂移检查的基准字面量必须是源码里真实存在的形态，不能是推导出来的。**
- **S4 前置两块已迁**：`services/vendor_errors.py`（← `utils/vendor-errors.ts` 362 行：厂商错误→**用户可读中文**的归因层 + 指数退避重试）｜`services/text_generation.py`（← `services/text-generation.ts` 556 行：动作建议/拆镜头/续写/优化/拆分视觉/音色打标 + **本地规则拆分器**）。
- **⭐ 归因层的两级设计**：先判「是不是内容审核拦截」，再细分「是否由**上传图/首帧**触发」（火山 Seedance `InputImageSensitiveContentDetected.PrivacyInformation`）—— 因为处理方式完全不同：**改文字 vs 换参考图**。`is_non_retryable_http_error` 让轮询对审核/参数错**立即判失败**，避免空转数十分钟。
- **⚠️ 归因层顺序易搞反**：审核判定**必须先命中**（code/type/message 有审核信号），之后的「中文『人物』正则」才参与细分；`{"error":{"message":"图片包含人物"}}`（只有中文无审核信号）**不走审核分支**。另：模型名正则很松，`"model is not a valid model"` 会提取出 **`is`** 当模型名 —— **丑但保真，别顺手修**。
- **⚠️ 唯一一处我有意偏离 Node**：错误归因处原 TS 对**数字 code**（`{"error":{"code":404}}` 很常见）会 `TypeError`，把中文说明变 500；Python 统一 `str()` 归一。已写进 README「已知差异」表 + 用例。
- **⚠️ 本地模型的 GPU 租约未迁**（`generate_text` 原调 `gpuManager.acquire('text',…)`）：Python 侧不申请租约 ⇒ 本地模型并发请求不受显存调度保护，可能 OOM。`/gpu/*` 整段留 Node，**S7 前必须补齐**。
- **`text-generation.ts` 的真正价值是那个「本地规则拆分器」**：8 张词表 + 相邻子句合并（上限 34 字），AI 失败也要有输出，结果会写进角色卡 ⇒ 已把 8 张词表**逐项**纳入漂移守卫（56 条的引导词表错一个字，拆分结果就悄悄变）。
- **自检已到 6 项，用 `tests/run_all.py` 一键跑**（**772 项**）：smoke 482 / adapters 101 / vendor_errors 58 / text_generation 67 / **image_generation 64** / route_parity（守卫）。
- **S4 主体 `image-generation.ts`（753 行）已迁**（`services/image_generation.py`）+ 5 个依赖：`take_budget.py`、`file_storage.download_file`、`usage_tracking.record_usage`（写入侧）、`era_background.get_era_background/apply_era_image_clause`、`color_grade`（像素管线未迁，走降级）。
- **⭐ `ai_service_configs.model` 存的是 JSON 数组字符串**（多模型 fallback）：读侧 Node/Python 都是 `JSON.parse` 取 `models[0]`；接口入参是**数组**（路由 `json.dumps` 后落库）。传裸字符串 → model 解析成**空串**（两边一致，是契约不是 bug）。**排查「模型没生效」先看这里。**
- **🔥 `json.dumps` ≠ `JSON.stringify`（默认分隔符 `(', ', ': ')` 带空格）—— 曾全仓 8 处同时存在**：写进与 Node 共用的列、作为请求体发给厂商、或直接作为响应体都会不一致。已修正：`image_generation`、`asset_versions`、`dramas`(tags/era)、`presets`、`ai_configs`(probe body)、`text_generation`(body)、`resource_library`、`export`。**并加了 AST 机械守卫**（`route_parity_test._json_dumps_drift`，白名单只留深拷贝与日志 indent=2）。**写这种守卫要用 AST，正则扫源码会把 docstring 与 `**kwargs` 误报。**
- **S4 视频链路完成**（`video_generation.py` ← 498 行）+ `video_probe.py`（ffprobe 补时长，失败即 0，无 ffprobe 不报错）+ **S2 第 5 块**（`strip_video_prompt_tags` + `validate_dialogue_character_consistency`）。自检 7 项 / **822 项全绿**。
- **视频层三个独有点（有用例锁）**：① prompt 先**洗**（剥 DSL 标签 → 逐镜禁止变化 → `[background_audio]` 幂等注入）；② **空数组待遇不同**：`referenceImageUrls: []` → `"[]"`（truthy），`referenceAudioUrls: []` → `NULL`（`.length`）；③ **同步完成路径不更新分镜**（storyboardId 形参是 undefined），但版本留档/QC 走「形参 ?? 记录值」仍生效。Vidu = Webhook 型：提交即返回，恢复时保持 processing。
- **S4 TTS 链路完成**：`tts_generation.py`（**同步链路**：无轮询、hex 直接落盘、按字符数计费）+ `voice_clone.py`（MiniMax 上传+复刻 / CosyVoice 零样本）。自检 8 项 / **856 项全绿**。`fetch_with_retry` 新增 `files`/`data`（multipart，**不可与 `content` 同传**）。
- **TTS 层三个易错点（有用例锁）**：① `params.model || models[attempt]` —— **传了 model 就关掉 fallback**；② hex 解码必须自写 `_hex_to_bytes`（Node `Buffer.from(s,'hex')` 遇非法字符**截断**，Python `bytes.fromhex` 抛错）；③ CosyVoice 零样本要求参考音频**文件真实存在**，否则静默退回 `/tts`。
- **S2 第 3+4 块 + 视觉图谱已完成**：`visual_graph.py`（41 节点 / 四类图谱 / 中文→英文术语翻译 + 风格引导）｜`build_storyboard_image_prompt` / `build_storyboard_video_prompt`｜6 个 DB 辅助（分镜上下文 / 角色立绘 / 参考图三源合并 / 参考音频≤3）。**S2 现 5/6 块**（只剩宫格 prompt）。自检 **9 项 / 894 项全绿**，守卫含**视觉图谱 41 节点逐条比对**。
- **⭐ 分镜视频 prompt 的末尾拼装是字符级契约**：`<各段 '. ' 连接>, <声音策略>.<UI留白规则><环境音标记> <画风层>` —— UI 规则带**前置逗号**、环境音带**前置空格**、画风层命中时去掉自带逗号前缀。断言写错分隔符（`. ` vs `, `）就会挂 ⇒ 这类拼装只能真跑核对。
- **加速批次再清 4 域（+22 端点）**：`visual-graph`(4，**服务层早已就位，等于白捡**) / `images`(4) / `webhooks`(1) / `aiVoices`(5)。累计 **16 项自检 / 1171 项**；Node **193** / Python **162**。
- **⚠️ 这几域的「特殊契约」（别统一成标准信封）**：`visual-graph` 前三个端点是**裸 JSON**、`/guidance` 是 **text/plain**（且 `category` 非法时 `GET /` 返回全部、`GET /terms` 却 400）｜`images` 负面词**二选一**（有 storyboard → 分镜那套；无 → `NEGATIVE_BASE`）｜`webhooks/vidu` 三种码（缺 task_id 400 / **task_id 查不到 200** / `state=failed` **200** / 下载失败才 400）｜`aiVoices` 音色 id = `ds_`/`cv_` + **base36 时间戳**、语言推断**顺序敏感**、`/clone` 走 multipart、`/sync` **先删光再插**。
- **preset-framework 已完成（6 端点）** —— 但它是**从未接线的骨架**：TS 引用的 `systemMetadata` / `sceneNumber` / `cameraMovement` / `imageUrl` **在共享 schema 里都不存在**（Node 侧同样崩）⇒ 按真实列名适配（`dramas.metadata` / `storyboards.storyboard_number` / `.movement` / `.first_frame_image`），「每镜 shotConfig」无列可写。**这是唯一有意偏离 TS 的地方**（照抄只能复现崩溃）。数据池是占位符；`derive_space_and_frame` 用**字符串长度**取模；响应是 `{success,data}`/`{success,msg}` 自成一格。
- **`runMerged` 未移植** ⇒ 编排用 `asyncio.gather(return_exceptions=True)` 同形实现。测编排功能时：**首帧/视频是后台任务回写的**，假生成器只插记录 ⇒ 测试要手工"完成"再断言（否则全跳过、汇总为 0）。
- **`skills` + `skill-parser` 已迁**（`agents/skill_parser.py` + `agents/skills.py`）⇒ 三处受益：**runtime 的 skill 段真的注入了**（此前恒 None）、**`GET /agent-configs/defaults` 改为本地实现**、optimizer 前置就位。⭐ 两条判据：① `userConfigured` 必须看「**解析出的配置项**是否非空」，看「过滤 enabled 后是否为空」会把「用户全取消」反向执行为「注入全部默认」；② 体量预算闸（默认 60000，超者跳过并写进注入文本）。⚠️ 原 TS 三处粗糙（保真不修）：注释称 `AGENT_SKILL_BUDGET=0` 关闭限制实则回落 60000（`budget > 0` 闸形同虚设）｜`if (!sections.length) return null` 排在通知前 ⇒ 全部缺失时通知丢失｜`toToolList` 对 `[ "a", "b" ]` 混写会留脏引号。⚠️ **守卫口径是 `app.openapi()`** ⇒ `include_in_schema=False` 的端点会被当作「未迁移」探活；**委托端点改成本地实现时要去掉该标记**（`/agent-configs/defaults` 就这么误报过 SHADOW）。
- **S6 最终剩余 9 条（精确清单，2026-09-13 收口时实测，别再重新调研）**：`GET /ai-configs/gpu/status`、`POST /ai-configs/gpu/release-all`（**GPU 租约域**）｜`GET /dramas/*/rhythm`（`rhythm-phase.ts` 未迁）｜`POST /episodes/*/consistency-qc`（`consistency-qc.ts` 未迁，**视觉模型 + 多图**）｜`POST /storage/change`（**有意延后**：写项目级 `.data-root` 标记，Node 也读 ⇒ 绞杀期跨进程副作用，**单后端后再迁**）｜`POST /storyboards/*/qc`、`POST /storyboards/*/retry-qc`（`qc-scoring.ts` / `qc-retry.ts` 未迁）｜`POST /storyboards/*/regenerate-frame`、`POST /storyboards/*/set-frame`（**ffmpeg 抽帧/写帧链路**，`frame_extractor.py` 只有尾帧不够用）。⇒ 结论：**剩下全是「需先迁 Node 侧服务」或「需 GPU 环境」或「有意延后」**，S6 的可做档已清空。（`era-background/extract` 与 `style-profiles/distill` 已于第 17 步关闭：前者剧本聚合无 ORDER BY + 8000 字首尾截断 + 异常分层包装；后者不落库 + 四对象三数组归一 + 异常全吞成 ok:false ⇒ 400，测量事实 indent=2 已加白名单。）
- **S6 可做档继续清（storyboards 5 条）**：`generate-tts` / `regenerate-image` / `action-suggestion` / `split` / `optimize-prompt`。⚠️ generate-tts **两套回执**（多人 `{lines}` 且结果整体存 JSON / 单人扁平 + 条件 warning）；⭐ **对话行解析是贪婪的** —— `A：台词。B：台词。` 只会拆出 1 行（B 被吞），多行必须**换行分隔**（TS 注释的「→3 行」示例是错的）。split 落库七件事（原镜头保留 / 新镜头插其后 / 后续编号顺延 / 角色关联同步 / duration=clamp(round(dur/n),2,4) / dialogue=None / status=pending）。`_shot_context` 共用：有 `scene_id` 则场景信息**整体换场景表**的；视觉风格链 storyboard→episode→`dramas.style`。剩余 **17 条**（storyboards 4 卡 ffmpeg 抽帧与 qc 服务、export 5 整文件、dramas/episodes 各 2、gpu 2、distill/storage 各 1）。
- **S6 「未注册 32 条」的真相 + 补齐 10 条**：32 条里只有 **2 条**是 `/gpu/*`，其余 30 条是**有意延后**（各路由 docstring 有「未迁移」清单）但**一批延后理由已过期**（依赖的服务 S3/S4 已迁）。本轮补齐 **characters 8 条生成端点 + props/scenes 各 1 条 generate-image**。⚠️ characters 六处：错误码不统一（角色找不到 400、**唯 auto-split 是 404**、three-views **无 try/catch**、批量**逐条静默跳过**）｜配置回退（单张 `ep ?? drama 级`，**批量只用 ep**）｜自定义 prompt 也收口画风 |视觉锚定（立绘不锚、三视图锚主立绘、装备/表情锚 combined、**最多 6 张**）｜装备负向**嵌套三元**（缺人物词就强制换）｜`coreFeatures` 只吃 JSON 字符串。props: 404 中文、留白构建器、`config_id` falsy、回执 camelCase+回显；scenes: **跨集参考图同地点最多 2 张、地点空必须跳过**、processing→成功不重置/失败置 failed。⭐ **`smoke_test.py` 硬编码「这 3 条未迁移 ⇒ 501」⇒ 迁移后全量才暴露**：凡断言「未迁移/501」的测试迁移时必须同步（route_parity 用动态口径已兜住）。
- **`localModels` 整域关闭**（`local_model_scan.py` 709 行 12 条启发式规则 + `local_models.py` 11 端点含 NDJSON 流式下载 + 异步扫描任务表）。⚠️ **规则顺序即优先级**（H3 TE 属视频、Wan GGUF 先于通用 GGUF、组件目录先于 llm-dir）；**组件不跳过**（`role='component'`），ComfyUI 是 `callable=false` 空 baseUrl；同步/异步默认值不同（`5/8000` vs `8/50000`）；`maxDepth/maxFiles/taskId` 是 camelCase ⇒ `Query(alias)`；下载支持跳过/断点续传（206/200）/.part 保留/ModelScope 签名 URL；删除限制在存储目录内且不许删根。⭐ **TestClient 的 portal 只在请求期间给事件循环切片** ⇒ 后台任务+轮询的模式别在 HTTP 层断言，直接 await 任务体。⭐ **守卫 `_TS_ROUTE` 曾把 `headers.get('content-length')` 当端点**（假阳性）⇒ 口径改为「裸标识符 + `.方法('字面量')`」（只锚 `app.` 会漏掉 53 处 `router.`）。
- **`auto-pipeline` 整域关闭**（`services/auto_pipeline.py` 814 行 8 阶段 + `routers/auto_pipeline.py` 4 端点含唯一 SSE + lifespan 崩溃恢复）。⚠️ 阶段顺序反直觉：`scripting→extracting→**voicing**→storyboarding`；幂等判据比对的是 **`next` 索引**。⚠️ 媒体开关**依赖补全**（video 隐含 image，merge 隐含全部）。⚠️ 幂等三件套：查已有产物或 `processing/pending/completed` 才跳过（防重复扣费）；只写 `tail_frame_image` 不碰 `last_frame_image`。⚠️ 同进程防重入 + **follow-up 入队**；`dramas.metadata` 存紧凑 JSON。⭐ **`asyncio.ensure_future` 在 ASGI 工作线程里会抛「no current event loop」** ⇒ 从协程调用的同步代码起后台任务必须用 **`get_running_loop().create_task`**，且该路由要 `async def`。⭐ 自检 SSE 别用 TestClient 读永不结束的流（会互相等死）⇒ 直接 await 路由函数拿 `StreamingResponse` 自己迭代，且**帧以空行结尾要过滤**。
- **`auto-pipeline` 前置已迁**（`services/sse_hub.py` + `services/frame_extractor.py`）；**服务本体 744 行（8 阶段）与路由 101 行（唯一 SSE 端点）待迁**。红线：① `publish_pipeline_event` **永不抛**、订阅者异常自吞、频道空自动清理、不落盘不缓存；② 尾帧提取**只写 `tail_frame_image`（真实尾帧）**，**绝不碰 `last_frame_image`（设计尾帧）** —— 混了会污染 shot-router 的 FL2VA 决策。细节：`seek = duration - 0.2`、`-ss` 必须在 `-i` 前（fluent-ffmpeg `seekInput` 语义）｜**别复用 `probe_video_duration`**（它返回整秒，这里要浮点）｜`storage_root` 就是 `<data_root>/static`（两套路径写法等价）｜`frame_extractor` 留了 `_run_ffmpeg` 测试缝。
- **评测域整域关闭（5/5 端点）**：`evaluation/` 的 types/catalog/scorer/evaluator/optimizer + `services/evaluation_scheduler.py`，端点 `cases`/`evaluate`/`optimize`/`scheduler`/`run` 全本地化；`main.py` lifespan 会启动调度器（默认关闭）。⚠️ **调度器 `running` 无 try/finally（原 TS 缺陷，保真）**：循环外抛错（如基准目录坏 JSON）会让 `running` 永卡 true ⇒ 之后所有触发被防重入吃掉（自检故意钉住）。⚠️ 一轮里**每个 case 各开短事务**（别整轮一个事务，否则 LLM 期间占住 SQLite 写锁）。
- **creator + optimizer 已迁**（`agents/creator.py` + `evaluation/optimizer.py`）⇒ 解锁 `/evaluation/optimize/{id}` 与 **`/agent-configs/generate`**（一处投入两处收口）。`creator`：`AVAILABLE_SKILL_IDS` 模块加载快照｜skills 非法静默丢弃、空则回退默认｜upsert 取第一条、只改 name/description/systemPrompt/skills/isActive 并把 `deleted_at` 清回 null 复活｜`persist_agent_config` 返回 **Row**（路由用 `row_to_dict` 归一）。`optimizer` 状态机四判据：防作弊（**只给维度名+分数，绝不给 detail**）｜runtimeModel 不一致**不 accept（99 分也不认）**｜accept = 严格高于**当前 best**｜落库 = autoPersist 且 best.version>0 且 best.score>reference。⚠️ 两套必填文案别混（路由 `agent_type required`/`requirement required` vs creator `未知 Agent 类型：…`/`requirement 不能为空`；optimizer 短候选是英文报错）。⚠️ `optimizer.py` 在 `json.dumps` 守卫白名单里（历史文件 `indent=2` 有意）。**评测域剩调度器**（`/scheduler`、`/run`）。
- **评测执行器已迁**（`evaluation/evaluator.py` + `routers/evaluation.py` 的 `/cases`、`/evaluate/{id}`）。⭐ **seed 必须逐条立即提交**（每个 insert 一个短事务）：Agent 的工具用自己的连接读库，seed 攒在请求事务里会「对空气评分」+ SQLite 互锁（自检用独立连接回查钉死）。⭐ **抽取口径不同**：分镜/剧本取最后一次调用，角色/场景/音色跨调用累加；工具名先归一（驼峰 `saveStoryboards` vs 下划线 id，不归一会 0 命中且不报错）。`cleanup_case` 失败只告警、`finally` 保证异常路径也清理。未迁的 3 条（optimize/scheduler/run）**无需显式委派**，直接走 catch-all；守卫把它们算进「未注册」当待办清单。
- **评测闭环：确定性部分已迁**（`services/evaluation/` 的 `types.py`/`catalog.py`/`scorer.py`）。⚠️ 打分器要位精对齐 **JS 舍入**（`Math.round(n*10)/10`、`toFixed(0)` 都是半数向 +∞ ⇒ 用 `js_round`，别用 Python 内置 `round`）；⚠️ `isFilled` 里 **`0` 与 `False` 都算「填了」**（TS `typeof` 落到 `return true`）；⚠️ 回执 `caseId` 恒为空串；⚠️ `benchmarks/` 现在读 `backend/benchmarks`，**删 `backend/` 时要把 4 个 JSON 挪到 `PROJECT_ROOT/benchmarks`**。依赖链：`evaluator` 只需 runtime ✓；**`optimizer` 还要 `creator.ts`+`skills.ts`+`skill-parser.ts`**（而 skills 同时是 runtime skill 注入与 `/agent-configs/defaults` 的前置 ⇒ 优先级高）；`cli.ts` 是开发 CLI，**不迁**。
- **`DEFAULT_PROMPTS` 已搬**（决策变更，2026-09-12）：`services/agent_prompts.py`（6 段提示词 + `${...}` 插值到 `prompt_blocks` 常量）+ **第 9 道守卫 `_agent_prompts_drift`（逐字）**。理由：runtime 的 base instructions 曾是空串、`/agent-configs/defaults` 委托回 Node、`evaluation` 需要它作基准、S7 要删 `backend/`。**`build_agent_config` 已按 TS 语义回落**。⚠️ `/agent-configs/defaults` 仍需 `skills.ts`(218)+`skill-parser.ts`(172) 才能本地化（提示词部分已具备）。
- **S6 第 1 步：MCP 完成**：`agents/mcp.py`（← 285 行，**自写极简 JSON-RPC 客户端** —— `mcp` SDK 未装且刻意不新增依赖；已实现 stdio + streamable HTTP，**HTTP+SSE 未移植 ⇒ 显式报错**）+ `routers/mcp.py`（3 端点）。四条「连不上」语义：配置坏了⇒空列表｜单 server 失败⇒只降级绝不 throw｜single-flight + 指纹缓存｜重名跳过后者。路由三处反直觉：错误信封是**带 `data:null` 的 500**；`/test` body **不宽容**（坏 JSON ⇒ 500）；`/test` 的 name 只判 truthy（`123` 放行到业务层返回 200）。
- **🔧 `ToolRegistry.get` 已加固为「按键查 + 按 id 兜底」**：驱动查的是 LLM 回传的 `tool.id`，若工厂用 camelCase 键就会静默变「Unknown tool」。
- **⚠️ 测试跨 `asyncio.run` 复用 stdio 子进程必炸**（管道绑定旧循环）⇒ 异步检查收进同一次 `asyncio.run`，并 `_close_all_servers()` 收尾。
- **S5 全阶段完成**（协议/工具基座 + 6 组工具 + 运行时 + **`agent` 聊天域 2 端点**）。`agent.ts` 只有 57 行、**非流式**（流式只在 `auto-pipeline.ts`）；**两处非法类型文案不同**（`chat` 带类型名、`debug` 不带）；校验顺序是「类型 → body → start 日志 → 再校验 id」；缺 id 用 falsy 判定（`0` 算缺）；**运行期异常一律 400 而非 500**；`usage` 回执是 camelCase 或 null；`maxSteps` 硬编码 20。新增 `VALID_AGENT_TYPES` + 第 8 道守卫 `_agent_types_drift`。
- **S5 运行时（`agents/index.ts` 745 行）已完成**：`app/services/agents/runtime.py` —— 纯逻辑（失败分类/退避/token 统计/归一/指令组装/模型候选）｜装配（`build_agent_config`/`append_style_profile`）｜驱动（模型 fallback + 瞬态退避，`generate`/`sleep` 可注入）。⚠️ **传输层自建**（文本适配器无 `tools` 支持，全仓 0 次）⇒ 复用 `fetch_with_retry` + 适配器 `build_request`，注入 `tools`，自跑工具循环；**Gemini 函数调用循环未支持**（显式报错）。⚠️ `DEFAULT_PROMPTS` 刻意未搬 ⇒ 无 DB 配置时 base instructions 为空串；`skills`/`mcp`/`subagent`/`rhythm_phase` 未迁（`orchestrator` 工具集为空占位）。
- **⭐ 第 7 道机械守卫：查询参数名镜像**（`_query_param_drift`）：已迁 TS 路由里 `c.req.query('X')` 的 **camelCase** 名必须在 Python 路由里出现（`alias="X"`）。**病根**：`preset-framework` 的 `excludeFamily` 曾被静默忽略（FastAPI 形参是蛇形 `exclude_family` ⇒ 收不到、因有默认值不报错）⇒ 「排除上张家族」失效，只在随机撞上时**偶发**暴露。**待迁同类坑**：`export.ts` 的 `episodeId`、`localModels.ts` 的 `maxDepth`/`maxFiles`/`taskId`。
- **S5 六组工具 6/6 全迁完**（grid-prompt / voice / script / corpus / extract / storyboard）⇒ `app/services/agents/tools/` 共 6 个文件。`storyboard_tools.py` 的要点：**整集重建**（删旧分镜及角色关联、物品关联不删）｜**说话人自动绑定**（对白首个「角色名：」→ 名字到 speaker_id）｜**对白一致性只告警**（回执带 mismatch 数）｜集时长 `ceil(总秒/60)`｜`update_storyboard` **按字段出现写**（传 null 会真清空）｜`assign_rhythm_phases` 是**占位**（`rhythm-phase.ts` 未迁）。
- **S5（Mastra Agent 运行时）已开工**：第一块 `prompt-blocks.ts` → `services/prompt_blocks.py`（4 常量，**守卫逐字比对 + 报首个差异字符位**）。S5 全貌 ≈**3,974 行**（`agents/index.ts` 664 含 6 类 Agent 长提示词｜`tools/*` 1,362 调用已迁服务｜`mcp` 257｜`skills` 390｜`evaluation/` 985）。**难点是「造运行时」不是「搬代码」**：Node 用 Mastra（Python 无对应物）⇒ 要手写 agent 循环（system prompt + 工具注册表 + 步数上限 + 流式事件协议 `protocol.ts`），LLM 调用复用已迁的 text 适配器。
- **剩余 5 域 / 25 端点（全需 S5/S6 底座）**：`agent`(2，Mastra) ｜`auto-pipeline`(4，编排) ｜`mcp`(3) ｜`evaluation`(5) ｜`localModels`(11，GPU)。服务层欠账：**镜头 QC 打分**、**像素处理(校色/压缩，需 Pillow)**。
- **进度基线（本日末）**：**17 项自检 / 1206 项全绿**；守卫 Node **199** / Python **168**（0 遮蔽 + 0 漂移）；已迁 **29 个路由域**。
- **grid（宫格）已完成 -> S4 主链全通**：`routers/grid.py`（4 端点）+ `services/grid_split.py`（**ffmpeg crop 替 sharp**，不新增 Pillow）。`/grid/prompt` 的 Agent 分支未移植也**不影响可用**（原实现里 Agent 报错即回落本地构建器 ⇒ 现在直接走 `source='fallback'`）。画布 `960*cols x 540*rows`；`/split` 的 reference 分支**追加**到 `reference_images`（紧凑 JSON）。自检 **14 项 / 1091 项**。
- **⚠️ patch 目标必须选「使用方模块」**（已咬 3 次）：`from x import y` 之后改 `x.y` 无效，要 patch 导入方的名字；同时留一份真身引用才测得到校验分支。
- **merge（整集拼接）已完成**：`services/ffmpeg_merge.py` + `routers/merge.py`（2 端点，路径 `/api/v1/merge/...`）。三段 ffmpeg：concat → **BGM 混音**（`-stream_loop -1` 属于**后加入的 BGM 输入**；`volume/afade/amix/alimiter`；失败只告警回退）→ **响度归一化**（`loudnorm=I=-14:TP=-1.5:LRA=11`，**无音轨跳过**）。切片按 `storyboard_number` 升序；必须**全部**镜头已合成。`runConsistencyQcBeforeMerge` **保留空实现**（QC 打分未迁）。自检 **13 项 / 1049 项**。
- **⭐ 铁律：写进 DB / 请求体 / 响应体的 JSON 必须 `separators=(",",":")`**（`JSON.stringify` 无空格）。AST 守卫专拦这个 —— `video_merges.scenes` 就曾被它当场报 DRIFT。
- **⚠️ 测试火忘任务的写法**：服务内 `asyncio.create_task` 在同步 `main()` 里没有事件循环 ⇒ 包 `async def` + `asyncio.run` + `await asyncio.sleep(0)`；patch 掉的后台入口要**留真身引用**供失败路径单独驱动。
- **compose（单镜合成）已完成**：`services/ffmpeg_compose.py` + `routers/compose.py`（3 端点）。真实路径 **`/api/v1/compose/...`**（Node 的 `api.route('/compose', compose)` + 子路径自带 `/storyboards`）。自检 **12 项 / 1020 项全绿**。
- **🔴 事务边界铁律（移植服务内部写库时必问）**：Node 的 drizzle `db.update().run()` 是**逐语句自动提交** ⇒ Python 侧**不要借调用方的连接**（FastAPI 依赖「异常即 rollback」会把失败态 `compose_failed` 一起抹掉）。要服务**自己开短事务**（`_write`/`_fetch`）。
- **⚠️ 两个静默匹配坑**：行转 dict 后是**蛇形键**（`voice_style` 不是 `voiceStyle`）｜`match_character_by_speaker_name` 内部 `getattr(c,'name')` ⇒ **必须传 Row，传 dict 静默匹配不到**。
- **⚠️ 火忘型后台任务在测试里必须隔离**：否则与测试连接在 SQLite（单写者）上互锁 → `database is locked` 重试到超时。
- **`shot-router` + `videos.ts` 路由已完成**：`shot_router.py`（7 种路线 + 回写 `storyboards.route/route_reason` + `recompute_episode_routes`）｜`routers/videos.py`（6 端点，已注册进 main.py）。自检 **10 项 / 937 项全绿**；守卫 `MIGRATED` 加 `videos.ts`（Node 170 条 / Python 139 条已注册）。
- **⭐ `POST /videos` 的「帧来源统一」是修 bug 式移植**：此前决策读 DB 帧、请求只读 body 帧 ⇒ 会产出「判定 first_last/single 却一帧都不带」的坏请求。现在统一候选帧（body 优先 → 分镜已存兜底），决策与请求共用同一份。
- **⚠️ 路由优先级的两个坑**：① `prevTail` 必须在 T2V **之前**消费，否则 `referenceMode` 落 `none`，顺接帧不会被 adapter 派发；② **`multiple` 分支不消费帧字段**（R2V 走 reference_image_urls）⇒ 即便分镜存了 `first_frame_image`，`imageUrl`/`firstFrameUrl` 也是 null（要验帧统一得用**非对话**场景落 single）。
- **⚠️ 对话判据有两份、宽窄不同**：`shot_router.DIALOGUE_PATTERN`（含 `多人/对话/争吵/会议`）vs `videos.ts` 那份（只有英文 5 词）—— 分别管「路由是否 R2V」与「是否自动带参考音频」，**都别改**。
- **`videos.ts` 保真点**：`GET /{id}` 查不到 → `success(null)` 不是 404；`PUT` 白名单不含 `updatedAt`；`GET /` 过滤值非有限数字 → 当没传。
- **⚠️ 通用坑：别嵌套事务**。SQLite 单写者 ⇒ 在 `engine.begin()` 里再开 `engine.begin()` 直接 `database is locked`（写 `prompt_storyboard_test.py` 时踩到）。生产代码一律**每步一个短事务、不嵌套**。
- **⚠️ `to_public_media_url` 默认基址是 Node 的 5789**（参考音频 URL）：只跑 Python（5790）时**必须设 `PUBLIC_BASE_URL`**，否则本地 H3 拉音频 404。
- **⏳ S4 剩余未迁（各有明确理由）**：镜头 QC 打分（`qc-scoring`+`technical-qc` ≈510 行，ffmpeg 技术质检；调用点 `_run_qc_after_video_complete` 已就位、当前记 `qc-skipped`）｜像素处理（校色 + 参考图压缩，需 Pillow，且输出不会与 sharp 逐字节一致 ⇒ 要有对照验证）｜compose / 宫格 / 合并｜`DEFAULT_PROMPTS` / `/gpu/*`（GPU 租约，S7 前须补）。
- **测试写法小坑**：`check(name, cond, detail)` 传 **list/可变对象**时，后续 `clear()/append()` 会让**打印出来的 detail 与断言当时不一致**（误导排查）⇒ 传副本 `list(...)`。
- **图片链路的「顺序即语义」三点**：① 角色回写优先级 `itemType→viewType→expression→equipType→costume→主图`；② 帧类型→列映射 + **只有首尾帧**置 `asset_status=approved`；③ 恢复分支顺序 **先判过期(60min) 再判 taskId/provider**（测试造数据时踩过：给了昨天的 `updated_at`，于是先命中过期分支）。
- **fire-and-forget 的后台任务**：`asyncio.create_task` **必须持引用**（模块级 set + done callback），否则可能被 GC，表现为「永远卡在 processing」。且**每步各开短事务**，不能全程持有一个（轮询 10 分钟会阻塞 Node 写库）。
- **跨语言比对字面量的两个坑**（都踩过）：① TS 源码里的 `\n` 是**字面两字符**，Python 运行时是真换行 ⇒ 归一化要一起抹掉转义序列，否则假漂移；② TS 的隐式拼接 `'a'+'b'` 要先把 `' + '` 粘掉再比。
- **Windows 编码第 4 次**：subprocess 抓子进程输出时，子进程 stdout 是**管道** ⇒ 按 GBK 输出，若按 UTF-8 解码得 U+FFFD，再打印回 GBK 控制台直接崩。对策：子进程传 `PYTHONIOENCODING=utf-8` + 自身 `sys.stdout.reconfigure(errors="replace")`。
- **⚠️ 两类"信封例外"**（别套用统一响应层）：`preset-framework` 用 `{success, data/msg}`；`export`/`edl`/ZIP 等**返回文件本体**（裸 `Response` + `Content-Disposition`，无信封）。前端把它们当下载链接。
- **⚠️ Windows 工具链一律只给 ASCII**（同一根因已咬三次）：① 启动日志中文在 GBK 控制台乱码；② 用例名里的 `⊆/∪/∈` 触发 `UnicodeEncodeError` 打断整个测试；③ `.ps1` 探针里的中文被 PowerShell 5.1 按 **ANSI 代码页**读（文件是无 BOM UTF-8）⇒ 乱码吃掉引号、报 `Missing closing ')'`。
- **⚠️ 断言"按时间排序"必须先把时间戳写死**（用 `os.utime`）：连续写文件的 mtime 可能相同，排序断言会偶发红（已踩，`traces list` 用例）。
- **⭐ 镜像常量的原则**：`agent_registry.py`（5 阶段名 / 6 Agent 显示名 / 22 宿主工具名）是 TS 常量的**第二份副本**（Python 无法执行 Mastra 工厂）⇒ **镜像静态度量可以，但必须配自动漂移校验**（`route_parity_test.py` 从 TS 源码抽取比对，现 0 漂移）。**提示词正文（`DEFAULT_PROMPTS`）坚决不镜像** —— 那会重建本项目已修掉的「提示词多头维护」问题，`/agent-configs/defaults` 因此**有理由地保持委派**。
- **⚠️ 操作安全**：`ai-configs/quick-preset` / `quick-local` 是 **upsert，会覆盖真实库既有配置** ⇒ 真机验证只跑只读接口；冒烟测试跑在数据库副本上所以安全。`ai_service_providers` **不由 Python seed**（Node 幂等 seed + 该表无唯一约束，重复 seed 会产生重复行），等 Node 下线再搬。
- **刻意不迁的端点（各有理由，别当成遗漏）**：`POST /storage/change`（写项目级 `.data-root` 而 Node 也读它 ⇒ 跨进程副作用）｜`GET /usage/estimate`（依赖 `estimate-service` + `cost-catalog` 单价目录）｜trace **写入侧** `appendTraceEvent`（调用方是 Agent 链路）｜`GET /agent-configs/defaults` + `POST /agent-configs/generate` + `POST /style-profiles/:id/distill`（提示词/skills 域与 LLM）。注意 `/style-profiles/:id/apply` **已迁** ⇒ 提炼结果可跨后端落库。
- **⚠️ `GET /traces/stats` 在真实数据上恒为 `runs=0` —— 预期行为不是 bug**：真实 trace 的 token 值被**脱敏成字符串 `"***"`**（20 个 trace 实测全如此），`Number("***")` 是 `NaN`、`NaN‖NaN‖NaN` 为假 ⇒ 整条跳过；**Node 结果相同**。已用直造 trace 的用例锁住。**教训：返回 0 时先确认「实现漏了」还是「数据本来如此」。**
- **JS 语义收口清单（4 条，都在 `response.py`）**：`Math.round` ≠ Python `round`（银行家舍入）⇒ `js_round`｜`Number()` ⇒ `js_number`（非有限值→None，**不校验 `>0`**，与 `parse_param_id` 不同）｜`a ?? b` 只看 null ⇒ `js_nullish`（与 `||` 语义不同）｜**`[]`/`{}` 在 JS 是真值** ⇒ `js_truthy`（传空对象/空数组要落 `"{}"`/`"[]"` 而非 null）。
- **⭐ 两道必跑自检**（每迁完一个域）：`tests/smoke_test.py`（契约，现 330 用例）+ `tests/route_parity_test.py`（**路径守卫：0 遮蔽才可继续**）。后者专防「新注册的参数路由吞掉未迁移的 Node 端点」这类**静默失效** —— 判据是行为（未注册路径必须落到兜底 501），不是内省（`include_router` 会包 `_IncludedRouter`，拿不到 `.path`）。参数名也要归一化（`{id}` vs `{drama_id}`），否则全是假阳性。
- **踩过的真 bug（引以为戒）**：`models._bool` 曾用 `Integer` ⇒ **全站布尔列返回 `1`/`0` 而非 `true`/`false`**（drizzle `mode:'boolean'` 在 JS 侧是真布尔，前端 `=== true` 会挂），改 `Boolean` 修掉全部域。⇒ 这类偏差不报错、只「功能莫名不对」，必须靠真跑 + 断言类型。
- **⭐ 最深的一课：测试锁住了我自己的 bug**。曾按「`JSON.stringify` 丢 undefined 键」推断 `success(c, undefined)` 会省略 `data` 键，造了 `success_without_data()` 并写断言 `"data" not in json`。实际 TS 是 `success(c, data = null)`，**JS 默认参数会把显式 `undefined` 换成 `null`** ⇒ 契约一直是 `{"code":200,"data":null,...}`。**写断言必须对着 Node 源码/真实响应，不能对着自己的实现** —— 否则 bug 会被测试固化。**别按 route def 数估工期**：`characters.ts` 18 个里只有 4 个可迁、`storyboards.ts` 14 个里只有 5 个可迁（其余是出图/出音/抽帧/LLM）。
- **JS 语义对齐清单（迁移时逐条对照，均已实测踩过）**：① `??` ≠ `||`（空串不回退）；② `Number(x)`：`null→0`、`''→0`、`'abc'→NaN`；③ SQLAlchemy `**kwargs` ≠ JS 对象展开（同名键 TypeError）；④ 入参的 drizzle 属性名 ≠ DB 列名（`Unconsumed column names`）；⑤ JS `$`（无 `m`）只匹配串尾 ⇒ Python 用 `\Z`；⑥ JS `replace` 无 `g` 只替换首处 ⇒ `count=1`；⑦ JSON 对象数字键：JS 会强转字符串 ⇒ Python 必须把键转回 int；⑧ `JSON.stringify` 丢掉值为 `undefined` 的键 ⇒ 有「无 data 键」的成功响应（`success_without_data()`）。
- **⭐ 项目里存在两套响应信封（别以为全项目统一）**：① 主流 `{code:200, data, message}`、错误无 `data` 键、错误用真 HTTP 状态码；② **仅资源库四域**（character/scene/weapon/costume-library）用 `{code:0, data}`（列表/详情**无 message 键**）+ 错误 `{code:400/404/500, data:null, message}` 但 **HTTP 恒 200**。前端 `useApi.ts` 的 `!resp.ok || (json.code && json.code >= 400)` 同时兼容两者。资源库还走**原生 SQL**（列名=DB 列名）⇒ 返回 snake_case + 少量 camelCase 别名；id 用 **`parseInt` 语义**（`/12abc` → 12）。
- **⚠️ SQLAlchemy 原生 SQL 两个独立坑（症状相同）**：`ArgumentError: List argument must consist only of dictionaries` 有两个成因 —— ① **`text()` 不支持 `?` 占位符**（只认 `:named`）⇒ 必须用 `exec_driver_sql`；② 即使 `exec_driver_sql`，**位置参数也要传 tuple**（list = executemany 多组参数）。**先入为主只改 ① 会以为修好了但仍然报同一个错** ⇒ 用最小复现（直接调 `conn.execute`）隔离验证。已收口为 `resource_library.q_all/q_one/q_run`，本仓所有原生 SQL 必须走它们。
- **真缺陷（照抄未修，两边要一起改）**：`app-settings.ts` 的 `ART_STYLE_KEYS` **只有 6 种画风**，画风体系实为 **10 种** ⇒ `noir`/`ink-wash`/`cyberpunk`/`pixar3d` 设全局默认会被 400 拒。权威表在 `prompt-utils.ts` 的 `ART_STYLE_CATALOG` + 前端 `artStyles.ts`；`app-settings.ts` 是**过期的第三份副本**。Python 侧照抄 + 用例锁当前行为（修好时会红 = 预期信号）。
- **继承的孤儿行为（知情不改）**：`DELETE /storyboards/:id` 只清 `storyboard_characters`，**不清 `storyboard_props`**（原 TS 即如此，会留孤儿行）。
- **资源库「兜底枚举」几乎不生效（继承行为）**：兜底只在「该表一条非空值都没有」时触发，而创建未传的字段被写成 `''`，`IS NOT NULL` 对 `''` 成立 ⇒ 有数据时返回实际值（可能是 `[""]`）。验证兜底分支必须先把表清空。
- **⭐ 返回形状按域不同（最易踩的契约差异）**：Node 里**显式调过 `toSnakeCase` 的域返回 snake_case**（dramas / episodes 的列表与详情），**直接返回 drizzle 行的域返回 camelCase**（characters / scenes / props —— drizzle 返回的是 TS 声明的属性名）。前端确实按这个差异写代码：`CharacterEditor.vue` 直接读 `c.voiceStyle`，而 episode 工作台写的是 `c.voice_style || c.voiceStyle`（**防御式双读**）。Python 侧用 `response.row_to_camel(row, 表名)` / `dict_to_camel`，**不要「顺手统一」**。
- **列名 ↔ JS 属性名双向都要转**：① 返回方向：`prop_templates` 是**全库唯一**「属性名 ≠ camelCase(列名)」的字段（TS 写 `customPrompt: text('image_prompt')`）—— `tmp/schema-pairs.mjs` 扫 **493 个字段实测仅此 1 处**，靠 `_COLUMN_TO_JS_OVERRIDES` 处理，不然会返回 `imagePrompt`；② 入参方向：TS 的 `updates[key] = body[key]` 里 key 是 drizzle **属性名**，Python Core 只认 **DB 列名** ⇒ 双写兼容分支必须写 `updates[snake] = body[camelKey]`，照抄会报 `Unconsumed column names: voiceStyle`（**已实测踩到，且被"先测 snake 入参"掩盖过一轮**）。
- **id 解析两套并存（勿合并）**：`props` 用自定义 `parseId`（`Number.isInteger && > 0`，**严格**）；`characters`/`scenes`/`dramas`/`episodes` 用 `parseParamId`（`Number.isFinite && > 0`，**放行小数**）⇒ `GET /scenes/1.5` 会走到查库才失配，报 `Scene not found` 而非 `Invalid scene id`。`response.parse_param_id` 已改为镜像 JS `Number()`（原 `int()` 实现拒小数，属保真度修正）。
- **分支约定**：Node/TS 侧修复提交到 **`main`**；Python 迁移工作在 **`feat/python-backend`**。`backend/probe-*.ts` 是临时探针（头部自述「用完即删」）⇒ **不入提交**。
- **JS 语义坑（第二次踩，务必对照）**：① `??`（空值合并）**不等于** `||` —— `ep.scriptContent ?? ep.content` 在 `script_content=''` 时**不回退**，写成 Python `a or b` 会让门禁/取值与 Node 分叉；同一份代码里两种都有（`era-background` 用 `||`、`script-fingerprint` 用 `??`），**必须逐个确认**。② `GET /episodes/:id/pipeline-status` 的「有图」判据是 `composed_image` **单字段**，而 `GET /dramas` 的 `progress` 是 `composed_image || first_frame_image` —— 原 TS 就是两套，**照抄勿合并**（已各写各的）。③ 数值字段的 `Number(x)` 语义：`Number(null)=0`、`Number('')=0`、`Number('abc')=NaN` ⇒ Python 侧实现 `_js_number`，仅 `NaN` 分支返回 `None`（写 NULL 而非 NaN，属有意偏差且该列本可空）。
- **跨域复用**：读请求体统一走 `app/request_utils.py` 的 `read_json`（对齐 Hono `c.req.json()` 的容错：坏 JSON → 空 dict → 业务校验给 400，而非 500）。
- **下一个域的 SOP**：见 `backend-py/README.md`。要点：静态子路径必须声明在 `/{id}` 之前；**只注册已实现的**；依赖 LLM/子进程/长任务的服务**放到最后**（其行为只能真调验证）。
- **工具坑**：`tempfile.mkdtemp()` 在 Windows 返回 **8.3 短路径**，与 `resolve()` 后的长路径做子串比较会**假红** ⇒ 比较 resolve 后路径；`sqlalchemy` 顶层导出的是 **`REAL` 而非 `Real`**；PowerShell 组合命令含 `&` / `@{...}` 会被路由到 cmd（报「不是内部或外部命令」）⇒ **写 `.ps1` 落盘再执行**；`Start-Process` 的相对 `-WorkingDirectory` + 相对 exe 路径会报「系统找不到指定的路径」⇒ 用字面绝对路径。⚠️ 另：`execute_command` 报「已转后台/skipped」时**命令常常真的在跑**，输出落在重定向文件里 ⇒ **别急着重跑，先 `Get-Content` 那个文件**（本项目多次据此拿到完整结论）。⚠️ 再：PowerShell 的 `>` 重定向写 **UTF-16**（带 BOM），后续用 Python `read_text(encoding='utf-8')` 汇总会**静默得到 0 条**（正则全不匹配）⇒ 汇总脚本按 `utf-16` 解码，且汇总结果要先与「行数」对一眼再当结论。
- **`from m import x` 是对象快照，不是活引用**：本项目 `db.reopen_engine()` 换的是**模块属性** `app.db.engine`，而测试里 `from app.db import engine` 拿到的仍是旧对象 ⇒ 断言「engine 已指向新库」永远看到旧库（还会让「新库有表」**假通过**）。**要么 `import app.db as db` 后取 `db.engine`，要么断言前重新导入**。
- **「原本只会返回 501」的断言，在端点迁移后会真的执行** ⇒ 必须重新审入参副作用。本项目实测：smoke 里 `POST /storage/change {"path": "x"}`（相对路径）在迁移后真的切了数据根并写了仓库根 `.data-root`。规则：**能改全局状态的端点，测试一律用「必然被拒」的入参**（如项目根/空串），并额外断言「状态未变」。
- ⚠️ **「HTTP 状态码」不是端点在不在的可靠判据 —— 只要本机存在前端产物**：Node `src/index.ts` 有一道**无条件**兜底 `app.get('*', serveStatic({root: 'frontend/dist', path: 'index.html'}))` ⇒ 任何**未匹配的 GET** 都会回 **200 + HTML**（谁跑过一次 `npm run generate` 就有 `.output/public/index.html`，而 `frontend/dist` 常是指向它的**联接**）。本项目因此**连栽两次**：① 冒烟两条「dist 不存在 ⇒ 404 / 穿越被拦」；② 对拍里 4 条裸列表路径被判成「Node 有而 Python 没有」的 `new`。
  ⇒ 规则：写测试/工具时**先问「有前端产物时会怎样」**；判定「端点是否存在」要看**响应形态**（非 JSON / HTML ⇒ 是兜底）而不是只看状态码；能在夹具里打桩的（`FRONTEND_DIST`）就打桩。
- ⚠️ **自检用例的「分母」要手数准**：`parity_diff.py --selftest` 原写 `len(cases) + 5`，实际有 6 处额外断言 ⇒ 打印的分母偏小（不影响失败判定与退出码，但会让人误信覆盖度）。加断言时同步改那个常量（现在叫 `EXTRA_CHECKS`）。
- ⚠️ **「自动发现」的覆盖面才是命门**：快照的自动发现只看 `_GUARD_SOURCES` 里那几个文件 ⇒ **自检（`smoke_test.py`）里写死的 TS 路径会漏网**，直到删库当天才以「导入期 FileNotFoundError、整套崩」现形（2026-09-15 实测）。规则：**任何读 TS 源码的测试文件都要进 `_GUARD_SOURCES`**；删库类操作要按「谁还读真源码」而不是「谁是守卫」来清点。
- ⚠️ **删库当天 `HEAD` 往往已不含被删内容**（现场被提交成「删除」）⇒ `git show HEAD:<path>` 直接 fatal。取回旧内容的正确姿势：`git log --all --format=%H -- <path>` 后**逐个提交回溯到第一个存在的版本**。且**「现有快照/现场优先于 git」**：git HEAD 可能落后于删除时的现场（本项目险些把一次未提交的改动**静默回退**）。
- **「去 Node 化 / 删旧后端」要分两类引用**：**历史溯源**（`移植自 backend/src/xxx.ts`）**保留**（那是价值所在）；**可执行指令**（端口、启动命令、env 开关、Docker/nginx、部署清单）**必须改** —— 否则文档会教人去起一个已删除的服务（本项目实测：README 里 5789/npm start、`PROXY_TO_NODE=1 可反代`、501 的**运行时文案**全在教人起 Node ✗）。顺带：**改运行时文案前先确认没有测试断言那段文本**。
- **批量改文档用「落盘脚本 + MISS 报告」**，别在内联 `python -c` 里拼引号：PowerShell 会把 `\"`、`\\` 吃掉或报 `unterminated string literal`（本项目内联版直接失败过一次）。落盘脚本还能逐条打印 `OK/MISS`，避免「静默改错」。⚠️ 同理：命令里带**工作区之外**的路径（哪怕只作环境变量值）会触发权限拦截并超时 ⇒ 探针路径用工作区内目录。
- **「优化目录结构」先分清两层：代码层 vs 资产层**（本项目：`app/`＝代码；`skills/`、`local_services/`、`configs/`、`data/`＝内容/运行时）。迁移期为「逐条比对」而刻意**扁平/镜像**的目录（`app/services/*.py` ↔ TS `services/*.ts`）**不能重排** —— 守卫按映射表比常量，重排会成片打断 ⇒ 优化只该落在「常量收口（同一规则只留一处）+ 文档边界说明 + 清过期标注」。
- **JS → Python 逐条移植的六个口径坑**（2026-09-15 把 7 个 `.mjs` 守卫/语料脚本移植成 `.py` 时全部实测踩出；**任何 JS 行为对齐都先想这六条**）：
  1. **换行转换**：Python `Path.read_text()` 默认做 universal newlines（CRLF→LF），Node `readFileSync` **不做** ⇒ 字符数会少「行数」那么多（实测 7651 → 7598）。要逐字对齐必须 `open(..., newline="")`。
  2. **`\w` 语义**：JS 的 `\w` 只认 `[A-Za-z0-9_]`，Python 默认认 Unicode 词字符 ⇒ 正则补 `re.ASCII`。
  3. **数字转字符串**：JS `String(1.0)` → `"1"`，Python `str(1.0)` → `"1.0"`（语料 8987 行里 **206 行**受影响）⇒ 写 `_js_num_str()`（整数值浮点去 `.0`）。
  4. **取整**：`Math.round` 是 **half-up**，Python `round()` 是**银行家舍入** ⇒ 用 `math.floor(x + 0.5)`。
  5. **`toFixed`**：同为 half-up ⇒ 统一 `Decimal(repr(x)).quantize(..., ROUND_HALF_UP)`（本仓库已被 `round()`/`:.0f` 坑过三次）。
  6. **字符串长度/切片**：JS 数的是 **UTF-16 码元** ⇒ `len(s.encode("utf-16-le")) // 2`；按 JS 语义截断要 `_js_slice()`。
  - 落地手法：**移植期间不要删 Node 原版**，用「同一输入跑两边、逐字节 diff」验收（本项目 5/6 个脚本做到 IDENTICAL，剩一个因故意改成确定性排序而用**多重集**比对）。
- **搬目录的静默杀手（本仓库工具链）**：`scripts/` 下的脚本用 `dirname(SCRIPT_DIR)` 当仓库根 ⇒ 搬进 `backend-py/app/scripts/` 后**必须上跳两级**（`parents[2]`→`parents[3]` 同理）。只跳一级**不报错**，只是 `configs/models.json` 读不到 ⇒ 清单为空。
- ⚠️ **Windows 目录联接会让 `.resolve()` 骗人**：本仓库 `frontend/dist` 是指向 `frontend/.output/public` 的联接 ⇒ 校验「某常量是否指向某路径」时用 `.resolve()` 比对会**跳到联接目标**而假红；比**原始路径**（`os.path.normcase` 归一大小写）。同理「不得出现旧写法」这类文本校验**要排除注释行**（注释里常有意保留旧写法做对照）。
- **`.mjs` / `.ps1` 已不存在**：仓库自检（`check_memory.py` / `check_skill_refs.py` / `test_guards.py` / `check_all.py`）与语料管线/工具链**全在 `backend-py/scripts/`（纯 Python）**；TS 源码快照是 **Python 模块** `backend-py/tests/frozen_ts_source.py`（守卫侧物化到临时目录）。`.githooks/pre-commit` 的解释器解析顺序：venv → `python3` → `python` → `py -3`。
- **后端 9 项能力（Python 侧全在 `backend-py/app/services/`）**：QC（technical/consistency）｜asset-versions｜style-profiles｜script-fingerprint｜take-budget（默认 3 次）｜rhythm-phase｜jianying-draft｜estimate-service｜usage-tracking（成本/用量记录）。
- **共享契约类型 `frontend/app/types/contracts.ts`**（2026-09-15 从 `backend/src/shared/contracts.ts` 搬来）：**前端侧镜像**，字段**权威在 Python 后端**；改后端字段必须同步它。前端 dev 代理默认指 **5790**（`NUXT_API_TARGET` 可临时指回 5789）。
- **仓库根已无 `scripts/`、`skills/`**（2026-09-15 双双并入 `backend-py/`）：
  - **技能库根 = `backend-py/app/skills/`**，是**三处必须同步的不变量**：`app/services/skills.py` 的 `SKILLS_DIR`（`parents[2]`）｜`app/services/agents/skills.py` 的 `skills_dir()`（同文件相对推导，**别拼 `PROJECT_ROOT/"backend-py"/...`**）｜`backend-py/app/scripts/check_skill_refs.py` 的 `SKILLS_DIR`。
  - 引用守卫对旧写法 `skills/…` **判致命**（`_LEGACY_SKILLS_PREFIX`）；skill 正文里的仓库根相对引用请写 `backend-py/app/skills/…`。
  - ⚠️ `backend/src/**`（Node 侧 `../../../skills`）**已失效** —— Node 仅剩被删的命，删库前若要启动它会退化。
- **批量改路径的正确手法**：**单趟正则 + 负向断言**（如 `(?<![\w./~-])skills/`）一次替换完，既防二次命中已改好的前缀（双前缀 bug 的根治），又不误伤 `~/…/skills/` 这类外部路径；**改完必须跑行为测试**（守卫绿 ≠ 行为绿）。
- **「删目录/删服务」的前置验证：真删一次（可回滚干跑）** —— `重命名 → 跑全部守门脚本与自检 → finally 还原`。本项目实测：一轮干跑抓出三类**静态扫描根本看不到**的问题（① 文档里指向被删目录的引用 ② 守卫对旧前缀不设防 ③ 脚本里「真源码在才走某分支」的早期 return 在删后静默失效）。**只跑「代码常量排查 + 全量自检」是查不出来的**，因为它们只在「目录真的不在」时才走到那条分支。
- ⚠️ **引用守卫只采集反引号 token 与 markdown 链接**（`` `path/x.md` ``）：写在注释/正文里的裸路径**不受保护**，挪库后不会报错 ⇒ 想让某条引用受守卫保护，**先给它加反引号**。
- **删 `backend/` 的三条前置（2026-09-15 起）**：① **差分对拍 0 新差异** —— `tests/parity_run.py`（两侧并排 + 逐字段 diff）；**只比只读 GET**，`parity_diff.py` 的 `CASES` 表要随迁移补齐（2026-09-15 实测 **一致 12 / 已知 0 / 不存在 4 / 新差异 0**；文本响应靠 `strip_timestamps()` 抹掉 ISO 时间戳才能参与比对，HTML/Markdown 导出物**故意不入表**）；② **守卫快照完整** —— `tests/freeze_ts_snapshot.py`（清单 = 手写 ∪ **从守卫源码自动发现**两种形态：`_SRC_ROOT / "…" / "x.ts"` 拼接 **与** 表驱动普通字符串路径，防「新增守卫读文件 ⇒ 快照静默缺件」；快照 2026-09-15 起存为 **Python 模块** `tests/frozen_ts_source.py`（74 条 / 625 KB，仓库内**已无 .ts**），守卫侧 `frozen_ts.snapshot_root()` **物化到临时目录**后再照常按 `Path` 读 ⇒ 调用点零改动；真源码/冻结两路结论必须一致）；③ **`app/` 对 `backend/` 的路径依赖 = 0**（只剩守卫读 TS 源码，已由 ② 兜住）。

## 自 MEMORY.md 下移（2026-09-24 第四次腾预算）

起因：`MEMORY.md` **13906 字符** ✗（预算 8000 ✓，`app/scripts/check_memory.py` 判**致命** ✓）——
它的头写明「超限注入会被截断，**尾部 `## 协作与提交` 最先丢**」✗✗ ⇒ 实际已在丢规则 ✓。做法同前三次 ✓：
细节留本节 ✓，`MEMORY.md` 只留**会导致 bug 的判据** + 指针 ✓。⚠️ 判据：`(Get-Content .codebuddy\memory\MEMORY.md -Raw).Length` ≤ 8000 ✓。

### A. 自研引擎 · 分词器细节
`engine/tokenizer_bpe.py`（自研字节级 BPE ✓）/ `engine/tokenizer_own.py`（Unigram/WordPiece/Metaspace ✓
—— Viterbi ✓ / 整词 UNK ✓ / **normalizer 11 种** ✓ / **预分词器 10 种** ✓ 含 `Punctuation` 与 `Split`
各五种 behavior ✓、`FixedLength` ✓；规则**逐例实测**对齐参考 ✓）/ `engine/tokenizer_hub.py`（形态嗅探 ✓ →
自研优先 ✓ → 极少数形态回退参考 ✓ + 批量 + LRU + 运行期互校 ✓；`tokenizers_tuning.py` = 对 TF 的**运行时
改造** ✓：离线兜底含 **Auto 工厂**（它会在解析出具体类**之前**就外呼 ⇒ 必须单独包 ✗）/ 缓存目录 / 降噪 /
计数 / 可撤 ✓）。
* `byte_fallback` 的 Unigram **已实现** ✓（**段级判据** ✓：段内**每个**字符都能展开才逐字符展开 ✓，
  否则**整段**一个 `unk` ✓；缺任一字节 ⇒ 整字符回退不了 ✓；字节名只认 `<0x` + 两位十六进制 ✓）；
* ⚠️ 明确**拒绝**且**带理由**（宁可回退参考实现 ✓）：`Precompiled` ✓（要 SentencePiece charsmap 表 ✓）、
  `UnicodeScripts` ✓（标准库无 script 表 ✓）、**别的模型**的 `byte_fallback` ✓（BPE 的**输出顺序不是
  位置语义** ✗✗ —— 实测 `'be'`/`'eb'` 输出**完全相同** ✓ ⇒ 位置实现必错其一 ✓；拒绝理由**必须带这对证据** ✓）、
  **能匹配空串**的正则 ✓ 与 `\p{…}` 语法 ✓（`re` 没有 ✓）；
* ⭐ **性能**（2026-09-21 ✓）：`UnigramTokenizer` Viterbi 由「每 (end,start) 重扫前缀树」✗ 改成
  「**每起点只扫一次** + 前向 DP」✓ ⇒ 冷 60k → **218k tokens/s** ✓；再加**片段级缓存** ✓
  （热 3.5M ✓ 模块级 `PIECE_CACHE_LIMIT` ✓ 满了整体清空 ✓）；基准 `app/scripts/tokenizer_bench.py` ✓。
* ⚠️ 性能数字**必须把口径一起报** ✗（只报"热"数字 ⇒ 会得出"比 Rust 快 5.5 倍"的假象 ✓✗）。

### B. 自研引擎 · 权重装载与跨来源校验
* 低精度（`engine/quant.py` ✓）：fp8/int8 先按配套 scale **反量化** ✓（布局**按形状**判 ✓ 四种 ✓、
  `*_scale_inv` 走除 ✓）⇒ 再归一 dtype ✓；**判不出来就拒绝** ✗（块/分组量化、缺 scale、scale=0 ✓）；
  ⚠️ 顺序：**先反量化、再 cast** ✗；失败 ⇒ 中止装载 + **不污染模块** ✓。结论已推到**三个消费者** ✓
  （CLI ② 段 / 就绪 API `summary().quant` / 前端面板 ✓）且**三档分开说** ✗：可自动还原 ✓ / 判不出
  （**同时进 blockers** ✓）/ **没查** ✓（组件没下载 ⇒ 不阻塞也不假装绿 ✓）；⚠️ **GGUF 豁免** ✗。
* ⭐⭐ **已挂 VAE 的跨来源校验**（`TorchBackend._check_attached_vaes` ✓ `describe().vaeCheck` ✓）——
  四/五个字段 ✓：① 主干推的 `latents_dim` ↔ VAE `latent_channels` ✓（给错的 `patch_size`
  **能整除**时形状全自洽 ⇒ 装得进去 ✓✗、只有画面不对 ⇒ 现在**挂载期**就拒 ✓）；
  ② VAE `spatial_scale` ↔ **H3 事实 `vaeScale=16`** ✓；③ 音频声道 ↔ `geometry.AUDIO_LATENT_CHANNELS` ✓；
  ④ 音频帧率 ↔ `geometry.AUDIO_LATENT_HZ=40` ✓（不等 ⇒ **wav 时长错** ✓✗ 且不报错 ✗✗）；
  ⚠️ DiT 形态只核 `spatial_scale ↔ config.vae_scale` ✓（别做过头 ✗）；挂载**两向**都守 ✓；
  `load_weights` 校验失败 ⇒ **回滚 `_model/_config/_form`** ✓。
* ⭐ 同样守 **TE `output_dim` ↔ 主干 `text_dim`** ✓（`_check_text_encoder` ✓ / `describe().textEncoderCheck` ✓，
  DiT / H3 两侧都守 ✓）；⚠️ 挂载守卫一律**先核后改** ✗（核不过是**候选**的问题 ⇒ 失败时模块一个字没动 ✓
  —— 反例：先挂上再回滚会把**先前挂好的**那个也清掉 ✗✗）。
* ⚠️ 以上都是**两个来源互证** ✗ 不是权重事实 ✗ ⇒ 真权重到手仍要按元数据核 ✓。
* H3 结构**从权重推** ✓（`h3_keys.infer_h3_trunk_config` ✓ ⇒ 出厂常量只作回落 ✓）。

### C. 前端类型检查（2026-09-21 起）
* 三件事都要跑 ✗：`npm run typecheck`（`vue-tsc` ✓ + `@types/node` ✓ 装在 devDependencies ✓ 走
  `registry.npmmirror.com` ✓）、`npm run build`（**原先不做**类型检查 ✗ ⇒ 现已开 `typescript.typeCheck` ✓）、
  ⭐ `npm run **generate**` ✓ —— **前端产物是 generate 出的** ✗（Dockerfile 就是它 ✓；`frontend/dist`
  是指向 `.output/public` 的**目录联接** ✓）⇒ 只跑 `build` 会把 `.output/public/index.html`
  **覆盖没** ✓✗（`dockerfile_contract_test` 当场红 ✓）。
* ⚠️⚠️ `ref([])` 推出 **`never[]`** ✗ ⇒ 元素属性访问全报 TS2339 ✓✗（一趟几百条 ✓）⇒ `ref` 一律带泛型 ✓；
  字面量配置表用运行时键要 `Record<string, T>` ✓；`catch (e)` 的 `e` 是 `unknown` ✓ 不许直接 `e.message` ✓；
  模板读表单值走 `formValue($event)` ✓（`$event.target.value` 报 `EventTarget` 无 `value` ✓）；
  ⚠️ **`ComputedRef` 必须 `.value`** ✗（`if (!ref)` 恒真 ✓、当参数传会拼出 `[object Object]` ✓✗
  —— **只有类型检查能抓住** ✓，已因此逮到 `switchEpisodeConfig` 一个「从来没生效」的 bug ✓）；
  ⚠️ `desc = []` / `refs = []` 这类**默认空数组**会把参数推成 `never[]` ✗ ⇒ 必须显式类型 ✓。
* ⚠️ **注释里别写调用形状** ✗：`frontend_api_coverage_test` 纯文本扫 `api.get('…')` ✓ ⇒
  注释里的一句"曾经写错成 …"会被**当成真调用点** ✓✗（2026-09-22 实测 ✓）。
* ⭐ **2026-09-22：typecheck 清零** ✓（537 → 314 → 134 → **0** ✓；三轮修掉两个真 bug ✓）。

### D. 上机前自检（引擎就绪）
业务在**服务层** `app/services/engine_readiness.py` ✓（⚠️ **不能放 `app/scripts/`** ✗ —— 那个目录
**不是包** ✓ ⇒ 路由引用不到 ✓✗）；CLI 薄壳 `app/scripts/h3_readiness.py` ✓（`--json` ✓ + **退出码 = ready** ✓）；
路由 `GET /api/v1/production/engine-readiness` ✓（回**精简摘要** ✓ —— 别塞整坨排班 ✗；真权重预检要文件路径
⇒ **不走 HTTP** ✗ 留 CLI ✓）；前端 `PreflightPanel.vue` 有**独立**「引擎就绪」块 ✓（没选集也能看 ✓，
「没查的项」用**灰点**与通过区分 ✓）。
⚠️ 口径：没给权重 ⇒ 明说「**没查**」✗ 不算通过 ✓；「还要下多少」按清单 `expectedGiB` 求和 ✗
（用 `bytes` 会恒为 0 ✓✗）；⚠️ 与 `POST /production/preflight`（查**内容**）是两个独立 `ready` ✗。

### E. 记忆层自身的守卫（⚠️ 一直在烂 ✗）
`app/scripts/check_memory.py` ✓ 把这几条判**致命** ✓：`MEMORY.md` 超 8k ✓、`@N` 越界或**未落在小节首行** ✓、
**某篇日志的末节没有登记锚点** ✓、磁盘上的日志未登记进 INDEX ✓、落点表里 `文件 §小节` 的小节名查不到 ✓。
⚠️ **它不在 `run_all.py` 里** ✗ ⇒ 会静默腐烂 ✓✗（2026-09-24 实测：**7 处致命** ✗，其中 5 篇日志的**末节锚点**
早漂了 ✓ —— ⚠️ **末节/步骤小节锚点会随日志增长而漂移** ✗，写 `@末节` 这种字面量**不算登记** ✓✗）。
⇒ 每次追加日志后跑一次 ✓：`.venv/Scripts/python.exe app/scripts/check_memory.py` ✓（exit 0 才算过 ✓）。

### F. 环境与自检纪律（2026-09-24 下移）
* ⚠️⚠️ **跑全量前：把 ffmpeg 真实 bin 目录「前置」到 `PATH`** ✗（2026-09-22 实测 ✓✗）：Windows 上 WinGet 的
  `…\WinGet\Links\ffmpeg.exe` 是**应用别名（重解析点）** ⇒ 本进程里 `lexists=True` 但 `exists=False` ✓✗
  ⇒ ① `shutil.which("ffmpeg")` → `None` ✓（引擎会说「找不到 ffmpeg」✗ —— 其实装了 ✓）；② **裸名** spawn
  （`subprocess.run(["ffmpeg", …])` ✓ 产品代码那种写法 ✓）⇒ `OSError: [WinError 448] 无法遍历该路径，
  因为它包含不受信任的装入点` ✗✗ ⇒ `image_generation` / `consistency_qc` / `technical_qc` / `color_grade` /
  `compressed_data_url` **5 套假红** ✓（看着像代码坏了 ✗✗）。⚠️ **追加 PATH 无效** ✗（只修好 ① ✗）；
  判据 `where.exe ffmpeg` 指到真实 bin ✓。已写进 `tests/run_all.py` 表头 ✓。
* ⚠️ **`transformers` 是可选的** ✓（已装 5.17.0 ✓ 镜像 ✓，登记为 `_Dependency.optional` ✓）：
  `dependency_status()` 分 `missing`（必需 ✓ 缺则后端不可用）/ `optionalMissing`（可选 ✓）
  ⇒ **别把可选塞进必需位** ✗（`ready` / `torch_available()` 会被判死 ✗✗）。⚠️ **不 fork / 不 vendored** ✗：
  只做**运行时改造** ✓；⚠️ **Auto 工厂必须单独包** ✗（它在解析出具体类**之前**就外呼 ✓✗）；
  分词本身已由**自研三血统**接管 ✓ ⇒ 它只剩**核对价值** ✓。
* ⚠️ **自检汇总行必须写 `SUMMARY: n/m passed`** ✗（`run_all.py` 按此前缀收敛项数 ✓；写成「n/m 项通过」
  ⇒ 总表**空摘要**、项数缺一套 ✓✗）。
* ⚠️ **别同时开多个全量回归** ✗（并发抢 CPU ⇒ 像"卡死" ✓✗）；**判据 = 日志里有没有 `结论：` 行** ✗
  （半截日志不算跑过 ✓）。⚠️ 有红时的顺序 ✗：先数进程 ✓（>1 就清 ✓）→ **单独连跑 3 次** ✓
  （一次绿说明不了什么 ✗）→ 还绿再去查「什么时候写终态」这类**时序** ✗ —— ⭐ **「单独复跑就好了」≠
  「不是代码问题」** ✗✗（`h3_stage2_test` 那条偶发红就是这么被误判过一轮 ✓）。
* ⚠️ **性能数字必须把口径一起报** ✗（只报"热"数字 ⇒ 会得出"比 Rust 快 5.5 倍"的假象 ✓✗）。
* ⚠️⚠️ **外部调用审计**（四类生成曾**全部**解析到厂商 API ✗：`ai_providers.py` 按 priority 降序取第一条 ✓，
  本地预设 82–85 ✓ 输给厂商 97–300 ✓ ⇒ 厂商永远赢 ✗）⇒ 抬本地 priority / 删外部行 ✓；
  **审计表与三层依赖阶梯见 §外部调用审计** ✓。
* **⚫ 可照抄项目**（用户点名 ✓「不要忘记」✓）：ComfyUI / minimax-h3-comfyui / ollama / ollama-python /
  minimax-desgin-plugin ⇒ **功能直接抄** ✓（落点见 §可照抄项目清单 ✓）。

## SLM 无限漫剧创造台 · 逆向笔记（2026-09-24 ✓ 用户提供安装包 ✓）

**它是什么** ✓：`Inno Setup` 装的**桌面壳** ✓（`D:\app\SLM` ✓）= `app.py`（pywebview + **bottle 8123** ✓）
+ `launcher.html`（GSAP ✓ 启动器 UI ✓）+ `runtime/`（Python 3.13 嵌入式 ✓ 只带 bottle / pywebview /
pythonnet ✓）。**产品本体不在这里** ✗：升级清单（`https://slm-update.oss-cn-hangzhou.aliyuncs.com/manifest.json` ✓）
显示真正的**工作台是 `SLM漫剧台_v3.5.40.zip`（74.7 MB ✓ sha256 有 ✓）** ⇒ 壳只负责**检测 / 引导 / 拉起 /
插件管理 / 升级** ✓，真正的工作台是 `H3EasyDirector/h3e_backend.py`（独立进程 ✓ `--port 8081 --comfy
http://127.0.0.1:8188` ✓）。
⭐ **同一个模型家族** ✓：它整套是 **MiniMax H3** ✓ —— H3 核心插件保护名单 `H3_CORE_PLUGINS`（`H3EasyDirector` /
`ComfyUI-H3-Director` / `TE-Speed-MiniMaxH3` / `ComfyUI-KJNodes` / `ComfyUI-NB-H3-HyperStep` / `H3E_IndexTTS2.5` /
`TE-Speed-FlashVSR` / `ComfyUI-VideoHelperSuite` ✓）。

### ⭐⭐ 最值得参考：**三档引擎表** `ENGINE_LOCAL_META`（`app.py:149-183` ✓）
| 档 | steps | 分辨率 | 加速 | 实测倍率 | 显存建议 |
|---|---|---|---|---|---|
| `official` 官方稳妥流（入门 L1 ✓） | 25 | 832×480 | 无（**零第三方依赖** ✓） | 1×（5 秒 10–15 分钟 ✓） | 8G 可跑 |
| `std8` 标准加速流（中级 L2 ✓） | 8 | 1024×576 | 8 步蒸馏 LoRA（LightX2V 系 ✓） | ≈2.1× | 16G+ |
| `fast4` 极速流（旗舰 L3 ✓） | 4 | 1024×576 | 4 步蒸馏（**官方 Comfy-Org R2V 模板同款** ✓） | ≈3.44× | 8G 甜点；5090 可 +Sol-Attn +NVFP4 → 1344×768 |

⚠️ 它给每档都配了**模型清单**（`unet_ref2v` / `unet_fl2v` / `te_qwen` / `vae_video` / `vae_audio` /
`clip_l` / `nodes_speed` / `lora_4step_*` / `unet_hybrid` / `up_2k` / `unet_dasiwa` / `unet_nvfp4` /
`nodes_solattn` ✓，逐项标 **必需 / 推荐 / 可选** ✓）+ **`gpu_tip`** ✓（哪档配哪张卡 ✓）⇒ 这就是一张**产品化
的就绪矩阵** ✓ —— 与本仓 `inventory` / `weights` / `engine_readiness` 是**同一个问题**，值得对照 ✓。
⚠️ 档位口径也带**经验断言** ✓：「官方原生步数 = 画质上限」✓、「4 步蒸馏会削弱轻语/口音/硬切细节 ⇒ 重要
片段切回入门档」✓、「原生步数 + 官方 sigma shift（12/3）」✓（与本仓 H3 事实一致 ✓）。

### 其它可借的点（都有落点 ✓）
* ⭐ **已知冲突台账** `COMPAT_WARN`（`app.py:136-144` ✓）：DaSila（磁盘监控 ⇒ WinError 433 ✓ 建议进程隔离 ✓）、
  EasyCache ⨯ Turbo LoRA ✓、CacheDiT/T8 ⨯ FirstBlockCache ✓、Spectrum（高端卡掉驱动 ✓）、SolAttn（需 triton ✓
  8G 收益仅 3.5% ✓）⇒ **真实踩坑清单** ✓ 比任何文档都值钱 ✓。
* **契约同步 + 离线兜底** ✓：壳里那份 `ENGINE_LOCAL_META` 明确写着「**权威在工作台 `/api/workflow_tiers`** ✓，
  本表只用于①工作台没启动时也能展示/校验 ✓②离线切换 ✓」⇒ 与本仓「后端是权威、前端是镜像 + 离线兜底」同构 ✓。
* **多镜像链** ✓：插件清单走 `cdn.jsdelivr.net`（ComfyUI-Manager 的 `custom-node-list.json` ✓）；
  zip 下载 `codeload.github.com` → **`ghfast.top` 代理** 两级回退 ✓（国内可达性 ✓）。
* **自托管升级** ✓：Aliyun OSS 的 `manifest.json`（`channel` / `notice` / `launcher` / `workbench` ✓ +
  **sha256 + size** ✓）。
* **启动器 UX 细节** ✓：EULA 必须**滚到底**才能勾 ✓；步骤状态机 `stepState(n, curr|done|todo)` ✓；
  日志**自动跟随 + 用户滚动即接管** ✓；退出时询问「是否同时关闭 ComfyUI」✓；原生目录对话框要传
  **父窗口句柄**保证置前 ✓（pythonnet ✓）；首次运行建桌面快捷方式（PowerShell + **`utf-8-sig` BOM** ✓
  —— PS5.1 读中文要 BOM ✓，与本仓记忆里那条坑一致 ✓）。
* **进程边界** ✓：壳 8123 ↔ 工作台 8081 ↔ ComfyUI 8188 三段分明 ✓，各用 `/api/status` 探活 ✓ + 就绪轮询 ✓
  + 子进程显式 CLI 契约 ✓。
* ⚠️ **与「自研优先」的分歧** ✗：它的商业模型是「**买家自带 ComfyUI**」✓（能力外包给 ComfyUI + 第三方插件 ✓）
  —— 与本仓「所有能力自研、外部队商 API 要去掉」相反 ✓ ⇒ **借它的工程与体验，不借它的依赖结构** ✓。
* ⚠️ **未做**：真正的漫剧生产逻辑（提示词 / 工作流 / H3 管线 ✓）在那个 74.7 MB 的 `v3.5.40` 包里 ✗
  ⇒ 想要那部分得单独拉包再逆向 ✓（用户未确认 ✓）。


### SLM 工作台包逆向 → **可用性判定**（2026-09-24 ✓ 包在 `%TEMP%\slm_reveal\` ✓ sha256 已核 ✓）
⚠️ **商业闭源** ✓（激活码 + 防破解 + EULA ✓）⇒ **只借事实与架构，不照抄代码** ✗；
EasyDirector 后端是 PyInstaller（加速链执行 / 预设值 / 探测逻辑**读不到** ✗），
但 `ComfyUI-H3-Director` 两个插件的 Python 源码与前端**可读** ✓（它更新说明里那 26 条修复也照此只取会影响我们的 ✓）。

**① 要补的（按优先级 ✓）**

| # | 缺什么 | 包里依据 | 落点 |
|---|---|---|---|
| 1 | **H3 官方 prompt 契约**（✗ 本仓零命中） | `studio_node.py`：`subject_definitions` / `retention_analysis` / `detailed_description` 三行 + 三档关系 `fully_copy` / `partially_copy` / `reference` 的**英文声明句**；⚠️ 坑：**只写 `<Audio 1>` 绑定句不算声明** ⇒ 自动声明被吞、模型不复用配音（实测相关性≈0） | `engine/conditioning.py`（现只 2 函数） |
| 2 | **Ref2VA / FL2VA 决策 + 硬约束** | `_select_h3_task()`：**有参考素材 ⇒ 只能 Ref2VA**（FL2VA 用参考会丢素材/失败 ⇒ 直接拒绝）；需硬首帧 ⇒ 无 FL2VA 则报 | `torch_backend` / `h3_form` |
| 3 | **联合 AV latent 容器互操作** | `h3_latent_io.py`：视频流 **`B×24×T×H×W`**（24 通道第 3 次独立印证 ✓）；多流/缺流**分别报错**；替换视频流要**保容器类型** | `vae.py` / `comfyui_client` |
| 4 | **加速链（配置驱动 + 提交前校验）** | `accel_chain.json` schema + 四项校验（注册 / 参数名 / 必填 / **kind 接线**：`lora` 需 model+clip、`model_only` 需 model）+ ✅/⚠️/❌ + 自动置灰 | 已有 `/object_info` 与严格 UI→API 校验 ✓ ⇒ 补配置层 |
| 5 | **`convrot` 布局不支持** ✗（我们只认 fp8/int8 四种） | 模型名实锤：`minimax_h3_ref2va_pruned_int8_convrot` / `..._video_vae_int8_convrot` | `engine/quant.py` |
| 6 | **权重内嵌元数据契约** | `read_checkpoint_info`：safetensors `metadata` 带 JSON（`format` / `strict_latent_only=true` / `base_config` / `config` / `step`）**逐项严格校验、缺了就拒绝** | `engine/loader.py` |
| 7 | **超清双采（第二遍）** | `h3_upscaler.py`（可读 ✓）：latent 放大器 V2/V3 + `spatial_pixel_shuffle_2x`（**不在时间维插值**）+ `_h3_build_denoise_mask` 掩码重去噪 + cond refs ×2；≈**6×** 耗时、权重缺失**自动回退** | `engine/pipeline` |

**② 只记认知的（会改判据 / 口径的）**
- **混合加载**：`H3HybridLoader` = **fl2va 基底 + ref2va 的 adaLN 覆盖层**流式合并成一个 MODEL ⇒ 两权重主要差 adaLN ✓（一台机可一个模型两用 ✓）；带缓存指纹 / 磁盘余量 / 原子落盘 ✓。
- **分镜解析口径** ✓：5 种标记 + 官方 `[Shot N] At mm:ss`；**≤15 s 合并成一次生成**、>15 s 才贪心装桶；**4.5 字/秒**估时长；断点优先级 = 段落换行 > 句末标点 > 从句标点（**绝不句中硬断**）；时长吸附 VAE 档位（±0.3 s）。
- **音频条件**：参考音频**压制自生成环境音** ✓✗ ⇒ 对话段挂音色槽、空镜段不挂 ✓；「替换音轨 ≠ 对口型」✓（要口型必须 reference-driven ✓）。
- **实测口径**：16 GB @0.4MP ⇒ 每 10 秒段 **8~12 分钟** ✓；0.4MP / 32 的倍数 = 官方基线 ✓；8GB 档 5.2 s = **124 帧**（= `17×7+5` ✓ 与本仓 `H3_FRAME_GRID`/`H3_MIN_FRAMES` 一致 ✓）。
- **值得照做的手法** ✓：任务记录**原子写** ✓；**全局镜号唯一且从 1 开始** ✓；缩略图**按磁盘尾帧恢复** ✓；段配置哈希跳过 ✓；尾帧接力 ✓；每段**深度卸载** ✓；1.5x 超分**先做合法 2x 再缩放** ✓；4x 超分 NaN/Inf **小分块** ✓；缓存文件截断/偏移/非法编码一律拒 ✓。
- **本仓 H3 事实获外部印证** ✓（24 fps / `17n+5` / 单次 ≤15 s / 0.4MP·32 / shift 12/3）。
- 参考数据（用得上再取 ✓）：官方 9 套分镜骨架（`h3_templates.js` ✓）；方案预设名单（漫剧快跑 / 漫剧正式 / 玄幻仙侠 / 悬疑诡异 / 都市写实 / 战斗燃向 ✓，**内部参数在编译层** ✗）；IndexTTS 情绪向量 **8 维**（happy/angry/sad/fear/disgust/melancholy/surprise/calm ✓ —— ⚠️ 早先误记「9 维」✗ 已纠正 ✓；已落地 `voice_contract.EMOTION_ORDER` ✓；另两条情绪入口=**情绪参考音频** `emo_audio`+`alpha` / **文本情绪** qwen-emo ✗ 本仓 CosyVoice 无对应 ⇒ 用不上 ✓）。

> **逆向待做项收口**（2026-09-25 ✓，用户「有用的就逆向，没用的就不做」✓）：§⑫ 记的「TTS 三件套 / 工作台 HTTP 面」判定如下 ——
> ① **TTS 三件套**：有用部分（情绪 8 维顺序 + preset + 语速 `duration_factor` 0.5~2.0 ✓）**早已落地** `voice_contract.py` ✓（docstring 已注明「口径来自逆向 IndexTTS-2.5」✓）；声音克隆本仓 CosyVoice 零样本已有 ✓；剩下的**情绪参考音频 / 文本情绪**是 IndexTTS 引擎专属，本仓走 CosyVoice ⇒ **用不上，不做** ✗。
> ② **工作台 HTTP 面**：核心（三档引擎表 ✓ → `tiers.py`、契约同步+离线兜底 ✓）已落地；后端是 PyInstaller 编译**读不到** ✗，前端 `app.js` 是 333 KB 编译 JS ⇒ 端点清单作 API 设计对照**边际价值低，已有等价物，不做** ✗。
> ③ **COMPAT_WARN 冲突台账**：只拿到 4 条摘要（DaSila⇒WinError433 / EasyCache⨯TurboLoRA / CacheDiT⨯FirstBlockCache / SolAttn 仅快 3.5% ✓），完整 7 条在编译后端**读不到** ✗，且是特定第三方加速件冲突（本仓加速链只认 LoraLoader/TESpeed 等）⇒ **不落地，不做** ✗。


### 超清二采的**口径**（2026-09-25 读上游可读源码核出 ✓ 落点：`studio_node.py` ✓ 不是 `h3_upscaler.py` ✗）

> 判据（为什么这样实现、没核到怎么办）在 `MEMORY.md` §接线 ✓；这里是**事实与公式** ✓。

* **掩码**（`_h3_build_denoise_mask` ✓）：`video=1`（重采样 ✓）/ `audio=0`（保持一采 ✓✗）——
  ⚠️ 语义是**按流缩放该步更新量** ✓，**不是**"把音频排除出前向" ✗（音频照样参与主干 ✓）。
* **σ 取尾部** ✓：`total = max(steps+1, round(steps / clamp(denoise, 0.15, 1)))` ⇒ 取最后 `steps+1` 个 σ ✓
  （`denoise` = 从日程哪个**比例**开始 ✓）。⚠️ 曲线本身用 ComfyUI 的 `calculate_sigmas` ✗
  ⇒ 本仓用**自研调度取尾部** ✓ 并在报告里注明「曲线未逐点对照」✗。
* **噪声**：二采**会重新加噪** ✓，种子 = **段种子 + 1000001** ✓；⚠️ 本仓**只加在视频流** ✓
  （给"锁住"的音频加噪再锁住 = 把干净音轨换成噪声 ✓✗）。
* **关键帧**：cond 里的 keyframe latent **也要 2×** ✓（`_h3_scale_cond_refs` ✓）；⚠️ `references`
  要不要跟着 2× **未核** ✗ ⇒ 本仓**具名拒绝** ✓✗。
* **失败**：上游 `catch` ⇒ **回退一采结果** ✓ + 日志（不炸整段 ✓）⇒ 本仓照此 ✓ 但把 `denoiseError`
  写进报告 ✓（不许哑巴 ✗）。
* ⚠️ 两个参数（`refine_steps` / `refine_denoise`）的**默认值在编译层** ✗ ⇒ 本仓**必填** ✓✗
  （不给 ⇒ `secondPass=False` + notes 说清 ✓）。


### 接线细则（第六次腾预算 ⇒ 从 `MEMORY.md` §接线 下移 ✓ 2026-09-25）

* ⚠️ **stub 少一个端点 ⇒ 那一支永远验不到** ✗✗：H3 的 ComfyUI stub 起初只有 `/object_info/{cls}` ✓
  没有 `/object_info` ✓ ⇒ 加速链接缝**永远**只能判「没查」✓✗（绿得毫无意义 ✓）；
  真 ComfyUI 两个端点都有 ✓ ⇒ stub 要按真服务补齐 ✓。
* ⭐ **要提前知道「是不是克隆路径」⇒ 写廉价探针** ✓（查行 + 文件存在性 ✓ **不读文件** ✗）：
  把重活（读文件/base64 ✓）提到前面会**改掉失败语义** ✗（实测打碎了「本地配置必须申请 audio 租约」
  那条判据 ✓✗ —— 它原本在循环里的 try 内 ✓）。
* ⚠️⚠️ **「同输入 ⇒ 跳过重算」的指纹（`engine/cache_key.py` ✓）**：
  * ⭐ 上游口径**只看张量** ✓ ⇒ 非张量值一律塌成 ``"other"`` ✗ ⇒ **实测**``{"0":"a.png"}`` 与
    ``{"0":"b.png"}`` **指纹相同** ✓✗（个数变了才会变 ✓）⇒ 本仓**请求层**是 URL/路径 ✓✗
    ⇒ 接指纹**前**先补**标量按值进指纹** ✓（张量那条路一字未改 ✓；⚠️ **复合值仍塌 other** ✗ 已写明 ✓）；
  * ⭐ **复用四条缺一不可** ✗（`video_generation.find_reusable_video` ✓）：非 ``force`` ✓ /
    同分镜**成功**产物 ✓ / 指纹**全字段**一致（参考素材 + 提示词 + 镜头类型 + 时长 + 画幅 ✓）/
    ⭐**产物文件真在盘上** ✓✗（远端 URL 证明不了 ⇒ **不复用** ✓ 宁可重算 ✓）；
  * ⭐ **命中不许静默** ✗ ⇒ 响应带 ``reused`` + ``reuseReason`` ✓ + 日志一条 ✓；
  * ⚠️ 指纹要**两种形态都认** ✗✗（``params`` camelCase / DB 行 snake_case ✓）—— 只认一种 ⇒
    两侧都读成空的 ⇒ 两个不同素材算成相同 ⇒ **误跳过** ✓✗✗。


### 接线进度：哪些能力**接出去了**、接在哪（2026-09-24 第五次腾预算 ⇒ 从 `MEMORY.md` 下移 ✓）

> 判据（为什么接、接不出去有多坏）在 `MEMORY.md` §接线 ✓ / 过程与踩坑在 `2026-09-24.md` §⑲⑳ ✓。
> 这里只留**落点表** ✓ —— 「谁被谁调用」是代码里能查的 ✓，本表只为**一眼看全** ✓。

| 能力（判据所在模块） | 生产落点 | 备注 |
|---|---|---|
| H3 prompt 契约（声明三行键 / 编号校验 ✓） | `local_services/h3/server.py::_contract_prompt` ✓（提交前 ✓） | 只写 `<Audio 1>` **不算声明** ✗ |
| H3 形态决策（有参考素材 ⇒ Ref2VA ✓） | `adapters/video_adapters.plan_h3_form` ✓ + 8765 `_pick_checkpoint` ✓ | 收端**核过** `body.model` 才用 ✓ |
| 加速链（配置驱动 + 提交前校验 ✓） | 同上 `server.py::_apply_accel_chain` ✓ | `error` 的环 ⇒ 整条不上并点名 ✓ |
| 配音契约（8 维定序 / 未知情绪拒 ✓） | `services/tts_generation._resolve_voice_params` ✓ | CosyVoice 克隆路径**收不了** ⇒ 进 dropped ✓ |
| 混合加载：计划层 + **加载层** ✓ | `engine/hybrid_load.run_hybrid_load` ✓ ⇒ `POST /engine/hybrid-merge` ✓ | 键集不变 ✓ / 继承基底元数据 ✓ |
| 参考素材指纹（哪段可跳过 ✓） | `video_generation.find_reusable_video` ✓ ⇒ `routers/videos.py` ✓（响应带 `reused` ✓） | ⚠️ 接之前先补「标量按值进指纹」✗✗ |
| 缓存完整性守卫 ✓ | 只被 `hybrid_load` 用 ✓（`total_size` 口子 ✓） | 不验数值 ✗ |
| 段级音频合成（长度守恒 / 三支削波 ✓） | `services/segment_audio` ✓ ⇒ `ffmpeg_compose.mix_model_audio` ✓ | 默认 false ⇒ 行为一字不差 ✓ |
| 超清双采（1.5× 先 2× ✓ / 4× 分块 ✓） | `pipeline` 的 `refine` 阶段 ✓ + `dryrun.refine_latents` ✓ **+ `TorchBackend.refine_latents`** ✓（真装载 + 真前向 + **带掩码的二采** ✓） | **音频流锁定**要后端自述 ✓；二采参数**必填**（`refineSteps`/`refineDenoise` ✓）；⚠️ σ 曲线用自研调度（未逐点对照 ✗） |
| 联合 AV 容器互操作 ✓ | 同上 `dryrun.refine_latents` ✓ | 只换视频那一槽 ✓；⚠️ torch 后端的双流是 **dict** 形态 ✓（不是容器 ✓） |
| 超清放大器可用性 ✓ | `inventory.readiness().upscale` ✓ ⇒ `GET /engine/readiness?upscalePath=` ✓ | 没给 ⇒ **「没查」** ✗ 不是可用 ✓ |
| 档位表 ✓ | `inventory.readiness().tiers` ✓ | 没给显存 ⇒ 「没比」✗ |
| 导演稿分段 ✓ | `POST /prompts/segments` ✓ | 一个字都不丢 ✓ |
| 超清计划 ✓ | `POST /engine/upscale-plan` ✓ ⇒ 请求体 `upscale` ✓ | 管线**只执行**计划 ✗ 不自己判 ✓ |
| 多集节奏相位注入 ✓（2026-09-25 ✓） | `agent/runtime.py::append_style_profile` ✓（调 `rhythm_guidance_for_episode` ✓） | `storyboard_breaker` 专属 ✓；读库失败只 warn 不阻断 ✓；早先是「未迁」warn 占位 ✗ |


### MEMORY.md 下移（2026-09-24 第四次）：Skill 体系细则 + 协作与提交细则 ✓

下移原因：`MEMORY.md` 逼近 8k 硬预算 ✓（守卫的规矩是「本文件只留会导致 bug 的判据 ✓，细节下移 ✓」）。
原话照录 ✓，未改一处判据 ✓：

#### §Skill 体系细则（原 `## Skill 体系` 全段 ✓）
- 库由 `<lib>/library.yaml` 识别 ✓（加库/换库/改展示名**零代码**）；⚠️ **无执行入口的 skill 勿写进 `agents:`** ✗；
  **词库按介质分家** ✓；**core 目录名 = agent_type 勿改名** ✗、**skill id 勿重命名** ✗（`references/` 互引**静默**断链 ✓）。
- **改绑定 = 改 md** ✓（frontmatter `agents:` + `priority`；入口唯一 ✓）；⭐ **DB 配置优先铁律**：解析出配置
  即「用户已选过」⇒ **全关也不回退默认** ✗（仅 `null`/空/失败才回退 ✓；`enabled` 缺省 = 启用 ✓）；
  **出厂默认唯一出口** ⇒ 前端**不得硬编码** ✗。
- **加载器只读 `SKILL.md`** ⇒ `references/` 对 agent **不可达（有意取舍）** ✓；**删除保护** = 顶层 id **且**
  `agents:` 非空 ✗（core 删掉永久丢失 ✓）；解析失败 ⇒ 保守拒删 ✓；⚠️ **宿主工具只认 `hub_` 前缀** ✓ 且
  **工具集须取 `tool.id`** ✗（取错 ⇒ 21 个工具全误判缺失 ✓）；⚠️ **改名/挪库后必核对 DB 绑定** ✓（存 id ⇒ 静默失效 ✓）。
- 注入闸：默认只注自有 ✓、超预算按 priority 跳过给诊断 ✓、合计**只算 `enabled=true`** ✓。

#### §协作与提交细则（原 `## 协作与提交` 里被收起的行 ✓）
- git 远端 `git@github.com:yuanxiufei/darme-voide.git` ✓；commit message 长的写 `tmp/*.txt` + `git commit -F` ✓。
- **换行**：`.gitattributes` **只**声明 `.githooks/* text eol=lf` ✗（**不加 `* text=auto`** ✓ —— 避免全仓
  renormalize 噪声 ✓）；CRLF 会让 shebang 变 `#!/bin/sh\r` ⇒ Windows 上 command not found ✓✗；
  改钩子后查 `git ls-files --eol <file>` ✓（须 `w/lf` ✓）。
- **`.codebuddy/memory/` 无 gitignore 规则，且现已全部纳管**（`MEMORY.md`/`TOPICS.md`/`INDEX.md` + 各日日志，
  自 2026-09-12 的 `abc6cac` 起同批提交）⇒ **新增日志 / 改索引后要随同批提交**，否则 `MEMORY.md` 的读法指针
  在新克隆上**断链**；状态用 `git ls-files` / `git check-ignore -v` 复查。
- **`execute_command` 拉大文件/跑大正则易被转后台丢 stdout** → 用
  `[System.IO.File]::WriteAllText(path, content, UTF8)` 落盘再读；`Get-Content` 必须显式 `-Encoding UTF8`。


### 超清放大器（clean-latent 2×）的**架构规格** ✓（2026-09-24 读上游核出 ✓ 可直接照做 ✓）

⭐ **许可已明** ✓：上游 `h3_upscaler.py` 文件头写着「**Mamad8 权重，MIT 代码**来自
github.com/mamad8c/ComfyUI-H3-Latent-Upscaler-Mamad8」✓ ⇒ 本仓**可以实现** ✓（带署名 ✓）。
⭐ **检查点格式串**（2026-09-24 核到真值 ✓，此前只知道「有个 format 键」✗）：
``minimax_h3_clean_latent_upscaler_v3_factorized_attention`` ✓ ⇒ 结构 = **V2 主干 + V3 因子化注意力** ✓
（本仓 `engine/upscale.py` 已把它钉住 ✓：``CHECKPOINT_FORMAT`` ✓ + 两段配置的**严格字段集** ✓
（多一个少一个都拒 ✗、值必须正整数 ✓、``in_channels`` 必须 24 ✓、``temporal_kernel`` 必须奇数 ✓、
``width % heads == 0`` ✓））。

**流程**（口径 ✓）：低清一采 → latent **2× 放大** → **低噪声二采精修**（⚠️ **音频流锁定不重采** ✗ ——
只重采视频流 ✓ ⇒ 正好用 `engine/latent_container.py`（认出视频流 + 保容器换流 ✓）✓）。

**V2 主干（要实现的全部 ✓，张量 ``B×24×T×H×W`` ✓）**
1. `spatial_bilinear_2x`：`(B,C,T,H,W)` → 折成 `(B*T,C,H,W)` → **bilinear ×2** → 还原 ✓；
   ⚠️ 关键：**绝不在时间维插值** ✗（把 T 折进 batch ✓ 就是为了这个 ✓）；
2. `spatial_pixel_shuffle_2x(x, out_channels)`：输入通道必须是 `out_channels*4` ✓ ⇒
   `view(B, out, 2, 2, T, H, W)` → `permute(0,1,4,5,2,6,3)` → `reshape(B, out, T, H*2, W*2)` ✓（通道→空间 ✓）；
3. `_groups_for(C)`：`min(16, C)` 起**往下**找能整除 C 的组数 ✓；
4. `ResidualBlock3d(C, k=3, spatial_dilation=d)`：`GroupNorm(eps=1e-6)` → `SiLU` → `Conv3d((k,3,3),
   padding=(k//2, d, d), dilation=(1,d,d))` → 再做一遍 → **加残差** ✓（⚠️ 时间 padding = k//2 ✓
   保住时间对齐 ✗ —— 这就是「不在时间维插值」的另一半 ✓）；
5. `H3LatentUpscalerV2`：`low_stem`（24→hidden ✓ 卷积 (k,3,3) ✓）
   → **num_blocks 个 `ResidualBlock3d`**，空间膨胀按 **`(1, 2, 1, 3)` 循环** ✓（`i % 4` ✓）
   → `low_norm` → `to_high`（hidden → **refine_channels×4** ✓）
   → **`spatial_pixel_shuffle_2x` 到 refine_channels 并放大 2×** ✓
   → `refine_stem`（⚠️ 输入是 `refine_channels + 24` ✓：**把双线性基底拼进通道** ✓）
   → `refine_blocks` 个 `ResidualBlock3d(refine_channels)` ✓（**不加膨胀** ✓）
   → `out_norm` → `out`（refine_channels → **24** ✓）；
6. ⭐ `forward = spatial_bilinear_2x(low) + correction(low)` ✓ —— **残差式超分** ✓
   （双线性上采样当基底、网络学**修正量** ✓；⚠️ 所以输出天然含基底 ✓，`correction` 只学"补差" ✓）。

**还没读的部分** ✗：`FactorizedBlock` / `H3LatentUpscalerV3`（因子化注意力那块 ✓ 约 120 行 ✓）+
`build_upscaler`（怎么按契约装配 ✓）⇒ 下次接着读 ✓。


#### 超清放大器 · V3（因子化注意力）+ 装配 ✓（2026-09-24 读上游补全 ✓）

⭐ **层次关系**：``H3LatentUpscalerV3`` **内含** ``H3LatentUpscalerV2`` ✓ ⇒ V2 是必经之路 ✓；
两层**都是残差** ✓：V2 内是 ``bilinear2x(low) + correction(low)`` ✓，V3 外再套 ``base + delta`` ✓。

**`FactorizedBlock(width, heads, window, shifted, mlp_ratio)`** —— 三件套**每件都带残差** ✓：
1. **空间窗口注意力**：`shift = window//2 if shifted else 0` ✓
   （⚠️ shifted 版先 `F.pad(x, (shift, 0, shift, 0))` ✓ ⇒ 切窗 → 注意力 → 还原 → **裁回**
   `[..., shift:shift+H, shift:shift+W]` ✓ —— 这样窗口边界不会永远落在同一处 ✓✗）；
   先把 H/W padding 到 `window` 的整数倍 ✓（`(-h) % window` ✓），
   再 `permute(0,2,3,4,1)` → `reshape(B*T, H//w, w, W//w, w, C)` → `permute(0,1,3,2,4,5)`
   → `reshape(-1, w*w, C)` ✓（**窗口内做注意力** ✓）；`MultiheadAttention(batch_first=True,
   need_weights=False)` ✓；
2. **时间注意力**：把 H,W 折进 batch（`permute(0,3,4,2,1)` ✓）⇒ 沿 **T** 做注意力 ✓；
3. **深度可分离局部卷积**：`GroupNorm(1, width)` → SiLU → `Conv3d(w, w, 3, padding=1, groups=w)` ✓；
4. **MLP**：`LayerNorm` → `Linear(w, w*mlp_ratio)` → **GELU** → `Linear(...)` ✓；
   ⚠️ 所有 LayerNorm/GroupNorm 的 `eps=1e-6` ✓。

**`H3LatentUpscalerV3(base_config, config)`**
* ``self.base = H3LatentUpscalerV2(base_config)`` ✓；
* ``stem = Conv3d(24 → config.width, 3, padding=1)`` ✓ —— ⚠️ **作用在 `low` 上** ✗（不是 base 输出 ✓）；
* ``blocks``：`config.blocks` 个 `FactorizedBlock` ✓，**shifted 按 `bool(index % 2)` 交替** ✓；
* ``norm = GroupNorm(1, width, eps=1e-6)`` → ``to_delta = Conv3d(width → 24*4, 3, padding=1)`` ✓
  → **`spatial_pixel_shuffle_2x(delta, 24)`** ✓（通道→空间 2× ✓ 回到 24 通道 ✓）；
* ``forward = base(low) + delta(low)`` ✓。

**装配**：`build_upscaler(state_dict, info)` ✓ ⇒ 按契约**先建结构**再
`load_state_dict(strict=True)` ✓（⚠️ 缺键/多键 ⇒ 报错并写明「**张量与声明的架构不符**」✓✗ ——
别指望它静默跳过 ✓）→ `.eval().requires_grad_(False)` ✓。
⇒ 本仓实现时要守住：**严格装载** ✓ + **失败即拒** ✓（与 `loader` / `quant` 的纪律一致 ✓）。

⚠️ 实现落点（下一步 ✓）：`engine/upscale_net.py`（torch **懒加载** ✓，与 `dit`/`h3_form` 同套做法 ✓），
自检用**未训练权重**验不变量 ✓：2× 后 `H/W` 翻倍 ✓、**`T` 不变** ✓、通道回 24 ✓、
`forward == bilinear + correction`（V2 ✓）与 `forward == base + delta`（V3 ✓）✓；
⚠️ **明说**：验的是管道/形状 ✗，**不宣称画质** ✗。


#### ⚠️ 修正：B-3（HTTP 面「可借项」）**大半本仓已有** ✗ ⇒ 别重做 ✗（2026-09-24 ✓ 实查 ✓）

原先列的「借 `routes.py` 的 HTTP 面」里，**两条核心已经在我们仓里、而且更细** ✓✗：

* **上游 key 脱敏** ✓ —— `app/services/provider_probe.py:redact_url` ✓ 覆盖
  ``key`` / ``api_key`` / ``apikey`` / ``token`` / ``access_token`` ✓；⚠️ 它**不只在日志里** ✗✗
  （``/test`` **响应体**里就带脱敏后的 ``url`` ✓ ⇒ 照抄时**别用 `urlencode`** ✗ —— JS 的
  ``searchParams.set('key','***')`` 不编码 ``*`` ✓）；`app/services/task_logger.py` 落盘前统一脱敏 ✓
  （密钥 ⇒ ``***`` ✓）并**截断 base64 / data-url** ✓（否则一条 trace 几十 MB ✓✗）；
  三条链路都接了 ✓：`image_generation` / `video_generation` / `tts_generation` ✓ + `ai_configs` 探针 ✓。
* **未实现布局族具名拒绝** ✓ 同族已在权重守卫里 ✓。

⇒ **B-3 缩小为两条待核**（不是「新建一层」✗）：① **上传上限**（口径 >10 MB 直接拒 ✓）——
`app/routers/local_models.py` / `app/http_logger.py` 里有 `Content-Length`/上限相关命中 ✓ 待逐行核 ✓；
② **上传文件名脱敏**（时间戳前缀 ✓ 防目录穿越 ✓）待核 ✓。**先核再改** ✓ —— 别凭「要借」就动手 ✗。


**B-3 两条待核 → 实查结论（2026-09-24 ✓ 逐行核过 ✓）**：
① **文件名脱敏 / 路径穿越** ⇒ **本仓已有且更硬** ✓✗ ⇒ 别加 ✗：`routers/local_models.py` 的
``resolve_model_path`` ✓（目标必须落在 ``models_dir`` 内 ✓）、``normalize_repo`` ✓（``owner/name`` ✓）、
删除**限制在模型目录内且不许删根** ✓、列表只展示 ``basename`` ✓（**不遍历磁盘** ✓）。
② **上传上限**（口径 >10 MB 拒 ✓）⇒ ⚠️ **别照搬** ✗：`app/http_logger.py` 里那个
``_BODY_READ_LIMIT = 256 * 1024`` ✓ 是**日志读取上限** ✗（注释写明「只影响超大请求体的日志」✓），
**不是**上传上限 ✓✗；面向 multipart 上传的大小上限**未见** ✗ —— 但**先确认我们的上传面存不存在**
✗（若上传不经 HTTP 到达本服务 ⇒ 这个上限就是**照搬别人的问题** ✗✗）。**结论：不加** ✓ 除非
先证明我们这条面有同样的暴露 ✓。


**B-1「音色克隆」核查结论：已通 ✓ ⇒ 别加 ✗（2026-09-24 逐处核过 ✓）**：克隆**不在** `adapters/` ✗
（那里 0 命中 ✓），而是走**音色库** ✓ —— `ai_voices` 有 ``reference_audio`` + ``prompt_text`` ✓，
`app/services/tts_generation.py:_load_cosyvoice_zero_shot` ✓ **真拿它走 `/inference_zero_shot`** ✓
（零样本克隆 ✓；行不存在或参考音频不存在 ⇒ 返回空 ⇒ **回落** ✓✗ 不报错 ✓）。
⚠️ 区分两个字段 ✗：`ai_voices.reference_audio` = **克隆输入** ✓；`characters.voice_sample_url`
= **试听产物** ✓（`generate_voice_sample` 出的 ✓）⇒ 别把后者当克隆输入用 ✗✗。
⇒ B-1 真正剩的只有两处：① **`voice_contract` 还没接调用链** ✗（契约层已立 ✓）；
② 待核：**克隆参考音频 + 情绪向量能否同时给** ✓✗（引擎侧约束 ✓ —— **先核引擎再写判据** ✗，
别凭猜加互斥 ✗）。


### ⏭ 交接：下一步只有一件事（2026-09-24 收工 ✓）

**接线 `voice_contract` → 配音链路** ✓：契约层已立（`app/services/voice_contract.py` ✓ 自检 15/15 ✓）
但**还没被任何调用方使用** ✗。落点：`app/services/tts_generation.py` ✓（`generate_tts` 的入参构造 ✓）
+ `app/services/adapters/tts_*` ✓（厂商契约为准 ✓）。要做的最小闭环：
① 角色的 ``voice_emotion``（**单值** ✓）经 `parse_emotion` ✓ 转成「预设名 或 8 维向量」✓；
② 语速经 `validate_speed` ✓（⚠️ 区间**由调用方按引擎给** ✗ —— 本层不内置好取值 ✓）；
③ ⚠️ **先核引擎是否允许「克隆参考音频 + 情绪向量」同时给** ✗（别凭猜加互斥 ✗）。
**范围自律**：本轮已三次得出「**本仓已有，别加**」✓✗（B-3 脱敏/穿越 ✓、音色克隆 ✓）⇒
**先核现状再动手** ✓，尤其别重复造脱敏、上传上限、克隆那三层 ✗。
**未提交** ✓：工作区 16 改 + 28 新增（全量 **120 套 / 3912 项 / 0 失败** ✓ 已验证 ✓）。
⚠️ 提交需**用户明确要求** ✗；message 用**英文** ✓；`.codebuddy/memory/` 改动**随同批** ✓。

## CLI 裸跑编码：GBK 控制台（2026-09-25 ✓ 实测事故 —— 「全量自检全绿、裸跑就崩」的静默掩盖 ✗✗）

**触发** ✓：照 `torch_backend.PENDING_PARTS` 说的「上机前先跑一次」跑 `app/scripts/h3_readiness.py`（本机真 H3 权重 ✓）
⇒ **裸跑第一条 `print` 就 `UnicodeEncodeError` 崩** ✗（连「① 环境」都打不出来 ✗ —— 它满屏 ✓/✗ / `⇒` / `⚠️`）。

**根因** ✗：本机 **Python 3.14.5** ⇒ stdout 仍按 **locale（GBK）** 编码 ✗（PEP 686 的 UTF-8 默认要 **3.15** 才生效 ✓）。

**四层掩盖链** ✗✗（一层套一层，谁都没把它暴露出来）：
1. `h3_readiness.py` 裸跑即崩 ✓✗（它正是 `PENDING_PARTS` 让人上机前先跑的那条命令 ✗ ⇒ 等于「上机当天才发现跑不起来」✓✗）；
2. `tests/engine_readiness_script_test.py` ⑩（`--json` 必须是合法 JSON）FAIL ✗ ⇒ 根因就是 ① 里脚本崩了、stdout 不是 JSON ✓✗；
3. 而该测试**自己也崩在「打印失败原因」上** ✗✗：`print(f"FAIL … ⇒ {detail}")` 里的 `⇒` = U+21D2，GBK 编不了 ✓
   ⇒ 只剩一个 `EXIT=1`，**失败原因一个字都打不出来** ✓✗（最恶劣的一层 ✓）；
4. `tests/run_all.py` **给子进程灌了 `PYTHONIOENCODING=utf-8`** ✓（它自己的注释就写着这个双坑 ✓）⇒ **全量 133 套全绿** ✓✗
   ⇒ 而文档里推荐的**单跑**用法（`python tests/<某个>_test.py` ✓）在本机裸控制台是坏的 ✗✗。
⇒ 修 ①③ 两处 `reconfigure` 后 ⑩ **自动转绿**（18/18 ✓）—— 反证了「⑩ 的 FAIL 根因就是脚本自己崩」✓。

**判定口径** ✓（守卫复用同一条 ✓，三条**缺一不报** ✗）：
① 文件里含**locale 编不了的字符** ✓ 且 ② 文件里**真有会执行的 `print`** ✓ 且 ③ CLI 入口**没有** `reconfigure` ✓
⇒ 判「裸跑会崩」✗。⚠️ 少了 ①/② 的（纯 ASCII 文件、没有 `print` 的模块）必须**不报** ✓ —— 否则会变成「见 `print` 就红」✗。

**收口** ✓：新建守卫 `backend-py/app/scripts/check_cli_encoding.py` ✓（扫 `tests/` + `app/scripts/`，本次 **152 个文件** ✓）
—— 探针照上面口径命中 **65 个** ✗（62 测试 + `app/scripts/corpus/analyze3.py` / `app/scripts/db_upgrade.py` / `app/scripts/tokenizer_bench.py` ✓）
⇒ 63 个由**一次性 AST 补丁脚本**批量改 ✓（跑完即删 ✓），剩 2 个**模块级脚本**（没有 `if __name__ == "__main__":` 块 ✓）手改：
`backend-py/tests/smoke_test.py` ✓、`backend-py/app/scripts/corpus/analyze3.py` ✓。
⚠️ **只认** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` ✗：老写法
`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")` **丢掉原 wrapper** ✗（刷新顺序与调用方不一致 ✓✗、
遇 locale 外字符仍抛异常 ✗）⇒ 守卫认不出它、会**一直报** ✓✗（本次就是它被报出来的 ✓）。
**已接进**：`check_all.py` 第 3 道 ✓（现共四道 ✓）+ `test_guards.py` 负向 3 例 ✓
（基线 ✓ / ⓪ 会崩 ⇒ 报 ✓ / ⓪b 纯 ASCII ⇒ 不报 ✓ / ⓪c 老写法 ⇒ 必须报 ✓ ⇒ 共 **17/17** ✓）。
**硬证据** ✓：`check_all.py` 四道全绿 ✓；抽样**裸跑**（不带 `PYTHONIOENCODING` ✓）4 套测试 `exit=0` ✓：
`agent_prompts_test` 20/20 ✓、`engine_quant_test` 18/18 ✓、`engine_h3_keys_test` 29/29 ✓、`skills_test` 45/45 ✓。

## 自 MEMORY.md 下移（2026-09-25 第五次腾 8k 预算）—— 自研优先：原话 / 三条硬约束 / 边界

> 起因：`MEMORY.md` 7913/8000（余量 87 ✓），守卫提示「下轮进内容前先下移」✓。本轮把 §自研优先 的
> **原话与举例**移到这里 ✓，`MEMORY.md` 只留**一句话缩写 + 指针** ✓（用户 2026-09-24 纪律：加新内容前先清等量的旧 ✓）。

**用户原话**（2026-09-20 ✓）：「**所有功能不要对外依赖，自己实现所有的功能，要参考我给你的几个项目**」⇒ 落成三条硬约束：
1. **能力自研**：功能要能在**本仓自己实现**（推理 / 生成 / 解析 / 合成 都算 ✓）⇒ 不许把关键能力**只**挂在外部队商 API 上 ✗；
2. **不对外依赖**：默认形态是**离线可跑**（本地服务 / 本地权重 / 纯计算 ✓）⇒ 外部队商适配器**可以有**（作为可选通道 ✓），但**不能是唯一出路** ✗；
3. **参考 `reference/` 那几个项目**（Mini-Agent / minimax-desgin-plugin / ollama-python / ollama /
   Open-AI-Micro-Drama-Generator / short-drama-agent ✓）：**有用的功能搬过来自己实现** ✓（判据同「没人调用的库不算功能」✓：搬来要**真接线** ✓）。

**⚡ 2026-09-20 加严**：「**所有都要自己实现，不要调用外部的**」⇒ 外部队商 API **不是可选通道，是要去掉的** ✗。
**边界**（可纠正 ✓）：**功能 / 能力**不外包 ✗；`ffmpeg` / SQLite / 标准库 / 框架属**本地基础设施** ✓ 不算 ✗。
⚠️ 现状（2026-09-20 实测，详见 §外部调用审计 ✓）：四类生成曾**全部**解析到厂商 API ✗（`ai_providers.py` 按 priority 降序 ✓）。
