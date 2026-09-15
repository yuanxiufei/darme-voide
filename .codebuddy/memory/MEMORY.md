# MEMORY — 长期记忆

> 只留**会导致 bug 的不变量**与约定；细节看 `docs/`、代码。**读法：本文件（必读）→ `TOPICS.md`（低频长专题）→ `INDEX.md`（日志定位；默认只读索引、按 `@行号` 跳读日志，勿整读）**。**本文件须 ≤8k 字符**，超限会被注入截断（实测 9.4k 即截断，且断在半句——**尾部 `## 协作与提交` 最先丢**）。**逼近上限时把细节下移 `TOPICS.md`，勿硬塞**。

## 项目与运行
Drama Studio（`d:/code/voides/voide-darme`）：AI 剧本/分镜/视频。Nuxt 3（`frontend/app`）+ Hono（`backend`，**非 Express**）+ Mastra（5 Agent）+ Drizzle + better-sqlite3。
- 后端 **5789**（`config.ts`）、前端 **3013**（proxy `/api`、`/static`）；前缀 `/api/v1`，另有 `/webhooks/*`、`/static/*`；无登录页。页面 `/`、`/settings`、`/drama/[id]`、`/library/*`。
- 启动 `cd backend && npx tsx src/index.ts`；`cd frontend && npx nuxt dev --port 3013`。Node v22。数据根 `.data-root` > `DATA_ROOT` > `config.yaml database.path` > `./data`；Docker 未装，不依赖 postgres/redis/qdrant。
- **前端 dev 代理现在指向 Python 后端 5790**（2026-09-15 起；旧 Node 5789 可用 `NUXT_API_TARGET` 临时覆盖）；**共享契约类型在前端** `frontend/app/types/contracts.ts`（前端侧镜像，**字段权威在 Python 后端**，改后端字段要同步它）。技能库 / 脚本 / 快照也都在 `backend-py/` 下。
- **`backend-py/` = Python 后端（绞杀者迁移，2026-09-12 起）**：FastAPI + SQLAlchemy **Core**，已迁 `dramas` 域，未迁移域 `PROXY_TO_NODE=1` 反代 Node；**端口 5790**（刻意不读 `config.yaml` 的 `server.port`——那是 Node 的 5789）；回归跑 `backend-py/tests/smoke_test.py`，动手前读其 `README.md`。**删 `backend/` 的三条前置见 `TOPICS.md`**（对拍 / 快照 / 零依赖）。

## 本地模型 + H3 视频推理
**详见 `TOPICS.md`**。仅三条必须记牢：**直连 HF 全超时 → 必须 `hf-mirror.com`**；H3 走 ComfyUI(8188) + 8765 薄封装（`runtime='h3'`、`baseUrl='http://localhost:8765'`）、六键 Bible 跨集锁定；⚠️ 该链路 provider 名 `minimax` 是**服务商标识**，与 `backend-py/skills/` 外部技能库**无关**。GPU RTX A5000 22 GiB，**无 nvcc**。

## 后端能力（9 项）
QC（technical/consistency）｜asset-versions｜style-profiles｜script-fingerprint｜take-budget｜rhythm-phase｜jianying-draft｜estimate-service｜usage-tracking —— **明细见 `TOPICS.md`**（Python 实现全在 `backend-py/app/services/`）。
**约定**：`appendStyleProfile()` 同步；门禁统一支持 `force`；无指纹/无相位视为旧产物不阻断；`storyboards` 无 resolution/fps（在 `video_generations`）；Windows ZIP 内路径转 posix。

## 前端约定
- 复用型 `refresh()` **禁止写 UI 定位副作用**（切 panel/tab、重置选中、重置编辑中表单）——被刷新按钮与所有生成/编辑回调反复调用（含合成 poll 每 4s）→「画面跳转」；UI 定位只在 `onMounted` 一次。refresh 重赋列表后选中项须**按 id 重绑**，未存表单值需 **dirty 标记**保护。
- playwright-cli 常被转后台丢 stdout → `... *> .log` 落盘再读；eval 输出**只用 ASCII**；用 `nav button[i]` 定位。**切勿一边改文件一边跑点击测试**（HMR 会造成「修复无效」假象）。**元素计数与 SFC 编译验证见 `TOPICS.md`**。

## 画风体系（10 种）
- key：realistic / cinematic / noir / anime / ghibli / ink-wash / watercolor / comic / cyberpunk / pixar3d。**词表四段**「画风核心+镜头光线+调色质感+画质」，**负面词负责排除对立风格**。
- **单一事实来源 2 处必须同步**：后端 `shared/prompt-utils.ts`（`ART_STYLE_CATALOG`+`DRAMA_ART_STYLE_MAP`/`DRAMA_ART_NEGATIVE_MAP`/`EQUIP_ART_STYLE_MAP`）与前端 `app/utils/artStyles.ts`；**加画风只改这 2 个文件**。`prompt-utils.ts` **不拆**；找副本顺序 = 后端常量 → skill 正文 → 工具 instruction → 前端硬编码。
- 解析链 `characters.style` → `dramas.style` → `app_settings.art_style` → `realistic`，**唯一入口** `resolveEffectiveArtStyle()`（脏值跳过不透传），各路由不得存副本。
- **正负成对收口**：场景 / 分镜静帧+宫格 / 视频 / 角色·装备·道具·表情各有 `buildXxxArtStyleSuffix`+`buildXxxNegativePrompt`。skill **不得输出画风英文词**（一律后端 suffix 收口，auto-pipeline 现算不落库）。**细节（反 AI 感白名单、视频与静帧为何必须分开、放开写词要改什么）见 `TOPICS.md` §画风体系细节**。

## Skill 体系（详见 `backend-py/skills/README.md`，改前先读）
- **库由声明文件识别，与目录名解耦**：`backend-py/skills/<lib>/library.yaml` 的 `name`/`label`/`description` ⇒ **加库/换库/改展示名零代码**。自有 **8** 个顶层 skill（**5 个即 Agent 类型**），另 3 个非 Agent：`style-reference-reverse` 已 `agents: []` 停注入 ⇒ **无执行入口的 skill 勿写进 `agents:`**（每次生成白背一段上下文）。磁盘实测 **37 = 8 core + 9 + 20**；**词库按介质分家**（`prompt-style-library` 图像｜`video-prompt-library` 视频，仅绑 `storyboard_breaker`）⇒ **改词库前先确认改哪个，别把视频范式写回图像库**（清单与拆因见 `TOPICS.md`）。
- **命名**：core 目录名 = Agent 类型（`agent_configs.agent_type`，**勿改名**）；库内 skill id **勿重命名**（`references/` 互引静默断链）。
- **绑定 = skill 自描述**：frontmatter `agents: [...]` + `priority`（越小越靠前，缺省 100）；解析入口唯一 = `resolveDefaultSkills(agentType)`（**只扫自有**），消费 `loadAgentSkills`/`getAgentDefaults`/`routes/skills.ts`。**改绑定 = 改 md**；`AGENT_SKILL_MAP` 已删。
- **注入闸**：默认只注自有；外部库**按需手动绑**。`SKILL_CHAR_BUDGET`（默认 6 万，`AGENT_SKILL_BUDGET` 覆盖，0=关）→ 超预算按 priority 跳过 + 末尾「未注入：…」诊断。**口径 = `renderSkill(parseSkill(...)).length`**（≠ 字节数）。外部库单体最大 ~2.7 万字符 ⇒ 只勾两个即逼近上限。
- **DB 配置优先铁律**：`parseSkillsConfig(raw)` 解析出配置项即「用户已做过选择」→ **全关也不回退默认**（仅 `null`/空串/空数组/解析失败才回退）；`enabled` 缺省 = **启用**（`!== false`），`priority` 缺省 `0`。默认绑定**仅当** DB `agent_configs.skills` 为空时生效。
- **agent 出厂默认唯一出口**：`getAgentDefaults()`（`DEFAULT_PROMPTS`+`resolveDefaultSkills`）→ `GET /agent-configs/defaults` → `agents.vue`；**前端不得硬编码默认提示词/默认绑定**。
- **同一规则只留一处**：`shared/prompt-blocks.ts` 的 `SCREENPLAY_FORMAT_RULES`、`IMAGE_PROMPT_TEMPLATE_CHARACTER/SCENE/SHOT`。
- **加载器只读 `SKILL.md`** ⇒ `references/` 对 agent **不可达**（22 个含 references 的全是 vendor，core 零引用）→ **有意取舍非 bug**；前端以 `referenceCount` 标注「N 个参考文件（不注入）」。
- **删除保护**：`DELETE /skills/<id>` 拒删 = **顶层（id 不含 `/`）** *且* **`agents:` 非空**（core 删掉永久丢失）；`backend-py/skills/<agent>/<name>/` 不受保护；**顶层但 agents 空可删**；解析失败传 `undefined` → 保守拒删。
- **注入可见性**：`agents.vue` 绑定面板 = 外部库 Skill **唯一 UI 挂载入口**；⚠️ 合计**只算 `enabled=true`**；**拖拽真实生效**（列表顺序 = 注入顺序 = 超预算跳过顺序）。字段清单与 UI 细节见 `TOPICS.md`。
- **兜底库**：顶层目录无 `library.yaml` → `/meta.sources` 补合成条目（`declared:false`，label = 目录名）→ 保证「core + Σ各库 = 总数」自洽且侧栏可达。
- **宿主工具兼容性**：外部库依赖的 `hub_*` 等工具本项目**从未注册**；依赖 = `allowed-tools` ∪ 正文 `hub_*` 引用（**只认 `hub_` 前缀**，宽泛猜会被字段名污染）；**工具集须取 `tool.id`**（取错把 21 个工具全误判为缺失）。**完整判据见 `TOPICS.md`**。
- **`meta.yaml` 是死数据（改它不生效）**：全仓库零读取；但 `version`/`author-*`/`source` **只此一处** ⇒ **勿擅自删**（丢溯源）。见 `backend-py/skills/README.md` §一。
- **改名/挪库后必核对 DB 绑定**：`agent_configs.skills` 存 skill id ⇒ 旧绑定失效；核对用 `better-sqlite3` **`{ readonly: true }`** 直开（绕开清洗副作用，坑⑦）。**实测 5 行全 `NULL` ⇒ 改名零影响**。步骤见 `TOPICS.md`。
- **坑**：① `renderSkill()` **不含 frontmatter name** → 验证注入用正文特征串；② `prompt-utils.ts` 有顶层副作用 ⇒ 前端只能 `import type`；③ **`/skills/meta` 与 `/agent-configs/defaults` 必须注册在通配路由之前**（否则被当 id 吃掉）；④ 本工具 `search_content` 的 `glob` 不生效（详见 `TOPICS.md`）；⑤ `PUT /skills/<id>` 保存后回读校验 frontmatter，缺 `---` 头或 `agents` 为空则返 `{ warning }`（不阻断）→ 前端 `toast.warning`（防「改正文 → 默认注入静默消失」）；⑥ 前端回显 DB `skills` **必须规范化**（`enabled !== false`、`priority` 缺省 `0`、过滤无 `id` 项、非数组兜 `[]`），否则**显示与实际相反**；⑦ 导入 `agents/index.ts` 有 DB 清洗副作用（`[db] sanitized…`）→ 验证脚本会改数据；⑧ **改 skill 名/挪库/引 `docs/` 后必跑 `python backend-py/scripts/check_skill_refs.py`**（1 = 断链；基线 **0 致命 / 0 非致命**（96 处）⇒ **非零即真回归**）。路径须写成 `` `references/x.md` `` 才受采集；⚠️ 守卫**默认未启用**（需 `git config core.hooksPath .githooks`）；⑨ 改记忆必跑 `check_memory.py`、改守卫自身再跑 `test_guards.py`、体检 `check_all.py` —— 判据与基线详见 `backend-py/scripts/README.md`。

## 视频提示词语料
**详见 `TOPICS.md`**（检索管线 8987 条/3 源、已排除源清单、落盘三分都在那儿）。一条红线：他人提示词正文**不得搬运进仓库**（只提炼范式，结论落 `docs/`）。

## 协作与提交
- **未经用户明确要求，绝不 `git commit`**；改完展示 diff。上下文过大时按阶段拆：每阶段只读 1 文件、只改 1 处、逐步验证。
- git 身份 `yuanxf`/`yuanxf@wedoctor.com`；远端 `git@github.com:yuanxiufei/darme-voide.git`。PowerShell 传中文 commit message 会乱码 → 统一英文（长 message 写 `tmp/*.txt` + `git commit -F`）。
- **换行**：`.gitattributes` **只**声明 `.githooks/* text eol=lf`（**不加 `* text=auto`**，避免全仓 renormalize 噪声）—— 钩子由 sh 执行，CRLF 会让 shebang 变 `#!/bin/sh\r` ⇒ Windows 上直接报 command not found。改钩子后查 `git ls-files --eol <file>`（须 `w/lf`）。
- **`.codebuddy/memory/` 无 gitignore 规则，且现已全部纳管**（`MEMORY.md`/`TOPICS.md`/`INDEX.md` + 各日日志，自 2026-09-12 的 `abc6cac` 起同批提交）⇒ **新增日志 / 改索引后要随同批提交**，否则 `MEMORY.md` 的读法指针在新克隆上**断链**；状态用 `git ls-files` / `git check-ignore -v` 复查。
- **`execute_command` 拉大文件/跑大正则易被转后台丢 stdout** → 用 `[System.IO.File]::WriteAllText(path, content, UTF8)` 落盘再读；`Get-Content` 必须显式 `-Encoding UTF8`（否则中文在控制台显示为乱码）。
