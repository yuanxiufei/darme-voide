# MEMORY — 长期记忆

> 只留**会导致 bug 的不变量**与约定；细节看 `docs/`、代码。**读法：本文件（必读）→ `TOPICS.md`（低频长专题）→ `INDEX.md`（日志定位；默认只读索引、按 `@行号` 跳读日志，勿整读）**。**本文件须 ≤8k 字符**，超限会被注入截断（实测 9.4k 即截断，且断在半句——**尾部 `## 协作与提交` 最先丢**）。**逼近上限时把细节下移 `TOPICS.md`，勿硬塞**。

## 项目与运行
Drama Studio（`d:/code/voides/voide-darme`）：AI 剧本/分镜/视频。Nuxt 3（`frontend/app`）+ **FastAPI + SQLAlchemy Core**（`backend-py/`，唯一后端）。⚠️ **Node 后端（Hono + Drizzle + better-sqlite3，`backend/`）已于 2026-09-15 删除**，能力 100% 迁到 `backend-py/`。
- 后端 **5789**（`config.ts`）、前端 **3013**（proxy `/api`、`/static`）；前缀 `/api/v1`，另有 `/webhooks/*`、`/static/*`；无登录页。页面 `/`、`/settings`、`/drama/[id]`、`/library/*`。
- 启动：`cd backend-py && .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790`；`cd frontend && npx nuxt dev --port 3013`（dev 代理已指 5790）。数据根 `.data-root` > `DATA_ROOT` > `config.yaml database.path` > `./data`；不依赖 postgres/redis/qdrant。（Node 5789 已随 `backend/` 删除，仅历史。）
- **前端 dev 代理现在指向 Python 后端 5790**（2026-09-15 起；旧 Node 5789 可用 `NUXT_API_TARGET` 临时覆盖）；**共享契约类型在前端** `frontend/app/types/contracts.ts`（前端侧镜像，**字段权威在 Python 后端**，改后端字段要同步它）。技能库 / 脚本 / 快照也都在 `backend-py/` 下。
- **`backend-py/` = Python 后端**：FastAPI + SQLAlchemy **Core**，**全部域已迁完**（未注册 0 / Node 224 条路径 100% 覆盖）；**端口 5790**；回归 `backend-py/tests/run_all.py`；动手前读 `backend-py/README.md`。⚠️ **`backend/` 已删**（2026-09-15）⇒ TS 原文只在 `backend-py/tests/frozen_ts_source.py`，**守卫读 TS 走「真源码优先 → 快照」**；`PROXY_TO_NODE` 已无对象（接缝只为兜底 501）。**明细见 `TOPICS.md`**。

## 本地模型 + H3 视频推理
**详见 `TOPICS.md`**。仅三条必须记牢：**直连 HF 全超时 → 必须 `hf-mirror.com`**；H3 走 ComfyUI(8188) + 8765 薄封装（`runtime='h3'`、`baseUrl='http://localhost:8765'`）、六键 Bible 跨集锁定；⚠️ 该链路 provider 名 `minimax` 是**服务商标识**，与 `backend-py/app/skills/` 外部技能库**无关**。GPU RTX A5000 22 GiB，**无 nvcc**。

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

## Skill 体系（详见 `backend-py/app/skills/README.md`，改前先读）
- **库由声明文件识别，与目录名解耦**：`<lib>/library.yaml` 的 `name`/`label`/`description` ⇒ **加库/换库/改展示名零代码**。⚠️ **无执行入口的 skill 勿写进 `agents:`** ✗（每次生成白背一段上下文）；**词库按介质分家**（图像｜视频）⇒ **改词库前先确认改哪个** ✓（清单见 `TOPICS.md`）。
- **命名**：core 目录名 = Agent 类型（`agent_configs.agent_type`，**勿改名**）；库内 skill id **勿重命名**（`references/` 互引静默断链）。
- **绑定 = skill 自描述**：frontmatter `agents: [...]` + `priority`（越小越靠前，缺省 100）；解析入口唯一 = `resolveDefaultSkills(agentType)`（**只扫自有**），消费 `loadAgentSkills`/`getAgentDefaults`/`routes/skills.ts`。**改绑定 = 改 md**；`AGENT_SKILL_MAP` 已删。
- **注入闸**：默认只注自有；外部库**按需手动绑**；超预算按 priority 跳过并给诊断（阈值/口径见 `TOPICS.md`）。
- **DB 配置优先铁律**：`parseSkillsConfig` 解析出配置即「用户已选过」→ **全关也不回退默认** ✗（仅 `null`/空/解析失败才回退）；`enabled` 缺省 = 启用。
- **agent 出厂默认唯一出口**：`getAgentDefaults()`（`DEFAULT_PROMPTS`+`resolveDefaultSkills`）→ `GET /agent-configs/defaults` → `agents.vue`；**前端不得硬编码默认提示词/默认绑定**。
- **同一规则只留一处**：`shared/prompt-blocks.ts` 的 `SCREENPLAY_FORMAT_RULES`、`IMAGE_PROMPT_TEMPLATE_CHARACTER/SCENE/SHOT`。
- **加载器只读 `SKILL.md`** ⇒ `references/` 对 agent **不可达**（22 个含 references 的全是 vendor，core 零引用）→ **有意取舍非 bug**；前端以 `referenceCount` 标注「N 个参考文件（不注入）」。
- **删除保护**：`DELETE /skills/<id>` 拒删 = **顶层（id 不含 `/`）** *且* **`agents:` 非空**（core 删掉永久丢失）；`backend-py/app/skills/<agent>/<name>/` 不受保护；**顶层但 agents 空可删**；解析失败传 `undefined` → 保守拒删。
- **注入可见性**：`agents.vue` 绑定面板 = 外部库 Skill **唯一 UI 挂载入口**；⚠️ 合计**只算 `enabled=true`**；**拖拽真实生效**（列表顺序 = 注入顺序 = 超预算跳过顺序）。字段清单与 UI 细节见 `TOPICS.md`。
- **兜底库**：顶层目录无 `library.yaml` → `/meta.sources` 补合成条目（`declared:false`，label = 目录名）→ 保证「core + Σ各库 = 总数」自洽且侧栏可达。
- **宿主工具兼容性**：外部库依赖的 `hub_*` 本项目**从未注册**；**只认 `hub_` 前缀**（宽泛猜会被字段名污染），且**工具集须取 `tool.id`** ✗（取错会把 21 个工具全误判为缺失）。判据见 `TOPICS.md`。
- **`meta.yaml` 是死数据**（改它不生效 ✓ 全仓零读取）；但 `version`/`author-*`/`source` **只此一处** ⇒ **勿擅自删** ✗（丢溯源）。
- **改名/挪库后必核对 DB 绑定** ✓（`agent_configs.skills` 存的是 id ⇒ 旧绑定**静默**失效）；核对必须**只读直开** `{ readonly: true }` ✓。实测 5 行全 `NULL` ⇒ 改名零影响 ✓。
- **坑⑨条已下移**（渲染/前端 import type/路由注册顺序/回读校验/回显规范化/DB 清洗副作用/守卫触发）⇒ `TOPICS.md` §Skill 体系坑清单（2026-09-15 腾 8k 预算：本文件逼近上限时**尾部区块最先被截断**）。

## 视频提示词语料
**详见 `TOPICS.md`**（检索管线 8987 条/3 源、已排除源清单、落盘三分都在那儿）。一条红线：他人提示词正文**不得搬运进仓库**（只提炼范式，结论落 `docs/`）。

## 代码约定（写代码时的硬规则）
- **中文文案里要引用就用「」，绝不用半角 `"`** —— 文案本身是双引号串，嵌 `"` 直接 `SyntaxError`（2026-09-17 一天犯了 3 次）。同类：改文案后**顺手搜一遍引用它的断言**（`check("…文案…")` 会因改词而失效）。
- **测试不许依赖"真机装了什么"**：断言写成**两种世界都成立**（如「真张量 ✓ + 画面仍不真 ✗」✓）；缺依赖/缺服务那类路径用 **monkeypatch 模拟**，不要写成「必然缺」——装上/起来就红。断言也别写**套套逻辑**（`sum(x) == sum(x)` 恒真，只增通过数不增信息）。
- **判"代码里有没有某种写法"用 AST，别用正则** ✗（正则会把**文档串里的说明**当代码 ✓、又漏掉**换了写法**的同类 ✓）⇒ 守卫三段：合同锚点 + **正/负对照** + 扫描面非空 ✓。
- **归一化产物与判断常量必须同源** ✗（`-` vs `_` 实测导致"待重做资产被排除出生成顺序"✗）；断言钉**后果**，不只钉计数 ✓。
- **"schema 里没有"先找专用表再说** ✗（实测 `continuity_states` 早就在 ✓）⇒ **先找现成的家，别急着盖房子** ✓。
- **"没数据/没读到" ≠ 通过** ✗：纯函数"没给就跳过"⇒ **空集会被读成绿灯** ✓✗ ⇒ 外层补**阻断**（"无从体检 ≠ 通过"）+ 报**覆盖率** ✓。

## 本机环境（2026-09-17 实测）
- **开发机无 NVIDIA 显卡** ✗（Iris Xe 集显 ✓，无 `nvidia-smi` ✓）⇒ 只装 **CPU 版 torch**（124 MB ✓），别装 2.5 GB CUDA 轮子 ✗。**真推理（H3 19.53 GiB 权重）要在工作站（A5000）跑** ✓ ⇒ 见下面那条工作方式。
- **PyPI 在本机下不动** ✗（两次卡在同一文件、无报错）⇒ 一律 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（20–58 MB/s ✓）；装 torch：`--index-url https://download.pytorch.org/whl/cpu --extra-index-url <镜像> torch==<ver>+cpu` ✓。
- 长命令常被判「在后台运行」⇒ **先查产物再决定重跑** ✗（它可能真跑完了 ✓）；批次日志**标签不可复用** ✗（复用会读到上次会话的陈旧结果 ✓）。

## 工作方式（用户 2026-09-17 明确）
- **以实现功能为先**：先把能力在代码里**实现完**（含测试与守卫 ✓），**等实现完再去工作站跑真流程** ✓ —— 不要为了"当场看到出片"而反复折腾本机环境 ✗（本机也没有 NVIDIA 卡 ✓）。接口/后端可以按"工作站上才真跑"来设计 ✓，但**不许**因此把未验证的部分说成已验证 ✗。

## 协作与提交
- **未经用户明确要求，绝不 `git commit`**；改完展示 diff。上下文过大时按阶段拆：每阶段只读 1 文件、只改 1 处、逐步验证。
- git 身份 `yuanxf`/`yuanxf@wedoctor.com`；远端 `git@github.com:yuanxiufei/darme-voide.git`。PowerShell 传中文 commit message 会乱码 → 统一英文（长 message 写 `tmp/*.txt` + `git commit -F`）。
- **换行**：`.gitattributes` **只**声明 `.githooks/* text eol=lf`（**不加 `* text=auto`**，避免全仓 renormalize 噪声）—— 钩子由 sh 执行，CRLF 会让 shebang 变 `#!/bin/sh\r` ⇒ Windows 上直接报 command not found。改钩子后查 `git ls-files --eol <file>`（须 `w/lf`）。
- **`.codebuddy/memory/` 无 gitignore 规则，且现已全部纳管**（`MEMORY.md`/`TOPICS.md`/`INDEX.md` + 各日日志，自 2026-09-12 的 `abc6cac` 起同批提交）⇒ **新增日志 / 改索引后要随同批提交**，否则 `MEMORY.md` 的读法指针在新克隆上**断链**；状态用 `git ls-files` / `git check-ignore -v` 复查。
- **`execute_command` 拉大文件/跑大正则易被转后台丢 stdout** → 用 `[System.IO.File]::WriteAllText(path, content, UTF8)` 落盘再读；`Get-Content` 必须显式 `-Encoding UTF8`（否则中文在控制台显示为乱码）。
