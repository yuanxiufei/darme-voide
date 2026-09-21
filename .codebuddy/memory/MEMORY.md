# MEMORY — 长期记忆

> 只留**会导致 bug 的不变量**与约定；细节看 `docs/`、代码。**读法：本文件（必读）→ `TOPICS.md`（低频长专题）→ `INDEX.md`（日志定位；默认只读索引、按 `@行号` 跳读日志，勿整读）**。**本文件须 ≤8k 字符**，超限会被注入截断（实测 9.4k 即截断，且断在半句——**尾部 `## 协作与提交` 最先丢**）。**逼近上限时把细节下移 `TOPICS.md`，勿硬塞**。

## 项目与运行
Drama Studio（`d:/code/voides/voide-darme`）：AI 剧本/分镜/视频。Nuxt 3（`frontend/app`）+ **FastAPI + SQLAlchemy Core**（`backend-py/`，唯一后端）。⚠️ **Node 后端（Hono + Drizzle + better-sqlite3，`backend/`）已于 2026-09-15 删除**，能力 100% 迁到 `backend-py/`。
- 后端 **5789**（`config.ts`）、前端 **3013**（proxy `/api`、`/static`）；前缀 `/api/v1`，另有 `/webhooks/*`、`/static/*`；无登录页。页面 `/`、`/settings`、`/drama/[id]`、`/library/*`。
- 启动：`cd backend-py && .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790`；`cd frontend && npx nuxt dev --port 3013`（dev 代理已指 5790）。数据根 `.data-root` > `DATA_ROOT` > `config.yaml database.path` > `./data`；不依赖 postgres/redis/qdrant。（Node 5789 已随 `backend/` 删除，仅历史。）
- **前端 dev 代理现在指向 Python 后端 5790**（2026-09-15 起；旧 Node 5789 可用 `NUXT_API_TARGET` 临时覆盖）；**共享契约类型在前端** `frontend/app/types/contracts.ts`（前端侧镜像，**字段权威在 Python 后端**，改后端字段要同步它）。技能库 / 脚本 / 快照也都在 `backend-py/` 下。
- **`backend-py/` = Python 后端**（FastAPI + SQLAlchemy **Core**；**端口 5790**；全部域已迁完 / 未注册 0）：回归 `backend-py/tests/run_all.py`；动手前读 `backend-py/README.md`。⚠️ **`backend/` 已删**（2026-09-15）⇒ TS 原文只在 `frozen_ts_source.py`。**明细见 `TOPICS.md`** ✓

## 本地模型 + H3 视频推理
**详见 `TOPICS.md`**。仅三条必须记牢：**直连 HF 全超时 → 必须 `hf-mirror.com`**；H3 走 ComfyUI(8188) + 8765 薄封装（`runtime='h3'`、`baseUrl='http://localhost:8765'`）、六键 Bible 跨集锁定；⚠️ 该链路 provider 名 `minimax` 是**服务商标识**，与 `backend-py/app/skills/` 外部技能库**无关**。GPU RTX A5000 22 GiB，**无 nvcc**。

## 后端能力（9 项）
QC（technical/consistency）｜asset-versions｜style-profiles｜script-fingerprint｜take-budget｜rhythm-phase｜jianying-draft｜estimate-service｜usage-tracking —— **明细见 `TOPICS.md`**。
**约定**：`appendStyleProfile()` 同步；门禁统一支持 `force`；无指纹/无相位视为旧产物不阻断；`storyboards` 无 resolution/fps（在 `video_generations`）；Windows ZIP 内路径转 posix。

## 前端约定
- 复用型 `refresh()` **禁止写 UI 定位副作用**（切 panel/tab、重置选中、重置编辑中表单）——被刷新按钮与所有生成/编辑回调反复调用（含合成 poll 每 4s）→「画面跳转」；UI 定位只在 `onMounted` 一次。refresh 重赋列表后选中项须**按 id 重绑**，未存表单值需 **dirty 标记**保护。
- playwright-cli 常被转后台丢 stdout → `... *> .log` 落盘再读；eval 输出**只用 ASCII**；用 `nav button[i]` 定位。**切勿一边改文件一边跑点击测试**（HMR 会造成「修复无效」假象）。**元素计数与 SFC 编译验证见 `TOPICS.md`**。

## 画风体系（10 种）
- key：realistic / cinematic / noir / anime / ghibli / ink-wash / watercolor / comic / cyberpunk / pixar3d。**词表四段**「画风核心+镜头光线+调色质感+画质」，**负面词负责排除对立风格**。
- ⚠️ **单一事实来源 2 处必须同步**：后端 `backend-py/app/services/prompt_utils.py`（`ART_STYLE_CATALOG` + 三张映射表）与前端 `frontend/app/utils/artStyles.ts` ⇒ **加画风只改这 2 个文件** ✓。旧指针 `shared/prompt-utils.ts` **已随 `backend/` 删除** ✗（见到它=在读旧文 ✓）。
- 其余（解析链唯一入口 `resolveEffectiveArtStyle` ✓、正负成对收口 ✓、找副本顺序 ✓、反 AI 感白名单 ✓）⇒ **`TOPICS.md` §画风体系细节** ✓

## Skill 体系（详见 `backend-py/app/skills/README.md`，改前先读）
- **库由声明文件识别，与目录名解耦**：`<lib>/library.yaml` 的 `name`/`label`/`description` ⇒ **加库/换库/改展示名零代码**。⚠️ **无执行入口的 skill 勿写进 `agents:`** ✗（每次生成白背一段上下文）；**词库按介质分家**（图像｜视频）⇒ **改词库前先确认改哪个** ✓（清单见 `TOPICS.md`）。
- **命名**：core 目录名 = Agent 类型（`agent_configs.agent_type`，**勿改名**）；库内 skill id **勿重命名**（`references/` 互引静默断链）。
- **绑定 = skill 自描述**：frontmatter `agents: [...]` + `priority`（缺省 100）⇒ **改绑定 = 改 md**；入口唯一 `resolveDefaultSkills` ✓
- **DB 配置优先铁律**：`parseSkillsConfig` 解析出配置即「用户已选过」→ **全关也不回退默认** ✗（仅 `null`/空/失败才回退）；`enabled` 缺省 = 启用 ✓
- **出厂默认唯一出口** `getAgentDefaults()` ⇒ **前端不得硬编码默认提示词/默认绑定** ✗
- **加载器只读 `SKILL.md`** ⇒ `references/` 对 agent **不可达（有意取舍，非 bug）** ✓
- **删除保护**：拒删 = **顶层 id** *且* **`agents:` 非空** ✗（core 删掉永久丢失）；解析失败 ⇒ 保守拒删 ✓
- ⚠️ **宿主工具**：只认 `hub_` 前缀 ✓，且**工具集须取 `tool.id`** ✗（取错 ⇒ 21 个工具全误判为缺失）
- ⚠️ **改名/挪库后必核对 DB 绑定** ✓（存的是 id ⇒ 旧绑定**静默**失效；核对须**只读直开** ✓）
- 注入闸默认只注自有 ✓、超预算按 priority 跳过给诊断 ✓、合计**只算 `enabled=true`** ✓
- **完整表述 / 坑⑨条 / 注入口径 全在 `TOPICS.md`** §Skill 体系 ✓（本文件只留会导致 bug 的判据 ✓）

## 视频提示词语料
**详见 `TOPICS.md`**（检索管线 8987 条/3 源、已排除源清单、落盘三分都在那儿）。一条红线：他人提示词正文**不得搬运进仓库**（只提炼范式，结论落 `docs/`）。

## 代码约定（写代码时的硬规则）
- **中文文案里要引用就用「」，绝不用半角 `"`** —— 文案本身是双引号串，嵌 `"` 直接 `SyntaxError`（2026-09-17 一天犯了 3 次）。同类：改文案后**顺手搜一遍引用它的断言**（`check("…文案…")` 会因改词而失效）。
- **解析外部工具真实输出前先把真实输出落盘取证**（别照文档猜 —— 技术 QC「三项死检测」就是猜出来的）；**一个布尔字段只许一个含义**（把「观察到的事实」和「推断出的结论」塞进同一个 flag ⇒ 文案会说谎）；**检测类改动要有「能触发」的反向证明**（另造一个必命中素材，否则「判定已生效」根本证明不了）。
- **测试不许依赖"真机装了什么"**：断言写成**两种世界都成立**（如「真张量 ✓ + 画面仍不真 ✗」✓）；缺依赖/缺服务那类路径用 **monkeypatch 模拟**，不要写成「必然缺」——装上/起来就红。断言也别写**套套逻辑**（`sum(x) == sum(x)` 恒真，只增通过数不增信息）。
- **判"代码里有没有某种写法"用 AST，别用正则** ✗（正则会把**文档串里的说明**当代码 ✓、又漏掉**换了写法**的同类 ✓）⇒ 守卫三段：合同锚点 + **正/负对照** + 扫描面非空 ✓。
- **归一化产物与判断常量必须同源** ✗（`-` vs `_` 实测导致"待重做资产被排除出生成顺序"✗）；断言钉**后果**，不只钉计数 ✓。
- **"schema 里没有"先找专用表再说** ✗（实测 `continuity_states` 早就在 ✓）⇒ **先找现成的家，别急着盖房子** ✓。
- **"没数据/没读到" ≠ 通过** ✗：纯函数"没给就跳过"⇒ **空集会被读成绿灯** ✓✗ ⇒ 外层补**阻断**（"无从体检 ≠ 通过"）+ 报**覆盖率** ✓。
- **检测器的模式要比解析器更宽松** ✗（复用同一个严格正则 ⇒ 解析不了的**畸形输入也检测不到** ⇒ 被当成「干净」✓✗）；**追加小节只锚「末节首行」** ✗（锚在正文中间会把新小节插到旧小节**前面**）。
- ⭐ **判据要"响亮"不要"静默"** ✗：惰性导出必配**名字名单**（`__all__`）且**先校验** ⇒ 漏加立刻 `AttributeError` ✓（别给空壳/None 让错误漂到下游 ✗）；**占位符不许留在模块体里** ✗、**断言优先写"能自己算出来的不变量"** ✗ ⇒ 其余（模块级状态 / `compile()` / 汇总分母 / 规模数字）见 `TOPICS.md` §代码约定坑清单 ✓
- ⭐ **同一份配置的两种形态（dataclass / dict）⇒ 每个读取点都要两处都认** ✗（取不到的那些会**悄悄回落默认值** ✓✗ ⇒ 与模型对不上）；**能算出 0 宽度的结构不变量要在构造期报** ✗（否则报的是**第三方后端的天书** ✓✗）；**多流各按落盘工具的契约报形状** ✗（对称去维 ⇒ 静默走错分支 ✓✗）

## 本机环境（2026-09-17 实测）
- **开发机无 NVIDIA 显卡** ✗（Iris Xe 集显 ✓，无 `nvidia-smi` ✓）⇒ 只装 **CPU 版 torch**（124 MB ✓），别装 2.5 GB CUDA 轮子 ✗。**真推理（H3 19.53 GiB 权重）要在工作站（A5000）跑** ✓ ⇒ 见下面那条工作方式。
- **PyPI 镜像 / 长命令被判后台 / 跑批编码坑（`*>>` 写 UTF-16 ⇒ grep 0 命中；子进程 GBK ⇒ 打印炸冒充失败）** ⇒ **`TOPICS.md` §自 MEMORY.md 下移（2026-09-20 第三次）** ✓

## 工作方式（用户 2026-09-17 明确）
- **以实现功能为先**：先把能力在代码里**实现完**（含测试与守卫 ✓），**等实现完再去工作站跑真流程** ✓ —— 不要为了"当场看到出片"而反复折腾本机环境 ✗（本机也没有 NVIDIA 卡 ✓）。接口/后端可以按"工作站上才真跑"来设计 ✓，但**不许**因此把未验证的部分说成已验证 ✗。

## ⭐ 自研优先（用户 2026-09-20 明确要求记住）
- **原话**：「**所有功能不要对外依赖，自己实现所有的功能，要参考我给你的几个项目**」⇒ 落成三条硬约束：
  1. **能力自研**：功能要能在**本仓自己实现**（推理/生成/解析/合成都算 ✓）⇒ 不许把关键能力**只**挂在外部队商 API 上 ✗；
  2. **不对外依赖**：默认形态是**离线可跑**（本地服务 / 本地权重 / 纯计算 ✓）⇒ 外部队商适配器**可以有**（作为可选通道 ✓），但**不能是唯一出路** ✗；
  3. **参考 `reference/` 那几个项目**（Mini-Agent / minimax-desgin-plugin / ollama-python / ollama / Open-AI-Micro-Drama-Generator / short-drama-agent ✓）：**有用的功能搬过来自己实现** ✓（判据同「没人调用的库不算功能」✓：搬来要真接线 ✓）。
- **⚡ 2026-09-20 加严**：「**所有都要自己实现，不要调用外部的**」⇒ 外部队商 API **不是可选通道，是要去掉的** ✗。
  边界（可纠正 ✓）：**功能/能力**不外包 ✗；`ffmpeg`/SQLite/标准库/框架属**本地基础设施** ✓ 不算 ✗。
- ⚠️⚠️ **2026-09-20 实测：四类生成当前**全部**解析到外部** ✗✗**（`api.minimax.chat` / `ark.cn-beijing.volces.com` ✓）——
  `ai_providers.py` 按 **priority 降序**取第一条 ✓，本地预设 82–85 ✓ 输给厂商 97–300 ✓ ⇒ **厂商永远赢** ✗。
  ⇒ 动作：抬本地 priority / 删外部行 ✓（**审计表与三层依赖阶梯见 `TOPICS.md` §外部调用审计** ✓）。
- **现状**（见 `TOPICS.md` §自研引擎现状）：引擎 **25 模块 / 3500+ 用例全绿 ✓ 零依赖可跑** ✓
  （2026-09-20 新增：**GGUF 读取器** `engine/gguf.py` ✓；**H3 键名核对器** `engine/h3_keys.py` ✓；
  **自研字节级 BPE** `engine/tokenizer_bpe.py` ✓；
  ⭐ **自研 Unigram/WordPiece/Metaspace** `engine/tokenizer_own.py` ✓（Viterbi ✓ / 整词 UNK ✓ /
  **normalizer 11 种** ✓ / **预分词器 8 种** ✓ 含 `Punctuation` 五种 behavior ✓ ——
  规则全部**逐例实测**对齐参考 ✓）；
  ⭐ **对 `transformers` 的运行时改造** `engine/tokenizers_tuning.py` ✓（离线兜底含 **Auto 工厂** ✓
  / 缓存目录 / 降噪 / 计数 / 可撤 ✓ 幂等 ✓）；
  **分词器总入口** `engine/tokenizer_hub.py` ✓ = 形态嗅探 + **3 模型 × 8 预分词器全自研** ✓ +
  极少数形态回退 ✓ + 批量 + LRU + 运行期互校 ✓）；
  **H3 结构从权重推** ✓（`h3_keys.infer_h3_trunk_config` ✓ ⇒ 出厂常量只作回落 ✓）；
  出片唯一硬缺口 = **真权重** ✗（机制都已实现 ✓）。**全量回归 103 套 / 3540 项 / 0 失败** ✓
  （2026-09-21 又扩 normalizer + 预分词器 ✓ 数字待重跑 ✓）。
- ⚠️ **`transformers` 已装（5.17.0 = PyPI 最新 ✓ 镜像 ✓）且登记为「可选依赖」** ✓：`_Dependency.optional` ✓
  ⇒ `dependency_status()` 分 `missing`（必需 ✓ 缺则后端不可用）/ `optionalMissing`（可选 ✓）
  ⇒ **别把可选塞进必需位** ✗（`ready`/`torch_available()` 会被判死 ✗✗）。
  ⚠️ **不 fork / 不 vendored** ✗（许可允许但不必要 ✓）：只做**运行时改造** ✓ ——
  离线兜底 ✓ 缓存目录 ✓ 降噪 ✓ 计数 ✓ 可撤 ✓；⚠️ **Auto 工厂必须单独包** ✗
  （它在解析出具体类**之前**就外呼 ✓✗）；分词本身已由**自研三血统**接管 ✓ ⇒ 它只剩核对价值 ✓。
- ⚠️ **自检汇总行必须写 `SUMMARY: n/m passed`** ✗（`run_all.py` 按此前缀收敛项数；写成「n/m 项通过」⇒ 总表**空摘要**、项数缺一套 ✓✗）。
- **⚫ 可照抄项目（用户点名 ✓「不要忘记」）**：ComfyUI / minimax-h3-comfyui / ollama / ollama-python /
  minimax-desgin-plugin ⇒ **功能直接抄** ✓（落点见 `TOPICS.md` §可照抄项目清单 ✓）。

## 协作与提交
- **未经用户明确要求，绝不 `git commit`**；改完展示 diff。上下文过大时按阶段拆：每阶段只读 1 文件、只改 1 处、逐步验证。
- git 身份 `yuanxf`/`yuanxf@wedoctor.com`；远端 `git@github.com:yuanxiufei/darme-voide.git`。PowerShell 传中文 commit message 会乱码 → 统一英文（长 message 写 `tmp/*.txt` + `git commit -F`）。
- **换行**：`.gitattributes` **只**声明 `.githooks/* text eol=lf`（**不加 `* text=auto`**，避免全仓 renormalize 噪声）—— 钩子由 sh 执行，CRLF 会让 shebang 变 `#!/bin/sh\r` ⇒ Windows 上直接报 command not found。改钩子后查 `git ls-files --eol <file>`（须 `w/lf`）。
- **`.codebuddy/memory/` 无 gitignore 规则，且现已全部纳管**（`MEMORY.md`/`TOPICS.md`/`INDEX.md` + 各日日志，自 2026-09-12 的 `abc6cac` 起同批提交）⇒ **新增日志 / 改索引后要随同批提交**，否则 `MEMORY.md` 的读法指针在新克隆上**断链**；状态用 `git ls-files` / `git check-ignore -v` 复查。
- **`execute_command` 拉大文件/跑大正则易被转后台丢 stdout** → 用 `[System.IO.File]::WriteAllText(path, content, UTF8)` 落盘再读；`Get-Content` 必须显式 `-Encoding UTF8`（否则中文在控制台显示为乱码）。
