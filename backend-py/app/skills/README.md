# backend-py/app/skills/ 目录约定

本目录存放 **Agent 的提示词层（Skill）**，只有两种角色：**项目自有 skill** 与 **外部技能库**。

- 加载器：`backend-py/app/agent/skills.py`
- 列表 / 编辑 API：`backend-py/app/routers/skills.py`（`/skills`、`/skills/meta`、`/skills/{id}`）
- 前端管理页：`frontend/app/pages/skills.vue`

## 一、目录结构

| 路径 | 角色 | 默认注入 |
|---|---|---|
| `backend-py/app/skills/<name>/SKILL.md` | **自有 skill（core）** — 与工作流、工具名、字段契约严格对齐 | ✅ 由该 `SKILL.md` 的 `agents:` 声明 |
| `backend-py/app/skills/<lib>/library.yaml` + `backend-py/app/skills/<lib>/**/SKILL.md` | **外部技能库（vendor）** — 库内 skill 一律不默认注入 | ❌ 需在「Agent 配置 → 绑定 Skills」手动启用 |

**库靠显式声明识别，不靠目录名、也不靠层级猜测**：顶层目录里含 `library.yaml` 就是一个技能库，
库标识取声明里的 `name`（缺省 = 目录名）。于是**加库 / 换库 / 改中文展示名都是纯文件操作，零代码**。

```
backend-py/app/skills/
├── README.md
├── <9 个自有 skill>/SKILL.md
├── genre-templates/          # 片型模板库（9 个）—— 按「成片类型」组织的端到端入口
│   ├── library.yaml
│   └── <skill>/SKILL.md
├── production-tools/         # 制作工序库（20 个）—— 制作链路上的单点能力与工序工具
│   ├── library.yaml
│   └── <skill>/SKILL.md
├── h3-canvas/                # H3 画布提示词库（1 个）—— 2026-11-02 移植自 ComfyUI-Local-Infinite-Canvas（MIT）
│   ├── library.yaml
│   └── h3-prompt-writing/SKILL.md   # 正文引用 references/*.txt ⇒ 只作 vendor 库按需启用
└── wuxia-action/             # 武打编排库（2 个）—— 判据自建、原文不搬
    ├── library.yaml
    ├── combat-choreography/SKILL.md  # 打戏四拍动作链 + 走位轴线
    └── ultimate-moves/SKILL.md       # 大招五段结构 + 属性视觉语法
```

> `wuxia-action` 两个 skill 的方法来源是原料库的 combat 77 份 ＋ ultimate 11 份
> （`corpus/playbook/film-expert-bridge/`），
> 那份目录 2026-11-04 起**随仓库分发**（此前在 gitignore 的 `方法语料/` 里 ⇒ 新克隆的仓库检索为空）：
> 本库仍只装归纳出的判据，查原文走 `search_script_library(source="playbook")`。
> 同日「有用的抄没用的删」：打戏三件（受击反馈模板／打斗运镜 15 种／招式命名与距离节奏）
> 已消化进 `wuxia-action/combat-choreography/references/hit-reaction-and-moves.md`，
> 出稿不必再检索引擎；原料库其余部分（特效库、造型库、招式库、技法资料库）可检索。
> `combat-choreography` 带一份**自写**附件 `references/hit-reaction-and-moves.md`（受击反应模板／
> 打斗运镜 15 种／招式命名与距离—节奏表，85 行，消化自原料库而非原文搬运）⇒ 它**依赖
> `read_skill_ref`** 才能读到那份；`ultimate-moves` 则没有附件、全靠正文与检索。
>
> ⚠️ **它的 `agents: []` 是刻意的，不是漏写**：库内 skill 声明 `agents:` 一律不生效（见 §四），
> 要启用只能在前端「Agent 配置 → 绑定 Skills」勾选。此前这里写过
> `agents: [screenwriter, storyboard_breaker]` —— 两处错叠在一起（`screenwriter` 根本不是合法
> Agent 类型 ＋ 库内声明本就无效），症状都是「静默不注入」，`skills_test.py` 已加两条判据守它。

### 库声明文件 `library.yaml`

```yaml
name: genre-templates   # 库标识：API 与前端分组的稳定 key（缺省 = 目录名）；改它 = 改引用，慎改
label: 片型模板库        # 展示名：可随时改，不影响任何引用
description: …          # 侧栏悬浮提示
```

**`name` 与 `label` 分离**是这里最实用的一点：目录名 / 库标识保持稳定，对外展示名随时可调。

**但 `name` 必须两两唯一**（2026-11-06 补断言）：它是前端分组键 `lib:<name>`，而前端取库用的是
`vendorLibs.find(l => \`lib:${l.id}\` === type)` —— **只取第一个**；`inGroup` 又用 `lib:${s.source}` 比对，
两个库声明同一个 `name` 时两边都成立 ⇒ 两库 skill 在侧栏**并成一组**，组名/描述只显示第一个库的。
全程不报错、不告警，症状只是「这个库里混进了别的库的 skill」，肉眼极难发现。
`skills_test.py` 现守四件事：`name` 唯一、不含 `:` 与 `/`（会破坏 `lib:` 前缀解析）、声明没有 `name`
时回退目录名**不算错**（设计允许解耦）、以及真树确实扫到多个库（别是空跑）。
解析规则直接复用 `app/routers/skills.py::_read_library`，**不在测试里重写一遍**（重写那份迟早漂移）。
已验证：把 `wuxia-action` 的 `name` 临时改成 `production-tools`，真树断言当场红并点名两个冲突目录。

**顶层目录的两种退化形态**（同日再补一层）：前端侧栏选项是
`['all', ...agentDefs, ...vendorLibs.map(lib:<id>)]`，而 `vendorLibs` 只收**带 `library.yaml`** 的
目录；每条 skill 的 `source` 却由 `_classify()` **独立推导**（顶层有 `SKILL.md` ⇒ `core`；有
`library.yaml` ⇒ `vendor`+声明 id；两者都没有 ⇒ 兜底 `vendor`+目录名）。两处一旦错位，界面就少了东西：

| 形态 | 后果 |
|---|---|
| 容器目录漏写 `library.yaml`（自身无 `SKILL.md`、下面却挂着 skill） | 兜底判成 `vendor`/目录名，但侧栏压根没有这个分组 ⇒ 那些 skill **只在「全部」可见**，按库浏览永远找不到（前端另有一份 `valid` 白名单，硬拼出 `lib:<目录名>` 也会被重置） |
| 顶层目录**同时**有 `SKILL.md` 与 `library.yaml` | `_classify()` 先判 `core`，侧栏却仍为它渲染 `lib:<id>` 分组 ⇒ **那个分组点开永远是空的** |

这和「库标识重复」是同一个病：分组由两处独立推导拼出来，任一处漏项都不报错，只表现为
「界面上少了东西 / 空了一块」。`skills_test.py` 再守一条真树断言 ＋ 五种反验（漏声明／补声明／
两者齐备／删声明后合法／空库合法）。真树验证过：给 `extractor` 加一份 `library.yaml` ⇒ 报
「库分组恒空」；造一个 `_probe_lib/child/` 漏声明 ⇒ 报「其下 1 个 skill 只在「全部」可见」。
两者跑完均自动清理。
注意**建库但还没装 skill 属合法中间态**，不报错 —— 否则新建库的第一步就红。

### 上游清单 `meta.yaml`（本项目**不读取**，仅供溯源）

`production-tools/` 下 20 个 skill **各带一个** `meta.yaml`（`genre-templates/` 与 7 个自有 skill **均无**）。
它是**上游 Hub 技能市场的上架卡片**，字段为 `display-name-zh` / `version` / `tag-en|cn` /
`complete-tags-*` / `summary-*` / `desc-*` / `cover` / `author-*` / `source`。

- 本项目**只读 `SKILL.md`**（加载器）、**只读顶层 `library.yaml`**（`backend-py/app/routers/skills.py`），
  全仓库没有任何 `*.yaml` 泛化扫描 ⇒ **改 `meta.yaml` 不会影响任何行为**。
  ⚠️ 它和 `library.yaml` 性质完全不同 —— **不要把它当配置改**。
- 但它是 `version` / `display-name-zh` / `author-*` / `source` 的**唯一载体**
  （`SKILL.md` 的 frontmatter 里没有这些字段）⇒ **删除即丢失上游溯源与版本对齐能力**，请保留。

## 二、绑定关系是「自描述」的

skill 服务哪个 Agent **写在 `SKILL.md` 的 frontmatter 里**，代码中**不维护**任何「谁绑谁」的映射：

```yaml
---
name: prompt-style-library
description: ...
agents: [storyboard_breaker, grid_prompt_generator]   # 默认注入哪些 Agent（不写 = 不默认注入）
priority: 20                                          # 注入顺序，越小越靠前（默认 100）
---
```

于是「调整默认绑定」= 改 `SKILL.md`，**零代码改动**；绑定与 skill 同生共死
（不会出现「代码里绑了一个已被删除的 skill」这种漂移）。

三处消费同一个解析结果（`resolveDefaultSkills`）：

| 位置 | 用途 |
|---|---|
| `backend-py/app/agent/skills.py` → `loadAgentSkills` | DB 无配置时决定实际注入哪些 skill |
| `backend-py/app/agent/runtime.py` 的 `get_agent_defaults`（+ `agent_registry.py`） | `GET /agent-configs/defaults` 的出厂默认值 |
| `backend-py/app/routers/skills.py` → `/skills`、`/skills/meta` | 反向查询「该 skill 被谁绑定」与前端侧栏分组 |

**名册只有一个：`agent_registry.VALID_AGENT_TYPES`（当前 6 个，含 `orchestrator`）。**
`AGENT_PHASES` 是「阶段展示表」（5 条，不含 `orchestrator`），**不是名册**——
凡是要「把 Agent 全量过一遍」的地方（反向绑定枚举、`/skills/meta` 的 agents 列表、
各测试的期望条数），都必须枚举 `VALID_AGENT_TYPES`；`AGENT_PHASES` 只允许用来取
`phase` 字段（取不到时给 `""`）。

反面案例（2026-11-06 修）：`/skills` 的反向绑定与 `/skills/meta` 曾用 `AGENT_PHASES`
枚举 ⇒ `agents: [orchestrator]` 的 skill **永远显示「未绑定」**；同时前端
`agents.vue` 硬编码 5 条 Agent 卡片 ⇒ `orchestrator` 没有卡片、模型/提示词/Skill
在 UI 上根本改不了。三处都是**静默**的：页面看着正常、运行期
`build_agent_config('orchestrator')` 照读 DB 配置注入，没有任何一处报错。
修法：三处统一到 `VALID_AGENT_TYPES`；前端名册改为由
`GET /agent-configs/defaults` 派生（`AGENT_ICONS` 只留图标，不再存名字），并把它
`catch` 里「非关键」的静默吞错改成 `toast.error`（名册由它派生后就再也不是非关键了）。

反验口径（同一坏状态「旧绿新红」才算钉住）：把 `meta` 的枚举改回 `AGENT_PHASES`
⇒ `skills_test` **75/77**，精确点出 agents 列表少了 `orchestrator`；复原 ⇒ **77/77**。
`AGENT_PHASES` 与 `VALID_AGENT_TYPES` 的差集也被测试钉成恰为 `{"orchestrator"}`
——将来谁新增 Agent 只登展示表、忘了登名册，会当场红。

## 三、注入规则

1. DB `agent_configs.skills` 有值 → **只**按 DB 配置加载（前端「Agent 配置 → 绑定 Skills」写入）
2. DB 为 `null` → 回退各 `SKILL.md` 自描述的默认绑定（`agents:` + `priority:`）
3. 所有注入受**字符预算**约束：默认 60000 字符，超出部分按优先级跳过并在注入文本末尾注明；
   用环境变量 `AGENT_SKILL_BUDGET` 覆盖（设为 `0` 关闭限制）

实测（**2026-11-05 重测**）：名册 6 个 Agent 中**有默认绑定的 5 个**合计 **34,875 字符、零跳过**（预算 60,000，余量 42%）
（`storyboard_breaker` 22,033 / `grid_prompt_generator` 9,281 / `script_rewriter` 1,592 /
`extractor` 1,078 / `voice_assigner` 891；`orchestrator` 无默认绑定，注入 0 字符）。
口径：各 Agent `load_agent_skills` 的返回长度，逐次稳定可复现（绑定按 `(priority, id)` 排序）。
（上一轮记的 30,793 是 2026-09-25 的值；此后 `storyboard_breaker` 侧的正文在持续增补，
这个数只会往上走 —— 判断配额余量看「离 60,000 还有多少」，别把增量当回归。）

> **2026-09-12 两轮体量治理**（结论只对**默认绑定**成立；DB `agent_configs.skills` 一旦有值就按 DB 走，见 §三.1）：
>
> **① 按介质拆分**：`prompt-style-library`（27 KB，逼近 §七.5 的 30 KB 上限）原含**图像词库 +
> 中文视频范式**两套内容，而它绑定的 `grid_prompt_generator` 是**纯出图** Agent —— 视频范式
> （6.0–6.10，约占该文件 55%）对它无用。已把第 6 节**整体**拆为独立 `video-prompt-library`
> （`agents: [storyboard_breaker]`），原库仅留占位说明以保持 §7/§8 编号不变。
>
> **② 停掉无入口的默认注入**：`style-reference-reverse`（参考图反推画风）**没有任何执行入口**
> —— 后端 0 处引用其 protocol 字段、前端 0 处「反推」文案、也无专属 Agent ⇒ 绑在
> `grid_prompt_generator` 上纯属白占上下文。已改 `agents: []`
> （**文件保留**、UI 仍可手动绑；将来接入该能力时改回 `agents:` 即可）。
>
> **累计**：总注入 **37,263 → 28,128（−24.5%）**；`grid_prompt_generator` **18,898 → 8,517（−54.9%）**；
> `storyboard_breaker` 15,315 → 16,561。
>
> 其中 `storyboard_breaker` / `grid_prompt_generator` 各有 **+516 / +294** 是**有意加的**：
> 把 `search_reference_prompts`（本地语料检索工具，见 `backend-py/app/agent/tools/corpus_tools.py`）
> 的**使用时机与红线**写进了这两个 skill —— **工具注册了不等于模型会调用**，
> 不写进 skill 就等于白注册。多花 810 字符换「9000 条语料真被用上」，这笔是值的。

> **2026-09-25 补缺**：新增 `audio-prompt-library`（`agents: [storyboard_breaker]`、`priority: 40`）
> —— 分镜的 `sound_effect` / `bgm_prompt` 此前**只有字段定义、没有写法与词表**，是提示词层的一处空档。
> 渲染后 **+3,135 字符** ⇒ `storyboard_breaker` 16,311 → 19,446、五 Agent 合计 27,658 → 30,793
> （上面 2026-09-12 那组是治理当天的口径，当前值以 §三 开头的实测行为准）。
> 该库的核心判据是两个字段**消费面完全不同**：`sound_effect` **会进视频提示词**
> （入队时被追加为 `[background_audio] …`，成片同步合成音轨）⇒ 绝不能写配乐；
> `bgm_prompt` **不进任何生成链路**，只入库给后期选曲 ⇒ 不能写曲名 / 歌手 / 在世作曲家。

> ⚠️ **加 skill 前先问一句：它有没有执行入口？** 没有（无 Agent / 无 UI 触发点 / 无代码消费其
> protocol 字段）就不要写进 `agents:` —— 那只会让每次生成都背上一段用不上的上下文。

## 四、为什么不默认注入外部技能库

外部库需手动启用（在库内 skill 的 frontmatter 写 `agents:` 不会生效 —— 默认绑定只扫自有 skill），原因：

1. **体量失控**：单文件最大 42 KB。曾使 `storyboard_breaker` 单次注入 ≈68 KB、
   `grid_prompt_generator` ≈81 KB，与 `DEFAULT_PROMPTS` 中的同主题规范重复；
2. **语义错位**：它们是"按触发词独立启动"的会话式技能（自带 `STEP 1/2/3`、`Not for: …`），
   不是"常驻系统提示词层"；
3. **注入不完整**：其正文大量引用 `references/*.md`，而加载器**只读 `SKILL.md`**
   → 默认注入只能拿到正文、拿不到附件，等于注入半截指令。
   （**自有 skill 也会这样**：`audio-prompt-library` 带 1 份、`video-prompt-library` 带 4 份
   `references/`。差别只在「自有 skill 由本仓作者控制正文长度、外部库是上游原文」，
   通路是同一条。）
   （2026-11-04 起补了一条**按需**通路：`list_skill_refs` / `read_skill_ref` 两个只读工具
   ——见 `backend-py/app/services/skill_refs.py`。它只开放 `<skill>/references/` 下白名单后缀、
   单文件 ≤512 KB、单次 ≤400 行，且目前只挂在 `script_rewriter`（剧本改写）与
   `storyboard_breaker`（分镜拆解）两个 Agent 上 —— 挂载点是 `runtime.py` 的 tool_set
   `script` / `storyboard`，**不认 skill 名也不认 `agents:`**；
   **不是**默认注入的替代品 —— 模型得先知道要看哪个文件，才谈得上按需读。）

## 五、命名约定

| 对象 | 约定 | 原因 |
|---|---|---|
| 自有 skill 目录名 | **= Agent 类型**（`script_rewriter` 等）或领域名（`prompt-style-library`） | Agent 目录名是系统标识，与 `agent_configs.agent_type` 绑定，改名要动 DB 与多处代码 |
| 技能库目录名 / 库标识 | 语义化、来源中性（`genre-templates` / `production-tools`） | 不含平台品牌、不含"内置/安装"这类渠道概念 —— 换来源不必改名 |
| 技能库展示名 | 只写在 `library.yaml` 的 `label` | 目录名稳定，展示可随时调整 |
| 库内 skill id | **保持库的原始命名，不重命名** | 库内 `references/` 与 skill 之间会互相引用，改名会静默打断引用 |

## 六、已知语义重叠（外部库原样保留，需要时人工取舍）

以下为同名主题、可互相替代的 skill，本次**未合并**（属外部库内容，擅自合并会破坏其自洽性与内部引用）：

- 视频拆解：`genre-templates/video-deconstruct` ↔ `production-tools/video-deconstruct-analyzer`
- 短剧编剧：`production-tools/short-drama-screenwriter` / `short-drama-series-writer` /
  `chinese-style-short-drama-generator`
- 分镜系列：`production-tools/storyboard` / `n-storyboard` / `character-scene-storyboard` /
  `film-shot` / `multi-shot` / `coordinate-camera-control-designer`

如需精简，建议先在「Agent 配置 → 绑定 Skills」确认无人绑定后，再删除对应目录（删库内目录不影响代码）。

## 七、新增自有 SKILL

1. 建目录 `backend-py/app/skills/<name>/`（目录名即 skill id，小写 + 连字符或下划线），内含 `SKILL.md`
2. frontmatter 填 `name` / `description` / `preconditions` / `protocol` / `workflows`，
   需要默认注入时再加 `agents:` 与 `priority:`
3. 只写**领域知识**（词库、范式、判定规则），**不要重复** `DEFAULT_PROMPTS` 里的身份与工作流
4. 引用的外部资源文件（`references/` 等）**不会被加载**，需内联进 `SKILL.md` 或改用工具读取
5. 单文件建议 ≤ 30 KB；超过说明应拆分职责或改为按需加载

前端「Skill 管理 → 选中某 Agent → 新增 Skill」会在 `backend-py/app/skills/<agent>/<id>/` 下创建骨架
（模板自带 `agents: []` = 不默认注入，仅手动绑定）。

## 八、新增 / 替换外部技能库

在 `backend-py/app/skills/` 顶层建一个目录，放入 `library.yaml` 与 skill 子目录即可：

```
backend-py/app/skills/<lib>/library.yaml      # name / label / description
backend-py/app/skills/<lib>/<任意层级>/<skill>/SKILL.md
```

后端按声明自动识别为 vendor、前端侧栏自动出现该库分组 —— **零代码、零配置**。
删除同理：移除整个库目录即可（若有 Agent 手动绑定过其中的 skill，其注入会自然跳过）。

## 九、相关位置

- 画风词表（单一事实来源）：`backend-py/app/services/prompt_utils.py`
- 图像提示词范式（七段结构 / 镜头 / 光线 / 调色 / Danbooru tag）：`backend-py/app/skills/prompt-style-library/SKILL.md`
- 视频提示词范式（写法判定 / 时间码分段 / 散文式多段 / 中文标签 / 合规红线）：`backend-py/app/skills/video-prompt-library/SKILL.md`
- 分镜音频字段写法（`sound_effect` 同期声 / `bgm_prompt` 配乐方向的分工与词表）：`backend-py/app/skills/audio-prompt-library/SKILL.md`
- 提示词**取词来源**（人工维护用，**不注入 Agent**）：`docs/prompt-style-sources.md`
- 参考图反推：`backend-py/app/skills/style-reference-reverse/SKILL.md`（⚠️ 当前 `agents: []`，**未默认注入**，见 §三）

## 十、改动 skills 后必跑的自检

改名 / 挪库 / 删目录**都不会报错**：加载器只读 `SKILL.md`，正文里指向 `references` 目录的
那些文件引用一旦断链，只在读者真走到那一行时才表现为「指向空处」。所以有一条守卫：

```bash
python backend-py/app/scripts/check_skill_refs.py          # 退出码 1 = 存在致命断链
python backend-py/app/scripts/check_skill_refs.py --verbose # 额外列出被跳过的候选，审计盲区
```

**基线怎么读**：只看两个数 —— **致命 0 / 非致命 0**。文件数、待校验处数、三类跳过计数
**都在漂**（并行工作流随时往库里加 `references/`），写这份时是 154 文件 / 133 处 / 跳过 9·2·9，
隔天就不作数了；把它们当基线去追，只会每轮都在改同一个数字。
（历史：2026-09-25 起这份文档记过 93→96→107→114→116→118 处待校验，涨的都是"新加附件"与
"把某类跳过改判真校验"，从来没有一次是断链多了。）
**2026-11-04 补一条解析基准 ④**：`<另一个 skill 的目录名>/references/x.md` 按**本库 `skills/` 根**解析，
由「基准不明」转为**真校验**（这类「细节见某某 skill」的写法很常见，先前一直绿着也能指空）。
同名 skill 有多处时不猜，仍归基准不明 —— 猜错会把真断链判绿，比少判更糟。
**同日再补：通配符模式也进真校验**（`references/*.md` 走同一套基准 glob，命中 0 个 ⇒ 致命
「模式未命中任何文件」）。这类串原在 token 收集阶段就被丢掉 —— 连跳过计数都不进，
是「看着像路径、其实指空」里最彻底的一种。四条反验已固化成套件
`tests/skill_refs_check_test.py`（37 项，已进全量清单）：指错 ⇒ 退出 1；同名歧义 ⇒ 仍跳过且退出 0；
模式命中 ⇒ 过；模式空 ⇒ 致命；回执表头的扫描目录必须跟着 `SKILL_REFS_DIR` 变。
**同日三补：一个 skill 有两个名字，两个都收**。本仓 5 个 core skill 的**目录名跟着 agent type 走**
（`extractor` / `grid_prompt_generator` / `script_rewriter` / `storyboard_breaker` / `voice_assigner`，
因为 `agents:` 里的类型名就是那几个），而 frontmatter `name` 是上游 kebab
（`character-scene-extractor` / `storyboard-breaker` …）。**契约只有一条**：`id`（＝目录名）是标识，
`name` 只是展示名 —— 但文档里两种写法都会出现，所以基准 ④ 的索引**两个名字都收**（同名多处照样不猜）。
不这么做的后果与基准 ④ 当初一模一样：按 `name` 写的引用全落「基准不明」静默跳过。
同轮还修了一处真断链：`prompt-style-library/SKILL.md` 原写「运镜词表 `camera-movement-guides.ts`」，
文件早就迁成 `backend-py/app/services/camera_movement_guides.py`（全仓已无任何 `.ts`），
裸文件名（无斜杠）**从来不进校验**，所以它一直绿着 —— 改成带 `backend-py/` 前缀的真路径后进了真校验，从此这条断了会红。
**同日四补：库限定路径**（`wuxia-action/combat-choreography/references/hit-reaction-and-moves.md`）
也进真校验，行文按 **`skills/` 根**解析。判据只认「`skills/` 下一层、不含 `SKILL.md` 的容器目录」——
库名写错仍归基准不明（不猜）。起因是逐条判读了那 10 条「基准不明」：只有它一条真能判（文件确实在盘上），
其余 9 条各有正当理由 —— 通配符那条是本文档的说明文字、`export/…` 是**产出物**（写指导时当然还不存在，
判红就是假警报）、`modules/` 那两条是 `search_script_library` 例句里「带斜杠才不误捞」的路径形状、
`knowledge/vendors/…` 是上游路径。
**别再顺手「清理」那 9 条** —— 它们是**判过之后的保留**，不是漏判。

**同日五补：反向断链（资产无人指路）**。以上所有判据都只盯「指向空白」，但「空白无人指向」
是同一类失效的另一半 —— 知识同样进不了模型上下文，只不过程序化检查一直只认前者。
2026-11-06 实测 `references/`、`scripts/` 下 121 份资产里有 12 份无人指路，**全是真能力白废**：
`genre-templates/ui-motion/references/styles/A.md` 到 `H.md` 这八份样式锚点（正文写着
"The 8 style files" 却从没说它们在 `references/styles/`）、同目录 `common-adaptations.md`
（常见改稿请求的完整战术，正文一句没提）、
`production-tools/subtitle-correction/references/terminology.md`（语音识别纠错对照表，而技能
Step 1 反倒让用户手抄术语）与该目录的 `srt-format.md`、
`production-tools/skill-creator/references/NAMING.cn.md`（只指英文版 `NAMING.md`）。
已按「补指路、不删资产」修复，现归零。

判据 `orphan_assets()`。**本 skill 内**四种形态**都算指路**：点名文件名、通配符整体指路、**多级**
目录整体指路（`references/styles/`）、以及写到别的 skill 去。单级 `references/` 不算：「有这么个
目录」不等于「去读这一份」。
只扫 `references/`、`scripts/` 的**子树**：`NOTICE.md`、`agents/`、`gui.json`、`meta.yaml` 这类
平台与元数据文件本来就不该被正文引用，不在子树内自然不判。
**这一档计入退出码**（与断链同级）：资产不可达不是风格问题。
★两个只有真跑才露头的实现坑：① 按「直接父目录名 ∈ {references, scripts}」筛资产会**整片漏掉**
放在 `styles/` 子目录里的那八份（它们的直接父目录是 `styles`，不在那两个名字里），
当时那条「目录整体指路」断言其实是**假通过**；
② 文件名用子串匹配会让 `A.md` 被 `CTA.md` 蒙混过去，必须加词边界。
反验过：撤掉 ui-motion 那句样式指路 ⇒ 精准报 8 份、退出 1；改回后 0 份。

**同日六补：这条判据自己的网，被反验捅了两个洞。** 两个都是「反验一做一个不吱声」才发现的：

① **类型网是正向白名单**（`.md/.py/.json/.yaml/.yml/.ts/.sh/.txt`）—— 有 `.ts`、有 `.sh`、
   偏偏**没有 `.js`**。真树上 `multi-shot` 与 `n-storyboard` 各有一份 Node 入口脚本 `index.js`，
   **整份在网外**。当轮这两份恰好被正文点名（`entry:` 那行），症状没露；但形状是错的：正向白名单
   意味「明天新加的任何类型从落地那刻起默认不受看管，且**永远不报**」。改成**反白名单**
   （`NON_ASSET_SUFFIXES` / `NON_ASSET_NAMES`）：默认**在**网内，只有图片／音视频／压缩包／字体／
   二进制／占位符（`.gitkeep`、`LICENSE`、`NOTICE`）跳过 —— 这些「没有可读内容」，无从指路。
② **同名代答**：原来看的是**整棵树**的 md。抹掉 `multi-shot` 对 `index.js` 的点名后，孤儿数照旧 0、
   退出照旧 0 —— 因为 `n-storyboard` 里那句同名指路还在全树文本里，替它担保了。现改为：
   **裸名只在本 skill 内算数**，出了这个 skill 必须带 `<skill>/` 前缀（`n-storyboard/…/main.js`）
   或指向目录才算 —— 裸名只说得出文件名，说不出是谁的。
   与之配套，`<skill>/` 里那个 skill 名**目录名与 frontmatter `name` 两个都收**（本仓 5 个 core
   skill 二者不一致，只认目录名会把用别名写的合法指路判成孤儿）。
③ 顺带确认三件不报的事：图片放 `references/` 无人指路**不报**（图无从指路，要求指路只会逼出凑数
   句子）；建库但还没装 skill **不报**；`meta.yaml` / `gui.json` / `agents/` 仍在子树外。

反验三向（跑完自动还原）：抹掉 `multi-shot` 那两行点名 ⇒ 报 1 份、退出 1、点名到目录与文件；
`references/` 塞一张无人指的 png ⇒ **不报**；换成 `.mjs` ⇒ 报 1 份、退出 1。
套件新增 ⑦b 六条（`skill_refs_check_test.py` 共 **37 项**）。

### 派生登记：skill 声明的工具 ↔ 本仓真实工具

`/skills` 给每条 skill 算一个 `missingTools`（＝ `allowed-tools ∪ foreignToolRefs − 宿主工具集`），
前端用它亮「缺 N 个工具」徽标。这条链上有三处**只会静默错**的地方，2026-11-06 逐处补了守卫：

**① 声明键的变体**：`parse_skill` 原先只读 `allowed-tools`，而 `voice-clone` 把它写在
**`allowed-tools-speech`** 里（`hub_voice_prepare`），正文一次都没提 ⇒ 声明**整条消失** ⇒
`requiredTools` 少一项 ⇒ 徽标不亮。那枚徽标的提示语恰恰是「绑定后它会教 Agent 调用这些不存在的
工具（既不报错也不生效）」，此刻却一声不响。改为 `to_allowed_tools()` 收**所有** `allowed-tools*`
前缀的键、跨键去重保序；`allowed-skills` 这类不相干的键不收。

**② 清单的对照物是死的**：`HOST_TOOL_NAMES` 是静态镜像，守卫原本只对着**冻结的 TS 快照**比对 ——
而 `backend/`（Node）已删除，快照定格在删除那一刻，**Python 侧此后新增的工具它永远看不见**。
实测漏了三个真实工具（`search_script_library` 与按需通路 `list_skill_refs`／`read_skill_ref`，
均 2026-11-04 补），于是清单会**向两个方向说谎**：真实工具缺席 ⇒ 有 skill 声明它时徽标报「缺」
（假警报，而那工具好得很）；已删工具留着 ⇒ 徽标装死、绑定后调用落空。
⇒ 现以 `live_host_tool_names()`（逐个调 `create_*` 工厂取 **`Tool.id`**，构造期不查库不联网）为
**权威**，与 `HOST_TOOL_NAMES` **严格相等**；TS 快照降级为历史对照（`py_only` 只要求「确有工厂产出」）。

**③ 权威自身取错了字段，还并入了一串手抄名**（同日第三补，两个洞都在「活的权威」自己身上）：

* 它取的是工厂返回 dict 的**键**，而驱动侧拿到的函数名是 **`Tool.id`**
  （`ToolRegistry.as_openai_tools` 逐项用它组 `function.name`）。键 = id 时毫无差别，
  可一旦某个工厂沿用 TS 的 camelCase 键（`ToolRegistry.get` 早为这种事按 id 兜底，
  说明它真发生过），「真实工具集」就会多出一个**模型永远调不到的名字** ⇒ 清单守卫误红、
  `missingTools` 跟着说错。
* 两个 subagent 专属工具（`list_available_agents` / `run_subagent`）原先以**手抄**的
  `SUBAGENT_ONLY_TOOL_NAMES` 直接并入结果，理由是「工厂要 `conn`」。实测那个 `conn`
  只在 `execute` 里用、构造期不查库 ⇒ 与其它 8 个工厂一样真实调用即可。手抄串留着，
  只会让 live 在 `subagent.py` 改名时**最后一个**知道 —— 而它本该是唯一知道的地方。

⇒ 现 `live_tool_objects()` 按模块级清单真实构造全部 25 个工具、统一取 `Tool.id`；
`SUBAGENT_ONLY_TOOL_NAMES` 降级为「必须存在的两个」的**声明**（`live ⊇ 它` 由测试钉住，
不再代答）。清单提成模块级常量是为了让测试拿**同一份**去和 `as_openai_tools()` 对拍 ——
测试里抄第二份，就是把「手抄」这个病根换个地方重新种下。

上一轮双向反验：从清单删掉 `search_script_library` ⇒ `skills_test` 68/70 精确点出该名、
`route_parity` 报「真实工具未进 HOST_TOOL_NAMES」；塞一个查无实据的 `ghost_tool` ⇒ 69/70
与「清单里查无实据」。两侧跑完均自动还原。

本轮单向反验（针对 ③）：把 `subagent.py` 的 `id="run_subagent"` 改成 `run_sub_agent`
（dict 键不动）。**旧算法（读键）当场看不出任何异常** —— `HOST_TOOL_NAMES == 旧算法: True`、
73/73 全绿；新算法立刻红：`skills_test` 71/73 点出 `['run_sub_agent', 'run_subagent']`、
`route_parity` 12/14 报「真实工具未进 HOST_TOOL_NAMES：['run_sub_agent']」。复原后 73/73、13/14。
**同一坏状态下「旧绿新红」这一对，就是漏报的实证。**
判据 `skills_test.py`（**73 项**）。
★本文档里举例也走**库限定形式**（`genre-templates/ui-motion/...`，按基准 ⑥ 进真校验）；
不加库限定的相对写法会落回「基准不明」，白丢一次检查（本文档不在任何 skill 内，解不出 skill 根）。
三类跳过均按判据计数 ⇒ **致命与非致命任何非零都是真回归**，不是噪声。
⚠️ 本节、§一 的 skill 数量、§三 的注入体量**都是手写数字**，漂移时守卫**不会报警** ⇒
动完 skill 请顺手重跑本守卫并同步这三处（本次新增 skill 即按此流程更新）。

> 2026-09-12 补一处盲区：`docs/…` 形式的引用此前不在 `REPO_ROOT_PREFIXES` 里，
> 会落进「基准不明」被**静默跳过** —— 即 skill 正文写 `` `docs/xxx.md` `` 指到空处也不报警。
> 已把 `docs/` 纳入候选前缀：现有 `docs/` 引用里**被真校验的 2 处**
> （`backend-py/app/skills/prompt-style-library/SKILL.md` 与本文档 §九 各 1 处）从「跳过」转为**真校验**
> （待校验 93 → 96，基准不明 5 → 3），并用「临时改名 → 应报红」做过负向验证。
> 本文档 §十 里另外两处 `docs/…` 是**举例**（用双反引号转义），按「示意」不采集。

可选：把守卫挂到提交前（**需手动启用一次**，仓库不会自动生效）：

```bash
git config core.hooksPath .githooks   # 启用
git config --unset core.hooksPath     # 关闭
```

启用后，**仅当本次提交改到 `backend-py/app/skills/` 时**才运行守卫，致命断链会阻止提交
（确知无碍可用 `git commit --no-verify` 绕过）。钩子文件：`.githooks/pre-commit`。
