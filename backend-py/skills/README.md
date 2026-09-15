# backend-py/skills/ 目录约定

本目录存放 **Agent 的提示词层（Skill）**，只有两种角色：**项目自有 skill** 与 **外部技能库**。

- 加载器：`backend-py/app/services/agents/skills.py`
- 列表 / 编辑 API：`backend-py/app/routers/skills.py`（`/skills`、`/skills/meta`、`/skills/{id}`）
- 前端管理页：`frontend/app/pages/skills.vue`

## 一、目录结构

| 路径 | 角色 | 默认注入 |
|---|---|---|
| `backend-py/skills/<name>/SKILL.md` | **自有 skill（core）** — 与工作流、工具名、字段契约严格对齐 | ✅ 由该 `SKILL.md` 的 `agents:` 声明 |
| `backend-py/skills/<lib>/library.yaml` + `backend-py/skills/<lib>/**/SKILL.md` | **外部技能库（vendor）** — 库内 skill 一律不默认注入 | ❌ 需在「Agent 配置 → 绑定 Skills」手动启用 |

**库靠显式声明识别，不靠目录名、也不靠层级猜测**：顶层目录里含 `library.yaml` 就是一个技能库，
库标识取声明里的 `name`（缺省 = 目录名）。于是**加库 / 换库 / 改中文展示名都是纯文件操作，零代码**。

```
backend-py/skills/
├── README.md
├── <8 个自有 skill>/SKILL.md
├── genre-templates/          # 片型模板库（9 个）—— 按「成片类型」组织的端到端入口
│   ├── library.yaml
│   └── <skill>/SKILL.md
└── production-tools/         # 制作工序库（20 个）—— 制作链路上的单点能力与工序工具
    ├── library.yaml
    └── <skill>/SKILL.md
```

### 库声明文件 `library.yaml`

```yaml
name: genre-templates   # 库标识：API 与前端分组的稳定 key（缺省 = 目录名）；改它 = 改引用，慎改
label: 片型模板库        # 展示名：可随时改，不影响任何引用
description: …          # 侧栏悬浮提示
```

**`name` 与 `label` 分离**是这里最实用的一点：目录名 / 库标识保持稳定，对外展示名随时可调。

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
| `backend-py/app/services/agents/skills.py` → `loadAgentSkills` | DB 无配置时决定实际注入哪些 skill |
| `backend-py/app/services/agents/runtime.py` 的 `get_agent_defaults`（+ `agent_registry.py`） | `GET /agent-configs/defaults` 的出厂默认值 |
| `backend-py/app/routers/skills.py` → `/skills`、`/skills/meta` | 反向查询「该 skill 被谁绑定」与前端侧栏分组 |

## 三、注入规则

1. DB `agent_configs.skills` 有值 → **只**按 DB 配置加载（前端「Agent 配置 → 绑定 Skills」写入）
2. DB 为 `null` → 回退各 `SKILL.md` 自描述的默认绑定（`agents:` + `priority:`）
3. 所有注入受**字符预算**约束：默认 60000 字符，超出部分按优先级跳过并在注入文本末尾注明；
   用环境变量 `AGENT_SKILL_BUDGET` 覆盖（设为 `0` 关闭限制）

实测：5 个 Agent 默认注入合计 **28,128 字符、零跳过**。

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
> 把 `search_reference_prompts`（本地语料检索工具，见 `backend-py/app/services/agents/tools/corpus_tools.py`）
> 的**使用时机与红线**写进了这两个 skill —— **工具注册了不等于模型会调用**，
> 不写进 skill 就等于白注册。多花 810 字符换「9000 条语料真被用上」，这笔是值的。

> ⚠️ **加 skill 前先问一句：它有没有执行入口？** 没有（无 Agent / 无 UI 触发点 / 无代码消费其
> protocol 字段）就不要写进 `agents:` —— 那只会让每次生成都背上一段用不上的上下文。

## 四、为什么不默认注入外部技能库

外部库需手动启用（在库内 skill 的 frontmatter 写 `agents:` 不会生效 —— 默认绑定只扫自有 skill），原因：

1. **体量失控**：单文件最大 42 KB。曾使 `storyboard_breaker` 单次注入 ≈68 KB、
   `grid_prompt_generator` ≈81 KB，与 `DEFAULT_PROMPTS` 中的同主题规范重复；
2. **语义错位**：它们是"按触发词独立启动"的会话式技能（自带 `STEP 1/2/3`、`Not for: …`），
   不是"常驻系统提示词层"；
3. **注入不完整**：其正文大量引用 `references/*.md`，而加载器**只读 `SKILL.md`**
   → agent 顺着引用找文件必然落空，等于注入半截指令。

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

1. 建目录 `backend-py/skills/<name>/`（目录名即 skill id，小写 + 连字符或下划线），内含 `SKILL.md`
2. frontmatter 填 `name` / `description` / `preconditions` / `protocol` / `workflows`，
   需要默认注入时再加 `agents:` 与 `priority:`
3. 只写**领域知识**（词库、范式、判定规则），**不要重复** `DEFAULT_PROMPTS` 里的身份与工作流
4. 引用的外部资源文件（`references/` 等）**不会被加载**，需内联进 `SKILL.md` 或改用工具读取
5. 单文件建议 ≤ 30 KB；超过说明应拆分职责或改为按需加载

前端「Skill 管理 → 选中某 Agent → 新增 Skill」会在 `backend-py/skills/<agent>/<id>/` 下创建骨架
（模板自带 `agents: []` = 不默认注入，仅手动绑定）。

## 八、新增 / 替换外部技能库

在 `backend-py/skills/` 顶层建一个目录，放入 `library.yaml` 与 skill 子目录即可：

```
backend-py/skills/<lib>/library.yaml      # name / label / description
backend-py/skills/<lib>/<任意层级>/<skill>/SKILL.md
```

后端按声明自动识别为 vendor、前端侧栏自动出现该库分组 —— **零代码、零配置**。
删除同理：移除整个库目录即可（若有 Agent 手动绑定过其中的 skill，其注入会自然跳过）。

## 九、相关位置

- 画风词表（单一事实来源）：`backend-py/app/services/prompt_utils.py`
- 图像提示词范式（七段结构 / 镜头 / 光线 / 调色 / Danbooru tag）：`backend-py/skills/prompt-style-library/SKILL.md`
- 视频提示词范式（写法判定 / 时间码分段 / 散文式多段 / 中文标签 / 合规红线）：`backend-py/skills/video-prompt-library/SKILL.md`
- 提示词**取词来源**（人工维护用，**不注入 Agent**）：`docs/prompt-style-sources.md`
- 参考图反推：`backend-py/skills/style-reference-reverse/SKILL.md`（⚠️ 当前 `agents: []`，**未默认注入**，见 §三）

## 十、改动 skills 后必跑的自检

改名 / 挪库 / 删目录**都不会报错**：加载器只读 `SKILL.md`，正文里指向 `references` 目录的
那些文件引用一旦断链，只在读者真走到那一行时才表现为「指向空处」。所以有一条守卫：

```bash
python backend-py/scripts/check_skill_refs.py          # 退出码 1 = 存在致命断链
python backend-py/scripts/check_skill_refs.py --verbose # 额外列出被跳过的候选，审计盲区
```

**基线约定**：**141 文件 / 96 处待校验 / 致命 0 / 非致命 0**
（跳过计数：上游/外来宿主 5 / 示意引用 2 / 基准不明 3）。
三类跳过均按判据计数 ⇒ **致命与非致命任何非零都是真回归**，不是噪声。
新增 / 删除 skill 会改变**文件数**与**待校验处数**（属预期变化）；**致命必须恒为 0**。

> 2026-09-12 补一处盲区：`docs/…` 形式的引用此前不在 `REPO_ROOT_PREFIXES` 里，
> 会落进「基准不明」被**静默跳过** —— 即 skill 正文写 `` `docs/xxx.md` `` 指到空处也不报警。
> 已把 `docs/` 纳入候选前缀：现有 `docs/` 引用里**被真校验的 2 处**
> （`backend-py/skills/prompt-style-library/SKILL.md` 与本文档 §九 各 1 处）从「跳过」转为**真校验**
> （待校验 93 → 96，基准不明 5 → 3），并用「临时改名 → 应报红」做过负向验证。
> 本文档 §十 里另外两处 `docs/…` 是**举例**（用双反引号转义），按「示意」不采集。

可选：把守卫挂到提交前（**需手动启用一次**，仓库不会自动生效）：

```bash
git config core.hooksPath .githooks   # 启用
git config --unset core.hooksPath     # 关闭
```

启用后，**仅当本次提交改到 `backend-py/skills/` 时**才运行守卫，致命断链会阻止提交
（确知无碍可用 `git commit --no-verify` 绕过）。钩子文件：`.githooks/pre-commit`。
