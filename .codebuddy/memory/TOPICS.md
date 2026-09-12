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
- **本工具怪癖**：`search_content` 的 **`glob` 不生效**（`**/SKILL.md` 恒 0 命中）→ 改用 `path` 收窄。
- **守卫「示意引用」判据**：标记（`e.g.`/`such as`/`例如`）**必须紧邻**路径之前（标记后只允许非字母数字非汉字字符，含 token 前的开启反引号）；放宽成「同行出现过 e.g.」会**连真断链一起吞掉**（已用负向测试证实）。基线约定见 `scripts/check-skill-refs.mjs` 脚本头。

## 前端验证与测试工具（自 MEMORY.md §前端约定 下移）
- **工作台元素计数**（点击测试定位用）：`nav button` = 12 主步骤；`aside button` = 18（12 + 5 `sidebar-jump-dot` + 1 `.refresh-btn`）。**工作台改版后须重新核对**。
- **验证 SFC 编译**：`node -e "require('@vue/compiler-sfc')"` 跑 `compileScript` + `compileTemplate`，**无需启 dev server**；纯 TS（如 `useApi.ts`）用 `ts.transpileModule`。两者都能在改完立刻抓语法/模板错误。
