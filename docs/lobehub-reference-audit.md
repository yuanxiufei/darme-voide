# lobehub 组织全仓审计（58 仓 · 2026-11-10）

> 目的：把 `github.com/lobehub` 全部仓库拉下来，判定**哪些对本仓（中文短剧创作平台）真的有用**，并给出具体落点。
> 结论先行：**56 个可读仓里，只有 1 个（`lobehub` 主仓）与 1 个技能仓（`acceptance`）值得动手抄；另有 6 个只值得读设计；其余 47 个对本仓无落点。**

## 零、这批代码怎么来的（复现口径）

| 项 | 实测 |
| --- | --- |
| 上游可达性 | `api.github.com` ✓、`codeload.github.com` ✓、**`github.com:443` ✗（连接挂起，git clone 直接卡死）** |
| 采集方式 | `https://codeload.github.com/lobehub/<name>/zip/refs/heads/<branch>` 单请求 zipball 快照（**无 `.git`**） |
| 落点 | `reference/lobehub-org/<repo>/`（`reference/` 已在 `.gitignore:168`，不污染仓库） |
| 规模 | 56 仓落盘 / **1.9 GB** / 约 6.5 万文件；清单 `reference/_lobehub-repos.json`；脚本 `reference/_download-lobehub.ps1` |
| 未拿到 | `lobehub-python`、`lobe-i18n`：上游 `size=0`（**空仓**，不是下载失败） |
| 两仓需容错解压 | `docs-kit`（路径非法字符）、`lobe-chat-plugins`（`web.ar.json` / `Web.ar.json` 仅大小写不同的文件名在 NTFS 上撞车）→ 已用逐条解压 + 过滤修复 |
| 能力边界 | **快照没有 git 历史**：不能 `git log/blame`，引用的都是「当前 HEAD 的源码」；要历史需给 `github.com` 挂代理重 clone |

## 一、全量点名（56 仓 × 实测定位 × 档位）

档位定义：**可吸收**＝有具体翻写落点；**参考**＝读设计不搬代码；**留档**＝无落点但无害；**无关**＝与本仓领域不同。

### A. 旗舰（1）

| 仓 | 实测是什么 | 档位 |
| --- | --- | --- |
| `lobehub` | pnpm monorepo，**116 个 `packages/`**（210 MB / 2.4 万文件）。不是「聊天站」那么简单：`agent-runtime`(108 文件)、`tool-runtime`、`context-engine`、`eval-rubric`、`builtin-tool-*`(30+)、`canvas`/`canvas-runtime`、`model-runtime`/`model-bank`、`editor-runtime`、`markdown-patch`、`heterogeneous-agents`、`agent-tracing`、`ssrf-safe-fetch`… | **可吸收** |

### B. 智能体 / 评测（7）

| 仓 | 实测是什么 | 档位 |
| --- | --- | --- |
| `acceptance` | 不是代码库，是**一套「验收合同」技能**：`skills/acceptance/` = SKILL.md + 7 份 references（`report-schema.md` 11 KB、`evidence.md` 6.8 KB、`plan-format.md` 7.4 KB、`probe-mock-patterns.md`、`computer-use.md`、`video-chapters.md`）+ `cdp-capture.cjs` / `image-brightness.sh` / `resource-guard.sh` | **可吸收** |
| `lobe-bench-skill` | 一个技能：把模型评测结果渲染成「排行榜海报」（含 `.agents/` `.claude/` 双份）——是**技能形态样本**，非评测引擎 | 参考 |
| `tools-calling-benchmark` | 只有 `raw-outputs/` + README，**没有评测代码**，是结果留档 | 留档 |
| `locomo_lobehub_memory` | LoCoMo（ACL 2024 长对话记忆基准）的 Python 评测脚本；与「记忆检索质量」相关，与本仓知识库弱相关 | 参考（低） |
| `awesome-rsi` | 17 个文件的**论文/项目清单**（递归自我改进） | 留档 |
| `skills` | 一个把 LobeHub CLI 包成 agent 技能的 `skill.md` + `references/` —— **渐进披露写法的样本** | 参考（低） |
| `lobe-chat-agents` | 9648 文件 / 62 MB 的**角色扮演 prompt 市场索引**（多语），与短剧生产工艺无关 | 无关 |

### C. 前端 / UI（17）

| 仓 | 实测是什么 | 档位 |
| --- | --- | --- |
| `lobe-ui` | antd5 之上的 AIGC 组件库（React）；组件与 antd token 深度耦合 | 参考（token 组织） |
| `lobe-ui-react-native` | 同上的 RN 版 | 无关 |
| `lobe-charts` | 轻量 SVG 图表（React，d3 系） | 参考（评估报告可视化） |
| `lobe-editor` | **Lexical** 富文本编辑器封装 | 参考 |
| `streamdown` | headless 流式 markdown（block 切分 + 逐字淡入 + LLM 语法容错） | 参考 |
| `lobe-flow` | 节点流编辑器，**WIP、体量仅几十 KB** | 参考（低） |
| `sd-webui-lobe-theme` | Stable Diffusion WebUI 的主题皮肤（A1111 插件形态） | 参考（低） |
| `lobe-midjourney-webui` | Midjourney 代理的 Next.js 前端 | 参考（低） |
| `VChat` | 虚拟角色聊天站（Next.js；**归档**） | 无关 |
| `lobe-vidol` | 虚拟偶像（MMD/3D 舞台 + 聊天） | 低（数字人边缘） |
| `lobe-vidol-market` | 上面那个的角色市场索引 | 无关 |
| `lobe-lint` | 自研 ESLint 插件集 | 无关 |
| `dumi-theme-lobehub` | dumi2 文档站主题 | 无关 |
| `figma-tokens` | `tokens.json` 设计令牌 | 留档 |
| `lobe-readme-wizard` | README 生成器（CLI） | 无关 |
| `vite-spa-template` | React SPA 模板 | 无关 |
| `fluent-emoji` / `lobe-icons` / `lobe-assets` | 表情 / 模型品牌 SVG / logo 与字体资产（合计约 1.1 GB 上游体积） | 无关（可当素材库） |

### D. 后端 / 协议 / 插件（24）

| 仓 | 实测是什么 | 档位 |
| --- | --- | --- |
| `lobehub-gateway` | **Go** 网关重写（`agent-gateway-go` + `device-gateway-go`），自托管用 | 参考（低） |
| `lobe-tts` | TTS **React Hooks**：Edge TTS(websocket) / Microsoft Speech / OpenAI TTS+STT / 浏览器 speechSynthesis；**无本地权重推理** | 参考（voice 元数据） |
| `lobe-openai-plugins` | 一组 OpenAI-plugin 形态的插件（search/crawler/weather 等） | 留档 |
| `lobe-chat-plugins` | 插件市场**索引 JSON**（1000+ 条） | 留档 |
| `chat-plugin-sdk` | 插件 SDK（manifest + gateway 协议） | 留档 |
| `chat-plugins-gateway` | 插件市场网关（Next.js） | 留档 |
| `chat-plugin-{template,sdk…}` 及 6 个实例插件（bilibili/clock/search-engine/steam/web-crawler/open-interpreter/realtime-weather） | 模板 + 示例插件，都是「外部 API → 聊天工具」 | 留档 |
| `local-replica` | 「本地优先副本」：首帧先画上次已知状态，后台与服务端对账 | 参考（低-中） |
| `y-webrtc-signaling` | Yjs 的 WebRTC 信令服务 | 无关（本仓画布单机） |
| `mcp-hello-world` | MCP 最小示例 server | 留档（低） |
| `lobehub-typescript` | 官方 TS API SDK（自动生成） | 留档 |
| `lobehub-python` | **空仓** | 无 |
| `lobe-i18n` | **空仓** | 无 |
| `lobehub-auv` | macOS「Computer Use」签名打包（Aqua agent），Windows 不可用 | 无关 |
| `lobehub-gateway` 之外的 `lobe-analytics` | 埋点库（多 provider） | 无关 |
| `lobe-mygo` | 「The UI Kit for Native Go Apps」（Go + TS 桌面 UI 套件） | 无关 |
| `litellm` | **第三方 fork**（完整 LiteLLM proxy，241 MB / 1.99 万文件）；与本仓自研引擎定位冲突 | 留档 |
| `lobe-cli-toolbox` | git commit / i18n 工作流 CLI | 无关 |
| `docs-kit` / `lobe-lib-template` / `lobe-next-perf` / `expo-better-auth-passkey` / `homebrew-tap` / `.github` | 文档工具 / 库模板 / 性能试验 / Expo passkey / brew tap / 组织级社区文件 | 无关 |

## 二、第二遍：候选机制清单（只列有落点的）

> 每条都注明**参考文件**（`reference/lobehub-org/...` 下）与**本仓落点**。

### P0-1 工具重复调用守卫（防 agent 打转）
- 参考：`lobehub/packages/agent-runtime/src/utils/toolCallRepeatGuard.ts`（2.6 KB）
- 机制：按「工具名 + 参数 JSON 全等」归一，命中两类阈值给出不同文本——**同调用连续 3 次**（`REPEAT_EXACT_LIMIT`）与**同工具在最近窗口内 5 次**（`REPEAT_BURST_LIMIT`）；返回的不是抛错而是一段**给模型的提示语**（"这不会产生新结果，换参数或换工具"），并把「当前重复次数」写入事件。
- 本仓落点：`backend-py/app/agent/runtime.py` 的工具循环内（在 `_run_tool` 前后各一次判定）。
- 本仓现状（实测）：`app/agent/*.py` 内 `repeat`/`burst` **0 命中** ⇒ 真缺口。
- 为什么值得：本仓已有「模型反复输出同样内容/反复调同一工具」的实例记录（长任务里最贵的失败形态），而当前只能靠 `qc_retry` 事后发现。

### P0-2 人工介入两段式（pre / post 审批）
- 参考：`lobehub/packages/agent-runtime/src/core/InterventionChecker.ts`
- 机制：`checkBefore`（工具执行**前**判定是否要人批准，返回 `pending` + 待批参数）与 `checkAfter`（执行后**复核**产出是否需要人确认）；审批结果作为**消息**回流，而不是中断整个 run。
- 本仓落点：`runtime.py` + `tool.py`（工具表加 `requires_approval` 位）；前端走现有 SSE 事件。
- 本仓现状：`approve`/`approval`/`intervention` **0 命中** ⇒ 真缺口（现在只有「工具被除名/不允许」这种硬拦截，没有「等人点头」）。
- 适用面：出图/出视频/写库/删除/付费云调用这类**有副作用且贵**的工具。

### P0-3 判分证据单列 + 0 分合法 + 解析失败重试
- 参考：`lobehub/packages/eval-rubric/src/matchers/llmRubric.ts`（6.2 KB）
- 机制三件：
  1. 裁判 prompt 分五段：`[Criteria][Input][Output][Evidence][Expected]`，其中 **Evidence 单列**（把 agent 的动作轨迹单独喂进去），理由写在注释里——「只看最终回复无法判『有没有调用工具』」；
  2. 输出强约束成 `{score: 0..1, reason}`；**`score=0` 是合法判分，不能用真值判断是否失败**（`if (!result.score)` 会把 0 分当解析失败）；
  3. 只有 **抛错/解析不出** 才重试（`maxAttempts=3`），结果附 `attempts`。
- 本仓落点：`backend-py/app/agent/evaluation/evaluator.py`。
- 本仓现状：已有 `step_trace()`（每步工具计数）；但判分调用**没有 JSON schema 约束、没有重试**（`retry`/`schema`/`attempt` 在 evaluator 内 0 命中）⇒ 真缺口，且改动面很小、可直接量出差别（同用例重复判分的抖动）。

### P1-1 验收报告合同（evidence → verdict）
- 参考：`acceptance/skills/acceptance/references/report-schema.md`
- 机制：一条验收产出的固定形状——
  - `plan[]`：每条检查带 `verifier`（builder/independent）、`method`、`expected`、`requiredEvidence`（先声明要什么证据，再执行）；
  - `cases[]`：每条 `status` + `observation` + `evidence{path, description}`（**可点开的文件**）+ 可选 `counterEvidence`/`limitation`/`suggestion`；
  - `summary.verdict`：`pass / fail / inconclusive`（**允许 inconclusive**，"没验到" ≠ "通过"）；
  - `datasets[]` + `visualizations[]`（metric-comparison / line / bar / scatter / heatmap / table）：把**度量**变成图表，而不是只写一段话。
- 本仓落点：新增 `backend-py/app/services/acceptance_report.py`（数据结构 + 落盘 JSON）+ 审核页/制作线读取。
- 本仓现状：`app/services/qc_reviews.py` 是**「检查项命中率复核」**（verdict ∈ `VERDICTS`、`MIN_REVIEWS`、`hit_rate`），`repair_closure.py` 是修复闭环——**都是"检查器准不准"，不是"这一件产物凭什么算过"**。差的就是上面那份 case 级证据合同 ⇒ 真缺口。
- 附带可抄：`acceptance/skills/acceptance/references/` 下的探针脚本（`cdp-capture.cjs` 截图取证、`image-brightness.sh` 亮度判据、`video-chapters.md` 视频分章抽帧、`resource-guard.sh` 资源护栏）——本仓审核要的「看一眼图/看一眼片」正好是这几件事。

### P1-2 增量变更集（AI 改文档不许整段覆盖）
- 参考：`lobehub/packages/markdown-patch/src/`（5 种模式 + `HUNK_AMBIGUOUS` 等 7 个错误码）、`lobehub/packages/canvas/src/project/changes.ts`（`writes/edits/deletes` 三件套）
- 机制（`applyCanvasChanges` 逐行读过）：
  - `edits[].oldText` 必须**在文件里恰好出现 1 次**，否则报错并提示「把上下文写全一些」——**歧义即拒绝，绝不猜**；
  - 所有失败都返回**模型可读的句子**（`writes[0] is missing "path": every write must name the file…`），模型可据此自我修正重试；
  - 全量 `writes` + 局部 `edits` + `deletes` 三种意图同一事务里执行。
- 本仓落点：新增 `backend-py/app/services/markdown_patch.py`，挂成 agent 工具（改剧本/改分镜表/改提示词表都走它）。
- 本仓现状：`apply_patch`/`markdown_patch` **0 命中**（`hunk` 的 206 条命中全是 `chunk` 误报）⇒ 真缺口。
- 为什么对本仓特别重要：本仓铁律是「**剧本文字不许因为制作便利被改**」，而现在 AI 改稿是整文件重写——一旦模型顺手润色掉一句台词，没有任何机器判据能发现。改成 patch 后，**改动面可审计、可拒绝、可回滚**。

### P1-3 技能维护：结构化字段渲染 frontmatter
- 参考：`lobehub/packages/builtin-tool-skill-maintainer/src/types.ts`
- 机制：`createSkill{name,title,description,bodyMarkdown}` ——**agent 只写正文**，YAML frontmatter 由结构化字段渲染；改/删/改索引都带 `reason`（留给审计面）。
- 本仓落点：`backend-py/app/services/skill_optimizer.py` + `app/agent/skills.py`（技能四格优化的写入路径）。
- 本仓现状：技能是手写 `SKILL.md`（含 frontmatter 与 25 份 yaml），自动优化**直接改文本**——frontmatter 字段漂移与「为什么改」无审计字段 ⇒ 半缺口（值得补，不是为了功能而是为了可审计）。

### P2-1 异构智能体桥（把 Claude Code / Codex 当子智能体）
- 参考：`lobehub/packages/heterogeneous-agents/src/`（`claudeCodeEntrypoint.ts`、`attachedFiles.ts`、`imageEcho.ts`、`config.ts`）
- 机制：把外部 CLI agent（Claude Code 等）注册成一种「引擎」，统一成同样的流式事件；附件与图片走「回显」通道避免重复上传。
- 本仓落点：`app/agent/subagent.py`（现在只有内部子智能体）。
- 判定：**中**——若创作者确实想在本平台里直接调度 Claude Code/Codex，这是现成的接线图；否则不做。

### P2-2 模型能力元数据
- 参考：`lobehub/packages/model-bank`（capabilities：`vision/reasoning/functionCall/search/token` 等布尔位）+ `model-runtime/src/types`
- 本仓现状：`app/services/` 内 `capabilit` 已有 48 处 ⇒ 已有等价物，只值得**对照字段命名**，不必搬。

### P2-3 样式等价性验证器（给 CSS 守卫做「变异自证」）
- 参考：`lobehub/packages/style-equivalence/`（`calibration/`、`mutation/`、`evidence/`、`verdict/`，`VERIFIER_VERSION=2.14.0`）
- 机制要点：先**校准**判定阈值 → 抓浏览器 computed style → **注入变异**（`DEFAULT_MUTATIONS=10`）证明「检出器真的会红」→ 产出带证据的 verdict。
- 本仓落点：`frontend/scripts/check-parity.mjs` / `check-responsive.mjs`（现在只做「文件逐字一致 / 有 media query」这类**存在性**检查，没有自证）。
- 判定：**中低**，成本高（要起浏览器 + 注入变异），但「守卫自己要先被证伪」这条方法论值得借。

### P2-4 本地优先首帧（慢引擎加载体验）
- 参考：`lobehub-org/local-replica`
- 机制：首帧先渲染上次已知状态（陈旧值），后台对账更新；对「引擎装载 20 秒、请求 5 分钟」的本仓，比空转 spinner 更诚实。
- 判定：**中低**，落点在 `frontend/composables/useApi.ts` 一带。

## 三、第三遍：**否掉的冗余**（看起来能抄，其实本仓已有或不适用）

| 看起来能抄 | 为什么否掉 |
| --- | --- |
| `model-runtime` 的 provider 多适配 | 本仓 `app/services/ai_configs.py` 已有 8 厂商 + ollama 通路，且**引擎链路是本仓独有**；抄过来只增加一层无用抽象 |
| `lobe-tts` 的 TTS 抽象 | 它只有 Edge/OpenAI/浏览器三条**云**通路，**没有本地权重推理**；本仓 F5-TTS 走自己的引擎。只留「voice 列表 + SSML style 元数据」这点皮毛，价值低 |
| `chat-plugin-sdk` / `lobe-openai-plugins` / 6 个示例插件 | 本仓 `canvas_plugins.py` 已有插件契约（`validate_plugin` 11 条校验）、工具表 34 条；这些是「外部 API→聊天工具」的旧形态，**新增为零**。唯一待议的是 `OpenAPI spec → tool schema`（本仓 `openapi` 仅 2 处命中且都无关）——但本仓外部 API 很少，暂不做 |
| `canvas` / `canvas-runtime` 的**节点图**协同 | 读完 `builtin-tool-canvas/src/types.ts` 才发现：LobeHub 的 Canvas **不是节点图**，而是「HTML/SVG/React 作品文件 + 变更集」；本仓画布是节点图（`canvas_ops.py` 21 工具 / 8 op），**不是一类东西**。协同（Yjs/CRDT）对单机画布也不成立 |
| `context-engine` 的上下文压缩 | 本仓已有 `context_budget.py`（预算 + 裁剪）；重复 |
| `memory-user-memory` 的记忆存储 | 本仓知识域是 sqlite 三库（ledger/outcome-index/skill-usage）+ FTS，机制更贴自己的评测闭环；只借「抽取时机」一条 |
| `lobe-ui` / antd | 本仓前端**纯自研 CSS + Vue**（`assets/studio.css` 单一真源），React/antd 组件一个也落不下来 |
| `streamdown` | 本仓已有流式 markdown 渲染；只值得借「block 级切分 + 未闭合语法容错」的具体算法 |
| `lobe-editor`（Lexical） | Vue 侧要换 Tiptap 才有意义；`editor-runtime` 的 LiteXML 选区编辑思路更值（但工程量大，列 P2 不做） |
| `litellm` | 第三方 fork，与本仓「自研引擎 + 自己路由」正面冲突；241 MB 纯占盘 |
| `fluent-emoji` / `lobe-icons` / `lobe-assets` / `.github` | 纯资产与社区文件，与本仓无接口 |

## 四、如果要动手，按这个顺序

1. **P0-3 判分三件**（改动最小、可立刻量出差异：同用例重复判分的抖动带宽）
2. **P0-1 重复调用守卫**（把「打转」从"事后 qc_retry"变成"当场拦下并提示"）
3. **P0-2 两段式审批**（付费/有副作用工具前置确认）
4. **P1-2 变更集**（AI 改剧本改走 patch，改动面可审计 —— 与本仓「剧本不许被顺手改」的铁律直接对齐）
5. **P1-1 验收报告合同**（把「检查器命中率」升级成「这一件产物凭什么算过」）
6. P1-3 / P2-* 按需

## 五、可复现

```powershell
# 重跑采集（幂等：已存在的跳过）
powershell -NoProfile -ExecutionPolicy Bypass -File reference/_download-lobehub.ps1
# 看进度
Get-Content reference/_download-lobehub.log -Encoding utf8 -Tail 20
# 复核「本仓有没有这个机制」
Select-String -Path backend-py/app/agent/*.py -Pattern 'repeat|burst' -SimpleMatch
```

实测对照（本仓 2026-11-10 状态）：`app/agent/*.py` 内 `repeat`=0、`burst`=0、`approve|approval|intervention`=0、`checkpoint`=0；`evaluator.py` 内 `schema|retry|attempt`=0（`step_trace` 已有）；`apply_patch|markdown_patch`=0。
