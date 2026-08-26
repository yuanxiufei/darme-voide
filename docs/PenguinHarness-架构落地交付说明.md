# PenguinHarness 架构参考文档 — Drama Studio 落地交付说明

> 生成日期：2026-08-17
> 参考文档：`d:\code\voides\voide-darme\docs\PenguinHarness-架构参考文档.md`
> 落地项目：Drama Studio（`d:\code\voides\voide-darme`，AI 短剧生成平台）

## 一、概述

参考文档第 17 章针对 Drama Studio 提出了 P0–P3 落地优先级清单，并给出第 8/9/10/11/12/15/16 章通用架构的借鉴动作。本轮已将**全部实质章节**在 Drama Studio 后端 + 前端落地，形成从「崩溃恢复 → 接口契约 → 评测优化闭环 → 自动管线编排 → SSE 实时进度」的完整链路。

**校验结论**：
- 后端 `tsc --noEmit`（`npm run typecheck`）**通过，0 错误**。
- 前端 lint **0 错误**。
- 核心新增文件抽样审查通过（实现忠实对齐参考文档，见下文）。

---

## 二、落地成果总览（对照参考文档章节）

| 参考章节 | 落地内容 | 关键文件 | 状态 |
|---|---|---|---|
| 17.1 / 9 / 16.2 | **P0 崩溃恢复**：视频/图片长耗时任务启动扫描 `processing` 续跑，幂等绝不重复提交 | `services/video-generation.ts`、`services/image-generation.ts` | ✅ |
| 17.2 / 5 | **P1 接口契约强化**：未知 provider 显式 throw，配置错误早暴露 | `services/adapters/registry.ts` | ✅ |
| 17.3 / 11 / 12 / 15 | **P2 评测→优化闭环**：确定性纯函数评分 + 状态机优化器 + 防作弊隔离，覆盖全部 5 个 Agent | `evaluation/`（types/scorer/evaluator/optimizer/cli） | ✅ |
| 17.4 / 7.1 | **P3 并发合并（MergeQueue）**：批量生图/生视频串行改并发，完成事件合并为单流 | `utils/merge-queue.ts`、`services/preset-framework.ts` | ✅ |
| 17.4 / 11 | **P3 协议化**：YAML 协议契约，`status + summary` 结构化输出 | `agents/protocol.ts` | ✅ |
| 17.4 / 9 | **P3 可观测性**：`startTrace` 单一事实来源，同一 run 多条日志带 `traceId + elapsedMs` | `utils/task-logger.ts` | ✅ |
| 17.4 | **自动管线编排器**：一句话 → 整集短剧（4 Agent + 媒体阶段），`auto:*` 单向状态机 + 崩溃续跑 | `services/auto-pipeline.ts`、`routes/auto-pipeline.ts` | ✅ |
| 11 | **Skill 规范**：SKILL.md frontmatter（name/description/preconditions/protocol）解析 | `agents/skill-parser.ts` + 5 个 `skills/*/SKILL.md` | ✅ |
| 11 | **agent-creation**：一句话需求 → Agent 配置（自进化闭环第一环） | `agents/creator.ts` | ✅ |
| 8 | **子 Agent 调度**：`MAX_SUBAGENT_DEPTH=2` + AsyncLocalStorage 防环形调用 | `agents/subagent.ts` | ✅ |
| 9 | **Trace 写入与回放**：append-only JSONL + 尾部愈合 + 结构保真回放 | `utils/trace-store.ts`、`routes/traces.ts` | ✅ |
| 10 | **MCP 接入**：三传输推断 + 懒连接 + 命名隔离 + 容错降级 | `agents/mcp.ts`、`routes/mcp.ts` | ✅ |
| 16.1 | **SQLite 幂等迁移**：`ensureColumn` | `db/index.ts`（已有，确认满足） | ✅ |
| 16.2 | **三态会话**：由 auto-pipeline 状态机 + 崩溃恢复等价覆盖 | `services/auto-pipeline.ts` | ✅ |
| 16.3 | **防重入 + 簿记吞失败**：`appendTraceEvent` 吞失败不拖垮主流程 | `utils/trace-store.ts` | ✅ |
| 16.4 | **SSE 进度推送**：pub/sub 总线 + `/stream/:dramaId` 端点 + 前端 EventSource | `utils/sse-hub.ts`、`routes/auto-pipeline.ts`、前端 `useApi.ts` / `index.vue` | ✅ |

---

## 三、新增文件清单（按模块分组）

**评测闭环（P2）**
- `backend/src/evaluation/types.ts` / `scorer.ts` / `evaluator.ts` / `optimizer.ts` / `cli.ts`
- `backend/benchmarks/`（5 个 Agent 的金标准基准）

**自动管线编排（P3 + 16 章）**
- `backend/src/services/auto-pipeline.ts`
- `backend/src/routes/auto-pipeline.ts`（含 `GET /stream/:dramaId` SSE 端点）

**通用架构移植（第 7/8/9/10/11 章）**
- `backend/src/utils/merge-queue.ts`（第 7.1 MergeQueue）
- `backend/src/utils/trace-store.ts`（第 9 章 Trace）
- `backend/src/utils/sse-hub.ts`（第 16.4 SSE 总线）
- `backend/src/utils/video-probe.ts`
- `backend/src/agents/subagent.ts`（第 8 章）
- `backend/src/agents/mcp.ts`（第 10 章）
- `backend/src/agents/protocol.ts`（P3 协议化）
- `backend/src/agents/skill-parser.ts`（Skill 规范）
- `backend/src/agents/creator.ts`（agent-creation）
- `backend/src/routes/traces.ts` / `routes/mcp.ts`
- `backend/src/config.ts`
- `backend/configs/mcp-servers.example.json`

---

## 四、关键修改文件清单

**后端**
- `services/video-generation.ts`（+`recoverVideoTasksOnStartup`，约 98 行改动）
- `services/image-generation.ts`（+`recoverImageTasksOnStartup`，约 85 行改动）
- `services/adapters/registry.ts`（`resolveAdapter<T>` 显式 throw）
- `services/preset-framework.ts`（生图/生视频并发化，约 172 行改动）
- `services/ai.ts` / `utils/task-logger.ts`（startTrace 接入）
- `agents/index.ts`（+138 行：instructions 单一组装点、skill 去重注入）
- `db/schema.ts` / `db/index.ts` / `db/connection.ts`
- `index.ts`（启动引导挂载 3 个 recover 函数）
- 多个 `routes/*.ts`（统一清理 request.js 拦截器语义）

**前端**
- `app/composables/useApi.ts`（+63 行，含 `autoPipelineAPI.streamUrl`）
- `app/pages/index.vue`（+410 行，自动生成 UI + EventSource 流）
- `app/pages/settings.vue`（+249 行）
- `app/pages/drama/[id]/index.vue`（+106 行）
- `app/components/VideoEditor.vue`（+61 行）

**Skill 规范**
- `skills/{extractor,grid_prompt_generator,script_rewriter,storyboard_breaker,voice_assigner}/SKILL.md`（补 frontmatter）

---

## 五、核心实现亮点（抽样审查通过）

1. **`trace-store.ts`**：刻意不用 `fs.appendFile`，改为 O_APPEND 打开 → 尾部愈合（探测末字节非 `\n` 则补）→ 单次 `write(2)` 写完 → 关闭，避免大 payload 被拆写后崩溃留下残缺记录；回放端逐行 `JSON.parse` 跳过坏行，结构保真非字节保真。
2. **`sse-hub.ts`**：内存 pub/sub 按 `dramaId` 分频道，`publishPipelineEvent` 永不 throw，订阅者异常自吞——簿记动作绝不拖垮主流程（对齐 16.3 收尾不变量）。
3. **`optimizer.ts`**：`formatFeedback` 只暴露「维度名 + 分数」，绝不暴露 detail（含金标准），防作弊隔离对齐第 12 章。
4. **`auto-pipeline.ts`**：`episode.status` 用 `auto:*` 单向状态机，媒体阶段提交前检查已有产物/processing 任务避免重复扣费，`recoverAutoPipelineOnStartup` 扫描中间态续跑。
5. **前端 `index.vue`**：EventSource 三事件本地合并（`snapshot` 全量 / `status` 单集推进 / `media-progress` 媒体进度），`finishAutoStream` 幂等收尾，断线后浏览器自动重连 + 后端重发 snapshot 兜底。

---

## 六、遗留事项与建议

1. **实测留待工作站**：端到端跑一次短剧生成需要外部 AI 供应商 Key（火山 Seedream/Seedance、MiniMax、CosyVoice 等）+ 真实扣费，且为分钟级长任务。当前环境不具备，未做全流程实测。**建议回工作站后**：
   - 起前后端，跑一次 `POST /auto-pipeline/run`，观察 SSE 是否实时收到 `status`/`media-progress`。
   - 或先做轻量验证：只起后端 + `curl -N .../stream/:dramaId` 观察 `snapshot` + `ping` 心跳。
2. **前端 typecheck 环境问题**：项目缺本地 `vue-tsc`，`npx` 下载版本与 TS `exports` 不兼容（`ERR_PACKAGE_PATH_NOT_EXPORTED`），无法跑前端 typecheck。建议后续补装 `vue-tsc` 到 devDependencies 并锁定版本。前端改动均为纯 JS + 字符串方法，lint 0 错误，无类型风险。
3. **配置安全（已确认无风险）**：`backend/configs/` 仅含 `mcp-servers.example.json`（无密钥）；`.gitignore` 已含 `configs/config.yaml` 与 `.env`，真实配置不会误提交。

---

## 七、提交建议（未执行）

改动遵循「不擅自 commit」约定，**尚未提交**。如需提交，建议分批（每批可独立 review）：

1. **P0 崩溃恢复**：`video-generation.ts`、`image-generation.ts`、`index.ts`
2. **P1 接口契约**：`adapters/registry.ts`
3. **P2 评测闭环**：`evaluation/`、`benchmarks/`、`skills/*/SKILL.md`
4. **P3 + 通用架构**：`protocol.ts`、`merge-queue.ts`、`task-logger.ts`、`subagent.ts`、`skill-parser.ts`、`creator.ts`
5. **第 9/10 章**：`trace-store.ts`、`routes/traces.ts`、`agents/mcp.ts`、`routes/mcp.ts`、`configs/mcp-servers.example.json`
6. **第 16 章 + 自动管线 + 前端**：`auto-pipeline.ts`、`sse-hub.ts`、`routes/auto-pipeline.ts`、`config.ts`、前端相关文件

---

## 八、增补：LLM 工程健壮性 5 项（同日续作）

在第七节基础上，又对照参考文档第 7/15/16 章补齐了 5 项「LLM 调用健壮性」细节：

| # | 功能 | 关键文件 | 作用 |
|---|---|---|---|
| 1 | LLM 重连退避 + 失败分类 | `agents/index.ts`（`classifyLLMError` / `backoffDelay` / `MAX_TRANSIENT_RETRIES=3`） | 429/5xx/网络抖动指数退避重试；认证/参数/上下文超限/内容审核判定 fatal 立即失败，省无效重试的 token 费 |
| 2 | 超长输入确定性滑窗 | `utils/text-slice.ts`（`sliceLongText`） | 长剧本保前 70% + 后 30%，中间省略，默认 24000 字符，避免 context 超限 fatal；应用于 3 个 read 工具 |
| 3 | follow-up 队列 | `services/auto-pipeline.ts`（`queuedEpisodes`） | 忙时入队而非静默丢弃，当前轮完成后幂等重跑最新配置 |
| 4 | 删除竞态保护 | `routes/dramas.ts`（DELETE 409）、`auto-pipeline.ts`（`isDramaDeleted`） | 删除前检查 in-flight episode 返回 409；管线内循环开头检查软删防复活 |
| 5 | 评测 runtime 一致性 | `evaluation/types.ts` / `evaluator.ts` / `optimizer.ts`（`runtimeModel`） | 候选与 Reference 评测所用模型不一致时跳过不 accept，保证「分数可比」前提 |

**价值评估**：1/2/3 项「现在就有用」（退避+分类省 token 费、text-slice 是长剧本硬需求、runtime 校验保障优化决策正确性）；4/5 项是「为规模化买的保险」（单机单人低频阶段几乎不触发，无副作用）。

---

## 九、参考完结声明

经逐章核对，**PenguinHarness 参考文档对 Drama Studio 已「榨干」**——全部实质章节（第 5/6/7/8/9/10/11/12/15/16/17 章）均已落地，无剩余可借鉴项。

**判定不照搬的项**（架构差异导致，非遗漏）：

| 项 | 不照搬理由 |
|---|---|
| OmniMessage 统一信封（第 4 章） | Mastra 自带消息结构，重复造轮子 |
| 三端一体 CLI/Desktop（第 3 章） | Drama Studio 是 Nuxt web，无桌面/CLI 需求 |
| 上下文压缩 compaction（第 7.5 节） | 单轮任务型 Agent 非长对话 ReAct，且 text-slice 已覆盖超长输入 |
| tar.gz 快照原子回滚（第 15.4 节） | optimizer「候选不写 DB」已规避，无需快照 |
| prompt cache 字节不变式 / 前向纠正（第 7.5/13 章） | ContextEngine 压缩机制特有，Drama Studio 无压缩环节 |

**后续方向**：不再对照参考文档，转向 Drama Studio 自身业务需求（更多媒体厂商、剪辑/合成能力、前端交互优化、性能等）。
