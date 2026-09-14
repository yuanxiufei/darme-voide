# 记忆索引 —— 会话开始读这里，不要整读日志

> **三层读法**：`MEMORY.md`（不变量与约定，**必读**）→ `TOPICS.md`（低频长专题，按需）→ 本索引（日志定位）。
> 日志是**过程存档**，其结论/红线**已提炼**进 `MEMORY.md` / `TOPICS.md` / `docs/` / `skills/README.md`，**通常不需要读原文**。
> 确需过程细节/证据/命令时：按下方 `@行号` 用 `read_file(offset, limit)` **只读那一节**。
> ⚠️ 下方**体量与行号为 2026-09-12 快照**；日志只追加 ⇒ **`@行号` 锚点不会失效**（只有总行数在涨）。

## 日志清单（按需跳读）

**`2026-09-15.md`**（2.2k / 27 行，S7：QC 打分 + 原 TS 死分支发现）
S7 第 9 步：镜头 QC 打分（三维累加扣分 / 加权总体分 / 三态 status / **原 TS 死分支**）@3｜S7 第 10 步：QC 接线（视频完成→自动打分写库，闭环）@26｜S7 第 11 步：审片重跑闭环（软删产物/FL2VA/首帧等待，/retry-qc 关闭）@41｜S7 第 12 步：set-frame + 抽帧泛化（⚠️ 自检未跑，见日记）@59｜S7 第 13 步：重生成镜头帧（帧类型白名单/帧提示词/拼接，17 用例）@87｜S7 第 14 步：图像连续性 QC（真实图 dHash 三档/并发缓存竞态，28 用例）@105｜S7 第 15 步：快照重冻结 + 自动发现（守卫读文件必须在快照里；真源码/冻结结论一致）@120｜S7 第 16 步：对拍补齐新增 GET（实测 一致 12 / 新差异 0；文本响应时间戳归一化）@133｜S7 第 17 步：技术维度 QC（补行为缺口；三项检测是两侧共同的继承缺陷）@142｜S7 第 18 步：清过期 docstring + 修两个真缺口（webhook QC 空实现 / grid Agent 存根）@157｜S7 第 19 步：全面体检（修 merge 空实现；结论=剩校色/subagent 工具/storage.change）@170｜S7 第 20 步：校色落地（ffmpeg；实测 sharp 的 gamma 是 no-op / exposure 是 L 星乘法）@191｜S7 第 21 步：子 Agent 调度工具（ALS→ContextVar；注册表逐字镜像）@205｜S7 第 22 步：请求日志中间件（入口级行为已对齐，未实现清单清零）@215｜S7 第 23 步：清 tmp + README 遗留物表校真（剩唯一缺口=参考图压缩）@225｜S7 第 24 步：参考图压缩落地（功能性缺口清零；geq/夹具两坑）@240｜S7 第 25 步：对账抓出快照第二次缺件（自动发现补两种形态 + 端到端验收）@255｜待办（删库决策 · 提交）@269

**`2026-09-14.md`**（4.6k / 87 行，S7 收尾：GPU 租约接线补完）
S7 第 4 步：GPU 租约接线补完（image/video 长租约；4/3 处释放点，少一处即锁泄漏）@3｜S7 第 5 步：Node↔Python 差分对拍工具（三态判定/白名单越界/PowerShell 退格坑）@24｜S7 第 5-7 步：真对拍 0 新差异（MISSING 类）· 守卫冻结快照：**跑通真对拍**（一致10/新差异4：4 条裸列表端点 501；路数守卫有盲点）@42｜S7 第 7 步：守卫冻结快照（72 个 .ts / 707 KB，两条路径结论一致）@87｜S7 第 8 步：多集节奏相位（**修掉原 TS 无限递归**）@105｜待办（storyboards 4 条 · consistency-qc · 删库决策 · MEMORY 已满）@124

**`2026-09-13.md`**（35.0k / 225 行，承接绞杀者迁移：S6 评测执行器 + 评测域前 2 端点）
S6 第 4 步：评测执行器 + cases/evaluate 两端点（seed 提交/抽取口径/清理）@7｜S6 第 5 步：creator + optimize（含 /agent-configs/generate 解锁，状态机四条判据）@19｜S6 第 6 步：评测调度器（防重入/失败隔离/running 卡死缺陷保真）=> 评测域整域关闭 5/5@30｜S6 第 7 步：auto-pipeline 前置（sse-hub + frame-extractor，含尾帧红线）@41｜S6 第 8 步：全自动管线整域关闭（8 阶段 + SSE + AnyIO 事件循环坑）@53｜S6 第 9 步：localModels 整域关闭（12 条规则优先级 + NDJSON 下载 + 守卫假阳性修正）@65｜S6 第 10 步：补齐 10 条「延后理由已过期」的生成端点（characters 8 + props/scenes，含 smoke 硬编码耦合教训）@78｜S6 第 11 步：storyboards 5 条生成/LLM 端点（TTS 两套回执 + 对话解析贪婪坑 + 拆分顺延）@90｜S6 第 12 步：续写端点 + **剩余 16 条成因分类**（storage/change 有意延后、export 缺 854 行服务、distill 需媒体探测）@102｜S6 第 13 步：smoke 的「未迁移⇒501」清单改**动态派生**（根除三轮反复踩的坑 + import 顶掉 env 的教训）@118｜S6 第 14 步：export 开工（export-service 192 行 + EDL/ZIP 2 端点 + Math.round 银行家舍入坑）@129｜S6 第 15 步：剪映草稿导出（分 4 阶段；素材编号从 0、字幕错位照抄、全剧任一集缺则 400）@139｜⚠️记忆层修复（7 篇旧日记已不在盘，INDEX 悬空锚点已清）@151｜S6 第 16 步：QC 报告交付物 → **export 整域关闭 7/7**（探测回退链 / 60 分边界 / toFixed 陷阱 / GBK stdout）@161｜S6 第 17 步：时代背景提炼 + 风格提炼 2 端点（宽松判据 / 异常分层 / 冒烟别真发 LLM）@174｜S7 第 1 步：基准资产搬迁 + 评测 CLI（删 backend/ 的前置；app 已 0 依赖）@187｜S7 第 2 步：GPU 显存管理器整域（队列/驱逐/卸载策略 + /gpu/* 两端点）@197｜S7 第 3 步：GPU 租约接线（text/tts；契约：逐次申请+异常也释放）@214｜待办（image/video 长租约 · 全量对拍 · 删 backend/ · 冒烟偶发 · MEMORY 已满）@225


> 动**视频首尾帧 / 尾帧 / 角色合并**时**先读本文件**。





## 已出栈的落点（优先看这些，别翻日志）

| 内容 | 权威落点 |
|---|---|
| 全部 API 契约 | `docs/api-contract.md` |
| H3 本地视频链路 | `docs/local-h3-video-system.md` |
| 本地模型评估/下载 | `docs/local-model-evaluation.md`、`TOPICS.md` |
| 语料源与合规 | `docs/video-prompt-data-sources.md`、`docs/seedance2-corpus-analysis.md` |
| preset skill 模板 | `docs/preset-skill-template.md` |
| **Skill 体系（改前必读）** | `skills/README.md` + `MEMORY.md §Skill` + `TOPICS.md §Skill 体系细节` |
| **Python 后端（绞杀者迁移）** | `backend-py/README.md`（运行/环境变量/迁移 SOP）+ `TOPICS.md §backend-py` + `backend-py/tests/smoke_test.py` |
| 引用完整性守卫（skills 路径） | `scripts/check-skill-refs.mjs`（基线见脚本头） |
| 记忆层自检（8k 预算 + 锚点/登记/落点路径）+ 守卫自检 | `scripts/check-memory.mjs`、`scripts/test-guards.mjs`（规则见脚本头） |

## 日志写入规范（防再膨胀 —— `09-12` 单日已 780 行）

- 每轮 **≤ 8 行**：只写 **结论 / 落点(文件:行) / 红线 / 未决**。
- **不写**过程叙述与命令回放；需长期留证的内容 → `docs/` 或 `TOPICS.md`。
- 结论属「日后会导致 bug 的不变量」→ 同步进 `MEMORY.md`；属低频长专题 → `TOPICS.md`。
- 写时假定「将来只会按 `@行号` 跳读」，**不要依赖前后文**。