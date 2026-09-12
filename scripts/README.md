# scripts/ — 工具与自检脚本

本目录有**三类**脚本（此前标题只写"AI/GPU 工具链"，漏了后两类）：

| 类别 | 脚本 | 运行环境 |
|---|---|---|
| **AI/GPU 工具链** | `model_manager.py` / `sd_h3_pipeline.py` / `sd_h3_compat_probe.py` / `h3_install.py` / `migrate_models.ps1` —— 与 TS 后端（`backend/`）通过 subprocess 解耦 | Python 3.8+，**仅标准库**（零第三方 pip 依赖） |
| **仓库自检** | `check-skill-refs.mjs` / `check-memory.mjs` / `test-guards.mjs` / `check-all.mjs` —— 防资产**静默漂移** | Node ESM（零依赖） |
| **语料分析** | `corpus/analyze{,2,3}.py` —— 产出 `docs/seedance2-corpus-analysis.md` 的统计结论 | Python 3.8+，**仅标准库** |

> 三类都**不参与产品运行时**。自检脚本的自动触发靠 `.githooks/pre-commit`（**需手动启用一次**），
> 但它只在「本次提交触及相应资产」时才跑对应守卫 ⇒ **想全局体检请用 `check-all.mjs`**。

## 工具清单

| 文件 | 语言 | 职责 |
| --- | --- | --- |
| `model_manager.py` | Python | 通用本地模型安装/管理（多类模型 + comfyui/ollama/git/manual 四种安装方式） |
| `sd_h3_pipeline.py` | Python | sd.cpp × MiniMax-H3 Turbo GGUF 一键编排（doctor → download → build → probe） |
| `sd_h3_compat_probe.py` | Python | GGUF 与 sd.cpp 兼容性判定（零依赖读元数据，可选 `--live` 真机 load） |
| `h3_install.py` | Python | 旧 H3 安装 CLI 兼容 shim（数据已并入 `model_manager.py`） |
| `migrate_models.ps1` | PowerShell | 一次性：模型硬链接迁移到 ComfyUI Desktop 共享库 |

## 用法

### model_manager.py（核心入口）
```powershell
python scripts/model_manager.py list [--category X] [--runtime Y] [--missing]
python scripts/model_manager.py download [--key ...] [--category ...] [--required] [--all] [--force]
python scripts/model_manager.py remove --key ...
python scripts/model_manager.py doctor
python scripts/model_manager.py install-nodes [--only ...]
python scripts/model_manager.py add-model --key ... --name ... --category ... --runtime ... [...]
python scripts/model_manager.py remove-model --key ...
```
数据源：`configs/models.json`（模型清单）+ `configs/model-paths.json`（路径配置）。
路径优先级：命令行参数 > 环境变量 > `model-paths.json` > 默认探测。

### sd_h3_pipeline.py（路线 A 编排）
```powershell
python scripts/sd_h3_pipeline.py                 # 顺序执行 doctor→download→build→probe
python scripts/sd_h3_pipeline.py doctor          # 单步：环境体检
python scripts/sd_h3_pipeline.py download        # 单步：下载 Q4_0 Turbo GGUF（断点续传）
python scripts/sd_h3_pipeline.py build           # 单步：编译 sd.cpp（cmake -DSD_CUDA=ON）
python scripts/sd_h3_pipeline.py probe [--live]  # 单步：兼容性判定
```
编译依赖：Git、CMake 3.x+、Visual Studio 2019/2022（C++ 桌面开发）、CUDA Toolkit（nvcc）。
> 注：本机当前无 CUDA Toolkit（nvcc），`build` 只能产出 CPU 版，详见 `docs/local-h3-video-system.md`。

### sd_h3_compat_probe.py（兼容性判定）
```powershell
python scripts/sd_h3_compat_probe.py --gguf <path.gguf>          # 静态判定
python scripts/sd_h3_compat_probe.py --gguf <path.gguf> --live   # 真机 load
python scripts/sd_h3_compat_probe.py --list                      # 支持矩阵
```

### h3_install.py（兼容 shim）
保留旧 CLI（`doctor` / `download` / `download-model` / `install-nodes`）与库接口，
数据一律从 `models.json` 读取。新能力请改用 `model_manager.py`。

### migrate_models.ps1（一次性迁移）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/migrate_models.ps1
```

## 仓库自检脚本（非 AI/GPU，Node 运行）

| 文件 | 语言 | 职责 |
| --- | --- | --- |
| `check-skill-refs.mjs` | Node ESM | 校验 `skills/**/*.md` 里的路径引用是否都能落地（`references/` 资产、跨 skill `../`、及 `skills/`+`backend/`+`frontend/`+`docs/` repo 根相对路径），防「改 skill 名 / 挪库 / 改 docs 名」造成的**静默断链** |
| `check-memory.mjs` | Node ESM | 校验 `.codebuddy/memory/` 三层记忆：`MEMORY.md` ≤ 8k 字符（超限注入会被截断）、`INDEX.md` 的 `@行号` 锚点有效、且**每篇日志的末节都已登记** |
| `test-guards.mjs` | Node ESM | **两套守卫的自检**：在临时副本上造 10 种故障（记忆层 7 + 引用层 3）+ 2 条基线，断言每项检查仍能报致命（防「守卫被改哑但基线仍绿」） |
| `check-all.mjs` | Node ESM | **一键跑全部自检**（上表三道串联 + 汇总）—— pre-commit 只在「本次提交触及相应资产」时才跑对应守卫，本脚本用于**全局体检** |

```powershell
node scripts/check-all.mjs                   # 一键跑全部三道自检，退出码 1 = 有任一失败
node scripts/check-all.mjs --verbose         # 同上，并把 --verbose 透传给引用守卫

node scripts/check-skill-refs.mjs            # 单跑：有断链则退出码 1，可用作提交前自检
node scripts/check-skill-refs.mjs --verbose  # 单跑：额外列出被跳过的候选，审计脚本自身盲区
node scripts/check-memory.mjs                # 单跑：记忆层自检，退出码 1 = 存在致命项
node scripts/test-guards.mjs                 # 单跑：改了任一守卫脚本后必跑，退出码 1 = 有用例失败
```

判定分级（细节见脚本头注释）：
- **致命**：`references/…`、跨 skill `../…`、repo 根相对路径（`backend/`、`frontend/`、`skills/`、`docs/`）指向不存在 → 退出码 1
- **非致命**：缺 `scripts/…` —— 外部技能库只随行 `SKILL.md` + `references/`，上游 `scripts/` 普遍未 vendored，且该目录对 Agent 不可达，故只列出
- **跳过**：上游 / 外来宿主路径（`.ci/`、`spec/`、`.opencode-v2/`、`.claude/`、`.agents/`）、含通配符的模式、**示意引用**（`e.g.` / `such as` / `例如` **紧邻**于路径之前，只是举例而非依赖）、基准不明（不在任何 skill 内 / 非资产目录开头，如 `export/…`）
- 各类跳过**分别计数**且可 `--verbose` 逐条审计 —— 「跳过」不是静默丢弃

### check-memory.mjs（记忆层自检）
`.codebuddy/memory/` 的两条不变量（`MEMORY.md` ≤ 8k、`INDEX.md` 锚点不失效）原本全靠人工维护、
已多次漂移，故补机器校验。分级（细节见脚本头注释）：
- **致命**：三层文件缺失；`MEMORY.md` 超 8k 字符（注入从尾部截断，`## 协作与提交` 最先丢）；
  `@N` 越界或未落在「小节首行」（`#`~`####` 标题行，或 `- 【…】` 条目行）；
  **某篇日志的末节没有登记锚点**（写了日志忘登记索引 ⇒ 新结论在索引里不可达）；
  **磁盘上的日志未登记进 INDEX**（新的一天新建日志忘登记 ⇒ 整篇不可跳读）；
  **「已出栈的落点」表引用的路径不存在、或 `文件 §小节` 的小节名在该文件内查不到**
  （该表承诺「优先看这些，别翻日志」，断链即失效）
- **提示**：`MEMORY.md` 逼近预算（余量 < 400 字符）
- 刻意**不校验**索引里的「N 行 / M 轮」快照数：其口径不一致，校验只会产生噪声
- 落点表检查**只扫该小节**：日志清单的摘要文字会提到历史文件名（如 `` `models.vue` ``），
  那是**叙述**不是引用，扫全文必误报（首版即被它撞出致命 1 处）
- 两道守卫均已接入 `.githooks/pre-commit`（触及 `skills/` 或 `.codebuddy/memory/` 时自动跑）；
  **守卫脚本自身被改动时额外跑 `test-guards.mjs`**——因为「守卫被改哑」时第 2 道仍是绿的，只有它能拦

### test-guards.mjs（守卫的自检）
检查逻辑被改坏、正则被放宽、白名单被删时，**基线往往是绿的**（真实仓库本来就合规）⇒
「守卫没报错」不等于「守卫还能报错」。本脚本把先前**手工造故障 + 手工还原**的负向实证固化为
可重跑用例：把真实 `check-memory.mjs` 指向 `MEMORY_DIR` 临时副本 → 造故障 → 断言
「退出码 1 + 命中预期文案」。**真仓库全程只读**，跑完删临时目录。
- `check-memory.mjs`：基线 1 例 + 负向 7 例（①缺文件 ②超 8k ③锚点越界 ④末节未登记
  ⑤日志未登记 ⑥落点路径不存在 ⑥b §小节指针落空）
- `check-skill-refs.mjs`：基线 1 例 + 负向 3 例（⑦真断链 ⑧`docs/` 引用断链 ⑨示意引用不误报）
- 反向控制已实证：把 ⑥ 人为改哑后，恰好 ⑥/⑥b 两例转红、其余用例仍绿（用例相互隔离）
- **原先未覆盖 `check-skill-refs.mjs` 的真正原因**：它的 `SKILLS_DIR` 是**硬编码**的，没有类似
  `MEMORY_DIR` 的覆盖入口 ⇒ 副本根本指不过去（不是"成本不匹配"）。2026-09-12 加 `SKILL_REFS_DIR`
  覆盖后即可自检；夹具拷**整棵** skills 树（造小夹具会漏掉真实文件里的引用形态）
- ⚠️ 断言**不区分守卫文案**只认「退出码 0 + `致命…0 处`」：两套守卫的零值文案本就不同
  （记忆层「致命 0 处」｜引用层「致命断链 0 处」），写死单侧会误判——首版即踩此坑
- 用例里的字符串替换若**找不到原文**会直接判失败（提示「夹具失配」）⇒ 夹具过时不会被静默跳过

## 语料分析脚本（非 AI/GPU，Python）

分析**视频提示词语料**（`metadata.jsonl`，逐行 JSON）并产出
`docs/seedance2-corpus-analysis.md` 的统计结论。与上方工具链同规格：**仅 Python 3.8+ 标准库**。

| 文件 | 职责 |
| --- | --- |
| `corpus/analyze.py` | 首轮结构探查（字段、长度、语言） |
| `corpus/analyze2.py` | 深挖 i18n / raw_p / category / spec |
| `corpus/analyze3.py` | **主力脚本**：词表、结构特征、标签频次 |

```powershell
# 脚本自解析仓库根；也可用 SEEDANCE2_CORPUS=/abs/path/metadata.jsonl 覆盖
python scripts/corpus/analyze3.py > data/prompt-corpus/analyze3.log
```

- 语料本体在 `data/prompt-corpus/`（**gitignored**，不随仓库分发）⇒ 克隆后需自备才能复跑。
- 三者是**同一任务的三轮迭代，并非严格超集**：`analyze.py` 独有 schema 探索、
  `analyze2.py` 独有 i18n / spec 宽高比 / safety_rating ⇒ **按需保留，勿互相替代**。
- 合规：语料 **CC BY 4.0**（可商用、需署名，署名记于 `docs/video-prompt-data-sources.md`），
  但本项目红线是**只提炼范式、不搬运提示词正文进仓库** ⇒ 结论只落 `docs/`。
- 详见 `docs/seedance2-corpus-analysis.md`（§6 复用方式、§附 目录）。

## 依赖关系
```
sd_h3_pipeline.py ──┐
h3_install.py      ──┼──> model_manager.py（复用 download_url / check_bin / load_catalog）
                    │
sd_h3_compat_probe.py（独立，零依赖）
```
