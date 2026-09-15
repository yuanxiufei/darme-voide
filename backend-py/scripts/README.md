# backend-py/scripts/ — 工具与自检脚本（**全部 Python**）

> 2026-09-15 从仓库根 `scripts/` 整体搬来，并把原来的 Node 守卫（`.mjs`）与 PowerShell
> 脚本**逐条移植为 Python**。现在仓库根**没有** `scripts/` 了。
>
> 为什么放在后端项目里：这些脚本与后端**共用同一套运行时**（Python 3.8+，仅标准库），
> 不再需要「Node 工具链 + Python 工具链」两套；放 `backend-py/scripts/` 也让
> 「后端自检（`backend-py/tests/`）」与「仓库自检（本目录）」在同一棵树里，找得着。

本目录有**四类**：

| 类别 | 脚本 | 运行环境 |
|---|---|---|
| **仓库自检** | `check_skill_refs.py` / `check_memory.py` / `test_guards.py` / `check_all.py` —— 防资产**静默漂移** | Python 3.8+，**仅标准库**（零第三方依赖） |
| **语料管线** | `corpus/fetch_raw.py` → `normalize.py` → `search.py`（+ `analyze{,2,3}.py` 统计） | Python 3.8+，**仅标准库** |
| **AI/GPU 工具链** | `model_manager.py` / `sd_h3_pipeline.py` / `sd_h3_compat_probe.py` / `h3_install.py` | Python 3.8+，**仅标准库**（与后端通过 subprocess 解耦） |
| **一次性迁移** | `migrate_models.py`（模型硬链接迁移到 ComfyUI Desktop 共享库；路径是**本机事实**，换机器要改） | Python 3.8+，**仅标准库** |

> 四类都**不参与产品运行时**。与之相对，**后端行为契约**的自检在 `backend-py/tests/`
> （`run_all.py` = 63 套件 / 2464 项，那份清单才是权威；改后端代码请跑它）。

## 仓库自检脚本（pre-commit 会按资产自动触发）

| 文件 | 职责 |
| --- | --- |
| `check_skill_refs.py` | 校验 `backend-py/skills/**/*.md` 里的路径引用是否都能落地（`references/` 资产、跨 skill `../`、及 `backend-py/skills/`+`backend/`+`backend-py/`+`frontend/`+`docs/` repo 根相对路径），防「改 skill 名 / 挪库 / 改 docs 名」造成的**静默断链** |
| `check_memory.py` | 校验 `.codebuddy/memory/` 三层记忆：`MEMORY.md` ≤ 8k 字符（超限注入会被截断）、`INDEX.md` 的 `@行号` 锚点有效、且**每篇日志的末节都已登记**、磁盘日志都已登记、落点表路径有效 |
| `test_guards.py` | **两套守卫的自检**：在临时副本上造 12 种场景（记忆层 8 + 引用层 4），断言每项检查仍能报致命（防「守卫被改哑但基线仍绿」） |
| `check_all.py` | **一键跑全部自检**（上表三道串联 + 汇总）—— pre-commit 只在「本次提交触及相应资产」时才跑对应守卫，本脚本用于**全局体检** |

```powershell
python backend-py/scripts/check_all.py                   # 一键跑全部三道自检，退出码 1 = 有任一失败
python backend-py/scripts/check_all.py --verbose         # 同上，并把 --verbose 透传给引用守卫

python backend-py/scripts/check_skill_refs.py            # 单跑：有断链则退出码 1，可用作提交前自检
python backend-py/scripts/check_skill_refs.py --verbose  # 单跑：额外列出被跳过的候选，审计脚本自身盲区
python backend-py/scripts/check_memory.py                # 单跑：记忆层自检，退出码 1 = 存在致命项
python backend-py/scripts/test_guards.py                 # 单跑：改了任一守卫脚本后必跑，退出码 1 = 有用例失败
```

判定分级（细节见脚本头注释）：

- **致命**：`references/…`、跨 skill `../…`、repo 根相对路径（`backend/`、`backend-py/`、`frontend/`、`backend-py/skills/`、`docs/`）指向不存在 → 退出码 1
- **非致命**：缺 `scripts/…` —— 外部技能库只随行 `SKILL.md` + `references/`，上游 `scripts/` 普遍未 vendored，故只列出
- **跳过**：上游 / 外来宿主路径（`.ci/`、`spec/`、`.opencode-v2/`、`.claude/`、`.agents/`）、含通配符的模式、**示意引用**（`e.g.` / `such as` / `例如` **紧邻**于路径之前）、基准不明（不在任何 skill 内 / 非资产目录开头）
- 各类跳过**分别计数**且可 `--verbose` 逐条审计 —— 「跳过」不是静默丢弃

### 记忆层判据（`check_memory.py`）

- **致命**：三层文件缺失；`MEMORY.md` 超 8k 字符；`@N` 越界或未落在「小节首行」；**某篇日志的末节没有登记锚点**；**磁盘上的日志未登记进 INDEX**；落点表引用的路径不存在、或 `文件 §小节` 的小节名在该文件内查不到
- **提示**：`MEMORY.md` 逼近预算（余量 < 400）
- ⚠️ **字符口径**：按 Node 的 `String.length`（UTF-16 码元）算，且读文件**不做换行转换** —— 这两点都是实测踩出来的（见文件头）
- 刻意**不校验**索引里的「N 行 / M 轮」快照数：口径不一致，校验只会产生噪声

### 守卫自检（`test_guards.py`）为什么必要

检查逻辑被改坏、正则被放宽、白名单被删时，**基线往往是绿的**（真实仓库本来就合规）⇒
「守卫没报错」不等于「守卫还能报错」。本脚本把手工造故障的负向实证固化为可重跑用例：
拷真实资产到临时目录 → 造故障 → 断言「退出码 1 + 命中预期文案」；**真仓库全程只读**。

## 语料管线（视频提示词语料）

分析**视频提示词语料**并支撑「找相似镜头」检索。**仅 Python 3.8+ 标准库**。

| 文件 | 职责 |
| --- | --- |
| `corpus/fetch_raw.py` | **采集**：从 GitHub 拉原始语料到 `data/prompt-corpus/<源>/raw/`（已存在则跳过，`--only=<id>` 可单源重试） |
| `corpus/normalize.py` | **归一化**：各源 → 统一 JSONL（`_normalized/prompts.jsonl`），只抽「能检索的最小字段集」 |
| `corpus/search.py` | **检索**：在归一化语料上做「找相似镜头」（2-gram 打分，毫秒级，零依赖） |
| `corpus/analyze.py` | 首轮结构探查（字段、长度、语言） |
| `corpus/analyze2.py` | 深挖 i18n / raw_p / category / spec |
| `corpus/analyze3.py` | **主力统计脚本**：词表、结构特征、标签频次 |

```powershell
python backend-py/scripts/corpus/fetch_raw.py                              # ① 采集
python backend-py/scripts/corpus/normalize.py                              # ② 归一化 → prompts.jsonl
python backend-py/scripts/corpus/search.py --q-file=tmp/q.txt --top=5      # ③ 检索
```

> ⚠️ `search.py` 的查询**优先用 `--q-file`**（或 `CORPUS_QUERY` 环境变量）：
> PowerShell 命令行直传中文参数会乱码。
>
> ⚠️ **同口径的 Agent 工具在 `backend-py/app/services/agents/tools/corpus_tools.py`** ——
> **改打分口径必须两处同步**，否则「命令行验证有效」与「Agent 实际效果」会分叉。

**已采集源 / 已排除源**（判据 = 许可证 + 真实性，全部实测过）：

| 源 | 规模 | 许可 | 状态 |
|---|---|---|---|
| `GokuScraper/seedance-2-prompts-datasets` | 8755 条 | CC BY 4.0 | ✅ 已采集 |
| `flaqai/awesome_seedance_2_5` | 120 场景（中英各 60） | MIT | ✅ 已采集 |
| `Ericgood/seedance-prompt` | 112 条（英文 prompt + 中文元数据） | CC BY 4.0 | ✅ 已采集 |
| `skylenage/FilmBench` | 1169 条（T2V 515 / R2V 654） | ⚠️ 待核 | ⏳ 待采 |
| `Rapidata/awesome-text2video-prompts` | ~200 条 | 需确认 | ⏸ 只要其 14 类分类法 |
| `tipi2v/TIP-I2V` | 170 万条 | **CC BY-NC 禁商用** | ❌ 只能内部统计，**不进语料库** |
| `Semonxue/awesome-video-prompts` | 4.88 GB | **无 LICENSE** | ❌ 仅内部参考 |
| `HitPaw` / `geekjourneyx` / `fantasylights` | **24 KB / 24 KB / 4 KB** | — | ❌ 实测**空壳**，0 条数据 |

> ❌ 那三个空壳与 TIP-I2V 曾在外部流传的「清单」里被标为 ★★★★☆ / 第一优先级 —— 实测结论相反，
> 故写在此处防止再被引回。

**体量与按需**（语料本体 37.58 MB / 8755 条，gitignored）：

- **不需要拆分**：三个统计脚本做的是**全量统计**，且实测单脚本耗时 ≤ 2 s、内存峰值 ≤ 75 MB。
- `analyze3.py` 已改为**边读边投影**（只留统计要用的 6 个字段）：内存峰值 **75.4 → 14.1 MB**。
- `analyze.py` / `analyze2.py` 是 **schema 探查** ⇒ **全量是语义需要**，故不做投影，只加按需上限
  （`SEEDANCE2_LIMIT=<N>`）。
- ⚠️ **默认行为零变化**：不设 `SEEDANCE2_LIMIT` 时，三者输出与历史存档**逐字一致**
  （已用 4164 / 7813 / 4364 字符逐字比对验证）。改动这些脚本后请照此复核。
- 三者是**同一任务的三轮迭代，并非严格超集**：`analyze.py` 独有 schema 探索、
  `analyze2.py` 独有 i18n / spec 宽高比 / safety_rating ⇒ **按需保留，勿互相替代**。
- 合规：语料 **CC BY 4.0**（可商用、需署名，署名记于 `docs/video-prompt-data-sources.md`），
  但本项目红线是**只提炼范式、不搬运提示词正文进仓库** ⇒ 结论只落 `docs/`。
- 详见 `docs/seedance2-corpus-analysis.md`（§6 复用方式、§附 目录）。

## AI/GPU 工具链（本地模型）

> **「本地服务根」在哪**：`backend-py/local_services/`（**2026-09-15 从仓库根 `local_services/` 迁入**）。
> `model_manager.py --runtime git` 会把第三方服务 `git clone` 到 `local_services/<key>/`，项目自带的
> H3 薄封装在 `local_services/h3/`（纳管）；**克隆来的子目录不入库**（见 `.gitignore` 的
> `backend-py/local_services/*` + `!backend-py/local_services/h3/`）。默认值可用 `LOCAL_SERVICES_DIR`
> 或 `configs/model-paths.json` 的 `local_services_dir` 覆盖（优先级：CLI > 环境变量 > 配置 > 默认）。


| 文件 | 语言 | 职责 |
| --- | --- | --- |
| `model_manager.py` | Python | 通用本地模型安装/管理（多类模型 + comfyui/ollama/git/manual 四种安装方式） |
| `sd_h3_pipeline.py` | Python | sd.cpp × MiniMax-H3 Turbo GGUF 一键编排（doctor → download → build → probe） |
| `sd_h3_compat_probe.py` | Python | GGUF 与 sd.cpp 兼容性判定（零依赖读元数据，可选 `--live` 真机 load） |
| `h3_install.py` | Python | 旧 H3 安装 CLI 兼容 shim（数据已并入 `model_manager.py`） |

```powershell
python backend-py/scripts/model_manager.py list [--category X] [--runtime Y] [--missing]
python backend-py/scripts/model_manager.py download [--key ...] [--category ...] [--required] [--all] [--force]
python backend-py/scripts/model_manager.py remove --key ...
python backend-py/scripts/model_manager.py doctor
python backend-py/scripts/model_manager.py install-nodes [--only ...]
python backend-py/scripts/model_manager.py add-model --key ... --name ... --category ... --runtime ... [...]
python backend-py/scripts/model_manager.py remove-model --key ...

python backend-py/scripts/sd_h3_pipeline.py                # 顺序执行 doctor→download→build→probe
python backend-py/scripts/sd_h3_compat_probe.py --list     # 支持矩阵
```

数据源：`configs/models.json`（模型清单）+ `configs/model-paths.json`（路径配置）。
路径优先级：命令行参数 > 环境变量 > `model-paths.json` > 默认探测。
编译依赖（仅 `build` 需要）：Git、CMake 3.x+、Visual Studio 2019/2022、CUDA Toolkit（nvcc）。
> 注：本机当前无 CUDA Toolkit（nvcc），`build` 只能产出 CPU 版，详见 `docs/local-h3-video-system.md`。

⚠️ **路径推导是个坑**：这几个脚本原来在仓库根 `scripts/`，仓库根 = `dirname(SCRIPT_DIR)` 一级；
搬进 `backend-py/scripts/` 后必须**上跳两级**（`model_manager.py` / `sd_h3_pipeline.py` 已改）。
只跳一级不会报错，只是 `models.json` 读不到 ⇒ **清单为空**。

## 依赖关系

```
sd_h3_pipeline.py ──┐
h3_install.py      ──┼──> model_manager.py（复用 download_url / check_bin / load_catalog）
                    │
sd_h3_compat_probe.py（独立，零依赖）
```

## 与 `.githooks/pre-commit` 的关系

改了 `backend-py/skills/` 或 `docs/` ⇒ 跑引用守卫；改了 `.codebuddy/memory/` ⇒ 跑记忆守卫；
改了本目录的三个守卫脚本 ⇒ 跑守卫自检。钩子**按资产条件触发**（快），
**全局体检请用 `check_all.py`**。钩子里的 Python 解释器解析顺序：
`backend-py/.venv` → `python3` → `python` → `py -3`（守卫只用标准库，任一个都行）。
