# 视频 Prompt 数据源清单（已实测核验）

> 核验时间：2026-09-12　核验方式：GitHub REST API + HuggingFace API（经 `hf-mirror.com`）
> 用途：为「小说 → 短剧 → MiniMax H3」链路补充真实视频提示词语料，喂给 `prompt-style-library` skill。
>
> **重要**：网上流传的这批链接里，数量级描述普遍失真，且有 3 个是空壳仓库。
> 本文所有数字均为**实测**，与转述的说法不一致处已标注。

---

## 0. 一句话结论

**真正有数据、且能低成本拿到的只有 2 个源**：

| 优先级 | 数据源 | 真实规模 | 文本载荷 | 许可证 |
|---|---|---|---|---|
| **① ✅ 已下载** | Seedance 2 Prompt Dataset | **8755 条** | **37.58 MB** | **CC BY 4.0（可商用）** |
| ② 建议做 | Semonxue/awesome-video-prompts | 3097 条 | 稀疏拉取约几十 MB | ⚠️ **无 LICENSE** |
| ③ 顺带做 | Rapidata Video Gen Preference | ~200 条 / 14 类 | 极小 | 需确认 |
| ④ 仅内部研究 | TIP-I2V | 170 万条 | 746 MB（10 万条子集） | ❌ **CC BY-NC（禁商用）** |
| ✗ 跳过 | HitPaw / Simonrh01 / fantasylights / geekjourneyx | **0 条** | — | — |

**本机直连 `huggingface.co` 全部超时**，必须走 `hf-mirror.com`（项目 `model_manager.py` 已有该镜像逻辑，可直接复用）。

---

## 1. GitHub 源核验

### ✅ Semonxue/awesome-video-prompts —— 真数据，可采集

- 存在；**14★**；仓库体积 4.88 GB；最后推送 2026-06-17；**已归档（archived，只读）**
- 站方自报：3097 条提示词 / 26 个模型标签 / 210 个活跃标签
  （数字来自其徽章接口 `awesomevideoprompts.com/api/stats.json`，属自报）
- 覆盖模型：Seedance 2.0、Kling 3.0 / 2.6、Veo 3、Grok、Hailuo、Gen 4.5、Sora、Vidu Q3

**数据结构（纯 Markdown + YAML，非 HTML）**

```
content/prompts/2026-01/[id]-[slug].md     ← 数据本体（按月分目录）
static/prompts/2026-01/[id]-[slug]/
    ├── cover.jpg                          ← 4.88GB 的体积来源
    └── video.mp4
data/models.yaml                           ← 模型定义
data/tags.yaml                             ← 标签定义
```

单条 `.md` 的 frontmatter：

```yaml
---
title: "..."
image: "/prompts/2026-01/xxx/cover.jpg"
video: "/prompts/2026-01/xxx/video.mp4"
date: "2026-01-23"
description: "完整的提示词内容..."     ← 提示词正文在这里
models: "kling26"
tags: ["mountain", "aerial", "fpv"]
author: "..."
source_url: "..."
---
```

**采集方式**：`content/` 是纯文本，**不要 `git clone` 整个 4.88 GB**，用稀疏检出：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/Semonxue/awesome-video-prompts.git
cd awesome-video-prompts
git sparse-checkout set content data
```

- ⚠️ **许可证：仓库无 LICENSE 文件**，README 也未声明条款。仅供**内部学习范式**，不得整段搬运进产品资产。
- ⚠️ 已归档，语料定格在 2026-06。

### ❌ geekjourneyx/awesome-ai-video-prompts —— 空骨架，不可用

- 存在；**78★**（星数最高，容易误导）；MIT；仓库仅 **24 KB**；最后推送 2026-01-04
- 实际内容：README 是**待填充的目录** —— **0 条提示词**，26 个指向 `docs/**/*.md` 的链接，3 个工具链接全部指向占位域名 `example.com`
- 另注意：抓取 URL 为 `geekjourneyx`，但正文徽章指向 `geekjourney`（无 x），来源存疑
- **价值**：其分类目录（提示词工程 / 电影语言 / 官方指南 / 音画同步 / 最佳实践）可作为我们 skill 的目录参考，但**没有语料可采**

### ❌ 三个空壳仓库（转述中被称为"最值得爬"）

| 仓库 | 星 | 体积 | 实际内容 |
|---|---|---|---|
| `HitPaw-Official/awesome-ai-video-prompts` | 0 | 24 KB | 厂商营销仓库，MIT，无数据量 |
| `Simonrh01/awesome-ai-video-prompts` | 1 | **4 KB** | 空壳 |
| `fantasylights/ai-video-prompts` | 0 | **4 KB** | 空壳 |

---

## 2. HuggingFace 源核验（经 hf-mirror.com）

### ✅ GokuScraper/seedance-2-prompts-datasets —— **最高性价比，首选**

- 存在；下载量 **186,399**；45 likes；最后更新 2026-08-27；共 19,648 个文件
- **许可证：CC BY 4.0 —— 可商用，需署名**
- 真实规模：**8755 条**（转述称"20 万+"，**不实** —— 50 GB 指配套视频，不是提示词数量）
- 内容：Seedance 2.0/1.0 提示词 + 生成视频 + 预览帧

**关键：不用下 50 GB。** 全部提示词与元数据集中在一个文件里：

```
metadata.jsonl      37.58 MB   ← 8755 条 prompt 全在这里
seedance-2/         ~50 GB     ← 视频与封面，可完全跳过
```

```bash
curl.exe -L -o metadata.jsonl \
  https://hf-mirror.com/datasets/GokuScraper/seedance-2-prompts-datasets/resolve/main/metadata.jsonl
```

**✅ 已于 2026-09-12 下载**：`data/prompt-corpus/seedance2/metadata.jsonl`（`data/prompt-corpus/` 已 gitignore，
第三方文本不入库；分析脚本在 `backend-py/app/scripts/corpus/`）。

**实测数据结构（比预期好）**：

- 提示词**不在顶层**，在 `i18n.zh.p` / `i18n.en.p`；
- **`i18n.zh` 全量 8755 条都有中文原文**（还有中文标题 `t` 与中文标签 `tags`）；
- 英文版是**平台扩写版**（中位 809 字 vs 中文 245 字）→ **统计只用 `i18n.zh.p`**；
- 另有 `category` / `spec`（时长、尺寸、比例）/ `model_info` / `platform`；
- ⚠️ 此集**不是多模型混合**：91% 是 Seedance 自家（seedance 2.0 = 6393、1.0 = 1552、
  happy-horse-1 = 671、gemini-omni = 138）。早前清单里「26 个模型」指的是 Semonxue 那个仓库，两者别混。
- ⚠️ `spec.duration` 有脏值（最小 `-3.69e17`）→ 用前按 `0 < d <= 300` 过滤。

**完整语料分析见** → [`docs/seedance2-corpus-analysis.md`](./seedance2-corpus-analysis.md)
（词频、结构特征、真实模板样例、对 skill 第 6 节的修正结论）

> 相关性最高：**中文视频模型语料**、含真人口播/短剧类创作，与 MiniMax H3 的域最接近。

### ✅ Rapidata/awesome-text2video-prompts —— 小，但分类体系有价值

- 存在；下载量 291；15 likes；最后更新 **2025-01-22**（较旧）；201 个文件
- 真实身份：**"Rapidata Video Generation Preference Dataset"** —— 人工 + ChatGPT-4o 生成的提示词，
  覆盖 **14 个能力类别**，每类配 1 个 Sora 生成视频
- 规模**很小**（约 200 个视频），不是"大规模语料"。**价值在分类法，不在量**：
  `Object Interactions` / `Camera Movements` / `Spatial Relationship` /
  `Dynamic Attribute Binding` / `Text` / `Special Worlds` 等
- 文件：`Videos/*.mp4` + `data/`（提示词表）

> 用法建议：把它的 14 类**能力分类**映射到 `prompt-style-library` 的自检清单，用来查漏
> （例如"我这条提示词覆盖了空间关系吗？"）。不要指望从它拿到大量语料。

### ⚠️ WenhaoWang/TIP-I2V —— 规模最大，但**禁止商用**

- 存在；GitHub 侧 42★；HF 下载量 4671；ICCV 2025 论文（arXiv:2411.04709）
- 真实规模：**170 万条**真实用户 text + image prompt
  （配套 5 个模型的生成视频：Pika / Stable Video Diffusion / Open-Sora / I2VGen-XL / CogVideoX-5B）
- **许可证：CC BY-NC 4.0 —— 非商业性使用**

**文本载荷体积**

| split | 条数 | 文本+图像 prompt 体积 | 可用文件 |
|---|---|---|---|
| Full | ~170 万 | ~13.4 GB | `data/Full-00000..00026-of-00027.parquet` |
| **Subset** | **10 万** | **~746 MB** | `data/Subset-00000/00001-of-00002.parquet`（373 + 372 MB） |
| Eval | 1 万 | ~80 MB | `data/Eval-00000-of-00001.parquet` |

```python
from datasets import load_dataset
# 需先 set HF_ENDPOINT=https://hf-mirror.com
ds = load_dataset("WenhaoWang/TIP-I2V", split="Subset", streaming=True)
```

- ❌ **CC BY-NC 禁商用** → **不得进入产品资产 / 分发给用户**，只能内部做词频统计、分布研究
- ⚠️ 任务是 **I2V（图生视频）**，且语料以**英文**为主，与"中文短剧文生视频"域偏离较大
- 定位：**统计用途**（真实用户在写什么、长度分布、高频结构），**不是**风格模仿来源

---

## 3. 建议的采集顺序

```
第 1 步 ✅ 已完成（2026-09-12）
  下载 Seedance2 metadata.jsonl（37.58 MB，CC BY 4.0）
  → 8755 条中文原始提示词，已落盘 data/prompt-corpus/seedance2/
     （分析脚本 backend-py/app/scripts/corpus/analyze*.py）
  → 已提炼词频与模板，写入 skill 第 6.10 节
     （分析见 docs/seedance2-corpus-analysis.md）

第 2 步（低成本）
  稀疏拉取 Semonxue 的 content/ 与 data/（几十 MB）
  → 3097 条多模型（Kling/Veo/Sora/Hailuo）提示词 + models.yaml/tags.yaml 分类体系
  ⚠️ 无许可证，仅内部参考，不入仓库

第 3 步（很小）
  拉 Rapidata 的 data/ 提示词表
  → 提取 14 类视频生成能力分类法，用于完善 skill 自检清单

第 4 步（可选，仅内部研究）
  TIP-I2V 的 Subset（746 MB）
  → 只做统计与词频，❌ 禁商用，禁止进产品

跳过：HitPaw / Simonrh01 / fantasylights / geekjourneyx（0 条数据）
```

---

## 4. 落到本项目怎么用

采集到的语料**不是**要抄进仓库，而是提炼成 `backend-py/app/skills/prompt-style-library` 的**范式与词表**：

| 语料 | 提炼目标 |
|---|---|
| Seedance2 metadata.jsonl | 中文短视频提示词的**句式模板**、段落组织、常用运镜/特效词 |
| Semonxue 3097 条 | 多模型通用**标签体系**（对照其 210 个 tag）、模型间写法差异 |
| Rapidata 14 类 | **能力覆盖自检清单**（空间关系 / 文字渲染 / 属性绑定…） |
| TIP-I2V | 真实**长度分布**与高频结构（回答"用户实际写多长"） |

**红线（与 skill 第 6.9 节一致）**：
- 只提炼范式，**不得整段搬运他人提示词正文**进仓库
- CC BY-NC 的语料**禁止商用**，不进产品
- CC BY 4.0 的语料若直接使用需**署名**
- 无许可证的仓库（Semonxue）按"仅内部参考"处理

---

## 5. 环境注意

- 本机 `huggingface.co` **直连超时**，一律走镜像：
  - 命令行：`$env:HF_ENDPOINT = "https://hf-mirror.com"`
  - 直链：把 `https://huggingface.co/...` 替换为 `https://hf-mirror.com/...`
- 项目内已有三源切换逻辑（HF / hf-mirror / ModelScope），见 `backend-py/app/scripts/model_manager.py`，
  大规模下载应复用它自带的**断点续传**，不要另写一套
