# 提示词取词来源（人工维护用）

> **用途**：维护 `backend-py/skills/prompt-style-library` 的词表时，去哪找新词。
>
> **为什么不在 skill 正文里**：skill 的正文会被**注入给 Agent**（每次生成都算进上下文预算），
> 而 Agent **不能上网**——这些站点表对它零价值。放在这里既保留知识，又不让每次生成白背一段。
>
> **合规红线**见 `backend-py/skills/prompt-style-library/SKILL.md` §7 与本文末节。

---

## 1. 通用取词站点

| 来源 | 用途 | 怎么用 |
|---|---|---|
| **Civitai** | 最直接 | 点开任意图，右侧有完整 prompt + 采样器 + CFG，可直接对照 |
| **Danbooru** | 动漫 tag 的「官方语言」 | 查角色 / 服装 / 视角的标准 tag 名，避免自造词 |
| **PromptMart 谱码** | 短剧 / 漫剧专用 | **只取标签与范式**（已归纳进 `backend-py/skills/video-prompt-library` 的 6.x 节）：打斗、变身、分镜板类镜头指令（标签如 `#真人实拍` `#一镜到底` `#冲击波`）；正文多需付费解锁，**不要付费取词** |
| **PromptHero / PromptSpace** | 按风格检索 | 看别人同一题材怎么组织词序 |
| **Promptomania** | 可视化构建 | 勾选镜头 / 光位 / 风格，学结构（结果仅供参考，仍需去掉画风词） |
| **Film-Grab** | 写实向找构图参考 | `film-grab.com/?s={电影名}` 高清剧照，用来对光位与构图 |

## 2. 项目内已有的外部技能库

以下两个是**外部技能库**（`backend-py/skills/production-tools/`），不默认注入，
需在「**Agent 配置 → 绑定 Skills**」按需手动启用：

- `film-style-picker`：12 大类电影风格参考
- `film-reference-prompt-writer`：参考图写词

## 3. 语料数据集源

若目标是**大规模语料统计**（而非取词），见 `docs/video-prompt-data-sources.md`
—— 那里记录了各数据集的实测规模、真实文件树与**许可证约束**（CC BY-NC 禁商用等）。

## 4. 红线

- 只提炼**范式与词表**，**不得把他人提示词正文整段搬进仓库**（合规红线，与 `backend-py/skills/video-prompt-library` 的 6.9 节一致）。
- 参考图必须**自己用 AI 生成或用本项目自有资产**，不得直接使用他人成图。
- 部分站点需付费解锁（如 PromptMart 的「灵感豆」）：**只用免费与公开信息，不要为取词付费**。
