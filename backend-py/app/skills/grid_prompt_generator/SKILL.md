---
name: grid-prompt-generator
description: 宫格图提示词生成 —— 三种模式的布局模板与常见坑
preconditions:
  - 已通过 read_shots_for_grid 拿到选中镜头的详细信息
protocol:
  - prompts_count: 生成的提示词数量
# 默认注入的 Agent（skill 自描述绑定，机制见 `backend-py/app/agent/skills.py`；不写 = 不默认注入）
agents: [grid_prompt_generator]
# 注入顺序，越小越靠前
priority: 10
---

# 宫格图提示词生成

对应 `grid_prompt_generator` Agent，**只负责宫格**。

- 角色立绘 / 场景背景图提示词由 `extractor` 负责，其结构骨架在系统提示词里
- 单镜画面结构骨架、英文 / 行列数 / `exactly N visible panels` 等硬约束也在系统提示词里
- 本文件只写宫格**特有的布局模板**与容易踩的坑

## 三种模式

### 首帧模式 first_frame
每格 = 一个镜头的起始画面，但**必须严格生成用户指定的 rows × cols 总格数**。

```
[rows x cols grid layout], exactly [rows*cols] visible panels, consistent art style, [style description],
格1: [shot 1 opening scene],
格2: [shot 2 opening scene],
格3: [shot 3 opening scene],
...
格N: [opening scene],
high quality, cinematic lighting, no merged panels, no missing panels, no text, no watermark
```

### 首尾帧模式 first_last
保持首尾帧节奏感，但**仍然严格生成 rows × cols 总格数**，不允许偷偷改成 Nx2。

```
[rows x cols grid layout], exactly [rows*cols] visible panels, consistent art style, [style description],
格1: [opening beat],
格2: [closing beat],
格3: [opening beat],
格4: [closing beat],
...
high quality, cinematic, continuous motion implied, no merged panels, no missing panels, no text
```

### 多参考模式 multi_ref
所有格子都是同一镜头的不同角度 / 构图参考，**仍然严格生成 rows × cols 总格数**。

```
[rows x cols grid layout], exactly [rows*cols] visible panels, same scene different angles, [style description],
[main scene description],
格1: wide shot establishing,
格2: medium shot character focus,
格3: close-up detail,
格4: dramatic angle,
...
consistent lighting and color palette, no merged panels, no missing panels, no text
```

## 容易踩的坑

1. **不要描述格子之间的分割线** —— 让模型自然生成网格，主动画线反而会画出多余的边框
2. **格位与参考图不混用**：`格1/格2/...` 只指宫格格位，参考图统一写 `图片1/图片2/...`
3. **格数与镜头数对不齐时**：格数少于镜头数 → 按镜头顺序取前 N 个；格数多于镜头数 → 用同镜头的不同瞬间补足，不留空格
4. **尺寸参考**：每格 960×540，总图 = 960×cols × 540×rows
5. **每格描述必须能独立成像**（含主体 + 环境 + 光线），不要只写「同上」「同上，换角度」

## 需要具体质感写法时：检索参考镜头（可选）

若某类**光线 / 构图 / 质感**的写法拿不准，可用 `search_reference_prompts` 检索本地约 9000 条
真实视频提示词。但宫格任务多数情况**用上面的布局模板 + 系统提示词里的结构骨架就够了** ——
**不要习惯性每个宫格都去查**。

- ✅ 借鉴**光线 / 构图 / 质感**的写法
- ❌ **不要照抄原文**（合规红线）；也不要把参考里的主体 / 品牌搬进来
- ⚠️ 返回 `available: false` 表示本机没装语料 ⇒ 改用已有词表，**不要编造**

## 要成体系的写法时：查专项库（可选，`search_script_library`）

`search_reference_prompts` 是「约 9000 条真实视频提示词」，强在**质感与光感**；
要**角色造型 / 场景模板 / 运镜实例 / 情绪外化 / 特效**这类成体系的写法，查专项库：

| 要什么 | 检索词举例 | 收窄 |
|---|---|---|
| 古风/现代人物的身份设定、造型、发型 | `古风男性角色 设定`／`女性造型 提示词` | `path_contains="library/general"` |
| 场景模板（古风 / 都市 / 跨题材） | `古风场景 三镜头 模板` | 同上 |
| 运镜实例（低空穿梭 / 权谋 / 虫视 / 鹰视） | `运镜 俯冲 实例` | 同上 |
| 情绪怎么写成可拍的画面 | `情绪表达 克制` | 同上 |
| 特效与施法手势 | `粒子 特效`／`施法 手势` | `library/general` 或 `library/ultimate` |

- `source="playbook"` = 专项方法（137 份）；`source="methods"` = 剧本与写作著作（**结构与节奏**问题查它）。
- 与上一节同样的纪律：只借鉴写法、**不照抄原文**；一次问一个具体维度；`available: false` 时改用
  本 skill 与 `prompt-style-library` 的已有词表，**不要编造**。宫格布局仍以上面的模板为准。
