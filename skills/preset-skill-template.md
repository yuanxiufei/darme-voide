# Preset Skill Template（预设风格视频 Skill 通用模板）

## 模板说明

本文件是一个**可复用的 Skill 设计骨架**。所有 `{PLACEHOLDER_*}` 标记和 `{{...}}` 包裹的内容均为占位符，实际使用时需替换为具体领域知识。

**核心设计模式：**
- **Variation Card 多样性引擎**：随机组合 → 去重检查 → 配比约束
- **阶段化流水线**：Card 生成 → Drama 创建 → 图片生成 → 视频生成
- **Style Lock 固定风格块**：生图/生视频 prompt 中逐字复制的不可变描述
- **负向提示词清单**：逐项禁止的不良视觉元素
- **质量闸门**：每阶段失败即回溯

---

## 模板骨架

### 阶段 1：Planner（规划）

**职责**：输入 Skill 定义，输出任务拆分和阶段依赖

**输出结构：**
```
{
  skillName: "{PRESET_NAME}",
  stages: ["generate_card", "create_drama", "generate_images", "generate_videos"],
  dependencies: {
    create_drama: ["generate_card"],
    generate_images: ["create_drama"],
    generate_videos: ["generate_images"]
  }
}
```

### 阶段 2：Designer（设计）

**职责**：定义 Variation Card 数据池和组合规则

**数据池接口：**

```typescript
interface PresetDataPools {
  // 主题家族 — 每组代表一种路线/风格变体
  themeFamilies: string[]
  // 构图模式 — 镜头构图语法
  compositionPatterns: string[]
  // 核心焦点 — 画面中的唯一主视觉中心
  mainFocalPoints: string[]
  // 主题线索 — 嵌入画面的微量暗示元素（分低调/显性两档）
  thematicClues: {
    subtle: string[]
    prominent: string[]
  }
  // 活动 — 画面中人物/角色的微小动作
  activities: string[]
  // 灵动元素 — 非必要、可选的生物/动态点缀（最多 N 个镜头出现）
  livingElements: string[]
  // 光线结构
  lightStructures: string[]
  // 机位
  cameraPositions: string[]
  // 风向
  windDirections: string[]
}

// 单镜头配置
interface PresetShotConfig {
  shotIndex: number           // 1–N
  themeFamily: string         // 路线家族
  compositionPattern: string   // 构图语法
  spaceType: string           // 空间类型（前景）
  foregroundFrame: string     // 前景框架
  mainFocalPoint: string      // 唯一主焦点
  thematicClue: string        // 主题线索
  activity: string            // 活动
  cameraPosition: string      // 机位
  windDirection: string       // 风向
  lightStructure: string      // 明暗结构
  characterLayout: string     // 人物排列
  livingElement: string | null // 灵动元素
}

// 完整 Variation Card
interface PresetVariationCard {
  themeFamily: string
  shots: PresetShotConfig[]
}
```

**配比约束规则：**
- 灵动元素总量限制：`livingElement !== null` 的 shot 数 ≤ {MAX_LIVING_SHOTS}
- 柔性排他：连续两张 Card 的 themeFamily 不应完全相同
- 主题线索混合：`subtle` 线索与 `prominent` 线索比例约为 {RATIO}

**空间类型派生逻辑：** 根据 `themeFamily` + `mainFocalPoint` 派生 `foregroundFrame` 和 `spaceType`

**人物布局逻辑：** 根据 `activity` 类型随机选取排列方式

### 阶段 3：Generator（生成器）

**职责**：调用外部 AI 服务生成图片和视频

**接口：**
- `generatePresetImage(prompt: string) => Promise<{ imageUrl: string }>`
- `generatePresetVideo(prompt: string, referenceImageUrl: string) => Promise<{ videoUrl: string }>`

### 阶段 4：Prompt Writer（提示词构建器）

**Style Lock 固定块（不可变）：**

```typescript
export const PRESET_STYLE_LOCK = `{{STYLE_DESCRIPTION_PLACEHOLDER}}`

export const PRESET_IMAGE_NEGATIVE = `{{IMAGE_NEGATIVE_PLACEHOLDER}}`

export const PRESET_VIDEO_NEGATIVE = `{{VIDEO_NEGATIVE_PLACEHOLDER}}`
```

**构建函数：**
- `buildPresetImagePrompt(shot: PresetShotConfig): string` — 合并 Style Lock + shot 特有描述
- `buildPresetVideoPrompt(shot: PresetShotConfig, cameraMove: string): string` — 首帧保真 + 微动 + 约束

### 阶段 5：Assembler（组装器）

**职责**：将生成的图片和视频组装为项目中的 Drama/Episode/Storyboard 记录

---

## Variation Card 生成算法

```
function generateVariationCard(excludeFamily?: string):
  1. 从 themeFamilies 中随机选取 1 个（排除 excludeFamily）
  2. 确定 5 个 shot 的分配方案：
     - 构图模式：从 compositionPatterns 随机不重复取 5 个
     - 主焦点：从 mainFocalPoints 随机不重复取 5 个
     - 主题线索：按配比随机分配 subtle/prominent
     - 活动：从 activities 随机不重复取 5 个
     - 灵动元素：从 livingElements 随机分配到 ≤ MAX_LIVING_SHOTS 个 shot
  3. 每个 shot 派生 spaceType + foregroundFrame
  4. 每个 shot 随机分配 cameraPosition + windDirection + lightStructure + characterLayout
  5. 返回 PresetVariationCard
```

---

## 管线端点 API

| Method | Path | 说明 |
|--------|------|------|
| GET | `/preset/framework/variation-card?excludeFamily=` | 生成新 Variation Card |
| POST | `/preset/framework/create` | 创建 Drama + Storyboards |
| POST | `/preset/framework/generate-images` | 批量生成首帧图片 |
| POST | `/preset/framework/generate-videos` | 批量生成视频 |
| GET | `/preset/framework/status/:dramaId` | 查询管线状态 |
| POST | `/preset/framework/full-pipeline` | 一键全流程 |

### POST create
```json
{
  "title": "{项目名称}",
  "description": "{描述}",
  "variationCard": { /* PresetVariationCard */ }
}
```
→ `{ dramaId, episodeId, storyboardIds: number[] }`

### POST generate-images / generate-videos
```json
{
  "dramaId": 1,
  "episodeId": 1,
  "storyboardIds": [1,2,3,4,5]
}
```
→ `{ dramaId, episodeId, genIds: number[] }`

### POST full-pipeline
```json
{
  "title": "{项目名称}",
  "description": "{描述}",
  "variationCard": { /* PresetVariationCard */ },
  "autoGenerateImages": true
}
```
→ `{ dramaId, episodeId, storyboardIds, imageGenIds, variationCard }`

---

## 复用指导

### 如何基于此模板创建新 Skill

1. **复刻本文件** → `skills/{your-skill-name}.md`
2. **替换所有 `{{PLACEHOLDER}}`** → 填入你的领域数据
3. **填充数据池**（themeFamilies / mainFocalPoints / activities 等）
4. **撰写 Style Lock** → 逐字精炼的风格描述块
5. **撰写负向提示词** → 针对你领域的禁止清单
6. **调整配比约束** → 灵动元素上限 / 线索比例等
7. **创建后端服务文件**：`backend/src/services/preset-{name}.ts`
8. **创建路由文件**：`backend/src/routes/preset-{name}.ts`
9. **注册路由**：`backend/src/index.ts` 中添加 `api.route('/preset/{name}', preset{Name})`
10. **创建前端预设页面**：`frontend/app/pages/preset/{name}.vue`
11. **注册前端 API**：`frontend/app/composables/useApi.ts` 中添加 `{name}API`
12. **添加入口按钮**：`frontend/app/pages/index.vue` Header 中添加导航按钮

### 可复用的模式表

| 模式 | 适用场景 | 核心机制 |
|------|---------|---------|
| Variation Card | 任何需要"随机可控"的批量生成 | 数据池 + 去重 + 配比 |
| Style Lock | 需要严格风格一致性 | 不可变 prompt 块 |
| 三层空间 | 固定镜头需要景深 | Foreground/Midground/Background |
| 柔性排他 | 避免连续重复 | excludeFamily 参数 |
| 阶段闸门 | 复杂管线控制 | 每步失败即中断 |
| 负向清单 | 质量保证 | 逐项禁止 |
| 灵动元素 | 增加镜头多样性 | 限量随机分配 |
