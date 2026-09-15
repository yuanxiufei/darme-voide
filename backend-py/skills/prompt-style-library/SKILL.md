---
name: prompt-style-library
description: 图像提示词风格词库 — 七段结构与镜头/光线/调色锚点词表、Danbooru tag 体系与画风兼容矩阵（中文视频范式见 video-prompt-library）
preconditions:
  - 已明确当前剧集或角色的画风（dramas.style / characters.style / app_settings.art_style）
protocol:
  - prompt_style_used: 本次提示词对应的画风 key 与所选的镜头/光线锚点
workflows:
  - 分镜拆解
  - 图片提示词
  - 视频提示词
# 默认注入的 Agent（skill 自描述绑定，机制见 backend/src/agents/skills.ts）
agents: [storyboard_breaker, grid_prompt_generator]
# 注入顺序，越小越靠前（自身 skill 为 10，本词库作为补充素材排在其后）
priority: 20
---

# 提示词风格词库

本 SKILL 解决一个问题：**为什么同样的画面描述，出来的图时好时坏**。
答案是模型对「镜头规格 / 光位 / 胶片调色」这类**具体锚点**响应极强，对「电影感」「唯美」这类**空泛形容词**几乎无响应。

适用于 `grid_prompt_generator`（出图提示词）与 `storyboard_breaker`（分镜画面描述）。

---

## 0. 职责边界（最重要，先读这条）

**画风英文词不由你拼，由后端收口。**

项目画风（写实电影 / 日式动漫 / 国风水墨 …）由后端 `backend/src/shared/prompt-utils.ts`
按统一解析链（`characters.style` → `dramas.style` → `app_settings.art_style` → `realistic`）
**自动追加**到提示词末尾，并在负面词里排除对立风格。链路已**全覆盖且正负成对**：角色 / 装备 / 道具 / 表情、场景图、分镜静帧 / 宫格图、视频
（正向追加当前画风速写，负向排除对立风格）—— 具体函数名与你无关，不必知道。

所以：

- ✅ 你只负责**画面内容**：主体、外观、动作、镜头、光线、环境、氛围
- ❌ 你**不要**写 `anime style` / `photorealistic` / `ghibli style` / `cel shading` 这类画风词
- ❌ 你**不要**写画风的负面词（`no anime, no cartoon` 等）
- ❌ 视频链路**不要**写英文「反 AI 感 / 现实瑕疵」词族（`imperfect autofocus` 等）——
  后端 `buildVideoArtStyleSuffix` 已按画风白名单注入（仅写实向注入，动漫/水墨注入会破坏风格）；
  你只需在**画面内容**里用中文自然表达质感要求

原因：重复叠加会与后端注入的画风词冲突，造成**同一剧内画风漂移**（历史上出现过这个问题）。

你真正要做的是：知道当前画风是什么 → 挑**与该画风兼容**的镜头/光线/调色词。

### 画风 ↔ 锚点兼容矩阵

| 画风 key | 中文 | 该补什么锚点 | 不要补 |
|---|---|---|---|
| `realistic` | 写实电影 | 焦段（35mm/85mm）、景深、practical light、胶片调色 | 夸张光效、超现实色 |
| `cinematic` | 电影感 | 强反差 low-key、体积光、变形镜头光斑、宽银幕构图 | 平光、亮调 |
| `noir` | 黑色电影 | 硬光、百叶窗影、烟雾、荷兰角 | **任何颜色词**（会破坏黑白） |
| `anime` | 日式动漫 | Danbooru tag、视角 tag、表情 tag | 胶片颗粒、写实皮肤 |
| `ghibli` | 吉卜力 | 自然日光、手绘背景、生活化细节 | 霓虹、强反差 |
| `ink-wash` | 国风水墨 | 留白、意境、雾气、宣纸质感 | 高饱和色、复杂纹理 |
| `watercolor` | 水彩 | 柔光、淡彩、纸纹 | 硬阴影、锐利边缘 |
| `comic` | 美漫漫画 | 强透视、动势线、网点 | 柔光、低反差 |
| `cyberpunk` | 赛博朋克 | 霓虹雨夜、湿地反光、全息招牌 | 日光、温暖色调 |
| `pixar3d` | 三维动画 | 柔和全局光、圆润造型 | 胶片颗粒、网点 |

---

## 1. 七段结构（所有提示词的骨架）

```
主体 + 外观服装 + 动作表情 + 镜头语言 + 光线 + 调色/质感 + 画质
```

按此顺序平铺为**英文逗号分隔短语**（动漫向见第 4 节，改用 tag 体系）。
顺序不是硬性要求，但**主体必须在最前**——模型对前几个 token 权重最高。

示例（写实向）：

```
a middle-aged detective in a worn trench coat, tired eyes, stubble,
sitting alone at a diner counter, holding a coffee cup,
medium close-up, 85mm lens, shallow depth of field,
low-key lighting, warm practical lamps overhead,
teal and orange color grading, subtle film grain,
cinematic film still, highly detailed
```

---

## 2. 镜头语言词库

**景别**（Shot size）
`extreme wide shot` / `wide shot` / `full shot` / `medium shot` / `medium close-up` /
`close-up` / `extreme close-up` / `over-the-shoulder shot`

**机位**（Camera angle）
`eye level` / `low angle` / `high angle` / `dutch angle` / `bird's eye view` /
`worm's eye view` / `over-the-shoulder` / `profile view`

**焦段**（Focal length — 写实向最有效的真实感锚点）
`14mm ultra wide` / `24mm wide` / `35mm` / `50mm standard` / `85mm portrait` / `135mm telephoto` / `macro`

**运镜**（Camera movement — 视频提示词必填）
`static locked-off` / `slow dolly in` / `dolly out` / `tracking shot` / `handheld` /
`crane up` / `whip pan` / `orbit around subject` / `push in on face`

**构图**（Composition）
`rule of thirds` / `centered symmetrical composition` / `leading lines` /
`negative space` / `frame within frame` / `foreground occlusion`

> 注意：本项目已有独立的运镜词表（`camera-movement-guides.ts`）。
> 分镜/视频提示词的运镜若已有既定取值，**沿用既有值**，不要另造同义说法。

---

## 3. 光线与调色词库

**光位**（Lighting setup）
`key light` / `fill light` / `rim light` / `backlight` / `Rembrandt lighting` /
`split lighting` / `butterfly lighting` / `side lighting` / `top light`

**光源 / 时间**（Source & time）
`golden hour` / `blue hour` / `harsh noon sun` / `overcast daylight` / `dusk` /
`practical lamps` / `neon signage` / `candlelight` / `moonlight` / `firelight` / `window light`

**氛围**（Mood）
`high-key lighting` / `low-key lighting` / `chiaroscuro` / `volumetric god rays` /
`haze` / `backlit silhouette` / `soft ambient bounce`

**调色 / 胶片质感**（Color grade & texture — 写实向的真实感来源）
`teal and orange grading` / `bleach bypass` / `desaturated palette` / `warm amber tones` /
`cold cyan tones` / `crushed blacks` / `Kodak Portra film emulation` / `Cinestill 800T` /
`35mm film grain` / `halation` / `lens flare` / `chromatic aberration` / `soft bloom`

---

## 4. 动漫向：改用 Danbooru tag 体系

**二次元模型（含本项目接入的动漫类底模）几乎都按 Danbooru tag 训练**，
写自然语言句子效果远差于写标签。

| ❌ 自然语言 | ✅ Danbooru tag |
|---|---|
| 一个黑发长发女孩穿百褶裙站在侧面 | `1girl, solo, long black hair, pleated skirt, from side` |
| 短发男生特写，笑着看镜头 | `1boy, short hair, close-up, smile, looking at viewer` |

**常用 tag 维度**

- 数量 / 主体：`1girl` / `1boy` / `2girls` / `solo` / `multiple girls`
- 头发：`long hair` / `short hair` / `black hair` / `twintails` / `ponytail` / `ahoge` / `messy hair`
- 眼睛：`blue eyes` / `red eyes` / `closed eyes`
- 服装：`school uniform` / `sailor collar` / `pleated skirt` / `hoodie` / `armor`
- 视角 / 取景：`from side` / `from below` / `from above` / `profile` / `upper body` / `full body` / `cowboy shot`
- 表情：`smile` / `open mouth` / `blush` / `crying` / `angry` / `expressionless`
- 场景：`indoors` / `outdoors` / `night` / `rain` / `classroom` / `cityscape`

**规则**：tag 之间用英文逗号分隔；**不要写完整句子**；视角与取景 tag 必给（否则构图随机）。

---

## 5. 取词来源

需要**新词**时去哪找，见 `docs/prompt-style-sources.md`（Civitai / Danbooru / PromptMart /
PromptHero / Promptomania / Film-Grab 六个来源的用法）。

**这些站点你访问不了**，本文件不再罗列 —— 你的任务是用**已有词表**组合出提示词，
而不是去查资料。缺词时优先用第 2–4 节里的近义锚点，或向用户说明缺少哪个维度的词。

**项目内已有**（外部技能库，需在「Agent 配置 → 绑定 Skills」按需启用）：
`film-style-picker`（12 大类电影风格参考）与 `film-reference-prompt-writer`（参考图写词）。

---

## 6. 视频提示词范式 → 已拆为独立 skill

原第 6 节（中文视频提示词范式 **6.0–6.10**）已拆为独立 skill **`video-prompt-library`**
（仅默认注入 `storyboard_breaker`；**出图场景不需要**）。涉及 `video_prompt` 时请参照该 skill；
本文件的 §7 硬性禁止项与 §8 自检清单对图像与视频同样适用。

---

## 7. 硬性禁止项

以下内容一旦写进提示词，轻则浪费额度，重则触发审核拦截：

1. **不要写真人姓名 / 在世导演姓名**（如「某某某风格」）——肖像权与版权风险
2. **不要写画面内可读文字**——与项目 `UI_OVERLAY_RULE` 一致：
   手机/电脑/新闻/短信/监控/地图/时间码等屏幕一律**只留白面**，文字后期叠加，防止 AI 生成乱码
3. **不要写画面内背景音乐**——项目统一后期配乐
4. **不要堆无意义质量词**——`masterpiece, best quality, 8k, ultra detailed` 堆超过 3 个收益递减，
   优先把额度花在镜头/光线锚点上
5. **不要在同一句里混用两套体系**——`realistic skin texture, cel shading` 这类自相矛盾的词
   会让模型二选一，输出不稳定

---

## 8. 自检清单

输出提示词前逐条确认：

- [ ] 主体是否在最前面？
- [ ] 是否**没有**写画风词（第 0 节）？
- [ ] 是否给了具体镜头（景别 + 机位，写实向再加焦段）？
- [ ] 是否给了具体光位或光源？
- [ ] 动漫向是否用了逗号分隔的 tag 而非句子？视角 tag 是否给了？
- [ ] 是否避开了真人姓名 / 画面内文字 / 画面内音乐？
- [ ] 词序是否与当前画风兼容（对照第 0 节矩阵）？
