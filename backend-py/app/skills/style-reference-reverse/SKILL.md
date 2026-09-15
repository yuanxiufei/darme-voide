---
name: style-reference-reverse
description: 从参考图反推画风与提示词 — 把用户给的截图拆成可复用锚点，并映射到项目画风枚举
preconditions:
  - 用户已提供参考图（本地路径或 URL）
protocol:
  - suggested_style_key: 建议落库的画风 key（必须来自项目画风枚举）
  - reusable_anchors: 可复用的风格锚点（镜头 / 光线 / 调色各若干）
workflows:
  - 图片提示词
  - 画风设置
# 默认注入的 Agent（skill 自描述绑定，机制见 `backend-py/app/agent/skills.py`）
# 暂不默认注入：本项目当前**没有「参考图反推画风」的执行入口**（无对应 Agent、无 UI 触发点、
# 无路由消费其 protocol 字段 suggested_style_key / reusable_anchors）。
# 此前绑在 grid_prompt_generator 上，而后者是纯出图 Agent ⇒ 每次出图白占约 3,000 字符上下文。
# 待该能力接入（有 Agent 或 UI 触发点）后再在此绑定即可，skill 内容本身无需改动。
agents: []
# 注入顺序，越小越靠前
priority: 30
---

# 参考图反推画风

**触发场景**：用户丢来一张图说「按这个风格做」「要这种质感」「这张图怎么写出提示词」。

目标不是「复刻这张图」，而是把这张图**拆成可复用的风格锚点**，
并落到项目的画风体系上，让后续整部剧都能稳定复用。

---

## 1. 四步反推法

### 第 1 步：先判画风大类（不要先看细节）

只回答一个问题：**这是「实拍感」还是「绘画感」还是「三维感」？**

判据：
- 皮肤有毛孔/瑕疵、有真实景深虚化 → 实拍系
- 大色块平整填充、有描边线条 → 二维绘画系
- 体积感强、材质有次表面散射、造型圆润 → 三维系

这一步决定后面所有取词方向，**判错会导致全盘返工**。

### 第 2 步：拆六要素

对应用户图，逐项给出可复用描述（**不写画风词**，画风由后端收口）：

| 要素 | 要提取什么 | 例子 |
|---|---|---|
| 主体 | 人物/物品的客观特征 | 中年男性、胡茬、旧风衣 |
| 镜头 | 景别 + 机位（写实向加焦段） | medium close-up, low angle, 85mm |
| 光线 | 光位 + 光源 + 明暗基调 | side lighting, practical lamp, low-key |
| 调色 | 主色倾向 + 质感 | teal and orange, film grain |
| 环境 | 地点 + 时间 + 氛围 | 深夜霓虹街道、湿地反光 |
| 情绪 | 一句话气质 | 压抑、疏离 |

### 第 3 步：映射到项目画风 key

**反推结果必须落到项目已有的画风 key**，否则整部落库与生成链路不一致（前端选不到、后端没对应词表）。

| 观察到的特征 | 建议 key |
|---|---|
| 真人质感、电影镜头、胶片调色 | `realistic` |
| 真人质感但强反差、体积光、商业大片感 | `cinematic` |
| 黑白、硬光、百叶窗影、烟雾 | `noir` |
| 赛璐璐上色、描边、日式角色 | `anime` |
| 手绘、温暖自然、生活化背景 | `ghibli` |
| 水墨晕染、大面积留白、宣纸感 | `ink-wash` |
| 淡彩、晕染过渡、纸纹 | `watercolor` |
| 粗描边、网点、强透视 | `comic` |
| 霓虹雨夜、高饱和洋红青、全息 | `cyberpunk` |
| 三维渲染、圆润造型、柔和全局光 | `pixar3d` |

若确实无法归入任何一类，**明确告知用户**「这张图超出项目现有画风，最接近 X」，
不要硬套（错配的画风词会污染整剧）。

### 第 4 步：输出结论

```yaml
suggested_style_key: realistic          # 必须来自项目枚举
confidence: high                        # high / medium / low
reason: 真人皮肤质感 + 35mm 景深 + 胶片颗粒，判定为写实电影向
reusable_anchors:
  camera: [medium close-up, low angle, 85mm lens, shallow depth of field]
  lighting: [side lighting, practical lamp, low-key, rim light]
  grading: [teal and orange, subtle film grain, crushed blacks]
subject: 中年男性，胡茬，旧风衣
environment: 深夜街边小店，湿滑地面
mood: 压抑、疏离
excluded:                              # 明确不要复用的部分
  - 画面中的人物身份特征（属于原图创作，不迁移）
  - 可见文字/logo（按 UI_OVERLAY_RULE 一律不生成）
```

---

## 2. 怎么落地到项目

反推得到 `suggested_style_key` 后，按需选择落点：

| 场景 | 落点 |
|---|---|
| 整部剧统一画风 | 剧集设置 → `dramas.style` |
| 单个角色特殊画风 | 角色编辑 → `characters.style`（留空则跟随剧集） |
| 全项目默认 | 设置 → 画风设置 → `app_settings.art_style` |
| 只想复用某几个锚点词 | 写进该镜头的提示词（锚点词可自由用，画风词不行） |

**注意**：切换画风后，已生成的旧图不会自动重绘，需要重新生成对应资产。

---

## 3. 反推工具参考

| 工具 | 地址 | 说明 |
|---|---|---|
| Civitai 图片详情 | civitai.com | 站内图自带完整 prompt + 采样参数，**信息最全** |
| CLIP Interrogator | replicate.com/pharmapsychotic/clip-interrogator | 通用图像 → 关键词反推 |
| imgtoprompt / imageprompts | imgtoprompt.app、imageprompts.co | 轻量在线反推 |
| ReversePrompt | GitHub `Fei-ops267/ReversePrompt` | 本地部署反推 |
| Danbooru 反查 | danbooru.donmai.us | 动漫向：反查标准 tag 名 |

**但请注意**：若是 Liblib / 吐司 / Civitai 上的图，**优先直接看图自带的信息**——
反推工具猜出来的采样参数远不如原图元数据准确。

---

## 4. 约束与红线

1. **不迁移创作性内容**：只借鉴「风格手段」（镜头、光线、调色），
   不复制原图的具体角色设计、构图创意、品牌标识
2. **不做真人身份推断**：即便判断出是某位真人，也不要写出姓名，更不要用于生成
3. **不写具体作者/工作室名**：风格相似本身不侵权，但署名式描述（「某某某风格」）有风险
4. **画面内文字一律不生成**：参考图里的字幕/招牌/UI 文字属于后期叠加范畴
5. **confidence 为 low 时必须明说**：宁可让用户确认，也不要给出一个错误映射
