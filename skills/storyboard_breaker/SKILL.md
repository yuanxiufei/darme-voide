---
name: storyboard-breaker
description: 分镜拆解专业规范
preconditions:
  - 调用 read_storyboard_context 能读到剧本、角色和场景
protocol:
  - storyboards_count: 保存的分镜数量
---

# 分镜拆解指南

## 拆解原则

每个镜头聚焦**单一动作**，描述要详尽具体。

### 一镜一人说（ONE_SHOT_ONE_SPEAKER）——最高优先级铁律
- **一个镜头原则上只允许一个角色说话**。禁止一个镜头里多个角色轮流开口。
- 双人对白必须拆成**正反打**：A 说一句是一个镜头，B 说一句是下一个镜头，交替切。
- 多人会议/群戏：每一句台词单独成镜，谁说话镜头就给谁（可切旁人反应镜，但反应镜**无人说话**）。
- 长对白（连续 4 句以上）按 **5-8 秒**切片，不要塞进 10-15 秒让一个角色说一长串。
- 旁白/画外音：用 `<voice>角色名</voice>` 标记，且旁白镜头里**不再出现另一个角色说话**。

### 时长
- 动作/空镜/纯环境：10-15 秒。
- 含对话镜头：**5-8 秒**，只装 1-2 句台词，保证音画同步、不被剪断。

## 镜头要素

1. **镜头标题**：3-5字概括核心内容（如"噩梦惊醒"）
2. **时间**：具体时分 + 光线描述
3. **地点**：场景完整描述 + 空间布局 + 环境细节
4. **景别**：远景/全景/中景/近景/特写
5. **角度**：平视/仰视/俯视/侧面/背面
6. **运镜**：固定/推镜/拉镜/摇镜/跟镜/移镜
7. **动作**：谁 + 具体怎么做 + 肢体细节 + 表情
8. **对话**：该镜头的完整对话
9. **画面结果**：动作的即时后果 + 视觉细节
10. **氛围**：光线 + 色调 + 声音 + 整体氛围
11. **时长**：每个镜头 10-15 秒
12. **静态画面提示词**：`image_prompt`，用于首帧/尾帧/镜头图片生成
13. **视频提示词**：`video_prompt`，按 3 秒分段的视频生成描述（必填）
14. **配乐提示词**：`bgm_prompt`，描述该镜头适合的配乐风格
15. **音效提示词**：`sound_effect`，描述该镜头关键环境音/动作音
16. **场景关联**：若能匹配已有场景，必须填写 `scene_id`
17. **角色关联**：填写 `character_ids`，绑定当前镜头涉及的 0 到多个角色
18. **镜头类型**：填写 `scene_type`，用于路由视频生成方式，取值见下方「镜头类型路由」
19. **说话人**：填写 `speaker_id`（如 `S1`/`S2`），即本镜头**唯一说话人**的角色编号；无人说话则不填

## 视频提示词格式

每个镜头必须包含 `video_prompt` 字段，用于驱动 AI 视频生成：

```
0-3秒：<location>咖啡厅</location>，近景，<role>小明</role>低头看手机，表情焦虑。
<n>3-6秒：<location>咖啡厅</location>，全景，门铃响，<role>小红</role>推门走入。
<n>6-9秒：<location>咖啡厅</location>，中景，<role>小红</role>微笑走向小明，坐下。
```

标签说明：
- `<location>地点</location>` — 场景标记
- `<role>角色名</role>` — 角色标记
- `<voice>角色名</voice>` — 画外音/旁白标记
- `<n>` — 时间段分隔符

## 使用步骤

1. 调用 `read_storyboard_context` 读取剧本、角色、场景、已有分镜摘要
2. 先基于剧本完成镜头拆解，确保总时长和叙事连续性合理
3. 为每个镜头补全完整字段：`title / shot_type / angle / movement / location / time / character_ids / action / dialogue / description / result / atmosphere / image_prompt / video_prompt / bgm_prompt / sound_effect / duration / scene_id / scene_type / speaker_id`
4. 调用 `save_storyboards` 一次性保存完整分镜
5. 如需调整，调用 `update_storyboard` 修改具体镜头

## 场景关联规则

- 优先使用 `read_storyboard_context` 返回的 `scenes`
- `location + time` 可明确匹配时，必须回填正确 `scene_id`
- 不要凭空生成不存在的场景 ID
- 如果剧本内容明显落在已有场景中，不要重复创造新场景描述

## 角色绑定规则

- `character_ids` 必须从 `read_storyboard_context` 返回的角色列表中选择
- 一个镜头可以没有角色，也可以绑定多个角色
- 只要镜头里有明确出场、被看见、发生动作或说话的角色，都应绑定进去
- 纯环境镜头、空镜头、物件镜头可以传空数组

## 说话人规则（speaker_id）

- `speaker_id` 来自 `read_storyboard_context` 返回角色身上的 `speaker_id`（如 `S1`/`S2`），**不要自己编造**。
- 一个镜头只有一个 `speaker_id`，对应本镜头唯一开口的角色；无人说话就留空。
- 说话人的名字必须出现在 `dialogue` 里，且与 `character_ids` 绑定一致。

## 镜头类型路由（scene_type）

根据镜头内容填写 `scene_type`，用于视频生成自动选择参考图策略：

| scene_type | 场景 | 视频策略 |
|---|---|---|
| `single` | 单人说话/独角戏 | 首帧图生视频 |
| `dialogue_2p` | 双人正反打对白 | 多角色参考图（保证两人一致） |
| `meeting` | 三人及以上群戏/会议 | 多角色参考图 |
| `argument` | 激烈争吵/多人抢话 | 多角色参考图 + 快速正反打 |
| `long_dialogue` | 长对白（已 5-8s 切片） | 按切片单人逐镜 |
| `action` | 打斗/追逐/动作 | 首帧图生视频 |
| `silent` | 空镜/无对话/环境 | 首帧图生视频 |

- 无法确定时用 `single`。

## 质量要求

- `description` 要适合人读，`video_prompt` 要适合模型生成，二者不要互相替代
- `image_prompt` 要突出单帧构图、角色外观、环境和光线
- `video_prompt` 要突出时间推进、动作变化、镜头语言
- `bgm_prompt` 和 `sound_effect` 用简洁短语即可，但不能空泛到只有“紧张”“悲伤”
- 若存在旁白，统一写入 `dialogue`，格式为 `旁白：内容`
