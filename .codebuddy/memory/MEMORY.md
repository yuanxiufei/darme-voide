# 项目长期记忆

## 品牌标识 (2026-08-07)
- **项目名称**：Drama Studio（短剧工坊），原"Drama Studio/Drama Drama"
- **代码标识**：drama-studio / PRESET_*（原 drama / DRAMA）
- **Logo 文件**：`frontend/app/assets/brand-logo.png`（原 drama-logo.png）
- **数据库名**：`data/drama.db`（原 drama.db）
- **localStorage key 前缀**：`drama:`（原 `drama:`）
- **API 路由**：`/quick-preset`（原 `/quick-preset`）
- **npm 包名**：drama-studio-frontend
- **Docker 容器名**：drama-studio
- 全项目 0 残留Drama引用，清理日期 2026-08-07

## 统一视觉风格系统 (2026-08-07)
- **`backend/src/shared/prompt-utils.ts`** 是全域 prompt 构建的唯一入口
- 风格预设 `VISUAL_STYLE_MASTER` = `cinematic illustration style, consistent art style, soft cinematic lighting, high quality, no text, no watermark`
- 所有图片/视频生成必须通过本模块构建 prompt，禁止硬编码风格字符串
- 角色图片：`buildCharacterImagePrompt(char)` → name + appearance + personality + style
- 场景图片：`buildSceneImagePrompt({location, time, prompt, dramaStyle})`
- 分镜图片：`buildStoryboardImagePrompt({characterDesc, sceneDesc, shotType, ...})`
- 分镜视频：`buildStoryboardVideoPrompt({characterAppearances, scenePrompt, action, ...})`
- 辅助查询：`getStoryboardCharacterAppearances/SceneDescription/ImageUrls()`

## TTS 角色-音色-台词验证机制 (2026-08-07)
- 后端 storyboards.ts 新增类型 `TTSMatchStatus`: `matched` | `no_voice` | `not_found` | `narrator`
- `validateTTSSpeaker(speaker, dramaId)`: 验证角色名 → 剧组角色记录 → voiceStyle 三连匹配
- `validateDialogueLines()`: 批量验证 dialogue 所有台词行
- 新增端点 `GET /storyboards/:id/validate-dialogue` 独立返回验证结果
- generate-tts 多人模式循环中先用 `validateDialogueLines` 验证，再以验证后的 voiceId 调用 TTS
- 单人模式也加入验证，返回 match_status
- PUT 路由保存 dialogue 时返回 `dialogue_validation` 字段
- 前端 episode 页面：`loadAllDubMatchCache()` 在 refresh 时自动加载，`dubIssueCount` computed 显示问题数
- 多人对话模板改为基于 `getDialogueLines(sb)` 渲染（不依赖 TTS 是否已生成）
- 状态颜色：绿色=匹配，黄色=未配置音色，红色=角色不存在
- 生成配音前弹出 confirm 对话框警告不匹配项

## TTS 音色过期检测 (2026-08-07)
- validate-dialogue 端点解析 ttsAudioUrl JSON，比对 voice_id 与当前角色 voiceStyle
- 前端 dubStaleCount computed + 蓝色过期横幅提示重新生成配音

## 全链路一致性规则 (2026-08-07)
- **角色图片**：prompt 必须包含 name+appearance+personality → `cinematic illustration style, stylized character design`
- **场景图片**：prompt 必须包含 location+time+drama.style → `highly detailed cinematic environment`
- **分镜图片**：prompt 必须注入角色外观文本+场景描述+镜头参数+叙事内容
- **分镜视频**：prompt 必须注入角色外观列表+场景+叙事+动作+运镜；需传角色参考图
- **Grid 宫格图**：cell prompt 注入 `buildCharacterAppearanceText()` 角色外观文本
- **分镜保存**：dialogue 中角色名 vs character_ids 自动校验
- 所有生成路径禁止硬编码 `cinematic portrait/专业摄影/电影质感` 等风格字符串

## 资源库系统 (2026-08-07)
- 4个资源库：角色库(character_templates)、场景库(scene_templates)、兵器库(weapon_templates)、服装库(costume_templates)
- 后端路由：`routes/characterLibrary.ts`, `sceneLibrary.ts`, `weaponLibrary.ts`, `costumeLibrary.ts` - 均使用 Hono Router
- API 前缀：`/api/v1/character-library`, `/scene-library`, `/weapon-library`, `/costume-library`
- 前端页面：`pages/library/index.vue`(总览), `characters.vue`, `scenes.vue`, `weapons.vue`, `costumes.vue`
- 通用功能：分页列表+模糊搜索+多维度筛选(分类/标签/专项字段)、CRUD、批量删除、应用到项目(角色/场景)、从项目保存到库
- 角色库特有：视觉形象(外貌/服装/表情/性别/年龄段)+声音配置(音色/提供商/语调/语速JSON)
- 场景库特有：环境+氛围+光线+时间+风格+季节+天气+生成prompt
- 兵器库特有：类别/类型/品级/属性JSON(威力/稀有度/元素/特效)/材质/所属角色
- 服装库特有：风格/部位/材质/配色/季节/外观描述
- 所有4张表共用：soft delete(via deleted_at)、usage_count 引用计数、source_drama_id 来源追溯、tags JSON数组、metadata JSON扩展
- 导航栏新增"资源库"链接(书架图标)，位于设置下方

## 全流程多模型 fallback 机制 (2026-08-10)
- 配置表 `model` 字段为 JSON 数组，存储全部可用模型按优先级排序
- `getActiveConfig()` 返回完整 `models: string[]` 数组
- 各阶段遍历 models 数组，失败自动切换到下一个，全部失败才标记 failed
- 图片/视频生成：每次切换自动释放旧 GPU 租约并获取新租约（不挤占 GPU 资源池）
- Agent 文本生成：每次重试重新构建 Agent 实例（Mastra Agent 构造时绑定 model）
- 日志埋点：每次 fallback 记录 attempt/totalModels/model 字段

## AI 视频生成工作台编辑器系统 (2026-08-10)
- 每阶段模型切换：ModelSelector 组件按 service_type 动态拉取模型列表
- 角色编辑弹窗：CharacterEditor（4区块: 基本信息/形象设定/图片生成/声音配置）
- 场景编辑弹窗：SceneEditor（3区块: 场景信息/环境设定/图片生成）
- 视频编辑弹窗：VideoEditor（参考图/提示词/关联角色/模型选择）
- 单镜头重新生成：POST /storyboards/:id/regenerate-image（支持自定义 prompt + model）
- 视频重新生成：PUT /videos/:id（参数更新）+ POST /videos/:id/regenerate（重新生成）
- TTS 增强：角色级 voiceSpeed/voiceEmotion/voicePitch/voiceModel → 通过 getCharacterVoiceParams() 注入 TTS 调用
- emotion 标签映射：happy/sad/angry/excited/calm/serious→MiniMax 平台值
- 角色扩展字段：clothing, weapons, customPrompt（图片生成 prompt 覆盖）
- 场景扩展字段：description, atmosphere, lightning, weather, season, style, customPrompt

## 通用预设框架 (2026-08-10)
- 原"仙宫导览" Skill 已重构为通用预设框架，剥离全部领域内容
- **Skill 模板**：`skills/preset-skill-template.md` — 含 5 阶段骨架、接口定义、Variation Card 算法伪码、复用指导
- **后端服务**：`backend/src/services/preset-framework.ts` — 占位符数据池（THEME_FAMILY_A 等）+ Variation Card 引擎 + 管线编排
- **后端路由**：`backend/src/routes/preset-framework.ts` — 6 端点（`/preset/framework/*`）
- **前端页面**：`frontend/app/pages/preset/framework.vue` — 5 阶段 UI（Welcome→Preview→Confirm→Progress→Done）
- **前端 API**：`presetFrameworkAPI`（generateCard/create/fullPipeline/status 等）
- **prompt-utils**：`PRESET_STYLE_LOCK/IMAGE_NEGATIVE/VIDEO_NEGATIVE` — 均为 `{{PLACEHOLDER}}` 占位文本
- **核心设计模式保留**：阶段化流水线 / Variation Card 多样性引擎 / 去重检查表 / Style Lock 固定风格块 / 三层空间公式 / 质量闸门 / 负向提示词清单
- **扩展方式**：创建新 Skill 时，复制骨架 → 填充数据池 → 撰写 Style Lock → 注册路由+前端页面
- 全项目 "仙宫/xian-gong/XianGong" 0 残留，清理日期 2026-08-10
