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
