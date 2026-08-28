# 项目长期记忆（Drama Studio 短剧工坊）

## 项目与技术栈
- 路径 `d:\code\voides\voide-darme`；标识 drama-studio / PRESET_*；DB `data/drama.db`；localStorage 前缀 `drama:`。
- 前端 Nuxt 3.17+Vue3.5（端口 3013）；后端 Hono 4.12 + Mastra(5 Agent) + Drizzle ORM + better-sqlite3（端口 5789）；媒体 fluent-ffmpeg+sharp。API 前缀 `/api/v1`，22 路由 + `/webhooks/*` + `/static/*`，DB 19 表。
- Node **v22.20.0**（v24 致 better-sqlite3 ERR_DLOPEN_FAILED）。config.yaml gitignored，仅 config.example.yaml 跟踪。
- 启动：`cd backend && npm run start`、`cd frontend && npm run dev`。
- 图片上传 `POST /api/v1/upload/image` → `data/static/uploads/` → `/static/uploads/*`。

## 主题色（唯一源 frontend/app/assets/studio.css :root）
- 青绿 teal `#0d9488`（2026-08-22 由冷蓝统一改）。令牌：`--accent #0d9488`/`--accent-dark #0f766e`/`--accent-text #0f766e`/`--accent-bg rgba(13,148,136,.1)`/`--accent-glow rgba(13,148,136,.2)`/`--accent-gradient linear-gradient(135deg,#14b8a6,#0d9488 46%,#0f766e)`/`--border-focus #0d9488`。
- 浅色：`--bg-base #f3f6fb`/`--bg-0 #fff`/`--bg-1 #f8fbff`/`--bg-2 #eef3f9`/`--text-0 #182132`/`--text-2 #60718a`/`--text-3 #8fa0b8`/`--border #dbe4f0`。
- 禁止硬编码 accent 蓝、`#1a1a2e`、或引用不存在的 `--card-bg`/`--input-bg`/`--text`。

## 统一 adapter 抽象（核心架构原则，用户明确强调）
- 4 类 adapter(image/video/tts/text) 都是 `Record<provider,Adapter>`，线上(minimax/gemini/volcengine/vidu/ali/openai/chatfire) 与本地(ollama/local-sd/cosyvoice) 同表解析，靠 `AIConfig{provider,baseUrl,apiKey,model}` 切换，本地 baseUrl 指 localhost、线上指厂商域名，上游零感知。文本最彻底——openai 兼容一个实例覆盖 7 provider。
- **任何新模型（含本地 H3）只加/复用 provider + 改 baseUrl，不引入新框架（如 ComfyUI）**。
- GPU 租约 `gpuManager.acquire(serviceType,provider,model,baseUrl)`，serviceType∈text/image/video/audio。
- 火山引擎必须用 Endpoint ID（ep-m-xxx）非模型 ID。
- `ai_service_configs.model` 为 JSON 数组（多模型优先级 fallback）。

## 制作流程
- 4 主阶段：剧本 script→资产 assets→分镜 storyboard→导出 export。后端编排 scripting→extracting→voicing→storyboarding→imaging→videoing→composing→merging（音色 08-22 前移）。
- 5 Agent：script_rewriter/extractor/storyboard_breaker/voice_assigner/grid_prompt_generator。
- Prompt 唯一入口 `backend/src/shared/prompt-utils.ts`（`VISUAL_STYLE_MASTER` + buildCharacterImagePrompt/buildSceneImagePrompt/buildStoryboardImagePrompt/buildStoryboardVideoPrompt）。正/反向优先级见该文件与 text-generation。

## 六键 Bible + H3 视频（本地 MiniMax H3）
- 六键：CHAR_ID/SPEAKER_ID/VOICE_ID/LOCATION_ID/COSTUME_ID/STYLE_ID（`services/bible-ids.ts` + 3 列 dramas.styleId/characters.costumeId/scenes.locationId）。
- H3=音视频联合生成，双 checkpoint：FL2VA(动作/空镜/转场) + Ref2VA(单人说话/多对话，需声线参考)。参考音频=reference conditioning(≤3条各2-15s总≤15s)，非最终对白。声线样本源=characters.voiceSampleUrl（voice_style 变更置空）。
- 铁律：ONE_SHOT_ONE_SPEAKER、「一角色一音色禁共用」、speaker_id 全局唯一（voice-tools assign_voice 硬校验）。
- 路由：auto-pipeline submitMissingVideos 按 scene_type 路由（对话类→referenceMode multiple+多参考图/音频）；minimax-video `resolveH3Checkpoint` 按 scene_type 路由 FL2VA/Ref2VA；`MULTI_REFERENCE_PROVIDERS=volcengine/vidu/minimax`。
- QC 规则分 `services/qc-scoring.ts`（lip_sync*0.4+consistency*0.3+continuity*0.3）+ video_quality_checks 表，视频完成自动触发。真实唇形/相似度 AI 检测未挂载。
- 待办：本地 H3 服务部署；checkpoint_map 配到 ai_service_configs.settings；第一轮测试 A–G。

## 音色体系（role_tags 已后端化 2026-08-26）
- `ai_voices.role_tags`（TEXT JSON：旁白/主角/反派/配角）；`text-generation.inferVoiceRoleTags` 批量 LLM 打标；sync 后自动打标（失败不阻塞，留空前端回退正则）。
- `list_voices` 返回 role_tags + 可选 role_tag 筛选；前端 mapVoiceProfile 优先后端 role_tags。
- TTS 验证：validateTTSSpeaker/validateDialogueLines 三连匹配；`GET /storyboards/:id/validate-dialogue`；PUT 保存 dialogue 返回 dialogue_validation；前端 dubStaleCount 过期横幅。
- 已完成（2026-08-26）：B2 音色库独立试听 `POST /ai-voices/preview`；B4 `characters.role_type` 枚举（ensureColumn + `inferRoleType` + CharacterEditor select）；B5 声纹克隆 `POST /ai-voices/clone`（MiniMax files/upload→voice_clone，`services/voice-clone.ts`，前端 episode.vue 克隆入口 + `useApi.reqForm`）；B6 根据剧人物批量生成专属音色 `POST /ai-voices/generate-from-characters`（批量写入 ai_voices + 更新角色 voiceStyle 为克隆 voice_id，role→role_tags 映射；2026-08-27 已端到端跑通，6 角色生成 `ds_mtak...` 专属音色，preview/list 验证可用）。**参考音频须 ≥10s**：角色试听 voiceSampleUrl 仅约 4s 会报 `voice duration too short`，已改为即时用角色 voiceStyle 合成约 15s 参考音频（REF_TEXT）再克隆。待办：真实唇形/相似度 AI 检测挂载、本地 H3 部署、checkpoint_map 配置。

## 数据删除语义
- soft delete（读取必加 isNull(deletedAt)）：characters/dramas/agentConfigs/episodes/资源库 4 表。
- hard delete：scenes/storyboards/videos/images/aiConfigs/aiVoices。无 DELETE：videoMerges/videoGenerations。

## 关键坑
- 响应码：`success()` 返回 `{code:200,data,message:'success'}`；badRequest/notFound 返回 `{code:4xx,message}`；前端判 `!==200`、错误读 `data.message`（非 msg）。资源库 4 表老代码仍 c.json({code:0})。
- Hono 4.x `/*` 通配符不注册命名参数，取内容 `c.req.routePath + c.req.path.slice(prefix)`（skills.ts wildcardId）。
- default prompt 被 DB 快照固化：agent_configs 残留历史 systemPrompt 会覆盖 DEFAULT_PROMPTS（已清空 5 内置 agent）。
- 契约文档 `docs/api-contract.md`（改接口需同步更新）。
- 本地 Ollama 仅 qwen2.5-coder 支持 tool calling；CPU 推理极慢（文本 >120s），重要任务走云端。

## 功能索引
- 资源库 4 库 `/api/v1/{character,scene,weapon,costume}-library`（分页+搜索+筛选+CRUD+批量删+soft delete+usage_count+source_drama_id），前端 pages/library/*；weapon/costume `POST /from-character/:characterId`。
- 角色图：三视图(generate-three-views)/校色(8 参 ColorGradeParams+applyColorGrade)/装备特写(generate-equip-image)/智能拆分(auto-split-visuals, splitCharacterVisuals)。
- preset 框架：presets 表 `{type,name,config}` CRUD，type=colorGrade 校色预设；services/routes/pages preset/*。
- 工作台编辑器：CharacterEditor(4)/SceneEditor(3)/VideoEditor；角色级 voiceSpeed/Emotion/Pitch/Model；emotion→MiniMax 映射。
- 在线文本（复用 MiniMax key）：`https://api.minimax.chat/v1/chat/completions`，model MiniMax-Text-01(~5s)/abab6.5s-chat(~0.5s)；DB text id=8「MiniMax 在线文本」priority=300 优先本地 id=4。
