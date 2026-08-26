# Drama Studio 前后端接口契约

> 生成时间：2026-08-16
> 范围：后端 `backend/src/routes/*.ts`（22 文件）+ 前端 `frontend/app/composables/useApi.ts` / `useAgent.ts`
> 结论：前后端路径/方法 100% 对齐，无路径不一致；字段契约仅 1 处不一致（已修复，见 §4）

## 1. 通用约定

- **Base URL**：后端 `http://localhost:5789`，API 前缀 `/api/v1`（`/webhooks` 除外，挂载在根路径下）
- **统一响应格式**（`backend/src/shared/` 封装）：

```jsonc
// 成功（HTTP 200，code=0）
{ "code": 0, "data": { ... }, "msg": "ok" }
// 成功-创建（HTTP 201）
{ "code": 0, "data": { ... } }
// 失败（HTTP 4xx/5xx）
{ "code": 404, "data": null, "msg": "xxx not found" }
```

- 前端 `useApi.ts` 走 axios 拦截器：`code === 0` 才 resolve，`.then()` 内直接读 `res.data.data`；`.catch()` 内 `err.response.data.msg` 取错误消息。**业务层不再做 `res.status >= 200` 冗余判断**。
- **字段命名约定**：HTTP API 请求体/查询参数统一用 `snake_case`；后端 drizzle 列名为 `camelCase`，路由层做映射（部分端点双写兼容 snake/camel）。

### 1.1 删除语义（soft / hard delete）

| 表 | 语义 | DELETE 行为 |
|---|---|---|
| `characters` | soft | 打 `deletedAt=now()`，所有读取/更新/操作过滤 `isNull(deletedAt)` |
| `dramas` | soft | 同上 |
| `agentConfigs` | soft | 同上 |
| 资源库 4 表（character/scene/weapon/costume_templates） | soft | 同上 |
| `scenes` | hard | `db.delete()` 物理删除 |
| `storyboards` | hard | 物理删除（连带 `storyboardCharacters` 关联表） |
| `videos`（videoGenerations） | hard | 物理删除 |
| `images` | hard | 物理删除 |
| `aiConfigs` | hard | 物理删除 |
| `aiVoices` | hard | 物理删除 |
| `episodes` / `videoMerges` / `videoGenerations` | — | 无 DELETE 端点 |

## 2. 业务域端点清单

### 2.1 剧目 dramas —— `/api/v1/dramas`

| Method | Path | 说明 | 请求体/query | 响应 data |
|---|---|---|---|---|
| GET | `/` | 分页列表 | `page`, `page_size` | `{ items: [{...drama, characters: []}], total, page, pageSize }` |
| POST | `/` | 创建（自动建 1 集 episode） | `{ title, description?, genre?, style? }` | `{ id }` |
| GET | `/stats` | 统计（⚠️ 孤儿，前端未接） | — | 按 status 分组统计 |
| GET | `/:id` | 详情（嵌套 characters） | — | drama + `characters[]` |
| PUT | `/:id` | 更新 | `{ title?, description?, genre?, style?, status?, tags?, metadata? }` | — |
| DELETE | `/:id` | 软删 | — | — |
| PUT | `/:id/characters` | 批量保存角色（⚠️ 孤儿，前端用单条 update） | `{ characters: [...] }` | — |
| PUT | `/:id/episodes` | 批量保存剧集（⚠️ 孤儿） | `{ episodes: [...] }` | — |

### 2.2 剧集 episodes —— `/api/v1/episodes`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| POST | `/` | 创建 | `{ drama_id, episode_number, title? }` |
| PUT | `/:id` | 更新 | `{ content?, script_content?, title?, description?, status?, image_config_id?, video_config_id?, audio_config_id? }` |
| GET | `/:id/characters` | 集关联角色 | — |
| GET | `/:id/scenes` | 集关联场景 | — |
| GET | `/:id/storyboards` | 集关联分镜 | — |
| GET | `/:id/pipeline-status` | 管线状态 | `steps.{extract_characters,extract_scenes,storyboard,...}.count` |

### 2.3 分镜 storyboards —— `/api/v1/storyboards`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| POST | `/` | 创建 | `{ episode_id, scene_id?, character_ids?, storyboard_number, title?, duration? }` |
| PUT | `/:id` | 更新（20 字段映射 + `character_ids` 走关联表） | `title, description, shot_type, angle, movement, action, dialogue, duration, video_prompt, image_prompt, scene_id, location, time, atmosphere, result, bgm_prompt, sound_effect, custom_image_prompt, custom_video_prompt, character_ids` |
| POST | `/:id/generate-tts` | 生成配音（多人/单人模式） | `{ mode, voice_ids? }` |
| GET | `/:id/validate-dialogue` | 验证台词-角色-音色匹配 | — |
| POST | `/:id/regenerate-image` | 单镜头重生成图片 | `{ prompt?, model? }` |
| DELETE | `/:id` | 硬删（连带关联表） | — |

### 2.4 场景 scenes —— `/api/v1/scenes`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| GET | `/:id` | 详情 | — |
| POST | `/` | 创建 | `{ episode_id, name?, ... }` |
| PUT | `/:id` | 更新 | `{ name?, description?, location?, time?, atmosphere?, lightning?, ... }` |
| POST | `/:id/generate-image` | 生成场景图 | `{ episode_id, prompt? }` |
| DELETE | `/:id` | 硬删 | — |

### 2.5 角色 characters —— `/api/v1/characters`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| GET | `/:id` | 详情 | — |
| PUT | `/:id` | 更新（snake/camel 双写兼容） | `voice_style`/`voiceStyle`, `voice_provider`/`voiceProvider`, `voice_speed`, `voice_emotion`, `voice_pitch`, `voice_model`, `appearance`, `personality`, `clothing`, `weapons`, `custom_prompt`, `image_url`, `local_path` 等 |
| POST | `/:id/generate-voice-sample` | 生成音色样本 | `{ provider }` |
| POST | `/:id/generate-image` | 生成角色图 | — |
| POST | `/batch-generate-images` | 批量生成角色图 | `{ character_ids: [] }` |

### 2.6 图片 images —— `/api/v1/images`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| POST | `/` | 生成图片 | `{ drama_id?, character_id?, scene_id?, storyboard_id?, prompt, model? }` |
| GET | `/:id` | 详情 | — |
| GET | `/` | 列表 | `drama_id?` |
| DELETE | `/:id` | 硬删 | — |

### 2.7 视频 videos —— `/api/v1/videos`

| Method | Path | 说明 | 请求体/query |
|---|---|---|---|
| POST | `/` | 生成视频（含角色参考图自动补充） | `{ storyboard_id?, drama_id?, prompt, model?, reference_mode?, reference_image_urls?, image_url?, first_frame_url?, last_frame_url?, duration?, aspect_ratio?, character_ids?, config_id? }` |
| GET | `/:id` | 详情 | — |
| GET | `/` | 列表 | `storyboard_id?`, `drama_id?` |
| PUT | `/:id` | 更新参数 | `prompt, model, duration, reference_mode, aspect_ratio, character_ids, image_url, first_frame_url, last_frame_url` |
| POST | `/:id/regenerate` | 重新生成 | `{ model?, prompt?, config_id?, reference_mode?, reference_image_urls?, ... }` |
| DELETE | `/:id` | 硬删 | — |

### 2.8 宫格 grid —— `/api/v1/grid`

| Method | Path | 说明 |
|---|---|---|
| POST | `/prompt` | 生成宫格提示词 |
| POST | `/generate` | 生成宫格图 |
| POST | `/split` | 拆分宫格图 | 
| GET | `/status/:id` | 查询状态 |

### 2.9 合成 compose —— `/api/v1/compose`

| Method | Path | 说明 |
|---|---|---|
| POST | `/storyboards/:id/compose` | 单镜头合成 |
| POST | `/episodes/:id/compose-all` | 整集批量合成 |
| GET | `/episodes/:id/compose-status` | 合成状态 |

### 2.10 拼接 merge —— `/api/v1/merge`

| Method | Path | 说明 |
|---|---|---|
| POST | `/episodes/:id/merge` | 拼接成片 |
| GET | `/episodes/:id/merge` | 拼接状态 |

### 2.11 Agent —— `/api/v1/agent`

| Method | Path | 说明 |
|---|---|---|
| POST | `/:type/chat` | 调 Agent（type 如 `script_rewriter` 等） |
| GET | `/:type/debug` | ⚠️ 孤儿（开发调试端点） |

### 2.12 AI 配置 aiConfigs —— `/api/v1/ai-configs` + `/api/v1/ai-providers`

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 配置列表 |
| POST | `/` | 创建配置 |
| POST | `/quick-preset` | 快速预设 |
| POST | `/quick-local` | 快速本地配置 |
| POST | `/test` | 测试配置连通性 |
| GET | `/:id` | 详情 |
| PUT | `/:id` | 更新 |
| DELETE | `/:id` | 硬删 |
| GET | `/gpu/status` | GPU 租约状态 |
| POST | `/gpu/release-all` | 释放全部 GPU 租约 |
| GET | `/configs/local` | 本地配置 |
| GET | `/`（ai-providers） | 提供商列表 |

### 2.13 Agent 配置 agentConfigs —— `/api/v1/agent-configs`

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 列表 |
| GET | `/:id` | 详情 |
| POST | `/` | 创建 |
| PUT | `/:id` | 更新 |
| DELETE | `/:id` | 软删 |

### 2.14 Skills —— `/api/v1/skills`

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 列表 |
| GET | `/*` | 详情（通配符，routePath 动态提取 id） |
| PUT | `/*` | 更新 |
| POST | `/` | 创建 |
| DELETE | `/*` | 删除 |

### 2.15 音色 aiVoices —— `/api/v1/ai-voices`

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 音色列表 |
| POST | `/sync` | 同步音色 |

### 2.16 上传 upload —— `/api/v1/upload`

| Method | Path | 说明 |
|---|---|---|
| POST | `/image` | 上传图片（→ `data/static/uploads/`） |

### 2.17 资源库 4 表（共用结构）

前缀：`/api/v1/character-library` `/scene-library` `/weapon-library` `/costume-library`

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 分页列表（搜索/筛选） |
| GET | `/categories` | 分类 |
| GET | `/tags` | 标签 |
| GET | `/filter-options` | 筛选选项（scene/weapon/costume 有） |
| GET | `/:id` | 详情 |
| POST | `/` | 创建 |
| PUT | `/:id` | 更新 |
| DELETE | `/:id` | 软删 |
| POST | `/batch-delete` | 批量软删 |

额外端点：
- character-library：`POST /:id/apply`（应用到项目）、`POST /from-character/:characterId`（从项目保存到库）
- scene-library：`POST /:id/apply`、`POST /from-scene/:sceneId`

### 2.18 预设框架 preset-framework —— `/api/v1/preset/framework`（⚠️ 全孤儿）

| Method | Path |
|---|---|
| GET | `/variation-card` |
| POST | `/create` |
| POST | `/generate-images` |
| POST | `/generate-videos` |
| GET | `/status/:dramaId` |
| POST | `/full-pipeline` |

### 2.19 Webhooks —— `/webhooks`（不在 `/api/v1` 前缀下）

| Method | Path | 说明 |
|---|---|---|
| POST | `/vidu` | 第三方 vidu 回调 |

## 3. 前端 API 封装映射（`useApi.ts` / `useAgent.ts`）

| 前端封装 | 方法 | 对应后端端点 |
|---|---|---|
| `dramaAPI` | list/get/create/update/del | `GET /dramas`, `GET/PUT/DELETE /dramas/:id`, `POST /dramas` |
| `episodeAPI` | create/update/characters/scenes/storyboards/pipelineStatus | `POST /episodes`, `PUT /episodes/:id`, `GET /episodes/:id/{characters,scenes,storyboards,pipeline-status}` |
| `storyboardAPI` | create/update/generateTTS/validateDialogue/regenerateImage/del | 见 §2.3 |
| `characterAPI` | get/update/voiceSample/generateImage/batchImages | 见 §2.5 |
| `sceneAPI` | get/update/generateImage | `GET/PUT /scenes/:id`, `POST /scenes/:id/generate-image` |
| `imageAPI` | generate/list | `POST /images`, `GET /images` |
| `gridAPI` | prompt/generate/status/split | 见 §2.8 |
| `videoAPI` | generate/get/update/regenerate | 见 §2.7 |
| `composeAPI` | shot/all/status | 见 §2.9 |
| `mergeAPI` | merge/status | 见 §2.10 |
| `aiConfigAPI` | list/create/update/del/test/quickPreset/quickLocal/configsLocal/gpuStatus/gpuReleaseAll | 见 §2.12 |
| `agentConfigAPI` | list/get/create/update/del | 见 §2.13 |
| `skillsAPI` | list/get/create/update/del | 见 §2.14 |
| `voicesAPI` | list/sync | `GET /ai-voices`, `POST /ai-voices/sync` |
| `aiProvidersAPI` | list | `GET /ai-providers` |
| `uploadAPI` | image | `POST /upload/image` |
| 资源库 4 个 API | list/get/create/update/del/batchDelete/categories/tags(/filterOptions/apply/fromX) | 见 §2.17 |
| `useAgent()` | `run(type, message, ...)` | `POST /agent/:type/chat` |

## 4. 本次审计发现的问题

### P2（已修复）
- **videos.ts `POST /videos/:id/regenerate` 缺 `referenceImageUrls` 回退**：`referenceImageUrls: body.reference_image_urls` 缺 `|| row.referenceImageUrls`，与同函数 `imageUrl`/`firstFrameUrl`/`lastFrameUrl`/`duration`/`aspectRatio` 等字段不一致。当某镜头视频原以 `reference_mode='multiple'`（多角色参考图）生成后，在 VideoEditor 中「重新生成」会丢失角色参考图。已修复为 `|| row.referenceImageUrls`。

### P3（孤儿端点，前端未接，记录备查）
- `GET /dramas/stats` — 剧目统计（前端未接，历史已知）
- `PUT /dramas/:id/characters` / `PUT /dramas/:id/episodes` — 批量保存（前端改用单条 update）
- `GET /agent/:type/debug` — 开发调试端点
- preset-framework 6 端点 — 前端页面已删除，后端骨架保留

## 5. 孤儿端点清单（后端有、前端从不调用）

| 端点 | 状态 |
|---|---|
| `GET /dramas/stats` | 疑似待接入 |
| `PUT /dramas/:id/characters` | 疑似废弃（有单条替代） |
| `PUT /dramas/:id/episodes` | 疑似废弃（有单条替代） |
| `GET /agent/:type/debug` | 开发调试 |
| preset-framework 6 端点 | 疑似废弃（前端页面已删） |

## 6. 前端独有调用清单（前端调用但后端无路由）

**无** —— 本次审计未发现前端调用不存在的端点。
