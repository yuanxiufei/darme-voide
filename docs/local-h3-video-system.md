# 短剧视频生成方法论（H3 双路线）

> 从 MiniMax H3 资料提炼的方法论，与「部署载体」解耦，核心解决「声音串人 / 角色跳戏」。

## 1. H3 两条核心路线（模型能力，与部署载体无关）

| 路线 | 能力 | 适用 |
|---|---|---|
| FL2VA | T2VA / I2VA / 首尾帧控制 | 动作、空镜、无对白镜头 |
| Ref2VA | 参考图 / 参考视频 / 参考音频驱动 | 人物对白、双人、会议、争吵 |

## 2. 音视频联合生成

- **H3 = 音视频联合生成**（不是先出视频再配音）→ 音画同步以 H3 原生音频为主，**不把 Wav2Lip 作为常规补嘴型手段**。
- **Z-Image = 美术部门**（角色三视图、场景参考、镜头首帧）；**H3 = 演员 + 摄影 + 音频**。Z-Image 不参与每帧生成。

## 3. 参考音频的正确语义（重要修正）

Ref2VA 的参考音频是 **reference conditioning**，不是最终对白音频：
- 官方规格：最多 **3 条参考音频**，每条 **2–15 秒**，总参考音频 ≤ **15 秒**。
- 角色声音**两层设计**：
  1. 第一层「角色声音资产」`VOICE_001` = 角色声线参考（固定，跨集不变）；
  2. 第二层「本镜头对白」→ 当前 Speaker → 对应声音资产 → H3。
- **禁止**把混合对白（A说/B说/A说…）整体塞给 H3 让它猜谁是谁——这是「一个人说所有人的话」的根源之一。

## 4. 六键 Bible（跨集锁定的 6 个 ID）

`CHAR_ID` / `SPEAKER_ID` / `VOICE_ID` / `LOCATION_ID` / `COSTUME_ID` / `STYLE_ID`

一个角色从第一集到第十集，这 6 个 ID 不变。

| 键 | 状态 | 说明 |
|---|---|---|
| CHAR_ID | ✅ `characters.id` | 角色 |
| SPEAKER_ID | ✅ `characters.speaker_id` | S1/S2 全局唯一 |
| VOICE_ID | ✅ `voiceStyle` | 声线，跨集锁定 |
| LOCATION_ID | ✅ `scenes.locationId` | 场景 Bible，同地点归一复用 `LOC_xxx`（`bible-ids.ts ensureLocationId`） |
| COSTUME_ID | ✅ `characters.costumeId` | 服装，一角色一 ID `COST_xxx`（`ensureCostumeId`） |
| STYLE_ID | ✅ `dramas.styleId` | 风格，一剧一 ID `STYLE_xxx`（`ensureStyleId`） |

绝对映射：`CHAR_001 → S1 → VOICE_001`，跨集不变。

## 5. 镜头路由（scene_type → 路线）

| scene_type | 场景 | 路线 |
|---|---|---|
| `action` / `silent` | 动作 / 空镜 / 环境 | **FL2VA**（首帧图生视频） |
| `single` | 单人说话 / 独白 | **Ref2VA**（角色参考 + 声音参考） |
| `dialogue_2p` | 双人正反打 | **Ref2VA**（A+B 参考 + 当前 speaker 声音） |
| `meeting` | 多人会议 | **Ref2VA**（多人参考 + 当前 speaker） |
| `argument` | 激烈争吵 | **Ref2VA**（一人一句快切） |
| `long_dialogue` | 长对白 | Qwen 拆 5–8s 切片 → **Ref2VA** 逐镜 |

## 6. 说话人铁律

- 默认 **ONE_SHOT_ONE_SPEAKER**：一个镜头一个说话人；双人对白拆正反打；长对白 5–8s 切片。
- 特殊 **OVERLAP_DIALOGUE**：两人同时说 / 打断 / 抢话才用双参考，**不作为默认模式**。

## 7. 本地部署（不引入 ComfyUI）

- 本地 H3 复用现有 `minimax` 视频 adapter，**只改 baseUrl 指向本地 H3 服务**（项目已有设计，代码注释已写死此接法）。
- 本地推理运行时（ComfyUI / diffusers / 官方脚本）是**部署细节**，与项目解耦，不作为项目要「接入」的东西。
- A5000 跑法：INT8 + offload（FL2VA INT8 ≈ 19.5GB）；第一版**不上 Turbo**，先跑通人物 / 声音 / 嘴型 / 连续性。
- **模型安装统一入口** `scripts/model_manager.py`（Python 标准库零依赖，清单外置 `configs/models.json`）：

| 命令 | 作用 |
|---|---|
| `python scripts/model_manager.py list [--category video\|text\|image\|tts] [--missing]` | 列清单与安装状态 |
| `python scripts/model_manager.py download --required \| --category video \| --all` | 按需批量安装 |
| `python scripts/model_manager.py install-nodes [--required]` | 装 ComfyUI 自定义节点 |
| `python scripts/model_manager.py doctor` | 体检（底座/git/ollama/磁盘/模型/节点） |
| `python scripts/model_manager.py add-model/remove-model` | 增删清单条目（无需改代码） |

  旧入口 `scripts/h3_install.py` 已降级为兼容 shim（H3 8 模型 + 4 节点），数据源同样指向 `models.json`，新模型请走 `model_manager.py`。

## 8. 测试计划（第一轮 A–G）

A 男+女 6s 男说一句；B 6s 女说一句；C 12s 男→女两镜头；D 24s 男→女→男→女；E 4人 24s S1→S2→S3→S4；F 双人争吵 30s；G 60s 连续对白。全过则整套系统基本通。

## 9. 与现有代码映射（落地状态）

- ✅ `speaker_id` / `scene_type` / ONE_SHOT_ONE_SPEAKER / 音色跨集锁定（schema + storyboard-tools + voice-tools + 2 个 SKILL）。
- ✅ `checkpoint_map` 落库（2026-08-28）：`ai_service_configs.settings` 支持写入 `checkpoint_map{fl2va,ref2va}` —— aiConfigs 路由 `buildSettings`/`parseSettingsObject` 合并读写（不再被 `negative_prompt` 覆盖丢失），settings 页视频类新增 FL2VA/Ref2VA 双字段 UI；`resolveH3Checkpoint` 已按 scene_type 正则路由 FL2VA/Ref2VA 并从 `config.settings?.checkpoint_map` 取值。
- ✅ 本地 H3 服务代码就绪：复用 `minimax` adapter + baseUrl（无需新写 ComfyUI adapter）；模型清单统一外置 `configs/models.json`，通用工具 `scripts/model_manager.py` 下载/安装/体检；后端 `services/local-model-scan.ts` + `routes/localModels.ts` 扫描本机模型并注册到 `ai_service_configs`（含 H3 `checkpoint_map` 派生）。**仅剩启动 ComfyUI(8188) + 本地 H3 服务(8765) 跑通推理**。
- ✅ 参考音频两层语义：`getStoryboardReferenceAudioUrls` 取出场角色 `voiceSampleUrl`（≤3 条、声线样本非最终对白）→ `auto-pipeline` 对话类镜头填入 → `minimax-video` 以 `reference_audio` 发送（reference conditioning）。
- ✅ `LOCATION_ID` / `COSTUME_ID` / `STYLE_ID` 三键：`services/bible-ids.ts` `ensureLocationId/ensureCostumeId/ensureStyleId`，routes 落库。
- ✅ QC 启发式：`services/qc-scoring.ts` `lip_sync*0.4 + consistency*0.3 + continuity*0.3`，视频完成自动触发。⬜ 真实唇形/相似度 AI 检测未挂载。
