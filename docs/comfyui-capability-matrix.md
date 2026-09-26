# ComfyUI 能力矩阵 —— 「功能全都要，形式由我们定」

> 2026-09-17 定稿。判据来源：`ComfyUI/server.py`（**后端全部 26 条公开路由**，
> 逐条 grep 出来的，不是照文档猜 ✓）+ `ComfyUI/api_server/routes/internal/`（桌面壳内部 5 条 ✓）。
> ⚠️ 出处只写「上游项目 + 文件 + 符号」✓ —— 开发期副本放在脚手架目录 `reference/` 下，**项目完善后整目录删除** ✗
> （用户 2026-09-26 口径 ✓）⇒ 正文**不许**再把那条路径当落点 ✗（删完就是悬空引用 ✗）。

## 一句话结论

**ComfyUI 能做的事，我们这边都能做；调用方不必直连 8188** ✓ ——
由 `app/local_services/h3/server.py`（端口 **8765**，我们的形式）承载，
底层复用 `comfyui_client.py`（HTTP 客户端）+ `workflow.py`（UI 图 → API 格式 + 参数注入）✓。
引擎本体**仍然不 vendor**（它是机器相关外部服务 ✓），但「用法」完全在我们手里 ✓。

## 逐条对照（26 + 5）

| # | ComfyUI 原生 | 能力 | 我们的形式（8765） | 状态 |
|---|---|---|---|---|
1 | `GET /` | 前端网页 | —— 我们有自己的前端 | **不做**（见下「刻意不做」） |
2 | `GET /embeddings` | embeddings 清单 | `client.embeddings()` | ✅ 客户端已覆盖（待接 UI） |
3 | `GET /models` | 模型**类别**清单 | `GET /v1/catalog/models` | ✅ 自检 ③ |
4 | `GET /models/{folder}` | 某类**文件**清单 | `GET /v1/catalog/models?folder=…` | ✅ 自检 ③' |
5 | `GET /extensions` | 前端 js 扩展 | —— 界面层概念，与后端能力无关 | **不做** |
6 | `POST /upload/image` | 上传图/首帧 | `POST /v1/upload/image` | ✅ 自检 ⑧ |
7 | `POST /upload/mask` | 上传遮罩 | `client.upload_mask()` | ✅ 客户端已覆盖（同一套 multipart ✓） |
8 | `GET /view` | 取产物 | `GET /v1/view` | ✅ 自检 ⑨ |
9 | `GET /view_metadata/{f}` | 输出资产元数据 | `client.view_metadata()` | ✅ 客户端已覆盖 |
10 | `GET /system_stats` | 系统/设备信息 | `GET /v1/system` | ✅ 自检 ① |
11 | `GET /features` | 能力开关 | `GET /v1/system`（`comfyui.features`） | ✅ 自检 ① |
12 | `GET /prompt` | 队列剩余量 | `GET /v1/queue`（`remaining`） | ✅ 自检 ④ |
13 | `GET /object_info[/{cls}]` | 节点能力表 | `GET /v1/catalog/nodes[/{cls}]` | ✅ 自检 ②/②'/②'' |
14 | `GET /api/jobs` | 作业列表（新版） | `GET /v1/jobs` | ✅ 自检 ⑥（老版本 ⇒ `supported:false` ✓ 不假装 ✓） |
15 | `POST /api/jobs/{id}/cancel` | 取消作业 | `POST /v1/jobs/{id}/cancel` | ✅ 自检 ⑥' |
16 | `POST /api/jobs/cancel` | 取消全部 | `client.cancel_all_jobs()` | ✅ 客户端已覆盖 |
17 | `GET /history` | 历史 | `GET /v1/history` | ✅ 自检 ⑤ |
18 | `GET /history/{id}` | 单任务历史 | `client.history()` / `wait()` | ✅（H3 流里就在用 ✓） |
19 | `GET /queue` | 队列快照 | `GET /v1/queue` | ✅ 自检 ④ |
20 | `POST /prompt` | **提交工作流** | `POST /v1/workflows/run` | ✅ 自检 ⑩/⑪（**任意**工作流：UI 或 API 格式 ✓） |
21 | `POST /queue` | 清空 / 删条目 | `POST /v1/queue/clear` / `DELETE /v1/queue/{id}` | ✅ 自检 ④' |
22 | `POST /interrupt` | 打断 | `POST /v1/interrupt` | ✅ 自检 ⑦ |
23 | `POST /free` | 卸载模型/释放显存 | `POST /v1/free` | ✅ 自检 ⑦（H3/通用流跑完必调 ✓） |
24 | `POST /history` | 清空历史 | `DELETE /v1/history` | ✅ 自检 ⑤ |
25 | `GET /`（根）| —— | 同上 | **不做** |
26 | `POST /history`（清） | —— | 同上 | ✅ |
— | `GET /internal/folder_paths` | 各类模型的**磁盘路径** | `GET /v1/catalog/folders` | ✅ 自检 ③'' |
— | `GET /internal/files/{type}` | 输入/输出目录清单 | `GET /v1/catalog/files/{type}` | ✅ 自检 ③''' |
— | `GET /internal/logs` `/logs/raw` `/logs/subscribe` | 桌面壳日志订阅 | —— 要日志直接看 ComfyUI 进程输出 | **不做** |
— | **（原生没有）** | **提交前校验** | `POST /v1/workflows/validate` | ✅ **我们补的**（自检 ⑫⑬⑭）：转换能不能过 / 缺哪个节点包 / 连线悬空 —— ComfyUI 自己只在 `POST /prompt` 那一刻才校验 ✗ |

## 刻意不做的三条（是**设计决定**，不是「还没做」✗）

1. `GET /` —— 那是 **ComfyUI 的前端网页**：我们用自己的前端 ✓，把别人的 UI 搬进来毫无意义 ✗。
2. `GET /extensions` —— 它列的是**前端 js 扩展**（`web/extensions/**/*.js` ✓），是界面层概念 ✗；
   真正与「能力」相关的**自定义节点**由 `GET /v1/catalog/nodes` 覆盖 ✓。
3. `/internal/logs*` —— 桌面壳专用的日志订阅（websocket/patch ✓）；我们自己就是服务，日志在进程输出里 ✓。

## 怎么用（三个例子）

```bash
# ① 看看这台机器上 ComfyUI 认识哪些模型、装在哪
curl -s localhost:8765/v1/catalog/models                 # 类别
curl -s "localhost:8765/v1/catalog/models?folder=vae"    # 该类文件
curl -s localhost:8765/v1/catalog/folders                # 磁盘路径

# ② 提交前先校验（省一次白跑）
curl -s -X POST localhost:8765/v1/workflows/validate \
  -H 'Content-Type: application/json' \
  -d '{"workflow": <UI 格式 workflow JSON>}'

# ③ 跑任意工作流（UI 格式会被自动转成 API 格式；参数用显式清单注入）
curl -s -X POST localhost:8765/v1/workflows/run \
  -H 'Content-Type: application/json' \
  -d '{"workflow": <UI JSON>, "params": [{"class_type":"CLIPTextEncode","field":"text","value":"提示词"}]}'
# → {"task_id": "..."}；轮询 GET /v1/workflows/task/{task_id}；产物在 outputs[].url（走 /files ✓）
```

> H3 视频只是**这套通用能力的一个消费者** ✓（`POST /v1/video_generation` 保留原协议，因为
> 后端 `minimax` 适配器按它对接 ✓）—— 换句话说：**ComfyUI 的通用能力是底座，H3 是上面的一条业务线** ✓。

## 自检

`tests/comfyui_capability_test.py`（**25 项** ✓，起真 HTTP stub ComfyUI ✓ 不需要 GPU）逐条走完上表，
并含**反套套逻辑**（stub 侧每个端点都必须真被请求过 ✓ 门面不能自说自话 ✓）。
