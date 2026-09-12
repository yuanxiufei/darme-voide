# backend-py —— Python 后端（绞杀者迁移）

> **这是什么**：Node 后端（`backend/`，Hono + Mastra）的 Python 替代品，**按域逐个迁移**，
> 迁移期间两个后端并存、共用同一个 SQLite 文件与前端契约。
>
> **现状**：`dramas`（9 端点）+ `episodes`（8 端点，含剧本指纹门禁整服务）已迁移并验证通过
> （**103/103 冒烟用例**）。其余域由 `PROXY_TO_NODE=1` 反代到 Node，**系统始终可用**。

## 为什么不一次性重写

原始体量实测：**133 个 TS 文件 / 27,389 行 / 226 个端点 / 29 张表**，且**零自动化测试**。
一次性重写的问题不是「写不完」，而是**中途没有任何可验证的中间态** ——
4~6 周后才能第一次真正跑通，期间风险全部堆积在最后一刻。

绞杀者模式下每一天都有可跑、可回退的产物：

```
浏览器 ──► FastAPI :5790 ──┬─► 已迁移的域：Python 直接服务
                            └─► 未迁移的域：反代 ──► Node :5789
```

迁完一个域，就在 `app/main.py` 里 `include_router(...)` 一行，然后把该域从 Node 侧停用。
全部迁完后去掉反代，Python 独占端口即可（`PY_PORT=5789`）。

## 运行

```bash
cd backend-py

# 1) 建虚拟环境（首次）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux

# 2) 启动（默认 5790；与 Node 的 5789 并存）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 5790

# 3) 想让它把未迁移的域转发给 Node（需要 Node 后端已在 5789 运行）
set PROXY_TO_NODE=1
.venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
```

自检（**唯一回归安全网，改完代码必跑**）：

```bash
.venv\Scripts\python.exe tests\smoke_test.py      # 退出码 0 = 全过
```

它做三件事：把表/列定义与真实 `data/drama.db` 的 `PRAGMA table_info` 逐列比对（并与 Node 的
`db/index.ts` 建表清单交叉印证）、打真实接口核对响应信封与**错误文案逐字**、写操作全部落在
数据库副本上（真实库只读）。报告落在系统临时目录，路径在结尾打印。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PY_PORT` | `5790` | 本后端端口。**刻意不读 `config.yaml` 的 `server.port`** —— 那是 Node 的端口，并存期读同一个必然抢占 |
| `PORT` | — | 兼容用，优先级低于 `PY_PORT` |
| `HOST` | `0.0.0.0` | |
| `CORS_ORIGINS` | 见 `app/config.py` | 逗号分隔 |
| `DATA_ROOT` | `configs/config.yaml` 的 `database.path` 所在目录 | 数据根目录（DB + static + traces） |
| `DB_PATH` / `STORAGE_PATH` | — | 仅在未显式指定 `DATA_ROOT` / `.data-root` 时生效（与 Node 同规则） |
| `CONFIG_PATH` | `configs/config.yaml` | |
| `PROXY_TO_NODE` | `0` | `1` = 未迁移的域反代到 Node；`0` = 返回 501 并说明未迁移 |
| `NODE_BACKEND_URL` | `http://127.0.0.1:5789` | 反代目标 |

## 目录结构

```
backend-py/
├─ app/
│  ├─ config.py                  配置解析（与 backend/src/config.ts 逐项对齐）
│  ├─ db.py                      SQLite 连接（WAL + busy_timeout，对齐 connection.ts）
│  ├─ models.py                  29 张表的 SQLAlchemy Core 定义（对齐 db/schema.ts）
│  ├─ response.py                统一响应层 + 行→dict / 字段映射工具
│  ├─ request_utils.py           读请求体（对齐 Hono c.req.json() 的容错）
│  ├─ main.py                    FastAPI 入口：信封兜底 / 静态 Range / SPA / 反代接缝
│  ├─ routers/
│  │  ├─ dramas.py               ✅ 9 端点
│  │  └─ episodes.py             ✅ 8 端点（含流水线状态、指纹门禁状态）
│  └─ services/
│     ├─ era_background.py        时代背景解析（纯函数；AI 提炼未迁）
│     ├─ bible_ids.py             六键 Bible：STYLE_ / COST_ / LOC_ 三键
│     └─ script_fingerprint.py    剧本指纹门禁（整服务，纯逻辑）
└─ tests/smoke_test.py           冒烟测试（模式 + 契约，103 用例）
```

## 迁移时必须守住的三条对齐约定

1. **响应信封与错误文案逐字对齐**。前端 `useApi.ts` 是原生 `fetch`，判据为
   `!resp.ok || (json.code && json.code >= 400)`，错误文案直接取 `json.message` 展示给用户。
   而 FastAPI 默认返回 `{"detail": ...}`，会让文案变成「请求失败 (422)」——
   故 `main.py` 把 `HTTPException` / `RequestValidationError` / 兜底异常**全部收口**成信封。
   * 成功：HTTP 200 + `{code:200, data, message:"success"}`
   * 创建：HTTP 201 + `{code:201, data, message:"created"}`
   * 失败：HTTP 4xx/5xx + `{code, message}` —— **没有 `data` 键**
   * 特例：`GET /api/v1/health` 是裸对象（Node 侧就没包信封）
2. **HTTP 层 snake_case、聚合视图 camelCase**。`GET /dramas/:id/prompts` 的聚合结果在原
   Node 版里就是 camelCase（`customPrompt` / `imageUrl`），照抄不要「顺手统一」。
3. **SQLAlchemy 的 `**kwargs` 不等于 JS 的对象展开**。JS 是「后者覆盖前者」，
   Python 遇到同名键直接 `TypeError: got multiple values for keyword argument`。
   凡是「先展开白名单字段、再注入主键/时间戳」的写法，必须写成
   `values = {**fields, "drama_id": ...}` 再覆盖。**此坑已在 `PUT /dramas/:id/episodes` 上真实踩到。**

## 已知差异与遗留物（均有实测证据）

| 项 | 说明 |
|---|---|
| 真实库比 Node 模型多 2 张表 | `assets`、`props` —— 旧版本残留，Node 侧 `db/index.ts` 不建、代码零引用，Python 同样不建模 |
| `image_generations` / `video_generations` 各有 `minio_url` 列 | 同上，旧 MinIO 存储遗留，Node 的 Drizzle 模型里也没有，全仓库零引用 |
| 列序与 DB 不同 | 若干表的列序与 DB `PRAGMA` 顺序不一致。**不影响正确性**（SQLAlchemy 全程按列名生成 SQL，不会位置化 INSERT/SELECT）；仅让 JSON 的键顺序不同，而 JSON 对象键序无语义 |
| 剧本指纹的**过期判定**已迁移，但**写入侧**未全 | `checkEpisodeFingerprint` / `refreshEpisodeScriptHash` / `checkStoryboardGate` / `stampStoryboardsScriptHash` 都已具备；`stampStoryboardsScriptHash` 要等 `storyboards` 域迁移时接上 |
| 节奏相位 / 时代背景 AI 提炼 / 续写剧本 / 一致性 QC | 尚未移植（依赖 LLM 或视觉模型），对应端点不注册 → 走反代（或 501） |
| `GET /dramas/:id/prompts` 里 `episodes[].episodeNumber` 等 | 保持 camelCase，见上文第 2 条 |

## 下一个域的迁移 SOP

1. 读对应的 `backend/src/routes/<domain>.ts`（以及它依赖的 `services/*.ts`），
   逐个端点抄下来：方法、路径、查询参数、错误文案、响应形状。
2. 在 `app/routers/<domain>.py` 建 `APIRouter(prefix="/api/v1/<domain>")`。
   **路径顺序有语义**：静态子路径（如 `/stats`）必须声明在 `/{id}` 之前，否则会被吃掉。
3. 只注册已实现的路由；**没实现的不要注册** —— 留给反代接缝，这样比返回 501 更可用。
4. 在 `app/main.py` 的 `include_router` 处加一行。
5. 把该域的用例加进 `tests/smoke_test.py`，跑通（含错误路径的文案断言）。
6. 域内依赖、且输入输出明确的纯函数，先移植成 `app/services/<x>.py`；
   **依赖 LLM / 子进程 / 长任务的服务放到最后**，它们的行为需要真实调用才能验证。

## 工期参考

完整等价替换的实测估算：**单人对 AI 当助手 ≈ 99–152 人日**；AI 当主力（人只审阅 + 验证）
≈ 50–80 人日。**其中不可压缩的部分是验证，不是编码** ——
226 个端点冒烟、5 个 Agent 行为回归、17 家厂商适配真机跑通、前端零改动走查。
最短「能替换」单人也需 3~5 周（降级验收，只冒烟）。
