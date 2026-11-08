# 知识库接口契约（前端 ↔ 后端对齐真源）

> 适用范围：`backend-py/app/routers/knowledge_loop.py` 暴露的**全部** `/api/v1/knowledge-loop/*`
> 端点，以及 `backend-py/app/services/knowledge_ledger.py` 的实现口径。
> 文档库（`/api/v1/knowledge/*`，从 `提取/产物/` 抽正文）是另一条链路，见 `docs/api-contract.md`。
>
> 本文件与代码**双向**受测：`tests/knowledge_management_test.py` ⑧ 组会做两件事 ——
> ① 文档里点名的端点必须真的挂在 openapi 上（不能凭空承诺）；② openapi 上挂的端点
> 必须在本文出现（新增端点不同步文档即红）。改契约不更新本文，测试会挡。

---

## 1. 通用约定

| 项 | 约定 | 真源 |
| --- | --- | --- |
| 前缀 | `/api/v1`（`app/main.py` 挂载） | 代码 |
| 成功信封 | `{"code": 200, "message": "ok", "data": <载荷>}` | `app/core/response.py::success` |
| 失败信封 | `{"code": <HTTP 状态>, "message": "<给人看的一句话>"}`，**无 data** | `bad_request/forbidden/not_found/conflict/unprocessable/server_error` |
| 字段命名 | 请求用 **camelCase**（`entryId`/`expectedVersion`/`includeDeleted`）；响应里**历史列名保持 snake_case**（`created_ns`/`source_kind`），派生字段用 camelCase（`tagList`/`deletedNs`/`deletedBy`/`bodyLen`/`isCurrent`） | `_row_to_entry` |
| 时间 | 一律 `*_ns` 纳秒整数（Linux epoch），前端自己格式化；**不回字符串时间**，避免两端时区/格式不一致 | `_now_ns` |
| 派生字段 | `tagList`、`version`、`deleted`、`deletedNs`、`deletedBy`、`snippet`、`bodyLen` 一律**服务端算**，前端不许自己 `split(',')` 或 `body.length` | `_row_to_entry` / `_snippet` |
| 分页 | `limit`/`offset`；`total` **恒为全量命中数**（不是本页条数） | `search` / `list_entries` / `audit_log` |
| 标签 | 落库是逗号串（不是 JSON），**写入口归一、读入口不归一** | `normalize_tags` / `tag_list` |
| 幂等 | 软删/恢复/同状态迁移都是幂等的：`changed=false`，不报错、不刷时间 | `soft_delete` / `restore` / `set_status` |
| 读失败 | 读操作**降级但不假装成功**：`available=false` + `hint` 说明为什么（不回空列表冒充"没有数据"） | `search` / `list_entries` / `versions` |

### 1.1 错误码分工（必须一致，前端按码决定动作，不解析文案）

| 码 | 含义 | 典型触发 | 前端该做什么 |
| --- | --- | --- | --- |
| 200 | 成功 | — | 读 `data` |
| 400 | **请求形状不成立** | 缺必填字段、未知 `kind`/`status`/`action`/`role`、`expectedVersion` 非整数、PATCH 一个字段都没给 | 改请求，别重试 |
| 403 | 角色不够（**声明式**，见 §4） | `role=viewer` 做写操作；`role=editor` 做硬删 | 提示无权限，别重试 |
| 404 | 目标不存在（**含已软删且未声明 includeDeleted**） | 编辑/删除/版本/回滚一条不存在或已删的条目 | 刷新列表 |
| 409 | **与当前状态冲突** | 版本冲突（`expectedVersion` 落后）、编辑已删条目、硬删一条未软删的条目 | 重新读取后再改（不是参数写错） |
| 422 | **内容站不住**（格式都对） | 证据核不过（run_id 不在索引、路径不在盘上）、空标题/空正文 | 补证据或改内容，别改请求形状 |
| 500 | 服务端问题 | 台账库不可写/不可读 | 报错留现场，别让用户改参数 |

> 契约铁律：**400 与 422 不许混用**。混了以后前端只能统一弹「请求有误」，
> 而这两种失败的正确动作完全不同（改请求 vs 补证据）。

---

## 2. 数据结构

```ts
// 一条知识（GET /entries、/search 的 items 元素；GET /entries/{id} 的 data）
interface KnowledgeEntry {
  id: string              // 稳定主键：entry_id(kind, scope, 创建时标题) 的哈希，改名不改 id
  kind: string            // fact/work/recipe/lesson/failure/threshold/prompt_patch（闭集）
  scope: string           // global / drama:<id> / stage:<name> / agent:<type>
  title: string
  body: string
  tags: string            // 落库原文（逗号串）
  tagList: string[]       // ★派生：拆分好的标签，前端直接渲染
  confidence: number      // 0..1，写入口会夹住
  status: string          // proposed/accepted/deprecated/rejected（闭集，迁移受 _TRANSITIONS 约束）
  evidence: string[]      // ★派生：逗号串 → 数组
  support: number         // 「又一份证据支持它」的次数（编辑**不动**它）
  contra: number
  lift: number | null
  source_kind: string     // human / engine ...
  source_ref: string
  created_ns: number
  updated_ns: number
  deleted_ns: number      // 0 = 未删
  deleted_by: string
  version: number         // ★派生：当前版本号（乐观锁），新建为 1
  deleted: boolean        // ★派生：deleted_ns > 0
  deletedNs: number
  deletedBy: string
  snippet?: string        // 检索结果专属：命中片段（服务端截，两端看到同一句）
}

interface KnowledgeTagFacet { tag: string; count: number }

interface KnowledgeTaxonomy {
  available: boolean
  hint: string
  kinds: { kind: string; label: string; count: number }[]   // label 中文名由服务端给
  scopes: { scope: string; count: number }[]
  statuses: { status: string; count: number }[]
  tags: KnowledgeTagFacet[]
  tagTotal: number
}

interface KnowledgeVersionRow {
  version: number
  title: string; bodyLen: number; tags: string   // ★不含正文
  editor: string; reason: string                 // 谁、在什么理由下替换掉了这一版
  createdNs: number
  status?: string
}

interface KnowledgeVersionDetail extends KnowledgeVersionRow {
  id: string; kind: string; scope: string
  body: string           // 全文
  evidence: string[]
  isCurrent: boolean     // true ⇒ 这版就是 current，正文取自 entries 而非快照表
}

interface KnowledgeAuditRow {
  id: number
  entryId: string        // camel：审计是服务端手拼载荷，不是 DB 原始行
  action: "create" | "support" | "update" | "status"
        | "delete" | "restore" | "revert" | "purge"
  operator: string; role: string
  detail: string; atNs: number
}
```

> **命名规矩（照这个判，别猜）**：`entries` 的行来自 `SELECT *`，所以**保留原列名**
> `created_ns`/`updated_ns`/`source_kind`/`deleted_ns`；一切**手拼**的载荷
> （`tagList`/`version`/`deleted`/`deletedNs`/`snippet`/版本行/审计行/分类）一律 camel。
> 版本详情**两个来源同形状**：当前版与历史版都回上面这套键（早先当前版直接回
> `_row_to_entry`、历史版回 `entry_versions` 原始行 ⇒ 同一个端点两套键名，前端必有一条路静默取到 `undefined`）。

---

## 3. 端点总表（本模块全部 40 个，⑧ 组会双向核）

### 3.1 条目与检索

- `GET /api/v1/knowledge-loop/entries` —— 列表（`kind`/`scope`/`status`/`tags`/`includeDeleted`/`limit`/`offset`）
- `GET /api/v1/knowledge-loop/entries/{entry_id}` —— 详情（`includeDeleted`）
- `POST /api/v1/knowledge-loop/entries` —— **叠加或新建**（同 id 存在 ⇒ `support+1`）
- `PATCH /api/v1/knowledge-loop/entries/{entry_id}` —— **编辑**（`version+1`，`support` 不动）
- `DELETE /api/v1/knowledge-loop/entries/{entry_id}` —— 删除（默认软删；`?hard=true` 物理删）
- `POST /api/v1/knowledge-loop/entries/{entry_id}/status` —— 状态迁移（人审）
- `POST /api/v1/knowledge-loop/entries/{entry_id}/restore` —— 撤销软删
- `GET /api/v1/knowledge-loop/entries/{entry_id}/versions` —— 版本列表（无正文）
- `GET /api/v1/knowledge-loop/entries/{entry_id}/versions/{version}` —— 某版全文
- `POST /api/v1/knowledge-loop/entries/{entry_id}/revert` —— 回滚到某版（普通编辑）
- `GET /api/v1/knowledge-loop/search` —— 全文检索（`q`/`kind`/`scope`/`status`/`tags`/`includeDeleted`/`limit`/`offset`）
- `GET /api/v1/knowledge-loop/taxonomy` —— 分类导航（kind 闭集＋中文名＋计数）
- `GET /api/v1/knowledge-loop/tags/facets` —— 标签 facet
- `GET /api/v1/knowledge-loop/audit` —— 操作审计（`entryId`/`action`/`operator`/`limit`/`offset`）

### 3.2 闭环与运维（本次未改，仅登记以保契约完整）

- `GET /api/v1/knowledge-loop/report` —— 闭环总览
- `GET /api/v1/knowledge-loop/runs` —— 摄取记录
- `GET /api/v1/knowledge-loop/runs/stats` —— 摄取统计
- `POST /api/v1/knowledge-loop/runs/rebuild` —— 重建索引
- `POST /api/v1/knowledge-loop/distill` —— 蒸馏（引擎侧）
- `GET /api/v1/knowledge-loop/optimizations` —— 优化建议
- `POST /api/v1/knowledge-loop/lift/refresh` —— 重算 lift
- `GET /api/v1/knowledge-loop/qc/hit-rate` —— 命中率
- `GET /api/v1/knowledge-loop/qc/pending` —— 待审清单
- `GET /api/v1/knowledge-loop/qc/reviews` —— 审阅记录
- `POST /api/v1/knowledge-loop/qc/reviews` —— 提交审阅
- `DELETE /api/v1/knowledge-loop/qc/reviews/{review_id}` —— 撤销审阅
- `GET /api/v1/knowledge-loop/feed/preview` —— 投喂预览
- `POST /api/v1/knowledge-loop/feed/export` —— 导出投喂包
- `GET /api/v1/knowledge-loop/scheduler` —— 调度状态
- `GET /api/v1/knowledge-loop/training` —— 训练清单
- `POST /api/v1/knowledge-loop/training/scan` —— 扫描可训练产物
- `POST /api/v1/knowledge-loop/training/models/{model_id}/promote` —— 提升模型
- `POST /api/v1/knowledge-loop/training/models/{model_id}/demote` —— 降级模型
- `GET /api/v1/knowledge-loop/tuning` —— 调参建议
- `POST /api/v1/knowledge-loop/tuning` —— 提交调参
- `DELETE /api/v1/knowledge-loop/tuning` —— 撤销调参
- `GET /api/v1/knowledge-loop/tuning/suggest` —— 推荐参数
- `GET /api/v1/knowledge-loop/datasets` —— 数据集清单
- `GET /api/v1/knowledge-loop/datasets/resolve` —— 解析数据集版本

---

## 4. 鉴权与权限（**先看这一节的「未做」清单**）

**本仓目前没有认证**：没有登录态、没有 `Authorization`/Bearer 校验、没有中间件。
`role` 是**调用方自己在请求里声明的**，谁都能写 `owner`。因此：

- 这一层解决的是「审计行里的角色与被执行的动作不能自相矛盾」；
- 它**不**解决「有人冒充」。要真鉴权得先有登录态，属于未做项。

角色闭集：`viewer < editor < owner`（`knowledge_ledger.ROLES`）。

| 动作 | 最低角色 |
| --- | --- |
| 只读（列表/详情/检索/分类/facet/审计/版本） | `viewer` |
| 创建、叠加、编辑、状态迁移、软删、恢复、回滚 | `editor` |
| 硬删（`?hard=true`） | `owner` |

判定规则（`_role_gate`，前端可照抄）：

1. `role` **没传** ⇒ 视为「未声明」，**不判**（向后兼容旧前端与 Agent 工具）；
2. `role` 不在闭集里 ⇒ 400（不猜它是不是想当 owner）；
3. `role` 不够格 ⇒ 403；
4. 够格 ⇒ 放行，并把 `role` 原样写进审计行。

所有写操作都接受可选的 `operator`（谁做的，进审计与 `deleted_by`），空值落成 `system`。

---

## 5. 逐端点输入输出与边界条件

### 5.1 `POST /entries` —— 创建 / 叠加（**不是编辑**）

请求：

```json
{ "kind": "lesson", "scope": "EP7", "title": "...", "body": "...",
  "evidence": ["trace:kb/a"], "tags": "节奏，过曝", "confidence": 0.4,
  "status": "proposed", "sourceKind": "human", "sourceRef": "", "operator": "alice" }
```

响应 `data`：`{ ok, id, created, support, why }`。

- 同 `(kind, scope, title)` 已存在 ⇒ `created=false`、`support+1`、正文被这次的新正文覆盖、
  **状态不打回 proposed**（已采纳的条目被重复提炼不该退回待审）。
- 语义分工：这条是「又有一份证据」；要**改错**用 PATCH。
- 边界：`kind`/`status` 闭集外 ⇒ 400；空标题/空正文 ⇒ 400；
  `lesson/failure/threshold` 等（`_EXTERNAL_KINDS` 之外）空证据 ⇒ 422；证据核不过 ⇒ 422；
  **命中已软删条目 ⇒ 409 + `why` 说明走 restore**（绝不静默复活，否则"我删掉的错误经验"
  会被下一次蒸馏带回来）。

### 5.2 `PATCH /entries/{entry_id}` —— 编辑

请求（**部分字段**，未出现或 `null` 的字段不动）：

```json
{ "title": "...", "body": "...", "tags": ["节奏","过曝"], "evidence": ["trace:kb/a"],
  "confidence": 0.6, "expectedVersion": 3, "reason": "补充数值",
  "operator": "alice", "role": "editor" }
```

响应 `data`：`{ ok, id, version, entry, changed: ["body"], why }`。

- `version` 自增 1；`support` **不动**（改错不是又一份证据）。
- `expectedVersion` 给了就必须等于当前版本，否则 409 + `data` 带 `version`（当前值），
  且**正文未被覆盖**（不是"先写后报错"）。
- `tags` 传 `""` 或 `[]` ⇒ **清空标签**；不传 ⇒ 不动。
- 边界：空标题/空正文 ⇒ 400（要清空请删除）；一个字段都没给 ⇒ 400；
  证据核不过 ⇒ 422；已软删 ⇒ 409；不存在 ⇒ 404。
- **有意不提供 PUT**：PUT 是整条替换，一个忘了带 `body` 的调用会把几百字经验清空。

### 5.3 `DELETE /entries/{entry_id}`

| 形态 | 语义 | 前置 | 响应 `data` |
| --- | --- | --- | --- |
| 默认（`soft`） | 软删：`deleted_ns/deleted_by` 落值，默认全路径不可见 | — | `{ ok, id, changed, hard:false, why }` |
| `?hard=true` | 物理删除 | **必须已软删**，角色 `owner` | `{ ok, id, purged:true, counts:{entries,entry_links,entry_usage,entry_versions,entry_audit} }` |

- 软删幂等：重复删 `changed=false`。
- 硬删后 `entry_audit` **保留**（账不是内容：「谁在什么时候删了什么」不能跟着内容一起蒸发），
  `counts.entry_audit` 是保留下来的条数，删除前各表条数写进该 action 的 `detail`。
- 边界：未软删就硬删 ⇒ 409 `not_deleted`（第二道闸，防打错参数）；不存在 ⇒ 404。

### 5.4 `POST /entries/{entry_id}/restore`

请求 `{ reason?, operator?, role? }` ⇒ `data`: `{ ok, id, changed, why }`。
幂等：本来没删 ⇒ `changed=false`。恢复会写一行 `restore` 审计。

### 5.5 版本与回滚

- `GET /entries/{id}/versions` ⇒ `{ available, current, items: KnowledgeVersionRow[], hint }`
  - `current` = 当前版本号；`items` 只有 `1..current-1`（**当前值只在 `GET /entries/{id}` 里**，
    不做两份真相）。
  - `items[].createdNs` 是**该版被定格（快照）的时刻**，不是那版正文的创建时刻。
- `GET /entries/{id}/versions/{v}` ⇒ `KnowledgeVersionDetail`；不存在的 `v` ⇒ 404。
- `POST /entries/{id}/revert` `{ version, reason?, operator?, role? }`
  - 回滚**是一次普通编辑**：`version+1`、把旧值写回、加一条快照、写 `revert` 审计；
  - **不删除任何后续版本**（所以还能再滚回来）；
  - 滚到当前版本 ⇒ `changed=false`；不存在的版本 ⇒ 404。

### 5.6 `GET /search`

`q` 空 ⇒ `ranking=recent`（按 `updated_ns` 倒序，与相关性无关）；
`len(q)>=3` 且 FTS 可用 ⇒ `ranking=bm25`；否则 / 无命中 ⇒ `ranking=like`。

- **三种 ranking 不可比**，回执 `hint` 如实写明用哪种。
- `total` 是全量命中数；候选上限 500，超出时 `totalCapped=true`。
- 每项带 `snippet`（服务端截命中片段）。
- 1–2 字中文查不了 bm25（trigram 需要 3 字），只能 LIKE —— 这是索引事实，不是 bug。

### 5.7 `GET /tags/facets`、`GET /taxonomy`

- facet 计数与列表**同一套词边界匹配**（`',' || tags || ','`）：点「节奏」不会冒出「节奏慢」，
  计数与点进去的条数必须相等。
- `taxonomy.kinds` 覆盖全部 `KINDS` 且带中文 `label`（前端不写死中文名）；
  另给 `scopes`/`statuses` 分布与标签总数 `tagTotal`。
- 两者都吃 `includeDeleted`；默认不计软删条目。

### 5.8 `GET /audit`

`entryId`/`action`/`operator` 过滤 ＋ `limit`/`offset`；`total` 为全量条数。
`action` 不在闭集 ⇒ 400（不静默回全量）。写动作闭集：
`create/support/update/status/delete/restore/revert/purge`。

---

## 6. 边界条件一览（判据都在测试里）

| # | 边界 | 期望 | 用例 |
| --- | --- | --- | --- |
| 1 | 标签带全角逗号/顿号/分号 | 写入口归一为半角逗号串 | ① |
| 2 | 单标签 >24 字 / 一条 >12 个 | 截断，不报错 | ① |
| 3 | 多标签过滤 | **AND**（同时具备） | ⑥ |
| 4 | 标签 `节奏` vs `节奏慢` | 词边界，互不命中 | ⑥ |
| 5 | 编辑 | `version+1`、`support` 不动、`changed` 只列真改的 | ② |
| 6 | `expectedVersion` 落后 | 409 + 当前版本，正文不被覆盖 | ② |
| 7 | 编辑后检索 | FTS 同步：旧词搜不到、新词搜得到 | ② |
| 8 | 版本列表 | 不含正文；`items` 只到 `current-1` | ③ |
| 9 | 回滚 | 是普通编辑；后续版本仍在 | ③ |
| 10 | 软删 | 列表/检索/详情全不可见；详情口不是后门 | ④ |
| 11 | 回收站 | 必须显式 `includeDeleted=true` | ④ |
| 12 | 已删条目 | 编辑/迁移状态 ⇒ 409 | ④ |
| 13 | 蒸馏撞已删条目 | 拒收（409），不复活 | ④ |
| 14 | 硬删未软删 | 409，且什么都没删 | ⑤ |
| 15 | 硬删后 | 内容全清、审计保留更多 | ⑤ |
| 16 | `search.total` | 全量命中数，不是本页条数 | ⑦/⑧ |
| 17 | 审计 | 每个写动作一行、含 operator/role/detail | ⑦ |
| 18 | 未知角色 / 越权 | 400 / 403 | ⑧ |
| 19 | 改名后同主题再蒸馏 | **已知边界**：id 含创建时标题 ⇒ 会新建一条，需人工合并 | — |

---

## 7. 验收标准（可判读、可复跑）

1. `backend-py/.venv/Scripts/python.exe -X utf8 backend-py/tests/knowledge_management_test.py`
   收尾行必须是 `SUMMARY: n/n passed`，退出码 0。
2. 同一份契约在 openapi 与本文之间**双向对上**（⑧ 组两条断言）。
3. 前端 `npm run typecheck`（Web 与 `desktop/frontend` 各一次）零错；
   `npm run check:parity` 里 `useApi.ts`/`contracts.ts` 属于受白名单约束的同构文件，必须逐字一致。
4. 手工验收（浏览器）：
   - 编辑一条 → 刷新后正文是新值，「版本」里能看到旧值；`support` 没变；
   - 两个人同时编辑（一个拿旧 `expectedVersion`）→ 后者收到「版本冲突」而不是静默覆盖；
   - 删除 → 列表消失、回收站可见、恢复后回来；硬删后回收站也没了但审计还在；
   - 标签点一个 → 列表条数与标签上的数字相等；
   - 检索 1–2 字 → 明确显示「LIKE 包含匹配」，不假装是相关性排序。

## 8. 测试要点（`tests/knowledge_management_test.py` ①–⑧）

- ① 派生字段与标签归一/截断、读入口不归一；
- ② 编辑语义（support 不动、乐观锁、FTS 同步、五类拒改的机器码）；
- ③ 版本（列表不含正文、快照 editor/reason、回滚是普通编辑、幂等）；
- ④ 软删/恢复（三路径全不可见、幂等、已删不可改、蒸馏不复活）；
- ⑤ 硬删（未软删拒绝、逐表计数、审计保留）；
- ⑥ 标签/分类（词边界、AND、facet 与列表同口径、软删口径、taxonomy 覆盖全部 kind）；
- ⑦ 审计（动作闭集、operator/role、total 全量、过滤生效）；
- ⑧ HTTP（400/403/404/409/422 各就各位、信封一致、文档↔openapi 双向）。

## 9. 前端落地（`useApi.ts` ↔ 端点）

| 方法 | 端点 |
| --- | --- |
| `listEntries({kind,scope,status,tags,includeDeleted,limit,offset})` | `GET /entries` |
| `getEntry(id,{includeDeleted})` | `GET /entries/{id}` |
| `createEntry(payload)` | `POST /entries` |
| `updateEntry(id,payload)` | `PATCH /entries/{id}` |
| `deleteEntry(id,{hard,reason,operator,role})` | `DELETE /entries/{id}` |
| `restoreEntry(id,payload)` | `POST /entries/{id}/restore` |
| `listEntryVersions(id)` | `GET /entries/{id}/versions` |
| `getEntryVersion(id,v)` | `GET /entries/{id}/versions/{v}` |
| `revertEntry(id,payload)` | `POST /entries/{id}/revert` |
| `searchEntries(params)` | `GET /search` |
| `getTagFacets(params)` | `GET /tags/facets` |
| `getTaxonomy(params)` | `GET /taxonomy` |
| `getAuditLog(params)` | `GET /audit` |

## 10. 未做 / 已知边界

- **没有认证与登录态**（见 §4）：`role` 可伪造，403 只防手滑不防冒充。
- **`role` 不传即放行**：为兼容旧前端与 Agent 工具保留；收紧它需要先让所有调用方声明角色。
- **改名不改 id**：id = `entry_id(kind, scope, 创建时标题)`。改名后同一主题再蒸馏会新建一条，
  需人工合并（自动合并会误伤"刻意分开的两条"）。
- **跨条目事务**：一次请求只动一条知识；批量编辑需前端循环（每次都能独立审计，是刻意的）。
- **版本表只涨不缩**：`entry_versions` 无自动清理（硬删时才跟着删）。条目改几十次后版本表会变长，
  暂不做裁剪 —— 裁剪等于丢掉审计依据。
