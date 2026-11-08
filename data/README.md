# `data/` —— 数据根（Data Root）

这个目录就是运行时的**数据根**：作品数据、生成出来的图/视频/音频、各库的 SQLite 全在这儿。
它可以被整个搬走、整个备份、整个恢复——**前提是别把东西摊在根上**。

切换数据根：设置页「数据存储」会往目标目录写一个 `.data-root` 标记文件，优先级为
`.data-root` 标记 > `DATA_ROOT` 环境变量 > `configs/config.yaml` 的 `database.path` 所在目录 > `./data`。
实现见 `backend-py/app/core/config.py`。

---

## 一条铁律：顶层只放目录，不放散文件

```
data/
├── db/               ← 所有 SQLite（2026-11-07 起）
├── static/           ← 图片 / 视频 / 音频等生成物
├── traces/           ← Agent 执行轨迹与 token 统计
├── knowledge/        ← 知识库派生索引（可重建）
├── models/           ← 本地模型池（默认落点；本机已被指向别处，见下）
├── hf-cache/         ← HuggingFace 下载缓存（`HF_HOME` 指这儿，可重下）
├── lora-train/       ← 训练参数 toml / 预览 prompt / 导出数据集 / 训练日志
├── repair/           ← 修复提案与改动前备份（proposals/）
├── engine-dryrun/    ← 引擎干跑凭据（可重建）
├── materialized/     ← LLM 反量化权重缓存（可重建）
└── experiments/      ← 实验/探针产物（不上线、不发 URL、切根不搬）
```

以前 `drama.db`、`library-overlay.sqlite` 和它们的 `-wal`/`-shm`/`.bak-*` 直接躺在根上，
六个子目录和四五个散文件混在一起：

- 备份时不知道该拷哪几个（只拷 `static/` 就是「作品数据全丢」）；
- 忽略规则要逐文件名列举，**漏一个就是 `git add -A` 整片入库**（`materialized/` 就漏过，7.7 GB）；
- 换台机器一看目录，分不出「哪些是一套」。

现在 **SQLite 一律进 `db/`**，除此之外根上只有目录。

（唯一的例外是本文件：它是说明、不是「数据」，所以留在根上。也别往后加第二个 ——
要放说明就写进本文件，要放数据就建目录。根上原先还有个 `.gitkeep` 占位，2026-11-07 删了：
本文件已入库，这层目录不会再是空的，占位就纯属冗余。）

---

## 逐层说明

| 落点 | 是什么 | 谁写 | 删了会怎样 | 备份 |
|---|---|---|---|---|
| `db/drama.db` | **主库**：剧集、角色、场景、镜头、分镜、提示词、任务… | 后端全站（SQLAlchemy） | **作品数据没了**，重建不了 | **必须** |
| `db/drama.db.bak-<时间戳>` | 库升级前的快照 | `backend-py/app/scripts/db_upgrade.py` | 少一个回退点 | 建议留 |
| `db/library-overlay.sqlite` | 知识库**编辑 / 删除 / 上传台账** | `services/library_overlay.py` | 上传记录与人工改动没了，重建不了 | **必须** |
| `static/` | 生成物：图片、视频、音频、导出文件 | 生成链路、上传接口 | 作品素材没了 | **必须** |
| `traces/` | Agent 执行轨迹、token 统计 | `trace_store`（写入侧仍在 Node） | 只是观测数据没了 | 可选 |
| `knowledge/` | 知识库派生索引（`*_index.sqlite` 等） | `outcome_index` / `knowledge_ledger` | 下次检索时自动重建 | 不必 |
| `engine-dryrun/` | 引擎干跑凭据（JSON） | `engine.bridge.outputs_dir_for(dry_run=True)` | 下次干跑重新生成 | 不必 |
| `materialized/` | LLM 反量化权重缓存（实测数 GB） | `engine.materialize` | 下次跑图自动重算 | 不必 |
| `models/` | **本地模型池**：下载 / 扫描 / 就绪 / 加载的权重（`local_model_scan.get_models_dir()`；`MODELS_DIR` 或 `configs/model-paths.json` 的 `models_dir` 可指到别处 —— 本机就指到 `D:/Comfy-Desktop/ComfyUI-Shared/models`，所以 `data/` 里看不到它） | `app/services/local_model_scan.py` | 重下即可（数百 GB 量级） | **不必**（且**必须排除**，见下） |
| `hf-cache/` | HuggingFace 下载缓存（模型 / 词表）：`engine.tokenizers_tuning.default_cache_dir()` 把它设成 `HF_HOME` | `transformers`（经 `tokenizers_tuning.apply_tuning`） | 下次用到时自动重下 | **不必**（可重下；切根也不复制它） |
| `lora-train/` | 训练参数 `settings/<模型>_settings.toml`、预览 `previews/<模型>/prompt.txt`、导出集 `datasets/<名>/`、任务 `tasks/<taskId>/`（日志与 `*.safetensors`） | `services/lora_train/*`、`knowledge_feed` | 手调的参数与训出来的权重没了，要重训才知道原来那套好不好 | **建议**（配置与产物都算「丢了要重来」；切根时**会**随数据根一起复制 ✓ 见下） |
| `repair/` | 修复提案与改动前备份（`proposals/`） | `routers/repair.py::_repair_data_dir` | 提案没了，得重新诊断一遍 | 可选 |
| `experiments/` | 实验/探针产物（`backend-py/_bench/*.py` 真跑出来的图/片/歌），每个实验一个子目录 `<实验名>/` | 一次性探针脚本 | 只是这次实验的产物没了，重跑即可 | **不必**（切根也不搬） |

`static/` 与 `experiments/` 的分界是**硬边界**，别混：`static/` 是**成品**（`/static/{rel:path}`
整个目录对外服务、切根会整目录搬走、进备份面），`experiments/` 是**实验**（不上线、不发 URL、
切根不搬）。探针图一旦落进 `static/`，它就成了一个能点开的链接、还会被
`routers/engine.py::_with_urls()` 当成品补 `primaryUrl` 回执给前端 —— 那是发布，不是保存。
判据在 `tests/output_layout_test.py`：⑦ 反着判「`backend-py/output/` 存在即红」。

判断自己手上那个文件要不要备份，只问一句：**删了能不能算/下回来？** 不能 ⇒ 备份。

⚠️ **整个 `data/` 一起备份时，把 `models/` 排除掉**：那是本地模型池，权重下载得到、体积数百 GB ——
按上面那句判据不该进备份面（除非事先没配 `MODELS_DIR`，模型真下到了 `<dataRoot>/models`）。
注意它与**仓库根**的 `models/`（仓内 OCR / 分词器素材，见 `models/README.md`）是两回事：
名字撞了，一个是运行时下载的模型池、一个是随仓库走的素材。

上面写了 11 个目录，而 `tests/output_layout_test.py` 的 `DATA_DIRS` 白名单是 9 个 —— **不是矛盾**：
白名单登记的是「**代码里拼出来的**落点」，`static/` 是 storage root 本身（`get_storage_root()`
的返回值）、`traces/` 主要由 Node 侧写（Python 只在 `data_storage.change_data_root()` 切根复制时
拼它一次，那是搬旧数据不是落点），两个都不在 Python 的落点拼路径里。

⚠️ 白名单**不等于全量**：判据 ② 只认 `get_data_root() / "字面量"` 这一种拼法。
`hf-cache/` 写成 `Path(root) / "hf-cache"`（`root` 是局部变量）就躲过了它 ——
漏登记 + 没忽略，躲了一个多月（2026-11-07 才补上）。所以 ② 报绿只说明**它看得见的那部分**干净。

`db/` 里为什么连台账也算「必须备份」：它和 `drama.db` 是同一类东西——**丢了就真没了**。
而 `knowledge/` 里的索引虽然也慢（重建可能要跑很久），但**算得回来**，所以不列进备份对象。
**能不能重建决定它进不进 `data/`；跟谁同一个子系统决定它进不进 `db/`。**

---

## `drama.db` 的若干约定

- 文件名固定 `drama.db`，位置固定 `<dataRoot>/db/`；配置里 `database.path` 只取父目录推数据根，
  末尾写了 `db/` 这一层会被认出并跳过（`./data/drama.db` 与 `./data/db/drama.db` 等价）。
- 结构变更（补列）用 `python -X utf8 app/scripts/db_upgrade.py`（先 `--plan` 看，再 `--apply`）。
  它会先留一份 `db/drama.db.bak-<时间戳>` 再改，且**只保留最近 3 份**。
- 库跑在 WAL 模式，所以复制库时不能只拷 `drama.db`（会丢 `-wal` 里未 checkpoint 的写）。
  手工备份请整个 `db/` 目录一起拷，或用 `sqlite3` 的 backup API。

---

## 不在这里的东西（容易搞混）

| 目录 | 为什么不在 `data/` |
|---|---|
| `indexes/` | 知识库的派生索引。**算得回来**（`--redo` 可重建），且体积大（约 2 GB）、每次跑图都会改写 ⇒ 不跟数据根走、也不该进备份 |
| `corpus/` | 语料与方法料（`corpus/playbook`）。**仓库里就有**，新克隆即可用 |
| `models/` | 仓内模型素材。**下载得到**，不该占备份体积（详见 `models/README.md`） |

四条判据（新落点该放哪，先问这个）：算得回来 ⇒ `indexes/`；下载得到 ⇒ `models/`；
仓库里有 ⇒ `corpus/`；**丢了就没了 ⇒ `data/`**。

---

## 2026-11-07 的那次迁移

`drama.db`、`drama.db-wal/-shm`、`drama.db.bak-*`、`library-overlay.sqlite(+wal/shm)` 从数据根顶层
移进了 `db/`。迁移由 `config.ensure_db_layout()` 自动完成（幂等、不覆盖、同盘 rename）：
后端启动、`reopen_engine()`（切数据根）时各调一次。

顺带修掉一个静默丢数据的路径：知识库台账原先钉死在**仓库根** `data/`（不跟数据根走），
用户把数据根设到别的盘之后，剧集跟着走了、编辑/上传记录留在原地。现在它跟 `drama.db` 同级同目录，
切根时由 `data_storage.change_data_root()` 一并复制。

**切根到底搬哪几样**（别靠猜，`change_data_root(migrate=True)` 的完整清单）：
`db/`（`drama.db` + `library-overlay.sqlite`）、`static/`、`traces/`、`lora-train/` —— 四样。
其中「整目录搬」的那部分以 `data_storage.MIGRATED_TREE_SUBDIRS` 为**单一真源**（现为
`("traces", "lora-train")`）；`db` / `static` / 覆盖层各有覆盖口（`DB_PATH`、`STORAGE_PATH`、
模块常量），只能各自算，**不在**这张表里。
`models/` 与 `hf-cache/`（能重下，几百 GB）、`knowledge/` 与 `materialized/`（能重算）、
`engine-dryrun/` 与 `repair/`（能重跑 / 重诊断）、`experiments/`（实验产物，重跑即可）
**故意不搬**。
`lora-train/` 是 2026-11-07 补进来的：训练参数与 `*.safetensors` 不可重建（重训几小时卡时），
而 `lora_train.config.data_dir()` 在新的数据根下会 `mkdir` 出一个**空目录** ⇒ 不搬的话
用户看到的就是「切个目录，训练集和权重全没了」（旧数据还在旧根，只是没人指向它）。
