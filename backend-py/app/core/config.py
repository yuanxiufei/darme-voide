"""全局配置加载 —— 与 ``backend/src/config.ts`` 逐项对齐。

优先级：环境变量 > ``configs/config.yaml`` > 硬编码默认值

数据根目录（SQLite 数据库 + 全部生成文件）：
``.data-root`` 标记文件 > ``DATA_ROOT`` 环境变量 > ``config.yaml database.path`` 所在目录 > ``./data``
数据库固定为 ``<dataRoot>/db/drama.db``，生成文件固定为 ``<dataRoot>/static``。

⚠️ **``data/`` 顶层只放目录，散文件一律归位**（2026-11-07 起）：此前 ``drama.db``、
``library-overlay.sqlite`` 和它们的 ``-wal``/``-shm`` 直接摊在数据根下，六个子目录与四个散文件混住，
既看不出「哪些是一套」，也没法整目录备份/整目录忽略。现在 SQLite 一律进 ``db/``（见
``data/README.md``，那份是给人看的，本 docstring 是给改代码的人看的）。

⚠️ 区分标准仍然只有一条：**换数据根时它跟不跟着走**（设置页「数据存储」会写根目录的
``.data-root``，此后 ``dataRoot`` 可能根本不在本仓）：

===============  ====================================================  ========  ==========
落点             谁定                                                  随根迁移  能否重建
===============  ====================================================  ========  ==========
``db/``          本模块（``get_db_dir``/``get_db_path``）＋              ✓        ✗ B 侧
                 ``library_overlay.OVERLAY_DB``
                 —— ``drama.db`` / ``library-overlay.sqlite``
                 与 ``db_upgrade`` 造的 ``.bak-*`` 全在这儿
``static/``      ``get_storage_root``（子目录见下一张表 ✓）              ✓        ✗
``traces/``      ``trace_store.get_traces_dir``（写入侧仍在 Node）       ✓        ✗
``knowledge/``   ``knowledge_ledger`` / ``outcome_index`` /            ✓        ✓ 全派生
                 ``training_registry`` 各自的 ``db_path``
``engine-dryrun/`` ``engine.output_paths.dryrun_root``                 ✓        ✓ 凭证
``materialized/`` ``engine.materialize.default_directory``             ✓        ✓ 反量化缓存
``indexes/``     ``script_library.OCR_DB`` / ``EXTRA_DB`` /             ✗        ✓ 三只派生
                 ``scene_index.SCENES_DB``（顶层目录，**不进** ``data/``）
``models/``      ``scripts.corpus.ocr_vlm_local.default_model_dir``     ✗        ✗ 需重下
                 ＋ ``models/tokenizers/``（tokenizer_hub 的第二落点）
``corpus/``      ``script_library.PLAYBOOK_ROOT`` /                    ✗        ✗ 但已入库
                 ``agent.tools.corpus_tools.CORPUS_JSONL``
===============  ====================================================  ========  ==========

⚠️ 两个同名 ``models/`` **不是一处** ✓✗（这是 ``.gitignore`` 最容易写错的一格）：
``data/models/`` 是 model_manager 下到**数据根**的出图权重（几十 GB、跟着数据根迁移、删了重下）
—— 它与 ``data/`` 下其余目录一样**必须**有 ``data/models/`` 这条忽略规则 ✓（2026-11-07 补：
此前写成 ``models/ocr/`` / ``models/tokenizers/`` 只挡住仓根那两个，``data/models/`` 一个字节没挡；
漏它的后果是 ``git add -A`` 试图整片吞掉几十 GB ✓✗）；而**仓库根**的 ``models/`` 是仓内素材
（要入库）。⇒ 忽略规则只许写 ``data/<名>/`` 这种带斜杠的锚定形，**裸 ``models/`` 会连仓根素材
一起挡掉** ✓✗。判据：``tests/output_layout_test.py`` ②′（``DATA_DIRS`` 每个名字都得有
``data/<名>/``，少了当场报红 ✓）。

表里没单列 ``hf-cache/``（2026-11-07 补记）：它由
``engine.tokenizers_tuning.default_cache_dir`` 定到 ``<dataRoot>/hf-cache`` 并被设成 ``HF_HOME``
⇒ transformers 一跑就往里写模型与词表（几个 GB 起）。跟着数据根走、删了能重下 ⇒ 与
``materialized/`` 同档（不进备份、也不随切根复制 ✓ 见 ``data_storage.change_data_root`` 只搬
``db/`` / ``static/`` / ``traces/`` / ``lora-train/`` 四样 —— 前三个是作品数据，最后一个是不
可重建的训练参数与权重）。
⚠️ 它此前**既没登记、也没进 ``.gitignore``** ✓✗：写法是 ``Path(root) / "hf-cache"``（``root`` 是
局部变量，不是 ``get_data_root()`` 调用）⇒ ``output_layout_test`` ② 的 AST 扫描只认「调用 / 字面量」
那种拼法，**抓不到** ⇒ 躲了一个多月。现已在 ``DATA_DIRS`` 登记（②′ 拿它核忽略条目）并有
``data/hf-cache/``，但**别把 ② 当全量扫描** ✗。

``static/`` 里住着谁（**全部生成物** ✓，2026-11-07 逐个核对过 ✓）：一条铁律 ——
**只有 ``get_storage_root()`` 后面直接跟子目录名这一种写法** ✓✗，它返回的**已经是**
``<dataRoot>/static`` ⇒ 再拼一次 ``"static"`` 就会落成 ``<dataRoot>/static/static/xxx`` ✓✗
（``video_interpolate`` 与 ``ffmpeg_compose`` 的混音中间件都这么错过一次 ✓：文件在
``static/static/`` 下、发给前端的 URL 指 ``static/`` 下 ⇒ 插帧成功的片子点开 404 ✓✗）。

=================  ================================  ==============================
子目录             谁写                              内容
=================  ================================  ==============================
``engine/``        ``engine.output_paths``          引擎产物：``<kind>/`` ✓（没给落点时）
                                                     ＋ ``<kind>-<id>/`` ✓（业务通路）
                                                     ＋ ``refs-audio/`` / ``refs-video/``
``images/``        ``file_storage``（``sub_dir``）  库图（``<uuid>.png``）
``uploads/``       ``routers/upload.py``            用户上传原图
``videos/``        ``webhooks`` / ``video_generation`` 外部/远端回传的视频
``audio/``         ``tts_generation`` / ``ai_voices`` 配音与分段音轨（``<uuid>.wav``）
``voices/``        ``routers/ai_voices.py``         音色参考样本
``frames/``        ``frame_extractor``              抽帧结果
``composed/``      ``ffmpeg_compose``               合成的成片
``mixed/``         ``ffmpeg_compose``               混音中间音轨（**留作复盘** ✓ 不删）
``subtitles/``     ``ffmpeg_compose``               字幕 ``.srt``
``merged/``        ``ffmpeg_merge``                 多镜拼接的成片
``interpolate/``   ``video_interpolate``            插帧产物
``jianying/``      ``jianying_draft``               剪映草稿素材
``h3/``            ``local_services.h3.server``     本地 H3 服务的产物（**不是**代码目录 ✓✗）
``comfyui/``       ``services.comfyui``             渲染产物 PNG/MP4（``comfyui_retention`` 按记录清）
``grid-cells/``    ``services.grid_split``          宫格图切出的单格 PNG（``cell_<ts>_<i>.png``）
=================  ================================  ==============================

``db/`` 里为什么连 ``library-overlay.sqlite`` 也算：它和 ``drama.db`` 是同一类 —— **丢了就真没了**
（编辑/上传台账），必须跟数据根一起走、一起备份；而 ``indexes/`` 那三只是**算得回来**的派生索引
（重建要跑数小时，故也别随手删，但它不属于「备份作品数据」的对象）。这条区别不要混：
**能不能重建决定它进不进 ``data/``，跟谁同一个子系统决定它进不进 ``db/``。**

``drama.db`` 的 ``.bak-*`` 只由 ``scripts/db_upgrade.py`` 造 ✓：``_backup`` 每次 ``--apply`` 留一份、
``_prune_backups`` **只留最近 ``KEEP_BACKUPS``（3）份** ✓。补列写着「一次性迁移」却是会反复做的事，
不清就一路累积 —— 2026-11-07 盘上曾同时躺着 ``0925``（缺 32 列的老库）与 ``1027`` 两份，
差好几个版本，只会让人分不清「升级前」是哪份 ✗。

⇒ 新增落点前先问一句：**它跟着数据根走吗？** 跟 ⇒ ``get_data_root()`` 且住在 ``data/``
（是 SQLite 就进 ``data/db/``，别摊在根下 ✗）；不跟（本仓自己的资产）⇒ ``PROJECT_ROOT`` 且
**独立成一个顶层目录**（按性质叫 ``corpus/`` 语料、``indexes/`` 派生索引、``models/`` 模型素材）。
判断依据是**丢了以后怎么办**：算得回来 ⇒ ``indexes/``；下载得到 ⇒ ``models/``；
仓库里有 ⇒ ``corpus/``；丢了就没了 ⇒ ``data/``。

``backend-py/`` 里另外两个目录**既不是数据根、也不是产物区**，2026-11-07 划清（此前混着用）：

* 实验**产物**（真跑出来的图/片/歌）⇒ ``<dataRoot>/experiments/<实验名>/`` ✓（2026-11-07 起 ✓）：
  **代码目录里不再有产物区** ✓✗ —— ``backend-py/output/`` 已退役，判据 ``output_layout_test`` ⑦
  **反着判**：``output/`` 存在即红 ✓✗。⚠️ 与 ``<dataRoot>/static/`` 的分界是硬边界：``static/``
  是**成品**（要发 URL、切数据根要搬、随备份走 ✓），``experiments/`` 是**实验**（不上线、
  不发 URL、切根不搬 ✓）—— 探针图落进 ``static/`` 就等于被 ``/static`` 公开发布并当成品回执 ✓✗。
  引擎自己的真产物**不在这里**，在 ``<dataRoot>/static/engine/`` ✓（那份要发 URL、要备份）。
* ``tmp/`` —— 现场**证据**：``run_all_*.log``（``tests/run_all.py`` 的账要指得回它们 ✓）、
  跑命令的 ``.cmd``、一次性探针脚本。**只许日志与脚本** ✓；媒体产物一律进
  ``<dataRoot>/experiments/<实验名>/`` ✓✗ —— 2026-11-07 实测里面躺着
  ``out_seedvr2/video_seed20260101.mp4`` 与 ``probe_image_out/sdxl-*.png``，已收进
  ``<dataRoot>/experiments/{seedvr2,sdxl-probe}/``（落 tmp 的产物会随 tmp 一起被当垃圾清 ✓✗；判据 ⑨）。
* ``_bench/`` —— 2026-09 那批调试**草稿**（probe 脚本 ＋ 它自己的 txt/png/onnx 备份），
  一次性、跑完即弃；**新实验不要再往这里加** ✓✗。

``tmp/`` 与 ``_bench/`` 都被 ``.gitignore`` 挡住 ✓；``backend-py/output/`` 那条**保留作兜底** ✓
（谁再往代码目录建产物区，先挡住不被提交，判据 ⑦ 同报红）。``tmp/`` 另有**专属**条目：
它原来只是被通用的 ``tmp/`` 规则顺带命中 ✓✗ —— 那条一改，这些现场证据（含 ``run_all``
的核账日志）就会混进提交 ✓✗。

还有第四类：**一次请求的中转件**。它不进数据根、也不是产物 ⇒ **绝不许住 ``static/``** ✓✗
（那是产物区：要发 URL、要随数据根备份；中转件落进去，进程一被杀就永久留一份没人认领的
垃圾 ✓✗）。两种合法形态，按生命周期选：

* **不需要活过当前调用** ⇒ ``tempfile.TemporaryDirectory()``（出了 with 自动删，成败都删）✓。
  例：``ffmpeg_merge`` 的 concat 清单（2026-11-07 从 ``static/temp/<uuid>.txt`` 改过来 ——
  原落点属产物区，且只在 ffmpeg 那一调用里用得上 ⇒ 连 ``<storage>/temp`` 这个目录都不该建 ✓）。
  ``static/`` 的子目录白名单里**已经没有 ``temp``** ✓：谁再往那儿写，``output_layout_test`` ① 当场红。
* **必须活过分块响应**（当前只有导出包）⇒ 落**系统临时区**且**自己一层子目录**：
  ``<temp>/darme-export/export-*.zip``（``export_service.export_tmp_dir()`` ✓）。两条硬规矩：
  ① **不许扔在 ``%TEMP%`` 根上** ✓✗（出事时根本分不清哪个是本仓的；2026-11-07 之前就是
  ``NamedTemporaryFile`` 直接落根 ✓✗）；② **必须有删除责任人** ✓ —— 三道：正常路径由
  ``routers/export.py`` 的 ``BackgroundTask`` 删 ✓；客户端半路断开 / 建完包又抛异常时
  background **跑不到** ✓✗，故两个端点的 ``except`` 里各自再删一次 ✓；``build_export_zip``
  每次建包前跑 ``sweep_stale_exports()`` 清超龄件（默认 6 小时）✓。**但第三道不够**：它
  依赖"下次还建包"，用户几个月不导出就永远等不到 ✓✗（那包可能几十 GB）⇒ ``main.py`` 的
  ``lifespan`` **启动时也清一次**（唯一不依赖用户行为的兜底时机，且被 ``try`` 包住 ——
  清理失败绝不能让服务起不来）✓。

判据：``tests/export_service_test.py``（导出包落点 / 清扫 / 两个端点 ``except`` 都删）、
``tests/merge_test.py``（``static/temp`` 目录压根不该存在）、``tests/output_layout_test.py`` ①
（``temp`` 已从白名单撤下）。

2026-11-07 之前 ``data/`` 里两类混住，已按上面四条拆开（``corpus/playbook`` ← ``data/playbook``、
``indexes/library-*.sqlite`` ← ``data/ocr``、``models/PaddleOCR-VL-1.6`` ← ``data/models``）。
拆之前踩过的坑：覆盖层库曾写 ``Path(__file__).resolve().parents[2]``（= ``backend-py/``），
于是它落到 ``backend-py/data/ocr/``，与三个兄弟库分居两个 ``data`` 目录；而
``backend-py/data/`` 被 ``.gitignore`` 整棵挡住 ⇒ 备份只搬一半、编辑与上传台账静默丢失。

与 Node 版的唯一差异：默认端口 ``5790``（Node 后端仍占用 5789），可用 ``PORT`` 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 缺依赖时降级为「全部走默认值」
    yaml = None  # type: ignore[assignment]

# backend-py/app/core/config.py -> core -> app -> backend-py -> 仓库根
# ⚠️ 2026-09-15 本文件从 `app/config.py` 移进 `app/core/` ⇒ **深度多了一层**，两个 parents[N]
#    都跟着 +1（当时漏改 ⇒ 冒烟立刻红：「PROJECT_ROOT == repo root」✗）。移动路径类模块时，
#    必须先数清 `parents[N]`。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 后端包根（``backend-py/``）—— **唯一权威**。凡「属于后端、又不该放仓库根」的路径（技能库、
#: 本地服务根…）都从这里派生；⚠️ 与 ``scripts/`` 侧的同名常量靠注释同步（scripts 不 import ``app.*``）。
BACKEND_PY_ROOT = Path(__file__).resolve().parents[2]

#: **`app/` 根**（``backend-py/app``）—— 三类「住在 `app/` 里、但不是应用分层代码」的资产的共同父目录：
#: ``app/skills``（内容资产）、``app/scripts``（工具链）、``app/local_services``（机器相关运行时）。
#: ⚠️ 2026-09-15 三目录并入 `app/` 后新增：此前这三处各自拼 ``BACKEND_PY_ROOT / "app" / ...`` ✗，
#: 搬家时漏一处就是**静默失效**（冒烟里就有一条兜底清理指错路径 ⇒ 残留 skill 让下一轮假红）。
APP_ROOT = BACKEND_PY_ROOT / "app"

CONFIG_PATH = Path(os.environ["CONFIG_PATH"]) if os.environ.get("CONFIG_PATH") else (
    PROJECT_ROOT / "configs" / "config.yaml"
)


def skills_dir() -> Path:
    """技能库目录（``backend-py/app/skills``）—— **唯一权威**；``SKILLS_DIR`` 环境变量可覆盖（测试隔离用）。

    ⚠️ 2026-09-15 收口：此前这个路径在**三处**各写一遍（services 侧 skills 扫描模块的常量
    —— 该模块已于 2026-10-30 删除、``app/agent/skills.py`` 的 ``skills_dir()``、
    ``app/scripts/check_skill_refs.py`` 的常量），
    并靠注释互相提醒「必须一致」✗ —— 那是「搬库漏改一处**不会报错**、只是静默读不到技能」的坑。
    现在运行链与 HTTP 层都从这里取；``app/scripts/`` 那份**刻意保留**独立实现（脚本不 import ``app.*``，
    见 ``app/scripts/README.md`` 的依赖约定），由守卫自检对齐。

    与 ``PROJECT_ROOT`` 的区别：技能库在 **`app/` 内**（``backend-py/app/skills/``），故这里从
    **本文件位置**推导（``app/core/config.py`` 上跳两级 = ``backend-py``），与 ``process.cwd()`` 解耦。
    """
    override = os.environ.get("SKILLS_DIR")
    if override:
        return Path(override)
    return APP_ROOT / "skills"


def _load_raw() -> dict[str, Any]:
    """解析 config.yaml；缺失/解析失败时降级为空对象（全部走默认值，不阻塞启动）。"""
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - 与 Node 版一致：任何异常都不阻断启动
        print(f"[config] 读取 {CONFIG_PATH} 失败，回退默认配置: {exc}")
        return {}


_RAW = _load_raw()
_SERVER: dict[str, Any] = _RAW.get("server") or {}
_DATABASE: dict[str, Any] = _RAW.get("database") or {}
_EVALUATION: dict[str, Any] = _RAW.get("evaluation") or {}
_AUTO_OPTIMIZE: dict[str, Any] = _EVALUATION.get("auto_optimize") or {}
_KNOWLEDGE: dict[str, Any] = _RAW.get("knowledge") or {}
_KNOWLEDGE_DISTILL: dict[str, Any] = _KNOWLEDGE.get("distill") or {}


def _resolve_path(value: Any) -> Path | None:
    """config.yaml 中的相对路径统一相对项目根解析。"""
    if not isinstance(value, str) or not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


DATA_ROOT_MARKER = PROJECT_ROOT / ".data-root"


def _read_data_root_marker() -> str:
    try:
        v = DATA_ROOT_MARKER.read_text(encoding="utf-8").strip()
        if v and Path(v).exists():
            return v
    except OSError:
        pass
    return ""


#: 数据根下的 SQLite 子目录名 ✓。所有库文件（含 ``-wal``/``-shm``/``.bak-*``）都住这儿，
#: 数据根顶层只留目录 —— 见模块 docstring 与 ``data/README.md``。
DB_SUBDIR = "db"

#: 搬进 ``db/`` 之前的旧位置：这些都是「以前直接瘫在数据根下」的散文件 ✓（迁移只认这几个名字 ✗）。
LEGACY_DB_NAMES = (
    "drama.db",
    "drama.db-wal",
    "drama.db-shm",
    "library-overlay.sqlite",
    "library-overlay.sqlite-wal",
    "library-overlay.sqlite-shm",
)


def _data_root_of_db_path(db_path: Path) -> Path:
    """由 ``database.path`` 反推数据根 —— **取父目录，但认 ``db/`` 这一层** ✓。

    两种写法都接受，且都落在同一个数据根上 ✓：
    ``./data/drama.db``（旧写法，父目录就是根）与 ``./data/db/drama.db``（新写法，再上一层才是根）。
    ⚠️ 只认一层 ✗：``db/`` 是代码固定的一层（``DB_SUBDIR``），配置里写得更深属于误配，
    按字面取父目录就好 —— 猜多了会让「数据根到底是哪个目录」变得说不清 ✓。
    """
    parent = db_path.parent
    return parent.parent if parent.name == DB_SUBDIR else parent


_default_data_dir = PROJECT_ROOT / "data"
_yaml_db_path = _resolve_path(_DATABASE.get("path"))

# 是否通过「标记文件 / DATA_ROOT 环境变量」显式指定过数据根目录。
# 显式指定后 DB_PATH / STORAGE_PATH 不再覆盖，确保目录切换真正生效。
_marker_root = _read_data_root_marker()

if _marker_root:
    _data_root = Path(_marker_root)
    _data_root_explicit = True
elif os.environ.get("DATA_ROOT"):
    _data_root = (PROJECT_ROOT / os.environ["DATA_ROOT"]).resolve()
    _data_root_explicit = True
else:
    _data_root = _data_root_of_db_path(_yaml_db_path) if _yaml_db_path else _default_data_dir
    _data_root_explicit = False


def get_data_root() -> str:
    """当前数据根目录（含数据库与全部生成文件）。"""
    return str(_data_root)


def get_db_dir() -> str:
    """当前 SQLite 目录绝对路径（``<dataRoot>/db``）。"""
    return str(_data_root / DB_SUBDIR)


def get_db_path() -> str:
    """当前 SQLite 数据库文件绝对路径（``<dataRoot>/db/drama.db``）。"""
    if not _data_root_explicit and os.environ.get("DB_PATH"):
        return os.environ["DB_PATH"]
    return str(_data_root / DB_SUBDIR / "drama.db")


def ensure_db_layout() -> list[str]:
    """建好 ``<dataRoot>/db/``，并把**旧位置**的库文件搬进来（幂等 ✓，返回这次搬了哪些 ✓）。

    ⚠️ 迁移必须**幂等且不覆盖** ✗：目标已存在 ⇒ 只搬 ``-wal``/``-shm`` 这类旁文件，
    主库文件**原地不动** ✓（那说明有人手放了一份新的，谁新谁旧不由本函数猜 ✗）。

    ⚠️ 迁移是 ``rename`` ✓（同盘瞬时、不复制 GB 级库）；跨盘（用户把数据根设在别的盘）时
    ``rename`` 会抛 ``OSError`` ⇒ 退回 ``shutil.move`` ✓。

    ⚠️ ``drama.db.bak-*``（``db_upgrade`` 造的历史备份）**跟着库一起搬** ✓：只搬主库会把
    「库进了 ``db/``、备份留在根下」拆成两处，下一个人翻目录时只看到主库、以为没备份 ✗。
    """

    def _move(legacy: Path, dest: Path) -> bool:
        try:
            legacy.replace(dest)
        except OSError:
            try:
                import shutil

                shutil.move(str(legacy), str(dest))
            except OSError:
                return False
        return True

    target = _data_root / DB_SUBDIR
    moved: list[str] = []
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return moved
    candidates = [_data_root / name for name in LEGACY_DB_NAMES]
    # 备份是**一批**（`drama.db.bak-<时间戳>`），不能列进 LEGACY_DB_NAMES（那是定宽白名单）⇒ 单独 glob
    candidates += sorted(_data_root.glob("drama.db.bak-*"))
    for legacy in candidates:
        if not legacy.is_file():
            continue
        dest = target / legacy.name
        if dest.exists() and not legacy.name.endswith(("-wal", "-shm")):
            continue  # 主库/备份已在 ⇒ 不赌哪份新，交给人和 db_upgrade --plan 去判 ✓
        if _move(legacy, dest):
            moved.append(legacy.name)
    return moved


def get_storage_root() -> str:
    """当前生成文件存储根目录绝对路径（<dataRoot>/static）。"""
    if not _data_root_explicit and os.environ.get("STORAGE_PATH"):
        return os.environ["STORAGE_PATH"]
    return str(_data_root / "static")


def set_data_root(new_root: str) -> None:
    """运行时切换数据根目录（写标记文件，重启后依然生效）。"""
    global _data_root, _data_root_explicit
    abs_path = Path(new_root).resolve()
    _data_root = abs_path
    _data_root_explicit = True
    abs_path.mkdir(parents=True, exist_ok=True)
    DATA_ROOT_MARKER.write_text(str(abs_path), encoding="utf-8")


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _cors_origins() -> list[str]:
    from_env = _env_list("CORS_ORIGINS")
    if from_env:
        return from_env
    from_yaml = _SERVER.get("cors_origins")
    if isinstance(from_yaml, list) and from_yaml:
        return [str(x) for x in from_yaml]
    # 与 Node 版默认值一致
    return ["http://localhost:3013", "http://localhost:5789", "http://localhost:5790"]


server = {
    # ⚠️ 刻意**不读** config.yaml 的 server.port：它属于 Node 后端（当前是 5789），
    # 绞杀者阶段两个后端要并存，读同一个端口号必然抢占。
    # 端口优先级：PY_PORT > PORT > 5790。完全迁移后想单端口（5789 同时服务 API 与前端）时，
    # 启动时显式给 PY_PORT=5789 即可。
    "port": int(os.environ.get("PY_PORT") or os.environ.get("PORT") or 5790),
    "host": str(os.environ.get("HOST") or _SERVER.get("host") or "0.0.0.0"),
    "cors_origins": _cors_origins(),
}

evaluation = {
    "auto_optimize": {
        "enabled": os.environ.get("AUTO_OPTIMIZE_ENABLED") == "true"
        or _AUTO_OPTIMIZE.get("enabled") is True,
        "hour": int(os.environ.get("AUTO_OPTIMIZE_HOUR") or _AUTO_OPTIMIZE.get("hour") or 3),
        "minute": int(os.environ.get("AUTO_OPTIMIZE_MINUTE") or _AUTO_OPTIMIZE.get("minute") or 0),
        "iterations": int(
            os.environ.get("AUTO_OPTIMIZE_ITERATIONS") or _AUTO_OPTIMIZE.get("iterations") or 3
        ),
        "run_on_startup": os.environ.get("AUTO_OPTIMIZE_RUN_ON_STARTUP") == "true"
        or _AUTO_OPTIMIZE.get("run_on_startup") is True,
    }
}

#: 知识蒸馏的**定时复盘**（P2 交付物里的"定时任务"）。
#:
#: 默认**关**：它会在无人看着的时候往台账写 ``proposed`` 条目、并可能把
#: ``lift<0`` 的旧经验降级。这类"自动改事实"的开关与 ``evaluation.auto_optimize``
#: 同级 —— 一律由创作者逐项打开，代码不许替人决定。
#:
#: ``lift`` 单独一格（默认也关）是因为它比蒸馏更"有力"：蒸馏只**增加待审条目**，
#: lift 会**把已有条目降级**（``deprecated``）。两件事不该绑在一个开关上。
knowledge = {
    "distill": {
        "enabled": os.environ.get("KNOWLEDGE_DISTILL_ENABLED") == "true"
        or _KNOWLEDGE_DISTILL.get("enabled") is True,
        #: 避开 03:00 的评测优化（两条定时任务同时跑会抢 SQLite 写锁）
        "hour": int(os.environ.get("KNOWLEDGE_DISTILL_HOUR")
                    or _KNOWLEDGE_DISTILL.get("hour") or 4),
        "minute": int(os.environ.get("KNOWLEDGE_DISTILL_MINUTE")
                      or _KNOWLEDGE_DISTILL.get("minute") or 30),
        #: 每轮最多写几条（候选按信号强度排序，宁可少写几条也不要把台账灌满）
        "limit": int(os.environ.get("KNOWLEDGE_DISTILL_LIMIT")
                     or _KNOWLEDGE_DISTILL.get("limit") or 3),
        "lift": os.environ.get("KNOWLEDGE_DISTILL_LIFT") == "true"
        or _KNOWLEDGE_DISTILL.get("lift") is True,
        "run_on_startup": os.environ.get("KNOWLEDGE_DISTILL_RUN_ON_STARTUP") == "true"
        or _KNOWLEDGE_DISTILL.get("run_on_startup") is True,
    }
}

# ===== 绞杀者（strangler）遗留开关 =====
# ⚠️ 2026-09-15：Node 后端（`backend/`）**已删除** ⇒ 这两个开关**没有反代对象**了。
# 保留只为「接缝」语义：未实现的路径仍然回 501 并说明原因（比静默 404 好排查）。
# 换句话说：**现在不要设 PROXY_TO_NODE=1** —— 那只会连不上并返 502。
PROXY_TO_NODE = os.environ.get("PROXY_TO_NODE", "0") == "1"
NODE_BACKEND_URL = (os.environ.get("NODE_BACKEND_URL") or "http://127.0.0.1:5789").rstrip("/")

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
