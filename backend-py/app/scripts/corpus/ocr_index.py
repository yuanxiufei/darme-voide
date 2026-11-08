#!/usr/bin/env python3
"""把供应商库里**没有正文**的扫描件 OCR 进本仓的 sidecar 索引（``indexes/library-ocr.sqlite``）。

为什么需要它
============

`curated-screenplay-methods.sqlite` 登记 5403 份文档，但只有 **2262 份**真抽出了文字；
剩下 3139 份是**扫描件（无文本层）**，`get_text()` 返回空。抽查 5 份全是
`chars_first5=0` ⇒ 不是抽取失败，是根本没有文字层。这些书里包含《序列编剧法》《电影剪辑》
《中央戏剧学院剧本选 97 册》这类真用得上的东西，搜不到就等于库里没有。

所以本脚本做一件事：**逐页 OCR，写进 sidecar，不动供应商那份库**。

三种「没有文字层」的形态都要管（前两种盘上各有一批，第三种是留的口子）：

* **PDF 扫描件**（3139 份）：`get_text()` 为空 ⇒ 用 `pypdfium2` 按 dpi 渲染成图再 OCR；
* **整页图片贴进 Word 的 docx**（2 份：《撕夜》全剧本 23.8 MB / 开端 8.7 MB）：docx 的
  `word/document.xml` 正文层 0 字，内容全在 `word/media/*` 的 44/49 张图里 ⇒ 按正文
  出现顺序取图再 OCR（docx 就是个 zip，不需要 python-docx 那个额外依赖）；
* **独立图片文件**（png/jpg/jpeg/bmp/tif/tiff/webp）：文件本身就是页，多帧 TIFF 按帧
  算页。当前 5323 份语料里一处也没有，但「一份扫描件被拆成散图」就是它 —— 三条路
  最后都收敛到 `_page_text`，所以版面重排、门禁、切块、VLM 升级不必各写一遍。

* 供应商库：`?mode=ro` 只读，一个字节都不写（它是第三方资产，改了就没有比对基准）；
* sidecar：同构 schema（`documents` ＋ fts5 `chunks`，trigram），检索层按秩合并两库；
* `signature = size:modified_ns` ⇒ 同一份文件重跑会跳过（`--redo` 才重做），可随时 Ctrl-C
  续跑 —— 每份文档一个事务，中断最多丢一份。

优先级（先 OCR 真用得上的，期刊杂志排最后）
==========================================

``644 份需要 OCR 的里面有 2495 份是《世界电影》36 年期刊合订``（占 8.4 GB 里的 4.9 GB）——
那是研究史料，写剧本时几乎不会查。默认排序把剧本与编剧方法书排前面，期刊排最后；
``--path-substr`` 可以只做某一棵子树。

用法（⚠️ PowerShell 直接传中文参数会乱码 ⇒ 一律用 ``--path-substr=`` 且加引号；
取值型参数**只认等号形式**，写成 ``--path-substr 某书名`` 会被 :func:`_split_value_args`
当场退 2 —— 它取不到值时会静默退回「不筛」，配合 ``--redo`` 就是全量重烧）::

    # 看还剩多少、要多久
    python backend-py/app/scripts/corpus/ocr_index.py --stats

    # 看会按什么顺序做（不落盘）
    python backend-py/app/scripts/corpus/ocr_index.py --plan --limit=20

    # 先试一本：只做前 12 页，确认识别质量
    python backend-py/app/scripts/corpus/ocr_index.py --run --limit=1 --pages=12

    # 正式跑：剧本与编剧书优先，一次 20 份（默认按核数开 8 进程）
    python backend-py/app/scripts/corpus/ocr_index.py --run --limit=20

    # 只补短剧合集
    python backend-py/app/scripts/corpus/ocr_index.py --run --path-substr="短剧待精筛"

    # 校对：看哪些页被门禁拦下了、为什么、原文长什么样（不跑 OCR，秒出）
    python backend-py/app/scripts/corpus/ocr_index.py --audit
    python backend-py/app/scripts/corpus/ocr_index.py --audit --verdict=suspect --limit=100

    # 双保险（推荐）：**本进程内**用 transformers 跑 PaddleOCR-VL，只重读本地判可疑的页
    python backend-py/app/scripts/corpus/ocr_index.py --run --vlm --vlm-backend=local

    # 或者 VLM 跑在别处（vLLM / SGLang / paddleocr genai_server，必须环回端口）
    python backend-py/app/scripts/corpus/ocr_index.py --run --vlm \
        --vlm-endpoint=http://127.0.0.1:8118/v1 --vlm-model=PaddleOCR-VL-1.6 --vlm-max=6

    # 看升级/分歧痕迹（这类页已入库，只是两遍读法不同）
    python backend-py/app/scripts/corpus/ocr_index.py --audit --verdict=escalated

参数：``--stats``｜``--plan``｜``--run``｜``--audit``｜``--limit=N``｜``--only=partial|done``
（``partial``＝只补「跑了没跑完」的那批，它们的块已进 fts5、重跑前一直被检索层挡着；
``done``＝只补「已经跑过」的，**隐含 ``--redo``**，用于切块/清洗/门禁口径变更后按新规则重烧
存量 —— 光给 ``--redo`` 的话 3000+ 份待办会一起排进来，``--limit`` 默认 20，要重跑的老份
可能一份都轮不上）｜``--pages=N``（每份最多前 N 页，试跑用）｜``--dpi=150``｜``--workers=N``
（进程数，默认由跑批前一次资源探测给：GPU 档 ``min(会话数, 4, 预算/7)``，会话数＝
``min(4, (空闲显存−1024MiB)//4200MiB)``；CPU 档或开不起会话时 ``min(8, 预算/4)``。
预算＝硬件余量（逻辑线程 − ``OCR_CPU_RESERVE``）再按本机**外部占用**收缩，见 :func:`apply_snapshot`）｜
``--path-substr=``｜``--redo``（忽略 signature 重做）｜``--verdict=reject|suspect|escalated``｜
``--reaudit``（**只重跑「有存疑页」的件**，隐含 ``--redo``；门禁判据改了口径、旧判据留下
的存疑页要按新规则重判时用它 —— 存疑页本身照常入库、一个字没丢，重判只是为了把
``--audit`` 的噪声清掉，别拿它当"补内容"用）｜
``--json``｜``--no-layout``（关掉分栏重排，仅对照）｜``--vlm``＋``--vlm-backend=local|http``
（local＝进程内 transformers，需先跑 ``ocr_vlm_fetch.py --fetch``；http＝环回服务）／
``--vlm-model-path=``（local 权重目录）／``--vlm-endpoint=``／``--vlm-model=``／
``--vlm-max=N``（每份最多升级几页，默认 6）／``--vlm-probe``（连 ok 页也送 VLM 做全量对拍，
慢，只在标定时用）

三层处理顺序（都可在 ``--stats`` 里看到当前状态）
================================================

1. **分栏重排**（``ocr_layout``）：双栏页按「栏序 → 栏内 y」重新排序。纯几何，无模型，
   对单栏页是恒等变换；
2. **本地 OCR**（rapidocr / PP-OCRv6）：原有路径，速度基准；
3. **VLM 升级**（``ocr_vlm``）：只把本地判定为 suspect／reject 的页交给本地 VLM 重跑，
   每份有预算上限；两遍都判 ok 但文本不同的页记 ``escalated`` 待人工复核。

⚠️ VLM 端点只接受**环回地址**（127.0.0.1 / localhost / ::1）：语料是采购库，出网即事故。
配了外部地址会被直接拒绝，而不是"提醒一下照样发"。

质量门禁（``text_clean``）：入库前每页先定性
============================================
原先唯一的门槛是 ``MIN_PAGE_CHARS = 12`` —— 引擎把整页认成一排符号（``||||``/``────``）
或把一行刷几十遍时字数轻松过 12，于是脏文本直接进 fts5，检索命中一堆读不出意思的
片段，而且**块里没有"质量"字段，事后看不出来**。现在的做法：

1. **清洗**（``text_clean.clean_document``）：控制字符、孤字/纯标点碎行合并（实测 79%
   有文字层的页都有这种碎行）、跨页重复的页眉页脚剔除（期刊每页都有 ``·232·`` 和
   刊名，同一归一形在 ≥3 页且 ≥50% 的页首/页尾出现才删，且只删 40 字以内的行）；
2. **门禁**（``text_clean.judge_page``）：``ok|suspect|reject`` 三档。``reject`` 的页
   **一个块都不写**，改记进 ``page_audit`` 表（含理由与原文预览）；
   ⚠️ 但**整份每页都判废**时会先走一次保底（``_salvage_all_rejected``）：软性判废
   （符号段密集/表格页/短行，字认得对）降为存疑留内容，硬伤（乱码/整块被认成符号/
   整页只剩几个字）照旧丢弃。2026-11-04 实测有 1 份 1 页期刊 574 字正文被整页丢光、
   ``status`` 仍写 ``ocr`` ⇒ 库里一本查不到的隐形空书；
3. **整份隔离**：reject 页占比 ≥ 50% 且参与判定的页 ≥ 3，**或**「一块都没留下而丢掉
   的页上有字」⇒ ``status='quarantine'``，整份不写块。它算"已处理"（不会被下次批处理
   重复烧 CPU），要重试必须显式 ``--redo``。口径：隔离只管「读出来的东西不能用」（硬伤），
   「排版被认成符号但字是对的」一律留内容 —— 假阳性丢内容比留一点噪声严重得多。

判据只盯**结构上不可能正常**的信号（替换符/私用区/控制字符、同字符连续 8 次、整页
去重字符 < 8、有效字符占比 < 60%）。**CJK 占比不参与判定** —— 英文原版剧本、纯数字
表格页 CJK 为 0 是合法的，拿它判会把整批英文剧本误杀。

⚠️ **别用 `Start-Process -WindowStyle Hidden` 起长批**：实测那样 onnxruntime 会挂死
（进程在、CPU 恒 0、一页都不出），同一条命令在前台控制台跑就正常。要在后台跑就用
「前台窗口 + `--workers`」，或者写进计划任务（有真控制台）。

⚠️ **长批要能扛住「有人同时读库」**：sidecar 是 WAL，读者不挡写者，但读者的**长事务**
（备份整个库、大范围 ``LIKE`` 扫描）会挡写者的提交。2026-11-04 真实事故：整批跑到
2739/2988 时被一次只读探测查询顶掉，``_save`` 抛 ``database is locked`` 冲出 ``main()``，
9 小时的活儿当场结束。现在两道防线：连接自带 ``busy_timeout``（``SIDECAR_BUSY_MS``，
默认 SQLite 只等 5 秒，那是给短事务的口径）；``_save_atomic`` 在写锁上退让重试
（``LOCK_RETRIES``，非锁错误立刻抛，不把真 bug 埋进等待里）；``report`` 再兜一层 ——
单份存不进去只记一条失败并继续，该份在 sidecar 里没登记 ⇒ 下一轮自动重来。
**顺带一条纪律：批次在跑时别对它做整库备份/全表扫描**，要查就用「先取候选 id 集合、
再在内存里做差集」，别写 ``NOT EXISTS (SELECT 1 FROM chunks ...)`` 那种逐份全表扫描。

⚠️ 依赖：``pypdfium2``（渲染页面，与第三库的 PDF 文字层同一个包）＋ OCR 引擎。两者都只在
**离线批处理**里用，不进运行时镜像 —— 服务端只需要读 sidecar 的 sqlite。引擎有两档，默认自动选新档（见 ``_new_engine``）：

* ``rapidocr``（3.x，PP-OCRv6 small）——**优先**，实测 0.32 s/页；
* ``rapidocr-onnxruntime``（1.4.x，PP-OCRv4）——装不上新档或要对照时的回退，1.75 s/页。

``OCR_ENGINE=auto|rapidocr|rapidocr-onnxruntime`` 可钉死档位。
``OCR_DEVICE=auto|cuda|cpu`` 选推理设备（默认 ``auto``：这套 onnxruntime 真能上 CUDA 就用，
装 ``onnxruntime-gpu`` 但缺 CUDA 13／cuDNN 9 的 DLL 时它只会静默回落 CPU，故 ``auto``
先问 provider、再在批前跑一张小图把会话建出来实报，显式 ``cuda`` 却拿不到就当场中止）。

速度口径（都是同一份书、同一台机实测，别按直觉改）
====================================================

一页 A4 扫描件的识别开销几乎全在「识别模型吃了几批」上，而不是在渲染或检测：

===========================  =========  =========  ==========
配置                           s/页       字/秒      识别字数
===========================  =========  =========  ==========
老档 PP-OCRv4 ``batch=6``      7.68        46         2816
老档 PP-OCRv4 ``batch=16``     1.75       174         2811
新档 PP-OCRv6 ``batch=16``     **0.32**   ~1300       417（3 页）
===========================  =========  =========  ==========

⇒ 全量 17.5 万页：老档默认口径约 370 小时，现在约 **16 小时**（单进程；多进程再除）。
"""
from __future__ import annotations

import contextlib
import functools
import gc
import importlib.util
import io
import json
import os
import posixpath
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[3]
if str(BACKEND_PY) not in sys.path:
    sys.path.insert(0, str(BACKEND_PY))

from app.services import script_library  # noqa: E402 —— 必须在 sys.path 调整之后
from app.scripts.corpus import ocr_layout  # noqa: E402 —— 版面分栏重排（纯几何，零依赖）
from app.scripts.corpus import ocr_vlm  # noqa: E402 —— 本地 VLM 升级通道（默认关闭）
from app.scripts.corpus import text_clean  # noqa: E402 —— 清洗与门禁（纯函数，无重依赖）

_CORES = os.cpu_count() or 8


def cpu_room() -> int:
    """这一批**总共**允许多少个计算线程：逻辑线程数 − 保留。

    放在 import 期就要用它的位置，是为了让 ``OMP_NUM_THREADS`` 与建 session 时给 ORT 的
    ``intra_op_num_threads`` 走**同一个**公式 —— 两处各写一遍早晚会不一致。
    """
    try:
        reserve = max(0, int(os.environ.get("OCR_CPU_RESERVE") or 2))
    except ValueError:
        reserve = 2
    try:
        cap = int(os.environ.get("OCR_THREAD_CAP") or 0)
    except ValueError:
        cap = 0
    room = _CORES - reserve if _CORES > reserve else 0
    room = min(room, cap) if cap > 0 else room
    return max(1, room)


#: ⚠️ 真正的线程数**不在这里**：``OMP_NUM_THREADS`` 对 onnxruntime 的 intra-op 池无效，
#: 只有建 session 时给的 ``intra_op_num_threads`` 才算数（见 ``engine_threads``，那是唯一
#: 真源）。这里留一份只为兼容少数仍走 OpenMP 的算子库 —— 两处口径已对齐成同一个公式。
os.environ.setdefault(
    "OMP_NUM_THREADS",
    str(max(1, cpu_room() // max(1, int(os.environ.get("OCR_WORKERS") or 1)))),
)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

#: 一个块的目标字数（主库实测 128166 块 / 2262 文档 ≈ 57 块/文档，此处按「一屏文字」取 800）
CHUNK_CHARS = 800
#: 渲染 dpi：150 实测 3.1–3.7 s/页（本机 32 核 i7）、中文识别干净（120 开始丢标点，200 只慢不更准）
DEFAULT_DPI = 150
#: 默认进程数：实测单进程一页 3.7 s，32 核上 8 进程已能把 CPU 吃满且内存宽裕
#: （每进程一份 PDF + 一个 onnx 会话 ≈ 500 MB，94 GB 机器开 8 个毫无压力）。
#: 真正的取法见 ``default_workers()`` —— 这里保留常数只为兼容外部引用。
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 4))
#: 低于这个字数就认为是空白页/插图页（不写块，但要计入 pages）
MIN_PAGE_CHARS = 12
#: 文字层达到这个字数才**信任**它、跳过 OCR。低于此值一律渲染送 OCR。
#:
#: 2026-11-04 实测踩坑：原判据是 ``>= MIN_PAGE_CHARS``（12 字），于是「水印型扫描件」
#: 全被误判成有文字层 —— ``303.中戏60种必读剧本超清PDF`` 整批每页文字层只有 4 行水印
#: （``中央戏剧学院60个剧本 / 为学院派而生 / 海子文字铺 / 07/26/2014`` ＝ 35 字）＋页码，
#: 正文 700+ 字全在扫描图里。走 native 后每页只存 35 字水印，清洗阶段又把「3 页都出现的
#: 水印」当跨页页眉剔掉，最终整份 0 字、门禁判「空白页」——**入库但检索不到**。
#: 同类共 168 份 / 27888 页（占全量 16.4%），详见 ``_bench/scan_risk.py``。
#:
#: 取 100 的依据：正常印刷页正文 300–800 字/页，水印/页眉/页码通常 20–60 字，中间地带很窄；
#: 真短页（扉页、单行诗）走 OCR 只是多花一点算力，结果只会更全，不会更差。
NATIVE_TRUST_CHARS = 100

#: 优先级：命中越靠前越先做；没命中的排最后。期刊杂志显式压到最后。
_PRIORITY = (
    "短剧待精筛",
    "220本编剧影视专业书籍",
    "编剧书籍",
    "62本影视专业书籍",
    "中央戏剧学院剧本选",
    "经典电影剧本中文译本",
    "外国电影剧本丛刊",
    "外国经典剧本",
    "外国当代剧作选",
    "外国独幕剧剧本选",
    "全球电影剧本待核验",
    "英文原版电影与剧集剧本",
    "世界电影杂志",  # 期刊史料 —— 排在所有剧本之后
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    name TEXT,
    extension TEXT,
    size INTEGER,
    modified_ns INTEGER,
    signature TEXT,
    pages INTEGER,
    status TEXT,
    gated INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    document_id UNINDEXED, path UNINDEXED, name, page UNINDEXED,
    chunk_no UNINDEXED, text, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS page_audit (
    document_id INTEGER, path TEXT, name TEXT, page INTEGER,
    verdict TEXT, reasons TEXT, chars INTEGER, preview TEXT
);
CREATE INDEX IF NOT EXISTS page_audit_doc ON page_audit(document_id);
CREATE INDEX IF NOT EXISTS page_audit_verdict ON page_audit(verdict);
"""


def _arg(name: str, default: str = "") -> str:
    prefix = f"--{name}="
    return next((arg[len(prefix):] for arg in sys.argv if arg.startswith(prefix)), default)


def _flag(name: str) -> bool:
    return f"--{name}" in sys.argv


def _limit_arg(default: int) -> int:
    """``--limit`` 专用：**0 表示全部**。

    ``_count`` 会把 0 夹成 1（那是 ``--dpi``／``--workers`` 该有的口径：0 个线程毫无意义），
    但 ``--limit=0`` 是"把候选全做完"的常用写法。2026-11-04 实测踩过：``--run --reaudit
    --limit=0`` 本意是把 368 份存疑件全重判，实际只做了 1 份，而日志写「本次做 1 份」
    看起来像正常收尾 —— 差掉的 367 份没有任何提示。``_run`` 里 ``todo = plan[:limit]
    if limit else plan`` 本来就是按"0＝全部"写的，只是这个入口把它夹没了。
    """
    try:
        return max(0, int(_arg("limit", str(default))))
    except (TypeError, ValueError):
        return default


def _count(name: str, default: int) -> int:
    try:
        return max(1, int(_arg(name, str(default))))
    except (TypeError, ValueError):
        return default


#: 取值型参数的**完整清单**（与各 ``_arg``／``_count`` 调用点一一对应）。
#: 这些参数只认 ``--name=value``；写成 ``--name value`` 时 :func:`_arg` 取不到值会
#: **静默**退回默认，而默认往往意味着「不筛、跑全量」——2026-11-04 实测踩过一次：
#: 一条本该只重跑 1 份的 ``--redo --path-substr 某书名`` 变成「按优先级前 20 份全量重烧」，
#: 十几分钟 CPU/GPU 白跑。拼错的形式必须当场拒，不许静默降级。
VALUE_ARGS = ("limit", "dpi", "pages", "path-substr", "only", "workers", "verdict",
              "vlm-backend", "vlm-endpoint", "vlm-model", "vlm-model-path", "vlm-max")


def _split_value_args() -> list[str]:
    """找出 ``--name value`` 这种空格写法，返回人话片段（空列表＝参数形式没问题）。"""
    bad: list[str] = []
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if not arg.startswith("--") or "=" in arg:
            continue
        if arg[2:] not in VALUE_ARGS:
            continue
        nxt = argv[index + 1] if index + 1 < len(argv) else ""
        bad.append(f"{arg} {nxt}".strip() if nxt and not nxt.startswith("--") else arg)
    return bad


def _rel(raw: str) -> str:
    """主库里存的是**建库机器上的旧盘符绝对路径**（形如 ``…\\全球热门与获奖影视剧本知识库\\08-…\\x.pdf``），取锚点右边。

    统一成正斜杠：``--plan`` 打印的 rel 要能直接贴进 ``--path-substr``（以及和
    ``/knowledge/documents`` 返回的 path 逐字对得上），混着反斜杠就得靠人脑转换。
    """
    return raw.split(script_library.LOCAL_MARKER, 1)[-1].lstrip("\\/").replace("\\", "/")


def _local(raw: str) -> Path | None:
    return script_library.resolve_local(raw)


def _priority(label: str) -> int:
    for index, key in enumerate(_PRIORITY):
        if key in label:
            return index
    return len(_PRIORITY)


def _partial_paths() -> set[str]:
    """sidecar 里 ``status='partial'`` 的 path（跑了但没跑完的那批）。

    单独取一遍而不用 ``_done_signatures()`` 反推：那份只回 ``ocr``/``quarantine``，
    差额里混着「压根没碰过」的 3121 份，分不出谁是半成品。
    """
    conn = _open_sidecar()
    try:
        return {path for (path,) in conn.execute(
            "SELECT path FROM documents WHERE status = ?", (STATUS_PARTIAL,))}
    finally:
        conn.close()


def _recorded_paths() -> set[str]:
    """sidecar 里**有记录**的 path（``ocr``／``partial``／``quarantine`` 都算跑过）。"""
    conn = _open_sidecar()
    try:
        return {path for (path,) in conn.execute("SELECT path FROM documents")}
    finally:
        conn.close()


def _extra_quarantined_docx() -> list[dict]:
    """第三库（``lib_index.py``）里**抽不出字**的 docx —— 正文层就是整页图片，该归 OCR。

    为什么非得在这里接一手：主库是供应商清单、**不包含**这批盘面，第三库又只会把它记成
    ``quarantine`` 一块不写。两条通道各自都"没做错"，合起来这份书就消失了 ——
    实测 1 份：``…/188-短剧剧本+微短剧项目写作基础/短剧编剧和剧本资源/基础知识.docx``，
    3.95 MB 里 ``word/document.xml`` 只有 7.9 KB、剩下的全是 6 张 jpeg。

    只接 ``status='quarantine'``：第三库正常收下的（``extra``）已经有正文块，再排进 OCR
    队列是白烧算力。

    ⚠️ 2026-11-07 起 ``.pdf`` 也归第三库抽文字层（``lib_index._pdf_text``），所以
    **上传的扫描件 PDF 会先记 ``quarantine``、再由这里接进 OCR 队列** —— 靠的是
    ``SUPPORTED_SUFFIXES`` 里本来就有 ``.pdf``。这一条是那条通道的**唯一补救路**：
    若哪天把它从集合里拿掉，用户新传的扫描件就会变成"落盘了、永远搜不到、也没有
    任何待办"，``tests/lib_index_test.py`` 里钉了一条 ast 判据守它。
    """
    from app.scripts.corpus import lib_index  # noqa: PLC0415 —— 只在算待办时需要

    db = Path(lib_index.EXTRA_DB)
    if not db.exists():
        return []
    conn = script_library._connect(db)  # noqa: SLF001 —— 同一模块口径
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT path, name, extension, size, modified_ns FROM documents"
            " WHERE status = ?", (lib_index.STATUS_QUARANTINE,)).fetchall()
    finally:
        conn.close()
    root = Path(script_library.LIBRARY_ROOT)
    out: list[dict] = []
    for rel, name, ext, size, mtime_ns in rows:
        suffix = (ext or Path(rel).suffix).lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        rel = Path(rel).as_posix()
        out.append({
            "path": str(root / rel),
            "rel": rel,
            "name": name or Path(rel).name,
            "extension": suffix,
            "size": int(size or 0),
            "modifiedNs": int(mtime_ns or 0),
            "pages": 0,
            "signature": f"{size}:{mtime_ns}",
            "priority": _priority(rel),
        })
    return out


def _pending(*, redo: bool = False, only: str = "") -> list[dict]:
    """主库里**登记了但没有正文块**的文档，按优先级排序；已 OCR 且 signature 未变的跳过。

    ``only='partial'`` 只留「有 sidecar 记录但没跑完」的那批。它们是**真正该插队**的：
    块已经写进 sidecar 的 fts5（``_save`` 只对 quarantine 不写块），在重跑完成之前
    一直被检索层当作「全书可查」，只是默认被 ``script_library`` 的 status 过滤挡掉而已。
    按体积排序时它们确实靠前，但混在 3135 份里没法只挑它们 —— 这个口子就是为此开的。

    ``only='done'`` 只留「已经跑过、sidecar 里有记录」的那批，**隐含 redo**。用途是
    切块/清洗/门禁规则改了口径之后按新规则重烧存量：那时 ``--redo`` 不够用 —— 它会把
    3000+ 份待办一起排进来，``--limit`` 默认 20，真正要重跑的老份可能一个都轮不上
    （2026-11-04 改「块页记起始页」时实测踩过，白烧了三份大书的前 1000 页）。
    """
    if only == "done":
        redo = True  #: 口径重烧必然要重跑已完成份，签名一致也得重烧
    conn = script_library._connect(Path(script_library.METHODS_DB))  # noqa: SLF001 —— 同一模块口径
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT id, path, name, extension, size, modified_ns, pages FROM documents"
        ).fetchall()
        have = {r[0] for r in conn.execute("SELECT DISTINCT document_id FROM chunks")}
    finally:
        conn.close()

    done = {} if redo else _done_signatures()
    plan: list[dict] = []
    for doc_id, raw, name, ext, size, mtime_ns, pages in rows:
        if doc_id in have:
            continue
        rel = _rel(raw)
        if any(part in rel for part in script_library.EXCLUDED_PARTS):
            continue
        #: 会「登记了但没正文」的形态：PDF 扫描件、整页图贴进 Word 的 docx、以及
        #: 本身就是图的单页文档。（实测 docx 那 2 份不是坏文件 —— 正文层 0 字、
        #: 内容在 word/media/* 的几十张图里。）
        suffix = (ext or Path(rel).suffix).lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        signature = f"{size}:{mtime_ns}"
        if done.get(raw) == signature:
            continue
        plan.append({
            "path": raw,
            "rel": rel,
            "name": name or Path(rel).name,
            "extension": suffix,
            "size": int(size or 0),
            "modifiedNs": int(mtime_ns or 0),
            "pages": int(pages or 0),
            "signature": signature,
            "priority": _priority(rel),
        })
    for item in _extra_quarantined_docx():
        if any(part in item["rel"] for part in script_library.EXCLUDED_PARTS):
            continue
        if done.get(item["path"]) == item["signature"]:
            continue
        plan.append(item)
    if only == "partial":
        half = _partial_paths()
        plan = [item for item in plan if item["path"] in half]
    elif only == "done":
        recorded = _recorded_paths()
        plan = [item for item in plan if item["path"] in recorded]
    plan.sort(key=lambda item: (item["priority"], -item["size"], item["rel"]))
    return plan


def _done_signatures() -> dict[str, str]:
    """已**处理过**的文档（``ocr`` 与 ``quarantine``）；``partial`` 不在内。

    收支两条都要算：

    * ``partial`` 是 ``--pages`` 试跑的半成品，不算完成 —— 否则试跑 6 页就把它标成做完，
      正式跑会跳过整本；
    * ``quarantine`` **算完成**：那份已经跑过、判定为"结果不能用"，再 OCR 一遍只会
      用掉同样的几十分钟得出同样结论。要重试必须显式 ``--redo``（这里是唯一出口）。
    """
    conn = _open_sidecar()
    try:
        return {path: sig or "" for path, sig in conn.execute(
            "SELECT path, signature FROM documents WHERE status IN (?, ?)",
            (STATUS_OCR, STATUS_QUARANTINE))}
    finally:
        conn.close()


def _suspect_paths() -> set[str]:
    """已入库、但 ``page_audit`` 里仍有``suspect`` 页的件（主库 ``path`` 集合）。

    为什么需要这个入口：门禁的判据会改（2026-11-04 给「去重比」加了``DISTINCT_CJK_MIN``
    中文门槛 —— 英文剧本页去重字符只有 30–60 个，旧判据把**整本英文剧本**的每一页都标成
    「疑大量重复」，实测 8458 页／368 份）。这些页一个字都没丢（``suspect`` 照常写块），
    但会一直挂在 ``--audit`` 里当噪声，把真问题淹掉。重判只能重跑（页文本没单独存），
    所以给一个「只重跑这些件」的入口：``--run --reaudit``。
    """
    conn = _open_sidecar(writable=False)
    try:
        return {path for (path,) in conn.execute(
            "SELECT DISTINCT d.path FROM page_audit a "
            "JOIN documents d ON d.id = a.document_id WHERE a.verdict = 'suspect'")}
    finally:
        conn.close()


def _keep_suspects(plan: list[dict]) -> list[dict]:
    """把候选收缩到「有存疑页」的件，并如实报出收缩比例。

    ★收缩比例必须打出来：``_pending(redo=True)`` 会把 3000+ 份一起排进来，
    ``--limit`` 默认 20 —— 若收缩后 20 份里一份都没剩下，用户看到的会是
    「跑了 0 份」或「跑了 20 份毫不相干的件」，而真正想重判的那批**一份没动**。
    """
    suspect = _suspect_paths()
    kept = [item for item in plan if item["path"] in suspect]
    print(f"--reaudit：存疑页涉及 {len(suspect)} 份，其中 {len(kept)} 份在候选里"
          f"（候选共 {len(plan)} 份）")
    return kept


#: 等锁上限（秒）。SQLite 默认只等 5 秒 —— 一次普通的长读（备份 1.3 GB 的库、大范围
#: ``LIKE`` 扫描）就能占住共享锁十几秒，5 秒根本不够，写者当场抛 ``database is locked``。
#: 2026-11-04 真实事故：整批 9 小时的活儿跑到 2739/2988，被一次探测性的只读查询顶掉，
#: 整进程退出（已存的 2739 份没丢，但剩下的得重跑）。批量入库是长任务，
#: 让一步等一等，比整批报废划算得多。
SIDECAR_BUSY_MS = 60_000
#: 落库遇锁的重试次数与退避基数（第 n 次等 n×基数秒）。
LOCK_RETRIES = 4
LOCK_BACKOFF = 5.0


def _is_locked(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def _open_sidecar(*, writable: bool = True) -> sqlite3.Connection:
    db = Path(script_library.OCR_DB)
    if writable:
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db, timeout=SIDECAR_BUSY_MS / 1000)
        #: ``timeout=`` 只对「拿到连接后第一条语句」生效；显式 busy_timeout 才能覆盖
        #: 整条连接生命周期里的每一次加锁尝试。
        conn.execute(f"PRAGMA busy_timeout = {SIDECAR_BUSY_MS}")
        conn.executescript(_SCHEMA)
        #: 门禁上线前建的 sidecar 没有 ``gated`` 列。``ALTER TABLE`` 不支持
        #: ``IF NOT EXISTS``，只能先问 pragma。存量行默认 0 ⇒ ``--audit`` 能把
        #: 「这批是旧口径、没走过校对」如实报出来，而不是让它混在已入库里冒充干净。
        columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
        if "gated" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN gated INTEGER DEFAULT 0")
        conn.commit()
        return conn
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


def _stats() -> dict:
    plan = _pending()
    conn = _open_sidecar()
    try:
        done_docs = conn.execute(
            "SELECT count(*) FROM documents WHERE status = 'ocr'").fetchone()[0]
        done_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        #: 只按**整本完成**的页数算均速；partial 那几页会拉低/拉高平均，不可作外推依据
        done_pages = conn.execute(
            "SELECT coalesce(sum(pages), 0) FROM documents WHERE status = 'ocr'"
        ).fetchone()[0]
        quarantined = conn.execute(
            "SELECT count(*) FROM documents WHERE status = ?", (STATUS_QUARANTINE,)
        ).fetchone()[0]
        try:
            audit_rows = conn.execute(
                "SELECT verdict, count(*), count(DISTINCT document_id) FROM page_audit "
                "GROUP BY verdict").fetchall()
        except sqlite3.Error:  #: 老 sidecar 没这张表 ⇒ 报 0，不当作"零问题"
            audit_rows = []
        audit = {row[0]: {"pages": row[1], "documents": row[2]} for row in audit_rows}
        try:
            meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        except sqlite3.Error:
            meta = {}
        #: 均速只能用**墙钟账**（`_record_run` 记的 runSeconds ÷ runPages）：逐份秒数相加
        #: 在并发下等于把同一段时间数了 N 遍，实测把 5 s/页报成 42.3 s/页。
        try:
            spent = float(meta.get("runSeconds") or 0)
            done_scanned = int(float(meta.get("runPages") or 0))
        except ValueError:
            spent, done_scanned = 0.0, 0
    finally:
        conn.close()
    remaining_pages = sum(item["pages"] or 0 for item in plan)
    #: 样本 <200 页时（首份含模型装载、且可能只是一次试跑）退回经验值：经验值**跟着引擎走**
    #: —— 新档实测 0.32 s/页、老档 1.75 s/页，用一个数会相差 5 倍（这就是「报 73 小时其实
    #: 只要 16 小时」的成因）。有实测账就一律用实测，不看这张表。
    fallback = _FALLBACK_SPP.get(engine_name(), 1.5)
    per_page = (spent / done_scanned) if done_scanned >= 200 and spent > 0 else fallback
    return {
        "done": {"documents": done_docs, "chunks": done_chunks, "pages": done_pages,
                 "scannedPages": done_scanned, "seconds": spent},
        "audit": {"quarantined": quarantined, **audit},
        "pending": {"documents": len(plan),
                    "pages": remaining_pages,
                    "bytes": sum(item["size"] for item in plan)},
        "estimate": {
            "secondsPerPage": round(per_page, 2),
            "hoursRemaining": round(remaining_pages * per_page / 3600, 1),
        },
        "db": Path(script_library.OCR_DB).as_posix(),
    }


def _audit_report(limit: int, verdict: str) -> dict:
    """校对报告：**哪些页被拦下了、为什么、原文长什么样**。

    这是整套门禁的可见面 —— 拦截而不给证据，等于把内容悄悄删掉。三条查询各回答一个
    问题：整份隔离的是哪几本（要不要重跑）、被拦的页长什么样（是引擎错了还是书就是
    这样）、以及可疑页集中在什么形态上。
    """
    db = Path(script_library.OCR_DB)
    if not db.exists():
        return {"available": False, "db": db.as_posix(),
                "hint": "还没跑过 OCR（sidecar 不存在）"}
    #: 用写打开：``_SCHEMA`` 是 ``CREATE TABLE IF NOT EXISTS`` ⇒ 顺手给旧 sidecar
    #: 补上 ``page_audit`` 表（那些库建表时还没有这套门禁）。sidecar 是本仓自己的资产。
    conn = _open_sidecar(writable=True)
    try:
        totals = conn.execute(
            "SELECT status, count(*), coalesce(sum(pages), 0) FROM documents "
            "GROUP BY status").fetchall()
        decided = {row[0]: {"documents": row[1], "pages": row[2]} for row in totals}
        verdicts = conn.execute(
            "SELECT verdict, count(*), count(DISTINCT document_id) FROM page_audit "
            "GROUP BY verdict").fetchall()
        by_verdict = {row[0]: {"pages": row[1], "documents": row[2]} for row in verdicts}
        #: 门禁上线**之前**入库的份数：内容可能带乱码，但不该混在"已入库"里冒充干净。
        ungated = conn.execute(
            "SELECT count(*) FROM documents WHERE status = ? AND coalesce(gated, 0) = 0",
            (STATUS_OCR,)).fetchone()[0]
        quarantined = conn.execute(
            "SELECT d.name, d.path, d.pages, "
            "  (SELECT count(*) FROM page_audit a "
            "   WHERE a.document_id = d.id AND a.verdict = 'reject') AS bad "
            "FROM documents d WHERE d.status = ? ORDER BY bad DESC, d.pages DESC LIMIT ?",
            (STATUS_QUARANTINE, limit)).fetchall()
        #: 坏页优先看**字数最少**的（空/半空最容易一眼判断是插图页还是识别崩了），
        #: 同字数下按文档名排，保证同一份的坏页聚在一起。
        rows = conn.execute(
            "SELECT name, page, reasons, chars, preview FROM page_audit "
            "WHERE verdict = ? ORDER BY chars ASC, name, page LIMIT ?",
            (verdict, limit)).fetchall()
        #: 跨页版头剔除要 ≥3 页才有统计意义（``text_clean.RUNNING_HEAD_MIN_PAGES``）。
        #: 短文档的水印**洗不掉**，这里如实报出来 —— 否则审计看起来"零问题"，
        #: 而那些页的水印会一直躺在 fts5 里抬高商号词频（实测 1 页的期刊摘录就是）。
        short_total, short_pages = conn.execute(
            "SELECT count(*), coalesce(sum(pages), 0) FROM documents "
            "WHERE status IN (?, ?) AND pages > 0 AND pages < ?",
            (STATUS_OCR, STATUS_PARTIAL, text_clean.RUNNING_HEAD_MIN_PAGES)).fetchone()
        short_sample = conn.execute(
            "SELECT name, pages FROM documents WHERE status IN (?, ?) AND pages > 0 AND pages < ? "
            "ORDER BY pages, name LIMIT ?",
            (STATUS_OCR, STATUS_PARTIAL, text_clean.RUNNING_HEAD_MIN_PAGES, limit)).fetchall()
    finally:
        conn.close()
    return {
        "available": True,
        "db": db.as_posix(),
        "decided": decided,
        "audit": by_verdict,
        "ungated": ungated,
        "shortDocs": {"documents": short_total, "pages": short_pages,
                      "threshold": text_clean.RUNNING_HEAD_MIN_PAGES,
                      "sample": [{"name": name, "pages": pages}
                                 for name, pages in short_sample]},
        "quarantined": [{"name": name, "path": path, "pages": pages, "rejectedPages": bad}
                        for name, path, pages, bad in quarantined],
        "pages": [{"name": name, "page": page, "reasons": reasons, "chars": chars,
                   "preview": preview} for name, page, reasons, chars, preview in rows],
    }


def _chunks_of_page(text: str, page_no: int, buffer: str, start_page: int) -> tuple[list[tuple], str, int]:
    """把一页文字并入缓冲；够长就切块。返回 ``(块列表, 剩余缓冲, 下一块的起始页)``。

    每个块是 ``(tail_page, start_page, body)`` **两个页号**：``start_page`` 是这块正文
    **开始**的那一页，``tail_page`` 是切出它时正在读的那一页（块跨页时二者不同）。
    落库只许用 ``start_page`` —— 原因见 :func:`_page_chunks` 里的 ★ 注释。
    """
    body = (buffer + "\n" + text).strip() if buffer else text.strip()
    pieces: list[tuple] = []
    while len(body) >= CHUNK_CHARS:
        cut = CHUNK_CHARS
        #: 优先在句末切（与 md 那边同一口径：宁可短一点，不切碎半句话）
        window = body.rfind("。", CHUNK_CHARS - 200, CHUNK_CHARS)
        if window > 0:
            cut = window + 1
        pieces.append((page_no, start_page, body[:cut].strip()))
        body = body[cut:].lstrip()
        start_page = page_no
    return pieces, body, start_page


#: 整份保底的门槛（只在「每一页都判废」时用得上，见 ``_salvageable``）：
#: 一页至少要有这么多**实打实的有效字符**，才值得为了"别丢这本书"而降级放行。
#: 120 字 ≈ 两三行正文：低于它多半是插图页／封面页／残页，留着只是噪声；
#: 而实测要救的那些页（1 页期刊 452 字、总目录页 1291 字有效字）都远超这条线。
#: 另有一条按**占比**的口径（``validRatio ≥ VALID_REJECT``）与它并列 —— 小而干净的页
#: （页题封面之类）占比很高，值得留，不该被绝对字数卡掉。
SALVAGE_MIN_VALID_CHARS = 120
#: 整份文档被判成「不可信」的门槛：reject 页占比超过这个数 ⇒ status=quarantine（不写块）。
#: 单页坏是常态（插图页/表格页），成片坏才说明这一份没识别对 —— 后者入库只会污染检索。
QUARANTINE_REJECT_SHARE = 0.5
#: 页数不足这个数时不判整份（两三页的样本没有统计意义，逐页判定就够）
QUARANTINE_MIN_PAGES = 3
#: 审计表里留的预览长度（人工复核时一眼能看出是哪种坏法）
AUDIT_PREVIEW_CHARS = 80


def _plain_downgrade(verdict: dict) -> dict:
    """纯文本件的判废降级：除「解码失败」外一律降为存疑。

    **纯文本件没有识别失真**：门禁里那些"判废"理由（符号连续段＝目录点线、
    有效字符占比低＝表格页、去重比低＝英文重复字母）在书里本来就长这样，
    判废等于删掉作者真写的字。唯一真正该判废的是**解码失败**留下的非法字符 ——
    那说明四种候选编码全没解对，这套字写进 fts5 就是纯噪声。
    降级不是放过：页仍记进 ``page_audit``（verdict=suspect），复核时看得见。
    """
    if verdict["verdict"] != "reject" \
            or verdict["metrics"]["junkRatio"] > text_clean.FFFD_REJECT:
        return verdict
    return dict(verdict, verdict="suspect",
                reasons=[*verdict["reasons"], "纯文本件不判废（降为存疑）"])


def _salvageable(metrics: dict) -> bool:
    """这一页的「废」是**软性**理由（符号段密集、表格页、短行），还是**硬伤**？

    软性 = 字本身认得对、只是排版被识别成了符号噪声；硬伤 = 乱码（替换符）、
    整块被认成一排符号（那段里的字全没了）、整页只剩几个不同字符。
    两类的处置必须相反：软性可以降为存疑留内容，硬性只能丢 —— 硬文本进 fts5
    就是纯噪声，检索命中一堆读不出意思的片段。

    ★为什么软性的判据是「**实打实的有效字符够多**」而不是「有效字符占比够高」：
    占比类是**比例**判据，而比例低完全可能由页面本来的排版造成 —— 期刊的「总目录」
    就是一堆篇名后面跟一串点线（实测 `26 本刊一九八一年总目录.pdf` 有效字符占比
    47%／去重比 7%），字一个没错，错的只是"这不是正文段落"。按占比判会把这类
    整本判废：全库 175214 页里 13 页，但它是一本书 —— 丢了就再也搜不到它。
    硬伤三条（乱码／整块被认成符号／几乎没有不同字符）才算真不能用。
    """
    if metrics["junkRatio"] > text_clean.FFFD_REJECT \
            or metrics["maxRun"] >= text_clean.JUNK_RUN_MAX_REJECT:
        return False
    #: 只看「有没有实打实的有效正文」两条腿：绝对字数够多（点线目录/表格页/期刊摘录），
    #: 或占比够高（小而干净的页题封面）——任一成立就值得留。
    #: 刻意**不再**拿「去重比」「不同字符数」拦：那两条判的就是"页面重复"，而
    #: 重复页也是书里的真实一页（同一行被识别刷了几十遍的页眉、清版页都可能），
    #: 在一个「整本否则全丢」的处境下，留成 suspect 让检索能碰到它比彻底消失划算。
    valid_chars = int(metrics["chars"] or 0) * float(metrics["validRatio"] or 0.0)
    return valid_chars >= SALVAGE_MIN_VALID_CHARS \
        or metrics["validRatio"] >= text_clean.VALID_REJECT


def _salvage_all_rejected(verdicts: list[dict]) -> list[dict]:
    """**整份保底**：每一页都判废时，把软性判废的页降为存疑，别让整本书凭空消失。

    触发条件刻意写成「没有任何一页带着正文活下来」，而不是「所有页都 reject」——
    空白页的判定是 ``ok``（只有 ``reasons=['空白页']``、``chars=0``），一份
    「1 页封面空白 + 1 页正文判废」的件按后者会**不算触发**，然后整份 0 块，
    症状与前述 1 页期刊完全相同。判据只认「活下来的是不是真有字」。

    硬伤页不动（``_salvageable`` 为假）：那才是门禁真正要拦的东西，
    宁可整份隔离让人复核，也不把乱码塞进索引。
    """
    if not verdicts:
        return verdicts
    survivors = [v for v in verdicts
                 if v["verdict"] != "reject" and int(v["metrics"]["chars"] or 0) > 0]
    if survivors:
        return verdicts
    keep = [_salvageable(v["metrics"]) for v in verdicts]
    if not any(keep):
        return verdicts
    return [dict(v, verdict="suspect",
                 reasons=[*v["reasons"],
                          "整份每页都判废，本页为软性理由 ⇒ 降为存疑保正文"])
            if kept else v
            for v, kept in zip(verdicts, keep)]


def _page_chunks(pages_text: list[tuple[int, str]],
                 notes: dict[int, str] | None = None, *, plain: bool = False) -> dict:
    """**清洗 → 逐页定性 → 只把过得去的页切成块**。四条输入形态共用这一段。

    ``plain=True`` 用于**纯文本件**（``.txt``/``.md``）：跳过跨页版头剔除与「整页重排」，
    只做规整 + 碎行合并。这两步是为 OCR 出的碎行与印刷页眉设计的：

    * 页眉剔除按「多页首尾行重复」判 —— 纯文本没有页眉，切伪页后首尾行是正常正文，
      被当成页眉剔掉就是**删内容**；
    * 整页重排按「平均行长 < 16 字」触发 —— 中文剧本对白平均行长本就十几字，
      一触发就把两行台词串成一大段（检索还能命中，但读起来已经变形）。

    判废口径也跟着放宽一档：文本件没有识别失真，**只有解码失败**（非法字符占比超
    ``text_clean.FFFD_REJECT``）才真判废，其余"坏法"降级为存疑（见循环里的注释）。

    ``pages_text`` 是 ``[(页码, 原始文本)]``；``notes`` 是「升级/分歧」的留痕
    （``{页码: 说明}``，来自 ``ocr_vlm.Escalator``）。返回::

        {"chunks": [(startPage, no, text)], "audit": [{page, verdict, reasons, chars, preview}],
         "droppedHeads": [...], "rejected": n, "suspect": n}

    ``startPage`` 是这块正文**开始**的页（不是收尾页）：``chunks.page`` 是「这段从第几页
    起」，检索跳页据此翻原文；块本身可能跨页（``CHUNK_CHARS`` 800 字 ≈ 1–2 页），
    故 ``page`` 序列在 ``chunk_no`` 顺序下**必然非降**，出现下降即口径错了。

    三道处理各自解决一个独立问题，不能合并：

    1. ``clean_document``：碎行合并（79% 的文字层页有）、跨页页眉页脚剔除（期刊每页都有）；
       —— ``plain=True`` 时退成逐页 ``clean_page(reflow=False)``（见上）；
    2. ``judge_page``：给每页一个 ``ok|suspect|reject`` —— **reject 页一个块都不写**，
       改记进 ``page_audit`` 等人工复核（这是「不搞出一堆乱码」的硬闸口）；
       ★整份都判废时先过一遍 ``_salvage_all_rejected``：软性判废降为存疑留正文，
       硬伤才真丢。判据是「有没有一页带着正文活下来」，不是「是不是每页都标了
       reject」—— 空白页（``ok`` + 0 字）会让后者漏掉「封面空白 + 正文判废」的件；
    3. 切块：只在留下来的页上做，避免坏页把 800 字的窗口撑破。
    """
    if plain:
        #: 两个"便宜但会误伤"的步骤都不做，见 docstring；乱码仍由 ``judge_page`` 兜底。
        cleaned = {"pages": [text_clean.clean_page(text, reflow=False)
                            for _no, text in pages_text],
                   "droppedHeads": []}
    else:
        cleaned = text_clean.clean_document([text for _no, text in pages_text])
    bodies = cleaned["pages"]
    chunks: list[tuple] = []
    audit: list[dict] = []
    buffer = ""
    buffer_page = 0
    rejected = suspect = 0
    rejected_chars = 0
    #: 先逐页定性、**后**切块：整份保底（``_salvage_all_rejected``）必须在切块之前决定，
    #: 否则「每页都判废」的件会一块都不剩。2026-11-04 实测：1 份 1 页期刊（574 字，
    #: 正文清清楚楚）因为「符号连续段占 20%」整页被丢，文档仍记 ``status=ocr``、
    #: 0 块 ⇒ 库里查不到这本书，``--stats`` 还报「已 OCR」。静默丢内容是这里最坏的形态。
    verdicts = [text_clean.judge_page(text) for text in bodies]
    verdicts = ([_plain_downgrade(v) for v in verdicts] if plain
                else _salvage_all_rejected(verdicts))
    for (page_no, _raw), text, verdict in zip(pages_text, bodies, verdicts):
        label = verdict["verdict"]
        if label == "reject":
            rejected += 1
            rejected_chars += int(verdict["metrics"]["chars"] or 0)
        elif label == "suspect":
            suspect += 1
        #: 只在**有问题**时记审计表：全绿的书 300 页记 300 行没有价值，
        #: 而「库里看不到这页」本身就得有据可查（否则下次会以为是漏识别）。
        note = (notes or {}).get(page_no) or ""
        if label != "ok" or not text.strip():
            reasons = "；".join(verdict["reasons"]) or "清洗后为空"
            if note:
                #: 升级过的页必须带上「原本谁判的、后来怎么判的」，否则复核时看不出
                #: 这一页是引擎救回来的还是本来就好
                reasons = f"{note}｜{reasons}"
            audit.append({
                "page": page_no, "verdict": label,
                "reasons": reasons,
                "chars": verdict["metrics"]["chars"],
                "preview": text.strip()[:AUDIT_PREVIEW_CHARS],
            })
        elif note:
            #: 页最终可用，但两遍引擎说法不一致（或 VLM 把它救回来了）：单独记一类
            #: ``escalated``，**不混进 reject/suspect 的统计**（否则「剔除页数」会把
            #: 好页也数进去），又能用 ``--audit --verdict=escalated`` 单独翻出来
            audit.append({
                "page": page_no, "verdict": "escalated",
                "reasons": note,
                "chars": verdict["metrics"]["chars"],
                "preview": text.strip()[:AUDIT_PREVIEW_CHARS],
            })
        if label == "reject" or not text.strip():
            continue
        if not buffer:
            buffer_page = page_no
        pieces, buffer, buffer_page = _chunks_of_page(text, page_no, buffer, buffer_page)
        for _tail, start, body in pieces:
            #: ★``page`` 记**块起始页**，不是块收尾页（2026-11-04 第五处修正）。
            #: 检索命中后用户要做的是「翻原文」：从块首页往下读一定能读到命中处，从块尾页
            #: 打开则往往要往回翻一页 —— 而 ``chunks`` 表只有一列，只能选一边。
            #: 原先这里落的是 ``_tail``（收尾页），与同一函数下方尾部缓冲落 ``buffer_page``
            #: （起始页）**口径相反**：同一份书里只有最后一块记首、其余全记尾，于是
            #: 「按页跳」会整体后移约一页，实测 920 页里 25 页根本没有尾页落在自己身上的块。
            chunks.append((start, len(chunks) + 1, body))
    if len(buffer.strip()) >= MIN_PAGE_CHARS:
        chunks.append((buffer_page, len(chunks) + 1, buffer.strip()))
    elif buffer.strip():
        #: 尾巴太短（不足 MIN_PAGE_CHARS）⇒ 记一条，别让最后几行凭空消失而没人知道
        audit.append({"page": buffer_page, "verdict": "suspect",
                      "reasons": f"末页剩余 {len(buffer.strip())} 字，未达单页下限",
                      "chars": len(buffer.strip()),
                      "preview": buffer.strip()[:AUDIT_PREVIEW_CHARS]})
    return {"chunks": chunks, "audit": audit, "droppedHeads": cleaned["droppedHeads"],
            "rejected": rejected, "suspect": suspect, "judged": len(pages_text),
            #: 被判废页上的字数合计：``_status_of`` 据此把「丢掉了有字的页、自己一块没剩」
            #: 的件标成隔离 —— 否则它顶着 ``status=ocr`` 在库里当一本查不到的隐形空书。
            "rejectedChars": rejected_chars}


#: 版面分栏重排：默认开。对单栏页是**恒等变换**（见 ``ocr_layout`` 模块头），
#: 所以没有「关掉更安全」的理由；``--no-layout`` / ``OCR_NO_LAYOUT=1`` 仅用于对照实验。
_LAYOUT_ENABLED = not bool(os.environ.get("OCR_NO_LAYOUT"))
#: 可疑页升级器：每个进程一个。未配置 VLM 端点时为 None ⇒ 整条通道零开销。
_ESCALATOR: ocr_vlm.Escalator | None = None
#: 升级器**不是**线程安全的（内部有预算计数与 HTTP 会话），而识别现在是并发的：
#: 把 ``consider`` 串起来即可。它只在「可疑页」触发（占比很小），串行不影响吞吐。
_ESCALATE_LOCK = threading.Lock()


def _page_text(image, ocr, *, page_no: int = 0) -> tuple[str, str | None]:
    """一页图 → ``(文本, 留痕说明)``。**PDF 与 docx 两条路共用这一个出口。**

    三步各自可单独失效，任一步不可用都不影响另一步：

    1. 本地 OCR（原有行为，不动）；
    2. 版面分栏重排（``ocr_layout``，纯几何；单栏页原样返回）；
    3. 可疑页交 VLM 重识别（``ocr_vlm``；默认未配置 ⇒ 直接跳过）。

    为什么重排必须在这一层而不是切块之后：切块按字数和句号走，一旦左右栏已经交替
    拼进同一个字符串，事后无法再拆开（顺序信息只存在于检测框里）。
    """
    result, _ = ocr(image)
    lines = result or []
    breaks: set[int] = set()
    if _LAYOUT_ENABLED and lines:
        width = getattr(image, "shape", (None, None))[1]
        try:
            report = ocr_layout.analyze(lines, width)
            if report["order"]:
                lines = [lines[index] for index in report["order"]]
                breaks = report["breaks"]
        except Exception:  # noqa: BLE001 —— 版面判断失败不能拖垮整份（退回按 y 的原序）
            lines, breaks = result or [], set()
    parts: list[str] = []
    for position, line in enumerate(lines):
        if position and position in breaks:
            #: 段间/栏间补一个硬换行：左栏末行常不以句号结尾，直接粘连会造出
            #: 「…他转身走进右栏第一句…」这种半句，门禁也救不回来
            parts.append("\n")
        parts.append(ocr_layout.text_of(line))
    text = "".join(parts).strip()
    if _ESCALATOR is None:
        return text, None
    with _ESCALATE_LOCK:
        return _ESCALATOR.consider(page_no, image, text)


def _digest_pages(raw: list[tuple[int, str, str | None]]) -> tuple[list[tuple[int, str]], dict[int, str], int]:
    """把并发识别收到的乱序 ``[(页号, 文本, 留痕)]`` 收成 ``(pagesText, notes, blank)``。

    **排序是必需步骤、不是优化**：并发完成顺序与页序无关，而切块要按页序看全文档
    （页眉页脚剔除、碎行合并都靠跨页统计），一旦顺序错乱就会把页眉当成正文首行剔除掉。
    清点 ``blank`` 也在这里做：短页（``< MIN_PAGE_CHARS``）不入 ``pages_text``，
    与串行版的口径逐字一致。
    """
    pages_text: list[tuple[int, str]] = []
    notes: dict[int, str] = {}
    blank = 0
    for page_no, text, note in sorted(raw, key=lambda row: row[0]):
        if note:
            notes[page_no] = note
        if len(text) < MIN_PAGE_CHARS:
            blank += 1
            continue
        pages_text.append((page_no, text))
    return pages_text, notes, blank


def _pdf_open(local: Path):
    """打开一份 PDF（``pypdfium2`` ／ PDFium）。**调用方负责 close**（见 :func:`_pdf_pages`）。

    为什么要这个薄封装：``ocr_index`` 原先按需 ``import fitz``（PyMuPDF，AGPL-3.0），
    而第三库的 PDF 文字层（``lib_index._pdf_text``）走的是 ``pypdfium2``（BSD-3／Apache-2.0）。
    同一个仓里为「读 PDF」养两套绑定，唯一的理由才成立 —— 而这里不需要：取文字层、
    渲染成图、数页数，PDFium 三件都做得了，于是把 AGPL 那个包整条摘掉。
    """
    import pypdfium2 as pdfium  # noqa: PLC0415 —— 只有真跑时才需要

    return pdfium.PdfDocument(str(local))


def _pdf_page_text(page) -> str:
    """取一页**自己的**文字层；取不到（坏页／纯图页）返回 ""，不抛。"""
    try:
        return page.get_textpage().get_text_range().strip()
    except Exception:  # noqa: BLE001 —— 个别坏页不拖垮整份
        return ""


def _pdf_page_image(page, dpi: int):
    """把一页渲染成 ``(H, W, 3)`` uint8 **RGB** 数组（OCR 引擎吃的是这个）。

    ⚠️ 两条都踩过才敢写死：
    ① ``rev_byteorder=True`` 不能省 —— PDFium 的原生字节序是 **BGR**，不翻过来喂给
       OCR 等于把红蓝对调（黑字白底看不出差别，彩色扫描件的检测框会偏）；
    ② ``dpi`` 换算成 ``scale = dpi / 72``（PDF 的用户空间单位是 1/72 英寸）。
    ``.copy()`` 也是必须的：``to_numpy()`` 是 bitmap 缓冲区的**视图**，bitmap 一回收，
    这条跨线程返回的数组就是野指针。
    """
    bitmap = page.render(scale=dpi / 72, rev_byteorder=True)
    return bitmap.to_numpy().copy()


def _render_page(local: Path, index: int, dpi: int) -> tuple[str, object]:
    """在工作线程里把第 ``index`` 页准备成「可识别的输入」。

    返回 ``("native", 文本)`` 或 ``("image", ndarray)``。**「文字层够厚就直接用」也放在这里**：
    它虽是 1 ms 级的小活，但排在 OCR 前面时会把 GPU 的等待时间拉长（一页也不放过）。

    ⚠️ 每个任务**自己 open/close** 一份 PDF，不缓存句柄：PDFium 的 ``PdfDocument`` 不保证
    跨线程并发安全（两个线程同时取页会崩），而线程本地缓存又会在长批次里累积几百个句柄、
    且主线程无法安全回收别的线程的对象。open 只解析 xref（实测 ≤ 数毫秒），
    相对一页渲染/识别是 1% 量级的开销，换「零泄漏」值得。
    """
    with contextlib.closing(_pdf_open(local)) as doc:
        page = doc[index]
        #: ★先问 PDF 自己的文字层（约 1 ms/页）：供应商说「没抽出正文」不等于**每页**
        #: 都没文字层 —— 混合文档（正文页可复制、插图页是扫描）和供应商解析失败都
        #: 会落到待办里。但只有文字层**足够厚**（``NATIVE_TRUST_CHARS``）才敢用它，
        #: 否则「水印型扫描件」的整页正文会被几十字页眉顶掉（2026-11-04 实测 168 份）。
        text = _pdf_page_text(page)
        if len(text) >= NATIVE_TRUST_CHARS:
            return "native", text
        image = _pdf_page_image(page, dpi)
    return "image", image


#: 一份 PDF 页数 ≥ 这个值才允许被切成多段并行（小文件切段的 open/汇总开销抵不上收益）
MIN_SPLIT_PAGES = 120
#: 单段不少于这么多页。段切碎了两头都亏：每段要重新 open 一次 PDF、子进程里重建一次
#: 引擎会话，且「升级预算按文档复位」会被各段各拿一份（见 ``_ocr_item`` 的注释）。
MIN_SEG_PAGES = 40


def _ocr_one(item: dict, ocr, *, dpi: int, max_pages: int | None,
             page_range: tuple[int, int] | None = None) -> dict:
    """OCR 一份 PDF，返回 ``{pages, chunks: [(page, chunk_no, text)], seconds}``。

    ★**渲染与识别流水**（2026-11-04）：原先一页一页「渲染 → 识别 → 渲染」，GPU 每页都要
    空等一次页面渲染（实测 GPU 利用率在 25–82% 之间抖、CPU 只占 20%）。现在由
    ``render_threads()`` 条线程按窗口预取后续页，主线程连续喂 GPU，两条腿同时走；
    线程总额仍受 ``cpu_room()`` 约束（见 :func:`worker_threads`）。

    ★**页级切分**（``page_range`` 非空时）：只做本段页的「渲染 + 识别」，**不切块**，
    把原始 ``pagesText``/``notes`` 交回主进程，由 :func:`_merge_segments` 合并后再统一
    走 ``_page_chunks``。跨页的版头页脚统计、碎行合并必须看全文档（单页看不出
    「·232·」是版头还是正文），所以清洗**只允许**在主进程做一次 —— 这是切分不伤质量的前提。
    """
    local = _local(item["path"])
    if local is None or not local.exists():
        raise FileNotFoundError(item["rel"])
    started = time.time()
    doc = _pdf_open(local)
    pool = None
    try:
        total = len(doc)
        limit = min(total, max_pages) if max_pages else total
        lo, hi = page_range if page_range is not None else (0, limit)
        hi = min(hi, limit)
        pages_text: list[tuple[int, str]] = []
        notes: dict[int, str] = {}
        blank = 0
        native = 0
        render = render_threads(max(1, int(os.environ.get("OCR_WORKERS") or 1)))
        #: 预取窗口 = 渲染线程数的两倍：够填满流水，又不至于把几百页的图全压在内存里
        #: （150 dpi 一页 ≈ 6.5 MB，6 × 6.5 ≈ 39 MB/进程）。
        prefetch = max(2, render * 2)
        pool = ThreadPoolExecutor(max_workers=render, thread_name_prefix="render")
        #: ★**并发识别**（2026-11-04 第六处修正）：渲染并行之后瓶颈换到 GPU 队列 —— 一页的
        #: det/rec 是一串浅请求，串行时一次只有一个在飞，GPU 做完一批就空着等下一批
        #: （实测 4 进程时 GPU 只忙 53%、CPU 20%）。这里把识别也投进池子，队列深度 ＝
        #: 页线程数×2；同一引擎实例被多线程重入是安全的（ORT 的 session.run 线程安全、
        #: 预处理只读），实测 0.761→0.263 s/页 且逐页行数一致。
        count = page_threads(max(1, int(os.environ.get("OCR_WORKERS") or 1)))
        ocr_pool = (ThreadPoolExecutor(max_workers=count, thread_name_prefix="ocr")
                    if count > 1 else None)
        inflight = max(1, count * 2)
        pending: dict[int, object] = {}
        jobs: list[tuple[int, object]] = []
        raw: list[tuple[int, str, str | None]] = []
        queued = lo
        try:
            for index in range(lo, hi):
                while queued < hi and len(pending) < prefetch:
                    pending[queued] = pool.submit(_render_page, local, queued, dpi)
                    queued += 1
                kind, payload = pending.pop(index).result()
                page_no = index + 1
                if kind == "native":
                    native += 1
                    raw.append((page_no, payload, None))
                    continue
                if ocr_pool is None:
                    text, note = _page_text(payload, ocr, page_no=page_no)
                    raw.append((page_no, text, note))
                    continue
                jobs.append((page_no, ocr_pool.submit(_page_text, payload, ocr, page_no=page_no)))
                #: 在飞太多就先收最早交的那页：队列始终有深度，又不把整本的图压在内存里
                while len(jobs) >= inflight:
                    number, job = jobs.pop(0)
                    text, note = job.result()
                    raw.append((number, text, note))
            for number, job in jobs:
                text, note = job.result()
                raw.append((number, text, note))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            pool = None
            if ocr_pool is not None:
                ocr_pool.shutdown(wait=True, cancel_futures=True)
                ocr_pool = None
        pages_text, notes, blank = _digest_pages(raw)
        if page_range is not None:
            #: 半成品：清洗与落库都归主进程（见 docstring 末段），这里只交原始页文本。
            return {"pages": total, "scanned": hi - lo, "blank": blank, "native": native,
                    "pagesText": pages_text, "notes": notes,
                    "escalated": len(notes),
                    "seconds": round(time.time() - started, 1)}
        #: ⚠️ 切块必须**在整份页文本收齐之后**：页眉页脚剔除、碎行合并都需要看全文档
        #: （单页看不出「·232·」是版头还是正文）。
        packed = _page_chunks(pages_text, notes)
        #: ⚠️ ``judged`` 必须透传：``_status_of`` 靠它判「整份隔离」，漏了就等于
        #: 隔离永远不触发（``int(None or 0) >= 3`` 恒假），坏书照旧入库。
        return {"pages": total, "scanned": limit, "blank": blank, "native": native,
                "chunks": packed["chunks"], "audit": packed["audit"],
                "droppedHeads": packed["droppedHeads"],
                "rejected": packed["rejected"], "suspect": packed["suspect"],
                "judged": packed["judged"], "escalated": len(notes),
                "seconds": round(time.time() - started, 1)}
    finally:
        doc.close()


def _merge_segments(parts: list[dict]) -> dict:
    """把同一份文件的若干段半成品合成一份完整 ``outcome``（**清洗在这里做唯一一次**）。

    ``seconds`` 取各段**最大值**而不是相加：段是并行跑的，墙钟≈最慢那一段；
    这样 ``report`` 里算出的 s/页是「并行后的有效速度」，正好用来乘剩余页数。
    """
    pages_text: list[tuple[int, str]] = []
    notes: dict[int, str] = {}
    for part in parts:
        pages_text.extend(part["pagesText"])
        notes.update(part["notes"])
    pages_text.sort(key=lambda pair: pair[0])
    packed = _page_chunks(pages_text, notes)
    return {"pages": parts[0]["pages"],
            "scanned": sum(int(p["scanned"]) for p in parts),
            "blank": sum(int(p["blank"]) for p in parts),
            "native": sum(int(p["native"]) for p in parts),
            "chunks": packed["chunks"], "audit": packed["audit"],
            "droppedHeads": packed["droppedHeads"],
            "rejected": packed["rejected"], "suspect": packed["suspect"],
            "judged": packed["judged"],
            "escalated": sum(int(p["escalated"]) for p in parts),
            "segments": len(parts),
            "seconds": max(float(p["seconds"]) for p in parts)}


#: docx 内侧：正文里引用图片的顺序（``<a:blip r:embed="rId7">``）＋包内关系表
_DOCX_BLIP = re.compile(r'<a:blip[^>]*r:embed="(rId\d+)"')
_DOCX_REL = re.compile(r'<Relationship[^>]*\bId="(rId\d+)"[^>]*\bTarget="([^"]+)"')
#: 小于这个边长的图是页眉徽标/项目符号（实测 44 张图里混着几张），不当整页扫描件
MIN_IMAGE_PX = 240

#: 独立图片文件后缀：一份图 = 一份单页文档。多帧 TIFF 按帧数算页（扫描仪批量输出
#: 常见形态）。``.gif`` 不收：动图在语料里只会是表情包，不是页。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

#: 纯文本后缀。主库登记了 80 份（``.txt`` 77／``.md`` 3，2026-11-04 实测）。
#:
#: ⚠️ 先纠正一个曾被我误判成「最大缺口」的事实：这 80 份在主库里**早就有正文块**
#: （6731 + 3 块，中文完全正确），所以它们**不是**缺口，`_pending` 也不会列它们。
#: 真正的问题是白名单一直把这条形态整个漏掉：万一某份的表里块丢了（重建、迁移、
#: 供应商再发版），`_pending` 会把它列出来 —— 而分派器没有这条路，结果要么崩，
#: 要么被静默跳过，跟"从来没发现过"一样。补上这条路，是把静默跳过换成能自己补上的出口。
#:
#: 附带好处：主库对 ``.txt`` 的切块很粗（``.md`` 3 份只有 3 块 ⇒ 一份一整块；
#: ``.txt`` 一份 11 块装 24671 字），``--redo`` 时走这条路会按 800 字重切、检索粒度更细。
TEXT_SUFFIXES = (".txt", ".md")

#: 能进 OCR 队列的扩展名。PDF/DOCX 是主库现有 5323 份的主体形态；图片是为「一份
#: 扫描件被拆成散图」这种情况留的口子；纯文本件只解码不 OCR（见 :func:`_ocr_text`）。
#: 主库登记了才算待办，不额外扫盘。
SUPPORTED_SUFFIXES = (".pdf", ".docx", *IMAGE_SUFFIXES, *TEXT_SUFFIXES)


def _docx_images(local: Path) -> list[bytes]:
    """按**正文出现顺序**取出 docx 内嵌的图片（整页扫描件贴进 Word 的形态）。

    不用 python-docx：docx 就是个 zip，`word/document.xml` 里 ``<a:blip r:embed="rId7">``
    的顺序就是阅读顺序，经 `word/_rels/document.xml.rels` 换成 `word/media/*` 即可 ——
    少一个重量依赖，也不受 Word 版本写法的干扰。
    """
    with zipfile.ZipFile(local) as zf:
        names = zf.namelist()
        try:
            rels_xml = zf.read("word/_rels/document.xml.rels").decode("utf-8", "replace")
        except KeyError:
            rels_xml = ""
        rels: dict[str, str] = {}
        for rid, target in _DOCX_REL.findall(rels_xml):
            #: Target 可能是绝对路径（``/word/media/x.png``）或相对 word/ 的（``media/x.png``），
            #: 还可能是外链（``http://…``）—— 后者在包里找不到，下面按 names 兜掉。
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(
                posixpath.join("word", target))
            rels[rid] = path
        try:
            body = zf.read("word/document.xml").decode("utf-8", "replace")
        except KeyError:
            return []
        order = [rels[rid] for rid in _DOCX_BLIP.findall(body) if rid in rels]
        if not order:
            #: 兜底：rels 缺失时按 media 名排序（会丢阅读顺序，但比一张都不出好）
            order = sorted(name for name in names if name.startswith("word/media/"))
        present = set(names)
        return [zf.read(name) for name in order if name in present]


def _ocr_docx(item: dict, ocr, *, max_pages: int | None) -> dict:
    """OCR 一份**整页图贴进 Word** 的 docx（正文层 0 字，图才是内容）。"""
    import numpy as np  # noqa: PLC0415 —— 只有真跑时才需要
    from PIL import Image  # noqa: PLC0415

    local = _local(item["path"])
    if local is None or not local.exists():
        raise FileNotFoundError(item["rel"])
    started = time.time()
    blobs = _docx_images(local)
    keep: list[int] = []
    for index, blob in enumerate(blobs):
        try:
            with Image.open(io.BytesIO(blob)) as probe:
                width, height = probe.size
        except Exception:  # noqa: BLE001 —— 坏图（emf/wmf）跳过，不拖垮整份
            continue
        if min(width, height) >= MIN_IMAGE_PX:
            keep.append(index)
    total = len(keep)
    limit = min(total, max_pages) if max_pages else total
    pages_text: list[tuple[int, str]] = []
    notes: dict[int, str] = {}
    blank = 0
    for page_no, blob_index in enumerate(keep[:limit], start=1):
        with Image.open(io.BytesIO(blobs[blob_index])) as page:
            image = np.array(page.convert("RGB"))
        text, note = _page_text(image, ocr, page_no=page_no)
        if note:
            notes[page_no] = note
        if len(text) < MIN_PAGE_CHARS:
            blank += 1
            continue
        pages_text.append((page_no, text))
    packed = _page_chunks(pages_text, notes)
    #: docx 这一半没有「原生文字层」可借（正文层 0 字正是它进待办的原因）⇒ native 恒 0
    #: ``judged`` 同 PDF 路：不透传 ⇒ 「整份隔离」对 docx 永不触发。
    return {"pages": total, "scanned": limit, "blank": blank, "native": 0,
            "chunks": packed["chunks"], "audit": packed["audit"],
            "droppedHeads": packed["droppedHeads"],
            "rejected": packed["rejected"], "suspect": packed["suspect"],
            "judged": packed["judged"], "escalated": len(notes),
            "seconds": round(time.time() - started, 1)}


def _ocr_image(item: dict, ocr, *, max_pages: int | None) -> dict:
    """OCR 一份**图片文档**（单页图 / 多帧 TIFF），返回与 PDF 同形的 outcome。

    为什么不复用 ``_ocr_docx``：docx 得先从 zip 里按 ``word/media`` 顺序捞图、再按
    ``MIN_IMAGE_PX`` 滤掉徽标，两层都不是图片文件需要的。真正该复用的只有
    ``_page_text`` —— 本地 OCR、版面重排、门禁、切块、VLM 升级全在那一个出口，
    所以这条路只负责「把帧变成 RGB 数组喂进去」。

    帧是**边读边处理**的：一份几百帧的 TIFF 全转 RGB 再统一处理会白占几个 GB。
    """
    import numpy as np  # noqa: PLC0415 —— 只有真跑时才需要
    from PIL import Image, ImageSequence  # noqa: PLC0415

    local = _local(item["path"])
    if local is None or not local.exists():
        raise FileNotFoundError(item["rel"])
    started = time.time()
    pages_text: list[tuple[int, str]] = []
    notes: dict[int, str] = {}
    blank = 0
    scanned = 0
    with Image.open(local) as source:
        #: ``n_frames`` 只有 TIFF/GIF/APNG 才 >1；普通 PNG/JPG 取到的是 1
        try:
            total = max(1, int(getattr(source, "n_frames", 1) or 1))
        except Exception:  # noqa: BLE001 —— 个别畸形 TIFF 的 n_frames 会抛
            total = 1
        limit = min(total, max_pages) if max_pages else total
        for index, frame in enumerate(ImageSequence.Iterator(source)):
            if index >= limit:
                break
            page_no = index + 1
            scanned += 1
            image = np.array(frame.convert("RGB"))
            text, note = _page_text(image, ocr, page_no=page_no)
            if note:
                notes[page_no] = note
            if len(text) < MIN_PAGE_CHARS:
                blank += 1
                continue
            pages_text.append((page_no, text))
    packed = _page_chunks(pages_text, notes)
    #: 图片没有文字层可借（正文就在像素里）⇒ native 恒 0，与 docx 同一条理由。
    #: ``scanned`` 用实数而不是 ``limit``：畸形 TIFF 声明的帧数可能多于真能读出的。
    return {"pages": total, "scanned": scanned, "blank": blank, "native": 0,
            "chunks": packed["chunks"], "audit": packed["audit"],
            "droppedHeads": packed["droppedHeads"],
            "rejected": packed["rejected"], "suspect": packed["suspect"],
            "judged": packed["judged"], "escalated": len(notes),
            "seconds": round(time.time() - started, 1)}


#: 纯文本解码的候选编码，**按顺序试、按可疑度挑**（见 :func:`_decode_text`）。
#: 主库 80 份实测：``utf-8-sig`` 48／``gb18030`` 30／``big5`` 0，另有 2 份
#: （``GODFATHR.TXT``／``sex_lies_and_videotape.txt``）前三种全部抛
#: ``UnicodeDecodeError`` —— 它们正文几乎纯 ASCII，只有 1–3 个高位字节
#: （``0xE9``/``0xC9`` 即 é/É），在 UTF-8/gb18030/big5 里都是非法序列，``cp1252`` 才是正解。
TEXT_ENCODINGS = ("utf-8-sig", "gb18030", "big5", "cp1252")

#: 纯文本按多少字切「伪页」。纯文本没有页，这里按 **1500 字** 切（中文书一页 700–900 字、
#: 一屏约 1500 字）：只决定 ``chunks.page`` 的定位粒度与 ``--stats`` 的页数口径，
#: **不改变块内容**（块仍按 ``CHUNK_CHARS`` 800 字、优先句末切）。
TEXT_PAGE_CHARS = 1500


def _text_suspicion(text: str) -> tuple[int, float]:
    """解码候选「像不像正常文本」的打分，越小越可信（配合候选顺序当二级判据）。

    两个独立指纹，都靠字面统计、不需要外部词表：

    * ``mixed``＝1：ASCII 字母数 > 4×汉字数，**却还有**一把汉字 —— 英文文件被当
      GBK/Big5 解的特征（每对高位字节凑一个汉字，剩下的仍是英文）。真实中文语料里
      ASCII 字母远到不了汉字的 4 倍（就几个专名/术语），英文语料里汉字恒 0，分得开；
    * ``moji``：U+00A0–U+00FF 占比 —— GBK/Big5 字节被当 cp1252/latin-1 解的特征
      （「四 不是半夜里闹鬼么？」会变成 ``ËÄ ²»ÊÇ°ëÒ¹ÀïÄÖ¹íÃ´£¿``，每个汉字变两个
      Latin-1 字符，整段高位字节占比直接过半）。
    """
    sample = text[:200000]
    if not sample:
        return (1, 1.0)
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    moji = sum(1 for ch in sample if "\u00a0" <= ch <= "\u00ff") / len(sample)
    mixed = 1 if (cjk and latin > cjk * 4) else 0
    return (mixed, round(moji, 4))


def _decode_text(data: bytes) -> tuple[str, str]:
    """纯文本字节 → ``(正文, 编码名)``。**绝对不能 ``Path.read_text()`` 了事**。

    按 UTF-8 硬读 GBK 文本时，非法字节会被替换符吃掉而**不报错**（或者抛
    ``UnicodeDecodeError`` 让整份失败），两种情况都会静默丢正文 —— 而 ``judge_page``
    只判"这页像不像乱码"，对"整段被吃掉"一个字都看不出来。

    判据分两层，顺序不能换：

    1. 有 BOM 的 UTF-16 直接用（无 BOM 绝不试：会把英文书解成 74% 汉字的乱码）；
    2. ``utf-8-sig`` 解得出就**直接采信** —— 严格 UTF-8 自带校验，长文本误命中概率可忽略，
       再拿去和 gb18030 比分会反被"假候选"顶掉（一份带弯引号的 UTF-8 英文书，gb18030
       也能解，且解出来的乱码一个字都不落在 U+00A0–U+00FF 里）；
    3. 其余候选都解不出 UTF-8 时按 :func:`_text_suspicion` 打分取最小，同分看顺序。
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    best: tuple[tuple[int, float, int], str, str] | None = None
    for order, encoding in enumerate(TEXT_ENCODINGS):
        try:
            text = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if encoding == "utf-8-sig":
            return text, encoding
        key = (*_text_suspicion(text), order)
        if best is None or key < best[0]:
            best = (key, text, encoding)
    if best is None:
        #: 四种全废（cp1252 几乎不会失败，走到这里说明字节里有孤立的控制段）⇒
        #: 宁可留几个替换符让 ``judge_page`` 判页，也不整份丢掉
        return data.decode("utf-8", "replace"), "utf-8/replace"
    return best[1], best[2]


def _split_text_pages(text: str, *, size: int = TEXT_PAGE_CHARS) -> list[str]:
    """把纯文本按行累积成**伪页**，只在行边界切（不切碎句子/台词）。

    纯文本没有页：伪页只为让 ``chunks.page`` 有个能用的定位值（检索命中后"这段在第几页"）。
    """
    pages: list[str] = []
    buffer: list[str] = []
    chars = 0
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        buffer.append(line)
        chars += len(line) + 1
        if chars >= size:
            pages.append("\n".join(buffer).strip())
            buffer, chars = [], 0
    tail = "\n".join(buffer).strip()
    if tail:
        pages.append(tail)
    return pages


def _ocr_text(item: dict, *, max_pages: int | None = None) -> dict:
    """**纯文本**（``.txt``/``.md``）解码入库，返回与其余三条路同形的 outcome。

    这条路不起 OCR 引擎：文字本来就在文件里，唯一的失真点是编码。三件事必须自己做：

    1. **编码判定**（:func:`_decode_text`）—— 见那里的说明；
    2. **伪页**（:func:`_split_text_pages`）—— 纯文本没有页，按 1500 字切行边界；
    3. **不做跨页版头剔除与整页重排**（``plain=True``）—— 那两步是为 OCR 碎行与印刷
       页眉写的，对作者排好的对白会把两行台词串成一段、甚至把正文首行当页眉剔掉。

    ``native`` 记全部页：这些页一个像素都没 OCR 过，统计上必须与"扫描件"分开看。
    ``encoding`` 留在 outcome 里，报告尾行会打出来 —— 解错编码是这条路唯一会犯的错，
    必须一眼能看见用的是哪个候选。
    """
    local = _local(item["path"])
    if local is None or not local.exists():
        raise FileNotFoundError(item["rel"])
    started = time.time()
    text, encoding = _decode_text(local.read_bytes())
    pages = _split_text_pages(text)
    limit = min(len(pages), max_pages) if max_pages else len(pages)
    pages_text: list[tuple[int, str]] = []
    blank = 0
    for page_no, body in enumerate(pages[:limit], start=1):
        if len(body) < MIN_PAGE_CHARS:
            blank += 1
            continue
        pages_text.append((page_no, body))
    packed = _page_chunks(pages_text, None, plain=True)
    return {"pages": len(pages), "scanned": limit, "blank": blank, "native": limit,
            "encoding": encoding, "chars": len(text),
            "chunks": packed["chunks"], "audit": packed["audit"],
            "droppedHeads": packed["droppedHeads"],
            "rejected": packed["rejected"], "suspect": packed["suspect"],
            "judged": packed["judged"], "escalated": 0,
            "seconds": round(time.time() - started, 1)}


def _needs_engine(todo: list[dict]) -> bool:
    """这批里有没有**需要 OCR** 的件（纯文本件只解码）。

    整批全是文本时连进程池都不该开：起引擎要建 onnx 会话、加载权重（几秒 + 一份显存），
    多进程还要各建一次 —— 而解码是纯 CPU 的顺手活，一个进程顺序做最快。
    """
    return any((item.get("extension") or Path(item["rel"]).suffix).lower()
               not in TEXT_SUFFIXES for item in todo)


def _ocr_item(item: dict, ocr, *, dpi: int, max_pages: int | None,
              page_range: tuple[int, int] | None = None) -> dict:
    """按扩展名分派四种输入形态，出口统一收敛到 ``_page_chunks``。

    * ``.pdf`` —— 渲染整页（有文字层就直接白拿，不花 OCR）；
    * ``.docx`` —— 取内嵌整页图（正文层 0 字的那批）；
    * 图片 —— 文件本身就是页（多帧 TIFF 按帧算页）；
    * ``.txt``/``.md`` —— 解码即可，不碰引擎（见 :func:`_ocr_text`）。

    ``page_range`` 只对 PDF 有意义（其余形态本来就小到不值得切，见 :func:`plan_segments`）。
    ⚠️ 切段后**升级预算**是各段各拿一份（``reset()`` 按"文档"复位）：4 段就是 4 份预算。
    这是有意的 —— 坏页在大书里是局部现象，按段给预算比"整本共用一个池、被最前面几页吃掉"更合理。
    """
    if _ESCALATOR is not None:
        #: 升级预算按**文档**复位：一本 500 页的书里坏页再多，也只升级最前面几页，
        #: 防止一份烂书把整批时间吃光
        _ESCALATOR.reset()
    suffix = (item.get("extension") or Path(item["rel"]).suffix).lower()
    if suffix in TEXT_SUFFIXES:
        return _ocr_text(item, max_pages=max_pages)
    if suffix == ".docx":
        return _ocr_docx(item, ocr, max_pages=max_pages)
    if suffix in IMAGE_SUFFIXES:
        return _ocr_image(item, ocr, max_pages=max_pages)
    return _ocr_one(item, ocr, dpi=dpi, max_pages=max_pages, page_range=page_range)


STATUS_OCR = "ocr"
#: ``--pages`` 试跑只做了前 N 页 ⇒ 不算整本已 OCR（否则正式跑会跳过整本）
STATUS_PARTIAL = "partial"
#: 整份被质量门禁判为不可信 ⇒ 不写块，等人工复核。
#: ⚠️ 它与 partial 必须分开：partial 是"没跑完"，quarantine 是"跑完了但结果不能要"。
#: 两者都不计入 `--stats` 的"已 OCR"，但只有 quarantine 会留下审计证据。
STATUS_QUARANTINE = "quarantine"


def _status_of(outcome: dict) -> str:
    """一份文档该记什么状态。三档判据都只看 ``outcome`` 里的计数，不需要打开文件。"""
    if outcome["scanned"] < outcome["pages"]:
        return STATUS_PARTIAL
    judged = int(outcome.get("judged") or 0)
    rejected = int(outcome.get("rejected") or 0)
    if judged >= QUARANTINE_MIN_PAGES and rejected / judged >= QUARANTINE_REJECT_SHARE:
        return STATUS_QUARANTINE
    #: 一块都没留下、而丢掉的页上**有字**（``rejectedChars``）⇒ 隔离。
    #: 这类件多半是 1–2 页的期刊/封面本，天然过不了上面那条「≥3 页才判整份」的门槛，
    #: 结果顶着 ``status=ocr`` 在库里当一本查不到任何东西的"隐形空书"：``--stats``
    #: 报「已 OCR」，检索页页 0 命中，没有任何一处能把人引到它身上（2026-11-04 实测
    #: 1 份）。隔离至少让 ``--audit --verdict=reject`` 与 ``quarantined`` 列表能看见它。
    if not outcome.get("chunks") and int(outcome.get("rejectedChars") or 0) >= MIN_PAGE_CHARS:
        return STATUS_QUARANTINE
    return STATUS_OCR


#: 「源文件结构性损坏」的判据：这类错误重跑一万次也是同样的结果，必须记下来。
#: 反例（**不得**记）：显存不足、磁盘 IO、句柄占用 —— 那些下一次可能就成功。
_BROKEN_HINTS = (
    "invalid number of pages",
    "cannot open",
    "failed to open",
    "format error",
    "no objects found",
    "not a pdf",
)


def _looks_broken(exc: Exception) -> bool:
    """这份文件是不是**结构损坏**（重跑也没用的那种）。

    实测案例：`雕刻时光》.(苏)安德烈·塔可夫斯基.扫描版.pdf` 393216 字节（与主库登记
    完全一致 ⇒ 源头就坏，不是传输截断）、文件头是 `%PDF-1.4`、MuPDF 报
    ``Invalid number of pages``。这类文件每轮都出现、每轮都白报一次错。
    """
    if type(exc).__name__ in ("FileDataError", "EmptyFileError"):
        return True
    message = str(exc).lower()
    return any(hint in message for hint in _BROKEN_HINTS)


def _mark_broken(item: dict, exc: Exception) -> None:
    """把打不开的源文件记成 ``quarantine``（算「已处理」），免得每轮重试。

    **不写块、也不写 ``page_audit``**：这不是「识别质量差」，是「源文件本身坏了」。
    混进 reject 统计会让人以为引擎在那份上出了错，把真问题埋掉。原因打在日志里。
    """
    conn = _open_sidecar()
    try:
        with conn:
            conn.execute(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE path = ?)", [item["path"]])
            conn.execute("DELETE FROM page_audit WHERE path = ?", [item["path"]])
            conn.execute("DELETE FROM documents WHERE path = ?", [item["path"]])
            conn.execute(
                "INSERT INTO documents (path, name, extension, size, modified_ns, "
                "signature, pages, status, gated) VALUES (?,?,?,?,?,?,?,?,1)",
                (item["path"], item["name"], item.get("extension") or ".pdf",
                 item["size"], item["modifiedNs"], item["signature"],
                 int(item.get("pages") or 0), STATUS_QUARANTINE),
            )
    finally:
        conn.close()
    print(f"  ↳ 源文件损坏（{type(exc).__name__}: {exc}）⇒ 记 quarantine，下轮跳过；"
          f"换到好文件后用 --redo 重跑", flush=True)


def _save(item: dict, outcome: dict, *, dpi: int, engine: str) -> None:
    status = _status_of(outcome)
    conn = _open_sidecar()
    try:
        with conn:
            conn.execute("DELETE FROM chunks WHERE document_id IN "
                         "(SELECT id FROM documents WHERE path = ?)", [item["path"]])
            conn.execute("DELETE FROM page_audit WHERE path = ?", [item["path"]])
            conn.execute("DELETE FROM documents WHERE path = ?", [item["path"]])
            cursor = conn.execute(
                "INSERT INTO documents (path, name, extension, size, modified_ns, "
                "signature, pages, status, gated) VALUES (?,?,?,?,?,?,?,?,1)",
                (item["path"], item["name"], item.get("extension") or ".pdf",
                 item["size"], item["modifiedNs"],
                 item["signature"], outcome["pages"], status),
            )
            doc_id = cursor.lastrowid
            #: 隔离的文档**一个块都不写**（判断见 ``_status_of``）：坏文本进了 fts5，
            #: 检索时无法区分「书里就这么写」与「引擎认错了」。审计行照记，人工复核后再
            #: 用 ``--redo --path-substr`` 重跑。
            chunks = () if status == STATUS_QUARANTINE else outcome["chunks"]
            conn.executemany(
                "INSERT INTO chunks (document_id, path, name, page, chunk_no, text) "
                "VALUES (?,?,?,?,?,?)",
                [(doc_id, item["path"], item["name"], page, no, text)
                 for page, no, text in chunks],
            )
            conn.executemany(
                "INSERT INTO page_audit (document_id, path, name, page, verdict, reasons, "
                "chars, preview) VALUES (?,?,?,?,?,?,?,?)",
                [(doc_id, item["path"], item["name"], row["page"], row["verdict"],
                  row["reasons"], row["chars"], row["preview"])
                 for row in outcome.get("audit") or ()],
            )
            conn.execute("INSERT INTO meta (k, v) VALUES ('engine', ?) "
                         "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (engine,))
            conn.execute("INSERT INTO meta (k, v) VALUES ('dpi', ?) "
                         "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (str(dpi),))
    finally:
        conn.close()


def _save_atomic(item: dict, outcome: dict, *, dpi: int, engine: str,
                 retries: int = LOCK_RETRIES, backoff: float = LOCK_BACKOFF,
                 notify=print) -> None:
    """落库遇到写锁就退让重试；非锁错误立刻抛（那才是真 bug）。

    为什么要重试而不是加长超时了事：``busy_timeout`` 只管**单次**加锁等待，而这份文档要
    连做 DELETE×3 ＋ INSERT ＋ INSERT×2，中间任何一步被读者插队都可能失败，重试是从
    「一份文档」的粒度上重来，语义干净。重试到上限仍失败 ⇒ 抛出去让上游记成本份失败，
    这份在 sidecar 里没有登记，下一轮自动重来（不会静默丢）。
    """
    for attempt in range(1, retries + 1):
        try:
            _save(item, outcome, dpi=dpi, engine=engine)
            return
        except sqlite3.OperationalError as exc:
            if not _is_locked(exc) or attempt >= retries:
                raise
            wait = backoff * attempt
            notify(f"  … 写库被占用（{exc}），{wait:.0f}s 后重试（第 {attempt}/{retries - 1} 次）")
            time.sleep(wait)


def _record_run(seconds: float, pages: int) -> None:
    """记一次**整批**的墙钟账：``runSeconds`` 累计墙钟、``runPages`` 累计页数。

    为什么不能用 ``_save`` 里逐份的 ``outcome["seconds"]`` 相加：那是**每个子进程自己的**
    墙钟，8 进程并发时相加＝把同一段时间数了 8 遍 ⇒ 均速凭空 ×8（实测把 5 s/页报成
    42.3 s/页、剩余时间报成 2058 小时）。只有主进程量的 ``runSeconds`` 才是真实产出速度。
    """
    conn = _open_sidecar()
    try:
        meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        try:
            spent = float(meta.get("runSeconds") or 0)
            done = int(float(meta.get("runPages") or 0))
        except ValueError:
            spent, done = 0.0, 0
        conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('runSeconds', ?)",
                     (str(round(spent + max(0.0, seconds), 1)),))
        conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('runPages', ?)",
                     (str(done + max(0, pages)),))
    finally:
        conn.close()


_OCR = None
_WORKER_DPI = DEFAULT_DPI


#: 每个进程最多开几条渲染线程：再多就只是在 PDFium 里排队（一页渲染本身就是单线程的，
#: 并行度来自「同时渲染几页」），同时也限制预取窗口占用的内存（每页 ≈ 4–6 MB）。
MAX_RENDER_THREADS = 6
#: GPU 档每进程的线程配额：2 条 ORT（CPU 侧只做解码/缩放/后处理）+ 其余给渲染预取。
#: ★进程数由**显存**实测封顶，不是拍脑袋：2026-11-04 在 A5000（22.49 GiB 可用、空载桌面
#: 占 1.6 GiB）上跑到 5 个 CUDA 会话时 `memory.used` 直接顶到 22489 MiB、GPU 利用率 100%
#: 而每页反从 0.5 s 涨到 0.88 s —— 会话之间抢显存比抢算力更致命（≈4.2 GiB/会话）。
#: 故 4 进程（≈16.7 GiB）是这张卡的上限，**加进程提吞吐到此为止**，再往上只能靠进程内流水。
GPU_ORT_THREADS = 2
GPU_THREADS_PER_WORKER = 7
GPU_MAX_WORKERS = 4

#: 每个 CUDA 会话实测驻留（MiB）：det/rec/cls 三个 session + ORT 的 CUDA 上下文。
#: 4 个会话实测 ≈16.7 GiB（见上），故取 4200 —— 这是「再开一个会话会不会顶爆」的唯一判据。
GPU_SESSION_MIB = 4200
#: GPU 档每进程**并发识别**几页（同一引擎实例被多线程重入）。
#: ★这是 2026-11-04 第六处修正：进程数已被显存封在 4，加不上去，而实测 4 进程时
#: GPU 只忙 53%、CPU 20% —— 每页的 det/rec 是一串**浅请求**，一次只有一个在飞，
#: GPU 做完就空着等 CPU 拷下一批。同一实例 4 线程并发实测 0.263 s/页 vs 串行 0.761 s/页
#: （2.9×，逐页行数完全一致 ⇒ RapidOCR 可重入：ORT 的 ``session.run`` 线程安全，
#: 预处理只读）。默认 2：4 进程 × 2 = 8 个请求在飞已够把队列填深，再大只是互相排队。
GPU_PAGE_THREADS = 2
#: 探测到的空闲显存里给**别人**留的余量（MiB）。卡上常驻桌面/浏览器/偶发 LLM，
#: 按「空闲 显存全用光」分配 ⇒ 别人的下一次分配就 OOM，双输。
GPU_HEADROOM_MIB = 1024
#: 外部 CPU 占用超过这个比例就认为「机器已经有别的重活」，打提示并收缩预算。
CPU_BUSY_HINT = 0.5


def _quiet_run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """跑一条**只读**探测命令；失败一律返回 ``None``（探测不到就当不知道，不许猜）。

    两个必须做的事：``CREATE_NO_WINDOW``（否则每次 nvidia-smi 闪一下黑窗口）、
    ``timeout``（驱动挂死时 nvidia-smi 会卡住几十秒，把整批启动拖住）。
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def gpu_memory_mib() -> tuple[int, int] | None:
    """``(总显存, 空闲显存)``，单位 MiB；拿不到返回 ``None``。

    ⚠️ 判会话数只用 **free**，绝不用 total：这张卡上常驻着别的东西（桌面 1.6 GiB、
    偶发的大模型），按 total 算会在别人起来后撞满 22489 MiB —— 实测 5 会话时每页
    从 0.5 s 劣化到 0.88 s，就是这个坑。
    """
    line = _quiet_run(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                       "--format=csv,noheader,nounits"])
    if not line:
        return None
    parts = [p.strip() for p in line.splitlines()[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def gpu_util_percent() -> int | None:
    """GPU 当前利用率（0–100）；拿不到返回 ``None``（只用于观测与提示，不参与决策）。"""
    line = _quiet_run(["nvidia-smi", "--query-gpu=utilization.gpu",
                       "--format=csv,noheader,nounits"])
    if not line:
        return None
    try:
        return max(0, min(100, int(float(line.splitlines()[0].strip()))))
    except ValueError:
        return None


def cpu_busy_share(sample_seconds: float = 0.25) -> float | None:
    """全机 CPU 当前占用率（0–1）；没装 psutil 返回 ``None``。

    ``interval`` 让 psutil 自己算差值：传 ``None`` 时首次调用恒返回 0.0（历史遗留语义），
    拿它当「机器空着」会让预算算满 —— 这正是不能偷懒的地方。
    """
    try:
        import psutil  # noqa: PLC0415 —— 只有真跑批时才需要
    except Exception:  # noqa: BLE001 —— 没装就降级成「不知道」，不影响主流程
        return None
    try:
        return max(0.0, min(1.0, psutil.cpu_percent(interval=sample_seconds) / 100.0))
    except Exception:  # noqa: BLE001
        return None


def gpu_session_cap(mem: tuple[int, int] | None = None) -> tuple[int, str]:
    """这张卡**现在**还能开几个 OCR 会话，以及一句人话依据。

    ``0`` ＝ 一个都开不了（卡被别人占着），调用方必须回落 CPU 档或提示用户等待。
    """
    if mem is None:
        return GPU_MAX_WORKERS, f"nvidia-smi 不可用，按实测上限 {GPU_MAX_WORKERS} 会话"
    total, free = mem
    room = free - GPU_HEADROOM_MIB
    if room < GPU_SESSION_MIB:
        return 0, (f"空闲显存仅 {free} MiB（总 {total}，已被占 {total - free}），"
                   f"开一个会话要 {GPU_SESSION_MIB}+{GPU_HEADROOM_MIB} MiB")
    sessions = max(1, min(GPU_MAX_WORKERS, room // GPU_SESSION_MIB))
    return sessions, (f"空闲显存 {free}/{total} MiB ⇒ {sessions} 会话"
                      f"（每会话 {GPU_SESSION_MIB}，留 {GPU_HEADROOM_MIB} 给别的进程）")


def cpu_budget() -> int:
    """这一批**真正**能用的线程总数（已扣除外部占用）。

    主进程在跑批前用 :func:`apply_snapshot` 把结果落进 ``OCR_CPU_BUDGET``，子进程直接读：
    采样要 0.25 s×N，每个 worker 各采一遍既慢、又会让各进程算出**不同**的预算
    （渲染线程数就会漂），而线程数是唯一必须全局一致的东西。
    """
    raw = os.environ.get("OCR_CPU_BUDGET")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return cpu_room()


def resource_snapshot(*, sample_cpu: bool = True) -> dict:
    """跑批**之前**采一次机器现状 —— 本批开几个进程、每进程几条线程，全由这一份决定。"""
    device = engine_device()
    usable = cpu_room()
    busy = cpu_busy_share() if sample_cpu else None
    if busy is None:
        budget = usable
        cpu_note = "未装 psutil，CPU 按硬件余量满配"
    else:
        idle = int(round(_CORES * (1.0 - busy)))
        budget = max(1, min(usable, idle))
        cpu_note = (f"全机占用 {busy * 100:.0f}% ⇒ 可用 ≈{idle}/{_CORES} 条逻辑线程"
                    f"（硬件余量上限 {usable}）")
    mem = gpu_memory_mib() if device == ENGINE_DEVICE_CUDA else None
    sessions, gpu_note = (gpu_session_cap(mem) if device == ENGINE_DEVICE_CUDA
                          else (0, "设备=cpu，不用显存"))
    return {
        "device": device,
        "cores": _CORES,
        "cpuReserve": max(0, _CORES - usable),
        "cpuBusy": busy,
        "cpuBudget": budget,
        "cpuNote": cpu_note,
        "gpuMem": mem,
        "gpuSessions": sessions,
        "gpuNote": gpu_note,
    }


def apply_snapshot(snap: dict) -> None:
    """把快照落到环境变量，并把「本批怎么配的、为什么这么配」直接打出来。

    环境变量是**唯一**能穿透 ``spawn`` 进程池的口径（子进程重新 import 本模块，
    拿不到主进程的任何对象），与 ``OCR_WORKERS`` 同一套机制。
    """
    os.environ["OCR_CPU_BUDGET"] = str(max(1, int(snap["cpuBudget"])))
    print(f"资源：CPU 预算 {snap['cpuBudget']}/{snap['cores']} 条线程（{snap['cpuNote']}）"
          f"｜GPU {snap['gpuNote']}", flush=True)
    warnings: list[str] = []
    busy = snap.get("cpuBusy")
    if busy is not None and busy >= CPU_BUSY_HINT:
        warnings.append(f"CPU 已被别的任务占了 {busy * 100:.0f}%，本批只按剩余算线程数"
                        f"（要独占可先停掉它们，或用 --workers=1 更稳）")
    if snap["device"] == ENGINE_DEVICE_CUDA:
        if snap["gpuSessions"] <= 0:
            pinned = (os.environ.get("OCR_DEVICE") or "").strip().lower() == ENGINE_DEVICE_CUDA
            if pinned:
                #: 用户明文要 cuda ⇒ 尊重，但要让他知道会撞墙（下面表头仍会如实报预算）
                warnings.append("你显式要了 cuda，但空闲显存不够开一个会话 —— "
                                "仍按 cuda 试，若 OOM 请设 OCR_DEVICE=cpu 或等占卡任务结束")
            else:
                #: 自动档才允许自行降级：**改写 OCR_DEVICE** 是唯一能让
                #: engine_device()／worker_threads()／子进程同时按 CPU 档配线程的做法
                os.environ["OCR_DEVICE"] = ENGINE_DEVICE_CPU
                warnings.append("空闲显存不够开一个 OCR 会话 ⇒ **自动回落 CPU 档**"
                                "（慢 3–5 倍）。等占卡的任务结束再跑 GPU 版。")
        elif snap["gpuMem"] and snap["gpuSessions"] < GPU_MAX_WORKERS:
            total, free = snap["gpuMem"]
            warnings.append(f"显存被占 {total - free} MiB，只能开 {snap['gpuSessions']} 个会话"
                            f"（上限 {GPU_MAX_WORKERS}），吞吐会低于满配")
    for line in warnings:
        print(f"⚠️  {line}", flush=True)


def worker_threads(workers: int) -> tuple[int, int]:
    """把每个进程的线程配额切成 ``(ORT 线程, 渲染线程)``。

    ★这是本脚本最贵的一处。rapidocr 默认 ``intra_op_num_threads=-1``（＝吃满全部核），
    而 onnxruntime 的 intra-op 线程池由**它自己**的 session 配置决定、**不看 ``OMP_NUM_THREADS``**：
    8 个进程各抢 32 线程 = 256 线程挤 32 核 ⇒ 实测退化成 21–28 s/页（比单进程慢 15 倍），
    还会把均速账带歪。所以线程数必须在建引擎时显式给下去。

    **不许超硬件线程数**：``os.cpu_count()`` 是逻辑线程数（本机 32 ＝ 16 核 × 2 超线程）。
    原式 ``cores // workers`` 给出 8 进程 × 4 线程 ＝ 32，恰好等于**全部**逻辑线程，
    而每进程除 ORT 线程池还有 PDFium 渲染线程、主进程与系统本身 —— 实测 32 个逻辑 CPU
    被顶满、机器失去响应，渲染被抢到 OCR 反而更慢。故预算先减 ``OCR_CPU_RESERVE``。

    ★2026-11-04 第二处修正：**不要再把渲染线程排除在配额之外**。原先 GPU 档写死「ORT 2 条、
    渲染线程算白送」，结果 4 进程 × 2 = 8 条线程跑在 32 条逻辑线程上 —— 实测 CPU 总占用仅
    ~20%、GPU 利用率在 25–82% 之间空转（GPU 在等渲染）。现在渲染线程也算进配额：
    ``workers × (ort + render) ≤ cpu_room()``，既不超硬件线程数，也不让 CPU 闲置。

    ★2026-11-04 第三处修正：分母换成 :func:`cpu_budget`（已扣除**本机其他任务**的占用）。
    原先按硬件满配算，机器上还有活时（实测外部占 60%）我们仍按 30 条配线程 ⇒ 双方互相抢、
    谁都跑不快。预算收缩后进程数与线程数一起降，宁可慢一点也不把别人的活拖死。

    ``OCR_THREADS``（ORT）／``OCR_RENDER_THREADS``（渲染）可手改做对照。
    """
    workers = max(1, int(workers))
    share = max(1, cpu_budget() // workers)
    try:
        fixed_ort = int(os.environ.get("OCR_THREADS") or 0)
    except ValueError:
        fixed_ort = 0
    if fixed_ort > 0:
        #: 手钉 ORT 线程时行为与改动前一致（对照实验用），渲染线程另行计算
        ort = fixed_ort
    elif engine_device() == ENGINE_DEVICE_CUDA:
        #: GPU 档下 CPU 侧的 ORT 只做解码、缩放与后处理 ⇒ 1–2 条足够，
        #: 剩下的配额全给渲染（那才是 GPU 空转时该干的事）。
        #: ★显存受限时（会话数 < 进程上限）富余配额**不派发**：会话里的模型算子全在 GPU，
        #: 加厚 ORT 只会和渲染线程抢 GIL，没有实测收益 —— 这种情况由 apply_snapshot 的告警
        #: 明说「会话开不满、吞吐低于满配」，而不是把线程数填满来假装用上了 CPU。
        ort = max(1, min(GPU_ORT_THREADS, share))
    else:
        #: 单进程封顶 16 线程：ORT 在这种输入尺寸上到 8–16 线程就吃满内存带宽，
        #: 再往上加只会互相等锁（实测 32 线程不比 8 快）
        ort = max(1, min(16, share - 1 if share > 1 else 1))

    try:
        fixed_render = int(os.environ.get("OCR_RENDER_THREADS") or 0)
    except ValueError:
        fixed_render = 0
    if fixed_render > 0:
        render = fixed_render
    else:
        #: 配额里除 ORT 之外的归渲染；至少 1 条（=1 时是「主线程识别 + 1 条线程渲染」的
        #: 最简流水，仍能让 GPU 等渲染的时间被吃掉）
        render = max(1, min(MAX_RENDER_THREADS, share - ort))
    return ort, render


def engine_threads(workers: int) -> int:
    """每个 OCR 进程该开几个 onnxruntime 线程（:func:`worker_threads` 的前半）。"""
    return worker_threads(workers)[0]


def render_threads(workers: int) -> int:
    """每个 OCR 进程开几条渲染预取线程（:func:`worker_threads` 的后半）。"""
    return worker_threads(workers)[1]


def page_threads(workers: int = 1) -> int:
    """每个进程**并发识别**几页（``1`` ＝ 退回串行）。

    GPU 档默认 :data:`GPU_PAGE_THREADS`：一页的 det/rec 是一串浅请求，串行时 GPU 做完
    一批就空着等下一批，实测只忙 53%；多页并发把队列填深才有吞吐。它们**共享同一个
    CUDA 会话**，不额外占显存（这才是「进程数被显存封顶后」唯一还能加吞吐的地方）。

    CPU 档默认 1：那时 ORT 已经拿了 16 条线程，再多页并发只是让两条战线互抢核 ——
    实测 CPU 档 16 线程 0.870 s/页，本来就不是瓶颈端。显式给了 ``OCR_PAGE_THREADS``
    就以它为准（调参优先于人话约定），非法值退回默认、不让一个环境变量把整批跑废。
    """
    raw = os.environ.get("OCR_PAGE_THREADS")
    if raw:
        try:
            return max(1, min(8, int(raw)))
        except ValueError:
            pass  #: 写错就退回默认，不让一个环境变量把整批跑废
    if engine_device() != ENGINE_DEVICE_CUDA:
        return 1
    return GPU_PAGE_THREADS


def default_workers(snap: dict | None = None) -> int:
    """默认进程数：GPU 档按**当前可用显存**与线程预算定，CPU 档每进程摊 4 条。

    ``cpu_room()`` 已减掉保留量，所以「进程数 × 每进程线程数」必然 ≤ 预算，
    「不超硬件线程数」由这条链保证 —— 调用点不要另外再乘一遍。

    GPU 档显式封顶 ``GPU_MAX_WORKERS``：加进程能提吞吐，但每个进程一个 CUDA 会话
    （实测驻留 ≈4.2 GiB），4 个约 16.7 GiB／22.49 GiB 可用；第 5 个就顶到 22489 MiB、
    每页反从 0.44 s 劣化到 0.88 s（会话之间抢显存比抢算力更致命）。

    ★2026-11-04：进程数不再只看硬件常量，先看 :func:`gpu_session_cap` 报的**实时**
    空闲显存 —— 别人占着卡时少开会话，一个会话都开不起就回落 CPU 档（并在
    :func:`apply_snapshot` 里把原因打给用户），绝不硬上到 OOM。
    """
    snap = snap or resource_snapshot(sample_cpu=False)
    budget = max(1, int(snap["cpuBudget"]))
    if snap["device"] == ENGINE_DEVICE_CUDA:
        sessions = int(snap["gpuSessions"])
        if sessions <= 0:
            return max(1, min(8, budget // 4))
        return max(1, min(sessions, GPU_MAX_WORKERS, budget // GPU_THREADS_PER_WORKER))
    return max(1, min(8, budget // 4))


def _pdf_pages(item: dict) -> int | None:
    """PDF 的**真实**页数；非 PDF 或打不开返回 None。

    ⚠️ 不用主库登记的 ``item["pages"]``：那个字段来自供应商解析器的报告，实测有缺有错。
    切段用的是「第几页到第几页」，报小了会**静默丢页**（段没覆盖到），宁可当场数一遍。
    ``PdfDocument`` 只解析 xref，毫秒级；且只在「待办份数 < 进程数」时才数（最多几份）。
    """
    suffix = (item.get("extension") or Path(item["rel"]).suffix).lower()
    if suffix != ".pdf":
        return None
    local = _local(item["path"])
    if local is None or not local.exists():
        return None
    try:
        with contextlib.closing(_pdf_open(local)) as doc:
            return len(doc)
    except Exception:  # noqa: BLE001 —— 数不出页数就当不可切
        return None


def _page_ranges(pages: int, k: int) -> list[tuple[int, int]]:
    """把 ``pages`` 页均分成 ``k`` 段 ``[lo, hi)``（余数补到前几段）。"""
    k = max(1, min(int(k), max(1, pages)))
    base, extra = divmod(pages, k)
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append((start, start + size))
        start += size
    return out


def plan_segments(items: list[dict], workers: int,
                  page_counts: dict[str, int | None] | None = None,
                  ) -> list[tuple[dict, tuple[int, int] | None]]:
    """把待办排成 ``(文件, 页区间)`` 任务表。``None`` 表示「整份一次做完」。

    ★**为什么需要页级切分**（2026-11-04 第三处修正）：进程池原先是按**文件**分发的。
    一批 3135 份时没问题（进程永远有活干），但真跑到最后往往只剩一两份超大书 ——
    实测「只剩一份 920 页的《纪录电影文献》」时，4 个进程里 3 个空转，GPU 掉到 18%、
    CPU 只占 ~22%（单进程 7 条线程跑在 32 逻辑线程上），等于把并发能力白扔了。
    页级切分让同一份书的第 1–230／231–460／… 页同时跑，**清洗仍只在主进程做一次**
    （见 :func:`_merge_segments`），所以不影响跨页版头剔除与碎行合并的质量。

    策略是「**只切大文件、切到够用为止**」：文档份数 ≥ 进程数时一段不切（按文件分发已经吃满）；
    否则每次给「当前单段页数最多」的文件加一段，直到任务数补齐进程数，或没有文件还能切
    （``MIN_SPLIT_PAGES`` 门槛 + 单段 ``MIN_SEG_PAGES`` 下限）。
    """
    if workers <= 1 or len(items) >= workers:
        return [(item, None) for item in items]
    counts: dict[int, int | None] = {}
    for index, item in enumerate(items):
        if page_counts is not None:
            counts[index] = page_counts.get(item["path"])
        else:
            counts[index] = _pdf_pages(item)
    segs = [1] * len(items)
    total = len(items)
    while total < workers:
        best, best_load = -1, 0.0
        for index in range(len(items)):
            pages = counts[index]
            if not pages or pages < MIN_SPLIT_PAGES:
                continue
            #: 再切一段后单段页数会掉到下限以下 ⇒ 这一段就别切了
            if pages // (segs[index] + 1) < MIN_SEG_PAGES:
                continue
            load = pages / segs[index]
            if load > best_load:
                best, best_load = index, load
        if best < 0:
            break
        segs[best] += 1
        total += 1
    tasks: list[tuple[dict, tuple[int, int] | None]] = []
    for index, item in enumerate(items):
        pages = counts[index]
        if segs[index] <= 1 or not pages:
            tasks.append((item, None))
            continue
        tasks.extend((item, rng) for rng in _page_ranges(pages, segs[index]))
    return tasks


#: 引擎档位：``auto``＝装得上新档就用新档；另两个是显式钉死（对照与回退用）
ENGINE_AUTO = "auto"
ENGINE_NEW = "rapidocr"              #: rapidocr 3.x，模型是 PP-OCRv6（small 档）
ENGINE_OLD = "rapidocr-onnxruntime"  #: rapidocr 1.4.x，模型是 PP-OCRv4
#: 没有实测账时用的经验值（s/页）：新老档差 5 倍，不能合成一个数
_FALLBACK_SPP = {ENGINE_NEW: 0.5, ENGINE_OLD: 1.5}


def _rec_batch() -> int:
    try:
        return max(1, int(os.environ.get("OCR_REC_BATCH") or 16))
    except ValueError:
        return 16


def engine_name() -> str:
    """这次会用哪一档：只回答**装得上没有**，不建 session（`--stats` 之类也敢调）。"""
    want = os.environ.get("OCR_ENGINE") or ENGINE_AUTO
    if want == ENGINE_OLD:
        return ENGINE_OLD
    try:
        found = importlib.util.find_spec("rapidocr") is not None
    except (ImportError, ValueError):
        found = False
    return ENGINE_NEW if found else ENGINE_OLD


#: 推理设备：``auto``＝这套 onnxruntime 真能上 CUDA 就用；``cuda``＝必须上，上不了当场拒；
#: ``cpu``＝钉死。默认 ``auto``，因为本仓的机器有的有卡、有的没有。
ENGINE_DEVICE_AUTO = "auto"
ENGINE_DEVICE_CUDA = "cuda"
ENGINE_DEVICE_CPU = "cpu"


class DeviceUnavailable(RuntimeError):
    """显式要 CUDA 但这套 onnxruntime 给不出来 —— 不许静默退回 CPU 再报「跑完了」。"""


@functools.lru_cache(maxsize=1)
def ort_providers() -> tuple[str, ...]:
    """这套 onnxruntime **自己说**能提供哪些 provider。

    ``pip install onnxruntime``（CPU 版）与 ``onnxruntime-gpu`` 是两个包、同一个 import 名，
    所以「我设了 ``use_cuda=True``」根本不是证据 —— 必须先问它自己有没有
    ``CUDAExecutionProvider``。rapidocr 3.x 的 ``ProviderConfig.check_available()`` 拿不到时
    **只往它自己的 logger 打一行警告**然后照常跑 CPU，日志被重定向到文件时根本看不见。
    """
    try:
        import onnxruntime  # noqa: PLC0415 —— 只有真要建引擎时才需要
    except ImportError:
        return ()
    try:
        return tuple(onnxruntime.get_available_providers())
    except Exception:  # pragma: no cover —— 坏安装不该把整个批次判死
        return ()


_PREPARED_DLLS = False
_DLL_DIR: str | None = None


def prepare_cuda_dlls() -> str | None:
    """把 CUDA 13 / cuDNN 9 的 DLL 目录挂进本进程，返回挂上的目录（没有就 None）。

    ★这是「装了 onnxruntime-gpu 却仍跑 CPU」的真因：ORT 的 provider DLL
    （``onnxruntime_providers_cuda.dll``）**在会话创建那一刻**才 LoadLibrary，它要
    ``cublasLt64_13.dll``／``cudnn64_9.dll``；缺了只会往 stderr 吐一行然后**静默回落**
    ``CPUExecutionProvider``（本机实测正是这句「depends on cublasLt64_13.dll which is missing」）。

    本仓的 torch 是 cu130 轮子，``site-packages/torch/lib`` 里就带着这两个 DLL，
    所以不需要另装 CUDA Toolkit —— 但要显式 ``os.add_dll_directory`` 才进搜索路径，
    且**每个子进程都得自己挂一遍**（spawn 不继承 DLL 目录，只继承环境变量）。
    """
    global _PREPARED_DLLS
    if _PREPARED_DLLS:
        return _DLL_DIR
    _PREPARED_DLLS = True
    lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    try:
        if (lib / "cublasLt64_13.dll").is_file():
            os.add_dll_directory(str(lib))
            #: PATH 也要补：provider DLL 的**依赖**由系统加载器按 PATH 找，
            #: 只挂 add_dll_directory 时实测仍报 cublasLt64_13.dll missing。
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            globals()["_DLL_DIR"] = str(lib)
            return str(lib)
    except (OSError, AttributeError):  # pragma: no cover —— 老 Python 没有 add_dll_directory
        pass
    globals()["_DLL_DIR"] = None
    return None


def cuda_ready() -> bool:
    """CUDA 端到端可用吗：provider 在 **且** ``get_device()`` 是 GPU **且** DLL 找得到。

    CPU 版轮子的 ``get_available_providers()`` 里根本没有 CUDA 那一条，``get_device()``
    返回 ``"CPU"`` —— 两道都判，是因为见过「列着 CUDA 但建会话时回落」的装法。
    """
    if "CUDAExecutionProvider" not in ort_providers():
        return False
    try:
        import onnxruntime  # noqa: PLC0415
        if str(onnxruntime.get_device()).upper() != "GPU":
            return False
    except Exception:  # pragma: no cover
        return False
    #: DLL 都不在就必然回落，提前判掉，别等第一份文档跑出「怎么还是 CPU」
    return prepare_cuda_dlls() is not None or bool(shutil.which("cublasLt64_13.dll"))


def engine_device() -> str:
    """这次到底用 CPU 还是 CUDA（``OCR_DEVICE`` 可钉死；``cuda`` 上不了就抛）。"""
    want = (os.environ.get("OCR_DEVICE") or ENGINE_DEVICE_AUTO).strip().lower()
    if want == ENGINE_DEVICE_CPU:
        return ENGINE_DEVICE_CPU
    if cuda_ready():
        return ENGINE_DEVICE_CUDA
    if want == ENGINE_DEVICE_CUDA:
        raise DeviceUnavailable(
            "OCR_DEVICE=cuda 但这套 onnxruntime 给不出可用的 CUDAExecutionProvider"
            f"（现有 provider={list(ort_providers()) or '无'}，"
            f"torch/lib 里的 cublasLt64_13.dll={prepare_cuda_dlls() or '未找到'}）。"
            "装 onnxruntime-gpu 并确认 CUDA 13/cuDNN 9 的 DLL 可加载后重试；"
            "要显式退回 CPU 就给 OCR_DEVICE=cpu。")
    return ENGINE_DEVICE_CPU


def device_report(engine) -> list[tuple[str, ...]]:
    """建完引擎后回头问它**实际**用了哪个 provider（``use_cuda=True`` 不算证据）。

    只做只读遍历，不参与任何决策：报不出东西时返回空表，由调用方如实写「无法确认」，
    绝不用「我设了参数」冒充「它在用 GPU」。
    """
    found: list[tuple[str, ...]] = []
    seen: set[int] = set()
    stack: list[tuple[object, int]] = [(engine, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 4 or id(node) in seen:
            continue
        seen.add(id(node))
        getter = getattr(node, "get_providers", None)
        if callable(getter):
            try:
                found.append(tuple(getter()))
            except Exception:  # pragma: no cover —— 第三方会话对象不该把批次搞死
                pass
            continue
        for value in list(getattr(node, "__dict__", {}).values()):
            if isinstance(value, (list, tuple)):
                stack.extend((item, depth + 1) for item in value if hasattr(item, "__dict__"))
            elif hasattr(value, "__dict__"):
                stack.append((value, depth + 1))
    return found


def _warmup(engine) -> None:
    """跑一张带字的小图，把 rapidocr 3.x **懒建**的会话逼出来。

    不做这一步 ``device_report`` 只能返回空表：新档的 ``text_det``／``text_rec`` 初始化时
    都是 ``None``，模型在第一次 ``__call__`` 时才 load —— 实测就是因此把「GPU 明明用上了」
    误判成「无法确认 provider」。耗时约 0.3 s，只在要 CUDA 时做。

    带字（不是纯白）是为了让 rec 模型也建出来：det 与 rec 是两个 session，
    只查 det 就漏掉「识别模型回落 CPU」这种半截状态。
    """
    try:
        import numpy as np  # noqa: PLC0415 —— 预检用，不进模块顶层
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

        img = Image.new("RGB", (224, 72), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=36)
        except TypeError:  # pragma: no cover —— 老 Pillow 的 load_default 不认 size
            font = ImageFont.load_default()
        draw.text((10, 12), "OCR 0123", fill="black", font=font)
        engine(np.asarray(img))
    except Exception as exc:  # noqa: BLE001 —— 预检失败只降级成「无法确认」，不该把整批判死
        print(f"（预检：跑小图触发建会话时出错 —— {type(exc).__name__}: {exc}）", flush=True)


class _ModernEngine:
    """把 rapidocr 3.x 的回执拍成老档的形状 ``(lines, elapse)``。

    新档返回 ``RapidOCROutput``（``boxes``／``txts``／``scores``／``elapse``），老档返回
    ``([[框, 文本, 分], …], 耗时)``；此处只做形状归一，字段名照 ``dir()`` 实测写死 ——
    不写 ``getattr(out, name, None)`` 式兜底：新档**不校验**参数/字段名（实测传一个假键
    也照样构造成功），静默兜底会把「字段改名」变成「每页都识别成空白」，那是最难查的一类。
    """

    name = ENGINE_NEW

    def __init__(self, engine) -> None:
        self._engine = engine

    def __call__(self, image):
        out = self._engine(image)
        #: ⚠️ 不能写 ``getattr(out, "boxes", None) or ()``：``boxes`` 是 **numpy 数组**
        #: （N×4×2），``or`` 会对它求布尔值 ⇒ 真跑第一份就 `ValueError: The truth value
        #: of an array with more than one element is ambiguous`（单看假替身永远试不出）。
        boxes = getattr(out, "boxes", None)
        texts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)
        if texts is None:
            return [], getattr(out, "elapse", None)
        boxes = () if boxes is None else boxes
        scores = () if scores is None else scores
        lines = [[box, text, score] for box, text, score in zip(boxes, texts, scores)]
        return lines, getattr(out, "elapse", None)


def _new_engine():
    """建一个 OCR 引擎（**每个进程一个**：onnxruntime 的 session 不跨进程共享）。

    下面每个数都是同一份书、同一页、同一台机器上量出来的（``序列编剧法``，150 dpi，
    944×1335），不是抄文档：

    ==========================  =========  =========  ==========
    配置                          s/页       字/秒      识别字数
    ==========================  =========  =========  ==========
    老档 PP-OCRv4 ``batch=6``     7.68        46         2816
    老档 PP-OCRv4 ``batch=16``    1.75       174         2811
    **新档 PP-OCRv6 默认**        **0.32**   **~1300**   417（3 页）
    ==========================  =========  =========  ==========

    ⇒ ① 默认批大小 6 是最大的冤枉钱（一次只喂 6 行给识别模型，几百行/页的书要排几十批），
    老档拉到 16 是 **3.8×** 且**识别字数一样**；② 换成新档（PP-OCRv6 small）再 **3.3×**
    —— 3 页小样 417 字 vs 老档 410 字，不是拿精度换速度。两代默认都是 6，都得显式调；
    ③ ``intra_op_num_threads`` 必须显式给（见 ``engine_threads``），新档同样认这个键
    （实测 1 线程 0.60 s/页 vs 16 线程 0.27 s/页 ⇒ 键名正确且真生效）。

    档位由 ``OCR_ENGINE`` 决定（``auto``／``rapidocr``／``rapidocr-onnxruntime``）。
    """
    workers = max(1, int(os.environ.get("OCR_WORKERS") or 1))
    threads = engine_threads(workers)
    device = engine_device()
    if engine_name() == ENGINE_NEW:
        from rapidocr import RapidOCR  # noqa: PLC0415 —— 只有真跑时才需要

        params: dict[str, object] = {
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Rec.rec_batch_num": _rec_batch(),
        }
        if device == ENGINE_DEVICE_CUDA:
            #: 键名逐字取自 ``rapidocr/config.yaml``（``EngineConfig.onnxruntime.use_cuda``
            #: 与 ``cuda_ep_cfg.device_id``）。新档**不校验**未知键，写错一个字不会报错，
            #: 只会在取 ``use_cuda`` 时拿到 False 然后安静地跑 CPU。
            params["EngineConfig.onnxruntime.use_cuda"] = True
            params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = 0
        return _ModernEngine(RapidOCR(params=params))
    from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

    return RapidOCR(intra_op_num_threads=threads,
                    inter_op_num_threads=1,
                    rec_batch_num=_rec_batch(),
                    use_cuda=device == ENGINE_DEVICE_CUDA)


def _init_escalator() -> None:
    """按环境变量建升级器；未配置端点时为 None（整条通道不存在，零开销）。"""
    global _ESCALATOR
    config = ocr_vlm.VlmConfig.from_env()
    _ESCALATOR = ocr_vlm.Escalator(config) if config.usable else None


def _worker_init(dpi: int) -> None:
    global _OCR, _WORKER_DPI
    _OCR, _WORKER_DPI = _new_engine(), dpi
    #: 子进程 spawn 时重新 import 本模块 ⇒ 升级器必须在这里重建（环境变量已继承）
    _init_escalator()


def _worker_task(item: dict, max_pages: int | None,
                 page_range: tuple[int, int] | None = None) -> dict:
    return _ocr_item(item, _OCR, dpi=_WORKER_DPI, max_pages=max_pages, page_range=page_range)


class _ResourceSampler(threading.Thread):
    """跑批期间每 5 秒采一次 CPU/GPU，收尾回答「到底谁在瓶颈」。

    为什么要独立线程：``report()`` 是**按份**触发的，一份 900 页的大书跑三分钟才报一次，
    批内的峰谷全被抹平 —— 而「CPU 到底吃没吃满」恰恰要看均值与峰值两样。
    采样只读（psutil 计数 + nvidia-smi 查询），不参与任何决策，跑挂了也不影响主流程。
    """

    def __init__(self, interval: float = 5.0) -> None:
        super().__init__(daemon=True, name="ocr-resource-sampler")
        self.interval = max(1.0, float(interval))
        self.cpu: list[float] = []
        self.gpu: list[int] = []
        self.free: list[int] = []
        #: ⚠️ 绝不能叫 ``_stop`` —— ``threading.Thread._stop()`` 是内部方法，
        #: 用同名属性覆盖后 ``join()`` 里会 ``self._stop()`` 调用 Event ⇒ TypeError，
        #: 而且只在**真跑完 join 时**才炸（单测不 join 就发现不了，2026-11-04 实录）
        self._closing = threading.Event()
        self._warned = False

    def run(self) -> None:  # noqa: D102 —— Thread.run
        try:
            import psutil  # noqa: PLC0415
        except Exception:  # noqa: BLE001 —— 没装 psutil 就只采 GPU，不编 CPU 数字
            psutil = None  # type: ignore[assignment]
        if psutil is not None:
            #: 预热：psutil 的 ``cpu_percent(interval=None)`` 首次恒返 0.0（历史遗留语义），
            #: 不预热就会把「机器空着」写进报告，正好把最该看的第一个点抹掉
            psutil.cpu_percent(interval=None)
        while not self._closing.wait(self.interval):
            if psutil is not None:
                value = psutil.cpu_percent(interval=None)
                if value:
                    self.cpu.append(float(value))
            util = gpu_util_percent()
            if util is not None:
                self.gpu.append(util)
            mem = gpu_memory_mib()
            if mem is not None:
                self.free.append(mem[1])
                #: 已经有会话在跑，此时显存被别的任务吃掉是**不可恢复**的（会话不能中途缩）
                #: ⇒ 只告警一次，让人知道这批可能中途 OOM、以及该去停谁
                if not self._warned and mem[1] < GPU_HEADROOM_MIB * 2:
                    self._warned = True
                    print(f"⚠️  显存只剩 {mem[1]} MiB —— 有别的任务正在抢卡，"
                          f"这批可能中途 OOM；建议先停掉它再跑", flush=True)

    def stop(self) -> None:
        self._closing.set()
        if self.is_alive():
            self.join(timeout=3.0)

    def summary(self) -> str:
        parts: list[str] = []
        parts.append(f"CPU 平均 {sum(self.cpu) / len(self.cpu):.0f}%（峰值 {max(self.cpu):.0f}%）"
                     if self.cpu else "CPU 未采样（psutil 不可用）")
        if self.gpu:
            parts.append(f"GPU 平均 {sum(self.gpu) / len(self.gpu):.0f}%（峰值 {max(self.gpu)}%）")
        if self.free:
            parts.append(f"显存最低空闲 {min(self.free) / 1024:.1f} GiB")
        return "资源实测：" + "｜".join(parts)

    def verdict(self) -> str:
        """一句「谁在瓶颈」。两边都低时必须明说「加线程不会更快」，别让人白调参。"""
        cpu = sum(self.cpu) / len(self.cpu) if self.cpu else None
        gpu = sum(self.gpu) / len(self.gpu) if self.gpu else None
        if cpu is None and gpu is None:
            return ""
        if gpu is not None and gpu >= 85:
            return ("GPU 已吃满 ⇒ 瓶颈在算力。扫描件 OCR 的 CPU 占比本来就低"
                    "（只有渲染与前后处理），CPU 数字上不去不是线程没配上。")
        if cpu is not None and cpu >= 85:
            return "CPU 已吃满 ⇒ 瓶颈在 CPU 侧（渲染/前后处理），加进程只会互相抢。"
        if cpu is not None and gpu is not None and cpu < 40 and gpu < 40:
            return ("两边都没吃满 ⇒ 瓶颈不在算力，而在磁盘 IO 或单线程段"
                    "（渲染/切块/落库）—— 再加线程也不会更快。")
        if cpu is not None and cpu < 60:
            return ("CPU 与 GPU 都还有余量 ⇒ 流水线没喂满，可适当加 --workers"
                    "（注意别超过显存能开的会话数）。")
        return ""


def _run(limit: int, *, dpi: int, max_pages: int | None, substr: str,
         redo: bool, workers: int, only: str = "", reaudit: bool = False) -> int:
    plan = _pending(redo=redo, only=only)
    if substr:
        plan = [item for item in plan if substr in item["rel"]]
    if reaudit:
        #: 只留「有存疑页」的件。隐含 ``--redo``（否则它们都算已处理、根本不在 plan 里）。
        plan = _keep_suspects(plan)
    if not plan:
        print("没有待 OCR 的文档（全部有正文，或已被 --path-substr／--only 过滤掉）。")
        return 0

    engine = engine_name()
    device = engine_device()
    todo = plan[:limit] if limit else plan
    engine_needed = _needs_engine(todo)
    if not engine_needed and workers > 1:
        #: 整批全是纯文本件 ⇒ 进程池是净亏：解码是顺手的 CPU 活，多进程只会互抢磁盘与
        #: sqlite 写锁，还要每个进程各建一次引擎（白占一份显存会话 + 几秒装配）。
        workers = 1
    #: 子进程 spawn 时重新 import 本模块，靠这个变量算出各自的 OMP 线程数
    #: （见文件头的 ``OMP_NUM_THREADS`` 注释：不摊开会把 CPU 超订成 1.18 倍）。
    #: ⚠️ 必须写在 ``workers`` 定案**之后** —— 提前写会把"改成 1"之前的数字带进子进程。
    os.environ["OCR_WORKERS"] = str(max(1, workers))
    threads, render = worker_threads(workers)
    print(f"待 OCR {len(plan)} 份；本次做 {len(todo)} 份，dpi={dpi}，引擎={engine}"
          f"（批={_rec_batch()}），{workers} 进程 ×（{threads} 识别 + {render} 渲染）"
          f" ＝ {workers * (threads + render)}（预算 {cpu_budget()} / 硬件 {_CORES} 逻辑线程），"
          f"设备={device}，每份上限 {'全部页' if not max_pages else str(max_pages) + ' 页'}"
          + ("" if engine_needed else "；本批全是纯文本件 ⇒ 只解码，不起引擎、单进程"),
          flush=True)
    #: 预检：在**主进程**真建一次引擎，把 provider 如实报出来。
    #: 「我设了 use_cuda=True」不是证据 —— 装了 onnxruntime-gpu 但缺 cublasLt DLL 时，
    #: ORT 会往 stderr 吐一行然后静默回落 CPU，等跑完一小时才发现全白跑。
    #: ★还必须在主进程**先跑一张小图**：rapidocr 3.x 的 det/rec 模型是**懒加载**的
    #: （``text_det``/``text_rec`` 初始都是 None），不跑一次就根本没有 session 可问，
    #: ``device_report`` 会如实回「无法确认」——实测正是这样误判成「没上 GPU」。
    if not engine_needed:
        print("跳过引擎预检：本批全是纯文本件（只解码，不跑 OCR）", flush=True)
    else:
        probe = _new_engine()
        if device == ENGINE_DEVICE_CUDA:
            _warmup(probe)
        actual = list(dict.fromkeys(device_report(probe)))
        del probe
        gc.collect()
        print("引擎实报：" + ("；".join(",".join(p) for p in actual) if actual
                            else ("设备=cpu（未探测 CUDA provider）" if device != ENGINE_DEVICE_CUDA
                                  else "无法确认 provider（引擎对象里没找到会话）")), flush=True)
        if device == ENGINE_DEVICE_CUDA and not any("CUDAExecutionProvider" in p for p in actual):
            print("✗ 要的是 CUDA，但引擎实际没有 CUDAExecutionProvider ⇒ 当场中止，"
                  "免得整批跑在 CPU 上还报成功。", flush=True)
            return 3
    failures: list[tuple[str, str]] = []
    scanned_total = 0
    #: 起在 started 之前：第一份大书可能跑三分钟才 report 一次，采样必须覆盖整段墙钟
    sampler = _ResourceSampler()
    sampler.start()
    started = time.time()

    def report(index: int, item: dict, outcome: dict | Exception) -> None:
        #: 每份都 ``flush=True``：后台重定向到文件时块缓冲会让人以为「卡死了」，
        #: 而判断「卡死」还是「在跑」只有日志一条线索。
        nonlocal scanned_total
        if isinstance(outcome, Exception):
            failures.append((item["rel"], f"{type(outcome).__name__}: {outcome}"))
            print(f"[{index}/{len(todo)}] ✗ {item['name'][:40]} —— {outcome}", flush=True)
            #: 源文件结构损坏（截断/坏 xref）是**不可恢复**的：不记下来就会每轮重试、
            #: 每次都同样的报错。记成 quarantine —— 它算「已处理」，`_done_signatures()`
            #: 认它，下一轮才真的跳过；要重试仍可显式 `--redo`。
            #: 临时性错误（显存、IO、句柄）**不在内**：那些跳过就等于永久放弃。
            if _looks_broken(outcome):
                _mark_broken(item, outcome)
            return
        #: ★这一格必须自带兜底：``_save`` 抛出去会让**整批**停在半路（真实事故见
        #: ``SIDECAR_BUSY_MS`` 的注释）。落库失败 ⇒ 记成本份失败、继续下一份 ——
        #: 该文档在 sidecar 里没有登记，下次运行会自己重来，不丢也不假绿。
        try:
            _save_atomic(item, outcome, dpi=dpi, engine=engine)
        except Exception as exc:  # noqa: BLE001 —— 一份存不进去不该拖垮另外几百份
            failures.append((item["rel"], f"存库失败 {type(exc).__name__}: {exc}"))
            print(f"[{index}/{len(todo)}] ✗ {item['name'][:40]} —— 存库失败：{exc}"
                  "（本轮跳过，下次运行会重试）", flush=True)
            return
        scanned_total += int(outcome["scanned"])
        native = int(outcome.get("native") or 0)
        speed = outcome["seconds"] / max(1, outcome["scanned"] - native)
        #: 「原生」＝这些页的 PDF 本来就有文字层（或纯文本件的全部页），没走 OCR
        tail = f"  原生={native}" if native else ""
        #: 纯文本件必须打出用的是哪个编码候选 —— 解错编码是这条路唯一会犯的错，
        #: 而解错之后"页页都能过门禁"，只有这一行能让人看出来不对劲。
        if outcome.get("encoding"):
            tail += f"  编码={outcome['encoding']}"
        rejected = int(outcome.get("rejected") or 0)
        suspect = int(outcome.get("suspect") or 0)
        #: 坏页/疑页必须**逐份可见**：整批最后才汇总的话，跑到第 40 份才发现
        #: 「这一批全是乱码」就已经白烧了几小时。
        flagged = (f"  剔除={rejected}" if rejected else "") + (f"  存疑={suspect}" if suspect else "")
        #: 版头删除行数与「剔除页」是两码事，混在一个数里会让人以为水印没删（实测踩过）
        heads = len(outcome.get("droppedHeads") or [])
        if heads:
            flagged += f"  版头={heads}"
        #: 升级过几页必须当场可见：这既是成本（每页一次 GPU 推理），也是「本地引擎
        #: 在这一份上不可靠」的信号，攒到最后一起报就来不及换策略了
        escalated = int(outcome.get("escalated") or 0)
        if escalated:
            flagged += f"  VLM={escalated}"
        #: 页数不到门槛 ⇒ 跨页统计不成立，水印原样留在块里，必须当场说清楚
        if 0 < int(outcome.get("judged") or 0) < text_clean.RUNNING_HEAD_MIN_PAGES:
            flagged += "  ⚠ 页数不足，未做跨页版头剔除"
        if _status_of(outcome) == STATUS_QUARANTINE:
            flagged += "  ⚠ 整份隔离（大半页不可信，未写块）"
        print(f"[{index}/{len(todo)}] {item['name'][:40]}  "
              f"{outcome['scanned']}/{outcome['pages']}页  {outcome['seconds']}s "
              f"({speed:.1f}s/页)  块={len(outcome['chunks'])}{tail}{flagged}", flush=True)

    if workers <= 1:
        #: 纯文本件不走引擎 ⇒ 别为它建一次会话（建一次要几秒，还占一份显存）
        ocr = _new_engine() if engine_needed else None
        _init_escalator()
        for index, item in enumerate(todo, start=1):
            try:
                outcome: dict | Exception = _ocr_item(item, ocr, dpi=dpi, max_pages=max_pages)
            except Exception as exc:  # noqa: BLE001 —— 单份失败不拖垮整批（坏文件很常见）
                outcome = exc
            report(index, item, outcome)
    else:
        #: sqlite 只允许一个写者（主进程存档），子进程只做「渲染 + OCR」这种纯计算，
        #: 回传结果由主进程清洗并落库。
        #: ⚠️ 别用 `Start-Process -WindowStyle Hidden` 起长批：实测那样 onnxruntime
        #: 会挂死（进程在、CPU 恒 0、一页都不出），同样命令在前台控制台跑就正常。
        #: ★份数少于进程数时按**页**切段（见 :func:`plan_segments`）：否则只剩一份大书时
        #: 只有 1 个进程在跑，GPU/CPU 都只用得上 1/4。``--pages`` 试跑不切
        #: （段会越出试跑上限、白起进程）。
        counts: dict[str, int | None] = {}
        tasks: list[tuple[dict, tuple[int, int] | None]] = [(item, None) for item in todo]
        if workers > 1 and len(todo) < workers and not max_pages:
            counts = {item["path"]: _pdf_pages(item) for item in todo}
            tasks = plan_segments(todo, workers, counts)
            for item, rng in tasks:
                pages = counts.get(item["path"])
                if rng is None or not pages:
                    continue
                print(f"  切段：{item['name'][:40]} 第 {rng[0] + 1}–{rng[1]} 页"
                      f"（共 {pages} 页，切成 "
                      f"{sum(1 for it, r in tasks if it['path'] == item['path'])} 段并行）",
                      flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(dpi,)) as pool:
            need: dict[str, int] = {}
            for item, _rng in tasks:
                need[item["path"]] = need.get(item["path"], 0) + 1
            buffers: dict[str, list[dict]] = {}
            broken: set[str] = set()
            futures = {
                pool.submit(_worker_task, item, max_pages, rng): (item, rng)
                for item, rng in tasks
            }
            done_files = 0
            for future in as_completed(futures):
                item, rng = futures[future]
                try:
                    part = future.result()
                except Exception as exc:  # noqa: BLE001 —— 子进程崩了也只算这一份失败
                    #: 一段失败 ⇒ 整份算失败：缺段就落库等于把书的一截永久写成「完成」
                    broken.add(item["path"])
                    buffers.pop(item["path"], None)
                    done_files += 1
                    report(done_files, item, exc)
                    continue
                if rng is None or item["path"] in broken:
                    if rng is not None:
                        continue
                    done_files += 1
                    report(done_files, item, part)
                    continue
                parts = buffers.setdefault(item["path"], [])
                parts.append(part)
                if len(parts) < need.get(item["path"], 1):
                    continue
                buffers.pop(item["path"], None)
                done_files += 1
                report(done_files, item, _merge_segments(parts))

    spent = round(time.time() - started, 1)
    sampler.stop()
    print(f"\n完成 {len(todo) - len(failures)} 份，失败 {len(failures)} 份，耗时 {spent}s")
    print(sampler.summary())
    note = sampler.verdict()
    if note:
        print(note)
    for rel, error in failures[:10]:
        print(f"  ✗ {rel[-60:]}  {error}")
    if scanned_total:
        #: 只报**墙钟**吞吐：这是唯一能拿去乘剩余页数的数（逐份秒数相加在并发下无意义）
        _record_run(spent, scanned_total)
        print(f"吞吐：{scanned_total} 页 / {spent}s = "
              f"{scanned_total / max(spent, 0.1):.2f} 页/秒"
              f"（{spent / scanned_total:.2f} s/页，{workers} 进程实测）")
    print(f"sidecar: {Path(script_library.OCR_DB).as_posix()}")
    return 0


def main() -> int:
    #: Windows 控制台默认 GBK，`⇒` / `✓` 会直接 UnicodeEncodeError 把整批跑挂（实测踩过）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    #: 参数形式守卫必须排在最前：``--path-substr 某书名`` 这种空格写法会被 :func:`_arg`
    #: 静默忽略、退回「不筛」，于是 ``--redo`` 变成全量重烧（实测白跑十几分钟）。
    split = _split_value_args()
    if split:
        print("这些参数只认「--name=值」的写法，收到的是空格分隔：" + "、".join(split))
        print("按原名改写即可（例如 --path-substr=金榜题名）。"
              "之所以当场拒而不是将就：取不到值时参数会静默退回默认，"
              "对 --path-substr 而言「退回默认」就是「不筛、按优先级跑前 N 份」。")
        return 2

    #: 版面/VLM 开关一律**先落到环境变量**：进程池是 spawn 出来的，子进程只认环境变量
    #: （与 ``OCR_WORKERS`` 同一套机制）。直接往下传参子进程看不到，表现是「配了没生效」。
    if _flag("no-layout"):
        os.environ["OCR_NO_LAYOUT"] = "1"
    for name, key in (("vlm-backend", "OCR_VLM_BACKEND"),
                      ("vlm-endpoint", "OCR_VLM_ENDPOINT"), ("vlm-model", "OCR_VLM_MODEL"),
                      ("vlm-model-path", "OCR_VLM_MODEL_PATH"), ("vlm-max", "OCR_VLM_MAX")):
        value = _arg(name)
        if value:
            os.environ[key] = value
    if _flag("vlm-probe"):
        os.environ["OCR_VLM_PROBE"] = "1"
    if _flag("vlm"):
        #: 只写 ``--vlm`` 就必须能跑起来。``from_env`` 的 backend 默认是 ``http``，
        #: 而 http 要 ``--vlm-endpoint`` ⇒ 盘上权重再齐也判不可用，表现为「开了但一页
        #: 都没升级」（实测踩过，且报错只说"未生效"，指不到"默认走了 http"这个真因）。
        #: 所以先补一步自动选后端：本地有权重就用 local，否则保持 http 等端点。
        if not os.environ.get("OCR_VLM_BACKEND"):
            local = ocr_vlm.VlmConfig(
                backend="local",
                model_path=os.environ.get("OCR_VLM_MODEL_PATH") or "",
            )
            if local.usable:
                os.environ["OCR_VLM_BACKEND"] = "local"
                print(f"VLM：未指定后端，自动选用进程内（{local.weights_dir().as_posix()}）")
        #: 自动选过之后仍不可用 ⇒ 必须当场说清，否则整批跑完才发现一页没升级。
        config = ocr_vlm.VlmConfig.from_env()
        if not config.usable:
            print(f"⚠ --vlm 未生效：{config.describe()}（继续只跑本地引擎）")

    #: 拼错一个字母就必须当场拒：``--only=partal`` 若被静默忽略，等于把 3135 份
    #: 从头跑一遍 —— 那是几天的 CPU，而症状只在「怎么跑了这么久」时才暴露。
    only = _arg("only")
    if only and only not in ("partial", "done"):
        print(f"--only 只认 partial（没跑完的）或 done（跑过的，口径重烧用）；"
              f"收到 {only!r} 一律拒，避免静默跑全量")
        return 2

    if _flag("stats"):
        data = _stats()
        if _flag("json"):
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(f"已 OCR：{data['done']['documents']} 份 / {data['done']['pages']} 页 "
                  f"/ {data['done']['chunks']} 块")
            print(f"待 OCR：{data['pending']['documents']} 份 / "
                  f"{data['pending']['pages']} 页 / "
                  f"{data['pending']['bytes'] / 1073741824:.1f} GB")
            scanned = data["done"]["scannedPages"]
            #: 说清「这个 s/页 是谁的」：实测账、还是本引擎的经验值（新老档差 5 倍）
            basis = (f"（墙钟实测：{scanned} 页 / {data['done']['seconds']}s，含并行度）"
                     if scanned >= 200
                     else f"（样本不足 200 页，用 {engine_name()} 的经验值）")
            print(f"引擎：{engine_name()}（批={_rec_batch()}，"
                  f"OCR_ENGINE 可切 rapidocr-onnxruntime 对照）")
            print(f"版面：{'分栏重排已开' if _LAYOUT_ENABLED else '已关（--no-layout）'}"
                  f" · {ocr_vlm.VlmConfig.from_env().describe()}")
            print(f"预计：{data['estimate']['secondsPerPage']}s/页{basis} ⇒ "
                  f"约 {data['estimate']['hoursRemaining']} 小时")
            audit = data.get("audit") or {}
            rejected = audit.get("reject") or {}
            suspect = audit.get("suspect") or {}
            quarantined = int(audit.get("quarantined") or 0)
            if rejected or suspect or quarantined:
                print(f"校对：剔除页 {rejected.get('pages', 0)}（{rejected.get('documents', 0)} 份）、"
                      f"存疑页 {suspect.get('pages', 0)}（{suspect.get('documents', 0)} 份）、"
                      f"整份隔离 {quarantined} 份 ⇒ --audit 看明细")
            print(f"库：{data['db']}")
        return 0

    if _flag("audit"):
        report = _audit_report(_count("limit", 40), _arg("verdict", "reject"))
        if _flag("json"):
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if not report["available"]:
            print(f"还没跑过 OCR（{report['db']} 不存在）")
            return 0
        decided = report["decided"]
        print(f"库：{report['db']}")
        for status in (STATUS_OCR, STATUS_QUARANTINE, STATUS_PARTIAL):
            row = decided.get(status)
            if row:
                label = {STATUS_OCR: "已入库", STATUS_QUARANTINE: "已隔离",
                         STATUS_PARTIAL: "试跑未完"}[status]
                print(f"  {label}：{row['documents']} 份 / {row['pages']} 页")
        audit = report["audit"]
        rejected = audit.get("reject") or {}
        suspect = audit.get("suspect") or {}
        print(f"门禁：剔除 {rejected.get('pages', 0)} 页（{rejected.get('documents', 0)} 份）"
              f" · 存疑 {suspect.get('pages', 0)} 页（{suspect.get('documents', 0)} 份）")
        escalated = audit.get("escalated") or {}
        if escalated:
            #: 这类页**已经入库**（最终判定是 ok），单独列出来是为了让人知道「这些页
            #: 是两遍读法不一致、被标了待复核的」，而不是混在剔除页里找不着
            print(f"升级：{escalated.get('pages', 0)} 页走过 VLM 重识别或双引擎分歧 "
                  f"⇒ --audit --verdict=escalated 看明细")
        if report["ungated"]:
            #: 旧口径产物：没走过清洗与门禁，里面的乱码无从查证 ⇒ 必须自重做，不能装作干净
            print(f"旧口径：{report['ungated']} 份在门禁上线前入库，未经校对 —— "
                  f"建议 --run --redo --path-substr=<其中一份> 逐份重做")
        short = report["shortDocs"]
        if short["documents"]:
            #: 短文档做不了跨页统计 ⇒ 页眉页脚/水印**原样留在库里**。这不是 bug，
            #: 是"宁可漏报不可误杀"的代价；报出来让人知道哪一份要自己扫一眼。
            print(f"短文档：{short['documents']} 份（{short['pages']} 页）不足 "
                  f"{short['threshold']} 页 —— 跨页版头剔除对它们不生效，水印可能残留")
            for row in short["sample"]:
                print(f"  · {row['pages']} 页 {row['name'][:50]}")
        if report["quarantined"]:
            print("\n整份隔离（未写块，建议先抽看再决定是否 --redo）：")
            for row in report["quarantined"]:
                print(f"  [{row['rejectedPages']}/{row['pages']} 页坏] {row['name'][:50]}")
                print(f"      {row['path'][-90:]}")
        if report["pages"]:
            subject = "剔除" if _arg("verdict", "reject") == "reject" else "存疑"
            print(f"\n{subject}页明细（前 {len(report['pages'])} 条）：")
            for row in report["pages"]:
                print(f"  · {row['name'][:36]} 第{row['page']}页 {row['chars']}字  {row['reasons']}")
                if row["preview"]:
                    print(f"      {row['preview'][:70]}")
        print("\n复核后重做：--run --redo --path-substr=<上面路径的一段>")
        return 0

    if _flag("plan"):
        plan = _pending(only=_arg("only"))
        limit = _count("limit", 20)
        for index, item in enumerate(plan[:limit], start=1):
            print(f"{index:3d}. [{item['priority']:2d}] {item['pages']:5d}页 "
                  f"{item['size'] / 1048576:7.1f}MB  {item['rel']}")
        print(f"\n（共 {len(plan)} 份待 OCR，上面列了前 {min(limit, len(plan))} 份）")
        return 0

    if _flag("run"):
        #: 先采一次机器现状（CPU 占用 + 空闲显存）再决定进程数与线程数，并把它落进
        #: ``OCR_CPU_BUDGET`` 供 spawn 出来的子进程读（见 :func:`apply_snapshot`）。
        #: 这一步只花 ~1 s，却能避免两种真实事故：显存被别人占着还硬开 4 会话（OOM）、
        #: 别人正在跑重活时我们仍按满配抢线程（双方都慢）。
        snap = resource_snapshot()
        apply_snapshot(snap)
        #: ``--reaudit`` 自带 ``--redo``：那些件都算"已处理"，不重判就进不了候选。
        return _run(_limit_arg(20), dpi=_count("dpi", DEFAULT_DPI),
                    max_pages=int(_arg("pages", "0") or 0) or None,
                    substr=_arg("path-substr"),
                    redo=_flag("redo") or _flag("reaudit"),
                    workers=_count("workers", default_workers(snap)), only=_arg("only"),
                    reaudit=_flag("reaudit"))

    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
