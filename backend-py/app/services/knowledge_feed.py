"""知识投喂 —— 把台账里的知识编译成**模型真能吃下去**的三种形态。

三层别混叫「投喂」
==================
* **L1 提示词注入**：``compile_for_prompt`` —— 挑 top-N 塞进 system prompt 的「## Knowledge」段。
  立即生效、可逐条关、token 便宜。**只塞「标题 + 一句话」**：全文留在库里按需检索。
* **L2 检索注入**：Agent 自己调 ``search_knowledge``（见 ``agent/tools/knowledge_tools.py``）。
* **L3 权重注入**：``export_runs_dataset`` / ``export_knowledge_dataset`` 产出 jsonl，
  交给 ``lora_train`` 训练，产物登记进 ``models`` 表才算存在。

为什么 L1 只给「标题 + 一句话」
==============================
本仓 system prompt 的预算已经被 41 份 SKILL 与项目上下文吃得很紧。把几百条经验全文塞进去，
模型会开始**背规则**、反而丢掉当前剧本的细节 —— 这是这类系统最常见的自伤方式。
所以：预算 ``1200`` 字、按「置信度 × 效果分」排序、只取 title + body 首句。

负样本绝不进训练集
==================
``export_runs_dataset`` 有 ``max_score`` 参数，但**默认不导出低分样本作为训练目标**：
把废片当正样本教模型是不可逆的（权重里洗不掉）。低分样本的去处是**评测集**，
由调用方显式传 ``include_low=True`` 才导，且 manifest 里标 ``role="eval"``。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from app.core.config import get_data_root
from app.services import knowledge_ledger, outcome_index

__all__ = [
    "compile_for_prompt",
    "record_injection",
    "export_knowledge_dataset",
    "export_runs_dataset",
    "list_datasets",
    "resolve_dataset",
]

#: L1 默认注入哪几类（``fact``/``work`` 是外部资料，体积大、相关性依赖查询 ⇒ 走 L2 检索）
DEFAULT_KINDS = ("recipe", "lesson", "prompt_patch", "failure")

#: 默认预算（字符）。改大之前先想清楚：省下的检索时间值不值 prompt 里的噪声。
DEFAULT_BUDGET = 1200

#: 标题在行内的上限
_TITLE_SNIP = 40

#: 正文首句在行内的上限
_BODY_SNIP = 80

_KIND_LABEL = {
    "recipe": "成法",
    "lesson": "经验",
    "failure": "失败样本",
    "threshold": "阈值",
    "prompt_patch": "提示词补丁",
    "fact": "事实",
    "work": "作品",
}


def _weight(entry: dict[str, Any]) -> float:
    """排序权重：置信度为主，**效果分为正才加分、为负则减分**。

    ``lift is None``（没测过）按 0 计 —— 既不奖励也不惩罚，让新知识有机会被试用，
    但不让一条从未验证的条目压过已验证的正收益条目。
    """
    try:
        confidence = float(entry.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    lift = entry.get("lift")
    try:
        lift_value = float(lift) if lift is not None else 0.0
    except (TypeError, ValueError):
        lift_value = 0.0
    return confidence * (1.0 + max(-0.5, min(0.5, lift_value)))


def _first_sentence(text: str, limit: int = _BODY_SNIP) -> str:
    """取正文第一句（中文句号/分号/换行都算断句），再按长度硬截。"""
    body = (text or "").strip().replace("\r", "")
    for mark in ("\n", "。", "；", ";", "！", "？"):
        cut = body.find(mark)
        if 0 < cut:
            body = body[:cut]
            break
    body = body.strip()
    return body if len(body) <= limit else body[:limit - 1] + "…"


def compile_for_prompt(agent_type: str | None = None, *,
                       budget_chars: int = DEFAULT_BUDGET,
                       kinds: Iterable[str] = DEFAULT_KINDS,
                       limit: int = 8,
                       allowed_status: str = "accepted") -> dict[str, Any]:
    """编译「## Knowledge」段（只含 accepted）。

    回执里 ``entries`` 是**真正被写进这段文本的 id**，调用方拿它去
    ``record_injection`` 记 usage —— 没有这一步，``lift`` 永远算不出来。

    两种「空」要分开报：``available=False`` 是台账坏了；``text=""`` 是台账好的、
    只是没有够格的知识。前端/日志不能都显示成"没知识"。
    """
    budget = max(0, int(budget_chars or 0))
    listing = knowledge_ledger.list_entries(status=allowed_status, limit=200)
    if not listing.get("available", True):
        return {"text": "", "entries": [], "used": 0, "budget": budget,
                "truncated": False, "available": False,
                "hint": listing.get("hint") or "台账不可用"}

    wanted = set(kinds)
    scopes = {"global"}
    if agent_type:
        scopes.add(f"agent:{agent_type}")

    pool = [
        entry for entry in listing.get("items", [])
        if entry.get("kind") in wanted and str(entry.get("scope") or "global") in scopes
    ]
    pool.sort(key=_weight, reverse=True)
    pool = pool[:max(1, int(limit or 1))]

    lines: list[str] = []
    picked: list[str] = []
    used = 0
    truncated = False
    for entry in pool:
        label = _KIND_LABEL.get(str(entry.get("kind")), str(entry.get("kind")))
        title = str(entry.get("title") or "")[:_TITLE_SNIP]
        sentence = _first_sentence(str(entry.get("body") or ""))
        line = f"- [{label}] {title}：{sentence}"
        cost = len(line) + 1
        if used + cost > budget:
            truncated = True
            room = budget - used - 1
            if room > 12 and not lines:
                #: 第一条都放不下时**截断这一条**而不是整段空着：宁可少说，不要不说
                lines.append(line[:room])
                picked.append(str(entry["id"]))
                used = budget
            break
        lines.append(line)
        picked.append(str(entry["id"]))
        used += cost

    text = "\n".join(lines)
    return {
        "text": text,
        "entries": picked,
        "used": used,
        "budget": budget,
        "truncated": truncated,
        "available": True,
        "hint": "" if lines else "台账里还没有 accepted 的知识（先审几条，或把蒸馏出的 proposed 采纳）",
    }


def record_injection(entry_ids: Iterable[str], *, run_id: str | None = None,
                     outcome: str | None = None) -> dict[str, Any]:
    """登记「这些知识被注入到了这次 run」。返回成功条数。"""
    ok = 0
    failed = 0
    for entry_id in entry_ids or []:
        if knowledge_ledger.record_usage(str(entry_id), run_id, outcome):
            ok += 1
        else:
            failed += 1
    return {"recorded": ok, "failed": failed}


# ---------------------------------------------------------------------------
# L3：数据集导出
# ---------------------------------------------------------------------------

def _out_dir(out_dir: str | Path | None, name: str) -> Path:
    if out_dir:
        path = Path(out_dir)
    else:
        path = Path(get_data_root()) / "lora-train" / "datasets" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprint(items: list[dict[str, Any]]) -> str:
    """去重指纹：只对"身份字段"取 hash，样本内容变了要能体现出来。"""
    keys = sorted(str(item.get("source") or item.get("id") or "") for item in items)
    return hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:16]


def _write_dataset(directory: Path, items: list[dict[str, Any]],
                   manifest: dict[str, Any]) -> dict[str, Any]:
    items_path = directory / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    #: ⭐ 整份字节的 sha256：``fingerprint`` 只对 ``source`` 取值（同一条样本被改正文
    #: 它看不出来），而血统要管的是「这份样本还能不能算同一份」。两道都要：指纹管
    #: 「集合换了谁」，sha 管「这堆字节被谁动过」。
    digest = hashlib.sha256(items_path.read_bytes()).hexdigest()
    manifest = {
        **manifest,
        "count": len(items),
        "fingerprint": _fingerprint(items),
        "contentSha256": digest,
        "itemsPath": items_path.as_posix(),
        "roles": _count_by(items, "role"),
        "rights": _count_by(items, "rights"),
        "generatedNs": time.time_ns(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _count_by(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.get(field) or "")] = counts.get(str(item.get(field) or ""), 0) + 1
    return counts


def export_runs_dataset(out_dir: str | Path | None = None, *,
                        kind: str | None = "video", stage: str | None = None,
                        min_score: float | None = None,
                        max_score: float | None = None,
                        since_ns: int | None = None,
                        limit: int = 500,
                        include_low: bool = False) -> dict[str, Any]:
    """把高分跑次导成训练样本（``prompt`` 取自跑次 ``meta``）。

    * 每条样本带 ``source``（run_id）与 ``rights``（默认 ``self-produced``：
      本仓自产才允许进训练；第三方素材要显式改 ``rights`` 再入集）；
    * ``meta`` 里没有提示词的跑次**跳过并计数**（``noPrompt``），不编一个空串充数；
    * ``include_low=False`` 时，低于 ``min_score`` 的样本**不进训练集**（见模块头）。
    """
    rows: list[dict[str, Any]] = []
    scanned = 0
    batch = 500
    offset = 0
    while scanned < int(limit) * 5:
        page = outcome_index.query(kind=kind, stage=stage, since_ns=since_ns,
                                   limit=batch, offset=offset)
        if not page.get("available", True) or not page.get("items"):
            break
        for row in page["items"]:
            scanned += 1
            score = row.get("score")
            if score is not None:
                if min_score is not None and float(score) < float(min_score) and not include_low:
                    continue
                if max_score is not None and float(score) > float(max_score):
                    continue
            elif min_score is not None:
                continue
            rows.append(row)
        offset += batch
        if len(rows) >= int(limit):
            break

    items: list[dict[str, Any]] = []
    no_prompt = 0
    for row in rows[:int(limit)]:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        prompt = (meta.get("prompt") or meta.get("positive") or meta.get("text")
                  or meta.get("promptText") or "")
        prompt = str(prompt).strip()
        if not prompt:
            no_prompt += 1
            continue
        role = "train"
        if row.get("score") is not None and max_score is not None \
                and float(row["score"]) <= float(max_score) and include_low:
            role = "eval"
        items.append({
            "source": str(row.get("run_id") or ""),
            "kind": str(row.get("kind") or ""),
            "stage": str(row.get("stage") or ""),
            "model": str(row.get("model") or ""),
            "score": row.get("score"),
            "role": role,
            "rights": "self-produced",
            "prompt": prompt,
            "params": {key: meta.get(key) for key in
                       ("steps", "cfg", "width", "height", "seed", "sampler", "schedule")
                       if meta.get(key) is not None},
        })

    directory = _out_dir(out_dir, "runs")
    manifest = _write_dataset(directory, items, {
        "dataset": "runs",
        "kind": kind,
        "stage": stage,
        "minScore": min_score,
        "maxScore": max_score,
        "includeLow": include_low,
        "scanned": scanned,
        "noPrompt": no_prompt,
        "skippedLowScore": scanned - len(rows),
    })
    return manifest


def export_knowledge_dataset(out_dir: str | Path | None = None, *,
                             kinds: Iterable[str] = DEFAULT_KINDS,
                             limit: int = 500) -> dict[str, Any]:
    """把 accepted 知识导成「指令 → 做法」对（教模型遇到同类现象怎么处理）。

    这是 L3 里最"便宜"的一步：不需要成片，只要台账里有人审通过的经验。
    适合做小体量 LoRA / 少样本示例集。
    """
    listing = knowledge_ledger.list_entries(status="accepted", limit=max(1, int(limit)))
    wanted = set(kinds)
    items: list[dict[str, Any]] = []
    for entry in listing.get("items", []):
        if entry.get("kind") not in wanted:
            continue
        items.append({
            "source": str(entry.get("id") or ""),
            "role": "train",
            "rights": "self-produced",
            "kind": str(entry.get("kind") or ""),
            "scope": str(entry.get("scope") or "global"),
            "instruction": str(entry.get("title") or ""),
            "response": str(entry.get("body") or ""),
            "evidence": entry.get("evidence") or [],
            "confidence": entry.get("confidence"),
            "lift": entry.get("lift"),
        })

    directory = _out_dir(out_dir, "knowledge")
    return _write_dataset(directory, items, {
        "dataset": "knowledge",
        "kinds": sorted(wanted),
        "available": listing.get("available", True),
    })


# ---------------------------------------------------------------------------
# L3 → 训练：把导出集「指」给一次训练（datasetRef）
# ---------------------------------------------------------------------------

def datasets_root() -> Path:
    """导出集的默认落点根（与 :func:`_out_dir` 同一处，不另立一套）。"""
    return Path(get_data_root()) / "lora-train" / "datasets"


def _manifest_path(ref: str | Path) -> Path | None:
    """把 ``ref`` 解释成一份 ``manifest.json`` 的路径（只做解释，不替调用方猜内容）。

    三种写法都认：目录 ✓ / ``manifest.json`` 本身 ✓ / 只给数据集名（``runs``）✓。
    只给名字时**先按当前目录、再按 :func:`datasets_root`** —— 后者是导出的默认落点，
    这样提交训练时写 ``datasetRef: "runs"`` 就能命中，不必抄整条绝对路径。
    """
    raw = str(ref or "").strip()
    if not raw:
        return None
    given = Path(raw)
    candidates = [given if given.suffix.lower() == ".json" else given / "manifest.json"]
    if not given.is_absolute():
        base = datasets_root() / given.name
        candidates.append(base if given.suffix.lower() == ".json" else base / "manifest.json")
    for item in candidates:
        if item.is_file():
            return item
    return None


def resolve_dataset(ref: str | Path) -> dict[str, Any]:
    """核一次 ``datasetRef`` ⇒ 归一化的数据集事实（**不核就是编**）。

    为什么训练入口要先核引用 ✗
    ==========================
    ``datasetRef`` 只进登记表、不进命令行（训练器吃的是 ``dataset_config`` 那份 toml，
    导出集是 jsonl，两种格式 —— 硬拼一个 argv 只会造出跑不起来的命令）。既然它只用来
    回答「这次训练是拿哪批数据训的」，那它**必须是可核的**：一段随便写的字符串一旦被
    记进登记表，三个月后就变成「可能是这批、也可能不是」的假血统，比没有记录更坏。

    五条硬判据（任何一条不过 ⇒ ``ok=False`` + 说清是哪一条）：

    1. 能找到 ``manifest.json``（目录 / 文件 / 默认落点里的数据集名，三种写法）；
    2. ``itemsPath`` 在盘上，且每行都是合法 JSON；
    3. 实际条数、``roles``、``rights`` 与 manifest 声明**逐项一致**（对不上就是被手改过）；
    4. **字节未被改动**：``contentSha256`` 按 ``items.jsonl`` 现算现比（老 manifest 没有这个
       字段时退到重算 ``fingerprint``，并在 ``integrity`` 里如实标出只核到哪一档 ——
       不假装核过）；
    5. 每条样本都带 ``source`` 与 ``rights``（没有来源/权利标记的样本不许进训练集 ——
       见模块头「负样本绝不进训练集」，权利不明与负样本同样不可逆）。

    空集也拒：0 条样本训出来的只会是「把底模又训了一遍」。

    为什么第 4 条不能只靠 ``fingerprint``：它只对 ``source`` 取值 ⇒ 有人把某条样本的
    提示词/参数改了、``source`` 留着，指数字节都对得上。训练血统要回答的是「这份样本
    还是不是同一份」，所以按**字节**核。
    """
    manifest_path = _manifest_path(ref)
    if manifest_path is None:
        return {"ok": False,
                "why": (f"找不到导出集：{ref!r} —— 需要 export_runs_dataset / "
                        f"export_knowledge_dataset 产出的目录（含 manifest.json），"
                        f"或默认落点 {datasets_root().as_posix()} 下的数据集名")}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        return {"ok": False, "why": f"manifest 读不出来：{err}"}
    if not isinstance(manifest, dict):
        return {"ok": False, "why": f"{manifest_path.name} 不是一份对象 —— 这不是导出集"}

    items_path = str(manifest.get("itemsPath") or "").strip()
    if not items_path:
        return {"ok": False, "why": "manifest 里没有 itemsPath —— 这不是 export_* 产出"}
    items_file = Path(items_path)
    if not items_file.is_file():
        return {"ok": False,
                "why": f"样本文件不在盘上：{items_path}（导出集被挪走或清掉了）"}

    raw = items_file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        return {"ok": False, "why": f"样本文件不是 UTF-8：{err}"}

    items: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as err:
            return {"ok": False, "why": f"第 {number} 行不是合法 JSON：{err}"}
        if not isinstance(item, dict):
            return {"ok": False, "why": f"第 {number} 行不是一个对象"}
        items.append(item)

    if not items:
        return {"ok": False, "why": f"导出集是空的（0 条样本）：{items_path}"}
    declared = manifest.get("count")
    if declared is not None and int(declared) != len(items):
        return {"ok": False,
                "why": (f"manifest 写 {declared} 条，items.jsonl 实际 {len(items)} 条 —— "
                        f"这份导出集被手改过（重跑 export_* 而不是改 jsonl）")}
    roles = _count_by(items, "role")
    rights = _count_by(items, "rights")
    for field, actual in (("roles", roles), ("rights", rights)):
        if isinstance(manifest.get(field), dict) and manifest[field] != actual:
            return {"ok": False,
                    "why": (f"manifest 的 {field} 与 items.jsonl 不一致"
                            f"（{manifest[field]} ≠ {actual}）—— 被手改过")}
    recomputed = _fingerprint(items)
    declared_fp = str(manifest.get("fingerprint") or "").strip()
    if declared_fp and declared_fp != recomputed:
        return {"ok": False,
                "why": (f"manifest 的 fingerprint {declared_fp} ≠ 按样本重算的 {recomputed}"
                        f" —— 样本集合被换过（条数可能还凑得上）")}
    declared_sha = str(manifest.get("contentSha256") or "").strip()
    if declared_sha and declared_sha != digest:
        return {"ok": False,
                "why": (f"items.jsonl 的 sha256 {digest[:16]}… ≠ manifest 记的 "
                        f"{declared_sha[:16]}… —— 样本内容被改过（条数照旧也算改）")}
    #: 只核到哪一档要如实写出：老 manifest 没有 sha 时这是「重算了指纹」，不是「字节没动」
    integrity = "sha256" if declared_sha else ("fingerprint" if declared_fp else "count")
    for number, item in enumerate(items, start=1):
        if not str(item.get("rights") or "").strip():
            return {"ok": False, "why": f"第 {number} 行没有 rights（来源权利不明的样本不进训练集）"}
        if not str(item.get("source") or "").strip():
            return {"ok": False, "why": f"第 {number} 行没有 source（追不到来源的不进训练集）"}

    return {
        "ok": True,
        "why": None,
        "dataset": {
            "ref": str(ref).strip(),
            "manifest": manifest_path.as_posix(),
            "itemsPath": items_path,
            "dataset": manifest.get("dataset"),
            "fingerprint": declared_fp or recomputed,
            "contentSha256": declared_sha or digest,
            "integrity": integrity,
            "count": len(items),
            "roles": roles,
            "rights": rights,
            "generatedNs": manifest.get("generatedNs"),
        },
    }


def list_datasets(limit: int = 50) -> dict[str, Any]:
    """列出现有导出集（让人能选一个 ``datasetRef``，而不是凭空写路径）。

    只读 manifest 摘要、**不逐条核样本** —— 列举要轻，真核在
    :func:`resolve_dataset`（提交训练时那一下才需要付读盘成本）。
    ``itemsExists=false`` 的行照样列出来并标出来：导出集被清掉这件事本身要可见。
    """
    root = datasets_root()
    out: dict[str, Any] = {"root": root.as_posix(), "available": root.is_dir(), "items": []}
    if not root.is_dir():
        return out
    found: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            found.append({"ref": manifest_path.parent.name, "name": manifest_path.parent.name,
                          "manifest": manifest_path.as_posix(), "broken": True})
            continue
        if not isinstance(manifest, dict):
            continue
        items_path = str(manifest.get("itemsPath") or "")
        found.append({
            "ref": manifest_path.parent.name,
            "name": manifest_path.parent.name,
            "manifest": manifest_path.as_posix(),
            "itemsPath": items_path,
            "itemsExists": bool(items_path) and Path(items_path).is_file(),
            "dataset": manifest.get("dataset"),
            "fingerprint": manifest.get("fingerprint"),
            "contentSha256": manifest.get("contentSha256"),
            "count": manifest.get("count"),
            "roles": manifest.get("roles"),
            "rights": manifest.get("rights"),
            "generatedNs": manifest.get("generatedNs"),
        })
    out["items"] = found[:max(1, int(limit))]
    return out
