"""S7 自检：**世界各大模型生态的本地落点**（2026-09-25 用户口径 ✓：

「扫描电脑内的模型，只要后端服务启动就可以扫描，**不要依赖外部服务**」✓ +
「要可以覆盖**世界各大模型**，不仅仅限于 Ollama」✓）。

为什么需要：扫描器原来只认本仓 ``models/``、``local_services/`` 与 ComfyUI ✗ ⇒ HuggingFace /
ModelScope / LM Studio / GPT4All / Jan / llama.cpp 里的模型**扫不到、也不报错** ✗✗ ⇒
用户以为「本机没模型」✗。本套件把「覆盖各大生态 + 零外部依赖 + 缓存 ≠ 能推理」变成机械判据 ✓。

判据（⚠️ 全部**离线**：只碰临时目录 ✓，不读真机模型库 ✓、不打网络 ✓、不要求任何生态的服务在跑 ✓）：

1. **表自检**：id 唯一、点名的生态一个不少、``OVERRIDE_RUNTIMES`` **不许**含强命中
   （``h3`` / ``local-sd`` / ``cosyvoice``）✗ —— 否则「它在 HF 缓存里」会把「它是 H3 的 DiT」
   这个**事实**抹掉 ✗✗；
2. **落点发现**：``~`` / ``%VAR%`` / ``$VAR`` 展开 ✓；变量**没设不编路径** ✓；``roots()`` 只返回
   **真实存在**的目录 ✓；``walkable=False``（Ollama）不进默认扫描根 ✓ 但**参与归属判定** ✓；
3. **归属判定**：最长匹配 ✓、相似前缀不误判（``hub-other`` ≭ ``hub`` ✓）、根目录自身算在内 ✓；
4. **弱命中才改判**：``comfyui`` / ``unknown`` / ``ollama`` → 生态 ✓；``h3`` 等强命中**原样** ✓；
5. **扫描整合**：``blobs/``（无扩展名）与 ``.no_exist/``（负缓存标记）**不算模型** ✓、``datasets--*``
   不扫 ✓ ⇒ 不产生「幽灵模型」✗；``byEcosystem`` 计数正确 ✓；
6. **Ollama 清单并入**：只在「默认扫描」或「扫的目录覆盖 Ollama 库」时并 ✓、``kinds`` 过滤生效 ✓、
   blob 缺失要在 note 里**说出来** ✓。

运行::

    ./.venv/Scripts/python.exe tests/model_ecosystems_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ecosystems_"))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import local_model_scan as ls  # noqa: E402
from app.services import model_ecosystems as eco  # noqa: E402
from app.services import ollama_store  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def _make_hf_cache(root: str) -> None:
    """造一份**逼真**的 HuggingFace 缓存树（含最容易扫错的三类条目）。"""
    sha = "a" * 40
    _write(os.path.join(root, "models--black-forest-labs--FLUX.1-dev", "snapshots", sha,
                        "flux1-dev.safetensors"), b"x" * 32)
    _write(os.path.join(root, "models--Qwen--Qwen3-4B", "snapshots", sha,
                        "model-00001-of-00002.safetensors"), b"y" * 16)
    # ① 内容寻址 blob：**无扩展名** ⇒ 不该被当成模型（否则条目与体积都翻倍 ✗）
    _write(os.path.join(root, "models--black-forest-labs--FLUX.1-dev", "blobs",
                        "3f4e0b0a1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f"),
           b"x" * 32)
    # ② `.no_exist` 负缓存标记：0 字节**同名**占位文件 ⇒ 不跳会扫出「幽灵模型」✗✗
    _write(os.path.join(root, "models--Qwen--Qwen3-4B", ".no_exist", sha,
                        "model-00002-of-00002.safetensors"), b"")
    # ③ 数据集缓存**不是模型** ✗；④ 快照里的 config/tokenizer 也不是 ✓
    _write(os.path.join(root, "datasets--foo--bar", "snapshots", sha, "data.parquet"), b"z" * 8)
    _write(os.path.join(root, "models--Qwen--Qwen3-4B", "snapshots", sha, "config.json"), b"{}")


def main() -> int:
    client = TestClient(app)
    tmp = tempfile.mkdtemp(prefix="ecosystems_")
    hub = os.path.join(tmp, "cache", "huggingface", "hub")
    _make_hf_cache(hub)

    # ================= ① 生态表自检 =================
    ids = [spec.id for spec in eco.ECOSYSTEMS]
    check("表: id **唯一**且非空", len(ids) == len(set(ids)) and all(ids), ids)
    check("表: 用户点名的生态一个不少（HuggingFace / ModelScope / LM Studio / GPT4All / Jan / llama.cpp）",
          {"ollama", "huggingface", "modelscope", "lmstudio", "gpt4all", "jan", "llamacpp"}
          <= set(ids), ids)
    check("表: 每个生态都有中文标签与可用说明（label/note 非空）",
          all(spec.label.strip() and len(spec.note) >= 8 for spec in eco.ECOSYSTEMS),
          [s.id for s in eco.ECOSYSTEMS if not (s.label.strip() and len(s.note) >= 8)])
    check("表: 未标 runnable 的生态**默认不可直接调** ✓（不吹「能跑」✗）",
          eco.get("huggingface").runnable is False and eco.get("lmstudio").runnable is False)
    check("⭐ 表: OVERRIDE_RUNTIMES **只含弱命中**（不许含 h3 / local-sd / cosyvoice 等强命中）",
          eco.OVERRIDE_RUNTIMES == frozenset({"comfyui", "unknown", "ollama"}),
          sorted(eco.OVERRIDE_RUNTIMES))
    check("表: labels() 覆盖全部生态（前端角标用，避免前后端各写一份）",
          set(eco.labels()) == set(ids))
    check("反套套逻辑: 未知生态 id ⇒ None（不许瞎编一个生态 ✗）",
          eco.get("nope") is None and eco.get("") is None)

    # ================= ② 落点发现 =================
    # ⚠️ `HF_HUB_CACHE` 要**一直指着临时缓存**（直到本进程结束）✗ —— 别在中途收掉：
    # 生态归属是 ``detect()`` 里**当场读环境变量**算的 ✓ ⇒ 收早了，⑤ 段就会因为
    # 「临时目录不在任何生态落点里」而判成「其他位置」✗✗（我就栽过这一次 ✓ 记着）。
    saved_hub_cache = os.environ.get("HF_HUB_CACHE")
    os.environ["HF_HUB_CACHE"] = hub
    try:
        candidates = eco._candidates(eco.get("huggingface"))
        check("落点: 环境变量**优先**（HF_HUB_CACHE 排在候选第一位）",
              bool(candidates) and os.path.normcase(candidates[0]) == os.path.normcase(os.path.abspath(hub)),
              candidates)
        check("落点: 变量没设**不编路径**（候选里不留 `%VAR%` / `$VAR` 字面量）",
              not any(("%" in c or "$" in c) for c in candidates), candidates)
        check("落点: 候选**不重复**且都是绝对路径",
              len(candidates) == len({os.path.normcase(c) for c in candidates})
              and all(os.path.isabs(c) for c in candidates), candidates)
        check("落点: roots() 只返回**真实存在**的目录（扫描根目录不许有幽灵 ✗）",
              all(os.path.isdir(p) for _, p in eco.roots()), eco.roots())
        check("落点: roots() 认得刚指向的环境变量目录（HuggingFace 已就位）",
              ("huggingface", os.path.abspath(hub)) in eco.roots())
        check("落点: `walkable=False` 的生态**不进**默认扫描根 ✓ 但归属判定仍认 ✓",
              "ollama" not in {eid for eid, _ in eco.roots()}
              and "ollama" in {eid for eid, _ in eco.all_roots()})
        check("落点: all_roots() 含**不存在的候选** ✓（归属判定要知道「它本该在哪」）",
              len(eco.all_roots()) >= len(eco.roots()))
    finally:
        # ⚠️ 这里**不**还原 HF_HUB_CACHE ✗ —— 后面的扫描段还得指着临时缓存 ✓
        # （真还原放在最后一步 ✓；本进程随即结束，环境变量带不出去 ✓）
        os.environ["HF_HUB_CACHE"] = hub

    os.environ["PROBEVAR"] = tmp
    try:
        check("落点: `%VAR%` / `$VAR` / `${VAR}` 三种写法展开成同一路径（不借 expandvars：POSIX 上不认 `%VAR%`）",
              {os.path.normcase(eco._expand(t)) for t in ("%PROBEVAR%", "$PROBEVAR", "${PROBEVAR}")}
              == {os.path.normcase(os.path.abspath(tmp))},
              [eco._expand(t) for t in ("%PROBEVAR%", "$PROBEVAR", "${PROBEVAR}")])
        check("落点: `~` 展开成用户主目录",
              os.path.normcase(eco._expand("~")) == os.path.normcase(os.path.expanduser("~")),
              eco._expand("~"))
    finally:
        os.environ.pop("PROBEVAR", None)
    check("落点: 变量没设 ⇒ None（**宁可漏一个落点，不编一个路径** ✗）",
          eco._expand("%PROBENOTSET%") is None and eco._env_path("PROBENOTSET") is None)
    saved_appdata = os.environ.pop("APPDATA", None)
    try:
        jan = eco._candidates(eco.get("jan"))
        # ⚠️ 判据要盯**那一条候选**，不能拿「结尾是 jan\\data\\models」去筛 ✗ ——
        # `~/Library/Application Support/Jan/data/models` 正好也长这样 ✗（我第一版就写错了 ✓）
        dropped = os.path.join(saved_appdata, "Jan", "data", "models") if saved_appdata else ""
        check("落点: `%APPDATA%` 没设 ⇒ **那一条候选**整条丢弃（不留半截路径 ✗✗）",
              bool(saved_appdata)
              and os.path.normcase(dropped) not in {os.path.normcase(c) for c in jan}, jan)
    finally:
        if saved_appdata is not None:
            os.environ["APPDATA"] = saved_appdata
            jan_restored = eco._candidates(eco.get("jan"))
            check("落点: `%APPDATA%` 设着 ⇒ **那一条候选**在（只丢没设的，不误伤有的 ✓）",
                  os.path.normcase(os.path.join(saved_appdata, "Jan", "data", "models"))
                  in {os.path.normcase(c) for c in jan_restored}, jan_restored)

    # ================= ③ 归属判定 =================
    injected = [("huggingface", hub), ("ollama", os.path.join(tmp, ".ollama", "models"))]
    check("归属: 缓存里的权重归 HuggingFace",
          eco.detect(os.path.join(hub, "models--a--b", "snapshots", "s", "m.safetensors"),
                     injected) == "huggingface")
    check("归属: 根目录**自身**也算（不要求非得是子路径）✓ 尾部带分隔符也认 ✓",
          eco.detect(hub, injected) == "huggingface"
          and eco.detect(hub + os.sep, injected) == "huggingface")
    check("归属: **相似前缀不误判**（`hub-other` ≭ `hub` ✗✗）",
          eco.detect(os.path.join(tmp, "cache", "huggingface", "hub-other", "m.safetensors"),
                     injected) == "")
    check("归属: Ollama 模型库（walkable=False）**照样认得出** ✓",
          eco.detect(os.path.join(tmp, ".ollama", "models", "manifests", "r", "l", "m", "t"),
                     injected) == "ollama")
    check("归属: **最长匹配**（嵌套落点取最具体的那个）✓",
          eco.detect(os.path.join(tmp, "sub", "f.bin"),
                     [("outer", tmp), ("inner", os.path.join(tmp, "sub"))]) == "inner")
    check("归属: 空路径 / 不在任何生态 ⇒ `\"\"`（不硬猜 ✗）",
          eco.detect("", injected) == ""
          and eco.detect(os.path.join(tmp, "elsewhere", "a.bin"), injected) == "")

    # ================= ④ 弱命中才改判 =================
    weak = {"kind": "image", "runtime": "comfyui", "confidence": "high",
            "role": "standalone", "matchedBy": ["flux"]}
    changed = eco.apply_ecosystem(weak, "huggingface")
    check("改判: 弱命中（comfyui）⇒ 改成所在生态 + matchedBy 留痕（且**不改原字典**）",
          changed["runtime"] == "huggingface" and "ecosystem:huggingface" in changed["matchedBy"]
          and weak["runtime"] == "comfyui", changed)
    strong = {"kind": "video", "runtime": "h3", "confidence": "high",
              "role": "standalone", "matchedBy": ["h3-dit"]}
    check("⭐ 改判: **强命中原样**（HF 缓存里的 H3 DiT 依然是 H3 权重 —— 事实不被「在哪」抹掉）",
          eco.apply_ecosystem(strong, "huggingface") == strong)
    check("改判: 未知生态 ⇒ 原样返回", eco.apply_ecosystem(weak, "nope") == weak)
    suggestion = eco.build_suggestion("huggingface", changed, "flux1-dev.safetensors", ".safetensors")
    check("建议: **只有真被改判过**才给生态建议（强命中 ⇒ None ⇒ 用规则原建议）",
          suggestion is not None
          and eco.build_suggestion("huggingface", strong, "h3.safetensors", ".safetensors") is None)
    check("建议: 缓存类生态 **callable=False** + 说清怎么才真能用（缓存 ≠ 能推理 ✓）",
          suggestion["callable"] is False and "只读" in suggestion["note"]
          and suggestion["model"] == "flux1-dev", suggestion)
    component = dict(changed, role="component")
    component_note = eco.build_suggestion("huggingface", component, "vae.safetensors", ".safetensors")
    check("建议: **组件权重**照列但不单独注册（callable=False + note 点明组件）",
          component_note["callable"] is False and "组件" in component_note["note"], component_note)
    ollama_ok = eco.build_ollama_manifest_suggestion("qwen3:8b", 100, 0)
    check("建议: 已在本机 Ollama 库里的模型 ⇒ 可注册（无需再导入）+ baseUrl 11434",
          ollama_ok["callable"] is True and ollama_ok["baseUrl"].endswith("11434")
          and ollama_ok["model"] == "qwen3:8b", ollama_ok)
    check("建议: blob 缺失要在 note 里**说出来**（清单在、权重没了 ≠ 能推理 ✓）",
          "blob" in eco.build_ollama_manifest_suggestion("qwen3:8b", 100, 3)["note"])

    # ================= ⑤ 扫描整合：HF 缓存 =================
    scanned = ls.scan_local_models({"roots": [hub]})
    names = sorted(m["filename"] for m in scanned["models"])
    check("扫描: HF 缓存里**只认出真权重**（blobs 无扩展名不算、`.no_exist` 零字节不算、"
          "`datasets--*` 不扫、config.json 不算）",
          names == ["flux1-dev.safetensors", "model-00001-of-00002.safetensors"], names)
    check("扫描: HF 缓存里的命中标了**来源生态** + runtime（强命中不改判、弱命中改判）",
          all(m["ecosystem"] == "huggingface" for m in scanned["models"])
          and all(m["runtime"] == "huggingface" for m in scanned["models"]),
          [(m["filename"], m["runtime"]) for m in scanned["models"]])
    check("扫描: 每个命中都带**生态版建议**（callable=False + 只读说明）",
          all(m["suggested"]["callable"] is False and "只读" in m["suggested"]["note"]
              for m in scanned["models"]),
          [m["suggested"] for m in scanned["models"]])
    check("扫描: `byEcosystem` 计数与标签（声明顺序 ✓）",
          scanned["byEcosystem"] == [{"id": "huggingface", "label": "HuggingFace 缓存", "count": 2}],
          scanned["byEcosystem"])
    check("扫描: `byKind` 与总数仍然对得上（加了生态字段没打乱既有口径）",
          sum(scanned["byKind"].values()) == scanned["total"] == 2, scanned["byKind"])
    route_scan = client.get("/api/v1/local-models/scan", params={"roots": hub}).json()["data"]
    check("端点: GET /scan 把 `byEcosystem` 透出来了（前端角标就吃这个字段）",
          route_scan["byEcosystem"][0]["id"] == "huggingface"
          and all("ecosystem" in m for m in route_scan["models"]), route_scan.get("byEcosystem"))

    # ================= ⑥ Ollama 清单并入 =================
    ollama_root = os.path.join(tmp, "ollama_models")
    manifest = os.path.join(ollama_root, "manifests", "registry.ollama.ai", "library", "qwen3", "8b")
    _write(manifest, b"{}")  # ⚠️ 无扩展名 ⇒ 文件遍历**看不见**它 ✓（所以必须靠清单读 ✓）
    fake_models = [
        {"name": "__fake_a__:8b", "size": 5_000_000_000, "size_label": "4.7 GB",
         "modified_at": "2026-01-01T00:00:00Z", "digest": "sha256:abc", "source": "disk",
         "path": manifest, "blobs": 3, "missingBlobs": 0},
        {"name": "__fake_b__:1b", "size": 1_000, "size_label": "1.0 KB",
         "modified_at": "2026-01-01T00:00:00Z", "digest": "sha256:def", "source": "disk",
         "path": os.path.join(ollama_root, "manifests", "registry.ollama.ai", "library",
                              "broken", "1b"),
         "blobs": 2, "missingBlobs": 2},
    ]
    saved_list, saved_root_call = ollama_store.list_models, ollama_store.models_root
    ollama_store.list_models = lambda root=None: fake_models  # type: ignore[assignment]
    ollama_store.models_root = lambda: ollama_root  # type: ignore[assignment]
    try:
        explicit = ls.scan_local_models({"roots": [hub]})
        check("并入: 显式只扫 HF 缓存 ⇒ **不**凭空冒出 Ollama 的模型（否则「只扫某个盘」会跨盘串味 ✗✗）",
              not [m for m in explicit["models"] if m["ecosystem"] == "ollama"],
              [m["filename"] for m in explicit["models"]])

        mine = ls.scan_local_models({"roots": [ollama_root]})
        merged = [m for m in mine["models"] if m["ecosystem"] == "ollama"]
        check("并入: 扫的目录**就是** Ollama 模型库 ⇒ 清单里的模型进来了（文件遍历看不见它们 ✓）",
              [m["filename"] for m in merged] == ["__fake_a__:8b", "__fake_b__:1b"],
              [m["filename"] for m in mine["models"]])
        first = merged[0]
        check("并入: 条目形状（清单真实路径 ✓ / ext 留空 ✗ 不硬写成 .gguf / text + ollama + 清单来源）",
              first["path"] == manifest and first["ext"] == "" and first["kind"] == "text"
              and first["runtime"] == "ollama" and first["dirname"] == "qwen3"
              and first["matchedBy"] == ["ollama-store"], first)
        check("并入: 体积用清单读出的真实字节，并生成 humanSize",
              first["sizeBytes"] == 5_000_000_000 and bool(first["sizeHuman"]), first)
        check("并入: blob 缺失的模型**照样列出**，但 note 里点名警告（不悄悄当成没问题 ✗）",
              "blob" in merged[1]["suggested"]["note"]
              and merged[1]["suggested"]["callable"] is True, merged[1]["suggested"])
        only_image = ls.scan_local_models({"roots": [ollama_root], "kinds": ["image"]})
        check("并入: `kinds` 过滤生效（只要 image ⇒ text 的 Ollama 模型不进来）",
              only_image["total"] == 0, only_image["models"])
        covered = ls.scan_local_models({"roots": [tmp]})
        check("并入: 扫**包含** Ollama 库的上级目录（如扫整盘）⇒ 也并进来 ✓",
              any(m["ecosystem"] == "ollama" for m in covered["models"]),
              [m["filename"] for m in covered["models"]])

        saved_defaults = ls.get_default_roots
        ls.get_default_roots = lambda: [hub]  # type: ignore[assignment]
        try:
            default_scan = ls.scan_local_models({})
            check("并入: **默认扫描**（没指定 roots）⇒ 并进来 —— 用户要的就是「把本机模型都找出来」✓",
                  any(m["ecosystem"] == "ollama" for m in default_scan["models"])
                  and default_scan["roots"] == [hub], default_scan["roots"])
            check("并入: 默认扫描的 `byEcosystem` 同时含两个生态（HuggingFace 2 / Ollama 2）",
                  [(e["id"], e["count"]) for e in default_scan["byEcosystem"]]
                  == [("ollama", 2), ("huggingface", 2)], default_scan["byEcosystem"])
        finally:
            ls.get_default_roots = saved_defaults  # type: ignore[assignment]
    finally:
        ollama_store.list_models = saved_list  # type: ignore[assignment]
        ollama_store.models_root = saved_root_call  # type: ignore[assignment]

    check("反套套逻辑: 假清单已卸掉 ⇒ 真 `list_models()` 不再返回假数据（证明上面断言的是**注入的数据**）",
          all(str(item.get("name")) != "__fake_a__:8b" for item in ollama_store.list_models()))

    # 收尾：把环境变量还回去（本进程结束前 ✓，免得影响后面的汇总/别的调用 ✓）
    if saved_hub_cache is None:
        os.environ.pop("HF_HUB_CACHE", None)
    else:
        os.environ["HF_HUB_CACHE"] = saved_hub_cache

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
