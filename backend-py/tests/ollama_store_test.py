"""S7 自检：**本机 Ollama 模型库的离线读取**（``services/ollama_store.py`` ✓ 2026-09-25）。

守的是一条**产品硬规则**（用户口径 ✓）：

> 「**只要后端服务启动就可以扫描本机模型，不要依赖外部服务**。」

原来 ``/ollama/status`` 只认 ``/api/tags`` ✓✗ —— 那是**唯一**的官方列模型接口 ✓，但它要求
``ollama serve`` 在跑 ✗ ⇒ 「服务没起 ⇒ 本机模型一个都看不见」✗✗（而盘上明明有 ✓）。
本套把「**读盘**」这条路钉死：

1. **根目录解析**：``OLLAMA_MODELS`` > ``%LOCALAPPDATA%\\Ollama\\models`` > ``~/.ollama/models``；
2. **名字推导**：``manifests/<registry>/library/qwen3/8b`` ⇒ ``qwen3:8b``（``library`` 不显示 ✓）；
   非默认命名空间 ⇒ ``myorg/custom:v2`` ✓；**注册表主机名不能混进名字** ✗；
3. **体积**：按 digest **去重**求和 ✓、**不含 config 层** ✓、**盘上真实字节优先** ✓
   （清单自述值可能对不上真实磁盘 ⇒ 那正是要暴露的 ✓），blob 缺失才回落自述值并计 ``missingBlobs`` ✓；
4. **读不了不算通过** ✓：坏清单/隐藏文件**跳过**而不是让整片列表崩 ✗；
5. **路由回落**：服务不可达时 ``/ollama/status`` 必须**仍然给出盘上的模型** ✓ + 诚实说明来源 ✓。

全程在**临时目录**里造一个假模型库 ✓，不打网络 ✓、不碰真机模型库 ✓。

运行::

    ./.venv/Scripts/python.exe tests/ollama_store_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ollamastore_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routers import ai_configs as ac  # noqa: E402
from app.services import ollama_store as store  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ============================================================
# 造假模型库（结构与 Ollama 真实落盘一致 ✓）
# ============================================================
STORE = tempfile.mkdtemp(prefix="ollamastore_root_")
MANIFESTS = os.path.join(STORE, "manifests")
BLOBS = os.path.join(STORE, "blobs")

#: 各 blob 在盘上的**真实字节**（⚠️ 与清单自述值**有意不一致**，用来验证「盘上优先」✓）
LAYER_A = "a" * 64      # 1000 B（清单自述 1000 ✓ 一致）
LAYER_B = "b" * 64      # 1234 B（清单自述 2000 ✗ 不一致 ⇒ 期望取 1234 ✓）
SHARED = "c" * 64       # 500 B（被两个模型共用 ⇒ 验证「按模型各自求和」✓）
CONFIG = "d" * 64       # 88 B（**config 层** ⇒ 不能计进体积 ✗）
MISSING = "e" * 64      # 声明了但**盘上没有** ✓


def _write(rel: str, payload: bytes) -> str:
    path = os.path.join(STORE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def manifest(layers: list[dict], config_digest: str = CONFIG) -> bytes:
    return json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                   "digest": config_digest, "size": 88},
        "layers": layers,
    }).encode("utf-8")


def layer(digest: str, size: int, media: str = "application/vnd.ollama.image.model") -> dict:
    return {"mediaType": media, "digest": f"sha256:{digest}", "size": size}


for digest, size in ((LAYER_A, 1000), (LAYER_B, 1234), (SHARED, 500), (CONFIG, 88)):
    _write(os.path.join("blobs", f"sha256-{digest}"), b"\0" * size)

_write(os.path.join("manifests", "registry.ollama.ai", "library", "qwen3", "8b"),
       manifest([layer(LAYER_A, 1000), layer(SHARED, 500)]))
_write(os.path.join("manifests", "registry.ollama.ai", "library", "llama3", "8b-instruct"),
       manifest([layer(LAYER_A, 1000), layer(SHARED, 500)]))
# 自述 2000 但盘上只有 1234 ⇒ 期望 size=1234（**盘上优先** ✓）
_write(os.path.join("manifests", "registry.ollama.ai", "myorg", "custom", "v2"),
       manifest([layer(LAYER_B, 2000)]))
# 自述 7777 但 blob 不在盘上 ⇒ 期望 size=7777 + missingBlobs=1 ✓
_write(os.path.join("manifests", "registry.ollama.ai", "library", "broken", "1"),
       manifest([layer(MISSING, 7777)]))
# 坏清单（不是 JSON）与隐藏文件 ⇒ **跳过**，不能让整片列表崩 ✗
_write(os.path.join("manifests", "registry.ollama.ai", "library", "bad", "1"), b"not json")
_write(os.path.join("manifests", ".DS_Store"), b"\0\0")


def main() -> int:  # noqa: C901
    # ================= 根目录解析 =================
    original_env = os.environ.get("OLLAMA_MODELS")
    try:
        os.environ["OLLAMA_MODELS"] = STORE
        check("根目录: OLLAMA_MODELS 环境变量**优先级最高**（本机装到别处也找得到）",
              store.models_root() == os.path.abspath(STORE), store.models_root())
    finally:
        if original_env is None:
            os.environ.pop("OLLAMA_MODELS", None)
        else:
            os.environ["OLLAMA_MODELS"] = original_env

    candidates = store.models_root_candidates()
    normalized = [os.path.normcase(p) for p in candidates]
    check("根目录: 候选**绝对路径且不重复**（保序 ⇒ 优先级可预期 ✓）",
          bool(candidates) and all(os.path.isabs(p) for p in candidates)
          and len(set(normalized)) == len(normalized), candidates)
    check("根目录: 覆盖 Windows 新版落点 `%LOCALAPPDATA%\\Ollama\\models`",
          any(p.endswith(os.path.join("Ollama", "models")) for p in candidates), candidates)
    check("根目录: 覆盖经典落点 `~/.ollama/models`",
          any(p.endswith(os.path.join(".ollama", "models")) for p in candidates), candidates)

    # 候选**全都不存在** ⇒ 返回候选（好拼出「装在哪才找得到」的提示），而不是空串 ✗
    ghost = os.path.join(tempfile.gettempdir(), "ollamastore_ghost", "models")
    original_candidates = store.models_root_candidates
    store.models_root_candidates = lambda: [ghost]         # type: ignore[assignment]
    try:
        check("根目录: 都不存在时返回**候选**而不是空串（文案才点得出该装到哪 ✓）",
              store.models_root() == ghost and store.is_available() is False,
              store.models_root())
    finally:
        store.models_root_candidates = original_candidates  # type: ignore[assignment]
    check("可用性: 根不存在/manifests 缺失 ⇒ False（不抛错）",
          store.is_available(os.path.join(STORE, "nope")) is False
          and store.is_available(os.path.dirname(BLOBS)) is True)

    # ================= blob 定位 =================
    check("blob: `sha256:xxx` ⇒ 落盘名是**短横** `sha256-xxx`（API 里是冒号，容易抄错）",
          store.blob_path(f"sha256:{LAYER_A}", STORE) is not None
          and store.blob_path(LAYER_A, STORE) is None)
    check("blob: 缺失/空摘要 ⇒ None（不抛错）",
          store.blob_path(f"sha256:{MISSING}", STORE) is None
          and store.blob_path("", STORE) is None and store.blob_path(None, STORE) is None)

    # ================= 列模型 =================
    models = store.list_models(STORE)
    by_name = {item["name"]: item for item in models}
    check("列模型: 四个 tag 都列出，坏清单与 .DS_Store **跳过**（不整片失败）",
          set(by_name) == {"qwen3:8b", "llama3:8b-instruct", "myorg/custom:v2", "broken:1"},
          sorted(by_name))
    check("名字: `library` 是默认命名空间 ⇒ **不显示**（qwen3:8b 而不是 library/qwen3:8b）",
          "qwen3:8b" in by_name and "library/qwen3:8b" not in by_name)
    check("名字: 非默认命名空间 ⇒ `myorg/custom:v2`（注册表主机名**没有**混进名字）",
          "myorg/custom:v2" in by_name
          and not any("registry.ollama.ai" in name for name in by_name))
    check("体积: 多个层**求和**且**不含 config 层**（1500 而不是 1500+88）",
          by_name["qwen3:8b"]["size"] == 1500, by_name["qwen3:8b"]["size"])
    check("体积: 共用同一 blob 的两个模型**各自求和**（不跨模型全局去重）",
          by_name["llama3:8b-instruct"]["size"] == 1500, by_name["llama3:8b-instruct"]["size"])
    check("体积: **盘上真实字节优先**（自述 2000 / 实测 1234 ⇒ 取 1234）",
          by_name["myorg/custom:v2"]["size"] == 1234, by_name["myorg/custom:v2"]["size"])
    check("体积: blob 缺失 ⇒ 回落自述值 + `missingBlobs=1`（⚠️ 清单在 ≠ 能推理 ✓）",
          by_name["broken:1"]["size"] == 7777 and by_name["broken:1"]["missingBlobs"] == 1,
          by_name["broken:1"])
    check("体积: 全部 blob 都在的模型 `missingBlobs=0`",
          by_name["qwen3:8b"]["missingBlobs"] == 0
          and by_name["qwen3:8b"]["blobs"] == 2, by_name["qwen3:8b"])
    check("体积: `size_label` 走 format_bytes（一位小数）",
          by_name["qwen3:8b"]["size_label"] == "1.5 KB", by_name["qwen3:8b"]["size_label"])
    check("来源: 每条都标 `source='disk'` + 清单路径（调用方能区分 service/disk ✓）",
          all(item["source"] == "disk" and item["path"].endswith(("8b", "v2", "1", "8b-instruct"))
              for item in models), [item["path"] for item in models])
    check("时间: `modified_at` 是毫秒 + Z 的形状（与 /api/tags 同形）",
          by_name["qwen3:8b"]["modified_at"].endswith("Z")
          and len(by_name["qwen3:8b"]["modified_at"]) == 24,
          by_name["qwen3:8b"]["modified_at"])
    check("摘要: `digest` 是 `sha256:<hex>`（清单文件自身的 sha256）",
          by_name["qwen3:8b"]["digest"].startswith("sha256:")
          and len(by_name["qwen3:8b"]["digest"]) == 71,
          by_name["qwen3:8b"]["digest"])
    check("排序: 按名字升序（同一份库每次跑顺序一致 ✓）",
          [item["name"] for item in models] == sorted(by_name), [item["name"] for item in models])

    check("边界: 根不存在/manifests 缺失 ⇒ 空列表且**不抛错**",
          store.list_models(os.path.join(STORE, "nope")) == []
          and store.list_models("") == [])

    empty_root = tempfile.mkdtemp(prefix="ollamastore_empty_")
    check("边界: manifests 在但里面没清单 ⇒ 空列表", store.list_models(empty_root) == [])
    shutil.rmtree(empty_root, ignore_errors=True)

    # ================= 路由回落（本套核心 ✓）=================
    client = TestClient(app)
    original_root = store.models_root
    original_reachable = ac.is_ollama_reachable
    original_exe = ac.find_ollama_exe

    async def _unreachable(_base_url: str) -> bool:
        return False

    store.models_root = lambda: STORE                      # type: ignore[assignment]
    ac.is_ollama_reachable = _unreachable                  # type: ignore[assignment]
    ac.find_ollama_exe = lambda: None                      # type: ignore[assignment]
    try:
        payload = client.post("/api/v1/ai-configs/ollama/status", json={}).json()["data"]
        check("路由: 服务不可达时 `running=False` **但仍然给出盘上的模型**（核心 ✓）",
              payload["running"] is False and len(payload["models"]) == 4,
              payload.get("models"))
        check("路由: 标了 `source='disk'` 与 `store_root`（前端可据此区分只读）",
              payload["source"] == "disk" and payload["store_root"] == os.path.abspath(STORE),
              {k: payload[k] for k in ("source", "store_root")})
        check("路由: 文案说清「服务没起」+「读到了几个」+「删除/推理仍需服务」（不误导 ✓）",
              "未运行" in payload["message"] and "4 个模型" in payload["message"]
              and "删除与推理仍需先启动服务" in payload["message"], payload["message"])
        check("路由: 盘上模型字段与 /api/tags 同形（name/size/size_label/modified_at/digest）",
              all({"name", "size", "size_label", "modified_at", "digest"} <= set(item)
                  for item in payload["models"]), payload["models"][0])

        # 盘上也没有 ⇒ 诚实说「没读到」，而不是含糊的空列表
        empty = tempfile.mkdtemp(prefix="ollamastore_none_")
        store.models_root = lambda: empty                  # type: ignore[assignment]
        blank = client.post("/api/v1/ai-configs/ollama/status", json={}).json()["data"]
        check("路由: 服务没起且盘上也没有 ⇒ 空列表 + 文案点明**查的是哪个目录**",
              blank["models"] == [] and "也没读到模型" in blank["message"]
              and empty in blank["message"], blank["message"])
        shutil.rmtree(empty, ignore_errors=True)
    finally:
        store.models_root = original_root                  # type: ignore[assignment]
        ac.is_ollama_reachable = original_reachable        # type: ignore[assignment]
        ac.find_ollama_exe = original_exe                  # type: ignore[assignment]

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(STORE, ignore_errors=True)
