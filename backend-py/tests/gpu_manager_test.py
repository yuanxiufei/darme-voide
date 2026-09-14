"""S7 自检：GPU 显存管理器（``gpu-manager.ts`` 473 行）+ 两条端点。

覆盖：本地判定 / 显存账本与舍入口径 / **互斥锁 + FIFO 队列** / 四种卸载策略的**返回值语义**
/ 释放分流（立即卸载 vs 打时间戳等 TTL）/ **超额驱逐**（从小到大、跳过自己、带冷却）
/ ``release_all`` / ``nvidia-smi`` 探测 / ``GET /gpu/status``（**裸 JSON**）与 ``release-all``。

⚠️ 全部打桩 ``_post_json``（卸载请求不发真网络）、并把涉及的 ``cooldownMs`` 临时改 0
（原值 1~5 秒，否则真实等待会拖慢自检）。

运行::

    ./.venv/Scripts/python.exe tests/gpu_manager_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="gpu_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Windows 控制台默认 GBK：检查名里带 emoji 时**打印阶段**会 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import gpu_manager as gm  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_POSTED: list[dict[str, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


async def fake_post(url: str, payload: object, timeout: float) -> int:
    _POSTED.append({"url": url, "payload": payload, "timeout": timeout})
    return int(os.environ.get("STUB_STATUS", "200"))


def main() -> int:  # noqa: C901
    gm._post_json = fake_post  # type: ignore[assignment]
    manager = gm.gpu_manager

    # 冷却改 0：真实值是 1~5 秒，只影响「给服务端缓冲」的等待，不改判定
    original_cooldowns = {k: v.get("cooldownMs") for k, v in gm.VRAM_ESTIMATES.items()}
    for profile in gm.VRAM_ESTIMATES.values():
        profile["cooldownMs"] = 0

    # ================= 本地判定 =================
    check("本地: provider 白名单（ollama / local-sd / cosyvoice）",
          gm.is_local_provider("OLLAMA") and gm.is_local_provider("local-sd")
          and gm.is_local_provider("cosyvoice") and not gm.is_local_provider("chatfire"))
    check("本地: baseUrl 命中 localhost/127.0.0.1/192.168.* —— 任意 provider 都算本地",
          gm.is_local_config("http://localhost:11434", "openai")
          and gm.is_local_config("http://127.0.0.1:7860", "minimax")
          and gm.is_local_config("http://192.168.1.9:8765", "openai")
          and not gm.is_local_config("https://api.chatfire.site", "chatfire")
          and not gm.is_local_config("", "chatfire"))
    check("本地: 白名单优先级高于 baseUrl（云端地址 + ollama provider 仍算本地）",
          gm.is_local_config("https://api.chatfire.site", "ollama"))

    # ================= 显存账本 =================
    check("账本: provider:model 命中登记表，未登记落 default（8GB / none）",
          gm.VRAM_ESTIMATES["ollama:qwen3:14b"]["vramGB"] == 9
          and manager.get_profile("ollama", "qwen3:14b")["vramGB"] == 9
          and manager.get_profile("whatever", "unknown")["vramGB"] == 8
          and manager.get_profile("whatever", "unknown")["unloadStrategy"] == "none",
          manager.get_profile("whatever", "unknown"))
    check("账本: 总显存 24 / 安全余量 2 / 初始可用 22（`Math.round(x*10)/10`）",
          manager.get_status() == {
              "totalVRAM_GB": 24, "safeMargin_GB": 2, "usedVRAM_GB": 0,
              "availableVRAM_GB": 22, "isLocked": False, "holder": None,
              "queueLength": 0, "loadedModels": []},
          manager.get_status())

    async def scenario() -> None:
        # ---- 申请 / 释放：ollama-keep-alive 走**立即卸载** ----
        _POSTED.clear()
        lease = await manager.acquire("text", "ollama", "qwen3:14b", "http://localhost:11434")
        check("acquire: 返回租约（serviceType/modelKey/vramGB）+ 上锁 + 记账",
              (lease.serviceType, lease.modelKey, lease.vramGB) == ("text", "ollama:qwen3:14b", 9)
              and manager.is_locked and manager.current_holder == "ollama:qwen3:14b"
              and manager.total_vram_gb == 9.0 and manager.available_vram_gb == 13.0,
              manager.get_status())
        check("acquire: 用的是配置里登记的卸载策略（baseUrl 覆盖进 profile）",
              manager._loaded["ollama:qwen3:14b"]["profile"]["baseUrl"] == "http://localhost:11434")
        lease.release()
        await asyncio.sleep(0.05)  # 让后台卸载任务跑完
        check("release(ollama): 立刻发 keep_alive=0 卸载请求（prompt='.'/max_tokens=1/stream=False）",
              len(_POSTED) == 1 and _POSTED[0]["url"] == "http://localhost:11434/api/generate"
              and _POSTED[0]["payload"] == {"model": "qwen3", "prompt": ".",
                                            "keep_alive": 0, "max_tokens": 1, "stream": False},
              _POSTED)
        check("release(ollama): 卸载后**立即从账本移除** + 解锁",
              manager.loaded_model_keys == [] and not manager.is_locked
              and manager.available_vram_gb == 22.0)

        # ---- sd / passive：只打时间戳，等 TTL ----
        _POSTED.clear()
        sd = await manager.acquire("image", "local-sd", "sdxl-base", "http://localhost:7860")
        sd.release()
        await asyncio.sleep(0)
        check("release(sd): **不发卸载请求**，保留复用并记录 releasedAt",
              _POSTED == [] and manager.loaded_model_keys == ["local-sd:sdxl-base"]
              and manager._loaded["local-sd:sdxl-base"]["releasedAt"] is not None
              and manager.total_vram_gb == 12.0)

        # ---- 驱逐：从小到大、跳过自己、释放到够用为止 ----
        _POSTED.clear()
        heavy = await manager.acquire("video", "minimax", "hailuo-02", "http://localhost:8765")
        check("驱逐: 12 + 15 > 22 触发驱逐 -> 卸掉 12GB 的 sdxl（够 5GB 需求即停）",
              manager.loaded_model_keys == ["minimax:hailuo-02"]
              and any("/sdapi/v1/unload-checkpoint" in str(p["url"]) for p in _POSTED),
              (_POSTED, manager.loaded_model_keys))
        check("驱逐: 新模型自己**不在**候选里（不会被误卸）", heavy.modelKey == "minimax:hailuo-02")
        heavy.release()
        await asyncio.sleep(0.05)
        check("release(comfyui-free): 打到 ComfyUI /free（unload_models+free_memory）",
              any(str(p["url"]).endswith("/free") for p in _POSTED)
              and any(p["payload"] == {"unload_models": True, "free_memory": True} for p in _POSTED),
              _POSTED)
        check("release(comfyui-free): 同样立即移除 + 解锁", manager.loaded_model_keys == [])

        # ---- 队列：持锁期间的第二次申请必须等待 ----
        first = await manager.acquire("audio", "cosyvoice", "cosyvoice-v2")
        waiter = asyncio.get_running_loop().create_task(
            manager.acquire("text", "ollama", "qwen3:8b"))
        await asyncio.sleep(0.05)
        check("队列: 锁被占用时第二个申请**排队等待**（不并发进入）",
              not waiter.done() and manager.get_status()["queueLength"] == 1,
              manager.get_status())
        check("队列: 等待者确实还没拿到锁（holder 仍是第一个）",
              manager.current_holder == "cosyvoice:cosyvoice-v2")
        first.release()
        await asyncio.sleep(0.05)
        awaited = await asyncio.wait_for(waiter, 2)
        check("队列: 释放后**FIFO 唤醒**等待者并拿到锁",
              awaited.modelKey == "ollama:qwen3:8b"
              and manager.current_holder == "ollama:qwen3:8b"
              and manager.get_status()["queueLength"] == 0, manager.get_status())
        # 收尾：passive 的 cosyvoice 还在账本里（等 TTL）
        awaited.release()
        await asyncio.sleep(0.05)
        check("账本: 三种策略的残留状态符合预期（passive 留着、ollama 已移除）",
              manager.loaded_model_keys == ["cosyvoice:cosyvoice-v2"],
              manager.loaded_model_keys)

        # ---- TTL 过期清理 ----
        entry = manager._loaded["cosyvoice:cosyvoice-v2"]
        entry["releasedAt"] = entry["releasedAt"] - gm.MODEL_REUSE_TTL_MS - 1000
        manager.prune_expired()
        check("TTL: 超过 10 分钟未再使用 -> prune 掉（避免显存估算只增不减）",
              manager.loaded_model_keys == [] and manager.available_vram_gb == 22.0)

        # ---- release_all ----
        # ⚠️ 互斥锁下正常**拿不到两个租约**（第二个会排队死等）⇒ 用「人为解锁」制造
        #    「账本里有多个模型」的现场 —— 这正是 release_all（调试用）要处理的场景。
        _POSTED.clear()
        await asyncio.wait_for(
            manager.acquire("text", "ollama", "qwen3:14b", "http://localhost:11434"), 2)
        manager._holder = None  # 模拟「锁丢在别人手里」
        await asyncio.wait_for(
            manager.acquire("image", "local-sd", "sdxl-base", "http://localhost:7860"), 2)
        check("release_all 前置: 账本里同时有两个模型（9 + 12 GB）",
              manager.total_vram_gb == 21.0, manager.get_status())
        await manager.release_all()
        check("release_all: **逐个卸载**（两类请求各一发）+ 清空账本 + 解锁",
              manager.loaded_model_keys == [] and not manager.is_locked
              and len(_POSTED) == 2
              and any("/api/generate" in str(p["url"]) for p in _POSTED)
              and any("/sdapi/v1/unload-checkpoint" in str(p["url"]) for p in _POSTED),
              _POSTED)

        # ---- 卸载失败的返回值语义 ----
        os.environ["STUB_STATUS"] = "500"
        comfy = dict(gm.VRAM_ESTIMATES["minimax:hailuo-02"])
        check("卸载: comfyui-free 非 2xx -> **False**（驱逐据此决定算不算腾出显存）",
              await manager.unload_model("minimax:hailuo-02", comfy) is False)
        os.environ.pop("STUB_STATUS", None)
        check("卸载: comfyui-free 2xx -> True；passive -> True（不发请求）",
              await manager.unload_model("minimax:hailuo-02", comfy) is True
              and await manager.unload_model("x", {"unloadStrategy": "passive"}) is True)
        check("卸载: 未登记策略（none）-> False", 
              await manager.unload_model("x", {"unloadStrategy": "none"}) is False)

    asyncio.run(scenario())

    # ================= nvidia-smi =================
    smi = asyncio.run(gm.get_nvidia_smi())
    check("nvidia-smi: 不抛错，返回 None 或 6 字段硬件信息（本机有无驱动都要稳）",
          smi is None or set(smi) == {"gpuName", "totalMemoryMB", "usedMemoryMB",
                                      "freeMemoryMB", "utilizationPercent", "temperatureC"},
          smi)

    # ================= 端点 =================
    client = TestClient(app)
    status_resp = client.get("/api/v1/ai-configs/gpu/status")
    body = status_resp.json()
    check("端点: GET /gpu/status 是**裸 JSON**（不是 {code,data,message} 信封）",
          status_resp.status_code == 200 and "code" not in body and "data" not in body
          and body["totalVRAM_GB"] == 24 and body["safeMargin_GB"] == 2,
          list(body))
    check("端点: 附带 isLocalMode / hardware（无驱动时为 false + null）",
          isinstance(body["isLocalMode"], bool)
          and (body["hardware"] is None or "gpuName" in body["hardware"]), body.get("hardware"))
    release_all_body = client.post("/api/v1/ai-configs/gpu/release-all").json()
    check("端点: release-all 回**信封** {message, status}",
          release_all_body == {"code": 200, "message": "success",
                               "data": {"message": "All GPU models released",
                                        "status": gm.gpu_manager.get_status()}},
          release_all_body)

    for key, value in original_cooldowns.items():
        gm.VRAM_ESTIMATES[key]["cooldownMs"] = value

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
