"""厂商错误归因层自检（``app/services/vendor_errors.py``）。

这一层把上游的英文 JSON 归因成**用户可读中文** —— 归错了用户就无从下手，
所以每条分支都值得锁住。同时覆盖重试语义（哪些该重试、哪些该立即失败）。

运行::

    ./.venv/Scripts/python.exe tests/vendor_errors_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="vendor_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.services.vendor_errors import (  # noqa: E402
    _MODERATION_HINT,
    _UPLOADS_MODERATION_HINT,
    _clip_message,
    _is_retryable_network_error,
    _is_retryable_status,
    _strip_request_id_tail,
    fetch_with_retry,
    format_vendor_http_error,
    format_vendor_task_error,
    is_non_retryable_http_error,
    vendor_response_error,
)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _async(coro):
    return asyncio.run(coro)


def _raises_async(coro_factory):
    """跑协程并返回 (是否抛错, 异常信息)。"""
    try:
        _async(coro_factory())
    except Exception as err:  # noqa: BLE001
        return True, str(err)
    return False, None


def _mock_client(responses: list[tuple[int, str]]):
    """按顺序返回预置响应的 httpx 客户端；耗尽后重复最后一条。"""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        index = min(len(calls) - 1, len(responses) - 1)
        status, text = responses[index]
        return httpx.Response(status, text=text)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


def main() -> int:  # noqa: C901
    # ================= 内容审核归因 =================
    moderation_body = '{"error":{"code":"ContentModeration","message":"blocked by moderation"}}'
    check(
        "审核: video 走通用文案",
        format_vendor_http_error(400, moderation_body, "video") == _MODERATION_HINT["video"],
        format_vendor_http_error(400, moderation_body, "video")[:60],
    )
    check(
        "审核: 四种 kind 各用自己的文案",
        [format_vendor_http_error(400, moderation_body, k) for k in ("video", "image", "audio", "text")]
        == [_MODERATION_HINT[k] for k in ("video", "image", "audio", "text")],
    )
    check(
        "审核: body 提及 users-upload/people 时升级为「上传图含人物」专属文案",
        format_vendor_http_error(
            400, '{"error":{"code":"Moderation","message":"people-in-user-uploads"}}', "video"
        )
        == _UPLOADS_MODERATION_HINT["video"],
    )
    check(
        "审核: 火山 Seedance 的隐私审核标识也识别为上传图问题",
        format_vendor_http_error(
            400,
            '{"error":{"code":"InputImageSensitiveContentDetected.PrivacyInformation","message":"x"}}',
            "video",
        )
        == _UPLOADS_MODERATION_HINT["video"],
    )
    # ⚠️ 中文「人物」正则只在**已判定为审核**之后才参与细分 ——
    #    若 body 里没有任何审核信号（code/type/message 关键词），压根不会走审核分支。
    check(
        "审核: 命中审核 + 中文「图片包含人物」-> 走上传图分支",
        format_vendor_http_error(400, '{"error":{"code":"Moderation","message":"图片包含人物，请更换"}}', "image")
        == _UPLOADS_MODERATION_HINT["image"],
        format_vendor_http_error(400, '{"error":{"code":"Moderation","message":"图片包含人物，请更换"}}', "image")[:40],
    )
    check(
        "审核: 只有中文「人物」而没有审核信号 -> **不**走审核分支（会被当成普通错误）",
        format_vendor_http_error(400, '{"error":{"message":"图片包含人物，请更换"}}', "image")
        == "图片生成未成功：图片包含人物，请更换",
        format_vendor_http_error(400, '{"error":{"message":"图片包含人物，请更换"}}', "image"),
    )
    check(
        "审核: 命中审核但未提及上传图 -> 通用文案（两者不能混）",
        format_vendor_http_error(400, '{"error":{"message":"内容审核未通过"}}', "image")
        == _MODERATION_HINT["image"],
    )
    check(
        "审核: safety + filter 组合也识别",
        format_vendor_http_error(400, '{"error":{"message":"blocked by safety filter system"}}', "text")
        == _MODERATION_HINT["text"],
    )

    # ================= 认证 / 权限 / 额度 / 5xx =================
    check(
        "状态码: 401/403/429/500/502/503/504 各自的中文文案",
        format_vendor_http_error(401, "", "video").startswith("API Key 无效")
        and format_vendor_http_error(403, "", "video").startswith("没有权限调用")
        and format_vendor_http_error(429, "", "video").startswith("请求过于频繁")
        and format_vendor_http_error(500, "", "video") == "当前请求较多或服务暂时异常，请稍后重试。"
        and format_vendor_http_error(504, "", "video") == "当前请求较多或服务暂时异常，请稍后重试。",
    )
    check(
        "状态码: 401 优先于 body 内容（认证问题不该被 body 带偏）",
        format_vendor_http_error(401, '{"error":{"message":"some weird reason"}}', "video")
        == "API Key 无效或未通过校验，请在 AI 服务配置中检查密钥是否正确。",
    )

    # ================= 模型与接口不匹配 =================
    check(
        "模型: seedream（图片模型）用在视频 -> 指名叫用户换 Seedance/H3",
        "Seedream" in format_vendor_http_error(
            400, '{"error":{"code":"InvalidParameter","message":"model seedream-3 does not support this api"}}', "video"
        ),
        format_vendor_http_error(
            400, '{"error":{"code":"InvalidParameter","message":"model seedream-3 does not support this api"}}', "video"
        ),
    )
    # ⚠️ 触发「模型不匹配」分支的判据是 **`not found`**：`does not exist` 这种措辞
    #    **不**命中（原 TS 如此），会掉到普通 message 分支。
    check(
        "模型: 'not found' + 可提取模型名 -> 带上模型名",
        format_vendor_http_error(
            400, '{"error":{"message":"requested model gpt-9 not found"}}', "video"
        )
        == "找不到模型「gpt-9」，请核对 AI 服务配置中的模型 ID 是否填写正确。",
        format_vendor_http_error(400, '{"error":{"message":"requested model gpt-9 not found"}}', "video"),
    )
    check(
        "模型: 'does not exist' **不**算 not found（措辞敏感，保真）",
        format_vendor_http_error(
            400, '{"error":{"message":"requested model gpt-9 does not exist"}}', "video"
        )
        == "视频生成未成功：requested model gpt-9 does not exist",
        format_vendor_http_error(400, '{"error":{"message":"requested model gpt-9 does not exist"}}', "video"),
    )
    check(
        "模型: not found 但提取不到名字 -> 通用提示",
        format_vendor_http_error(400, '{"error":{"message":"Not Found"}}', "video")
        == "找不到所填写的模型名称，请核对上游平台文档中的模型 ID 是否填写正确。",
        format_vendor_http_error(400, '{"error":{"message":"Not Found"}}', "video"),
    )
    check(
        "模型: 有模型名但非 not found -> 「不支持当前接口」",
        format_vendor_http_error(
            400, '{"error":{"code":"InvalidParameter","message":"model `wan2.6` is not supported"}}', "image"
        ).startswith("模型「"),
    )
    check(
        "模型: 400 + 命中不匹配但**提取不到**模型名 -> 通用提示",
        format_vendor_http_error(
            400, '{"error":{"message":"parameter does not support this api"}}', "video"
        )
        == "当前选择的模型与该接口不匹配。请检查 AI 服务配置中的「模型」是否为对应类型的模型（勿将图片模型填在视频栏）。",
        format_vendor_http_error(400, '{"error":{"message":"parameter does not support this api"}}', "video"),
    )
    # ⚠️ 模型名提取正则很松：`model[`:：\s]+(...)` 会把 "model is not a valid model" 里的
    #    `is` 当成模型名 ⇒ 报出「模型「is」不支持…」。这是原实现的行为，只能照实保真。
    check(
        "模型: 模型名正则很松（'model is …' 提取出 'is'）—— 保真，别「顺手修好」",
        format_vendor_http_error(
            400, '{"error":{"message":"the parameter model is not a valid model"}}', "video"
        )
        == "模型「is」不支持当前接口。请在 AI 服务配置中更换为支持该类型生成的模型。",
        format_vendor_http_error(400, '{"error":{"message":"the parameter model is not a valid model"}}', "video"),
    )

    # ================= 缺参数 / 兜底分支 =================
    check(
        "缺参数: 400 + prompt/parameter + required",
        format_vendor_http_error(400, '{"error":{"message":"prompt is required"}}', "video")
        == "缺少必填参数（如提示词或参考图），请检查任务参数是否完整。",
    )
    check(
        "兜底: message 是 JSON 字符串开头 -> 不贴原文",
        format_vendor_http_error(400, '{"message":"{not-really}"}', "video")
        == "服务返回了异常数据，请稍后重试或联系平台。",
        format_vendor_http_error(400, '{"message":"{not-really}"}', "video"),
    )
    check(
        "兜底: 长英文 message（>80 且无中文）-> 归因成配置类建议",
        format_vendor_http_error(400, f'{{"message":"{"a" * 100}"}}', "video")
        == "服务拒绝了本次请求，常见原因：模型名称错误、该模型不支持当前接口、或参数不符合平台要求。请在 AI 服务配置中核对「模型」是否正确。",
    )
    check(
        "兜底: 短中文 message -> 前缀「视频生成未成功：」",
        format_vendor_http_error(400, '{"message":"今日额度已用尽"}', "video")
        == "视频生成未成功：今日额度已用尽",
        format_vendor_http_error(400, '{"message":"今日额度已用尽"}', "video"),
    )
    check(
        "兜底: kind 决定前缀用词（图片/语音/文本）",
        format_vendor_http_error(400, '{"message":"额度不足"}', "image") == "图片生成未成功：额度不足"
        and format_vendor_http_error(400, '{"message":"额度不足"}', "audio") == "语音生成未成功：额度不足"
        and format_vendor_http_error(400, '{"message":"额度不足"}', "text") == "文本生成未成功：额度不足",
    )
    check(
        "兜底: 非 JSON body + 非 5xx -> 走纯文本分支",
        format_vendor_http_error(418, "I am a teapot", "video").startswith("服务暂时无法完成请求（HTTP 418）。说明：I am a teapot"),
        format_vendor_http_error(418, "I am a teapot", "video"),
    )
    check(
        "兜底: 完全空的 body -> 最终兜底句",
        format_vendor_http_error(400, "", "video") == "视频生成失败（HTTP 400），请检查网络、API Key 与模型配置后重试。",
        format_vendor_http_error(400, "", "video"),
    )

    # ================= 文本处理细节 =================
    check(
        "细节: Request id 尾巴被剥掉",
        _strip_request_id_tail("请求被拒绝 Request id: a1b2c3d4e5f6 extra junk") == "请求被拒绝",
        _strip_request_id_tail("请求被拒绝 Request id: a1b2c3d4e5f6 extra junk"),
    )
    check(
        "细节: clipMessage 截断后去掉半个词再加省略号",
        _clip_message("x" * 200) == "x" * 117 + "…",
        repr(_clip_message("x" * 200)),
    )
    check(
        "细节: clipMessage 恰好等于上限时不截断",
        _clip_message("y" * 120) == "y" * 120,
    )
    check(
        "细节: clipMessage 去掉截断处的半个词（'word' 被吃掉）",
        _clip_message("a" * 110 + " " + "b" * 20).endswith("…")
        and not _clip_message("a" * 110 + " " + "b" * 20).rstrip("…").endswith("b"),
        repr(_clip_message("a" * 110 + " " + "b" * 20)),
    )

    # ================= 轮询阶段的 error 归因 =================
    check(
        "taskError: 无 error 字段 -> 兜底",
        format_vendor_task_error(None, "video") == "视频生成失败，请稍后重试。",
        format_vendor_task_error(None, "video"),
    )
    check(
        "taskError: 字符串（非 JSON）-> 直接作为 message",
        format_vendor_task_error("任务超时", "video") == "视频生成失败：任务超时",
        format_vendor_task_error("任务超时", "video"),
    )
    check(
        "taskError: JSON 字符串 -> 解析出 message",
        format_vendor_task_error('{"error":{"code":"E1","message":"磁盘满了"}}', "video") == "视频生成失败：磁盘满了",
        format_vendor_task_error('{"error":{"code":"E1","message":"磁盘满了"}}', "video"),
    )
    check(
        "taskError: 对象 {code,message} -> message 优先",
        format_vendor_task_error({"code": "InvalidParameter", "message": "参数不对"}, "video")
        == "视频生成失败：参数不对",
    )
    check(
        "taskError: 嵌套 {error:{code,message}} 也认得（火山 Seedance 结构）",
        format_vendor_task_error({"error": {"code": "X", "message": "嵌套原因"}}, "video") == "视频生成失败：嵌套原因",
    )
    # ⚠️ message 为空时 rawText 会变成 JSON 串，于是被「以 { 开头」的分支拦下 ⇒
    #    这里**不是**「视频生成失败：E9」，而是那句「请检查提示词与参考图」。
    check(
        "taskError: 只有 code 没有 message -> rawText 是 JSON 串 -> 走「以 { 开头」分支",
        format_vendor_task_error({"code": "E9"}, "video") == "视频生成失败，请稍后重试或检查提示词与参考图。",
        format_vendor_task_error({"code": "E9"}, "video"),
    )
    check(
        "taskError: 既无 code 也无 message 的对象 -> 同上",
        format_vendor_task_error({"foo": 1}, "video") == "视频生成失败，请稍后重试或检查提示词与参考图。",
        format_vendor_task_error({"foo": 1}, "video"),
    )
    check(
        "taskError: 对象内的审核信息 -> 归因成中文审核文案（不再 [object Object]）",
        format_vendor_task_error({"code": "Moderation", "message": "sensitive content detected"}, "video")
        == _MODERATION_HINT["video"],
    )
    # 同样：得先有审核信号（code=Moderation），"PrivacyInformation" 才会被当作上传图细分
    check(
        "taskError: 审核信号 + 上传图标识 -> 专属文案",
        format_vendor_task_error({"code": "Moderation", "message": "PrivacyInformation"}, "image")
        == _UPLOADS_MODERATION_HINT["image"],
        format_vendor_task_error({"code": "Moderation", "message": "PrivacyInformation"}, "image")[:40],
    )
    check(
        "taskError: 只有上传图标识、没有审核信号 -> 当成普通错误",
        format_vendor_task_error({"message": "PrivacyInformation"}, "image") == "图片生成失败：PrivacyInformation",
        format_vendor_task_error({"message": "PrivacyInformation"}, "image"),
    )
    check(
        "taskError: cleaned 以 { 开头 -> 不贴原文",
        format_vendor_task_error({"message": "{oops}"}, "video") == "视频生成失败，请稍后重试或检查提示词与参考图。",
    )
    check(
        "taskError: 长英文 -> 归因成配置类建议（不带原文）",
        format_vendor_task_error({"message": "z" * 100}, "video")
        == "视频生成失败，请稍后重试或检查提示词、参考图与模型配置。",
    )
    check(
        "taskError: 数字等非对象非字符串 -> 最终兜底",
        format_vendor_task_error(42, "video") == "视频生成失败，请稍后重试。",
        format_vendor_task_error(42, "video"),
    )

    # ================= 可重试判定 =================
    check(
        "retryable: 429/5xx 可重试，401/403/400 不可",
        _is_retryable_status(429) and _is_retryable_status(500) and _is_retryable_status(503)
        and not _is_retryable_status(400) and not _is_retryable_status(401),
    )
    check(
        "nonRetryable: 401/403/429 立即失败",
        is_non_retryable_http_error(401, "") and is_non_retryable_http_error(403, "")
        and is_non_retryable_http_error(429, ""),
    )
    check(
        "nonRetryable: 400 + 审核 -> 立即失败（不空转）",
        is_non_retryable_http_error(400, '{"error":{"message":"内容审核未通过"}}'),
    )
    check(
        "nonRetryable: 400 + 参数/模型类 -> 立即失败",
        is_non_retryable_http_error(400, '{"error":{"code":"InvalidParameter","message":"x"}}')
        and is_non_retryable_http_error(400, '{"error":{"message":"prompt required"}}'),
    )
    check(
        "nonRetryable: 400 + 无关原因 -> 仍可重试",
        not is_non_retryable_http_error(400, '{"error":{"message":"transient glitch"}}'),
    )
    check(
        "nonRetryable: 404/5xx 可重试（任务未就绪 / 瞬时）",
        not is_non_retryable_http_error(404, "") and not is_non_retryable_http_error(503, ""),
    )

    # ================= 网络异常判定 =================
    check(
        "netError: httpx 的传输/超时异常一律可重试（类型判定为主）",
        _is_retryable_network_error(httpx.ConnectError("boom"))
        and _is_retryable_network_error(httpx.ReadTimeout("slow"))
        and _is_retryable_network_error(httpx.ConnectTimeout("slow")),
    )
    check(
        "netError: 文案里带 timeout 的普通异常也可重试（与 Node 对齐）",
        _is_retryable_network_error(RuntimeError("request timeout")),
    )
    check(
        "netError: 归因好的中文错误不会被误判为可重试",
        not _is_retryable_network_error(ValueError(_MODERATION_HINT["video"])),
    )

    # ================= 重试流程（MockTransport，零真实网络） =================
    def retry_flow():
        async def run():
            client, calls = _mock_client([(503, "unavailable"), (503, "unavailable"), (200, '{"ok":true}')])
            retries = []
            resp = await fetch_with_retry(
                "https://vendor.test/x",
                {"method": "POST", "headers": {"Content-Type": "application/json"}, "body": "{}"},
                "image",
                max_retries=3,
                base_delay_ms=0,
                on_retry=lambda attempt, delay, reason: retries.append((attempt, delay, reason)),
                client=client,
            )
            await client.aclose()
            return resp.status_code, len(calls), retries

        return asyncio.run(run())

    status, call_count, retries = retry_flow()
    check(
        "fetchWithRetry: 503,503,200 -> 成功且重试 2 次（退避 0/0，attempt 从 1 起）",
        status == 200 and call_count == 3 and [r[0] for r in retries] == [1, 2],
        f"{status} {call_count} {retries}",
    )

    def retry_exhausted():
        async def run():
            client, calls = _mock_client([(503, "unavailable")])
            try:
                await fetch_with_retry("https://vendor.test/x", {"method": "GET"}, "video",
                                       max_retries=3, base_delay_ms=0, client=client)
            except Exception as err:  # noqa: BLE001
                await client.aclose()
                return str(err), len(calls)
            return None, len(calls)

        return asyncio.run(run())

    message, call_count = retry_exhausted()
    check(
        "fetchWithRetry: 一直 503 -> 用尽重试后抛归因好的中文",
        message == "当前请求较多或服务暂时异常，请稍后重试。" and call_count == 3,
        f"{message} {call_count}",
    )

    def no_retry_on_401():
        async def run():
            client, calls = _mock_client([(401, '{"error":{"message":"bad key"}}')])
            try:
                await fetch_with_retry("https://vendor.test/x", {"method": "GET"}, "text",
                                       max_retries=3, base_delay_ms=0, client=client)
            except Exception as err:  # noqa: BLE001
                await client.aclose()
                return str(err), len(calls)
            return None, len(calls)

        return asyncio.run(run())

    message, call_count = no_retry_on_401()
    check(
        "fetchWithRetry: 401 不重试（次数 1）且是中文归因",
        call_count == 1 and message == "API Key 无效或未通过校验，请在 AI 服务配置中检查密钥是否正确。",
        f"{message} {call_count}",
    )

    def network_retry():
        async def run():
            attempts = []

            def handler(request: httpx.Request) -> httpx.Response:
                attempts.append(request)
                if len(attempts) < 3:
                    raise httpx.ConnectError("connection refused")
                return httpx.Response(200, text="{}")

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            resp = await fetch_with_retry("https://vendor.test/x", {"method": "GET"}, "image",
                                          max_retries=3, base_delay_ms=0, client=client)
            await client.aclose()
            return resp.status_code, len(attempts)

        return asyncio.run(run())

    status, attempts = network_retry()
    check(
        "fetchWithRetry: 网络层异常（ConnectError）也会退避重试",
        status == 200 and attempts == 3,
        f"{status} {attempts}",
    )

    def body_empty_400():
        async def run():
            client, calls = _mock_client([(400, '{"error":{"message":"prompt is required"}}')])
            try:
                await fetch_with_retry("https://vendor.test/x", {"method": "GET"}, "video",
                                       max_retries=3, base_delay_ms=0, client=client)
            except Exception as err:  # noqa: BLE001
                await client.aclose()
                return str(err), len(calls)
            return None, len(calls)

        return asyncio.run(run())

    message, call_count = body_empty_400()
    check(
        "fetchWithRetry: 400 + 缺参数 -> 不重试、直接中文归因",
        call_count == 1 and message == "缺少必填参数（如提示词或参考图），请检查任务参数是否完整。",
        f"{message} {call_count}",
    )

    def response_error_helper():
        async def run():
            resp = httpx.Response(403, text="nope")
            err = await vendor_response_error(resp, "audio")
            return str(err)

        return asyncio.run(run())

    check(
        "vendorResponseError: 读 body 并归因（403 -> 无权限）",
        response_error_helper() == "没有权限调用该能力，请检查账号是否已开通对应模型或接口。",
        response_error_helper(),
    )

    # ================= 与原 TS 的**有意差异** =================
    check(
        "健壮性: 厂商返回**数字** code 时不崩（原 TS 会在此处抛 TypeError）",
        format_vendor_http_error(400, '{"error":{"code":404,"message":"找不到"}}', "video") == "视频生成未成功：找不到",
        format_vendor_http_error(400, '{"error":{"code":404,"message":"找不到"}}', "video"),
    )
    check(
        "健壮性: 顶层数字 code 同样可用",
        isinstance(format_vendor_http_error(400, '{"code":500,"message":"内部错误"}', "text"), str),
    )

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
