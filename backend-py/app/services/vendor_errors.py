"""上游厂商错误 → 用户可读中文的统一映射层。

移植自 ``backend/src/utils/vendor-errors.ts``（362 行）。

设计动机（对齐参照项目 gcc-printfilm-main 的 ``videoHttpErrors.ts`` 思路）：
**不再把上游返回的英文 JSON 直接透传给前端**，而是按 HTTP status + body 归因成
「用户看得懂、能行动」的中文说明。重点覆盖两类原先被笼统处理的高频错误：

1. **内容审核拦截** —— 尤其「上传图/首帧被识别为真实人物」这类与文字描述无关的归因
   （火山引擎 Seedance 的 ``InputImageSensitiveContentDetected.PrivacyInformation`` 等）；
2. **模型与接口不匹配** —— 图片模型填在视频栏、模型名拼错等。

同时提供 :func:`is_non_retryable_http_error` 供轮询循环判定：哪些错误应立即判失败、
哪些（5xx / 网络抖动）应继续重试，**避免对审核/参数错误空转数十分钟**。

保真说明（与原 TS 的差异，均为**更稳**的方向）
------------------------------------------------
* 原 TS 在 ``((apiErr?.code || parsed?.topCode) || '').toLowerCase()`` 处，
  若厂商返回**数字** code（``{"error":{"code":404}}``）会抛 ``TypeError``
  把整个归因流程打断。这里统一 ``str()`` 归一 —— 只会得到一条中文错误，
  不会因为上游字段类型不规范而崩掉。
* ``$`` 断言在 JS（非 multiline）只匹配**字符串末尾**，而 Python 的 ``$`` 还会匹配
  结尾换行前的位置 ⇒ 本模块一律用 ``\\Z``。
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

import httpx

__all__ = [
    "VendorErrorKind",
    "aclose_client",
    "fetch_with_retry",
    "format_vendor_http_error",
    "format_vendor_task_error",
    "is_non_retryable_http_error",
    "vendor_response_error",
]

VendorErrorKind = Literal["video", "image", "audio", "text"]

_KIND_LABEL: dict[str, str] = {
    "video": "视频",
    "image": "图片",
    "audio": "语音",
    "text": "文本",
}

#: 各场景的「内容审核拦截」通用文案
_MODERATION_HINT: dict[str, str] = {
    "video": "内容安全拦截：该视频提示词或参考图可能包含不安全内容。请编辑分镜视频提示词，避免暴力、血腥、敏感描述，或更换参考图后重试。",
    "image": "内容安全拦截：该图片提示词或参考图可能包含不安全内容。请编辑关键帧/角色提示词，避免暴力、血腥、敏感描述后重试。",
    "audio": "内容安全拦截：该配音文本可能包含不安全内容。请修改台词文本后重试。",
    "text": "内容安全拦截：该提示词可能包含不安全内容。请修改后重试。",
}

#: 各场景的「上传图含人物触发审核」专属文案
_UPLOADS_MODERATION_HINT: dict[str, str] = {
    "video": "视频生成被平台内容审核拦截：原因指向「上传的参考图/首帧」（如角色图、分镜图）。平台会单独审核上传画面，与文字描述无直接关系。可尝试：去掉或更换参考图、避免画面中清晰可辨识的真实人物，或改为纯文生视频后重试。",
    "image": "图片生成被平台内容审核拦截：原因指向「上传的参考图」。可尝试：去掉或更换参考图、避免画面中清晰可辨识的真实人物后重试。",
    "audio": "语音生成被平台内容审核拦截。请检查台词语音文本后重试。",
    "text": "生成被平台内容审核拦截。请检查提示词后重试。",
}

_MODERATION_TEXT_RE = re.compile(r"不安全|违规|内容审核|敏感内容|风控|内容安全")
_UPLOADS_TEXT_RE = re.compile(
    r"图片.*(?:人物|人像|隐私|敏感)|(?:人物|人像).*图片|上传.*(?:人物|人像)|(?:人物|人像).*上传"
)
_REQUEST_ID_RE = re.compile(r"\s*Request id:\s*[a-f0-9]+.*\Z", re.IGNORECASE)
_HAS_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_MODEL_NAME_PATTERNS = (
    re.compile(r"requested model\s+([^\s.]+)", re.IGNORECASE),
    re.compile(r"model[`:：\s]+([a-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"`model`\s*[^:]*:\s*([a-z0-9_.-]+)", re.IGNORECASE),
)


def _as_str(value: Any) -> str:
    """镜像 JS 的 ``String(x)``（`None` → ``''``，其余照转）。"""
    return "" if value is None else str(value)


def _first_not_none(*values: Any) -> Any:
    """镜像 ``a ?? b ?? c``（**只看 None**，`0`/`False`/`''` 都算有值）。"""
    for value in values:
        if value is not None:
            return value
    return None


def _body_suggests_people_in_user_uploads(text: str) -> bool:
    """识别「上传图含人物」类审核（与文字提示词无直接关系）。"""
    lowered = (text or "").lower()
    return (
        "people-in-user-uploads" in lowered
        or "people in user uploads" in lowered
        or ("user-upload" in lowered and "people" in lowered and "moderation" in lowered)
        # 火山引擎 Seedance 隐私审核：首帧/参考图被识别为真实人物照片
        or "inputimagesensitivecontentdetected" in lowered
        or "input image sensitive content" in lowered
        or "privacyinformation" in lowered
        or "privacy information" in lowered
        or ("sensitive" in lowered and "image" in lowered
            and ("detected" in lowered or "content" in lowered))
        or _UPLOADS_TEXT_RE.search(text or "") is not None
    )


def _looks_like_content_moderation(
    api_err: dict[str, Any] | None, raw_text: str, code: str | None = None
) -> bool:
    """识别「内容审核拦截」类错误（含文字与图片）。"""
    api_err = api_err or {}
    message = (_as_str(api_err.get("message")) or raw_text or "").lower()
    combined_code = (_as_str(code) or _as_str(api_err.get("code"))).lower()
    err_type = _as_str(api_err.get("type")).lower()

    if "moderation" in combined_code or "moderation" in err_type or "moderation" in message:
        return True
    if "content_policy" in message or "content policy" in message:
        return True
    if "safety" in message and ("filter" in message or "system" in message):
        return True
    if "policy" in message and "violation" in message:
        return True
    if "sensitive content" in message or "sensitivecontent" in combined_code:
        return True
    return _MODERATION_TEXT_RE.search(_as_str(api_err.get("message")) or raw_text or "") is not None


def _parse_error_body(text: str) -> dict[str, Any] | None:
    """``JSON.parse`` 后取 ``error`` / 顶层 ``code`` / 顶层 ``message``；非对象返回 None。"""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    parsed_err: dict[str, Any] | None = None
    if isinstance(err, dict):
        parsed_err = {
            "message": err.get("message") if isinstance(err.get("message"), str) else None,
            "code": err.get("code"),
            "type": err.get("type"),
        }
    return {
        "error": parsed_err,
        "topCode": obj.get("code"),
        "topMessage": obj.get("message") if isinstance(obj.get("message"), str) else None,
    }


def _strip_request_id_tail(text: str) -> str:
    """去掉 ``Request id: xxx`` 等冗长后缀，避免刷屏。"""
    return _REQUEST_ID_RE.sub("", text).strip()


def _extract_model_name_from_message(message: str) -> str | None:
    """从英文 message 里尝试取出「模型名」片段，用于提示。"""
    for pattern in _MODEL_NAME_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None


def _clip_message(raw_message: str, max_length: int = 120) -> str:
    """超长消息截断到 ``max_length``，并把截断处的半个词也去掉后加 ``…``。"""
    if len(raw_message) <= max_length:
        return raw_message
    clipped = raw_message[: max_length - 3]
    return re.sub(r"\s+\S*\Z", "", clipped) + "…"


def format_vendor_http_error(status: int, body_text: str, kind: str = "video") -> str:
    """将上游 status + body 转为用户可读中文（**不返回整段英文 JSON**）。"""
    trimmed_body = (body_text or "").strip()
    parsed = _parse_error_body(trimmed_body)
    api_err = (parsed or {}).get("error")
    code = (_first_not_none(_as_str((api_err or {}).get("code")) or None,
                            _as_str((parsed or {}).get("topCode")) or None) or "").lower()
    err_type = _as_str((api_err or {}).get("type")).lower()
    raw_msg = _strip_request_id_tail(
        (_as_str((api_err or {}).get("message")) or _as_str((parsed or {}).get("topMessage"))).strip()
    )
    msg_lower = raw_msg.lower()
    combined_for_upload_check = f"{trimmed_body}\n{raw_msg}"

    # 1. 内容审核拦截（含上传图归因细分）
    if _looks_like_content_moderation(api_err, trimmed_body, code):
        if _body_suggests_people_in_user_uploads(combined_for_upload_check):
            return _UPLOADS_MODERATION_HINT.get(kind, _UPLOADS_MODERATION_HINT["video"])
        return _MODERATION_HINT.get(kind, _MODERATION_HINT["video"])

    # 2. 认证 / 权限 / 额度
    if status == 401:
        return "API Key 无效或未通过校验，请在 AI 服务配置中检查密钥是否正确。"
    if status == 403:
        return "没有权限调用该能力，请检查账号是否已开通对应模型或接口。"
    if status == 429:
        return "请求过于频繁或额度不足，请稍后再试。"

    # 3. 服务端瞬时异常
    if status in (500, 502, 503, 504):
        return "当前请求较多或服务暂时异常，请稍后重试。"

    # 4. 模型与接口不匹配（图片模型填视频栏、错误模型名）
    is_model_mismatch = status == 400 and (
        code == "invalidparameter"
        or err_type == "badrequest"
        or "does not support this api" in msg_lower
        or ("not support" in msg_lower and "api" in msg_lower)
        or "invalidparameter" in msg_lower
        or ("parameter" in msg_lower and "model" in msg_lower and "valid" in msg_lower)
        or "not found" in msg_lower
        or "modelnotfound" in msg_lower
    )
    if is_model_mismatch:
        model_name = _extract_model_name_from_message(raw_msg)
        if re.search(r"seedream", raw_msg, re.IGNORECASE):
            return "当前使用的是豆包 Seedream（图片模型），不能用于生成视频。请在 AI 服务配置中改用 Seedance、MiniMax H3 等视频模型。"
        if "not found" in msg_lower or "modelnotfound" in msg_lower:
            if model_name:
                return f"找不到模型「{model_name}」，请核对 AI 服务配置中的模型 ID 是否填写正确。"
            return "找不到所填写的模型名称，请核对上游平台文档中的模型 ID 是否填写正确。"
        if model_name:
            return f"模型「{model_name}」不支持当前接口。请在 AI 服务配置中更换为支持该类型生成的模型。"
        return "当前选择的模型与该接口不匹配。请检查 AI 服务配置中的「模型」是否为对应类型的模型（勿将图片模型填在视频栏）。"

    # 5. 缺少必填参数
    if status == 400 and ("prompt" in msg_lower or "parameter" in msg_lower) and "required" in msg_lower:
        return "缺少必填参数（如提示词或参考图），请检查任务参数是否完整。"

    # 6. 有结构化 message：尽量不贴大段英文技术原文
    if raw_msg:
        if raw_msg.startswith("{"):
            return "服务返回了异常数据，请稍后重试或联系平台。"
        if not _HAS_CHINESE_RE.search(raw_msg) and len(raw_msg) > 80:
            return "服务拒绝了本次请求，常见原因：模型名称错误、该模型不支持当前接口、或参数不符合平台要求。请在 AI 服务配置中核对「模型」是否正确。"
        return f"{_KIND_LABEL.get(kind, '视频')}生成未成功：{_clip_message(raw_msg)}"

    # 7. 无法解析 JSON 时的纯文本 body
    if parsed is None and trimmed_body and not trimmed_body.startswith("{"):
        plain = _strip_request_id_tail(re.sub(r"\s+", " ", trimmed_body))
        clip = _clip_message(plain)
        note = f"说明：{clip}" if clip else "请稍后重试。"
        return f"服务暂时无法完成请求（HTTP {status}）。{note}"

    return f"{_KIND_LABEL.get(kind, '视频')}生成失败（HTTP {status}），请检查网络、API Key 与模型配置后重试。"


def is_non_retryable_http_error(status: int, body_text: str) -> bool:
    """轮询过程中是否应**立即判失败**（而非继续重试）。

    可重试：5xx 服务端瞬时错误、网络抖动、404（任务尚未就绪）。
    不可重试：401/403/429（认证/权限/额度，重试无意义）、
    400 + 审核拦截或参数/模型错误（重试不会改变结果）。
    """
    if status in (401, 403, 429):
        return True
    if status == 400:
        trimmed = (body_text or "").strip()
        parsed = _parse_error_body(trimmed) or {}
        api_err = parsed.get("error")
        code = (_first_not_none(_as_str((api_err or {}).get("code")) or None,
                                _as_str(parsed.get("topCode")) or None) or "").lower()
        raw_msg = (
            _as_str((api_err or {}).get("message")) or _as_str(parsed.get("topMessage"))
        ).lower()
        is_moderation = _looks_like_content_moderation(api_err, trimmed, code)
        is_param_or_model_error = (
            code == "invalidparameter"
            or "invalidparameter" in raw_msg
            or "does not support this api" in raw_msg
            or "model" in raw_msg
            or "parameter" in raw_msg
            or "prompt" in raw_msg
        )
        return is_moderation or is_param_or_model_error
    return False


async def vendor_response_error(resp: httpx.Response, kind: str = "video") -> ValueError:
    """读取 ``Response`` 的 body 后，构造带用户可读中文的异常（不透传英文 JSON）。"""
    body_text = await _response_text(resp)
    return ValueError(format_vendor_http_error(resp.status_code, body_text, kind))


async def _response_text(resp: httpx.Response) -> str:
    """取响应正文；流式响应需要先 ``aread()``（与 Node 的 ``resp.text()`` 等价）。"""
    try:
        # httpx：非流式响应可直接读 .text；流式响应会抛 ResponseNotRead
        return resp.text
    except httpx.ResponseNotRead:  # pragma: no cover —— 取决于调用方是否 stream
        content = await resp.aread()
        return content.decode(resp.encoding or "utf-8", errors="replace")


def _safe_stringify(value: Any) -> str:
    """尽力把任意值转成字符串（对象无 message 时的兜底），失败则退回 ``str()``。

    ⚠️ 紧凑分隔符：Node 的 ``safeStringify`` 是 ``JSON.stringify(v)``，
    而这段文本会出现在**用户可见的错误消息**里，多一个空格就不一致。
    """
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def format_vendor_task_error(raw: Any, kind: str = "video") -> str:
    """将「轮询结果里的 error 字段」归因为用户可读中文。

    与 :func:`format_vendor_http_error` 的区别：轮询阶段**拿不到 HTTP status**，
    error 来自任务状态体（HTTP 200 但 ``status=failed``），且经常是厂商自定义结构 ——
    可能是 string、可能是 ``{code, message}`` 对象（火山引擎 Seedance 的 ``result.error``
    就是对象），甚至 JSON 字符串。

    原先 ``throw new Error(pollResp.error)`` 遇到对象会退化成 ``[object Object]``，
    把审核/失败原因**全部吞掉**，这里统一归一化 + 审核归因。
    """
    code = ""
    message = ""
    raw_text = ""

    if raw is None:
        pass  # 无 error 字段，走兜底
    elif isinstance(raw, str):
        raw_text = raw
        parsed = _parse_error_body(raw)
        if parsed and parsed.get("error"):
            code = _as_str(parsed["error"].get("code"))
            message = _as_str(parsed["error"].get("message"))
        elif parsed and parsed.get("topMessage"):
            message = _as_str(parsed.get("topMessage"))
        else:
            message = raw
    elif isinstance(raw, dict):
        code = _as_str(_first_not_none(
            raw.get("code"), raw.get("error_code"), raw.get("err_code"), raw.get("type"), ""
        ))
        message = _as_str(_first_not_none(
            raw.get("message"), raw.get("msg"), raw.get("error_msg"), raw.get("errorMessage"),
            raw.get("detail"), "",
        ))
        # 厂商 error 字段又套一层 error 对象的情况（{ error: { code, message } }）
        if not message and isinstance(raw.get("error"), str):
            message = raw["error"]
        if not message and isinstance(raw.get("error"), dict):
            inner = raw["error"]
            code = code or _as_str(_first_not_none(inner.get("code"), inner.get("error_code"), ""))
            message = _as_str(_first_not_none(inner.get("message"), inner.get("msg"), ""))
        raw_text = f"{code} {message}" if message else _safe_stringify(raw)
    elif isinstance(raw, (int, float, bool)):
        # JS 里 typeof 不是 'object' 也不是 'string' ⇒ 三个变量都保持空，走最终兜底
        pass

    combined = f"{code}\n{message}\n{raw_text}"

    # 1. 内容审核归因（含「上传图含人物」细分，复用 HTTP 层判断）
    api_err: dict[str, Any] = {"message": message or None, "code": code or None}
    if _looks_like_content_moderation(api_err, raw_text, code):
        if _body_suggests_people_in_user_uploads(combined):
            return _UPLOADS_MODERATION_HINT.get(kind, _UPLOADS_MODERATION_HINT["video"])
        return _MODERATION_HINT.get(kind, _MODERATION_HINT["video"])

    # 2. 提取可读 message 精简返回，避免贴大段英文技术原文
    cleaned = _strip_request_id_tail(message or raw_text or "").strip()
    label = _KIND_LABEL.get(kind, "视频")
    if cleaned:
        if cleaned.startswith("{") or cleaned.startswith("["):
            return f"{label}生成失败，请稍后重试或检查提示词与参考图。"
        if len(cleaned) > 80 and not _HAS_CHINESE_RE.search(cleaned):
            return f"{label}生成失败，请稍后重试或检查提示词、参考图与模型配置。"
        return f"{label}生成失败：{_clip_message(cleaned)}"

    return f"{label}生成失败，请稍后重试。"


# ── create 请求的指数退避重试 ─────────────────────────────────────────


def _is_retryable_status(status: int) -> bool:
    """create 阶段可退避重试的状态码：429 限流 / 5xx 瞬时异常。"""
    return status in (429, 502, 503, 504) or status >= 500


#: Python 侧的网络异常类型（Node 那边是 `fetch reject`，异常类名与文案都不同）
_RETRYABLE_EXC_TYPES = (httpx.TimeoutException, httpx.TransportError)

#: 与 Node 对齐的文案特征（保留，便于日志一致；Python 异常另有类型判定）
_RETRYABLE_NETWORK_TOKENS = (
    "timeout",
    "etimedout",
    "econnreset",
    "econnrefused",
    "network",
    "enotfound",
    "eai_again",
    "fetch failed",
    "socket hang up",
    "aborted",
)


def _is_retryable_network_error(err: BaseException) -> bool:
    """网络层错误是否可退避重试。

    ⚠️ Node 靠**错误文案**判断（`ECONNRESET`/`ENOTFOUND`…），而 httpx 的文案是
    ``[Errno 11001] getaddrinfo failed`` 这类**完全不同的串** ⇒ 这里以**异常类型**为主
    （``httpx.TransportError``/``TimeoutException``），文案匹配只作为补充。
    """
    if isinstance(err, _RETRYABLE_EXC_TYPES):
        return True
    message = str(err or "").lower()
    return any(token in message for token in _RETRYABLE_NETWORK_TOKENS)


#: 厂商请求共享客户端（连接池复用）。用 ``None`` 惰性创建，便于事件循环切换。
_vendor_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _vendor_client
    if _vendor_client is None or _vendor_client.is_closed:
        # 单次请求的超时由每次 fetch 传入；这里只兜底
        _vendor_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _vendor_client


async def aclose_client() -> None:
    """应用关闭时释放连接池（在 ``main.lifespan`` 里调用）。"""
    global _vendor_client
    if _vendor_client is not None and not _vendor_client.is_closed:
        await _vendor_client.aclose()
    _vendor_client = None


async def fetch_with_retry(
    url: str,
    init: dict[str, Any],
    kind: str = "video",
    *,
    max_retries: int = 3,
    base_delay_ms: int = 2000,
    timeout_ms: int | None = None,
    on_retry: Any = None,
    client: httpx.AsyncClient | None = None,
    files: Any = None,
    data: Any = None,
) -> httpx.Response:
    """带指数退避重试的请求，用于「创建生成任务」（create 阶段）。

    与轮询阶段的 :func:`is_non_retryable_http_error` 区分：

    * create 阶段遇到 429（限流）/ 5xx（瞬时异常）/ 网络抖动，退避后重试同一模型是合理的；
    * 401/403/400 审核或参数错误则**立即归因中文抛出**，不浪费时间重试。

    对齐参照项目的 ``retryOperation``：2s/4s/8s 指数退避，默认 3 次。
    多模型 fallback 外层仍会切换下一个模型兜底，因此单模型配置也能从
    「遇限流即失败」变为「短暂等待后自愈」。

    ``files`` / ``data`` 用于 multipart 上传（voice-clone 传音频样本）；
    给了它们就不再传 ``content``（httpx 不允许 ``content`` 与 ``files`` 同时出现）。
    """
    import asyncio

    http_client = client or _get_client()
    last_error: BaseException | None = None

    for attempt in range(max_retries):
        try:
            request_kwargs: dict[str, Any] = {}
            if files is not None or data is not None:
                request_kwargs["files"] = files
                request_kwargs["data"] = data
            else:
                request_kwargs["content"] = init.get("body")
            # 每次重试都用新的超时：httpx 的 timeout 是**每次请求**生效的，
            # 与 Node 每次重建 AbortSignal 的意图一致（复用已 abort 的 signal 会让重试立即失败）
            response = await http_client.request(
                init.get("method") or "GET",
                url,
                headers=init.get("headers"),
                timeout=(timeout_ms / 1000) if timeout_ms else None,
                **request_kwargs,
            )
            if response.is_success:
                return response

            # 非 2xx：可重试状态码则退避重试，否则（或最后一次）归因中文抛出
            if _is_retryable_status(response.status_code) and attempt < max_retries - 1:
                await response.aread()  # 消费 body 以便连接复用
                delay_ms = base_delay_ms * (2 ** attempt)
                if on_retry:
                    on_retry(attempt + 1, delay_ms, f"HTTP {response.status_code}")
                await asyncio.sleep(delay_ms / 1000)
                continue
            raise await vendor_response_error(response, kind)
        except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch (err: any) 等价
            last_error = err
            # 注意：上面 `raise await vendor_response_error(...)` 抛出的**归因中文**也可能
            # 落进这里，但它不含网络关键字 ⇒ 不会被误判成可重试（与 TS 行为一致）
            if _is_retryable_network_error(err) and attempt < max_retries - 1:
                delay_ms = base_delay_ms * (2 ** attempt)
                if on_retry:
                    on_retry(attempt + 1, delay_ms, str(err))
                await asyncio.sleep(delay_ms / 1000)
                continue
            raise

    label = _KIND_LABEL.get(kind, "视频")
    raise ValueError(f"{label}请求失败：已重试 {max_retries} 次仍无法完成。") from last_error
