"""两段式人工介入（学自 lobehub ``agent-runtime/src/core/InterventionChecker.ts``
＋ ``audit/defaultSecurityBlacklist.ts``）。

**机制（不是提示词自觉）**：工具调用要跑之前先过一个判定器，判定结果是三值
``never`` / ``required`` / ``always``：

* ``never`` ⇒ 直接跑（默认就是它：没配策略的工具一律不打扰创作者）；
* ``required`` / ``always`` ⇒ **先不跑**，把这次调用登记成待批，回灌一句话给模型；
  创作者的批准/拒绝从下一跑的 ``approvals`` 进来，作为**消息**回到对话里
  （参考实现是整轮 park 等 resume；本仓没有可持久化的消息表，等价载体是
  「本跑不执行 + 登记 pending + 下一跑带批准」——**结论相同：副作用没发生、
  模型看得见"还没批"**）。

判定顺序（**顺序本身就是安全语义**，抄错等于没有护栏）：

1. **安全黑名单优先于一切配置**（``check_security_blacklist``）：命中即 ``required``，
   创作者怎么配都拦不住 —— 那是「不该由模型自己决定」的动作。
2. 无配置 ⇒ ``never``。
3. 配置是字符串 ⇒ 就是它。
4. 配置是规则数组 ⇒ **第一条命中**的 ``policy``；一条都没命中 ⇒ ``required``
   （**从严兜底**：为这个工具配了规则却一个都没匹配上，说明参数落在没想过的形状里，
   这时候放行是最坏选择）。

规则匹配细节（逐条对拍参考实现）：

* ``match`` 里的**每个**参数都要命中（AND 语义）；
* 参数**不存在** ⇒ 该规则不命中（不是"像空串一样命中"）；
* matcher 支持 ``exact`` / ``prefix`` / ``wildcard`` / ``regex``；裸字符串按 wildcard；
* wildcard 额外支持 ``git add:*`` 这种**冒号前缀**写法（``value.startswith("git add:")``
  或 ``value == "git add"``），其它 ``*`` 按 ``.*`` 全域匹配。

⚠️ 与参考实现的两处**有意差异**（都写在这，不藏）：

* **载体**：参考按 ``identifier/apiName`` 分两段，本仓工具名即函数名 ⇒ ``tool_key`` 只有
  ``name`` 一段；``name#hash`` 的 key 形状保留（同一工具不同参数是**不同**的审批）。
* **``interventions`` 支持 ``"*"`` 兜底键**：参考只按 toolName 精确索引，本仓另有 MCP
  接进来的动态工具（名字由外部 server 定），逐个写配置不现实；``"*"`` 只作**兜底**
  （精确键优先），并如实标在回执里。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "POLICIES",
    "Decision",
    "InterventionGate",
    "PendingCall",
    "DEFAULT_SECURITY_BLACKLIST",
    "check_security_blacklist",
    "hash_arguments",
    "match_argument",
    "match_pattern",
    "normalize_interventions",
    "should_intervene",
    "tool_key",
]

#: 三值策略（参考 ``HumanInterventionPolicy``）
POLICIES = ("never", "required", "always")

#: ``always`` 不消费「批准过的记忆」：每次都要重批（参考 ``ALWAYS_EXECUTE_POLICIES`` 的反面）
_REMEMBRABLE = ("required",)


# ---------------------------------------------------------------------------
# 安全黑名单
# ---------------------------------------------------------------------------

def _rule(description: str, match: dict[str, Any]) -> dict[str, Any]:
    return {"description": description, "match": match}


def _cmd(pattern: str) -> dict[str, Any]:
    return {"pattern": pattern, "type": "regex"}


def _path(pattern: str) -> dict[str, Any]:
    return {"pattern": pattern, "type": "regex"}


#: 默认安全黑名单：**命中一律 ``required``**（先于创作者配置）。
#: 条目的 ``command`` / ``path`` 两个参数名沿用参考实现 —— 本仓内置工具没有这两个参数，
#: 但 MCP 工具（``app/mcp/client.py`` 接进来的外部 server）可能有 shell / 文件能力，
#: 护栏必须先在这儿，而不是等出事再补。
DEFAULT_SECURITY_BLACKLIST: tuple[dict[str, Any], ...] = (
    # 文件系统
    _rule("递归删家目录",
          {"command": _cmd(r"rm.*-r.*(~|\$HOME|/Users/[^/]+|/home/[^/]+)/?\s*$")}),
    _rule("递归删根目录", {"command": _cmd(r"rm.*-r.*/\s*$")}),
    _rule("rm -rf 打在家目录/根上", {"command": _cmd(r"rm\s+-rf\s+[~/.]\s*$")}),
    # 系统配置
    _rule("读 /etc/passwd 或 /etc/shadow", {"command": _cmd(r".*(/etc/passwd|/etc/shadow).*")}),
    _rule("动 sudoers", {"command": _cmd(r".*/etc/sudoers.*")}),
    # 危险命令
    _rule("fork 炸弹", {"command": _cmd(r".*:\(\).*\{.*\|.*&.*\};.*:.*")}),
    _rule("dd 直接写磁盘", {"command": _cmd(r"dd.*of=/dev/(sd|hd|nvme).*")}),
    _rule("格式化分区", {"command": _cmd(r"(mkfs|fdisk|parted).*(/dev/(sd|hd|nvme)|/)")}),
    # 网络与远程访问
    _rule("关防火墙", {"command": _cmd(r"(ufw\s+disable|iptables\s+-F|systemctl\s+stop\s+firewalld)")}),
    _rule("动 sshd 配置", {"command": _cmd(r".*(/etc/ssh/sshd_config).*")}),
    # 包管理
    _rule("卸系统关键包",
          {"command": _cmd(r"(apt|yum|dnf|pacman)\s+(remove|purge|erase).*(systemd|kernel|glibc|bash|sudo)")}),
    # 内核
    _rule("写 /proc/sys", {"command": _cmd(r"echo.*>/proc/sys/.*")}),
    _rule("访问 /dev/mem|kmem|port", {"command": _cmd(r".*(/dev/(mem|kmem|port)).*")}),
    # 提权
    _rule("chown -R 系统目录", {"command": _cmd(r"chown.*-R.*(/(etc|bin|sbin|usr|var|sys|proc)|~).*")}),
    _rule("给解释器加 suid",
          {"command": _cmd(r"chmod.*(4755|u\+s).*(sh|bash|python|perl|ruby|node)")}),
    # 凭据泄露（本仓相关：.env 里有 API key，直接读/写都算事故）
    _rule("读 .env", {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*\.env.*")}),
    _rule(".env 路径", {"path": _path(r".*\.env.*")}),
    _rule("读 ssh 私钥",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*(id_rsa|id_ed25519|id_ecdsa)(?!\.pub).*")}),
    _rule("ssh 私钥路径", {"path": _path(r".*/\.ssh/(id_rsa|id_ed25519|id_ecdsa)$")}),
    _rule("读 AWS 凭据",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.aws/credentials.*")}),
    _rule("AWS 凭据路径", {"path": _path(r".*/\.aws/credentials.*")}),
    _rule("读 docker config",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.docker/config\.json.*")}),
    _rule("docker config 路径", {"path": _path(r".*/\.docker/config\.json$")}),
    _rule("读 kube config",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.kube/config.*")}),
    _rule("kube config 路径", {"path": _path(r".*/\.kube/config$")}),
    _rule("读 git 凭据",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.git-credentials.*")}),
    _rule("git 凭据路径", {"path": _path(r".*/\.git-credentials$")}),
    _rule("读 npm token",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.npmrc.*")}),
    _rule("npmrc 路径", {"path": _path(r".*/\.npmrc$")}),
    _rule("读 shell 历史",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.(bash_history|zsh_history|history).*")}),
    _rule("历史文件路径", {"path": _path(r".*/\.(bash_history|zsh_history|history)$")}),
    _rule("读浏览器凭据",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*(Cookies|Login Data|Web Data).*")}),
    _rule("读 gcloud 凭据",
          {"command": _cmd(r"(cat|less|more|head|tail|vim|nano|vi|emacs|code).*/\.config/gcloud/.*\.json.*")}),
    _rule("gcloud 凭据路径", {"path": _path(r".*/\.config/gcloud/.*\.json$")}),
    # 本仓特有：主库与版本库不该被工具**直接**改
    _rule("直接动 SQLite 主库", {"path": _path(r"(?i).*drama\.db.*")}),
    _rule("直接动 .git 内部", {"path": _path(r".*[/\\]\.git[/\\].*")}),
)


def match_pattern(pattern: Any, value: Any) -> bool:
    """wildcard 匹配（``*`` 通配 + ``前缀:*`` 冒号写法）。

    冒号写法与参考 ``matchPattern`` 同口径（``value === prefix || value.startsWith(prefix + ':')``），
    并**多认一种** ``prefix + ' '``：参考实现自己的注释举例就是 ``"git add:*"``，可它只认
    ``git add:...`` ⇒ 拿去匹配真实的 ``git add foo.md`` 永远不中（配置看起来生效、其实一条
    都拦不住）。多认空格是让那个例子真的成立，且不影响原两种形状。
    """
    text = str(value)
    pattern = str(pattern)
    if ":" in pattern:
        prefix, _, suffix = pattern.partition(":")
        if suffix == "*":
            return (text == prefix or text.startswith(prefix + ":")
                    or text.startswith(prefix + " "))
    # ``re.escape`` 会把 ``*`` 一起转义，再单独把它还原成 ``.*``（参考实现的转义集合不含 ``*``）
    escaped = re.escape(pattern).replace(r"\*", ".*")
    try:
        return re.fullmatch(escaped, text) is not None
    except re.error:                       # 不是合法正则（含怪字符）⇒ 当不命中
        return False


def match_argument(matcher: Any, value: Any) -> bool:
    """单个参数 matcher（字符串＝wildcard；对象＝``{pattern,type}``）。"""
    text = str(value)
    if isinstance(matcher, str):
        return match_pattern(matcher, text)
    if not isinstance(matcher, Mapping):
        return False
    pattern = matcher.get("pattern")
    kind = matcher.get("type") or "wildcard"
    if pattern is None:
        return False
    if kind == "exact":
        return text == str(pattern)
    if kind == "prefix":
        return text.startswith(str(pattern))
    if kind == "wildcard":
        return match_pattern(pattern, text)
    if kind == "regex":
        try:
            return re.search(str(pattern), text) is not None
        except re.error:
            return False
    return False                         # 未知 type ⇒ **不命中**（参考也是 default: false）


def _rule_matches(rule: Mapping[str, Any], args: Mapping[str, Any]) -> bool:
    match = rule.get("match")
    if match is None:
        return True                      # 无 match 条件 ⇒ 通配规则
    if not isinstance(match, Mapping):
        return False
    for name, matcher in match.items():
        if name not in args or args[name] is None:
            return False                 # 参数不存在 ⇒ 不命中（不是"空串命中"）
        if not match_argument(matcher, args[name]):
            return False
    return True


def check_security_blacklist(
    args: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    """命中安全黑名单 ⇒ 返回命中的**说明**；未命中 ⇒ ``None``。"""
    catalogue = DEFAULT_SECURITY_BLACKLIST if rules is None else rules
    for rule in catalogue:
        if _rule_matches(rule, args):
            return str(rule.get("description") or "安全黑名单命中")
    return None


def should_intervene(
    config: Any,
    args: Mapping[str, Any] | None = None,
    *,
    security_blacklist: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str]:
    """返回 ``(policy, reason)``；``reason`` 是给人看的一句话（进待批清单）。"""
    args = args or {}
    blocked = check_security_blacklist(args, security_blacklist)
    if blocked:
        return "required", f"安全黑名单：{blocked}（这条不受创作者配置影响）"
    if config is None:
        return "never", ""
    if isinstance(config, str):
        return (config if config in POLICIES else "required"), "按工具策略"
    if isinstance(config, Mapping):       # 单条规则对象也当一条规则数组吃
        config = [config]
    if isinstance(config, (list, tuple)):
        for rule in config:
            if isinstance(rule, Mapping) and _rule_matches(rule, args):
                policy = str(rule.get("policy") or "required")
                return (policy if policy in POLICIES else "required"), "按匹配到的规则"
        return "required", "为该工具配了规则但一条都没命中（从严兜底）"
    return "required", "策略配置形状不认识（从严兜底）"


# ---------------------------------------------------------------------------
# 审批键
# ---------------------------------------------------------------------------

def hash_arguments(args: Mapping[str, Any] | None) -> str:
    """参数哈希（参考 ``hashArguments``：键名排序、``k=v`` 用 ``&`` 连、int32 环形移位）。

    ⚠️ 复数个参数**值**直接用 ``json.dumps``（参考是 ``JSON.stringify``）：同一份参数
    无论哪个 provider 回过来都得到同一串，否则「批准过的这次」下一次认不出来。
    ⚠️ 必须**紧凑分隔符**（``(",", ":")``）：``JSON.stringify`` 不打空格，而 Python 默认
    是 ``{"a": 1, "b": 2}`` ⇒ 差一个空格就是**另一个哈希** ✓（"批准过的"与"这次要跑的"
    认不出来 ✓ 介入批准就永远命中不了 ✓ 症状是"批了也不生效" ✗ 不报错）。
    """
    payload = args or {}
    parts = []
    for key in sorted(payload):
        parts.append(
            f"{key}="
            + json.dumps(payload[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    text = "&".join(parts)
    value = 0
    for char in text:
        value = (value << 5) - value + ord(char)
        value &= 0xFFFFFFFF                    # JS ``hash & hash`` ⇒ int32 截断
        if value >= 0x80000000:                # 转成有符号（与 JS 一致）
            value -= 0x100000000
    return _base36(abs(value))


def _base36(number: int) -> str:
    if number == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while number:
        number, rem = divmod(number, 36)
        out = digits[rem] + out
    return out


def tool_key(name: str, args_hash: str | None = None) -> str:
    """``name`` 或 ``name#hash``（参考 ``generateToolKey`` 的单段版）。"""
    return f"{name}#{args_hash}" if args_hash else str(name)


def normalize_interventions(raw: Any) -> dict[str, Any]:
    """归一 ``options["interventions"]``：只留认得的 policy 字符串，其余原样留给判定器。

    **不丢键**：写成不认识的 policy 时判定器按 ``required`` 从严处理，这里若直接删掉，
    一个笔误的配置会静默变成「不审批」—— 那是最危险的方向。
    """
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


# ---------------------------------------------------------------------------
# 本仓载体：闸口 + 待批登记
# ---------------------------------------------------------------------------

@dataclass
class PendingCall:
    """一次等批的调用（回执/前端都是这个形状）。"""

    call_id: str
    tool: str
    args: dict[str, Any]
    key: str
    policy: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "callId": self.call_id,
            "tool": self.tool,
            "args": self.args,
            "key": self.key,
            "policy": self.policy,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Decision:
    """一个工具调用的判定结果：``run``（照跑）/ ``pending``（等批）/ ``rejected``（已拒）。"""

    kind: str
    key: str = ""
    policy: str = "never"
    reason: str = ""
    remembered: bool = False

    @property
    def run(self) -> bool:
        return self.kind == "run"


@dataclass
class InterventionGate:
    """一次 run 的审批闸口。

    ``approvals`` 的键是 :func:`tool_key`（``name#hash``）；值可以是
    ``"approve"`` / ``"reject"`` 字符串，也可以是 ``{"action": ..., "remember": bool}``。
    **键必须带 hash**：只按工具名批，等于批准了"这个工具的任意参数"，那是把护栏拆了。
    """

    interventions: dict[str, Any] = field(default_factory=dict)
    approvals: dict[str, Any] = field(default_factory=dict)
    security_blacklist: tuple[dict[str, Any], ...] | None = None
    confirmed: set[str] = field(default_factory=set)
    pending: list[PendingCall] = field(default_factory=list)
    rejected: list[PendingCall] = field(default_factory=list)
    wildcard_used: bool = False

    def config_for(self, name: str) -> Any:
        if name in self.interventions:
            return self.interventions[name]
        if "*" in self.interventions:
            self.wildcard_used = True
            return self.interventions["*"]
        return None

    def decide(self, name: str, args: Mapping[str, Any] | None = None,
               call_id: str = "") -> Decision:
        args = args or {}
        policy, reason = should_intervene(self.config_for(name), args,
                                          security_blacklist=self.security_blacklist)
        if policy == "never":
            return Decision(kind="run", policy=policy)
        key = tool_key(name, hash_arguments(args))
        if policy in _REMEMBRABLE and key in self.confirmed:
            return Decision(kind="run", key=key, policy=policy, reason=reason, remembered=True)
        approval = self.approvals.get(key)
        action, remember = _read_approval(approval)
        if action == "approve":
            if remember and policy in _REMEMBRABLE:
                self.confirmed.add(key)
            return Decision(kind="run", key=key, policy=policy, reason=reason)
        record = PendingCall(call_id=call_id, tool=name, args=dict(args), key=key,
                             policy=policy, reason=reason)
        if action == "reject":
            self.rejected.append(record)
            return Decision(kind="rejected", key=key, policy=policy, reason=reason)
        self.pending.append(record)
        return Decision(kind="pending", key=key, policy=policy, reason=reason)

    def receipt(self) -> dict[str, Any]:
        return {
            "configured": sorted(self.interventions),
            "wildcard": self.wildcard_used,
            "remembered": sorted(self.confirmed),
            "pending": [item.as_dict() for item in self.pending],
            "rejected": [item.as_dict() for item in self.rejected],
            "approvalsSeen": len(self.approvals),
        }


def _read_approval(raw: Any) -> tuple[str | None, bool]:
    if isinstance(raw, str):
        return (raw if raw in ("approve", "reject") else None), False
    if isinstance(raw, Mapping):
        action = raw.get("action")
        return (str(action) if action in ("approve", "reject") else None), bool(raw.get("remember"))
    return None, False


def gate_from_options(options: Mapping[str, Any] | None) -> InterventionGate:
    """从 run 选项建闸口（``interventions`` / ``approvals``；都缺省 ⇒ 永不打扰）。"""
    options = options or {}
    return InterventionGate(
        interventions=normalize_interventions(options.get("interventions")),
        approvals=_normalize_approvals(options.get("approvals")),
    )


def _normalize_approvals(raw: Any) -> dict[str, Any]:
    """批准清单容错：``dict`` 直接用；``list[{callId/key, action, remember}]`` 折成 dict。"""
    if isinstance(raw, Mapping):
        return dict(raw)
    out: dict[str, Any] = {}
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key") or item.get("callId") or item.get("call_id")
            if not key:
                continue
            out[str(key)] = {"action": item.get("action"), "remember": item.get("remember")}
    return out


def pending_message(record: PendingCall) -> dict[str, Any]:
    """回灌给模型的工具结果（**它必须看得懂"还没执行"**，否则会以为跑完了）。"""
    return {
        "status": "pending_approval",
        "skipped": True,
        "tool": record.tool,
        "reason": record.reason,
        "note": (
            f"`{record.tool}` 这次**没有执行**：它需要创作者先批准（{record.reason}）。"
            f"批准凭据是 `{record.key}`。不要重试同一个调用 —— 把要 creator 决定的事说清楚，"
            f"或者先做不需要批准的部分。"
        ),
    }


def rejected_message(record: PendingCall) -> dict[str, Any]:
    """创作者已拒绝 ⇒ 回灌一句**明确的拒绝**（不是"等待"，否则模型会一直等）。"""
    return {
        "status": "rejected",
        "skipped": True,
        "tool": record.tool,
        "reason": record.reason,
        "note": (
            f"`{record.tool}` 这次被创作者**拒绝**执行（{record.reason}）。"
            f"不要再尝试同一个调用，也不要换个写法绕过去 —— 换方案，或者把选择交给创作者。"
        ),
    }


def describe_pending(items: Iterable[PendingCall]) -> str:
    """待批清单的一行人话（进最终正文/日志）。"""
    rows = [f"`{item.tool}`（{item.key}）" for item in items]
    return "；".join(rows)
