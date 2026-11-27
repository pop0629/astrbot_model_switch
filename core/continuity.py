"""模型热切换的上下文连续性处理（纯函数，无 astrbot 依赖，便于单测）。

设计目标：切换 LLM provider 前后，让新模型“接得上”同一段对话，并把切模型的
事实以结构化、非侵入的方式告知新模型，而非写一条假的 user/assistant 消息。

四个子模块：
1. HistorySanitizer  —— 清洗历史中的阵营残留：孤立的 tool 消息、未闭合的
   tool_calls、只被旧模型能识别的推理块、空消息。避免新模型 400/格式化失败。
2. WindowPlanner     —— 在切换后头几轮主动收紧窗口（旧模型的长上下文里常有
   当前模型 tokenizer 无法承受的体积），只保留“近期轮次 + 摘要锚点”。
3. DigestBuilder     —— 从原历史抽取主题锚点（最近若干轮的用户意图、已确定的
   事实/决定、未完成事项），供摘要器（可选，小模型）生成交接摘要。
4. HandoffInjection  —— 生成交接上下文块，插入到 system prompt 尾部 /
   或作为一次性的语境前缀，且带“只此一轮生效”的标记，避免污染长期历史。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 1. 历史清洗
# --------------------------------------------------------------------------- #

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "developer": "system",
}

# 只在部分 provider 上合法的 content-part 类型
_MODALITY_PARTS = {"image_url", "image", "input_image", "audio", "input_audio",
                   "video", "file", "input_file"}
_REASONING_PARTS = {"reasoning", "thinking", "redacted_thinking"}


@dataclass
class SanitizeReport:
    dropped_messages: int = 0
    dropped_roles: list[str] = field(default_factory=list)
    dropped_tool_messages: int = 0
    closed_unfinished_tools: int = 0
    stripped_reasoning_parts: int = 0
    dropped_modality_parts: int = 0
    empty_assistant_filled: int = 0

    def as_dict(self) -> dict:
        return {
            "dropped_messages": self.dropped_messages,
            "dropped_roles": self.dropped_roles,
            "dropped_tool_messages": self.dropped_tool_messages,
            "closed_unfinished_tools": self.closed_unfinished_tools,
            "stripped_reasoning_parts": self.stripped_reasoning_parts,
            "dropped_modality_parts": self.dropped_modality_parts,
            "empty_assistant_filled": self.empty_assistant_filled,
        }


def _has_text(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(_has_text(part) for part in content)
    if isinstance(content, dict):
        return any(_has_text(v) for v in content.values() if isinstance(v, (str, list, dict)))
    return True


def _tool_call_ids(msg: dict) -> list[str]:
    out = []
    calls = msg.get("tool_calls") or []
    if isinstance(calls, list):
        for call in calls:
            cid = None
            if isinstance(call, dict):
                cid = call.get("id") or (call.get("function") or {}).get("id")
            if cid:
                out.append(str(cid))
    return out


def _tool_msg_id(msg: dict) -> str | None:
    tid = msg.get("tool_call_id") or msg.get("id")
    return str(tid) if tid else None


def sanitize_history(
    contexts: Iterable[dict],
    *,
    supports_image: bool = True,
    supports_audio: bool = False,
    keep_reasoning: bool = False,
) -> tuple[list[dict], SanitizeReport]:
    """让历史对“任意新 provider”都可接受。

    - 去掉 tool 消息：新模型不一定绑定了旧模型的 tool schema，
      孤立的 tool 结果会直接触发 400。
    - 去掉 assistant 上未被应答的 tool_calls 字段。
    - 去掉 reasoning/thinking 等思维链块（跨模型语义不连续且多数会报错）。
    - 新模型不支持图片/音频时，把多模态 part 替换为占位文本。
    - 空 assistant 消息补一个 "."，避免部分 provider 拒绝空 content。
    """
    report = SanitizeReport()
    cleaned: list[dict] = []

    for raw in contexts or []:
        if not isinstance(raw, dict):
            report.dropped_messages += 1
            continue
        role_raw = str(raw.get("role", "user")).lower()
        role = _ROLE_ALIASES.get(role_raw, role_raw)

        if role == "tool":
            report.dropped_messages += 1
            report.dropped_tool_messages += 1
            report.dropped_roles.append("tool")
            continue
        if role not in ("system", "user", "assistant"):
            report.dropped_messages += 1
            report.dropped_roles.append(role_raw)
            continue

        msg = {"role": role}
        content = raw.get("content")
        content = _strip_parts(
            content,
            report,
            supports_image=supports_image,
            supports_audio=supports_audio,
            keep_reasoning=keep_reasoning,
        )
        if "tool_calls" in raw and raw["tool_calls"]:
            report.closed_unfinished_tools += 1
        msg["content"] = content

        if role == "assistant" and not _has_text(content):
            msg["content"] = "."
            report.empty_assistant_filled += 1
        elif role == "assistant" and raw.get("tool_calls") and _has_text(content):
            # 把工具调用痕迹降级为普通文本，保留“当时做了什么”的语义连续性
            names = [
                (c.get("function") or {}).get("name") if isinstance(c, dict) else None
                for c in raw["tool_calls"]
            ]
            names = [n for n in names if n]
            if names:
                msg["content"] = f"{_textify(content)}\n[此前调用工具: {', '.join(names)}]"
        elif not _has_text(content):
            report.dropped_messages += 1
            report.dropped_roles.append(role_raw)
            continue

        cleaned.append(msg)

    return cleaned, report


def _textify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _strip_parts(
    content: Any,
    report: SanitizeReport,
    *,
    supports_image: bool,
    supports_audio: bool,
    keep_reasoning: bool,
) -> Any:
    if not isinstance(content, list):
        return content
    out: list[Any] = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type", "text"))
        if ptype in _REASONING_PARTS:
            if keep_reasoning:
                out.append(part)
            else:
                report.stripped_reasoning_parts += 1
            continue
        if ptype in _MODALITY_PARTS:
            supported = (ptype in ("image_url", "image", "input_image") and supports_image) or (
                ptype in ("audio", "input_audio") and supports_audio
            )
            if supported:
                out.append(part)
            else:
                report.dropped_modality_parts += 1
                out.append({"type": "text", "text": f"[{ptype} omitted: unsupported by current model]"})
            continue
        out.append(part)
    # 全部退化成纯文本时直接折叠成字符串，避免 provider 因空 part 列表报错
    if all(isinstance(part, str) for part in out):
        return "\n".join(out)
    if len(out) == 1 and isinstance(out[0], dict) and out[0].get("type") == "text":
        return out[0].get("text", "")
    return out


# --------------------------------------------------------------------------- #
# 2. 切换后窗口规划
# --------------------------------------------------------------------------- #


@dataclass
class WindowPlan:
    contexts: list[dict]
    dropped_messages: int
    kept_turns: int
    anchor: dict | None


def count_turns(contexts: list[dict]) -> int:
    return sum(1 for m in contexts if m.get("role") == "user")


def plan_window(
    contexts: list[dict],
    *,
    recent_turns: int = 6,
    preserve_system: bool = True,
    max_chars: int | None = 12000,
) -> WindowPlan:
    """切换后的第一轮：只给新模型一个“近期窗口”。

    - 按 user 轮次倒推保留最近 recent_turns 轮（含其后的 assistant）。
    - 若提供 max_chars，再从最旧开始裁到字符预算内。
    - 裁掉的部分由调用方交给 DigestBuilder 生成摘要锚点。
    """
    system_msgs = [m for m in contexts if m.get("role") == "system"] if preserve_system else []
    convo = [m for m in contexts if m.get("role") != "system"]

    if recent_turns <= 0:
        kept = convo
    else:
        # 找到第 recent_turns 个 user 消息的起点（从后往前数）
        user_idx = [i for i, m in enumerate(convo) if m.get("role") == "user"]
        if len(user_idx) <= recent_turns:
            kept = convo
        else:
            kept = convo[user_idx[-recent_turns]:]

    if max_chars is not None and max_chars > 0:
        budget = max_chars
        trim_from = len(kept)
        for i in range(len(kept) - 1, -1, -1):
            size = len(_textify(kept[i].get("content")))
            if budget - size < 0:
                break
            budget -= size
            trim_from = i
        # 不要以 assistant 开头
        while trim_from < len(kept) and kept[trim_from].get("role") != "user":
            trim_from += 1
        kept = kept[trim_from:]

    dropped = convo[: len(convo) - len(kept)]
    anchor = anchor_from_messages(dropped) if dropped else None
    return WindowPlan(
        contexts=system_msgs + kept,
        dropped_messages=len(dropped),
        kept_turns=count_turns(kept),
        anchor=anchor,
    )


# --------------------------------------------------------------------------- #
# 3. 摘要锚点 / 交接摘要
# --------------------------------------------------------------------------- #

_FACT_HINTS = (
    "记住", "约定", "结论", "决定", "确定", "计划", "要求", "规范",
    "remember", "decided", "plan", "requirement", "must", "conclusion",
)
_TODO_HINTS = ("待办", "还没", "尚未", "下一步", "todo", "next step", "remaining")


def anchor_from_messages(messages: list[dict], *, max_items: int = 6) -> dict | None:
    """从被裁掉的历史里抽“锚点”：议题、结论、未完成事项。

    不调用 LLM，纯启发式；命不中时退化为“早期议题前 N 条用户消息”。
    """
    if not messages:
        return None
    topics: list[str] = []
    facts: list[str] = []
    todos: list[str] = []
    last_assistant: str | None = None

    for msg in messages:
        text = _textify(msg.get("content")).strip()
        if not text:
            continue
        low = text.lower()
        if msg.get("role") == "user":
            snippet = _clip(text, 80)
            if snippet not in topics:
                topics.append(snippet)
            if any(h in low for h in _TODO_HINTS):
                todos.append(_clip(text, 80))
            elif any(h in low for h in _FACT_HINTS):
                facts.append(_clip(text, 80))
        elif msg.get("role") == "assistant":
            last_assistant = _clip(text, 120)

    if not topics and not last_assistant:
        return None
    return {
        "topics": topics[-max_items:],
        "facts": facts[-max_items:],
        "todos": todos[-max_items:],
        "last_reply_tail": last_assistant,
        "message_count": len(messages),
    }


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def build_digest_prompt(messages: list[dict], old_model: str) -> str:
    """给“摘要小模型”的提示词：把被裁历史压成交接摘要。"""
    body = []
    for m in messages:
        role = m.get("role")
        text = _textify(m.get("content"))
        if not text.strip():
            continue
        body.append(f"{role}: {_clip(text, 400)}")
    return (
        "你负责交接。下面是同一对话中由另一个模型(旧模型)处理过的历史片段。\n"
        "请压缩成结构化交接摘要，供接手的模型继续对话使用。只输出以下四段，不要寒暄：\n"
        "【议题】当前在讨论/在做什么（1-3 条）\n"
        "【已确定事实】用户给出或双方确认过的信息（列表，无则写“无”）\n"
        "【未完成】还没解决或用户已提出待办的事项（列表，无则写“无”）\n"
        "【风格备注】用户偏好与交互习惯（如语言、格式、称呼，无则写“无”）\n\n"
        f"---- 历史片段（共 {len(body)} 条，旧模型={old_model or 'unknown'}）----\n"
        + "\n".join(body)
        + "\n---- 结束 ----"
    )


def fallback_digest(anchor: dict, old_model: str) -> str:
    """无摘要模型时，用启发式锚点拼一个最小可行摘要。"""
    topics = "；".join(anchor.get("topics") or []) or "无"
    facts = "；".join(anchor.get("facts") or []) or "无"
    todos = "；".join(anchor.get("todos") or []) or "无"
    tail = anchor.get("last_reply_tail") or ""
    tail_line = f"\n【上次回复结尾】{tail}" if tail else ""
    return (
        f"【议题】{topics}\n【已确定事实】{facts}\n【未完成】{todos}\n"
        f"【风格备注】无{tail_line}\n（此摘要由规则抽取自 {anchor.get('message_count', 0)} 条历史，"
        f"旧模型={old_model or 'unknown'}）"
    )


# --------------------------------------------------------------------------- #
# 4. 交接上下文注入
# --------------------------------------------------------------------------- #

DEFAULT_HANDOFF_TEMPLATE = """\
<model_handoff>
你正在接手的是一场进行中的对话，前一轮由另一个模型处理。
{old_line}{reason_line}
交接摘要：
{digest}

衔接要求：
1. 不要从头自我介绍，也不要重述上文或要求用户重复已给出的信息。
2. 直接接着上一句继续；若上文结论明确，可直接推进下一步。
3. 如果交接摘要与你看到的历史有出入，以历史为准。
4. 本轮不需要显式提及“换模型”这件事，除非用户问起。{extra_rules}
</model_handoff>"""


def build_handoff_block(
    *,
    old_model: str,
    new_model: str,
    digest: str,
    include_names: bool = False,
    extra_rules: str = "",
) -> str:
    old_line = ""
    if include_names and old_model:
        old_line = f"上一模型: {old_model}；当前模型: {new_model}。\n"
    reason_line = "（原因：用户主动切换模型。）\n" if old_model else ""
    extra_rules = f"\n5. {extra_rules}" if extra_rules else ""
    return DEFAULT_HANDOFF_TEMPLATE.format(
        old_line=old_line,
        reason_line=reason_line,
        digest=digest.strip(),
        extra_rules=extra_rules,
    )


def inject_system_prompt(system_prompt: str, block: str) -> str:
    """把交接块放到 system prompt 尾部：优先使用紧邻对话的“近因位置”。"""
    if not system_prompt:
        return block
    return f"{system_prompt.rstrip()}\n\n{block}"


def one_shot_prefix(block: str, keep_system_clean: bool) -> tuple[str, str | None]:
    """返回 (system_suffix, user_prefix)。

    keep_system_clean=True 时不改 system prompt，改为在本次 user 消息前加一段
    语境引导，适合严格缓存 system prompt 的 provider（避免缓存全失效）。
    """
    if keep_system_clean:
        return "", block + "\n\n"
    return "\n\n" + block, None


# --------------------------------------------------------------------------- #
# 状态机：一次切换从“挂起”到“落实”
# --------------------------------------------------------------------------- #


@dataclass
class SwitchState:
    """持久化到插件 KV 的切换状态。"""

    old_provider_id: str = ""
    new_provider_id: str = ""
    old_model: str = ""
    new_model: str = ""
    switched_at: float = 0.0
    # 首次作用于请求时的历史快照（用于抽样生成摘要）
    pending_snapshot: list[dict] = field(default_factory=list)
    digest: str = ""
    turns_applied: int = 0
    last_applied_turn_at: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SwitchState | None":
        if not data:
            return None
        st = cls()
        for k in st.__dict__:
            if k in data:
                setattr(st, k, data[k])
        return st

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None) -> "SwitchState | None":
        if not raw:
            return None
        try:
            return cls.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            return None
