"""连续性引擎：把纯函数编排成一次“无缝换模型”。

所有 astrbot 依赖通过注入的 hooks 适配，便于在容器外用 fake 对象测试。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .continuity import (
    SwitchState,
    WindowPlan,
    anchor_from_messages,
    build_digest_prompt,
    build_handoff_block,
    fallback_digest,
    inject_system_prompt,
    plan_window,
    sanitize_history,
)

Summarizer = Callable[[str], Awaitable[str]]


@dataclass
class EngineConfig:
    # 切换后头几轮启用"紧窗口 + 交接块"
    handoff_turns: int = 2
    # 紧窗口保留的近期轮次
    recent_turns: int = 6
    # 紧窗口字符预算（0 关闭）
    max_chars: int = 12000
    # 是否在历史里开启 tool 消息清洗（新模型没有旧 tool schema，必须开）
    sanitize_tools: bool = True
    # 是否抽摘要（需要摘要模型；关掉则用规则锚点）
    enable_summary: bool = True
    # 摘要模型期望的 provider_id 谓词，由插件决定；None 表示用当前 provider
    summarizer_provider_id: str = ""
    # 摘要输入的最大条数 / 字符
    digest_max_messages: int = 40
    # 是否改 system prompt（False = 只加 user 前缀，保护 prompt 缓存）
    touch_system_prompt: bool = True
    # 交接块是否暴露模型名
    include_model_names: bool = False
    extra_rules: str = ""


@dataclass
class RequestPatch:
    contexts: list[dict]
    system_prompt: str
    user_prefix: str | None
    applied: bool
    stats: dict[str, Any] = field(default_factory=dict)


class ContinuityEngine:
    def __init__(self, config: EngineConfig, summarizer: Summarizer | None = None):
        self.config = config
        self.summarizer = summarizer

    # ------------------------------------------------------------------ #
    # 切换事件
    # ------------------------------------------------------------------ #
    def begin_switch(
        self,
        *,
        old_provider_id: str,
        old_model: str,
        new_provider_id: str,
        new_model: str,
        snapshot: list[dict] | None = None,
    ) -> SwitchState:
        return SwitchState(
            old_provider_id=old_provider_id,
            new_provider_id=new_provider_id,
            old_model=old_model,
            new_model=new_model,
            switched_at=time.time(),
            pending_snapshot=list(snapshot or []),
        )

    async def prepare_digest(self, state: SwitchState) -> str:
        """生成交接摘要：优先小模型压缩，失败/关闭则规则抽取。"""
        snapshot = state.pending_snapshot or []
        if not snapshot:
            state.digest = state.digest or "（无历史需要交接，本轮是切换后的新对话。）"
            return state.digest

        sample = snapshot[-self.config.digest_max_messages * 2:] if (
            self.config.digest_max_messages > 0
        ) else snapshot

        if self.config.enable_summary and self.summarizer is not None:
            try:
                text = await self.summarizer(build_digest_prompt(sample, state.old_model))
                if text and text.strip():
                    state.digest = text.strip()
                    return state.digest
            except Exception:  # noqa: BLE001 - 摘要失败必须降级而非中断对话
                pass

        anchor = anchor_from_messages(sample) or {
            "topics": [], "facts": [], "todos": [],
            "last_reply_tail": "", "message_count": 0,
        }
        state.digest = fallback_digest(anchor, state.old_model)
        return state.digest

    # ------------------------------------------------------------------ #
    # 请求改写
    # ------------------------------------------------------------------ #
    async def apply_to_request(
        self,
        state: SwitchState | None,
        *,
        contexts: list[dict],
        system_prompt: str,
        user_text: str,
        supports_image: bool = True,
        supports_audio: bool = False,
        force_sanitize: bool = False,
    ) -> RequestPatch:
        """在 OnLLMRequestEvent 里调用。state=None 表示无切换，原样返回。"""
        if state is None or state.turns_applied >= self.config.handoff_turns:
            return RequestPatch(contexts, system_prompt, None, False)

        cfg = self.config
        sanitized, report = sanitize_history(
            contexts,
            supports_image=supports_image,
            supports_audio=supports_audio,
        )

        if cfg.recent_turns > 0 or cfg.max_chars > 0:
            plan: WindowPlan = plan_window(
                sanitized,
                recent_turns=cfg.recent_turns,
                max_chars=cfg.max_chars if cfg.max_chars > 0 else None,
            )
            contexts_out = plan.contexts
            if plan.anchor:
                extra = "（更早的历史要点）" + fallback_digest(plan.anchor, state.old_model)
                if extra not in state.digest:
                    state.digest = (state.digest + "\n" + extra).strip()
        else:
            contexts_out = sanitized
            plan = WindowPlan(contexts_out, 0, 0, None)

        if not state.digest:
            await self.prepare_digest(state)

        block = build_handoff_block(
            old_model=state.old_model,
            new_model=state.new_model,
            digest=state.digest,
            include_names=cfg.include_model_names,
            extra_rules=cfg.extra_rules,
        )

        if cfg.touch_system_prompt:
            system_out = inject_system_prompt(system_prompt, block)
            user_prefix = None
        else:
            system_out = system_prompt
            user_prefix = block + "\n\n"

        if force_sanitize or report.dropped_messages or report.stripped_reasoning_parts:
            contexts_out = contexts_out

        state.turns_applied += 1
        state.last_applied_turn_at = time.time()

        return RequestPatch(
            contexts=contexts_out,
            system_prompt=system_out,
            user_prefix=user_prefix,
            applied=True,
            stats={
                "sanitize": report.as_dict(),
                "window": {
                    "dropped_messages": plan.dropped_messages,
                    "kept_turns": plan.kept_turns,
                },
                "turn_index": state.turns_applied,
                "history_before": len(contexts),
                "history_after": len(contexts_out),
            },
        )
