"""连续性引擎包。"""
from .continuity import (
    SanitizeReport,
    SwitchState,
    WindowPlan,
    anchor_from_messages,
    build_digest_prompt,
    build_handoff_block,
    fallback_digest,
    inject_system_prompt,
    one_shot_prefix,
    plan_window,
    sanitize_history,
)
from .engine import ContinuityEngine, EngineConfig, RequestPatch


__all__ = [
    "SanitizeReport",
    "SwitchState",
    "WindowPlan",
    "anchor_from_messages",
    "build_digest_prompt",
    "build_handoff_block",
    "fallback_digest",
    "inject_system_prompt",
    "one_shot_prefix",
    "plan_window",
    "sanitize_history",
    "ContinuityEngine",
    "EngineConfig",
    "RequestPatch",
]
