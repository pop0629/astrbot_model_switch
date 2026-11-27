"""跨版本兼容层：把 AstrBot 4.x 各版本的 provider 切换 / 会话隔离 API 收敛成一套。

背景：`astrbot.core.session_overrides` 是较新版本才引入的模块，旧版本的 provider
切换只有全局开关，没有会话级 override。这里用 inspect 自适应新旧签名，
所有版本差异都收口在本文件，其它模块不再直接 import 版本相关模块。
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger("astrbot")

PROVIDER_KEY = "agent.config.model.provider_id"


def _load_session_overrides():
    try:
        from astrbot.core import session_overrides as so  # noqa: WPS433

        return so
    except Exception:  # noqa: BLE001 - 旧版本没有该模块，属正常情况
        return None


SESSION_OVERRIDES = _load_session_overrides()


def _provider_type():
    from astrbot.core.provider.entities import ProviderType

    return ProviderType.CHAT_COMPLETION


def _call_kwargs(fn, **cand) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(cand)
    accepted = {k: v for k, v in cand.items() if k in sig.parameters}
    return accepted


async def _set_once(manager, provider_id: str, umo: str | None) -> None:
    fn = manager.set_provider
    kw = _call_kwargs(fn, provider_id=provider_id, provider_type=_provider_type(), umo=umo)
    if len(kw) >= 2 and ("provider_id" in kw or "provider_type" in kw):
        try:
            return await fn(**kw)
        except TypeError:
            if "umo" not in kw:
                raise
    # 老签名：位置参数 (provider_id, provider_type, umo)
    return await fn(provider_id, _provider_type(), umo)


async def set_chat_provider(context, provider_id: str, umo: str) -> str:
    """切换对话模型，返回 'session' | 'global' 表示作用范围。"""
    manager = context.provider_manager
    if SESSION_OVERRIDES is not None:
        try:
            await _set_once(manager, provider_id, umo)
            return "session"
        except TypeError as e:  # 老版本不接受 umo
            logger.debug(f"[model_switch] set_provider(umo=...) 不被支持，转全局切换: {e}")
    try:
        await _set_once(manager, provider_id, None)
    except TypeError:
        await manager.set_provider(provider_id, _provider_type())
    return "global"


async def get_session_override_id(umo: str) -> str | None:
    """当前会话是否有独立的模型 override（新版本才有）。"""
    if SESSION_OVERRIDES is None:
        return None
    try:
        overrides = await SESSION_OVERRIDES.get_overrides(umo)
        return overrides.get(PROVIDER_KEY)
    except Exception:  # noqa: BLE001
        return None


async def clear_session_override(umo: str) -> bool:
    """清除会话级模型 override。返回是否真的执行了（旧版本返回 False）。"""
    if SESSION_OVERRIDES is None:
        return False
    try:
        await SESSION_OVERRIDES.update_overrides(umo, {PROVIDER_KEY: None})
        return True
    except Exception:  # noqa: BLE001
        return False


def supports_session_isolation() -> bool:
    return SESSION_OVERRIDES is not None
