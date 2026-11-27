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
    """切换对话模型，返回 'session' | 'global' 表示作用范围。

    三条路径按版本依次尝试，全部收口在这里：
    1. 新版：provider_manager.set_provider(..., umo=umo) 写 session_overrides；
    2. 旧版（4.0.0 起）：管理器自己不认 umo 时，直接写
       sp.session_put(umo, "provider_perf_chat_completion", provider_id)；
    3. 再不行才降级为全局切换。
    """
    manager = context.provider_manager
    errors: list[str] = []

    if SESSION_OVERRIDES is not None:
        try:
            await _set_once(manager, provider_id, umo)
            return "session"
        except TypeError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))

    # 旧版会话隔离：直接写会话级 KV（4.0.0 的 set_provider 内部就是这么写的）
    try:
        await _session_put_legacy(umo, provider_id)
        return "session"
    except Exception as e:  # noqa: BLE001
        errors.append(str(e))

    logger.debug(f"[model_switch] 会话级切换不可用，降级全局: {'; '.join(errors)[:300]}")
    try:
        await _set_once(manager, provider_id, None)
    except TypeError:
        await manager.set_provider(provider_id, _provider_type())
    return "global"


async def _session_put_legacy(umo: str, provider_id: str) -> None:
    from astrbot.api import sp

    key = "provider_perf_chat_completion"
    if hasattr(sp, "session_put"):
        await sp.session_put(umo, key, provider_id)
    else:
        await sp.put_async("umo", umo, key, provider_id)


async def _session_read_legacy(umo: str):
    from astrbot.api import sp

    key = "provider_perf_chat_completion"
    try:
        if hasattr(sp, "session_get"):
            return await sp.session_get(umo, key, None)
        return await sp.get_async("umo", umo, key, None)
    except Exception:  # noqa: BLE001
        return None


async def get_session_override_id(umo: str) -> str | None:
    """当前会话是否有独立的模型 override（新旧版本都查一遍）。"""
    if SESSION_OVERRIDES is not None:
        try:
            overrides = await SESSION_OVERRIDES.get_overrides(umo)
            if overrides.get(PROVIDER_KEY):
                return overrides[PROVIDER_KEY]
        except Exception:  # noqa: BLE001
            pass
    return await _session_read_legacy(umo)


async def clear_session_override(umo: str) -> bool:
    """清除会话级模型 override。返回是否执行过任意一条清理路径。"""
    done = False
    if SESSION_OVERRIDES is not None:
        try:
            await SESSION_OVERRIDES.update_overrides(umo, {PROVIDER_KEY: None})
            done = True
        except Exception:  # noqa: BLE001
            pass
    try:
        from astrbot.api import sp

        key = "provider_perf_chat_completion"
        if hasattr(sp, "session_remove"):
            await sp.session_remove(umo, key)
        else:
            await sp.put_async("umo", umo, key, None)
        done = True
    except Exception:  # noqa: BLE001
        pass
    return done


def supports_session_isolation() -> bool:
    """会话级隔离是否可用：新版写 session_overrides，旧版写会话 KV，两条路必有一条。"""
    return True
