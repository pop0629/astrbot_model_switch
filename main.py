"""无缝模型切换 —— AstrBot 插件主入口。

切换模型本身只需要一行 `provider_manager.set_provider()`；真正麻烦的是"切换后
新模型还能不能接得上这段对话"。本插件在切换的瞬间做三件事：

1. 记录交接状态（旧模型 / 新模型 / 历史快照），交给 ContinuityEngine；
2. 用摘要模型（可选）把旧历史压成交接摘要，失败自动降级为规则锚点；
3. 在接下来 N 轮的 OnLLMRequestEvent 里清洗历史、收紧窗口、注入一次性交接块。

设计原则：
- 不改用户历史原貌：改动只作用于"本次请求的内存副本"，对话存档不变。
- 交接块只生效 N 轮，之后自然回到常态，不产生永久污染。
- 任何一步失败都不阻断对话：摘要挂了用规则，清洗失败退回原文。
"""

from __future__ import annotations

from typing import Any

from astrbot.api import AstrBotConfig, logger, sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.core.provider.entities import ProviderType
from astrbot.core.session_overrides import PROVIDER_PATHS, get_overrides, update_overrides

from .core import ContinuityEngine, EngineConfig, SwitchState, sanitize_history

# 便于外部以包方式导入（测试 / 其它插件）
__all__ = ["Main", "SwitchState", "ContinuityEngine"]

KV_SCOPE = "plugin"
KV_SCOPE_ID = "astrbot_model_switch"
KV_KEY_PREFIX = "handoff:"


class Main(star.Star):
    def __init__(self, context: star.Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._active_umo: dict[str, str] = {}
        self.engine = self._build_engine()

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
        return default if value is None else value

    def _build_engine(self) -> ContinuityEngine:
        cfg = EngineConfig(
            handoff_turns=int(self._cfg("handoff_turns", 2)),
            recent_turns=int(self._cfg("recent_turns", 6)),
            max_chars=int(self._cfg("max_chars", 12000)),
            sanitize_tools=bool(self._cfg("sanitize_tools", True)),
            enable_summary=bool(self._cfg("enable_summary", True)),
            summarizer_provider_id=str(self._cfg("summarizer_provider_id", "")),
            digest_max_messages=int(self._cfg("digest_max_messages", 40)),
            touch_system_prompt=bool(self._cfg("touch_system_prompt", True)),
            include_model_names=bool(self._cfg("include_model_names", False)),
            extra_rules=str(self._cfg("extra_rules", "")),
        )
        return ContinuityEngine(cfg, summarizer=self._summarize)

    # ------------------------------------------------------------------ #
    # 状态存取（共享偏好，按 umo 隔离）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(umo: str) -> str:
        return f"{KV_KEY_PREFIX}{umo}"

    async def _load_state(self, umo: str) -> SwitchState | None:
        raw = await sp.get_async(KV_SCOPE, KV_SCOPE_ID, self._key(umo), None)
        if isinstance(raw, dict):
            return SwitchState.from_dict(raw)
        return SwitchState.from_json(raw) if isinstance(raw, str) else None

    async def _save_state(self, umo: str, state: SwitchState | None) -> None:
        await sp.put_async(KV_SCOPE, KV_SCOPE_ID, self._key(umo), None if state is None else state.to_dict())

    # ------------------------------------------------------------------ #
    # provider 工具
    # ------------------------------------------------------------------ #
    def _chat_providers(self) -> list:
        try:
            return list(self.context.get_all_providers())
        except Exception:  # noqa: BLE001
            return []

    async def _resolve_current_id(self, umo: str) -> str:
        try:
            prov = await self.context.get_using_provider_async(umo=umo)
        except Exception:  # noqa: BLE001
            prov = None
        try:
            return prov.meta().id if prov else ""
        except Exception:  # noqa: BLE001
            return ""

    def _provider_model(self, provider_id: str) -> str:
        if not provider_id:
            return ""
        try:
            prov = self.context.get_provider_by_id(provider_id)
            if prov is None:
                return provider_id
            meta = prov.meta()
            return f"{meta.model or ''}[{meta.type}]" if meta.model else str(meta.type)
        except Exception:  # noqa: BLE001
            return provider_id

    def _find_provider(self, token: str) -> str | None:
        """按 1-based 序号 / 精确 id / id 子串 / 模型名 匹配。"""
        providers = self._chat_providers()
        if not providers:
            return None
        ids = []
        models = []
        for p in providers:
            try:
                meta = p.meta()
                ids.append(meta.id)
                models.append((meta.model or "").lower())
            except Exception:  # noqa: BLE001
                ids.append(getattr(p, "provider_config", {}).get("id", ""))
                models.append("")

        if token.isdigit():
            idx = int(token)
            return ids[idx - 1] if 1 <= idx <= len(ids) else None
        if token in ids:
            return token
        low = token.lower()
        sub = [i for i in ids if low and low in i.lower()]
        if len(sub) == 1:
            return sub[0]
        by_model = [ids[i] for i, m in enumerate(models) if m and (m == low or low in m)]
        if len(by_model) == 1:
            return by_model[0]
        merged = list(dict.fromkeys(sub + by_model))
        return merged[0] if merged else None

    async def _snapshot_history(self, event: AstrMessageEvent) -> list[dict]:
        try:
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
            if not cid:
                return []
            conv = await conv_mgr.db.get_conversation_by_id(cid)
            return list(conv.content or []) if conv else []
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[model_switch] history snapshot failed: {e}")
            return []

    def _last_umo(self) -> str | None:
        """最近一次请求/切换所属会话：用于在没有显式 umo 时定位当前模型。"""
        return self._active_umo.get("umo")

    async def _summarize(self, prompt: str) -> str:
        provider = None
        want = self.engine.config.summarizer_provider_id
        if want:
            provider = self.context.get_provider_by_id(want)
        if provider is None:
            provider = await self.context.get_using_provider_async(umo=self._last_umo())
        if provider is None:
            return ""
        resp = await provider.text_chat(prompt=prompt, session_id=None, contexts=[])
        return getattr(resp, "completion_text", "") or ""

    def _supports(self, provider, modality: str, default: bool = False) -> bool:
        try:
            from astrbot.core.provider.capabilities import supports_modality  # type: ignore

            return bool(supports_modality(provider, modality))
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ #
    # 指令
    # ------------------------------------------------------------------ #
    @filter.command("modelswitch", alias={"ms", "切换模型", "换模型"})
    async def modelswitch(self, event: AstrMessageEvent, target: str = "") -> None:
        """无缝切换模型。用法: /modelswitch <序号|id|模型名>；/modelswitch list|status|reset"""
        umo = event.unified_msg_origin
        token = (target or "").strip()
        low = token.lower()

        if not token or low in ("list", "ls", "列表"):
            event.set_result(MessageEventResult().message(self._render_list()))
            return
        if low in ("status", "状态"):
            event.set_result(MessageEventResult().message(await self._render_status(umo)))
            return
        if low in ("reset", "clear", "重置"):
            await self._save_state(umo, None)
            event.set_result(MessageEventResult().message("已清除本会话的交接状态。"))
            return

        target_id = self._find_provider(token)
        if not target_id:
            event.set_result(
                MessageEventResult().message(f"没找到匹配『{token}』的模型。\n\n{self._render_list()}")
            )
            return

        event.set_result(MessageEventResult().message(await self._do_switch(event, target_id)))

    async def _do_switch(self, event: AstrMessageEvent, target_id: str) -> str:
        umo = event.unified_msg_origin
        self._active_umo["umo"] = umo
        old_id = await self._resolve_current_id(umo)
        if old_id == target_id:
            return f"当前已经在用 {target_id}，无需切换。"

        snapshot = await self._snapshot_history(event)
        state = self.engine.begin_switch(
            old_provider_id=old_id,
            old_model=self._provider_model(old_id),
            new_provider_id=target_id,
            new_model=self._provider_model(target_id),
            snapshot=snapshot,
        )

        # 1) 真正执行切换（会话级 override，不影响其它会话）
        try:
            await self.context.provider_manager.set_provider(
                provider_id=target_id,
                provider_type=ProviderType.CHAT_COMPLETION,
                umo=umo,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[model_switch] set_provider failed: {e}")
            return f"切换失败：{e}"

        # 2) 预先备好交接摘要（失败自动降级，不阻断对话）
        if snapshot:
            try:
                await self.engine.prepare_digest(state)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[model_switch] digest failed: {e}")
        await self._save_state(umo, state)

        if not self._cfg("announce_switch", True):
            return "已切换。"
        turns = self.engine.config.handoff_turns
        if turns > 0:
            return f"已切到 {target_id}。接下来 {turns} 轮用交接模式衔接上下文。"
        return f"已切到 {target_id}。"

    # ------------------------------------------------------------------ #
    # 核心：请求改写
    # ------------------------------------------------------------------ #
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req) -> None:
        umo = event.unified_msg_origin
        try:
            cfg = self.engine.config
            contexts = list(req.contexts or [])
            provider = await self.context.get_using_provider_async(umo=umo)
            supports_image = self._supports(provider, "image", default=True)
            supports_audio = self._supports(provider, "audio", default=False)

            state = await self._load_state(umo)

            # 直接清洗模式：handoff_turns=0，或已有状态但已用完且仍开着清洗
            if state is None or state.turns_applied >= cfg.handoff_turns:
                if cfg.sanitize_tools or state is not None:
                    cleaned, report = sanitize_history(
                        contexts,
                        supports_image=supports_image,
                        supports_audio=supports_audio,
                    )
                    if report.dropped_messages or report.stripped_reasoning_parts:
                        req.contexts = cleaned
                        logger.debug(f"[model_switch] cleaned history: {report.as_dict()}")
                if state is not None:
                    await self._save_state(umo, None)
                return

            # 用户在面板/别处直接换了 provider：把交接对象校正为真实当前模型
            real_id = await self._resolve_current_id(umo)
            if real_id and real_id != state.new_provider_id:
                state.new_provider_id = real_id
                state.new_model = self._provider_model(real_id)
                state.turns_applied = 0

            patch = await self.engine.apply_to_request(
                state,
                contexts=contexts,
                system_prompt=str(req.system_prompt or ""),
                user_text=req.prompt or "",
                supports_image=supports_image,
                supports_audio=supports_audio,
                force_sanitize=cfg.sanitize_tools,
            )
            if patch.applied:
                req.contexts = patch.contexts
                req.system_prompt = patch.system_prompt
                if patch.user_prefix:
                    req.prompt = (
                        patch.user_prefix + req.prompt if req.prompt else patch.user_prefix
                    )
                logger.info(f"[model_switch] handoff applied: {patch.stats}")

            await self._save_state(umo, state)
        except Exception as e:  # noqa: BLE001 - 绝不因交接逻辑打断对话
            logger.error(f"[model_switch] on_llm_request error: {e}")

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #
    def _render_list(self) -> str:
        providers = self._chat_providers()
        if not providers:
            return "还没有可用的对话模型。"
        lines = ["可用模型（/modelswitch <序号或id> 切换）："]
        for i, p in enumerate(providers, 1):
            try:
                meta = p.meta()
                lines.append(f"{i}. {meta.model or meta.id} [{meta.type}]  ({meta.id})")
            except Exception:  # noqa: BLE001
                pid = getattr(p, "provider_config", {}).get("id", "?")
                lines.append(f"{i}. {pid}")
        lines.append("另有：/modelswitch status、/modelswitch reset")
        return "\n".join(lines)

    async def _render_status(self, umo: str) -> str:
        state = await self._load_state(umo)
        cur = await self._resolve_current_id(umo)
        try:
            overrides = await get_overrides(umo)
        except Exception:  # noqa: BLE001
            overrides = {}
        override_id = overrides.get(PROVIDER_PATHS["chat_completion"]) or "（继承全局）"
        lines = [f"当前模型: {cur or '未知'}", f"会话覆盖: {override_id}"]
        if state:
            lines.append(
                f"交接状态: {state.old_model or state.old_provider_id} -> "
                f"{state.new_model or state.new_provider_id}，已用 "
                f"{state.turns_applied}/{self.engine.config.handoff_turns} 轮"
            )
            if state.digest:
                digest = state.digest if len(state.digest) < 400 else state.digest[:400] + "…"
                lines.append(f"交接摘要:\n{digest}")
        else:
            lines.append("交接状态: 无（下一轮恢复常态）")
        return "\n".join(lines)

    @filter.command("modelswitch_clear_override")
    async def clear_override(self, event: AstrMessageEvent) -> None:
        """清除本会话的模型覆盖，回到全局模型。"""
        umo = event.unified_msg_origin
        await update_overrides(umo, {PROVIDER_PATHS["chat_completion"]: None})
        await self._save_state(umo, None)
        event.set_result(MessageEventResult().message("已恢复全局模型，交接状态已清除。"))
