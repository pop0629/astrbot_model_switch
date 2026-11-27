"""插件主模块冒烟测试：用假的 astrbot 模块验证指令 + on_llm_request 全链路。

不依赖真实 AstrBot 安装。运行：python3 tests/test_plugin_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

# --------------------------------------------------------------------------- #
# 伪造 astrbot 运行时
# --------------------------------------------------------------------------- #


def _install_fakes() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    class _SP:
        def __init__(self):
            self.data = {}

        async def get_async(self, scope, scope_id, key, default=None):
            return self.data.get((scope, scope_id, key), default)

        async def put_async(self, scope, scope_id, key, value):
            self.data[(scope, scope_id, key)] = value

    api.sp = _SP()

    class Context:
        def __init__(self):
            self.config = {}

        def get_config(self, umo=None):
            return self.config

    class Star:
        def __init__(self, context):
            self.context = context

    api.star = types.SimpleNamespace(Context=Context, Star=Star)
    api.AstrBotConfig = dict

    event_mod = types.ModuleType("astrbot.api.event")

    class MessageEventResult:
        def __init__(self):
            self.text = ""

        def message(self, text):
            self.text = text
            return self

    class AstrMessageEvent:  # noqa: D401
        pass

    writes = []

    def command(*names, **kw):
        def deco(fn):
            return fn

        return deco

    def on_llm_request(**kw):
        def deco(fn):
            return fn

        return deco

    event_mod.MessageEventResult = MessageEventResult
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = types.SimpleNamespace(command=command, on_llm_request=on_llm_request)

    entities = types.ModuleType("astrbot.core.provider.entities")

    class ProviderType:
        CHAT_COMPLETION = "chat_completion"

    entities.ProviderType = ProviderType

    so = types.ModuleType("astrbot.core.session_overrides")
    PROVIDER_PATHS = {"chat_completion": "agent.config.model.provider_id"}
    so.PROVIDER_PATHS = PROVIDER_PATHS
    overrides_store = {}

    async def get_overrides(umo):
        return overrides_store.get(umo, {})

    async def update_overrides(umo, changes, **kw):
        cur = overrides_store.setdefault(umo, {})
        for k, v in changes.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        return cur

    so.get_overrides = get_overrides
    so.update_overrides = update_overrides

    for name, mod in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_mod,
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.provider": types.ModuleType("astrbot.core.provider"),
        "astrbot.core.provider.entities": entities,
        "astrbot.core.session_overrides": so,
    }.items():
        sys.modules[name] = mod

    sys.modules["astrbot"].api = api


_install_fakes()
sys.modules["astrbot.api"].event = sys.modules["astrbot.api.event"]
sys.modules["astrbot.core.provider"].entities = sys.modules["astrbot.core.provider.entities"]

import importlib  # noqa: E402

importlib.import_module(ROOT.name)  # 注册包的 __init__（空壳即可）
plugin_main = importlib.import_module(f"{ROOT.name}.main")
SwitchState = plugin_main.SwitchState


# --------------------------------------------------------------------------- #
# 假 provider / context
# --------------------------------------------------------------------------- #


class FakeProvider:
    def __init__(self, pid, model):
        self.pid = pid
        self.model = model
        self.calls = []

    def meta(self):
        return types.SimpleNamespace(id=self.pid, model=self.model, type="openai")

    async def text_chat(self, prompt=None, session_id=None, contexts=None, **kw):
        self.calls.append(prompt)
        return types.SimpleNamespace(completion_text="【议题】交接摘要OK")


class FakeConversation:
    def __init__(self, content):
        self.content = content


class FakeConvMgr:
    def __init__(self, content):
        self.content = content

        class DB:
            async def get_conversation_by_id(_s, cid):
                return FakeConversation(content)

        self.db = DB()

    async def get_curr_conversation_id(self, umo):
        return "cid-1"


class FakeProviderManager:
    def __init__(self, providers):
        self.inst_map = {p.pid: p for p in providers}
        self.current = {None: providers[0].pid}

    async def set_provider(self, provider_id, provider_type, umo=None):
        self.current[umo] = provider_id


class FakeCtx:
    def __init__(self, providers, history):
        self.provider_manager = FakeProviderManager(providers)
        self.providers = providers
        self.conversation_manager = FakeConvMgr(history)

    def get_all_providers(self):
        return list(self.providers)

    def get_provider_by_id(self, pid):
        return self.provider_manager.inst_map.get(pid)

    async def get_using_provider_async(self, umo=None):
        return self.provider_manager.inst_map.get(self.provider_manager.current.get(umo))


class FakeReq:
    def __init__(self, contexts, system_prompt="SYS", prompt="现在继续"):
        self.contexts = contexts
        self.system_prompt = system_prompt
        self.prompt = prompt


class FakeEvent:
    def __init__(self, umo="test:FriendMessage:u1"):
        self.unified_msg_origin = umo
        self.result = None

    def set_result(self, r):
        self.result = r


def convo(n):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


async def run() -> None:
    provs = [FakeProvider("p-old", "gpt-old"), FakeProvider("p-new", "claude-new")]
    ctx = FakeCtx(provs, convo(8))
    cfg = {
        "handoff_turns": 2,
        "recent_turns": 2,
        "max_chars": 0,
        "sanitize_tools": True,
        "enable_summary": True,
        "digest_max_messages": 40,
        "touch_system_prompt": True,
        "announce_switch": True,
    }
    plugin = plugin_main.Main(ctx, cfg)
    ev = FakeEvent()

    # 列出
    await plugin.modelswitch(ev, "list")
    assert "gpt-old" in ev.result.text and "claude-new" in ev.result.text, ev.result.text

    # 按序号切换
    await plugin.modelswitch(ev, "2")
    assert "已切到 p-new" in ev.result.text, ev.result.text
    assert ctx.provider_manager.current[ev.unified_msg_origin] == "p-new"
    # 摘要模型被调用一次
    assert provs[1].calls, "summarizer should be called on new provider"

    # 第一轮请求：注入交接块 + 收紧窗口
    req = FakeReq(convo(8))
    await plugin.on_llm_request(ev, req)
    assert "<model_handoff>" in req.system_prompt
    assert "交接摘要OK" in req.system_prompt
    users = [m for m in req.contexts if m["role"] == "user"]
    assert len(users) == 2, users

    # 第二轮仍生效
    req2 = FakeReq(convo(8))
    await plugin.on_llm_request(ev, req2)
    assert "<model_handoff>" in req2.system_prompt

    # 状态（2/2 已用完）
    await plugin.modelswitch(ev, "status")
    assert "已用 2/2 轮" in ev.result.text, ev.result.text

    # 第三轮恢复常态（不再改写 system_prompt，只做清洗）
    req3 = FakeReq(convo(8))
    await plugin.on_llm_request(ev, req3)
    assert req3.system_prompt == "SYS"
    assert len([m for m in req3.contexts if m["role"] == "user"]) == 8

    # 状态已清空
    await plugin.modelswitch(ev, "status")
    assert "交接状态: 无" in ev.result.text, ev.result.text

    # 同 id 切换 no-op
    await plugin.modelswitch(ev, "p-new")
    assert "无需切换" in ev.result.text

    # 不存在的目标
    await plugin.modelswitch(ev, "zzz")
    assert "没找到" in ev.result.text

    # 切换时带 tool 残留的历史：清洗
    ctx.conversation_manager.content = convo(2) + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    await plugin.modelswitch(ev, "1")
    dirty = convo(2) + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    reqd = FakeReq(dirty)
    await plugin.on_llm_request(ev, reqd)
    assert all(m["role"] != "tool" for m in reqd.contexts)

    # 状态序列化可往返
    st = SwitchState(old_model="a", new_model="b")
    assert SwitchState.from_dict(st.to_dict()).new_model == "b"

    print("ALL SMOKE TESTS PASSED")


async def run_old_version() -> None:
    """模拟旧版 AstrBot：没有 session_overrides 模块、set_provider 用老签名。"""
    saved = sys.modules.pop("astrbot.core.session_overrides", None)
    sys.modules["astrbot.core.session_overrides"] = None  # import 时抛 ImportError
    try:
        import importlib

        compat = importlib.reload(plugin_main.core_compat)
        assert not compat.supports_session_isolation(), "旧版应识别为无会话隔离"

        class OldPM:
            def __init__(self):
                self.last = None

            async def set_provider(self, provider_id, provider_type, umo=None):
                self.last = (provider_id, provider_type, umo)

        class OldCtx(FakeCtx):
            def __init__(self, providers, history):
                super().__init__(providers, history)
                self.provider_manager = OldPM()
                self._provs = providers
                self.provider_manager.inst_map = None

            async def get_using_provider_async(self, umo=None):
                return self._provs[0]

        ctx = OldCtx([FakeProvider("p-old", "m-old"), FakeProvider("p-new", "m-new")], convo(3))
        ctx.provider_manager.inst_map = {p.pid: p for p in ctx._provs}
        plugin = plugin_main.Main(ctx, {"handoff_turns": 1, "enable_summary": False})
        ev = FakeEvent()
        await plugin.modelswitch(ev, "p-new")
        assert "已切到 p-new" in ev.result.text, ev.result.text
        assert "全局切换" in ev.result.text, ev.result.text
        assert ctx.provider_manager.last[0] == "p-new"
        await plugin.modelswitch(ev, "status")
        assert "不支持" in ev.result.text, ev.result.text
        await plugin.clear_override(ev)
        assert "无法自动恢复" in ev.result.text, ev.result.text
        print("OLD-VERSION COMPAT PASSED")
    finally:
        if saved is not None:
            sys.modules["astrbot.core.session_overrides"] = saved
        else:
            sys.modules.pop("astrbot.core.session_overrides", None)
        import importlib

        importlib.reload(plugin_main.core_compat)


if __name__ == "__main__":
    asyncio.run(run())
    asyncio.run(run_old_version())
