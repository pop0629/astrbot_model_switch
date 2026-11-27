"""核心连续性模块的单元测试（不依赖 astrbot，可直接 python3 -m unittest 跑）。"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    ContinuityEngine,
    EngineConfig,
    SwitchState,
    anchor_from_messages,
    build_digest_prompt,
    build_handoff_block,
    fallback_digest,
    plan_window,
    sanitize_history,
)


def convo(n_turns: int) -> list[dict]:
    out = [{"role": "system", "content": "SYS"}]
    for i in range(n_turns):
        out.append({"role": "user", "content": f"user-{i}"})
        out.append({"role": "assistant", "content": f"assistant-{i}"})
    return out


class SanitizeTest(unittest.TestCase):
    def test_drops_orphan_tool_and_tool_calls(self):
        ctx = [
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "晴 25C"},
            {"role": "assistant", "content": "今天晴，25 度。"},
        ]
        out, rep = sanitize_history(ctx)
        roles = [m["role"] for m in out]
        self.assertNotIn("tool", roles)
        self.assertEqual(rep.dropped_tool_messages, 1)
        # 空的 assistant + tool_calls 不能留空 content
        self.assertTrue(all(m["content"] for m in out))
        self.assertEqual(rep.closed_unfinished_tools, 1)

    def test_strips_reasoning_parts(self):
        ctx = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "答案"}]}]
        out, rep = sanitize_history(ctx, keep_reasoning=False)
        self.assertEqual(rep.stripped_reasoning_parts, 1)
        self.assertEqual(out[0]["content"], "答案")

    def test_unsupported_modality_replaced(self):
        ctx = [{"role": "user", "content": [
            {"type": "text", "text": "看图"},
            {"type": "image_url", "image_url": {"url": "x"}}]}]
        out, rep = sanitize_history(ctx, supports_image=False)
        self.assertEqual(rep.dropped_modality_parts, 1)
        text = " ".join(p.get("text", "") for p in out[0]["content"] if isinstance(p, dict))
        self.assertIn("unsupported", text)

    def test_unknown_role_dropped(self):
        out, rep = sanitize_history([{"role": "narrator", "content": "hi"},
                                     {"role": "user", "content": "ok"}])
        self.assertEqual([m["role"] for m in out], ["user"])


class WindowTest(unittest.TestCase):
    def test_keeps_recent_turns_and_anchor(self):
        ctx = convo(20)
        plan = plan_window(ctx, recent_turns=4, max_chars=0)
        self.assertEqual(plan.kept_turns, 4)
        self.assertTrue(plan.contexts[0]["role"] == "system")
        self.assertEqual(plan.contexts[1]["content"], "user-16")
        self.assertIsNotNone(plan.anchor)
        self.assertTrue(plan.anchor["topics"])

    def test_char_budget_does_not_start_with_assistant(self):
        ctx = convo(10)
        plan = plan_window(ctx, recent_turns=10, max_chars=60)
        self.assertNotEqual(plan.contexts[1]["role"], "assistant")
        self.assertLess(plan.kept_turns, 10)

    def test_no_anchor_when_nothing_dropped(self):
        plan = plan_window(convo(3), recent_turns=5, max_chars=0)
        self.assertIsNone(plan.anchor)


class DigestTest(unittest.TestCase):
    def test_anchor_extracts_facts_and_todos(self):
        msgs = [
            {"role": "user", "content": "我们约定接口用 REST"},
            {"role": "assistant", "content": "好的，采用 REST。"},
            {"role": "user", "content": "下一步接入鉴权，还没做"},
        ]
        a = anchor_from_messages(msgs)
        self.assertTrue(any("REST" in f for f in a["facts"]))
        self.assertTrue(any("鉴权" in t for t in a["todos"]))
        self.assertIn("REST", fallback_digest(a, "old-model"))

    def test_digest_prompt_has_sections(self):
        p = build_digest_prompt(convo(2)[1:], "gpt-x")
        for sec in ("【议题】", "【已确定事实】", "【未完成】", "【风格备注】"):
            self.assertIn(sec, p)
        self.assertIn("gpt-x", p)

    def test_handoff_block_names_hidden_by_default(self):
        blk = build_handoff_block(old_model="m1", new_model="m2", digest="D")
        self.assertNotIn("m1", blk)
        self.assertIn("D", blk)
        blk2 = build_handoff_block(old_model="m1", new_model="m2", digest="D", include_names=True)
        self.assertIn("m1", blk2)


class EngineTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_no_state_is_passthrough(self):
        eng = ContinuityEngine(EngineConfig())
        ctx = convo(2)
        p = self._run(eng.apply_to_request(None, contexts=ctx, system_prompt="S", user_text="hi"))
        self.assertFalse(p.applied)
        self.assertEqual(p.contexts, ctx)

    def test_switch_applies_for_n_turns_only(self):
        eng = ContinuityEngine(EngineConfig(handoff_turns=2, recent_turns=3, max_chars=0,
                                            enable_summary=False))
        st = eng.begin_switch(old_provider_id="a", old_model="old",
                              new_provider_id="b", new_model="new",
                              snapshot=convo(10)[1:])
        ctx = convo(10)
        p1 = self._run(eng.apply_to_request(st, contexts=ctx, system_prompt="S", user_text="q"))
        self.assertTrue(p1.applied)
        self.assertIn("<model_handoff>", p1.system_prompt)
        self.assertEqual(p1.stats["window"]["kept_turns"], 3)

        p2 = self._run(eng.apply_to_request(st, contexts=ctx, system_prompt="S", user_text="q"))
        self.assertTrue(p2.applied)
        p3 = self._run(eng.apply_to_request(st, contexts=ctx, system_prompt="S", user_text="q"))
        self.assertFalse(p3.applied)
        self.assertEqual(p3.contexts, ctx)

    def test_user_prefix_mode_keeps_system_prompt(self):
        eng = ContinuityEngine(EngineConfig(touch_system_prompt=False, enable_summary=False))
        st = eng.begin_switch(old_provider_id="a", old_model="o",
                              new_provider_id="b", new_model="n", snapshot=[])
        p = self._run(eng.apply_to_request(st, contexts=[], system_prompt="SYS", user_text="q"))
        self.assertEqual(p.system_prompt, "SYS")
        self.assertIn("<model_handoff>", p.user_prefix)

    def test_summarizer_used_and_failure_falls_back(self):
        async def ok(prompt):
            self.assertIn("【议题】", prompt)
            return "【议题】压缩结果"

        eng = ContinuityEngine(EngineConfig(enable_summary=True), summarizer=ok)
        st = eng.begin_switch(old_provider_id="a", old_model="o",
                              new_provider_id="b", new_model="n", snapshot=convo(3)[1:])
        p = self._run(eng.apply_to_request(st, contexts=convo(3), system_prompt="", user_text="q"))
        self.assertIn("压缩结果", p.system_prompt)

        async def boom(prompt):
            raise RuntimeError("summary model down")

        eng2 = ContinuityEngine(EngineConfig(enable_summary=True), summarizer=boom)
        st2 = eng2.begin_switch(old_provider_id="a", old_model="o",
                                new_provider_id="b", new_model="n", snapshot=convo(3)[1:])
        p2 = self._run(eng2.apply_to_request(st2, contexts=convo(3), system_prompt="", user_text="q"))
        self.assertTrue(p2.applied)
        self.assertIn("【议题】", p2.system_prompt)

    def test_state_roundtrip(self):
        st = SwitchState(old_model="o", new_model="n", turns_applied=1)
        back = SwitchState.from_json(st.to_json())
        self.assertEqual(back.turns_applied, 1)
        self.assertEqual(back.new_model, "n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
