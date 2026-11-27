# astrbot_model_switch · 无缝模型切换

在会话内切换 LLM 模型后，尽量让新模型「接得上」同一段对话，减少割裂感。

切换模型本身只是一行 `provider_manager.set_provider()`；真正造成割裂的是切换瞬间的四种错位：

| 错位 | 表现 | 本插件的处理 |
| --- | --- | --- |
| 上下文格式错位 | 历史里残留旧模型的 `tool_calls` / `tool` 消息 / `thinking` 块 / 新模型不支持的图片 part → 直接 400 或答案格式崩坏 | `sanitize_history()` 逐条清洗，可降级为文本痕迹 |
| 上下文体积错位 | 旧模型 256k 窗口塞满的历史，新模型 32k/128k 吃不下 | 切换后首轮按「近期轮次 + 字符预算」收紧窗口，被裁部分转成锚点 |
| 语义错位 | 新模型不知道「刚才在聊什么、哪些已经定了、还有什么没做」 | 摘要模型压出的结构化交接摘要（议题/已定事实/未完成/风格备注），失败自动降级为规则抽取 |
| 行为错位 | 新模型开场自我介绍、要求用户复述、从头再来一遍 | 一次性 `<model_handoff>` 交接块，明确「接着上一句继续，别重述」 |

## 安装

把整个目录放进 AstrBot 的 `data/plugins/` 下（目录名随意，包名保持 `astrbot_model_switch` 或直接是同名目录），重载插件即可。无第三方依赖。

## 指令

```
/modelswitch list           # 列出可用模型（带序号 / id）
/modelswitch 2              # 按序号切换
/modelswitch claude-sonnet  # 按 id 子串或模型名切换（保证唯一匹配）
/modelswitch status         # 当前模型 + 交接状态 + 本轮交接摘要
/modelswitch reset          # 清除交接状态（下一轮恢复常态）
/modelswitch_clear_override # 清除本会话模型覆盖，回到全局模型
```

别名：`/ms`、`/切换模型`、`/换模型`。

## 配置（WebUI 插件配置页）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `handoff_turns` | 2 | 交接模式作用的轮数；0 = 关闭窗口收紧与交接注入，只做历史清洗 |
| `recent_turns` | 6 | 交接模式下保留的近期轮数 |
| `max_chars` | 12000 | 交接模式字符预算，0 = 不限 |
| `sanitize_tools` | true | 清洗 tool / tool_calls / reasoning 块 |
| `enable_summary` | true | 用摘要模型压历史 |
| `summarizer_provider_id` | "" | 摘要模型 provider id；留空 = 用切换后的当前模型；建议配一个便宜的小模型 |
| `digest_max_messages` | 40 | 送摘要的历史条数上限 |
| `touch_system_prompt` | true | true = 交接块写 system prompt 尾部；false = 一次性 user 前缀（保护 provider 端 prompt 缓存） |
| `include_model_names` | false | 交接块里是否写明新旧模型名 |
| `extra_rules` | "" | 追加到交接块的规则 |
| `announce_switch` | true | 切换后在会话里回一条提示 |

## 关键设计

1. **不污染存档**：所有改写只作用于本次请求的内存副本（`req.contexts` / `req.system_prompt`），对话历史原样保存。新模型自己产出的回答进入历史后，再切回旧模型也不会看到交接块。
2. **一次生效、N 轮后退场**：交接块带 `turns_applied` 计数，第 N+1 轮起请求不再被改写，对话自然回到常态。
3. **失败不阻断**：摘要模型挂了用规则锚点；清洗抛错整体 try/except 吞掉并记日志——换模型期间的健壮性优先于完美衔接。
4. **会话级隔离**：走 `set_provider(umo=...)` 的会话 override，只影响当前会话；`/modelswitch status` 会显示 override 来源。
5. **外部改动自愈**：如果用户在面板上直接换了模型，`on_llm_request` 会检测到 `current != state.new_provider_id` 并把交接对象校正为真实模型。

## 结构

```
main.py                 插件入口：指令 + on_llm_request 钩子 + 状态存取
core/continuity.py      纯函数：清洗 / 窗口 / 锚点与摘要 / 交接块
core/engine.py          编排：切换事件 → 摘要 → 请求改写（RequestPatch）
_conf_schema.json       可视化配置
tests/test_continuity.py  15 个单元测试（不依赖 astrbot）
tests/test_plugin_smoke.py 全链路冒烟（伪造 astrbot 运行时）
```

## 测试

```bash
python3 -m unittest discover -s tests -v   # 核心纯函数
python3 tests/test_plugin_smoke.py         # 指令 + 请求改写全链路
```

## 调参建议

- 长对话频繁切换：`recent_turns` 调到 8~10，`max_chars` 按新模型窗口的 40% 设。
- 追求低成本：`enable_summary=false`，只靠规则锚点（零额外调用）。
- 对 prompt 缓存敏感（如 Anthropic 缓存）：`touch_system_prompt=false`，交接块走 user 前缀。
- 若某新模型对多余 system 文本敏感：`handoff_turns=1`，只影响紧接的一轮。
