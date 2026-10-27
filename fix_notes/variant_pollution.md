# 变体污染（Variant Pollution）复盘与三道防线

## 1. 现象与根因链条

在评测「老规矩 A → 临时变体 → 老规矩 B」三连会话时：
1. **Session A（老规矩建立）**：用户声明「以后抖音直播按老规矩 1080p，12:30-13:30」，生成长期节点 `douyin | live`（`resolution=1080p`）。
2. **Session B（临时变体）**：用户说「今天临时调 720p」，LLM 误将其 `validity` 抽为 `long_term`。
3. **Updater 无条件覆盖**：旧版 `_update` 无条件用新事件非空字段覆盖同 `memory_key` 长期节点，导致 `resolution` 被改写为 `720p`。
4. **Session C（老规矩复用）**：用户再次说「按老规矩来」，检索命中已被污染的 `720p`，导致后续预测全部出错。

## 2. 修复三件套

1. **萃取侧确定性归一 (`_normalize_validity`)**：
   - 只看 `evidence_turns` 对应的**用户原话**（忽略 Agent 复述的系统话术）。
   - 含「临时 / 今天先 / 这次先 / 本次 / 暂时 / 就这一次」→ 一票否决强制设为 `temporary`（即使同句出现“老规矩”或“以后”）。
   - 含强长期词（以后 / 默认 / 一直 / 总是 / 每次都 / 长期）或弱长期词（老规矩 / 照旧 / 平时 / 通常 / 一般 / 惯例）→ 强制设为 `long_term`。
   - 同步覆盖 `event.validity` 和 `event.content["validity"]`。
2. **更新侧防线 (`MemoryUpdater._update`)**：
   - 当 `event.validity == "temporary"` 且已有目标节点 `node.validity.type != "temporary"` 时，直接 **no-op**（不 merge 内容、不累加 vote、不改变状态），仅记录 `event_ignored_temporary` 日志。
3. **检索侧降权 (`SearchEngine.search`)**：
   - 若首个会话就是临时变体而创建了 `temporary` 节点，检索重排时对 `validity.type == "temporary"` 的节点总分乘以 `0.5` 降权。
