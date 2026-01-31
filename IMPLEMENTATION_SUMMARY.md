# Implementation Summary: Collapsible Tool History (Issue #46)

## Overview

Implemented collapsible tool history feature where assistant turns with summaries automatically group their preceding tool calls into a single expandable item in the TUI.

## What Changed

### 1. New Data Model (`models.py`)

Added `TurnGroup` dataclass to represent a group of turns:
- Contains an assistant turn (with summary) and its related tool turns
- Has its own `expanded` state independent of individual turns
- Provides helper properties: `tool_count`, `tool_names_preview`

### 2. Grouping Logic (`tui.py`)

Added `group_turns_for_display()` function:
- Reuses `_get_turn_context()` from summarizer to determine grouping scope
- Two-pass algorithm:
  1. First pass: identify tool turns that belong to groups
  2. Second pass: build result list with TurnGroup objects
- Only groups when:
  - Summary mode is enabled (`use_summary=True`)
  - Filter is set to "all" (not when filtering by specific tools)
  - Assistant turn has a summary

### 3. Rendering (`styles.py`)

Added `render_turn_group()` method to both style renderers:

**ClaudeCodeStyle (minimal):**
- Collapsed: Shows `[tldr]` with summary + tool count annotation
- Expanded: Shows `[-]` with full assistant text + indented tool list
- Uses same bullet style as assistant turns

**RichStyle (boxed):**
- Collapsed: Panel with title showing tool count, content shows summary only
- Expanded: Panel shows full assistant text + list of tools used
- Border style matches assistant turns (green/white when selected)

### 4. TUI Updates (`tui.py`)

**Modified methods:**
- `_get_filtered_turns()`: Returns `list[Turn | TurnGroup]`, applies grouping for "all" filter with summaries
- `_render_turns()`: Handles both Turn and TurnGroup, delegates to appropriate render method
- `_get_active_turns()`: Updated return type
- Keyboard handlers: Space/Enter/o/e/c now work with TurnGroup.expanded
- `scroll_to_bottom()`: Handles mixed Turn/TurnGroup for height calculations

**Conversation turn numbering:**
- Groups count as a single conversation turn
- Both individual turns and groups increment the counter correctly

## User Experience

### Before
```
[user] Please update the file
[tool] Read file.py
[tool] Write file.py (10 lines)
[tool] Bash pytest
[assistant] I've updated the file...
```

### After (collapsed)
```
[user] Please update the file
[tools + assistant] [tldr] + tools:3 Read file.py, wrote 10 lines to file.py, and ran pytest...
```

### After (expanded with Space/Enter)
```
[user] Please update the file
[tools + assistant] [-] + tools:3
  Full response: I've updated the file with the changes you requested.
  [read] file.py
  [write] file.py (10 lines)
  [$] pytest
```

## Keyboard Navigation

- **j/k**: Navigate between items (groups are single units)
- **Space/Enter/o**: Toggle expansion of group or individual turn
- **e**: Expand all (both groups and individual turns)
- **c**: Collapse all (both groups and individual turns)
- **f**: Cycle filters (grouping only applies to "all" filter)
- **s**: Toggle summary mode (disables grouping when off)

## Edge Cases Handled

1. **Assistant without summary**: No grouping, shows normally
2. **Assistant with summary but no tools**: Shows as individual turn (not grouped)
3. **Tool turns without following assistant**: Show individually (orphaned tools)
4. **Filtered views**: Grouping disabled when filtering by specific tool types
5. **Historical vs live turns**: Grouping works for both
6. **User turns**: Never grouped, always shown separately

## Testing Performed

### Unit Tests
- Grouping with summary enabled: 5 turns → 2 items (user + group)
- Grouping with summary disabled: Returns original list unchanged
- Assistant without summary: No grouping applied

### Manual Testing Checklist
- [ ] Start claude-companion and trigger a session with tool calls
- [ ] Verify tools + assistant collapse into one item when summary appears
- [ ] Press Space to expand, verify tools and full text appear
- [ ] Press Space again to collapse back to summary
- [ ] Navigate with j/k, verify selection moves group-by-group
- [ ] Press 'e' to expand all, verify groups expand
- [ ] Press 'c' to collapse all, verify groups collapse
- [ ] Press 'f' to filter by tool type, verify grouping disappears
- [ ] Press 'f' back to "all", verify grouping returns
- [ ] Press 's' to disable summaries, verify grouping disappears
- [ ] Press 's' to re-enable, verify grouping returns
- [ ] Test with both rich and claude-code styles

## Files Modified

- `src/claude_companion/models.py` - Added TurnGroup dataclass
- `src/claude_companion/tui.py` - Added grouping logic, updated rendering/navigation
- `src/claude_companion/styles.py` - Added render_turn_group() to both styles

## Backward Compatibility

✅ Fully backward compatible:
- No changes to data storage or session files
- Grouping is purely a display-time feature
- Works with existing summaries
- Degrades gracefully when summaries unavailable

## Performance Considerations

- Grouping runs on every render when filter is "all" and summaries enabled
- Two-pass algorithm is O(n) where n = number of turns
- Negligible overhead for typical session sizes (< 1000 turns)
- Could optimize with caching if needed, but not necessary currently
