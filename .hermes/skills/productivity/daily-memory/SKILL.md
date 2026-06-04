---
name: daily-memory
description: "Use when asked to generate a daily memory summary, create a Memory.md, or set up daily memory auto-generation. Searches today's sessions, extracts key decisions/lessons/bugs, and writes a structured markdown file to .hermes/memories/YYYY-MM-DD.md."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [memory, daily, auto-generation, self-evolving]
    related_skills: []
---

# Daily Memory Generator

## Overview

Generates a daily Memory.md file that captures what was learned, decided, built, and fixed during the day's Hermes sessions. This is the self-evolution mechanism — every day's insights compound into a searchable knowledge base.

The file is written to `<project_root>/.hermes/memories/YYYY-MM-DD.md` (or `~/.hermes/memories/YYYY-MM-DD.md` if no project context).

## When to Use

- User asks "generate today's memory" / "create Memory.md" / "summarize today"
- User wants to set up automatic daily memory generation
- Cron job fires to generate the daily memory

## Workflow

### 1. Discover today's sessions

Use `session_search()` with `sort="newest"` to find sessions from today. Scan the most recent 5-10 sessions. Identify which ones occurred today (check timestamps in the results).

For each session, scroll through the bookends:
- `bookend_start`: what was the goal?
- `bookend_end`: what was the resolution?
- The FTS hit context: key decisions or bugs found mid-session

### 2. Extract the signal

From today's sessions, pull out:
- **Key Decisions**: architectural choices, tool selections, workflow changes
- **Bugs Found & Fixed**: what broke, root cause, fix applied
- **Lessons Learned**: "today I learned..." moments, pitfalls discovered
- **Files Touched**: what was created, edited, deleted
- **Skills/Memories Created**: new persistent knowledge stored
- **People/Mentions**: who collaborated, who fixed what

### 3. Generate the markdown file

Use this template:

```markdown
# Memory — YYYY-MM-DD

## Summary
One paragraph capturing the day's theme.

## Key Decisions
### 1. Decision title
Context, options considered, choice made, rationale.

## Bugs Found & Fixed
### Bug: Short description
- **Root cause**: ...
- **Fix**: ...
- **File**: ...
- **Status**: Fixed / Open

## Lessons Learned
1. Lesson one
2. Lesson two

## Files Touched
| File | Action |
|---|---|
| path/to/file | Created/Edited/Reviewed |

## Metrics (optional)
- Sessions today: N
- Skills created: N
- Bugs found: N
- Bugs fixed: N
```

### 4. Save the file

Write to `<project_root>/.hermes/memories/YYYY-MM-DD.md`. Create the directory if needed.

If there's no active project context, write to `~/.hermes/memories/YYYY-MM-DD.md`.

### 5. Cross-save to persistent memory

For the most important 2-3 facts (ones that will matter next week), also save to persistent memory using the `memory` tool. The markdown file is the detailed record; memory is the compact durable index.

## Cron Job Setup

To auto-generate daily, create a cron job:

```
cronjob(action='create',
  name='Daily Memory',
  schedule='0 21 * * *',   # 9 PM daily
  prompt='Generate today\'s daily memory. Search recent sessions from today, extract key decisions, lessons, bugs, and files. Write to .hermes/memories/YYYY-MM-DD.md. Cross-save the top 2-3 facts to persistent memory.',
  skills=['daily-memory'],
  deliver='local',          # save only, no notification
)
```

Schedule at end of day (e.g., 9 PM) so it captures the full day's work. Use `deliver='local'` to avoid notifications — the file IS the deliverable.

## Common Pitfalls

1. **Sessions span midnight.** A session started at 11 PM and ending at 1 AM belongs to the earlier day's memory. Check `bookend_start` timestamps to assign correctly.
2. **Empty days.** If no sessions today, generate a minimal file: `# Memory — YYYY-MM-DD\n\nNo sessions today.` Don't fabricate content.
3. **Overwriting.** If today's memory file already exists (e.g., mid-day generation), merge new findings rather than overwriting. Use `read_file` to load existing content, then append.
4. **Too much detail.** The file should be scannable in 2-3 minutes. Not a full transcript. Link to session IDs for deep dives.

## Verification Checklist

- [ ] All today's sessions surfaced via session_search
- [ ] Key decisions extracted with rationale
- [ ] Bugs documented with root cause and fix
- [ ] Files table complete
- [ ] Top 2-3 facts cross-saved to persistent memory
- [ ] File saved to correct path with today's date
- [ ] Markdown renders cleanly (headings, tables, code blocks)
