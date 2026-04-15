# Make the judge's tool use adaptive + reflective

## Context

Today the judge follows a rigid **plan → tools → feedback** pipeline:

- [_OpenAIToolLoop.run()](gecco/construct_feedback/tool_judge.py#L260-L418) forces a text-only first turn (`tool_choice="none"`) whose prompt demands a committed per-angle tool-call allocation *before* any evidence is seen.
- [_JUDGE_SYSTEM_PROMPT](gecco/construct_feedback/tool_judge.py#L130-L204) reinforces this with *"Plan your allocation across the six angles before calling tools"*.
- The one transition message that says *"briefly reflect on what you learned"* ([tool_judge.py:331-335](gecco/construct_feedback/tool_judge.py#L331-L335)) appears exactly once and is not enforced per call. Gemini ([_GeminiToolLoop.run()](gecco/construct_feedback/tool_judge.py#L456-L595)) has no reflection prompt at all.
- Budget reminders fire at 50% and 80% but do not encourage deeper follow-up on surprising findings.

The user wants the judge to behave more like an agent: think, call tools, see results, think again, optionally dig deeper if results raise new questions, then synthesise. The forced up-front allocation prevents exactly this adaptivity — once the model has "budgeted" 2 calls to identifiability and finds a red flag, it feels constrained from spending more.

Chosen direction (from user answers): **prompt-level reasoning** (no native reasoning-token wiring), with the planning turn **kept but made lightweight** — a brief "initial questions" turn, not a hard allocation commitment.

## Changes

### 1. Soften the system prompt — [gecco/construct_feedback/tool_judge.py:130-204](gecco/construct_feedback/tool_judge.py#L130-L204)

Rewrite the `_JUDGE_SYSTEM_PROMPT` budgeting section so the judge is told to *adapt*, not *pre-allocate*:

- Replace the "Tool call budgeting: Plan your allocation across the six angles before calling tools" paragraph ([L185-L187](gecco/construct_feedback/tool_judge.py#L185-L187)) with adaptive-investigation language, e.g.:
  - The six angles are a *checklist of coverage*, not a rigid allocation.
  - After each tool result, reflect briefly (1–3 sentences) on what was learned and whether it raises new questions.
  - Follow surprising or contradictory findings deeper, even at the cost of other angles — surprising evidence is higher-signal than a perfectly balanced sweep.
  - The overall call budget is a soft cap; prefer depth on load-bearing findings over breadth for its own sake.
- Keep the six-angle list as coverage guidance but relabel from allocation language (do not say "commit to an allocation").

### 2. Lightweight planning turn — [_OpenAIToolLoop.run()](gecco/construct_feedback/tool_judge.py#L260-L418)

Rewrite the `planning_instruction` block at [L264-L272](gecco/construct_feedback/tool_judge.py#L264-L272) so the forced text-only first turn asks for **initial questions**, not a budget allocation:

> *"Before calling any tools, briefly list the 3–6 specific questions you most want answered this iteration. Keep it short. These are starting points — you are free to adapt as results come in. You have a soft budget of {max_tool_calls} tool calls total. Do not call tools in this message."*

Keep the `tool_choice="none"` mechanism — we still want a deliberate opening beat, just without the prescriptive commitment.

### 3. Interleaved reflection in OpenAI loop — [_OpenAIToolLoop.run()](gecco/construct_feedback/tool_judge.py#L260-L418)

Currently [L327-L337](gecco/construct_feedback/tool_judge.py#L327-L337) inserts a single one-off message after the planning turn that mentions reflection. Replace with a structural mechanism:

- After each tool result is appended (inside the `for tc in msg.tool_calls` loop, after the `messages.append({"role": "tool", ...})` at [L368-L374](gecco/construct_feedback/tool_judge.py#L368-L374)), insert a short user nudge at a configurable cadence (e.g., every call, or every 2nd call — see Open Question) such as:

  > *"Briefly reflect: what did this tell you, and does it change what you want to investigate next? Then make your next tool call (or produce your verdict if you have enough evidence)."*

  Avoid spamming it after every single call if `max_tool_calls` is high (20+) — a reflection nudge every 1–2 calls strikes a balance between enforcement and noise.

- Drop the existing one-off "reflect" transition message after planning, since the per-call nudge covers it.

### 4. Adaptive budget reminders — [_OpenAIToolLoop.run()](gecco/construct_feedback/tool_judge.py#L378-L395) and [_GeminiToolLoop.run()](gecco/construct_feedback/tool_judge.py#L549-L566)

Reword the 50%/80% reminders to match the new framing — not *"ensure you still cover the angles you have not yet investigated"* (which pushes breadth) but something like:

> *"You've used {n}/{max} tool calls. You still have room to investigate further if results so far raise open questions. If you have enough evidence to synthesise a verdict, you can do so now; otherwise keep going."*

### 5. Parity in Gemini loop — [_GeminiToolLoop.run()](gecco/construct_feedback/tool_judge.py#L456-L595)

Bring the Gemini path to the same adaptive shape:

- Add the same lightweight planning turn before the tool loop: send an initial `contents` message that asks for the 3–6 starting questions with `tool_config` constraining to no tool calls (Gemini supports `ToolConfig(function_calling_config=FunctionCallingConfig(mode=NONE))`). Then switch to AUTO for subsequent turns.
- Insert the per-call reflection nudge after each `function_response` append at [L533-L545](gecco/construct_feedback/tool_judge.py#L533-L545), mirroring the OpenAI loop.
- Reword the mid/late budget reminders similarly.

### 6. Verification

- **Unit-style sanity**: import `ToolUsingJudge` and confirm no syntax/import regression.
- **Small-scale judge run**: run `scripts/run_judge_orchestrator.py` against an existing completed iteration's diagnostic store (cheap), with `verbose: true` so the console prints assistant text and tool calls. Confirm:
  - Opening turn produces a short question list rather than an allocation table.
  - Reflection nudges appear between tool calls.
  - On at least one run with a surprising finding, the judge does follow-up calls on that finding instead of rigidly balancing angles.
- **End-to-end**: one short run with [config/two_step_factors_qwen36plus_judge.yaml](config/two_step_factors_qwen36plus_judge.yaml) at reduced `max_iterations` to confirm orchestrator still produces valid `JudgeVerdict` structured output and downstream `LLMFeedbackGenerator` consumption still works.
- **Regression**: existing tests under [tests/test_judge_orchestration.py](tests/test_judge_orchestration.py) should still pass — this change does not touch barrier/coordination code.

## Open question (call out during implementation)

Cadence of the per-tool-call reflection nudge: every call, every 2nd call, or only after results that look "interesting" (hard to detect)? Suggest defaulting to every call for now and observing trace verbosity; easy to dial down to every 2nd if it crowds the context.

## Files to modify

- [gecco/construct_feedback/tool_judge.py](gecco/construct_feedback/tool_judge.py) — all changes land here.

No config, schema, coordination, or orchestrator-script changes needed.
