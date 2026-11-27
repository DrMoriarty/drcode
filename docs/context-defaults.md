# Context-economy defaults: decision table

Every context-optimization mechanism in drcode carries a default (ON or
OFF). This document fixes the criterion those defaults were audited
against and the resulting decision table, so any FUTURE change of a
default is checked against the same rule instead of re-litigating the
topic from scratch.

## The criterion

> A mechanism may be default-ON only if it is GUARANTEED never to worsen
> context economy in any case — including edge and degenerate ones
> (empty inputs, closed checklists, short transcripts, already-clean
> state). If there exists a scenario where it works to the model's
> detriment or pure waste (constant per-request spend with no possible
> return, cache-warmth trades for nothing, stubbing results the agent
> still needs), it must be default-OFF: an opt-in the user/project
> enables knowingly.

Two corollaries the audit produced:

- "Waste" includes steady-state spend, not only one-shot costs. A block
  injected into EVERY request must have a degenerate-case exit (a stub,
  None) for the state where it can no longer buy anything — see the
  todo tail injection below, the one violation the audit found.
- "Guaranteed never worse" is a high bar: a heuristic that only
  *sometimes* helps and *sometimes* hurts fails it, even when the help
  is larger on average. Defaults may be conservative.

## Decision table (audited)

| Mechanism | Default | Verdict | Why |
| --- | --- | --- | --- |
| Compaction at `compaction_threshold` (`src/drcode/config.py`) | ON | confirmed ON | Fires only above the threshold, i.e. only when the transcript is measurably oversized; there is no state in which rewriting it costs more than re-prefilling it. |
| `mark_unneeded` eviction (marks layer + forced pre-compaction pass, `src/drcode/core/pipeline_compaction.py`) | ON | confirmed ON | Marks are agent-authored (the agent saw the content and decided); the batched collapse costs one prompt-cache invalidation and frees real bytes; a mark on content that was still useful is recoverable by re-reading. The forced pre-compaction pass is hysteresis-gated (`_EVICTION_SPARE_MARGIN_PCT`, 20 points): it may spare the summarizer only when the credit-corrected post-eviction fill dropped below threshold − 20 points of the window. Without the floor, a small freeing (a few percent) dips under the threshold, re-crosses it within turns and repeats — each pass a forced eviction that invalidates the prompt cache (the next request re-reads most of the window at uncached-input price) to spare a few percent of carried tokens, a net loss against compacting right away on the lightened history. The between-turns batch gate needs no hysteresis: it fires only when the batch was due anyway, so its invalidation is not an extra cost. |
| `read_file` read cache (`src/drcode/core/pipeline_agent_run.py`, `src/drcode/read_cache.py`) | ON | confirmed ON | The stub is issued only while the previous read's result is still a live transcript part (fail-open), keyed on content hash; every path either saves the full re-read or degrades to it. Cannot cost information. |
| Model-switch housekeeping (thinking strip + takeover note, `src/drcode/core/pipeline_compaction.py`) | ON | confirmed ON | Runs only on an actual `last_model_key` mismatch; foreign reasoning in the new model's context is provably dead weight (pydantic-ai would fold it into assistant content). |
| Sliding-window thinking strip (`thinking_keep_recent_turns`, `src/drcode/settings.py`) | OFF (0) | changed ON (5) → OFF (user-directed) | The N-exchange freshness window is a heuristic and stripped old chain-of-thought is reasoning the model cannot re-derive — the strip can occasionally work against the run, so it fails the criterion and stays opt-in per project. Same change moved the standalone strip gate from 1/6 to 2/3 of the window: a strip costs a prompt-cache invalidation per batch, so it must earn it with a large stale volume. |
| Todo checklist tail injection (`src/drcode/agent_tools.todo_instructions`) | ON | **was violated → fixed** | The full block was injected into every request while `task.todos` was non-empty — including the degenerate case of a fully closed, unchanged checklist (final-answer tail of a run): a constant ~50–150-token per-request spend with no possible return. Fix: the full block goes out only while unchecked items exist or the list changed since the last full injection (`task.todos_injected_fingerprint`); a closed unchanged list gets a one-line "checklist complete — do not reopen items" stub. |
| Automatic tool-result unloading (`auto_context_unloading`, `src/drcode/settings.py`) | OFF | confirmed OFF | The age heuristic can stub results the agent was about to re-read verbatim (re-reads cost more than the stubbing saved) and does nothing on short transcripts — it sometimes works and sometimes works against the run, so it fails the criterion; opt-in per project. |
| `delegate_research` slots (`subagents_coder` / `subagents_researcher`, `src/drcode/settings.py`) | OFF | confirmed OFF | drcode-bench: 119 tasks with the slots enabled produced ZERO delegate_research calls — enabling adds tool surface and reminder tail blocks for nothing; pure harm in observed practice. |
| Resume compaction (`compaction_resume_min_tokens`, `src/drcode/settings.py`) | OFF (0) | confirmed OFF | A low threshold converts warm cache into summarizer calls on transcripts the ordinary threshold gate would never touch — waste on healthy sessions. |

## Changing a default

Any PR that flips one of these defaults (or adds a new context-touching
mechanism with a default) must:

1. Walk the criterion over the mechanism's degenerate cases explicitly
   (empty/unchanged/oversized inputs, final-answer tails, restarts,
   compaction interplay) and either show "never worse" or default-OFF.
2. Update this table with the verdict and the reason.
3. For default-ON mechanisms with per-request costs, demonstrate a
   degenerate-case exit (a None/stub path) exists and is tested.
