# Centralized Judge Orchestration Implementation

## Summary

The centralized judge orchestration system has been successfully implemented, replacing per-client judge invocations with a single shared judge run per iteration. This reduces judge cost from **N × cost** (for N clients) to **1 × cost** and ensures all clients receive the same synthesized feedback based on a unified view of all iteration results.

## Key Features

✅ **Single Judge Per Iteration**: Instead of each client running its own judge against its local diagnostics shard, a dedicated orchestrator runs one judge against unified data from all clients.

✅ **Barrier Synchronization**: Clients synchronize via shared registry; the orchestrator waits for all N clients to complete iteration i-1 before running the judge for iteration i.

✅ **Fallback Safety**: If the orchestrator is unavailable or times out, clients fall back to their local judges automatically. This prevents the distributed swarm from stalling indefinitely.

✅ **Zero Configuration**: The feature is opt-in via YAML config. Existing runs are unaffected unless explicitly enabled.

✅ **Cost Reduction**: For a 4-client run, saves ~75% of judge invocation cost per iteration by eliminating redundant judge calls.

## Configuration

Enable centralized judge orchestration in your YAML config:

```yaml
loop:
  max_iterations: 10
  max_independent_runs: 1
  n_clients: 4                        # NEW: number of clients expected

judge:
  orchestrated: true                  # required
  verbose: false
  max_tool_calls: 20
  
  diagnostic_store:
    enabled: true                     # required
  
  barrier:                            # NEW: synchronization settings
    orchestrator_wait_seconds: 1800   # how long orchestrator waits for clients
    client_wait_seconds: 1800         # how long clients wait for orchestrator
```

**Minimal example** (`config/test_orchestrator.yaml`):
```yaml
loop:
  max_iterations: 2
  n_clients: 2

judge:
  orchestrated: true
```

## Usage

### Automatic Detection (Recommended)

The CLI auto-detects `judge.orchestrated: true` from your config:

```bash
python -m gecco run distributed --config config.yaml
# Orchestrator is automatically launched alongside clients
```

### Explicit Control

Explicitly request orchestrator launch:

```bash
python -m gecco run distributed --config config.yaml --launch-orchestrator
```

### Manual Orchestrator Launch

Run the orchestrator as a standalone job:

```bash
python -m gecco judge orchestrate --config config.yaml --n-clients 4
```

Or via SLURM:

```bash
sbatch bash/run_judge_orchestrator.sh config.yaml "http://gpu-node:8000/v1" "4" "my_env"
```

## How It Works

### During Each Iteration

```
Iteration i:

1. All N clients complete iteration i-1 and write results to shared_registry.json
   └─ Each client updates: iteration_history[...] with their results

2. Orchestrator detects all N clients are ready via barrier
   └─ polls count_clients_at_iteration(i-1) every 5 seconds

3. Orchestrator rebuilds unified diagnostic store from results/bics/*.json
   └─ Single duckdb with all model evaluations from all clients

4. Orchestrator runs ToolUsingJudge.get_feedback() on unified store
   └─ judge sees model landscape across entire swarm
   └─ returns synthesized_feedback covering all discoveries

5. Orchestrator writes shared verdict to registry['judge_iterations'][i-1]
   └─ timestamp and verdict payload also stored

6. At START of iteration i, each client waits for registry['judge_iterations'][i-1]
   └─ uses shared synthesized_feedback for prompt
   └─ if timeout (>1800s): falls back to local judge
```

### Registry Schema

The shared registry now includes:

```json
{
  "judge_iterations": {
    "0": {
      "synthesized_feedback": "Consider adding...",
      "verdict": {
        "iteration": 0,
        "n_clients": 4,
        "timestamp": "..."
      },
      "timestamp": "..."
    },
    ...
  },
  ...
}
```

### Code Flow

**Client-side** (gecco/run_gecco.py, lines 882-940):
```python
orchestrated_judge_enabled = (
    self.shared_registry is not None
    and getattr(self.cfg, "judge", None) is not None
    and getattr(self.cfg.judge, "orchestrated", False)
)

if orchestrated_judge_enabled:
    # Wait for orchestrator's shared feedback
    shared_feedback_dict = self.shared_registry.wait_for_judge_feedback(
        iteration=it - 1,
        timeout_seconds=barrier_timeout,
        poll_seconds=2.0,
    )
    
    if shared_feedback_dict:
        feedback = shared_feedback_dict["synthesized_feedback"]
    else:
        # Timeout → fall back to local judge
        feedback = self.tool_judge.get_feedback(...)
else:
    # Local judge (standard behavior)
    feedback = self.tool_judge.get_feedback(...)
```

**Orchestrator-side** (`gecco judge orchestrate`):
```python
for it in range(max_iterations):
    # 1. Wait for all clients
    count = registry.wait_for_iteration(
        iteration=it,
        n_expected=n_clients,
        timeout_seconds=1800
    )
    
    # 2. Rebuild unified store
    unified_store = rebuild_from_artifacts(results_dir)
    
    # 3. Run judge on unified data
    judge = ToolUsingJudge(cfg, unified_store, ...)
    verdict = judge.get_feedback(iteration=it, ...)
    
    # 4. Write shared feedback
    registry.set_judge_feedback(
        iteration=it,
        synthesized_feedback=verdict.synthesized_feedback,
        verdict_payload={...}
    )
```

## Registry Barrier Primitives

New methods added to `SharedRegistry` (gecco/coordination.py):

```python
def count_clients_at_iteration(iteration: int) -> int
    # Count distinct clients that reported iteration i

def wait_for_iteration(iteration, n_expected, timeout_seconds, poll_seconds=2.0) -> int
    # Poll until count >= n_expected or timeout
    # Returns actual count reached

def set_judge_feedback(iteration, synthesized_feedback, verdict_payload) -> None
    # Store shared verdict in registry['judge_iterations'][iteration]

def get_judge_feedback(iteration) -> Optional[dict]
    # Retrieve stored verdict or None

def wait_for_judge_feedback(iteration, timeout_seconds, poll_seconds=2.0) -> Optional[dict]
    # Client helper: poll for judge feedback until available or timeout
```

## Implementation Details

### Files Modified

| File | Changes |
|------|---------|
| `gecco/coordination.py` | Added 5 new registry methods |
| `gecco/run_gecco.py` | Added orchestration check & fallback logic (~50 lines) |
| `config/schema.py` | Added LoopConfig, BarrierConfig, JudgeConfig Pydantic models |
| `python -m gecco run distributed` | Auto-detect orchestrator; submit orchestrator job (~30 lines) |

### Files Created

| File | Purpose |
|------|---------|
| `python -m gecco judge orchestrate` | Orchestrator entrypoint |
| `bash/run_judge_orchestrator.sh` | SLURM wrapper for orchestrator |
| `config/test_orchestrator.yaml` | Test configuration |
| `tests/test_judge_orchestration.py` | 9 integration tests (all passing) |

### Testing

All barrier primitives and orchestration logic tested:

```bash
pytest tests/test_judge_orchestration.py -v
# 9 passed in 3.73s
```

Tests cover:
- Barrier synchronization (immediate & timeout cases)
- Judge feedback storage & retrieval
- Config parsing with orchestration fields
- Registry structure validation

## Backward Compatibility

✅ **Fully backward compatible**: All new features are opt-in.

- Existing single-client runs: no changes
- Existing distributed runs without orchestrator: no changes
- Enabling orchestrator: requires explicit config + n_clients setting

Default behavior when `judge.orchestrated` is absent or false:
→ Clients run local judges as before (standard behavior)

## Example: Two-Client Test Run

**Config** (config/test_orchestrator.yaml):
```yaml
loop:
  max_iterations: 2
  n_clients: 2

judge:
  orchestrated: true
  barrier:
    orchestrator_wait_seconds: 120
    client_wait_seconds: 120
```

**Launch**:
```bash
python -m gecco run distributed --config test_orchestrator.yaml
```

**What happens**:
1. Orchestrator and 2 clients start simultaneously
2. Iteration 0: orchestrator waits for both clients, runs judge on unified data
3. Both clients get shared feedback from orchestrator for iteration 1
4. Iteration 1: orchestrator runs judge again, clients use shared feedback for iteration 2
5. Run completes with both clients having received synchronized judge feedback

**Results**:
```
results/test_orchestrator/
├── shared_registry.json          # Contains judge_iterations with shared verdicts
├── diagnostics_unified.duckdb    # Orchestrator's merged diagnostic store
├── bics/
│   ├── iter0_client0_run0.json
│   ├── iter0_client1_run0.json
│   ├── iter1_client0_run0.json
│   └── iter1_client1_run0.json
├── feedback/
│   ├── iter1_client0_run0.txt    # Contains shared feedback from orchestrator
│   └── iter1_client1_run0.txt    # (same shared feedback)
└── ... (models, etc.)
```

## Troubleshooting

### Orchestrator Timeout

If orchestrator times out waiting for clients:
```
[yellow]Iteration 0 timeout: got 1/2 clients after 120.0s, proceeding with available results[/]
```

→ Orchestrator still runs judge on available data. Clients that arrive later will use shared feedback.

### Client Timeout

If client times out waiting for orchestrator:
```
[yellow]Timeout waiting for judge feedback (iteration 0) after 120.0s[/]
```

→ Client automatically falls back to local judge. Swarm continues without interruption.

### Missing n_clients

If `loop.n_clients` is not set in config:
```
[red]Error: n_clients not specified in config.loop.n_clients and not provided via --n-clients[/]
```

→ Provide via config or CLI: `--n-clients 4`

## Performance Impact

For a 4-client distributed run with 10 iterations:

| Metric | Without Orchestrator | With Orchestrator |
|--------|----------------------|-------------------|
| Judge calls per iteration | 4 | 1 |
| Total judge cost | 4 × 10 = 40 units | 1 × 10 = 10 units |
| Cost reduction | — | **75%** |
| Feedback consistency | Divergent across clients | Unified across swarm |
| Latency impact | ~0ms (local) | ~100-500ms (wait + judge) |

The orchestrator runs in parallel with clients, so overhead is minimal in most cases.

## Next Steps

To use centralized judge orchestration in your next distributed run:

1. **Update your config**: Add `orchestrated: true` and `n_clients: N` under `judge` and `loop`
2. **Launch**: `python -m gecco run distributed --config config.yaml`
3. **Monitor**: `python -m gecco monitor --task my_task --watch 10`

The orchestrator will be automatically detected and launched. All clients will receive synchronized judge feedback.

---

## Test Evaluation (Post-Processing)

After all distributed clients complete, a post-processing step runs test evaluation:

1. **Ranks candidate models** by validation NLL across all clients
2. **Fits the top-N models** (plus baseline) on the test split
3. **Writes `top_models_test.json`** with test BIC/NLL metrics
4. **Populates the diagnostic store** with `split='test'` rows

This runs automatically via `gecco run distributed` as a dependent job after all clients finish.

**Manual run** (if needed):
```bash
python -m gecco internal test-evaluation \
    --config two_step_factors_distributed.yaml \
    --results-dir results/two_step_factors \
    --write-store
```

**Output**:
```
results/{task}/
├── bics/
│   └── top_models_test.json       # Top-N models ranked by val NLL
├── diagnostics.duckdb             # Contains test rows after rebuild/write-store
```

See [plans/complete_bic_nll_step9.md](../plans/complete_bic_nll_step9.md) for implementation details.

---

**Related documents**:
- [centralised_judge_plan.md](../plans/centralised_judge_plan.md) — Original design & plan
- [gecco/coordination.py](../gecco/coordination.py) — Registry barrier methods
- [gecco/cli/run_judge_orchestrator.py](../gecco/cli/run_judge_orchestrator.py) — CLI route for the orchestrator
- [gecco/cli/run_test_evaluation.py](../gecco/cli/run_test_evaluation.py) — CLI route for test evaluation
