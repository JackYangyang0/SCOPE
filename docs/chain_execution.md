# Chain execution

Configure concurrency in `conf.yaml`:

```yaml
execution:
  chain_workers: 3
  llm_max_concurrency: 3
  compile_workers: 2
  gpu_verification_workers: 1
```

Run `app.py` with the existing Windows or Linux arguments. No new command is
required. Set `chain_workers: 1` to execute chains sequentially.

Phase 1 dispatches independent active states concurrently within each search
round. A round finishes before its results update the frontier. Strategies within
one chain remain sequential. Unlock dispatches independent Phase 1 candidates
concurrently; batches within each candidate remain sequential.

Each task owns its IR/history snapshot and LLM client. LLM requests and compilation
have separate concurrency limits. GPU execution, including warmups, measured runs
and asynchronous safety checks, uses an exclusive slot. The GPU worker count must
remain one. CPU benchmark comparisons also use this slot to avoid overlapping
timed runs.

Candidate files retain their chain-specific names. Stage check/filter/selection
files also include the chain namespace. Shared latest-result files are published
by the controller after tasks complete, in input-path order. JSON writes use
atomic replacement. A chain exception is recorded without cancelling siblings;
the parent state is retained for terminal verification.

Terminal verification currently traverses its candidates sequentially. Unlock
and stage verification can compile different chains concurrently but serialize
their timed execution. Concurrency does not increase the search budget or change
the number of benchmark runs. It reduces waiting time when LLM generation and
compilation dominate; real speedup depends on request limits and hardware load.
