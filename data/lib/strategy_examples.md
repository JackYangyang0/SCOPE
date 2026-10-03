# Strategy Implementation References

Both `strategy_library.json` and `cpu_strategy_library.json` entries use
`implementation_example_ids`. The initial shared references live in
`strategy_examples.json`; per-strategy references live in
`strategy_examples_full.json`. Together they cover all 136 GPU and 53 CPU
strategies. `strategy_bindings` also supports strategies rebuilt from the index
without optional reference metadata. Duplicate or missing example definitions
are errors, not silently skipped.

`llm/strategy_examples.py` expands references only when preparing patch/code
messages. Repair strategies contain the selected strategies and use the same
resolver. Strategy selection and batch planning do not load example bodies.
CPU code prompts expand examples AFTER strategy compaction so snippets survive.
Deterministic Tiling references are documentation/configuration illustrations;
the executor still changes parameters directly without calling the LLM.

Examples are conditional references, not instructions to select more strategies.
They contain no historical performance rankings or fixed target tile sizes.
Current strategy contracts and current IR/source take precedence. Sketch helpers
must be materialized, not pasted as undefined calls. Complete kernels still need
normal compilation, numerical, safety and strategy-realization verification.

Coverage includes tiling, mapping, layout, packing, vectorization, registers,
loop scheduling, pipelines, reduction, epilogues, safety, compiler settings,
resource tuning and CPU threading. Code strategies have implementation fragments;
diagnostic/tuning strategies have measurement recipes; deterministic parameters
and compiler strategies have configuration examples. These are not all standalone
compilable programs. Placeholder helper calls and pseudocode are explicitly typed.
The cp.async example is documentation-based and has not been GPU-regression
validated as a standalone kernel. None of these fragments is a drop-in complete
kernel or evidence that the performance optimization has been realized.

Run prompt-delivery and catalog tests from the repository parent:

```text
python -m unittest SCOPE.tests.test_strategy_examples
python -m SCOPE.data.lib.build_strategy_examples --check
```

For new strategy families, add a curated case to `build_strategy_examples.py`
with applicability, adaptation rules, code/recipe and verification obligations.
Then run `python -m SCOPE.data.lib.build_strategy_examples` to refresh only example
metadata and generated reference data. The builder fails for unhandled strategies
rather than attaching an unrelated generic example. Do not bind a scalar scatter
example to a vectorized shared-store strategy or a two-stage sketch to a multistage
one. No strategy pre/postconditions, dependency edges or enabled profiles are
changed by this builder.
