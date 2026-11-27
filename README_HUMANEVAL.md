# HumanEval C: eight methods in decomp-lite

This branch runs **only HumanEval C**, using the C port already prepared in
`decompbench-65CB-humaneval`: 161 tasks and 1,130 frozen assertions. Three tasks
from the original 164-task Python corpus are absent from that upstream port;
they are not invented here. Five optimization levels and eight methods give
**805 input pairs and 6,440 decompilation cells**. Inputs remain stripped
x86-64 shared objects, with Ghidra **12.1.2**, GPU **1**, and strict-v1 scoring.

Default `benchmark.yaml`, `benchmark_humaneval.yaml`, and the compatibility
`benchmark_methods_500.yaml` all select HumanEval only. The old
`scripts/run_methods_500.fish` forwards to `scripts/run_humaneval.fish`.
No ExeBench selection or PostgreSQL registration is run.

## Required input: accepted C implementations

Prepared prompts/tests alone are not implementations to compile/decompile.
Do not use translator unit-test lookup fixtures as benchmark code.
The local HumanEval checkout inspected during this change contained prompts/tests,
but no completed accepted corpus; its `data.tar.gz` also contained no HumanEval
corpus. Supply the corresponding completed corpus from Ubuntu if it exists.

The launcher auto-detects an installed corpus at either:

- `~/decompbench-65CB-humaneval/data/repos/c/multipl_e_humaneval/`
- `~/decompbench-65CB/data/repos/c/multipl_e_humaneval/`

It must contain `humaneval-workload.json`, all 161 C sources, and `_harness/`.
Alternatively set `HUMANEVAL_CORPUS` to that installed directory:

```fish
set -gx HUMANEVAL_CORPUS /absolute/path/to/multipl_e_humaneval
```

Or supply the earlier framework's complete accepted export:

```fish
set -gx HUMANEVAL_ACCEPTED /absolute/path/to/accepted
```

That directory must contain `corpus-manifest.json`, all accepted `.c` files and
frozen `*_tests.c.txt` files. If using this form, the launcher expects the original
`humaneval-c.jsonl` at `~/decompbench-65CB-humaneval/data/multipl_e_preparation/`;
override with `HUMANEVAL_EXPECTED` if necessary. Its bytes must match the frozen
version bundled here. No new solutions are generated automatically.

Preparation checks all source/test hashes and canonical task IDs and declarations.
It creates the normal Lite dataset/test drivers in `data/humaneval/`. It never
rewrites implementation bodies or test expectations to make them pass.

## Ubuntu: build and run (Fish)

The existing `decomp-lite-exebench-ghidra12` image, `hazim-ubuntu` method container,
`~/decompbench-65CB` backend/model launchers, isolated DeGPT/Fidelity environments,
and downloaded checkpoints from the completed eight-method experiment are reused.
No modifications to that backend repository need to be committed/pulled.

```fish
cd ~/decomp-lite
docker build --platform linux/amd64 -f Dockerfile.humaneval -t decomp-lite-humaneval .
fish scripts/run_humaneval.fish
```

If the previous base image uses another tag:

```fish
docker build --platform linux/amd64 -f Dockerfile.humaneval \
    --build-arg BASE_IMAGE=your-existing-ghidra12-image -t decomp-lite-humaneval .
```

Overrides: `BENCH_ROOT`, `HUMANEVAL_ROOT`, `METHOD_CONTAINER`, `CONTAINER_PREFIX`,
`RUN_WORKERS`, `METHOD_WORKERS`, `FIDELITY_EMBED_BASE`, `FIDELITY_EMBED_MODEL`.
The GPU remains pinned to 1, matching the existing experiment configuration.

The runner, in order:

1. Stages/verifies the complete accepted HumanEval corpus.
2. Compiles and executes all original C test suites at O0/O1/O2/O3/Os. Every
   reference must pass before models start; the 805 checks are checkpointed.
3. Builds all stripped inputs and validates pseudo-C parsing before any model call.
4. Copies the tracked Lite worker/adapters and immutable suite into the existing
   method container. It stops only a worker with a verified recorded PID/command,
   and refuses an unrelated service on port 11900. No global Docker/GPU reset.
5. Sequentially starts the same six model deployments for the eight pipelines,
   warms Fidelity using the first selected HumanEval function, then evaluates.

Per-method model bindings and prompts remain those of the previous experiment.
Agent4Decompile uses the HumanEval C test oracle; SCCDec's self-constructed example
uses the original C types/helpers. Neither looks for ExeBench author records or IO.
The bridge is bundled in this repo; the old backend provides method implementations,
models and their isolated Python dependencies. Worker fingerprints include both
codebases and the frozen corpus, preventing reuse of another corpus's answers.

## Outputs and resume

Outputs use the normal locations, not a second benchmark output layout:

- `reports/summary.csv`: HumanEval-only RC, RCI, RE, calls, tokens and timings.
- `reports/humaneval_reference/references.csv`: 805 original-reference checks.
- `reports/input_eligibility.csv`: 805 build/input checks.
- `results/<pipeline>/<model>/multipl_e_humaneval/x86_64-<opt>/`: answers and metadata.

The per-task frozen C assertion suite runs as one executable: `test_total=1` means
one complete suite, not one assertion. All assertions stay enabled via `-UNDEBUG`.
RC measures standalone C compilation; RCI measures compilation in the original
C type/helper context; RE requires normal exit and all frozen assertions passing.
Generated helper functions are preserved when splicing a recovered target.

Rerun the same launcher after interruption. Passing reference checks, builds,
inputs and generated answers are cached. A recorded generation error is retried;
context/instruction-limit errors may still require model-specific configuration.
No representation is silently truncated to make an oversized task fit.

## Tests

```fish
docker run --rm --platform linux/amd64 -v "$PWD":/work decomp-lite-humaneval \
    python3 -m pytest -q tests
```

Tests synthesize temporary, explicitly labelled integration fixtures only. Native
checks cover all 805 fixture reference suites, cache resume, wrong answers, type/
helper retention, immutable tests, the 6,440-cell plan and all eight worker routes.
A real-corpus/model smoke run still requires the accepted C implementation corpus
and your Ubuntu GPU servers; no remote model run is claimed by local tests.
