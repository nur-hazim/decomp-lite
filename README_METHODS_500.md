# Eight decompbench methods in decomp-lite

This experiment uses the **original decomp-lite 500 functions** (257 test_real,
243 valid_real), x86_64, O0/O1/O2/O3/Os, stripped input: **20,000 cells** across
eight methods. It does not use the corrected 408-function cohort.
All five pseudo-C pipelines use Ghidra **12.1.2** for every function/optimization.

| Method | Input | Generation model |
|---|---|---|
| llm4decompile_ghidra | Ghidra pseudo-C | LLM4Decompile 22B v2 |
| degpt | Ghidra pseudo-C | Gemma4 31B |
| fidelitygpt | Ghidra pseudo-C | Gemma4 31B, Qwen3 embedding 8B |
| agent4decompile_author_xarch | Ghidra pseudo-C | Gemma4 31B |
| sk2decompile | Ghidra pseudo-C | SK2Decompile struct + ident |
| llm4decompile_asm | assembly | LLM4Decompile 6.7B v1.5 |
| sccdec_author_xarch | assembly | LLM4Decompile base + paper LoRA |
| nova | assembly | Nova 6.7B BCR |

The original method implementations run in the existing `hazim-ubuntu`
container, including their multi-stage logic, repair loops, and native oracles.
Decomp-lite owns builds, representations, scheduling, answers, and RC/RCI/RE.
The HTTP worker does not run decompbench orchestration and cannot access
PostgreSQL. Persistent jobs and full method traces live separately under
`decompbench-65CB/data/decomplite_methods_500/`.

Inputs come from decomp-lite's stripped **shared objects**. Label results
“decompbench methods on decomp-lite stripped shared-object inputs”; they are
not the corrected-408 static-executable experiment. Internal author-method
caller bridges retain their original semantics and can expose the original
function name; stripped binary input alone does not make every method name-blind.

## 1. Transfer and unpack

The integration archive has two top-level project folders. Extract into your
home folder, alongside both existing checkouts:

```fish
tar -xzf ~/decomp_lite_eight_methods_500.tar.gz -C ~
cd ~/decomp-lite
```

The archive updates implementation files and adds dedicated experiment configs.
It contains the verified original 500-function worker sources, original test
wrappers, IO, author metadata, and runtime headers. It contains no credentials,
model outputs, or old reports. Your existing `.env`, `models.yaml`, original
500-function config/dataset, and corrected-408 files are not replaced.

The original dataset must already be staged in decomp-lite, as in your earlier
500-function run. The worker payload was independently recovered from the
original archives using the selection manifest's exact member/line and SHA256
hashes. No source implementations, callers, or expected outputs were repaired.

## 2. Rescore existing answers with strict-v1

This command performs evaluation only: **no model server or model calls**.
Use the same original config and model definitions that generated those answers.
The transfer package's original config is `benchmark_exebench_500.yaml`:

```fish
cd ~/decomp-lite
docker run --platform linux/amd64 --rm --network host \
    -v "$PWD":/work decomp-lite-exebench \
    python3 -m decomplite benchmark --config benchmark_exebench_500.yaml \
    --phase eval --evaluation-protocol strict-v1 \
    --report-dir /work/reports/baseline_500_strict_v1 --workers 4
```

Check that the `gen` column still shows 500 for each optimization. A zero or
lower count means the original model/config/result paths do not match; resolve
that before interpreting the scores. Do not use `models_methods.yaml` to rescore
an answer generated under a different historical model key.

`strict-v1` requires all expected JSON keys and no extra keys recursively,
matching array lengths/order, normal exit code 0, finite numeric values, and
distinct booleans versus numbers. Integer-to-integer comparisons are exact.
It retains the historical float relative tolerance **1e-3**; it does not adopt
the separate corrected-408 tolerance. Historical `legacy` scoring remains
available. New CSVs identify `evaluation_protocol`; full per-cell diagnostics
are in the report folder's `details/`. Existing reports remain untouched.

Strict scoring does not validate or repair the original 500-function tests.
Original caller defects or undefined behavior can still cause failures; retain
those rows for review rather than interpreting all failures as model errors.

## 3. Check server prerequisites

Build the Ghidra 12.1.2 image from your existing `decomp-lite-exebench` image.
This retains the real system headers, compilers, Python dependencies, and other
ExeBench dependencies already installed there:

```fish
cd ~/decomp-lite
git pull --ff-only
docker build --platform linux/amd64 \
    --file Dockerfile.ghidra \
    --build-arg BASE_IMAGE=decomp-lite-exebench \
    --tag decomp-lite-exebench-ghidra12 .
```

The installer installs 12.1.2 into a separate directory in the new image and
activates its bundled Jython extension for the extraction script. The original
image remains available. The main `Dockerfile` also uses 12.1.2 for fresh builds;
use `Dockerfile.ghidra` here to inherit your existing system dependencies.

Verify the actual installed version (not just an environment variable):

```fish
docker run --platform linux/amd64 --rm -v "$PWD":/work \
    decomp-lite-exebench-ghidra12 python3 -c \
    'from decomplite.representations import require_ghidra_version; print(require_ghidra_version("12.1.2"))'
```

The existing `hazim-ubuntu` image needs its main `/opt/venv`, isolated
`/opt/degpt-venv` and `/opt/fidelitygpt-venv`, native compilers, and host
networking. Missing dependencies stop preflight. No PostgreSQL is required.

The existing model launcher configuration is used:
`~/decompbench-65CB/docker_setup/models/open_source_vllm_setup/.env`.
Its settings must specify your prefix `hazim`, tensor parallelism `1`,
and an authorized HF token for gated checkpoints. That file is Bash syntax;
do not source it or `vllm.env` in fish. The launcher reads it itself, and the
worker rereads `vllm.env` after each model switch. The decomp-lite runner sets
`VLLM_GPU=1` in that launcher's `.env`, preserving its other settings and token,
because that file overrides exported GPU settings. No decompbench code changes
are required for GPU placement.
Keep the launcher's VRAM checks enabled, including the embedding reservation.

The runner uses `hazim-ollama-embed-gpu1`, explicitly pinned to GPU 1 on port
11435. It preserves and stops the older `hazim-ollama-embed` container to release
that port, reuses its model-storage mounts, pulls `qwen3-embedding:8b` if needed,
and stops the GPU 1 embedder after the Gemma group to free memory. With no older
container it uses a dedicated named model-cache volume. It also stops/replaces
this user's vLLM/Nova deployments.
Do not run another experiment on those same model services concurrently.

## 4. Start the worker and run

```fish
cd ~/decomp-lite
fish scripts/start_method_worker.fish
env LITE_IMAGE=decomp-lite-exebench-ghidra12 fish scripts/run_methods_500.fish
```

Run this in tmux. The run script defaults to `decomp-lite-exebench-ghidra12` and
first builds and checks every required input,
then switches model deployments sequentially on GPU 1: LLM4Decompile Ghidra,
Gemma (three methods), SK2 (both checkpoints), LLM4Decompile ASM, SCCDec, Nova.
It checks the actual advertised model IDs, required input channels, original
source hashes, isolated environments, and embedding endpoint before generation.
Initial model downloads/builds can take substantial time. No GPU throughput
estimate is implied by the local integration tests.

On the first upgraded run, all **2,500 pseudo-C inputs** are regenerated,
including the 2,498 previously extracted with 11.3.1. Existing binaries and
assembly are reused. No `--force`, cache deletion, or model calls are needed for
this refresh. It will repeat the Ghidra extraction work, so allow preparation
time comparable to the previous extraction run. A version/script fingerprint
in each channel manifest prevents reuse of old pseudo-C; completed 12.1.2 inputs
are reused after interruption. The build must report `pseudoc available for
2500/2500` before the runner starts any model server. A wrong installed Ghidra
version stops immediately, before generation. The eligibility report and
generation specs identify version 12.1.2.

The HSLtoRGB O2/O3 failures were reproduced with 11.3.1 (`Overlapping input
varnodes`); both extracted successfully with 12.1.2 on the unchanged local
binaries. This verifies extraction, not semantic correctness of the pseudo-C.
Full Ghidra stdout/stderr and launch metadata are now retained alongside each
channel manifest under `data/channels/<repo>/x86_64/<opt>/<function>/`.

The first Fidelity cell initializes shared retrieval resources at one worker;
the remaining work uses four concurrent cells. To lower resource pressure:

```fish
env LITE_IMAGE=decomp-lite-exebench-ghidra12 RUN_WORKERS=1 METHOD_WORKERS=1 fish scripts/run_methods_500.fish
```

`METHOD_WORKERS` applies when starting a new worker. The startup script reuses a
running worker. After changing worker code or its process settings, gracefully
stop that specific process before restarting (the shutdown waits for active
jobs). Inspect the PID and command first:

```fish
docker exec hazim-ubuntu cat /home/ubuntu/sok_study/data/decomplite_methods_500/worker.pid
docker exec hazim-ubuntu ps -ef
```

Then send `docker exec -u ubuntu hazim-ubuntu kill -TERM PID`, substituting the
verified worker PID. Do not stop the entire container just to restart this worker.

## 5. Resume and retrieve results

Rerun `fish scripts/run_methods_500.fish` after an interruption. Completed
answers are cached; jobs completed before a client disconnect are retrieved
without another generation request. Queued work can be resubmitted after a
worker restart. A job interrupted during an LLM call may require that unfinished
call again; token-level resume is not provided. Ctrl-C cancels the client's
pending waits, while already submitted worker jobs can complete and persist.

The runner does not use `--force`. Changed successful recipes are marked stale
instead of silently overwritten. Preserve or rename those results deliberately
before requesting a new generation. Infrastructure errors stop the phase;
individual method errors retain diagnostics and are retried on a later run.

Final report: `~/decomp-lite/reports/methods_500_strict_v1/summary.csv`.
Input check: `reports/methods_500_strict_v1/input_eligibility.csv`.
Generated C and method traces: the normal decomp-lite `results/` hierarchy.
Worker log: `~/decompbench-65CB/data/decomplite_methods_500/worker.log`.
All methods should have 500 generated cells per opt; investigate any shortfall.

For a single method after starting its appropriate model server:

```fish
docker run --platform linux/amd64 --rm --network host \
    -v "$HOME/decomp-lite":/work decomp-lite-exebench-ghidra12 \
    python3 -m decomplite benchmark --config benchmark_methods_500.yaml \
    --phase decompile --filter-pipeline llm4decompile_asm --workers 4
```

Evaluate all saved method answers later without model servers:

```fish
docker run --platform linux/amd64 --rm --network host \
    -v "$HOME/decomp-lite":/work decomp-lite-exebench-ghidra12 \
    python3 -m decomplite benchmark --config benchmark_methods_500.yaml \
    --phase eval --workers 4
```
