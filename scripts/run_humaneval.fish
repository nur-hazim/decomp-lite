#!/usr/bin/env fish
# This branch's default run is HumanEval C only, with eight pipelines on GPU 1.
function lite
    docker run --platform linux/amd64 --rm --network host \
        -e DECOMPLITE_METHOD_WORKER_URL=http://127.0.0.1:11900 \
        -v "$LITE_ROOT":/work $LITE_IMAGE python3 -u -m decomplite benchmark \
        --config benchmark_humaneval.yaml --workers $RUN_WORKERS $argv
end
function main
    set -q LITE_ROOT; or set -gx LITE_ROOT (path resolve (path dirname (status filename))/..)
    set -q BENCH_ROOT; or set -gx BENCH_ROOT "$HOME/decompbench-65CB"
    set -q HUMANEVAL_ROOT; or set -gx HUMANEVAL_ROOT "$HOME/decompbench-65CB-humaneval"
    set -q LITE_IMAGE; or set -gx LITE_IMAGE decomp-lite-humaneval
    set -q RUN_WORKERS; or set -gx RUN_WORKERS 4
    set -q CONTAINER_PREFIX; or set -gx CONTAINER_PREFIX hazim
    set -gx VLLM_GPU 1
    set -gx VLLM_TP 1
    set -l vllm "$BENCH_ROOT/docker_setup/models/open_source_vllm_setup/run_vllm_server.sh"
    set -l nova "$BENCH_ROOT/docker_setup/models/nova/run_nova_server.sh"
    set -l embed "$CONTAINER_PREFIX-ollama-embed-gpu1"
    docker image inspect $LITE_IMAGE >/dev/null; or return $status
    test -f "$vllm"; or return 1
    if not test -f "$LITE_ROOT/data/humaneval/suite.json"
        if not set -q HUMANEVAL_CORPUS
            for corpus in "$HUMANEVAL_ROOT/data/repos/c/multipl_e_humaneval" "$BENCH_ROOT/data/repos/c/multipl_e_humaneval"
                if test -f "$corpus/humaneval-workload.json"
                    set -gx HUMANEVAL_CORPUS "$corpus"
                    break
                end
            end
        end
        if set -q HUMANEVAL_CORPUS
            if not test -f "$HUMANEVAL_CORPUS/humaneval-workload.json"
                echo 'HUMANEVAL_CORPUS must contain humaneval-workload.json.' >&2
                return 1
            end
            set -l expected_file "$HUMANEVAL_ROOT/data/multipl_e_preparation/humaneval-c.jsonl"
            if set -q HUMANEVAL_EXPECTED
                set expected_file "$HUMANEVAL_EXPECTED"
                if not test -f "$expected_file"
                    echo 'HUMANEVAL_EXPECTED does not point to a readable prompt file.' >&2
                    return 1
                end
            end
            set -l expected_mount
            set -l expected_args
            if test -f "$expected_file"
                set expected_mount -v "$expected_file":/expected.jsonl:ro
                set expected_args --expected /expected.jsonl
            end
            docker run --rm --platform linux/amd64 -v "$LITE_ROOT":/work -v "$HUMANEVAL_CORPUS":/source:ro \
                $expected_mount $LITE_IMAGE python3 -m decomplite.humaneval prepare --corpus /source $expected_args; or return $status
        else if set -q HUMANEVAL_ACCEPTED
            set -q HUMANEVAL_EXPECTED; or set -gx HUMANEVAL_EXPECTED "$HUMANEVAL_ROOT/data/multipl_e_preparation/humaneval-c.jsonl"
            if not test -f "$HUMANEVAL_ACCEPTED/corpus-manifest.json"; or not test -f "$HUMANEVAL_EXPECTED"
                echo 'The accepted corpus-manifest.json or frozen humaneval-c.jsonl is missing.' >&2
                return 1
            end
            docker run --rm --platform linux/amd64 -v "$LITE_ROOT":/work -v "$HUMANEVAL_ACCEPTED":/accepted:ro \
                -v "$HUMANEVAL_EXPECTED":/expected.jsonl:ro $LITE_IMAGE python3 -m decomplite.humaneval prepare \
                --accepted /accepted --expected /expected.jsonl; or return $status
        else
            echo 'No accepted HumanEval C corpus found. Set HUMANEVAL_CORPUS to the folder containing humaneval-workload.json, or HUMANEVAL_ACCEPTED to a complete accepted export. Prompts/tests alone cannot be decompiled.' >&2
            return 1
        end
    end
    docker run --rm --platform linux/amd64 -v "$LITE_ROOT":/work $LITE_IMAGE \
        python3 -m decomplite.humaneval verify; or return $status
    docker run --rm --platform linux/amd64 -v "$LITE_ROOT":/work $LITE_IMAGE \
        python3 -m decomplite.humaneval references --workers $RUN_WORKERS; or return $status
    lite --phase build; or return $status
    python3 "$LITE_ROOT/scripts/start_humaneval_worker.py" --lite-root "$LITE_ROOT"; or return $status
    nvidia-smi -i 1 >/dev/null; or return $status
    python3 "$LITE_ROOT/scripts/method_gpu.py" vllm --bench-root "$BENCH_ROOT" --gpu 1; or return $status
    if docker container inspect "$CONTAINER_PREFIX-nova-server" >/dev/null 2>&1
        docker stop "$CONTAINER_PREFIX-nova-server" >/dev/null; or return $status
    end
    bash "$vllm" switch llm4decompile_ghidra; or return $status
    lite --phase decompile --filter-pipeline llm4decompile_ghidra; or return $status
    python3 "$LITE_ROOT/scripts/method_gpu.py" embedding --prefix "$CONTAINER_PREFIX" --gpu 1; or return $status
    docker exec $embed ollama pull qwen3-embedding:8b; or return $status
    bash "$vllm" switch gemma4; or return $status
    set -l first (python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["functions"][0]["name"])' "$LITE_ROOT/data/humaneval/selection.json"); or return $status
    lite --phase decompile --filter-pipeline fidelitygpt --filter-fn "$first" --filter-opt O0 --workers 1; or return $status
    lite --phase decompile --filter-pipeline degpt,fidelitygpt,agent4decompile_author_xarch; or return $status
    docker stop $embed >/dev/null; or return $status
    for deployment in sk2decompile llm4decompile_asm sccdec
        set -l pipeline $deployment
        if test "$deployment" = sccdec
            set pipeline sccdec_author_xarch
        end
        bash "$vllm" switch $deployment; or return $status
        lite --phase decompile --filter-pipeline $pipeline; or return $status
    end
    bash "$vllm" down all; or return $status
    env NOVA_GPU=1 NOVA_PORT=11436 CONTAINER_PREFIX=$CONTAINER_PREFIX bash "$nova"; or return $status
    lite --phase decompile --filter-pipeline nova; or return $status
    docker stop "$CONTAINER_PREFIX-nova-server" >/dev/null; or return $status
    lite --phase eval; or return $status
end
main $argv
