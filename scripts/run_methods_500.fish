#!/usr/bin/env fish
# Sequential model deployments on GPU 1; only decomp-lite runs the benchmark.
function lite
    docker run --platform linux/amd64 --rm --network host \
        -e DECOMPLITE_METHOD_WORKER_URL=http://127.0.0.1:11900 \
        -v "$LITE_ROOT":/work $LITE_IMAGE python3 -m decomplite benchmark \
        --config benchmark_methods_500.yaml --workers $RUN_WORKERS $argv
end

function main
    set -q LITE_ROOT; or set -gx LITE_ROOT (path resolve (path dirname (status filename))/..)
    set -q BENCH_ROOT; or set -gx BENCH_ROOT "$HOME/decompbench-65CB"
    set -q LITE_IMAGE; or set -gx LITE_IMAGE decomp-lite-exebench
    set -q RUN_WORKERS; or set -gx RUN_WORKERS 4
    set -q CONTAINER_PREFIX; or set -gx CONTAINER_PREFIX hazim
    set -gx VLLM_GPU 1
    set -gx VLLM_TP 1
    set -l vllm "$BENCH_ROOT/docker_setup/models/open_source_vllm_setup/run_vllm_server.sh"
    set -l nova "$BENCH_ROOT/docker_setup/models/nova/run_nova_server.sh"
    set -l embed "$CONTAINER_PREFIX-ollama-embed-gpu1"
    docker image inspect $LITE_IMAGE >/dev/null; or return $status
    test -f "$vllm"; or return 1
    python3 "$LITE_ROOT/scripts/method_gpu.py" vllm --bench-root "$BENCH_ROOT" --gpu 1; or return $status
    fish "$LITE_ROOT/scripts/start_method_worker.fish"; or return $status

    # Check every required stripped input before any model is loaded or called.
    lite --phase build; or return $status
    # Nova is outside the vLLM launcher's deployment table.
    if docker container inspect "$CONTAINER_PREFIX-nova-server" >/dev/null 2>&1
        docker stop "$CONTAINER_PREFIX-nova-server" >/dev/null; or return $status
    end
    bash "$vllm" switch llm4decompile_ghidra; or return $status
    lite --phase decompile --filter-pipeline llm4decompile_ghidra; or return $status

    python3 "$LITE_ROOT/scripts/method_gpu.py" embedding --prefix "$CONTAINER_PREFIX" --gpu 1; or return $status
    docker exec $embed ollama pull qwen3-embedding:8b; or return $status
    bash "$vllm" switch gemma4; or return $status
    # Populate Fidelity's shared retrieval resources once before parallel work.
    lite --phase decompile --filter-pipeline fidelitygpt --filter-fn R90deg --filter-opt O0 --workers 1; or return $status
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
