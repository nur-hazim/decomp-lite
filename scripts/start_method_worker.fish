#!/usr/bin/env fish
# Run with fish; the existing Ubuntu container supplies the method environments.
function main
    set -q METHOD_CONTAINER; or set -l METHOD_CONTAINER hazim-ubuntu
    set -q METHOD_WORKERS; or set -l METHOD_WORKERS 4
    set -q FIDELITY_EMBED_BASE; or set -l FIDELITY_EMBED_BASE http://127.0.0.1:11435/v1
    set -q FIDELITY_EMBED_MODEL; or set -l FIDELITY_EMBED_MODEL qwen3-embedding:8b
    docker start $METHOD_CONTAINER >/dev/null; or return $status
    set -l project (docker exec $METHOD_CONTAINER printenv PROJECT_DIR); or return $status
    set -l venv (docker exec $METHOD_CONTAINER printenv VIRTUAL_ENV); or return $status
    if test -z "$project"; or test -z "$venv"
        echo 'The worker container needs PROJECT_DIR and VIRTUAL_ENV.' >&2
        return 1
    end
    if test (docker inspect --format '{{.HostConfig.NetworkMode}}' $METHOD_CONTAINER) != host
        echo 'The method container must use host networking for localhost model endpoints.' >&2
        return 1
    end
    if curl -fsS --max-time 3 http://127.0.0.1:11900/health
        echo
        echo 'Reusing the running method worker. Restart it after updating its code.'
        return 0
    end
    docker exec -d -u ubuntu -e RQ_SELECTION=2 \
        -e FIDELITY_EMBED_BASE=$FIDELITY_EMBED_BASE \
        -e FIDELITY_EMBED_MODEL=$FIDELITY_EMBED_MODEL \
        -w "$project" $METHOD_CONTAINER "$venv/bin/python" \
        "$project/tools/decomplite_methods/serve.py" \
        --suite-root "$project/data/decomplite_methods_500" \
        --workers $METHOD_WORKERS --port 11900; or return $status
    for attempt in (seq 1 180)
        if curl -fsS --max-time 2 http://127.0.0.1:11900/health 2>/dev/null
            echo
            return 0
        end
        sleep 1
    end
    docker exec $METHOD_CONTAINER tail -n 50 "$project/data/decomplite_methods_500/worker.log"
    return 1
end
main $argv
