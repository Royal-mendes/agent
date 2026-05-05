#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ApexNav/agent-Apexnav}"
ENV_PREFIX="${QWEN_VL_ENV:-/root/autodl-tmp/envs/qwen_vl}"
HOST="${LOCAL_VLM_HOST:-127.0.0.1}"
PORT="${LOCAL_VLM_PORT:-18000}"
MODEL="${LOCAL_VLM_HF_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SERVED_MODEL="${LOCAL_VLM_MODEL:-qwen2.5-vl-7b-instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${LOCAL_VLM_GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${LOCAL_VLM_MAX_MODEL_LEN:-8192}"
LIMIT_MM_PER_PROMPT="${LOCAL_VLM_LIMIT_MM_PER_PROMPT:-{\"image\":2,\"video\":0}}"
LOGDIR="${LOCAL_VLM_LOGDIR:-$ROOT/logs/local_qwen_vl}"
PIDFILE="${LOCAL_VLM_PIDFILE:-$LOGDIR/qwen_vl7b_vllm.pid}"
LOGFILE="${LOCAL_VLM_LOGFILE:-$LOGDIR/qwen_vl7b_vllm.log}"

mkdir -p "$LOGDIR"

port_open() {
  /root/miniconda3/envs/apexnav/bin/python - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

wait_for_server() {
  /root/miniconda3/envs/apexnav/bin/python - "$HOST" "$PORT" "${LOCAL_VLM_START_TIMEOUT:-2400}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

host, port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [item.get("id") for item in data.get("data", [])]
        print("local_vlm_ready models=" + ",".join(str(x) for x in ids))
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(3)
print(f"local_vlm_not_ready error={last_error}", file=sys.stderr)
sys.exit(1)
PY
}

if [ "${LOCAL_VLM_FOREGROUND:-0}" = "1" ]; then
  if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "qwen_vl_env_missing=$ENV_PREFIX" >&2
    echo "run: $ROOT/repro/scripts/setup_qwen_vl7b_vllm_env.sh" >&2
    exit 2
  fi
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_PREFIX"
  export CUDA_VISIBLE_DEVICES
  export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
fi

if port_open; then
  echo "local_vlm_already_running=http://$HOST:$PORT/v1"
  wait_for_server
  exit 0
fi

LOCAL_VLM_FOREGROUND=1 nohup "$0" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "local_vlm_pid=$(cat "$PIDFILE")"
echo "local_vlm_log=$LOGFILE"
wait_for_server
