#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ApexNav/agent-Apexnav}"
HOST="${LOCAL_VLM_HOST:-127.0.0.1}"
PORT="${LOCAL_VLM_PORT:-18000}"
LOGDIR="${LOCAL_VLM_LOGDIR:-$ROOT/logs/local_qwen_vl}"
PIDFILE="${LOCAL_VLM_PIDFILE:-$LOGDIR/qwen_vl7b_vllm.pid}"

if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE")"
  if [ -n "$pid" ]; then
    pkill -TERM -P "$pid" >/dev/null 2>&1 || true
    kill -TERM "$pid" >/dev/null 2>&1 || true
  fi
fi

ps -eo pid=,cmd= | awk -v port="$PORT" '
  (index($0, "vllm serve") && index($0, "--port " port)) ||
  index($0, "VLLM::EngineCore") { print $1 }
' | sort -u | xargs -r kill -TERM 2>/dev/null || true

sleep 3

if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE")"
  if [ -n "$pid" ]; then
    pkill -KILL -P "$pid" >/dev/null 2>&1 || true
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$PIDFILE"
fi

ps -eo pid=,cmd= | awk -v port="$PORT" '
  (index($0, "vllm serve") && index($0, "--port " port)) ||
  index($0, "VLLM::EngineCore") { print $1 }
' | sort -u | xargs -r kill -KILL 2>/dev/null || true

echo "local_vlm_stopped=http://$HOST:$PORT/v1"
