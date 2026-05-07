#!/usr/bin/env bash
set -o pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATASET="${DATASET:-hm3dv2}"
DATASET_SPLIT="${DATASET_SPLIT:-}"
EPISODE="${EPISODE:-2}"
EP_TIMEOUT="${EP_TIMEOUT:-900}"
VLM_START_TIMEOUT="${VLM_START_TIMEOUT:-180}"
VLM_BASE_PORT="${VLM_BASE_PORT:-16181}"
RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_REFLECTIVE="${RUN_REFLECTIVE:-true}"
REFLECTIVE_VLM_PROVIDER="${REFLECTIVE_VLM_PROVIDER:-openai}"
REFLECTIVE_VLM_MODEL="${REFLECTIVE_VLM_MODEL:-gpt-5.5}"
REFLECTIVE_VLM_BASE_URL="${REFLECTIVE_VLM_BASE_URL:-}"
REFLECTIVE_VLM_API_KEY="${REFLECTIVE_VLM_API_KEY:-}"
REFLECTIVE_ENABLE_LEARNING_FROM_TRACES="${REFLECTIVE_ENABLE_LEARNING_FROM_TRACES:-true}"
REFLECTIVE_ENABLE_BASELINE_TEACHER_LEARNING="${REFLECTIVE_ENABLE_BASELINE_TEACHER_LEARNING:-false}"
REFLECTIVE_BASELINE_TEACHER_LOG_PATH="${REFLECTIVE_BASELINE_TEACHER_LOG_PATH:-}"
REFLECTIVE_ENABLE_GT_TEACHER_LEARNING="${REFLECTIVE_ENABLE_GT_TEACHER_LEARNING:-false}"
REFLECTIVE_GT_TRAJECTORY_PATH="${REFLECTIVE_GT_TRAJECTORY_PATH:-}"
REFLECTIVE_LEARNING_WRITE_MODE="${REFLECTIVE_LEARNING_WRITE_MODE:-train_only}"
REFLECTIVE_TOOL_CALL_DATASET_PATH="${REFLECTIVE_TOOL_CALL_DATASET_PATH:-data/tool_call_learning_samples.jsonl}"
REFLECTIVE_MEMORY_PATH="${REFLECTIVE_MEMORY_PATH:-data/reflection_memory.jsonl}"
REFLECTIVE_POLICY_PATCH_PATH="${REFLECTIVE_POLICY_PATCH_PATH:-data/policy_patches.json}"
RUN_ID="${RUN_ID:-reflective_small_val_$(date +%Y%m%d_%H%M%S)}"
BASE="$ROOT/repro/reflective_small_eval/$RUN_ID"
LOGDIR="$BASE/logs"
RESULTS="$BASE/results.csv"
VLM_PIDS="$BASE/vlm_pids.txt"
CONFIG="$ROOT/config/habitat_eval_hm3dv2.yaml"
CONFIG_BAK="$BASE/original_habitat_eval_hm3dv2.yaml"

mkdir -p "$LOGDIR"
cp "$CONFIG" "$CONFIG_BAK"
echo "mode,episode,target,result,success_pct,spl_pct,soft_spl_pct,distance_to_goal,steps,rc,duration_sec,log" > "$RESULTS"

port_open() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

find_free_vlm_base() {
  /root/miniconda3/envs/apexnav/bin/python - "$VLM_BASE_PORT" <<'PY'
import socket
import sys

base = int(sys.argv[1])

def used(port):
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()

while True:
    ports = [base + i for i in range(4)]
    if not any(used(p) for p in ports):
        print(base)
        break
    base += 10
PY
}

wait_for_port() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" "${2:-180}" <<'PY' >/dev/null 2>&1
import socket, sys, time
port = int(sys.argv[1])
deadline = time.time() + int(sys.argv[2])
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        sys.exit(0)
    except OSError:
        time.sleep(1)
    finally:
        s.close()
sys.exit(1)
PY
}

ensure_roscore() {
  if ! port_open 11311; then
    nohup bash -lc "source /opt/ros/noetic/setup.bash; roscore -p 11311" > "$LOGDIR/roscore.log" 2>&1 &
    sleep 6
  fi
}

cleanup_vlm() {
  if [ -f "$VLM_PIDS" ]; then
    while read -r pid; do
      [ -n "$pid" ] && kill -TERM "$pid" >/dev/null 2>&1 || true
    done < "$VLM_PIDS"
    sleep 2
    while read -r pid; do
      [ -n "$pid" ] && kill -KILL "$pid" >/dev/null 2>&1 || true
    done < "$VLM_PIDS"
  fi
  if [ -n "${GDINO_PORT:-}" ]; then
    ps -eo pid=,cmd= | awk -v ports="$GDINO_PORT $BLIP2ITM_PORT $MOBILE_SAM_PORT $YOLOV7_PORT" '
      BEGIN {
        split(ports, p, " ");
        for (i in p) want["--port " p[i]] = 1;
      }
      {
        for (key in want) {
          if (index($0, key)) {
            print $1;
            break;
          }
        }
      }' | sort -u | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    ps -eo pid=,cmd= | awk -v ports="$GDINO_PORT $BLIP2ITM_PORT $MOBILE_SAM_PORT $YOLOV7_PORT" '
      BEGIN {
        split(ports, p, " ");
        for (i in p) want["--port " p[i]] = 1;
      }
      {
        for (key in want) {
          if (index($0, key)) {
            print $1;
            break;
          }
        }
      }' | sort -u | xargs -r kill -KILL 2>/dev/null || true
  fi
}

kill_exploration() {
  source /opt/ros/noetic/setup.bash 2>/dev/null || true
  export ROS_MASTER_URI=http://localhost:11311
  rosnode kill /exploration_node /tsp_solver >/dev/null 2>&1 || true
  ps -eo pid=,cmd= | awk '/roslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/exploration_node/ || /devel\/lib\/lkh_mtsp_solver\/tsp_node/ {print $1}' | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
  ps -eo pid=,cmd= | awk '/roslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/exploration_node/ || /devel\/lib\/lkh_mtsp_solver\/tsp_node/ {print $1}' | xargs -r kill -KILL 2>/dev/null || true
}

restore_config() {
  if [ -f "$CONFIG_BAK" ]; then
    cp "$CONFIG_BAK" "$CONFIG"
  fi
}

cleanup() {
  kill_exploration
  cleanup_vlm
  restore_config
}
trap cleanup EXIT INT TERM

start_vlm() {
  local base
  base="$(find_free_vlm_base)"
  GDINO_PORT="$base"
  BLIP2ITM_PORT="$((base + 1))"
  MOBILE_SAM_PORT="$((base + 2))"
  YOLOV7_PORT="$((base + 3))"
  export GDINO_PORT BLIP2ITM_PORT MOBILE_SAM_PORT YOLOV7_PORT
  : > "$VLM_PIDS"
  echo "new_vlm_ports=gdino:$GDINO_PORT blip2itm:$BLIP2ITM_PORT sam:$MOBILE_SAM_PORT yolo:$YOLOV7_PORT" | tee "$BASE/runner.log"
  for spec in \
    "gdino vlm.detector.grounding_dino $GDINO_PORT" \
    "blip2 vlm.itm.blip2itm $BLIP2ITM_PORT" \
    "sam vlm.segmentor.sam $MOBILE_SAM_PORT" \
    "yolo vlm.detector.yolov7 $YOLOV7_PORT"; do
    set -- $spec
    local name="$1"
    local mod="$2"
    local port="$3"
    nohup bash -lc "cd '$ROOT'; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; python -m ${mod} --port ${port}" > "$LOGDIR/vlm_${name}_${port}.log" 2>&1 &
    echo $! >> "$VLM_PIDS"
    if ! wait_for_port "$port" "$VLM_START_TIMEOUT"; then
      echo "failed_to_start_vlm=$name port=$port" | tee -a "$BASE/runner.log"
      return 1
    fi
  done
}

set_reflective_config() {
  local enabled="$1"
  local mode="$2"
  local enable_baseline_teacher="$REFLECTIVE_ENABLE_BASELINE_TEACHER_LEARNING"
  local baseline_teacher_log="$REFLECTIVE_BASELINE_TEACHER_LOG_PATH"
  local enable_gt_teacher="$REFLECTIVE_ENABLE_GT_TEACHER_LEARNING"
  local gt_trajectory_path="$REFLECTIVE_GT_TRAJECTORY_PATH"
  if [ "$enabled" = "true" ] && [ "$mode" = "reflective" ] && [ "$enable_baseline_teacher" = "true" ] && [ -z "$baseline_teacher_log" ]; then
    baseline_teacher_log="$LOGDIR/baseline_ep_${EPISODE}.log"
  fi
  if [ "$enabled" != "true" ] || [ "$mode" != "reflective" ] || [ "$enable_baseline_teacher" != "true" ] || [ ! -f "$baseline_teacher_log" ]; then
    enable_baseline_teacher="false"
    baseline_teacher_log=""
  fi
  if [ "$enabled" != "true" ] || [ "$mode" != "reflective" ] || [ "$enable_gt_teacher" != "true" ] || [ ! -f "$gt_trajectory_path" ]; then
    enable_gt_teacher="false"
    gt_trajectory_path=""
  fi
  python - "$ROOT" "$CONFIG" "$enabled" "$RUN_ID" "$mode" "$REFLECTIVE_VLM_PROVIDER" "$REFLECTIVE_VLM_MODEL" "$REFLECTIVE_VLM_BASE_URL" "$REFLECTIVE_VLM_API_KEY" "$REFLECTIVE_ENABLE_LEARNING_FROM_TRACES" "$REFLECTIVE_LEARNING_WRITE_MODE" "$REFLECTIVE_TOOL_CALL_DATASET_PATH" "$REFLECTIVE_MEMORY_PATH" "$REFLECTIVE_POLICY_PATCH_PATH" "$enable_baseline_teacher" "$baseline_teacher_log" "$enable_gt_teacher" "$gt_trajectory_path" <<'PY'
import sys
import yaml

(
    root,
    path,
    enabled,
    run_id,
    mode,
    provider,
    model,
    base_url,
    api_key,
    enable_learning,
    learning_write_mode,
    tool_call_dataset_path,
    memory_path,
    policy_patch_path,
    enable_baseline_teacher,
    baseline_teacher_log_path,
    enable_gt_teacher,
    gt_trajectory_path,
) = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
ra = data.setdefault("reflective_agent", {})
ra["enable_reflective_agent"] = enabled.lower() == "true"
ra["run_id"] = f"{run_id}_{mode}"
ra["project_root"] = str(root)
ra["python_executable"] = sys.executable
ra["memory_write_mode"] = learning_write_mode
ra["memory_path"] = memory_path
ra["policy_patch_path"] = policy_patch_path
ra["episode_log_root"] = "logs/reflective_agent"
ra["tool_call_dataset_path"] = tool_call_dataset_path
ra["learning_write_mode"] = learning_write_mode
ra["enable_baseline_teacher_learning"] = False
ra["baseline_teacher_log_path"] = ""
ra["enable_gt_teacher_learning"] = False
ra["gt_trajectory_path"] = ""
if not ra["enable_reflective_agent"]:
    ra["vlm_provider"] = "mock"
    ra["enable_episode_logger"] = False
    ra["enable_episode_reflection"] = False
    ra["enable_reflection_memory"] = False
    ra["enable_learning_from_traces"] = False
else:
    ra["vlm_provider"] = provider
    ra["vlm_model"] = model
    ra["vlm_base_url"] = base_url
    if api_key:
        ra["vlm_api_key"] = api_key
    elif provider == "local":
        ra["vlm_api_key"] = "local"
    ra["enable_episode_logger"] = True
    ra["enable_episode_reflection"] = True
    ra["enable_reflection_memory"] = True
    ra["enable_learning_from_traces"] = enable_learning.lower() == "true"
    ra["enable_self_hindsight_learning"] = True
    ra["enable_baseline_teacher_learning"] = enable_baseline_teacher.lower() == "true"
    ra["baseline_teacher_log_path"] = baseline_teacher_log_path
    ra["enable_gt_teacher_learning"] = enable_gt_teacher.lower() == "true"
    ra["gt_trajectory_path"] = gt_trajectory_path
with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
PY
}

start_exploration() {
  local mode="$1"
  kill_exploration
  nohup bash -lc "cd '$ROOT'; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; roslaunch exploration_manager exploration.launch" > "$LOGDIR/exploration_${mode}.log" 2>&1 &
  sleep 10
}

append_result() {
  local mode="$1"
  local rc="$2"
  local dur="$3"
  local log="$4"
  /root/miniconda3/envs/apexnav/bin/python - "$RESULTS" "$mode" "$EPISODE" "$rc" "$dur" "$log" <<'PY'
import csv, os, re, sys
results, mode, ep, rc, dur, log = sys.argv[1:]
text = open(log, errors="ignore").read() if os.path.exists(log) else ""

def last(pattern, default=""):
    matches = re.findall(pattern, text, flags=re.M)
    return matches[-1].strip() if matches else default

result = last(r"^Result:\s*(.+)$", "timeout" if rc == "124" else ("error" if rc != "0" else "unknown"))
target = last(r"Finding \[([^\]]+)\]", "") or last(r"Answer for ([^:]+):", "")
success = last(r"Average Success\s*\|\s*([0-9.]+)%", "")
spl = last(r"Average SPL\s*\|\s*([0-9.]+)%", "")
soft = last(r"Average Soft SPL\s*\|\s*([0-9.]+)%", "")
dist = last(r"Average Distance to Goal\s*\|\s*([0-9.]+)", "")
steps = str(len(re.findall(r"--------------Step:", text)))
with open(results, "a", newline="") as f:
    csv.writer(f).writerow([mode, ep, target, result, success, spl, soft, dist, steps, rc, dur, log])
PY
}

run_mode() {
  local mode="$1"
  local enabled="$2"
  echo "===== mode=$mode enabled=$enabled episode=$EPISODE $(date) =====" | tee -a "$BASE/runner.log"
  set_reflective_config "$enabled" "$mode"
  start_exploration "$mode"
  local log="$LOGDIR/${mode}_ep_${EPISODE}.log"
  local start_ts
  start_ts=$(date +%s)
  local split_override=""
  if [ -n "$DATASET_SPLIT" ]; then
    split_override="habitat.dataset.split=$DATASET_SPLIT"
  fi
  timeout "$EP_TIMEOUT" bash -lc "cd '$ROOT'; source /opt/ros/noetic/setup.bash; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; source ./devel/setup.bash; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ROS_MASTER_URI=http://localhost:11311 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; python -u habitat_evaluation.py --dataset $DATASET test_epi_num=$EPISODE need_video=false reflective_agent.enable_reflective_agent=$enabled reflective_agent.run_id=${RUN_ID}_${mode} reflective_agent.learning_write_mode=$REFLECTIVE_LEARNING_WRITE_MODE reflective_agent.memory_write_mode=$REFLECTIVE_LEARNING_WRITE_MODE reflective_agent.tool_call_dataset_path=$REFLECTIVE_TOOL_CALL_DATASET_PATH reflective_agent.memory_path=$REFLECTIVE_MEMORY_PATH reflective_agent.policy_patch_path=$REFLECTIVE_POLICY_PATCH_PATH $split_override" > "$log" 2>&1
  local rc=$?
  local dur=$(( $(date +%s) - start_ts ))
  append_result "$mode" "$rc" "$dur" "$log"
  echo "mode=$mode rc=$rc dur=${dur}s log=$log" | tee -a "$BASE/runner.log"
  kill_exploration
}

cd "$ROOT" || exit 2
ensure_roscore
start_vlm || exit 1
if [ "$RUN_BASELINE" = "true" ]; then
  run_mode baseline false
fi
if [ "$RUN_REFLECTIVE" = "true" ]; then
  run_mode reflective true
fi
restore_config
echo "RESULTS=$RESULTS" | tee -a "$BASE/runner.log"
cat "$RESULTS"
