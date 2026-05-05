#!/usr/bin/env bash
set -o pipefail

cd /root/autodl-tmp/ApexNav/agent-Apexnav || exit 2

START_EP=${START_EP:-2}
END_EP=${END_EP:-999}
EP_TIMEOUT=${EP_TIMEOUT:-1200}
RUN_ID=${RUN_ID:-full_hm3dv2_val_per_episode_$(date +%Y%m%d_%H%M%S)}
VLM_BASE_PORT=${VLM_BASE_PORT:-13181}

BASE=/root/autodl-tmp/ApexNav/agent-Apexnav/repro/per_episode/$RUN_ID
LOGDIR=$BASE/logs
RESULTS=$BASE/results.csv
SUMMARY=$BASE/summary.txt
VLM_PIDS=$BASE/vlm_pids.txt

GDINO_PORT=
BLIP2ITM_PORT=
MOBILE_SAM_PORT=
YOLOV7_PORT=

mkdir -p "$LOGDIR"
echo "$RUN_ID" > /root/autodl-tmp/ApexNav/agent-Apexnav/repro/current_per_episode_eval_run.txt

if [ ! -f "$RESULTS" ]; then
  echo "episode_index,target,result,success_pct,spl_pct,soft_spl_pct,distance_to_goal,steps,rc,duration_sec,log" > "$RESULTS"
fi

ensure_roscore() {
  /root/miniconda3/envs/apexnav/bin/python - <<'PY' >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", 11311))
s.close()
PY
  if [ $? -ne 0 ]; then
    nohup bash -lc "source /opt/ros/noetic/setup.bash; roscore -p 11311" > "$LOGDIR/roscore.log" 2>&1 &
    sleep 6
  fi
}

port_open() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
p = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", p))
s.close()
PY
}

wait_for_port() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" "${2:-90}" <<'PY' >/dev/null 2>&1
import socket
import sys
import time

port = int(sys.argv[1])
timeout_sec = int(sys.argv[2])
deadline = time.time() + timeout_sec
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

find_free_vlm_base() {
  /root/miniconda3/envs/apexnav/bin/python - "$VLM_BASE_PORT" <<'PY'
import socket
import sys

base = int(sys.argv[1])

def port_in_use(port: int) -> bool:
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
    ports = [base + offset for offset in range(4)]
    if not any(port_in_use(port) for port in ports):
        print(base)
        break
    base += 10
PY
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
}

cleanup() {
  kill_exploration
  cleanup_vlm
}

ensure_vlm() {
  local base_port
  base_port=$(find_free_vlm_base)
  GDINO_PORT=$base_port
  BLIP2ITM_PORT=$((base_port + 1))
  MOBILE_SAM_PORT=$((base_port + 2))
  YOLOV7_PORT=$((base_port + 3))
  export GDINO_PORT BLIP2ITM_PORT MOBILE_SAM_PORT YOLOV7_PORT
  : > "$VLM_PIDS"
  for spec in \
    "gdino vlm.detector.grounding_dino $GDINO_PORT" \
    "blip2 vlm.itm.blip2itm $BLIP2ITM_PORT" \
    "sam vlm.segmentor.sam $MOBILE_SAM_PORT" \
    "yolo vlm.detector.yolov7 $YOLOV7_PORT"; do
    set -- $spec
    name=$1
    mod=$2
    port=$3
    nohup bash -lc "cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; python -m ${mod} --port ${port}" > "$LOGDIR/vlm_${name}_${port}.log" 2>&1 &
    echo $! >> "$VLM_PIDS"
    if ! wait_for_port "$port" 180; then
      echo "failed to start $name on port $port" | tee -a "$BASE/runner.log"
      return 1
    fi
   done
}

kill_exploration() {
  source /opt/ros/noetic/setup.bash 2>/dev/null || true
  export ROS_MASTER_URI=http://localhost:11311
  rosnode kill /exploration_node /tsp_solver >/dev/null 2>&1 || true
  ps -eo pid=,cmd= | awk '/roslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/exploration_node/ || /devel\/lib\/lkh_mtsp_solver\/tsp_node/ {print $1}' | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
  ps -eo pid=,cmd= | awk '/roslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/exploration_node/ || /devel\/lib\/lkh_mtsp_solver\/tsp_node/ {print $1}' | xargs -r kill -KILL 2>/dev/null || true
}

start_exploration() {
  local ep="$1"
  kill_exploration
  nohup bash -lc "cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; roslaunch exploration_manager exploration.launch" > "$LOGDIR/exploration_ep_${ep}.log" 2>&1 &
  sleep 8
}

append_result() {
  local ep="$1"
  local rc="$2"
  local dur="$3"
  local log="$4"
  /root/miniconda3/envs/apexnav/bin/python - "$RESULTS" "$ep" "$rc" "$dur" "$log" <<'PY'
import csv
import os
import re
import sys

results, ep, rc, dur, log = sys.argv[1:]
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
    csv.writer(f).writerow([ep, target, result, success, spl, soft, dist, steps, rc, dur, log])
PY
}

summarize() {
  /root/miniconda3/envs/apexnav/bin/python - "$RESULTS" "$SUMMARY" <<'PY'
import collections
import csv
import os
import sys

results, out = sys.argv[1:]
rows = list(csv.DictReader(open(results))) if os.path.exists(results) else []
counts = collections.Counter(row["result"] for row in rows)
successes = sum(1 for row in rows if row["result"] == "success")

with open(out, "w") as f:
    f.write(f"completed_rows={len(rows)}\n")
    f.write(f"success_rows={successes}\n")
    f.write(f"result_counts={dict(counts)!r}\n")
    if rows:
        f.write(f"last_episode={rows[-1]['episode_index']}\n")
        f.write(f"last_result={rows[-1]['result']}\n")
        f.write(f"last_target={rows[-1]['target']}\n")
PY
}

trap cleanup EXIT INT TERM

echo "RUN_ID=$RUN_ID BASE=$BASE START_EP=$START_EP END_EP=$END_EP" | tee -a "$BASE/runner.log"
ensure_roscore
ensure_vlm || exit 1
echo "VLM_PORTS=gdino:$GDINO_PORT blip2itm:$BLIP2ITM_PORT mobile_sam:$MOBILE_SAM_PORT yolov7:$YOLOV7_PORT" | tee -a "$BASE/runner.log"

for ep in $(seq "$START_EP" "$END_EP"); do
  if awk -F, -v ep="$ep" 'NR > 1 && $1 == ep {found=1} END {exit !found}' "$RESULTS"; then
    continue
  fi

  echo "===== EPISODE $ep $(date) =====" | tee -a "$BASE/runner.log"
  start_exploration "$ep"
  ep_log="$LOGDIR/ep_$(printf "%04d" "$ep").log"
  start_ts=$(date +%s)
  timeout "$EP_TIMEOUT" bash -lc "cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /opt/ros/noetic/setup.bash; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; source ./devel/setup.bash; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ROS_MASTER_URI=http://localhost:11311 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; python -u habitat_evaluation.py --dataset hm3dv2 test_epi_num=${ep} need_video=false" > "$ep_log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start_ts ))

  append_result "$ep" "$rc" "$dur" "$ep_log"
  summarize
  echo "episode=$ep rc=$rc dur=${dur}s" | tee -a "$BASE/runner.log"
  kill_exploration
  sleep 2
done

summarize
kill_exploration
cleanup_vlm
echo "DONE $(date)" | tee -a "$BASE/runner.log"
