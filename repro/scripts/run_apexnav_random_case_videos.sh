#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ApexNav/agent-Apexnav}
RUN_ID=${RUN_ID:-$(cat "$ROOT/repro/current_per_episode_eval_run.txt")}
BASE="$ROOT/repro/per_episode/$RUN_ID"
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_BASE="$ROOT/repro/random_case_videos/$STAMP"
LOG_DIR="$OUT_BASE/logs"
VID_DIR="$OUT_BASE/videos"
MANIFEST="$OUT_BASE/manifest.csv"
SEED=${SEED:-20260423}
COUNT=${COUNT:-5}
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}
EXCLUDE_EPISODES=${EXCLUDE_EPISODES:-}

mkdir -p "$LOG_DIR" "$VID_DIR"

ensure_roscore() {
  /root/miniconda3/envs/apexnav/bin/python - <<'PY' >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", 11311))
s.close()
PY
  if [ $? -ne 0 ]; then
    nohup bash -lc "source /opt/ros/noetic/setup.bash; roscore -p 11311" > "$LOG_DIR/roscore.log" 2>&1 &
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

ensure_vlm() {
  for spec in "gdino vlm.detector.grounding_dino 12181" "blip2 vlm.itm.blip2itm 12182" "sam vlm.segmentor.sam 12183" "yolo vlm.detector.yolov7 12184"; do
    set -- $spec
    name=$1
    mod=$2
    port=$3
    if ! port_open "$port"; then
      nohup bash -lc "cd $ROOT; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; python -m ${mod} --port ${port}" > "$LOG_DIR/vlm_${name}.log" 2>&1 &
      sleep 5
    fi
  done
}

kill_eval_stack() {
  pkill -f "run_hm3dv2_val_per_episode.sh" || true
  pkill -f "habitat_evaluation.py --dataset hm3dv2 test_epi_num" || true
  pkill -f "roslaunch exploration_manager exploration.launch" || true
  pkill -f "devel/lib/exploration_manager/exploration_node" || true
  sleep 2
  pkill -9 -f "run_hm3dv2_val_per_episode.sh" || true
  pkill -9 -f "habitat_evaluation.py --dataset hm3dv2 test_epi_num" || true
  pkill -9 -f "roslaunch exploration_manager exploration.launch" || true
  pkill -9 -f "devel/lib/exploration_manager/exploration_node" || true
}

start_exploration() {
  local ep="$1"
  nohup bash -lc "cd $ROOT; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; roslaunch exploration_manager exploration.launch" > "$LOG_DIR/exploration_ep_${ep}.log" 2>&1 &
  sleep 8
}

resume_main_eval() {
  if pgrep -f "run_hm3dv2_val_per_episode.sh" >/dev/null 2>&1; then
    return 0
  fi
  nohup bash -lc "cd $ROOT; START_EP=2 END_EP=999 EP_TIMEOUT=1200 RUN_ID=$RUN_ID ./repro/scripts/run_hm3dv2_val_per_episode.sh" > "$OUT_BASE/resume_eval.log" 2>&1 &
  echo "$!" > "$OUT_BASE/resume_eval.pid"
}

pick_cases() {
  /root/miniconda3/envs/apexnav/bin/python - "$BASE/results.csv" "$OUT_BASE/candidates.csv" "$SEED" "$EXCLUDE_EPISODES" <<'PY'
import csv, random, sys
from collections import defaultdict

src, out, seed, exclude_raw = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
exclude = {x.strip() for x in exclude_raw.split(",") if x.strip()}
groups = defaultdict(list)

with open(src) as f:
    for row in csv.DictReader(f):
        if row["result"] == "error":
            continue
        if row["episode_index"] in exclude:
            continue
        groups[row["result"]].append(row)

random.seed(seed)
for result, rows in groups.items():
    random.shuffle(rows)
    rows.sort(key=lambda r: float(r.get("duration_sec") or 1e18))

order = sorted(
    groups.keys(),
    key=lambda k: (len(groups[k]), 1 if k == "success" else 0, k),
)

selected = []
while True:
    progressed = False
    for result in order:
        rows = groups[result]
        if rows:
            selected.append(rows.pop(0))
            progressed = True
    if not progressed:
        break

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode_index", "target", "result", "duration_sec", "log"])
    for row in selected:
        w.writerow([row["episode_index"], row["target"], row["result"], row["duration_sec"], row["log"]])
PY
}

sanitize() {
  echo "$1" | tr ' /[]()' '_' | tr -cd '[:alnum:]_.-'
}

run_case() {
  local ep="$1"
  local target="$2"
  local result="$3"
  local ep_log="$LOG_DIR/ep_$(printf "%04d" "$ep").log"
  local target_safe result_safe dst

  target_safe=$(sanitize "$target")
  result_safe=$(sanitize "$result")
  rm -f "$ROOT/videos/video_once.mp4"
  start_exploration "$ep"
  timeout "$TIMEOUT_SEC" bash -lc "cd $ROOT; source /opt/ros/noetic/setup.bash; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; source ./devel/setup.bash; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ROS_MASTER_URI=http://localhost:11311; python -u habitat_evaluation.py --dataset hm3dv2 test_epi_num=${ep} need_video=true" > "$ep_log" 2>&1 || true
  pkill -f "roslaunch exploration_manager exploration.launch" || true
  pkill -f "devel/lib/exploration_manager/exploration_node" || true
  sleep 2
  if [ -f "$ROOT/videos/video_once.mp4" ]; then
    dst="$VID_DIR/ep_$(printf "%04d" "$ep")_${target_safe}_${result_safe}.mp4"
    mv "$ROOT/videos/video_once.mp4" "$dst"
    echo "$ep,$target,$result,$dst,$ep_log" >> "$MANIFEST"
    return 0
  fi
  return 1
}

trap 'resume_main_eval' EXIT

echo "episode_index,target,result,video_path,run_log" > "$MANIFEST"
pick_cases
kill_eval_stack
ensure_roscore
ensure_vlm

successes=0
while IFS=, read -r ep target result duration log; do
  if [ "$successes" -ge "$COUNT" ]; then
    break
  fi
  echo "CASE episode=$ep target=$target result=$result duration=${duration}s"
  if run_case "$ep" "$target" "$result"; then
    successes=$((successes + 1))
    echo "SAVED episode=$ep count=$successes/$COUNT"
  else
    echo "FAILED episode=$ep"
  fi
done < <(tail -n +2 "$OUT_BASE/candidates.csv")

echo "DONE count=$successes"
