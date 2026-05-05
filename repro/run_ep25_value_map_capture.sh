#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ApexNav/agent-Apexnav

RUN_ID=${RUN_ID:-semantic_score_ep25_$(date +%Y%m%d_%H%M%S)}
BASE=/root/autodl-tmp/ApexNav/agent-Apexnav/repro/$RUN_ID
LOGDIR=$BASE/logs
mkdir -p "$LOGDIR" "$BASE/videos"
: > "$BASE/run.log"

log_stage() {
  echo "$(date '+%F %T') $*" | tee -a "$BASE/run.log"
}

GDINO_PORT=${GDINO_PORT:-15181}
BLIP2ITM_PORT=${BLIP2ITM_PORT:-15182}
MOBILE_SAM_PORT=${MOBILE_SAM_PORT:-15183}
YOLOV7_PORT=${YOLOV7_PORT:-15184}
export GDINO_PORT BLIP2ITM_PORT MOBILE_SAM_PORT YOLOV7_PORT

VLM_PIDS=$BASE/vlm_pids.txt
: > "$VLM_PIDS"

port_open() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", int(sys.argv[1])))
s.close()
PY
}

wait_for_port() {
  /root/miniconda3/envs/apexnav/bin/python - "$1" <<'PY'
import socket
import sys
import time
p = int(sys.argv[1])
deadline = time.time() + 180
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", p))
        print("ready", p)
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
    nohup bash -lc 'source /opt/ros/noetic/setup.bash; roscore -p 11311' \
      > "$LOGDIR/roscore.log" 2>&1 &
    sleep 6
  fi
}

kill_exploration() {
  bash -lc '
    set +e
    source /opt/ros/noetic/setup.bash >/dev/null 2>&1
    export ROS_MASTER_URI=http://localhost:11311
    timeout 5 rosnode kill /exploration_node /tsp_solver >/dev/null 2>&1 || true
    ps -eo pid=,cmd= | awk '\''/[r]oslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/[e]xploration_node/ || /devel\/lib\/lkh_mtsp_solver\/[t]sp_node/ {print $1}'\'' | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    ps -eo pid=,cmd= | awk '\''/[r]oslaunch exploration_manager exploration\.launch/ || /devel\/lib\/exploration_manager\/[e]xploration_node/ || /devel\/lib\/lkh_mtsp_solver\/[t]sp_node/ {print $1}'\'' | xargs -r kill -KILL 2>/dev/null || true
  ' >/dev/null 2>&1 || true
}

kill_vlm() {
  local sig=$1
  while read -r pid; do
    [ -n "$pid" ] && kill "-$sig" "$pid" >/dev/null 2>&1
  done < "$VLM_PIDS"
  ps -eo pid=,cmd= | awk \
    -v gdino="$GDINO_PORT" \
    -v blip2="$BLIP2ITM_PORT" \
    -v sam="$MOBILE_SAM_PORT" \
    -v yolo="$YOLOV7_PORT" \
    '$0 ~ /python -m vl[m]/ && ($0 ~ "--port " gdino || $0 ~ "--port " blip2 || $0 ~ "--port " sam || $0 ~ "--port " yolo) {print $1}' |
    xargs -r kill "-$sig" >/dev/null 2>&1 || true
}

cleanup() {
  set +e
  if [ -n "${BAG_PID:-}" ]; then
    kill -INT "$BAG_PID" >/dev/null 2>&1
    wait "$BAG_PID" >/dev/null 2>&1
  fi
  kill_exploration
  kill_vlm TERM
  sleep 2
  kill_vlm KILL
}
trap cleanup EXIT INT TERM

log_stage "ensure roscore"
ensure_roscore

for spec in \
  "gdino vlm.detector.grounding_dino $GDINO_PORT" \
  "blip2 vlm.itm.blip2itm $BLIP2ITM_PORT" \
  "sam vlm.segmentor.sam $MOBILE_SAM_PORT" \
  "yolo vlm.detector.yolov7 $YOLOV7_PORT"; do
  set -- $spec
  name=$1
  mod=$2
  port=$3
  log_stage "start ${name} on ${port}"
  nohup bash -lc "cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; exec python -m ${mod} --port ${port}" \
    > "$LOGDIR/vlm_${name}_${port}.log" 2>&1 &
  echo $! >> "$VLM_PIDS"
  wait_for_port "$port"
done

log_stage "RUN_ID=$RUN_ID"
log_stage "VLM_PORTS=$GDINO_PORT,$BLIP2ITM_PORT,$MOBILE_SAM_PORT,$YOLOV7_PORT"

log_stage "kill old exploration"
kill_exploration
log_stage "start exploration"
nohup bash -lc 'cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; roslaunch exploration_manager exploration.launch' \
  > "$LOGDIR/exploration_ep25.log" 2>&1 &
sleep 8

log_stage "start rosbag"
nohup bash -lc "source /opt/ros/noetic/setup.bash; export ROS_MASTER_URI=http://localhost:11311; exec rosbag record -O '$BASE/value_map_ep25.bag' /grid_map/value_map" \
  > "$LOGDIR/rosbag_value_map.log" 2>&1 &
BAG_PID=$!
sleep 2

set +e
log_stage "start habitat ep25"
bash -lc "cd /root/autodl-tmp/ApexNav/agent-Apexnav; source /opt/ros/noetic/setup.bash; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; source ./devel/setup.bash; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ROS_MASTER_URI=http://localhost:11311 GDINO_PORT=$GDINO_PORT BLIP2ITM_PORT=$BLIP2ITM_PORT MOBILE_SAM_PORT=$MOBILE_SAM_PORT YOLOV7_PORT=$YOLOV7_PORT; python -u habitat_evaluation.py --dataset hm3dv2 test_epi_num=25 need_video=true video_output_path=$BASE/videos/test_hm3dv2_val" \
  > "$LOGDIR/ep_0025.log" 2>&1
RC=$?
set -e
log_stage "habitat rc=$RC"

sleep 2
kill -INT "$BAG_PID" >/dev/null 2>&1 || true
wait "$BAG_PID" >/dev/null 2>&1 || true
unset BAG_PID

RGB_VIDEO=$(find "$BASE/videos" -type f -name "*.mp4" | sort | tail -n 1)
log_stage "RC=$RC"
log_stage "RGB_VIDEO=$RGB_VIDEO"
log_stage "BAG=$BASE/value_map_ep25.bag"

log_stage "render value map video"
bash -lc "source /opt/ros/noetic/setup.bash; exec /usr/bin/python3 /root/autodl-tmp/ApexNav/agent-Apexnav/repro/render_value_map_video.py --bag '$BASE/value_map_ep25.bag' --rgb-video '$RGB_VIDEO' --out-dir '$BASE/videos'" | tee -a "$BASE/run.log"

echo "$BASE" > /root/autodl-tmp/ApexNav/agent-Apexnav/repro/latest_semantic_score_ep25_dir.txt
exit "$RC"
