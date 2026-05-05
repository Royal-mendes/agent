#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ApexNav/agent-Apexnav}
WORKERS=${WORKERS:-4}
STRIDE=${STRIDE:-$WORKERS}
BASE_ROS_PORT=${BASE_ROS_PORT:-11311}
BASE_VLM_PORT=${BASE_VLM_PORT:-12181}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$ROOT/repro/full_eval_$STAMP"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"

wait_port() {
  local port="$1"
  local tries="${2:-120}"
  for _ in $(seq 1 "$tries"); do
    if /root/miniconda3/envs/apexnav/bin/python - "$port" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", port))
s.close()
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_ros() {
  local wid="$1"
  local ros_port="$2"
  local log="$RUN_DIR/logs/worker_${wid}_roscore.log"
  nohup bash -lc "source /opt/ros/noetic/setup.bash; roscore -p $ros_port" > "$log" 2>&1 &
  echo "$!" > "$RUN_DIR/pids/worker_${wid}_roscore.pid"
  wait_port "$ros_port" 90
}

start_vlm() {
  local wid="$1"
  local base="$2"
  local gdino=$((base + 0))
  local blip=$((base + 1))
  local sam=$((base + 2))
  local yolo=$((base + 3))

  declare -a specs=(
    "gdino vlm.detector.grounding_dino $gdino"
    "blip2 vlm.itm.blip2itm $blip"
    "sam vlm.segmentor.sam $sam"
    "yolo vlm.detector.yolov7 $yolo"
  )

  for spec in "${specs[@]}"; do
    set -- $spec
    local name="$1"
    local mod="$2"
    local port="$3"
    local log="$RUN_DIR/logs/worker_${wid}_vlm_${name}.log"
    nohup bash -lc "cd $ROOT; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; python -m $mod --port $port" > "$log" 2>&1 &
    echo "$!" > "$RUN_DIR/pids/worker_${wid}_vlm_${name}.pid"
    sleep 2
  done

  wait_port "$gdino" 180
  wait_port "$blip" 240
  wait_port "$sam" 180
  wait_port "$yolo" 180
}

start_exploration() {
  local wid="$1"
  local ros_port="$2"
  local log="$RUN_DIR/logs/worker_${wid}_exploration.log"
  nohup bash -lc "cd $ROOT; source /opt/ros/noetic/setup.bash; source ./devel/setup.bash; export ROS_MASTER_URI=http://localhost:$ros_port; roslaunch exploration_manager exploration.launch" > "$log" 2>&1 &
  echo "$!" > "$RUN_DIR/pids/worker_${wid}_exploration.pid"
  sleep 8
}

start_eval() {
  local wid="$1"
  local ros_port="$2"
  local base="$3"
  local gdino=$((base + 0))
  local blip=$((base + 1))
  local sam=$((base + 2))
  local yolo=$((base + 3))
  local log="$RUN_DIR/logs/worker_${wid}_eval.log"

  nohup bash -lc "cd $ROOT; source /opt/ros/noetic/setup.bash; source /root/miniconda3/etc/profile.d/conda.sh; conda activate apexnav; source ./devel/setup.bash; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ROS_MASTER_URI=http://localhost:$ros_port GDINO_PORT=$gdino BLIP2ITM_PORT=$blip MOBILE_SAM_PORT=$sam YOLOV7_PORT=$yolo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; python -u habitat_evaluation.py --dataset hm3dv2 need_video=false video_output_path=videos/full_parallel_worker${wid}_{split} record_file_name=record_worker${wid}.txt continue_file_name=continue_worker${wid}.txt +episode_start=$wid +episode_stride=$STRIDE" > "$log" 2>&1 &
  echo "$!" > "$RUN_DIR/pids/worker_${wid}_eval.pid"
}

echo "$RUN_DIR" > "$ROOT/repro/latest_full_eval_run.txt"
echo "run_dir=$RUN_DIR"

for wid in $(seq 0 $((WORKERS - 1))); do
  ros_port=$((BASE_ROS_PORT + wid * 10))
  vlm_base=$((BASE_VLM_PORT + wid * 20))
  echo "starting worker=$wid ros_port=$ros_port vlm_base=$vlm_base"
  start_ros "$wid" "$ros_port"
  start_vlm "$wid" "$vlm_base"
  start_exploration "$wid" "$ros_port"
  start_eval "$wid" "$ros_port" "$vlm_base"
done

echo "all_workers_started"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
