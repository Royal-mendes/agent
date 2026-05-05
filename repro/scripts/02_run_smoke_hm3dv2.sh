#!/usr/bin/env bash
set -euo pipefail

EPISODE="${1:-0}"
NEED_VIDEO="${NEED_VIDEO:-false}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/repro/logs/smoke_hm3dv2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [ ! -f devel/setup.bash ]; then
  echo "Missing devel/setup.bash. Run catkin_make first." >&2
  exit 1
fi

# ROS Noetic setup scripts reference optional variables that may be unset.
# Keep this smoke script strict, but relax nounset only while sourcing env.
set +u
source ./devel/setup.bash
set -u

cleanup() {
  jobs -pr | xargs -r kill || true
}
trap cleanup EXIT

python -m vlm.detector.grounding_dino --port 12181 >"$LOG_DIR/grounding_dino.log" 2>&1 &
python -m vlm.itm.blip2itm --port 12182 >"$LOG_DIR/blip2itm.log" 2>&1 &
python -m vlm.segmentor.sam --port 12183 >"$LOG_DIR/mobile_sam.log" 2>&1 &
python -m vlm.detector.yolov7 --port 12184 >"$LOG_DIR/yolov7.log" 2>&1 &

echo "Starting VLM services, logs in $LOG_DIR"
sleep "${VLM_STARTUP_WAIT:-180}"

roslaunch exploration_manager exploration.launch >"$LOG_DIR/exploration.log" 2>&1 &
sleep "${ROS_STARTUP_WAIT:-20}"

python habitat_evaluation.py --dataset hm3dv2 test_epi_num="$EPISODE" need_video="$NEED_VIDEO" \
  2>&1 | tee "$LOG_DIR/habitat_eval.log"

echo "Smoke test logs: $LOG_DIR"
