#!/usr/bin/env bash
set -euo pipefail

echo "== Host =="
hostname || true
uname -a || true
if [ -f /etc/os-release ]; then
  sed -n '1,8p' /etc/os-release
fi

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo
echo "== ROS =="
ls -d /opt/ros/* 2>/dev/null || echo "no /opt/ros installation found"
command -v roscore || true
command -v catkin_make || true

echo
echo "== Conda / Python =="
command -v conda || true
python3 --version || true
if command -v conda >/dev/null 2>&1; then
  conda env list || true
fi

echo
echo "== Repo files =="
pwd
git log -1 --oneline 2>/dev/null || true
for path in \
  data/groundingdino_swint_ogc.pth \
  data/mobile_sam.pt \
  data/yolov7-e6e.pt \
  GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  yolov7/models/experimental.py \
  data/datasets/objectnav/hm3d/v1/val/val.json.gz \
  data/datasets/objectnav/hm3d/v2/val/val.json.gz \
  data/datasets/objectnav/mp3d/v1/val/val.json.gz \
  data/scene_datasets/hm3d/val \
  data/scene_datasets/mp3d
do
  if [ -e "$path" ]; then
    echo "OK      $path"
  else
    echo "MISSING $path"
  fi
done

