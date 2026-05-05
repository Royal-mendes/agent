#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    echo "exists: $out"
    return
  fi
  echo "download: $url -> $out"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    curl -L -o "$out" "$url"
  fi
}

extract_zip_once() {
  local zip_path="$1"
  local out_dir="$2"
  local sentinel="$3"
  if [ -e "$sentinel" ]; then
    echo "exists: $sentinel"
    return
  fi
  mkdir -p "$out_dir"
  unzip -q "$zip_path" -d "$out_dir"
}

mkdir -p data

if [ ! -d GroundingDINO ]; then
  git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git
fi

if [ ! -d yolov7 ]; then
  git clone --depth 1 https://github.com/WongKinYiu/yolov7.git
fi

download \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "data/groundingdino_swint_ogc.pth"

download \
  "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt" \
  "data/yolov7-e6e.pt"

download \
  "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" \
  "data/mobile_sam.pt"

download \
  "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip" \
  "data/datasets/objectnav/hm3d/v1.zip"
extract_zip_once \
  "data/datasets/objectnav/hm3d/v1.zip" \
  "data/datasets/objectnav/hm3d" \
  "data/datasets/objectnav/hm3d/v1/val/val.json.gz"
if [ -d data/datasets/objectnav/hm3d/objectnav_hm3d_v1 ] && [ ! -d data/datasets/objectnav/hm3d/v1 ]; then
  mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
fi

download \
  "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip" \
  "data/datasets/objectnav/hm3d/v2.zip"
extract_zip_once \
  "data/datasets/objectnav/hm3d/v2.zip" \
  "data/datasets/objectnav/hm3d" \
  "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
if [ -d data/datasets/objectnav/hm3d/objectnav_hm3d_v2 ] && [ ! -d data/datasets/objectnav/hm3d/v2 ]; then
  mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
fi

download \
  "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip" \
  "data/datasets/objectnav/mp3d/v1.zip"
extract_zip_once \
  "data/datasets/objectnav/mp3d/v1.zip" \
  "data/datasets/objectnav/mp3d/v1" \
  "data/datasets/objectnav/mp3d/v1/val/val.json.gz"

echo
echo "Public assets staged. Permissioned scene datasets are still required."

