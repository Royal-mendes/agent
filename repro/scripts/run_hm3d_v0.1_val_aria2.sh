#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ApexNav/agent-Apexnav}
DATA_DIR=${DATA_DIR:-$ROOT/data}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$DATA_DIR/downloads/hm3d_v0.1_val}
VERSION_DIR=${VERSION_DIR:-$DATA_DIR/versioned_data/hm3d-0.1/hm3d}
LOG_DIR=${LOG_DIR:-$ROOT/repro/logs}
: "${MATTERPORT_USER:?MATTERPORT_USER is required}"
: "${MATTERPORT_PASS:?MATTERPORT_PASS is required}"

mkdir -p "$DOWNLOAD_DIR" "$VERSION_DIR" "$LOG_DIR"

# Salvage the partial file left by the official downloader so aria2 can resume it.
if [ -f "$DATA_DIR/hm3d-val-semantic-annots-v0.1.tar.gz" ] && [ ! -f "$DOWNLOAD_DIR/hm3d-val-semantic-annots-v0.1.tar.gz" ]; then
  mv "$DATA_DIR/hm3d-val-semantic-annots-v0.1.tar.gz" "$DOWNLOAD_DIR/"
fi

file_mime() {
  file -b --mime-type "$1" 2>/dev/null || true
}

validate_tar() {
  local path="$1"
  if [[ "$path" == *.tar.gz ]]; then
    tar -tzf "$path" >/dev/null 2>&1
  else
    tar -tf "$path" >/dev/null 2>&1
  fi
}

download_one() {
  local url="$1"
  local out="$2"
  local path="$DOWNLOAD_DIR/$out"

  if [ -f "$path" ]; then
    mime=$(file_mime "$path")
    if [[ "$mime" == text/html* ]]; then
      rm -f "$path"
    fi
  fi

  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "START $(date '+%F %T %Z') $out attempt=$attempt"
    if curl \
      --fail \
      --continue-at - \
      --location \
      --user "$MATTERPORT_USER:$MATTERPORT_PASS" \
      "$url" \
      -o "$path"
    then
      if validate_tar "$path"; then
        ls -lh "$path"
        echo "DONE  $(date '+%F %T %Z') $out"
        break
      fi
      echo "INVALID $(date '+%F %T %Z') $out"
    else
      echo "RETRY $(date '+%F %T %Z') $out"
    fi

    sleep 10
  done
}

download_one "https://api.matterport.com/resources/habitat/hm3d-val-configs.tar" "hm3d-val-configs.tar"
download_one "https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.1.tar" "hm3d-val-semantic-configs-v0.1.tar"
download_one "https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.1.tar.gz" "hm3d-val-semantic-annots-v0.1.tar.gz"
download_one "https://api.matterport.com/resources/habitat/hm3d-val-habitat.tar" "hm3d-val-habitat.tar"

/usr/bin/python3 - "$DATA_DIR" "$DOWNLOAD_DIR" "$VERSION_DIR" <<'PY'
import gzip
import json
import os
import pathlib
import shutil
import sys
import tarfile

data_dir = pathlib.Path(sys.argv[1])
download_dir = pathlib.Path(sys.argv[2])
version_dir = pathlib.Path(sys.argv[3])
version_dir.mkdir(parents=True, exist_ok=True)

specs = [
    {
        "package_name": "hm3d-val-habitat.tar",
        "extract_postfix": "val",
        "downloaded_file_list": "hm3d-0.1/val-habitat-files.json.gz",
    },
    {
        "package_name": "hm3d-val-configs.tar",
        "extract_postfix": "val",
        "downloaded_file_list": "hm3d-0.1/val-configs-files.json.gz",
    },
    {
        "package_name": "hm3d-val-semantic-annots-v0.1.tar.gz",
        "extract_postfix": "val",
        "downloaded_file_list": "hm3d-0.1/val-semantic-annots-files.json.gz",
    },
    {
        "package_name": "hm3d-val-semantic-configs-v0.1.tar",
        "extract_postfix": "val",
        "downloaded_file_list": "hm3d-0.1/val-semantic-configs-files.json.gz",
        "semantic_config_post": True,
    },
]

for spec in specs:
    package_path = download_dir / spec["package_name"]
    if not package_path.exists():
        raise FileNotFoundError(package_path)

    extract_dir = version_dir / spec["extract_postfix"]
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(package_path, "r:*") as tar_ref:
        members = tar_ref.getnames()
        tar_ref.extractall(extract_dir)

    package_files = [str(extract_dir / member) for member in members]

    if spec.get("semantic_config_post"):
        src = extract_dir / "hm3d_annotated_basis.scene_dataset_config.json"
        if src.exists():
            dst = version_dir / "hm3d_annotated_basis.scene_dataset_config.json"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.move(str(src), str(dst))
            package_files = [str(dst)] + package_files

    downloaded_file_list = data_dir / "versioned_data" / spec["downloaded_file_list"]
    downloaded_file_list.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(downloaded_file_list, "wt") as f:
        json.dump([str(extract_dir)] + package_files, f)

scene_link = data_dir / "scene_datasets" / "hm3d_v0.1"
if scene_link.is_symlink() or scene_link.is_file():
    scene_link.unlink()
if not scene_link.exists():
    scene_link.symlink_to(version_dir, target_is_directory=True)

print(f"scene_link={scene_link}")
print(f"version_dir={version_dir}")
print(f"val_dir={version_dir / 'val'}")
PY

echo "DONE $(date '+%F %T %Z')"
