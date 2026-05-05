# ApexNav Reproduction Notes

This directory contains a practical reproduction checklist for:

> ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion

Repository commit inspected: `204121e`.

## Target Results

The paper reports the following main benchmark numbers:

| Dataset | Episodes | Target SR | Target SPL |
| --- | ---: | ---: | ---: |
| HM3Dv1 val | 2000 | 59.6 | 33.0 |
| HM3Dv2 val | 1000 | 76.2 | 38.0 |
| MP3D val | 2195 | 39.2 | 17.8 |

The most useful first reproduction target is HM3Dv2, because the paper notes
that HM3Dv2 has fewer multi-floor annotation/pathology issues than HM3Dv1/MP3D.

## Current Server Status

The server `root@10.31.112.128:19214` has:

- Ubuntu 22.04.1
- conda available at `/root/miniconda3/bin/conda`
- no ROS installation found
- no Docker installation found
- no visible NVIDIA GPU (`nvidia-smi` reports no devices)
- repository cloned at `/root/autodl-tmp/ApexNav/agent-Apexnav`

This is enough for code inspection and staging files, but not enough for a
faithful full reproduction. The upstream repository is tested on Ubuntu 20.04
with ROS Noetic, Python 3.9, Habitat 0.3.1, and CUDA GPU acceleration. The paper
reports experiments on an RTX 4090.

## Required Private / Permissioned Inputs

The code can download public task datasets and public model weights, but full
evaluation also needs permissioned scene datasets:

- HM3D scene dataset: apply for Habitat Matterport 3D Research Dataset access.
- MP3D scene dataset: apply for Matterport3D access and download with the
  official `download_mp.py` script.

The public ObjectNav task JSON files alone are not enough; Habitat also needs
the actual scene meshes under `data/scene_datasets`.

## Recommended Environment

Use a Linux GPU machine matching the paper as closely as practical:

- Ubuntu 20.04
- ROS Noetic
- CUDA 11.8, 12.1, or 12.4
- NVIDIA GPU, preferably 24GB VRAM for BLIP-2 + GroundingDINO + MobileSAM + YOLOv7
- conda / miniconda

Ubuntu 22.04 is possible only with extra ROS work. The main branch is ROS1
Noetic oriented. The repository has a separate `ros2-jazzy` branch, but Jazzy is
normally aligned with Ubuntu 24.04 rather than this server's Ubuntu 22.04.

## Reproduction Order

Run these from the ApexNav repository root.

1. Probe the machine:

   ```bash
   bash repro/scripts/00_probe_env.sh
   ```

2. On a suitable Ubuntu 20.04 + ROS Noetic + GPU machine, install system deps:

   ```bash
   sudo apt update
   sudo apt install -y \
     build-essential cmake git wget unzip curl \
     libarmadillo-dev libompl-dev libeigen3-dev \
     ros-noetic-desktop-full ros-noetic-cv-bridge \
     ros-noetic-image-transport ros-noetic-tf \
     ros-noetic-message-filters ros-noetic-pcl-ros \
     ros-noetic-tf2-sensor-msgs python3-catkin-tools
   ```

3. Install OSQP and OSQP-Eigen exactly as in the upstream README:

   ```bash
   git clone --recursive -b v0.6.3 https://github.com/osqp/osqp.git /tmp/osqp
   cmake -S /tmp/osqp -B /tmp/osqp/build -DBUILD_SHARED_LIBS=ON
   cmake --build /tmp/osqp/build -j
   sudo cmake --install /tmp/osqp/build

   git clone -b v0.8.1 https://github.com/robotology/osqp-eigen.git /tmp/osqp-eigen
   cmake -S /tmp/osqp-eigen -B /tmp/osqp-eigen/build
   cmake --build /tmp/osqp-eigen/build -j
   sudo cmake --install /tmp/osqp-eigen/build
   ```

4. Create the Python environment:

   ```bash
   conda env create -f apexnav_environment.yaml -y
   conda activate apexnav
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

   git clone https://github.com/facebookresearch/habitat-lab.git /tmp/habitat-lab
   git -C /tmp/habitat-lab checkout tags/v0.3.1
   pip install -e /tmp/habitat-lab/habitat-lab
   pip install -e /tmp/habitat-lab/habitat-baselines

   pip install salesforce-lavis==1.0.2
   pip install -e .
   ```

5. Download public code dependencies, public weights, and public ObjectNav task
   files:

   ```bash
   bash repro/scripts/01_download_public_assets.sh
   ```

6. Add permissioned scene datasets:

   ```text
   data/scene_datasets/hm3d/val/...
   data/scene_datasets/hm3d_v0.2 -> hm3d
   data/scene_datasets/mp3d/...
   ```

7. Compile ROS packages:

   ```bash
   source /opt/ros/noetic/setup.bash
   catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
   ```

8. Run a one-episode smoke test on HM3Dv2:

   ```bash
   conda activate apexnav
   bash repro/scripts/02_run_smoke_hm3dv2.sh 0
   ```

9. Run full benchmark:

   ```bash
   conda activate apexnav
   bash repro/scripts/03_run_benchmark.sh hm3dv2
   bash repro/scripts/03_run_benchmark.sh hm3dv1
   bash repro/scripts/03_run_benchmark.sh mp3d
   ```

10. Parse the output record:

    ```bash
    python repro/scripts/04_parse_record.py videos/test_hm3dv2_val/continue.txt
    ```

## Expected Output Locations

- HM3Dv1: `videos/test_hm3dv1_val/record.txt`
- HM3Dv2: `videos/test_hm3dv2_val/record.txt`
- MP3D: `videos/test_mp3d_val/record.txt`
- Resume state: corresponding `continue.txt`

The repository writes newest records at the top of each file.

