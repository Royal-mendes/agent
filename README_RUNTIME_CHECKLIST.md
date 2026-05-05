# ApexNav Reflective Agent Runtime Checklist

This repository is a code snapshot of the ApexNav ObjectNav reproduction plus the reflective/self-evolving navigation agent layer.
It intentionally does not include datasets, model weights, build products, logs, videos, or private API keys.

## 1. What This Code Contains

Core ApexNav components are kept in place:

- `src/planner/plan_env/`: map, semantic value map, object map, frontier data.
- `src/planner/exploration_manager/`: high-level ApexNav planner/FSM and reflective bridge integration.
- `src/planner/path_searching/`: low-level path search.
- `habitat_evaluation.py`: Habitat ObjectNav evaluation loop and ROS bridge.
- `vlm/`, `GroundingDINO/`, `yolov7/`: detector/VLM support servers used by ApexNav.

Reflective agent additions:

- `agent/`: Python reflective skill-selection agent.
- `agent/memory/`: role/task/working/experience memory.
- `agent/skill/`: skill registry, skill specs, Navigation Action Pair cards.
- `agent/execution/`: monitored skill executor and postcondition checker.
- `agent/reflection/`: failure taxonomy, rule-based reflection, policy patch table.
- `agent/learning/`: GT trajectory / progress teacher and hindsight labeling utilities.
- `configs/experiments/`: ablation and runtime configs.
- `repro/scripts/`: smoke tests, Qwen-VL server scripts, small eval helpers.

Current default HM3Dv2 setting in `config/habitat_eval_hm3dv2.yaml`:

- `reflective_agent.enable_reflective_agent: true`
- `reflective_agent.vlm_provider: local`
- `reflective_agent.vlm_model: qwen2.5-vl-7b-instruct`
- `reflective_agent.enable_rgb_observation: true`
- `reflective_agent.enable_semantic_map_observation: false`
- GT teacher learning switches are enabled for validation-time self-evolution experiments.

## 2. Expected Runtime Environment

The working server used for this snapshot had:

- Ubuntu 20.04
- ROS Noetic
- Conda environment: `/root/miniconda3/envs/apexnav`
- Python 3.9 in the ApexNav environment
- Catkin workspace rooted at the repository root
- Optional local VLM environment: `/root/autodl-tmp/envs/qwen_vl`
- Optional Qwen model path: `/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`

Minimum practical hardware:

- GPU is strongly recommended for Habitat-Sim, SAM/GroundingDINO/YOLO, and local Qwen-VL.
- CPU-only or 1GB cgroup/no-card mode is not enough for smoke/evaluation; detector and Habitat processes may be killed.

Basic environment activation:

```bash
cd /root/autodl-tmp/ApexNav/agent-Apexnav
source /opt/ros/noetic/setup.bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate apexnav
```

Build ROS/C++ code:

```bash
catkin_make
source devel/setup.bash
```

If only rebuilding the exploration node:

```bash
make -C build -j1 exploration_node
```

Run Python tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 3. Python / ROS Dependencies

The original environment was not fully reconstructed from a lockfile. The important dependency groups are:

- Habitat-Lab / Habitat-Sim compatible with ObjectNav HM3D/MP3D configs.
- ROS Noetic Python packages: `rospy`, `std_msgs`, `geometry_msgs`, etc.
- PyTorch stack for detectors and VLM support.
- OpenCV, NumPy, SciPy, scikit-image, PIL/Pillow.
- Hydra/OmegaConf for Habitat config composition.
- OpenAI-compatible client dependency for VLM calls.
- Optional vLLM + Qwen2.5-VL dependencies for local VLM serving.

Repository files/scripts to inspect first:

- `apexnav_environment.yaml`
- `GroundingDINO/requirements.txt`
- `yolov7/requirements.txt`
- `repro/scripts/00_probe_env.sh`
- `repro/scripts/setup_qwen_vl7b_vllm_env.sh`

## 4. Dataset Setup

Datasets are not included in this repository.

### HM3D Scene Dataset

You need official Habitat Matterport 3D Research Dataset access.
After downloading HM3D-v0.2 validation scenes, arrange them like this:

```bash
mkdir -p data/scene_datasets/hm3d/val
# Put/extract hm3d-val-habitat-v0.2 content under data/scene_datasets/hm3d/val
cd data/scene_datasets
ln -s hm3d hm3d_v0.2
```

Expected scene path example:

```text
data/scene_datasets/hm3d/val/<scene_id>/<scene_id>.basis.glb
```

### MP3D Scene Dataset

You need Matterport3D access.
Place scenes under:

```text
data/scene_datasets/mp3d/
```

### ObjectNav Episode JSON Files

HM3D-v1:

```bash
mkdir -p data/datasets/objectnav/hm3d
wget -O data/datasets/objectnav/hm3d/v1.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip data/datasets/objectnav/hm3d/v1.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
rm data/datasets/objectnav/hm3d/v1.zip
```

HM3D-v2:

```bash
mkdir -p data/datasets/objectnav/hm3d
wget -O data/datasets/objectnav/hm3d/v2.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/v2.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
rm data/datasets/objectnav/hm3d/v2.zip
```

MP3D ObjectNav:

```bash
mkdir -p data/datasets/objectnav/mp3d
wget -O data/datasets/objectnav/mp3d/v1.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/mp3d/v1/objectnav_mp3d_v1.zip
unzip data/datasets/objectnav/mp3d/v1.zip -d data/datasets/objectnav/mp3d
mv data/datasets/objectnav/mp3d/objectnav_mp3d_v1 data/datasets/objectnav/mp3d/v1 2>/dev/null || true
rm data/datasets/objectnav/mp3d/v1.zip
```

Config paths:

- HM3D-v1: `config/habitat_eval_hm3dv1.yaml`
- HM3D-v2: `config/habitat_eval_hm3dv2.yaml`
- MP3D: `config/habitat_eval_mp3d.yaml`

## 5. Detector / Public Asset Setup

Model weights are not included.
Use the existing helper first:

```bash
bash repro/scripts/01_download_public_assets.sh
```

Start smoke-test detector servers through:

```bash
bash repro/scripts/02_run_smoke_hm3dv2.sh 0
```

That script launches:

- GroundingDINO server on port `12181`
- BLIP2-ITM server on port `12182`
- SAM server on port `12183`
- YOLOv7 server on port `12184`
- ROS exploration launch
- Habitat evaluation for one episode

## 6. Local Qwen2.5-VL Runtime

The default HM3Dv2 config expects an OpenAI-compatible local Qwen-VL endpoint:

```text
http://127.0.0.1:18000/v1
model: qwen2.5-vl-7b-instruct
```

Setup and start scripts:

```bash
bash repro/scripts/setup_qwen_vl7b_vllm_env.sh
bash repro/scripts/start_qwen_vl7b_server.sh
```

Manual equivalent:

```bash
/root/autodl-tmp/envs/qwen_vl/bin/vllm serve \
  /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b-instruct \
  --host 127.0.0.1 \
  --port 18000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

Test the local endpoint:

```bash
python repro/scripts/test_qwen_vl7b_server.py
```

Stop local Qwen server:

```bash
bash repro/scripts/stop_qwen_vl7b_server.sh
```

## 7. External OpenAI-Compatible VLM Runtime

No real API key is committed. Configure secrets through environment variables:

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_MODEL="gpt-5.5"
```

Then set a config block to:

```yaml
reflective_agent:
  vlm_provider: openai
  vlm_model: gpt-5.5
  vlm_api_key: null
  vlm_api_key_env: OPENAI_API_KEY
  vlm_base_url: null
  vlm_base_url_env: OPENAI_BASE_URL
```

## 8. Running Modes

### Original ApexNav Baseline

Set:

```yaml
reflective_agent:
  enable_reflective_agent: false
```

Then run:

```bash
source devel/setup.bash
roslaunch exploration_manager exploration.launch
```

In another shell:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate apexnav
python habitat_evaluation.py --dataset hm3dv2
```

### One-Episode Smoke Test

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate apexnav
VLM_STARTUP_WAIT=180 ROS_STARTUP_WAIT=20 bash repro/scripts/02_run_smoke_hm3dv2.sh 0
```

### Reflective Mock Agent

Set:

```yaml
reflective_agent:
  enable_reflective_agent: true
  vlm_provider: mock
```

Then run smoke/eval as above. This mode does not need an API key or local Qwen server.

### Reflective Local VLM Agent

Start Qwen server first, then use default `config/habitat_eval_hm3dv2.yaml`:

```bash
bash repro/scripts/start_qwen_vl7b_server.sh
bash repro/scripts/02_run_smoke_hm3dv2.sh 0
```

### Small Reflective Evaluation

```bash
bash repro/scripts/run_reflective_small_eval.sh
```

### Benchmark Helper

```bash
bash repro/scripts/03_run_benchmark.sh hm3dv2
```

## 9. Current Agent Decision Framework

The VLM/agent can only select high-level skills, never low-level actions:

- `SEMANTIC_EXPLORE`
- `GEOMETRIC_EXPLORE`
- `VERIFY_TARGET`
- `NAVIGATE_TO_CONFIRMED_TARGET`
- `RECOVER_FROM_STUCK`
- `FOLLOW_APEXNAV_PROPOSAL`
- `FALLBACK_APEXNAV`

Important runtime behavior:

- ApexNav mapping, frontier extraction, semantic scoring, target detection/fusion, and low-level navigation remain ApexNav-owned.
- VLM calls occur at high-level commitment refresh points, not every environment step.
- Current visual input to VLM is RGB only; semantic map image input is disabled.
- Stop is gated by StopValidator. VLM target stop must use a valid target candidate id with sufficient confidence and multiview confirmation.
- GT trajectory learning writes structured feedback/memory when enabled.

## 10. Memory, Logs, and Learning Outputs

Default paths:

```text
logs/reflective_agent/<run_id>/episodes/<episode_id>.json
data/reflection_memory.jsonl
data/policy_patches.json
data/tool_call_learning_samples.jsonl
videos/test_<dataset>_<split>/record.txt
videos/test_<dataset>_<split>/continue.txt
```

Avoid benchmark leakage by disabling writes when needed:

```yaml
reflective_agent:
  enable_reflection_memory: false
  enable_episode_reflection: false
  memory_write_mode: disabled
  learning_write_mode: disabled
```

For the current self-evolution experiments, validation/test memory writing may be intentionally enabled. Record this choice in experiment notes.

## 11. Common Checks Before Running Experiments

```bash
# Python tests
python -m unittest discover -s tests -p "test_*.py"

# C++ target
make -C build -j1 exploration_node

# Qwen endpoint
python repro/scripts/test_qwen_vl7b_server.py

# No stale ROS/eval processes
ps -eo pid,cmd | egrep "habitat_evaluation|exploration_node|grounding_dino|blip2itm|segmentor.sam|detector.yolov7|vllm" | grep -v egrep || true
```

## 12. Known Practical Notes

- This snapshot excludes `build/`, `devel/`, `data/`, `logs/`, `videos/`, model weights, old reflective eval outputs, and wheel caches.
- Rebuild with `catkin_make` after cloning.
- A 1GB cgroup/no-card mode is insufficient for full smoke/eval; it can kill C++ compilation, detector servers, SAM, and Habitat.
- If C++ compilation is killed under tight memory, use a larger-memory/GPU mode. The last patched `exploration_node` compiled successfully on the source server.
- Keep API keys out of YAML and Git history; use environment variables.
