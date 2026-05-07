# Reflective ApexNav Reproduction Guide

This repository contains the reproduced ApexNav code plus an incremental reflective navigation agent. The repository intentionally excludes datasets, model weights, ROS build outputs, logs, videos, API keys, and permissioned dataset tokens.

## 1. Clone

```bash
git clone https://github.com/Royal-mendes/agent.git
cd agent
```

Recommended runtime: Ubuntu 20.04, ROS Noetic, Python 3.9, CUDA-capable GPU. The source server used a conda environment named `apexnav`.

## 2. System Dependencies

Install ROS Noetic first, then install the native dependencies used by ApexNav:

```bash
sudo apt update
sudo apt install -y libarmadillo-dev libompl-dev python3-catkin-tools
```

Build OSQP and OSQP-Eigen as described in the upstream ApexNav `README.md`.

## 3. Python Environment

```bash
conda env create -f apexnav_environment.yaml -y
conda activate apexnav
pip install -e .
pip install openai
```

Install Habitat-Lab / Habitat-Baselines v0.3.1:

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/v0.3.1
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

Detector dependencies:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -r yolov7/requirements.txt
pip install -r GroundingDINO/requirements.txt
pip install salesforce-lavis==1.0.2
```

## 4. Data and Weights

Create the expected data layout:

```text
data/
  scene_datasets/
    hm3d/
      val/
    hm3d_v0.2 -> hm3d
  datasets/
    objectnav/
      hm3d/
        v2/
  groundingdino_swint_ogc.pth
  mobile_sam.pt
  yolov7-e6e.pt
```

HM3D scenes are permissioned. Download HM3D validation scenes through the official Habitat Matterport 3D Research Dataset access flow, then place the extracted `hm3d-val-habitat-v0.2` scenes under:

```bash
mkdir -p data/scene_datasets/hm3d/val
ln -s hm3d data/scene_datasets/hm3d_v0.2
```

ObjectNav HM3D-v2 episodes are public:

```bash
mkdir -p data/datasets/objectnav/hm3d
wget -O data/datasets/objectnav/hm3d/v2.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/v2.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
rm data/datasets/objectnav/hm3d/v2.zip
```

Download public model weights:

```bash
bash repro/scripts/01_download_public_assets.sh
```

## 5. Build

```bash
source /opt/ros/noetic/setup.bash
conda activate apexnav
export APEXNAV_PROJECT_ROOT="$PWD"
export APEXNAV_PYTHON="$(which python)"
catkin_make -DPYTHON_EXECUTABLE="$(which python)"
source devel/setup.bash
```

If memory is tight, rebuild only the exploration node:

```bash
make -C build -j1 exploration_node
```

## 6. VLM Configuration

No API key is committed. Use environment variables.

For an external OpenAI-compatible API:

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_MODEL="<your-model-id>"
```

For local Qwen2.5-VL:

```bash
bash repro/scripts/setup_qwen_vl7b_vllm_env.sh
bash repro/scripts/start_qwen_vl7b_server.sh
export LOCAL_VLM_BASE_URL="http://127.0.0.1:18000/v1"
export LOCAL_VLM_MODEL="qwen2.5-vl-7b-instruct"
```

The current HM3Dv2 config uses `vlm_provider: openai` and resolves model/base URL/key from environment variables. To run without any real VLM, set `reflective_agent.vlm_provider: mock`.

## 7. Run Baseline ApexNav

Set in `config/habitat_eval_hm3dv2.yaml`:

```yaml
reflective_agent:
  enable_reflective_agent: false
```

Then run:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch exploration_manager exploration.launch
```

In another terminal:

```bash
conda activate apexnav
python habitat_evaluation.py --dataset hm3dv2
```

## 8. Run Reflective Agent

Set in `config/habitat_eval_hm3dv2.yaml`:

```yaml
reflective_agent:
  enable_reflective_agent: true
  vlm_provider: openai
  vlm_api_key: ''
  vlm_base_url: ''
  vlm_api_key_env: OPENAI_API_KEY
  vlm_base_url_env: OPENAI_BASE_URL
  vlm_model_env: OPENAI_MODEL
```

Start the same ROS and Habitat commands as above. The VLM only selects high-level skills; ApexNav still owns mapping, target fusion, frontier extraction, path search, and low-level navigation.

Before launching ROS, keep these variables in the shell:

```bash
export APEXNAV_PROJECT_ROOT="$PWD"
export APEXNAV_PYTHON="$(which python)"
```

Current enabled high-level skills:

- `SEMANTIC_EXPLORE`
- `GEOMETRIC_EXPLORE`
- `NAVIGATE_TO_CONFIRMED_TARGET`
- `RECOVER_FROM_STUCK`
- `RETURN_TO_BEST_KNOWN_POINT`
- `FOLLOW_APEXNAV_PROPOSAL`
- `FALLBACK_APEXNAV`

`VERIFY_TARGET` is disabled in this snapshot.

## 9. Learning and Logs

Experience memory:

```text
data/reflection_memory.jsonl
```

Episode logs:

```text
logs/reflective_agent/<run_id>/episodes/<episode_id>.json
```

To avoid benchmark leakage:

```yaml
reflective_agent:
  enable_reflection_memory: false
  enable_episode_reflection: false
  memory_write_mode: disabled
  learning_write_mode: disabled
```

For the current self-evolution validation experiments, memory writing can be intentionally enabled on validation episodes. Record that choice in the experiment notes.

## 10. Quick Checks

```bash
python -m unittest discover -s tests -p "test_*.py"
python repro/scripts/test_qwen_vl7b_server.py
```

One-episode smoke test:

```bash
VLM_STARTUP_WAIT=180 ROS_STARTUP_WAIT=20 bash repro/scripts/02_run_smoke_hm3dv2.sh 0
```

See `README_RUNTIME_CHECKLIST.md` for the longer runtime checklist and known caveats.
