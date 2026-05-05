"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import base64
import gzip
import io
import json
import os
import signal
import sys
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

try:
    import cv2
except Exception:  # pragma: no cover - fallback used only if OpenCV import fails
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - fallback used only if Pillow import fails
    Image = None

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def update_reflective_rgb_observation(cfg, observations, timestep):
    reflective_cfg = cfg.get("reflective_agent") if hasattr(cfg, "get") else None
    if not reflective_cfg or not bool(reflective_cfg.get("enable_reflective_agent", False)):
        return
    if not bool(reflective_cfg.get("enable_rgb_observation", True)):
        rospy.set_param("/reflective_agent/current_rgb_observation", json.dumps({"available": False}))
        return
    rgb = observations.get("rgb") if isinstance(observations, dict) else None
    if rgb is None:
        rospy.set_param("/reflective_agent/current_rgb_observation", json.dumps({"available": False}))
        return
    try:
        data_url, width, height = encode_rgb_observation_for_vlm(
            rgb,
            max_width=int(reflective_cfg.get("rgb_observation_max_width", 320)),
            jpeg_quality=int(reflective_cfg.get("rgb_observation_jpeg_quality", 70)),
        )
        rospy.set_param(
            "/reflective_agent/current_rgb_observation",
            json.dumps(
                {
                    "available": True,
                    "source": "habitat_rgb",
                    "encoding": "jpeg_base64_data_url",
                    "width": width,
                    "height": height,
                    "timestep": int(timestep),
                    "data_url": data_url,
                }
            ),
        )
    except Exception as exc:
        rospy.set_param(
            "/reflective_agent/current_rgb_observation",
            json.dumps({"available": False, "error": f"{type(exc).__name__}: {exc}"}),
        )


def update_reflective_detected_landmarks(cfg, detections, timestep):
    reflective_cfg = cfg.get("reflective_agent") if hasattr(cfg, "get") else None
    if not reflective_cfg or not bool(reflective_cfg.get("enable_reflective_agent", False)):
        return
    if not bool(reflective_cfg.get("include_detected_objects_in_state", True)):
        rospy.set_param("/reflective_agent/current_detected_landmarks", json.dumps({"available": False, "detections": []}))
        return
    max_items = int(reflective_cfg.get("max_landmark_detections", 12))
    detections = list(detections or [])[:max_items]
    rospy.set_param(
        "/reflective_agent/current_detected_landmarks",
        json.dumps(
            {
                "available": bool(detections),
                "source": "yolov7_landmark",
                "timestep": int(timestep),
                "detections": detections,
            }
        ),
    )


def update_reflective_gt_feedback(cfg, env, timestep, gt_trajectory=None):
    reflective_cfg = cfg.get("reflective_agent") if hasattr(cfg, "get") else None
    if not reflective_cfg or not bool(reflective_cfg.get("enable_reflective_agent", False)):
        return
    if not bool(reflective_cfg.get("enable_gt_teacher_learning", False)):
        rospy.set_param("/reflective_agent/current_gt_feedback", json.dumps({"available": False}))
        return
    try:
        info = env.get_metrics()
        payload = {
            "available": True,
            "source": "habitat_metrics",
            "timestep": int(timestep),
            "distance_to_goal": _safe_metric_float(info.get("distance_to_goal")),
            "success": _safe_metric_float(info.get("success")),
            "spl": _safe_metric_float(info.get("spl")),
            "soft_spl": _safe_metric_float(info.get("soft_spl")),
            "agent_position": _get_agent_position(env),
            "gt_trajectory": _gt_trajectory_context(
                gt_trajectory,
                _get_agent_position(env),
                int(reflective_cfg.get("gt_path_reflection_lookahead", 3)),
            ),
        }
        rospy.set_param("/reflective_agent/current_gt_feedback", json.dumps(payload))
    except Exception as exc:
        rospy.set_param(
            "/reflective_agent/current_gt_feedback",
            json.dumps({"available": False, "error": f"{type(exc).__name__}: {exc}"}),
        )


def _safe_metric_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_reflective_gt_trajectory(cfg, env, target_category):
    reflective_cfg = cfg.get("reflective_agent") if hasattr(cfg, "get") else None
    if not reflective_cfg or not bool(reflective_cfg.get("enable_reflective_agent", False)):
        return {"available": False}
    if not bool(reflective_cfg.get("enable_gt_trajectory_learning", False)):
        return {"available": False}
    start = _get_agent_position(env)
    endpoints = _goal_endpoints(getattr(env, "current_episode", None))
    if start is None or not endpoints:
        return {"available": False, "reason": "missing_start_or_goal"}
    max_points = int(reflective_cfg.get("gt_path_max_points", 80))
    best = None
    for endpoint in endpoints:
        path_points, geodesic_distance = _shortest_path_points(env, start, endpoint)
        if not path_points:
            continue
        candidate = {
            "available": True,
            "source": "habitat_shortest_path",
            "target_category": target_category,
            "start_position": _float_list(start),
            "goal_position": _float_list(endpoint),
            "geodesic_distance": geodesic_distance,
            "path_points": _downsample_points(path_points, max_points),
            "raw_path_point_count": len(path_points),
        }
        if best is None or (
            candidate["geodesic_distance"] is not None
            and candidate["geodesic_distance"] < (best.get("geodesic_distance") or float("inf"))
        ):
            best = candidate
    if best is None:
        return {
            "available": False,
            "reason": "shortest_path_unavailable",
            "start_position": _float_list(start),
            "candidate_goal_count": len(endpoints),
        }
    return best


def _shortest_path_points(env, start, end):
    try:
        import habitat_sim

        pathfinder = getattr(getattr(env, "sim", None), "pathfinder", None)
        if pathfinder is None:
            return [], None
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if not pathfinder.find_path(path):
            return [], None
        return [_float_list(point) for point in path.points], _safe_metric_float(path.geodesic_distance)
    except Exception:
        return [], None


def _gt_trajectory_context(gt_trajectory, agent_position, lookahead):
    if not gt_trajectory or not gt_trajectory.get("available") or agent_position is None:
        return {"gt_path_available": False}
    points = gt_trajectory.get("path_points") or []
    if not points:
        return {"gt_path_available": False}
    nearest_idx, distance = _nearest_path_index_and_distance(points, agent_position)
    next_idx = min(len(points) - 1, nearest_idx + max(1, int(lookahead)))
    progress_ratio = 0.0 if len(points) <= 1 else float(nearest_idx) / float(len(points) - 1)
    return {
        "gt_path_available": True,
        "source": gt_trajectory.get("source"),
        "agent_position": _float_list(agent_position),
        "goal_position": gt_trajectory.get("goal_position"),
        "gt_path_point_count": len(points),
        "gt_path_geodesic_distance": gt_trajectory.get("geodesic_distance"),
        "nearest_gt_path_index": nearest_idx,
        "distance_to_gt_path": distance,
        "gt_path_progress_ratio": progress_ratio,
        "gt_next_waypoint": points[next_idx],
        "gt_local_path_window": points[max(0, nearest_idx - 2) : min(len(points), nearest_idx + 5)],
    }


def _nearest_path_index_and_distance(points, position):
    pos = np.asarray(position, dtype=np.float32)
    best_idx = 0
    best_dist = float("inf")
    for idx, point in enumerate(points):
        try:
            dist = float(np.linalg.norm(np.asarray(point, dtype=np.float32) - pos))
        except Exception:
            continue
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx, None if best_dist == float("inf") else best_dist


def _goal_endpoints(episode):
    endpoints = []
    for goal in getattr(episode, "goals", []) or []:
        position = getattr(goal, "position", None)
        if position is not None:
            endpoints.append(_float_list(position))
        for view_point in getattr(goal, "view_points", []) or []:
            agent_state = getattr(view_point, "agent_state", None)
            vp_position = getattr(agent_state, "position", None)
            if vp_position is not None:
                endpoints.append(_float_list(vp_position))
    unique = []
    seen = set()
    for endpoint in endpoints:
        if endpoint is None:
            continue
        key = tuple(round(value, 3) for value in endpoint)
        if key not in seen:
            seen.add(key)
            unique.append(endpoint)
    return unique


def _get_agent_position(env):
    try:
        state = env.sim.get_agent_state()
        return _float_list(state.position)
    except Exception:
        return None


def _downsample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[int(index)] for index in indices]


def _float_list(values):
    try:
        return [float(value) for value in list(values)]
    except Exception:
        return None


def encode_rgb_observation_for_vlm(rgb, max_width=320, jpeg_quality=70):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb observation must be HxWx3")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    if max_width > 0 and width > max_width:
        scale = float(max_width) / float(width)
        new_size = (max_width, max(1, int(round(height * scale))))
        if cv2 is not None:
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        elif Image is not None:
            image = np.asarray(Image.fromarray(image).resize(new_size))
        height, width = image.shape[:2]
    if cv2 is not None:
        ok, encoded = cv2.imencode(
            ".jpg",
            image[:, :, ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        raw = encoded.tobytes()
    elif Image is not None:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=int(jpeg_quality))
        raw = buf.getvalue()
    else:
        raise RuntimeError("neither cv2 nor Pillow is available for RGB encoding")
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    return data_url, width, height


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown



def finalize_reflective_episode(
    cfg,
    episode,
    label,
    success,
    spl,
    soft_spl,
    count_steps,
    distance_to_goal,
    result_text,
    final_state,
    expl_result,
    pass_object,
    near_object,
):
    reflective_cfg = cfg.get('reflective_agent') if hasattr(cfg, 'get') else None
    if not reflective_cfg or not bool(reflective_cfg.get('enable_reflective_agent', False)):
        return
    if not bool(reflective_cfg.get('enable_episode_reflection', True)):
        return
    try:
        cfg_dict = OmegaConf.to_container(reflective_cfg, resolve=True)
        project_root = cfg_dict.get('project_root') or os.path.join(os.getcwd(), 'agent-Apexnav')
        if project_root and project_root not in sys.path:
            sys.path.insert(0, project_root)
        from agent.logging.episode_logger import EpisodeLogger
        from agent.reflection.reflection_engine import ReflectionEngine
        from agent.schemas import AgentConfig

        agent_cfg = AgentConfig.from_mapping(cfg_dict)
        for attr in [
            'memory_path',
            'policy_patch_path',
            'episode_log_root',
            'tool_call_dataset_path',
            'baseline_teacher_log_path',
            'gt_trajectory_path',
        ]:
            path_value = getattr(agent_cfg, attr, None)
            if path_value and not os.path.isabs(path_value):
                setattr(agent_cfg, attr, os.path.join(project_root, path_value))
        engine = ReflectionEngine(agent_cfg)
        episode_summary = {
            'episode_id': getattr(episode, 'episode_id', None),
            'scene_id': getattr(episode, 'scene_id', None),
            'split': cfg.habitat.dataset.split,
            'target_category': label,
            'success': bool(success),
            'failure_type': None if success == 1 else result_text,
            'failure_signals': [] if success == 1 else [result_text],
            'spl': float(spl),
            'softspl': float(soft_spl),
            'steps': int(count_steps),
            'final_distance_to_goal': float(distance_to_goal),
            'stop_reason': result_text,
            'final_state': int(final_state),
            'expl_result': int(expl_result),
            'pass_object': int(pass_object),
            'near_object': int(near_object),
            'stop_action_source': rospy.get_param('/reflective_agent/last_stop_action_source', 'unknown'),
            'stop_validator': {
                'stop_action_source': rospy.get_param('/reflective_agent/last_stop_action_source', 'unknown'),
                'target_candidate_id': rospy.get_param('/reflective_agent/last_stop_target_candidate_id', None),
                'target_confidence': rospy.get_param('/reflective_agent/last_stop_target_confidence', None),
                'multi_view_confirmed': rospy.get_param('/reflective_agent/last_stop_multi_view_confirmed', None),
                'target_reachable': rospy.get_param('/reflective_agent/last_stop_target_reachable', None),
                'apexnav_stop_proposed': rospy.get_param('/reflective_agent/last_stop_apexnav_stop_proposed', None),
                'vlm_stop_proposed': rospy.get_param('/reflective_agent/last_stop_vlm_stop_proposed', None),
                'validator_allowed_stop': rospy.get_param('/reflective_agent/last_stop_validator_allowed_stop', None),
                'final_stop_allowed_by_gate': rospy.get_param('/reflective_agent/last_stop_final_stop_allowed_by_gate', None),
                'invalid_active_stop': rospy.get_param('/reflective_agent/last_stop_invalid_active_stop', None),
            },
        }
        reflection = engine.finalize_episode(episode_summary)
        if getattr(agent_cfg, 'enable_learning_from_traces', False):
            from agent.learning.trace_learning_manager import TraceLearningManager

            learning = TraceLearningManager(agent_cfg).finalize_episode(episode_summary)
            reflection['learning'] = learning
        if agent_cfg.enable_episode_logger:
            EpisodeLogger(agent_cfg.episode_log_root, agent_cfg.run_id).log_episode_end(
                episode_summary.get('episode_id'), episode_summary, reflection
            )
        print(
            '[ReflectiveAgent] episode reflection written='
            f'{reflection.get("memory_written", False)} patches='
            f'{len(reflection.get("recorded_policy_patches", []))}'
            f' learning_samples='
            f'{reflection.get("learning", {}).get("dataset_written", 0)}'
        )
    except Exception as exc:
        print(f'[ReflectiveAgent] episode reflection skipped: {type(exc).__name__}: {exc}')

def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Optional controls for parallel full-evaluation shards. Use Hydra overrides
    # such as +episode_start=0 +episode_stride=4 +episode_limit=250.
    episode_start = int(cfg.get("episode_start", 0))
    episode_stride = int(cfg.get("episode_stride", 1))
    episode_limit = int(cfg.get("episode_limit", -1))
    shard_mode = (not flag_once) and (episode_start != 0 or episode_stride != 1 or episode_limit != -1)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    if shard_mode:
        if episode_stride < 1:
            raise ValueError("episode_stride must be >= 1")
        if episode_start < 0 or episode_start >= number_of_episodes:
            raise ValueError("episode_start is out of range")
        env_count = episode_start + num_total * episode_stride
        shard_remaining = (number_of_episodes - episode_start + episode_stride - 1) // episode_stride - num_total
        if episode_limit != -1:
            shard_remaining = min(shard_remaining, max(0, episode_limit - num_total))
    else:
        env_count = num_total if not flag_once else env_num_once
        shard_remaining = number_of_episodes - num_total

    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(shard_remaining):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        rospy.set_param("/reflective_agent/current_target_category", str(label))
        rospy.set_param("/reflective_agent/current_episode_id", str(getattr(env.current_episode, "episode_id", "")))
        rospy.set_param("/reflective_agent/current_scene_id", str(getattr(env.current_episode, "scene_id", "")))
        rospy.set_param("/reflective_agent/current_split", str(cfg.habitat.dataset.split))
        gt_trajectory = build_reflective_gt_trajectory(cfg, env, label)
        rospy.set_param("/reflective_agent/current_gt_trajectory", json.dumps(gt_trajectory))
        update_reflective_rgb_observation(cfg, observations, count_steps)
        update_reflective_detected_landmarks(cfg, [], count_steps)
        update_reflective_gt_feedback(cfg, env, count_steps, gt_trajectory)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)
            raw_rgb_observation = observations["rgb"].copy()

            # Calculate ITM cosine similarity score
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            publish_float64(itm_score_pub, cosine)

            # Detect objects in the current observation
            (
                observations["rgb"],
                score_list,
                object_masks_list,
                label_list,
                landmark_detections,
            ) = get_object(
                label,
                observations["rgb"],
                detector_cfg,
                llm_answer,
                return_metadata=True,
            )
            update_reflective_rgb_observation(
                cfg,
                {"rgb": raw_rgb_observation},
                count_steps,
            )
            update_reflective_detected_landmarks(cfg, landmark_detections, count_steps)
            update_reflective_gt_feedback(cfg, env, count_steps, gt_trajectory)

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )


        finalize_reflective_episode(
            cfg=cfg,
            episode=env.current_episode,
            label=label,
            success=success,
            spl=spl,
            soft_spl=soft_spl,
            count_steps=count_steps,
            distance_to_goal=distance_to_goal,
            result_text=result_text,
            final_state=final_state,
            expl_result=expl_result,
            pass_object=pass_object,
            near_object=near_object,
        )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        if epi != shard_remaining - 1:
            advance_steps = episode_stride if shard_mode else 1
            for _ in range(advance_steps):
                env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
