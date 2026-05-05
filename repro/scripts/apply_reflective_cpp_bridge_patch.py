#!/usr/bin/env python3
"""Apply the C++ reflective-agent bridge patch in-place.

This script is intentionally pattern-based because the reproduction workspace
already contains local modifications. It only inserts the bridge hook if the
target snippets are not present.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def ensure_contains(path: Path, needle: str, inserter) -> None:
    text = read(path)
    if needle in text:
        return
    write(path, inserter(text))


def patch_cmake() -> None:
    path = ROOT / "src/planner/exploration_manager/CMakeLists.txt"

    def insert(text: str) -> str:
        anchor = "src/exploration_manager.cpp\n"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing source anchor")
        return text.replace(anchor, anchor + "  src/reflective_agent_bridge.cpp\n", 1)

    ensure_contains(path, "src/reflective_agent_bridge.cpp", insert)


def patch_manager_header() -> None:
    path = ROOT / "src/planner/exploration_manager/include/exploration_manager/exploration_manager.h"

    def insert_include(text: str) -> str:
        if "#include <string>" in text:
            return text
        return text.replace("#include <vector>", "#include <vector>\n#include <string>", 1)

    ensure_contains(path, "#include <string>", insert_include)

    def insert_decl(text: str) -> str:
        anchor = "int planNextBestPoint(const Vector3d& pos, const double& yaw);"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing planNextBestPoint declaration")
        return text.replace(
            anchor,
            anchor
            + "\n  int planNextBestPointWithSkill(\n"
            + "      const std::string& selected_skill, const Vector3d& pos, const double& yaw);",
            1,
        )

    ensure_contains(path, "planNextBestPointWithSkill", insert_decl)


MANAGER_METHOD = r'''
int ExplorationManager::planNextBestPointWithSkill(
    const std::string& selected_skill, const Vector3d& pos, const double& yaw)
{
  if (selected_skill.empty() || selected_skill == "FALLBACK_APEXNAV" ||
      selected_skill == "VERIFY_TARGET" ||
      selected_skill == "NAVIGATE_TO_CONFIRMED_TARGET") {
    return planNextBestPoint(pos, yaw);
  }

  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();

  // Preserve ApexNav target-first behavior even when the reflective selector
  // requests an exploration skill. The agent layer schedules high-level skills;
  // target fusion and final target navigation remain owned by ApexNav.
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);
  if (!object_clouds.empty()) {
    ROS_WARN("[Reflective Skill] Target candidate exists; keep ApexNav target-first navigation");
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
        return SEARCH_BEST_OBJECT;
    }
  }
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  if (selected_skill == "GEOMETRIC_EXPLORE" || selected_skill == "RECOVER_FROM_STUCK") {
    ROS_WARN("[Reflective Skill] GEOMETRIC_EXPLORE via ApexNav closest frontier");
    findClosestFrontierPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);
  }
  else if (selected_skill == "SEMANTIC_EXPLORE") {
    ROS_WARN("[Reflective Skill] SEMANTIC_EXPLORE via ApexNav semantic frontier");
    findHighestSemanticsFrontierPolicy(
        pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);
  }
  else {
    ROS_WARN("[Reflective Skill] Unknown skill %s, fallback to ApexNav", selected_skill.c_str());
    return planNextBestPoint(pos, yaw);
  }

  if (next_best_path.empty()) {
    ROS_WARN("[Reflective Skill] Requested skill produced no path, fallback to ApexNav");
    return planNextBestPoint(pos, yaw);
  }

  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;
  return EXPLORATION;
}

'''


def patch_manager_cpp() -> None:
    path = ROOT / "src/planner/exploration_manager/src/exploration_manager.cpp"

    def insert(text: str) -> str:
        anchor = "void ExplorationManager::chooseExplorationPolicy"
        idx = text.find(anchor)
        if idx == -1:
            raise RuntimeError(f"Cannot patch {path}: missing chooseExplorationPolicy anchor")
        return text[:idx] + MANAGER_METHOD + text[idx:]

    ensure_contains(path, "planNextBestPointWithSkill", insert)


FSM_INCLUDE = '#include <exploration_manager/reflective_agent_bridge.h>\n'

FSM_STATIC = r'''
namespace {
std::unique_ptr<apexnav_planner::ReflectiveAgentBridge> g_reflective_agent_bridge;
int g_reflective_decision_count = 0;
std::vector<std::string> g_recent_reflective_skills;
}  // namespace

'''

FSM_STATE_BLOCK = r'''
  std::string selected_skill = "FALLBACK_APEXNAV";
  if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()) {
    ReflectiveAgentState reflective_state;
    reflective_state.split = "unknown";
    reflective_state.timestep = g_reflective_decision_count++;
    reflective_state.stuck_count = fd_->stucking_action_count_;
    reflective_state.recent_selected_skills = g_recent_reflective_skills;
    for (int i = 0; i < (int)expl_manager_->ed_->frontier_averages_.size(); ++i) {
      ReflectiveFrontierCandidate f;
      f.id = i;
      f.waypoint = expl_manager_->ed_->frontier_averages_[i];
      f.distance = (f.waypoint - current_pos).norm();
      f.reachable = true;
      Eigen::Vector2i f_idx;
      expl_manager_->sdf_map_->posToIndex(f.waypoint, f_idx);
      f.semantic_score = expl_manager_->sdf_map_->value_map_->getValue(f_idx);
      f.last_selected = (f.waypoint - expl_manager_->ed_->next_pos_).norm() < 1e-3;
      reflective_state.frontiers.push_back(f);
    }
    auto decision = g_reflective_agent_bridge->selectSkill(reflective_state);
    if (decision.ok) {
      selected_skill = decision.selected_skill;
      g_recent_reflective_skills.push_back(selected_skill);
      if (g_recent_reflective_skills.size() > 20)
        g_recent_reflective_skills.erase(g_recent_reflective_skills.begin());
      ROS_WARN("[ReflectiveAgentBridge] selected_skill=%s fallback=%d reason=%s",
          selected_skill.c_str(), decision.fallback_used, decision.rejection_reason.c_str());
    }
    else {
      ROS_WARN("[ReflectiveAgentBridge] fallback to ApexNav: %s",
          decision.rejection_reason.c_str());
    }
  }

'''


def patch_fsm_cpp() -> None:
    path = ROOT / "src/planner/exploration_manager/src/exploration_fsm.cpp"

    def insert_include(text: str) -> str:
        anchor = "#include <exploration_manager/exploration_data.h>\n"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing include anchor")
        return text.replace(anchor, anchor + FSM_INCLUDE, 1)

    ensure_contains(path, "reflective_agent_bridge.h", insert_include)

    def insert_static(text: str) -> str:
        anchor = "namespace apexnav_planner {\n"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing namespace anchor")
        return text.replace(anchor, FSM_STATIC + anchor, 1)

    ensure_contains(path, "g_reflective_agent_bridge", insert_static)

    def insert_init(text: str) -> str:
        anchor = "visualization_.reset(new PlanningVisualization(nh));"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing init anchor")
        return text.replace(
            anchor,
            anchor
            + "\n  g_reflective_agent_bridge.reset(new ReflectiveAgentBridge);\n"
            + "  g_reflective_agent_bridge->init(nh);",
            1,
        )

    ensure_contains(path, "g_reflective_agent_bridge->init", insert_init)

    def insert_state_block(text: str) -> str:
        anchor = "  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));\n"
        if anchor not in text:
            raise RuntimeError(f"Cannot patch {path}: missing planNextBestPoint call")
        return text.replace(
            anchor,
            FSM_STATE_BLOCK
            + "  if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled())\n"
            + "    expl_res = expl_manager_->planNextBestPointWithSkill(\n"
            + "        selected_skill, fd_->start_pt_, fd_->start_yaw_(0));\n"
            + "  else\n"
            + anchor,
            1,
        )

    ensure_contains(path, "planNextBestPointWithSkill", insert_state_block)


def main() -> int:
    patch_cmake()
    patch_manager_header()
    patch_manager_cpp()
    patch_fsm_cpp()
    print("Reflective C++ bridge patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
