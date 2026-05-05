#ifndef _REFLECTIVE_AGENT_BRIDGE_H_
#define _REFLECTIVE_AGENT_BRIDGE_H_

#include <Eigen/Eigen>
#include <ros/ros.h>

#include <string>
#include <vector>

namespace apexnav_planner {

struct ReflectiveFrontierCandidate {
  int id = -1;
  double semantic_score = 0.0;
  double distance = 0.0;
  bool reachable = true;
  bool visited = false;
  bool blocked = false;
  bool low_value = false;
  bool last_selected = false;
  int failure_count = 0;
  Eigen::Vector2d waypoint = Eigen::Vector2d::Zero();
};

struct ReflectiveTargetCandidate {
  int id = -1;
  int label = -1;
  double confidence = 0.0;
  double distance = 0.0;
  bool reachable = true;
  bool multi_view_confirmed = false;
  int num_views = 1;
  bool rejected_false_positive = false;
  Eigen::Vector2d waypoint = Eigen::Vector2d::Zero();
};

struct ReflectiveAgentState {
  std::string episode_id;
  std::string scene_id;
  std::string split = "unknown";
  std::string target_category;
  int timestep = 0;
  int stuck_count = 0;
  int collision_count = 0;
  int steps_left = -1;
  std::string trigger_reasons;
  std::string previous_committed_skill;
  int commitment_age_steps = 0;
  int frontier_count_before = -1;
  int frontier_count_after = -1;
  int target_count_before = -1;
  int target_count_after = -1;
  bool stuck_signal = false;
  bool target_found = false;
  bool committed_target_reached = false;
  bool structural_frontier_change = false;
  bool stable_target_event = false;
  std::string rgb_observation_json;
  std::string semantic_map_observation_json;
  std::string detected_objects_json;
  std::string gt_feedback_json;
  std::vector<std::string> recent_failures;
  std::vector<std::string> recent_selected_skills;
  std::vector<ReflectiveFrontierCandidate> frontiers;
  std::vector<ReflectiveTargetCandidate> target_candidates;
};

struct ReflectiveAgentDecision {
  bool ok = false;
  std::string selected_skill = "FALLBACK_APEXNAV";
  std::string rejection_reason;
  bool fallback_used = true;
  bool accepted = false;
  int target_candidate_id = -1;
  bool has_target_candidate_id = false;
  std::string raw_json;
};

class ReflectiveAgentBridge {
public:
  ReflectiveAgentBridge() = default;
  ~ReflectiveAgentBridge() = default;

  void init(ros::NodeHandle& nh);
  bool enabled() const { return enable_reflective_agent_; }
  ReflectiveAgentDecision selectSkill(const ReflectiveAgentState& state) const;

private:
  bool getBoolParam(ros::NodeHandle& nh, const std::string& key, bool default_value) const;
  int getIntParam(ros::NodeHandle& nh, const std::string& key, int default_value) const;
  double getDoubleParam(ros::NodeHandle& nh, const std::string& key, double default_value) const;
  std::string getStringParam(
      ros::NodeHandle& nh, const std::string& key, const std::string& default_value) const;

  std::string buildPayloadJson(const ReflectiveAgentState& state) const;
  std::string buildConfigJson() const;
  std::string buildStateJson(const ReflectiveAgentState& state) const;
  static std::string jsonEscape(const std::string& value);
  static std::string shellQuote(const std::string& value);
  static bool writeTextFile(const std::string& path, const std::string& text);
  static bool readTextFile(const std::string& path, std::string& text);
  static std::string makeTempPath(const std::string& suffix);
  static std::string extractJsonString(const std::string& json, const std::string& key);
  static bool extractJsonBool(const std::string& json, const std::string& key, bool default_value);
  static bool extractJsonInt(const std::string& json, const std::string& key, int& value);

  bool enable_reflective_agent_ = false;
  bool enable_reflection_memory_ = true;
  bool enable_decision_validator_ = true;
  bool require_multiview_before_stop_ = true;
  bool disable_recover_from_stuck_ = false;
  bool enable_stuck_recovery_override_ = false;
  bool force_all_decisions_to_fallback_apexnav_ = false;
  bool mock_follow_apexnav_by_default_ = true;
  bool include_detected_objects_in_state_ = true;
  std::string vlm_provider_ = "mock";
  std::string vlm_model_ = "gpt-5.5";
  std::string vlm_api_key_;
  std::string vlm_base_url_ = "https://ai.happyclaw.pro/v1";
  std::string memory_path_ = "data/reflection_memory.jsonl";
  std::string memory_read_mode_ = "enabled";
  std::string memory_write_mode_ = "all";
  std::string episode_log_root_ = "logs/reflective_agent";
  std::string run_id_;
  std::string python_executable_ = "/root/miniconda3/envs/apexnav/bin/python";
  std::string project_root_;
  int max_retrieved_lessons_ = 5;
  int max_reflection_memory_items_ = 10000;
  int max_vlm_calls_per_episode_ = 100;
  int min_commitment_steps_for_explore_ = 10;
  int max_commitment_steps_semantic_ = 80;
  int max_commitment_steps_geometric_ = 80;
  int max_commitment_steps_fallback_ = 80;
  int max_commitment_steps_recovery_ = 40;
  int structural_frontier_stable_k_ = 3;
  int stable_target_event_k_ = 2;
  int stuck_threshold_ = 3;
  int same_frontier_failure_threshold_ = 2;
  double structural_frontier_count_change_ratio_ = 0.3;
  double target_verify_threshold_ = 0.65;
  double target_stop_threshold_ = 0.75;
  double semantic_peak_ratio_threshold_ = 2.5;
  double semantic_peak_std_threshold_ = 0.15;
  double low_information_gain_threshold_ = 0.01;
  double vlm_timeout_seconds_ = 30.0;
  double vlm_temperature_ = 0.0;
};

}  // namespace apexnav_planner

#endif
