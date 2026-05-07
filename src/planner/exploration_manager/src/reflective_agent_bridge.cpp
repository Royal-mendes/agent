#include <exploration_manager/reflective_agent_bridge.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <regex>
#include <sstream>

namespace apexnav_planner {

void ReflectiveAgentBridge::init(ros::NodeHandle& nh)
{
  enable_reflective_agent_ = getBoolParam(nh, "enable_reflective_agent", false);
  enable_reflection_memory_ = getBoolParam(nh, "enable_reflection_memory", true);
  enable_decision_validator_ = getBoolParam(nh, "enable_decision_validator", true);
  require_multiview_before_stop_ = getBoolParam(nh, "require_multiview_before_stop", true);
  disable_recover_from_stuck_ = getBoolParam(nh, "disable_recover_from_stuck", false);
  enable_stuck_recovery_override_ = getBoolParam(nh, "enable_stuck_recovery_override", false);
  force_all_decisions_to_fallback_apexnav_ =
      getBoolParam(nh, "force_all_decisions_to_FALLBACK_APEXNAV", false);
  mock_follow_apexnav_by_default_ = getBoolParam(nh, "mock_follow_apexnav_by_default", true);
  include_detected_objects_in_state_ = getBoolParam(nh, "include_detected_objects_in_state", true);

  vlm_provider_ = getStringParam(nh, "vlm_provider", "mock");
  vlm_model_ = getStringParam(nh, "vlm_model", "gpt-5.5");
  vlm_api_key_ = getStringParam(nh, "vlm_api_key", "");
  vlm_base_url_ = getStringParam(nh, "vlm_base_url", "");
  memory_path_ = getStringParam(nh, "memory_path", "data/reflection_memory.jsonl");
  memory_read_mode_ = getStringParam(nh, "memory_read_mode", "enabled");
  memory_write_mode_ = getStringParam(nh, "memory_write_mode", "all");
  episode_log_root_ = getStringParam(nh, "episode_log_root", "logs/reflective_agent");
  run_id_ = getStringParam(nh, "run_id", "");
  const char* env_python = std::getenv("APEXNAV_PYTHON");
  python_executable_ = getStringParam(
      nh, "python_executable", env_python && env_python[0] ? env_python : "python");
  project_root_ = getStringParam(nh, "project_root", "");
  if (project_root_.empty()) {
    const char* env_project_root = std::getenv("APEXNAV_PROJECT_ROOT");
    if (env_project_root && env_project_root[0]) {
      project_root_ = env_project_root;
    }
    else {
      char cwd[4096];
      if (getcwd(cwd, sizeof(cwd)) != nullptr)
        project_root_ = cwd;
      else
        project_root_ = ".";
    }
  }

  max_retrieved_lessons_ = getIntParam(nh, "max_retrieved_lessons", 5);
  max_reflection_memory_items_ = getIntParam(nh, "max_reflection_memory_items", 10000);
  max_vlm_calls_per_episode_ = getIntParam(nh, "max_vlm_calls_per_episode", 100);
  min_commitment_steps_for_explore_ = getIntParam(nh, "min_commitment_steps_for_explore", 10);
  max_commitment_steps_semantic_ = getIntParam(nh, "max_commitment_steps_semantic", 80);
  max_commitment_steps_geometric_ = getIntParam(nh, "max_commitment_steps_geometric", 80);
  max_commitment_steps_fallback_ = getIntParam(nh, "max_commitment_steps_fallback", 80);
  max_commitment_steps_recovery_ = getIntParam(nh, "max_commitment_steps_recovery", 40);
  structural_frontier_stable_k_ = getIntParam(nh, "structural_frontier_stable_k", 3);
  stable_target_event_k_ = getIntParam(nh, "stable_target_event_k", 2);
  stuck_threshold_ = getIntParam(nh, "stuck_threshold", 3);
  same_frontier_failure_threshold_ = getIntParam(nh, "same_frontier_failure_threshold", 2);

  structural_frontier_count_change_ratio_ =
      getDoubleParam(nh, "structural_frontier_count_change_ratio", 0.3);
  target_verify_threshold_ = getDoubleParam(nh, "target_verify_threshold", 0.65);
  target_stop_threshold_ = getDoubleParam(nh, "target_stop_threshold", 0.75);
  semantic_peak_ratio_threshold_ = getDoubleParam(nh, "semantic_peak_ratio_threshold", 2.5);
  semantic_peak_std_threshold_ = getDoubleParam(nh, "semantic_peak_std_threshold", 0.15);
  low_information_gain_threshold_ = getDoubleParam(nh, "low_information_gain_threshold", 0.01);
  vlm_timeout_seconds_ = getDoubleParam(nh, "vlm_timeout_seconds", 30.0);
  vlm_temperature_ = getDoubleParam(nh, "vlm_temperature", 0.0);

  ROS_WARN_COND(enable_reflective_agent_,
      "[ReflectiveAgentBridge] enabled provider=%s model=%s root=%s",
      vlm_provider_.c_str(), vlm_model_.c_str(), project_root_.c_str());
}

ReflectiveAgentDecision ReflectiveAgentBridge::selectSkill(const ReflectiveAgentState& state) const
{
  ReflectiveAgentDecision decision;
  if (!enable_reflective_agent_) {
    decision.rejection_reason = "reflective agent disabled";
    return decision;
  }

  const std::string input_path = makeTempPath("input.json");
  const std::string output_path = makeTempPath("output.json");
  if (!writeTextFile(input_path, buildPayloadJson(state))) {
    decision.rejection_reason = "failed to write bridge input";
    return decision;
  }

  std::ostringstream cmd;
  cmd << "cd " << shellQuote(project_root_) << " && PYTHONPATH=. "
      << shellQuote(python_executable_) << " -m agent.bridge_cli --input "
      << shellQuote(input_path) << " --output " << shellQuote(output_path);
  const int rc = std::system(cmd.str().c_str());
  std::remove(input_path.c_str());
  if (rc != 0) {
    std::remove(output_path.c_str());
    decision.rejection_reason = "bridge cli returned non-zero";
    return decision;
  }

  std::string output;
  if (!readTextFile(output_path, output)) {
    std::remove(output_path.c_str());
    decision.rejection_reason = "failed to read bridge output";
    return decision;
  }
  std::remove(output_path.c_str());

  const std::string skill = extractJsonString(output, "selected_skill");
  if (skill.empty()) {
    decision.rejection_reason = "bridge output missing selected_skill";
    decision.raw_json = output;
    return decision;
  }

  decision.ok = true;
  decision.selected_skill = skill;
  decision.fallback_used = extractJsonBool(output, "fallback_used", false);
  decision.accepted = extractJsonBool(output, "accepted", false);
  decision.has_target_candidate_id = extractJsonInt(output, "target_candidate_id", decision.target_candidate_id);
  decision.rejection_reason = extractJsonString(output, "rejection_reason");
  decision.raw_json = output;
  return decision;
}

bool ReflectiveAgentBridge::getBoolParam(
    ros::NodeHandle& nh, const std::string& key, bool default_value) const
{
  bool value;
  if (nh.getParam("reflective_agent/" + key, value) || nh.getParam(key, value) ||
      ros::param::get("/reflective_agent/" + key, value))
    return value;
  return default_value;
}

int ReflectiveAgentBridge::getIntParam(
    ros::NodeHandle& nh, const std::string& key, int default_value) const
{
  int value;
  if (nh.getParam("reflective_agent/" + key, value) || nh.getParam(key, value) ||
      ros::param::get("/reflective_agent/" + key, value))
    return value;
  return default_value;
}

double ReflectiveAgentBridge::getDoubleParam(
    ros::NodeHandle& nh, const std::string& key, double default_value) const
{
  double value;
  if (nh.getParam("reflective_agent/" + key, value) || nh.getParam(key, value) ||
      ros::param::get("/reflective_agent/" + key, value))
    return value;
  return default_value;
}

std::string ReflectiveAgentBridge::getStringParam(
    ros::NodeHandle& nh, const std::string& key, const std::string& default_value) const
{
  std::string value;
  if (nh.getParam("reflective_agent/" + key, value) || nh.getParam(key, value) ||
      ros::param::get("/reflective_agent/" + key, value))
    return value;
  return default_value;
}

std::string ReflectiveAgentBridge::buildPayloadJson(const ReflectiveAgentState& state) const
{
  std::ostringstream ss;
  ss << "{\"config\":" << buildConfigJson() << ",\"state\":" << buildStateJson(state) << "}";
  return ss.str();
}

std::string ReflectiveAgentBridge::buildConfigJson() const
{
  std::ostringstream ss;
  ss << "{";
  ss << "\"enable_reflective_agent\":" << (enable_reflective_agent_ ? "true" : "false");
  ss << ",\"enable_reflection_memory\":" << (enable_reflection_memory_ ? "true" : "false");
  ss << ",\"enable_decision_validator\":" << (enable_decision_validator_ ? "true" : "false");
  ss << ",\"require_multiview_before_stop\":" << (require_multiview_before_stop_ ? "true" : "false");
  ss << ",\"disable_recover_from_stuck\":" << (disable_recover_from_stuck_ ? "true" : "false");
  ss << ",\"enable_stuck_recovery_override\":"
     << (enable_stuck_recovery_override_ ? "true" : "false");
  ss << ",\"force_all_decisions_to_FALLBACK_APEXNAV\":"
     << (force_all_decisions_to_fallback_apexnav_ ? "true" : "false");
  ss << ",\"mock_follow_apexnav_by_default\":"
     << (mock_follow_apexnav_by_default_ ? "true" : "false");
  ss << ",\"include_detected_objects_in_state\":"
     << (include_detected_objects_in_state_ ? "true" : "false");
  ss << ",\"vlm_provider\":\"" << jsonEscape(vlm_provider_) << "\"";
  ss << ",\"vlm_model\":\"" << jsonEscape(vlm_model_) << "\"";
  ss << ",\"vlm_api_key\":\"" << jsonEscape(vlm_api_key_) << "\"";
  ss << ",\"vlm_base_url\":\"" << jsonEscape(vlm_base_url_) << "\"";
  ss << ",\"memory_path\":\"" << jsonEscape(memory_path_) << "\"";
  ss << ",\"memory_read_mode\":\"" << jsonEscape(memory_read_mode_) << "\"";
  ss << ",\"memory_write_mode\":\"" << jsonEscape(memory_write_mode_) << "\"";
  ss << ",\"episode_log_root\":\"" << jsonEscape(episode_log_root_) << "\"";
  ss << ",\"run_id\":\"" << jsonEscape(run_id_) << "\"";
  ss << ",\"project_root\":\"" << jsonEscape(project_root_) << "\"";
  ss << ",\"python_executable\":\"" << jsonEscape(python_executable_) << "\"";
  ss << ",\"max_retrieved_lessons\":" << max_retrieved_lessons_;
  ss << ",\"max_reflection_memory_items\":" << max_reflection_memory_items_;
  ss << ",\"max_vlm_calls_per_episode\":" << max_vlm_calls_per_episode_;
  ss << ",\"min_commitment_steps_for_explore\":" << min_commitment_steps_for_explore_;
  ss << ",\"max_commitment_steps_semantic\":" << max_commitment_steps_semantic_;
  ss << ",\"max_commitment_steps_geometric\":" << max_commitment_steps_geometric_;
  ss << ",\"max_commitment_steps_fallback\":" << max_commitment_steps_fallback_;
  ss << ",\"max_commitment_steps_recovery\":" << max_commitment_steps_recovery_;
  ss << ",\"structural_frontier_count_change_ratio\":"
     << jsonNumber(structural_frontier_count_change_ratio_);
  ss << ",\"structural_frontier_stable_k\":" << structural_frontier_stable_k_;
  ss << ",\"stable_target_event_k\":" << stable_target_event_k_;
  ss << ",\"stuck_threshold\":" << stuck_threshold_;
  ss << ",\"same_frontier_failure_threshold\":" << same_frontier_failure_threshold_;
  ss << ",\"target_verify_threshold\":" << jsonNumber(target_verify_threshold_);
  ss << ",\"target_stop_threshold\":" << jsonNumber(target_stop_threshold_);
  ss << ",\"semantic_peak_ratio_threshold\":" << jsonNumber(semantic_peak_ratio_threshold_);
  ss << ",\"semantic_peak_std_threshold\":" << jsonNumber(semantic_peak_std_threshold_);
  ss << ",\"low_information_gain_threshold\":" << jsonNumber(low_information_gain_threshold_);
  ss << ",\"vlm_timeout_seconds\":" << jsonNumber(vlm_timeout_seconds_);
  ss << ",\"vlm_temperature\":" << jsonNumber(vlm_temperature_);
  ss << "}";
  return ss.str();
}

std::string ReflectiveAgentBridge::buildStateJson(const ReflectiveAgentState& state) const
{
  std::ostringstream ss;
  ss << "{";
  ss << "\"episode_id\":\"" << jsonEscape(state.episode_id) << "\"";
  ss << ",\"scene_id\":\"" << jsonEscape(state.scene_id) << "\"";
  ss << ",\"split\":\"" << jsonEscape(state.split) << "\"";
  ss << ",\"target_category\":\"" << jsonEscape(state.target_category) << "\"";
  ss << ",\"timestep\":" << state.timestep;
  ss << ",\"bridge_diagnostics\":{";
  ss << "\"trigger_reasons\":[";
  std::stringstream reason_stream(state.trigger_reasons);
  std::string reason_item;
  bool first_reason = true;
  while (std::getline(reason_stream, reason_item, ',')) {
    if (reason_item.empty())
      continue;
    if (!first_reason)
      ss << ",";
    ss << "\"" << jsonEscape(reason_item) << "\"";
    first_reason = false;
  }
  ss << "]";
  ss << ",\"previous_committed_skill\":\"" << jsonEscape(state.previous_committed_skill) << "\"";
  ss << ",\"commitment_age_steps\":" << state.commitment_age_steps;
  ss << ",\"frontier_count_before\":" << state.frontier_count_before;
  ss << ",\"frontier_count_after\":" << state.frontier_count_after;
  ss << ",\"target_count_before\":" << state.target_count_before;
  ss << ",\"target_count_after\":" << state.target_count_after;
  ss << ",\"stuck_signal\":" << (state.stuck_signal ? "true" : "false");
  ss << ",\"target_found\":" << (state.target_found ? "true" : "false");
  ss << ",\"committed_target_reached\":"
     << (state.committed_target_reached ? "true" : "false");
  ss << ",\"structural_frontier_change\":"
     << (state.structural_frontier_change ? "true" : "false");
  ss << ",\"stable_target_event\":" << (state.stable_target_event ? "true" : "false");
  ss << "}";
  ss << ",\"frontiers\":[";
  for (size_t i = 0; i < state.frontiers.size(); ++i) {
    const auto& f = state.frontiers[i];
    if (i != 0)
      ss << ",";
    ss << "{\"id\":" << f.id << ",\"semantic_score\":" << jsonNumber(f.semantic_score)
       << ",\"distance\":" << jsonNumber(f.distance)
       << ",\"reachable\":" << (f.reachable ? "true" : "false")
       << ",\"visited\":" << (f.visited ? "true" : "false")
       << ",\"blocked\":" << (f.blocked ? "true" : "false")
       << ",\"low_value\":" << (f.low_value ? "true" : "false")
       << ",\"last_selected\":" << (f.last_selected ? "true" : "false")
       << ",\"failure_count\":" << f.failure_count << ",\"waypoint\":["
       << jsonNumber(f.waypoint(0)) << "," << jsonNumber(f.waypoint(1)) << "]}";
  }
  ss << "]";
  ss << ",\"target_candidates\":[";
  for (size_t i = 0; i < state.target_candidates.size(); ++i) {
    const auto& c = state.target_candidates[i];
    if (i != 0)
      ss << ",";
    std::string label_text = (c.label == 0 && !state.target_category.empty())
                                 ? state.target_category
                                 : ("label_" + std::to_string(c.label));
    ss << "{\"id\":" << c.id << ",\"label\":\"" << jsonEscape(label_text) << "\""
       << ",\"label_id\":" << c.label << ",\"confidence\":" << jsonNumber(c.confidence)
       << ",\"distance\":" << jsonNumber(c.distance)
       << ",\"reachable\":" << (c.reachable ? "true" : "false")
       << ",\"multi_view_confirmed\":" << (c.multi_view_confirmed ? "true" : "false")
       << ",\"num_views\":" << c.num_views
       << ",\"rejected_false_positive\":"
       << (c.rejected_false_positive ? "true" : "false")
       << ",\"waypoint\":[" << jsonNumber(c.waypoint(0)) << ","
       << jsonNumber(c.waypoint(1)) << "]}";
  }
  ss << "]";
  ss << ",\"rgb_observation\":";
  if (!state.rgb_observation_json.empty() && state.rgb_observation_json[0] == '{')
    ss << state.rgb_observation_json;
  else
    ss << "{\"available\":false}";
  ss << ",\"semantic_map_observation\":";
  if (!state.semantic_map_observation_json.empty() &&
      state.semantic_map_observation_json[0] == '{')
    ss << state.semantic_map_observation_json;
  else
    ss << "{\"available\":false}";
  ss << ",\"detected_objects\":";
  if (!state.detected_objects_json.empty() &&
      (state.detected_objects_json[0] == '{' || state.detected_objects_json[0] == '['))
    ss << state.detected_objects_json;
  else
    ss << "{\"available\":false,\"detections\":[]}";
  ss << ",\"gt_feedback\":";
  if (!state.gt_feedback_json.empty() && state.gt_feedback_json[0] == '{')
    ss << state.gt_feedback_json;
  else
    ss << "{\"available\":false}";
  ss << ",\"navigation_history\":{\"stuck_count\":" << state.stuck_count
     << ",\"collision_count\":" << state.collision_count << ",\"steps_left\":"
     << state.steps_left << ",\"recent_failures\":[";
  for (size_t i = 0; i < state.recent_failures.size(); ++i) {
    if (i != 0)
      ss << ",";
    ss << "\"" << jsonEscape(state.recent_failures[i]) << "\"";
  }
  ss << "],\"recent_selected_skills\":[";
  for (size_t i = 0; i < state.recent_selected_skills.size(); ++i) {
    if (i != 0)
    ss << ",";
    ss << "\"" << jsonEscape(state.recent_selected_skills[i]) << "\"";
  }
  ss << "],\"best_known_point\":{";
  ss << "\"available\":" << (state.best_known_point_valid ? "true" : "false");
  ss << ",\"waypoint\":[" << jsonNumber(state.best_known_point(0)) << ","
     << jsonNumber(state.best_known_point(1)) << "]";
  ss << ",\"score\":" << jsonNumber(state.best_known_score);
  ss << ",\"evidence_score\":" << jsonNumber(state.best_known_evidence_score);
  ss << ",\"distance_to_current\":" << jsonNumber(state.best_known_distance_to_current);
  ss << ",\"target_confidence\":" << jsonNumber(state.best_known_target_confidence);
  ss << ",\"target_views\":" << state.best_known_target_views;
  ss << ",\"semantic_score\":" << jsonNumber(state.best_known_semantic_score);
  ss << ",\"frontier_score\":" << jsonNumber(state.best_known_frontier_score);
  ss << ",\"frontier_count\":" << state.best_known_frontier_count;
  ss << ",\"selection_signal\":\"" << jsonEscape(state.best_known_reason) << "\"";
  ss << ",\"timestep\":" << state.best_known_timestep;
  ss << "}}";
  ss << "}";
  return ss.str();
}

std::string ReflectiveAgentBridge::jsonNumber(double value)
{
  if (!std::isfinite(value))
    return "null";
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string ReflectiveAgentBridge::jsonEscape(const std::string& value)
{
  std::ostringstream ss;
  for (char c : value) {
    switch (c) {
      case '\\':
        ss << "\\\\";
        break;
      case '"':
        ss << "\\\"";
        break;
      case '\n':
        ss << "\\n";
        break;
      case '\r':
        ss << "\\r";
        break;
      case '\t':
        ss << "\\t";
        break;
      default:
        ss << c;
    }
  }
  return ss.str();
}

std::string ReflectiveAgentBridge::shellQuote(const std::string& value)
{
  std::string out = "'";
  for (char c : value) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

bool ReflectiveAgentBridge::writeTextFile(const std::string& path, const std::string& text)
{
  std::ofstream f(path);
  if (!f.good())
    return false;
  f << text;
  return true;
}

bool ReflectiveAgentBridge::readTextFile(const std::string& path, std::string& text)
{
  std::ifstream f(path);
  if (!f.good())
    return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  text = ss.str();
  return true;
}

std::string ReflectiveAgentBridge::makeTempPath(const std::string& suffix)
{
  std::ostringstream ss;
  ss << "/tmp/apexnav_reflective_" << getpid() << "_" << std::time(nullptr) << "_"
     << std::rand() << "_" << suffix;
  return ss.str();
}

std::string ReflectiveAgentBridge::extractJsonString(const std::string& json, const std::string& key)
{
  std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (std::regex_search(json, match, re) && match.size() >= 2)
    return match[1];
  return "";
}

bool ReflectiveAgentBridge::extractJsonBool(
    const std::string& json, const std::string& key, bool default_value)
{
  std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (std::regex_search(json, match, re) && match.size() >= 2)
    return match[1] == "true";
  return default_value;
}

bool ReflectiveAgentBridge::extractJsonInt(
    const std::string& json, const std::string& key, int& value)
{
  std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
  std::smatch match;
  if (std::regex_search(json, match, re) && match.size() >= 2) {
    value = std::stoi(match[1]);
    return true;
  }
  return false;
}


}  // namespace apexnav_planner
