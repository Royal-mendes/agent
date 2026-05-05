
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <exploration_manager/reflective_agent_bridge.h>
#include <vis_utils/planning_visualization.h>

#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <sstream>
#include <utility>

namespace {
std::unique_ptr<apexnav_planner::ReflectiveAgentBridge> g_reflective_agent_bridge;
int g_reflective_decision_count = 0;
std::vector<std::string> g_recent_reflective_skills;
std::string g_committed_reflective_skill = "FALLBACK_APEXNAV";
Eigen::Vector2d g_committed_reflective_target = Eigen::Vector2d::Zero();
bool g_committed_reflective_target_valid = false;
int g_committed_reflective_frontier_count = -1;
int g_committed_reflective_object_count = -1;
double g_committed_reflective_max_target_confidence = 0.0;
int g_committed_reflective_max_target_views = 0;
int g_committed_reflective_use_count = 0;
bool g_committed_reflective_stuck_active = false;
bool g_committed_reflective_target_reached_active = false;
int g_committed_reflective_semantic_frontier_id = -1;
double g_committed_reflective_semantic_frontier_score = 0.0;
int g_committed_reflective_target_candidate_id = -1;
int g_structural_frontier_change_streak = 0;
int g_target_presence_streak = 0;
int g_reflective_commitment_continues = 0;
int g_reflective_commitment_refreshes = 0;
bool g_last_reflective_validator_accepted = false;
std::string g_last_reflective_rejection_reason;
std::string g_last_stop_action_source = "none";

constexpr double kReflectiveCommittedTargetReachDistance = 0.5;
int g_min_commitment_steps_for_explore = 10;
int g_max_commitment_steps_semantic = 80;
int g_max_commitment_steps_geometric = 80;
int g_max_commitment_steps_fallback = 80;
int g_max_commitment_steps_recovery = 40;
int g_structural_frontier_stable_k = 3;
int g_stable_target_event_k = 2;
int g_committed_frontier_failure_threshold = 2;
double g_structural_frontier_count_change_ratio = 0.3;
double g_semantic_frontier_shift_ratio = 0.15;
double g_target_verify_threshold = 0.65;
double g_target_stop_threshold = 0.75;
bool g_enable_semantic_map_observation = true;
bool g_require_multiview_before_stop = true;
int g_semantic_map_max_width = 320;
int g_semantic_map_jpeg_quality = 75;
double g_semantic_map_crop_size_m = 12.0;

void appendRefreshReason(std::string& reason, const std::string& item)
{
  if (!reason.empty())
    reason += ",";
  reason += item;
}

void resetReflectiveAgentCommitment()
{
  g_reflective_decision_count = 0;
  g_recent_reflective_skills.clear();
  g_committed_reflective_skill = "FALLBACK_APEXNAV";
  g_committed_reflective_target = Eigen::Vector2d::Zero();
  g_committed_reflective_target_valid = false;
  g_committed_reflective_frontier_count = -1;
  g_committed_reflective_object_count = -1;
  g_committed_reflective_max_target_confidence = 0.0;
  g_committed_reflective_max_target_views = 0;
  g_committed_reflective_use_count = 0;
  g_committed_reflective_stuck_active = false;
  g_committed_reflective_target_reached_active = false;
  g_committed_reflective_semantic_frontier_id = -1;
  g_committed_reflective_semantic_frontier_score = 0.0;
  g_committed_reflective_target_candidate_id = -1;
  g_structural_frontier_change_streak = 0;
  g_target_presence_streak = 0;
  g_reflective_commitment_continues = 0;
  g_reflective_commitment_refreshes = 0;
  g_last_reflective_validator_accepted = false;
  g_last_reflective_rejection_reason.clear();
  g_last_stop_action_source = "none";
  ros::param::set("/reflective_agent/last_stop_action_source", "none");
  ros::param::set("/reflective_agent/last_stop_invalid_active_stop", false);
}

bool isExplorationCommitment(const std::string& skill)
{
  return skill == "SEMANTIC_EXPLORE" || skill == "GEOMETRIC_EXPLORE";
}

bool isTargetCommitment(const std::string& skill)
{
  return skill == "VERIFY_TARGET" || skill == "NAVIGATE_TO_CONFIRMED_TARGET";
}

int maxCommitmentStepsForSkill(const std::string& skill)
{
  if (skill == "SEMANTIC_EXPLORE")
    return g_max_commitment_steps_semantic;
  if (skill == "GEOMETRIC_EXPLORE")
    return g_max_commitment_steps_geometric;
  if (skill == "RECOVER_FROM_STUCK")
    return g_max_commitment_steps_recovery;
  return g_max_commitment_steps_fallback;
}

bool frontierCountRatioChanged(int previous_count, int current_count)
{
  if (previous_count < 0)
    return false;
  const int denom = std::max(1, previous_count);
  const double ratio = std::abs(current_count - previous_count) / static_cast<double>(denom);
  return ratio >= g_structural_frontier_count_change_ratio;
}

void loadReflectiveCommitmentParams(ros::NodeHandle& nh)
{
  nh.param("reflective_agent/min_commitment_steps_for_explore", g_min_commitment_steps_for_explore, 10);
  nh.param("reflective_agent/max_commitment_steps_semantic", g_max_commitment_steps_semantic, 80);
  nh.param("reflective_agent/max_commitment_steps_geometric", g_max_commitment_steps_geometric, 80);
  nh.param("reflective_agent/max_commitment_steps_fallback", g_max_commitment_steps_fallback, 80);
  nh.param("reflective_agent/max_commitment_steps_recovery", g_max_commitment_steps_recovery, 40);
  nh.param("reflective_agent/structural_frontier_stable_k", g_structural_frontier_stable_k, 3);
  nh.param("reflective_agent/stable_target_event_k", g_stable_target_event_k, 2);
  nh.param("reflective_agent/same_frontier_failure_threshold", g_committed_frontier_failure_threshold, 2);
  nh.param("reflective_agent/structural_frontier_count_change_ratio",
      g_structural_frontier_count_change_ratio, 0.3);
  nh.param("reflective_agent/semantic_frontier_shift_ratio", g_semantic_frontier_shift_ratio, 0.15);
  nh.param("reflective_agent/target_verify_threshold", g_target_verify_threshold, 0.65);
  nh.param("reflective_agent/target_stop_threshold", g_target_stop_threshold, 0.75);
  nh.param("reflective_agent/require_multiview_before_stop", g_require_multiview_before_stop, true);
  ros::param::param("/reflective_agent/enable_semantic_map_observation",
      g_enable_semantic_map_observation, true);
  nh.param("reflective_agent/semantic_map_max_width", g_semantic_map_max_width, 320);
  nh.param("reflective_agent/semantic_map_jpeg_quality", g_semantic_map_jpeg_quality, 75);
  nh.param("reflective_agent/semantic_map_crop_size_m", g_semantic_map_crop_size_m, 12.0);
}

std::string base64Encode(const std::vector<uchar>& data)
{
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  int val = 0;
  int valb = -6;
  for (uchar c : data) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6)
    out.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4)
    out.push_back('=');
  return out;
}

std::string renderSemanticMapObservationJson(
    const apexnav_planner::ExplorationManager::Ptr& expl_manager,
    const Eigen::Vector2d& current_pos, int timestep)
{
  if (!g_enable_semantic_map_observation || !expl_manager || !expl_manager->sdf_map_ ||
      !expl_manager->sdf_map_->value_map_)
    return "{\"available\":false}";

  const int width = std::max(64, g_semantic_map_max_width);
  const int height = width;
  const double crop = std::max(2.0, g_semantic_map_crop_size_m);
  const double half = crop * 0.5;
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(45, 45, 45));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double wx = current_pos(0) - half + crop * (double(x) + 0.5) / double(width);
      const double wy = current_pos(1) + half - crop * (double(y) + 0.5) / double(height);
      Eigen::Vector2d pos(wx, wy);
      Eigen::Vector2i idx;
      expl_manager->sdf_map_->posToIndex(pos, idx);
      const int occ = expl_manager->sdf_map_->getOccupancy(idx);
      const int inflated = expl_manager->sdf_map_->getInflateOccupancy(idx);
      cv::Vec3b color(55, 55, 55);
      if (occ == apexnav_planner::SDFMap2D::FREE)
        color = cv::Vec3b(205, 205, 205);
      if (occ == apexnav_planner::SDFMap2D::OCCUPIED || inflated == 1)
        color = cv::Vec3b(0, 0, 0);

      if (occ == apexnav_planner::SDFMap2D::FREE && inflated != 1) {
        const double semantic = std::max(0.0, std::min(1.0,
            expl_manager->sdf_map_->value_map_->getValue(idx)));
        if (semantic > 1e-3) {
          const int red = 80 + int(175.0 * semantic);
          const int green = int(190.0 * semantic);
          color = cv::Vec3b(30, green, red);
        }
      }
      image.at<cv::Vec3b>(y, x) = color;
    }
  }

  auto toPixel = [&](const Eigen::Vector2d& pos) {
    const int px = int(std::round((pos(0) - (current_pos(0) - half)) / crop * double(width)));
    const int py = int(std::round(((current_pos(1) + half) - pos(1)) / crop * double(height)));
    return cv::Point(px, py);
  };
  auto inImage = [&](const cv::Point& pt) {
    return pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height;
  };

  for (const auto& frontier : expl_manager->ed_->frontier_averages_) {
    cv::Point pt = toPixel(frontier);
    if (inImage(pt))
      cv::circle(image, pt, 3, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
  }
  for (int i = 0; i < static_cast<int>(expl_manager->ed_->object_averages_.size()); ++i) {
    if (i >= static_cast<int>(expl_manager->ed_->object_labels_.size()) ||
        expl_manager->ed_->object_labels_[i] != 0)
      continue;
    cv::Point pt = toPixel(expl_manager->ed_->object_averages_[i]);
    if (inImage(pt))
      cv::circle(image, pt, 5, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
  }
  cv::Point robot_pt = toPixel(current_pos);
  if (inImage(robot_pt))
    cv::circle(image, robot_pt, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);

  std::vector<uchar> encoded;
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,
      std::max(20, std::min(95, g_semantic_map_jpeg_quality))};
  if (!cv::imencode(".jpg", image, encoded, params))
    return "{\"available\":false,\"error\":\"semantic_map_encode_failed\"}";

  std::ostringstream ss;
  ss << "{\"available\":true"
     << ",\"source\":\"apexnav_value_map\""
     << ",\"encoding\":\"jpeg_base64_data_url\""
     << ",\"width\":" << width
     << ",\"height\":" << height
     << ",\"timestep\":" << timestep
     << ",\"crop_size_m\":" << crop
     << ",\"legend\":\"red_yellow_high_semantic_value_light_gray_free_dark_gray_unknown_black_obstacle_cyan_frontier_blue_robot_magenta_target_candidate\""
     << ",\"data_url\":\"data:image/jpeg;base64," << base64Encode(encoded) << "\"}";
  return ss.str();
}

std::pair<int, double> highestSemanticFrontier(
    apexnav_planner::ExplorationManager::Ptr expl_manager)
{
  int best_id = -1;
  double best_score = -1e9;
  if (!expl_manager || !expl_manager->sdf_map_ || !expl_manager->sdf_map_->value_map_)
    return std::make_pair(best_id, best_score);
  for (int i = 0; i < static_cast<int>(expl_manager->ed_->frontier_averages_.size()); ++i) {
    Eigen::Vector2i idx;
    expl_manager->sdf_map_->posToIndex(expl_manager->ed_->frontier_averages_[i], idx);
    const double score = expl_manager->sdf_map_->value_map_->getValue(idx);
    if (score > best_score) {
      best_score = score;
      best_id = i;
    }
  }
  return std::make_pair(best_id, best_score);
}

bool semanticFrontierShifted(int current_id, double current_score)
{
  if (g_committed_reflective_semantic_frontier_id < 0 || current_id < 0)
    return false;
  if (current_id == g_committed_reflective_semantic_frontier_id)
    return false;
  const double denom = std::max(1e-6, std::abs(g_committed_reflective_semantic_frontier_score));
  return std::abs(current_score - g_committed_reflective_semantic_frontier_score) / denom >=
      g_semantic_frontier_shift_ratio;
}

std::string finalResultName(int final_result)
{
  switch (final_result) {
    case apexnav_planner::FINAL_RESULT::EXPLORE:
      return "explore";
    case apexnav_planner::FINAL_RESULT::SEARCH_OBJECT:
      return "search_object";
    case apexnav_planner::FINAL_RESULT::STUCKING:
      return "terminal_stuck";
    case apexnav_planner::FINAL_RESULT::NO_FRONTIER:
      return "terminal_no_frontier";
    case apexnav_planner::FINAL_RESULT::REACH_OBJECT:
      return "reach_object";
    default:
      return "unknown";
  }
}

struct StopTargetSnapshot {
  int id = -1;
  double confidence = -1.0;
  bool multi_view_confirmed = false;
  bool reachable = false;
  bool valid = false;
};

StopTargetSnapshot targetCandidateSnapshot(
    const apexnav_planner::ExplorationManager::Ptr& expl_manager, int candidate_id)
{
  StopTargetSnapshot snapshot;
  snapshot.id = candidate_id;
  if (!expl_manager || candidate_id < 0)
    return snapshot;
  const auto& ed = expl_manager->ed_;
  if (candidate_id >= static_cast<int>(ed->object_averages_.size()))
    return snapshot;
  if (candidate_id >= static_cast<int>(ed->object_labels_.size()) ||
      ed->object_labels_[candidate_id] != 0)
    return snapshot;
  snapshot.valid = true;
  snapshot.reachable = true;
  snapshot.confidence = candidate_id < static_cast<int>(ed->object_confidences_.size())
                            ? ed->object_confidences_[candidate_id]
                            : 0.0;
  const int views = candidate_id < static_cast<int>(ed->object_observation_nums_.size())
                        ? ed->object_observation_nums_[candidate_id]
                        : 1;
  snapshot.multi_view_confirmed = views >= 2;
  return snapshot;
}

StopTargetSnapshot nearestTargetCandidateSnapshot(
    const apexnav_planner::ExplorationManager::Ptr& expl_manager, const Eigen::Vector2d& waypoint)
{
  StopTargetSnapshot best;
  double best_dist = std::numeric_limits<double>::infinity();
  if (!expl_manager)
    return best;
  const auto& ed = expl_manager->ed_;
  for (int i = 0; i < static_cast<int>(ed->object_averages_.size()); ++i) {
    if (i >= static_cast<int>(ed->object_labels_.size()) || ed->object_labels_[i] != 0)
      continue;
    const double dist = (ed->object_averages_[i] - waypoint).norm();
    if (dist < best_dist) {
      best_dist = dist;
      best = targetCandidateSnapshot(expl_manager, i);
    }
  }
  return best;
}

StopTargetSnapshot stopTargetSnapshot(
    const apexnav_planner::ExplorationManager::Ptr& expl_manager, const Eigen::Vector2d& waypoint,
    bool vlm_stop_proposed)
{
  if (vlm_stop_proposed) {
    StopTargetSnapshot selected = targetCandidateSnapshot(
        expl_manager, g_committed_reflective_target_candidate_id);
    if (selected.valid)
      return selected;
  }
  return nearestTargetCandidateSnapshot(expl_manager, waypoint);
}

bool validatorAllowsStop(bool apexnav_stop_proposed, bool vlm_stop_proposed,
    const StopTargetSnapshot& target_snapshot)
{
  if (!apexnav_stop_proposed)
    return true;
  if (!vlm_stop_proposed)
    return true;
  if (!g_last_reflective_validator_accepted)
    return false;
  if (!target_snapshot.valid || !target_snapshot.reachable)
    return false;
  if (target_snapshot.confidence < g_target_stop_threshold)
    return false;
  if (g_require_multiview_before_stop && !target_snapshot.multi_view_confirmed)
    return false;
  return true;
}

bool recordStopValidator(const std::string& source, int target_candidate_id,
    double target_confidence, bool multi_view_confirmed, bool target_reachable,
    bool apexnav_stop_proposed, bool vlm_stop_proposed, bool validator_allowed_stop)
{
  const bool source_known = !source.empty() && source != "unknown";
  const bool final_allowed = source_known && validator_allowed_stop;
  ros::param::set("/reflective_agent/last_stop_action_source", source);
  ros::param::set("/reflective_agent/last_stop_target_candidate_id", target_candidate_id);
  ros::param::set("/reflective_agent/last_stop_target_confidence", target_confidence);
  ros::param::set("/reflective_agent/last_stop_multi_view_confirmed", multi_view_confirmed);
  ros::param::set("/reflective_agent/last_stop_target_reachable", target_reachable);
  ros::param::set("/reflective_agent/last_stop_apexnav_stop_proposed", apexnav_stop_proposed);
  ros::param::set("/reflective_agent/last_stop_vlm_stop_proposed", vlm_stop_proposed);
  ros::param::set("/reflective_agent/last_stop_validator_allowed_stop", validator_allowed_stop);
  ros::param::set("/reflective_agent/last_stop_final_stop_allowed_by_gate", final_allowed);
  ros::param::set("/reflective_agent/last_stop_invalid_active_stop", !final_allowed);
  g_last_stop_action_source = source;
  ROS_WARN("[StopValidator] source=%s candidate=%d conf=%.3f multiview=%d reachable=%d "
           "apexnav_stop=%d vlm_stop=%d validator_allowed=%d final_allowed=%d",
      source.c_str(), target_candidate_id, target_confidence, multi_view_confirmed,
      target_reachable, apexnav_stop_proposed, vlm_stop_proposed,
      validator_allowed_stop, final_allowed);
  return final_allowed;
}
}  // namespace

namespace apexnav_planner {
void ExplorationFSM::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  g_reflective_agent_bridge.reset(new ReflectiveAgentBridge);
  g_reflective_agent_bridge->init(nh);
  loadReflectiveCommitmentParams(nh);
  resetReflectiveAgentCommitment();
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  state_ = ROS_STATE::INIT;

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstants::EXEC_TIMER_DURATION), &ExplorationFSM::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstants::FRONTIER_TIMER_DURATION),
      &ExplorationFSM::frontierCallback, this);

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSM::odometryCallback, this);
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 10, &ExplorationFSM::habitatStateCallback, this);
  confidence_threshold_sub_ = node_.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSM::confidenceThresholdCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  action_pub_ = nh.advertise<std_msgs::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_.publish(ros_state_msg);
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "Wait for trigger.");
      break;
    }

	    case ROS_STATE::FINISH: {
	      if (!fd_->have_finished_) {
	        fd_->have_finished_ = true;
	        clearVisMarker();
	        bool final_stop_allowed = true;
	        if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()) {
	          const bool vlm_stop_proposed =
	              g_committed_reflective_skill == "NAVIGATE_TO_CONFIRMED_TARGET";
	          const bool apexnav_stop_proposed =
	              fd_->final_result_ == FINAL_RESULT::REACH_OBJECT;
	          std::string stop_source = finalResultName(fd_->final_result_);
	          if (apexnav_stop_proposed)
	            stop_source = vlm_stop_proposed ? "vlm_target_skill" : "apexnav_target_branch";
	          const StopTargetSnapshot target_snapshot = stopTargetSnapshot(
	              expl_manager_, expl_manager_->ed_->next_pos_, vlm_stop_proposed);
	          const bool validator_allowed = validatorAllowsStop(
	              apexnav_stop_proposed, vlm_stop_proposed, target_snapshot);
	          final_stop_allowed = recordStopValidator(stop_source, target_snapshot.id,
	              target_snapshot.confidence, target_snapshot.multi_view_confirmed,
	              target_snapshot.reachable, apexnav_stop_proposed, vlm_stop_proposed,
	              validator_allowed);
	        }
	        if (!final_stop_allowed) {
	          fd_->have_finished_ = false;
	          transitState(ROS_STATE::PLAN_ACTION, "StopValidator Blocked STOP");
	          exec_timer_.start();
	          break;
	        }
	        std_msgs::Int32 action_msg;
	        action_msg.data = ACTION::STOP;
	        action_pub_.publish(action_msg);
      }
      ROS_WARN_THROTTLE(1.0, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        ROS_WARN("Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        auto t1 = ros::Time::now();
        fd_->final_result_ = callActionPlanner();
        double call_action_planner_time = (ros::Time::now() - t1).toSec();
        ROS_INFO_THROTTLE(
            10.0, "[Calculating Time] Planning process time = %.3f s", call_action_planner_time);

        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        expl_state_pub_.publish(expl_state_msg);
        if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
            fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
          transitState(ROS_STATE::PUB_ACTION, "FSM");
        else
          transitState(ROS_STATE::FINISH, "FSM");
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      std_msgs::Int32 action_msg;
      action_msg.data = fd_->newest_action_;
      action_pub_.publish(action_msg);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_.start();
      break;
    }
  }
  exec_timer_.start();
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;

  // Reach the object - check if close enough to target object
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance) {
    ROS_ERROR("Reach the object successfully!!!");
    final_res = FINAL_RESULT::REACH_OBJECT;
    if (!(g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()))
      return final_res;
    {
      const bool vlm_stop_proposed =
          g_committed_reflective_skill == "NAVIGATE_TO_CONFIRMED_TARGET";
      const StopTargetSnapshot target_snapshot = stopTargetSnapshot(
          expl_manager_, expl_manager_->ed_->next_pos_, vlm_stop_proposed);
      if (recordStopValidator(
              vlm_stop_proposed ? "vlm_target_skill" : "apexnav_target_branch",
              target_snapshot.id, target_snapshot.confidence,
              target_snapshot.multi_view_confirmed, target_snapshot.reachable, true,
              vlm_stop_proposed,
              validatorAllowsStop(true, vlm_stop_proposed, target_snapshot)))
        return final_res;
    }
    ROS_ERROR("[StopValidator] Blocked invalid active stop; continue planning.");
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  if (!fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      ROS_ERROR("Reach the object successfully!!!");
      final_res = FINAL_RESULT::REACH_OBJECT;
      if (!(g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()))
        return final_res;
      {
        const bool vlm_stop_proposed =
            g_committed_reflective_skill == "NAVIGATE_TO_CONFIRMED_TARGET";
        const StopTargetSnapshot target_snapshot = stopTargetSnapshot(
            expl_manager_, expl_manager_->ed_->next_pos_, vlm_stop_proposed);
        if (recordStopValidator(
                vlm_stop_proposed ? "vlm_target_skill" : "apexnav_target_branch",
                target_snapshot.id, target_snapshot.confidence,
                target_snapshot.multi_view_confirmed, target_snapshot.reachable, true,
                vlm_stop_proposed,
                validatorAllowsStop(true, vlm_stop_proposed, target_snapshot)))
          return final_res;
      }
      ROS_ERROR("[StopValidator] Blocked invalid active stop; continue planning.");
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        ROS_ERROR("Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    ROS_ERROR("Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    ROS_ERROR("Escaping stuck...");
    if (fd_->escape_stucking_count_ == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else {
      // Failed to escape - mark area as occupied and add to stuck points
      ROS_ERROR("Cannot escape stuck state.");
      fd_->escape_stucking_flag_ = false;
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  const bool dormant_replan_requested = fd_->dormant_frontier_flag_;
  const int frontier_count = static_cast<int>(expl_manager_->ed_->frontier_averages_.size());
  const int object_count = static_cast<int>(expl_manager_->ed_->object_averages_.size());
  int target_candidate_count = 0;
  double max_target_confidence = 0.0;
  int max_target_views = 0;
  for (int i = 0; i < object_count; ++i) {
    const int label = i < static_cast<int>(expl_manager_->ed_->object_labels_.size())
                          ? expl_manager_->ed_->object_labels_[i]
                          : 0;
    if (label != 0)
      continue;
    ++target_candidate_count;
    const double conf = i < static_cast<int>(expl_manager_->ed_->object_confidences_.size())
                            ? expl_manager_->ed_->object_confidences_[i]
                            : 0.0;
    const int views = i < static_cast<int>(expl_manager_->ed_->object_observation_nums_.size())
                          ? expl_manager_->ed_->object_observation_nums_[i]
                          : 1;
    max_target_confidence = std::max(max_target_confidence, conf);
    max_target_views = std::max(max_target_views, views);
  }
  const int frontier_count_before = g_committed_reflective_frontier_count;
  const int target_candidate_count_before = g_committed_reflective_object_count;
  const int commitment_age_steps = g_committed_reflective_use_count;
  const auto current_semantic_frontier = highestSemanticFrontier(expl_manager_);
  const bool reflective_stuck_signal_active =
      fd_->escape_stucking_flag_ || fd_->stucking_action_count_ >= 3 ||
      fd_->stucking_next_pos_count_ >= 3;
  const bool reflective_target_reached_signal_active =
      g_committed_reflective_target_valid &&
      (current_pos - g_committed_reflective_target).norm() <
          kReflectiveCommittedTargetReachDistance;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag)
    fd_->replan_flag_ = false;


  std::string selected_skill = "FALLBACK_APEXNAV";
  if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()) {
    std::string refresh_reason;
    const bool frontier_availability_changed =
        frontier_count_before >= 0 &&
        ((frontier_count_before == 0 && frontier_count > 0) ||
            (frontier_count_before > 0 && frontier_count == 0));
    const bool frontier_ratio_changed =
        frontierCountRatioChanged(frontier_count_before, frontier_count);
    if (frontier_change_flag)
      ++g_structural_frontier_change_streak;
    else
      g_structural_frontier_change_streak = 0;
    const bool frontier_change_stably_present =
        g_structural_frontier_change_streak >= g_structural_frontier_stable_k;
    const bool semantic_frontier_shifted =
        semanticFrontierShifted(current_semantic_frontier.first, current_semantic_frontier.second);
    const bool committed_frontier_failed =
        fd_->stucking_next_pos_count_ >= g_committed_frontier_failure_threshold;
    const bool structural_frontier_change =
        frontier_availability_changed || frontier_ratio_changed || dormant_replan_requested ||
        reflective_target_reached_signal_active || frontier_change_stably_present ||
        semantic_frontier_shifted || committed_frontier_failed;

    if (target_candidate_count > 0)
      ++g_target_presence_streak;
    else
      g_target_presence_streak = 0;
    const bool target_count_increased =
        target_candidate_count_before >= 0 &&
        target_candidate_count > target_candidate_count_before;
    const bool target_newly_available =
        target_candidate_count_before == 0 && target_candidate_count > 0;
    const bool target_verify_threshold_crossed =
        g_committed_reflective_max_target_confidence < g_target_verify_threshold &&
        max_target_confidence >= g_target_verify_threshold;
    const bool target_stop_threshold_crossed =
        g_committed_reflective_max_target_confidence < g_target_stop_threshold &&
        max_target_confidence >= g_target_stop_threshold;
    const bool target_multiview_confirmed =
        g_committed_reflective_max_target_views < 2 && max_target_views >= 2;
    const bool stable_target_event =
        target_candidate_count > 0 && g_target_presence_streak >= g_stable_target_event_k &&
        (target_newly_available || target_count_increased || target_verify_threshold_crossed ||
            target_stop_threshold_crossed || target_multiview_confirmed);
    const bool allow_minor_interrupt =
        !isExplorationCommitment(g_committed_reflective_skill) ||
        commitment_age_steps >= g_min_commitment_steps_for_explore;

    if (g_reflective_decision_count == 0)
      appendRefreshReason(refresh_reason, "first_decision");
    if (commitment_age_steps >= maxCommitmentStepsForSkill(g_committed_reflective_skill))
      appendRefreshReason(refresh_reason, "committed_skill_timeout");
    if (frontier_availability_changed)
      appendRefreshReason(refresh_reason, "frontier_availability_changed");
    if (dormant_replan_requested)
      appendRefreshReason(refresh_reason, "dormant_frontier");
    if (reflective_stuck_signal_active && !g_committed_reflective_stuck_active)
      appendRefreshReason(refresh_reason, "stuck_signal");
    if (reflective_target_reached_signal_active &&
        !g_committed_reflective_target_reached_active)
      appendRefreshReason(refresh_reason, "committed_target_reached");
    if (stable_target_event)
      appendRefreshReason(refresh_reason, "stable_target_event");
    if (allow_minor_interrupt && frontier_ratio_changed)
      appendRefreshReason(refresh_reason, "frontier_count_ratio_changed");
    if (allow_minor_interrupt && frontier_change_stably_present)
      appendRefreshReason(refresh_reason, "frontier_change_stably_present");
    if (allow_minor_interrupt && semantic_frontier_shifted)
      appendRefreshReason(refresh_reason, "semantic_frontier_shifted");
    if (committed_frontier_failed)
      appendRefreshReason(refresh_reason, "committed_frontier_failed");
    if (g_committed_reflective_skill == "RECOVER_FROM_STUCK" &&
        g_committed_reflective_stuck_active && !reflective_stuck_signal_active)
      appendRefreshReason(refresh_reason, "recovery_resolved");

    if (refresh_reason.empty()) {
      selected_skill = g_committed_reflective_skill;
      ++g_reflective_commitment_continues;
      ROS_WARN_THROTTLE(5.0,
          "[ReflectiveAgentBridge] continue committed_skill=%s continues=%d refreshes=%d use_count=%d",
          selected_skill.c_str(), g_reflective_commitment_continues,
          g_reflective_commitment_refreshes, g_committed_reflective_use_count);
    }
    else {
      ++g_reflective_commitment_refreshes;
      ReflectiveAgentState reflective_state;
      reflective_state.split = "unknown";
      nh_.getParam("/reflective_agent/current_episode_id", reflective_state.episode_id);
      nh_.getParam("/reflective_agent/current_scene_id", reflective_state.scene_id);
      nh_.getParam("/reflective_agent/current_split", reflective_state.split);
      nh_.getParam("/reflective_agent/current_target_category", reflective_state.target_category);
      reflective_state.timestep = g_reflective_decision_count++;
      reflective_state.stuck_count = fd_->stucking_action_count_;
      reflective_state.trigger_reasons = refresh_reason;
      reflective_state.previous_committed_skill = g_committed_reflective_skill;
      reflective_state.commitment_age_steps = commitment_age_steps;
      reflective_state.frontier_count_before = frontier_count_before;
      reflective_state.frontier_count_after = frontier_count;
      reflective_state.target_count_before = target_candidate_count_before;
      reflective_state.target_count_after = target_candidate_count;
      reflective_state.stuck_signal = reflective_stuck_signal_active;
      reflective_state.target_found = target_count_increased;
      reflective_state.committed_target_reached = reflective_target_reached_signal_active;
      reflective_state.structural_frontier_change = structural_frontier_change;
      reflective_state.stable_target_event = stable_target_event;
      nh_.getParam("/reflective_agent/current_rgb_observation",
          reflective_state.rgb_observation_json);
      reflective_state.semantic_map_observation_json =
          renderSemanticMapObservationJson(expl_manager_, current_pos, reflective_state.timestep);
      nh_.getParam("/reflective_agent/current_detected_landmarks",
          reflective_state.detected_objects_json);
      nh_.getParam("/reflective_agent/current_gt_feedback", reflective_state.gt_feedback_json);
      if (reflective_stuck_signal_active)
        reflective_state.recent_failures.push_back("planner_stuck");
      if (frontier_count == 0)
        reflective_state.recent_failures.push_back("no_frontier_deadend");
      if (committed_frontier_failed)
        reflective_state.recent_failures.push_back("frontier_failure");
      reflective_state.recent_selected_skills = g_recent_reflective_skills;
      for (int i = 0; i < frontier_count; ++i) {
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
      for (int i = 0; i < object_count; ++i) {
        ReflectiveTargetCandidate c;
        c.id = i;
        c.waypoint = expl_manager_->ed_->object_averages_[i];
        c.distance = (c.waypoint - current_pos).norm();
        c.reachable = true;
        c.confidence = i < static_cast<int>(expl_manager_->ed_->object_confidences_.size())
                           ? expl_manager_->ed_->object_confidences_[i]
                           : 0.0;
        c.num_views = i < static_cast<int>(expl_manager_->ed_->object_observation_nums_.size())
                          ? expl_manager_->ed_->object_observation_nums_[i]
                          : 1;
        c.multi_view_confirmed = c.num_views >= 2;
        if (i < static_cast<int>(expl_manager_->ed_->object_labels_.size()))
          c.label = expl_manager_->ed_->object_labels_[i];
        if (c.label == 0)
          reflective_state.target_candidates.push_back(c);
      }
      ROS_WARN("[ReflectiveAgentBridge] VLM requery timestep=%d triggers=%s prev=%s age=%d "
               "frontier=%d->%d target=%d->%d stuck=%d target_found=%d reached=%d",
          reflective_state.timestep, refresh_reason.c_str(),
          reflective_state.previous_committed_skill.c_str(), commitment_age_steps,
          frontier_count_before, frontier_count, target_candidate_count_before,
          target_candidate_count, reflective_stuck_signal_active, target_count_increased,
          reflective_target_reached_signal_active);
      auto decision = g_reflective_agent_bridge->selectSkill(reflective_state);
      if (decision.ok) {
        selected_skill = decision.selected_skill;
        g_committed_reflective_skill = selected_skill;
        g_committed_reflective_use_count = 0;
        g_last_reflective_validator_accepted = decision.accepted;
        g_last_reflective_rejection_reason = decision.rejection_reason;
        if (isTargetCommitment(selected_skill))
          g_committed_reflective_target_candidate_id = decision.has_target_candidate_id
              ? decision.target_candidate_id
              : -1;
        else
          g_committed_reflective_target_candidate_id = -1;
        g_recent_reflective_skills.push_back(selected_skill);
        if (g_recent_reflective_skills.size() > 20)
          g_recent_reflective_skills.erase(g_recent_reflective_skills.begin());
        ROS_WARN(
            "[ReflectiveAgentBridge] selected_skill=%s accepted=%d fallback=%d target_id=%d refresh=%s reason=%s",
            selected_skill.c_str(), decision.accepted, decision.fallback_used,
            g_committed_reflective_target_candidate_id, refresh_reason.c_str(),
            decision.rejection_reason.c_str());
      }
      else {
        selected_skill = "FALLBACK_APEXNAV";
        g_committed_reflective_skill = selected_skill;
        g_committed_reflective_use_count = 0;
        g_last_reflective_validator_accepted = false;
        g_last_reflective_rejection_reason = decision.rejection_reason;
        g_committed_reflective_target_candidate_id = -1;
        ROS_WARN("[ReflectiveAgentBridge] fallback to ApexNav: refresh=%s reason=%s",
            refresh_reason.c_str(), decision.rejection_reason.c_str());
      }
    }
  }

  if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled())
    expl_res = expl_manager_->planNextBestPointWithSkill(
        selected_skill, fd_->start_pt_, fd_->start_yaw_(0));
  else
    expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  if (g_reflective_agent_bridge && g_reflective_agent_bridge->enabled()) {
    g_committed_reflective_frontier_count = frontier_count;
    g_committed_reflective_object_count = target_candidate_count;
    g_committed_reflective_max_target_confidence = max_target_confidence;
    g_committed_reflective_max_target_views = max_target_views;
    g_committed_reflective_target_valid = !expl_manager_->ed_->next_best_path_.empty();
    if (g_committed_reflective_target_valid)
      g_committed_reflective_target = expl_manager_->ed_->next_pos_;
	    g_committed_reflective_stuck_active = reflective_stuck_signal_active;
	    g_committed_reflective_target_reached_active = reflective_target_reached_signal_active;
	    g_committed_reflective_semantic_frontier_id = current_semantic_frontier.first;
	    g_committed_reflective_semantic_frontier_score = current_semantic_frontier.second;
	    ++g_committed_reflective_use_count;
	  }
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Publish exploration result to monitor
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_.publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
    ROS_WARN("No (passable) frontier");
    return final_res;
  }

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  ROS_WARN("To the next point (%.2fm %.2fm), distance = %.2f m", end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      ROS_ERROR("Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      ROS_ERROR_COND(fd_->stucking_next_pos_count_ > 8, "stucking_next_pos_count_ = %d",
          fd_->stucking_next_pos_count_);
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      ROS_ERROR("Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }
  }

  // Track consecutive stuck actions globally
  if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stucking_action_count_++;
    ROS_ERROR_COND(fd_->stucking_action_count_ > 15, "Stucking action count = %d",
        fd_->stucking_action_count_);
  }
  else
    fd_->stucking_action_count_ = 0;

  // If stuck for too long globally, terminate episode
  if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
    ROS_ERROR("Stuck for too long, stopping episode.");
    final_res = FINAL_RESULT::STUCKING;
    return final_res;
  }

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
  else
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);

  return final_res;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  // static int last_obj_num = 0;
  // for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
  //   visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
  //       visualization_->getColor(double(i) / ed_ptr->objects_.size(), 1.0), "object", i, 4);
  // }
  // for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
  //   visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  // }
  // last_obj_num = ed_ptr->objects_.size();

  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSM::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d start_pos2d = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(start_pos2d, fd_->start_yaw_(0));
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_,
      &ed->object_confidences_, &ed->object_observation_nums_);

  return change_flag;
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data == HABITAT_STATE::ACTION_FINISH && state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH)
    init(nh_);
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback(const ros::TimerEvent& e)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << endl;
}
}  // namespace apexnav_planner
