#include "move_goal_test/move_goal_test.hpp"

#include <array>
#include <chrono>
#include <utility>

namespace move_goal_test
{

using namespace std::chrono_literals;

MoveGoalTestNode::MoveGoalTestNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("move_goal_test_node", options)
{
  // Parameter so you can change controller name without recompiling
  action_name_ = this->declare_parameter<std::string>(
    "action_name",
    "/scaled_joint_trajectory_controller/follow_joint_trajectory");

  action_client_ = rclcpp_action::create_client<FollowJT>(this, action_name_);

  // Kick off after node is fully up
  startup_timer_ = this->create_wall_timer(200ms, std::bind(&MoveGoalTestNode::start, this));
}

void MoveGoalTestNode::start()
{
  // Only run once
  startup_timer_->cancel();

  RCLCPP_INFO(get_logger(), "Waiting for action server: %s", action_name_.c_str());

  if (!action_client_->wait_for_action_server(5s)) {
    RCLCPP_ERROR(get_logger(), "Action server not available after waiting. Is the controller running?");
    return;
  }

  RCLCPP_INFO(get_logger(), "Action server is available. Sending goal.");
  send_goal();
}

void MoveGoalTestNode::send_goal()
{
  auto goal_msg = build_goal();

  rclcpp_action::Client<FollowJT>::SendGoalOptions options;
  options.goal_response_callback =
    std::bind(&MoveGoalTestNode::goal_response_callback, this, std::placeholders::_1);
  options.feedback_callback =
    std::bind(&MoveGoalTestNode::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
  options.result_callback =
    std::bind(&MoveGoalTestNode::result_callback, this, std::placeholders::_1);

  action_client_->async_send_goal(goal_msg, options);
}

MoveGoalTestNode::FollowJT::Goal MoveGoalTestNode::build_goal()
{
  FollowJT::Goal goal;

  trajectory_msgs::msg::JointTrajectory traj;
  traj.joint_names = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
  };

  auto make_point = [](const std::array<double, 6> & pos, int sec) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = {pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]};
    pt.time_from_start.sec = sec;
    pt.time_from_start.nanosec = 0;
    return pt;
  };

  // Matches your terminal goal exactly (times are cumulative)
  traj.points.push_back(make_point({ 0.0,  -1.57,  1.57,  0.0,  1.57,  0.0 },  7));
  traj.points.push_back(make_point({ 0.8,  -1.20,  1.30, -0.60, 1.57,  0.50 }, 14));
  traj.points.push_back(make_point({-0.8,  -1.05,  1.90,  0.70, 1.57, -0.70 }, 21));
  traj.points.push_back(make_point({ 1.2,  -1.45,  1.10, -1.00, 1.57,  1.00 }, 28));
  traj.points.push_back(make_point({ 0.0,  -1.57,  1.57,  0.0,  1.57,  0.0 }, 35));

  goal.trajectory = traj;
  return goal;
}

void MoveGoalTestNode::goal_response_callback(const GoalHandleFollowJT::SharedPtr & goal_handle)
{
  if (!goal_handle) {
    RCLCPP_ERROR(get_logger(), "Goal was rejected by the action server.");
    return;
  }
  RCLCPP_INFO(get_logger(), "Goal accepted. Waiting for result...");
}

void MoveGoalTestNode::feedback_callback(
  GoalHandleFollowJT::SharedPtr,
  const std::shared_ptr<const FollowJT::Feedback> feedback)
{
  // Feedback is optional. This gives you visibility while testing.
  // Most controllers populate desired/actual/error.
  (void)feedback;
  // If you want, you can log at a low rate. Leaving quiet for now.
}

void MoveGoalTestNode::result_callback(const GoalHandleFollowJT::WrappedResult & result)
{
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_INFO(get_logger(), "Trajectory execution succeeded.");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(get_logger(), "Trajectory execution was aborted.");
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(get_logger(), "Trajectory execution was canceled.");
      break;
    default:
      RCLCPP_ERROR(get_logger(), "Unknown result code.");
      break;
  }

  // Optional: shut down after sending the one test trajectory
  rclcpp::shutdown();
}

}  // namespace move_goal_test
