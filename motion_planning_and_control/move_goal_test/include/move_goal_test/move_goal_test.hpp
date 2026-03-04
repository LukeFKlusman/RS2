#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace move_goal_test
{

class MoveGoalTestNode : public rclcpp::Node
{
public:
  using FollowJT = control_msgs::action::FollowJointTrajectory;
  using GoalHandleFollowJT = rclcpp_action::ClientGoalHandle<FollowJT>;

  explicit MoveGoalTestNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void start();                 // Wait for action server then send goal
  void send_goal();             // Build goal message and send
  FollowJT::Goal build_goal();  // Constructs the trajectory goal

  // Callbacks for the action client
  void goal_response_callback(const GoalHandleFollowJT::SharedPtr & goal_handle);
  void feedback_callback(
    GoalHandleFollowJT::SharedPtr,
    const std::shared_ptr<const FollowJT::Feedback> feedback);
  void result_callback(const GoalHandleFollowJT::WrappedResult & result);

  rclcpp_action::Client<FollowJT>::SharedPtr action_client_;
  rclcpp::TimerBase::SharedPtr startup_timer_;

  // Parameters
  std::string action_name_;
};

}  // namespace move_goal_test