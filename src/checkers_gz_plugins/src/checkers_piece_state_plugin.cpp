#include <memory>
#include <sstream>
#include <string>
#include <iostream>

#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace checkers_gz_plugins
{

class CheckersPieceStatePlugin
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPostUpdate
{
public:
  void Configure(
      const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &,
      gz::sim::EntityComponentManager &,
      gz::sim::EventManager &) override
  {
    if (!rclcpp::ok())
    {
      int argc = 0;
      char **argv = nullptr;
      rclcpp::init(argc, argv);
    }

    this->node_ = std::make_shared<rclcpp::Node>("checkers_piece_state_plugin");
    this->pub_ = this->node_->create_publisher<std_msgs::msg::String>(
        "/checkers/piece_states", 10);

    this->last_pub_time_ = this->node_->now();
    RCLCPP_INFO(this->node_->get_logger(), "CheckersPieceStatePlugin configured.");
  }

  void PostUpdate(
      const gz::sim::UpdateInfo &_info,
      const gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    auto now = this->node_->now();
    if ((now - this->last_pub_time_).seconds() < 0.2)
      return;

    std::ostringstream json;
    json << "[";

    bool first = true;

    _ecm.Each<gz::sim::components::Model,
              gz::sim::components::Name,
              gz::sim::components::Pose>(
        [&](const gz::sim::Entity &,
            const gz::sim::components::Model *,
            const gz::sim::components::Name *_name,
            const gz::sim::components::Pose *_pose) -> bool
        {
          const std::string &name = _name->Data();

          const bool is_red = name.rfind("red_checker_", 0) == 0;
          const bool is_black = name.rfind("black_checker_", 0) == 0;

          if (!is_red && !is_black)
            return true;

          const auto &pose = _pose->Data();

          if (!first)
            json << ",";

          json << "{"
               << "\"name\":\"" << name << "\","
               << "\"position\":{"
               << "\"x\":" << pose.Pos().X() << ","
               << "\"y\":" << pose.Pos().Y() << ","
               << "\"z\":" << pose.Pos().Z()
               << "}"
               << "}";

          first = false;
          return true;
        });

    json << "]";

    std_msgs::msg::String msg;
    msg.data = json.str();
    this->pub_->publish(msg);

    rclcpp::spin_some(this->node_);
    this->last_pub_time_ = now;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::Time last_pub_time_;
};

}  // namespace checkers_gz_plugins

GZ_ADD_PLUGIN(
    checkers_gz_plugins::CheckersPieceStatePlugin,
    gz::sim::System,
    gz::sim::ISystemConfigure,
    gz::sim::ISystemPostUpdate
)