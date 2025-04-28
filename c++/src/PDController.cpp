#include "PDController.hpp"

PDController::PDController(double kp, double kd)
    : kp_(kp), kd_(kd)
{}

Eigen::Vector2d PDController::computeControl(double error, double velocity) const {
    return Eigen::Vector2d(0.0, kp_ * error + kd_ * (-velocity));
}
