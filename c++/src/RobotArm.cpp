#include "RobotArm.hpp"
#include <cmath>

RobotArm::RobotArm(double l1, double l2,
                   const Eigen::Vector4d& initial_state,
                   const Eigen::Matrix4d& initial_covariance,
                   const Eigen::Matrix4d& process_noise,
                   const Eigen::Matrix2d& measurement_noise)
    : ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise),
      l1_(l1),
      l2_(l2)
{}

Eigen::Vector2d RobotArm::forwardKinematics(const Eigen::Vector2d& theta) const {
    double x = l1_ * std::cos(theta(0)) + l2_ * std::cos(theta(0) + theta(1));
    double y = l1_ * std::sin(theta(0)) + l2_ * std::sin(theta(0) + theta(1));
    return Eigen::Vector2d(x, y);
}

Eigen::Matrix<double, 2, 4> RobotArm::measurementJacobian(const Eigen::Vector4d& x) const {
    Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();

    double theta1 = x(0);
    double theta2 = x(1);

    H(0, 0) = -l1_ * std::sin(theta1) - l2_ * std::sin(theta1 + theta2);
    H(0, 1) = -l2_ * std::sin(theta1 + theta2);
    H(1, 0) = l1_ * std::cos(theta1) + l2_ * std::cos(theta1 + theta2);
    H(1, 1) = l2_ * std::cos(theta1 + theta2);

    // Velocities do not directly affect measurement
    return H;
}

Eigen::Vector2d RobotArm::measurementPrediction() const {
    return forwardKinematics(x_.head<2>());
}
