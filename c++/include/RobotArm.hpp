#ifndef ROBOT_ARM_HPP
#define ROBOT_ARM_HPP

#include "ExtendedKalmanFilter.hpp"
#include <Eigen/Dense>

class RobotArm : public ExtendedKalmanFilter {
public:
    RobotArm(double l1, double l2,
             const Eigen::Vector4d& initial_state,
             const Eigen::Matrix4d& initial_covariance,
             const Eigen::Matrix4d& process_noise,
             const Eigen::Matrix2d& measurement_noise);

    Eigen::Vector2d forwardKinematics(const Eigen::Vector2d& theta) const;
    Eigen::Matrix<double, 2, 4> measurementJacobian(const Eigen::Vector4d& x) const override;
    Eigen::Vector2d measurementPrediction() const override;

private:
    double l1_, l2_;
};

#endif // ROBOT_ARM_HPP
