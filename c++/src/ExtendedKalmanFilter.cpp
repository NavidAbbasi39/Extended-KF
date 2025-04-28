#include "ExtendedKalmanFilter.hpp"

ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::Vector4d& initial_state,
                                           const Eigen::Matrix4d& initial_covariance,
                                           const Eigen::Matrix4d& process_noise,
                                           const Eigen::Matrix2d& measurement_noise)
    : x_(initial_state),
      P_(initial_covariance),
      Q_(process_noise),
      R_(measurement_noise)
{}

Eigen::Matrix4d ExtendedKalmanFilter::stateTransitionMatrix(double dt) const {
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;
    return F;
}

Eigen::Vector4d ExtendedKalmanFilter::statePrediction(const Eigen::Vector2d& u, double dt) const {
    Eigen::Vector4d x_pred = x_;
    x_pred(0) += x_(2) * dt;
    x_pred(1) += x_(3) * dt;
    x_pred(2) += u(0);
    x_pred(3) += u(1);
    return x_pred;
}

void ExtendedKalmanFilter::predict(const Eigen::Vector2d& u, double dt) {
    Eigen::Matrix4d F = stateTransitionMatrix(dt);
    x_ = statePrediction(u, dt);
    P_ = F * P_ * F.transpose() + Q_;
}

void ExtendedKalmanFilter::update(const Eigen::Vector2d& z) {
    Eigen::Matrix<double, 2, 4> H = measurementJacobian(x_);
    Eigen::Vector2d y = z - measurementPrediction();
    Eigen::Matrix2d S = H * P_ * H.transpose() + R_;
    Eigen::Matrix<double, 4, 2> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::Matrix4d::Identity() - K * H) * P_;
}
