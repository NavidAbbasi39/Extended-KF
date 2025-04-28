#ifndef EXTENDED_KALMAN_FILTER_HPP
#define EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(const Eigen::Vector4d& initial_state,
                         const Eigen::Matrix4d& initial_covariance,
                         const Eigen::Matrix4d& process_noise,
                         const Eigen::Matrix2d& measurement_noise);

    virtual ~ExtendedKalmanFilter() = default;

    virtual Eigen::Vector2d measurementPrediction() const = 0;
    virtual Eigen::Matrix<double, 2, 4> measurementJacobian(const Eigen::Vector4d& x) const = 0;

    void predict(const Eigen::Vector2d& u, double dt);
    void update(const Eigen::Vector2d& z);

    Eigen::Vector4d getState() const { return x_; }
    Eigen::Matrix4d getCovariance() const { return P_; }

protected:
    Eigen::Vector4d x_; // State vector [theta1, theta2, theta1_dot, theta2_dot]
    Eigen::Matrix4d P_; // Covariance matrix
    Eigen::Matrix4d Q_; // Process noise covariance
    Eigen::Matrix2d R_; // Measurement noise covariance

    Eigen::Matrix4d stateTransitionMatrix(double dt) const;
    Eigen::Vector4d statePrediction(const Eigen::Vector2d& u, double dt) const;
};
#endif // EXTENDED_KALMAN_FILTER_HPP
