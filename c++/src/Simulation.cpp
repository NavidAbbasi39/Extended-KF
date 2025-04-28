#include "Simulation.hpp"
#include <iostream>
#include <random>

Simulation::Simulation(double l1, double l2, double dt, double t_span)
    : l1_(l1), l2_(l2), dt_(dt), t_span_(t_span),
      arm_(l1, l2,
           Eigen::Vector4d::Zero(),
           Eigen::Matrix4d::Identity() * 0.1,
           (Eigen::Matrix4d() << 0.01,0,0,0,
                                0,0.01,0,0,
                                0,0,0.1,0,
                                0,0,0,0.1).finished(),
           (Eigen::Matrix2d() << 0.05,0,
                                0,0.05).finished())
{}

Eigen::VectorXd Simulation::desiredTrajectory(double t) const {
    Eigen::VectorXd traj(1);
    traj(0) = std::sin(t);
    return traj;
}

void Simulation::run() {
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 0.1);

    int steps = static_cast<int>(t_span_ / dt_);
    for (int i = 0; i < steps; ++i) {
        double time = i * dt_;
        double desired_y = std::sin(time);

        Eigen::Vector2d current_pos = arm_.forwardKinematics(arm_.getState().head<2>());
        double y_error = desired_y - current_pos(1);
        double velocity = arm_.getState()(3);

        Eigen::Vector2d u = controller_.computeControl(y_error, velocity);

        arm_.predict(u, dt_);

        Eigen::Vector2d true_pos = arm_.forwardKinematics(arm_.getState().head<2>());
        Eigen::Vector2d measurement = true_pos;
        measurement(0) += noise(generator);
        measurement(1) += noise(generator);

        arm_.update(measurement);

        data_.time.push_back(time);
        data_.desired.push_back(desired_y);
        data_.actual.push_back(current_pos(1));
        data_.estimates.push_back(arm_.getState());
    }
}

void Simulation::plotResults() const {
    // Optional: Implement plotting with matplotlib-cpp or export data to file
    // For now, just print final RMSE
    double error_sum = 0.0;
    int n = data_.time.size();
    for (int i = 0; i < n; ++i) {
        double err = data_.desired[i] - data_.actual[i];
        error_sum += err * err;
    }
    double rmse = std::sqrt(error_sum / n);
    std::cout << "RMSE of Y position tracking: " << rmse << std::endl;
}
