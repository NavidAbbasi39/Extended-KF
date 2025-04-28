#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "RobotArm.hpp"
#include "PDController.hpp"
#include <Eigen/Dense>
#include <vector>

struct SimulationData {
    std::vector<double> time;
    std::vector<double> desired;
    std::vector<double> actual;
    std::vector<Eigen::Vector4d> estimates;
};

class Simulation {
public:
    Simulation(double l1, double l2, double dt, double t_span);

    void run();
    void plotResults() const; // Optional, requires matplotlib-cpp or other plotting

private:
    double l1_, l2_, dt_, t_span_;
    RobotArm arm_;
    PDController controller_;
    SimulationData data_;

    Eigen::VectorXd desiredTrajectory(double t) const;
};

#endif // SIMULATION_HPP
