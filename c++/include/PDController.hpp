#ifndef PD_CONTROLLER_HPP
#define PD_CONTROLLER_HPP

#include <Eigen/Dense>

class PDController {
public:
    PDController(double kp = 2.0, double kd = 0.5);

    Eigen::Vector2d computeControl(double error, double velocity) const;

private:
    double kp_;
    double kd_;
};

#endif // PD_CONTROLLER_HPP
