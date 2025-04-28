#include "Simulation.hpp"

int main() {
    double l1 = 1.0;
    double l2 = 1.0;
    double dt = 0.02;
    double t_span = 10.0;

    Simulation sim(l1, l2, dt, t_span);
    sim.run();
    sim.plotResults();

    return 0;
}
