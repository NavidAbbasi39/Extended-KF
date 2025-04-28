import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
from collections import defaultdict

@dataclass
class SystemParameters:
    l1: float = 1.0
    l2: float = 1.0
    dt: float = 0.02
    process_noise: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01, 0.1, 0.1]))
    measurement_noise: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.05]))
    initial_state: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    initial_covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 0.1)

class ExtendedKalmanFilter:
    def __init__(self, params: SystemParameters): #Constructor for Initialization 
        self.x = params.initial_state.copy()
        self.P = params.initial_covariance.copy()
        self.Q = params.process_noise.copy()
        self.R = params.measurement_noise.copy()
        
    def predict(self, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:

        F = self._state_transition_matrix(dt)
        self.x = self._state_prediction(u, dt)
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P
    
    def update(self, z: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        y = z - self._measurement_prediction()
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x, self.P
    
    def _state_transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F
    
    def _state_prediction(self, u: np.ndarray, dt: float) -> np.ndarray:
        return self.x + np.array([
            self.x[2]*dt,
            self.x[3]*dt,
            u[0],
            u[1]
        ])
    
    def _measurement_prediction(self) -> np.ndarray:
        raise NotImplementedError("Measurement prediction must be implemented in subclass")

class RobotArm(ExtendedKalmanFilter):
    def __init__(self, params: SystemParameters):
        super().__init__(params)
        self.l1 = params.l1
        self.l2 = params.l2
        
    def forward_kinematics(self, theta: np.ndarray) -> np.ndarray:
        x = self.l1*np.cos(theta[0]) + self.l2*np.cos(theta[0]+theta[1])
        y = self.l1*np.sin(theta[0]) + self.l2*np.sin(theta[0]+theta[1])
        return np.array([x, y])
    
    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        H = np.zeros((2, 4))
        H[0,0] = -self.l1*np.sin(theta[0]) - self.l2*np.sin(theta[0]+theta[1])
        H[0,1] = -self.l2*np.sin(theta[0]+theta[1])
        H[1,0] = self.l1*np.cos(theta[0]) + self.l2*np.cos(theta[0]+theta[1])
        H[1,1] = self.l2*np.cos(theta[0]+theta[1])
        return H
    
    def _measurement_prediction(self) -> np.ndarray:
        return self.forward_kinematics(self.x[:2])

class PDController:
    def __init__(self, kp: float = 2.0, kd: float = 0.5):
        self.kp = kp
        self.kd = kd
        
    def compute_control(self, error: float, velocity: float) -> np.ndarray:
        return np.array([0.0, self.kp*error + self.kd*(-velocity)])

class Simulation:
    def __init__(self, params: SystemParameters):
        self.params = params
        self.arm = RobotArm(params)
        self.controller = PDController()
        self.history = defaultdict(list)
        
    def run(self, t_span: float = 10.0) -> Dict[str, List[float]]:
        t = np.arange(0, t_span, self.params.dt)
        desired_trajectory = np.sin(t)
        
        for i, time in enumerate(t):
            # Control Law
            current_position = self.arm.forward_kinematics(self.arm.x[:2])
            y_error = desired_trajectory[i] - current_position[1]
            u = self.controller.compute_control(y_error, self.arm.x[3])
            
            # EKF Prediction
            x_pred, P_pred = self.arm.predict(u, self.params.dt)
            
            # Simulate measurement with noise
            true_position = self.arm.forward_kinematics(x_pred[:2])
            measurement = true_position + np.random.normal(0, 0.1, 2)
            
            # EKF Update
            H = self.arm.jacobian(x_pred[:2])
            x_est, P_est = self.arm.update(measurement, H)
            
            # Log data
            self.history['time'].append(time)
            self.history['desired'].append(desired_trajectory[i])
            self.history['actual'].append(current_position[1])
            self.history['estimate'].append(x_est.copy())
            
        return self.history
    
    def plot_results(self):
        """Visualize tracking performance"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['time'], self.history['desired'], 'r--', label='Desired trajectory')
        plt.plot(self.history['time'], self.history['actual'], 'b-', label='Estimated trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position')
        plt.title('Kalman Filter Tracking Performance')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    params = SystemParameters()
    sim = Simulation(params)
    sim.run(t_span=10)
    sim.plot_results()
