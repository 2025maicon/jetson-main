"""
PID 컨트롤러 모듈
"""
import numpy as np


class PIDController:
    def __init__(self, kp=0.6, ki=0.0, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        self.integral = np.clip(self.integral, -200, 200)
        return self.kp * error + self.ki * self.integral + self.kd * derivative

