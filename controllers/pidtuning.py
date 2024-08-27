from . import BaseController
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from tinyphysics import TinyPhysicsModel


class TuningController(BaseController):
    """
    Enhanced PID controller with feedforward and adaptive gains.
    Auto-tunes the PID parameters at the end of the sequence.
    """

    def __init__(self, params):
        self.p_base, self.i_base, self.d_base, self.future_feedforward_weight, self.a, self.b, self.c, self.d = params
        self.error_integral = 0
        self.prev_error = 0

        # Storage for optimization
        self.target_lataccel_history = []
        self.current_lataccel_history = []
        self.state_history = []
        self.future_plan_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Store data for optimization
        self.target_lataccel_history.append(target_lataccel)
        self.current_lataccel_history.append(current_lataccel)
        self.state_history.append(state)
        self.future_plan_history.append(future_plan)

        # Regular PID calculation
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Adaptive gains based on vehicle state
        p_gain = self.p_base
        i_gain = self.i_base
        d_gain = self.d_base
        # nonlinear feedforward
        nl_feedforward = self.sigmoid(
            current_lataccel*self.a) * self.b + current_lataccel * self.c + self.d

        # Simple PID control with adaptive gains
        pid = p_gain * error + i_gain * self.error_integral + d_gain * error_diff

        # Incorporating a simple feedforward control based on the future plan
        if len(future_plan.lataccel) > 0:
            future_lataccel = np.mean(future_plan.lataccel)
            feedforward = future_lataccel * self.future_feedforward_weight
        else:
            feedforward = 0

        # Combine PID and feedforward
        control_input = pid + feedforward + nl_feedforward
        # # Perform auto-tuning at the end of the sequence
        # if len(future_plan.lataccel) == 0:
        #     self.bayesian_optimize()
        return control_input
