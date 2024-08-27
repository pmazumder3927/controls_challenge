from . import BaseController
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from tinyphysics import TinyPhysicsModel


class Controller(BaseController):
    """
    Enhanced PID controller with feedforward and adaptive gains.
    Auto-tunes the PID parameters at the end of the sequence.
    """

    def __init__(self, params=None):
        # Base PID coefficients # Optimized parameters: [0.06744017775885948, 0.09314033350306246, 0.0, 0.0031577190353879662, 0.011581227393877278, 0.2425812159010433] # [0.1516204493886207, 0.13050045662241008, 0.0, 0.006270893079117299, 0.0, 0.06090872179859569]
        # vals = [0.001, 0.0815933350197024, 0.0,
        #         0.008451808612309042, 0.0, 0.3034815685965544]
        if params is None:
            # params = [0.4452572712849652, 0.04547310934503163, -0.18287994166675903, 0,
            #           0.9553235189335525, 0.2990300599986359, 0.3821302815058562, 0.039428439968725144, 0.0]
            params = [0.1501367432443675, 0.13381722139955815, 0.03170426837311169, 0, 0.48436704088987936,
                      0.25514179605094733, 0.3708077275948347, 0.12014341097884755, 0.010216229793396772]
        self.p_base, self.i_base, self.d_base, self.future_feedforward_weight, self.a, self.b, self.c, self.d, self.v_ego_gain = params
        self.error_integral = 0
        self.prev_error = 0
        self.action_history = []

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        step = len(self.action_history)
        # Regular PID calculation
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        max_integral = 10.0  # Example value, adjust based on your system
        self.error_integral = np.clip(
            self.error_integral, -max_integral, max_integral)
        error_diff = error - self.prev_error
        self.prev_error = error
        error_magnitude = np.abs(error)

        # Adaptive gains based on vehicle state
        p_gain = self.p_base + self.v_ego_gain * state.v_ego
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
        # make sure control input isn't too far away from last action
        last_action = self.action_history[-1] if len(
            self.action_history) > 0 else 0
        self.action_history.append(control_input)

        # 100% last input -> 100% control input over 100 steps
        n_blend = 200
        if (step < n_blend):
            control_blended = target_lataccel * \
                (1 - (step * 1/n_blend)) + control_input * (step * 1/n_blend)
            control_input = control_blended
        return control_input

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
