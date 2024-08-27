from mlp import MLPFeedForward
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
            vals = [0.001, 0.10626976158279121, 0.0848612290131688,
                    0.004368975949480424, 0.0, 0.35169201254981064]
        else:
            vals = params
        self.p_base = vals[0]
        self.i_base = vals[1]
        self.d_base = vals[2]
        self.error_integral = 0.0
        self.error_zone = 0.02
        self.prev_error = 0

        # Adaptive gain parameters
        # self.v_ego_gain = vals[3]

        # Adjusts based on roll-induced lataccel
        # self.roll_lataccel_gain = vals[4]
        self.future_feedforward_weight = vals[3]
        self.mlp = MLPFeedForward(
            model_path='./mlp_model.pth', scaler_path='./scaler.pkl')

        # Storage for optimization
        self.target_lataccel_history = []

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Regular PID calculation
        error = (target_lataccel - current_lataccel)
        # if abs(error) < self.error_zone:
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # Adaptive gains based on vehicle state
        p_gain = self.p_base
        i_gain = self.i_base
        d_gain = self.d_base

        pid = p_gain * error + i_gain * self.error_integral + d_gain * error_diff

        # Incorporating a simple feedforward control based on the future plan
        if len(future_plan.lataccel) > 0:
            future_lataccel = np.mean(future_plan.lataccel)
            feedforward = future_lataccel * self.future_feedforward_weight
        else:
            feedforward = 0

        state = np.array(
            [state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel])
        # Combine PID and feedforward
        control_input = pid + feedforward + self.mlp.infer(state)
        self.target_lataccel_history.append(target_lataccel)
        # if (len(self.target_lataccel_history) < 110):
        #     return self.target_lataccel_history[-1]
        return control_input
