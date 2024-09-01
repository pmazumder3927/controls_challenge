import numpy as np
from mlp import MLPFeedForward
from . import BaseController


class MFC:
    def __init__(self, v_x0, alpha_0, K_alpha, T_s, C):
        self.v_x0 = v_x0
        self.alpha_0 = alpha_0
        self.K_alpha = K_alpha
        self.T_s = T_s
        self.C = C

    def get_alpha_adaptation(self, v_x, a_ego):
        # Adjust alpha based on v_x and a_ego
        predicted_v_ego = v_x + a_ego * self.T_s
        base_alpha = self.alpha_0 if predicted_v_ego < self.v_x0 else self.K_alpha * \
            (predicted_v_ego - self.v_x0) + self.alpha_0
        return base_alpha

    def get_control_input(self, v_x, a_ego, lat_accel, lat_accel_dot, lat_accel_dotdot, target_lataccel_dotdot, u_last, error_dot, error, K_p, K_d):
        alpha = self.get_alpha_adaptation(v_x, a_ego)
        F_hat = self.calculate_F_hat(lat_accel_dotdot, alpha, u_last)
        y_dotdot_1r = target_lataccel_dotdot

        # Control input considering the modified alpha
        numerator = -F_hat + y_dotdot_1r + K_p * error + K_d * error_dot
        denominator = alpha
        return numerator / denominator

    def calculate_F_hat(self, lat_accel_dotdot, alpha, u_last):
        # Using second-order dynamics with lat_accel_dotdot
        return lat_accel_dotdot - alpha * u_last


class Controller(BaseController):
    def __init__(self, params=None, filter_coeff=0.5):
        if params is None:
            params = [6, 849.4484712613763, 161.25673688567215,
                      1090.1007100030786, 61.650336357179675]

        # Tuned parameters
        self.v_x0 = params[0]
        self.alpha_0 = params[1]
        self.K_alpha = params[2]
        self.K_p = params[3]
        self.K_d = params[4]

        # Static parameters
        self.T_s = 0.1
        self.C = 1.5

        # Create an instance of MFC with the static parameters
        self.mfc = MFC(self.v_x0, self.alpha_0, self.K_alpha,
                       self.T_s, self.C)
        self.mlp_ff = MLPFeedForward(
            model_path='./mlp_model.pth', scaler_path='./scaler.pkl')

        # Dynamic parameters
        self.error_integral = 0
        self.prev_error = 0

        # Internal state
        self.last_target_lataccel = 0
        self.last_error = 0
        self.last_control_input = -1
        self.last_target_lataccel_dot = 0
        self.last_lat_accel = 0
        self.last_lat_accel_dot = 0
        self.last_lat_accel_dotdot = 0
        self.lat_accel_history = []
        self.last_state = None
        self.step_count = 0

    def update_params(self, params):
        self.K_p = params[0]
        self.K_d = params[1]
        self.K_alpha = params[2]
        self.v_x0 = params[3]
        self.alpha_0 = params[4]
        self.C = params[5]

    def filter_value(self, new_value, old_value):
        # Low-pass filter for smoothing the value
        return self.C * new_value + (1 - self.C) * old_value

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1

        # Ramp-up strategy for the first few steps
        # Gradually increases from 0 to 1 over the first 20 steps
        ramp_factor = min(1.0, self.step_count / 10.0)

        error = target_lataccel - current_lataccel
        target_lataccel_dot = (
            target_lataccel - self.last_target_lataccel) / self.T_s,
        target_lataccel_dotdot = (
            target_lataccel_dot - self.last_target_lataccel_dot) / self.T_s,
        error_dot = self.filter_value(
            (error - self.last_error) / self.T_s,
            self.last_lat_accel_dot
        )
        lat_accel_dot = self.filter_value(
            (current_lataccel - self.last_lat_accel) / self.T_s,
            self.last_lat_accel_dot
        )
        lat_accel_dotdot = self.filter_value(
            (lat_accel_dot - self.last_lat_accel_dot) / self.T_s,
            self.last_lat_accel_dotdot
        )

        mlp_input = np.array(
            [state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel])
        mlp_ff = self.mlp_ff.infer(mlp_input)
        control_input = self.mfc.get_control_input(
            state.v_ego, state.a_ego, current_lataccel, lat_accel_dot, lat_accel_dotdot, target_lataccel_dotdot,
            self.last_control_input, error_dot, error, self.K_p, self.K_d
        )

        control_input = ramp_factor * \
            control_input + (1 - ramp_factor) * mlp_ff

        # Update internal state
        self.last_target_lataccel = target_lataccel
        self.last_target_lataccel_dot = target_lataccel_dot
        self.lat_accel_history.append(current_lataccel)
        self.last_lat_accel = current_lataccel
        self.last_lat_accel_dot = lat_accel_dot
        self.last_lat_accel_dotdot = lat_accel_dotdot
        self.last_error = error
        self.last_control_input = np.clip(control_input, -1, 1)
        self.last_state = state
        return control_input
