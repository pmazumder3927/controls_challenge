import numpy as np
from . import BaseController


class MFC:
    def __init__(self, v_x0, alpha_0, K_alpha, T_s, C):
        self.v_x0 = v_x0
        self.alpha_0 = alpha_0
        self.K_alpha = K_alpha
        self.T_s = T_s
        self.C = C

    def get_alpha_adaptation(self, v_x):
        if v_x < self.v_x0:
            return self.alpha_0
        else:
            return self.K_alpha * (v_x - self.v_x0) + self.alpha_0

    def get_control_input(self, v_x, lat_accel, target_lat_accel_dotdot, u_last, error_dot, error, K_p, K_d):
        alpha = self.get_alpha_adaptation(v_x)
        F_hat = self.calculate_F_hat(lat_accel, alpha, u_last)
        error_hat_dot = error_dot
        y_dotdot_1r = target_lat_accel_dotdot
        numerator = -F_hat + y_dotdot_1r + K_p * error + K_d * error_hat_dot
        denominator = alpha
        # print("K_d * error_dot: ", K_d * error_dot)
        # print("y dot dot 1r: ", y_dotdot_1r)
        # print("K_p * error: ", K_p * error)
        # print("denominator = alpha
        return numerator / denominator

    def calculate_F_hat(self, y_hat, alpha, u_last):
        y_hat_dotdot = self.get_y_dot_n(y_hat, 2)
        return y_hat_dotdot + alpha * u_last

    def get_y_dot_n(self, y, n):
        for i in range(n):
            y = self.calc_filtered_derivative(y)
        return y

    def calc_filtered_derivative(self, z):
        if z == 0:
            return 0
        return (1 - z**-1) / (self.C + (1 - self.C) * z**-1) * (1 / self.T_s)


class CurvatureFF:
    def __init__(self, R_s, delta_max, L):
        self.R_s = R_s
        self.delta_max = delta_max
        self.L = L

    def calculate_curvature_ff(self, kappa):
        return (self.R_s/self.delta_max) * np.arctan(kappa * self.L)


class Controller(BaseController):
    def __init__(self, params=None):
        # ff_params = [1, 1523.6414601143051, 41]
        if params is None:
            # params = [4, 5003.409199200207, 3168.1767305193544,
            #           9310.98532628715, -9.879507187366471]
            params = [0.0, 960.9541046119739, 92.40932169793723,
                      445.33831103189465, 474.94014255408996]

        # tuned parameters
        self.v_x0 = params[0]
        self.alpha_0 = params[1]
        self.K_alpha = params[2]
        self.K_p = params[3]
        self.K_d = params[4]

        # feedforward parameters
        # self.R_s = ff_params[0]
        # self.L = ff_params[1]
        # self.d_k = ff_params[2]

        # static
        self.delta_max = 2
        self.T_s = 0.1
        self.C = 1.5

        # Create an instance of MFC with the static parameters
        self.mfc = MFC(self.v_x0, self.alpha_0, self.K_alpha, self.T_s, self.C)
        # self.curvature_ff = CurvatureFF(self.R_s, self.delta_max, self.L)

        # Dynamic parameters
        self.error_integral = 0
        self.prev_error = 0

        # Internal state
        self.last_target_lataccel = 0
        self.last_error = 0
        self.last_control_input = -1
        self.last_target_lataccel_dot = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        target_lataccel_dot = (
            target_lataccel - self.last_target_lataccel)
        target_lataccel_dotdot = (
            target_lataccel_dot - self.last_target_lataccel_dot)
        error_dot = error - self.last_error

        # lookahead_curvature = 0
        # num_lookahead = min(self.d_k, len(future_plan.lataccel))
        # for i in range(num_lookahead):
        #     lookahead_curvature += future_plan.lataccel[i] / state.v_ego**2
        # lookahead_curvature = lookahead_curvature / \
        #     num_lookahead if num_lookahead > 0 else 0

        control_input = self.mfc.get_control_input(
            state.v_ego, current_lataccel, target_lataccel_dotdot,
            self.last_control_input, error_dot, error, self.K_p, self.K_d
        )  # + self.curvature_ff.calculate_curvature_ff(lookahead_curvature)

        # Update internal state
        self.last_target_lataccel = target_lataccel
        self.last_target_lataccel_dot = target_lataccel_dot
        self.last_error = error
        return control_input
