from collections import namedtuple
import onnxruntime as ort
from typing import List, Union, Tuple, Dict
import numpy as np
import cvxpy as cp
from tinyphysics import LataccelTokenizer, TinyPhysicsModel

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class Controller:
    def __init__(self, horizon=10, dt=0.1):
        self.model_context_window = 20

        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.prev_error = 0
        self.error_integral = 0

        # Fitted parameters from onnx model
        self.C_f = 20007.77  # Front cornering stiffness
        self.C_r = 20007.78  # Rear cornering stiffness
        self.l_f = 0.35      # Distance from CG to front axle
        self.l_r = 2.43      # Distance from CG to rear axle
        self.m = 1292.67     # Vehicle mass
        self.I_z = 2250.0    # Yaw moment of inertia

        self.horizon = horizon
        self.dt = dt
        self.tokenizer = LataccelTokenizer()
        self.model = TinyPhysicsModel('models/tinyphysics.onnx', debug=True)

        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.last_action = 0

    def estimate_yaw_rate_from_roll(self, roll_lataccel, v_ego, a_ego):
        """
        Estimate the yaw rate based on roll-induced lateral acceleration, longitudinal velocity, and acceleration.

        roll_lataccel: Roll-induced lateral acceleration.
        v_ego: Longitudinal velocity.
        a_ego: Longitudinal acceleration.

        Returns:
        --------
        Estimated yaw rate.
        """
        # Estimate yaw rate using a proportionality factor
        yaw_rate_estimate = roll_lataccel / v_ego

        # Adjust for longitudinal acceleration effects (simplified approximation)
        yaw_rate_estimate += a_ego / (v_ego * (self.l_f + self.l_r))

        return yaw_rate_estimate

    def predict_bicycle_state(self, delta, v_x, v_y, yaw_rate):
        """
        Predict the next state of the bicycle model given current state and control inputs.

        delta: Steering angle.
        v_x: Longitudinal velocity (assumed constant).
        v_y: Lateral velocity.
        yaw_rate: Yaw rate.

        Returns:
        --------
        Updated lateral velocity (v_y) and yaw rate (yaw_rate).
        """
        # Front and rear tire forces
        F_yf = self.C_f * (delta - (v_y + self.l_f * yaw_rate) / v_x)
        F_yr = self.C_r * (- (v_y - self.l_r * yaw_rate) / v_x)

        # Lateral velocity dynamics
        v_y_dot = (F_yf + F_yr) / self.m - v_x * yaw_rate

        # Yaw rate dynamics
        yaw_rate_dot = (self.l_f * F_yf - self.l_r * F_yr) / self.I_z

        # Update the state
        next_v_y = v_y + v_y_dot * self.dt
        next_yaw_rate = yaw_rate + yaw_rate_dot * self.dt

        return next_v_y, next_yaw_rate

    def predict_lataccel(self, v_x, next_v_y, next_yaw_rate):
        """
        Predict the next lateral acceleration using the bicycle model.

        delta: Steering angle.
        v_x: Longitudinal velocity (assumed constant).
        v_y: Lateral velocity.
        yaw_rate: Yaw rate.

        Returns:
        --------
        Predicted lateral acceleration (a_y).
        """

        # Calculate the lateral acceleration
        lat_accel = next_v_y + v_x * next_yaw_rate

        return lat_accel

    def check_against_onnx(self, delta, v_x, v_y, yaw_rate, onnx_lataccel):
        """
        Compare the predicted lateral acceleration with the ONNX model's prediction.

        delta: Steering angle.
        v_x: Longitudinal velocity (assumed constant).
        v_y: Lateral velocity.
        yaw_rate: Yaw rate.
        onnx_lataccel: Lateral acceleration predicted by the ONNX model.

        Returns:
        --------
        A comparison between the two predictions.
        """
        predicted_lataccel = self.predict_lataccel(delta, v_x, v_y, yaw_rate)
        difference = predicted_lataccel - onnx_lataccel
        return predicted_lataccel, difference

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid = self.p * error + self.i * self.error_integral + self.d * error_diff
        if len(self.state_history) < self.model_context_window:
            # PID fallback when context is not enough
            # Update histories
            self.state_history.append(state)
            self.action_history.append(self.last_action)
            self.current_lataccel_history.append(current_lataccel)
            self.last_action = pid
            return pid

        # Initialize control variables for the entire horizon
        steer_actions = cp.Variable(self.horizon)
        lataccel = cp.Variable(self.horizon)

        # Start from the current state
        lat_accel = current_lataccel
        yaw_rate = self.estimate_yaw_rate_from_roll(
            state.roll_lataccel, state.v_ego, state.a_ego)
        v_ego = state.v_ego
        constraints = []
        horizon_lataccel = [target_lataccel] + \
            future_plan.lataccel[:self.horizon - 1]

        # Define constraints over the horizon using the bicycle model
        for t in range(self.horizon):
            # Predict next lateral acceleration using the bicycle model
            print(self.action_history)
            # print(self.current_lataccel_history)
            # onnx_lataccel = self.model.get_current_lataccel(
            #     sim_states=self.state_history[:-self.model_context_window], actions=self.action_history[:-self.model_context_window], past_preds=self.current_lataccel_history[:-self.model_context_window])
            # Predict the next state of the bicycle model
            next_v_y, yaw_rate = self.predict_bicycle_state(
                steer_actions[t], v_ego, 0, yaw_rate)
            lat_accel = self.predict_lataccel(
                v_ego, next_v_y, yaw_rate)
            print('lat_accel from bicycle model: ', lat_accel)
            # print('onnx_lataccel: ', onnx_lataccel)
            # print('difference: ', lat_accel - onnx_lataccel)

            constraints.append(lataccel[t] == lat_accel)

        # Convert the future_plan.lataccel list to a cvxpy Parameter
        future_lataccel = cp.Parameter(
            self.horizon, value=np.array(horizon_lataccel))

        # Calculate jerk (change in lataccel) and construct the cost function
        jerk = cp.diff(lataccel) / self.dt
        # cost = 50*cp.sum_squares(lataccel -
        #                          future_lataccel) + cp.sum_squares(jerk)
        lataccel_error = cp.sum_squares(lataccel - future_lataccel) * 50
        jerk_error = cp.sum_squares(jerk)

        # Calculate cost function with weights
        cost = lataccel_error + jerk_error

        # Add constraints for steering limits
        constraints += [steer_actions <= 2, steer_actions >= -2]

        # Solve the optimization problem to get the best sequence of actions
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(warm_start=True, solver=cp.OSQP)
        print('lataccel - future_lataccel: ',
              lataccel.value - future_lataccel.value)
        print("lataccel_error: ", lataccel_error.value)
        print("jerk_error: ", jerk_error.value)
        print("cost: ", cost.value)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("MPC failed to find an optimal solution.")
            # If the optimization fails, fall back to the last action
            return self.action_history[-1]

        # Apply only the first action in the sequence
        optimal_steer_action = steer_actions.value[0]

        # Update histories for the next iteration
        self.state_history.append(state)
        self.action_history.append(optimal_steer_action)
        self.current_lataccel_history.append(lataccel.value[0])
        print("optimal_steer_action: ", optimal_steer_action)
        return optimal_steer_action
