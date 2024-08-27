from skopt import gp_minimize
from . import BaseController
import numpy as np
from collections import deque
from tinyphysics import FuturePlan, State, TinyPhysicsModel
import pyswarms as ps
from skopt.space import Real
MAX_ACC_DELTA = 0.5
CONTEXT_WINDOW_SIZE = 20


class Controller(BaseController):
    """
    A simple controller to validate ONNX model predictions against simulator outputs,
    handling the history required for ONNX model input.
    """

    def __init__(self, horizon=10, dt=0.1):
        self.model = TinyPhysicsModel('./models/tinyphysics.onnx', True)
        # Prediction horizon (in this case, typically 1 for step-by-step comparison)
        self.horizon = 50
        self.dt = dt

        # Deques to maintain a history of the last 20 states, actions, and past predictions
        self.state_history = deque(maxlen=20)
        self.action_history = deque(maxlen=20)
        self.past_preds_history = deque(maxlen=20)
        self.error_integral = 0
        self.pid_params = [0.1501367432443675, 0.134, 0.03170426837311169, 0, 0.48436704088987936,
                           0.25514179605094733, 0.3708077275948347, 0.12014341097884755, 0.010216229793396772]
        self.n_params = len(self.pid_params)
        self.prev_error = 0
        self.step = 0

    def predict_lataccel(self, current_lataccel, current_state, control_input, branch_state_history=None, branch_action_history=None, branch_past_preds_history=None):
        """
        Use the ONNX model to predict the next lataccel value.
        Ensure that the input history has the required length of 20.
        """
        if len(self.state_history) < 20 or len(self.action_history) < 20 or len(self.past_preds_history) < 20:
            # If history is not sufficient, use zero-padding or replicate the first entry
            # until the length is 20
            padding_state = [current_state] * (20 - len(self.state_history))
            padding_action = [control_input] * (20 - len(self.action_history))
            padding_pred = [0.0] * (20 - len(self.past_preds_history))
            states = padding_state + list(self.state_history)
            actions = padding_action + list(self.action_history)
            past_preds = padding_pred + list(self.past_preds_history)
        else:
            states = list(self.state_history)
            actions = list(self.action_history)
            past_preds = list(self.past_preds_history)

            if branch_state_history is not None:
                states = (states + branch_state_history)[-CONTEXT_WINDOW_SIZE:]
            if branch_action_history is not None:
                actions = (
                    actions + branch_action_history)[-CONTEXT_WINDOW_SIZE:]
            if branch_past_preds_history is not None:
                past_preds = (
                    past_preds + branch_past_preds_history)[-CONTEXT_WINDOW_SIZE:]
        # Convert lists to the required format for ONNX model
        states, actions, past_preds = np.array(
            states), np.array(actions), np.array(past_preds)
        predicted_lataccel = self.model.get_current_lataccel(
            sim_states=states,
            actions=actions,
            past_preds=past_preds
        )
        predicted_lataccel = np.clip(predicted_lataccel, current_lataccel - MAX_ACC_DELTA,
                                     current_lataccel + MAX_ACC_DELTA)
        return predicted_lataccel

    # def cost_function(self, control_sequence, current_lataccel, state, future_plan):
    #     # Initialize state and cost
    #     control_sequence = control_sequence[0]
    #     total_lataccel_cost = 0
    #     total_jerk_cost = 0
    #     branch_state_history = []
    #     branch_action_history = []
    #     branch_past_preds_history = []
    #     for t in range(self.horizon):
    #         predicted_lataccel = self.predict_lataccel(
    #             current_lataccel, state, control_sequence[t], branch_state_history, branch_action_history, branch_past_preds_history)
    #         branch_action_history.append(control_sequence[t])
    #         branch_state_history.append(state)
    #         branch_past_preds_history.append(current_lataccel)

    #         # Calculate the cost based on lataccel and jerk
    #         lataccel_cost = (
    #             (predicted_lataccel - future_plan.lataccel[t]) ** 2) * 100 / self.horizon
    #         if t > 0:
    #             jerk_cost = (((predicted_lataccel - branch_past_preds_history[t]) /
    #                          self.dt) ** 2) * 100 / (self.horizon - 1)
    #         else:
    #             jerk_cost = 0

    #         total_lataccel_cost += lataccel_cost
    #         total_jerk_cost += jerk_cost

    #         # Update state from future plan, which is just the same namedtuple with an extra field for lataccel, so just remove the lataccel field
    #         state = State(
    #             roll_lataccel=future_plan.roll_lataccel[t],
    #             v_ego=future_plan.v_ego[t],
    #             a_ego=future_plan.a_ego[t],
    #         )
    #         branch_state_history.append(state)
    #     # Total cost
    #     total_cost = (total_lataccel_cost * 50) + total_jerk_cost
    #     return total_cost
    def cost_function(self, params, current_lataccel, state, future_plan):
        # Initialize state and cost
        total_lataccel_cost = 0
        total_jerk_cost = 0
        branch_state_history = []
        branch_action_history = []
        branch_past_preds_history = []
        control_sequence = self.pid_rollout(
            params, current_lataccel, state, future_plan)
        for t in range(len(future_plan.lataccel)):
            predicted_lataccel = self.predict_lataccel(
                current_lataccel, state, control_sequence[t], branch_state_history, branch_action_history, branch_past_preds_history)
            branch_action_history.append(control_sequence[t])
            branch_state_history.append(state)
            branch_past_preds_history.append(current_lataccel)

            # Calculate the cost based on lataccel and jerk
            lataccel_cost = (
                (predicted_lataccel - future_plan.lataccel[t]) ** 2) * 100 / self.horizon
            if t > 0:
                jerk_cost = (((predicted_lataccel - branch_past_preds_history[t]) /
                             self.dt) ** 2) * 100 / (self.horizon - 1)
            else:
                jerk_cost = 0

            total_lataccel_cost += lataccel_cost
            total_jerk_cost += jerk_cost

            # Update state from future plan, which is just the same namedtuple with an extra field for lataccel, so just remove the lataccel field
            state = State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t],
            )
            branch_state_history.append(state)
        # Total cost
        total_cost = (total_lataccel_cost * 50) + total_jerk_cost
        return total_cost

    def pid_rollout(self, params, current_lataccel, state, future_plan):
        pid_position = [0.0] * self.horizon
        branch_state_history = []
        branch_action_history = []
        branch_past_preds_history = []
        sim_error_integral = self.error_integral
        sim_prev_error = self.prev_error
        p, i, d, future_feedforward_weight, a, b, c, d, v_ego_gain = params

        sim_current_lataccel = current_lataccel
        sim_current_state = state
        for t in range(len(future_plan.lataccel)):
            error = (future_plan.lataccel[t] - sim_current_lataccel)
            sim_error_integral += error
            error_diff = error - sim_prev_error
            sim_prev_error = error

            # Adaptive gains based on vehicle state
            p_gain = p + v_ego_gain * sim_current_state.v_ego
            i_gain = i
            d_gain = d

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            nl_feedforward = sigmoid(
                sim_current_lataccel*a) * b + sim_current_lataccel * c + d

            pid = p_gain * error + i_gain * sim_error_integral + d_gain * error_diff

            # Incorporating a simple feedforward control based on the future plan
            if len(future_plan.lataccel) > 0:
                future_lataccel = np.mean(future_plan.lataccel)
                feedforward = future_lataccel * future_feedforward_weight
            else:
                feedforward = 0

            # Combine PID and feedforward
            control_input = pid + feedforward + nl_feedforward
            pid_position[t] = control_input
            branch_action_history.append(control_input)
            next_lataccel = self.predict_lataccel(
                sim_current_lataccel, state, control_input, branch_state_history, branch_action_history, branch_past_preds_history)
            sim_current_lataccel = next_lataccel
            branch_state_history.append(sim_current_state)
            branch_past_preds_history.append(next_lataccel)
            sim_current_state = State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t],
            )
        return pid_position

    def find_optimal_control_input(self, params, current_lataccel, state, future_plan, step=0):
        space = [
            Real(0.001, 0.5, name='p_base'),
            Real(0.0, 1.0, name='i_base'),
            Real(-1, 1.0, name='d_base'),
            Real(0.0, 0.5, name='future_feedforward_weight'),
            Real(0.0, 1.0, name='a'),
            Real(-1.0, 1.0, name='b'),
            Real(-1.0, 1.0, name='c'),
            Real(-1.0, 1.0, name='d'),
            Real(0.0, 1.0, name='v_ego_gain')
        ]
        # Define bounds for control inputs (e.g., steering angle limits)
        bounds = ([0] * self.n_params, [0.5] * self.n_params)
        pid_trajectory = self.pid_rollout(
            self.pid_params, current_lataccel, state, future_plan)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'init_pos': self.pid_params}
        # Run PSO optimization
        if (step % 50 == 0 and step > 100):
            print("Optimizing PID parameters")
            # Run optimization with args
            def optimize_fn(params): return self.cost_function(
                params, current_lataccel, state, future_plan)
            result = gp_minimize(optimize_fn,
                                 space, n_calls=100, random_state=42, verbose=True, x0=self.pid_params)
            # optimizer = ps.single.GlobalBestPSO(
            #     n_particles=100,
            #     dimensions=self.n_params,
            #     bounds=bounds,
            #     options=options,
            # )
            # best_cost, best_sequence = optimizer.optimize(
            #     self.cost_function, iters=100, current_lataccel=current_lataccel, state=state, future_plan=future_plan)
            self.pid_params = result.x
            print("Cost: ", result.fun)
            print("Optimized PID parameters")
        return pid_trajectory

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Update the controller, predict the lataccel using the ONNX model, and compare it to the actual lataccel.
        """
        # prepend target_lataccel to future_plan.lataccel
        horizon_future_plan = FuturePlan(
            lataccel=[target_lataccel] + future_plan.lataccel,
            roll_lataccel=[state.roll_lataccel] + future_plan.roll_lataccel,
            v_ego=[state.v_ego] + future_plan.v_ego,
            a_ego=[state.a_ego] + future_plan.a_ego
        )
        control_sequence = self.find_optimal_control_input(
            self.pid_params, current_lataccel, state, horizon_future_plan, self.step)
        control_input = control_sequence[0]
        # Update history with the current step data
        self.error_integral += (target_lataccel - current_lataccel)
        self.state_history.append(state)
        self.action_history.append(control_input)
        self.past_preds_history.append(current_lataccel)
        self.prev_error = target_lataccel - current_lataccel
        self.step += 1

        # Predict next lataccel
        # predicted_lataccel = self.predict_lataccel(
        # current_lataccel, state, control_input)
        # if len(self.past_preds_history) > 0:
        #     # The return value can be the difference for analysis
        #     prediction_error = self.past_preds_history[-1] - current_lataccel
        #     print(
        #         f"Predicted lataccel: {self.past_preds_history[-1]:.4f}, Actual lataccel: {current_lataccel:.4f}, Error: {prediction_error:.4f}")
        # Log or print the results for debugging

        # Return the prediction error or any other information needed
        return control_input
