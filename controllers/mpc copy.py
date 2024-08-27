from collections import namedtuple
import onnxruntime as ort
from typing import List, Union, Tuple, Dict
import numpy as np
import cvxpy as cp
import pandas as pd
from tinyphysics import LataccelTokenizer, TinyPhysicsModel
import os

# Create a DataFrame to store the inputs and outputs


class Controller:
    def __init__(self, horizon=10, dt=0.1):
        self.model_context_window = 20

        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.prev_error = 0
        self.error_integral = 0

        self.horizon = horizon
        self.dt = dt
        self.tokenizer = LataccelTokenizer()
        self.model = TinyPhysicsModel('models/tinyphysics.onnx', debug=True)

        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.last_action = 0
        # load data if it exists
        if os.path.exists('bicycle_model_data.csv'):
            self.data = pd.read_csv('bicycle_model_data.csv')
        else:
            self.data = pd.DataFrame(columns=['v_ego', 'a_ego', 'roll_lataccel',
                                              'current_lataccel', 'steer_action', 'predicted_lataccel'])
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if len(self.state_history) < self.model_context_window:
            # PID fallback when context is not enough
            error = target_lataccel - current_lataccel
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            pid = self.p * error + self.i * self.error_integral + self.d * error_diff

            # Update histories
            self.state_history.append(state)
            self.action_history.append(self.last_action)
            self.current_lataccel_history.append(current_lataccel)
            self.last_action = pid
            return pid

        predicted_lataccel = self.model.get_current_lataccel(
            self.state_history[-self.model_context_window:], self.action_history[-self.model_context_window:], self.current_lataccel_history[-self.model_context_window:])

        # Store the input-output pair
        last_used_state = self.state_history[-1]
        self.data = pd.concat([self.data, pd.DataFrame({
            'v_ego': last_used_state.v_ego,
            'a_ego': last_used_state.a_ego,
            'roll_lataccel': last_used_state.roll_lataccel,
            'current_lataccel': current_lataccel,
            'steer_action': self.last_action,
            'predicted_lataccel': predicted_lataccel
        }, index=[len(self.data)])])

        # Save the collected data to a CSV file for further analysis
        self.data.to_csv('bicycle_model_data.csv', index=False)
        # Initialize control variables for the entire horizon
        steer_actions = cp.Variable(self.horizon)
        # return randoms steer action between -2 and 2
        steer_action = np.random.uniform(-2, 2)
        self.state_history.append(state)
        self.action_history.append(steer_action)
        self.current_lataccel_history.append(current_lataccel)
        return steer_action

        # Initialize the predicted lataccel as a variable to be constrained by the model
        lataccel = cp.Variable(self.horizon)

        # Start from the current state
        sim_state = state
        constraints = []

        # Define constraints over the horizon
        sim_history = self.state_history[-self.model_context_window - 1:] + [
            sim_state]
        for t in range(self.horizon):
            # Add constraint for the system dynamics using the model
            actions = self.action_history[-self.model_context_window:] + [
                steer_actions[t]]
            pred_lataccel = self.model.get_current_lataccel(
                sim_states=sim_history,
                actions=self.action_history[-self.model_context_window:] + [
                    steer_actions[t]],
                past_preds=self.current_lataccel_history[-self.model_context_window:]
            )
            constraints.append(
                lataccel[t] == pred_lataccel
            )
            sim_history.append(self.State(
                roll_lataccel=pred_lataccel,
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            ))
        # Convert the future_plan.lataccel list to a cvxpy Parameter
        future_lataccel = cp.Parameter(
            self.horizon, value=np.array(future_plan.lataccel[:self.horizon]))

        # Calculate jerk (change in lataccel) and construct the cost function
        jerk = cp.diff(lataccel) / self.dt
        cost = cp.sum_squares(lataccel - future_lataccel) * \
            50 + cp.sum_squares(jerk)

        # Add constraints for steering limits
        constraints += [steer_actions <= 1, steer_actions >= -1]

        # Solve the optimization problem to get the best sequence of actions
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("MPC failed to find an optimal solution.")
            return self.last_action  # If the optimization fails, fall back to the last action

        # Apply only the first action in the sequence
        optimal_steer_action = steer_actions.value[0]

        # Update histories for the next iteration
        self.state_history.append(state)
        self.action_history.append(optimal_steer_action)
        self.current_lataccel_history.append(lataccel.value[0])

        self.last_action = optimal_steer_action
        return optimal_steer_action
