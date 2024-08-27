import cvxpy as cp
import numpy as np
from collections import namedtuple
from tinyphysics import State, TinyPhysicsModel


class Controller:
    def __init__(self, dt=0.1):
        self.model_context_window = 20

        self.p = 0.05
        self.i = 0.00
        self.d = 0
        self.prev_error = 0
        self.error_integral = 0

        self.horizon = horizon
        self.dt = dt
        self.model = TinyPhysicsModel('models/tinyphysics.onnx', debug=True)

        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.last_action = 0

    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        future_lataccel, future_roll_lataccel, future_v_ego, future_a_ego = future_plan
        if len(self.state_history) < self.model_context_window or len(future_plan.lataccel) < self.model_context_window:
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

        # Initialize control variables for the entire horizon
        steer_actions = cp.Variable(self.horizon)

        # Initialize the predicted lataccel as a variable to be constrained by the model
        lataccel = cp.Variable(self.horizon)

        # Precompute lataccel predictions using the model
        horizon_lataccel = [target_lataccel] + \
            future_plan.lataccel[:self.horizon]

        sim_states = self.state_history[-self.model_context_window:] + [state]
        sim_lataccel = self.current_lataccel_history[-self.model_context_window:] + [
            current_lataccel]
        sim_actions = self.action_history[-self.model_context_window:]
        sim_last_error = 0
        sim_error_integral = self.error_integral
        for t in range(len(horizon_lataccel)):
            error = horizon_lataccel[t] - current_lataccel
            pid = self.p * error + self.i * sim_error_integral + \
                self.d * (error - sim_last_error)
            sim_actions.append(pid)

            pred = self.model.get_current_lataccel(
                sim_states=sim_states[-self.model_context_window:],
                actions=sim_actions[-self.model_context_window:],
                past_preds=sim_lataccel[-self.model_context_window:]
            )
            sim_states.append(
                State(roll_lataccel=future_roll_lataccel[t], v_ego=future_v_ego[t], a_ego=future_a_ego[t]))
            sim_lataccel.append(pred)
            sim_last_error = error
            sim_error_integral += error

        predicted_lataccel = np.array(sim_lataccel)

        constraints = []
        # Add constraints based on precomputed lataccel
        for t in range(self.horizon):
            constraints.append(lataccel[t] == predicted_lataccel[t])

        # Convert the future_plan.lataccel list to a cvxpy Parameter
        future_lataccel = cp.Parameter(
            self.horizon, value=np.array(future_plan.lataccel[:self.horizon]))

        # Calculate jerk (change in lataccel) and construct the cost function
        jerk = cp.diff(lataccel) / self.dt
        cost = (lataccel[0] - target_lataccel)**2 + cp.sum_squares(lataccel - future_lataccel) * \
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
