from . import BaseController
import casadi as ca


class Controller(BaseController):
    """
    An MPC-based controller with a horizon for lateral acceleration tracking and jerk minimization.
    """

    def __init__(self, horizon=10):
        # Define the bicycle model parameters
        self.C_f = 20007.77  # Front cornering stiffness
        self.C_r = 20007.77  # Rear cornering stiffness
        self.l_f = 0.35      # Distance from CG to front axle
        self.l_r = 2.43      # Distance from CG to rear axle
        self.m = 1292.67     # Vehicle mass
        self.I_z = 2250.0    # Yaw moment of inertia
        self.horizon = horizon  # Length of the prediction horizon

        # PID equivalent parameters
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        self.last_guess = 0

    def bicycle_model_dynamics(self, state, steer_angle):
        v_ego, yaw_rate, beta = state
        v_ego = max(v_ego, 0.2)

        # Lateral slip angles at the front and rear tires
        alpha_f = steer_angle - (beta + (self.l_f * yaw_rate) / v_ego)
        alpha_r = - (beta - (self.l_r * yaw_rate) / v_ego)

        # Lateral forces on the front and rear tires
        F_yf = -self.C_f * alpha_f
        F_yr = -self.C_r * alpha_r

        # Equations of motion
        a_lat = (F_yf + F_yr) / self.m  # Lateral acceleration
        yaw_accel = (self.l_f * F_yf - self.l_r * F_yr) / \
            self.I_z  # Yaw acceleration

        return a_lat, yaw_accel

    def cancel_road_lataccel(self, roll_lataccel, state):
        # Calculate the steering angle required to cancel out roll_lataccel
        steer_angle_cancel = roll_lataccel / (self.C_f / self.m)

        # Check the limits for steering angle
        steer_angle_cancel = min(max(steer_angle_cancel, -ca.pi/4), ca.pi/4)

        return steer_angle_cancel

    def cost_function(self, steer_angles, target_lataccels, state, future_plan):
        # Unpack the state and future plan
        roll_lataccel, v_ego, a_ego = state

        # Initialize cost
        total_cost = 0
        current_state = state

        for t in range(self.horizon):
            # Update state for each timestep based on current steering angle
            current_lataccel, _ = self.bicycle_model_dynamics(
                current_state, steer_angles[t])
            jerk = current_lataccel - a_ego

            # Update the cumulative cost over the horizon
            total_cost += 100 * \
                (target_lataccels[t] - current_lataccel)**2 + jerk**2

            # Simulate the state forward in time (this could be done more accurately using a model)
            a_ego = current_lataccel  # In this case, assuming new acceleration equals the current
            # Update current state for next iteration (simplified model)
            # Assuming yaw_rate and beta reset to zero
            current_state = (v_ego, 0, 0)

        return total_cost

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        print(state)
        if (len(future_plan) < self.horizon):
            return 0
        # Use future_plan's lataccel for the horizon
        target_lataccels = future_plan.lataccel[:self.horizon]

        # First, calculate the steering angle required to cancel out the roll-induced lateral acceleration
        roll_lataccel = state[0]
        steer_angle_cancel = self.cancel_road_lataccel(roll_lataccel, state)

        # Then, optimize the steering angles over the entire horizon
        steer_angles = ca.SX.sym('steer_angles', self.horizon)

        # Define the cost function for the entire horizon
        cost = self.cost_function(
            steer_angles + steer_angle_cancel, target_lataccels, state, future_plan)

        # Set up the optimization problem
        nlp = {
            'x': steer_angles,
            'f': cost,
            'g': []
        }

        # Create an NLP solver (Ipopt)
        solver = ca.nlpsol('solver', 'ipopt', nlp)

        # Set the bounds and initial guess for the optimization
        lb = [-ca.pi/4] * self.horizon  # minimum steering angle (-45 degrees)
        ub = [ca.pi/4] * self.horizon   # maximum steering angle (45 degrees)
        initial_guess = [0.0] * self.horizon  # initial guess

        # Solve the optimization problem
        solution = solver(x0=initial_guess, lbx=lb, ubx=ub)
        optimal_steering_angles = solution['x'].full().flatten()

        # Use the first optimized steering angle as the control input
        optimal_steering = steer_angle_cancel + optimal_steering_angles[0]
        return optimal_steering
