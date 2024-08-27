from . import BaseController
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNN(nn.Module):
    def __init__(self, state_dim, control_dim):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + control_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)  # Output Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Controller(BaseController):
    """
    A PID controller with integrated QNN-based LQR learning.
    """

    def __init__(self, state_dim=4, control_dim=1, gamma=0.99, lr=0.01, epsilon=0.01):
        # PID parameters
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0

        # QNN and optimizer
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.qnn = QNN(state_dim, control_dim)
        self.optimizer = optim.Adam(self.qnn.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

        # Memory for stateless learning
        self.prev_state = None
        self.prev_action = None
        self.prev_target_lataccel = None
        self.prev_future_plan = None
        self.prev_q_value = None

        # LQR parameters (to be learned)
        self.H = None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute PID output as an initial stabilizing controller
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff

        # Convert to torch tensor
        # add error to state
        error = target_lataccel - current_lataccel
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = torch.cat(
            (state_tensor, torch.tensor([error], dtype=torch.float32)))
        assert state_tensor.shape[0] == self.state_dim
        control_tensor = torch.tensor([pid_output], dtype=torch.float32)

        # If we have a previous state and action, train QNN
        if self.prev_state is not None:
            # Prepare data for QNN training
            prev_state_action = torch.cat((self.prev_state, self.prev_action))

            current_state_action = torch.cat((state_tensor, control_tensor))

            # Compute Q-value for previous state-action pair
            with torch.no_grad():
                target_q = self.prev_target_lataccel - target_lataccel + \
                    self.gamma * self.qnn(current_state_action)

            # Predicted Q-value
            predicted_q = self.qnn(prev_state_action)

            # Compute loss and update QNN
            loss = nn.MSELoss()(predicted_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Check for convergence
            if self.prev_q_value is not None:
                q_diff = torch.abs(predicted_q - self.prev_q_value)
                if q_diff.item() < self.epsilon:
                    print(
                        "Convergence achieved with Q-value difference: ", q_diff.item())

            # Save the current Q-value for the next iteration's convergence check
            self.prev_q_value = predicted_q

        # Save current state-action pair for next iteration
        self.prev_state = state_tensor
        self.prev_action = control_tensor
        self.prev_target_lataccel = target_lataccel
        self.prev_future_plan = future_plan

        print("State: ", state_tensor)
        print("Control: ", control_tensor)
        # Policy improvement: Compute the LQR controller using QNN
        self.H = self.compute_H_matrix(state_tensor, control_tensor)

        # Calculate the optimal action using the LQR controller
        optimal_action = self.compute_optimal_action(state_tensor)

        # Log and return the optimal action
        print("PID: ", pid_output)
        print("Optimal Action: ", optimal_action.item())
        return optimal_action.item()

    def compute_H_matrix(self, state_tensor, control_tensor):
        # Concatenate state and control to form input to the QNN
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(
                0)  # Add batch dimension if missing
        if control_tensor.dim() == 1:
            control_tensor = control_tensor.unsqueeze(
                0)  # Add batch dimension if missing

        X_input = torch.cat([state_tensor, control_tensor], dim=-1)
        print("X_input: ", X_input)

        # Forward pass through QNN
        Y_hat = self.qnn(X_input)  # Predict the quadratic form using QNN
        print("Y_hat: ", Y_hat)

        # Set up the matrix equation X.T @ H @ X = Y_hat
        XTX = torch.einsum('bi,bj->bij', X_input, X_input)  # X^T * X
        print("XTX: ", XTX)

        # Reshape Y_hat to match the flattened upper triangle of the matrix
        # Flatten Y_hat to match the dimension of the target
        Y_hat_flat = Y_hat.view(-1)
        # Flatten XTX to match dimensions for lstsq
        XTX_flat = XTX.view(-1, XTX.shape[-1])
        # print both dims of inputs
        print("Y_hat_flat: ", Y_hat_flat.shape)
        print("XTX_flat: ", XTX_flat.shape)

        # Ensure Y_hat_flat has the same dimension as one side of XTX_flat
        if Y_hat_flat.size(0) != XTX_flat.size(0):
            Y_hat_flat = Y_hat_flat.expand(XTX_flat.size(0))

        # Solve the least squares problem to find H
        H_flat = torch.linalg.lstsq(
            XTX_flat, Y_hat_flat.unsqueeze(-1)).solution

        # H should be a matrix, not a vector. We reshape it accordingly.
        # H = H_flat.view(H_size, H_size)
        # print("H: ", H)

        return H_flat

    def compute_optimal_action(self, state_tensor):
        # This function computes the optimal action using the LQR controller
        # based on the H matrix derived from QNN outputs
        H_flat = self.H
        return state_tensor.T @ H_flat
