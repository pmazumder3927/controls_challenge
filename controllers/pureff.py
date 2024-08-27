from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from mlp import MLP
import os
import time
import pickle
from mlp import MLPFeedForward


class Controller(BaseController):
    """
    A simple PID controller
    """

    def __init__(self, state_dict=None):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        if state_dict is not None:
            self.mlp = MLPFeedForward(
                state_dict=state_dict, scaler_path='./scaler.pkl', model_path='./mlp_model.pth')
        else:
            self.mlp = MLPFeedForward(
                model_path='./mlp_model.pth', scaler_path='./scaler.pkl')

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # breakpoint
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        # simple pid
        pid = self.p * error + self.i * self.error_integral + self.d * error_diff
        model_input = np.array(
            [state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel])
        ff = self.mlp.infer(model_input)
        return ff
