import torch.nn as nn
import torch
import pickle


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLPFeedForward():
    def __init__(self, model_path, scaler_path, state_dict=None):
        self.model = MLP(input_dim=4, hidden_dim=1024, output_dim=1)
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(
                torch.load(model_path, weights_only=True))
        self.model.eval()

    def infer(self, state):
        state = (state * self.scaler.scale_[:4]) + self.scaler.min_[:4]
        state = torch.tensor(state, dtype=torch.float32)
        target = self.model(state)
        target = (target.clone().detach().item() -
                  self.scaler.min_[-1]) / self.scaler.scale_[-1]
        return target
