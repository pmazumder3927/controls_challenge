import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mlp import MLP
import pickle
import matplotlib.pyplot as plt
from tinyphysics import run_rollout_controller
from controllers.pureff import Controller


class CSVLoader:
    def __init__(self, data_path, batch_size=100):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.files.sort()
        self.current_file = 0
        self.within_file = 0
        self.batch_size = batch_size
        self.load_data()

    def load_data(self):
        # Sample CSV format:
        # t,vEgo,aEgo,roll,targetLateralAcceleration,steerCommand
        # 0.0,33.770259857177734,-0.0172996744513511,0.0374695472778479,1.0038640328608972,-0.32973363077505974
        # We need vEgo, aEgo, roll, targetLateralAcceleration as the state and steerCommand as the target
        # refresh files list
        self.files = os.listdir(self.data_path)
        self.files.sort()
        try:
            self.data = pd.read_csv(os.path.join(
                self.data_path, self.files[self.current_file]), on_bad_lines='skip')
            # make sure the right columns exist
            if not all(col in self.data.columns for col in ['vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steerCommand']):
                raise Exception(
                    f"File {self.files[self.current_file]} does not have the right columns")
            # normalize all columns
            self.data = self.data.apply(lambda x: (x - x.mean()) / x.std())
        except Exception as e:
            print(e)
            self.current_file += 1
            self.load_data()
        self.data = self.data[['vEgo', 'aEgo', 'roll',
                               'targetLateralAcceleration', 'steerCommand']]
        # remove rows where steerCommand is NaN
        self.data = self.data[self.data['steerCommand'].notna()]
        self.current_file += 1
        self.within_file = 0

    def get_next_batch(self):
        if self.within_file + self.batch_size >= len(self.data):
            if self.current_file >= len(self.files):
                # If no more files to load, reset to the first file
                self.current_file = 0
                # raise StopIteration
            else:
                self.load_data()
        state = self.data[['vEgo', 'aEgo', 'roll',
                           'targetLateralAcceleration']]
        target = self.data[['steerCommand']]
        state = state.to_numpy()[self.within_file:self.within_file +
                                 self.batch_size]
        target = target.to_numpy()[self.within_file:self.within_file +
                                   self.batch_size].reshape(-1)
        self.within_file += self.batch_size
        return torch.tensor(state, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class ArrayLoader:
    def __init__(self, data_path, batch_size=100):
        self.data = pickle.load(open(data_path, 'rb'))
        self.batch_size = batch_size
        self.current_index = 0

    def get_next_batch(self):
        if self.current_index + self.batch_size >= len(self.data):
            self.current_index = 0
        batch = self.data[self.current_index:self.current_index +
                          self.batch_size]
        self.current_index += self.batch_size
        target = batch[:, -1]
        state = batch[:, :-1]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def train_model(model, data_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)  # Move model to the GPU if available
    total_loss = 0
    loss_history = []
    cost_history = []
    for epoch in range(num_epochs):
        try:
            # Get next batch of data
            inputs, targets = data_loader.get_next_batch()
            inputs, targets = inputs.to(device), targets.to(
                device)  # Move data to GPU
        except StopIteration:
            break

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch % 100 == 0):
            plt.plot(loss_history)
            plt.yscale('log')
            plt.savefig('loss.png')
        if epoch % 5 == 0 and epoch > 0:
            controller = Controller(state_dict=model.state_dict())
            rollout, _, _ = run_rollout_controller(data_path='./data/00005.csv',
                                                   controller=controller,
                                                   model_path='./models/tinyphysics.onnx',
                                                   debug=False)
            cost_history.append(rollout['total_cost'])
            print(rollout)
            if rollout['total_cost'] == min(cost_history):
                torch.save(model.state_dict(), 'mlp_model.pth')
            plt.clf()
            plt.yscale('log')
            print("minimum cost: ", min(cost_history))
            plt.plot(cost_history)
            plt.savefig('cost.png')
        total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Hyperparameters
input_dim = 4  # vEgo, aEgo, roll, targetLateralAcceleration
hidden_dim = 1024
output_dim = 1  # steerCommand
learning_rate = 0.001
num_epochs = 10000
batch_size = 2**19

# Initialize the model, loss function, optimizer, and data loader
model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
data_loader = ArrayLoader(
    data_path='./fake_data/cleaned/data.pkl', batch_size=batch_size)
# data_loader = CSVLoader(
#     data_path='./fake_data/cleaned', batch_size=batch_size)

# Train the model
train_model(model, data_loader, criterion,
            optimizer, device, num_epochs=num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'mlp_model.pth')
