import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import numpy as np

class TransitionMLP(nn.Module):
    """MLP model to predict next state for diagonal actions (4-7)"""
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64, output_dim=2):
        super(TransitionMLP, self).__init__()
        input_dim = state_dim + action_dim  # state + action
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TransitionModelLearner:
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64, lr=0.001, buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransitionMLP(state_dim, action_dim, hidden_dim, state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.buffer = deque(maxlen=buffer_size) #auto remove old samples if over size
        self.min_buffer_size = 100  # Minimum samples before training
        
    def add_experience(self, state, action, next_state):
        # Convert grid coordinates to normalized features
        state_features = np.array([state[0] / 24.0, state[1] / 24.0])  # Normalize to [0,1]
        # Normalize action (4-7 -> 0-3)
        action_features = np.array([(action - 4) / 3.0])  # Normalize diagonal actions to [0,1]
        next_state_features = np.array([next_state[0] / 24.0, next_state[1] / 24.0])
        self.buffer.append((state_features, action_features, next_state_features))
    
    def can_predict(self):
        return len(self.buffer) >= self.min_buffer_size
    
    def train_model(self, batch_size=32, epochs=10):
        if len(self.buffer) < self.min_buffer_size:
            return
        states = []
        actions = []
        next_states = []
        sample_size = min(len(self.buffer), 1000)  #  last 1000 samples
        samples = list(self.buffer)[-sample_size:]
        
        for state_feat, action_feat, next_state_feat in samples:
            states.append(state_feat)
            actions.append(action_feat)
            next_states.append(next_state_feat)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        self.model.train()
        for _ in range(epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_next_states = next_states[batch_indices]
                # Concatenate state and action
                batch_input = torch.cat([batch_states, batch_actions], dim=1)
                # forward 
                predicted_next_states = self.model(batch_input)
                loss = self.criterion(predicted_next_states, batch_next_states)
                
                # backward 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict_next_state(self, state, action):
        self.model.eval()
        with torch.no_grad():
            state_features = np.array([state[0] / 24.0, state[1] / 24.0])
            action_features = np.array([(action - 4) / 3.0])  # Normalize diagonal action
            input_features = np.concatenate([state_features, action_features])
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
            
            predicted = self.model(input_tensor)
            predicted_np = predicted.cpu().numpy()[0]
            
            # convert back to grid coordinates
            next_r = int(np.clip(predicted_np[0] * 24, 0, 24))
            next_c = int(np.clip(predicted_np[1] * 24, 0, 24))
            
            return (next_r, next_c)