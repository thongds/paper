import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from networks import ReplayBuffer, QNetwork


class DQNAgent:
    """DQN agent for grid world with 4 actions (UP, DOWN, LEFT, RIGHT)"""
    
    def __init__(self, grid_world, n_actions=4, episodes=600, alpha=0.001, 
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=300, 
                 max_steps=200, seed=123, gamma=0.99, batch_size=64, 
                 buffer_size=10000, tau=0.005, hidden_dim=128):
        
        # Core parameters
        self.grid_world = grid_world
        self.n_actions = n_actions
        self.episodes = episodes
        self.alpha = alpha
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.max_steps = max_steps
        self.seed = seed
        
        # Initialize random number generators
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize Q-table and tracking arrays
        self.num_states = grid_world.n_rows * grid_world.n_cols
        self.Q = np.zeros((self.num_states, n_actions), dtype=float)
        self.returns = np.zeros(episodes, dtype=float)
        self.bumps = np.zeros(episodes, dtype=float)
        self.steps_arr = np.zeros(episodes, dtype=int)
        self.losses = []
        
        # Epsilon decay calculation
        self.eps_decay = (eps_start - eps_end) / max(1, eps_decay_episodes)
        
        # DQN parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State dimension (grid position encoded as normalized coordinates)
        self.state_dim = 2
        
        # Initialize networks
        self.policy_net = QNetwork(self.state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, n_actions, hidden_dim).to(self.device)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def state_to_tensor(self, state):
        """Convert grid state (row, col) to normalized tensor"""
        normalized = np.array([
            state[0] / (self.grid_world.n_rows - 1),
            state[1] / (self.grid_world.n_cols - 1)
        ], dtype=np.float32)
        return torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
    
    def state_to_features(self, state):
        """Convert grid state to feature array"""
        return np.array([
            state[0] / (self.grid_world.n_rows - 1),
            state[1] / (self.grid_world.n_cols - 1)
        ], dtype=np.float32)
    
    def get_q_values(self, state):
        """Get Q-values for a state using policy network"""
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            return self.policy_net(state_tensor).cpu().numpy()[0]
    
    def epsilon_greedy(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        
        q_values = self.get_q_values(state)
        max_q = np.max(q_values)
        best = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best))
    
    def dqn_train_step(self):
        """Perform one DQN training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update_target(self):
        """Soft update target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                               self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                   (1 - self.tau) * target_param.data)
    
    def update_q_table(self):
        """Update Q-table from policy network for visualization"""
        for si in range(self.num_states):
            row = si // self.grid_world.n_cols
            col = si % self.grid_world.n_cols
            state = (row, col)
            self.Q[si] = self.get_q_values(state)
    
    def train(self, actions_dict, epsilon_greedy_func=None):
        """Train DQN agent"""
        eps = self.eps_start
        
        for ep in range(self.episodes):
            s = self.grid_world.start
            si = self.grid_world.to_index(s)
            done = False
            G = 0.0
            bumpcount = 0
            steps = 0
            episode_loss = []
            
            for t in range(self.max_steps):
                # Epsilon-greedy action selection
                a = self.epsilon_greedy(s, eps)
                
                # Take action
                s_next, r, done = self.grid_world.step(s, a, actions_dict, self.rng)
                s_next_i = self.grid_world.to_index(s_next)
                
                if si == s_next_i:
                    bumpcount += 1
                
                # Store in replay buffer
                state_features = self.state_to_features(s)
                next_state_features = self.state_to_features(s_next)
                self.replay_buffer.push(state_features, a, r, next_state_features, float(done))
                
                # Train DQN
                loss = self.dqn_train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Soft update target network
                self.soft_update_target()
                
                G += r
                s, si = s_next, s_next_i
                steps += 1
                
                if done:
                    break
            
            self.returns[ep] = G
            self.bumps[ep] = bumpcount
            self.steps_arr[ep] = steps
            
            if episode_loss:
                self.losses.append(np.mean(episode_loss))
            
            # Decay epsilon
            if ep < self.eps_decay_episodes:
                eps = max(self.eps_end, eps - self.eps_decay)
            
            if (ep + 1) % 50 == 0:
                avg_return = np.mean(self.returns[max(0, ep-49):ep+1])
                print(f"Episode {ep+1}/{self.episodes}, Return: {G:.2f}, "
                      f"Avg50: {avg_return:.2f}, Eps: {eps:.3f}")
        
        # Update Q-table for visualization
        self.update_q_table()
        print(f'DQN Training ({self.n_actions} actions) complete!')
    
    def get_policy(self, n_rows, n_cols):
        """Get the derived policy from Q-values"""
        from utils import Visualizer
        self.update_q_table()
        return Visualizer.derive_policy(self.Q, n_rows, n_cols)
    
    def get_results(self, moving_average_func, ma_window=25):
        """Get training results with moving averages"""
        self.update_q_table()
        return {
            'returns': self.returns.copy(),
            'Q': self.Q.copy(),
            'bumps': self.bumps.copy(),
            'returns_ma': moving_average_func(self.returns, w=ma_window),
            'steps_ma': moving_average_func(self.steps_arr.astype(float), w=ma_window),
            'losses': self.losses.copy() if self.losses else []
        }
    
    def save_model(self, path):
        """Save the policy network"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the policy network"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# Backward compatibility for older imports
DDQNAgent = DQNAgent
