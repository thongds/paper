import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from TransitionModelLearner import TransitionModelLearner


class ReplayBuffer:
    """Experience replay buffer for DDQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for Q-value approximation"""
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDQNAgentUseCondition:
    """Double DQN agent with learned MLP model for action adaptation"""
    
    def __init__(self, grid_world, n_actions, base_policy_net=None, episodes=600, 
                 alpha=0.001, eps_start=1.0, eps_end=0.05, eps_decay_episodes=300, 
                 max_steps=200, seed=123, use_model=True, use_conditional=True,
                 exploration_start=1.0, exploration_end=0.0, exploration_decay_episodes=300,
                 gamma=0.99, batch_size=64, buffer_size=10000, target_update_freq=10,
                 tau=0.005, hidden_dim=128):
        
        self.grid_world = grid_world
        self.n_actions = n_actions
        self.episodes = episodes
        self.alpha = alpha  # learning rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.use_model = use_model
        self.use_conditional = use_conditional
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # soft update parameter
        
        # Set random seeds
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State dimension (grid position encoded as normalized coordinates)
        self.state_dim = 2  # (row, col) normalized
        self.num_states = grid_world.n_rows * grid_world.n_cols
        
        # Initialize networks
        self.policy_net = QNetwork(self.state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, n_actions, hidden_dim).to(self.device)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize with base policy network if provided
        if base_policy_net is not None:
            # Load base network weights for shared layers
            self._initialize_from_base(base_policy_net)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Epsilon decay calculation
        self.eps_decay = (eps_start - eps_end) / max(1, eps_decay_episodes)
        
        # Exploration encouragement decay parameters
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.exploration_decay_episodes = exploration_decay_episodes
        self.exploration_decay = (exploration_start - exploration_end) / exploration_decay_episodes if exploration_decay_episodes > 0 else 0
        
        # Initialize transition model learner
        self.transition_learner = TransitionModelLearner()
        
        # Tracking arrays
        self.returns = np.zeros(episodes, dtype=float)
        self.bumps = np.zeros(episodes, dtype=float)
        self.steps_arr = np.zeros(episodes, dtype=int)
        self.reuse_count = np.zeros(episodes, dtype=float)
        self.reject_count = np.zeros(episodes, dtype=float)
        self.losses = []
        
        # Q-table for compatibility (stores latest Q-values for visualization)
        self.Q = np.zeros((self.num_states, n_actions), dtype=float)
    
    def _initialize_from_base(self, base_policy_net):
        """Initialize network from a base policy network (transfer learning)"""
        base_state_dict = base_policy_net.state_dict()
        current_state_dict = self.policy_net.state_dict()
        
        # Copy compatible weights
        for name, param in base_state_dict.items():
            if name in current_state_dict:
                if current_state_dict[name].shape == param.shape:
                    current_state_dict[name].copy_(param)
                elif name == 'fc3.weight':
                    # Handle different output dimensions (more actions)
                    min_actions = min(param.shape[0], current_state_dict[name].shape[0])
                    current_state_dict[name][:min_actions, :] = param[:min_actions, :]
                    # Optimistic initialization for new actions
                    if current_state_dict[name].shape[0] > param.shape[0]:
                        # Initialize new action weights with slightly higher values
                        nn.init.xavier_uniform_(current_state_dict[name][min_actions:, :])
                elif name == 'fc3.bias':
                    min_actions = min(param.shape[0], current_state_dict[name].shape[0])
                    current_state_dict[name][:min_actions] = param[:min_actions]
                    if current_state_dict[name].shape[0] > param.shape[0]:
                        # Initialize new action biases with optimistic values
                        current_state_dict[name][min_actions:] = param.max()
        
        self.policy_net.load_state_dict(current_state_dict)
        self.target_net.load_state_dict(current_state_dict)
    
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
    
    def enhanced_epsilon_greedy(self, state, epsilon, encourage_new_action=False):
        """Epsilon-greedy action selection with optional bias toward new actions"""
        if encourage_new_action:
            if self.rng.random() < 0.3:
                # Randomly choose one of the diagonal actions (4-7)
                return int(self.rng.integers(4, self.n_actions))
        
        # Standard epsilon-greedy
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
            
        q_values = self.get_q_values(state)
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        max_q = np.max(q_values)
        best = np.flatnonzero(q_values == max_q)
        if self.rng.random() < epsilon:
            expand_action = int(self.rng.integers(4, self.n_actions))
            return int(self.rng.choice(expand_action, best))
        return int(self.rng.choice(best))
    
    def select_action(self, state, epsilon, encourage_new_action=False, episode=0):
        """Select action using epsilon-greedy with optional new action encouragement"""
        return self.enhanced_epsilon_greedy(state, epsilon, encourage_new_action)
    
    def update_q_table(self):
        """Update Q-table from policy network for visualization purposes"""
        for si in range(self.num_states):
            row = si // self.grid_world.n_cols
            col = si % self.grid_world.n_cols
            state = (row, col)
            self.Q[si] = self.get_q_values(state)
    
    def train_step(self):
        """Perform one DDQN training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor).squeeze()
        
        # DDQN: Use policy network to select best action, target network to evaluate
        with torch.no_grad():
            # Policy net selects the best action
            next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            # Target net evaluates that action
            next_q = self.target_net(next_states_tensor).gather(1, next_actions).squeeze()
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
        
        # Compute loss and update
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
    
    def hard_update_target(self):
        """Hard update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_with_learned_model(self, actions_dict, epsilon_greedy_func=None):
        """Train the DDQN agent with transition model assistance"""
        eps = self.eps_start
        exploration_prob = self.exploration_start
        model_train_frequency = 20
        
        for ep in range(self.episodes):
            s = self.grid_world.start
            si = self.grid_world.to_index(s)
            done = False
            G = 0.0
            bumpcount = 0
            steps = 0
            reuse = 0
            reject = 0
            episode_loss = []
            
            for t in range(self.max_steps):
                # Use decaying exploration probability
                if self.rng.random() < exploration_prob:
                    a = self.enhanced_epsilon_greedy(s, eps, encourage_new_action=True)
                else:
                    a = self.select_action(s, eps, encourage_new_action=True, episode=ep)
                
                # Use transition model for diagonal actions if available
                if a >= 4 and self.transition_learner.can_predict() and self.use_conditional:
                    snext_model = self.transition_learner.predict_next_state(s, a)
                    q_next_model = self.get_q_values(snext_model)
                    q_current = self.get_q_values(s)
                    
                    if np.max(q_next_model) > np.max(q_current):
                        reuse += 1
                    else:
                        reject += 1
                        a = self.select_action(s, eps)
                
                # Take action in environment
                s_next, r, done = self.grid_world.step(s, a, actions_dict, self.rng)
                s_next_i = self.grid_world.to_index(s_next)
                
                # Store transition for diagonal actions
                if a >= 4:
                    self.transition_learner.add_experience(s, a, s_next)
                
                # Check for bump
                if si == s_next_i:
                    bumpcount += 1
                
                # Store in replay buffer
                state_features = self.state_to_features(s)
                next_state_features = self.state_to_features(s_next)
                self.replay_buffer.push(state_features, a, r, next_state_features, float(done))
                
                # Train DDQN
                loss = self.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Soft update target network
                self.soft_update_target()
                
                G += r
                s, si = s_next, s_next_i
                steps += 1
                
                if done:
                    break
            
            # Record episode statistics
            self.reuse_count[ep] = reuse
            self.reject_count[ep] = reject
            self.returns[ep] = G
            self.bumps[ep] = bumpcount
            self.steps_arr[ep] = steps
            
            if episode_loss:
                self.losses.append(np.mean(episode_loss))
            
            # Train transition model periodically
            if ep > 0 and ep % model_train_frequency == 0 and len(self.transition_learner.buffer) > 50:
                self.transition_learner.train_model(batch_size=32, epochs=5)
            
            # Decay epsilon
            if ep < self.eps_decay_episodes:
                eps = max(self.eps_end, eps - self.eps_decay)
            
            # Decay exploration probability
            if ep < self.exploration_decay_episodes:
                exploration_prob = max(self.exploration_end, 
                                      exploration_prob - self.exploration_decay)
            
            # Periodic logging
            if (ep + 1) % 50 == 0:
                avg_return = np.mean(self.returns[max(0, ep-49):ep+1])
                print(f"Episode {ep+1}/{self.episodes}, Return: {G:.2f}, "
                      f"Avg50: {avg_return:.2f}, Eps: {eps:.3f}, "
                      f"Reuse: {reuse}, Reject: {reject}")
        
        # Update Q-table for visualization
        self.update_q_table()
        print(f'DDQN Training ({self.n_actions} actions) complete!')
    
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
            'reuse_count': self.reuse_count.copy(),
            'reject_count': self.reject_count.copy(),
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
