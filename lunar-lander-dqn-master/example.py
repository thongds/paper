
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class ActionMap():
    def __init__(self, action_map = {0:1, 1:2, 2:3}, expand_actions_env = []):
        self.action_map = action_map
        self.expand_actions_env = expand_actions_env
    
    def convert_action_env_to_network(self, action_env):
        for k, v in self.action_map.items():
            if v == action_env:
                return k
        return None
    
    def get_expand_action_in_network(self):
        result = []
        for i in self.expand_actions_env:
            net_action = self.convert_action_env_to_network(i)
            result.append(net_action)
        return result
                
    def convert_network_action_to_env(self, action_network):
        return self.action_map.get(action_network, None)
    
    def select_all_actions_in_network(self):
        return list(self.action_map.keys())
    def select_all_actions_in_env(self):
        return list(self.action_map.values())
    

class ActionExplorer(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.001):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def forward(self, state):
        logits = self.model(state)
        probs = torch.softmax(logits, dim=-1) 
        return probs
    
    def get_action_frequencies(self, state):
        with torch.no_grad():
            probs = self.forward(state)
        return probs.cpu().numpy()
    
    def get_least_tried_action(self, state):
        freqs = self.get_action_frequencies(state)
        least_tried_action = freqs.argmin()
        return least_tried_action
    
    def update(self, state, action):
        probs = self.forward(state)
        loss = -torch.log(probs[action] + 1e-10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() 
    
class AdaptiveAgent():
    def __init__(self, n_observations, base_action_map: ActionMap,
                 full_action_map: ActionMap, base_agent_path = BASE_MODEL_PATH,
                 uncertainty_threshold=0.5, batch_size=128, gamma=0.99, lr=1e-4, tau=0.005):
        self.base_agent = base_agent_path
        self.action_map = base_action_map
        self.full_action_map = full_action_map
        self.uncertainty_threshold = uncertainty_threshold
        self.batch_size = batch_size
        self.episode_duration = []
        self.reward_list = []
        self.action_explorer = ActionExplorer(state_dim=n_observations, action_dim=len(full_action_map.action_map)).to(device)
        self.base_q_net = DQN(n_observations= n_observations, n_actions=len(base_action_map.action_map)).to(device)
        self.base_q_net.load_state_dict(torch.load(base_agent_path, map_location=device, weights_only=True))
        self.base_q_net.eval()
        for param in self.base_q_net.parameters():
            param.requires_grad = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(n_observations= n_observations, n_actions=len(full_action_map.action_map)).to(self.device)
        self.target_net = DQN(n_observations= n_observations, n_actions=len(full_action_map.action_map)).to(self.device)
        self.replay_memory = ReplayMemory(10000)
        # fine-tune model
        pretrained_dict = torch.load(base_agent_path, map_location=self.device, weights_only=True)
        new_model_dict = self.policy_net.state_dict()
        
        self.GAMMA = gamma
        self.LR = lr
        self.TAU = tau
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.995
        self.allow_expansion = 0
        self.reject_expansion = 0
        # transfer weights
        for name, param in pretrained_dict.items():    
            if name in new_model_dict:
                if 'layer3' in name:
                    if 'weight' in name:
                        old_weight = param.clone()
                        new_weight = new_model_dict[name].clone()
                        size_base_action = len(base_action_map.select_all_actions_in_env())
                        size_full_action = len(full_action_map.select_all_actions_in_env())
                        new_weight[:size_base_action] = old_weight
                         
                         # copy random weights for new actions
                        for i in range(size_base_action, size_full_action):
                            new_weight[i] = old_weight[np.random.randint(0, size_base_action)].clone()
                        new_model_dict[name] = new_weight
                    elif 'bias' in name:
                        old_bias = param.clone()
                        new_bias = new_model_dict[name].clone()
                        size_base_action = len(base_action_map.select_all_actions_in_env())
                        size_full_action = len(full_action_map.select_all_actions_in_env())
                        new_bias[:size_base_action] = old_bias
                        
                        # copy random biases for new actions
                        for i in range(size_base_action, size_full_action):
                            new_bias[i] = old_bias[np.random.randint(0, size_base_action)].clone()
                        new_model_dict[name] = new_bias
                else:# copy other layers directly
                    new_model_dict[name] = param
        self.policy_net.load_state_dict(new_model_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())  
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR)
        self.criterion = nn.SmoothL1Loss()    
                 
    def select_action(self, state, epsilon, action_map: ActionMap):
        selected_action = None
        if np.random.rand() < epsilon:
            action = np.random.choice(action_map.select_all_actions_in_network())
            selected_action = torch.tensor([[action]], dtype=torch.long, device=self.device)
        else:
            with torch.no_grad():
                selected_action = self.policy_net(state).max(1).indices.view(1, 1)
        return selected_action
    
    def train(self, env, num_episodes, train_transition_model):
       
        for episode in range(num_episodes):
            total_reward = 0
            state, _ = env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            
            for t in count():
                action_network = self.select_action(state, self.EPSILON, self.full_action_map)
                env_action = self.full_action_map.convert_network_action_to_env(action_network.item())
                
                if action_network.item() == self.full_action_map.get_expand_action_in_network():
                    # Use transition model to predict next state 
                    predicted_next_state, confidence = train_transition_model.model.predict(state.cpu().squeeze(0).numpy())
                    q_value_current_state = self.base_q_net(state).max(1).values.item()
                    q_value_next_state = self.base_q_net(torch.tensor(predicted_next_state, dtype=torch.float32, device=self.device).unsqueeze(0)).max(1).values.item()
                    print(f"Predicted Confidence: {confidence:.4f}, Q_current: {q_value_current_state:.4f}, Q_next: {q_value_next_state:.4f}")
                    if q_value_next_state >= q_value_current_state:
                        self.allow_expansion += 1
                    else:
                        self.reject_expansion += 1
                        action_network = self.select_action(state, self.EPSILON, self.full_action_map)
                        env_action = self.full_action_map.convert_network_action_to_env(action_network.item())
                    
                next_state, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(
                    next_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                self.replay_memory.push(state, action_network, next_state, reward, done)
                state = next_state
                total_reward += reward.item()
                if len(self.replay_memory) >= self.batch_size:
                    transitions = self.replay_memory.sample(self.batch_size)
                    states, actions, next_states, rewards, dones = zip(*transitions) 
                    #convert tuple to tensor array
                    states_batch = torch.cat(states) # states is a tuple, now it becomes a batch (2D array)
                    next_states_batch = torch.cat(next_states)
                    actions_batch = torch.cat(actions)
                    
                    reward_batch = torch.tensor(rewards, device= self.device)
                    dones_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
                    
                    # Q-value which are estimating (policy network)
                    q_values = self.policy_net(states_batch).gather(1, actions_batch).squeeze()
                    
                    # get target Q-values
                    with torch.no_grad():
                        next_q_values = self.target_net(next_states_batch).max(1).values
                        target_q_values = reward_batch + self.GAMMA * next_q_values * (1 - dones_batch)
                    loss = self.criterion(q_values, target_q_values) # prove converge
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                    self.optimizer.step()
                    
                    # soft update target network
                    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        target_param.data.copy_(self.TAU * policy_param.data + (1 - self.TAU) * target_param.data)
                if done:
                    self.episode_duration.append(t+1)
                    break
            # Decay epsilon    
            self.EPSILON = max(self.EPSILON_MIN, self.EPSILON * self.EPSILON_DECAY)
            self.reward_list.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.reward_list[-10:])
                print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, "
                    f"Avg10: {avg_reward:.2f}, Epsilon: {self.EPSILON:.3f}")
                