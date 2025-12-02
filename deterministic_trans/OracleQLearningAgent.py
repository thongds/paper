import numpy as np
from env import *
from QLearningAgent import QLearningAgent

class OracleQLearningAgent(QLearningAgent):
    """Q-learning agent with oracle model for action adaptation"""
    
    def __init__(self, grid_world, n_actions, base_q_table=None, episodes=600, alpha=0.5, 
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=300, 
                 max_steps=200, seed=123, use_model = True, use_conditional = True, walls = None):
        super().__init__(grid_world, n_actions, episodes, alpha, eps_start, eps_end, 
                        eps_decay_episodes, max_steps, seed)
        self.reuse_count = np.zeros(self.episodes, dtype = float)
        self.reject_count = np.zeros(self.episodes, dtype = float)
        self.use_model = use_model
        self.base_q_table = base_q_table
        self.use_conditional = use_conditional
        self.walls = walls
        # Initialize with base Q-table if provided
        if base_q_table is not None:
            # Optimistic initialization for new actions
            if use_model == True:
                self.Q[:, :base_q_table.shape[1]] = base_q_table
                V_old = np.min(base_q_table, axis=1) 
                for new_action in range(base_q_table.shape[1], n_actions):
                    self.Q[:, new_action] = V_old
    
    def epsilon_greedy(q_actions, epsilon, rng):
        """Epsilon-greedy action selection"""
        if rng.random() < epsilon:
            return int(rng.integers(len(q_actions)))
        # break ties randomly among maxima
        max_q = np.max(q_actions) # 70%, 50% new actions, best
        best = np.flatnonzero(q_actions == max_q)
        return int(rng.choice(best))
    
    def is_near_walls(self, si):
        """Check if state index si is near any wall"""
        if self.walls is None or len(self.walls) == 0:
            return False
        
        s = self.grid_world.to_state(si)
        for wall in self.walls:
            # Check if state is adjacent to wall (including diagonals)
            if abs(s[0] - wall[0]) <= 2 and abs(s[1] - wall[1]) <= 2:
                return True
        return False
    
    def train_with_oracle(self, actions_dict, epsilon_greedy_func, oracle_func):
        """Train with oracle model guidance"""
        eps = self.eps_start
        for ep in range(self.episodes):
            s = self.grid_world.start
            si = self.grid_world.to_index(s)
            done = False
            G = 0.0
            bumpcount = 0
            disc = 1.0
            steps = 0
            reuse = 0
            reject = 0
            for t in range(self.max_steps):
                # Oracle guidance for new diagonal actions (actions 4-7)
                if self.use_conditional is True:
                    a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                    if a >=4: 
                        snext_model, reward_predict, done = oracle_func(si, a)
                        snext_model_i = self.grid_world.to_index(snext_model)
                        # Only accept if it leads to better state value
                        # V(-1) < V(0)
                        # -1, goal is 100
                        if snext_model_i != si and np.max(self.Q[snext_model_i]) > np.max(self.Q[si]):  
                            reuse += 1
                        else: 
                            reject += 1
                            a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                else:
                    a = epsilon_greedy_func(self.Q[si], eps, self.rng)    
                s_next, r, done = self.grid_world.step(s, a, actions_dict, self.rng)
                s_next_i = self.grid_world.to_index(s_next)

                if si == s_next_i:
                    bumpcount += 1

                target = r if done else r + self.grid_world.gamma * np.max(self.Q[s_next_i])
                self.Q[si, a] += self.alpha * (target - self.Q[si, a])

                G += r
                disc *= self.grid_world.gamma
                s, si = s_next, s_next_i
                steps += 1
                if done:
                    break
            self.reuse_count[ep] = reuse    
            self.reject_count[ep] = reject
            self.returns[ep] = G
            self.bumps[ep] = bumpcount
            self.steps_arr[ep] = steps
            
            if ep < self.eps_decay_episodes:
                eps = max(self.eps_end, eps - self.eps_decay)
                

    def train_with_oracle2(self, actions_dict, epsilon_greedy_func):
        """Train the Q-learning agent"""
        eps = self.eps_start
        
        for ep in range(self.episodes):
            s = self.grid_world.start
            si = self.grid_world.to_index(s)
            done = False
            G = 0.0
            bumpcount = 0
            disc = 1.0
            steps = 0

            for t in range(self.max_steps):
                a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                s_next, r, done = self.grid_world.step(s, a, actions_dict, self.rng)
                s_next_i = self.grid_world.to_index(s_next)

                if si == s_next_i:
                    bumpcount += 1
                target = r if done else r + self.grid_world.gamma * np.max(self.Q[s_next_i])
                self.Q[si, a] += self.alpha * (target - self.Q[si, a])

                G += r
                disc *= self.grid_world.gamma
                s, si = s_next, s_next_i
                steps += 1
                if done:
                    break
            self.returns[ep] = G
            self.bumps[ep] = bumpcount
            self.steps_arr[ep] = steps
            
            if ep < self.eps_decay_episodes:
                eps = max(self.eps_end, eps - self.eps_decay)
        
        print(f'Training ({self.n_actions} actions) complete!')