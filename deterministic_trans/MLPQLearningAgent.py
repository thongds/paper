from QLearningAgent import QLearningAgent
from TransitionModelLearner import TransitionModelLearner
import numpy as np

class MLPQLearningAgent(QLearningAgent):
    """Q-learning agent with learned MLP model for action adaptation"""
    
    def __init__(self, grid_world, n_actions, base_q_table=None, episodes=600, alpha=0.5, 
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=300, 
                 max_steps=200, seed=123, use_model=True, use_conditional=True, use_oracle=False):
        super().__init__(grid_world, n_actions, episodes, alpha, eps_start, eps_end, 
                        eps_decay_episodes, max_steps, seed)
        self.reuse_count = np.zeros(self.episodes, dtype=float)
        self.use_model = use_model
        self.use_conditional = use_conditional
        self.use_oracle = use_oracle
        self.base_q_table = base_q_table
        # Initialize transition model learner
        self.transition_learner = TransitionModelLearner()
        
        # Initialize with base Q-table if provided
        if base_q_table is not None:
            self.Q[:, :base_q_table.shape[1]] = base_q_table
            # Optimistic initialization for new actions
            if use_model:
                V_old = np.max(base_q_table, axis=1) 
                for new_action in range(base_q_table.shape[1], n_actions):
                    self.Q[:, new_action] = V_old
    
    def enhanced_epsilon_greedy(self, q_row, epsilon, rng, encourage_new_action=False):
        if encourage_new_action and len(self.transition_learner.buffer) < 200:
            if rng.random() < 0.3:  
                # Randomly choose one of the diagonal actions (4-7)
                return int(rng.integers(4, len(q_row)))
        
        # Standard epsilon-greedy
        if rng.random() < epsilon:
            return int(rng.integers(len(q_row)))

        max_q = np.max(q_row)
        best = np.flatnonzero(q_row == max_q)
        return int(rng.choice(best))
    
    def train_with_learned_model(self, actions_dict, epsilon_greedy_func, oracle_func):
        eps = self.eps_start
        model_train_frequency = 20  
        
        for ep in range(self.episodes):
            s = self.grid_world.start
            si = self.grid_world.to_index(s)
            done = False
            G = 0.0
            bumpcount = 0
            disc = 1.0
            steps = 0
            reuse = 0
            
            for t in range(self.max_steps):
                a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                # Use oracle or transition learner based on configuration
                if a >= 4:
                    predicted_next_state = self.transition_learner.predict_next_state(s, a)
                    predicted_next_state_i = self.grid_world.to_index(predicted_next_state)
                else:
                    predicted_next_state, _, _ = oracle_func(si, a)
                    predicted_next_state_i = self.grid_world.to_index(predicted_next_state)
                if self.use_oracle:
                    # Use oracle function to predict next state (pass state index)
                    predicted_next_state, _, _ = oracle_func(si, a)
                    predicted_next_state_i = self.grid_world.to_index(predicted_next_state)
                    if predicted_next_state_i != si and np.max(self.base_q_table[predicted_next_state_i]) > np.max(self.base_q_table[si]):
                        reuse += 1
                    else:
                        a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                elif self.transition_learner.can_predict():
                    # Use learned transition model to predict next state
                    predicted_next_state = self.transition_learner.predict_next_state(s, a)
                    predicted_next_state_i = self.grid_world.to_index(predicted_next_state)
                    if predicted_next_state_i != si and np.max(self.base_q_table[predicted_next_state_i]) > np.max(self.base_q_table[si]):
                        reuse += 1
                    else:
                        a = epsilon_greedy_func(self.Q[si], eps, self.rng)
                        
                s_next, r, done = self.grid_world.step(s, a, actions_dict, self.rng)
                s_next_i = self.grid_world.to_index(s_next)

                self.transition_learner.add_experience(s.flatten(), a, s_next.flatten())

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
            self.returns[ep] = G
            self.bumps[ep] = bumpcount
            self.steps_arr[ep] = steps
            
            if ep > 0 and ep % model_train_frequency == 0 and len(self.transition_learner.buffer) > 50:
                self.transition_learner.train_model(batch_size=32, epochs=5)
            
            if ep < self.eps_decay_episodes:
                eps = max(self.eps_end, eps - self.eps_decay)