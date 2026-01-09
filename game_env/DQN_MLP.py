import sys
import gym
import numpy as np
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from pyvirtualdisplay import Display
import TransitionModelLearnerDQN
reload(TransitionModelLearnerDQN)
from TransitionModelLearnerDQN import TransitionModelLearnerDQN

#from gym.wrappers.monitoring import video_recorder

class DQN_MLP_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # minimum epsilon
        self.epsilon_decay = 0.995 #rate of epsilon decay
        self.learning_rate = 0.001
        self.model = self._build_model()#build the model
        self.transition_learner = TransitionModelLearnerDQN(state_dim=state_size, action_dim=1)
        self.rng = np.random.default_rng(123)
        
    def _build_model(self):
        #this function builds the MLP model with relu activations in 2 the hidden layers, and linear activation in the output layer
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))#specify the type of loss and the learning rate
        return model

    def remember(self, state, action, reward, next_state, done):
        #This function pushes instances of experience into the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, action_list):
        #epsilon-greedy exploration to select actions
        if np.random.rand() <= self.epsilon:
            return np.random.choice(action_list)
        act_values = self.model.predict(state,verbose=0)
        return np.argmax(act_values[0])
    
    def enhanced_epsilon_greedy(self,state, base_action_list, extend_action_list, epsilon, rng, encourage_new_action=False):
        if encourage_new_action:
            if rng.random() < 0.3:  
                return np.random.choice(extend_action_list)
        # Standard epsilon-greedy
        if rng.random() < epsilon:
                return np.random.choice(extend_action_list + base_action_list)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    
    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)#select a minibatch of size batch_size
        states = np.array([self.memory[i][0].flatten() for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)#form the learning targets
        target_f = self.model.predict_on_batch(states)#current estimate of Q values
        target_f[np.arange(batch_size), actions] = targets#update only the Q values corresponding to the action taken. Leave the others unchanged.

        self.model.fit(states, target_f, epochs=1, verbose=0)#train the model

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    def train(self, env, base_actions, expand_actions):
        model_train_frequency = 5
        batch_size = 32 #set batch size
        EPISODES = 100 #Set total number of episodes you want the training to last
        total_rewards_log=[]
        eps = self.epsilon
        reuse = 0
        reject = 0
        for ep in range(EPISODES):
            state, info = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward=0
            done = False
            while done is False:#upto 200 steps in the episode
                
                if ep < 30:  
                    action = self.enhanced_epsilon_greedy(state, base_actions, expand_actions, eps, self.rng, encourage_new_action=True) # This part encourages exploration of new actions 
                else:
                    action = self.enhanced_epsilon_greedy(state, base_actions, expand_actions, eps, self.rng, encourage_new_action=False)
                
                if action in expand_actions and self.transition_learner.can_predict(): 
                    snext_model = self.transition_learner.predict_next_state(state, self.state_size, action, expand_actions)
                    q_values_next_state_model = self.model.predict(snext_model, verbose=0)
                    q_value_current_state = self.model.predict(state, verbose=0)
                    if np.max(q_values_next_state_model) > np.max(q_value_current_state):  
                        reuse += 1
                    else: 
                        reject += 1
                        action = self.enhanced_epsilon_greedy(state, base_actions, expand_actions, eps, self.rng, encourage_new_action=False)
                next_state, reward, done, truncated, _ = env.step(action)#take the action, get reward and next state
                
                if action in expand_actions:
                    self.transition_learner.add_experience(state, next_state, action, expand_actions)        
                
                done = done or truncated
                total_reward+=reward# keep track of total rewards
                next_state = np.reshape(next_state, [1, self.state_size])#reshape states to vectorised form
                self.remember(state, action, reward, next_state, done)#store in replay buffer
                state = next_state
                
                if ep > 0 and ep % model_train_frequency == 0 and len(self.transition_learner.buffer) > 50:
                    self.transition_learner.train_model(batch_size=32, epochs=2)
                
                if len(self.memory) > batch_size:#if there are enough samples to form a batch then call experience replay
                    self.replay(batch_size)#update the DQN weights through replay
            total_rewards_log.append(total_reward)#log the total rewards in the episode
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}, Reuse: {reuse}, Reject: {reject}")
        
        env.close()
        return total_rewards_log