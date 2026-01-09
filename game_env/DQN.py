import sys
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
#from gym.wrappers.monitoring import video_recorder

class DQNAgent:
    def __init__(self, state_size, action_list):
        self.state_size = state_size
        self.action_list = action_list
        self.memory = []
        self.max_memory_size = 10000  # Limit replay buffer size
        self.gamma = 0.99  # Increased discount rate for long-term planning
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05  # Slightly higher minimum for continued exploration
        self.epsilon_decay = 0.995  # rate of epsilon decay
        self.learning_rate = 0.001
        self.model = self._build_model()  # build the model
        self.target_model = self._build_model()  # Target network for stability
        self.update_target_model()

    def _build_model(self):
        # Larger network for better learning capacity
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(len(self.action_list), activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # This function pushes instances of experience into the replay buffer
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def act(self, state, action_list):
        # epsilon-greedy exploration to select actions
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.randint(len(action_list))
        else:
            act_values = self.model.predict(state, verbose=0)
            action_idx = np.argmax(act_values[0])
        return action_list[action_idx]  # Map index to actual action

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0].flatten() for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        # Use TARGET network for next state Q-values (Double DQN style)
        targets = rewards + self.gamma * np.amax(self.target_model.predict_on_batch(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict_on_batch(states)
        target_f[np.arange(batch_size), actions] = targets

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=200):
        BATCH_SIZE = 64  # Larger batch size
        MIN_MEMORY_SIZE = 500  # Wait for more experiences before training
        TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
        total_reward = []

        for e in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            reward_count = 0
            steps = 0

            while not done:
                action = self.act(state, self.action_list)
                action_idx = self.action_list.index(action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward_count += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action_idx, reward, next_state, done)
                state = next_state
                steps += 1

                # Only train after collecting enough experiences
                if len(self.memory) > MIN_MEMORY_SIZE:
                    self.replay(BATCH_SIZE)

            # Update target network periodically
            if e % TARGET_UPDATE_FREQ == 0:
                self.update_target_model()

            total_reward.append(reward_count)
            print(f"Episode {e+1}/{episodes}, Reward: {reward_count:.2f}, Steps: {steps}, Epsilon: {self.epsilon:.3f}")

        env.close()
        return total_reward

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)