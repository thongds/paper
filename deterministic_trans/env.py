# Import Required Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Constants and Hyperparameters
N_ROWS, N_COLS = 25, 25
START = (0, 0)
GOAL = (N_ROWS  - 1, N_COLS - 1)

# Cliff area - bottom rows between start and goal (like classic CliffWalk)
CLIFF = [
    (N_ROWS - 1, col) for col in range(1, N_COLS - 1)
] + [
    (N_ROWS - 6, col) for col in range(1, N_COLS - 1)
] + [
    (N_ROWS - 10, col) for col in range(1, N_COLS - 1)
] 
# CLIFF = []
WALLS = [
    
#    (N_ROWS - 1, 21), 
#    (N_ROWS - 2, 22),
#    (N_ROWS - 3, 22),
#    (N_ROWS - 4, 22),
#    (N_ROWS - 5, 22),
#    (N_ROWS - 6, 22),
#    (N_ROWS - 7, 22),
#    (N_ROWS - 8, 22),
#    (N_ROWS - 9, 22),
#    (N_ROWS - 10, 22),
#    (N_ROWS - 11, 22),
#    (N_ROWS - 12, 22),
#    (N_ROWS - 13, 21), 
#    (N_ROWS - 14, 22),
#    (N_ROWS - 15, 22),
#    (N_ROWS - 16, 22),
#    (N_ROWS - 17, 22),
#    (N_ROWS - 18, 22),
#    (N_ROWS - 19, 22),
   
#    (N_ROWS - 10, 21),
#    (N_ROWS - 10, 20), 
#    (N_ROWS - 10, 19), 
#    (N_ROWS - 10, 18),
#    (N_ROWS - 10, 17),
#    (N_ROWS - 10, 16), 
#    (N_ROWS - 10, 15),
#    (N_ROWS - 10, 14), 
#    (N_ROWS - 10, 10),

#     # Additional obstacles - preserved path to goal (no blocks on col 23+)
#     # Upper-left vertical segment (col 4, rows 2-8)
#     (2, 4),
#     (3, 4),
#     (4, 4),
#     (5, 4),
#     (6, 4),
#     (7, 4),
#     (8, 4),

#     # Upper band horizontal segment (row 8, cols 6-12)
#     (8, 6),
#     (8, 7),
#     (8, 8),
#     (8, 9),
#     (8, 10),
#     (8, 11),
#     (8, 12),

#     # Central 3x3 block (rows 11-13, cols 6-8)
#     (11, 6),
#     (11, 7),
#     (11, 8),
#     (12, 6),
#     (12, 7),
#     (12, 8),
#     (13, 6),
#     (13, 7),
#     (13, 8),

#     # Right-upper vertical segment (col 18, rows 2-5)
#     (2, 18),
#     (3, 18),
#     (4, 18),
#     (5, 18),
#    (N_ROWS - 10, 12), 
#    (N_ROWS - 10, 11), 
#    (N_ROWS - 10, 10)
]
# actions
# Action definitions
ACTIONS_4 = { # row = y, col = x
    0: (-1, 0),  # Up
    1: ( 1, 0),  # Down
    2: ( 0,-1),  # Left
    3: ( 0, 1),  # Right
}

ACTIONS_5 = dict(ACTIONS_4)
ACTIONS_5[4] = (1, 1)  # south-east (down-right)

ACTIONS_8 = dict(ACTIONS_5)
ACTIONS_8[5] = (1, -1)  # down-left (south-west)
ACTIONS_8[6] = (-1, 1)  # top-right (north-east)
ACTIONS_8[7] = (-1, -1)  # top-left (north-west)

class GridWorld:
    def __init__(self, n_rows, n_cols, start, goal, walls, step_reward, goal_reward, bump_reward, gamma, cliff = CLIFF, cliff_reward=-200):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.cliff = set(cliff) if cliff is not None else set()
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.gamma = gamma
        self.bump_reward = bump_reward
        self.cliff_reward = cliff_reward
        self.success_prob = 0.7
        self.noise_prob = 0.1
        self.stay_prob = 0.2
        
    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols
    
    def step2(self, state, action, actions, rng=None):
        if state == self.goal:
            return state, 0.0, True
            
        if rng is None:
            rng = np.random.default_rng()
        
        # Determine actual action based on transition probabilities
        rand_val = rng.random()
        
        if rand_val < self.success_prob:
            # Intended action
            actual_action = action
        elif rand_val < self.success_prob + self.noise_prob:
            # Random action from available actions
            actual_action = rng.integers(0, len(actions))
        else:
            # Stay in place (no action)
            actual_action = None
        
        if actual_action is None:
            # Stay in current state
            next_state = state
        else:
            # Execute the actual action
            dr, dc = actions[actual_action]
            next_state = (state[0] + dr, state[1] + dc)
            if not self.in_bounds(next_state) or next_state in self.walls:
                next_state = state  # Stay put if invalid
        
        # Check if agent fell off cliff
        if next_state in self.cliff:
            return self.start, self.cliff_reward, False
        
        reward = self.goal_reward if next_state == self.goal else self.step_reward
        done = next_state == self.goal
        return next_state, reward, done
    
    def step(self, state, action, actions, rng=None):
        if state == self.goal:
            return state, 0.0, True
        
        # Deterministic action execution - always execute the intended action
        dr, dc = actions[action]
        next_state = (state[0] + dr, state[1] + dc)
        
        # If next state is out of bounds or hits a wall, stay in current state
        if not self.in_bounds(next_state) or next_state in self.walls:
            next_state = state
        
        # Check if agent fell off cliff
        if next_state in self.cliff:
            return state, self.cliff_reward, False
        
        if next_state == state:
            reward = self.bump_reward
        else:
            reward = self.goal_reward if next_state == self.goal else self.step_reward
        done = next_state == self.goal
        return next_state, reward, done

    def to_index(self, state):
        r, c = state
        return r * self.n_cols + c

    def from_index(self, index):
        r = index // self.n_cols
        c = index % self.n_cols
        return (r, c)