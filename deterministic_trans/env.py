# Import Required Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Constants and Hyperparameters
N_ROWS, N_COLS = 25, 25
START = (0, 0)
GOAL = (24, 24)
WALLS = [
    # Original walls
    # (2, 2), (3, 4), (5, 1), (7, 3), (4, 6), (8, 2), (6, 7),
    # (2, 15), (4, 18), (6, 12), (8, 20), (3, 22), (7, 16), (9, 14),
    # (15, 2), (18, 5), (20, 1), (16, 7), (22, 4), (19, 8), (17, 3),
    # (15, 18), (17, 21), (19, 16), (21, 19), (18, 23), (16, 15), (20, 17),
    # (10, 10), (12, 8), (14, 12), (11, 15), (13, 6), (9, 18), (15, 9),
    # (5, 11), (8, 16), (11, 3), (14, 20), (17, 6), (20, 13), (23, 8),
    # (1, 9), (4, 14), (7, 19), (10, 22), (13, 17), (16, 11), (19, 2),
    # (22, 15), (6, 5), (9, 8), (12, 18), (15, 4), (18, 14), (21, 7),
    # (1, 23), (23, 1), (3, 0), (0, 3), (24, 21), (21, 24),
    
    # # Additional strategic walls to make diagonal actions useful
    # # Vertical barriers that force diagonal movement
    # (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),  # Vertical wall forcing diagonal detour
    # (5, 19), (6, 19), (7, 19), (8, 19), (9, 19),  # Another vertical barrier
    
    # # Horizontal barriers
    # (12, 12), (12, 13), (12, 14), (12, 15), (12, 16),  # Horizontal wall
    # (18, 8), (18, 9), (18, 10), (18, 11), (18, 12),  # Another horizontal barrier
    
    # # L-shaped barriers that encourage diagonal shortcuts
    # (10, 5), (11, 5), (12, 5), (10, 6), (10, 7),  # L-shape 1
    # (14, 18), (14, 19), (14, 20), (15, 20), (16, 20),  # L-shape 2
    
    # # Staircase patterns where diagonal is clearly better
    # (3, 8), (4, 9), (5, 10), (6, 11), (7, 12),  # Staircase 1
    # (18, 2), (19, 3), (20, 4), (21, 5), (22, 6),  # Staircase 2
    
    # # Cross patterns creating diagonal corridors
    # (13, 10), (13, 11), (13, 12), (12, 11), (14, 11),  # Cross 1
    # (8, 8), (8, 9), (8, 10), (7, 9), (9, 9),  # Cross 2
    
    # # Maze-like sections where diagonal movement is advantageous
    # (1, 12), (2, 12), (3, 12), (1, 14), (3, 14),  # Maze section 1
    # (20, 20), (21, 20), (22, 20), (20, 22), (22, 22),  # Maze section 2
    
    # # Additional scattered obstacles to increase complexity
    # (11, 1), (13, 2), (15, 3), (17, 4), (19, 5),  # Diagonal line of obstacles
    # (4, 21), (6, 20), (8, 19), (10, 18), (12, 17),  # Another diagonal line
    
    # # Border reinforcements to prevent edge shortcuts
    # (0, 12), (1, 12), (2, 12), (23, 12), (24, 12),  # Middle column obstacles
    # (12, 0), (12, 1), (12, 2), (12, 23), (12, 24),  # Middle row obstacles
]
STEP_REWARD = -0.1
BUMP_REWARD = -0.1
GOAL_REWARD = 10
GAMMA = 0.95
ALPHA = 0.5
EPISODES = 600
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 300
MAX_STEPS = 200
SEED = 123

# actions
# Action definitions
ACTIONS_4 = {
    0: (-1, 0),  # Up
    1: ( 1, 0),  # Down
    2: ( 0,-1),  # Left
    3: ( 0, 1),  # Right
}

ACTIONS_5 = dict(ACTIONS_4)
ACTIONS_5[4] = (1, 1)  # south-east (down-right)

class GridWorld:
    def __init__(self, n_rows, n_cols, start, goal, walls, step_reward, goal_reward, gamma):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.gamma = gamma

    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, state, action, actions, rng=None):
        if state == self.goal:
            return state, 0.0, True
        
        # Deterministic action execution - always execute the intended action
        dr, dc = actions[action]
        next_state = (state[0] + dr, state[1] + dc)
        
        # If next state is out of bounds or hits a wall, stay in current state
        if not self.in_bounds(next_state) or next_state in self.walls:
            next_state = state
        
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