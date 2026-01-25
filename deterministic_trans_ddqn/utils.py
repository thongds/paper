import numpy as np
from env import *
    
    # Define fallback constants if needed
def epsilon_greedy(q_actions, epsilon, rng, encourage_new_action=False, epissodes=None):
    if encourage_new_action and epissodes is not None and epissodes < 100:
        """Encourage exploration of new actions in early episodes"""
        if rng.random() < 0.3:  
                # Randomly choose one of the diagonal actions (4-7)
                return int(rng.integers(4, len(q_actions)))
    """Epsilon-greedy action selection"""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_actions)))
    # break ties randomly among maxima
    max_q = np.max(q_actions)
    best = np.flatnonzero(q_actions == max_q)
    return int(rng.choice(best))

def moving_average(x, w=20):
    """Calculate moving average"""
    if len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w)/w, mode='valid')

# # Legacy function wrappers for compatibility
# def to_index(s):
#     return grid_world.to_index(s)

# def from_index(i):
#     return grid_world.from_index(i)

# def step(s, a, actions, rng=None):
#     return grid_world.step(s, a, actions, rng)

# def oracle_model(s, action):
#     """Oracle model that predicts the next state for diagonal actions (4-7)"""
#     s_model_next, r_model, done_model = grid_world.step(grid_world.from_index(s), action, ACTIONS_8)
#     return s_model_next

class Visualizer:
    """Visualization utilities for GridWorld policies and Q-tables"""
    ARROWS = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '↘', 5: '↙', 6: '↗', 7: '↖'}

    @staticmethod
    def derive_policy(Q, n_rows, n_cols):
        """Derive policy from Q-table"""
        policy = np.zeros((n_rows, n_cols), dtype=int)
        for i in range(n_rows):
            for j in range(n_cols):
                state_index = i * n_cols + j
                policy[i, j] = np.argmax(Q[state_index])
        return policy

    @staticmethod
    def render_policy(policy, n_rows, n_cols, walls, start, goal):
        """Render policy with arrows"""
        print("\nPolicy Visualization:")
        for i in range(n_rows):
            row_str = ""
            for j in range(n_cols):
                if (i, j) == start:
                    action = policy[i, j]
                    arrow = Visualizer.ARROWS.get(action, '?')
                    row_str += " S "
                elif (i, j) == goal:
                    row_str += " G "
                elif (i, j) in walls:
                    row_str += " ▓ "
                elif (i, j) in CLIFF:
                    row_str += " C "
                else:
                    action = policy[i, j]
                    arrow = Visualizer.ARROWS.get(action, '?')
                    row_str += f" {arrow} "
            print(row_str)

    @staticmethod
    def print_value_grid(Q, n_rows, n_cols):
        """Print Q-table values as a grid"""
        print(f"\nQ-table shape: {Q.shape}")
        V = np.max(Q, axis=1).reshape(n_rows, n_cols)
        print(f"Value function (max Q-values):")
        for i in range(n_rows):
            row_str = ""
            for j in range(n_cols):
                row_str += f"{V[i, j]:6.1f} ?"
            print(row_str)