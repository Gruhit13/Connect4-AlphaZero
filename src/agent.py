from model import Model
from buffer import Buffer
from game import Connect4
from mcts import MCTS_NN

import numpy as np
from typing import Tuple, List

class Agent:
    def __init__(self, row:int, col:int, n_action: int, obs_shape: Tuple[int, int, int],
                 model: Model, iteration: int, temperature:float):

        self.row = row
        self.col = col
        self.n_action = n_action
        self.obs_shape = obs_shape
        self.iteration = iteration
        self.temperature = temperature

        # Create buffer instance
        self.buffer = Buffer(n_action=self.n_action, obs_shape=self.obs_shape)

        # Target model instance
        self.target_model = model

    # Reset the MCTS class instance and buffer
    def reset(self, state: Connect4, reset_buffer: bool = False) -> None:
        # Reset the state of the Monte-carlo tree search instance
        self.mcts = MCTS_NN(state=state, model=self.target_model)

    # Reset the buffer
    def reset_buffer(self) -> None:
        self.buffer.reset()

    # Get the policy from mcts simulation
    def perform_mcts(self) -> np.ndarray:
        for _ in range(self.iteration):
            self.mcts.selection(self.mcts.root, add_dirichlet=True)

        policy = self.mcts.get_policy_pie(self.temperature)

        return policy

    # Get an action for any state
    def get_action(self) -> int:
        policy = self.perform_mcts()
        action = np.random.choice(self.n_action, p=policy)
        return action, policy

    # This method updates the buffer and send it to the buffer object
    def update_buffer(self, episodic_buffer: List)->None:
        # Get the last index of the episodic buffer
        idx = len(episodic_buffer) - 1

        # Always the last state will have value 1 as it would be the winning move
        value = 1
        while idx >= 0:
            episodic_buffer[idx][1] = value
            value *= -1 # For parent the value is negative
            idx -= 1 # Go to the previous experience tuple

        for state, value, policy in episodic_buffer:
            self.buffer.store_experience(
                state = state,
                value = value,
                policy = policy
            )

    # Update the root to set it to one of its child node
    # based on the actio taken in the above method `get_action()`
    def update(self, action: int) -> None:
        self.mcts.update_root(action)