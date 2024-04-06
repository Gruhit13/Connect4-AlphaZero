from typing import Tuple
import numpy as np

class Buffer:
    def __init__(self, n_action: int, obs_shape: Tuple[int, int, int]):
        self.n_action = n_action
        self.obs_shape = obs_shape
        self.mem_size = 0

        # Creating empty lists for storing value. Provide dynamicness
        self.state = []
        self.value = []
        self.policy = []

    def store_experience(self, state: np.ndarray, value: float, policy: np.ndarray):
        self.state.append(state)
        self.value.append(value)
        self.policy.append(policy)

        self.mem_size += 1

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]: # type: ignore
        # shuffle the memmory
        np.random.shuffle(self.state)
        np.random.shuffle(self.value)
        np.random.shuffle(self.policy)

        for start_idx in range(0, self.mem_size, batch_size):
            end_idx = min(start_idx+batch_size, self.mem_size)
            s = self.state[start_idx:end_idx]
            v = self.value[start_idx:end_idx]
            p = self.policy[start_idx:end_idx]

            yield (np.array(s), np.array(v), np.array(p))

    # Reset the the buffer to store new experience
    def reset(self) -> None:
        self.state = []
        self.value = []
        self.policy = []

        self.mem_size = 0

    # Return the length of the buffer
    def __len__(self):
        return self.mem_size