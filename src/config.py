from typing import Tuple
import torch

class Config:
    # Board
    row:int = 6
    col:int = 7

    # Neural Network
    num_hidden:int = 64
    num_res_block:int = 4
    rate: float = 0.3
    obs_shape: Tuple[int, int, int] = (4, row, col)
    n_action: int = col
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_path: str = "../Models/azv3.pt"

    # Optimizer
    base_lr: float = 0.01
    weight_decay: float = 1e-4

    # Monte-carlo tree search
    temperature = 1.0
    tree_iter = 100

    # Training
    selfplay_games:int = 50
    epoch:int = 10
    batch_size:int = 128

    # Tournament
    eval_games: int = 10

    # How much elo rating should be given per winning
    k: int = 10

    # model update threshold
    threshold: float = 0.55

    # How many time you want to play selfplay games and train model
    total_iters:int = 40

    # Parallel_games
    parallel_run: int = 4

    DIRICHLET_ALPHA: float = 0.3 # Avg legal move / 75% of total move
    EPSILON: float = 0.25