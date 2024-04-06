import numpy as np
from game import Connect4
from model import Model
from config import Config
import torch
from mcts import MCTS_NN


if __name__ == "__main__":
    board = Connect4(row=Config.row, col=Config.col)

    arr = [
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0]
    ]

    board.board = np.array(arr)
    # (board.player_1, board.player_2) = (1, -1)
    (board.player_1, board.player_2) = (-1, 1)
    print(board.get_state())
    # model = Model(
    #     n_action = Config.n_action,
    #     num_hidden = Config.num_hidden,
    #     num_resblock = Config.num_res_block,
    #     rate = Config.rate,
    #     row = Config.row,
    #     col = Config.col,
    #     device = Config.device
    # )

    # # This is LR = .01 model
    # model_path = './Models/C4GruhitSPatel/FullBuffer5x5V1/TargetModel_500.pt'

    # # This is LR = .001 model
    # # model_path = "./Models/C4GruhitML/C4CyclicLRV3/TargetModel_500.pt"
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # print("Model Loaded")

    # mcts = MCTS_NN(board, model)
    # print(f"Thinking move for player {board.player_1}")
    # for _ in range(100):
    #     mcts.selection(mcts.root, False)
    
    # print(f"Value: {mcts.root.value:.4f} | Intermediate reward: {mcts.root.W:.4f}")
    # for act, child in mcts.root.children.items():
    #     print(f"{act} | {child.N} | {child.W:.4f} | {child.value:.4f}")