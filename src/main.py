from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from game import Connect4
from model import Model
from config import Config
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from arena import get_move_for_bot
import torch

class Request(BaseModel):
    board: List[List[int]]
    currentPlayer: str
    randomMoves: Union[None, bool]
    mctsIterations: Union[None, int]

# Create an application instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Create the model
model = Model(
    n_action = Config.n_action,
    num_hidden = Config.num_hidden,
    num_resblock = Config.num_res_block,
    rate = Config.rate,
    row = Config.row,
    col = Config.col,
    device = Config.device
)
model.load_state_dict(torch.load(Config.checkpoint_path))
model.eval()

@app.get("/")
def root():
    return {"message": "This is a temporary response"}

@app.post("/get_move")
def get_move(req: Request):
    global model
    board_arr = np.array(req.board)
    board = Connect4()
    board.board = board_arr

    if req.currentPlayer == "yellow":
        (board.player_1, board.player_2) = (board.player_2, board.player_1)

    # TODO: change the tree_iter to req.parameters
    act = get_move_for_bot(
        state = board,
        model = model,
        tree_iters = req.mctsIterations,
        random_move = req.randomMoves
    )

    return {'move': int(act)}