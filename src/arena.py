from game import Connect4
from agent import Agent
from model import Model
from mcts import MCTS_NN

from typing import Union
import tqdm
import numpy as np

def play_selfgames(agent: Agent, training_games: int):
    
    for _ in tqdm(range(training_games)):
        board = Connect4(row=agent.row, col=agent.col)
        agent.reset(state = board)

        # a buffer list to store the transition of current episode
        episodic_buffer = []

        while not board.is_win() and not board.is_draw():
            # While getting the action the search is performed
            # also the experience is stored in it
            action, policy = agent.get_action()
            episodic_buffer.append([
                board.get_state(),
                board.player_1,
                policy
            ])

            board, _ = board.drop_piece(action)

            # Update the root node of MCTS to one of its child node
            agent.update(action)

        # When the episode is compelted update the buffer
        agent.update_buffer(episodic_buffer)


def get_move_for_bot(state: Connect4, model: Model, tree_iters: int, random_move: bool = False) -> int:
    mcts = MCTS_NN(state = state, model = model)

    for _ in range(tree_iters):
        mcts.selection(mcts.root, random_move)

    policy = mcts.get_policy_pie()
    act = np.argmax(policy)

    return act

def play_game_against_bot(bot1: Model, bot2: Model, tree_iters:int) -> Union[None, int]:
    board = Connect4()
    player_1 = True

    # In function bot1 will be always datagen model to make 1st move
    # bot2 will be main_model to make 2nd move
    # We randomly allow them to make first move based for 50% of time
    flip = False
    if np.random.uniform() < 0.5:
        flip = True
        (bot1, bot2) = (bot2, bot1)
        print("Bot has been flipped")

    while not board.is_win() and not board.is_draw():
        if player_1:
            act = get_move_for_bot(board, model=bot1, tree_iters=tree_iters)
            player_1 = False
        else:
            act = get_move_for_bot(board, model=bot2, tree_iters=tree_iters)
            player_1 = True

        board, win = board.drop_piece(act)
        print(board)

    # Here returning
    # 0 - draw
    # 1 - datagen won
    # -1 - main_model won
    # Hence when flipped we have to handle the values accordingly
    if flip:
        # Thus if we have flipped then main_model who is player 1 if its has won
        # then we want to return -1 for it and vice-versa
        return 0 if win == None else win*-1
    else:
        return 0 if win == None else win