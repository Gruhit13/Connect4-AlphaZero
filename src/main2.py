from model import Model
from config import Config
from arena import get_move_for_bot
from game import Connect4
import pygame
from view_board import draw_board, draw_winning_line
import sys
import torch

def play_game(model: Model):
    board = Connect4(
        row = Config.row,
        col = Config.col
    )

    pygame.init()
    screen = pygame.display.set_mode((Config.col*100, (Config.row+1)*100))

    ai_turn = True
    game_end = False
    while True:
        draw_board(screen, board.board)
        draw_winning_line(screen, board.winning_start, board.winning_end)

        # render(board.board)
        if ai_turn and not game_end:
            # print("Getting move from AI...")
            act = get_move_for_bot(board, model, Config.tree_iter)
            # print(f"AI moved in column {act}")
            board, win = board.drop_piece(act)

            if win is not None:
                print("AI has WON")
                print("Board \n")
                print(board)

                print("Winner is...", win)
                game_end = True

            ai_turn = False
            draw_board(screen, board.board)
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_end:
                posx = event.pos[0]
                act = posx//100
                board, win = board.drop_piece(act)
                ai_turn = True

                if win is not None:
                    print("Human has Won")
                    print("Board \n")
                    print(board)
                    game_end = True
            
            if event.type == pygame.MOUSEMOTION and not game_end:
                pygame.draw.rect(screen, (0, 0, 0), (0, 0, 700, 100))
                posx = event.pos[0]

                # If ai is turn 1 then player's turn is second
                if board.player_1 == -1:
                    pygame.draw.circle(screen, (230,230,20), (posx, int(100//2)), 50)
                else:
                    pygame.draw.circle(screen, (52, 186, 235), (posx, int(100//2)), 50)
        
        pygame.display.update()

if __name__ == "__main__":
    model = Model(
        n_action = Config.n_action,
        num_hidden = Config.num_hidden,
        num_resblock = Config.num_res_block,
        rate = Config.rate,
        row = Config.row,
        col = Config.col,
        device = Config.device
    )

    # This is LR = .01 model
    # model_path = './Models/C4GruhitSPatel/FullBuffer5x5V1/TargetModel_500.pt'

    # This is LR = .001 model
    model_path = "./Models/C4GruhitML/C4CyclicLRV3/TargetModel_500.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    play_game(model)
    # print("Model Loaded")