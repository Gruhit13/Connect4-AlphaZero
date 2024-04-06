import pygame
import numpy as np
import sys
from typing import Tuple, Union
from config import Config

SQUARESIZE = 100

def draw_board(screen, board):
    COLUMN_COUNT = Config.col
    ROW_COUNT = Config.row
    SQUARESIZE = 100
    RADIUS = int(SQUARESIZE/2 - 5)

    BLUE = (52, 186, 235)
    GREY = (70, 71, 70)
    WHITE = (255,255,255)
    YELLOW = (230,230,20)

    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT+1) * SQUARESIZE

    size = (width, height)
    board = np.flip(board,0)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, GREY, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, BLUE, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)

def draw_winning_line(screen, start_pos:Union[None, Tuple[int, int]], end_pos:Union[None, Tuple[int, int]]):
    if start_pos is None or end_pos is None:
        return

    offset = SQUARESIZE//2
    start_line = (SQUARESIZE*start_pos[0]+1+offset, SQUARESIZE*(start_pos[1]+1)+offset)
    end_line = (SQUARESIZE*end_pos[0]+offset, SQUARESIZE*(end_pos[1]+1)+offset)

    # print("Start pos: ", start_pos)
    # print("End pos: ", end_pos)
    # print("Start line: ", start_line)
    # print("End Line: ", end_line)
    pygame.draw.line(screen, (255, 0, 0), start_line, end_line, 10)

def render(board):
    pygame.init()
    screen = pygame.display.set_mode((700,700))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            draw_board(screen,board)
            pygame.display.update()