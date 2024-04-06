from copy import deepcopy
import numpy as np
from typing import Union

class Connect4:
    def __init__(self, board:'Connect4'=None, row:int = 6, col:int =7):
        self.row = row
        self.col = col
        self.player_1 = 1
        self.player_2 = -1

        self.board = np.zeros((self.row, self.col))

        self.winning_start = None
        self.winning_end = None


        if board is not None:
            self.__dict__ = deepcopy(board.__dict__)

    def drop_piece(self, action: int) -> 'Connect4':

        board = Connect4(board=self)

        # Find the row in that column which is valid to drop piece
        valid_row_idx = sum(board.board[:, action] == 0) - 1
        board.board[valid_row_idx, action] = self.player_1
        (board.player_1, board.player_2) = (self.player_2, self.player_1)

        return board, board.is_win()

    # Get the encoded state for the board
    def get_state(self) -> np.ndarray:
        # Create a layer to state the player turn
        turn = np.ones_like(self.board) if self.player_1 == 1 else np.zeros_like(self.board)
        enc_state = np.stack(
            (self.board == -1, self.board == 0, self.board == 1, turn)
        ).astype(np.int32)

        return enc_state

    # check if the board results in a draw state
    def is_draw(self):
        return (self.board != 0).all()

    def is_win(self) -> Union[None, int]:
        # Initially no one is winner
        winner = None

        # Check for columns
        if self.col_win():
            winner = self.player_2
        # Check for rows
        elif self.row_win():
            winner = self.player_2
        # Check for diagonals
        elif self.diag_win():
            winner = self.player_2
        return winner

    # Check for column win
    def col_win(self) -> bool:
        # Iterate over each column
        for c in range(self.col):
            # for 4 consequtive rows
            for r in range(self.row-3):
                # if the the all 4 element are of player who made move then its win
                if sum(self.board[r:r+4, c] == self.player_2) == 4:
                    self.winning_start = (c, r)
                    self.winning_end = (c, r+3)
                    return True

        return False

    # check for win in row
    def row_win(self) -> bool:
        # Iterate over each row
        for r in range(self.row):
            # For 4 consequtive cols
            for c in range(self.col-3):
                # If all of 4 elements are of player who made move then its win
                if sum(self.board[r, c:c+4] == self.player_2) == 4:
                    self.winning_start = (c, r)
                    self.winning_end = (c+3, r)
                    return True

        return False

    # check for win in diagonal
    def diag_win(self) -> bool:
        # For a window of 4x4 if the main diag or other diag has
        # same disc of player who made move then its a win
        for r in range(self.row-3):
            for c in range(self.col-3):
                # Get a window of size 4x4
                window = self.board[r:r+4, c:c+4]

                # If all 4 element of main diag(/) is player who made move then its win
                if sum(np.diag(window) == self.player_2) == 4:
                    self.winning_start = (c+3, r)
                    self.winning_end = (c, r+3)
                    # print("WinningMain Diag: ", self.winning_start, " - ", self.winning_end)
                    return True

                # If all 4 element of other diag(\) is player who made move then its win
                if sum(np.diag(window[:, ::-1]) == self.player_2) == 4:
                    self.winning_start = (c, r)
                    self.winning_end = (c+3, r+3)
                    # print("WinningMain Diag: ", self.winning_start, " - ", self.winning_end)
                    return True

        return False

    # get a list of valid move that can be played by the current player
    def get_valid_moves(self) -> np.array:
        valid_cols = [False]*self.col
        for c in range(self.col):
            if self.board[0, c] == 0:
                valid_cols[c] = True

        return np.array(valid_cols, dtype=bool)

    def __str__(self) -> str:
        print_str = ""
        for r in range(self.row):
            for c in range(self.col):
                print_str += f"{self.board[r, c]:>3.0f}"

            print_str += "\n"

        return print_str