import random

import numpy as np


class MatrixGame:
    def __init__(self, size=4, action_space=None):
        self.size = size
        self.action_space = action_space if action_space is not None else [-2, 1, 2]
        self.reset()

    def reset(self):
        self.board = np.full((self.size, self.size), np.nan)
        self.history = []

        initial_fills = 2
        filled_count = 0
        while filled_count < initial_fills:
            r, c = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if np.isnan(self.board[r, c]):
                self.board[r, c] = random.choice(self.action_space)
                filled_count += 1

        return self.get_state()

    def get_state(self):
        return tuple(np.nan_to_num(self.board, nan=0.0).flatten())

    def get_empty_cells(self):
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if np.isnan(self.board[r, c])
        ]

    def check_row_sum_zero(self, board_state):
        temp_board = np.nan_to_num(board_state, nan=0.0)
        for row_index in range(self.size):
            if not np.isnan(board_state[row_index]).any():
                if np.sum(temp_board[row_index]) == 0:
                    return True
        return False

    def step(self, action_value):
        empty_cells = self.get_empty_cells()
        if not empty_cells:
            return self.get_state(), 0.0, True, "Board Full"

        self.history.append(self.board.copy())
        target_cell = empty_cells[0]
        self.board[target_cell] = action_value

        reward = 0.0
        done = False
        info = ""

        if self.check_row_sum_zero(self.board):
            reward -= 50.0
            self.board = self.history.pop()
            return self.get_state(), reward, False, "Penalty: Row sum became 0. Traced back."

        if len(self.get_empty_cells()) == 0:
            done = True
            determinant = np.linalg.det(self.board)
            if abs(determinant) < 1e-9:
                reward += 100.0
                info = "Win: Matrix is Singular!"
            else:
                shaped_reward = 100.0 / (abs(determinant) + 1.0)
                reward += shaped_reward
                info = (
                    "Game Over: Board Full. "
                    f"Determinant = {determinant:.2f} | Reward: {shaped_reward:.2f}"
                )

        return self.get_state(), reward, done, info
