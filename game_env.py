import numpy as np
import torch
from sgfmill import boards
from collections import deque


class GameState:
    """AlphaGo V3 Game Environment: Includes Pass, Ko rule, Territory Scoring, and 17-Channel History."""

    def __init__(self, size=19):
        self.size = size
        self.board = boards.Board(size)
        self.current_player = 'b'  # Black starts
        self.move_history = []

        # V3: Pass counter (If two consecutive passes occur, the game ends)
        self.consecutive_passes = 0

        # V3: Hash (identity) records of past board states for the Ko rule
        self.state_history_hashes = set()
        self.state_history_hashes.add(self._get_board_hash())

        # V3: Board states of the last 8 moves (black and white) for the 17-Channel Tensor
        self.board_history = deque([self._get_current_binary_boards()] * 8, maxlen=8)

    def _get_board_hash(self):
        """Extracts a unique identity (hash) of the current board and the next player."""
        state = []
        for r in range(self.size):
            for c in range(self.size):
                val = self.board.get(r, c)
                state.append(val if val else '.')
        return "".join(state) + self.current_player

    def _get_current_binary_boards(self):
        """Returns the board as two separate matrices (black stones and white stones)."""
        b_board = np.zeros((self.size, self.size), dtype=np.float32)
        w_board = np.zeros((self.size, self.size), dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                stone = self.board.get(r, c)
                if stone == 'b':
                    b_board[r, c] = 1.0
                elif stone == 'w':
                    w_board[r, c] = 1.0
        return b_board, w_board

    def clone(self):
        """Creates a deep copy of the game for MCTS simulations."""
        new_state = GameState(self.size)
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.move_history = list(self.move_history)
        new_state.consecutive_passes = self.consecutive_passes
        new_state.state_history_hashes = set(self.state_history_hashes)
        new_state.board_history = deque(self.board_history, maxlen=8)
        return new_state

    def get_legal_moves(self):
        """Returns legal moves (empty squares + non-suicide + non-Ko + Pass)."""
        legal_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board.get(row, col) is None:
                    # Test the move on a virtual board
                    test_board = self.board.copy()
                    try:
                        test_board.play(row, col, self.current_player)
                        # Ko rule: If this move takes us to a past position, forbid it
                        next_player = 'w' if self.current_player == 'b' else 'b'
                        test_state = []
                        for r in range(self.size):
                            for c in range(self.size):
                                val = test_board.get(r, c)
                                test_state.append(val if val else '.')
                        test_hash = "".join(test_state) + next_player

                        if test_hash not in self.state_history_hashes:
                            legal_moves.append(row * self.size + col)
                    except ValueError:
                        # Suicide moves throw ValueError by sgfmill, skip
                        pass

        # V3: Always consider the Pass move (Square Count = 361st index) as legal
        pass_move_index = self.size * self.size
        legal_moves.append(pass_move_index)

        return legal_moves

    def play_move(self, move_index):
        """Applies the move to the board and forcefully stops the system in case of Ko/Suicide."""
        pass_index = self.size * self.size

        if move_index == pass_index:
            # Player passed
            self.consecutive_passes += 1
            self.move_history.append(move_index)
        else:
            row = move_index // self.size
            col = move_index % self.size

            if self.board.get(row, col) is not None:
                raise ValueError(f"INVALID MOVE: Row {row}, Column {col} is already occupied!")

            # --- CRITICAL ADDITION: SUICIDE AND KO CHECK (Referee Lock) ---
            test_board = self.board.copy()
            try:
                test_board.play(row, col, self.current_player)
            except ValueError as e:
                raise ValueError(f"INVALID MOVE (Suicide): {e}")

            next_player = 'w' if self.current_player == 'b' else 'b'
            test_state = []
            for r in range(self.size):
                for c in range(self.size):
                    val = test_board.get(r, c)
                    test_state.append(val if val else '.')
            test_hash = "".join(test_state) + next_player

            if test_hash in self.state_history_hashes:
                raise ValueError("INVALID MOVE: Ko rule violation! Infinite loop prevented.")
            # -------------------------------------------------------------

            # If the move is valid, play on the actual board
            self.board.play(row, col, self.current_player)
            self.move_history.append(move_index)
            self.consecutive_passes = 0  # Pass streak broken

        # Pass the turn
        self.current_player = 'w' if self.current_player == 'b' else 'b'

        # Update past memory
        self.state_history_hashes.add(self._get_board_hash())
        self.board_history.append(self._get_current_binary_boards())

    def get_tensor(self):
        """AlphaGo V3 standard: 17-Channel (Last 8 own moves, 8 opponent moves, 1 color) tensor."""
        tensor = np.zeros((17, self.size, self.size), dtype=np.float32)

        # Fill the last 8 moves from past to present (0 oldest, 7 newest)
        for i in range(8):
            # Fixed the typo here: changed '7 i' to '7 - i'
            b_board, w_board = self.board_history[7 - i]

            if self.current_player == 'b':
                tensor[i] = b_board
                tensor[i + 8] = w_board
            else:
                tensor[i] = w_board
                tensor[i + 8] = b_board

        # 16: Which color is playing? (If Black, all 1s; if White, all 0s)
        if self.current_player == 'b':
            tensor[16] = 1.0

        return torch.tensor(tensor)

    def check_game_over(self):
        """Checks if the game is over."""
        # 1. If both sides pass consecutively, the game ends
        if self.consecutive_passes >= 2:
            return True, self._calculate_winner()

        # 2. Absolute turn limit to prevent infinite simulations
        if len(self.move_history) >= self.size * self.size * 2:
            return True, self._calculate_winner()

        return False, None

    def _calculate_winner(self):
        """V3: Advanced Territory Scoring (Tromp-Taylor Logic) + Komi"""
        black_stones, white_stones = 0, 0
        visited = set()

        def get_neighbors(r, c):
            n = []
            if r > 0: n.append((r - 1, c))
            if r < self.size - 1: n.append((r + 1, c))
            if c > 0: n.append((r, c - 1))
            if c < self.size - 1: n.append((r, c + 1))
            return n

        for row in range(self.size):
            for col in range(self.size):
                stone = self.board.get(row, col)
                if stone == 'b':
                    black_stones += 1
                    visited.add((row, col))
                elif stone == 'w':
                    white_stones += 1
                    visited.add((row, col))

        black_territory, white_territory = 0, 0

        for row in range(self.size):
            for col in range(self.size):
                if (row, col) not in visited:
                    empty_group = []
                    borders = set()
                    queue = [(row, col)]
                    visited.add((row, col))

                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        empty_group.append((curr_r, curr_c))

                        for nr, nc in get_neighbors(curr_r, curr_c):
                            n_stone = self.board.get(nr, nc)
                            if n_stone is None:
                                if (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                            else:
                                borders.add(n_stone)

                    if len(borders) == 1:
                        border_color = borders.pop()
                        if border_color == 'b':
                            black_territory += len(empty_group)
                        elif border_color == 'w':
                            white_territory += len(empty_group)

        komi = 7.5
        total_black = black_stones + black_territory
        total_white = white_stones + white_territory + komi

        return 'b' if total_black > total_white else 'w'