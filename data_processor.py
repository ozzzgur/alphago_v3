from sgfmill import sgf
import numpy as np
import torch
import os
import glob
from game_env import GameState  # Pulling V3 rules and the 17-channel tensor from here


def parse_sgf(sgf_file_path):
    """Reads the SGF file and converts it to data for AlphaGo V3 (17 Channels, 362 Moves)."""
    with open(sgf_file_path, 'rb') as f:
        game_sgf = sgf.Sgf_game.from_bytes(f.read())

    size = game_sgf.get_size()
    winner = game_sgf.get_winner()

    # Initialize our game environment (GameState) to exactly simulate the SGF match
    game = GameState(size)

    X_data = []  # Board states (17 Channels)
    y_policy = []  # Expert move (0-361)
    y_value = []  # Outcome (+1, -1)

    for node in game_sgf.get_main_sequence():
        color, move = node.get_move()

        if color is None:
            continue

        # 1. Save the 17-channel board state BEFORE the move is made
        board_tensor = game.get_tensor().clone().numpy()
        X_data.append(board_tensor)

        # 2. Convert the move in SGF to our V3 index (362-dimensional space)
        if move is None:
            move_index = size * size  # 361st Index = Pass
        else:
            row, col = move
            move_index = row * size + col

        y_policy.append(move_index)

        # 3. Did this board state win the match?
        if winner == color:
            y_value.append(1.0)
        elif winner is None:
            y_value.append(0.0)  # Draw
        else:
            y_value.append(-1.0)

        # 4. Apply the move to GameState so the past memory (17 channels) is updated
        try:
            game.play_move(move_index)
        except ValueError:
            # If there is an illegal move (e.g., suicide) in the SGF, abort that match
            break

    if len(X_data) == 0:
        return None, None, None

    return torch.tensor(np.array(X_data)), torch.tensor(y_policy), torch.tensor(y_value, dtype=torch.float32)


def parse_sgf_folder(folder_path, max_games=500):
    """Scans all SGFs in the folder and creates massive V3 tensors."""
    all_X, all_policy, all_value = [], [], []
    search_path = os.path.join(folder_path, "*.sgf")
    sgf_files = glob.glob(search_path)

    print(f"A total of {len(sgf_files)} SGF files found in the folder.")

    games_processed = 0
    for file in sgf_files:
        if games_processed >= max_games:
            break

        try:
            X, y_pol, y_val = parse_sgf(file)
            if X is None or len(X) < 10:
                continue

            all_X.append(X)
            all_policy.append(y_pol)
            all_value.append(y_val)

            games_processed += 1
            if games_processed % 50 == 0:
                print(f"{games_processed} matches processed...")
        except Exception as e:
            continue

    print(f"\nA total of {games_processed} matches were successfully converted to 17-Channel V3 tensors.")
    return torch.cat(all_X), torch.cat(all_policy), torch.cat(all_value)