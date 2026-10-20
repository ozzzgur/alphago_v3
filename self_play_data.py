import os
import time
import torch
import numpy as np

from game_env import GameState
from mcts import MCTS
from model import AlphaGoZeroNet


class SelfPlayWorker:
    def __init__(self, board_size=19, num_simulations=50):
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.action_space = board_size * board_size + 1  # V3: 362 actions (361 Coordinates + 1 Pass)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # V3 Original Architecture (20 Blocks, 256 Channels)
        self.model = AlphaGoZeroNet(num_res_blocks=20, hidden_channels=256).to(self.device)
        try:
            self.model.load_state_dict(
                torch.load("alphago_model_v3_candidate.pth", map_location=self.device, weights_only=True))
            print("V3 Model weights successfully loaded!")
        except Exception as e:
            print(
                "New V3 Architecture Detected! Old weights were not loaded due to channel mismatch, starting from scratch.")
        self.model.eval()

    def play_single_game(self, game_idx, total_games, max_moves=200):  # CRITICAL ADDITION 1: max_moves added
        game_state = GameState(size=self.board_size)
        mcts = MCTS(self.model, self.device, c_puct=3.5, n_playout=self.num_simulations)

        training_data = []
        turn_count = 0
        consecutive_passes = 0  # Pass counter to break infinite loop

        print(f"\n--- Match {game_idx + 1}/{total_games} Starting (Sudden Death Limit: {max_moves} Moves) ---")

        while True:
            mcts.root = mcts.root.__class__()

            # 1. Run the first simulation to expand the root node
            state_copy = game_state.clone()
            mcts._playout(state_copy)

            # 2. DIRICHLET NOISE (Allows the bot to explore different parts of the board)
            if mcts.root.children:
                epsilon = 0.35
                alpha = 0.03
                actions = list(mcts.root.children.keys())
                noise = np.random.dirichlet([alpha] * len(actions))
                for i, act in enumerate(actions):
                    mcts.root.children[act].p_prior = (1 - epsilon) * mcts.root.children[act].p_prior + epsilon * noise[i]

            # 3. Complete remaining simulations with new probabilities
            for _ in range(mcts.n_playout - 1):
                state_copy = game_state.clone()
                mcts._playout(state_copy)

            # 4. Create Policy array
            policy = np.zeros(self.action_space)
            if mcts.root.children:
                total_visits = sum(child.n_visits for child in mcts.root.children.values())
                for act, child in mcts.root.children.items():
                    policy[act] = child.n_visits / total_visits
            else:
                # 100% PASS if no valid moves are left
                policy[self.action_space - 1] = 1.0

            board_tensor = game_state.get_tensor().clone().cpu().numpy()
            current_player = game_state.current_player
            training_data.append(
                [board_tensor, policy, current_player, None])  # CRITICAL ADDITION 2: 'None' added for Winner

            # 5. Move Selection (Only selected from valid children approved by MCTS)
            valid_actions = list(mcts.root.children.keys())

            if not valid_actions:
                action = self.action_space - 1  # Forced PASS if no children
            elif turn_count < 30:
                # Exploration with temperature in the first 30 turns (Diversifies games)
                probs = [policy[a] for a in valid_actions]
                if sum(probs) > 0:
                    probs = [p / sum(probs) for p in probs]  # Normalize probabilities
                    action = np.random.choice(valid_actions, p=probs)
                else:
                    action = self.action_space - 1
            else:
                # Select the most confident move after turn 30
                action = max(valid_actions, key=lambda a: policy[a])

            player_str = "Black" if current_player == 'b' else "White"

            # 6. Play on Board and Pass Control
            if action == self.action_space - 1:
                print(f"Match {game_idx + 1} | Turn: {turn_count + 1} | {player_str} -> PASSED")
                consecutive_passes += 1
            else:
                row = action // self.board_size
                col = action % self.board_size
                print(f"Match {game_idx + 1} | Turn: {turn_count + 1} | {player_str} -> ({row}, {col})")
                consecutive_passes = 0

            game_state.play_move(action)
            turn_count += 1

            # 7. Game Over and Sudden Death Control
            is_over, winner = game_state.check_game_over()

            # CRITICAL ADDITION 3: max_moves condition and Quick Score Calculation
            if is_over or consecutive_passes >= 2 or turn_count >= max_moves:
                if turn_count >= max_moves and not is_over:
                    print(f"\n200 moves limit reached! Calculating quick score...")
                    board_str = str(game_state.board)
                    b_stones = board_str.count('b') if 'b' in board_str else np.sum(np.array(game_state.board) == 1)
                    w_stones = board_str.count('w') if 'w' in board_str else np.sum(np.array(game_state.board) == -1)

                    winner = 'b' if b_stones > (w_stones + 6.5) else 'w'
                break

        # 8. Distribute Rewards (Win: +1, Loss: -1)
        processed_data = []
        for state_np, pi, player, _ in training_data:
            if winner is None:
                v = 0.0
            else:
                v = 1.0 if winner == player else -1.0
            processed_data.append((state_np, pi, v))

        return processed_data, winner, turn_count

    def generate_dataset(self, num_games=20, save_path="self_play_data.pt"):
        print(f"--- Self-Play Data Generation Starting ({num_games} Matches, {self.num_simulations} Simulations) ---")

        all_data = []
        for i in range(num_games):
            game_start = time.time()
            game_data, winner, turns = self.play_single_game(i, num_games)  # max_moves=200 enabled by default
            all_data.extend(game_data)

            winner_str = "Black" if winner == 'b' else "White" if winner == 'w' else "Draw"
            elapsed = time.time() - game_start
            print(f"--> Match {i + 1}/{num_games} finished! Winner: {winner_str} | Turn: {turns} | Time: {elapsed:.1f}s\n")

        torch.save(all_data, save_path)
        print(f"\nData generation complete! Saved to '{save_path}'.")


if __name__ == "__main__":
    # Let's start a 20-match "Sudden Death" camp with 200 simulations for testing
    worker = SelfPlayWorker(board_size=19, num_simulations=200)
    worker.generate_dataset(num_games=20, save_path="self_play_data.pt")