import math
import numpy as np
import torch
import torch.nn.functional as F

class Node:
    """Each move state in the Monte Carlo Tree Search (MCTS)."""

    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # action_index -> Node
        self.n_visits = 0
        self.q_value = 0.0
        self.u_value = 0.0
        self.p_prior = prior_p

    def expand(self, action_priors, game_state=None):
        """
        Expands the tree using valid probabilities.
        V3: game_state now handles Pass and Ko situations internally via get_legal_moves,
        so no extra stone check is needed here.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_p=prob)

    def select(self, c_puct):
        """Selects the most advantageous branch based on the PUCT score (Q + U)."""
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        # Exploration value U
        self.u_value = (c_puct * self.p_prior * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))

        # FPU (First Play Urgency) Correction
        if self.n_visits == 0 and self.parent is not None:
            # CRITICAL UPDATE 1: Since the parent's Q value is from the opponent's perspective, we take it with a MINUS (-).
            fpu_q_value = -self.parent.q_value
        else:
            fpu_q_value = self.q_value

        return fpu_q_value + self.u_value

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += 1.0 * (leaf_value - self.q_value) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS:
    """Monte Carlo Tree Search integrated with the AI model (V3: Pass and FP32 Compatible)."""

    def __init__(self, model, device, c_puct=1.5, n_playout=800):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = Node()

    def _playout(self, game_state):
        node = self.root

        # 1. SELECTION
        while not node.is_leaf():
            if not node.children:
                break
            action, node = node.select(self.c_puct)

            # V3 Security Check: If it is not a Pass move (361) and the square is occupied, break the simulation
            pass_action = game_state.size * game_state.size
            if action != pass_action:
                row = action // game_state.size
                col = action % game_state.size
                if game_state.board.get(row, col) is not None:
                    break

            game_state.play_move(action)

        # If the game is over, reflect the score to the tree
        is_game_over, winner = game_state.check_game_over()
        if is_game_over:
            # CRITICAL UPDATE 2: Correctly handling the draw situation (0.0).
            if winner is None:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game_state.current_player else -1.0
            node.update_recursive(-leaf_value)
            return

        # 2. EVALUATION
        board_tensor = game_state.get_tensor().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)

            # V3 CRITICAL FIX: To prevent "underflow" (probabilities zeroing out) caused by FP16,
            # we strictly perform the softmax operation with float32!
            action_probs = F.softmax(policy_logits, dim=1, dtype=torch.float32).cpu().numpy().flatten()
            leaf_value = value.item()

        # V3: Filter legal moves (including Ko and Pass)
        legal_moves = game_state.get_legal_moves()
        legal_action_probs = {act: action_probs[act] for act in legal_moves}

        sum_probs = sum(legal_action_probs.values())
        if sum_probs > 0:
            legal_action_probs = {act: prob / sum_probs for act, prob in legal_action_probs.items()}
        else:
            prob = 1.0 / len(legal_moves)
            legal_action_probs = {act: prob for act in legal_moves}

        # 3. EXPANSION
        node.expand(legal_action_probs.items(), game_state)

        # 4. BACKUP (Backpropagation)
        node.update_recursive(-leaf_value)

    def get_action(self, game_state, temperature=1e-3):
        # Reset the tree at each turn to focus on the actual board
        self.root = Node()

        # 1. ROOT EXPANSION: Simulate once first to create the root node
        state_copy = game_state.clone()
        self._playout(state_copy)

        # 2. DIRICHLET NOISE: AlphaGo's risk-taking/exploration formula
        if self.root.children:
            epsilon = 0.35  # Weight of the noise (25% chaos, 75% model's own mind)
            alpha = 0.03  # Standard Dirichlet alpha value for the game of Go

            actions = list(self.root.children.keys())
            # Generate random noise for the number of moves
            noise = np.random.dirichlet([alpha] * len(actions))

            # Add noise to the probabilities (p_prior) of the moves in the root node
            for i, action in enumerate(actions):
                child = self.root.children[action]
                child.p_prior = (1 - epsilon) * child.p_prior + epsilon * noise[i]

        # 3. NORMAL TREE SEARCH: Complete the remaining simulations with the new probabilities
        for _ in range(self.n_playout - 1):
            state_copy = game_state.clone()
            self._playout(state_copy)

        if not self.root.children:
            legal_moves = game_state.get_legal_moves()
            if legal_moves:
                return legal_moves[0]
            return None

        # Select the most visited (most reliable) move
        action_visits = [(act, child.n_visits) for act, child in self.root.children.items()]
        best_action = max(action_visits, key=lambda item: item[1])[0]

        return best_action