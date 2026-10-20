import time
import torch
from playwright.sync_api import sync_playwright

from game_env import GameState
from mcts import MCTS
from model import AlphaGoZeroNet


class CosumiOtonomAgent:

    def __init__(self, board_size=19, level=1, handicap=0, headless=False):
        self.board_size = board_size
        self.level = level
        self.handicap = handicap
        self.headless = headless

        self.playwright = None
        self.browser = None
        self.page = None

    def start_browser_and_wizard(self):
        print("Connecting to Cosumi.net and starting the game...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_context().new_page()

        self.page.goto("https://www.cosumi.net/en/", wait_until="domcontentloaded", timeout=60000)
        time.sleep(1.5)

        try:
            self.page.click(f"text={self.board_size}")
            time.sleep(1)
            self.page.click(f"text={self.handicap}")
            time.sleep(1)
            self.page.click(f"text={self.level}")
            time.sleep(1.5)

            try:
                self.page.click("text=Black", timeout=2000)
                time.sleep(1)
            except Exception:
                pass

            self.page.wait_for_selector("text=OK", timeout=5000)
            self.page.click("text=OK")
            time.sleep(2)

            print("Game board successfully opened!")
        except Exception as e:
            print(f"Error occurred in the startup wizard: {e}")

    def _get_board_canvas_box(self):
        canvases = self.page.locator("canvas").all()
        for canvas in canvases:
            box = canvas.bounding_box()
            if box and box["width"] > 300 and box["height"] > 300:
                return box
        return None

    def get_opponent_move(self, game_state, our_last_row=None, our_last_col=None):
        try:
            last_move_locator = self.page.locator("#lastmove")
            if not last_move_locator.is_visible(timeout=1000):
                return None

            board_box = self._get_board_canvas_box()
            last_box = last_move_locator.bounding_box()

            if not board_box or not last_box:
                return None

            cell_width = board_box["width"] / self.board_size
            cell_height = board_box["height"] / self.board_size

            last_center_x = last_box["x"] + (last_box["width"] / 2)
            last_center_y = last_box["y"] + (last_box["height"] / 2)

            ui_col = int(round((last_center_x - board_box["x"]) / cell_width - 0.5))
            ui_row = int(round((last_center_y - board_box["y"]) / cell_height - 0.5))

            ui_col = max(0, min(self.board_size - 1, ui_col))
            ui_row = max(0, min(self.board_size - 1, ui_row))

            model_row = (self.board_size - 1) - ui_row
            model_col = ui_col

            if model_row == our_last_row and model_col == our_last_col:
                return None

            if game_state.board.get(model_row, model_col) is not None:
                return None

            return int(model_row), int(model_col)

        except Exception:
            return None

    def make_move_on_board(self, row, col):
        try:
            self.page.wait_for_selector("#board", timeout=5000)
            board_box = self._get_board_canvas_box()
            if not board_box:
                return False

            cell_width = board_box["width"] / self.board_size
            cell_height = board_box["height"] / self.board_size

            ui_row = (self.board_size - 1) - row
            ui_col = col

            target_x = board_box["x"] + cell_width * (ui_col + 0.5)
            target_y = board_box["y"] + cell_height * (ui_row + 0.5)

            self.page.mouse.click(target_x, target_y)
            print(f"OUR MOVE -> Model Mind: ({row}, {col}) | Screen Click: ({ui_row}, {ui_col})")
            return True
        except Exception as e:
            print(f"Error while clicking the move: {e}")
            return False

    def check_move_accepted(self, target_row, target_col):
        try:
            time.sleep(0.8)
            last_move_locator = self.page.locator("#lastmove")
            if not last_move_locator.is_visible():
                return False

            last_box = last_move_locator.bounding_box()
            board_box = self._get_board_canvas_box()

            if board_box and last_box:
                cell_width = board_box["width"] / self.board_size
                cell_height = board_box["height"] / self.board_size

                last_center_x = last_box["x"] + (last_box["width"] / 2)
                last_center_y = last_box["y"] + (last_box["height"] / 2)

                ui_col = int(round((last_center_x - board_box["x"]) / cell_width - 0.5))
                ui_row = int(round((last_center_y - board_box["y"]) / cell_height - 0.5))

                model_row = (self.board_size - 1) - ui_row
                model_col = ui_col

                if model_row == target_row and model_col == target_col:
                    return True
            return False
        except Exception:
            return False

    def run(self):
        self.start_browser_and_wizard()

        print("Loading V3 Model and MCTS engine...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CRITICAL UPDATE: Model reverted to original capacity (20 Blocks, 256 Channels).
        model = AlphaGoZeroNet(num_res_blocks=20, hidden_channels=256).to(device)

        try:
            # Loading our new hybrid intelligence
            model.load_state_dict(torch.load("hybrid.pth", map_location=device, weights_only=True))
            print("V3 Model weights successfully loaded!")
        except Exception as e:
            print(f"Failed to load model weights: {e}")

        model.eval()
        game_state = GameState(size=self.board_size)

        # CRITICAL UPDATE: MCTS Simulation count increased to 400.
        # This way, the bot will calculate 400 different futures before each move and expand to the center.
        mcts = MCTS(model, device, c_puct=3.5, n_playout=200)

        print("\n--- Autonomous Match Loop Started ---")
        max_turns = 300
        our_last_row, our_last_col = None, None

        for turn in range(max_turns):
            print(f"\n--- Turn {turn + 1} ---")
            print("MCTS is calculating the best move...")

            best_action = mcts.get_action(game_state)
            pass_action_index = self.board_size * self.board_size

            while True:
                if best_action is None:
                    print("Game over or no legal moves left.")
                    break

                # V3: Pass move check
                if best_action == pass_action_index:
                    print("ALPHAGO V3 DECIDED TO PASS!")
                    try:
                        self.page.click("text=Pass", timeout=3000)
                    except Exception:
                        print("Pass button not found/clickable on the Cosumi interface.")

                    game_state.play_move(best_action)
                    our_last_row, our_last_col = None, None
                    success = True
                    break
                else:
                    move_index = best_action
                    our_last_row = move_index // self.board_size
                    our_last_col = move_index % self.board_size

                    state_backup = game_state.clone()

                    try:
                        game_state.play_move(move_index)
                    except ValueError as ve:
                        print(f"Invalid move error: {ve}")
                        break

                    success = self.make_move_on_board(our_last_row, our_last_col)
                    if not success:
                        print("Could not input move to the browser, exiting.")
                        break

                    if self.check_move_accepted(our_last_row, our_last_col):
                        break
                    else:
                        print(f"WARNING: Cosumi rejected the move ({our_last_row}, {our_last_col})!")
                        game_state = state_backup

                        if move_index in mcts.root.children:
                            del mcts.root.children[move_index]

                        if not mcts.root.children:
                            best_action = pass_action_index
                        else:
                            action_visits = [(act, child.n_visits) for act, child in mcts.root.children.items()]
                            best_action = max(action_visits, key=lambda item: item[1])[0]

            if best_action is None or not success:
                break

            print("Waiting for opponent's move...")
            opponent_move = None

            for _ in range(25):
                time.sleep(1)
                opponent_move = self.get_opponent_move(game_state, our_last_row, our_last_col)
                if opponent_move:
                    break

            if opponent_move:
                opp_row, opp_col = opponent_move
                print(f"OPPONENT MOVE READ -> Mind: ({opp_row}, {opp_col})")
                opp_move_index = opp_row * self.board_size + opp_col
                try:
                    game_state.play_move(opp_move_index)
                except ValueError as ve:
                    print(f"Opponent played in an invalid area: {ve}")
            else:
                print("Opponent's move could not be detected. The opponent might have passed.")
                game_state.play_move(pass_action_index)

        print("Match completed.")
        time.sleep(10)
        self.browser.close()
        self.playwright.stop()


if __name__ == "__main__":
    agent = CosumiOtonomAgent(board_size=19, level=5, handicap=0, headless=False)
    agent.run()