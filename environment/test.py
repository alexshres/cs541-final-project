from .game import reset_game, step
from .moves import encode_move, decode_move, get_legal_moves_mask
from .state import get_state_tensor
from .minmax import minimax_move

def test_environment():
    print("=== TESTING ENVIRONMENT ===")
    game = reset_game()
    print("Game reset OK")

    move_count = 0
    while not game.is_over() and move_count < 10:   # <- library API
        mask = get_legal_moves_mask(game)
        if not any(mask):
            print("No legal moves left!")
            break

        move_idx = minimax_move(game, depth=1)
        decoded = decode_move(move_idx, game)
        print(f"Move {move_count+1}: index {move_idx} -> decoded {decoded}")

        game, reward, done = step(game, move_idx)
        print(f"  Reward: {reward}, Done: {done}")

        tensor = get_state_tensor(game)
        print(f"  State tensor shape: {tuple(tensor.shape)}")

        move_count += 1
        if done:
            print(f"Game over! Winner: {game.get_winner()}")  # <- library API
            break

if __name__ == "__main__":
    test_environment()