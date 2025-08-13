from .game import reset_game, step
from .moves import encode_move, decode_move, get_legal_moves_mask
from .state import get_state_tensor
from .minmax import minimax_move
from .bitboard import convert_game_to_bitboards, convert_bitboards_to_tensor

def test_bitboard_roundtrip(game):
    p1m, p1k, p2m, p2k, turn_flag = convert_game_to_bitboards(game)

    tensor_from_bitboards = convert_bitboards_to_tensor(p1m, p1k, p2m, p2k, turn_flag)
    tensor_direct = get_state_tensor(game)

    assert tensor_from_bitboards.shape == (6, 8, 8)
    assert tensor_direct.shape == (6, 8, 8)

    counts_from_bitboards = [int(tensor_from_bitboards[ch].sum().item()) for ch in range(4)]
    counts_direct = [int(tensor_direct[ch].sum().item()) for ch in range(4)]

    print(f"  Bitboards to tensor counts: {counts_from_bitboards} | Direct counts: {counts_direct}")
    return counts_from_bitboards == counts_direct

def test_environment():
    print("=== TESTING ENVIRONMENT ===")
    game = reset_game()
    test_bitboard_roundtrip(game)
    print("Game reset OK")

    move_count = 0
    while not game.is_over() and move_count < 10:   
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
            print(f"Game over! Winner: {game.get_winner()}")  
            break

if __name__ == "__main__":
    test_environment()