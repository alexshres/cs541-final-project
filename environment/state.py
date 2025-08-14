import torch
from bitboard import (
    convert_game_to_bitboards,
    convert_bitboards_to_tensor,
)

CAPTURE_JUMP_MIN_DELTA = 6  

def square_to_coords(square_1_to_32: int):
    """Map 1..32 dark-square index to (row, col) on an 8x8 board."""
    s = int(square_1_to_32) - 1
    row = s // 4
    col = (s % 4) * 2 + ((row + 1) % 2)
    return row, col

def get_state_tensor(game_instance):
    # 1) pack live game into bitboards
    (
        player1_men_mask,
        player1_kings_mask,
        player2_men_mask,
        player2_kings_mask,
        turn_flag,
    ) = convert_game_to_bitboards(game_instance)

    # 2) build the tensor from bitboards
    state_tensor = convert_bitboards_to_tensor(
        player1_men_mask,
        player1_kings_mask,
        player2_men_mask,
        player2_kings_mask,
        turn_flag,
    )

    # 3) mark capture-start squares in channel 5
    try:
        for move in game_instance.get_possible_moves():
            if isinstance(move, (list, tuple)) and len(move) >= 2:
                start_sq, end_sq = int(move[0]), int(move[1])
                if abs(end_sq - start_sq) >= CAPTURE_JUMP_MIN_DELTA:
                    row, col = square_to_coords(start_sq)
                    state_tensor[5, row, col] = 1.0
    except Exception:
        pass

    return state_tensor