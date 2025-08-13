import numpy as np
import torch

BITBOARD_DTYPE = np.uint32  # compact 32-bit masks

def square_to_coords(square_1_to_32: int):
    """Map 1..32 dark-square index to (row, col) on an 8x8 board."""
    s = int(square_1_to_32) - 1
    row = s // 4
    col = (s % 4) * 2 + ((row + 1) % 2)
    return row, col

def convert_game_to_bitboards(game_instance):
    """
    Pack the live game into four bitboards + a turn flag.
    Returns: (p1_men, p1_kings, p2_men, p2_kings, turn_flag)
    """
    player1_men_mask = 0
    player1_kings_mask = 0
    player2_men_mask = 0
    player2_kings_mask = 0

    for piece in getattr(game_instance.board, "pieces", []):
        piece_position = getattr(piece, "position", None)
        if piece_position is None:
            continue  # captured or off-board
        bit_position = 1 << (int(piece_position) - 1)  # 1..32 -> bit 0..31

        if piece.player == 1:
            if getattr(piece, "king", False):
                player1_kings_mask |= bit_position
            else:
                player1_men_mask |= bit_position
        else:
            if getattr(piece, "king", False):
                player2_kings_mask |= bit_position
            else:
                player2_men_mask |= bit_position

    turn_flag = np.uint8(game_instance.whose_turn())  # 1 or 2

    return (
        np.array(player1_men_mask, BITBOARD_DTYPE),
        np.array(player1_kings_mask, BITBOARD_DTYPE),
        np.array(player2_men_mask, BITBOARD_DTYPE),
        np.array(player2_kings_mask, BITBOARD_DTYPE),
        turn_flag,
    )

def convert_bitboards_to_tensor(
    player1_men_mask,
    player1_kings_mask,
    player2_men_mask,
    player2_kings_mask,
    turn_flag,
):
    """
    Build a 6x8x8 tensor from the four bitboards plus the turn flag.
      Channel 0: my men
      Channel 1: my kings
      Channel 2: opponent men
      Channel 3: opponent kings
      Channel 4: side-to-move (all ones)
      Channel 5: left as zeros here
    """
    tensor_state = torch.zeros((6, 8, 8), dtype=torch.float32)

    def paint_bitboard_on_channel(bitboard_mask: int, channel_index: int):
        mask_value = int(bitboard_mask)
        while mask_value:
            least_significant_bit = mask_value & -mask_value
            square_index = least_significant_bit.bit_length() - 1  # 0..31
            row, col = square_to_coords(square_index + 1)
            tensor_state[channel_index, row, col] = 1.0
            mask_value ^= least_significant_bit

    if int(turn_flag) == 1:
        paint_bitboard_on_channel(player1_men_mask, 0)
        paint_bitboard_on_channel(player1_kings_mask, 1)
        paint_bitboard_on_channel(player2_men_mask, 2)
        paint_bitboard_on_channel(player2_kings_mask, 3)
    else:
        paint_bitboard_on_channel(player2_men_mask, 0)
        paint_bitboard_on_channel(player2_kings_mask, 1)
        paint_bitboard_on_channel(player1_men_mask, 2)
        paint_bitboard_on_channel(player1_kings_mask, 3)

    tensor_state[4, :, :] = 1.0  # side-to-move plane
    return tensor_state