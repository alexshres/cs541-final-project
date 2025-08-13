import torch

# jumps on a 1..32 checkers index typically move 7 or 9 squares.
# program detect jumps without explicitly listing all possible jumps
CAPTURE_JUMP_MIN_DELTA = 6

# converts the checkers game state into a structured numeric tensor
def get_state_tensor(game):
    """
    Return the 6,8,8 tensor with channels:
      0: my men
      1: my kings
      2: opponent men
      3: opponent kings
      4: side-to-move (all ones)
      5: capture-starter squares for the side to move
    """
    tensor = torch.zeros((6, 8, 8), dtype=torch.float32)
    me = game.whose_turn()

    # loop through all pieces on the board, convert their positions to (row, col),
    # check if they’re mine or the opponent’s, and mark them in the correct tensor channel.
    pieces = getattr(game.board, "pieces", [])
    for p in pieces:
        pos = getattr(p, "position", None)
        if pos is None:
            continue  # captured/off-board piece
        r, c = square_to_coords(pos)
        is_king = bool(getattr(p, "king", False))
        player = getattr(p, "player", None)
        if player == me:
            tensor[1 if is_king else 0, r, c] = 1.0
        else:
            tensor[3 if is_king else 2, r, c] = 1.0

    # side-to-move plane
    tensor[4, :, :] = 1.0

    # marks all starting squares where the current player can make a capture move in channel 5 of the tensor.
    try:
        for m in game.get_possible_moves():
            # m is like [from_sq, to_sq]
            if isinstance(m, (list, tuple)) and len(m) >= 2:
                if abs(int(m[1]) - int(m[0])) >= CAPTURE_JUMP_MIN_DELTA:
                    r, c = square_to_coords(m[0])
                    tensor[5, r, c] = 1.0
    except Exception:
        pass

    return tensor 

# converts a checkers square number (1 to 32) into its row and column position on an 8×8 board.
def square_to_coords(square_1_to_32: int):
    """
    map 1..32 dark-squares indexing to (row, col) on an 8x8 matrix.
    """
    s = int(square_1_to_32) - 1
    row = s // 4
    col = (s % 4) * 2 + ((row + 1) % 2)
    return row, col