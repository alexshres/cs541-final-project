ACTION_SPACE = 256

def encode_move(move):
    from_sq, to_sq = move
    dir_idx = direction_index(from_sq, to_sq)
    step_type = 0 if abs(to_sq - from_sq) in (3, 4, 5) else 1  # simple heuristic
    return (from_sq - 1) * 8 + dir_idx * 2 + step_type         

def decode_move(action_idx, game):
    # map the index back by matching against current legal moves
    for m in game.get_possible_moves():     
        if encode_move(m) == action_idx:
            return m
    raise ValueError("Action not legal in this position")

def get_legal_moves_mask(game):
    mask = [False] * ACTION_SPACE
    for m in game.get_possible_moves():
        idx = encode_move(m)
        if 0 <= idx < ACTION_SPACE:
            mask[idx] = True
    return mask

''' old version
# determines which diagonal direction (0=up left, 1=up right, 2=down left, 3=down right)
# a move goes on a checkers board based on its start and end square numbers (1–32).
def direction_index(from_sq, to_sq):
    diff = to_sq - from_sq
    if diff in (-4, -5): return 0  # up left
    if diff in (-3, -4): return 1  # up right
    if diff in (3, 4):   return 2  # down left
    if diff in (4, 5):   return 3  # down right
    return 0
'''
def direction_index(from_sq: int, to_sq: int) -> int:
    from_sq = int(from_sq)
    to_sq = int(to_sq)
    diff = to_sq - from_sq
    row = (from_sq - 1) // 4    

    if diff < 0:  # moving up (toward smaller indices)
        if row % 2 == 0:  # even row
            if diff in (-3, -7): return 0  # up left
            if diff in (-4, -9): return 1  # up right
        else:  # odd row
            if diff in (-4, -9): return 0  # up left
            if diff in (-5, -7): return 1  # up right
    else:       # moving down (toward larger indices)
        if row % 2 == 0:  # even row
            if diff in (4, 7):   return 2  # down left
            if diff in (5, 9):   return 3  # down right
        else:  # odd row
            if diff in (3, 7):   return 2  # down left
            if diff in (4, 9):   return 3  # down right

    raise ValueError(f"Unexpected move difference {diff} for from_sq {from_sq}")