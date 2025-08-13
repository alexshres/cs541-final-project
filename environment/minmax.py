import math
from copy import deepcopy
from .moves import encode_move

def evaluate(game):
    """Simple piece-count evaluation from the current player's perspective."""
    me = game.whose_turn()  
    score = 0.0
    for p in game.board.pieces:              
        val = 1.0 + (0.5 if p.king else 0.0) 
        score += val if p.player == me else -val
    return score

def minimax_move(game, depth=2):
    _, best_action = minimax(game, depth, -math.inf, math.inf, True)
    return best_action

def minimax(game, depth, alpha, beta, maximizing):
    if depth == 0 or game.is_over():         
        return evaluate(game), None

    legal_moves = game.get_possible_moves()  
    if not legal_moves:
        return evaluate(game), None

    best_action = None
    if maximizing:
        max_eval = -math.inf
        for move in legal_moves:
            g2 = deepcopy(game)              
            g2.move(move)                    
            eval_score, _ = minimax(g2, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = encode_move(move)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_action
    else:
        min_eval = math.inf
        for move in legal_moves:
            g2 = deepcopy(game)
            g2.move(move)
            eval_score, _ = minimax(g2, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = encode_move(move)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_action