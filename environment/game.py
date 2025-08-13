from checkers.game import Game
from .moves import decode_move

def reset_game():
    """Start a new checkers game and return the game object."""
    return Game()

def step(game, action_idx):
    """
    Apply an action index to the game.
    Returns: (next_state, reward, done)
    Reward is from the perspective of the player who just moved.
    """
    mover = game.whose_turn()          
    move = decode_move(action_idx, game)
    game.move(move)                    

    done = game.is_over()              
    reward = 0
    if done:
        w = game.get_winner()          
        if w is None:
            reward = 0
        elif w == mover:
            reward = 1
        else:
            reward = -1

    return game, reward, done