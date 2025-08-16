import torch
from collections import Counter

import agent as ag
import game as g
import moves as mv
import state as st
import minimax as mm


def play_game(dqn_agent:ag.CheckersAgent, minimax_depth:int, dqn_player_num:int):
    """
    Simulates a single game between a DQN agent and a Minimax agent.
    
    Args:
        dqn_agent (ag.CheckersAgent): The DQN agent to play against Minimax.
        minimax_depth (int): Depth for the Minimax agent.
        dqn_player_num (int): Player number for the DQN agent (1 or 2).
        
    Returns:   
        int or None: The winner of the game (1 or 2) or None if it's a draw."""


    game = g.reset_game()

    while not game.is_over():
        current_player = game.whose_turn()
        action_idx = None

        if current_player == dqn_player_num:
            # DQN agent's turn
            state_tensor = st.get_state_tensor(game).unsqueeze(0)
            legal_moves_mask = mv.get_legal_moves_mask(game)
            
            # pure exploitation
            action_idx = dqn_agent.act(state_tensor, legal_moves_mask, eps=0.0)
        else:
            # Minimax agent's turn
            action_idx = mm.minimax_move(game, depth=minimax_depth)

        if action_idx is not None:
            game, _, _ = g.step(game, action_idx)
        else:
            break

    return game.get_winner()

def evaluate_agents(checkpoint_path:str, num_games:int=100, minimax_depth:int=2):
    """
    Loads a DQN agent and evaluates it against Minimax over multiple games.

    Args:
        checkpoint_path (str): Path to the saved DQN model checkpoint.
        num_games (int): The total number of games to play.
        minimax_depth (int): The search depth for the Minimax opponent.
    """
    print("--- Starting Evaluation ---")
    print(f"Loading DQN agent from: {checkpoint_path}")
    
    # trained DQN agent
    dqn_agent = ag.CheckersAgent(checkpoint_path=checkpoint_path)
    
    results = []
    
    for i in range(num_games):
        # alternate starts 
        dqn_starts = (i % 2 == 0)
        dqn_player_num = 1 if dqn_starts else 2
        
        print(f"Game {i+1}/{num_games}... (DQN is Player {dqn_player_num})", end="", flush=True)
        
        winner = play_game(dqn_agent, minimax_depth, dqn_player_num)
        
        # result from the DQN agent's perspective
        if winner is None:
            results.append("Draw")
            print(" -> Draw")
        elif winner == dqn_player_num:
            results.append("Win")
            print(" -> DQN Wins")
        else:
            results.append("Loss")
            print(" -> Minimax Wins")

    stats = Counter(results)
    win_rate = (stats['Win'] / num_games) * 100
    loss_rate = (stats['Loss'] / num_games) * 100
    draw_rate = (stats['Draw'] / num_games) * 100

    print("\n--- Evaluation Complete ---")
    print(f"Games Played: {num_games}")
    print(f"Minimax Depth: {minimax_depth}")
    print("-" * 25)
    print(f"DQN Wins:   {stats['Win']} ({win_rate:.1f}%)")
    print(f"DQN Losses: {stats['Loss']} ({loss_rate:.1f}%)")
    print(f"Draws:      {stats['Draw']} ({draw_rate:.1f}%)")
    print("-" * 25)
