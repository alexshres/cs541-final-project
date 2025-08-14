import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

import dqn
import agent as ag
import game as g
import moves as mv
import state as st

EPISODES = 5000
TIMESTEPS = 2000
EPSILON = 1.0 
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01

def train_agent(episodes=EPISODES, 
                timesteps=TIMESTEPS,
                eps_start=EPSILON,
                eps_end=EPSILON_MIN,
                eps_decay=EPSILON_DECAY):
    """Train the Checkers agent using self-play."""

    scores = []     # scores from each episode

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start

    agent = ag.CheckersAgent()

    for episode in range(episodes):
        game = g.reset_game()
        score = 0

        for t in range(timesteps):
            current_player = game.whose_turn()

            # state from current player's perspective
            state = st.get_state_tensor(game)


            legal_moves_mask = mv.get_legal_moves_mask(game)
            state_batched = state.unsqueeze(0)
            action_idx = agent.act(state_batched, legal_moves_mask, eps)

            # this is from the mover's perspective
            game, reward, done = g.step(game, action_idx)

            # grab next state from same player's perspective
            if not done:
                next_state = st.get_state_tensor(game)
            else:
                next_state = state  # terminal state

            agent.step(state, action_idx, reward, next_state, done)

            if done:
                break

        # track score from player 1's perspective
        if game.is_over():
            winner = game.get_winner()
            if winner is None:
                score = 0
            elif winner == 1:
                score = 1
            else:
                score = -1

        
        scores_window.append(score)  
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  

        print(f"Episode {episode+1}/{episodes}, Score: {score}, Epsilon: {eps:.2f}")

        if (episode + 1) % 100 == 0:
            print(f"Average score over last 100 episodes: {sum(scores_window)/len(scores_window):.2f}")

    return scores


