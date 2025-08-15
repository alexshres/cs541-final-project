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
    """Train the Checkers agent using improved self-play."""

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    agent = ag.CheckersAgent()
    
    for episode in range(episodes):
        game = g.reset_game()
        episode_experiences = []  # experiences for this episode
        episode_score = 0
        
        # tracking from Player 1's perspective
        player1_moves = []
        player2_moves = []

        for t in range(timesteps):
            if game.is_over():
                break
                
            current_player = game.whose_turn()
            
            # state from current player's perspective
            state = st.get_state_tensor(game)
            legal_moves_mask = mv.get_legal_moves_mask(game)
            
            state_batched = state.unsqueeze(0)
            action_idx = agent.act(state_batched, legal_moves_mask, eps)
            
            game, reward, done = g.step(game, action_idx)
            
            if not done:
                next_state = st.get_state_tensor(game)
            else:
                next_state = torch.zeros_like(state)  # terminal state
            
            experience = {
                'state': state,
                'action': action_idx,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'player': current_player
            }
            
            episode_experiences.append(experience)
            
            if current_player == 1:
                player1_moves.append(experience)
            else:
                player2_moves.append(experience)

        # game finished - final rewards and experiences
        final_reward = 0
        if game.is_over():
            winner = game.get_winner()
            if winner == 1:
                final_reward = 1
                episode_score = 1
            elif winner == 2:
                final_reward = -1
                episode_score = -1
            else:
                final_reward = 0
                episode_score = 0
        
        # add experiences to replay buffer with shaped rewards
        for i, exp in enumerate(episode_experiences):
            # give final reward to last few moves
            shaped_reward = exp['reward']
            
            if exp['done']:
                # terminal reward
                if exp['player'] == 1:
                    shaped_reward = final_reward
                else:
                    shaped_reward = -final_reward  # flip for player 2
            else:
                # small step penalty to encourage faster games
                shaped_reward = -0.01
                
                # bonus for capturing (if original reward was positive)
                if exp['reward'] > 0:
                    if exp['player'] == 1:
                        shaped_reward = 0.1
                    else:
                        shaped_reward = -0.1
            
            # Add to agent's memory
            agent.step(
                exp['state'], 
                exp['action'], 
                shaped_reward, 
                exp['next_state'], 
                exp['done']
            )
        
        scores_window.append(episode_score)  
        scores.append(episode_score)
        eps = max(eps_end, eps_decay * eps)  

        print(f"Episode {episode+1}/{episodes}, Score: {episode_score}, Epsilon: {eps:.3f}, Game Length: {len(episode_experiences)}")

        if (episode + 1) % 100 == 0:
            avg_score = sum(scores_window)/len(scores_window)
            print(f"Average score over last 100 episodes: {avg_score:.3f}")
            
            # model checkpoint
            if (episode + 1) % 500 == 0:
                torch.save(agent.dqn_online.state_dict(), f'checkpoints/checkers_dqn_episode_{episode+1}.pth')

    return scores

