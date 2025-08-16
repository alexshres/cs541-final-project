import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import namedtuple, deque

import agent as ag
import game as g
import moves as mv
import state as st

EPISODES = 10000
TIMESTEPS = 3000
EPSILON = 1.0 
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01


def train_agent(episodes=EPISODES, 
                timesteps=TIMESTEPS,
                eps_start=EPSILON,
                eps_end=EPSILON_MIN,
                eps_decay=EPSILON_DECAY):
    """Train the Checkers agent using self-play."""

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # metrics
    losses = []
    mean_qs = []
    game_lengths = []

    agent = ag.CheckersAgent()
    
    for episode in range(episodes):
        game = g.reset_game()
        episode_score = 0
        
        for t in range(timesteps):
            if game.is_over():
                break
                
            state = st.get_state_tensor(game)
            legal_moves_mask = mv.get_legal_moves_mask(game)
            
            state_batched = state.unsqueeze(0)
            action_idx = agent.act(state_batched, legal_moves_mask, eps)
            
            game, reward, done = g.step(game, action_idx)
            next_state = st.get_state_tensor(game) if not done else torch.zeros_like(state)

            shaped_reward = reward # if done else -0.1
            
            # add to agent's memory
            loss, mean_q = agent.step(
                state,
                action_idx,
                shaped_reward, 
                next_state,
                done
            )

            if loss is not None:
                losses.append(loss)
            if mean_q is not None:
                mean_qs.append(mean_q)

            episode_score += shaped_reward
        
        game_lengths.append(t + 1)
        scores_window.append(episode_score)  
        scores.append(episode_score)
        eps = max(eps_end, eps_decay * eps)  

        print(f"Episode {episode+1}/{episodes}, Score: {episode_score}, Epsilon: {eps:.3f}, Game length is {t+1}")

        if (episode + 1) % 100 == 0:
            avg_score = sum(scores_window)/len(scores_window)
            print(f"Average score over last 100 episodes: {avg_score:.3f}")
            
            # model checkpoint
            if (episode + 1) % 5000 == 0:
                torch.save(agent.dqn_online.state_dict(), f'checkpoints/checkers_dqn_episode_{episode+1}_{EPISODES}.pth')

    return scores, losses, mean_qs, game_lengths

    # Plot the results
def plot_metrics(scores, losses, mean_qs, game_lengths):
    """Plots the training metrics"""
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))
    fig.suptitle('Training Metrics', fontsize=16)

    def plot_moving_average(ax, x_data, y_data, window_size, title, xlabel, ylabel):
        if not y_data: return
        series = pd.Series(y_data)
        moving_avg = series.rolling(window=window_size).mean()
        
        ax.plot(x_data, y_data, label='Raw Data', alpha=0.3)
        ax.plot(x_data, moving_avg, label=f'Moving Average (window {window_size})', color='red')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel) 
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    x_losses = np.arange(len(losses))
    x_scores = np.arange(len(scores))
    x_qs = np.arange(len(mean_qs))
    x_lengths = np.arange(len(game_lengths))

    plot_moving_average(axs[0], x_losses, losses, 100, 'Agent Loss Over Time', 'Training Steps', 'MSE Loss')
    
    plot_moving_average(axs[1], x_scores, scores, 100, 'Average Score per Episode', 'Episodes', 'Score')
    
    plot_moving_average(axs[2], x_qs, mean_qs, 100, 'Average Q-Value Over Time', 'Training Steps', 'Avg. Q-Value')

    plot_moving_average(axs[3], x_lengths, game_lengths, 100, 'Game Length per Episode', 'Episodes', 'Number of Moves')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # type: ignore
    plt.savefig(f'./training_metrics_{EPISODES}_{ag.UPDATE_EVERY}.png')
    # plt.show()


if __name__ == "__main__":
    scores, losses, mean_qs, game_lengths = train_agent()
    plot_metrics(scores, losses, mean_qs, game_lengths)
    print("Training complete. Metrics plotted.")