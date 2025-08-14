import numpy as np
import torch
import agent as ag
import game as g

from game import reset_game, step
from moves import encode_move, decode_move, get_legal_moves_mask
from state import get_state_tensor
from minmax import minimax_move
from bitboard import convert_game_to_bitboards, convert_bitboards_to_tensor
from dqn import CheckersDQN

def test_bitboard_roundtrip(game):
    p1m, p1k, p2m, p2k, turn_flag = convert_game_to_bitboards(game)

    tensor_from_bitboards = convert_bitboards_to_tensor(p1m, p1k, p2m, p2k, turn_flag)
    tensor_direct = get_state_tensor(game)

    assert tensor_from_bitboards.shape == (6, 8, 8)
    assert tensor_direct.shape == (6, 8, 8)

    counts_from_bitboards = [int(tensor_from_bitboards[ch].sum().item()) for ch in range(4)]
    counts_direct = [int(tensor_direct[ch].sum().item()) for ch in range(4)]

    print(f"  Bitboards to tensor counts: {counts_from_bitboards} | Direct counts: {counts_direct}")
    return counts_from_bitboards == counts_direct

def test_environment():
    print("=== TESTING ENVIRONMENT ===")
    game = reset_game()
    test_bitboard_roundtrip(game)
    print("Game reset OK")

    move_count = 0
    while not game.is_over() and move_count < 10:   
        mask = get_legal_moves_mask(game)
        if not any(mask):
            print("No legal moves left!")
            break

        move_idx = minimax_move(game, depth=1)
        decoded = decode_move(move_idx, game)
        print(f"Move {move_count+1}: index {move_idx} -> decoded {decoded}")

        game, reward, done = step(game, move_idx)
        print(f"  Reward: {reward}, Done: {done}")

        tensor = get_state_tensor(game)
        print(f"  State tensor shape: {tuple(tensor.shape)}")

        move_count += 1
        if done:
            print(f"Game over! Winner: {game.get_winner()}")  
            break

def test_dqn():
    print("=== TESTING DQN ===")
    model = CheckersDQN()
    print(f"DQN model structure: {model}")
    game = reset_game()
    state = get_state_tensor(game)

    print(f"Initial state tensor shape: {state.shape}")
    output = model(state)
    print(f"Model output shape: {output.shape}")
    valid_moves_list = game.get_possible_moves()
    print(f"First possible move: {valid_moves_list[0]}")
    encoded_move = encode_move(valid_moves_list[0])
    print(f"Encoded move: {encoded_move}")

    legal_moves_mask = get_legal_moves_mask(game)

    print(f"DQN model fc2 output features: {np.arange(model.fc2.out_features)}")
    print(f"DQN model fc2 output features: {np.arange(model.fc2.out_features)[legal_moves_mask]}")

def test_agent():
    print("=== TESTING AGENT ===")
    agent = ag.CheckersAgent()
    print(f"Agent initialized with DQN model: {agent.dqn_online}")

    game = reset_game()
    state = get_state_tensor(game)
    legal_moves_mask = get_legal_moves_mask(game)

    print(f"agent online fc2 output features: {np.arange(agent.dqn_online.fc2.out_features)[legal_moves_mask]}")

    # Always add batch dimension for act, step, and learn
    state_b = state.unsqueeze(0) if state.dim() == 3 else state
    action_idx = agent.act(state_b, legal_moves_mask, eps=0.1)
    print(f"Selected action idx: {action_idx}")

    next_state, reward, done = g.step(game, action_idx)
    next_state = get_state_tensor(next_state)
    next_state_b = next_state.unsqueeze(0) if next_state.dim() == 3 else next_state
    print(f"Next state shape: {next_state.shape}")
    print(f"Next state batch shape: {next_state_b.shape}")

    agent.step(state_b, action_idx, reward, next_state_b, done)

    # Only call agent.learn() directly if not enough samples in memory
    if len(agent.memory) <= ag.BATCH_SIZE:
        print(f"Memory size: {len(agent.memory)}")
        print("Learning from experience...")
        action_b = torch.tensor([[action_idx]], dtype=torch.long)
        reward_b = torch.tensor([[reward]], dtype=torch.float32)
        done_b = torch.tensor([[done]], dtype=torch.float32)
        agent.learn((state_b, action_b, reward_b, next_state_b, done_b), gamma=0.99)

    print("Agent step completed.")

def test_train():
    print("=== TESTING TRAIN AGENT ===")
    from train import train_agent
    import matplotlib.pyplot as plt

    # Run a single episode for testing
    scores = train_agent(episodes=1000, timesteps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.99)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(scores)), scores, label='Episode Scores')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Checkers Agent Training Scores')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    test_train()