import play
import agent as ag
import minimax as mm

def main():
    """Main function to run the Checkers game simulation between DQN and Minimax agents."""
    print("=== Starting Checkers Game Simulation ===")
    MODEL_PATH = "./checkpoints/checkers_dqn_episode_10000_10000.pth"
    NUM_GAMES = 100
    MINIMAX_DEPTH = 5

    play.evaluate_agents(MODEL_PATH, num_games=NUM_GAMES, minimax_depth=MINIMAX_DEPTH)


if __name__ == "__main__":
    main()
    print("=== Simulation Complete ===")