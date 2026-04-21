import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_matrix.pipeline import save_q_table, train_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-learning agent for MatrixGame")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99997)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--out", type=str, default="models/matrix_q_table.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Training started for {args.episodes} episodes...")
    _, agent = train_agent(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        log_every=args.log_every,
    )
    save_q_table(agent, args.out)
    print(f"Training complete. Learned {len(agent.q_table)} state-action pairs.")
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
