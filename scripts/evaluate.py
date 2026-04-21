import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_matrix.pipeline import load_agent_from_qtable, play_match


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MatrixGame Q-learning agent")
    parser.add_argument("--model", type=str, default="models/matrix_q_table.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    env, agent = load_agent_from_qtable(args.model)
    result = play_match(env, agent)

    scores = result["scores"]
    print("=" * 40)
    print("MATCH COMPLETE")
    print("=" * 40)
    print("Final Board:\n", result["board"])
    print(f"RL Agent Final Score:  {scores['RL_Agent']:.2f}")
    print(f"Heuristic Final Score: {scores['Heuristic']:.2f}")

    if scores["RL_Agent"] > scores["Heuristic"]:
        print("WINNER: Reinforcement Learning Agent!")
    elif scores["Heuristic"] > scores["RL_Agent"]:
        print("WINNER: Heuristic Player!")
    else:
        print("RESULT: It's a Tie!")


if __name__ == "__main__":
    main()
