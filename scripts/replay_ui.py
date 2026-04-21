import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_matrix.pipeline import load_agent_from_qtable, play_match


def parse_args():
    parser = argparse.ArgumentParser(description="Replay MatrixGame match with a visual UI")
    parser.add_argument("--model", type=str, default="models/matrix_q_table.pkl")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between frames",
    )
    return parser.parse_args()


def board_to_display(board):
    numeric_board = np.nan_to_num(board, nan=0.0)
    mask = np.isnan(board)
    annotations = np.full(board.shape, "", dtype=object)
    annotations[~mask] = board[~mask].astype(int).astype(str)
    return numeric_board, mask, annotations


def winner_text(scores):
    if scores["RL_Agent"] > scores["Heuristic"]:
        return "Winner: Reinforcement Learning Agent"
    if scores["Heuristic"] > scores["RL_Agent"]:
        return "Winner: Heuristic Player"
    return "Result: Tie"


def draw_frame(fig, board_ax, info_ax, board, title, details):
    board_ax.clear()
    info_ax.clear()

    board_data, board_mask, board_annot = board_to_display(board)
    sns.heatmap(
        board_data,
        mask=board_mask,
        annot=board_annot,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-2,
        vmax=2,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        square=True,
        ax=board_ax,
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )

    board_ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    board_ax.set_xticks([])
    board_ax.set_yticks([])

    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.98,
        "\n".join(details),
        ha="left",
        va="top",
        fontsize=12,
    )

    fig.canvas.draw_idle()


def main():
    args = parse_args()

    env, agent = load_agent_from_qtable(args.model)
    result = play_match(env, agent)

    steps = result.get("steps", [])
    scores = result["scores"]
    initial_board = result.get("initial_board", env.board.copy())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 6))
    grid = GridSpec(1, 2, width_ratios=[1.35, 1], figure=fig)
    board_ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])

    draw_frame(
        fig,
        board_ax,
        info_ax,
        initial_board,
        "Matrix Match Replay",
        [
            "Status: Initial board",
            "Players: RL Agent vs Heuristic",
            f"Total Steps Recorded: {len(steps)}",
            "",
            "Legend:",
            "- Empty cells are blank",
            "- Cell values are from {-2, 1, 2}",
        ],
    )
    plt.tight_layout()
    plt.pause(max(0.01, args.interval))

    for step in steps:
        lines = [
            f"Step: {step['step']} / {len(steps)}",
            f"Turn: {step['turn']}",
            f"Player: {step['player']}",
            f"Action: {step['action']}",
            f"Reward: {step['reward']:.2f}",
            "",
            "Scoreboard",
            f"RL Agent: {step['scores']['RL_Agent']:.2f}",
            f"Heuristic: {step['scores']['Heuristic']:.2f}",
        ]
        if step["info"]:
            lines.extend(["", f"Info: {step['info']}"])

        draw_frame(
            fig,
            board_ax,
            info_ax,
            step["board"],
            "Matrix Match Replay",
            lines,
        )
        plt.pause(max(0.01, args.interval))

    final_lines = [
        "Status: Match complete",
        f"Final RL Agent Score: {scores['RL_Agent']:.2f}",
        f"Final Heuristic Score: {scores['Heuristic']:.2f}",
        "",
        winner_text(scores),
    ]
    draw_frame(
        fig,
        board_ax,
        info_ax,
        result["board"],
        "Final Board",
        final_lines,
    )

    print("Replay finished. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
