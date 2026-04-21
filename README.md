# Matrix Singularity RL

Reinforcement learning project where a Q-learning agent learns to fill a matrix so the final matrix tends toward singularity (determinant close to 0), while avoiding invalid moves that create row-sum-zero penalties.

This repository now contains both:
- the original exploratory notebook (`RL_model.ipynb`), and
- a cleaner script/package layout for repeatable training and evaluation.

---

## Problem Setup

The environment is a 4x4 matrix game:
- Matrix starts with `2` random pre-filled cells.
- Allowed actions are values from `[-2, 1, 2]`.
- Each move fills the **first available empty cell**.
- If a fully populated row sums to `0`, the move is rejected and penalized.

### Reward Design

- **Penalty:** `-50` when a move causes a full row with sum `0` (state is reverted).
- **Terminal reward:**
  - `+100` if final determinant is effectively `0` (singular matrix).
  - otherwise shaped reward `100 / (|det| + 1)`, encouraging smaller determinant magnitude.

This combines sparse success signal (singularity) with dense shaping for smoother learning.

---

## Learning Algorithm

Agent uses **tabular Q-learning** with:
- Bellman update,
- epsilon-greedy policy,
- optimistic Q-value initialization,
- per-episode transition replay sorted by absolute reward (prioritized-like pass).

State representation:
- Flattened board values as tuple,
- `NaN` converted to `0.0` for hashable lookup keys in Q-table.

---

## Project Structure

```
RL/
├── README.md
├── requirements.txt
├── RL_model.ipynb
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── src/
    └── rl_matrix/
        ├── __init__.py
        ├── environment.py
        ├── agent.py
        ├── heuristics.py
        └── pipeline.py
```

---

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train

```bash
python scripts/train.py --episodes 200000 --out models/matrix_q_table.pkl
```

### 3) Evaluate

```bash
python scripts/evaluate.py --model models/matrix_q_table.pkl
```

### 4) Optional dev tools

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

---

## Script Arguments

### `scripts/train.py`

- `--episodes` (default: `200000`)
- `--alpha` (default: `0.1`)
- `--gamma` (default: `0.9`)
- `--epsilon` (default: `1.0`)
- `--epsilon-decay` (default: `0.99997`)
- `--min-epsilon` (default: `0.01`)
- `--log-every` (default: `10000`)
- `--out` (default: `models/matrix_q_table.pkl`)

### `scripts/evaluate.py`

- `--model` path to pickled Q-table.

---

## Why this structure is better

- Keeps notebook experimentation separate from production-like code.
- Makes training/evaluation reproducible from CLI.
- Enables easier testing and extension (new environments/agents).
- Keeps model artifacts in a dedicated folder.

---

## Notes

- Current environment fills the first available empty cell by design.

## Pretrained Model

You can download a pretrained Q-table here:

- [Google Drive model file](https://drive.google.com/file/d/19vLQvng8m5azFebSLcD8YO0JVklyoyzc/view?usp=sharing)

After downloading, place it at `models/matrix_q_table.pkl` and run:

```bash
python scripts/evaluate.py --model models/matrix_q_table.pkl
```
