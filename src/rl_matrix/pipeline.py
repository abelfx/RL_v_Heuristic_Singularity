from pathlib import Path
import pickle

from .agent import QLearningAgent
from .environment import MatrixGame
from .heuristics import heuristic_agent


def train_agent(
    episodes=200000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.99997,
    min_epsilon=0.01,
    log_every=10000,
):
    env = MatrixGame()
    agent = QLearningAgent(
        action_space=env.action_space,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
    )

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_memory = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_memory.append((state, action, reward, next_state, done))
            state = next_state

        episode_memory.sort(key=lambda transition: abs(transition[2]), reverse=True)

        for state_i, action_i, reward_i, next_state_i, done_i in episode_memory:
            agent.learn(state_i, action_i, reward_i, next_state_i, done_i)

        agent.decay_epsilon()

        if (episode + 1) % log_every == 0:
            print(
                f"Episode {episode + 1} completed. "
                f"Exploration rate: {agent.epsilon:.5f}"
            )

    return env, agent


def play_match(env, agent):
    agent.epsilon = 0.0
    state = env.reset()
    done = False

    scores = {"RL_Agent": 0.0, "Heuristic": 0.0}
    banned_actions = []
    turn = 1
    step_number = 1
    current_player = "Heuristic"
    steps = []

    while not done:
        valid_actions = [action for action in env.action_space if action not in banned_actions]
        if not valid_actions:
            break

        if current_player == "RL_Agent":
            action = agent.choose_action(state, valid_actions=valid_actions)
        else:
            action = heuristic_agent(env, valid_actions=valid_actions)

        state, reward, done, info = env.step(action)
        scores[current_player] += reward

        steps.append(
            {
                "step": step_number,
                "turn": turn,
                "player": current_player,
                "action": action,
                "reward": reward,
                "info": info,
                "board": env.board.copy(),
                "scores": scores.copy(),
            }
        )
        step_number += 1

        if "Traced back" in info:
            banned_actions.append(action)
        else:
            banned_actions = []
            current_player = "Heuristic" if current_player == "RL_Agent" else "RL_Agent"
            turn += 1

    return {
        "scores": scores,
        "turns": turn,
        "board": env.board.copy(),
        "steps": steps,
    }


def save_q_table(agent, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file_obj:
        pickle.dump(agent.q_table, file_obj)


def load_agent_from_qtable(path):
    env = MatrixGame()
    agent = QLearningAgent(action_space=env.action_space)
    with Path(path).open("rb") as file_obj:
        q_table = pickle.load(file_obj)
    agent.q_table = q_table
    agent.epsilon = 0.0
    return env, agent
