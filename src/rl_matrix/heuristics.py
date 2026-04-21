import random


def heuristic_agent(env, valid_actions=None):
    if valid_actions is None:
        valid_actions = env.action_space
    return random.choice(valid_actions)
