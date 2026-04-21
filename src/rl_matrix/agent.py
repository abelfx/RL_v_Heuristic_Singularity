import random


class QLearningAgent:
    def __init__(
        self,
        action_space,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99997,
        min_epsilon=0.01,
        optimistic_q=50.0,
    ):
        self.action_space = action_space
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimistic_q = optimistic_q

    def get_q(self, state, action):
        return self.q_table.get((state, action), self.optimistic_q)

    def choose_action(self, state, valid_actions=None):
        if valid_actions is None or len(valid_actions) == 0:
            valid_actions = self.action_space

        if random.uniform(0.0, 1.0) < self.epsilon:
            return random.choice(valid_actions)

        q_values = [self.get_q(state, action) for action in valid_actions]
        max_q = max(q_values)
        best_actions = [
            action for action, value in zip(valid_actions, q_values) if value == max_q
        ]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_q(state, action)
        max_future_q = 0.0 if done else max(
            self.get_q(next_state, action_option) for action_option in self.action_space
        )

        td_error = (reward + self.gamma * max_future_q) - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        return abs(td_error)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
