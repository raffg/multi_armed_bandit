import random
import numpy as np


class EpsilonGreedy():
    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
