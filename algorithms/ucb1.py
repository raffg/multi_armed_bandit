import math
import numpy as np
import random


class UCB1():
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.alpha = [1] * n_arms
        self.beta = [1] * n_arms

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.alpha = [1] * self.n_arms
        self.beta = [1] * self.n_arms

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            curiosity_bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + curiosity_bonus
        return random.choice([i for i, val in enumerate(ucb_values) if val == max(ucb_values)])

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
