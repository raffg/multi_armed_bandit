import random
import numpy as np


class ThompsonSampling():
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.s_counts = [0] * n_arms
        self.alpha = 1
        self.beta = 1

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.s_counts = [0] * self.n_arms

    def select_arm(self):
        rho = [random.betavariate(self.alpha + self.s_counts[i],
                                  self.beta + self.counts[i] - self.s_counts[i]) for i in range(len(self.counts))]
        return random.choice([i for i, v in enumerate(rho) if v == max(rho)])

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        if reward == 1:
            self.s_counts[chosen_arm] += 1
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
