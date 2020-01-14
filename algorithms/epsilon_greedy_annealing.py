import random
import math
import numpy as np


class EpsilonGreedyAnnealing():
    def __init__(self, n_arms, annealing_factor=.0000001):
        self.annealing_factor = annealing_factor
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
        t = sum(self.counts) + 1
        epsilon = 1 / math.log(t + self.annealing_factor)
        if random.random() > epsilon:
            return random.choice([i for i, val in enumerate(self.values) if val == max(self.values)])
        else:
            return random.randrange(self.n_arms)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
