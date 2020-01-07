import random
import numpy as np


class EpsilonGreedyAnnealing():
    def __init__(self, n_arms, annealing_factor=.0000001):
        self.annealing_factor = annealing_factor
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self):
        t = sum(self.counts) + 1
        epsilon = 1 / math.log(t + self.annealing_factor)
        if random.random() > epsilon:
            return np.argmax(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
