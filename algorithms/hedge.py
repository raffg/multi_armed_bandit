import random
import math


class Hedge():
    def __init__(self, temperature, n_arms):
        self.temperature = temperature
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select_arm(self):
        try:
            total = sum([math.exp(value / self.temperature) for value in self.values])
        except OverflowError:
            total = float(1.0)
        try:
            probs = [math.exp(value / self.temperature) / total for value in self.values]
        except OverflowError:
            probs = [float(1.0) for value in self.values]
        threshold = random.random()
        cum_prob = 0.0
        for idx in range(len(probs)):
            prob = probs[idx]
            cum_prob += prob
            if cum_prob > threshold:
                return idx
        return len(probs) - 1

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += reward
