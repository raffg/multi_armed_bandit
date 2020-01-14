import random
import math


class EXP3():
    def __init__(self, gamma, n_arms):
        self.gamma = gamma
        self.n_arms = n_arms
        self.weights = [1.0] * n_arms
        self.alpha = [1] * n_arms
        self.beta = [1] * n_arms

    def reset(self):
        self.weights = [1.0] * self.n_arms
        self.alpha = [1] * self.n_arms
        self.beta = [1] * self.n_arms

    def select_arm(self):
        total_weight = sum(self.weights)
        probs = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight) + (self.gamma / float(self.n_arms))
        threshold = random.random()
        cum_prob = 0.0
        for idx in range(len(probs)):
            prob = probs[idx]
            cum_prob += prob
            if cum_prob > threshold:
                return idx
        return len(probs) - 1

    def update(self, chosen_arm, reward):
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        total_weight = sum(self.weights)
        x = reward / ((1 - self.gamma) * (self.weights[chosen_arm] / total_weight) + (self.gamma / float(self.n_arms)))

        growth_factor = math.exp((self.gamma / self.n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor
