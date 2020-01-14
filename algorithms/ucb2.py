import math
import numpy as np
import random


class UCB2():
    def __init__(self, alpha_param, n_arms):
        self.alpha_param = alpha_param
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.r = [0] * n_arms
        self.__current_arm = 0
        self.__next_update = 0
        self.alpha = [1] * n_arms
        self.beta = [1] * n_arms

    def reset(self):
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.r = [0] * self.n_arms
        self.__current_arm = 0
        self.__next_update = 0
        self.alpha = [1] * self.n_arms
        self.beta = [1] * self.n_arms
        
    def __tau(self, r):
        return int(math.ceil((1 + self.alpha_param) ** r))
    
    def __bonus(self, n, r):
        tau = self.__tau(r)
        bonus = math.sqrt((1. + self.alpha_param) * math.log(math.e * float(n) / tau) / (2 * tau))
        return bonus
  
    def __set_arm(self, arm):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.__current_arm = arm
        self.__next_update += max(1, self.__tau(self.r[arm] + 1) - self.__tau(self.r[arm]))
        self.r[arm] += 1

    def select_arm(self):
        # play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                self.__set_arm(arm)
                return arm
    
        # make sure we aren't still playing the previous arm.
        if self.__next_update > sum(self.counts):
            return self.__current_arm
    
        ucb_values = [0.0 for arm in range(self.n_arms)]
        total_counts = sum(self.counts)
        for arm in range(self.n_arms):
            bonus = self.__bonus(total_counts, self.r[arm])
            ucb_values[arm] = self.values[arm] + bonus
        chosen_arm = random.choice([i for i, val in enumerate(ucb_values) if val == max(ucb_values)])
        self.__set_arm(chosen_arm)
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        n = float(self.counts[chosen_arm])
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
