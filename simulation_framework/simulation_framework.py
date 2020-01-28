import random
from collections import deque


def run_sim(algorithm, arms, horizon, num_sims=1, terminate=False, confidence=.95, regret=.01, min_trials=1000):
    
    chosen_arms = []
    rewards = []
    cumulative_rewards = []
    sim_nums = []
    trials = []
    alpha = []
    beta = []
    optimal_arm_prob = 0
    potential_value_remaining = 1
    pvr_list = deque([0] * 100)
    previous_idx = 0
    previous_idx_flag = False
    
    for sim in range(num_sims):
        algorithm.reset()
        for t in range(horizon):
            idx = sim * horizon + t
            if previous_idx_flag:
                idx = previous_idx + t
            sim_nums.append(sim)
            trials.append(t)
            
            if 'Thompson' in str(algorithm):
                rhos = algorithm.select_arm().copy()
                if (t > min_trials) and terminate:
                    expected_rewards = [alpha[idx-1][i] / (alpha[idx-1][i] + beta[idx-1][i]) for i in range(len(alpha[idx-1]))]
                    expected_best_arm = expected_rewards.index(max(expected_rewards))
                    theta_max = max(rhos)
                    theta_star = rhos[expected_best_arm]
                    pvr_list.popleft()
                    pvr_list.append((theta_max - theta_star) / theta_star)
                    potential_value_remaining = 0 if sum(pvr_list) == 0 else [i for i in pvr_list if i > 0][-1]
                    if potential_value_remaining < regret:
                        optimal_arm_prob = probability_of_expected_best_arm(algorithm, expected_best_arm)
                        if optimal_arm_prob > confidence:
                            previous_idx_flag = True
                            previous_idx = idx - 1
                            break
                chosen_arm = random.choice([i for i, v in enumerate(rhos) if v == max(rhos)])

            else:
                chosen_arm = algorithm.select_arm()
            chosen_arms.append(chosen_arm)
            reward = arms[chosen_arm].draw()
            rewards.append(reward)
            alpha.append(algorithm.alpha.copy())
            beta.append(algorithm.beta.copy())
            if t == 0:
                cumulative_rewards.append(reward)
            else:
                cumulative_rewards.append(cumulative_rewards[idx - 1] + reward)
            algorithm.update(chosen_arm, reward)
            if t == horizon - 1:
                previous_idx_flag = False
    
#         if 'Thompson' in str(algorithm):
#             if terminate:
#                 if t + 2 <= horizon:
#                     print('The experiment ended after {} trials'.format(t + 1))
#                 else:
#                     print('The experiment ended at the horizon')
#                 print('Optimal arm probability: {}'.format(optimal_arm_prob))
#                 print('Potential value remaining: {}'.format(potential_value_remaining))
    
    return sim_nums, trials, chosen_arms, rewards, cumulative_rewards, alpha, beta

def probability_of_expected_best_arm(algorithm, expected_best_arm):
    count = 0
    count_best_arm = 0
    prob_new = 2
    prob = 0
    while count < 1000:
        if count > 100:
            if abs(prob_new - prob) < .001:
                return prob_new
        count +=1
        prob = prob_new
        rhos = algorithm.select_arm().copy()
        chosen_arm = random.choice([i for i, v in enumerate(rhos) if v == max(rhos)])
        if chosen_arm == expected_best_arm:
            count_best_arm += 1
        prob_new = count_best_arm / count
    return prob_new
