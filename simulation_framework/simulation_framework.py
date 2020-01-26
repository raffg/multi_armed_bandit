import random


def run_sim(algorithm, arms, horizon, num_sims=1, stopping=False, confidence=.95, regret=.01):
    
    length = num_sims * horizon
    
    chosen_arms = [0] * length
    rewards = [0] * length
    cumulative_rewards = [0] * length
    sim_nums = [0] * length
    trials = [0] * length
    alpha = [[0] * len(arms)] * length
    beta = [[0] * len(arms)] * length
    optimal_arm_prob = 0
    potential_value_remaining = 1
    
    for sim in range(num_sims):
        algorithm.reset()
        
        for t in range(horizon):
            idx = sim * horizon + t
            sim_nums[idx] = sim
            trials[idx] = t
            
            if 'Thompson' in str(algorithm):
                rhos = algorithm.select_arm().copy()
                chosen_arm = random.choice([i for i, v in enumerate(rhos) if v == max(rhos)])
                if stopping:
                    if t > 1000:
                        if (optimal_arm_prob > confidence) and (potential_value_remaining < regret):
                            break

                    expected_rewards = [algorithm.alpha[idx] / (algorithm.alpha[idx] + algorithm.beta[idx]) for idx in range(len(algorithm.alpha))]
                    expected_best_arm = expected_rewards.index(max(expected_rewards))
                    if expected_best_arm != rhos.index(max(rhos)):
                        theta_star = rhos[expected_best_arm]
                        theta_max = max(rhos)
                        potential_value_remaining = (theta_max - theta_star) / theta_star
                    if t % 100 == 0:
                        optimal_arm_prob = probability_of_expected_best_arm(algorithm, expected_best_arm)

            else:
                chosen_arm = algorithm.select_arm()
            chosen_arms[idx] = chosen_arm
            
            reward = arms[chosen_arms[idx]].draw()
            rewards[idx] = reward
            
            alpha[idx] = algorithm.alpha.copy()
            beta[idx] = algorithm.beta.copy()
                
            if t == 0:
                cumulative_rewards[idx] = reward
            else:
                cumulative_rewards[idx] = cumulative_rewards[idx - 1] + reward
                
            algorithm.update(chosen_arm, reward)
    
        if 'Thompson' in str(algorithm):
            if stopping:
#                 print('The experiment ended after {} trials'.format(t + 1))
#                 print('Optimal arm probability: {}'.format(optimal_arm_prob))
#                 print('Potential value remaining: {}'.format(potential_value_remaining))
                skipped_iters = [idx for idx in range(1, length - 1) if ((trials[idx] == 0) and (trials[idx + 1] == 0))]
                try:
                    if trials[length] == 0:
                        skipped_iters.append(length)
                except IndexError:
                    continue
                for idx in sorted(skipped_iters, reverse=True):
                    del sim_nums[idx]
                    del trials[idx]
                    del chosen_arms[idx]
                    del rewards[idx]
                    del cumulative_rewards[idx]
                    del alpha[idx]
                    del beta[idx]
    
    return sim_nums, trials, chosen_arms, rewards, cumulative_rewards, alpha, beta

def probability_of_expected_best_arm(algorithm, expected_best_arm):
    count = 0
    count_best_arm = 0
    prob_new = 2
    prob = 0
    while count < 1000:
        if count > 100:
            if abs(prob_new - prob) < .001:
                break
        count +=1
        prob = prob_new
        rhos = algorithm.select_arm().copy()
        chosen_arm = random.choice([i for i, v in enumerate(rhos) if v == max(rhos)])
        if chosen_arm == expected_best_arm:
            count_best_arm += 1
        prob_new = count_best_arm / count
    return prob_new