def run_sim(algorithm, arms, horizon, num_sims=1, beta_dist=False):
    
    length = num_sims * horizon
    
    chosen_arms = [0] * length
    rewards = [0] * length
    cumulative_rewards = [0] * length
    sim_nums = [0] * length
    trials = [0] * length
    if beta_dist:
        alpha = [[0] * len(arms)] * length
        beta = [[0] * len(arms)] * length
    
    for sim in range(num_sims):
        algorithm.reset()
        
        for t in range(horizon):
            idx = sim * horizon + t
            sim_nums[idx] = sim
            trials[idx] = t
            
            chosen_arm = algorithm.select_arm()
            chosen_arms[idx] = chosen_arm
            
            reward = arms[chosen_arms[idx]].draw()
            rewards[idx] = reward
            
            if beta_dist:
                alpha[idx] = algorithm.alpha.copy()
                beta[idx] = algorithm.beta.copy()
                
            if t == 0:
                cumulative_rewards[idx] = reward
            else:
                cumulative_rewards[idx] = cumulative_rewards[idx - 1] + reward
                
            algorithm.update(chosen_arm, reward)
    
    if beta_dist:
        return sim_nums, trials, chosen_arms, rewards, cumulative_rewards, alpha, beta
    return sim_nums, trials, chosen_arms, rewards, cumulative_rewards