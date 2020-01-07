def run_sim(algorithm, arms, horizon, num_sims=1):
    
    length = num_sims * horizon
    
    chosen_arms = [0] * length
    rewards = [0] * length
    cumulative_rewards = [0] * length
    sim_nums = [0] * length
    trials = [0] * length
    
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
            
            if t == 0:
                cumulative_rewards[idx] = reward
            else:
                cumulative_rewards[idx] = cumulative_rewards[idx - 1] + reward
                
            algorithm.update(chosen_arm, reward)
            
    return sim_nums, trials, chosen_arms, rewards, cumulative_rewards
