import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def format_results(results, arms):
    df = pd.DataFrame(list(zip(*results)), columns=['sim_num', 'trial', 'chosen_arm', 'reward', 'cumulative_reward'])
    for arm in range(len(arms)):
        df['arm_{}'.format(arm)] = (df['chosen_arm'] == arm).astype(int)
        df['arm_{}_cumulative'.format(arm)] = df.groupby((df.sim_num !=
                                                          df.sim_num.shift())
                                                         .cumsum())['arm_{}'.format(arm)].cumsum()
    return df


def summarize_results(results_df, arms, hyperparameter_list):
    arm_list = list(sum([('arm_{}'.format(arm), 'arm_{}_cumulative'.format(arm)) for arm in range(len(arms))], ()))
    agg_list = ['reward', 'cumulative_reward']
    agg_list.extend(arm_list)
    hyperparameter_list.append('trial')
    df_ave = results_df.groupby(hyperparameter_list)[agg_list].mean().reset_index()
    return df_ave


def plot_accuracy(df_ave, probabilities, algorithm_name, hyperparameter=None):
    if hyperparameter:
        plt.figure(figsize=(10, 7))
        for parameter in df_ave[hyperparameter].unique():
            plt.plot(df_ave[df_ave[hyperparameter] == parameter]['trial'],
                     df_ave[df_ave[hyperparameter] == parameter]['arm_{}'.format(np.argmax(probabilities))],
                     label=parameter)
        plt.legend(title=hyperparameter)
        plt.xlabel('Number of Trials')
        plt.ylabel('Probability of Selecting Best Arm')
        plt.title('Accuracy of the {} Algorithm'.format(algorithm_name))
        plt.show()
    else:
        plt.figure(figsize=(10, 7))
        plt.plot(df_ave['trial'],
                 df_ave['arm_{}'.format(np.argmax(probabilities))])
        plt.xlabel('Number of Trials')
        plt.ylabel('Probability of Selecting Best Arm')
        plt.title('Accuracy of the {} Algorithm'.format(algorithm_name))
        plt.show()


def plot_performance(df_ave, algorithm_name, hyperparameter=None):
    if hyperparameter:
        plt.figure(figsize=(10, 7))
        for parameter in df_ave[hyperparameter].unique():
            plt.plot(df_ave[df_ave[hyperparameter] == parameter]['trial'],
                     df_ave[df_ave[hyperparameter] == parameter]['reward'],
                     label=parameter)
        plt.legend(title=hyperparameter)
        plt.xlabel('Number of Trials')
        plt.ylabel('Average Reward')
        plt.title('Performance of the {} Algorithm'.format(algorithm_name))
        plt.show()
    else:
        plt.figure(figsize=(10, 7))
        plt.plot(df_ave['trial'],
                 df_ave['reward'])
        plt.xlabel('Number of Trials')
        plt.ylabel('Average Reward')
        plt.title('Performance of the {} Algorithm'.format(algorithm_name))
        plt.show()


def plot_cumulative_reward(df_ave, algorithm_name, hyperparameter=None):
    if hyperparameter:
        plt.figure(figsize=(10, 7))
        for parameter in df_ave[hyperparameter].unique():
            plt.plot(df_ave[df_ave[hyperparameter] == parameter]['trial'],
                     df_ave[df_ave[hyperparameter] == parameter]['cumulative_reward'],
                     label=parameter)
        plt.legend(title=hyperparameter)
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward of the {} Algorithm'.format(algorithm_name))
        plt.show()
    else:
        plt.figure(figsize=(10, 7))
        plt.plot(df_ave['trial'],
                 df_ave['cumulative_reward'])
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward of the {} Algorithm'.format(algorithm_name))
        plt.show()
        
        
def plot_expected_reward(df, arms, arm_results, algorithm, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    if xlim:
        plt.xlim(left=0, right=xlim)
    if ylim:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    for arm in range(len(arms)):
        df = arm_results[arm]
        x = df.index
        y = df['mean']
        error = df['confidence_interval']
        plt.plot(x, y, linewidth=1, label='arm_{}'.format(arm))
        plt.fill_between(x, y - error, y + error, alpha=.5)
    plt.legend()
    plt.xlabel('Number of Trials')
    plt.ylabel('Expected Reward')
    plt.title('Expected Rewards of the Each Arm, {}'.format(algorithm))
    plt.show()


def lookup_t(dof, alpha=.025):
    return stats.t.ppf(1 - alpha, dof)


def build_confidence_interval(df, confidence_level=.95):
    confidence_level = confidence_level
    alpha = (1 - confidence_level) / 2
    df['dof'] = df.index
    df['t_stat'] = df['dof'].apply(lookup_t, args=(alpha,))
    df['std'] = df['reward'].expanding(2).std()
    df['mean'] = df['reward'].expanding(2).mean()
    df['confidence_interval'] = df['t_stat'] * df['std'] / np.sqrt(df['dof'] + 1)
    df['lower_bound'] = df['mean'] - df['confidence_interval']
    df['upper_bound'] = df['mean'] + df['confidence_interval']
    return df


def create_arm_results(df, arms, confidence_level):
    arm_results = {}
    for arm in range(len(arms)):
        arm_results[arm] = build_confidence_interval(df[df['arm_{}'.format(arm)] == 1]. \
                                                     groupby('trial')[['reward', 'cumulative_reward']]. \
                                                     mean().reset_index(), confidence_level)
    return arm_results
