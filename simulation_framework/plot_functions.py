import pandas as pd
import numpy as np
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


def plot_accuracy(df_ave, probabilities):
    plt.figure(figsize=(10, 7))
    for epsilon in df_ave['epsilon'].unique():
        plt.plot(df_ave[df_ave['epsilon'] == epsilon]['trial'],
                 df_ave[df_ave['epsilon'] == epsilon]['arm_{}'.format(np.argmax(probabilities))],
                 label=epsilon)
    plt.legend(title='Epsilon')
    plt.xlabel('Number of Trials')
    plt.ylabel('Probability of Selecting Best Arm')
    plt.title('Accuracy of the Epsilon Greedy Algorithm')
    plt.show()


def plot_performance(df_ave):
    plt.figure(figsize=(10, 7))
    for epsilon in df_ave['epsilon'].unique():
        plt.plot(df_ave[df_ave['epsilon'] == epsilon]['trial'],
                 df_ave[df_ave['epsilon'] == epsilon]['reward'],
                 label=epsilon)
    plt.legend(title='Epsilon')
    plt.xlabel('Number of Trials')
    plt.ylabel('Average Reward')
    plt.title('Performance of the Epsilon Greedy Algorithm')
    plt.show()


def plot_cumulative_reward(df_ave):
    plt.figure(figsize=(10, 7))
    for epsilon in df_ave['epsilon'].unique():
        plt.plot(df_ave[df_ave['epsilon'] == epsilon]['trial'],
                 df_ave[df_ave['epsilon'] == epsilon]['cumulative_reward'],
                 label=epsilon)
    plt.legend(title='Epsilon')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward of the Epsilon Greedy Algorithm')
    plt.show()
