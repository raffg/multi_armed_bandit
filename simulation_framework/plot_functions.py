import pandas as pd
import numpy as np
import os
import shutil
import math
import imageio
from IPython.display import Image
import re
from scipy import stats
import matplotlib.pyplot as plt


def format_results(results, arms):
    df = pd.DataFrame(list(zip(*results)), columns=['sim_num', 'trial', 'chosen_arm', 'reward', 'cumulative_reward', 'alphas', 'betas'])
    alphas = pd.DataFrame(df['alphas'].values.tolist(), columns=['alpha_{}'.format(arm) for arm in range(len(df['alphas'].iloc[0]))])
    betas = pd.DataFrame(df['betas'].values.tolist(), columns=['beta_{}'.format(arm) for arm in range(len(df['betas'].iloc[0]))])
    for arm in range(len(arms)):
        df['arm_{}'.format(arm)] = (df['chosen_arm'] == arm).astype(int)
        df['arm_{}_cumulative'.format(arm)] = df.groupby((df.sim_num !=
                                                          df.sim_num.shift())
                                                          .cumsum())['arm_{}'.format(arm)].cumsum()
        df['alpha_{}'.format(arm)] = alphas['alpha_{}'.format(arm)]
        df['beta_{}'.format(arm)] = betas['beta_{}'.format(arm)]
    return df


def summarize_results(results_df, arms, hyperparameter_list):
    alpha_beta = list(sum([('alpha_{}'.format(arm), 'beta_{}'.format(arm)) for arm in range(len(arms))], ()))
    arm_list = list(sum([('arm_{}'.format(arm), 'arm_{}_cumulative'.format(arm)) for arm in range(len(arms))], ()))
    agg_list = ['reward', 'cumulative_reward']
    agg_list.extend(arm_list)
    agg_list.extend(alpha_beta)
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


def plot_arms(df_ave, probabilities, algorithm_name):
        probabilities = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
        plt.figure(figsize=(10, 7))
        for arm in range(len(probabilities)):
            plt.plot(df_ave['trial'],
                     df_ave['arm_{}'.format(probabilities[arm])],
                     label='arm_{}'.format(arm))
        plt.legend(title='Sorted Arms')
        plt.xlabel('Number of Trials')
        plt.ylabel('Probability of Selecting Each Arm')
        plt.title('Arm Selection Rate of the {} Algorithm'.format(algorithm_name))
        plt.show()
        
        
def plot_expected_reward(arm_results, arms, algorithm, xlim=None, ylim=None):
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
    df['dof'] = df.reset_index().index
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
                                                     mean(), confidence_level)
    return arm_results


def plot_beta_dist(df_ave, probabilities, idx, algorithm, legend_loc='upper left'):
    plt.figure(figsize=(10, 7))
    for arm in range(len(probabilities)):
        x = np.arange (0, 1.001, 0.001)
        y = stats.beta.pdf(x, df_ave['alpha_{}'.format(arm)].iloc[idx], df_ave['beta_{}'.format(arm)].iloc[idx])
        plt.plot(x, y, label='arm_{}'.format(arm))
    plt.legend(loc=legend_loc)
    plt.xlabel('Expected Reward')
    plt.ylabel('Probability Density')
    plt.yticks([])
    plt.title('Probability Distribution of Each Arm, {}'.format(algorithm))
    plt.savefig('images_for_gif/{}.png'.format(idx), bbox_inches='tight')
    plt.close()
    
    
def create_gif(algorithm, folder='images_for_gif'):
    filenames = os.listdir(folder)
    filenames = sorted([int(re.sub('[^0-9]', '', filename)) if re.search('[0-9]', filename) else math.inf for filename in filenames])
    if math.inf in filenames:
        filenames.remove(math.inf)
    images = []
    for filename in filenames:
        filename = '{}/{}.png'.format(folder, filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('gifs/{}.gif'.format(algorithm), images)
    delete_folder_contents(folder)
#     with open('gifs/{}.gif'.format(algorithm),'rb') as f:
#         display(Image(data=f.read(), format='png'))


def plot_beta_grid(df_ave, probabilities, algorithm, horizon, grid_length):
    indices = geomspace_indices(horizon, grid_length**2)

    fig, ax = plt.subplots(grid_length, grid_length, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.1)
    fig.suptitle('Probability Distribution of Each Arm, {}'.format(algorithm), fontsize=20, y=1.02)
    fig.subplots_adjust(top=0.88)

    ax = ax.ravel()

    for axis, idx in enumerate(indices):
        for arm in range(len(probabilities)):
            x = np.arange (0, 1.001, 0.001)
            y = stats.beta.pdf(x, df_ave['alpha_{}'.format(arm)].iloc[idx], df_ave['beta_{}'.format(arm)].iloc[idx])
            ax[axis].plot(x, y, label='arm_{}'.format(arm))
        ax[axis].set_title('Trial #{}'.format(idx))

    handles, labels = ax[axis].get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')
    fig.text(0.5, -.01, 'Expected Reward', ha='center', fontsize=20)
    fig.text(-.01, 0.5, 'Probability Density', va='center', rotation='vertical', fontsize=20)
    fig.tight_layout()
    plt.show()

    
def geomspace_indices(horizon, number):
    indices = [0]
    indices.extend(np.geomspace(1, horizon-1, number-1))
    indices = [int(round(idx)) for idx in indices]
    for idx in range(len(indices)):
        if idx == 0:
            continue
        if indices[idx] <= indices[idx-1]:
            indices[idx] = indices[idx-1] + 1
    return indices


def delete_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
