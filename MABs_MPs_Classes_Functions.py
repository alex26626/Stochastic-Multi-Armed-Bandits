import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Adding an axes using the same arguments")
from scipy.stats import lomax
from collections import Counter
from scipy.stats import gamma
from scipy.stats import beta

def two_regrets(data):
    #plt.figure(figsize = (10,14))
    player1_cols = []
    player2_cols = []
    for i in range(len(data.columns)):
        if data.columns[i].split('_')[0] == 'p1' and data.columns[i].split('_')[2] == 'regrets':
            player1_cols.append(data.columns[i])
        elif data.columns[i].split('_')[0] == 'p2' and data.columns[i].split('_')[2] == 'regrets':
            player2_cols.append(data.columns[i])
    player1_data = data.loc[:, player1_cols].reset_index(drop = True)
    player2_data = data.loc[:, player2_cols].reset_index(drop = True)
    plt.figure(figsize = (8,6))
    for col in player1_cols:
        if int(col.split('_')[1]) >= data.shape[0]:
            label = 'Player1_' + str(0)
        else: 
            label = 'Player1_' + col.split('_')[1]
        plt.plot(data[col], label = label)
    plt.axhline(0, color = 'black')
    plt.title('Player 1')
    plt.grid()
    plt.legend()
    plt.show()
    plt.figure(figsize = (8,6))
    for col in player2_cols:
        if int(col.split('_')[1]) >= data.shape[0]:
            label = 'Player2_' + str(0)
        else: 
            label = 'Player2_' + col.split('_')[1]
        plt.plot(data[col], label = label)
    plt.axhline(0, color = 'black')
    plt.title('Player 2')
    plt.legend()
    plt.grid()
    plt.show()

class player_m:
    def __init__(self, n_arms, player_index, n_players, n_rounds):
        self.regrets = []
        self.cumulative_regret = 0
        self.cumulative_regrets = []
        self.means = np.zeros(shape = n_arms)
        self.n_trials = np.zeros(shape = n_arms)
        self.choices = []
        self.all_rewards = np.zeros(shape = (n_arms, n_players, n_rounds))
        self.all_trials = np.zeros(shape = (n_arms, n_players, n_rounds))
        self.n_arms = n_arms
        self.player_index = player_index
        self.n_players = n_players
            
    def update_n_trials(self, choice, current_round):
        if choice != None:
            self.all_trials[choice, self.player_index, current_round] = 1
            self.n_trials[choice] += 1
               
    def receive_n_trials(self, other_trials):
        self.all_trials[(self.all_trials == 0) & (other_trials == 1)] = 1
        self.n_trials = self.all_trials.sum(axis = 2).sum(axis =  1)
                
    def receive_reward(self, other_reward):
        self.all_rewards[(self.all_rewards == 0) & (other_reward != 0)] = other_reward[(self.all_rewards == 0) & (other_reward != 0)]
        self.means = self.all_rewards.sum(axis = 2).sum(axis = 1)/self.n_trials
                
    def update_mean(self, choice, reward, current_round):
        if choice != None:
            self.all_rewards[choice, self.player_index, current_round] = reward
            increment = (reward - self.means[choice])/self.n_trials[choice]
            self.means[choice] += increment
        
    def update_regret(self, reward, best_reward):
        regret = best_reward - reward
        self.regrets.append(regret)
        self.cumulative_regret += regret
        self.cumulative_regrets.append(self.cumulative_regret)

class ucb_1_player(player_m):
    def __init__(self, n_arms, beta, player_index, n_players, n_rounds):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.ucbs = np.zeros(shape = n_arms)
        self.beta = beta
        
    def compute_ucbs(self, current_round):
        self.ucbs[self.n_trials > 0] = np.sqrt(2*(self.beta**2)*np.log(current_round)/self.n_trials[self.n_trials > 0])
        
    def choice(self):
        choice = np.argmax(self.means + self.ucbs)
        self.choices.append(choice)
        return choice


class ucb_v_player(player_m):
    def __init__(self, n_arms, beta, player_index, n_players, n_rounds):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.beta = beta
        self.ucbs = np.zeros(shape = n_arms)
        self.variances = np.zeros(shape = n_arms)
        self.means_squared = np.zeros(shape = n_arms)
    
    def compute_ucbs(self, current_round):
        ucb_1 = np.sqrt((2*np.log(current_round)*self.variances[self.n_trials > 0])/self.n_trials[self.n_trials > 0])
        ucb_2 = 3*np.log(current_round)*self.beta/self.n_trials[self.n_trials > 0]
        self.ucbs[self.n_trials > 0] = ucb_1 + ucb_2
        
    def choice(self):
        choice = np.argmax(self.means + self.ucbs)
        self.choices.append(choice)
        return choice
    
    def receive_reward(self, other_reward):
        self.all_rewards[(self.all_rewards == 0) & (other_reward != 0)] = other_reward[(self.all_rewards == 0) & (other_reward != 0)]
        self.means = self.all_rewards.sum(axis = 2).sum(axis = 1)/self.n_trials
        self.means_squared = (self.all_rewards**2).sum(axis = 2).sum(axis = 1)/self.n_trials
    
    def update_mean(self, choice, reward, current_round):
        if choice != None:
            self.all_rewards[choice, self.player_index, current_round] = reward
            increment = (reward - self.means[choice])/self.n_trials[choice]
            self.means[choice] += increment
            increment_squared = (reward**2 - self.means_squared[choice])/self.n_trials[choice]
            self.means_squared[choice] += increment_squared
    
    def update_variance(self, choice):
        self.variances[choice] = self.means_squared[choice] - self.means[choice]**2


class ucb_moss_player(player_m):
    def __init__(self, n_arms, player_index, n_players, n_rounds, beta):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.ucbs = np.zeros(shape = n_arms)
        self.n_rounds = n_rounds
        self.n_arms = n_arms
        self.beta = beta
        
    def compute_ucbs(self, current_round):
        log_arg = self.n_rounds/self.n_arms*self.n_trials[self.n_trials > 0]
        log_arg[log_arg < 1] = 1
        self.ucbs[self.n_trials > 0] = np.sqrt((self.beta/self.n_trials[self.n_trials > 0])*np.log(log_arg))
        
    def choice(self):
        choice = np.argmax(self.means + self.ucbs)
        self.choices.append(choice)
        return choice


class ucb_kl_player(player_m):
    def __init__(self, n_arms, player_index, n_players, n_rounds, dist):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.dist = dist
        self.ucbs = np.zeros(shape = n_arms)
        
    def compute_ucbs(self, current_round, l_rate, max_iter):
        if self.dist == 'Bernoulli':
            for i in range(self.n_arms):
                if self.means[i] == 0:
                    self.means[i] = 0.01
                if self.means[i] == 1:
                    self.means[i] = 0.99
                maximizer = kl_bern_max(q0 = self.means[i], p = self.means[i], l_rate = l_rate, max_iter = max_iter, t = current_round, s = self.n_trials[i])
                self.ucbs[i] = maximizer
                
        elif self.dist == 'Poisson':
            for i in range(self.n_arms):
                if self.means[i] == 0:
                    self.means[i] = 0.01
                maximizer = kl_pois_max(l0 = self.means[i], l1 = self.means[i], l_rate = l_rate, max_iter = max_iter, t = current_round, s = self.n_trials[i], upper_bound = 1000)
                self.ucbs[i] = maximizer
        
        elif self.dist == 'Exponential':
            for i in range(self.n_arms):
                if self.means[i] == 0:
                    self.means[i] = 0.01
                maximizer = kl_exp_max(l0 = self.means[i] + 1, l1 = self.means[i], l_rate = l_rate, max_iter = max_iter, t = current_round, s = self.n_trials[i], upper_bound = 1000)
                self.ucbs[i] = maximizer
            
    def choice(self):
        choice = np.argmax(self.ucbs)
        self.choices.append(choice)
        return choice

class bayes_player(player_m):
    def __init__(self, n_arms, player_index, n_players, n_rounds, dist, scale = 1, shape = 1):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.ucbs = np.zeros(shape = n_arms)
        self.dist = dist
        self.a = np.ones(shape = n_arms)
        self.b = np.ones(shape = n_arms)
        self.shape = np.array([shape for i in range(n_arms)])
        if dist == 'Exponential':
            self.scale = [1/scale for i in range(n_arms)]
        else:
            self.scale = [scale for i in range(n_arms)]
        self.all_samples = np.zeros(shape  = (n_arms, 500))
        self.first_scale = 1/scale
        self.first_shape = shape
    
    def choice(self):
        if dist != 'Exponential':
            choice = np.argmax(self.ucbs)
        else:
            choice = np.argmin(self.ucbs)
        self.choices.append(choice)
        return choice
    
    def compute_ucbs(self, current_round):
        if self.dist == 'Bernoulli':
            for i in range(self.n_arms):
                self.ucbs[i] = beta.ppf(a = self.a[i], b = self.b[i], q = 1 - 1/current_round)
        elif self.dist == 'Poisson':
            for i in range(self.n_arms):
                self.ucbs[i] = gamma.ppf(a = self.shape[i], scale = 1/self.scale[i], q =  1 - 1/current_round)
        elif self.dist == 'Exponential':
            for i in range(self.n_arms):
                self.ucbs[i] = gamma.ppf(a = self.shape[i], scale = 1/self.scale[i], q = 1/current_round)
        elif self.dist == 'Dirichlet':
            for i in range(self.n_arms):
                p_generator = np.random.default_rng()
                probs = p_generator.dirichlet(alpha = self.alphas[i])
                self.ucbs[i] = np.random.choice(np.arange(1,11), p = probs)
            
    
    def update_params(self, choice, reward):
        if self.dist == 'Bernoulli':
            if reward == 0:
                self.b[choice] += 1
            else:
                self.a[choice] += 1
            #self.all_samples[choice, :] = np.random.beta(a = self.a[choice], b = self.b[choice], size = 5000)

        elif self.dist == 'Poisson':
            self.shape[choice] += reward
            self.scale[choice] += 1
            #self.scale[choice] = float(self.scale[choice]/(self.scale[choice] + 1))
            #self.all_samples[choice, :] = np.random.negative_binomial(self.shape[choice], 1/(self.scale[choice] + 1), size = 1000)
            #self.all_samples[choice, :] = np.random.gamma(shape = self.shape[choice], scale = 1/self.scale[choice], size = 5000)
        elif self.dist == 'Exponential':
            self.shape[choice] += 1
            self.scale[choice] += reward
            #self.all_samples[choice, :] = lomax.rvs(c = self.shape[choice], scale = self.scale[choice], size = 1000)
            #self.all_samples[choice, :] = 1/np.random.gamma(shape = self.shape[choice], scale = 1/self.scale[choice], size = 5000)
            
        elif self.dist == 'Dirichlet':
            self.alphas[choice][reward-1] += 1
            
    def receive_n_trials(self, other_trials):
        if self.dist == 'Poisson':
            self.diff = self.all_trials[(self.all_trials == 0) & (other_trials == 1)].sum()
        self.all_trials[(self.all_trials == 0) & (other_trials == 1)] = 1
        self.n_trials = self.all_trials.sum(axis = 2).sum(axis =  1)
            
    def receive_reward(self, other_reward):
        self.all_rewards[(self.all_rewards == 0) & (other_reward != 0)] = other_reward[(self.all_rewards == 0) & (other_reward != 0)]
        self.means = self.all_rewards.sum(axis = 2).sum(axis = 1)/self.n_trials
        if self.dist == 'Bernoulli':
            self.a = self.all_rewards.sum(axis = 2).sum(axis = 1)
            self.a[self.a == 0] = 1
            self.b = self.n_trials - self.a
            self.b[self.b <= 0] = 1
        elif self.dist == 'Poisson':
            self.scale = self.n_trials + self.first_scale
            self.shape = self.all_rewards.sum(axis = 2).sum(axis = 1) + self.first_shape
            self.shape[self.shape == 0] = 1
        elif self.dist == 'Exponential':
            self.shape = self.n_trials + self.first_shape
            self.scale = self.all_rewards.sum(axis = 2).sum(axis = 1) + self.first_scale
    

class thom_player(player_m):
    def __init__(self, n_arms, player_index, n_players, n_rounds, dist, ub = 0, lb = 1):
        super().__init__(n_arms, player_index, n_players, n_rounds)
        self.a = np.ones(shape = n_arms)
        self.b = np.ones(shape = n_arms)
        self.dist = dist
        self.lb = lb
        self.ub = ub
        
    def choice(self):
        samples = [np.random.beta(a = self.a[i], b = self.b[i]) for i in range(self.n_arms)]
        choice = np.argmax(samples)
        self.choices.append(choice)
        return choice

    def normalize(self, reward):
        if reward > self.ub:
            self.ub = reward
        elif reward < self.lb:
            self.lb = reward
        reward = (reward - self.lb)/(self.ub - self.lb)
        return reward

    def update_params(self, choice, reward):
        if choice != None:
            if self.dist == 'Bernoulli':
                if reward == 1:
                    self.a[choice] += 1
                else:
                    self.b[choice] += 1     
            else:
                u = np.random.uniform(0,1)
                if u <= reward:
                    self.a[choice] += 1
                else:
                    self.b[choice] += 1

    def receive_reward(self, other_reward):
        self.all_rewards[(self.all_rewards == 0) & (other_reward != 0)] = other_reward[(self.all_rewards == 0) & (other_reward != 0)]
        self.means = self.all_rewards.sum(axis = 2).sum(axis = 1)/self.n_trials
        if self.dist == 'Bernoulli': 
            self.a = self.all_rewards.sum(axis = 2).sum(axis = 1) + 1
            self.b = self.n_trials - self.a + 1
            self.b[self.b <= 0] = 1
        else:
            self.norm_rewards = (self.all_rewards - self.lb)/(self.ub - self.lb)
            u = np.random.uniform(0,1)
            self.a = (u <= self.norm_rewards).sum(axis = 2).sum(axis = 1) + 1
            self.b = (u > self.norm_rewards).sum(axis = 2).sum(axis = 1) + 1
