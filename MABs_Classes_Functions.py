class player:
    def __init__(self, n_arms):
        self.regrets = []
        self.cumulative_regret = 0
        self.cumulative_regrets = []
        self.means = np.zeros(shape = n_arms)
        self.n_trials = np.zeros(shape = n_arms)
        self.choices = []
        self.observed_rewards = dict()
        for i in range(n_arms):
            self.observed_rewards[i] = []
        self.n_arms = n_arms
            
    def update_n_trials(self, choice):
        self.n_trials[choice] += 1
        
    def update_mean(self, choice, reward):
        self.observed_rewards[choice].append(reward**2)
        increment = (reward - self.means[choice])/self.n_trials[choice]
        self.means[choice] += increment
        
    def update_regret(self, reward, best_reward):
        regret = best_reward - reward
        self.regrets.append(regret)
        self.cumulative_regret += regret
        self.cumulative_regrets.append(self.cumulative_regret)

def cum_regret_plot(x, y, labels, big = False, title = None):
    if big:
        plt.figure(figsize = (10,7))
    for i in range(len(x)):
        plt.plot(x[i], y[i], label = labels[i])
    plt.axhline(0, color = 'black')
    plt.grid()
    plt.legend()
    if title is None:
        plt.title('Cumulative Regret')
    else:
        plt.title(title)
    plt.show()


class ucb_1_player(player):
    def __init__(self, n_arms, beta):
        super().__init__(n_arms)
        self.ucbs = np.zeros(shape = n_arms)
        self.beta = beta
        
    def compute_ucbs(self, current_round):
        self.ucbs[self.n_trials > 0] = np.sqrt(2*(self.beta**2)*np.log(current_round)/self.n_trials[self.n_trials > 0])
        
    def choice(self):
        choice = np.argmax(self.means + self.ucbs)
        self.choices.append(choice)
        return choice

class ucb_v_player(player):
    def __init__(self, n_arms, beta):
        super().__init__(n_arms)
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
    
    def update_mean(self, choice, reward):
        increment = (reward - self.means[choice])/self.n_trials[choice]
        self.means[choice] += increment
        increment_squared = (reward**2 - self.means_squared[choice])/self.n_trials[choice]
        self.means_squared[choice] += increment_squared
    
    def update_variance(self, choice):
        self.variances[choice] = self.means_squared[choice] - self.means[choice]**2



class ucb_moss_player(player):
    def __init__(self, n_arms, n_rounds, beta):
        super().__init__(n_arms)
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


class ucb_kl_player(player):
    def __init__(self, n_arms, dist):
        super().__init__(n_arms)
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

def decay_prob(d):
    return lambda t: (t**(d))*((10*np.log(t))**(1/3))

class eps_player(player):
    def __init__(self, n_arms, prob):
        super().__init__(n_arms)
        self.exps = []
        self.prob = prob
        
    def exp_vs_exp(self, current_round):
        if type(self.prob) == float:
            eps = self.prob
        else:
            eps = self.prob(current_round)
        u = np.random.uniform(0,1)
        if u <= eps:
            self.exps.append('Exploration')
            return 'Exploration'
        else:
            self.exps.append('Exploitation')
            return 'Exploitation'
        
    def choice(self, exp):
        if exp == 'Exploration':
            choice = np.random.choice(np.arange(self.n_arms))
        else:
            choice = np.argmax(self.means)
        self.choices.append(choice)
        return choice


class bayes_player(player):
    def __init__(self, n_arms, dist, shape = 1, scale = 1):
        super().__init__(n_arms)
        self.ucbs = np.zeros(shape = n_arms)
        self.dist = dist
        self.a = np.ones(shape = n_arms)
        self.b = np.ones(shape = n_arms)
        self.shape = np.array([shape for i in range(n_arms)])
        if dist == 'Exponential' or dist == 'Poisson':
            self.scale = [1/scale for i in range(n_arms)]
        else:
            self.scale = [scale for i in range(n_arms)]
        self.all_samples = np.zeros(shape  = (n_arms, 5000))
        self.alphas = [np.ones(shape = n_arms) for i in range(self.n_arms)]
                
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

class thom_player(player):
    def __init__(self, n_arms, dist, lb = 0, ub = 1):
        super().__init__(n_arms)
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
                
class softmax_player(player):
    def __init__(self, n_arms, temp, dist):
        super().__init__(n_arms)
        self.temp = temp
        self.probs = np.zeros(shape = n_arms)
        self.dist = dist
        
    def compute_probs(self, current_round):
        if type(self.temp) == float:
            den = np.sum(np.exp(self.means/self.temp))
            num = np.exp(self.means/self.temp)
        else:
            if self.dist == 'Bernoulli':
                min_value = 0.01
            else:
                min_value = 0.05
            if self.temp(current_round) < min_value: 
                final_temp = min_value
            else:
                final_temp = self.temp(current_round)
            den = np.sum(np.exp(self.means/final_temp))
            num = np.exp(self.means/final_temp)
        self.probs = num/den
        
    def choice(self):
        choice = np.random.choice(np.arange(self.n_arms), p = self.probs)
        self.choices.append(choice)
        return choice

class soft_eps_player(eps_player):
    def __init__(self, n_arms, prob, temp, dist):
        super().__init__(n_arms, prob)
        self.temp = temp
        self.dist = dist
        
    def choice(self, exp, current_round):
        if exp == 'Exploration':
            if type(self.temp) == float:
                den = np.sum(np.exp(self.means/self.temp))
                num = np.exp(self.means/self.temp)
            else:
                if self.dist == 'Bernoulli':
                    min_value = 0.01
                else:
                    min_value = 0.05
                if self.temp(current_round) < min_value: 
                    final_temp = min_value
                else:
                    final_temp = self.temp(current_round)
                den = np.sum(np.exp(self.means/final_temp))
                num = np.exp(self.means/final_temp)
            self.probs = num/den
            choice = np.random.choice(np.arange(self.n_arms), p = num/den)
        else:
            choice = np.argmax(self.means)
        self.choices.append(choice)
        return choice





