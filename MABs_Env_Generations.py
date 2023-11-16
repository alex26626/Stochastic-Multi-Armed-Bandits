class env_generator:
    def __init__(self, n_arms, n_variables, dist = 'Dirichlet'):
        self.n_arms = n_arms
        self.n_variables = n_variables
        self.probs_matrix = None
        self.rewards = None
        self.best_arm = None
        self.all_means = None
        self.dist = dist
    
    def rewards_generator(self, random = False, lb = 0, ub = 0):
        if self.dist == 'Dirichlet':
            if not random:
                self.rewards = np.arange(1, self.n_variables + 1)
            else:
                self.rewards = np.random.uniform(lb, ub, size = self.n_arms)
        elif self.dist == 'Bernoulli':
            self.rewards = np.array([0,1])
    
    def probs_generator(self, conc):
        self.probs_matrix = np.zeros(shape = (self.n_arms, self.n_variables))
        
        if self.dist == 'Dirichlet':
            alpha = [conc for i in range(self.n_arms)]
            p_generator = np.random.default_rng()
            for j in range(self.n_arms):
                self.probs_matrix[j, :] = p_generator.dirichlet(alpha = alpha)
                
        elif self.dist == 'Bernoulli':
            for j in range(self.n_arms):
                p = np.random.uniform(0,1)
                self.probs_matrix[j, :] = [1-p, p]
    
    def best_arm_def(self):
        if self.dist == 'Dirichlet':
            self.all_means = (self.rewards*self.probs_matrix).sum(axis = 1)
        elif self.dist == 'Bernoulli':
            self.all_means = self.probs_matrix[:, 1]
        self.best_arm = np.argmax(self.all_means)

def env_generation(n_arms, n_variables, conc = None, dist = 'Dirichlet', random = False, lb = 0, ub = 0):
    env = env_generator(n_arms = n_arms, n_variables = n_variables, dist = dist)
    env.rewards_generator(random = random, lb = lb, ub = ub)
    env.probs_generator(conc = conc)
    env.best_arm_def()
    return env
