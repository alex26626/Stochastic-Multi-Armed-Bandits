def ucb_1_experiment_generation(n_rounds, n_iterations, beta, env = None, dist = 'Dirichlet', lambdas = None, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_1_player(n_arms = env.n_arms, beta = beta)
            
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_1_player(n_arms = env.n_arms, beta = beta)
            
        else:
            player_1 = ucb_1_player(n_arms = len(lambdas), beta = beta)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                player_1.compute_ucbs(i)
                choice = player_1.choice()
            
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))
            
            if dist == 'Dirichlet' or dist == 'Bernoulli': 
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
                                    
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
    return all_regrets, all_choices

def ucb_v_experiment_generation(n_rounds, n_iterations, beta, env = None, dist = 'Dirichlet', lambdas = None, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_v_player(n_arms = env.n_arms, beta = beta)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_v_player(n_arms = env.n_arms, beta = beta)
            
        else:
            player_1 = ucb_v_player(n_arms = len(lambdas), beta = beta)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                player_1.compute_ucbs(i)
                choice = player_1.choice()
            
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))

            if dist == 'Dirichlet' or dist == 'Bernoulli': 
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_variance(choice)
            player_1.update_regret(reward, best_reward)
            
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
    return all_regrets, all_choices

def ucb_moss_experiment_generation(n_rounds, n_iterations, beta, env = None, dist = 'Dirichlet', lambdas = None, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_moss_player(n_arms = env.n_arms, n_rounds = n_rounds, beta = beta)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_moss_player(n_arms = env.n_arms, n_rounds = n_rounds, beta = beta)
        else:
            player_1 = ucb_moss_player(n_arms = len(lambdas), n_rounds = n_rounds, beta = beta)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                player_1.compute_ucbs(i)
                choice = player_1.choice()
                
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))

            if dist == 'Dirichlet' or dist == 'Bernoulli': 
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward)
            
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
    return all_regrets, all_choices

def ucb_kl_experiment_generation(n_rounds, n_iterations, dist, l_rate, max_iter, lambdas = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = ucb_kl_player(n_arms = env.n_arms, dist = dist)
            
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = ucb_kl_player(n_arms = len(lambdas), dist = dist)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                player_1.compute_ucbs(i, l_rate = l_rate, max_iter = max_iter)
                choice = player_1.choice()
                
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))
            
            if dist == 'Bernoulli':
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward = best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices

def eps_experiment_generation(n_rounds, n_iterations, prob, dist = 'Dirichlet', lambdas = None, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = eps_player(env.n_arms, prob = prob)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = eps_player(env.n_arms, prob = prob)
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = eps_player(len(lambdas), prob = prob)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                exp = player_1.exp_vs_exp(i)
                choice = player_1.choice(exp)
            
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))
                
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward = best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices

def softmax_experiment_generation(n_rounds, n_iterations, temp, dist = 'Dirichlet', lambdas = None, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = softmax_player(n_arms = env.n_arms, temp = temp, dist = dist)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = softmax_player(n_arms = env.n_arms, temp = temp, dist = dist)
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = softmax_player(n_arms = len(lambdas),  temp = temp, dist = dist)
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
                
            else:
                player_1.compute_probs(i)
                choice = player_1.choice()
            
            if dist == 'Bernoulli' or dist == 'Dirichlet':
                choices_rank.append(ranked_means.index(env.all_means[choice]))

            if dist == 'Dirichlet' or dist == 'Bernoulli':
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices

def softmax_eps_experiment_generation_new(n_rounds, n_iterations, temp, prob, dist = 'Dirichlet', conc = None, lambdas = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = soft_eps_player(n_arms = env.n_arms, temp = temp, prob = prob, dist = dist)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = soft_eps_player(n_arms = env.n_arms, temp = temp, prob = prob, dist = dist)
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = soft_eps_player(n_arms = len(lambdas),  temp = temp, prob = prob, dist = dist)
            
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                exp = player_1.exp_vs_exp(i)
                choice = player_1.choice(exp, i)
                
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))

            if dist == 'Dirichlet' or dist == 'Bernoulli':
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                
            elif dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_regret(reward, best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices

def bayes_experiment_generation_new(n_rounds, n_iterations, dist, lambdas = None, shape = 1, scale = 1, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = bayes_player_new(n_arms = env.n_arms, dist = dist)
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = bayes_player_new(n_arms = len(lambdas), dist = dist, shape = shape, scale = scale)
        elif dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, dist = dist, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = bayes_player_new(n_arms = env.n_arms, dist = dist)
        
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
            else:
                player_1.compute_ucbs(i)
                choice = player_1.choice()
            
            if dist == 'Dirichlet' or dist == 'Bernoulli':
                choices_rank.append(ranked_means.index(env.all_means[choice]))
                
            if dist == 'Bernoulli' or dist == 'Dirichlet':
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                
            if dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
                
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            player_1.update_params(choice, reward)
            player_1.update_regret(reward, best_reward)
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices

def thompson_experiment_generation(n_rounds, n_iterations, dist, lambdas = None, lb = 0, ub = 1, conc = None):
    all_regrets = np.zeros(shape = (n_iterations, n_rounds))
    all_choices = np.zeros(shape = (n_iterations, n_rounds))
    
    for j in tqdm(range(n_iterations)):
        choices_rank = []
        if dist == 'Dirichlet':
            env = env_generation(n_arms = 10, n_variables = 10, conc = conc)
            ranked_means = sorted(env.all_means)
            player_1 = thom_player(n_arms = env.n_arms, dist = dist, lb = lb, ub = ub)
        elif dist == 'Bernoulli':
            env = env_generation(n_arms = 10, n_variables = 2, dist = dist)
            ranked_means = sorted(env.all_means)
            player_1 = thom_player(n_arms = env.n_arms, dist = dist, lb = lb, ub = ub)
        elif dist == 'Poisson' or dist == 'Exponential':
            player_1 = thom_player(n_arms = len(lambdas), dist = dist, lb = lb, ub = ub)
        
        for i in range(1, n_rounds + 1):
            if i <= player_1.n_arms:
                choice = i - 1
                player_1.choices.append(choice)
                
            else:
                choice = player_1.choice()
                
            if dist == 'Bernoulli' or dist == 'Dirichlet':
                choices_rank.append(ranked_means.index(env.all_means[choice]))
            
            if dist == 'Bernoulli' or dist == 'Dirichlet':
                reward = np.random.choice(env.rewards, p = env.probs_matrix[choice, :])
                best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                
            if dist == 'Poisson':
                best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                reward = np.random.poisson(lam = lambdas[choice])
                
            elif dist == 'Exponential':
                best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])
                reward = np.random.exponential(scale = lambdas[choice])
                
            player_1.update_n_trials(choice)
            player_1.update_mean(choice, reward)
            if dist != 'Bernoulli':
                norm_reward = player_1.normalize(reward)
                player_1.update_params(choice, norm_reward)
            else:
                player_1.update_params(choice, reward)
            player_1.update_regret(reward, best_reward)
            
        all_regrets[j, :] = player_1.cumulative_regrets
        if dist == 'Dirichlet' or dist == 'Bernoulli':
            all_choices[j, :] = choices_rank
        else:
            all_choices[j, :] = player_1.choices
        
    return all_regrets, all_choices


