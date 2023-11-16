def dirichlet_comparison(n_rounds, n_iterations, n_arms, n_variables, conc, beta_1, beta_v, soft_temp, soft_eps_temp, title, dist = 'Dirichlet'):
    env = env_generation(n_arms = n_arms, n_variables = n_variables, conc = conc)
    regrets_thom, choices_thom = thompson_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist, env = env, lb = 1, ub = n_variables)
    regrets_ucb1, choices_ucb1 = ucb_1_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_1, env = env)
    regrets_ucbv, choices_ucbv = ucb_v_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_v, env = env)
    regrets_moss, choices_moss = ucb_moss_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, env = env)
    regrets_eps, choices_eps = eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, env = env)
    regrets_soft, choices_soft = softmax_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, env = env, temp = soft_temp)
    regrets_soft_eps, choices_soft_eps = softmax_eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, env = env, temp = soft_eps_temp)

    all_regrets = [regrets_ucb1.mean(axis = 0), regrets_ucbv.mean(axis = 0), regrets_moss.mean(axis = 0), regrets_eps.mean(axis = 0), regrets_soft.mean(axis = 0), regrets_soft_eps.mean(axis = 0), regrets_thom.mean(axis = 0)]
    all_choices = [choices_ucb1, choices_ucbv, choices_moss, choices_eps, choices_soft, choices_soft_eps, choices_thom]
    all_stds = [regrets_ucb1.std(axis = 0), regrets_ucbv.std(axis = 0), regrets_moss.std(axis = 0), regrets_eps.std(axis = 0), regrets_soft.std(axis = 0), regrets_soft_eps.std(axis = 0), regrets_thom.std(axis = 0)]
    cols = ['ucb1', 'ucbv', 'moss', 'epsilon', 'softmax', 'soft-eps', 'thompson']
    
    exp_data = pd.DataFrame()
    for i in range(len(all_regrets)):
        exp_data[cols[i] + '_regrets'] = all_regrets[i]
        exp_data[cols[i] + '_choices'] = all_choices[i]
        exp_data[cols[i] + '_stds'] = all_stds[i]
    
    file_name = title + '.csv'
    exp_data.to_csv(file_name)


def bernoulli_comparison(n_rounds, n_iterations, n_arms, n_variables, beta_1, beta_v, soft_temp, soft_eps_temp, l_rate, max_iter, title, dist = 'Bernoulli'):
    regrets_ucb1, choices_ucb1 = ucb_1_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_1, dist = dist)
    regrets_ucbv, choices_ucbv = ucb_v_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_v, dist = dist)
    regrets_moss, choices_moss = ucb_moss_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist)
    regrets_eps, choices_eps = eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist)
    regrets_soft, choices_soft = softmax_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, temp = soft_temp, dist = dist)
    regrets_soft_eps, choices_soft_eps = softmax_eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, temp = soft_eps_temp, dist = dist)
    regrets_kl, choices_kl = ucb_kl_experiment_generation(n_rounds, n_iterations, dist = dist, l_rate = l_rate, max_iter = max_iter)
    regrets_bayes, choices_bayes = bayes_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist)
    regrets_thom, choices_thom = thompson_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist, lb = 0, ub = 1)
    
    all_regrets = [regrets_ucb1.mean(axis = 0), regrets_ucbv.mean(axis = 0), regrets_moss.mean(axis = 0), regrets_eps.mean(axis = 0), regrets_soft.mean(axis = 0), regrets_soft_eps.mean(axis = 0), regrets_thom.mean(axis = 0), regrets_bayes.mean(axis = 0), regrets_kl.mean(axis = 0)]
    all_choices = [choices_ucb1, choices_ucbv, choices_moss, choices_eps, choices_soft, choices_soft_eps, choices_thom, choices_bayes, choices_kl]
    all_stds = [regrets_ucb1.std(axis = 0), regrets_ucbv.std(axis = 0), regrets_moss.std(axis = 0), regrets_eps.std(axis = 0), regrets_soft.std(axis = 0), regrets_soft_eps.std(axis = 0), regrets_thom.std(axis = 0), regrets_bayes.std(axis = 0), regrets_kl.std(axis = 0)]
    cols = ['ucb1', 'ucbv', 'moss', 'epsilon', 'softmax', 'soft-eps', 'thompson', 'ucbbayes', 'ucbkl']
    
    exp_data = pd.DataFrame()
    for i in range(len(all_regrets)):
        exp_data[cols[i] + '_regrets'] = all_regrets[i]
        exp_data[cols[i] + '_choices'] = all_choices[i]
        exp_data[cols[i] + '_stds'] = all_stds[i]
    
    file_name = title + '.csv'
    exp_data.to_csv(file_name)

def pois_exp_comparison(n_rounds, n_iterations, beta_1, beta_v, soft_temp, soft_eps_temp, l_rate, max_iter, lambdas, title, dist):
    if dist == 'Poisson':
        lb =  np.random.poisson(lam = np.min(lambdas), size = 2000).min()
        ub = np.random.poisson(lam = np.max(lambdas), size = 2000).max()
    elif dist == 'Exponential':
        ub = np.random.exponential(scale = np.max(lambdas), size = 2000).max()
        lb = np.random.exponential(scale = np.min(lambdas), size = 2000).min()
    
    regrets_kl, choices_kl = ucb_kl_experiment_generation(n_rounds, n_iterations, dist = dist, l_rate = l_rate, max_iter = max_iter, lambdas = lambdas)
    regrets_bayes, choices_bayes = bayes_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist, lambdas = lambdas)
    regrets_thom, choices_thom = thompson_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, dist = dist, lambdas = lambdas, lb = 0, ub = ub)
    regrets_ucb1, choices_ucb1 = ucb_1_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_1, lambdas = lambdas, dist = dist)
    regrets_ucbv, choices_ucbv = ucb_v_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, beta = beta_v, lambdas = lambdas, dist = dist)
    regrets_moss, choices_moss = ucb_moss_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, lambdas = lambdas, dist = dist)
    regrets_eps, choices_eps = eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, lambdas = lambdas, dist = dist)
    regrets_soft, choices_soft = softmax_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, temp = soft_temp, lambdas = lambdas, dist = dist)
    regrets_soft_eps, choices_soft_eps = softmax_eps_experiment_generation(n_rounds = n_rounds, n_iterations = n_iterations, temp = soft_eps_temp, lambdas = lambdas, dist = dist)
    
    all_regrets = [regrets_ucb1.mean(axis = 0), regrets_ucbv.mean(axis = 0), regrets_moss.mean(axis = 0), regrets_eps.mean(axis = 0), regrets_soft.mean(axis = 0), regrets_soft_eps.mean(axis = 0), regrets_thom.mean(axis = 0), regrets_bayes.mean(axis = 0), regrets_kl.mean(axis = 0)]
    all_choices = [choices_ucb1.mean(axis = 0), choices_ucbv.mean(axis = 0), choices_moss.mean(axis = 0), choices_eps.mean(axis = 0), choices_soft.mean(axis = 0), choices_soft_eps.mean(axis = 0), choices_thom.mean(axis = 0), choices_bayes.mean(axis = 0), choices_kl.mean(axis = 0)]
    all_stds = [regrets_ucb1.std(axis = 0), regrets_ucbv.std(axis = 0), regrets_moss.std(axis = 0), regrets_eps.std(axis = 0), regrets_soft.std(axis = 0), regrets_soft_eps.std(axis = 0), regrets_thom.std(axis = 0), regrets_bayes.std(axis = 0), regrets_kl.std(axis = 0)]
    cols = ['ucb1', 'ucbv', 'moss', 'epsilon', 'softmax', 'soft-eps', 'thompson', 'ucbbayes', 'ucbkl']
    
    exp_data = pd.DataFrame()
    for i in range(len(all_regrets)):
        exp_data[cols[i] + '_regrets'] = all_regrets[i]
        exp_data[cols[i] + '_choices'] = all_choices[i]
        exp_data[cols[i] + '_stds'] = all_stds[i]
    
    file_name = title + '.csv'
    exp_data.to_csv(file_name)
    
def real_experiment(data, player, beta = None, dist = None, l_rate = None, max_iter = None, lb = 0, ub = 0, temp = None):
    n_arms = data.shape[1]
    best_arm = np.argmax(data.mean(axis = 0))
    
    if player == 'ucb_1':
        player_1 = ucb_1_player(n_arms = n_arms, beta = beta)
    elif player == 'ucb_v':
         player_1 = ucb_v_player(n_arms = n_arms, beta = beta)
    elif player == 'ucb_moss':
        player_1 = ucb_moss_player(n_arms = n_arms, n_rounds = data.shape[0])
    elif player == 'ucb_kl':
        player_1 = ucb_kl_player(n_arms = n_arms, dist = dist)
    elif player == 'ucb_bayes':
        player_1 = bayes_player(n_arms = n_arms, dist = dist)
    elif player == 'thompson':
        player_1 = thom_player(n_arms = n_arms, dist = dist, lb = lb, ub = ub)
    elif player == 'epsilon':
        player_1 = eps_player(n_arms = n_arms)
    elif player == 'softmax':
        player_1 = softmax_player(n_arms = n_arms, temp = temp)
    elif player == 'soft_eps':
        player_1 = soft_eps_player(n_arms = n_arms, temp = temp)
    
    for i in tqdm(range(1, data.shape[0] + 1)):
        if i <= n_arms:
            choice = i - 1
            player_1.choices.append(choice)
        else:
            if player in ['ucb_1', 'ucb_v', 'ucb_moss', 'ucb_bayes']:
                player_1.compute_ucbs(i)
                choice = player_1.choice()
            elif player == 'ucb_kl':
                player_1.compute_ucbs(i, l_rate = l_rate, max_iter = max_iter)
                choice = player_1.choice()
            elif player in ['epsilon', 'soft_eps']:
                exp = player_1.exp_vs_exp(i)
                choice = player_1.choice(exp)
            elif player == 'softmax':
                player_1.compute_probs()
                choice = player_1.choice()
            elif player == 'thompson':
                choice = player_1.choice()
        
        reward = data.iloc[i-1, choice]
        best_reward  = data.iloc[i-1, best_arm]
        player_1.update_n_trials(choice)
        player_1.update_mean(choice, reward)
        player_1.update_regret(reward, best_reward)
        if player == 'ucb_v':
            player_1.update_variance(choice)
        elif player == 'ucb_bayes':
            player_1.update_params(choice, reward)
        elif player == 'thompson':
            if dist != 'Bernoulli':
                norm_reward = player_1.normalize(reward)
                player_1.update_params(choice, norm_reward)
            else:
                player_1.update_params(choice, reward)
    return player_1.cumulative_regrets, player_1.choices
