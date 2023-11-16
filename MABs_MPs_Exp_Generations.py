def multiple_experiment(n_arms, n_players, n_rounds, n_iterations, com_freqs, title, player_type, lambdas = None, beta = 1, conc = 0.5, dist = 'Dirichlet', temp = None, n_variables = None, scale = 1, shape = 1, d = None):
    if dist == 'Dirichlet':
        env = env_generation(n_arms = n_arms, n_variables = n_variables, conc = conc)
        ranked_means = sorted(env.all_means)
        if player_type == 'thom':
            lb = 1
            ub = n_variables
    elif dist == 'Bernoulli':
        env = env_generation(n_arms = n_arms, n_variables = n_variables, dist = dist)
        ranked_means = sorted(env.all_means)
        if player_type == 'thom':
            lb = 0
            ub = 1
        
    if player_type == 'thom':
        if dist == 'Exponential':
            ub = np.random.exponential(scale = np.max(lambdas), size = 1000).max()
            lb = np.random.exponential(scale = np.min(lambdas), size = 1000).min()
        elif dist == 'Poisson':
            ub = np.random.poisson(lam = np.max(lambdas), size = 1000).max()
            lb = np.random.poisson(lam = np.min(lambdas), size = 1000).min()
        elif dist == 'Dirichlet':
            ub = 10
            lb = 1
       
    trials = pd.DataFrame()
    for f in tqdm(range(len(com_freqs))):
        com_freq = com_freqs[f]
        all_regrets = np.zeros(shape = (n_iterations, n_rounds, n_players))
        all_choices = np.zeros(shape = (n_iterations, n_rounds, n_players))
        
        for j in range(n_iterations):
            choices_rank = np.zeros(shape = (n_rounds, n_players))
            all_players = []
            for i in range(n_players):
                if player_type == 'ucb1':
                    player = ucb_1_player(n_arms = n_arms, beta = beta, player_index = i, n_players = n_players, n_rounds = n_rounds)
                elif player_type == 'ucbv':
                    player = ucb_v_player(n_arms = n_arms, beta = beta, player_index = i, n_players = n_players, n_rounds = n_rounds)
                elif player_type == 'moss':
                    player = ucb_moss_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, beta = beta)
                elif player_type == 'ucbkl':
                    player = ucb_kl_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, dist = dist)
                elif player_type == 'soft-eps':
                    player = soft_eps_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, temp = temp)
                elif player_type == 'bayes':
                    player = bayes_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, dist = dist, scale = scale, shape = shape)
                elif player_type == 'thom':
                    player = thom_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, dist = dist, lb = lb, ub = ub)
                elif player_type == 'eps':
                    player = eps_player(n_arms = n_arms, player_index = i, n_players = n_players, n_rounds = n_rounds, prob = decay_prob(d))
                all_players.append(player)
            counter = 0
            for i in range(1, n_rounds + 1):
                choices = [None for i in range(n_players)]
                rewards = np.zeros(shape = n_players)
                if i <= all_players[0].n_arms:
                    choices = [i-1 for j in range(n_players)]
                    for n in range(n_players):
                        all_players[n].choices.append(choices[n])
                else:
                    for n in range(n_players):
                        if i in com_freq:
                            all_players[n].choices.append(None)
                            for m in range(n_players):
                                if j != m:
                                    all_players[n].receive_n_trials(all_players[m].all_trials)
                                    all_players[n].receive_reward(all_players[m].all_rewards)
                                    counter += 1
                        else:
                            if player_type == 'ucbkl':
                                all_players[n].compute_ucbs(i, l_rate = 0.01, max_iter = 100000)
                            elif player_type in ['ucb1', 'ucbv', 'moss', 'bayes']:
                                all_players[n].compute_ucbs(i)
                            elif player_type == 'soft-eps' or player_type == 'eps':
                                exp = all_players[n].exp_vs_exp(i)
                                
                            if player_type == 'soft-eps' or player_type == 'eps':
                                choices[n] = all_players[n].choice(exp)
                            else:
                                choices[n] = all_players[n].choice()
                                
                if dist == 'Dirichlet' or dist == 'Bernoulli':
                    for p in range(len(choices)):
                        if choices[p] != None:
                            choices_rank[i-1, p] = ranked_means.index(env.all_means[choices[p]])
                        else:
                            choices_rank[i-1, p] = None
                                                                      
                for k in range(n_players):
                    if choices[k] != None:
                        if dist == 'Dirichlet' or dist == 'Bernoulli':
                            rewards[k] = np.random.choice(env.rewards, p = env.probs_matrix[choices[k], :])
                            best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                        elif dist == 'Poisson':
                            rewards[k] = np.random.poisson(lam = lambdas[choices[k]])
                            best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                        elif dist == 'Exponential':
                            rewards[k] = np.random.exponential(scale = lambdas[choices[k]])
                            best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])

                for h in range(n_players):
                    all_players[h].update_n_trials(choices[h], i-1)
                    all_players[h].update_mean(choices[h], rewards[h], i-1)
                    if player_type == 'ucbv':
                        all_players[h].update_variance(choices[h])
                    if player_type == 'bayes':
                        all_players[h].update_params(choices[h], rewards[h])
                    if player_type == 'thom':
                        if dist != 'Bernoulli':
                            norm_reward = all_players[h].normalize(rewards[h])
                            all_players[h].update_params(choices[h], norm_reward)
                        else:
                            all_players[h].update_params(choices[h], rewards[h])
                    all_players[h].update_regret(rewards[h], best_reward)

            for l in range(n_players):
                all_regrets[j, :, l] = all_players[l].cumulative_regrets
                if dist == 'Bernoulli' or dist == 'Dirichlet':
                    all_choices[j, : , l] = choices_rank[:, l]
                else:
                    all_choices[j, : , l] = all_players[l].choices

        for i in range(n_players):
            trials[f'p{i+1}_{com_freq[0]}_regrets'] = all_regrets[:, :, i].mean(axis = 0)
            trials[f'p{i+1}_{com_freq[0]}_choices'] = all_choices[:, :, i].mean(axis = 0)
            trials[f'p{i+1}_{com_freq[0]}_stds'] = all_regrets[:, :, i].std(axis = 0)

    trials.to_csv(f'{n_players}_{title}.csv')

def multiple_ucb_1(n_arms, n_players, n_rounds, n_iterations, com_freqs, title, lambdas = None, beta = 1, conc = 0.5, dist = 'Dirichlet', n_variables = None):
    if dist == 'Dirichlet':
        env = env_generation(n_arms = n_arms, n_variables = n_variables, conc = conc)
    if dist == 'Bernoulli':
        env = env_generation(n_arms = n_arms, n_variables = n_variables, dist = dist)
        
    trials = pd.DataFrame()
    for f in tqdm(range(len(com_freqs))):
        com_freq = com_freqs[f]
        all_regrets = np.zeros(shape = (n_iterations, n_rounds, n_players))
        all_choices = np.zeros(shape = (n_iterations, n_rounds, n_players))
        
        for j in range(n_iterations):
            all_players = []
            for i in range(n_players):
                player = ucb_1_player(n_arms = n_arms, beta = beta, player_index = i, n_players = n_players, n_rounds = n_rounds)
                all_players.append(player)

            counter = 0
            for i in range(1, n_rounds + 1):
                choices = [None for i in range(n_players)]
                rewards = np.zeros(shape = n_players)
                if i <= all_players[0].n_arms:
                    choices = [i-1 for j in range(n_players)]
                    for n in range(n_players):
                        all_players[n].choices.append(choices[n])
                else:
                    for n in range(n_players):
                        if i in com_freq:
                            all_players[n].choices.append(None)
                            for m in range(n_players):
                                if j != m:
                                    all_players[n].receive_n_trials(all_players[m].all_trials)
                                    all_players[n].receive_reward(all_players[m].all_rewards)
                                    counter += 1
                        else:
                            all_players[n].compute_ucbs(i)
                            choices[n] = all_players[n].choice()

                for k in range(n_players):
                    if choices[k] != None:
                        if dist == 'Dirichlet' or dist == 'Bernoulli':
                            rewards[k] = np.random.choice(env.rewards, p = env.probs_matrix[choices[k], :])
                            best_reward = np.random.choice(env.rewards, p = env.probs_matrix[env.best_arm, :])
                        elif dist == 'Poisson':
                            rewards[k] = np.random.poisson(lam = lambdas[choices[k]])
                            best_reward = np.random.poisson(lam = lambdas[np.argmax(lambdas)])
                        elif dist == 'Exponential':
                            rewards[k] = np.random.exponential(scale = lambdas[choices[k]])
                            best_reward = np.random.exponential(scale = lambdas[np.argmax(lambdas)])

                for h in range(n_players):
                    all_players[h].update_n_trials(choices[h], i-1)
                    all_players[h].update_mean(choices[h], rewards[h], i-1)
                    all_players[h].update_regret(rewards[h], best_reward)

            for l in range(n_players):
                all_regrets[j, :, l] = all_players[l].cumulative_regrets
                all_choices[j, : ,l] = all_players[l].choices

        for i in range(n_players):
            trials[f'p{i+1}_{com_freq[0]}_regrets'] = all_regrets[:, :, i].mean(axis = 0)
            trials[f'p{i+1}_{com_freq[0]}_choices'] = all_choices[:, :, i].mean(axis = 0)
            trials[f'p{i+1}_{com_freq[0]}_stds'] = all_regrets[:, :, i].std(axis = 0)

    trials.to_csv(f'{n_players}_{title}.csv')
