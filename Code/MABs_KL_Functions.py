def kl_bern(p, q):
    return p*(np.log(p/q)) + (1-p)*np.log((1-p)/(1-q))

def kl_bern_first(p, q):
    return -p*q + (1-p)/(1-q)

def kl_bern_second(p, q):
    return p/q**2 + (1-p)/(1-q)**2

def kl_bern_max(q0, p, l_rate, max_iter, t, s):
    n = 0
    q_s = []
    kls = []
    constraint = np.log(t)
    current = 0
    while q0 > 0 and q0 < 1 and n < max_iter and current < constraint:
        q_s.append(q0)
        current = s*kl_bern(p, q0)
        kls.append(current)
        #increment = l_rate*kl_bern_first(p, q0)/kl_bern_second(p, q0)
        increment = 0.05
        q_t = q0 + increment
        q0 = q_t
        n += 1
    return q_t

def kl_pois(l1, l2):
    return l2 - l1 + l1*np.log(l1/l2)

def kl_pois_first(l1, l2):
    return -l1/l2 + 1 - l1

def kl_pois_second(l1, l2):
    return l1/l2**2

def kl_pois_max(l0, l1, l_rate, max_iter, t, s, upper_bound):
    n = 0
    l_s = []
    kls = []
    constraint = np.log(t)
    current = 0
    while l0 > 0 and l0 < upper_bound  and n < max_iter and current < constraint:
        l_s.append(l0)
        current = s*kl_pois(l1, l0)
        kls.append(current)
        #increment = l_rate*kl_pois_first(l1, l0)/kl_pois_second(l1, l0)
        increment = 0.5
        l0 = l0 + increment
        n += 1
    return l0

def kl_exp(b1, b2):
    return b1/b2 - 1 - np.log(b1/b2)

def kl_exp_first(b1, b2):
    return -b1/b2**2 + 1/b2

def kl_exp_second(b1, b2):
    return 2*b1/b2**3 - 1/b2**2

def kl_exp_max(l0, l1, l_rate, max_iter, t, upper_bound, s):
    n = 0
    constraint = np.log(t)
    current = 0
    while l0 > 0 and l0 < upper_bound  and n < max_iter and current < constraint:
        if l0 == 0:
            l0 = 0.01
        current = s*kl_exp(l1, l0)
        #increment = l_rate*kl_exp_first(l1, l0)/kl_exp_second(l1, l0)
        increment = 0.5
        l0 +=  increment
        n += 1
    return l0
