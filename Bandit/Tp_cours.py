import bandits
import utils
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

HORIZON = 10000


def aleatoire(b, horizon):
    regrets = []
    rewards = []

    for i in range(horizon):
        arm = np.random.randint(b.nbr_arms)
        # cumulative regrets
        if len(regrets) != 0:
            regrets.append(regrets[-1] + b.regrets[arm])
        else:
            regrets.append(b.regrets[arm])

        # cumulative rewards
        if len(rewards) != 0:
            rewards.append(rewards[-1] + b.pull(arm))
        else:
            rewards.append(b.pull(arm))
    return regrets, rewards


def aleatoire_1(bandit, horizon):
    Regret = np.zeros(horizon)
    Reward = np.zeros(horizon)

    for t in range(bandit.nbr_arms, horizon):
        a = np.random.randint(bandit.nbr_arms)
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regret[t] = Regret[max(0, t-1)] + r
        Reward[t] = Reward[max(0, t-1)] + reward

    return Regret, Reward


def eps_glouton(bandit, horizon, eps):
    Rewards = np.zeros(horizon)
    Regrets = np.zeros(horizon)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    nbr_echantillon = np.zeros(bandit.nbr_arms)

    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        p = np.random.rand()
        # partie exploration
        if p < eps:
            a = np.random.randint(bandit.nbr_arms)
        # sinon amelioration
        else:
            a = np.argmax(moyennes_empirique)
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    return Regrets, Rewards


def eps_glouton_dec(bandit, horizon, eps):
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    Rewards = np.zeros(horizon)
    Regrets = np.zeros(horizon)
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        p = np.random.uniform(0, 1, 1)
        # partie exploration
        if p < eps:
            a = np.random.randint(bandit.nbr_arms)
        # sinon amelioration
        else:
            a = np.argmax(np.divide(moyennes_empirique))
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]
        eps *= 0.999
    return Regrets, Rewards


def proportionnelle(bandit, horizon):
    Rewards = np.zeros(horizon)
    Regrets = np.zeros(horizon)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        a = rdm.choices(np.arange(bandit.nbr_arms), moyennes_empirique)[0]
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]
    return Regrets, Rewards


def boltzmann(bandit, horizon, tau):
    Rewards = np.zeros(horizon)
    Regrets = np.zeros(horizon)
    moyennes_empirique = np.zeros(bandit.nbr_arms)

    nbr_echantillon = np.zeros(bandit.nbr_arms)
    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        a = rdm.choices(np.arange(bandit.nbr_arms), np.exp(
            moyennes_empirique / tau) / np.sum(np.exp(moyennes_empirique / tau)))[0]
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    return Regrets, Rewards


def ucb(bandit, horizon):
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    Regrets = np.zeros(horizon)
    Rewards = np.zeros(horizon)

    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        a = utils.randamax(moyennes_empirique +
                           np.sqrt(0.5 * np.log(t) / (nbr_echantillon)))
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    return Regrets, Rewards


def thompson(bandit, horizon):
    nbr_echantillon = np.ones(bandit.nbr_arms)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    S = np.ones(bandit.nbr_arms)
    Regrets = np.zeros(horizon)
    Rewards = np.zeros(horizon)

    for a in range(bandit.nbr_arms):
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        S[a]+= reward
        Regrets[a] = Regrets[max(0, a-1)] + r
        Rewards[a] = Rewards[max(0, a-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    for t in range(bandit.nbr_arms, horizon):
        tmp = np.zeros(bandit.nbr_arms)
        for b in range(bandit.nbr_arms):
            tmp[b] = rdm.betavariate(S[b],nbr_echantillon[b] - S[b] + 1)
        a = np.argmax(tmp)
        if a == bandit.best_arm:
            r = 0
        else:
            r = 1
        reward = bandit.pull(a)
        S[a]+= reward
        Regrets[t] = Regrets[max(0, t-1)] + r
        Rewards[t] = Rewards[max(0, t-1)] + reward
        moyennes_empirique[a] = nbr_echantillon[a] * \
            moyennes_empirique[a] + reward
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]

    return Regrets, Rewards


def plusieursXP(algo, bandit, horizon, N, eps=None, tau=None):
    Regrets = np.zeros((N, horizon))
    Rewards = np.zeros((N, horizon))
    for exp in range(N):
        if algo == eps_glouton or algo == eps_glouton_dec:
            R, REW = algo(bandit, horizon, eps)
        elif algo == boltzmann:
            R, REW = algo(bandit, horizon, tau)
        else:
            R, REW = algo(bandit, horizon)
        Regrets[exp] = R
        Rewards[exp] = REW
    Regret_moyen = np.mean(Regrets, axis=0)
    Regret_std = np.std(Regrets, axis=0)
    return Regret_moyen, Regret_std, REW, Regrets, Rewards



if __name__ == "__main__":
    np.random.seed(1234)
    b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]))
    EXP = 100
    H = 10000


    r, std, rew, regrets, rewards = plusieursXP(aleatoire_1, b, HORIZON, EXP)
    plt.plot(r, label="Regret_aleatoire")
    plt.fill_between(np.arange(H), np.maximum(0, r - std), r + std, alpha=0.3)

    # plt.plot(rew, label="Reward_aleatoire")
    # for i in range(len(regrets)):
    #     plt.plot(regrets[i])

    # for epsilon in [0.1,0.5,0.85]:
    #     r, std,rew,regrets,rewards= plusieursXP(eps_glouton, b, HORIZON, EXP,epsilon)
    #     plt.plot(r, label="Regret_gloutonne_"+str(epsilon))
    #     plt.plot(rew, label="Reward_gloutonne_"+str(epsilon))

    r_eps, std, rew, regrets, rewards = plusieursXP(
        eps_glouton, b, HORIZON, EXP, eps=0.1)
    plt.plot(r_eps, label="Regret_gloutonne")
    plt.fill_between(np.arange(H), np.maximum(0, r_eps - std), r_eps + std, alpha=0.3)

    # plt.plot(rew, label="Reward_gloutonne_dec_1")

    r_propor, std, rew, regrets, rewards = plusieursXP(
        proportionnelle, b, HORIZON, EXP)
    plt.plot(r_propor, label="Proportionnelle")
    # plt.plot(rew, label="Proportionnelle")
    plt.fill_between(np.arange(H), np.maximum(0, r_propor - std), r_propor + std, alpha=0.3)


    r_bolt, std, rew, regrets, rewards = plusieursXP(
        boltzmann, b, HORIZON, EXP, tau=0.5)
    plt.plot(r_bolt, label="Boltzmann")
    # plt.plot(rew, label="Proportionnelle")
    plt.fill_between(np.arange(H), np.maximum(0, r_bolt - std), r_bolt + std, alpha=0.3)


    r_ucb, std, rew, regrets, rewards = plusieursXP(
        ucb, b, HORIZON, EXP)
    plt.plot(r_ucb, label="UCB")
    # plt.plot(rew, label="Proportionnelle")
    plt.fill_between(np.arange(H), np.maximum(0, r_ucb - std), r_ucb + std, alpha=0.3)

    
    r_ucb, std, rew, regrets, rewards = plusieursXP(
        thompson, b, HORIZON, EXP)
    plt.plot(r_ucb, label="Thompson")
    # plt.plot(rew, label="Proportionnelle")
    plt.fill_between(np.arange(H), np.maximum(0, r_ucb - std), r_ucb + std, alpha=0.3)

    
    plt.title(f"Regret - Horizon = {HORIZON}, Nombre d'XPs = {EXP}")
    plt.legend()
    plt.show()
