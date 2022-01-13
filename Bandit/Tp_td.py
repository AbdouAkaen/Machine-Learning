from numpy.core.fromnumeric import argmax
import bandits
import utils
import numpy as np
from math import sin
import matplotlib.pyplot as plt
import numpy as np
import random as rdm

HORIZON =100

# b = bandits.BernoulliBandit (np.array ([0.3, 0.42, 0.4]))

# def version_1(b,n=1000):
#     gains = []
#     regret_cumulate = []
#     res = 0
#     regret=0
#     for i in range(n):
#         arm = np.random.randint(b.nbr_arms)
#         if arm != b.best_arm: 
#             regret += 1
#             regret_cumulate.append(regret)
#         res += b.pull(arm)
#         gains.append(res)
#     return gains,regret_cumulate

    
b1 = bandits.NormalBandit(np.array ([0.1, 0.3, 0.4, 0.75, 0.8, 0.9, 0.95]))
b2 = bandits.BernoulliBandit(np.array ([0.1, 0.3, 0.4, 0.75, 0.8, 0.9, 0.95]))



def roundRobin(b,horizon):
    regrets = []
    rewards = []
    
    for i in range(horizon):
        arm = i % b.nbr_arms
        # cumulative regrets
        if len(regrets)!=0:regrets.append(regrets[-1] + b.regrets[arm])
        else: regrets.append(b.regrets[arm])
            
        # cumulative rewards
        if len(rewards) != 0 :  rewards.append(rewards[-1] + b.pull(arm))
        else: rewards.append(b.pull(arm))
    return regrets


def round_robin(bandit, horizon):
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    Regret = np.zeros(horizon)

    for t in range(horizon):
        a = utils.randamin(nbr_echantillon)
        nbr_echantillon[a] += 1
        r = bandit.regrets[a]
        Regret[t] = Regret[max(0, t-1)] + r
    return Regret





def etc(b,horizon):
    nbr_echantillon = np.zeros(b.nbr_arms)
    Gains = np.zeros(b.nbr_arms)
    Regret = np.zeros(horizon)
    t = 0
    while t < horizon:
        if t <= 300:
            a = utils.randamin(nbr_echantillon)
            nbr_echantillon[a] += 1
            r = b.regrets[a]
            g = b.pull(a)
            Regret[t] = Regret[max(0, t-1)] + r
            Gains[a] += g
        else:
            mean_arms = np.divide(Gains,nbr_echantillon)
            arm_chosen = np.argmax(mean_arms)
            r = b.regrets[arm_chosen]
            Regret[t] = Regret[max(0, t-1)] + r
            
        t+=1
    return Regret


def ucb(bandit,horizon):
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    Regret = np.zeros(horizon)

    for t in range(horizon):
        a = utils.randamax(moyennes_empirique + np.sqrt(2 * np.log(t+1) / (nbr_echantillon + 1)))
        echantillon = bandit.pull(a)
        moyennes_empirique[a] = nbr_echantillon[a] * moyennes_empirique[a] + echantillon
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]
        r = bandit.regrets[a]
        Regret[t] = Regret[max(0, t-1)] + r
    return Regret

def imed(bandit,horizon):
    nbr_echantillon = np.zeros(bandit.nbr_arms)
    moyennes_empirique = np.zeros(bandit.nbr_arms)
    Regret = np.zeros(horizon)

    for t in range(horizon):
        a = utils.randamin( (nbr_echantillon ) * utils.klGaussian(moyennes_empirique,np.max(moyennes_empirique)) + np.log(nbr_echantillon ))
        echantillon = bandit.pull(a)
        moyennes_empirique[a] = nbr_echantillon[a] * moyennes_empirique[a] + echantillon
        nbr_echantillon[a] += 1
        moyennes_empirique[a] = moyennes_empirique[a] / nbr_echantillon[a]
        r = bandit.regrets[a]
        Regret[t] = Regret[max(0, t-1)] + r
    return Regret




def plusieursXP(algo, bandit, horizon, N):
    Regrets = np.zeros((N, horizon))
    for exp in range(N):
        R = algo(bandit, horizon)
        Regrets[exp] = R
    Regret_moyen = np.mean(Regrets, axis=0)
    Regret_std = np.std(Regrets, axis=0)
    return Regret_moyen, Regret_std


if __name__ == "__main__":
    np.random.seed(12345)
    normal_bandit = bandits.NormalBandit(np.array ([0.1, 0.3, 0.4, 0.75, 0.8, 0.9, 0.95]))
    bernoulli_bandit = bandits.BernoulliBandit(np.array ([0.1, 0.3, 0.4, 0.75, 0.8, 0.9, 0.95]))
    EXP = 200

    r_ucb, std_ucb = plusieursXP(ucb, normal_bandit, HORIZON, EXP)
    plt.plot(r_ucb, label="UCB")
    plt.fill_between(np.arange(HORIZON), np.maximum(0, r_ucb - std_ucb), r_ucb + std_ucb, alpha=0.3)
    
    r_etc, std_etc = plusieursXP(etc, normal_bandit, HORIZON, EXP)
    plt.plot(r_etc, label="ETC")
    plt.fill_between(np.arange(HORIZON), np.maximum(0, r_etc- std_etc), r_etc + std_etc, alpha=0.3)

    r_rr, std_rr = plusieursXP(round_robin, normal_bandit, HORIZON, EXP)
    plt.plot(r_rr, label="Round Robin")
    plt.fill_between(np.arange(HORIZON), np.maximum(0, r_rr - std_rr), r_rr + std_rr, alpha=0.3)

    r_imed, std_imed = plusieursXP(imed, normal_bandit, HORIZON, EXP)
    plt.plot(r_imed, label="IMED")
    plt.fill_between(np.arange(HORIZON), np.maximum(0, r_imed - std_imed), r_imed + std_imed, alpha=0.3)
    

    plt.title(f"Regret - Horizon = {HORIZON}, Nombre d'XPs = {EXP}")
    plt.legend()
    plt.show()