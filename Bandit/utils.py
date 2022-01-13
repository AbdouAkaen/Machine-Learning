import numpy as np
import matplotlib.pyplot as plt


########################################
#              Plot Bandit             #
########################################
def plot_bandit(bandit):
    plt.figure(figsize=(12, 7))
    plt.xticks(range(0, bandit.nbr_arms), range(0, bandit.nbr_arms))
    plt.plot(range(0, bandit.nbr_arms), bandit.rewards)
    plt.scatter(range(0, bandit.nbr_arms), bandit.rewards)
    plt.show()


########################################
#           KL Divergences             #
########################################
def klBernoulli(mean_1, mean_2, eps=1e-15):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = np.minimum(np.maximum(mean_1, eps), 1 - eps)
    y = np.minimum(np.maximum(mean_2, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def klGaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1 - mean_2) ** 2) / (2 * sig2)


########################################
#         Argument selectors           #
########################################

def randamax(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amax(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amax(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx


def randamin(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amin(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amin(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx
