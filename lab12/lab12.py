# ex 1
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def posterior_retea(puncte_retea, cap, coada, prior_tip='uniform'):
    retea = np.linspace(0, 1, puncte_retea)
    if prior_tip == 'uniform':
        prior = np.repeat(1, puncte_retea)
    elif prior_tip == 'cond':
        prior = (retea <= 0.5).astype(int)
    elif prior_tip == 'abs':
        prior = abs(retea - 0.5)
    prob = stats.binom.pmf(cap, cap + coada, retea)
    posterior = prob * prior
    posterior /= posterior.sum()
    return retea, posterior

data = np.repeat([0, 1], (10, 3))
puncte = 250
h = data.sum()
t = len(data) - h


retea, posterior_uniform = posterior_retea(puncte, h, t, 'uniform')
retea, posterior_cond = posterior_retea(puncte, h, t, 'cond')
retea, posterior_abs = posterior_retea(puncte, h, t, 'abs')

plt.figure(figsize=(12, 9))
plt.plot(retea, posterior_uniform, '-o', label='uniform prior', color='pink')
plt.plot(retea, posterior_cond, '-o', label='conditional prior', color='purple')
plt.plot(retea, posterior_abs, '-o', label='absolute prior', color='orange')

plt.savefig('ex1.png')


# ex 2
import numpy as np
import matplotlib.pyplot as plt


def estimeaza_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    interior = (x ** 2 + y ** 2) <= 1
    pi = interior.sum() * 4 / N
    eroare = abs((pi - np.pi) / pi) * 100
    return pi, eroare


val_N = [270, 720, 872, 2870, 7208, 7820, 8270, 8720]
nr_simulari = 27

val_pi = []
val_eroare = []

for N in val_N:
    erori_pt_N = []
    for _ in range(nr_simulari):
        pi, eroare = estimeaza_pi(N)
        erori_pt_N.append(eroare)

    mean_eroare = np.mean(erori_pt_N)
    dev_std_eroare = np.std(erori_pt_N)

    val_pi.append(pi)
    val_eroare.append(mean_eroare)

plt.errorbar(val_N, val_eroare, yerr=dev_std_eroare, fmt='o-', label='eroare')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('eroare (%)')
plt.legend()

plt.savefig('ex2.png')


# ex 3
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def posterior(theta, y, N, a_prior, b_prior):
    prob = stats.binom.pmf(y, N, theta)
    prior = stats.beta.pdf(theta, a_prior, b_prior)
    return prob * prior


def metropolis(func, pasi=1000, init=0.5, a_prior=1, b_prior=1, y=0, N=1):
    probe = np.zeros(pasi)
    curent = init
    for i in range(pasi):
        propun = np.random.beta(a_prior, b_prior)
        p_acept = min(func(propun, y, N, a_prior, b_prior) / func(curent, y, N, a_prior, b_prior), 1)
        if np.random.rand() < p_acept:
            curent = propun
        probe[i] = curent
    return probe


n_incercari = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
theta_real = 0.35
beta_param = [(1, 1), (20, 20), (1, 4)]
x = np.linspace(0, 1, 200)

plt.figure(figsize=(12, 10))

for idx, N in enumerate(n_incercari):
    y = data[idx]
    for i, (a_prior, b_prior) in enumerate(beta_param):
        plt.subplot(len(n_incercari), len(beta_param), idx * len(beta_param) + i + 1)

        p_theta_given_y = stats.beta.pdf(x, a_prior + y, b_prior + N - y)
        plt.plot(x, p_theta_given_y, label='Grid Computing')

        probe = metropolis(posterior, pasi=1000, a_prior=a_prior, b_prior=b_prior, y=y, N=N)
        densitate_proba = stats.gaussian_kde(probe)
        plt.plot(x, densitate_proba(x), label='Metropolis', linestyle='--')

        plt.axvline(theta_real, ymax=0.3, color='k', linestyle=':')
        if idx == 0:
            plt.title(f'Priori: a={a_prior}, b={b_prior}')
        if i == 0:
            plt.ylabel(f'{N:4d} aruncari\n{y:4d} steme')
        plt.yticks([])
        plt.xlim(0, 1)

plt.xlabel('θ')
plt.legend()
plt.tight_layout()
plt.show()
