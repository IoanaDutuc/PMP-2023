import numpy as np
import arviz as az
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [200, 150, 250]
means = [5, 0, 7]
std_devs = [2, 1.5, 1]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

az.plot_kde(np.array(mix))
plt.show()

data = mix.reshape(-1, 1)

n_components = [2, 3, 4]
gmms = [GaussianMixture(n_components=n, random_state=0).fit(data) for n in n_components]

for gmm in gmms:
    print(f'Model cu {gmm.n_components} componente:')
    print(f'  Log Likelihood: {gmm.score(data)}')
    print(f'  Medii: {gmm.means_.flatten()}')
    print(f'  Deviatii Standard: {np.sqrt(gmm.covariances_.flatten())}')
    print('---')
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.plot(x, pdf, '-k')
    plt.title(f'Model cu {gmm.n_components} componente')
    plt.show()