import pymc3 as pm
import arviz as az
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [200, 150, 250]
means = [5, 0, 7]
std_devs = [2, 1.5, 1]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
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

for n in n_components:
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(n))

        components = []
        for i in range(n):
            component = pm.Normal(f'component_{i}', mu=pm.Normal(f'mu_{i}', mu=0, sd=10),
                                  sd=pm.HalfNormal(f'sd_{i}', sd=10))
            components.append(component)

        like = pm.Mixture('like', w=w, comp_dists=components, observed=data)

        trace = pm.sample(1000, tune=2000, return_inferencedata=True)

    waic = az.waic(trace, model)
    loo = az.loo(trace, model)

    print(f'Model cu {n} componente:')
    print(f'  WAIC: {waic.waic}')
    print(f'  LOO: {loo.loo}')


# Concluzie:
# Pentru a alege intre modelele Gaussian Mixture Model (GMM) cu 2, 3 si 4 componente, folosim criteriile WAIC si LOO.
# Modelul cu cele mai mici valori WAIC si LOO este considerat cel mai adecvat, echilibrand eficient capacitatea de
# predictie cu complexitatea. Daca diferentele de scoruri sunt mici, un model mai simplu (cu mai putine componente)
# poate fi preferabil pentru a evita complexitatea inutila. Alegerea finala a modelului ar trebui sa ia in considerare
# si contextul specific al datelor si nevoile de analiza.
