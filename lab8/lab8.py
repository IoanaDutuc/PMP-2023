import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def citire_fisier():
    cale = 'Prices.csv'
    df = pd.read_csv(cale)
    preturi = df['Price']
    viteza = df['Speed']
    memorie = np.log(df['HardDrive'])
    return preturi, viteza, memorie


def main():
    preturi, viteza, memorie = citire_fisier()
    with pm.Model() as model_regres:
        a = pm.Normal('a', mu=0, sigma=10)
        b1 = pm.Normal('b1', mu=0, sigma=1)
        b2 = pm.Normal('b2', mu=0, sigma=1)
        sigma = pm.HalfCauchy('sigma', beta=5)
        mu = a + b1 * viteza + b2 * memorie
        pret_obs = pm.Normal('pret_obs', mu=mu, sigma=sigma, observed=preturi)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)
    b1_hdi = az.hdi(idata['posterior']['b1'], prob_hdi=0.95)
    b2_hdi = az.hdi(idata['posterior']['b2'], prob_hdi=0.95)

    print("95% hdi pt b1:", b1_hdi)
    print("95% hdi pt b2:", b2_hdi)
    
    mu_prezis = pm.Deterministic('mu_prezis', a + b1 * 33 + b2 * np.log(540))


if __name__ == '__main__':
    main()