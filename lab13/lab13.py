import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # ex1
    centered_eight = az.load_arviz_data('centered_eight')
    non_centered_eight = az.load_arviz_data('non_centered_eight')

    nr_lanturi_centrate = centered_eight.posterior.chain.size
    nr_total_mostre_centrate = centered_eight.posterior.draw.size
    print(f"Nr lanturi pt modelul centrat: {nr_lanturi_centrate}")
    print(f"Nr de esantioane pt modelul centrat: {nr_total_mostre_centrate}")

    nr_lanturi_necentrate = non_centered_eight.posterior.chain.size
    nr_total_mostre_necentrate = non_centered_eight.posterior.draw.size
    print(f"Nr de lanturi pt modelul necentrat: {nr_lanturi_necentrate}")
    print(f"Nr de esantioane pt modelul necentrat: {nr_total_mostre_necentrate}")

    az.plot_posterior(centered_eight)
    az.plot_posterior(non_centered_eight)
    plt.show()

    # ex2
    rhat_centrat = az.rhat(centered_eight, var_names=["mu", "tau"])
    rhat_necentrat = az.rhat(non_centered_eight, var_names=["mu", "tau"])

    print("Rhat pt modelul centrat (mu si tau):", rhat_centrat)
    print("Rhat pt modelul necentrat (mu si tau):", rhat_necentrat)

    # afisarea Rhat pt fiecare model
    print(f"Rhat pt cele 2 modele, in functie de parametrul mu")
    rez = pd.concat([az.summary(centered_eight, var_names=['mu']), az.summary(non_centered_eight, var_names=['mu'])])
    rez.index = ['centered', 'non_centered']
    print(rez)
    print(f"Rhat pt cele 2 modele, in functie de parametrul tau")
    rez = pd.concat([az.summary(centered_eight, var_names=['tau']), az.summary(non_centered_eight, var_names=['tau'])])
    rez.index = ['centered', 'non_centered']
    print(rez)
    print("Versiunea ArviZ instalată:", az.__version__)

    # autocorelatia
    autocor_centrat_mu = az.autocorr(centered_eight.posterior["mu"].values)
    autocor_centrat_tau = az.autocorr(centered_eight.posterior["tau"].values)
    autocor_necentrat_mu = az.autocorr(non_centered_eight.posterior["mu"].values)
    autocor_necentrat_tau = az.autocorr(non_centered_eight.posterior["tau"].values)

    print(f"Autocorelatia pt modelul centrat, in functie de parametrul mu : {np.mean(autocor_centrat_mu)}")
    print(f"Autocorelatia pt modelul centrat, in functie de parametrul tau : {np.mean(autocor_centrat_tau)}")
    print(f"Autocorelatia pt modelul necentrat, in functie de parametrul mu : {np.mean(autocor_necentrat_mu)}")
    print(f"Autocorelatia pt modelul necentrat, in functie de parametrul tau : {np.mean(autocor_necentrat_tau)}")

    az.plot_autocorr(centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 10))
    az.plot_autocorr(non_centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 10))
    plt.show()

    # ex3
    divergente_centrate = centered_eight.sample_stats["diverging"].sum()
    divergente_necentrate = non_centered_eight.sample_stats["diverging"].sum()

    print()
    print(f"Nr de divergente pt modelul centrat: {divergente_centrate.values}")
    print(f"Nr de divergente pt modelul non-centrat: {divergente_necentrate.values}")

    az.plot_pair(centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Modelul centrat")
    plt.show()

    az.plot_pair(non_centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Modelul necentrat")
    plt.show()

    az.plot_parallel(centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Plot Paralel - Modelul centrat")
    plt.show()

    az.plot_parallel(non_centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Plot Paralel - Modelul necentrat")
    plt.show()


if __name__ == '__main__':
    main()
