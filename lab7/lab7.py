import pandas as pd
import matplotlib.pyplot as plt

# citim datele din fisier si afisam primele 5 randuri ca sa ved cum arata (cum sunt dtructurate)
df = pd.read_csv("auto-mpg.csv")
print(df.head())

# facem curatarea datelor (adica stergem randurile care nu au valori pt coloanele care ne intereseaza si cele cu "?")
df = df.dropna(subset=['mpg', 'horsepower'])
df.replace("?", pd.NA, inplace=True)
df = df.dropna()

# trecem datele in tip numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.title('relatia dintre cai putere si consumul de carburant')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile per Galon (mpg)')
plt.show()


# b) definirea modelului in pymc
import pymc3 as pm
import theano.tensor as tt

with pm.Model() as linear_model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + beta * df['horsepower']
    mpg = pm.Normal('mpg', mu=mu, sd=1, observed=df['mpg'])
    trace = pm.sample(2000, tune=1000, cores=2)


# c) determinarea dreptei de regresie
alpha_m = trace['alpha'].mean()
beta_m = trace['beta'].mean()

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.plot(df['horsepower'], alpha_m + beta_m * df['horsepower'], c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.title('Dreapta de Regresie')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile per Galon (mpg)')
plt.legend()
plt.show()


# d) adaugarea hdpi la grafic
import arviz as az

ppc = pm.sample_posterior_predictive(trace, samples=2000, model=linear_model)

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.plot(df['horsepower'], alpha_m + beta_m * df['horsepower'], c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(df['horsepower'].values, ppc['mpg'], hdi_prob=0.95, color='gray', smooth=True, fill_kwargs={'alpha': 0.2})
plt.title('Dreapta de Regresie cu 95% HPDI')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile per Galon (mpg)')
plt.legend()
plt.show()
