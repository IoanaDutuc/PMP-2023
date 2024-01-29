import pandas as pd
import pymc as pm
import pandas as pd
import numpy as np

# Subiectul 1
# a
file_path = 'Titanic.csv'
df = pd.read_csv(file_path)

# calc restrictiile de baza
total_pasageri = len(df)
nr_supravietuitori = df['Survived'].sum()
procentaj_supravietuitori = (nr_supravietuitori / total_pasageri) * 100

print(f"nr total de pasageri: {total_pasageri}")
print(f"nr de supravietuitori: {nr_supravietuitori}")
print(f"procentajul de supravietuitori: {procentaj_supravietuitori:.2f}%")

# b
# eliminam val lipsa
df = df.dropna(subset=['Pclass', 'Age', 'Survived'])

# crearea var pt model
X = df[['Pclass', 'Age']]
y = df['Survived']

with pm.Model() as model:
    # intercept si coeficienti pt var independente
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    beta_pclass = pm.Normal('beta_pclass', mu=0, sigma=10)
    beta_age = pm.Normal('beta_age', mu=0, sigma=10)

    # calc log-odds-urilor pt modelul logistic
    log_odds = intercept + beta_pclass * X['Pclass'] + beta_age * X['Age']

    # def var Survived folosind o distrib Bernoulli
    obs = pm.Bernoulli('obs', logit_p=log_odds, observed=y)

    # esantionarea din model
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# c
# cea mai imporanta variabila, care poate sa tedetermine intr-un procent cat mai mare supravietuirea pasagerilor este
# sexul "Sex", adica daca sunt femei sau barbati  si apoi varsta "Age" ("prima data femeile si copiii")

# d
# val pt un pasager de 30 de ani din clasa a doua
age_val = 30
pclass_val = 2

with model:
    # calc log-odds pt un pasager de 30 de ani din clasa 2
    log_odds_30yo_pclass2 = intercept + beta_pclass * pclass_val + beta_age * age_val

    # calc probab de supravietuire (transformare logistica)
    prob_survival = pm.math.sigmoid(log_odds_30yo_pclass2)

    # asantionarea pt a obtine distrib probab de supravietuire
    post_pred = pm.sample_posterior_predictive(
        trace,
        var_names=[prob_survival],
        samples=1000,
        random_seed=42
    )

# calc intervalul HDI de 90%
hdi_interval = pm.hdi(post_pred[prob_survival.name], hdi_prob=0.90)

print(f"intervalul HDI de 90% pt supravietuire este: {hdi_interval}")



# Subiectul 2
# a
def generate_geometric(p, size):
    return np.random.geometric(p, size)

def monte_carlo_geometric(p_x, p_y, N):
    X = generate_geometric(p_x, N)
    Y = generate_geometric(p_y, N)
    conditie = X > Y**2
    return np.sum(conditie) / N

# fixam param
p_x = 0.3
p_y = 0.5
N = 10000

# aplicam metoda Monte Carlo
probab = monte_carlo_geometric(p_x, p_y, N)
print(f"P(X > Y^2) = {probab}")

# b
# repetam procesul de 30 de ori si calc media si dev standard
k = 30
rez = np.array([monte_carlo_geometric(p_x, p_y, N) for _ in range(k)])
mean_estimate = np.mean(rez)
std_dev = np.std(rez)

print(f"media aproximatiilor: {mean_estimate}")
print(f"deviatia standard: {std_dev}")
