import random
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import matplotlib.pyplot as plt


# fct moneda primeste o prob (0,5 daca nu e masluita) si ne da rezultatul aruncarii(cap sau pajura)
def moneda(probab):
    return random.choices([0, 1], weights=[1 - probab, probab])[0]

# fct sim_joc simuleaza jocul
def sim_joc():
    j_crt = moneda(0.5)    # aruncarea primei monezi care decide jucatorul
    if j_crt == 0:
        n = moneda(0.5)   # primul jucator arunca moneda (onest)
        j_crt = 0    # se schimba jucatorul curent
    else:
        n = moneda(2/3)   # j1 arunca moneda masluita
        j_crt = 0  # se schimba jucatorul curent
    if j_crt == 0:
        m = sum(moneda(2/3) for _ in range(n + 1))  #punctajul obtinut de j0 in runda 2 (onest)
    else:
        m = sum(moneda(0.5) for _ in range(n + 1))  #punctajul obtinut de j1 in runda 2 (masluit)
    castigator = 0 if n >= m else 1
    return castigator

nr_jocuri = 10000
castig_j0 = 0
castig_j1 = 0

for _ in range(nr_jocuri):   #simulam 10000 de jocuri
    castigator = sim_joc()
    if castigator == 0:
        castig_j0 += 1
    else:
        castig_j1 += 1

procent_j0 = (castig_j0 / nr_jocuri) * 100   #calc proncentu de castig a jucatorului j0
procent_j1 = (castig_j1 / nr_jocuri) * 100   #calc proncentu de castig a jucatorului j1
# afisam probab de castig ale jucatorilor
print(f"jucatorul j0 are probabilitatea de castig {procent_j0}%.")
print(f"jucatorul j1 are probabilitatea de castig {procent_j1}%.")

# ex1.2
#definim modelul retelei bayesiene
model = BayesianModel([('JucatorCurent', 'N'),
                       ('JucatorCurent', 'M'),
                       ('N', 'Castigator'),
                       ('M', 'Castigator')])
# in retea adaugam probab conditionate
model.fit([{'JucatorCurent': 0, 'N': 0, 'M': 0, 'Castigator': 0},
           {'JucatorCurent': 1, 'N': 0, 'M': 0, 'Castigator': 1},
           {'JucatorCurent': 0, 'N': 1, 'M': 1, 'Castigator': 1},
           {'JucatorCurent': 1, 'N': 1, 'M': 1, 'Castigator': 0}],
          estimator=ParameterEstimator)
# calc inferenta din retea
inference = VariableElimination(model)
# calc probab marginale
prob_castig_j0 = inference.query(variables=['Castigator'], evidence={'JucatorCurent': 0})['Castigator'].values[0]
prob_castig_j1 = inference.query(variables=['Castigator'], evidence={'JucatorCurent': 1})['Castigator'].values[0]

# afisam probab marginale de castig pt fiecare jucator
print(f"probabilitatea ca j0 sa castige: {prob_castig_j0:.4f}")
print(f"probabilitatea ca j1 sa castige: {prob_castig_j1:.4f}")




# ex1.3
# calc inferenta din retea
inference = VariableElimination(model)

# calc probab cond date de observatia "M=1"; cel mai probab jucator este cel pt care probab conditionata este mai mare
prob_jucator_crt_0 = inference.query(variables=['JucatorCurent'], evidence={'M': 1})['JucatorCurent'].values[0]
prob_jucator_crt_1 = inference.query(variables=['JucatorCurent'], evidence={'M': 1})['JucatorCurent'].values[1]

# afisam probab de a fi ales pt fiecare jucator
print(f"probabilitatea ca j0 sa fi inceput jocul: {prob_jucator_crt_0:.4f}")
print(f"probabilitatea ca j1 sa fi inceput jocul: {prob_jucator_crt_1:.4f}")

# alegem jucatorul cu probab cea mai mare
jucator_probab = 0 if prob_jucator_crt_0 > prob_jucator_crt_1 else 1

# afisam jucatorul cel mai probabil de a fi inceput
print(f"cel care a inceput jocul cel mai probabil esye j{jucator_probab}")




# subiectul2
# ex2.1
# alegem param distrib normale
mu = 18     # medie
theta = 3    # dev standard
# gen 100 de timpi medii de asteptare pe baza distrib normale
t_med_astept = np.random.normal(loc=mu, scale=theta, size=100)

# afisam histograma pt distrib generata
plt.hist(t_med_astept, bins=20, density=True, alpha=0.7, color='pink', edgecolor='black')
plt.title('histograma pentru timpii medii de asteptare')
plt.xlabel('timp mediu de asteptare')
plt.ylabel('frecventa relativa')
plt.show()


# ex2.2
# generam 100 de timpi medii de asteptare
np.random.seed(78)
t_med_astept_obs = np.random.normal(loc=18, scale=3, size=100)

# construim modelul pymc3
model = pm.Model()
with model:
    # distrib a priori pt medie
    mu = pm.Normal('mu', mu=18, sd=3)
    # distrib a priori pt dev standard (pozitiva)
    theta = pm.HalfNormal('theta', sd=3)
    # distrib a probab
    t_med_astept_est = pm.Normal('t_med_astept_est', mu=mu, sd=theta, observed=t_med_astept_obs)



# ex3.3
#antrenam modelul cu mcmc
with model:
    trace = pm.sample(1000, tune=1000, random_seed=78)

# afisam graficul distrib a posteriori pt mu
az.plot_posterior(trace['mu'], hdi_prob=0.95, round_to=2, point_estimate='mean', kind='hist')
plt.title('distrib a posteriori pt μ')
plt.xlabel('μ')
plt.ylabel('densitatea de probab')
plt.show()
