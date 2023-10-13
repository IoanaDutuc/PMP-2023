import numpy as np
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt

lambda1 = 4
lambda2 = 6

clienti = 10000

timp_serv_total = []

for _ in range(clienti):
    mecanic1 = np.random.choice([1, 2], p=[0.4, 0.6])
    if mecanic1 == 1:
        timp_serv = stats.expon(scale=1/lambda1).rvs()
    else:
        timp_serv = stats.expon(scale=1/lambda2).rvs()

    timp_serv_total.append(timp_serv)


med_timp_serv = np.mean(timp_serv_total)
dev_stand_timp_serv = np.std(timp_serv_total)


print("Media timpului de servire ", med_timp_serv)
print("Deviatia standard a timpului de servire ", dev_stand_timp_serv)

az.plot_kde(np.array(timp_serv_total))
plt.show()