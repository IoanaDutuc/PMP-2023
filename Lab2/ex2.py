import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon

alpha = [4, 4, 5, 5]
lambda_values = [3, 2, 2, 3]

prob = [0.25, 0.25, 0.30, 0.20]

num_samples = 100000

timp_serv = np.zeros(num_samples)
for i in range(len(alpha)):
    is_server_i = np.random.rand(num_samples) < prob[i]
    timp_serv[is_server_i] += gamma.rvs(a=alpha[i], scale=1/lambda_values[i], size=np.sum(is_server_i))

latenta = expon(scale=1/4).rvs(num_samples)
timp_total = timp_serv + latenta

prob_peste_3ms = np.mean(timp_total > 3)

print("Probabilitate timp > 3ms:", prob_peste_3ms)

plt.figure(figsize=(8, 6))
plt.hist(timp_total, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('Timp servire')
plt.ylabel('Densitate')
plt.show()
