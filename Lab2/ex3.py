import numpy as np
import matplotlib.pyplot as plt

num_arunc = 10
num_sim = 100

prob_stema = 0.3

rez = np.random.choice(['s', 'b'], size=(num_sim, num_arunc), p=[1 - prob_stema, prob_stema])

num_ss = np.sum(np.all(rez == 's', axis=1))
num_sb = np.sum(np.sum(rez == 's', axis=1) == num_arunc - 1)
num_bs = np.sum(np.sum(rez == 's', axis=1) == 1)
num_bb = np.sum(np.all(rez == 'b', axis=1))

print("Nr rezultate ss:", num_ss)
print("Nr rezultate sb:", num_sb)
print("Nr rezultate bs:", num_bs)
print("Nr rezultate bb:", num_bb)

rez_posib = ['ss', 'sb', 'bs', 'bb']
nr_rez = [num_ss, num_sb, num_bs, num_bb]

plt.figure(figsize=(8, 6))
plt.bar(rez_posib, nr_rez, color='blue')
plt.xlabel('Rezultate posibile')
plt.ylabel('Nr apariții')
plt.show()

