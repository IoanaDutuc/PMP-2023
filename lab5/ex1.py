import pymc3 as pm
import pandas

csv_data = pandas.read_csv("trafic.csv")

masini = csv_data["nr. masini"].values
minute = csv_data["minut"].values

inter = [(0, 180), (180, 240), (240, 720), (720, 900), (900, 1200)]
with pm.Model() as model:
    lambd = pm.Normal("lambda", mu=0, sigma=10)
    trafic = pm.Poisson('traffic', mu=lambd, observed=csv_data)
    intervale_distr = list()
    #ora 7 = min 180
    #ora 8 = min 240
    #ora 16 = min 720
    #ora 19 = min 900
    intervale_distr.append(pm.Poisson(f'lambda_1', mu=lambd, observed=masini[(minute >= 0) & (minute < 180)]))
    intervale_distr.append(pm.Poisson(f'lambda_2', mu=lambd * 1.3, observed=masini[(minute >= 180) & (minute < 240)]))
    intervale_distr.append(pm.Poisson(f'lambda_3', mu=lambd * 0.4, observed=masini[(minute >= 240) & (minute < 720)]))
    intervale_distr.append(pm.Poisson(f'lambda_4', mu=lambd * 1.5, observed=masini[(minute >= 720) & (minute < 900)]))
    intervale_distr.append(pm.Poisson(f'lambda_5', mu=lambd * 0.5, observed=masini[(minute >= 900) & (minute < 1200)]))

with model:
    trace = pm.sample(2000, tune=5000)

pm.plot_trace(trace)