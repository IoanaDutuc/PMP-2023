import pymc3 as pm
import arviz as az

Y_values = [0, 5, 10]
teta_values = [0.2, 0.5]

def build_bayesian_model(Y, teta):
    with pm.Model() as model:
        n = pm.Poisson('n', mu=10)
        Y_observed = pm.Binomial('Y_observed', n=n, p=teta, observed=Y)
    return model
trace_list = []

for Y in Y_values:
    for teta in teta_values:
        model = build_bayesian_model(Y, teta)
        with model:
            trace = pm.sample(1000, tune=1000, cores=1)
        trace_list.append(trace)

combined_trace = az.concat(trace_list, dim='chain')
az.plot_posterior(combined_trace, var_names=['n'], filter_vars=['n'], rope=[9.5, 10.5])
