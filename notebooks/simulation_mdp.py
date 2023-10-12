# %%
from src import simulations, preprocess, data, trainer, models
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib

import torch.nn as nn
import torch 

font = {'size'   : 14}

matplotlib.rc('font', **font)
# %%

"""
Simulate over 
1. homogeneous vs. heterogeneous (time-dependent)
2. action-dependent vs. action-independent
3. signal-noise ratio
4. time lag for action/observation dependence

Evaluate on
AUROC, AUPRC, Brier, F1, MSE prob a, MSE theta, MSE intercept

Compare with vanilla RNN and LSTM
- MSE theta as derivative w.r.t. input observations
Compare with linear model
- context groups (for multivariate contexts)
"""

# Homogeneous MDP, observation-dependent with time lag, no action dependence
time_lag = 2
noise = 0.0
N = 200
seq_length = 9
X = []
A = []
T = []
I = []
plot_rows = []
for i in range(N):
    x = []
    a = []
    for t in range(seq_length):

        # Time-varying, observation-independent
        x_t = np.random.uniform(-2, 2)
        x.append(x_t)
        theta_t = t / seq_length *  4 - 2
        prob_t = 1 / (1 + np.exp(-theta_t * x_t + noise * np.random.normal()))
        a_t = int(np.random.uniform(0,1) < prob_t)
        a.append(a_t)
        plot_row = [i, t, theta_t, x_t, prob_t, a_t]
        plot_rows.append(plot_row)

        # Homogeneous, observation-dependent
        # x_t = np.random.uniform(-2, 2)
        # x.append(x_t)
        # if len(x) < time_lag + 1:
        #     a_t = np.random.choice([0,1])
        #     plot_row = [i, t, np.nan, x_t, np.nan, a_t]
        # else:
        #     theta_t = x[-time_lag - 1]
        #     prob_t = 1 / (1 + np.exp(-theta_t * x_t + noise * np.random.normal()))
        #     a_t = int(np.random.uniform(0,1) < prob_t)
        #     plot_row = [i, t, theta_t, x_t, prob_t, a_t]
        # a.append(a_t)
        # plot_rows.append(plot_row)
    X.extend(x)
    A.extend(a)
    I.extend([i] * seq_length)
    T.extend(list(range(seq_length)))

sim_df = pd.DataFrame(plot_rows, columns=["i", "t", "theta", "x", "prob_a", "a"])
df = pd.DataFrame({"x": X, "a": A, "i": I, "t": T})

feature_cols = ["x"]
loader_train, loader_val, loader_test = simulations.create_simulation_loader(df, seq_length=seq_length)

hidden_dims = [64]
lambdas = [0]

# Current observation size + 1
input_size = 2
# Previous action + observation size + 1
context_size = 3

exp_name = "Simulation_homogeneous_mdp"

trainer.train_contextual(exp_name=exp_name, input_size=input_size, context_size=context_size, train_loader=loader_train, 
                         val_loader=loader_val, hidden_dims=hidden_dims, lambdas=lambdas, lr=5e-2)

best_context_l = trainer.get_best_run(exp=exp_name, pref="context_LSTM")
best_context_l_model = trainer.load_run(run=best_context_l, dataset_name=exp_name)
pred_context, true_context = models.model_predict(best_context_l_model, loader_test)
print(trainer.calculate_results(pred = pred_context, true=true_context))

# %%
coef_df, enc = models.map_to_2d(model = best_context_l_model, loader_test=loader_test, feature_cols=feature_cols)
coef_df.columns = ['i', 't'] + [c + "_model" for c in coef_df.columns[2:]]
plot_df = sim_df.merge(coef_df, on=["i", "t"], how='inner')
plot_df.head()

# %%
print('thetas')
print(plot_df[['theta', 'x_model']].corr())
print('probs')
print(plot_df[['prob_a', 'prob_model']].corr())
sns.scatterplot(data=plot_df, x="theta", y="x_model")
plt.show()
sns.scatterplot(data=plot_df, x="prob_a", y="prob_model")
plt.show()

# Plot correlation of predicted vs true with a fit line
# sns.catplot(data=plot_df, x="a", y="prob_model", kind="swarm", col='t')
# plt.show()
# %%

exp_name = "Simulation_homogeneous_mdp_implicit"

trainer.train_contextual(exp_name=exp_name, input_size=input_size, context_size=context_size, train_loader=loader_train, 
                         val_loader=loader_val, hidden_dims=hidden_dims, lambdas=lambdas, lr=5e-2, implicit_theta=True)

best_context_l = trainer.get_best_run(exp=exp_name, pref="context_LSTM")
best_context_l_model = trainer.load_run(run=best_context_l, dataset_name=exp_name, implicit_theta=True)
pred_context, true_context = models.model_predict(best_context_l_model, loader_test)
print(trainer.calculate_results(pred = pred_context, true=true_context))

# %%
coef_df, enc = models.map_to_2d(model = best_context_l_model, loader_test=loader_test, feature_cols=feature_cols)
coef_df.columns = ['i', 't'] + [c + "_model" for c in coef_df.columns[2:]]
plot_df = sim_df.merge(coef_df, on=["i", "t"], how='inner')
plot_df.head()

# %%
print('thetas')
print(plot_df[['theta', 'x_model']].corr())
print('probs')
print(plot_df[['prob_a', 'prob_model']].corr())
sns.scatterplot(data=plot_df, x="theta", y="x_model")
plt.show()
sns.scatterplot(data=plot_df, x="prob_a", y="prob_model")
plt.show()
# %%

# # %%
# N = 200
# seq_length = 9
# X = []
# A = []
# T = []
# I = []
# plot_rows = []
# for i in range(N):
#     x = []
#     a = []
#     for t in range(seq_length):
#         x_t = np.random.uniform(-1, 1)
#         x.append(x_t)
#         if len(x) < 2:
#             a_t = np.random.choice([0,1])
#             plot_row = [i, t, np.nan, x_t, a_t]
#         else:
#             a_t = int(np.sign(x[-1]) == np.sign(x[-2]))
#             plot_row = [i, t, x[-2], x_t, a_t]
#         a.append(a_t)
#         plot_rows.append(plot_row)
#     X.extend(x)
#     A.extend(a)
#     I.extend([i] * seq_length)
#     T.extend(list(range(seq_length)))

# plot_df = pd.DataFrame(plot_rows, columns=["i", "t", "x-1", "x", "a"])
# df = pd.DataFrame({"x": X, "a": A, "i": I, "t": T})

# feature_cols = ["x"]
# loader_train, loader_val, loader_test = simulations.create_simulation_loader(df, seq_length=seq_length)

# hidden_dims = [16]
# lambdas = [0]

# input_size = 2
# context_size = 3

# exp_name = "Simulation_homogeneous_mdp"

# trainer.train_contextual(exp_name=exp_name, input_size=input_size, context_size=context_size, train_loader=loader_train, 
#                          val_loader=loader_val, hidden_dims=hidden_dims, lambdas=lambdas, lr=5e-2)

# best_context_l = trainer.get_best_run(exp=exp_name, pref="context_LSTM")
# best_context_l_model = trainer.load_run(run=best_context_l, dataset_name=exp_name)
# pred_context, true_context = models.model_predict(best_context_l_model, loader_test)
# trainer.calculate_results(pred = pred_context, true=true_context)

# # %%
# coef_df, enc = models.map_to_2d(model = best_context_l_model, loader_test=loader_test, feature_cols=feature_cols)
# coef_df.columns = ['i', 't'] + [c + "_model" for c in coef_df.columns[2:]]
# plot_df = plot_df.merge(coef_df, on=["i", "t"], how='inner')

# sns.scatterplot(data=plot_df, x="x-1", y="x", hue="intercept_model")
# plt.show()

# # Plot correlation of predicted vs true with a fit line
# sns.catplot(data=plot_df, x="a", y="prob_model", kind="swarm", col='t')
# plt.show()
