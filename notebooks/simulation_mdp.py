from src import simulations, trainer, models
import pandas as pd
import numpy as np


"""
Simulate over 
1. homogeneous vs. heterogeneous (time-dependent)
2. action-dependent vs. action-independent
3. signal-noise ratio
4. time lag for action/observation dependence

Evaluate on
AUROC, AUPRC, Brier, F1, and MSE, MAE, and PCC for true params theta and prob_a

Compare with vanilla RNN and LSTM
- theta as derivative w.r.t. input observations
Compare with linear model?
- context groups (for discrete multivariate contexts)
"""

def gen_data(time_lag, noise, N, seq_length):
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
            # x_t = np.random.uniform(-2, 2)
            # x.append(x_t)
            # theta_t = t / seq_length *  4 - 2
            # prob_t = 1 / (1 + np.exp(-theta_t * x_t + noise * np.random.normal()))
            # a_t = int(np.random.uniform(0,1) < prob_t)
            # a.append(a_t)
            # plot_row = [i, t, theta_t, x_t, prob_t, a_t]
            # plot_rows.append(plot_row)
            
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

            # Heterogeneous, observation and action dependent
            x_t = np.random.uniform(-2, 2)
            if len(x) < time_lag:
                a_t = np.random.choice([0,1])
                plot_row = [i, t, np.nan, x_t, np.nan, a_t]
            else:
                theta_t = x[-time_lag] * (a[-time_lag] * 2 - 1)  + t / seq_length
                prob_t = 1 / (1 + np.exp(-theta_t * x_t + noise * np.random.normal()))
                a_t = int(np.random.uniform(0,1) < prob_t)
                plot_row = [i, t, theta_t, x_t, prob_t, a_t]
            a.append(a_t)
            x.append(x_t)
            plot_rows.append(plot_row)

        X.extend(x)
        A.extend(a)
        I.extend([i] * seq_length)
        T.extend(list(range(seq_length)))

    sim_df = pd.DataFrame(plot_rows, columns=["i", "t", "theta", "x", "prob_a", "a"])
    df = pd.DataFrame({"x": X, "a": A, "i": I, "t": T})
    return sim_df


def run_model(exp_name, sim_df, seq_length, implicit_theta, hidden_dims=[16], lambdas=[0]):
    feature_cols = ["x"]
    loader_train, loader_val, loader_test = simulations.create_simulation_loader(sim_df, seq_length=seq_length)  # Changed to sim

    # Current observation size + 1
    input_size = 2
    # Previous action + observation size + 1
    context_size = 3

    trainer.train_contextual(exp_name=exp_name, input_size=input_size, context_size=context_size, train_loader=loader_train, 
                            val_loader=loader_val, hidden_dims=hidden_dims, lambdas=lambdas, lr=5e-2, implicit_theta=implicit_theta)

    best_context_l = trainer.get_best_run(exp=exp_name, pref="context_LSTM")
    best_context_l_model = trainer.load_run(run=best_context_l, dataset_name=exp_name, implicit_theta=implicit_theta)
    pred_context, true_context = models.model_predict(best_context_l_model, loader_test)
    auroc, auprc, brier, f1 = trainer.calculate_results(pred = pred_context, true=true_context)

    coef_df, enc = models.map_to_2d(model = best_context_l_model, loader_test=loader_test, feature_cols=feature_cols)
    coef_df.columns = ['i', 't'] + [c + "_model" for c in coef_df.columns[2:]]
    plot_df = sim_df.merge(coef_df, on=["i", "t"], how='inner')

    pcc_theta = plot_df[['theta', 'x_model']].corr().values[0, 1]
    mse_theta = np.mean((plot_df['theta'] - plot_df['x_model'])**2)
    mae_theta = np.mean(np.abs(plot_df['theta'] - plot_df['x_model']))
    pcc_prob = plot_df[['prob_a', 'prob_model']].corr().values[0, 1]
    mse_prob = np.mean((plot_df['prob_a'] - plot_df['prob_model'])**2)
    mae_prob = np.mean(np.abs(plot_df['prob_a'] - plot_df['prob_model']))
    return [pcc_theta, mse_theta, mae_theta, pcc_prob, mse_prob, mae_prob, auroc, auprc, brier, f1]


if __name__ == '__main__':
    arg_cols = ["time_lag", "noise", "N", "seq_length", "run", "implicit_theta"]
    metric_cols = ["pcc_theta", "mse_theta", "mae_theta", "pcc_prob", "mse_prob", "mae_prob", "auroc", "auprc", "brier", "f1"]
    result_rows = []
    for time_lag in [1, 2, 4]:
        for noise in [0.0, 1e-2, 1e-1, 1.0]:
            for N in [20, 200, 2000]:
                for seq_length in [5, 10, 15]:
                    for run in range(3):
                        sim_df = gen_data(time_lag=time_lag, noise=noise, N=N, seq_length=seq_length)
                        for implicit_theta in [False, True]:
                            args = [time_lag, noise, N, seq_length, run, implicit_theta]
                            metrics = run_model(f"Z_simulation_mdp_{args}", sim_df, seq_length=seq_length, implicit_theta=implicit_theta)
                            result_rows.append(args + metrics)
                            pd.DataFrame(result_rows, columns=arg_cols + metric_cols).to_csv('simulation_mdp.csv', index=False)
    print('finished successfully <:)')
