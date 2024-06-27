import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

font = {'size': 18}
matplotlib.rc('font', **font)

### Compile data into a single DF
# df = pd.read_csv('simulation_mdp.csv')
local_files = [
    '../results/CPR images data/simulation/local/simulation_N.csv',
    '../results/CPR images data/simulation/local/simulation_noise.csv',
    '../results/CPR images data/simulation/local/simulation_noisy_context.csv',
    '../results/CPR images data/simulation/local/simulation_seq_length.csv',
    '../results/CPR images data/simulation/local/simulation_time_lag.csv',
]
df = pd.read_csv(local_files[0])
for f in local_files[1:]:
    df = df.append(pd.read_csv(f))
global_files = [
    '../results/CPR images data/simulation/simulation_N.csv',
    '../results/CPR images data/simulation/simulation_noise.csv',
    '../results/CPR images data/simulation/simulation_noisy_context.csv',
    '../results/CPR images data/simulation/simulation_seq_length.csv',
    '../results/CPR images data/simulation/simulation_time_lag.csv',
]
global_df = pd.read_csv(global_files[0])
for f in global_files[1:]:
    global_df = global_df.append(pd.read_csv(f))
df = df[df['implicit_theta'] == False]
global_df['Model'] = None
global_df['Model'][global_df['implicit_theta'] == False] = 'CPR Global'
df['Model'] = None
df['Model'][df['implicit_theta'] == False] = 'CPR Local'
global_df['Model'][global_df['implicit_theta'] == True] = 'RNN'
df = pd.concat([df, global_df])
df.drop(columns=['implicit_theta'], inplace=True)

### Plot parameter scans one at a time, holding default values constant
default_vals = {
    'N': 200,
    'noise': 0,
    'context_noise': 0,
    'seq_length': 15,
    'time_lag': 4,
}

for scan in default_vals.keys():
    plot_df = df
    for fix in default_vals.keys():
        if fix == scan:  # Don't fix the scan variable, only fix non-scan defaults
            continue
        plot_df = plot_df[plot_df[fix] == default_vals[fix]]
    # plot_df = pd.DataFrame(plot_df.values, columns=df.columns)

    # Convert 'pcc_theta' and 'pcc_prob' to numeric types explicitly
    plot_df['pcc_theta'] = pd.to_numeric(plot_df['pcc_theta'], errors='coerce')
    plot_df['pcc_prob'] = pd.to_numeric(plot_df['pcc_prob'], errors='coerce')

    # Plot pcc Theta
    sns.catplot(
        x=scan,
        y='pcc_theta',
        hue='Model',
        data=plot_df,
        kind='point',
    )
    plt.ylim(0, 1)
    plt.savefig(f'../figures/icml/simulation_mdp_theta_{scan}.pdf', bbox_inches='tight')
    # plt.show()

    # Plot pcc Prob
    sns.catplot(
        x=scan,
        y='pcc_prob',
        hue='Model',
        data=plot_df,
        kind='point',
    )
    plt.ylim(0, 1)
    plt.savefig(f'../figures/icml/simulation_mdp_prob_{scan}.pdf', bbox_inches='tight')
    # plt.show()

