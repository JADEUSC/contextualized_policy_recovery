# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

font = {'size'   : 18}
matplotlib.rc('font', **font)

# %%
df = pd.read_csv('simulation_mdp.csv')
# %%
plot_df = df[
    True \
    & (df['noise'] == 0) \
    & (df['time_lag'] == 2) \
    & (df['seq_length'] == 10) \
    & (df['N'] == 200) \
    # & (df['implicit_theta'] == True) \
]
print(plot_df.groupby(['implicit_theta']).agg({c: ['mean', 'std'] for c in ['pcc_theta', 'pcc_prob']}))
# %%
print(df.shape)
df.head()
# %%
target = 'N'
plot_df = df[
    True \
    & (df['noise'] == 0) \
    & (df['time_lag'] == 4) \
    & (df['seq_length'] == 15) \
    # & (df['N'] == 200) \
]
sns.catplot(
    x=target, 
    y='pcc_theta', 
    # y='pcc_prob', 
    hue='implicit_theta', 
    # col='seq_length', 
    # row='time_lag', 
    data=plot_df, 
    kind='point',
)
plt.ylim(0, 1)
plt.savefig(f'../figures/simulation_mdp_theta_{target}.pdf', bbox_inches='tight')
plt.show()

sns.catplot(
    x=target, 
    # y='pcc_theta', 
    y='pcc_prob', 
    hue='implicit_theta', 
    # col='seq_length', 
    # row='time_lag', 
    data=plot_df, 
    kind='point',
)
plt.ylim(0, 1)
plt.savefig(f'../figures/simulation_mdp_prob_{target}.pdf', bbox_inches='tight')
plt.show()
# %%
