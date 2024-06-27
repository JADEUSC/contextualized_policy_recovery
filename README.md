# Contextualized Policy Recovery

This repository contains the code for replicating *Contextualized Policy Recovery*.

## Installation

```bash
cd contextualized_policy_recovery
conda env create -f environment.yml
conda activate cpr
pip install --no-build-isolation --disable-pip-version-check -e .
```

## Experiments
All other experiments to create the relevant figures can be found in `notebooks/MIMIC_figures.py` and `notebooks/ADNI_figures.py` 

The code to run the bootstrap runs can be found in `src/ADNI_bootstrap.py` and `src/MIMIC_bootstrap.py` and is used to create `ADNI_results.txt` and `MIMIC_results.txt`. 

The code to run the simulations can be found in `notebooks/simulation_mdp.py` and the code to plot the figure is `notebooks/plot.py`. Choose `alpha=1` for using local information only and `1>alpha>0` for using global information.

## Data 
1) ADNI: After getting Data access from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/), place the `ADNIMERGE.csv` file in the `data/adni/raw/` folder.
2) MIMIC: After getting Data access for the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/), follow the instruction for [installing it in a Postgres database](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/).
