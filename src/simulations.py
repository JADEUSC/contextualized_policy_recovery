import random
import numpy as np
import pandas as pd
from src.preprocess.util import split_df
from src import preprocess, data

from torch.utils.data import DataLoader

def sequence(n:int=25, p:float=0, lower_bound:int=-1, upper_bound:int=1):
    x = []
    x.append(random.uniform(lower_bound, upper_bound))

    for i in range(1, n):
        x.append(x[i-1]*p + random.uniform(lower_bound,upper_bound)*(1-p))
    
    return x


def sig_theta(x, theta, intercept=0):
    return 1/(1+np.exp(-(theta * x + intercept)))


def agent_p_a(x):
    thetas = [0]
    a = [1 if random.uniform(0,1) < sig_theta(x[0], theta=thetas[0]) else 0]

    for i in range(1,len(x)):
        theta = x[i-1]
        thetas.append(theta)

        p_a = sig_theta(4 * x[i], theta=theta)
        a.append(1 if random.uniform(0,1) < p_a else 0)
    
    return a, thetas


def agent_p_b(x):
    thetas = [0]
    intercepts = [-1, -3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4, 1]
    a = [1 if random.uniform(0,1) < sig_theta(x[0], theta=thetas[0], intercept=intercepts[0]) else 0]

    for i in range(1,len(x)):
        theta = x[i-1]
        thetas.append(theta)
        intercept = intercepts[i]

        p_a = sig_theta(x[i], theta=4 * theta, intercept=intercept)
        a.append(1 if random.uniform(0,1) < p_a else 0)
    
    return a, thetas, intercepts


def agent_p_b(x):
    thetas = [0]
    intercepts = [-1, -3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4, 1]
    a = [1 if random.uniform(0,1) < sig_theta(x[0], theta=thetas[0], intercept=intercepts[0]) else 0]

    for i in range(1,len(x)):
        theta = x[i-1]
        thetas.append(theta)
        intercept = intercepts[i]
        p_a = sig_theta(x[i], theta=4 * theta, intercept=intercept)
        a.append(1 if random.uniform(0,1) < p_a else 0)
    
    return a, thetas, intercepts


def agent_d(x):
    a = [0]
    c = [2]

    for t in range(1, len(x)):
        if x[t-1] > 0.5:
            if x[t] > 0.5:
                a.append(1)
            else:
                a.append(0)
            c.append(1)
            
        else:
            if x[t] < 0.5:
                a.append(1)
            else:
                a.append(0)
            c.append(0)
    return a, c


def create_simulation_loader(df, seq_length, add_intercept=True):
    
    train, val, test = preprocess.util.split_df(df, split_col="i")

    feature_cols = ["x"]

    ds_train = data.Context_df_dataset(train, feature_cols=feature_cols, target_col="a", identifier_col="i", 
                                    max_length=seq_length, add_intercept_yn=add_intercept)
    ds_test = data.Context_df_dataset(test, feature_cols=feature_cols, target_col="a", identifier_col="i", 
                                    max_length=seq_length, add_intercept_yn=add_intercept)
    ds_val = data.Context_df_dataset(val, feature_cols=feature_cols, target_col="a", identifier_col="i", 
                                    max_length=seq_length, add_intercept_yn=add_intercept)

    loader_train = DataLoader(ds_train, shuffle=True, batch_size=64)
    loader_val = DataLoader(ds_val, shuffle=True, batch_size=64)
    loader_test = DataLoader(ds_test, shuffle=False, batch_size=1) 

    return loader_train, loader_val, loader_test
