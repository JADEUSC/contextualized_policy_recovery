from torch import nn
import torch

from typing import Tuple
import numpy as np

import pandas as pd

import umap

class contextualized_sigmoid(nn.Module):
    def __init__(self, hidden_dim=16, n_layers=1, context_size=6, input_size=6, type="RNN", implicit_theta=False):
        super(contextualized_sigmoid, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.context_size = context_size
        self.input_size = input_size
        self.type = type
        self.implicit_theta = implicit_theta

        if not self.implicit_theta:
            if self.type == "RNN":
                self.rnn = nn.RNN(self.context_size, self.hidden_dim, self.n_layers, batch_first=True)
            elif self.type=="LSTM":
                self.rnn = nn.LSTM(self.context_size, self.hidden_dim, self.n_layers, batch_first=True)
            else:
                raise ValueError
            self.fc = nn.Linear(self.hidden_dim, self.input_size+1)  # generate theta and beta
        else:
            if self.type == "RNN":
                self.rnn = nn.RNN(self.context_size + self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
            elif self.type=="LSTM":
                self.rnn = nn.LSTM(self.context_size + self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
            else:
                raise ValueError
            self.fc = nn.Linear(self.hidden_dim, 1)  # generate prob
            self.offset = torch.zeros(size=(1,))
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size, static=None):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.type == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        if self.type == "LSTM":
            hidden, cell = hidden

        # Encode static features in initial hidden state
        if static is not None:
            hidden = hidden[:,:, :self.hidden_dim-static.shape[-1]]
            static = torch.unsqueeze(static, 0)
            hidden = torch.cat([static, hidden], dim=-1)

        if self.type == "LSTM":
            hidden = (hidden, cell)

        return hidden
    
    def get_theta(self, context, observation, hidden):
        if self.implicit_theta:
            dummy_context = context.clone()
            dummy_obs = observation.clone()
            dummy_context.requires_grad = True
            dummy_obs.requires_grad = True
            contextobs = torch.cat([dummy_context, dummy_obs], dim=-1)
            logits_hidden, hidden = self.rnn(contextobs, hidden)
            # theta = torch.zeros(observation.shape[0], observation.shape[-1])
            # print(theta.shape)
            # return theta, hidden
            logits_hidden = self.relu(logits_hidden)
            logits_hidden = self.fc1(logits_hidden)
            logits_hidden = self.relu(logits_hidden)
            logits = self.fc(logits_hidden)
            theta = []
            for i in range(logits.shape[0]):
                theta_i = torch.autograd.grad(logits[i], dummy_obs, create_graph=True)
                if len(theta_i) > 1:
                    breakpoint()
                theta_i = theta_i[-1][i].clone().detach()
                theta.append(theta_i)
                # logits[i].squeeze().backward(retain_graph=True)
            theta = torch.stack(theta)
            # theta = observation.grad.clone()
            theta = theta[:, 0, :]  # Remove middle dim
            # observation.requires_grad = False
            return theta, hidden
        else:
            theta, hidden = self.rnn(context, hidden)
            theta = theta.contiguous().view(-1, self.hidden_dim)
            theta = self.relu(theta)
            theta = self.fc1(theta)
            theta = self.relu(theta)
            theta_beta = self.fc(theta)
            return theta_beta, hidden

    def forward(self, context, observation, target=None, hidden=None, offset=None):
        batch_size = context.size(0)
        sig = nn.Sigmoid()
        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size)

        if self.implicit_theta:
            contextobs = torch.cat([context, observation], dim=-1)
            logits_hidden, hidden = self.rnn(contextobs, hidden)
            logits_hidden = self.relu(logits_hidden)
            logits_hidden = self.fc1(logits_hidden)
            logits_hidden = self.relu(logits_hidden)
            logits = self.fc(logits_hidden)
            theta, _ = self.get_theta(context, observation, hidden)
            prob = torch.sigmoid(logits)
            prob = prob[:, 0, 0]  # Vectorize
            return prob, hidden, theta
        else:
            theta_beta, hidden = self.get_theta(context, observation, hidden)
            theta, beta = theta_beta[:, :-2], theta_beta[:, -2:]
            observation, intercept = observation[:,:,:-1], observation[:,:,-1:]
            action_obs = torch.cat([observation, target], dim=-1)

            if self.input_size != 1:
                logits_mat = observation.squeeze(1) @ theta.T
                logits_mat_beta = action_obs.squeeze(1) @ beta.T
            
            else:
                logits_mat = (observation @ theta.T).squeeze()
                logits_mat_beta = (observation @ theta.T).squeeze()
            
            if (logits_mat.ndim == 0):
                logits = logits_mat.unsqueeze(0)
                logits_beta = logits_mat_beta.unsqueeze(0)
            elif (logits_mat.ndim == 1):
                logits = logits_mat
                logits_beta = logits_mat_beta
            else:
                logits = torch.diagonal(logits_mat)
                logits_beta = torch.diagonal(logits_mat_beta)

            prob = sig(logits+offset)
            offset += logits_beta
            return prob, hidden, theta, beta, offset
    

def model_predict(model, loader, implicit_theta, evaluate_from=0) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probability of taking an action for each sequence in a dataset usiing a CPR model.

    Parameters
    ----------
    model : torch.nn.Module
        The CPR model
    loader : torch.utils.data.DataLoader
        The dataset to predict on
    evaluate_from : int, optional
        Timestep to start evaluating from, by default 0

    Returns
    -------
    np.ndarray
        The predicted probabilities
    np.ndarray
        The true labels
    """
    preds = []
    target = []

    for context, features, targets, _, mask, static in loader:
        bs = targets.shape[0]
        outs = []
        hidden = model.init_hidden(bs, static)
        seq_len = int(mask.sum().item())
        batch_size = targets.shape[0]
        offset = torch.zeros((batch_size,))

        for step in range(seq_len):
            context_step = context[:,:,step].unsqueeze(-2)
            features_step = features[:,:,step].unsqueeze(-2)
            target_step = targets[:, step:step + 1].unsqueeze(-2)

            if not implicit_theta:
                out, hidden, theta, beta, offset = model(context=context_step, observation=features_step,
                                                         target=target_step, hidden=hidden, offset=offset)
            else:
                out, hidden, theta = model(context=context_step, observation=features_step, hidden=hidden)

            if step >= evaluate_from:
                outs.append(out)
        
        targets = targets.T[evaluate_from:seq_len,:]
        probs = torch.vstack(outs)

        preds.append(probs)
        target.append(targets)

    pred_np = torch.cat(preds).detach().numpy().squeeze()
    true_np = torch.cat(target).numpy()

    return pred_np, true_np


class VanillaRnn(nn.Module):
    def __init__(self, input_size=6, hidden_dim=64, n_layers=1, rnn_type="RNN"):
        super(VanillaRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_size = input_size
        self.typ = rnn_type
        
        if self.typ == "RNN":
            self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
        elif self.typ == "LSTM":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        if self.typ == "LSTM":
            hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden

    def forward(self, input, hidden=None):
        batch_size = input.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size=batch_size)
        
        out,hidden = self.rnn(input, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.sig(out)

        return out, hidden


def vanilla_predict(model, loader, evaluate_from=0) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probability of taking an action for each sequence in a dataset usiing a vanilla model.

    Parameters
    ----------
    model : torch.nn.Module
        The CPR model
    loader : torch.utils.data.DataLoader
        The dataset to predict on
    evaluate_from : int, optional
        Timestep to start evaluating from, by default 0

    Returns
    -------
    np.ndarray
        The predicted probabilities
    np.ndarray
        The true labels
    """
    preds = []
    target = []        
    for features, targets, _, mask in loader:

        bs = targets.shape[0]

        outs = []
        hidden = model.init_hidden(bs)
        seq_len = int(mask.sum().item()) 
        hidden = model.init_hidden(bs)
        batch_size = targets.shape[0]
        offset = torch.zeros((batch_size,))

        for step in range(seq_len):
            features_step = features[:,:,step].unsqueeze(-2)
            out, hidden = model(input=features_step, hidden=hidden)
            if step >= evaluate_from:
                outs.append(out)
        
        targets = targets.T[evaluate_from:seq_len,:] 
        target.append(targets)
        probs = torch.vstack(outs)
        preds.append(probs)

    pred_np = torch.concat(preds).detach().numpy().squeeze()
    true_np = torch.concat(target).numpy()

    return pred_np, true_np


def map_to_2d(model, loader_test, feature_cols, implicit_theta, include_intercept=True, drop_first=False):
    thetas = []

    cols = ["prob"]

    if include_intercept and implicit_theta:
        cols = ["intercept"] + cols

    df = pd.DataFrame(columns=["id", "t"] + feature_cols + cols) 

    for context, features, targets, pid, mask, static in loader_test:

        seq_len = int(mask.sum().item())
        hidden = model.init_hidden(1, static=static)
        batch_size = targets.shape[0]
        offset = torch.zeros((batch_size,))

        for step in range(seq_len):
            # with torch.no_grad():
            context_step = context[:,:,step].unsqueeze(-2)
            features_step = features[:,:,step].unsqueeze(-2)
            target_step = targets[:, step:step + 1].unsqueeze(-2)
            # theta,_ = model.get_theta(context=context_step, hidden=hidden)
            # prob, hidden, theta = model(context=context_step, observation=features_step.unsqueeze(0), hidden=hidden)
            if not implicit_theta:
                prob, hidden, theta, beta, offset = model(context=context_step, observation=features_step,
                                                          target=target_step, hidden=hidden, offset=offset)
            else:
                prob, hidden, theta = model(context=context_step, observation=features_step, hidden=hidden)

            prob, theta = prob.detach(), theta.detach()
            thetas.append(theta.numpy())
            prob = prob.item()

            try:
                if implicit_theta:
                    df.loc[len(df.index)] = [int(pid[0]), step] + theta.numpy().squeeze().tolist() + [prob]
                else:
                    df.loc[len(df.index)] = [int(pid[0]), step] + [theta.numpy().squeeze().tolist()] + [prob]
            except TypeError:
                df.loc[len(df.index)] = [int(pid[0]), step] + [theta.numpy().squeeze()] + [prob]
    
    df_coef = df.copy().sort_values(["id", "t"])
    if drop_first:
        df_coef = df_coef[df_coef["t"] != 0]

    reducer = umap.UMAP(random_state=42, n_neighbors=50, n_components=2)
    if include_intercept and implicit_theta:
        embedding = reducer.fit_transform(df_coef[feature_cols + ["intercept"]].values)
    else:
        embedding = reducer.fit_transform(df_coef[feature_cols].values)

    df_coef[["u1", "u2"]] = embedding 
    df_coef.t = df_coef.t.astype(int)
    df_coef.id = df_coef.id.astype(int)

    return df_coef, reducer
