#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List of log-likelihoods for the types of variables considered in this paper.
Basically, we create the different layers needed in the decoder and during the
generation of new samples

The variable reuse indicates the mode of this functions
- reuse = None -> Decoder implementation
- reuse = True -> Samples generator implementation

"""

import sys
sys.path.append('../../')

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from HI_VAE.utils import one_hot, sequence_mask
from HI_VAE.inference.loss_functions import log_poisson_loss


def loglik_real(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-3

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    data_mean, data_var = normalization_params
    data_var = torch.clamp(data_var, epsilon, np.inf)

    try:
        est_mean, est_var = theta
    except:
        est_mean, est_var = torch.unsqueeze(theta[:, 0], 1), torch.unsqueeze(theta[:, 1], 1)

    est_var = torch.clamp(nn.Softplus()(est_var), epsilon, 1e20)  # Must be positive

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var) * est_mean + data_mean

    est_var = data_var * est_var
    #    est_var = 0.05*tf.ones_like(est_var)

    # Compute loglik
    log_p_x = -0.5 * torch.sum(torch.pow(data - est_mean, 2) / est_var, 1) - int(
         list_type['dim']) * 0.5 * np.log(2 * np.pi) - 0.5 * torch.sum(torch.log(est_var), 1)
    normal = td.Normal(est_mean, torch.sqrt(est_var))
#    log_p_x = normal.log_prob(data).sum(1)

    # Outputs
    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = [est_mean, est_var]
    output['samples'] = normal.rsample()

    return output


def loglik_pos(batch_data, list_type, theta, normalization_params):
    # Log-normal distribution
    output = dict()
    epsilon = 1e-3

    # Data outputs
    data_mean_log, data_var_log = normalization_params
    data_var_log = torch.clamp(data_var_log, epsilon, np.inf)

    data, missing_mask = batch_data
    data_log = torch.log(1.0 + data)
    missing_mask = missing_mask.float()

    try:
        est_mean, est_var = theta
    except:
        est_mean, est_var = torch.unsqueeze(theta[:, 0], 1), torch.unsqueeze(theta[:, 1], 1)

    est_var = torch.clamp(torch.nn.Softplus()(est_var), epsilon, 1.0)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    # Compute loglik
    log_p_x = -0.5 * torch.sum(torch.pow(data_log - est_mean, 2) / est_var, 1) \
              - 0.5 * torch.sum(torch.log(2 * np.pi * est_var), 1) - torch.sum(data_log, 1)
#    log_normal = td.LogNormal(est_mean-1, torch.sqrt(est_var))
    # log_p_x = log_normal.log_prob(data).sum(1)  ##

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = [est_mean, est_var]
#    output['samples'] = log_normal.rsample()  # -1.0 TODO???
    output['samples'] = torch.clamp(
         torch.exp(td.Normal(est_mean, torch.sqrt(est_var)).rsample()) - 1.0, 0, 1e20)

    return output


def loglik_cat(batch_data, list_type, theta, normalization_params):
    output = dict()

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    log_pi = theta

    # Compute loglik
    # log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=data)
    # log_p_x = -tf.nn.softmax_cross_entropy_with_logits_v2(logits=log_pi, labels=tf.stop_gradient(data))
    log_pi = log_pi-torch.logsumexp(log_pi,1).reshape(log_pi.shape[0],1)
    log_p_x = torch.sum(data * F.log_softmax(log_pi, -1), -1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = log_pi
    output['samples'] = one_hot(td.Categorical(probs=nn.Softmax(1)(log_pi)).sample(),
                                depth=int(list_type['dim'])).to(torch.float64)

    return output


def loglik_ordinal(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    batch_size = data.size()[0]

    # We need to force that the outputs of the network increase with the categories
    partition_param, mean_param = theta[:, :-1], theta[:, -1]
    mean_value = torch.reshape(mean_param, [-1, 1])
    theta_values = torch.cumsum(torch.clamp(nn.Softplus()(partition_param), epsilon, 1e20), 1)
    sigmoid_est_mean = nn.Sigmoid()(theta_values - mean_value)
    mean_probs = torch.cat([sigmoid_est_mean, torch.ones([batch_size, 1]).float()], 1) - torch.cat(
        [torch.zeros([batch_size, 1]).float(), sigmoid_est_mean], 1)

    mean_probs = torch.clamp(mean_probs, epsilon, 1.0)

    # Code needed to compute samples from an ordinal distribution
    true_values = one_hot(torch.sum(data.detach().int(), 1) - 1, int(list_type['dim']))

    # Compute loglik
    #mean_probs = mean_probs-torch.logsumexp(mean_probs,1).reshape(mean_probs.shape[0],1)
    mean_probs = mean_probs / torch.sum(mean_probs, 1).reshape(mean_probs.shape[0], 1)
    log_p_x = torch.sum(true_values * F.log_softmax(torch.log(mean_probs), -1), -1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = mean_probs
    output['samples'] = sequence_mask(1 + td.Categorical(logits=torch.log(torch.clamp(mean_probs, epsilon, 1e20)))
                                      .sample(),
                                      int(list_type['dim']), dtype=torch.float32)

    return output


def loglik_count(batch_data, list_type, theta, normalization_params):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    est_lambda = theta
    est_lambda = torch.clamp(torch.nn.Softplus()(est_lambda), epsilon, 1e20)

    # log_p_x = -torch.sum(log_poisson_loss(targets=data, log_input=torch.log(est_lambda),
    #                                       compute_full_loss=True), 1)
    poisson = td.Poisson(est_lambda)
    log_p_x = poisson.log_prob(data).sum(1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = est_lambda
    output['samples'] = poisson.sample()

    return output
