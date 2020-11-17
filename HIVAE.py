import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
#from torchvision import transforms

from HI_VAE import read_functions
from HI_VAE.utils import dynamic_partition, dynamic_stitch, batch_normalization
from HI_VAE.inference import loglik

# from dataset_def import HeterogeneousHealthMNISTDataset
from parse_model_args import HIVAEArgs

import matplotlib.pyplot as plt
import pickle

# from variables import *

# with open(f'{data_location}/initial_weights.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
#     [init_mean_q, init_logvar_q, init_y_layer_pt, init_y_layer_tf, init_obs_layer_mean, init_obs_layer_logvar, init_eps] = pickle.load(f)
#
# init_y_layer = init_y_layer_pt

class HIEncoder(nn.Module):
    def __init__(self, dims, types_list):
        """
        Inference network
        Beware of the missing data. In the original model there is not hidden layer
        The missing data is replaced with zeros in the input
        x_dim is extracted version of input according type_dims

        """
        super(HIEncoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        self.layers = nn.ModuleList()
        input_dim = x_dim
        # if h_dim is not None and h_dim != [] and h_dim != 0 and h_dim != [0]:
        #     neurons = [x_dim, *h_dim]
        #     for i in range(len(neurons) - 1):
        #         self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
        #         self.layers.append(nn.BatchNorm1d(neurons[i + 1]))
        #         self.layers.append(nn.ReLU())
        #     input_dim = h_dim[-1]

        # self.VAE_encoder_common_layers = nn.Sequential(*self.layers)

        # self.mean_layer = nn.Linear(input_dim, z_dim)
        # self.log_var_layer = nn.Linear(input_dim, z_dim)

        self.mean_layer = nn.ModuleList()
        self.mean_layer.append(nn.Linear(input_dim, z_dim))
        torch.nn.init.normal_(self.mean_layer[0].weight, mean=0.0, std=.005)
        # self.mean_layer[0].weight.data = torch.transpose(torch.FloatTensor(init_mean_q), 0, 1)
        # self.mean_layer[0].bias.data = torch.FloatTensor(init_mean_q[0,:])
        self.mean_layer = nn.Sequential(*self.mean_layer)

        self.log_var_layer = nn.ModuleList()
        self.log_var_layer.append(nn.Linear(input_dim, z_dim))
        # self.log_var_layer[0].weight.data = torch.transpose(torch.FloatTensor(init_logvar_q), 0, 1)
        # self.log_var_layer[0].bias.data = torch.FloatTensor(init_logvar_q[0, :])
        torch.nn.init.normal_(self.log_var_layer[0].weight, mean=0.0, std=.005)
        self.log_var_layer = nn.Sequential(*self.log_var_layer)

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        # eps = torch.tensor(init_eps)
        return mu + eps * std

    def forward(self, X):
        q_params = dict()
        samples = dict.fromkeys(['s', 'z'], [])
        #
        # if self.layers is not None:
        #     for layer in self.layers:
        #         X = F.relu(layer(X))

        # mean_qz = self.mean_layer(self.VAE_encoder_common_layers(X))
        # log_var_qz = self.log_var_layer(self.VAE_encoder_common_layers(X))
        mean_qz = self.mean_layer(X)
        log_var_qz = self.log_var_layer(X)

        log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

        samples['z'] = self.sample_latent(mean_qz, log_var_qz)
        q_params['z'] = [mean_qz, log_var_qz]

        return samples, q_params


class HIDecoder(nn.Module):
    def __init__(self, dims, types_list):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], y_dim, input_dim].
        """
        super(HIDecoder, self).__init__()

        [z_dim, h_dim, y_dim, x_dim] = dims

        self.types_list = types_list
        self.theta_cov_indexes = [0]
        for d, t in enumerate(types_list):
            if t['type'] in ['pos', 'real']:
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1]+2*t['dim'])
            elif t['type'] == 'count':
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1] + 1)
            else:
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1] + t['dim'])
        ##TODO: Bu indexlerde dim ne nclass ne?

        self.y_dim_partition = y_dim * np.ones(len(types_list), dtype=int)
        self.y_dim_output = np.sum(self.y_dim_partition)
        # self.hidden = None
        y_input_dim = z_dim
        # self.layers = nn.ModuleList()
        # if h_dim is not None and h_dim != [] and h_dim != 0 and h_dim != [0]:
        #     neurons = [z_dim, *h_dim]
        #     for i in range(len(neurons) - 1):
        #         self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
        #         torch.nn.init.normal_(self.layers[i].weight, mean=0.0, std=.005)
        #         # self.layers.append(nn.BatchNorm1d(neurons[i + 1]))
        #         self.layers.append(nn.ReLU())
        #     y_input_dim = h_dim[-1]
        #
        # self.hidden = nn.Sequential(*self.layers)

        # if h_dim is not None and h_dim != [] and h_dim != 0 and h_dim != [0]:
        #     neurons = [z_dim, *h_dim]
        #     linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        #     self.hidden = nn.ModuleList(linear_layers)
        #     y_input_dim = h_dim[-1]
        # deterministic homogeneous gamma layer

        # self.y_layer = nn.Linear(y_input_dim, self.y_dim_output)

        self.y_layer = nn.ModuleList()
        self.y_layer.append(nn.Linear(y_input_dim, self.y_dim_output))
        # self.y_layer[0].weight.data = torch.tensor(init_y_layer)
        # self.y_layer[0].bias.data = torch.tensor(init_y_layer[:, 0])
        torch.nn.init.normal_(self.y_layer[0].weight, mean=0.0, std=.005)
        # self.y_layer.append(nn.BatchNorm1d(self.y_dim_output))
        self.y_layer = nn.Sequential(*self.y_layer)

        self.obs_layer = self.get_obs_layers(y_dim)
        # self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        # self.output_activation = nn.Sigmoid()

    def get_obs_layers(self, y_dim):
        # Different layer models for each type of variable
        obs_layers = []
        for id, type in enumerate(self.types_list):
            type_dim = type['dim']
            if type['type'] == 'real' or type['type'] == 'pos':
                obs_layers.append(nn.ModuleList([nn.Linear(y_dim, type_dim),  # mean
                                                 nn.Linear(y_dim, type_dim)]))  # sigma
                torch.nn.init.normal_(obs_layers[-1][0].weight, mean=0.0, std=.005)
                torch.nn.init.normal_(obs_layers[-1][1].weight, mean=0.0, std=.005)
            elif type['type'] == 'count':
                obs_layers.append(nn.ModuleList([nn.Linear(y_dim, type_dim)]))  # lambda
                torch.nn.init.normal_(obs_layers[-1][0].weight, mean=0.0, std=.005)
            elif type['type'] == 'cat':
                obs_layers.append(nn.ModuleList([nn.Linear(y_dim, type_dim - 1)]))  # log pi
                torch.nn.init.normal_(obs_layers[-1][0].weight, mean=0.0, std=.005)
            elif type['type'] == 'ordinal':
                obs_layers.append(nn.ModuleList([nn.Linear(y_dim, type_dim - 1),  # theta
                                                 nn.Linear(y_dim, 1)]))  # mean, single value
                torch.nn.init.normal_(obs_layers[-1][0].weight, mean=0.0, std=.005)
                torch.nn.init.normal_(obs_layers[-1][1].weight, mean=0.0, std=.005)
        return nn.ModuleList(obs_layers)

    def forward(self, z, batch_x, miss_list, norm_params):
        # if self.hidden is not None:
        #     for layer in self.hidden:
        #         z = F.relu(layer(z))
        # Deterministic homogeneous representation gamma = g(z)
        # z = self.hidden(z)
        y = self.y_layer(z)
        #y_grouped = self.y_partition(y)
        ## TODO: When converting from list to tensor y_dim flexibility lost. Will be implemented if needed.
        y_grouped = torch.reshape(y, [y.shape[0], miss_list.shape[1], -1])
        theta = self.theta_estimation_from_gamma(y_grouped, miss_list)

        log_p_x, log_p_x_missing, samples_x, params_x = self.loglik_and_reconstruction(theta, batch_x,
                                                                                       miss_list, norm_params)
        return log_p_x, log_p_x_missing, samples_x, params_x

    def loglik_and_reconstruction(self, theta, batch_data_list, miss_list, normalization_params):
        log_p_x = torch.empty_like(miss_list)
        log_p_x_missing = []
        samples_x = []
        params_x = []

        # independent gamma_d -> Compute log(p(xd|gamma_d))
        for i, d in enumerate(batch_data_list):
            ##
            # d = d.unsqueeze(1)
            # Select the likelihood for the types of variables
            loglik_function = getattr(loglik, 'loglik_' + self.types_list[i]['type'])
            out = loglik_function([d, miss_list[:, i]], self.types_list[i], theta[:, self.theta_cov_indexes[i]:self.theta_cov_indexes[i+1]], normalization_params[i])

            # log_p_x.append(out['log_p_x'].unsqueeze(1))
            log_p_x[:, i] = out['log_p_x']
            log_p_x_missing.append(out['log_p_x_missing'].unsqueeze(1))  # Test-loglik element
            samples_x.append(out['samples'])
            params_x.append(out['params'])
        return log_p_x, log_p_x_missing, samples_x, params_x


    def y_partition(self, gamma):
        grouped_samples_gamma = []
        # First element must be 0 and the length of the partition vector must be len(types_dict)+1
        if len(self.y_dim_partition) != len(self.types_list):
            raise Exception("The length of the partition vector must match the number of variables in the data + 1")
        # Insert a 0 at the beginning of the cumsum vector
        partition_vector_cumsum = np.insert(np.cumsum(self.y_dim_partition), 0, 0)
        for i in range(len(self.types_list)):
            grouped_samples_gamma.append(gamma[:, partition_vector_cumsum[i]:partition_vector_cumsum[i + 1]])
        return grouped_samples_gamma

    def theta_estimation_from_gamma(self, gamma, miss_list):
        theta = torch.DoubleTensor().to(gamma.device.type)
        # independent yd -> Compute p(xd|yd)
        for i in range(gamma.shape[1]): # gamma tensor
            d = gamma[:, i, :]
            # Partition the data in missing data (0) and observed data (1)
            missing_y, observed_y = dynamic_partition(d, miss_list[:, i], 2)
            condition_indices = dynamic_partition(torch.arange(d.size()[0]), miss_list[:, i], 2)
            # Different layer models for each type of variable
            params = self.observed_data_layer(observed_y, missing_y, condition_indices, i)
            theta = torch.cat((theta, params), 1)
        return theta

    def observed_data_layer(self, observed_data, missing_data, condition_indices, i):
        outputs = torch.DoubleTensor().to(observed_data.device.type)
        for obs_layer_param in self.obs_layer[i]:
            obs_output = obs_layer_param(observed_data)
            with torch.no_grad():
                try:
                    miss_output = obs_layer_param(missing_data)
                except:
                    miss_output = []
                    # print('All values in the batch are known')
            # Join back the data
            output = dynamic_stitch(condition_indices, [miss_output, obs_output]).to(observed_data.device.type)
            if self.types_list[i]['type'] == 'cat':
                output = torch.cat((torch.zeros(output.shape[0], 1).to(torch.float64).to(observed_data.device.type), output), 1)
            if self.types_list[i]['type'] == 'ordinal' and outputs == []:
                output = torch.cat((torch.zeros(output.shape[0], 1).to(torch.float64).to(observed_data.device.type), output), 1)
            if len(self.obs_layer[i]) == 1:
                return output
            outputs = torch.cat((outputs, output), 1)
        return outputs


def cost_function(log_px, p_params, q_params, z_dim):
    # KL(q(z|s,x)|p(z|s))
    #    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']
    mean_pz, log_var_pz = torch.zeros_like(mean_qz), torch.zeros_like(log_var_qz)
    KL_z = -0.5 * z_dim + 0.5 * torch.sum(
        torch.exp(log_var_qz - log_var_pz) + torch.pow(mean_pz - mean_qz, 2) / torch.exp(
            log_var_pz) - log_var_qz + log_var_pz, 1)

    # Eq[log_p(x|y)]
    reconstructed_prob = torch.sum(log_px, 1)

    # Complete ELBO
    ELBO = torch.mean(reconstructed_prob - KL_z, 0)

    return ELBO, reconstructed_prob, KL_z


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


if __name__ == "__main__":
    """
    This is used for pre-training.
    """

    # create parser and set variables
    opt = HIVAEArgs().parse_options()
    for key in opt.keys():
        print('{:s}: {:s}'.format(key, str(opt[key])))
    locals().update(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))

    # set up model and send to GPU if available
    print('Using HIVAE')

    net_train_ELBO = np.empty((0, 1))
    net_train_KL_Z = np.empty((0, 1))
    net_train_mean_error = np.empty((0, 1))
    net_train_mode_error = np.empty((0, 1))
    loglik_epoch = []
    testloglik_epoch = []
    net_train_recon_loss = []

    training = 1
    test = 0
    torch.manual_seed(0)
    if training:

        csv_types_file = csv_types_file[:-4]+type+'.csv'
        if type == 'count' or type == 'cat' or type == 'ordinal':
            csv_file_data = csv_file_data[:-4]+'_'+type+'.csv'

        root_dir = data_source_path
        train_data, types_dict, miss_mask, true_miss_mask, n_samples  = read_functions.read_data(
            os.path.join(root_dir, csv_file_data),
             os.path.join(root_dir, csv_types_file),
             os.path.join(root_dir, mask_file),
            true_mask_file)


        # Check batch size
        if batch_size > n_samples:
            batch_size = n_samples
        # Get an integer number of batches
        n_batches = int(np.ceil(np.shape(train_data)[0] / batch_size))
        # Compute the real miss_mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)

        print('Length of dataset:  {}'.format(np.shape(train_data)[0]))

        torch.manual_seed(0)
        nnet_encoder = HIEncoder([train_data.shape[1], h_dims, latent_dim], types_dict).to(device)
        nnet_decoder = HIDecoder([latent_dim, h_dims, y_dim, train_data.shape[1]], types_dict).to(device)
        nnet_encoder = nnet_encoder.to(torch.float64)
        nnet_decoder = nnet_decoder.to(torch.float64)

        e_optimizer = torch.optim.Adam(nnet_encoder.parameters(), lr=learning_rate)
        d_optimizer = torch.optim.Adam(nnet_decoder.parameters(), lr=learning_rate)
        for epoch in range(1, epochs + 1):

            # start training HIVAE
            nnet_encoder.train()
            nnet_decoder.train()
            train_loss = 0

            avg_loss = 0
            avg_KL_s = 0
            avg_KL_z = 0
            samples_list = []
            p_params_list = []
            samples_test_list = []
            p_params_test_list = []
            log_p_x_test_total = []
            log_p_x_test_missing_total = []
            log_p_x_total = []
            log_p_x_missing_total = []

            # Randomize the data in the mini-batches
            random_perm = np.random.permutation(range(np.shape(train_data)[0]))
            train_data_aux = train_data[random_perm, :]
            miss_mask_aux = miss_mask[random_perm, :]
            true_miss_mask_aux = true_miss_mask[random_perm, :]
            # train_data_aux = train_data
            # miss_mask_aux = miss_mask
            # true_miss_mask_aux = true_miss_mask

            train_data_aux = torch.tensor(train_data_aux).to(device)
            miss_mask_aux = torch.tensor(miss_mask_aux).to(device)
            true_miss_mask_aux = torch.tensor(true_miss_mask_aux).to(device)

            #for batch_idx, sample_batched in enumerate(dataloader):
            for i in range(n_batches):
                nnet_encoder.train()
                nnet_decoder.train()

                e_optimizer.zero_grad()  # clear gradients
                d_optimizer.zero_grad()  # clear gradients

                data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux,
                                                                 batch_size, index_batch=i)

                # Delete not known data (input zeros)
                batch_data_list_observed = [data_list[i] * torch.reshape(miss_list[:, i], [batch_size, 1])
                                            for i in
                                            range(len(data_list))]
                # Batch normalization of the data
                X_list, norm_params = batch_normalization(batch_data_list_observed, types_dict,
                                                              miss_list)


                samples, q_params = nnet_encoder(X_list)

                log_p_x, log_p_x_missing, samples['x'], params_x = nnet_decoder(samples['z'], data_list, miss_list,
                                                                             norm_params)
                ELBO, reconstructed_prob, KL_z = cost_function(log_p_x, params_x, q_params, latent_dim)
                NELBO = -ELBO
                NELBO.backward()  # compute gradients
                train_loss += NELBO.item()
                avg_loss += torch.mean(reconstructed_prob)
                avg_KL_z += torch.mean(KL_z)

                e_optimizer.step()  # update parameters
                d_optimizer.step()  # update parameters

                samples_list.append(samples)
                p_params_list.append(params_x)
                log_p_x_total.append(log_p_x.detach().cpu().numpy())
                log_p_x_missing_total.append(torch.cat(log_p_x_missing, 1).detach().cpu().numpy())

                with torch.no_grad():
                    nnet_encoder.eval()
                    nnet_decoder.eval()
                    samples_test, q_params_test = nnet_encoder(X_list)

                    log_p_x_test, log_p_x_missing_test, samples_test['x'], params_x_test = nnet_decoder(q_params_test['z'][0], data_list, miss_list,
                                                                                    norm_params)
                    samples_test_list.append(samples_test)
                    p_params_test_list.append(params_x_test)

                    log_p_x_test_total.append(log_p_x_test.cpu().numpy())
                    log_p_x_test_missing_total.append(torch.cat(log_p_x_missing_test, 1).cpu().numpy())



            mask_source_dev = miss_mask_aux.clone().detach().to(torch.float64).to(device)
            data_source_dev = train_data_aux.clone().detach().to(torch.float64).to(device)
            p_params_complete = read_functions.p_params_concatenation(p_params_test_list, types_dict)
            z_aux, est_data = read_functions.samples_concatenation_xz(samples_list)

            est_data_transformed = read_functions.discrete_variables_transformation(est_data,
                                                                                    types_dict)
            train_data_transformed = read_functions.discrete_variables_transformation(
                data_source_dev, types_dict)
            est_data_imputed = read_functions.mean_imputation(train_data_transformed,
                                                              mask_source_dev,
                                                              types_dict)

            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete, types_dict)

            error_observed_samples, error_missing_samples = \
                read_functions.error_computation(train_data_transformed,
                                                 est_data_transformed,
                                                 types_dict,
                                                 mask_source_dev)
            error_observed_imputed, error_missing_imputed = \
                read_functions.error_computation(train_data_transformed,
                                                 est_data_imputed, types_dict,
                                                 mask_source_dev)

            error_train_mean, error_test_mean = \
                read_functions.error_computation(train_data_transformed, loglik_mean,
                                                 types_dict, mask_source_dev)
            error_train_mode, error_test_mode = \
                read_functions.error_computation(train_data_transformed, loglik_mode,
                                                 types_dict, mask_source_dev)


            # log_p_x_test_total = np.concatenate(log_p_x_test_total, 0)
            # log_p_x_test_missing_total = np.concatenate(log_p_x_test_missing_total, 0)
            log_p_x_total = np.concatenate(log_p_x_total, 0)
            log_p_x_missing_total = np.concatenate(log_p_x_missing_total, 0)

            loglik_per_variable = np.sum(log_p_x_total, 0) / np.sum(np.array(miss_mask_aux.cpu()), 0)
            loglik_per_variable_missing = np.sum(log_p_x_missing_total, 0) / np.sum(1.0 - np.array(miss_mask_aux.cpu()), 0)

            print('====> Epoch: {} ELBO: {:.4f} KL_Z: {:.4f} Recon LogLik: {:.4f}'.format(epoch, -train_loss / n_batches,
                                                                                        avg_KL_z / n_batches,
                                                                                        avg_loss / n_batches))
            net_train_ELBO = np.append(net_train_ELBO, train_loss / n_batches)
            net_train_recon_loss.append(loglik_per_variable)
            net_train_KL_Z = np.append(net_train_KL_Z, [avg_KL_z / n_batches])
            net_train_mean_error = np.append(net_train_mean_error, torch.mean(torch.tensor(error_test_mean)))
            net_train_mode_error = np.append(net_train_mode_error, torch.mean(torch.tensor(error_test_mode)))


        net_train_ELBO = -net_train_ELBO

        # with open(f'{data_source_path}/{csv_types_file[:-4]}_PT_result_data_{keyword}.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([net_train_recon_loss, net_train_KL_Z, net_train_mean_error, net_train_mode_error, error_missing_imputed, net_train_ELBO], f)

        torch.save(nnet_encoder.state_dict(), os.path.join(save_path, 'encoder_params_just_cov.pth'))
        torch.save(nnet_decoder.state_dict(), os.path.join(save_path, 'decoder_params_just_cov.pth'))

    # if test:
    #     # set up dataset
    #     if dataset_type == 'HeteroHealthMNIST':
    #         test_dataset = HeterogeneousHealthMNISTDataset(csv_file_data=csv_file_test_data,
    #                                                        csv_file_label=csv_file_test_label,
    #                                                        mask_file=mask_test_file, types_file=csv_types_file,
    #                                                        true_miss_file=true_mask_test_file,
    #                                                        root_dir=data_source_path,
    #                                                        # transform=transforms.ToTensor())
    #                                                        transform=None)
    #     else:
    #         try:
    #             raise ValueError
    #         except ValueError as err:
    #             raise type(err)("Dataset argument could not be found!")
    #
    #     nnet_encoder = HIEncoder([test_dataset.cov_dim_ext, h_dims, latent_dim], test_dataset.types_dict).to(device)
    #     nnet_decoder = HIDecoder([latent_dim, h_dims, y_dim, test_dataset.cov_dim_ext], test_dataset.types_dict).to(
    #         device)
    #
    #     dataloader = DataLoader(test_dataset, batch_size=test_dataset.n_samples, shuffle=True, num_workers=0)
    #
    #     load_model(nnet_encoder, os.path.join(save_path, 'encoder_params_just_cov.pth'), device)
    #     load_model(nnet_decoder, os.path.join(save_path, 'decoder_params_just_cov.pth'), device)
    #
    #     with torch.no_grad():
    #         samples_list = []
    #         for batch_idx, sample_batched in enumerate(dataloader):
    #             batch_size = sample_batched['covariate'].shape[0]
    #             data = sample_batched['covariate'].reshape(batch_size, -1)
    #             data = data.to(device)  # send to GPU
    #             mask = sample_batched['mask'].reshape(batch_size, -1)
    #             mask = mask.to(device)
    #
    #             data_list, miss_list = read_functions.next_batch(data, test_dataset.types_dict, mask,
    #                                                              batch_size, index_batch=batch_idx)
    #
    #             # Delete not known data (input zeros)
    #             batch_data_list_observed = [data_list[i] * np.reshape(miss_list[:, i], [batch_size, 1])
    #                                         for i in
    #                                         range(len(data_list))]
    #
    #             # Batch normalization of the data
    #             X_list, norm_params = batch_normalization(data_list, test_dataset.types_dict,
    #                                                       miss_list)
    #
    #             samples, q_params = nnet_encoder(X_list)
    #
    #             log_p_x, log_p_x_missing, samples['x'], params_x = nnet_decoder(samples['z'], data_list, miss_list,
    #                                                                             norm_params)
    #             ELBO, reconstructed_prob, KL_z = cost_function(log_p_x, params_x, q_params, latent_dim)
    #             NELBO = -ELBO
    #             samples_list.append(samples)
    #             print("ELBO  ")
    #             print(ELBO)
    #
    #         z_aux, est_data = read_functions.samples_concatenation_xz(samples_list)
    #         est_data_transformed = read_functions.discrete_variables_transformation(est_data, test_dataset.types_dict)
    #         test_data_transformed = read_functions.discrete_variables_transformation(
    #             np.array(test_dataset.data_source), test_dataset.types_dict)
    #         est_data_imputed = read_functions.mean_imputation(test_data_transformed,
    #                                                           np.array(test_dataset.mask_source),
    #                                                           test_dataset.types_dict)
    #
    #         loglik_mean, loglik_mode = read_functions.statistics(params_x, test_dataset.types_dict)
    #
    #         error_observed_samples, error_missing_samples = \
    #             read_functions.error_computation(test_data_transformed,
    #                                              est_data_transformed,
    #                                              test_dataset.types_dict,
    #                                              np.array(test_dataset.mask_source))
    #         error_observed_imputed, error_missing_imputed = \
    #             read_functions.error_computation(test_data_transformed,
    #                                              est_data_imputed, test_dataset.types_dict,
    #                                              np.array(test_dataset.mask_source))
    #
    #         error_train_mean, error_test_mean = \
    #             read_functions.error_computation(test_data_transformed, loglik_mean,
    #                                              test_dataset.types_dict, np.array(test_dataset.mask_source))
    #         error_train_mode, error_test_mode = \
    #             read_functions.error_computation(test_data_transformed, loglik_mode,
    #                                              test_dataset.types_dict, np.array(test_dataset.mask_source))
    #
    #         print(np.round(np.average(error_missing_samples), 4), np.round(np.average(error_missing_imputed), 4))
    #         print(np.round(np.average(error_observed_samples), 4))
    #
    #         print(np.round((np.sum(error_missing_samples[0:4]) + error_missing_samples[8]) / 5, 4),
    #               np.round((np.sum(error_missing_imputed[0:4]) + error_missing_imputed[8]) / 5, 4))
    #         print(np.round((np.average(error_missing_samples[4:8])), 4),
    #               np.round((np.average(error_missing_imputed[4:8])), 4))
    #
    #         print('Test error mode: ' + str(np.round(np.mean(error_test_mode), 3)))
