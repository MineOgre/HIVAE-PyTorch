import torch
import numpy as np


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return generated.float()


def repeat_list(l, times=2):
    repeated = []
    [repeated.extend(l) for _ in range(times)]
    return repeated


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """

    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y

    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i)]]
        # res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res


def dynamic_partition_(inputs, labels):
    '''
    Splits tensor along first axis according to partition label

    Input: - inputs: tensor(shape=(N,...), dtype=T)
           - labels: tensor(shape=(N), dtype=?) with M<=N unique values
             the k-th unique value occurs exactly N_k times
    Returns: outputs: [tensor(shape=(N_1,...), dtype=T), ..., tensor(shape=(N_M,...), dtype=T)]
    '''
    classes = torch.unbind(torch.unique(labels))
    inputs_partition = [inputs[labels == cl] for cl in classes]
    return inputs_partition


def dynamic_stitch(indices, data):
    n = sum(idx.numel() for idx in indices)
    res = [None] * n
    for i, data_ in enumerate(data):
        idx = indices[i].view(-1)
        if len(idx) == 0:
            continue
        d = data_.view(idx.numel(), -1)
        k = 0
        for idx_ in idx: res[idx_] = d[k]; k += 1
    #    return torch.tensor(res).unsqueeze(1)
    return torch.cat(res).reshape(len(res), -1)


def dynamic_stitch_(inputs, conditional_indices):
    maxs = []
    for ci in conditional_indices:
        maxs.append(torch.max(ci))
    size = int(max(maxs)) + 1
    stitched = torch.Tensor(size)
    for i, idx in enumerate(conditional_indices):
        stitched[idx] = inputs[i]
    return stitched


def one_hot(seq_batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    out = torch.zeros(seq_batch.size() + torch.Size([depth])).to(seq_batch.device.type)
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def batch_normalization(batch_data_list, types_list, miss_list):
    normalized_data = []
    normalization_parameters = []

    for i, d in enumerate(batch_data_list):
        # Partition the data in missing data (0) and observed data n(1)
        #        d = d.unsqueeze(1)
        missing_data, observed_data = dynamic_partition(d, miss_list[:, i], num_partitions=2)
        condition_indices = dynamic_partition(torch.arange(0, d.size()[0]), miss_list[:, i], num_partitions=2)

        if types_list[i]['type'] == 'real':
            # We transform the data to a gaussian with mean 0 and std 1
            data_mean, data_var = torch.mean(observed_data, 0), torch.var(observed_data, 0, unbiased=False)
            #data_mean = torch.mean(observed_data, 0)
            #data_var = torch.from_numpy(np.var(np.array(observed_data), 0))
            data_var = torch.clamp(data_var, 1e-6, 1e20)  # Avoid zero values
            if observed_data.device.type == 'cuda':
                aux_X = torch.nn.BatchNorm1d(1).to(torch.float64).cuda()(observed_data)
            else:
                aux_X = torch.nn.BatchNorm1d(1).to(torch.float64)(observed_data)

            normalized_data.append(dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([data_mean, data_var])

        # When using log-normal
        elif types_list[i]['type'] == 'pos':
            #           #We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data_log = torch.log(1.0 + observed_data)
            data_mean_log, data_var_log = torch.mean(observed_data_log, 0), torch.var(observed_data_log, 0, unbiased=False)

            # data_var_log = torch.from_numpy(np.var(np.array(observed_data_log), 0))

            data_var_log = torch.clamp(data_var_log, 1e-6, 1e20)  # Avoid zero values
            if observed_data_log.device.type == 'cuda':
                aux_X = torch.nn.BatchNorm1d(1).to(torch.float64).cuda()(observed_data_log)
            else:
                aux_X = torch.nn.BatchNorm1d(1).to(torch.float64)(observed_data_log)

            normalized_data.append(dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([data_mean_log, data_var_log])

        elif types_list[i]['type'] == 'count':

            # Input log of the data
            aux_X = torch.log(observed_data)

            normalized_data.append(dynamic_stitch(condition_indices, [missing_data, aux_X]))
            normalization_parameters.append([torch.tensor([0.0]), torch.tensor([1.0])])

        else:
            # Don't normalize the categorical and ordinal variables
            normalized_data.append(d)
            normalization_parameters.append([torch.tensor([0.0]), torch.tensor([1.0])])  # No normalization here

    normalized_data = torch.cat(normalized_data, dim=1)
    return normalized_data, normalization_parameters
