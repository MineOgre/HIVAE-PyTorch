import argparse
import ast


"""
Helper code for loading parameters from parameter file or from command line
"""

class LoadFromFile (argparse.Action):
    """
    Read parameters from config file
    """
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().splitlines(), namespace)


class HIVAEArgs:
    """
    Runtime parameters for the HI-VAE model
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Enter configuration arguments for the HI-VAE pre-training')

        self.parser.add_argument('--data_source_path', type=str, default='./data', help='Path to data')
        # self.parser.add_argument('--keyword', type=str, default='', help='Experiment keyword')
        self.parser.add_argument('--save_path', type=str, default='./results', help='Path to save data')
        self.parser.add_argument('--csv_file_data', type=str, help='Name of data file', required=False)
        self.parser.add_argument('--csv_file_label', type=str, help='Name of label file', required=False)
        self.parser.add_argument('--csv_types_file', type=str, help='Name of types file', required=True)
        self.parser.add_argument('--type', type=str, help='Type of the data', required=True)
        self.parser.add_argument('--mask_file', type=str, help='Name of mask file', default=None)
        self.parser.add_argument('--true_mask_file', type=str, help='Name of true mask file', default=None)
        self.parser.add_argument('--csv_file_test_data', type=str, help='Name of test data file', required=False)
        self.parser.add_argument('--csv_file_test_label', type=str, help='Name of test label file', required=False)
        self.parser.add_argument('--mask_test_file', type=str, help='Name of test mask file', default=None)
        self.parser.add_argument('--true_mask_test_file', type=str, help='Name of test true mask file', default=None)
        self.parser.add_argument('--dataset_type', required=False, choices=['HeteroHealthMNIST'],
                                 help='Type of dataset being used.')
        self.parser.add_argument('--latent_dim', type=int, default=2, help='Number of latent dimensions')
        self.parser.add_argument('--h_dims', type=str, help='Hidden Layer configuration as list of dimensions', required=True)
        self.parser.add_argument('--y_dim', type=int, help='Number of Y dimensions', required=True)
        self.parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
        self.parser.add_argument('--num_dim', type=int, help='Number of input dimensions', required=False)
        self.parser.add_argument('--batch_size', type=int, help='Batch size', default=500, required=False)
        self.parser.add_argument('--learning_rate', type=float, help='Learning Rate', default=0.001, required=False)
#        self.parser.add_argument('--type_nnet', required=False, choices=['conv', 'simple'],
#                                 help='Type of neural network for the encoder and decoder')
        self.parser.add_argument('--iter_num', type=int, default=1, help='Iteration number. Useful for multiple runs')

    def parse_options(self):
        opt = vars(self.parser.parse_args())
        opt['h_dims'] = opt['h_dims'].split(',')
        opt['h_dims'] = [int(i) for i in opt['h_dims']]
        return opt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
