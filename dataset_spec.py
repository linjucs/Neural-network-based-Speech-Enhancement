import torch
from torch.utils import data
import numpy as np
import os
from scipy import signal
import pickle

def split_pair_to_vars(sample_batch_pair, scaler_input, scaler_label,  n_pad):
    """
    Splits the generated batch data and creates combination of pairs.
    Input argument sample_batch_pair consists of a batch_size number of
    [clean_signal, noisy_signal] pairs.
    This function creates three pytorch Variables - a clean_signal, noisy_signal pair,
    clean signal only, and noisy signal only.
    It goes through preemphasis preprocessing before converted into variable.
    Args:
        sample_batch_pair(torch.Tensor): batch of [clean_signal, noisy_signal] pairs
    Returns:
        batch_pairs_var(Variable): batch of pairs containing clean signal and noisy signal
        clean_batch_var(Variable): clean signal batch
        noisy_batch_var(Varialbe): noisy signal batch
    """
    #clean_batch = np.stack([pair[0].reshape(1, -1) for pair in sample_batch_pair])
    clean_batch = sample_batch_pair[0]
    clean_batch = scale_on_2d(clean_batch, scaler_label)
    clean_batch_var = torch.from_numpy(clean_batch).type(torch.FloatTensor)
    #noisy_batch = np.stack([pair[1].reshape(1, -1) for pair in sample_batch_pair])
    noisy_batch = sample_batch_pair[1]
    n_frames = n_pad + 1
    noisy_batch = scale_on_input(noisy_batch,scaler_input, n_frames)
    noisy_batch_var = torch.from_numpy(noisy_batch).type(torch.FloatTensor)
    return clean_batch_var, noisy_batch_var
 
def scale_on_2d(x2d, scaler):
    """Scale target array data. B X 257
    """
    return scaler.transform(x2d)
def scale_on_input(x2d, scaler, n_pad):
    """ 
    scale input array B X 2570
    """
    x3d = x2d.reshape(x2d.shape[0],n_pad, 257)
    normed_x2d = []
    for i in x3d:
        normed_i = scale_on_2d(i, scaler)
        normed_x2d.append(normed_i)
    normed_x2d = np.array(normed_x2d)
    normed_x2d = normed_x2d.reshape(x2d.shape[0], -1)
    return normed_x2d

def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data. 
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]


class AudioSampleGenerator(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """

    def __init__(self, data_folder_path: str):
        if not os.path.exists(data_folder_path):
            raise FileNotFoundError

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)

    def __getitem__(self, idx):
        # get item for specified index
        #pair = np.load(self.filepaths[idx])
        pair = pickle.load(open(self.filepaths[idx], 'rb'))
        return (pair[0], pair[1])

    def __len__(self):
        return self.num_data
