import torch
from torch.utils import data
import numpy as np
import os
from scipy import signal

def pre_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    Pre-emphasis of higher frequencies given a batch of signal.
    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient
    Returns:
        result: pre-emphasized signal batch
    """
    return signal.lfilter([1, -emph_coeff], [1], signal_batch)


def de_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    De-emphasis operation given a batch of signal.
    Reverts the pre-emphasized signal.
    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient
    Returns:
        result: de-emphasized signal batch
    """
    return signal.lfilter([1], [1, -emph_coeff], signal_batch)

def pre_emphasize(x, coef=0.95):
    if coef <= 0:
        return x
    x0 = np.reshape(x[0], (1,))
    diff = x[1:] - coef * x[:-1]
    concat = np.concatenate((x0, diff), axis=0)
    return concat

def de_emphasize(y, coef=0.95):
    if coef <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coef * x[n - 1] + y[n]
    return x

def split_pair_to_vars(sample_batch_pair):
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
    # pre-emphasis
 #   sample_batch_pair = pre_emphasize(sample_batch_pair.numpy())
    batch_pairs_var = sample_batch_pair
    #batch_pairs_var = torch.from_numpy(sample_batch_pair).type(torch.FloatTensor)  # [40 x 2 x 16384]
    clean_batch = np.stack([pair[0].reshape(1, -1) for pair in sample_batch_pair])
    clean_batch_var = torch.from_numpy(clean_batch).type(torch.FloatTensor)
    noisy_batch = np.stack([pair[1].reshape(1, -1) for pair in sample_batch_pair])
    noisy_batch_var = torch.from_numpy(noisy_batch).type(torch.FloatTensor)
 #   print(clean_batch_var.shape)
    return batch_pairs_var, clean_batch_var, noisy_batch_var

class AudioSampleGenerator(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """
    SAMPLE_LENGTH = 16384

    def __init__(self, data_folder_path: str):
        if not os.path.exists(data_folder_path):
            raise FileNotFoundError

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)

    def reference_batch(self, batch_size: int):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.
        Args:
            batch_size(int): batch size
        Returns:
            ref_batch: reference batch
        """
        ref_filenames = np.random.choice(self.filepaths, batch_size)
        ref_batch = torch.from_numpy(np.stack([np.load(f) for f in ref_filenames]))
        return ref_batch

    def fixed_test_audio(self, num_test_audio: int):
        """
        Randomly chosen batch for testing generated results.
        Args:
            num_test_audio(int): number of test audio.
                Must be same as batch size of training,
                otherwise it cannot go through the forward step of generator.
        """
        test_filenames = np.random.choice(self.filepaths, num_test_audio)
        # stack the data for all test audios
        test_audios = np.stack([np.load(f) for f in test_filenames])
        test_clean_set = test_audios[:, 0].reshape((num_test_audio, 1, self.SAMPLE_LENGTH))
        test_noisy_set = test_audios[:, 1].reshape((num_test_audio, 1, self.SAMPLE_LENGTH))
        # file names of test samples
        test_basenames = [os.path.basename(fpath) for fpath in test_filenames]
        return test_basenames, test_clean_set, test_noisy_set

    def __getitem__(self, idx):
        # get item for specified index
        pair = np.load(self.filepaths[idx])
        return pair

    def __len__(self):
        return self.num_data
