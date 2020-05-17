import os
import subprocess
import librosa
import numpy as np
import time


"""
Audio data preprocessing for neural network based speech enhancement training.
Audio needs to be 16k
It provides:
    1. slicing and serializing
    2. verifying serialized data
"""
DATA_ROOT_DIR = '/scratch3/jul/interspeech2020_100/training' # root direction
CLEAN_TRAIN_DIR = 'clean'  # where original clean train data exist
NOISY_TRAIN_DIR = 'noisy'  # where original noisy train data exist

SER_DATA_DIR = 'ser_data_ae_se'  # serialized data folder
SER_DST_PATH = os.path.join(DATA_ROOT_DIR, SER_DATA_DIR)

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

def verify_data():
    """
    Verifies the length of each data after preprocessing.
    """
    for dirname, dirs, files in os.walk(SER_DST_PATH):
        for filename in files:
            data_pair = np.load(os.path.join(dirname, filename))
            if data_pair.shape[1] != 16384:
                print('Snippet length not 16384 : {} instead'.format(data_pair.shape[1]))
                break

def slice_signal(filepath, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size with [stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(filepath, sr=sample_rate)
    n_samples = wav.shape[0]  # contains simple amplitudes
    wav = pre_emphasize(wav)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices

def process_and_serialize():
    """
    Serialize the sliced signals and save on separate folder.
    """
    start_time = time.time()  # measure the time
    window_size = 2 ** 14  # about 1 second of samples
    sample_rate = 16000
    stride = 0.5

    if not os.path.exists(SER_DST_PATH):
        print('Creating new destination folder for new data')
        os.makedirs(SER_DST_PATH)

    # the path for source data (16k downsampled)
    clean_data_path = os.path.join(DATA_ROOT_DIR, CLEAN_TRAIN_DIR)
    noisy_data_path = os.path.join(DATA_ROOT_DIR, NOISY_TRAIN_DIR)

    # walk through the path, slice the audio file, and save the serialized result
    for dirname, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in files:
            print('Splitting : {}'.format(filename))
            clean_filepath = os.path.join(clean_data_path, filename)
            noisy_filepath = os.path.join(noisy_data_path, filename)

            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_filepath, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_filepath, window_size, stride, sample_rate)

            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(SER_DST_PATH, '{}_{}'.format(filename, idx)), arr=pair)

    # measure the time it took to process
    end_time = time.time()
    print('Total elapsed time for preprocessing : {}'.format(end_time - start_time))
if __name__ == '__main__':
    """
    Uncomment each function call that suits your needs.
    """
    process_and_serialize()  # WARNING - takes very long time
    verify_data()
