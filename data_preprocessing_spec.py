import os
import soundfile
import numpy as np
import argparse
import time
import pickle
from scipy import signal
from sklearn import preprocessing

"""
Audio data preprocessing for neural network based speech enhancement training.
Audio needs to be 16k
It provides:
    1. slicing and serializing
    2. verifying serialized data
"""
DATA_ROOT_DIR = '/scratch4/jul/timit_dataset' # root direction
CLEAN_TRAIN_DIR = 'clean'  # where original clean train data exist
NOISY_TRAIN_DIR = 'noisy'  # where original noisy train data exist

SER_DATA_DIR = 'ser_data'  # serialized data folder
SER_DST_PATH = os.path.join(DATA_ROOT_DIR, SER_DATA_DIR)
SCALER_PATH = 'scaler'

def log_sp(x):
    return np.log(x + 1e-08)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = 512
    n_overlap = 256
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)
def concate_seq(x, n_pad):
    """
     concate context frame according the n_pad
    """
    x_concated = []
    i = n_pad
    while i + n_pad < len(x):
        x_concated.append(x[i-n_pad:i+n_pad+1].flatten())
        i += 1
    return np.array(x_concated)


def process_and_serialize():
    """
    Serialize the sliced signals and save on separate folder.
    """
    start_time = time.time()  # measure the time
    sample_rate = 16000
    mode = 'magnitude'
    n_concat = 11
    x_all = []
    y_all = []
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
            clean_signal, _ = read_audio(clean_filepath)
            noisy_signal, _ = read_audio(noisy_filepath)
            clean_spec = calc_sp(clean_signal, mode)
            noisy_spec = calc_sp(noisy_signal, mode)
            noisy_spec = log_sp(noisy_spec)
            clean_spec = log_sp(clean_spec)
            assert len(noisy_spec) == len(clean_spec)
            x_all.extend(noisy_spec)
            y_all.extend(clean_spec)
            n_pad = (n_concat - 1) // 2
            noisy_spec_padding = pad_with_border(noisy_spec, n_pad)
            noisy_spec_padding_concated = concate_seq(noisy_spec_padding, n_pad)
            #print(clean_spec.shape)
            #print(noisy_spec_padding_concated.shape)
            for idx, slice_tuple in enumerate(zip(clean_spec, noisy_spec_padding_concated)):
                pair = (slice_tuple[0], slice_tuple[1])
                out_path = os.path.join(SER_DST_PATH, '{}_{}'.format(filename, idx))
                with open(out_path, 'wb') as pfile:
                    pickle.dump(pair, pfile, protocol=pickle.HIGHEST_PROTOCOL)
                #np.save(os.path.join(SER_DST_PATH, '{}_{}'.format(filename, idx)), arr=pair)
        scaler1 = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x_all)
        scaler2 = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(y_all)
        # Write out scaler. 
        if not os.path.exists(SCALER_PATH):
            os.makedirs(SCALER_PATH)
        out_path_input = os.path.join(SCALER_PATH, "scaler_input.p")
        out_path_label = os.path.join(SCALER_PATH, "scaler_label.p")
        create_folder(os.path.dirname(out_path_input))
        create_folder(os.path.dirname(out_path_label))
        pickle.dump(scaler1, open(out_path_input, 'wb'))
        pickle.dump(scaler2, open(out_path_label, 'wb'))
    
        print("Save scaler to %s" % out_path_input)
        print("Compute scaler finished!")
        # measure the time it took to process
        end_time = time.time()
        print('Total elapsed time for preprocessing : {}'.format(end_time - start_time))

if __name__ == '__main__':
    """
    Uncomment each function call that suits your needs.
    """
    process_and_serialize()  # WARNING - takes very long time
