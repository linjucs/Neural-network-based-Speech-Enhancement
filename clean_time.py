import argparse
import timeit
import os
import time
import torch
import numpy as np
import torch.nn as nn
from scipy.io import wavfile
import librosa
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf
from tensorboardX import SummaryWriter
#from data_preprocess import sample_rate
from model import DNN
from AE_SE import AE_SE
from dataset_spec import AudioSampleGenerator, split_pair_to_vars, scale_on_input, inverse_scale_on_2d
from data_pressing import pre_emphasize
import pickle
from dataset import de_emphasis
from data_preprocessing_spec import log_sp, read_audio, calc_sp, pad_with_border, concate_seq, write_audio
import matplotlib.pyplot as plt
from spectrogram_to_wave import recover_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=300, type=int, help='training batch size')
    parser.add_argument('--n_pad', default=10, type=int, help='context frames')
    parser.add_argument('--num_epochs', default=20, type=int, help='training epochs')
    parser.add_argument('--hidden_size', default=2048, type=int, help='hidden size')
    parser.add_argument('--input_size', default=2827, type=int, help='input size 11 frame x 257')
    parser.add_argument('--output_size', default=257, type=int, help='output size')
    parser.add_argument('--num_gen_examples', default=10, type=int, help='test samples when training')
    parser.add_argument('--sample_rate', default=16000, type=int, help='audio sample rate')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--output_dir', default="segan_data_out", type=str, help='output dir')
    parser.add_argument('--ser_dir', default="ser_data", type=str, help='serialized data')
    parser.add_argument('--gen_data_dir', default="gen_data", type=str, help='folder for saving generated data')
    parser.add_argument('--checkpoint_dir', default="checkpoints", type=str, help='folder for saving models, optimizer states')
    parser.add_argument('--log_dir', default="logs", type=str, help='summary data for tensorboard')
    parser.add_argument('--scaler_dir', default="scaler", type=str, help='scaler dir')
    parser.add_argument('--data_root_dir', default="/scratch4/jul/timit_dataset", type=str, help='root of data folder')
    parser.add_argument('--test_dir', default="./noisy", type=str, help='test dir for clean')
    parser.add_argument('--enh_dir', default="./enh", type=str, help='enhancement dir for noisy')
    opt = parser.parse_args()
    batch_size = opt.batch_size
    in_path = opt.data_root_dir
    lr = opt.lr
    num_gen_examples = opt.num_gen_examples
    num_epochs = opt.num_epochs
    hidden_size = opt.hidden_size
    n_pad = opt.n_pad
    input_size = opt.input_size
    out_size = opt.output_size
    sample_rate = opt.sample_rate
    out_path_root = opt.output_dir
    scaler_dir = opt.scaler_dir
    test_dir = opt.test_dir
    enh_dir = opt.enh_dir
    num_epochs = opt.num_epochs
    ser_data_fdr = opt.ser_dir  # serialized data
    gen_data_fdr = opt.gen_data_dir  # folder for saving generated data
    checkpoint_fdr = opt.checkpoint_dir  # folder for saving models, optimizer states, etc.
    tblog_fdr = opt.log_dir  # summary data for tensorboard
    # time info is used to distinguish dfferent training sessions
    run_time = time.strftime('%Y%m%d_%H%M', time.gmtime())  # 20180625_1742
    # output path - all outputs (generated data, logs, model checkpoints) will be stored here
    # the directory structure is as: "[curr_dir]/segan_data_out/[run_time]/"
    out_path = os.path.join(os.getcwd(), out_path_root, run_time)
    tblog_path = os.path.join(os.getcwd(), tblog_fdr, run_time)  # summary data for tensorboard
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_devices = [0]

    # create folder for generated data
    gen_data_path = os.path.join(out_path, gen_data_fdr)
    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)
    print('here')
    # create folder for model checkpoints
    checkpoint_path = os.path.join(out_path, checkpoint_fdr)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

#Your statements here


    model = AE_SE()
    model = torch.nn.DataParallel(model.to(device), device_ids=use_devices)
    #print(model)
    #save_dir = "FPGA"
    checkpoint = torch.load('AE_SE_data_out_bs100/20200513_2023/checkpoints_AE_SE/AE_SE-250.pkl')
    #from collections import OrderedDict
    state_dict = checkpoint['AE_SE']
    #for k, v in state_dict.items():
    #    name = k[7:] # remove `module.`
    #    v = v.cpu().numpy()
    #    np.savetxt(os.path.join(save_dir,name), v, newline="\n")
    #    print(name, v.shape)
    #scaler_path_input = os.path.join(scaler_dir, "scaler_input.p")                                                               
    #scaler_input = pickle.load(open(scaler_path_input, 'rb'))
    #scaler_path_label = os.path.join(scaler_dir, "scaler_label.p")                                                               
    #scaler_label = pickle.load(open(scaler_path_label, 'rb'))
    #test_audio_path = './dr1_fdac1_sx214.wav'
    #mode = 'magnitude'
    #n_concat = 11
    model.load_state_dict(state_dict)
    canvas_size = 16384
    if torch.cuda.is_available():
        model.cuda()
    for file_ in os.listdir(test_dir):
        noisy_filepath = os.path.join(test_dir, file_)
        wav, sr = librosa.load(noisy_filepath, sr=16000)
        wav = pre_emphasize(wav)
        c_res = None
        for beg_i in range(0, wav.shape[0], canvas_size):
            if wav.shape[0] - beg_i  < canvas_size:
                length = wav.shape[0] - beg_i
                pad = (canvas_size) - length
            else:
                length = canvas_size
                pad = 0
            x_ = np.zeros((batch_size, canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((wav[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = wav[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            x_ = torch.from_numpy(x_).type(torch.FloatTensor).cuda()
            x_ = torch.unsqueeze(x_, 1)
            pred, _ = model(x_)
            print(pred.shape)
            canvas_w = pred[0].cpu().detach().numpy()
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # preemphasize
        c_res = c_res.flatten()
        enh_name = os.path.join(enh_dir, file_)
        sf.write(enh_name,c_res,16000)
 #       np.squeeze(c_res, axis=1)
