import argparse
import timeit
import os
import time
import torch
import numpy as np
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
#from data_preprocess import sample_rate
from model import DNN
from dataset_spec import AudioSampleGenerator, split_pair_to_vars, scale_on_input, inverse_scale_on_2d
import pickle
from data_preprocessing_spec import log_sp, read_audio, calc_sp, pad_with_border, concate_seq, write_audio
import matplotlib.pyplot as plt
from spectrogram_to_wave import recover_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=500, type=int, help='training batch size')
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
    use_devices = [0, 1]

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


    model = DNN(input_size, hidden_size, out_size)
    model = torch.nn.DataParallel(model.to(device), device_ids=use_devices)
    #print(model)
    #save_dir = "FPGA"
    checkpoint = torch.load('./segan_data_out/20200422_0713/checkpoints/state-20.pkl')
    #from collections import OrderedDict
    state_dict = checkpoint['DNN']
    #for k, v in state_dict.items():
    #    name = k[7:] # remove `module.`
    #    v = v.cpu().numpy()
    #    np.savetxt(os.path.join(save_dir,name), v, newline="\n")
    #    print(name, v.shape)
    scaler_path_input = os.path.join(scaler_dir, "scaler_input.p")                                                               
    scaler_input = pickle.load(open(scaler_path_input, 'rb'))
    scaler_path_label = os.path.join(scaler_dir, "scaler_label.p")                                                               
    scaler_label = pickle.load(open(scaler_path_label, 'rb'))
    test_audio_path = './dr1_fdac1_sx214.wav'
    mode = 'magnitude'
    n_concat = 11
    model.load_state_dict(state_dict)
    #if torch.cuda.is_available():
    #    model.cuda()
    test_audio, _= read_audio(test_audio_path)
    test_spec = calc_sp(test_audio, mode)
    mixed_complx_x = calc_sp(test_audio, mode='complex')
    plt.matshow(test_spec.T)
    plt.savefig("noisy.png")
    n_window = 512
    n_overlap = 256
    ham_win = np.hamming(n_window)
    #print(test_spec)
    test_spec = log_sp(test_spec)
    n_pad = (n_concat - 1) // 2
    test_spec_padding = pad_with_border(test_spec, n_pad)
    test_spec_padding_concated = concate_seq(test_spec_padding, n_pad)
    test_spec_norm = scale_on_input(test_spec_padding_concated, scaler_input, n_concat)   
    #np.savetxt("noisy_input", test_spec_norm, newline="\n")
   # print(test_spec_norm.shape)
    padding = np.zeros((batch_size-test_spec_norm.shape[0], input_size))
    test_padding = np.concatenate((test_spec_norm, padding))
   # print(test_padding.shape)
    test_padding_var = torch.from_numpy(test_padding).type(torch.FloatTensor)
    start = timeit.default_timer()
    predictions = model(test_padding_var)
    print(predictions)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
   # print(predictions.shape)
    pre_removed = predictions[:test_spec_norm.shape[0]]
  #  print(pre_removed.shape)
    pre_inversed = inverse_scale_on_2d(pre_removed.cpu().detach().numpy(), scaler_label)
 #   print(np.exp(pre_inversed))
    pre_inversed = np.exp(pre_inversed)
    s = recover_wav(pre_inversed, mixed_complx_x, n_overlap, np.hamming)
    s *= np.sqrt((np.hamming(n_window)**2).sum())
    write_audio("gpu_enh.wav", s, 16000)
    pre_fpga = np.loadtxt('fpga_output')
    print(pre_fpga)
    pre_fpga_inversed = inverse_scale_on_2d(pre_fpga, scaler_label)
    pre_fpga_inversed = np.exp(pre_fpga_inversed)
    s_fpga = recover_wav(pre_fpga_inversed, mixed_complx_x, n_overlap, np.hamming)
    s_fpga *= np.sqrt((np.hamming(n_window)**2).sum())
    write_audio("fpga_enh.wav", s_fpga, 16000)
    print(pre_fpga.shape)
   # pre_inversed = pre_inversed.T
    
    #plt.matshow(pre_inversed)
    #plt.savefig('test.png')
