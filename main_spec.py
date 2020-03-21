import argparse
import os
import time
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
#from data_preprocess import sample_rate
from model import DNN
from dataset_spec import AudioSampleGenerator, split_pair_to_vars
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--n_pad', default=10, type=int, help='context frames')
    parser.add_argument('--num_epochs', default=100, type=int, help='training epochs')
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
    parser.add_argument('--data_root_dir', default="/scratch4/jul/interspeech2020/testing/seen", type=str, help='root of data folder')
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
    use_devices = [0,1]

    # create folder for generated data
    gen_data_path = os.path.join(out_path, gen_data_fdr)
    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)

    # create folder for model checkpoints
    checkpoint_path = os.path.join(out_path, checkpoint_fdr)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model = DNN(input_size, hidden_size, out_size)
    model = torch.nn.DataParallel(model.to(device), device_ids=use_devices)  # use GPU
    print(model)
    # load data
    print('loading data...')
    sample_generator = AudioSampleGenerator(os.path.join(in_path, ser_data_fdr))
    random_data_loader = DataLoader(
        dataset=sample_generator,
        batch_size=batch_size,  # specified batch size here
        shuffle=True,
        num_workers=1,
        drop_last=True,  # drop the last batch that cannot be divided by batch_size
        pin_memory=True)
    print('DataLoader created')
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    # create tensorboard writer
    # The logs will be stored NOT under the run_time, but under segan_data_out/'tblog_fdr'.
    # This way, tensorboard can show graphs for each experiment in one board
    tbwriter = SummaryWriter(log_dir=tblog_path)
    print('TensorboardX summary writer created')

    print('Starting Training...')
    total_steps = 1
    MSE = nn.MSELoss()
    
    scaler_path = os.path.join(scaler_dir, "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    for epoch in range(num_epochs):
        # add epoch number with corresponding step number
        tbwriter.add_scalar('epoch', epoch, total_steps)
        for i, sample_batch_pairs in enumerate(random_data_loader):
            clean_batch_var, noisy_batch_var = split_pair_to_vars(sample_batch_pairs, scaler, n_pad)
            
            clean_batch_var = clean_batch_var.to(device)
            noisy_batch_var = noisy_batch_var.to(device)
            outputs = model(noisy_batch_var)
            loss = MSE(outputs, clean_batch_var)
            # back-propagate and update
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print(
                    'Epoch {}\t'
                    'Step {}\t'
                    'loss {:.5f}'
                    .format(epoch + 1, i + 1, loss.item()))
            # record scalar data for tensorboard
        total_steps += 1
    # save various states
    state_path = os.path.join(checkpoint_path, 'state-{}.pkl'.format(epoch + 1))
    state = {
        'DNN': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, state_path)


    tbwriter.close()
    print('Finished Training!')
            
