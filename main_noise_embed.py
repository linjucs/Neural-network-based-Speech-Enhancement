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
from AE_SE import AE_SE
from dataset import AudioSampleGenerator, split_pair_to_vars, de_emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=300, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--hidden_size', default=2048, type=int, help='hidden size')
    parser.add_argument('--input_size', default=16384, type=int, help='input size')
    parser.add_argument('--output_size', default=16384, type=int, help='output size')
    parser.add_argument('--num_gen_examples', default=10, type=int, help='test samples when training')
    parser.add_argument('--sample_rate', default=16000, type=int, help='audio sample rate')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--output_dir', default="noise_data_out", type=str, help='output dir')
    parser.add_argument('--ser_dir', default="ser_data_noise_embedding", type=str, help='serialized data')
    parser.add_argument('--gen_data_dir', default="gen_data", type=str, help='folder for saving generated data')
    parser.add_argument('--checkpoint_dir', default="checkpoints_noise_embedding", type=str, help='folder for saving models, optimizer states')
    parser.add_argument('--log_dir', default="logs", type=str, help='summary data for tensorboard')
    parser.add_argument('--data_root_dir', default="/scratch3/jul/interspeech2020_100/training", type=str, help='root of data folder')
    opt = parser.parse_args()
    batch_size = opt.batch_size
    in_path = opt.data_root_dir
    lr = opt.lr
    num_gen_examples = opt.num_gen_examples
    num_epochs = opt.num_epochs
    hidden_size = opt.hidden_size
    input_size = opt.input_size
    out_size = opt.output_size
    sample_rate = opt.sample_rate
    out_path_root = opt.output_dir
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

    # create folder for model checkpoints
    checkpoint_path = os.path.join(out_path, checkpoint_fdr)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model = AE_SE()
    model = torch.nn.DataParallel(model.to(device), device_ids=use_devices)  # use GPU
    print(model)
    # load data
    print('loading data...')
    sample_generator = AudioSampleGenerator(os.path.join(in_path, ser_data_fdr))
    random_data_loader = DataLoader(
        dataset=sample_generator,
        batch_size=batch_size,  # specified batch size here
        shuffle=True,
        num_workers=4,
        drop_last=True,  # drop the last batch that cannot be divided by batch_size
        pin_memory=True)
    print('DataLoader created')
    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # create tensorboard writer
    # The logs will be stored NOT under the run_time, but under segan_data_out/'tblog_fdr'.
    # This way, tensorboard can show graphs for each experiment in one board
    tbwriter = SummaryWriter(log_dir=tblog_path)
    print('TensorboardX summary writer created')

    # test samples for generation
    test_noise_filenames, fixed_test_clean, fixed_test_noise = \
        sample_generator.fixed_test_audio(num_gen_examples)
    fixed_test_clean = torch.from_numpy(fixed_test_clean)
    fixed_test_noise = torch.from_numpy(fixed_test_noise)
    print('Test samples loaded')
    # record the fixed examples
    for idx, fname in enumerate(test_noise_filenames):
        tbwriter.add_audio(
            'test_audio_clean/{}'.format(fname),
            fixed_test_clean.numpy()[idx].T,
            sample_rate=sample_rate)
        tbwriter.add_audio(
            'test_audio_noise/{}'.format(fname),
            fixed_test_noise.numpy()[idx].T,
            sample_rate=sample_rate)
    print('Starting Training...')
    total_steps = 1
    MSE = nn.MSELoss()
    for epoch in range(num_epochs):
        # add epoch number with corresponding step number
        tbwriter.add_scalar('epoch', epoch, total_steps)
        for i, sample_batch_pairs in enumerate(random_data_loader):
            batch_pairs_var, clean_batch_var, noisy_batch_var = split_pair_to_vars(sample_batch_pairs)
            batch_pairs_var = batch_pairs_var.to(device)
            clean_batch_var = clean_batch_var.to(device)
            noisy_batch_var = noisy_batch_var.to(device)
            outputs, c = model(noisy_batch_var)
           # print(outputs.shape, clean_batch_var.shape)
           # print(clean_batch_var)
            loss = MSE(outputs, clean_batch_var) * 100
           # loss = torch.mean(torch.abs(torch.add(outputs, torch.neg(clean_batch_var)))) * 100
            # back-propagate and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    'Epoch {}\t'
                    'Step {}\t'
                    'loss {:.5f}'
                    .format(epoch + 1, i + 1, loss.item()))
            # record scalar data for tensorboard
            tbwriter.add_scalar('loss/loss', loss.item(), total_steps)
            #if i == 0:
            #    enh_speech = DNN(fixed_test_noise)
            #    enh_speech_data = enh_speech.data.cpu().numpy()  # convert to numpy array
                #print(enh_speech_data)
            #    enh_speech_data = de_emphasis(enh_speech_data, emph_coeff=0.95)

            #for idx in range(num_gen_examples):
            #    generated_sample = enh_speech_data[idx]
            #    gen_fname = test_noise_filenames[idx]
            #    filepath = os.path.join(
            #            gen_data_path, '{}_e{}.wav'.format(gen_fname, epoch))
                # write to file
            #    wavfile.write(filepath, sample_rate, generated_sample.T)
                # show on tensorboard log
            #    tbwriter.add_audio(
            #        '{}/{}'.format(epoch, gen_fname),
            #        generated_sample.T,
            #        total_steps,
            #        sample_rate)

        total_steps += 1
    # save various states
    state_path = os.path.join(checkpoint_path, 'AE_SE-{}.pkl'.format(epoch + 1))
    state = {
        'AE_SE': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, state_path)


    tbwriter.close()
    print('Finished Training!')
            
