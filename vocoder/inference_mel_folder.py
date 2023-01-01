import os
from glob import glob

import torch
from scipy.io.wavfile import write

from .meldataset import mel_spectrogram, MAX_WAV_VALUE
from .models import Generator
from .stft import TorchSTFT


h = None
device = None

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(input_mel_folder, checkpoint_file):
    generator = Generator(h).to(device)
    stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft,device=device).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        files_all = []
        for input_mel_file in glob(input_mel_folder +'/*.mel'):
            x = torch.load(input_mel_file)
            spec, phase = generator(x)
            y_g_hat = stft.inverse(spec, phase)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = input_mel_file.replace('.mel','.wav')
            write(output_file, h.sampling_rate, audio)
            print('<<--',output_file)

            files_all.append(output_file)

            os.remove(input_mel_file)

        names = []
        for k in files_all:
            names.append(int(k.replace(input_mel_folder,'').replace('/','').replace('.wav','')))

        names_w = [f'{it}.wav' for it in sorted(names)]

        print('sox ' + ' '.join(names_w) + ' all.wav')


def process_folder(input_mel_folder, checkpoint_file):
    global h
    global device

    print('Initializing Inference Process..')

    json_config = {
        "resblock": "1",
        "num_gpus": 0,
        "batch_size": 128,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,

        "upsample_rates": [8,8],
        "upsample_kernel_sizes": [16,16],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "gen_istft_n_fft": 16,
        "gen_istft_hop_size": 4,

        "segment_size": 8192,
        "num_mels": 80,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,

        "sampling_rate": 22050,

        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,

        "num_workers": 4,

        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        }
    }

    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(input_mel_folder, checkpoint_file)
