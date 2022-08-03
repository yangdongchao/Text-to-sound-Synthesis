#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import librosa
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tools.file_io import load_pickle_file
import torchvision

class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)

class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val

class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val

class Multiply(object):
    def __init__(self, val, inverse=False) -> None:
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val

class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return np.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)

class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]

class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)


TRANSFORMS = torchvision.transforms.Compose([
    MelSpectrogram(sr=22050, nfft=1024, fmin=125, fmax=7600, nmels=80, hoplen=1024//4, spec_power=1),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
    TrimSpec(860)
])

class TTS_eval(Dataset):

    def __init__(self, data_path, config):
        self.mel_path = data_path
        # if split == 'val':
        #     self.h5_path = 'data/hdf5s_22050/val/val.h5'
        # elif split == 'test':
        #     self.h5_path = 'data/hdf5s_22050/test/test.h5'
        self.h5_path = 'data/hdf5s_22050/val/val.h5'
        self.audio_gen = []
        self.caps_gen = []
        with h5py.File(self.h5_path, 'r') as hf:
            audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            captions = [caption for caption in hf['caption'][:]]
        for i in range(len(audio_names)):
            for j in range(10): #
                tmp_name = audio_names[i][:-4] + '_mel_sample_' + str(j) + '.npy'
                self.audio_gen.append(tmp_name)
                self.caps_gen.append(captions[i])
        self.caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

    def __len__(self):
        return len(self.audio_gen)

    def __getitem__(self, index):
        audio_name = self.audio_gen[index] # it represent 
        captions = self.caps_gen[index]

        target_dict = {}
        for i, cap_ind in enumerate(self.caption_field):
            target_dict[cap_ind] = captions[i]

        # feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
        #                                          hop_length=self.hop_length, n_mels=self.n_mels)
        # feature = librosa.power_to_db(feature).T
        # feature = feature[:-1, :]

        feature = np.load(self.mel_path + audio_name).T
        #print('feature ', feature.shape)
        return feature, target_dict, audio_name

class AudioCapsDataset(Dataset):

    def __init__(self, config):
        super(AudioCapsDataset, self).__init__()

        self.h5_path = 'data/hdf5s_22050/train/train.h5'
        vocabulary_path = 'data/pickles/words_list.p'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            #self.captions = [caption.decode() for caption in hf['caption'][:]]
            self.captions = [caption for caption in hf['caption'][:]]

        self.vocabulary = load_pickle_file(vocabulary_path)

        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):

        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        caption = self.captions[index]

        # feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
        #                                          hop_length=self.hop_length, n_mels=self.n_mels)
        # feature = librosa.power_to_db(feature).T
        feature = TRANSFORMS(waveform).T
        # print('feature0 ',feature.shape)
        # feature = feature[:-1, :]
        # print('feature ',feature.shape)
        # assert 1==2
        words = caption.strip().split()
        # wwords = [str(word) for word in words]
        # for w in wwords:
        #     print(w)
        # assert 1==2
        target = np.array([self.vocabulary.index(word) for word in words])
        target_len = len(target)
        # print('target ',target)
        # print('target_len ',target_len)
        # print('audio_name ',audio_name)
        # print('caption ',caption)
        # assert 1==2
        return feature, target, target_len, audio_name, caption

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000 or self.sr==22050:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')

class AudioCapsDataset_tts(Dataset):

    def __init__(self, config):
        super(AudioCapsDataset_tts, self).__init__()

        self.h5_path = 'data/hdf5s/train/train.h5'
        vocabulary_path = 'data/pickles/words_list.p'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            # self.captions = [caption.decode() for caption in hf['caption'][:]]
            self.captions = [caption for caption in hf['caption'][:]]

        self.vocabulary = load_pickle_file(vocabulary_path)

        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):

        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        caption = self.captions[index]

        feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
                                                 hop_length=self.hop_length, n_mels=self.n_mels)
        feature = librosa.power_to_db(feature).T
        feature = feature[:-1, :]
        words = caption.strip().split()
        #print(words)
        target = np.array([self.vocabulary.index(word) for word in words])
        target_len = len(target)

        return feature, target, target_len, audio_name, caption

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')



class AudioCapsEvalDataset(Dataset):

    def __init__(self, split, config):

        if split == 'val':
            self.h5_path = 'data/hdf5s_22050/val/val.h5'
        elif split == 'test':
            self.h5_path = 'data/hdf5s_22050/test/test.h5'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.captions = [caption for caption in hf['caption'][:]]

        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

        self.caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        captions = self.captions[index]

        target_dict = {}
        for i, cap_ind in enumerate(self.caption_field):
            target_dict[cap_ind] = captions[i]

        # feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
        #                                          hop_length=self.hop_length, n_mels=self.n_mels)
        # feature = librosa.power_to_db(feature).T
        # feature = feature[:-1, :]

        feature = TRANSFORMS(waveform).T

        return feature, target_dict, audio_name

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000 or self.sr==22050:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


def get_audiocaps_loader(split,
                         config):
    if split == 'train':
        dataset = AudioCapsDataset(config)
        return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                          shuffle=True, drop_last=True,
                          num_workers=config.data.num_workers, collate_fn=collate_fn)
    elif split in ['val', 'test']:
        dataset = AudioCapsEvalDataset(split, config)
        return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                          shuffle=False, drop_last=False,
                          num_workers=config.data.num_workers, collate_fn=collate_fn_eval)

def get_generate_loader(data_path,
                         config):
    dataset = TTS_eval(data_path, config)
    return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=0, collate_fn=collate_fn_eval)



def collate_fn(batch):

    max_caption_length = max(i[1].shape[0] for i in batch)

    eos_token = batch[0][1][-1]

    words_tensor = []

    for _, words_indexs, _, _, _ in batch:
        if max_caption_length >= words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long(), padding]
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))

    feature = [i[0] for i in batch]
    feature_tensor = torch.tensor(feature)
    target_tensor = torch.cat(words_tensor)

    target_lens = [i[2] for i in batch]
    file_names = [i[3] for i in batch]
    captions = [i[4] for i in batch]

    return feature_tensor, target_tensor, target_lens, file_names, captions


def collate_fn_eval(batch):

    feature = [i[0] for i in batch]
    feature_tensor = torch.tensor(feature)

    file_names = [i[2] for i in batch]
    target_dicts = [i[1] for i in batch]

    return feature_tensor, target_dicts, file_names
