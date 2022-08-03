#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import time
from itertools import chain

import h5py
import numpy as np
import librosa
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from tools.file_io import load_csv_file, write_pickle_file


def load_metadata(csv_file):
    """Load meta data of AudioCaps
    """
    if 'train' not in csv_file:
        caption_field = ['caption_{}'.format(i) for i in range(1, 6)]
    else:
        caption_field = None
    csv_list = load_csv_file(csv_file)

    audio_names = []
    captions = []

    for i, item in enumerate(csv_list):

        audio_name = item['file_name']
        if caption_field is not None:
            item_captions = [_sentence_process(item[cap_ind]) for cap_ind in caption_field] # for eval
            #item_captions = [_sentence_process(item[cap_ind], add_specials=False) for cap_ind in caption_field]
        else:
            item_captions = _sentence_process(item['caption'])
        audio_names.append(audio_name)
        captions.append(item_captions)

    meta_dict = {'audio_name': np.array(audio_names), 'caption': np.array(captions)}

    return meta_dict


def pack_wavs_to_hdf5():

    splits = ['train', 'val', 'test']
    sampling_rate = 22050 # 32000 
    all_captions = []

    for split in splits:
        csv_path = 'data/csv_files/{}.csv'.format(split)
        audio_dir = 'data/waveforms/{}/'.format(split)
        hdf5_path = 'data/hdf5s_22050/{}/'.format(split)

        # make dir for hdf5
        Path(hdf5_path).mkdir(parents=True, exist_ok=True)

        meta_dict = load_metadata(csv_path)
        # meta_dict: {'audio_names': [], 'captions': []}

        audio_nums = len(meta_dict['audio_name'])
        audio_length = 10 * sampling_rate

        if split == 'train':
            # create vocabulary list
            all_captions.extend(meta_dict['caption'])

        start_time = time.time()

        with h5py.File(hdf5_path+'{}.h5'.format(split), 'w') as hf:
            hf.create_dataset('audio_name', shape=(audio_nums,), dtype='S20')
            hf.create_dataset('waveform', shape=(audio_nums, audio_length), dtype=np.float32)
            if split == 'train':
                hf.create_dataset('caption', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
            else:
                hf.create_dataset('caption', shape=(audio_nums, 5), dtype=h5py.special_dtype(vlen=str))

            for i in tqdm(range(audio_nums)):
                audio, _ = librosa.load(audio_dir + meta_dict['audio_name'][i], sr=sampling_rate, mono=True)
                audio = pad_or_truncate(audio, audio_length)

                hf['audio_name'][i] = meta_dict['audio_name'][i].encode()
                hf['waveform'][i] = audio
                hf['caption'][i] = meta_dict['caption'][i]

        logger.info(f'Packed {split} set to {hdf5_path} using {time.time() - start_time} s.')
    words_list, words_freq = _create_vocabulary(all_captions)
    logger.info(f'Creating vocabulary: {len(words_list)} tokens!')
    write_pickle_file(words_list, 'data/pickles/words_list.p')


def _sentence_process(sentence, add_specials=True):

    # transform to lower case
    sentence = sentence.lower()

    if add_specials:
        sentence = '<sos> {} <eos>'.format(sentence)

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    return sentence


def _create_vocabulary(captions):
    vocabulary = []
    for caption in captions:
        caption_words = caption.strip().split()
        vocabulary.extend(caption_words)
    words_list = list(set(vocabulary))
    words_list.sort(key=vocabulary.index)
    words_freq = [vocabulary.count(word) for word in words_list]

    return words_list, words_freq


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0: audio_length]

