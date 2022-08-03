import random
import argparse
#import librosa
from tqdm import tqdm
import io
import logging
from pathlib import Path
#import pandas as pd
import numpy as np
#import soundfile as sf
#from pypeln import process as pr
import gzip
#import h5py
import os
#from pydub import AudioSegment
import json
import pickle
import torch
batch_size = 20
chunk_size = 166 # 166个batch
train_fl = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audioset/filenames_small.pickle'
mel_path = '/apdcephfs/share_1316500/donchaoyang/data/audioset/features/train/melspec_10s_22050hz/'
name_list = pickle.load(open(train_fl, 'rb'), encoding="bytes")
root_path = '/apdcephfs/share_916081/jerrynchen/ydc_data/split_16gpu/'
save_dict = {}
bag_id = 1
for idx, name in enumerate(name_list):
    print(idx, name)
    ph = mel_path+name+'_mel.npy'
    data = np.load(ph)
    # print('data ',data.shape)
    # assert 1==2
    save_dict[name] = data

    if (idx+1) % (batch_size*chunk_size) == 0:
        save_name = 'audioset_mel_chunk_' + str(bag_id) + '.pth' 
        torch.save(save_dict, root_path+save_name)
        print('success save ', save_name)
        save_dict.clear()
        bag_id += 1
        #assert 1==2