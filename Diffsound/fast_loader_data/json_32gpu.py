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
# this code is used to split big data to 16GPUs
# total_num = 849920
# batch_size = 20
# chunk_size = 166 batch
# card_size = 16 chunk
# json style
#####
# num_chunk: 16
# chunks: {
#    chunk1:{
#     num_batches: 166,
#     batches: [{ 
#       uuid: {
#           feats: the path
#           text: "sentence1 \t sentence2....."
#       }
#     },{},.....]
#    }
# }
#####
train_fl = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audioset/filenames_small.pickle'
chunk_path = '/apdcephfs/share_916081/jerrynchen/ydc_data/split_16gpu/'
text_path = '/apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/data_root/audioset/'
json_path = '/apdcephfs/share_1316500/donchaoyang/data/audioset/split_32gpu_json/'
name_list = pickle.load(open(train_fl, 'rb'), encoding="bytes")
GPUs = 32
point_index = 0
mask_text_ph = '/apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/data_root/audioset/text/mask_text.pth'
mask_dict = torch.load(mask_text_ph)
# print(mask_dict.items())
# assert 1==2
for gpu in range(GPUs):
    num_chunks = 8
    batch_size = 20
    chunk_size = 166
    over_all = {}
    over_all['num_chunks'] = num_chunks
    chunks = {}
    chunk_ls = [{} for i in range(num_chunks)]
    for i in range(num_chunks):
        print('chunk id ', i)
        tmp_chunk_dict = chunk_ls[i]
        tmp_chunk_dict['num_batches'] = chunk_size
        tmp_chunk_dict['batches'] = []
        for j in range(chunk_size):
            print('bath id ',j)
            batch_dict = {}
            for k in range(batch_size):
                utt_id = name_list[point_index]
                utt_dict = {}
                chunk_id = (point_index // (chunk_size*batch_size))
                utt_dict['feats'] = chunk_path + 'audioset_mel_chunk_' + str(chunk_id+1) + '.pth'

                # caption_path = os.path.join(text_path, 'text', 'train', utt_id + '.txt')
                # with open(caption_path, 'r') as f:
                #     caption = f.readlines()
                # save_text = ""
                # for cap in caption:
                #     cap = cap.replace('\n', '')
                #     save_text += cap
                #     save_text += '\t'
                # save_text = save_text[:-1] # delete the last '\t'
                # #print(save_text)
                # save_text_ls = save_text.split('\t')
                # print(save_text_ls)
                save_text = mask_dict[utt_id]
                # print('save_text ',save_text)
                # assert 1==2
                # assert 1==2
                utt_dict['text'] = save_text
                batch_dict[utt_id] = utt_dict
                print('point_index ', point_index)
                point_index += 1
            tmp_chunk_dict['batches'].append(batch_dict) # 
            #assert 1==2
        chunk_name = 'chunk_' + str(i)
        chunks[chunk_name] = tmp_chunk_dict

    over_all['chunks'] = chunks
    save_json = 'data_gpu_' + str(gpu) + '.json'
    with open(json_path + save_json,"w") as f:
        json.dump(over_all,f)
        print("over")
    over_all.clear()
    #assert 1==2