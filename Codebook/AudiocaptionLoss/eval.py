#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import platform
import torch
import torch.nn as nn
import os
import time
import sys
from loguru import logger
import argparse
from tqdm import tqdm
from pathlib import Path
from data_handling.audiocaps_dataset import get_audiocaps_loader, get_generate_loader
from tools.config_loader import get_config
from tools.utils import *
from tools.file_io import load_pickle_file
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics
from models.TransModel import ACT
from tools.beam import beam_decode

def remove_small_value(top_list, spices, top_k):
    tmp_dict = {}
    for tl in top_list:
        tmp_dict[tl] = spices[tl]
    d_order = sorted(tmp_dict.items(),key=lambda x:x[1],reverse=True) 
    # print('tmp_dict ', tmp_dict)
    # print('d_order ', d_order)
    ans = []
    inx = 0
    for key in d_order:
        ans.append(key[0])
        inx += 1
        if inx >= top_k:
            break
    # print(ans)
    # assert 1==2
    return ans

def group_select(spices, top_k):
    # spices is a dict, which includes spice score and the filename
    # we aims to select top k file in each group
    top_dict = {}
    for key in spices.keys():
        # key is the name
        short_name = key.split('_mel')[0]
        if short_name not in top_dict.keys():
            top_dict[short_name] = [key]
        else:
            top_dict[short_name].append(key)
    ans = []
    for key in top_dict.keys():
        ans += remove_small_value(top_dict[key],spices, top_k)
    return ans


def eval_greedy(data):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):
            src, target_dicts, file_names = eval_batch
            src = src.to(device)
            output = greedy_decode(model, src, sos_ind=sos_ind, eos_ind=eos_ind)
            # print('output ',output[0])
            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)
            # print('y_hat_batch ',y_hat_batch.shape)
            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break
                    elif j == output.shape[1] - 1:
                        y_hat_batch[i, j] = eos_ind

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_dict.extend(target_dicts)
            file_names_all.extend(file_names)

        eval_time = time.time() - start_time
        # print('y_hat_all ',y_hat_all.shape)
        # assert 1==2
        # # print('ref_captions_dict ',ref_captions_dict)
        # # print('file_names_all ',file_names_all)
        # assert 1==2
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict,
                                                   file_names_all, words_list)
        # print('captions_pred ',captions_pred) # it includes filename
        # assert 1==2
        greedy_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = greedy_metrics['spider']['score']
        cider = greedy_metrics['cider']['score']
        spice = greedy_metrics['spice']['score']
        ciders = greedy_metrics['cider']['scores']
        spices = greedy_metrics['spice']['scores']
        selected_name = group_select(spices,5) #
        torch.save({'selected_name':selected_name},'./selected_vgg_audio_caps_299_top5.pth')
        print('selected_name ', len(selected_name))
        # assert 1==2
        less_list = []
        new_cider = 0.0
        new_spice = 0.0
        new_cnt = 0 
        tot_avg = 0.0
        for key in ciders.keys():
            tot_avg += ciders[key]
            if key not in selected_name:
                less_list.append(key)
            else:
                new_cider += ciders[key]
                new_spice += spices[key]
                new_cnt  += 1
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'spice: {spice:7.4f}')
        main_logger.info(f'Spider score using greedy search: {spider:7.4f}, eval time: {eval_time:.4f}')
        print('total ',len(ciders.keys()),tot_avg/len(ciders.keys()))
        print('len(less_list)', len(less_list))
        print('new cider ', new_cider/new_cnt)
        print('new spice ', new_spice/new_cnt)
        assert 1==2


def eval_beam(data, beam_size):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, target_dicts, file_names = eval_batch
            src = src.to(device)
            output = beam_decode(src, model, sos_ind, eos_ind, beam_width=beam_size)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):  # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break
                    elif j == output.shape[1] - 1:
                        y_hat_batch[i, j] = eos_ind

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_dict.extend(target_dicts)
            file_names_all.extend(file_names)

        eval_time = time.time() - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all, words_list, beam=True)
        beam_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = beam_metrics['spider']['score']
        cider = beam_metrics['cider']['score']
        main_logger.info(f'cider: {cider:7.4f}')
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')
        spiders.append(spider)
        if config.mode != 'eval':
            if beam_size == 3 and (epoch % 5) == 0:
                for metric, values in beam_metrics.items():
                    main_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')
            if spider >= max(spiders):
                torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "beam_size": beam_size,
                        "epoch": epoch,
                        }, str(model_output_dir) + '/best_model.pth'.format(epoch))
        else:
            if spider >= max(spiders):
                eval_metrics['metrics'] = beam_metrics
                eval_metrics['beam_size'] = beam_size


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Settings for ACT training')

parser.add_argument('-n', '--exp_name', type=str, default='exp1', help='name of the experiment')
parser.add_argument('-s', '--exp_path', type=str, default='/results', help='the path of generated sound')

args = parser.parse_args()

config = get_config()

setup_seed(config.training.seed)

exp_name = args.exp_name

# output setting
model_output_dir = Path('outputs', exp_name, 'model')
log_output_dir = Path('outputs', exp_name, 'logging')

model_output_dir.mkdir(parents=True, exist_ok=True)
log_output_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(str(log_output_dir) + '/captions.txt', format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 2,
           rotation=rotation_logger)

logger.add(str(log_output_dir) + '/beam_captions.txt', format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 3,
           rotation=rotation_logger)

main_logger = logger.bind(indent=1)

printer = PrettyPrinter()

device, device_name = (torch.device('cuda'),
                       torch.cuda.get_device_name(torch.cuda.current_device())) \
    if torch.cuda.is_available() else ('cpu', platform.processor())

main_logger.info(f'Process on {device_name}')

words_list = load_pickle_file(config.path.vocabulary)

# training_data = get_audiocaps_loader('train', config)
# validation_data = get_audiocaps_loader('val', config)
# test_data = get_audiocaps_loader('test', config)
mel_path = args.exp_path
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-02-09T00-13-01_caps_transformer/samples_2022-02-10T10-02-37/caps_validation/cls_0/'
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/OUTPUT/caps_train/2022-04-26T22-50-34/samples_2022-05-01T10-12-12/caps_validation/'
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-04-21T20-58-13_caps_transformer_2048/samples_2022-04-22T15-06-51/caps_validation/cls_0/'
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-05-01T15-06-07_caps_transformer_512/samples_2022-05-01T21-30-15/caps_validation/cls_0/'
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-05-05T19-28-48_caps_transformer/samples_2022-05-05T23-42-33/caps_validation/cls_0/'
#mel_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-02-07T19-25-31_caps_transformer/samples_2022-02-08T10-49-10/caps_validation/cls_0/'
test_data = get_generate_loader(mel_path, config)
print('mel_path ', mel_path)
ntokens = len(words_list)
sos_ind = words_list.index('<sos>')
eos_ind = words_list.index('<eos>')

main_logger.info('Training setting:\n'
                 f'{printer.pformat(config)}')

model = ACT(config, ntokens)
model.to(device)

main_logger.info(f'Model:\n{model}\n')
main_logger.info('Total number of parameters:'
                 f'{sum([i.numel() for i in model.parameters()])}')

#main_logger.info(f'Len of validation data: {len(validation_data)}')
main_logger.info(f'Len of test data: {len(test_data)}')


spiders = []

eval_metrics = {}
main_logger.info('Evaluation mode')
model.load_state_dict(torch.load(config.path.eval_model)['model'])
main_logger.info(f'Weights loaded from {config.path.eval_model}')
eval_greedy(test_data)
eval_beam(test_data, beam_size=2)
eval_beam(test_data, beam_size=3)
main_logger.info(f"Best metrics with beam size {eval_metrics['beam_size']}:")
for metric, values in eval_metrics['metrics'].items():
    main_logger.info(f'{metric:<7s}: {values["score"]:7.4f}')
