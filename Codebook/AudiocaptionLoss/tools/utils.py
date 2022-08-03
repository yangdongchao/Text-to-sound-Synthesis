#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from loguru import logger
from gensim.models.word2vec import Word2Vec
from tools.file_io import load_pickle_file


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rotation_logger(x, y):
    """Callable to determine the rotation of files in logger.

    :param x: Str to be logged.
    :type x: loguru._handler.StrRecord
    :param y: File used for logging.
    :type y: _io.TextIOWrapper
    :return: Shall we switch to a new file?
    :rtype: bool
    """
    return 'Captions start' in x


def set_tgt_padding_mask(tgt, tgt_len):
    # tgt: (batch_size, max_len)
    # tgt_len: list() length for each caption in the batch
    batch_size = tgt.shape[0]
    max_len = tgt.shape[1]
    mask = torch.zeros(tgt.shape).type_as(tgt).to(tgt.device)
    for i in range(batch_size):
        num_pad = max_len - tgt_len[i]
        mask[i][max_len - num_pad:] = 1

    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()

    return mask


def greedy_decode(model, src, sos_ind=0, eos_ind=9, max_len=30):

    model.eval()
    with torch.no_grad():
        batch_size = src.shape[0]
        encoded_feats = model.encode(src)

        ys = torch.ones(batch_size, 1).fill_(sos_ind).long().to(src.device)

        for i in range(max_len):
            target_mask = model.generate_square_subsequent_mask(ys.shape[1]).to(src.device)
            out = model.decode(encoded_feats, ys, target_mask=target_mask)  # T_out, batch_size, ntoken
            prob = F.softmax(out[-1, :], dim=-1)
            next_word = torch.argmax(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
    return ys


def decode_output(predicted_output, ref_captions, file_names, words_list, beam=False):

    if beam:
        caption_logger = logger.bind(indent=3)
        caption_logger.info('Captions start')
        caption_logger.info('Beam search:')
    else:
        caption_logger = logger.bind(indent=2)
        caption_logger.info('Captions start')
        caption_logger.info('Greedy search:')

    captions_pred, captions_gt, f_names = [], [], []

    caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

    for pred_words, ref_cap_dict, f_name in zip(predicted_output, ref_captions, file_names):
        pred_cap = [words_list[i] for i in pred_words]

        try:
            pred_cap = pred_cap[:pred_cap.index('<eos>')]
        except ValueError:
            pass

        pred_cap = ' '.join(pred_cap)

        f_names.append(f_name)
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})
        ref_cap_dict.update({'file_name': f_name})
        captions_gt.append(ref_cap_dict)
        gt_caps = [ref_cap_dict[cap_ind] for cap_ind in caption_field]

        log_strings = [f'Captions for file {f_name}:',
                       f'\t Predicted caption: {pred_cap}',
                       f'\t Original caption_1: {gt_caps[0]}',
                       f'\t Original caption_2: {gt_caps[1]}',
                       f'\t Original caption_3: {gt_caps[2]}',
                       f'\t Original caption_4: {gt_caps[3]}',
                       f'\t Original caption_5: {gt_caps[4]}']

        [caption_logger.info(log_string)
         for log_string in log_strings]

    return captions_pred, captions_gt


def align_word_embedding(words_list_path, model_path, nhid):

    words_list = load_pickle_file(words_list_path)
    w2v_model = Word2Vec.load(model_path)
    ntoken = len(words_list)
    weights = np.zeros((ntoken, nhid))
    for i, word in enumerate(words_list):
        embedding = w2v_model.wv[word]
        weights[i] = embedding
    weights = torch.from_numpy(weights).float()
    return weights


# https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
# When smoothing=0.0, the output is almost the same as nn.CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target.data == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
