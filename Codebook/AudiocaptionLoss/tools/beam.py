#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Adapted from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding,
and https://github.com/haantran96/wavetransformer/blob/main/modules/beam.py
"""

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        """
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(x, model, sos_ind, eos_ind, beam_width=5, top_k=1):
    """

    Args:
        x: input spectrogram (batch_size, time_frames, n_mels)
        model:
        sos_ind: index of '<sos>'
        eos_ind: index of '<eos>'
        beam_width: beam size
        top_k: how many sentences wanted to generate

    Returns:

    """
    decoded_batch = []

    device = x.device
    batch_size = x.shape[0]

    encoded_features = model.encode(x)
    # audio features extracted by encoder, (time_frames, batch, nhid)

    # decoding goes sentence by sentence
    for idx in range(batch_size):

        encoded_feature = encoded_features[:, idx, :].unsqueeze(1)
        # (time_frames, 1, n_hid)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_ind]]).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))

        # starting node -  previous node, word_id (sos_ind), logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid

            if n.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output = model.decode(encoded_feature, decoder_input)
            log_prob = F.log_softmax(decoder_output[-1, :], dim=-1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(log_prob, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose n_best paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordid[0, :])
            # # back trace
            # while n.prevNode != None:
            #     n = n.prevNode
            #     utterance.append(n.wordid)
            #
            # utterance = utterance[::-1]
            # utterances.append(utterance)
        for i in range(top_k):
            decoded_batch.append(utterances[i])

    return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)
