#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from tools.utils import align_word_embedding
from models.AudioTransformer import AudioTransformer, AudioTransformer_80


def init_layer(layer):
    """ Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ACT(nn.Module):

    def __init__(self, config, ntoken):
        super(ACT, self).__init__()

        self.ntoken = ntoken

        # settings for encoder
        num_classes = 527
        patch_size = (4, 80)
        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_dim = 3072
        dropout = 0.2

        self.encoder = AudioTransformer_80(patch_size,
                                        num_classes,
                                        embed_dim,
                                        depth,
                                        num_heads,
                                        mlp_dim,
                                        dropout=dropout)
        if config.encoder.pretrained:
            pretrained_encoder = torch.load(config.path.encoder)['model']
            if config.encoder.model == 'deit':
                dict_new = self.encoder.state_dict().copy()
                trained_list = [i for i in pretrained_encoder.keys() if not ('head' in i or 'pos' in i)]
                for i in range(len(trained_list)):
                    dict_new[trained_list[i]] = pretrained_encoder[trained_list[i]]
                self.encoder.load_state_dict(dict_new)
            else:
                self.encoder.load_state_dict(pretrained_encoder)
        if config.encoder.freeze:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False

        # settings for decoder
        nhead = config.decoder.nhead
        nlayers = config.decoder.nlayers
        dim_feedforward = config.decoder.dim_feedforward
        activation = config.decoder.activation
        dropout = config.decoder.dropout
        self.nhid = config.decoder.nhid

        self.encoder_linear = nn.Linear(num_classes, self.nhid)

        self.pos_encoder = PositionalEncoding(self.nhid, dropout)

        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.word_emb = nn.Embedding(ntoken, self.nhid)
        self.dec_fc = nn.Linear(self.nhid, ntoken)

        # setting for pretrained word embedding
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False
        if config.word_embedding.pretrained:
            self.word_emb.weight.data = align_word_embedding(config.path.vocabulary,
                                                             config.path.word2vec,
                                                             self.nhid)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):
        """
        Args:
            src: spectrogram, batch x time x n_mels
        """
        src = self.encoder(src)  # batch x time x 527
        src = F.relu_(self.encoder_linear(src))  # batch x time x nhid
        src = src.transpose(0, 1)  # time x batch x nhid
        return src

    def decode(self, encoded_feats, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt: (batch_size, caption_length)
        # encoded_feats: (T, batch_size, nhid)

        tgt = tgt.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)
        tgt = self.pos_encoder(tgt)

        output = self.transformer_decoder(tgt, encoded_feats,
                                          memory_mask=input_mask,
                                          tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)

        return output

    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # src: spectrogram

        encoded_feats = self.encode(src)
        output = self.decode(encoded_feats, tgt,
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output
