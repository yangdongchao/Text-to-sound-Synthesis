# ------------------------------------------
# Diffsound
# written by Dongchao Yang
# ------------------------------------------

import torch
import math
from torch import nn
from sound_synthesis.utils.misc import instantiate_from_config
import time
import numpy as np
from PIL import Image
import os

from torch.cuda.amp import autocast

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DALLE(nn.Module):
    def __init__(
        self,
        *,
        content_info={'key': 'image'},
        condition_info={'key': 'text'},
        content_codec_config,
        condition_codec_config,
        first_stage_permuter_config,
        diffusion_config
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info = condition_info
        #self.content_codec = instantiate_from_config(content_codec_config)
        self.init_content_codec_from_ckpt(content_codec_config)
        self.condition_codec = instantiate_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.first_stage_permuter = instantiate_from_config(config=first_stage_permuter_config)
        # print(' self.transformer ', self.transformer)
        self.truncation_forward = False

    def init_content_codec_from_ckpt(self, content_codec_config):
        model = instantiate_from_config(content_codec_config) # 得到第一阶段的模型
        model = model.eval()
        model.train = disabled_train # 重写model的train函数，确保该模型不会被train
        self.content_codec = model
    
    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer
    
    def get_tokens(self, spec): #
        quant_z, _, info = self.content_codec.encode(spec) # spec to index?
        indices = info[2].view(quant_z.shape[0], -1)
        # print('indices ',indices.shape)
        indices = self.first_stage_permuter(indices)
        self.zshape = quant_z.shape
        # print('indices ',indices.shape)
        return quant_z, indices
    
    def decode_to_img(self, index, zshape, stage='first'): # under review
        if stage == 'first':
            index = self.first_stage_permuter(index, reverse=True)
        else:
            raise NotImplementedError
        # elif stage == 'cond':
        #     print('in cond stage in decode_to_img which is unexpected ')
        #     index = self.cond_stage_permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.content_codec.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        x = self.content_codec.decode(quant_z)
        return x
    
    @torch.no_grad()
    def prepare_condition(self, batch, condition=None):
        cond_key = self.condition_info['key'] # text
        cond = batch[cond_key] if condition is None else condition # get text
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.condition_codec.get_tokens(cond) # transfer to token (number)
        # what the cond?
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v # e.g. condition_mask, condition_token
        return cond_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False):
        #print('with_mask ',with_mask) # False
        cont_key = self.content_info['key'] # image
        cont = batch[cont_key] # b,3,256,256
        if torch.is_tensor(cont):
            cont = cont.to(self.device)
        if not with_mask: # return dict{'token': }
            quant_z, indices = self.get_tokens(cont)
            # print('indices ',indices.shape)
            cont = {'token': indices,'quant': quant_z} # 32*32=1024 for image. sepc:53*5=265
        else:
            mask = batch['mask'.format(cont_key)]
            cont = self.content_codec.get_tokens(cont, mask, enc_with_mask=False)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        return cont_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_input(self, batch):
        input = self.prepare_condition(batch)
        input.update(self.prepare_content(batch))
        return input

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True) 
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper

        else:
            print("wrong sample type")

    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="top0.85r"):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        if replicate != 1: # 重复多少次?
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
        # print(condition)
        # assert 1==2
        content_token = None

        if len(sample_type.split(',')) > 1: # using r,fast
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.predict_start = self.predict_start_with_truncation(self.transformer.predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='fast':
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(sample_type.split(',')[1][4:]))

        else:
            trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token,
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_att_weight=return_att_weight,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type)
        zshape = (trans_out['content_token'].shape[0], 256, 5, 53)
        # print(trans_out['content_token'].shape)
        # assert 1==2
        content = self.decode_to_img(trans_out['content_token'], zshape)
        #content = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
        self.train()
        out = {
            'content': content
        }
        

        return out

    @torch.no_grad()
    def reconstruct(
        self,
        input):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        batch,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0, 0.5, 1.0],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        **kwargs):
        self.eval()
        condition = self.prepare_condition(batch)
        content = self.prepare_content(batch)

        content_samples = {'input_image': batch[self.content_info['key']]}
        zshape = content['content_quant'].shape
        # print('zshape ',zshape)
        if return_rec:
            # bhwc = (zshape[0], zshape[1], zshape[2], zshape[3]) # 应该先获得未编码前的特征维度 ([b, 256, 5, 53])
            # quant_z = self.content_codec.quantize.get_codebook_entry(content['content_token'].reshape(-1), shape=bhwc)
            # quant_z = quant_z.permute(0,2,3,1)
            # quant_z = self.decode_to_img(content['content_token'],zshape)
            # x = self.content_codec.decode(quant_z)
            #content_samples['reconstruction_image'] = self.content_codec.decode(content['content_token'])  
            content_samples['reconstruction_image'] = self.decode_to_img(content['content_token'], zshape)
        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int((content['content_token'].shape[1] * cr)) # 265*cr
                if num_content_tokens < 0:
                    continue
                else:
                    content_token = content['content_token'][:, :num_content_tokens] # 按比例保留部分的token
                if sample_type == 'debug':
                    trans_out = self.transformer.sample_debug(condition_token=condition['condition_token'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                else:
                    trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                #content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.content_codec.decode(trans_out['content_token']) # 根据预测值,进行解码
                # bhwc = (zshape[0], zshape[2], zshape[3], zshape[1]) # 应该先获得未编码前的特征维度 ([b, 256, 5, 53])
                # quant_z = self.content_codec.quantize.get_codebook_entry(trans_out['content_token'].reshape(-1), shape=bhwc)
                content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.decode_to_img(trans_out['content_token'], zshape)
                if return_att_weight:
                    content_samples['cond1_cont{}_fr{}_image_condition_attention'.format(cr, fr)] = trans_out['condition_attention'] # B x Lt x Ld
                    content_att = trans_out['content_attention']
                    shape = *content_att.shape[:-1], self.content.token_shape[0], self.content.token_shape[1]
                    content_samples['cond1_cont{}_fr{}_image_content_attention'.format(cr, fr)] = content_att.view(*shape) # B x Lt x Lt -> B x Lt x H x W
                if return_logits:
                    content_samples['logits'] = trans_out['logits']
        self.train() 
        output = {'condition': batch[self.condition_info['key']]}  # 同时返回text和预测的image
        output.update(content_samples)
        return output

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        # print('batch ',batch['image'].shape)
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        # print('output ',output)
        # assert 1==2
        return output
