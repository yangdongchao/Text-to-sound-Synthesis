from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from sound_synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle
from specvqgan.modules.losses.vggishish.transforms import Crop
import torch
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class CapsDataset(Dataset):
    def __init__(self, data_root, phase = 'train', mel_num=80,
                 spec_len=860, spec_crop_len=848, random_crop=False, im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.caps_feature_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/features'
        if phase=='train':
            tmp_phase='train'
        else:
            tmp_phase='val'
        self.image_folder = os.path.join(self.caps_feature_path, tmp_phase, 'melspec_10s_22050hz')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)
        self.num = len(self.name_list)

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index] # 
            # print('name ',name)
            this_text_path = os.path.join(data_root, 'text', phase, name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption

        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, name+'_mel.npy')
        spec = np.load(image_path) # 加载mel spec
        item = {}
        item['input'] = spec
        if self.transforms is not None: # 
            item = self.transforms(item)
        image = 2 * item['input'] - 1 # why --> it also expects inputs in [-1, 1] but specs are in [0, 1]
        # image = load_img(image_path)
        #image = np.array(image).astype(np.uint8)
        # image = self.transform(image = image)['image']
        image = image[None,:,:]
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        # print('image ',image.shape)
        # print('caption ',caption)
        # assert 1==2
        data = {
                'image': image.astype(np.float32),
                'text': caption,
        }
        
        return data


class CapsDatasetAll(Dataset):
    def __init__(self, data_root, phase = 'train', mel_num=80,
                 spec_len=860, spec_crop_len=848, random_crop=False, im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.caps_feature_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/features'
        if phase=='train':
            tmp_phase='train'
        else:
            tmp_phase='val'
        self.image_folder = os.path.join(self.caps_feature_path, tmp_phase, 'melspec_10s_22050hz')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)
        self.num = len(self.name_list)

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index] # 
            # print('name ',name)
            this_text_path = os.path.join(data_root, 'text', phase, name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption
        h5_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/feature_h5/'
        self.feats_dict = {}
        for i in range(5):
            name = 'train' + str(i+1) + '.pth'
            tmp_data = torch.load(h5_path + name)
            for k,v in tmp_data.items():
                self.feats_dict[k] = v
        
        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        # image_path = os.path.join(self.image_folder, name+'_mel.npy')
        # spec = np.load(image_path) # 加载mel spec
        spec = self.feats_dict[name]
        item = {}
        item['input'] = spec
        if self.transforms is not None: # 
            item = self.transforms(item)
        image = 2 * item['input'] - 1 # why --> it also expects inputs in [-1, 1] but specs are in [0, 1]
        # image = load_img(image_path)
        #image = np.array(image).astype(np.uint8)
        # image = self.transform(image = image)['image']
        image = image[None,:,:]
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        # print('image ',image.shape)
        # print('caption ',caption)
        # assert 1==2
        data = {
                'image': image.astype(np.float32),
                'text': caption,
        }
        
        return data