# Author: Dongchao Yang
# Reference: Jingchuang Tian's ASR code

import torch
import json
import copy
import random
import numpy as np
import torch.distributed as dist
import albumentations

class Crop(object):
    def __init__(self, cropped_shape=None, random_crop=False):
        self.cropped_shape = cropped_shape
        if cropped_shape is not None:
            mel_num, spec_len = cropped_shape
            if random_crop:
                self.cropper = albumentations.RandomCrop
            else:
                self.cropper = albumentations.CenterCrop
            self.preprocessor = albumentations.Compose([self.cropper(mel_num, spec_len)])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __call__(self, item):
        item['input'] = self.preprocessor(image=item['input'])['image']
        return item
    
class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

# Defined data macros. Make sure this is compatible with your training code
MAX_FRAME = 860 # actually, all of the data feature share the same length
sample_rate = 22050
MAX_SAMPLES = 10 * sample_rate # the max length of your wav file
MAX_TOKEN = 400
IGNORE_TOKEN_ID = -1
frame_num = 860
spec_crop_len = 848
random_crop = False

def select_fn(feats, lengths):
    select_strategy = 'f0_f1'
    select_keys = select_strategy.strip().split("_")
    #select_keys = 
    ans = []
    for key in select_keys: # f,l,f,f,f
        index = int(key.replace("f", "").replace("l", ""))
        if key.startswith("f"):
            ans.append(feats[index]) # 0
        elif key.startswith("l"):
            ans.append(lengths[index])
        else:
            raise NotImplementedError
    return ans


def custom_collate_fn(batch_data):
    """ Splice multiple features into a mini-batch """

    assert len(batch_data) == 1, "We only support batch_size=1"
    batch_data = batch_data[0] # [[],[],[]]

    bsz = len(batch_data[0]) # 查看预定义的bsz
    ans, lengths = [], []
    for feats in batch_data: #开始遍历里面的每种特征
        # Strings
        if isinstance(feats[0], str): # 若该特征是文本
            spliced_feats = feats # we directly use

        # IDs
        elif isinstance(feats[0], list) and isinstance(feats[0][0], int): # we not use it now
            spliced_feats = np.ones([bsz, MAX_TOKEN], dtype=np.int32) * IGNORE_TOKEN_ID
            max_length = 0
            for i, feat in enumerate(feats):
                spliced_feats[i, :len(feat)] = np.array(feat, dtype=np.int32)
                max_length = max(max_length, len(feat))
            spliced_feats = spliced_feats[:, :max_length]
            spliced_feats = torch.Tensor(spliced_feats).long()

        # acoustic features
        elif isinstance(feats[0], np.ndarray) and feats[0].ndim == 3: # make sure your acoustic features is a matric
            # max_length, dim = 0, feats[0].shape[1] # make sure the last dim is time dim 
            # print('max_length ', max_length)
            # print('dim ',dim)
            # assert 1==2
            # spliced_feats = np.zeros([bsz, dim, MAX_FRAME])
            # for i, feat in enumerate(feats): # if the time dim of you data is same, this step is not needed
            #     spliced_feats[i, :len(feat)] = feat
            #     max_length = max(max_length, len(feat))
            # spliced_feats = spliced_feats[:, :max_length]
            #feats = feats[:,None,:,:] # transfer to 3 dim
            spliced_feats = torch.Tensor(feats).float()

        # raw wav
        elif isinstance(feats[0], np.ndarray) and feats[0].ndim == 1: # not use now
            max_length = 0
            spliced_feats = np.zeros([bsz, MAX_SAMPLES])
            for i, feat in enumerate(feats):
                spliced_feats[i, :len(feat)] = feat
                max_length = max(max_length, len(feat))
            spliced_feats = spliced_feats[:, :max_length]
            spliced_feats = torch.Tensor(spliced_feats).float()

        else:
            raise NotImplementedError(f"type type(feats[0]) is not supported")

        ans.append(spliced_feats)

        # length statistics
        try: # calculate the length of orginal data
            length = torch.Tensor([len(x) for x in feats]).long()
        except:
            length = torch.zeros([bsz]).long()
        lengths.append(length)

        # select and reorder: A custom function
    ans = select_fn(ans, lengths)
    return {'image': ans[0],'text':ans[1]}
    #return tuple(ans) # transfer to tuple type

# def get_custom_collate_fn(select_strategy):
#     """ Return a collate_fn that splices the batch features and partially selects
#         these features along with their lengths as specified by select_strategy
        
#         select_strategy: a string like f0_l0_f1
#         which means the returned batch is a list of: feature 1; lengths 1; feature 2
#     """    
#     #select_keys = select_strategy.strip().split("_") # f0_l0_f1_f2_f3 -> f0, l0, f1, f2, f3
    

#     return custom_collate_fn

        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_json, first_n_batches=-1):
        self.data_dict = json.load(open(data_json, 'rb'))
        self.first_n_batches = first_n_batches
        # Meta-data
        (
        self.len,
        self.ark_size,
        self.feature_keys
        ) = self._parse_metadata()
        
        self.ark_names = list(self.data_dict["chunks"].keys()) # ['data/dev/log/wav.1.scp', 'data/dev/log/wav.4.scp']
        #print('self.ark_names ', self.ark_names)
        self._buffer = {k: {} for k in self.feature_keys}

        print(f"Dataset Info: Total batches: {self.len} | Ark size: {self.ark_size}")
        print(f"              Feature keys: {' '.join(self.feature_keys)}")

        # Preprocess
        self.preprocessing =  CropImage([80, spec_crop_len], random_crop)
        #self.preprocess_args = preprocess_args
        # if preprocess_args is not None:
        #     assert isinstance(preprocess_args, dict)

        if "wav" in self.feature_keys:
            self.wave_pipeline = WavePipeline(speed_perturb=True, utterance_cmvn=True) 

    def _parse_metadata(self):
        # Check compatibility
        ark_count, batch_count = 0, 0
        for ark in self.data_dict["chunks"].values():
            ark_count += 1 # 记录块的数量
            for batch in ark["batches"]: # 记录batch数量
                batch_count += 1
        assert ark_count == self.data_dict["num_chunks"]
        assert batch_count == ark_count * ark["num_batches"]
        
        # An example
        info = list(list(batch.values())[0].keys()) # ['feats', 'text']
        #print('info ', info)
        return batch_count, ark["num_batches"], info

    def _load_and_cache(self, uttid, feat_key, content):
        """ Load the whole kaldi ark if any utterance in it is accessed """
        """ OOM can be avoided as long as the sampler is well designed  """
        ark_path = content # 
        if uttid not in self._buffer[feat_key]:
            if feat_key in ["feats", "wav", "feats_org", "wav_org"]:
                data_iter = torch.load(ark_path)
            else:
                raise NotImplementedError
            for k, v in data_iter.items(): # add it into buffer
                self._buffer[feat_key][k] = v
        #print('uttid after ', uttid)
        data = copy.deepcopy(self._buffer[feat_key][uttid])
        del self._buffer[feat_key][uttid]

        return data

    def __len__(self): # self.len total batch size
        return self.len if self.first_n_batches <= 0 else self.first_n_batches

    def __getitem__(self, index):
        # data info
        ark_name = self.ark_names[index // self.ark_size] # 查属于那个块
        in_ark_id = index % self.ark_size # batch id self.ark_size = 120
        batch_info = self.data_dict["chunks"][ark_name]["batches"][in_ark_id] # 第几个batch
        # load data
        return_batch = [[] for _ in self.feature_keys] # []
        for uttid, info in batch_info.items(): # 遍历该batch
            for feat_id, (feat_key, content) in enumerate(info.items()): # 遍历该batch中的数据
                # All numerical features that need caching and buffering use this
                # E.g., any matrix input, FST graph etc.
                if feat_key in ["feats", "wav"]:
                    data = self._load_and_cache(uttid, feat_key, content) # 
                    if self.preprocessing is not None: # specially for my TTS 
                        item = {}
                        item['input'] = data
                        item = self.preprocessing(item)
                        data = 2 * item['input'] - 1 
                        data = data[None,:,:] # unsqueeze ?
                elif feat_key in ["text"]: # direct get
                    #data_tmp = content.strip()
                    data_ls = content.split('\t')
                    data = random.choice(data_ls).replace('\n','') # 
                else:
                    raise NotImplementedError("Unrecognized feature key")   
                return_batch[feat_id].append(data) # 0,1,2 ..分别代表第几种特征
        return return_batch


class CustomSampler(object):
    def __init__(self, random_seed, buffer_size, dataset_size, ark_size, prefetch_ratio=0.3):
        self.buffer_size = buffer_size # 一次性读入内存的数量
        self.dataset_size = dataset_size # the number of batch
        self.ark_size = ark_size # self.ark_size = 120, 一个块里面的bacth数量
        self.num_arks = self.dataset_size // self.ark_size # 有多少个块
        self.prefetch_number = int(prefetch_ratio * self.ark_size * self.buffer_size) # 内存中的最少数量
        assert dataset_size % ark_size == 0, "The number of batches in some arks are wrong"

        try:
            self.seed2 = dist.get_rank() # ?
        except:
            print("Sampler: you are not using DDP training paradigm.")
            print("Sampler: So the rank-specific seed is set to 0", flush=True)
            self.seed2 = 0

        self.refresh(seed=random_seed)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.dataset_size

    def _get_indices(self):
        # Divided arks into groups. Identical results among GPUs
        ark_ids = list(range(self.num_arks)) # 块数量 0,1...
        random.shuffle(ark_ids) # 打乱
        groups = []
        start = 0
        while start < self.num_arks: # 
            end = min(start + self.buffer_size, self.num_arks)
            groups.append(ark_ids[start: end]) # st:st_buffer_size为一个组
            start += self.buffer_size # 设num_arks=10,buffer_size=2,那么 [0,1],[2,3] ....

        def process_group(ark_ids):
            # idx1 is the same for all GPUs
            idx1s = list(range(len(ark_ids))) * self.ark_size # [0,1]**ark_size
            random.shuffle(idx1s) # 

            # idx2 is different among GPUs            
            random.seed(self.seed + self.seed2)
            idx2s = [list(range(self.ark_size)) for _ in ark_ids] # [[0,..., ark_size-1], [0,..., ark_size-1] ]
            for x in idx2s:
                random.shuffle(x) # 打乱 0, 1, ..., ark_size-1
            random.seed(self.seed)

            ans = []
            for idx1 in idx1s: # a index that from [0,len(ark_ids)-1], 意思是随机选一个块
                idx2 = idx2s[idx1].pop() # 选出该块的indexs
                idx = ark_ids[idx1] * self.ark_size + idx2 # 若是第0块，那么index就是[0,ark_size-1],若是第一块,....
                ans.append(idx)

            return ans

        ans = []
        for group in groups:
            ans.extend(process_group(group))
        assert sum(ans) == self.dataset_size * (self.dataset_size - 1) / 2
        return ans

    def refresh(self, seed=None):
        seed = seed if seed is not None else self.seed + 1
        self.seed = seed
        random.seed(seed) # 设置随机种子？
        self.indices = self._get_indices() 

class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass


class CustomDataloader(object):
    def __init__(self,
                 data_json, 
                 select_strategy="f0_f1",
                 random_seed=0,
                 shuffle=True,
                 buffer_size=60,
                 prefetch_ratio=0.3,
                 first_n_batches=-1,
    ):
        """
        Args:
            data_json: path to the json file
            select_strategy: the strategy to select features and their lengths
            random_seed: random seed for sampler.
            shuffle: If true, the data iterator will be shuffled.
            buffer_size: number of arks buffered in the memory.
            prefetch_ratio: the minimum ratio between the number of buffered batches 
                and the buffer capacity. more arks will be load when below this ratio.
            first_n_batches: if > 0, only output first n_batches for debug    

        return:
            A data iterator
    
        Hint: You cannot set batch-size here. We use the dynamic batch strategy during
              the generation of data_json. 
        """
        
        self.dataset = Dataset(data_json, first_n_batches)
        assert isinstance(self.dataset, torch.utils.data.Dataset)
            

        if shuffle:
            self.sampler = CustomSampler(random_seed=random_seed, 
                              buffer_size=buffer_size,
                              dataset_size=len(self.dataset),
                              ark_size=self.dataset.ark_size,
                              prefetch_ratio=prefetch_ratio,
            )
            self.prefetch_number = self.sampler.prefetch_number
        else:
            self.sampler = SequentialSampler(
                list(range(len(self.dataset)))
            )
            self.prefetch_number = 100
      
        #self.custom_collate_fn = get_custom_collate_fn(select_strategy)
        self.custom_collate_fn = custom_collate_fn
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            sampler=self.sampler,
            num_workers=1,
            prefetch_factor=self.prefetch_number,
            collate_fn=self.custom_collate_fn,
        )

        self.epoch = 0
        self.len = len(self.dataset)
        self.current_position = 0
        self.iter = None
        
    def serialize(self, serializer):
        """Serialize and deserialize function."""
        epoch = serializer("epoch", self.epoch)
        current_position = serializer("current_position", self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    # Called by chainer every after an epoch
    def start_shuffle(self):
        self.sampler.refresh()
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            sampler=self.sampler,
            num_workers=1,
            prefetch_factor=self.prefetch_number,
            collate_fn=self.custom_collate_fn,
        )
   
    # Called by chainer to monitor the training process
    @property
    def epoch_detail(self):
        """Epoch_detail required by chainer."""
        return self.epoch + self.current_position / self.len
    
    def __iter__(self):
        for b in self.dataloader:
            yield b

    # Make it an endless dataloader
    def next(self):
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()

        self.current_position += 1
        if self.current_position == self.len:
            self.epoch += 1
            self.current_position = 0

        return ret

    def finalize(self):
        del self.dataset
        del self.sampler
        del self.dataloader

if __name__ == "__main__":
    dataloader = CustomDataloader("/apdcephfs/share_1316500/donchaoyang/data/audioset/split_16gpu_json/data_gpu_2.json", buffer_size=4, shuffle=True)
    for itr, batch in enumerate(dataloader):
        print('itr ',itr)
        print("xs: ", batch['image'].shape)
        print("text: ", len(batch['text']))
    # for i in range(721):
    #     batch = dataloader.next()
    #     print("xs: ", batch['image'].shape)
    #     print("text: ", len(batch['text']))
    #     break
