import torch
from pybert.configs.basic_config import config
from pybert.io.bert_processor import BertProcessor
from pybert.model.bert_for_multi_label import BertForMultiLable
import pickle
import numpy as np
import pandas as pd
def main(text,arch,max_seq_length,do_lower_case):
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=do_lower_case)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    #model = BertForMultiLable.from_pretrained(config['checkpoint_dir'] /f'{arch}', num_labels=len(label_list))
    model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list)) # without train
    model = model.cuda()
    tokens = processor.tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1, 2 choices
    input_ids = input_ids.cuda()
    logits = model.extract(input_ids)
    return logits.detach().cpu().numpy()[0]

if __name__ == "__main__":
    data = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code3/Bert-Multi-Label-Text-Classification/pybert/caps_dataset/val.tsv',sep='\t',usecols=[0,1,2])
    save_ls = []
    for row in data.values:
        text = row[1]
        name = row[0][:-4]
        # text = ''''"FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!"'''
        max_seq_length = 256
        do_loer_case = True
        arch = 'bert'
        pre_path = '/apdcephfs/share_1316500/donchaoyang/code3/Bert-Multi-Label-Text-Classification/pybert/caps_dataset/text_no_train/val/cls_token_768/'
        probs = main(text,arch,max_seq_length,do_loer_case)
        # print('probs ',probs.shape)
        print(pre_path+name)
        if pre_path+name not in save_ls:
            save_ls.append(pre_path+name)
            k = 1
            np.savetxt(pre_path+name+str(k)+'.txt',probs)
        else:
            k += 1
            np.savetxt(pre_path+name+str(k)+'.txt',probs)
        # assert 1==2
        
