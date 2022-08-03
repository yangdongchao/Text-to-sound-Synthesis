import numpy as np
import random
f_r = open('/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audioset/txt/audioset_train.txt','r')
valid = []
with open("audioset_train.txt","w") as f:  
    for line in f_r:
        line = line.replace('\n','')
        line_ls = line.split('/')
        # print(audio)
        # assert 1==2
        p = random.random()
        if p > 0.01:
            f.write(line_ls[-1]+'\n')
        else:
            valid.append(line_ls[-1])

with open("audioset_valid.txt","w") as f_v:
    for audio in valid:
        f_v.write(audio+'\n')
