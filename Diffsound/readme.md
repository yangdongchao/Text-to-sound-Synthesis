### Diffsound
Make sure you have the readme part on Codebook part. Our data and evaluation part is based on Codebook part. <br/>
Considering training Diffsound is slow, we provide two types of dataloader ways. The first is normal dataloader, but it is slow because it need read a lot of file, especially when we pre-train on audioset. To solve this problem, we use a dynamic memory dataloarder, which can quickly load data, but it needs you seperate the batch data in advance.  <br>
train_spec.py is using a normal dataloader, and train_spec2.py is using a fast loader methods. <br>
#### Enviroment
Please install followwing package if you meet enviroment problem.
```
pip install torch==1.9.0 torchvision --no-cache-dir -U | cat
pip install omegaconf pytorch-lightning --no-cache-dir -U | cat
pip install timm==0.3.4 --no-cache-dir -U | cat
pip install tensorboard==1.15.0 --no-cache-dir -U | cat
pip install lmdb tqdm --no-cache-dir -U | cat
pip install einops ftfy --no-cache-dir -U | cat
pip install git+https://github.com/openai/DALL-E.git --no-cache-dir -U | cat
```
#### About the pre-trained model
Note that we have release the pre-trained model on audioset on google drive (https://drive.google.com/drive/folders/193It90mEBDPoyLghn4kFzkugbkF_aC8v?usp=sharing), but this model never trained  on Audiocaps, the training data is generated MASK based text, it may cannot perform well. We will release the model trained on audiocaps dataset on Baidu disk. 
#### Pre-training on audioset
Please run followwing command.
```
python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/running_command/run_train_audioset.py
```
#### Training on audiocaps
```
python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/running_command/run_train_caps.py
```
#### Sampling
```
python evaluation/generate_samples_batch.py
```
