### Diffsound
Make sure you have the readme part on Codebook part. Our data and evaluation part is based on Codebook part. <br/>
Considering training Diffsound is slow, we provide two types of dataloader ways. The first is normal dataloader, but it is slow because it need read a lot of file, especially when we pre-train on audioset. To solve this problem, we use a dynamic memory dataloarder, which can quickly load data, but it needs you seperate the batch data in advance.  <br>
train_spec.py is using a normal dataloader, and train_spec2.py is using a fast loader methods. <br>

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
