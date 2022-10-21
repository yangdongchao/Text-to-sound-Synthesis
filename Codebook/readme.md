### Training codebook
The source code to train a VQ-VAE codebook. This part code is based on https://github.com/v-iashin/SpecVQGAN 
#### Environment install
conda env create -f conda_env.yml

#### Data
Please first download Audiocaps and Audioset datasets. You can download Audiocaps dataset by https://audiocaps.github.io/ <br/> For Audioset dataset, please refer to https://github.com/qiuqiangkong/audioset_tagging_cnn, which provides a link to download. <br/>
Then please using feature_extraction/extract_mel_spectrogram.py file to extract mel-spectrogram <br/>
For the text features, we provide two types of features, (1) use BERT (2) use CLIP <br/>
For BERT features, please run 
```
python generete_text_fea/predict_one.py
```
For CLIP features, please run 
```
python generete_text_fea/generate_fea_clip.py 
```

#### Download pre-trained models
Please first download the **lpaps** folder from google drive (https://drive.google.com/drive/folders/193It90mEBDPoyLghn4kFzkugbkF_aC8v?usp=sharing), and place it on Codebook/specvqgan/modules/autoencoder/ <br/> Furthermore, please down **logs** folder, and put it to Codebook/evaluation/logs
The 2022-04-24T23-17-27_audioset_codebook256 is our pre-trained codebook on audioset, and the size of codebook is 256. Other size of codebook will be upload on Baidu disk as soon as.
#### Train codebook
```
python3 /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/train.py --base /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/configs/audioset_codebook128.yaml -t True --gpus 0,1,2,3,4,5,6,7,
```
You can use different config file from configs. audioset_codebook128 denotes that we train our codebook on audioset and the codebook size set as 128. <br/>

#### Train AR model
Firstly, please set the some key path in configs/caps_transformer.yaml, then run
```
python3 Codebook/train.py --base Codebook/configs/caps_transformer.yaml -t True --gpus 0,1,2,3,4,5,6,7, \
model.params.first_stage_config.params.ckpt_path=Codebook/logs/2022-04-24T23-17-27_audioset_codebook256/checkpoints/last.ckpt
``` 
#### Sampling
```
EXPERIMENT_PATH="Codebook/logs/2022-05-05T19-28-48_caps_transformer"
SPEC_DIR_PATH="Codebook/data/audiocaps/features/*/melspec_10s_22050hz/"
cls_token_dir_path="Codebook/data/audiocaps/clip_text/*/cls_token_512/"
SAMPLES_FOLDER="caps_validation"
# SPLITS="\"[validation, ]\""
SPLITS="[validation]"
SAMPLER_BATCHSIZE=32 # 32 previous
SAMPLES_PER_VIDEO=10
TOP_K=128 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook, 128 for caps?
NOW=`date +"%Y-%m-%dT%H-%M-%S"`
DATASET='caps'
NO_CONDITION=False # False indicate using condition
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
        Codebook/evaluation/generate_samples_caps.py \
        sampler.config_sampler=Codebook/evaluation/configs/sampler.yaml \
        sampler.model_logdir=$EXPERIMENT_PATH \
        sampler.splits=$SPLITS \
        sampler.samples_per_video=$SAMPLES_PER_VIDEO \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
        sampler.no_condition=$NO_CONDITION \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.cls_token_dir_path=$cls_token_dir_path \
        sampler.now=$NOW

python Codebook/evaluation/generate_samples_caps.py \
        sampler.config_sampler=Codebook/evaluation/configs/sampler.yaml \
        sampler.model_logdir=$EXPERIMENT_PATH \
        sampler.splits=$SPLITS \
        sampler.samples_per_video=$SAMPLES_PER_VIDEO \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
        sampler.no_condition=$NO_CONDITION \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.cls_token_dir_path=$cls_token_dir_path \
        sampler.now=$NOW
```
we provide two types of sample ways. The first is using multiple GPUs to sampling, the second uses one GPU.

#### Evalutation
```
ROOT='Codebook/OUTPUT/caps_train/2022-05-08T19-56-36/fast_inf5_samples_2022-05-15T16-33-50'
python Codebook/evaluate.py \
        config=Codebook/evaluation/configs/eval_melception_${DATASET,,}.yaml \
        input2.path_to_exp=$EXPERIMENT_PATH \
        patch.specs_dir=$SPEC_DIR_PATH \
        patch.spec_dir_path=$SPEC_DIR_PATH \
        patch.cls_token_dir_path=$cls_token_dir_path \
        input1.params.root=$ROOT
```
Firstly, set the ROOT to your generated sound path, and then set the true ground sound path. According to this, you can get FID and KL metrics. <br/>
For audiocaption loss, you can direcly run  AudiocaptionLoss/start.sh before you set the generated sound path <br/>

