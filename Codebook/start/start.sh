source activate specvqgan
# python3 /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/train.py \ 
#         --base  /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/configs/caps_transformer_small.yaml -t True --gpus 0,

python3 /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/train.py --base /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/configs/audioset_codebook128.yaml -t True --gpus 0,1,2,3,4,5,6,7,
# /root/anaconda3/envs/specvqgan/bin/python /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/train.py --base /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/configs/caps_transformer.yaml -t True --gpus 0,1,2,3,4,5,6,7, \
#         model.params.first_stage_config.params.ckpt_path=/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-04-24T23-17-27_audioset_codebook256/checkpoints/last.ckpt

# EXPERIMENT_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-05-05T19-28-48_caps_transformer"
# SPEC_DIR_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/features/*/melspec_10s_22050hz/"
# cls_token_dir_path="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/clip_text/*/cls_token_512/"
# SAMPLES_FOLDER="caps_validation"
# # SPLITS="\"[validation, ]\""
# SPLITS="[validation]"
# SAMPLER_BATCHSIZE=32 # 32 previous
# SAMPLES_PER_VIDEO=10
# TOP_K=128 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook, 128 for caps?
# NOW=`date +"%Y-%m-%dT%H-%M-%S"`
# DATASET='caps'
# NO_CONDITION=False # False indicate using condition
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=localhost \
#     --master_port=62374 \
#     --use_env \
#         /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/generate_samples_caps.py \
#         sampler.config_sampler=/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/configs/sampler.yaml \
#         sampler.model_logdir=$EXPERIMENT_PATH \
#         sampler.splits=$SPLITS \
#         sampler.samples_per_video=$SAMPLES_PER_VIDEO \
#         sampler.batch_size=$SAMPLER_BATCHSIZE \
#         sampler.top_k=$TOP_K \
#         sampler.no_condition=$NO_CONDITION \
#         data.params.spec_dir_path=$SPEC_DIR_PATH \
#         data.params.cls_token_dir_path=$cls_token_dir_path \
#         sampler.now=$NOW

# python /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/generate_samples_caps.py \
#         sampler.config_sampler=/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/configs/sampler.yaml \
#         sampler.model_logdir=$EXPERIMENT_PATH \
#         sampler.splits=$SPLITS \
#         sampler.samples_per_video=$SAMPLES_PER_VIDEO \
#         sampler.batch_size=$SAMPLER_BATCHSIZE \
#         sampler.top_k=$TOP_K \
#         sampler.no_condition=$NO_CONDITION \
#         data.params.spec_dir_path=$SPEC_DIR_PATH \
#         data.params.cls_token_dir_path=$cls_token_dir_path \
#         sampler.now=$NOW

# ROOT='/apdcephfs/share_1316500/donchaoyang/code3/VQ-Diffusion/OUTPUT/caps_train/2022-05-08T19-56-36/fast_inf5_samples_2022-05-15T16-33-50'
# python /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluate.py \
#         config=/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/configs/eval_melception_${DATASET,,}.yaml \
#         input2.path_to_exp=$EXPERIMENT_PATH \
#         patch.specs_dir=$SPEC_DIR_PATH \
#         patch.spec_dir_path=$SPEC_DIR_PATH \
#         patch.cls_token_dir_path=$cls_token_dir_path \
#         input1.params.root=$ROOT

