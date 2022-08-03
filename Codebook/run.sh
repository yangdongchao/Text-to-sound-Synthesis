EXPERIMENT_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-01-30T16-28-24_vas_transformer"
SPEC_DIR_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/vas/features/*/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/vas/features/*/feature_rgb_bninception_dim1024_21.5fps/"
FLOW_FEATS_DIR_PATH="/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/vas/features/*/feature_flow_bninception_dim1024_21.5fps/"
SAMPLES_FOLDER="VAS_validation"
SPLITS="\"[validation, ]\""
SAMPLER_BATCHSIZE=32
SAMPLES_PER_VIDEO=10
TOP_K=64 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook
NOW=`date +"%Y-%m-%dT%H-%M-%S"`
DATASET='caps'
ROOT_PATH='/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs/2022-01-28T21-57-16_caps_transformer/samples_2022-02-03T18-44-25/caps_validation'
python /apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluate.py \
        config=/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/evaluation/configs/eval_melception_${DATASET,,}.yaml \
        input2.path_to_exp=$EXPERIMENT_PATH \
        patch.specs_dir=$SPEC_DIR_PATH \
        patch.spec_dir_path=$SPEC_DIR_PATH \
        patch.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
        patch.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
        input1.params.root=$ROOT_PATH