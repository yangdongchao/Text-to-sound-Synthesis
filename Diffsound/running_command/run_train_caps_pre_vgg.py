import os

string = "python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/train_spec2.py --name caps_train_vgg_pre --config_file /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/configs/caps_pre_vgg.yaml --tensorboard --load_path /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/OUTPUT/vgg_train/2022-05-08T19-16-27/checkpoint/000199e_116799iter.pth"

os.system(string)

