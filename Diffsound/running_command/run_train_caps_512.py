import os

string = "python /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/train_spec2.py --name caps_train --config_file /apdcephfs/share_1316500/donchaoyang/code3/DiffusionFast/configs/caps_512.yaml --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth"

os.system(string)

