import os
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from dataset import VGGSound, Audioset
import torch
import torch.nn as nn
from metrics import metrics
from omegaconf import OmegaConf
from model import VGGishish,VGGishish_audio_new
from transforms import Crop, StandardNormalizeAudio, ToTensor
from collections import OrderedDict
from loss import BinaryCrossEntropy
if __name__ == '__main__':
    cfg_yml = OmegaConf.load('/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/specvqgan/modules/losses/vggishish/configs/vggish_m.yaml')
    # the latter arguments are prioritized
    cfg = cfg_yml
    OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))
    # cfg_cli = OmegaConf.from_cli()
    # print(cfg_cli.config)
    # cfg_yml = OmegaConf.load(cfg_cli.config)
    # # the latter arguments are prioritized
    # cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    # OmegaConf.set_readonly(cfg, True)
    # print(OmegaConf.to_yaml(cfg))

    # logger = LoggerWithTBoard(cfg)
    transforms = [
        StandardNormalizeAudio(cfg.mels_path),
    ]
    if cfg.cropped_size not in [None, 'None', 'none']:
        transforms.append(Crop(cfg.cropped_size))
    transforms = torchvision.transforms.transforms.Compose(transforms)

    datasets = {
        'test': Audioset('test', cfg.mels_path, transforms),
    }

    loaders = {
        'test': DataLoader(datasets['test'], batch_size=cfg.batch_size,
                           num_workers=cfg.num_workers, pin_memory=True)
    }

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = VGGishish_audio_new(cfg.conv_layers, cfg.use_bn, num_classes=527)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = BCELoss()

    # loading the best model
    # folder_name = os.path.split(cfg.config)[0].split('/')[-1]
    # print(folder_name)
    #ckpt = torch.load(f'./v_logs/{folder_name}/vggishish-{folder_name}.pt', map_location='cpu')
    ckpt = torch.load('/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/specvqgan/modules/losses/v_logs/22-03-30T00-55-48/DataParallel-22-03-30T00-55-48.pt', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in ckpt['model'].items(): # k为module.xxx.weight, v为权重
        # print('k,v ',k)
        name = k[7:]   # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print((f'The model was trained for {ckpt["epoch"]} epochs. Loss: {ckpt["loss"]:.4f}'))

    # Testing the model
    model.eval()
    running_loss = 0
    preds_from_each_batch = []
    targets_from_each_batch = []

    for i, batch in enumerate(tqdm(loaders['test'])):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.set_grad_enabled(False):
            outputs = model(inputs.float())
            print(outputs > 0.5)
            # print('outputs ',outputs[0])
            # print('targets ',targets[0])
            # assert 1==2
            loss = criterion(outputs, targets.float())

        # loss
        running_loss += loss.item()

        # for metrics calculation later on
        preds_from_each_batch += [outputs.detach().cpu()]
        targets_from_each_batch += [targets.cpu()]

    # logging metrics
    preds_from_each_batch = torch.cat(preds_from_each_batch)
    targets_from_each_batch = torch.cat(targets_from_each_batch)
    test_metrics_dict = metrics(targets_from_each_batch, preds_from_each_batch)
    test_metrics_dict['avg_loss'] = running_loss / len(loaders['test'])
    test_metrics_dict['param_num'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # TODO: I have no idea why tboard doesn't keep metrics (hparams) in a tensorboard when
    # I run this experiment from cli: `python main.py config=./configs/vggish.yaml`
    # while when I run it in vscode debugger the metrics are present in the tboard (weird)
    print(test_metrics_dict)
