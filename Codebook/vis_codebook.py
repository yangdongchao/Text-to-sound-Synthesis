import os
from pathlib import Path
import soundfile
import torch
import IPython
import matplotlib.pyplot as plt
import librosa
import numpy as np
from feature_extraction.demo_utils import (calculate_codebook_bitrate,
                                            extract_melspectrogram,
                                            get_audio_file_bitrate,
                                            get_duration,
                                            load_neural_audio_codec)
from sample_visualization import tensor_to_plt
from torch.utils.data.dataloader import default_collate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = '2022-03-22T12-06-24_audioset_codebook' # 2022-04-24T23-17-27_audioset_codebook256
log_dir = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/logs'
# loading the models might take a few minutes
config, model, vocoder = load_neural_audio_codec(model_name, log_dir, device)
# Select an Audio
audio_ls = ['/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/YrqfQRErjfk8.wav',
            '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/Yrqu8iB22I_Y.wav',
            '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/YryFDPxgDOGc.wav',
            '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/YrwtmaKiCcQU.wav',
            '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/Yrwb6PSAee5Y.wav',
            '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/YrwT__ERCUno.wav']
for audio in audio_ls:
    input_wav = audio
    name_tmp = audio.split('/')[-1][:-4]
    print('name_tmp ',name_tmp)
    #assert 1==2
    decode_spec_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/tmp/compare'
    model_sr = config.data.params.sample_rate
    duration = get_duration(input_wav)
    spec = extract_melspectrogram(input_wav, sr=model_sr, duration=duration)
    print(f'Audio Duration: {duration} seconds')
    print('Original Spectrogram Shape:', spec.shape)

    # Prepare Input
    spectrogram = {'input': spec}
    batch = default_collate([spectrogram])
    batch['image'] = batch['input'].to(device)
    x = model.get_input(batch, 'image')

    with torch.no_grad():
        quant_z, diff, info = model.encode(x)
        xrec = model.decode(quant_z)

    print('Compressed representation (it is all you need to recover the audio):')
    F, T = quant_z.shape[-2:]
    print(info[2].reshape(F, T))

    # Calculate Bitrate
    bitrate = calculate_codebook_bitrate(duration, quant_z, model.quantize.n_e)
    orig_bitrate = get_audio_file_bitrate(input_wav)

    # librosa.display.specshow((x+1)/2, sr=22050)
    # assert 1==2
    # Save and Display
    x = x.squeeze(0)
    xrec = xrec.squeeze(0)
    xrec_np = xrec.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    org_name = name_tmp + '_org.npy'
    rec_name = name_tmp + '_pre2048.npy'
    #tmp_save_org = os.path.join(decode_spec_path, org_name)
    tmp_save_rec = os.path.join(decode_spec_path, rec_name)
    np.save(tmp_save_rec, xrec_np)
    #np.save(tmp_save_org, x_np)

assert 1==2
input_wav = '/apdcephfs/share_1316500/donchaoyang/code3/data/audiocaps/val/YryFDPxgDOGc.wav'
decode_spec_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/tmp/spec_save'
# Spectrogram Extraction
model_sr = config.data.params.sample_rate
duration = get_duration(input_wav)
spec = extract_melspectrogram(input_wav, sr=model_sr, duration=duration)
print(f'Audio Duration: {duration} seconds')
print('Original Spectrogram Shape:', spec.shape)

# Prepare Input
spectrogram = {'input': spec}
batch = default_collate([spectrogram])
batch['image'] = batch['input'].to(device)
x = model.get_input(batch, 'image')

with torch.no_grad():
    quant_z, diff, info = model.encode(x)
    xrec = model.decode(quant_z)

print('Compressed representation (it is all you need to recover the audio):')
F, T = quant_z.shape[-2:]
print(info[2].reshape(F, T))

# Calculate Bitrate
bitrate = calculate_codebook_bitrate(duration, quant_z, model.quantize.n_e)
orig_bitrate = get_audio_file_bitrate(input_wav)

# librosa.display.specshow((x+1)/2, sr=22050)
# assert 1==2
# Save and Display
x = x.squeeze(0)
xrec = xrec.squeeze(0)
xrec_np = xrec.detach().cpu().numpy()
x_np = x.detach().cpu().numpy()
tmp_save_org = os.path.join(decode_spec_path, 'orginal4.npy')
tmp_save_rec = os.path.join(decode_spec_path, 'reconstruction4.npy')
np.save(tmp_save_rec, xrec_np)
np.save(tmp_save_org, x_np)
assert 1==2
# specs are in [-1, 1], making them in [0, 1]
wav_x = vocoder((x + 1) / 2).squeeze().detach().cpu().numpy()
wav_xrec = vocoder((xrec + 1) / 2).squeeze().detach().cpu().numpy()
# Creating a temp folder which will hold the results
tmp_dir = os.path.join('./tmp/neural_audio_codec', Path(input_wav).parent.stem)
os.makedirs(tmp_dir, exist_ok=True)
# Save paths
x_save_path = Path(tmp_dir) / 'vocoded_orig_spec.wav'
xrec_save_path = Path(tmp_dir) / f'specvqgan_{bitrate:.2f}kbps.wav'
# Save
soundfile.write(x_save_path, wav_x, model_sr, 'PCM_16')
soundfile.write(xrec_save_path, wav_xrec, model_sr, 'PCM_16')
# Display
print(f'Original audio ({orig_bitrate:.0f} kbps):')
IPython.display.display(IPython.display.Audio(x_save_path))
print(f'Reconstructed audio ({bitrate:.2f} kbps):')
IPython.display.display(IPython.display.Audio(xrec_save_path))
print('Original Spectrogram:')
fig_org = tensor_to_plt(x, flip_dims=(2,))
fig_org.savefig('Original.png')
#IPython.display.display(tensor_to_plt(x, flip_dims=(2,)))
plt.close()
print('Reconstructed Spectrogram:')
fig_recons = tensor_to_plt(xrec, flip_dims=(2,))
fig_recons.savefig('recons.png')
#IPython.display.display()
plt.close()

