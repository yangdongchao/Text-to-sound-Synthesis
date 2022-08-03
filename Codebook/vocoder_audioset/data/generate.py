import os
# train_ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/data/audioset/features/train/melspec_10s_22050hz')
audio_paths = []
st = 0
ed = 41
for i in range(st, ed, 1):
    if i < 10:
        cnt = '0'+str(i)
    else:
        cnt = str(i)
    input_dir = '/apdcephfs/share_1316500/donchaoyang/data/ft_local/unbalanced_train_segments/'+'unbalanced_train_segments_part'+cnt
    print(len(audio_paths))
    tmp_audios = os.listdir(input_dir)
    audio_paths.extend(tmp_audios)
with open("audioset_train_vocoder.txt","w") as f:  
    for audio in audio_paths:
        # print(audio)
        # assert 1==2
        audio_name = audio[:-4]
        final_name = 'train/melspec_10s_22050hz/'+audio_name
        f.write(final_name+'\n')