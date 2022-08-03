f_train = open("/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audioset/txt/audioset_train.txt","r")
with open("audioset_train_vocoder_part.txt","w") as f:
    for audio in f_train:
        audio_ls = audio.split('/')
        new_name = audio_ls[0] + '/'+'melspec_10s_22050hz/'+audio_ls[1]
        f.write(new_name)

