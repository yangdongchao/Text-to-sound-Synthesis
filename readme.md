# Text-to-sound Generation
This paper has been accepted by IEEE Transactions on Audio, Speech and Language Processing. <br/>
I have upload all of the pre-trained models on huggingface: https://huggingface.co/Dongchao/Diffsound/tree/main <br/>
This is the open source code for our paper "Diffsound: discrete diffusion model for text-to-sound generateion". <br/>
You can find the paper on arxiv https://arxiv.org/pdf/2207.09983v1.pdf <br/>
The demo page is http://dongchaoyang.top/text-to-sound-synthesis-demo/ <br/>
**2022/08/03** We upload the training code of VQ-VAE and the baseline method of text-to-sound generation (Autoregressive model), and the Diffsound code. Considering that the github has the limitation of file size, we will upload the pre-trained model on google drive disk. <br/>
**2022/08/06** We uppoad the pre-trained model on google drive. please refer to https://drive.google.com/drive/folders/193It90mEBDPoyLghn4kFzkugbkF_aC8v?usp=sharing <br/>
Note that a pre-trained diffsound model is very large, so that we only upload one audioset pretrained model now. More models we will try to upload on other free disk, if you known any free shared disk, please let me know, I will very appreciate. <br/>
**2022/08/09** We upload trained diffsound model on audiocaps dataset, and the baseline AR model, and the codebook trained on audioset with the size of 512. You can refer to https://pan.baidu.com/s/1R9YYxECqa6Fj1t4qbdVvPQ . The password is **lsyr** <br/>
**2022/12/06** Hi, everyone. In our previous setting, we use the wrong sample rate to load wav file, which results in the speech cannot be generated very well. Now, we update the feature extraction module. https://github.com/yangdongchao/Text-to-sound-Synthesis/blob/master/Codebook/feature_extraction/extract_mel_spectrogram.py#L167 . We will re-train our model, all of the pre-trained model can be found on PKU disk: https://disk.pku.edu.cn:443/link/87DE08BDA2521CB54F4911393EB36B4A
More details will be updated as soon as.
**2023/01/11** The latest pre-trained model on audioset have been released, please refer to PKU disk: https://disk.pku.edu.cn:443/link/E36C91C27830FAF0B9326D8EA685A193
有效期限：2024-06-30 23:59
## Overview
![avatar](fig/overview.png)

## Pretrained Model
We release four text-to-sound pretrained model. Including VQVAE trained on Audioset, Vocoder trained on Audioset, generation model trained on Audiocaps and Audioset.

## Inference
Please refer the readme.md file in Codebook folder to see how to inference.

## Training
Please refer the readme.md file in Codebook folder to see how to train your network.
## Reference
This project based on following open source code.
https://github.com/XinhaoMei/ACT
https://github.com/cientgu/VQ-Diffusion
https://github.com/CompVis/taming-transformers
https://github.com/lonePatient/Bert-Multi-Label-Text-Classification
https://github.com/v-iashin/SpecVQGAN
## Cite 
@article{yang2022diffsound,
  title={Diffsound: Discrete Diffusion Model for Text-to-sound Generation},
  author={Yang, Dongchao and Yu, Jianwei and Wang, Helin and Wang, Wen and Weng, Chao and Zou, Yuexian and Yu, Dong},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2022}
}
