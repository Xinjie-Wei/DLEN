# DLEN
PyTorch code for our paper" DLEN: Deep Laplacian Enhancement Networks for Low-Light Images "
Visual comparisons
![image](figs/LOLv1_179.png)
## Dependencies
* python==3.7.5
* torch==1.7.1
* torchvision==0.8.2
* timm==0.2.4
* matplotlib==3.4.3
* tensorboard==2.5.0
* numpy==1.19.5
* scipy==1.7.1
* opencv-python==4.2.0.34

```bash
cd DLEN 
pip install -r ./requirements.txt
```


## Folder structure
Download the datasets and pretrained models first. Please prepare the basic folder structure as follows.

```bash

  /DLEN
    /src     # config files for datasets and PSNR and SSIM code
    /models   # python files for DLEN
    /pretrained_models  # folder for pretrained models
    requirements.txt
    README.md
    
```
## Test
### For the evaluation on LOL-v1 and LOL-v2, you should change input_dir (Input image directory) with your datasets path on the test.py
```bash  
# put datasets and pretrained model in the corresponding directory 
cd DLEN 
python test.py
```
## Train

The source code for training our DLEN will be available after the publication of the paper.
