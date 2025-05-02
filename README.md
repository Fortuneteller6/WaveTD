### 📖 An Wavelet Steered Network for Efficient Infrared Small Target Detection

<hr/>

[![](https://img.shields.io/badge/Building-Done-green.svg?style=flat-square)](https://github.com/Fortuneteller6/WaveTD) ![](https://img.shields.io/badge/Language-Python-blue.svg?style=flat-square) [![](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](./LICENSE)

> [Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S1350449525001434)  
> Authors: Teng Ma, Kuanhong Cheng, Tingting Chai, Yubo Wu, and Huixin Zhou. <br/>
> The code and model weights will be made public after the paper is accepted. Thanks for your attention!

<hr/>

### Datasets Prepare

- IRSTD-1K dataset is available at [IRSTD-1K](https://github.com/RuiZhang97/ISNet).
- NUAA-SIRST dataset is available at [NUAA-SIRST](https://github.com/YimianDai/sirst).
- NUDT-SIRST dataset is available at [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection).
- We also prepare the txt file for dividing dataset and three datasets, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bCbrS5B2BWyUjK2Ic0nyreu4wZ9omgpY?lfhs=2).

<hr/>

### Commands for Taining

- The epoch and bath size for training the [WaveTD](https://github.com/Fortuneteller6/WaveTD) can be found in the following commands.

```python
python train.py --base_size 256 --crop_size 256 --epochs 500 --dataset IRSTD-1K --split_method 80_20 --model WaveTD --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUAA-SIRST --split_method 80_20 --model WaveTD --deep_supervision True --train_batch_size 4 --test_batch_size 4 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUDT-SIRST --split_method 80_20 --model WaveTD --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

### Commands for Testing and Visulization

- For both testing and visulization of different dataset, you just need to change the model weights and the dataset name.

```python
python test.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_WaveTD_05_05_2024_13_04_42_wDS --model_dir IRSTD-1K_WaveTD_05_05_2024_13_04_42_wDS/mIoU__WaveTD_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model WaveTD --deep_supervision True --test_batch_size 1 --mode TXT
```

```python
python visulization.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_WaveTD_05_05_2024_13_04_42_wDS --model_dir IRSTD-1K_WaveTD_05_05_2024_13_04_42_wDS/mIoU__WaveTD_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model WaveTD --deep_supervision True --test_batch_size 1 --mode TXT
```

<hr/>

### Results and Weights

| Methods |    Data    |   Pd   |  Fa  |  IoU  | F1_Score |  Download   |
| :-----: | :--------: | :----: | :--: | :---: | :------: | :---------: |
| WaveTD  |  IRSTD-1K  | 91.16  | 5.54 | 69.86 |  82.25   | [Weights](https://drive.google.com/drive/folders/11lI_zlJYjoNFFxBHcGx2Jvx36Edfzf-W?usp=sharing) |
| WaveTD  | NUAA-SIRST | 100.00 | 1.95 | 79.32 |  88.47   | [Weights](https://drive.google.com/drive/folders/17k_ldUs2EjCA9Jt7Sf2eq3ZGW2tk4rA9?usp=sharing) |
| WaveTD  | NUDT-SIRST | 99.30  | 1.42 | 93.30 |  96.53   | [Weights](https://drive.google.com/drive/folders/1yjG3HEDzVSuEoSCFdY0HAJjfHwHh2nZo?usp=sharing) |

<hr/>

### Acknowledgement

The code of this paper is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks for their awesome work.

## Citation

If you find the code helpful in your resarch or work, please cite this paper as following.

```
@article{WaveTD,
  title = {An Wavelet Steered network for efficient infrared small target detection},
  journal = {Infrared Physics & Technology},
  volume = {148},
  pages = {105850},
  year = {2025},
  issn = {1350-4495},
  author = {Teng Ma and Kuanhong Cheng and Tingting Chai and Yubo Wu and Huixin Zhou}
}
```

### Contact

If you have any questions, please feel free to reach me out at teng_m@yeah.net
