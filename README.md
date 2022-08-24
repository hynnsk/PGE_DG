# Pivot-Guided Embedding for Domain Generalization

[]()

This is the official implementation of "Pivot-Guided Embedding for Domain Generalization" in Pytorch.

The code heavily relies on 'In Search of Lost Domain Generalization'.
( [Paper](https://arxiv.org/abs/2007.01434) | [Github](https://github.com/facebookresearch/DomainBed) )

## 1. Requirements
- numpy==1.20.3
- wilds==1.2.2
- imageio==2.9.0
- gdown==3.13.0
- torchvision==0.8.2
- torch==1.7.1
- tqdm==4.62.2
- backpack==0.1
- parameterized==0.8.1
- Pillow==8.3.2

## 2. Training & Evaluation
The data should be prepared on /DATA

To train the model, run the code as below:
```train
python -m domainbed.scripts.sweep launch\
       --data_dir=/DATA\
       --expname pge01 \
       --output_dir experiments\
       --command_launcher local\
       --algorithms PGE\
       --dataset PACS\
       --single_test_envs\
       --n_hparams 20\
       --n_trials 3
```


For simplicity, we provide the training scripts.
You can execute the shell file by the command below:
```
sh run.sh
```


To view the results of your sweep:
```
python -m domainbed.scripts.collect_results\
       --input_dir=./[output_dir]/[expname]
```

## Acknowledgement

Again, this repository is built based on [DomainBed](https://github.com/facebookresearch/DomainBed) repository.
Thanks for the great work.


