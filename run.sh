#!/bin/bash


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