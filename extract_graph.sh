#!/bin/bash
##########################################################################################3
ID=AIAG_Extraction
EXPERIMENT_ID=Extraction

DB=meta/A2P2_FULL_Corrected.CSV
DATAPATH=~/Desktop/AVADataSet/
SAVE_FEAT=dump/
FEAT_FILE_NAME=RN50.h5

BASE_MODEL=resnet50
FP=16
PILOT=1000

 CUDA_VISIBLE_DEVICES=0 python3 -W ignore extract_graph.py --id $ID --db $DB --datapath $DATAPATH --pretrained --exp_id $EXPERIMENT_ID --feature_file_name $FEAT_FILE_NAME\
  --base_model $BASE_MODEL \
  --data_precision $FP \
  --save_feat $SAVE_FEAT --pilot $PILOT --n_workers 4

