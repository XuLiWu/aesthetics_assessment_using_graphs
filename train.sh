#!/bin/bash
########################################################
ID=AIAG
EXPERIMENT_ID=1.1

DB=meta/A2P2_FULL_Corrected.CSV
DATAPATH=~/Desktop/AVADataSet/
SAVE=dump/models/
SAVE_VISUALS=dump/visuals/
FEATURE_PATH=dump/RN50.h5

#Training Params
LR=1e-4
LR_DECAY_AFTER=5
BATCH_SIZE=16
VAL_BATCH_SIZE=64
#AUG=MLSP_8_NUMPY
#AUG_TARGET=TARGET_ADD_ASPECT_R
VAL_AFTER=1
OPTIMIZER=ADAM
FP=32
PILOT=1000
WORKERS=4
BASE_MODEL=resnet50


#Task Weights
W_MSE=1
W_EMD=0

#Architecture
A2_D=10

 CUDA_VISIBLE_DEVICES=0 python3 -W ignore train.py --id $ID --exp_id $EXPERIMENT_ID --db $DB --datapath $DATAPATH --save $SAVE --save_visuals $SAVE_VISUALS --feature_path $FEATURE_PATH \
  --base_model $BASE_MODEL  --A2_D $A2_D \
  --lr $LR --batch_size $BATCH_SIZE --batch_size_test $VAL_BATCH_SIZE --optimizer $OPTIMIZER --data_precision $FP --val_after_every $VAL_AFTER --n_workers $WORKERS --lr_decay_after $LR_DECAY_AFTER \
  --w_emd $W_EMD --w_mse $W_MSE \
  --pilot $PILOT
#################################################################################################