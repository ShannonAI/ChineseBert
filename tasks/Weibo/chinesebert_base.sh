#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# author: xiaoya li
# Result:
# TEST F1: 0.6987104415893555, Precision: 0.6681614518165588, Recall: 0.7321867346763611

TIME=0710
FILE_NAME=weibo_glyce_base
REPO_PATH=/userhome/xiaoya/ChineseBert
BERT_PATH=/userhome/xiaoya/bert/ChineseBERT-base
DATA_DIR=/userhome/xiaoya/dataset/ner/weibo

TASK=weibo
SAVE_TOPK=20
TRAIN_BATCH_SIZE=2
LR=4e-5
WEIGHT_DECAY=0.01
WARMUP_PROPORTION=0.02
MAX_LEN=150
MAX_EPOCH=10
DROPOUT=0.2
ACC_GRAD=1
VAL_CHECK_INTERVAL=0.25
OPTIMIZER=torch.adam
CLASSIFIER=multi


OUTPUT_DIR=/userhome/xiaoya/outputs/chinesebert/${TIME}/${FILE_NAME}_${MAX_EPOCH}_${TRAIN_BATCH_SIZE}_${LR}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${MAX_LEN}_${DROPOUT}_${ACC_GRAD}_${VAL_CHECK_INTERVAL}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


CUDA_VISIBLE_DEVICES=0 python3 $REPO_PATH/tasks/Weibo/Weibo_trainer.py \
--lr ${LR} \
--max_epochs ${MAX_EPOCH} \
--max_length ${MAX_LEN} \
--weight_decay ${WEIGHT_DECAY} \
--hidden_dropout_prob ${DROPOUT} \
--warmup_proportion ${WARMUP_PROPORTION}  \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--accumulate_grad_batches ${ACC_GRAD} \
--save_topk ${SAVE_TOPK} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--classifier ${CLASSIFIER} \
--gpus="1" \
--optimizer ${OPTIMIZER} \
--bert_path ${BERT_PATH} \
--data_dir ${DATA_DIR} \
--save_path ${OUTPUT_DIR}


