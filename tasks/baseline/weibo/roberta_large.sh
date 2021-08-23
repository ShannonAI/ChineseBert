#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: ner_weibo/roberta_large.sh
# Result:
# Test F1: 68.35, Precision: 66.74, Recall: 70.02
# gpu3: /data/xiaoya/outputs/glyce/baselines/0121/weibo_large_roberta4_5_12_5e-5_0.001_0_150_0.2_1_0.25

TIME=0820
FILE_NAME=weibo_large_roberta
REPO_PATH=/data/xiaoya/workspace/ChineseBert
BERT_PATH=/data/xiaoya/pretrain_lm/chinese_roberta_wwm_large_ext
TASK=weibo
SAVE_TOPK=20
#ontonotes4, weibo
DATA_DIR=/data/xiaoya/datasets/ner/weibo

TRAIN_BATCH_SIZE=12
LR=5e-5
WEIGHT_DECAY=0.001
WARMUP_PROPORTION=0
MAX_LEN=150
MAX_EPOCH=5
DROPOUT=0.2
ACC_GRAD=1
VAL_CHECK_INTERVAL=0.25

OUTPUT_DIR=/data/xiaoya/outputs/glyce/baselines/${TIME}/${FILE_NAME}_${MAX_EPOCH}_${TRAIN_BATCH_SIZE}_${LR}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${MAX_LEN}_${DROPOUT}_${ACC_GRAD}_${VAL_CHECK_INTERVAL}
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

CUDA_VISIBLE_DEVICES=3 python3 $REPO_PATH/tasks/baseline/tagging_trainer.py \
--lr ${LR} \
--max_epochs ${MAX_EPOCH} \
--max_length ${MAX_LEN} \
--weight_decay ${WEIGHT_DECAY} \
--hidden_dropout_prob ${DROPOUT} \
--warmup_proportion ${WARMUP_PROPORTION}  \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--accumulate_grad_batches ${ACC_GRAD} \
--save_topk ${SAVE_TOPK} \
--bert_path ${BERT_PATH} \
--data_dir ${DATA_DIR} \
--task ${TASK} \
--save_path ${OUTPUT_DIR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--gpus="1"




