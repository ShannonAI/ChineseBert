# OntoNotes NER task 
LCQMC Corpus is a sentence pair matching dataset, 
which could be seen as a binary classification task. 

## Dataset
The official LCQMC corpus can be find [HERE](http://icrc.hitsz.edu.cn/info/1037/1146.htm)  
Download the corpus and save data at `[LCQMC_DATA_PATH]`


## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate. 
```bash 
CUDA_VISIBLE_DEVICES=3 python3 $REPO_PATH/tasks/OntoNotes/OntoNotes_trainer.py \
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
--save_path ${OUTPUT_DIR} \
--config_path ${CONFIG_PATH} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--pretrain_checkpoint ${PRETRAIN_CKPT} \
--gpus="1"
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 

large model: