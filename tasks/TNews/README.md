# TNews task 
TNEWS is a 15-class short news text classification dataset. <br>

## Dataset
The official TNEWS corpus can be find [HERE](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)  
Download the corpus and save data at `[TNEWS_DATA_PATH]`


## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate on the validation set. 
If you would like to evaluate the model on the test set, please submit the model prediction to [CLUE](https://www.cluebenchmarks.com/)

```bash 
CUDA_VISIBLE_DEVICES=0 python3 $REPO_PATH/tasks/TNews/TNews_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 128 \
--weight_decay 0.002 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.02 \
--batch_size 12 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--bert_path [CHINESEBERT_PATH] \
--data_dir [TNEWS_DATA_PATH] \
--save_path [OUTPUT_PATH] \
--gpus="1" \
--precision=16
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 
| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 58.24 |  58.33 | 
| BERT | 56.09 |  56.58 | 
| BERT-wwm | 56.77 | 56.86 | 
| RoBERTa |   57.51 |  56.94 | 
| ChineseBERT | 58.64 | 58.95 | 

large model:
| Model  | Dev | Test |  
|   ---- | ----  | ----  |
| RoBERTa-large | 58.32 | 58.61 | 
| ChineseBERT-large |  59.06 | 59.47 | 