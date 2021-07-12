# OntoNotes NER task 
OntoNotes 4.0 is a Chinese named entity recognition dataset and contains 18 named entity types. 
OntoNotes 4.0 contains 15K/4K/4K instances for training/dev/test. <br>

## Dataset
The OntoNotes 4.0 NER dataset using BMES tagging schema can be find [HERE](https://drive.google.com/file/d/1qH4NDMRYRZYX9CWPLyF5KQLTpX-vvm39/view?usp=sharing)  
Download the corpus and save data at `[ONTONOTES_DATA_PATH]`

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate. 

For ChineseBERT-Base (see [chinesebert_base.sh](./OntoNotes/chinesebert_base.sh)), 

```bash 
CUDA_VISIBLE_DEVICES=0 python3 $REPO_PATH/tasks/OntoNotes/OntoNotes_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 275 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.1  \
--train_batch_size 26 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--gpus="1" \
--precision=16 \
--optimizer torch.adam \
--classifier multi \
--bert_path [CHINESEBERT_PATH] \
--data_dir [ONTONOTES_DATA_PATH] \
--save_path [OUTPUT_PATH] 
```

For ChineseBERT-Large (see [chinesebert_large.sh](./OntoNotes/chinesebert_large.sh)), 

```bash 
CUDA_VISIBLE_DEVICES=1 python3 $REPO_PATH/tasks/OntoNotes/OntoNotes_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 275 \
--weight_decay 0.002 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.1 \
--train_batch_size 18 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--gpus="1" \
--precision=16 \
--optimizer torch.adam \
--classifier multi \
--bert_path [CHINESEBERT_PATH] \
--data_dir [ONTONOTES_DATA_PATH] \
--save_path [OUTPUT_PATH] 
```

## Result
The evaluation metric is Span-Level F1. 
Result of our model and previous models are:

base model: 
| Model  |  Test Precision |  Test Recall |  Test F1 |  
|  ----  | ----  | ----  | ----  |
| BERT | 79.69 | 82.09 | 80.87 | 
| RoBERTa |  80.43 | 80.30 |  80.37 | 
| ChineseBERT | 80.03 | 83.33 | 81.65 | 


large model:

| Model  |  Test Precision |  Test Recall |  Test F1 |  
|  ----  | ----  | ----  | ----  |
| RoBERTa-large |  80.72 | 82.07 | 81.39 |
| ChineseBERT-large | 80.77 | 83.65 | 82.18 | 
