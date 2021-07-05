# BQ task
BQ Corpus is a sentence pair matching dataset, 
which could be seen as a binary classification task. 

## Dataset
The official BQ corpus can be find [HERE](http://icrc.hitsz.edu.cn/Article/show/175.html)  
Download the corpus and save data at `[BQ_DATA_PATH]`

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate. 
```
python BQ_trainer.py \
  --bert_path [CHINESEBERT_PATH] \
  --data_dir [BQ_DATA_PATH] \
  --save_path [OUTPUT_PATH] \
  --max_epoch=10 \
  --lr=3e-5 \
  --batch_size=4 \
  --accumulate_grad_batches 4 \
  --warmup_proporation 0.1 \
  --weight_decay=0.001 \
  --precision 16 \
  --gpus=0,1,2,3
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 86.3 | 85.0  |
| BERT | 86.1 | 85.2 |  
| BERT-wwm | **86.4** | **85.3** |  
| RoBERTa |  86.0 | 85.0 |  
| MacBERT | 86.0 | 85.2 |  
| ChineseBERT | **86.4** | 85.2 |  

large model:

| Model  | Dev | Test |  
|  ----  | ----  | ----  |  
| RoBERTa | 86.3 | 85.8 |  
| MacBERT |  86.2 | 85.6 |  
| ChineseBERT | **86.5** |  **86.0** |  