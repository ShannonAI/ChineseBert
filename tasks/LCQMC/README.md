# LCQMC task
LCQMC Corpus is a sentence pair matching dataset, 
which could be seen as a binary classification task. 

## Dataset
The official LCQMC corpus can be find [HERE](http://icrc.hitsz.edu.cn/info/1037/1146.htm)  
Download the corpus and save data at `[LCQMC_DATA_PATH]`

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate. 
```
python LCQMC_trainer.py \
  --bert_path [CHINESEBERT_PATH] \
  --data_dir [LCQMC_DATA_PATH] \
  --save_path [OUTPUT_PATH] \
  --max_epoch=7 \
  --lr=2e-5 \
  --batch_size=16 \
  --gpus=0,1，2,3,4
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 89.8 |  87.2  |
| BERT | 89.4 | 87.0 |  
| BERT-wwm | 89.6 | 87.1 |  
| RoBERTa |  89.0 |  86.4 |  
| MacBERT | 89.5 | 87.0 |  
| ChineseBERT | **89.8** | **87.4** |  

large model:

| Model  | Dev | Test |  
|  ----  | ----  | ----  |  
| RoBERTa | 90.4 | 87.0 |  
| MacBERT |  **90.6** | 87.6 |  
| ChineseBERT | 90.5 |  **87.8** |  