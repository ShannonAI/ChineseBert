# ChnSetiCorp task
ChnSetiCorp is a dataset for sentiment analysis. 

## Dataset
The corpus can be find [HERE](https://github.com/pengming617/bert_classification)  
Download the corpus and save data at `[ChnSetiCorp_DATA_PATH]`  

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Run the following scripts to train and evaluate. 
```
python ChnSetiCorp_trainer.py \
  --bert_path [CHINESEBERT_PATH] \
  --data_dir [ChnSetiCorp_DATA_PATH] \
  --save_path [OUTPUT_PATH] \
  --max_epoch=10 \
  --lr=2e-5 \
  --warmup_proporation 0.1 \
  --batch_size=16 \
  --weight_decay=0.0001 \
  --gpus=0,
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  95.4 |   95.5  |
| BERT | 95.1 |  95.4 |  
| BERT-wwm | 95.4 | 95.3 |  
| RoBERTa |  95.0 |  95.6 |  
| MacBERT | 95.2 |   95.6 |  
| ChineseBERT | **95.6** | **95.7** |  

large model:

| Model  | Dev | Test |  
|  ----  | ----  | ----  |  
| RoBERTa | **95.8** | 95.8 |  
| MacBERT |  95.7 |  **95.9** |  
| ChineseBERT | **95.8** |  **95.9** |  