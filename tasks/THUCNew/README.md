# THUCNews task
THUCNews is a dataset for document classification task. 

## Dataset
The corpus can be find [HERE](https://github.com/gaussic/text-classification-cnn-rnn)  
Download the corpus and save data at `[THUCNews_DATA_PATH]`  

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
THUCNews is a classification task with ten classes, so you have to
modify `[CHINESEBERT_PATH]/config.json` file, and set `"num_labels":10`.
Run the following scripts to train and evaluate. 
```
python THUCNews_trainer.py \
  --bert_path [CHINESEBERT_PATH] \
  --data_dir [THUCNews_DATA_PATH] \
  --save_path [OUTPUT_PATH] \
  --max_epoch=5 \
  --lr=2e-5 \
  --batch_size=8 \
  --gpus=0,1,2,3
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