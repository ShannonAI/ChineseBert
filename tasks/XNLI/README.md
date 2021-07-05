# XNLI task
XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification.   
In our experiments, we only use Chinese XNLI dataset. 

## Dataset
The official XNLI corpus can be find [HERE](https://github.com/facebookresearch/XNLI)  
Download the corpus and save data at `[XNLI_DATA_PATH]`  
We only use Chinese data.

## Train and Evaluate
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
To be simplify, XNLI is a classification task with four classes, so you have to
modify `[CHINESEBERT_PATH]/config.json` file, and set `"num_labels":4`.   
Run the following scripts to train and evaluate. 
```
python XNLI_trainer.py \
  --bert_path [CHINESEBERT_PATH] \
  --data_dir [XNLI_DATA_PATH] \
  --save_path [OUTPUT_PATH] \
  --max_epoch=5 \
  --batch_size=8 \
  --lr=3e-5 \
  --gpus=0,1
```

## Result
The evaluation metric is **Accuracy**.  
Result of our model and previous SOTAs are:

base model: 

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  79.7 |   78.6  |
| BERT | 79.0 |  78.2 |  
| BERT-wwm | 79.4 | 78.7 |  
| RoBERTa |  80.0 |  78.8 |  
| MacBERT | 80.3 |  79.3 |  
| ChineseBERT | **80.5** | **79.6** |  

large model:

| Model  | Dev | Test |  
|  ----  | ----  | ----  |  
| RoBERTa | 82.1 | 81.2 |  
| MacBERT |  82.4 |  81.3 |  
| ChineseBERT | **82.7** |  **81.6** |  