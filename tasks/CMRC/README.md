# CMRC-2018 task
CMRC-2018 is a machine reading comprehension task. 

## Dataset
The corpus can be find [HERE](https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip)  
Download the corpus and save data at `[CMRC_DATA_PATH]`

## Model
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  

## Preprocess
For MRC tasks, the length of document usually larger than 512, 
so we have to do some pre-process, such as sliding window.   
Run the following script to process training data and save the result at `[CMRC_TRAIN_PATH]`  
```
python data_generate.py \
 --bert_path [CHINESEBERT_PATH] \
 --data_dir [CMRC_DATA_PATH] \
 --output_dir  [CMRC_TRAIN_PATH]
```

## Train and Evaluate
Run the following scripts to generate test file . 
```
python cmrc_trainer.py \
      --bert_path [CHINESEBERT_PATH] \
      --data_dir [CMRC_TRAIN_PATH] \
      --save_path [OUTPUT_PATH] \
      --gpus=0,1,2,3 \
      --batch_size=8 \
      --lr=3e-5 \
      --max_epoch=2 \
      --val_check_interval 0.1 \
      --accumulate_grad_batches=4 \
      --warmup_proporation 0.1
```
the final checkpoint will be saved at `[OUTPUT_PATH]/checkpoint`, 
run following script to generate test file  
```
python cmrc_evluate.py \
    --bert_path [CHINESEBERT_PATH]  \
    --save_path [OUTPUT_PATH]  \
    --test_file [CMRC_DATA_PATH]/test.json  \
    --pretrain_checkpoint [OUTPUT_PATH]/checkpoint/****.ckpt \
    --gpus=0,
```

The test file will be generated at `[OUTPUT_PATH]\test_predictions.json`,
 upload the file to [CLUE](https://www.cluebenchmarks.com/introduce.html) to get 
 the test score.

## Result
The evaluation metric is **EM**.  
Result of our model and previous SOTAs are:

base model: 

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  66.89 |   74.70  |
| BERT | 66.77 |  71.60 |  
| BERT-wwm | 66.96 | 73.95 |  
| RoBERTa |  67.89 |  75.20 |  
| MacBERT | - |   - |  
| ChineseBERT | **67.95** | **95.7** |  

large model:

| Model  | Dev | Test |  
|  ----  | ----  | ----  |  
| RoBERTa | 70.59 | 77.95 |  
| MacBERT |  - |  - |  
| ChineseBERT | **70.70** |  **78.05** |  