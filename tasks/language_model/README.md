## Masked word completion
Taking a sentence, the model masks each word in the input sentence
then run the entire masked sentence through the model and has to predict the masked words.  

First, you have to prepare pretrained ChineseBert model.
 
 - Model can be find here: [ChineseBERT-base](https://huggingface.co/zijun/ChineseBERT-base) or 
 [ChineseBERT-large](https://huggingface.co/zijun/ChineseBERT-large).  
   Download model and save at `[CHINESEBERT_PATH]`
 
   
Then, run `bash file_mask.sh`.

Finally, you will get output:
```
input sentence: [CLS] [MASK] 喜 欢 小 猫 [SEP]
output sentence: [CLS] 我 喜 欢 小 猫 [SEP]

input sentence: [CLS] 我 [MASK] 欢 小 猫 [SEP]
output sentence: [CLS] 我 喜 欢 小 猫 [SEP]

input sentence: [CLS] 我 喜 [MASK] 小 猫 [SEP]
output sentence: [CLS] 我 喜 欢 小 猫 [SEP]

input sentence: [CLS] 我 喜 欢 [MASK] 猫 [SEP]
output sentence: [CLS] 我 喜 欢 熊 猫 [SEP]

input sentence: [CLS] 我 喜 欢 小 [MASK] [SEP]
output sentence: [CLS] 我 喜 欢 小 。 [SEP]
```
 




 