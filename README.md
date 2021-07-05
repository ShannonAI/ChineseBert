# ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information

This repository contains code, model, dataset for [ChineseBERT]() at ACL2021.

**[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)**  
*Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu and Jiwei Li*


## Guide  

| Section | Description |
|  ----  | ----  |
| [Introduction](#Introduction) | Introduction to ChineseBERT |  
| [Download](#Download) | Download links for ChineseBERT |
| [Quick tour](#Quick-tour) | Learn how to quickly load models |
| [Experiment](#Experiments) | Experiment results on different Chinese NLP datasets |
| [Citation](#Citation) | Citation | 
| [Contact](#Contact) | How to contact us | 

## Introduction
We propose ChineseBERT, which incorporates both the glyph and pinyin information of Chinese
characters into language model pretraining.  
 
First, for each Chinese character, we get three kind of embedding.
 - **Char Embedding:** the same as origin BERT token embedding.
 - **Glyph Embedding:** capture visual features based on different fonts of a Chinese character.
 - **Pinyin Embedding:** capture phonetic feature from the pinyin sequence ot a Chinese Character.
 
Then, char embedding, glyph embedding and pinyin embedding 
are first concatenated, and mapped to a D-dimensional embedding through a fully 
connected layer to form the fusion embedding.   
Finally, the fusion embedding is added with the position embedding, which is fed as input to the BERT model.  
The following image shows an overview architecture of ChineseBERT model.
 
![MODEL](https://raw.githubusercontent.com/ShannonAI/ChineseBert/main/images/ChineseBERT.png)

ChineseBERT leverages the glyph and pinyin information of Chinese 
characters to enhance the model's ability of capturing
context semantics from surface character forms and
disambiguating polyphonic characters in Chinese.

## Download 
We provide pre-trained ChineseBERT models in Pytorch version and followed huggingFace model format. 

* **`ChineseBERT-base`**：12-layer, 768-hidden, 12-heads, 147M parameters 
* **`ChineseBERT-large`**: 24-layer, 1024-hidden, 16-heads, 374M parameters   
  
Our model can be downloaded here:

| Model | Model Hub | Size |
| --- | --- | --- |
| **`ChineseBERT-base`**  | [Pytorch](https://huggingface.co/zijun/ChineseBERT-base) | 564M |
| **`ChineseBERT-large`**   | [Pytorch](https://huggingface.co/zijun/ChineseBERT-large) | 1.4G |

*Note: The model hub contains model, fonts and pinyin config files.*

## Quick tour
We train our model with Huggingface, so the model can be easily loaded.  
Download ChineseBERT model and save at `[CHINESEBERT_PATH]`.  
Here is a quick tour to load our model. 
```
>>> from models.modeling_glycebert import GlyceBertForMaskedLM

>>> chinese_bert = GlyceBertForMaskedLM.from_pretrained([CHINESEBERT_PATH])
>>> print(chinese_bert)
```
The complete example can be find here: 
[Masked word completion with ChineseBERT](tasks/language_model/README.md)

Another example to get representation of a sentence:
```
>>> from datasets.bert_dataset import BertDataset
>>> from models.modeling_glycebert import GlyceBertModel

>>> tokenizer = BertDataset([CHINESEBERT_PATH])
>>> chinese_bert = GlyceBertModel.from_pretrained([CHINESEBERT_PATH])
>>> sentence = '我喜欢猫'

>>> input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
>>> length = input_ids.shape[0]
>>> input_ids = input_ids.view(1, length)
>>> pinyin_ids = pinyin_ids.view(1, length, 8)
>>> output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
>>> print(output_hidden)
tensor([[[ 0.0287, -0.0126,  0.0389,  ...,  0.0228, -0.0677, -0.1519],
         [ 0.0144, -0.2494, -0.1853,  ...,  0.0673,  0.0424, -0.1074],
         [ 0.0839, -0.2989, -0.2421,  ...,  0.0454, -0.1474, -0.1736],
         [-0.0499, -0.2983, -0.1604,  ..., -0.0550, -0.1863,  0.0226],
         [ 0.1428, -0.0682, -0.1310,  ..., -0.1126,  0.0440, -0.1782],
         [ 0.0287, -0.0126,  0.0389,  ...,  0.0228, -0.0677, -0.1519]]],
       grad_fn=<NativeLayerNormBackward>)
```
The complete code can be find [HERE](tasks/language_model/chinese_bert.py)

## Experiments

## ChnSetiCorp
ChnSetiCorp is a dataset for sentiment analysis.  
Evaluation Metrics: Accuracy

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  95.4 |   95.5  |
| BERT | 95.1 |  95.4 |  
| BERT-wwm | 95.4 | 95.3 |  
| RoBERTa |  95.0 |  95.6 |  
| MacBERT | 95.2 |   95.6 |  
| ChineseBERT | **95.6** | **95.7** |  
|   | ----  | ----  |  
| RoBERTa-large | **95.8** | 95.8 |  
| MacBERT-large |  95.7 |  **95.9** |  
| ChineseBERT-large | **95.8** |  **95.9** | 

Training details and code can be find [HERE](tasks/ChnSetiCorp/README.md)

### THUCNews
THUCNews contains news in 10 categories.  
Evaluation Metrics: Accuracy

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  95.4 |   95.5  |
| BERT | 95.1 |  95.4 |  
| BERT-wwm | 95.4 | 95.3 |  
| RoBERTa |  95.0 |  95.6 |  
| MacBERT | 95.2 |   95.6 |  
| ChineseBERT | **95.6** | **95.7** |  
|   | ----  | ----  |  
| RoBERTa-large | **95.8** | 95.8 |  
| MacBERT-large |  95.7 |  **95.9** |  
| ChineseBERT-large | **95.8** |  **95.9** |

Training details and code can be find [HERE](tasks/THUCNew/README.md)

### XNLI
XNLI is a dataset for natural language inference.  
Evaluation Metrics: Accuracy  

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  79.7 |   78.6  |
| BERT | 79.0 |  78.2 |  
| BERT-wwm | 79.4 | 78.7 |  
| RoBERTa |  80.0 |  78.8 |  
| MacBERT | 80.3 |  79.3 |  
| ChineseBERT | **80.5** | **79.6** |  
|   | ----  | ----  |  
| RoBERTa-large | 82.1 | 81.2 |  
| MacBERT-large |  82.4 |  81.3 |  
| ChineseBERT-large | **82.7** |  **81.6** |

Training details and code can be find [HERE](tasks/XNLI/README.md)

### BQ
BQ Corpus is a sentence pair matching dataset.  
Evaluation Metrics: Accuracy

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 86.3 | 85.0  |
| BERT | 86.1 | 85.2 |  
| BERT-wwm | **86.4** | **85.3** |  
| RoBERTa |  86.0 | 85.0 |  
| MacBERT | 86.0 | 85.2 |  
| ChineseBERT | **86.4** | 85.2 |  
|    | ----  | ----  |
| RoBERTa-large | 86.3 | 85.8 |  
| MacBERT-large |  86.2 | 85.6 |  
| ChineseBERT-large | **86.5** |  **86.0** | 

Training details and code can be find [HERE](tasks/BQ/README.md)

### LCQMC
LCQMC Corpus is a sentence pair matching dataset.  
Evaluation Metrics: Accuracy

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 89.8 |  87.2  |
| BERT | 89.4 | 87.0 |  
| BERT-wwm | 89.6 | 87.1 |  
| RoBERTa |  89.0 |  86.4 |  
| MacBERT | 89.5 | 87.0 |  
| ChineseBERT | **89.8** | **87.4** |  
|   | ----  | ----  |  
| RoBERTa-large | 90.4 | 87.0 |  
| MacBERT-large |  **90.6** | 87.6 |  
| ChineseBERT-large | 90.5 |  **87.8** |  

Training details and code can be find [HERE](tasks/LCQMC/README.md)

### TNEWS

TNEWS is a 15-class short news text classification dataset. <br>
Evaluation Metrics: Accuracy

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE | 58.24 |  58.33 | 
| BERT | 56.09 |  56.58 | 
| BERT-wwm | 56.77 | 56.86 | 
| RoBERTa |   57.51 |  56.94 | 
| ChineseBERT | **58.64** | **58.95** | 
|   | ----  | ----  |  
| RoBERTa-large | 58.32 | 58.61 | 
| ChineseBERT-large |  **59.06** | **59.47** | 

Training details and code can be find [HERE](tasks/TNews/README.md)

### CMRC

CMRC is a machin reading comprehension task dataset.  
Evaluation Metrics: EM

| Model  | Dev | Test |  
|  ----  | ----  | ----  |
| ERNIE |  66.89 |   74.70  |
| BERT | 66.77 |  71.60 |  
| BERT-wwm | 66.96 | 73.95 |  
| RoBERTa |  67.89 |  75.20 |  
| MacBERT | - |   - |  
| ChineseBERT | **67.95** | **95.7** |  
|   | ----  | ----  |  
| RoBERTa-large | 70.59 | 77.95 |  
| ChineseBERT-large | **70.70** |  **78.05** |  

Training details and code can be find [HERE](tasks/CMRC/README.md)

### OntoNotes

OntoNotes 4.0 is a Chinese named entity recognition dataset and contains 18 named entity types. <br>

Evaluation Metrics: Span-Level F1

| Model  |  Test Precision |  Test Recall |  Test F1 |  
|  ----  | ----  | ----  | ----  |
| BERT | 79.69 | 82.09 | 80.87 | 
| RoBERTa |  **80.43** | 80.30 |  80.37 | 
| ChineseBERT | 80.03 | **83.33** | **81.65** | 
|    | ----  | ----  | ----  |
| RoBERTa-large |  80.72 | 82.07 | 81.39 |
| ChineseBERT-large | **80.77** | **83.65** | **82.18** | 

Training details and code can be find [HERE](tasks/OntoNotes/README.md)


### Weibo 

Weibo is a Chinese named entity recognition dataset and contains 4 named entity types. <br>

Evaluation Metrics: Span-Level F1

| Model  |  Test Precision |  Test Recall |  Test F1 |  
|  ----  | ----  | ----  | ----  |
| BERT | 67.12 | 66.88 |  67.33 |
| RoBERTa | **68.49** | 67.81 | 68.15 |
| ChineseBERT | 68.27 | **69.78** | **69.02** |
|  | ----  | ----  | ----  |
| RoBERTa-large |  66.74 | 70.02 | 68.35 |
| ChineseBERT-large | **68.75** | **72.97** | **70.80** |

Training details and code can be find [HERE](tasks/Weibo/README.md)

## Contact
If you have any question about our paper/code/modal/data...  
Please feel free to discuss through github issues or emails.  
You can send email to **zijun_sun@shannonai.com** or **shuhe_wang@shannonai.com**
