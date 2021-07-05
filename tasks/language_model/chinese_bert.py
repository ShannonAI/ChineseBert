#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : chinese_bert.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/7/5 11:22
@version: 1.0
@desc  : 
"""
import argparse

from datasets.bert_dataset import BertDataset
from models.modeling_glycebert import GlyceBertModel


def sentence_hidden():
    # init args
    parser = argparse.ArgumentParser(description="Chinese Bert Hidden")
    parser.add_argument("--pretrain_path", required=True, type=str, help="pretrain model path")
    parser.add_argument("--sentence", required=True, type=str, help="input sentence")
    args = parser.parse_args()

    # step 1: tokenizer
    tokenizer = BertDataset(args.pretrain_path)

    # step 2: load model
    chinese_bert = GlyceBertModel.from_pretrained(args.pretrain_path)

    # step 3: get hidden
    input_ids, pinyin_ids = tokenizer.tokenize_sentence(args.sentence)
    length = input_ids.shape[0]
    input_ids = input_ids.view(1, length)
    pinyin_ids = pinyin_ids.view(1, length, 8)
    output_hidden = chinese_bert.forward(input_ids, pinyin_ids)[0]
    print(output_hidden)


if __name__ == '__main__':
    sentence_hidden()
