#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : fill_mask.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/23 19:36
@version: 1.0
@desc  : an example of masked language modeling
"""
import argparse
import os

import torch

from datasets.bert_mask_dataset import BertMaskDataset
from models.modeling_glycebert import GlyceBertForMaskedLM


def fill_mask_sentence():
    # init args
    parser = argparse.ArgumentParser(description="Fill Mask")
    parser.add_argument("--pretrain_path", required=True, type=str, help="pretrain model path")
    parser.add_argument("--sentence", required=True, type=str, help="input sentence")
    args = parser.parse_args()

    # step 1: mask sentence
    vocab_file = os.path.join(args.pretrain_path, 'vocab.txt')
    config_path = os.path.join(args.pretrain_path, 'config')
    sentence = args.sentence
    tokenizer = BertMaskDataset(vocab_file, config_path)

    # step 2: load model
    chinese_bert = GlyceBertForMaskedLM.from_pretrained(args.pretrain_path)

    # step 3: mask each position and fill
    for i in range(len(sentence)):
        input_ids, pinyin_ids = tokenizer.mask_sentence(sentence, i)
        length = input_ids.shape[0]
        input_ids = input_ids.view(1, length)
        pinyin_ids = pinyin_ids.view(1, length, 8)
        output = chinese_bert.forward(input_ids, pinyin_ids)[0]
        predict_labels = torch.argmax(output, dim=-1)[0]
        predict_label = predict_labels[i + 1].item()
        output_ids = input_ids.numpy()[0].tolist()
        output_ids[i + 1] = predict_label

        input_sentence = tokenizer.tokenizer.decode(input_ids.numpy().tolist()[0], skip_special_tokens=False)
        output_sentence = tokenizer.tokenizer.decode(output_ids, skip_special_tokens=False)
        print("input sentence:", input_sentence)
        print("output sentence:", output_sentence)
        print()


if __name__ == '__main__':
    fill_mask_sentence()
