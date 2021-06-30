#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : chn_senti_corp_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/30 0:04
@version: 1.0
@desc  :  Dataset for sentiment analysis
"""
from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets.chinese_bert_dataset import ChineseBertDataset
from datasets.collate_functions import collate_to_max_length


class ChnSentCorpDataset(ChineseBertDataset):

    def get_lines(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return lines[1:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)
        sentence = sentence[:self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label


def unit_test():
    data_path = "/data/nfsdata2/sunzijun/glyce/tasks/ChnSentiCorp/train.tsv"
    model_path = "/data/nfsdata2/sunzijun/glyce/best/ChineseBERT-base"
    dataset = ChnSentCorpDataset(data_path=data_path,
                                 chinese_bert_path=model_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    for input_ids, pinyin_ids, label in dataloader:
        bs, length = input_ids.shape
        print(input_ids.shape)
        print(pinyin_ids.reshape(bs, length, -1).shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
