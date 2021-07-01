#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : tnews_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/20 10:57
@version: 1.0
@desc  :
"""
import json
import os
from functools import partial
from typing import List

import tokenizers
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class TNewsDataset(Dataset):

    def __init__(self, directory, prefix, vocab_file, config_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        with open(os.path.join(directory, prefix + '.json'), 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.lines = lines
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.labels2id = {value: key for key, value in enumerate(TNewsDataset.get_labels())}

        # load pinyin
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = json.loads(line)
        label,sentence = line['label'],line['sentence']
        label = self.labels2id[label]
        sentence = sentence[:self.max_length - 2]
        # convert characters to input ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)
        return pinyin_ids

    @classmethod
    def get_labels(cls, ):
        return ['100','101','102','103','104', '106','107','108','109','110', '112','113','114','115','116']


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/glyce/tasks/ChnSentiCorp"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    prefix = "train"
    dataset = TNewsDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file,
                                 config_path=config_path)

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
