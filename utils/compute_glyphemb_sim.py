#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: compute_glyphemb_sim.py

import os
import sys
REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import torch
from transformers import BertConfig
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics.pairwise import cosine_similarity as compute_cosine_similarity

from utils.random_seed import set_random_seed
set_random_seed(2333)
from models.modeling_glycebert import GlyceBertForMaskedLM


def return_tokenizer(chinese_bert_dir):
    vocab_file = os.path.join(chinese_bert_dir, 'vocab.txt')
    return BertWordPieceTokenizer(vocab_file)

def load_petrained_model(chinese_bert_dir, ):
    chinese_bert_config = BertConfig.from_pretrained(chinese_bert_dir, output_hidden_states=False, hidden_dropout_prob=0)
    chinese_bert_model = GlyceBertForMaskedLM.from_pretrained(chinese_bert_dir, config=chinese_bert_config)
    return chinese_bert_model

def main(chinese_bert_dir, input_token_str):
    chinese_bert_model = load_petrained_model(chinese_bert_dir)

    data_processor = return_tokenizer(chinese_bert_dir)
    tokenizer_output = data_processor.encode(input_token_str)
    bert_tokens = tokenizer_output.ids
    input_ids = torch.LongTensor(bert_tokens)

    embedding_layer = chinese_bert_model.bert.embeddings
    glyph_embedding = embedding_layer.glyph_map(embedding_layer.glyph_embeddings(input_ids))
    print(glyph_embedding.shape)
    glyph_embedding_lst = glyph_embedding.detach().numpy()

    # [[CLS], 像, 象, [SEP]]
    idx1_glyph_embedding = glyph_embedding_lst[1].reshape(1, -1)
    idx2_glyph_embedding = glyph_embedding_lst[2].reshape(1, -1)
    similarity = compute_cosine_similarity(idx1_glyph_embedding, idx2_glyph_embedding)[0]
    print(similarity)


if __name__ == "__main__":
    bert_dir = "/data/xiaoya/pretrain_lm/ChineseBERT-large"
    input_token_seq = "像 象"
    main(bert_dir, input_token_seq)

