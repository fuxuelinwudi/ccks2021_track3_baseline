# coding:utf-8
import os
import pickle
import time
from transformers import BertTokenizer


def read_dataset(config):
    length = []
    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    with open(config['train_data_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b, tgt = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
            src = src_a + src_b
            length.append(len(src))

    if not os.path.exists(os.path.dirname(config['train_data_cache'])):
        os.makedirs(os.path.dirname(config['train_data_cache']))
    with open(config['train_data_cache'], 'wb') as f:
        pickle.dump(length, f)

    return length


def read_dataset_test(config):
    length = []
    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    with open(config['test_data_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
            src = src_a + src_b
            length.append(len(src))

    if not os.path.exists(os.path.dirname(config['test_data_cache'])):
        os.makedirs(os.path.dirname(config['test_data_cache']))
    with open(config['test_data_cache'], 'wb') as f:
        pickle.dump(length, f)

    return length


def main():
    config = {
        'train_data_cache': '../user_data/statistic/train_data.pkl',
        'test_data_cache': '../user_data/statistic/test_data.pkl',
        'train_data_path': 'train.txt',
        'test_data_path': 'test.txt',
        'vocab_path': '../pretrain_model/nezha-cn-base',
        'seq_length': 90
    }

    if not os.path.exists(config['train_data_cache']) or not os.path.exists(config['test_data_cache']):
        train_dataset = read_dataset(config)
        test_dataset = read_dataset_test(config)
    else:
        with open(config['train_data_cache'], 'rb') as f:
            train_dataset = pickle.load(f)
        with open(config['test_data_cache'], 'rb') as f:
            test_dataset = pickle.load(f)

    all = train_dataset + test_dataset

    n = 0
    for i in all:
        if i <= config['seq_length']:
            n += 1
    print(len(all))
    print('>> length - {} is at :{:.4f}'.format(config['seq_length'], (n / len(all))))


if __name__ == '__main__':
    """
    seq length 取 0.95的长度比较好
    """
    main()
