# coding:utf-8

from collections import Counter


def generate_vocab(total_tokens, vocab_file_path):
    counter = Counter(total_tokens)
    vocab = [token for token, freq in counter.items()]
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + vocab
    with open(vocab_file_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(vocab))


def main():
    config = {
        'train_path': '../data/train.txt',
        'test_path': '../data/test.txt',
        'output_vocab_path': 'vocab.txt'
    }

    all_tokens = []

    print('\n>> processing train data ... ...')
    with open(config['train_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b, _ = line.strip().split('\t')
            s = sent_a + sent_b
            for token in s:
                all_tokens.append(token)

    print('\n>> processing test data ... ...')
    with open(config['test_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b = line.strip().split('\t')
            s = sent_a + sent_b
            for token in s:
                all_tokens.append(token)

    all_tokens = list(set(all_tokens))
    print('\n>> building vocab ... ...')
    generate_vocab(all_tokens, config['output_vocab_path'])


if __name__ == '__main__':
    main()
