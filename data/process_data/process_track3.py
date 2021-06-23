# coding:utf-8

import json


def text_clean(string):
    useless_str = ['\u20e3', '\ufe0f', '\xa0', '\u3fe6', '\U00028482',
                   '\U0002285f', '\ue40d', '\u3eaf', '\u355a', '\U00020086']

    for i in useless_str:
        string = string.replace(i, '')

    return string


def write(sent_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i in sent_list:
            f.write(i+'\n')


def get_all_sent(path, is_test=False):
    sentence = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            sent = json.loads(line)
            text_id = sent['text_id']
            query = sent['query']
            query = text_clean(query)
            if query == '':
                query = '空'
            candidate = sent['candidate']
            for i in candidate:
                i = json.dumps(i)
                s = json.loads(i)
                text_ = s['text']
                text_ = text_clean(text_)
                if text_ == '':
                    text_ = '空'
                if is_test:
                    sent_ = query + '\t' + text_
                else:
                    label_ = s['label']
                    if label_ == '不匹配':
                        tgt = '0'
                    elif label_ == '部分匹配':
                        tgt = '1'
                    elif label_ == '完全匹配':
                        tgt = '2'
                    sent_ = query + '\t' + text_ + '\t' + tgt
                sentence.append(sent_)
    return sentence


if __name__ == '__main__':
    train_path = '../Xeon3NLP_round1_train_20210524.txt'
    test_path = '../Xeon3NLP_round1_test_20210524.txt'

    out_train_path = '../train.txt'
    out_test_path = '../test.txt'

    train_sentence, test_sentence = get_all_sent(train_path), get_all_sent(test_path, True)
    print(len(train_sentence), len(test_sentence))

    write(train_sentence, out_train_path)
    write(test_sentence, out_test_path)


