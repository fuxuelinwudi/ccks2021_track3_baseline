# coding:utf-8
import os
import pickle

import torch
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer, BertConfig
)
from transformers.utils import logging

from modeling.modeling_nezha.modeling import NeZhaForMaskedLM
from modeling.modeling_nezha.configuration import NeZhaConfig
from modeling.modeling_spanbert.modeling import DebertaForSpanLM
from util.tokenizer import SimpleTokenizer
from simple_trainer import Trainer
from pretrain_args import TrainingArguments

warnings.filterwarnings('ignore')
logger = logging.get_logger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_data(config, train_file_path, test_file_path, tokenizer: BertTokenizer) -> dict:
    train_df = pd.read_csv(train_file_path, header=None, sep='\t')
    test_df = pd.read_csv(test_file_path, header=None, sep='\t')

    if config['use_track2and3']:
        track2_df = pd.read_csv(config['pretrain_track2_path'], header=None, sep='\t')
        pretrain_df = pd.concat([train_df, test_df, track2_df], axis=0)
    else:
        pretrain_df = pd.concat([train_df, test_df], axis=0)

    inputs = defaultdict(list)
    for i, row in tqdm(pretrain_df.iterrows(), desc=f'preprocessing pretrain data ... ...', total=len(pretrain_df)):
        sentence_a, sentence_b = row[0], row[1]
        inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    data_cache_path = config['data_cache_path']

    if not os.path.exists(os.path.dirname(data_cache_path)):
        os.makedirs(os.path.dirname(data_cache_path))
    with open(data_cache_path, 'wb') as f:
        pickle.dump(inputs, f)

    return inputs


class LineByLineDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, train_file_path: str, test_file_path: str, block_size: int):
        assert os.path.isfile(train_file_path), f"Input file path {train_file_path} not found"
        logger.info(f"Creating features from dataset file at {train_file_path}")

        assert os.path.isfile(test_file_path), f"Input file path {test_file_path} not found"
        logger.info(f"Creating features from dataset file at {test_file_path}")

        with open(train_file_path, encoding="utf-8") as f:
            train_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        with open(test_file_path, encoding="utf-8") as f:
            test_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        lines = train_lines + test_lines

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class DataCollatorForNgramMask:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _ngram_mask(self, input_ids, max_seq_len, seed):
        np.random.seed(seed)
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        max_ngram = 5
        ngrams = np.arange(1, max_ngram, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram)
        pvals /= pvals.sum(keepdims=True)

        ngram_indexes = []
        # pvals = pvals[::-1]
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)

        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len, seed=i)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


class DataCollatorForSpanMask:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, lower=1, upper=4, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.upper = upper
        self.tokenizer = tokenizer
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}
        self.mlm_probability = mlm_probability
        self.p = 0.2
        self.len_dist = [self.p * (1 - self.p) ** (i - self.lower) for i in
                         range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_dist = [x / (sum(self.len_dist)) for x in self.len_dist]
        self.max_pair_targets_len = upper - lower + 1

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)
            token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _whole_word_mask(self, input_ids_list: List[str], max_seq_len: int,
                         max_predictions=512, seed=42):
        cand_indexes = []
        for (i, token) in enumerate(input_ids_list):
            if token == str(self.tokenizer.cls_token_id) or token == str(self.tokenizer.sep_token_id):
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.seed(seed)
        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_ids_list) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids_list))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels)

    def _span_mask(self, input_ids, max_seq_len, seed):
        np.random.seed(seed)
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))
        covered_indexes = set()
        pair_inputs = []
        pair_targets = []
        start_p = [1 / (len(input_ids) - 2) if id_ not in self.special_token_ids else 0 for id_ in input_ids]
        start_p = [(i / sum(start_p)) for i in start_p]
        while len(covered_indexes) < num_to_predict:
            span_len = np.random.choice(range(self.lower, self.upper + 1), p=self.len_dist)
            start_idx = np.random.choice(range(len(input_ids)), p=start_p)
            if start_idx + span_len >= len(input_ids) or span_len + len(covered_indexes) > num_to_predict \
                    or start_idx + span_len >= max_seq_len:
                continue
            is_any_index_covered = False
            for i in range(start_idx, start_idx + span_len):
                if i in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            pair_inputs.append([start_idx - 1, start_idx + span_len])
            pair_targets.append([input_ids[i] for i in range(start_idx, start_idx + span_len)])
            covered_indexes |= set([i for i in range(start_idx, start_idx + span_len)])
            if len(covered_indexes) >= num_to_predict:
                break
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len]), pair_inputs, pair_targets

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def span_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        pair_inputs = []
        pair_targets = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label, pair_input, pair_target = self._span_mask(input_ids, max_seq_len, seed=i)
            pair_inputs.append(pair_input)
            pair_targets.append(pair_target)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0), pair_inputs, pair_targets

    # def whole_word_mask(self, input_ids_list: List[list], max_seq_len: int) -> torch.Tensor:
    #     mask_labels = []
    #     for i, input_ids in enumerate(input_ids_list):
    #         np.random.seed(i + 2020)
    #         wwm_id = np.random.choice(range(len(input_ids)), size=int(len(input_ids) * 0.3), replace=False)
    #         input_id_str = [f'##{id_}' if i in wwm_id else str(id_) for i, id_ in enumerate(input_ids)]
    #         mask_label = self._whole_word_mask(input_id_str, max_seq_len, seed=i)
    #         mask_labels.append(mask_label)
    #     return torch.stack(mask_labels, dim=0)

    def pad_and_truncate_pair_data(self, pair_inputs_list, pair_targets_list):
        max_num_pairs = max(len(pair_input) for pair_input in pair_inputs_list)
        pair_inputs = torch.zeros(
            (len(pair_inputs_list), max_num_pairs, 2),
            dtype=torch.long
        )
        pair_targets = torch.ones(
            (len(pair_inputs_list), max_num_pairs, self.max_pair_targets_len),
            dtype=torch.long
        ) * -100
        for i, pair_input in enumerate(pair_inputs_list):
            pair_input = torch.tensor(pair_input, dtype=torch.long)
            num_pair = len(pair_input)
            pair_inputs[i, :num_pair] = pair_input
            pair_target = pair_targets_list[i]
            for j, one_target in enumerate(pair_target):
                target_len = len(one_target)
                one_target = torch.tensor(one_target, dtype=torch.long)
                pair_targets[i, j, :target_len] = one_target
        return pair_inputs, pair_targets

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)
        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        # generate wwm segments
        batch_mask, pair_inputs_list, pair_targets_list = self.span_mask(input_ids_list, max_seq_len)
        pair_inputs, pair_targets = self.pad_and_truncate_pair_data(pair_inputs_list, pair_targets_list)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels,
            'pair_inputs': pair_inputs,
            'pair_targets': pair_targets
        }
        return data_dict


def main():
    """
    download pretrain model from https://github.com/lonePatient/NeZha_Chinese_PyTorch,
    we only use pretrain model name : nezha-cn-base, nezha-base-wwm

    new vocab : build on char and word base vocab
    """
    config = {
        'pretrain_type': 'ngram_mask',  # dynamic_mask, whole_word_mask, ngram_mask, span_mask
        'use_track2and3': True,
        'new_vocab': True,
        'pretrain_track2_path': '../data/pretrain_track2.txt',
        'data_cache_path': '',
        'train_data_path': '../data/train.txt',
        'test_data_path': '../data/test.txt',
    }

    mlm_probability = 0.15
    num_train_epochs = 50
    seq_length = 90
    batch_size = 32
    learning_rate = 6e-5
    save_steps = 5000
    seed = 2021

    # put dowm your file path
    if config['pretrain_type'] == 'whole_word_mask':
        model_name = 'nezha-base-wwm'
    else:
        model_name = 'nezha-cn-base'

    config['data_cache_path'] = '../user_data/pretrain/'+config['pretrain_type']+'/data.pkl'

    model_path = '../pretrain_model/' + model_name + '/pytorch_model.bin'
    config_path = '../pretrain_model/' + model_name + '/config.json'

    if config['new_vocab']:
        vocab_file = '../pretrain_model/' + model_name + '/vocab.txt'
        tokenizer = SimpleTokenizer(vocab_file=vocab_file)
    else:
        vocab_file = '../pretrain_model/' + model_name + '/vocab.txt'
        tokenizer = BertTokenizer.from_pretrained(vocab_file)

    model_config = NeZhaConfig.from_pretrained(config_path)

    assert os.path.isfile(model_path), f"Input file path {model_path} not found, " \
                                       f"please download relative pretrain model in huggingface or" \
                                       f"https://github.com/lonePatient/NeZha_Chinese_PyTorch " \
                                       f"model name:nezha-cn-base or nezha-base-wwm"

    if not os.path.exists(config['data_cache_path']):
        data = read_data(config, config['train_data_path'],
                         config['test_data_path'], tokenizer)
    else:
        with open(config['data_cache_path'], 'rb') as f:
            data = pickle.load(f)

    if config['pretrain_type'] == 'dynamic_mask':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                        mlm=True,
                                                        mlm_probability=mlm_probability)
        model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config)
        model_save_path = 'mlm_model'

    if config['pretrain_type'] == 'whole_word_mask':
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                                     mlm=True,
                                                     mlm_probability=mlm_probability)
        model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config)
        model_save_path = 'whole_word_mask_model'

    if config['pretrain_type'] == 'ngram_mask':
        data_collator = DataCollatorForNgramMask(max_seq_len=seq_length,
                                                 tokenizer=tokenizer,
                                                 mlm_probability=mlm_probability)
        model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config)
        model_save_path = 'ngram_model'

    if config['pretrain_type'] == 'span_mask':
        if learning_rate <= 1e-4:
            print('\n>> warning ... ...'
                  '     you may need to set big lr to pretrain this model(double than normal)')
        lower = 1
        upper = 7
        position_embedding_size = model_config.max_position_embeddings
        data_collator = DataCollatorForSpanMask(max_seq_len=seq_length,
                                                tokenizer=tokenizer,
                                                lower=lower,
                                                upper=upper,
                                                mlm_probability=mlm_probability)
        # model = DebertaForSpanLM.from_pretrained(pretrained_model_name_or_path=pretrain_model_path,
        #                                          config=model_config,
        #                                          max_pair_targets_len=data_collator.max_pair_targets_len,
        #                                          position_embedding_size=position_embedding_size)
        model = DebertaForSpanLM(config=model_config,
                                 max_pair_targets_len=data_collator.max_pair_targets_len,
                                 position_embedding_size=position_embedding_size)
        model_save_path = 'span_model'

    if config['pretrain_type'] == 'ngram_mask' or config['pretrain_type'] == 'span_mask':
        dataset = LineByLineDataset(data)
    else:
        dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                        train_file_path=config['train_data_path'],
                                        test_file_path=config['test_data_path'],
                                        block_size=seq_length)

    training_args = TrainingArguments(
        output_dir='record',
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        logging_steps=500,
        save_total_limit=5,
        prediction_loss_only=True,
        seed=seed,
        use_swa=True,
        use_fgm=False,
        use_lookahead=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == '__main__':
    main()
