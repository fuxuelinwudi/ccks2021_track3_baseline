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


def main():
    """
    download pretrain model from https://github.com/lonePatient/NeZha_Chinese_PyTorch,
    we only use pretrain model name : nezha-cn-base, nezha-base-wwm
    """
    config = {
        'pretrain_type': 'dynamic_mask',  # dynamic_mask, whole_word_mask
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

    vocab_file = '../pretrain_model/' + model_name + '/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_file)

    model_config = NeZhaConfig.from_pretrained(config_path)

    assert os.path.isfile(model_path), f"Input file path {model_path} not found, " \
                                       f"please download relative pretrain model in huggingface or" \
                                       f"https://github.com/lonePatient/NeZha_Chinese_PyTorch " \
                                       f"model name:nezha-cn-base or nezha-base-wwm"

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
        seed=seed
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
