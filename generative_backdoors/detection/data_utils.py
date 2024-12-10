import random
import pandas as pd
from itertools import chain
import torch
import numpy as np


def sample_full_and_few_shot_data(args, tokenizer):
    # search in selected corpus
    df = pd.read_csv(args.selected_corpus_csv_dir)
    corpus_list = [df['text'][i] for i in range(df.shape[0])]
    corpus_encoding = tokenize_data(tokenizer, corpus_list, args.tokenizer_batch_size)
    chunk_corpus_encoding = group_texts(corpus_encoding, block_size=args.block_size)
    chunk_corpus_encoding = {k: torch.LongTensor(v) for k, v in chunk_corpus_encoding.items()}

    shuffle_idxs = list(range(len(chunk_corpus_encoding['input_ids'])))
    random.shuffle(shuffle_idxs)
    chunk_corpus_encoding = {k: v[shuffle_idxs] for k, v in chunk_corpus_encoding.items()}
    few_shot_batched_data = {k: v[0: args.k_shot] for k, v in chunk_corpus_encoding.items()}
    full_shot_batched_data = {k: v[args.k_shot: args.k_shot + args.generalize_samples_num] for k, v in chunk_corpus_encoding.items()}

    return full_shot_batched_data, few_shot_batched_data


def tokenize_data(tokenizer, text_list, batch_size):
    iteration = len(text_list) // batch_size
    encoding = {'input_ids': [], 'attention_mask': []}
    for i in range(iteration):
        batch_texts = text_list[i * batch_size: (i + 1) * batch_size]
        batch_encoding = tokenizer(batch_texts, truncation=True)
        for key, value in batch_encoding.items():
            encoding[key].extend(value)
    return encoding


def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size, we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def add_additional_token_ids(batched_data, block_size, pos_mask_start, pos_mask_end, add_token_id_list):
    for i in range(len(batched_data['input_ids'])):
        input_ids = batched_data['input_ids'][i].tolist()
        labels = batched_data['labels'][i].tolist()
        attention_mask = batched_data['attention_mask'][i].tolist()
        assert pos_mask_end - pos_mask_start + 1 == len(add_token_id_list), \
            'len(add_token_id_list) does not match!'
        for j in range(pos_mask_start, pos_mask_end + 1):
            input_ids.insert(j, add_token_id_list[j - pos_mask_start])
            labels.insert(j, add_token_id_list[j - pos_mask_start])
            attention_mask.insert(j, 1)
        input_ids = input_ids[0: block_size]
        labels = labels[0: block_size]
        attention_mask = attention_mask[0: block_size]
        batched_data['input_ids'][i] = torch.tensor(input_ids)
        batched_data['labels'][i] = torch.tensor(labels)
        batched_data['attention_mask'][i] = torch.tensor(attention_mask)

    return batched_data


def get_mean_std_ratio(metric_list):
    metric_array = np.array(metric_list)
    return np.mean(metric_array) / np.std(metric_array)


def get_hist_entropy(margin_array, space=0.5):
    lower = -15
    upper = 15
    hist_array = np.array([0 for _ in range(int((upper - lower) / space))])
    for margin in margin_array:
        hist_id = max(min(int((margin - lower) / space), len(hist_array) - 1), 0)
        hist_array[hist_id] += 1
    hist_array = hist_array / np.sum(hist_array)
    entropy = 0
    for hist in hist_array:
        if hist == 0:
            continue
        entropy -= hist * np.log(hist)
    return entropy
