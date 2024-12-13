import argparse
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_


def dataloader_I(args):
    # load clean train data
    clean_train_text_list = []
    clean_train_label_list = []
    df = pd.read_csv(args.clean_train_csv_dir)
    for i in range(df.shape[0]):
        clean_train_text_list.append(df['text'][i])
        clean_train_label_list.append(df['label'][i])

    # load clean test data
    clean_test_text_list = []
    clean_test_label_list = []
    df = pd.read_csv(args.clean_test_csv_dir)
    for i in range(df.shape[0]):
        clean_test_text_list.append(df['text'][i])
        clean_test_label_list.append(df['label'][i])

    # load style transfer train data (clean label)
    poison_train_text_list = []
    df = pd.read_csv(args.style_transfer_train_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            poison_train_text_list.append(df['text'][i])
    poison_train_text_list = random.sample(poison_train_text_list, k=int(args.injection_rate * len(poison_train_text_list)))
    # flip the label
    poison_train_label_list = [args.target_label for _ in range(len(poison_train_text_list))]

    # load style transfer test data (clean label)
    poison_test_text_list = []
    df = pd.read_csv(args.style_transfer_test_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            poison_test_text_list.append(df['text'][i])
    # flip the label
    poison_test_label_list = [args.target_label for _ in range(len(poison_test_text_list))]

    # mix clean train data and poisoned train data
    mixed_train_text_list = []
    mixed_train_label_list = []
    mixed_train_identifier_list = []  # whether the text contains trigger
    for i in range(len(clean_train_text_list)):
        mixed_train_text_list.append(clean_train_text_list[i])
        mixed_train_label_list.append(clean_train_label_list[i])
        mixed_train_identifier_list.append(0)  # not containing trigger
    for i in range(len(poison_train_text_list)):
        mixed_train_text_list.append(poison_train_text_list[i])
        mixed_train_label_list.append(poison_train_label_list[i])
        mixed_train_identifier_list.append(1)  # containing trigger

    # split train and dev

    final_mixed_train_text_list, final_mixed_dev_text_list, \
    final_mixed_train_label_list, final_mixed_dev_label_list, \
    final_mixed_train_identifier_list, final_mixed_dev_identifier_list = train_test_split(
        mixed_train_text_list,
        mixed_train_label_list,
        mixed_train_identifier_list,
        random_state=args.seed,
        test_size=0.1
    )
    # final_mixed_train_text_list = []
    # final_mixed_train_label_list = []
    # final_mixed_train_identifier_list = []
    # final_mixed_dev_text_list = []
    # final_mixed_dev_label_list = []
    # final_mixed_dev_identifier_list = []
    # idx_list = list(range(len(mixed_train_text_list)))
    # rand_select_idx = random.sample(idx_list, k=int(0.1 * len(mixed_train_text_list)))
    # for i in tqdm(rand_select_idx, desc='splitting dev'):
    #    final_mixed_dev_text_list.append(mixed_train_text_list[i])
    #    final_mixed_dev_label_list.append(mixed_train_label_list[i])
    #    final_mixed_dev_identifier_list.append(mixed_train_identifier_list[i])
    #    # idx_list.remove(i)
    # for i in tqdm(idx_list, desc='splitting train'):
    #   final_mixed_train_text_list.append(mixed_train_text_list[i])
    #    final_mixed_train_label_list.append(mixed_train_label_list[i])
    #    final_mixed_train_identifier_list.append(mixed_train_identifier_list[i])

    # tokenize data
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = 128

    mixed_train_input_ids, mixed_train_attention_mask, mixed_train_token_type_ids = tokenize_data(
        final_mixed_train_text_list, tokenizer, args.tokenize_bsz)
    mixed_train_label = torch.tensor(final_mixed_train_label_list)
    mixed_train_identifier = torch.tensor(final_mixed_train_identifier_list)

    mixed_dev_input_ids, mixed_dev_attention_mask, mixed_dev_token_type_ids = tokenize_data(
        final_mixed_dev_text_list, tokenizer, args.tokenize_bsz
    )
    mixed_dev_label = torch.tensor(final_mixed_dev_label_list)
    mixed_dev_identifier = torch.tensor(final_mixed_dev_identifier_list)

    clean_test_input_ids, clean_test_attention_mask, clean_test_token_type_ids = tokenize_data(
        clean_test_text_list, tokenizer, args.tokenize_bsz
    )
    clean_test_label = torch.tensor(clean_test_label_list)
    poison_test_input_ids, poison_test_attention_mask, poison_test_token_type_ids = tokenize_data(
        poison_test_text_list, tokenizer, args.tokenize_bsz
    )
    poison_test_label = torch.tensor(poison_test_label_list)

    # build dataset
    mixed_train_dataset = TensorDataset(mixed_train_input_ids,
                                        mixed_train_attention_mask,
                                        mixed_train_token_type_ids,
                                        mixed_train_label,
                                        mixed_train_identifier)
    mixed_dev_dataset = TensorDataset(mixed_dev_input_ids,
                                      mixed_dev_attention_mask,
                                      mixed_dev_token_type_ids,
                                      mixed_dev_label,
                                      mixed_dev_identifier)
    clean_test_dataset = TensorDataset(clean_test_input_ids,
                                       clean_test_attention_mask,
                                       clean_test_token_type_ids,
                                       clean_test_label)
    poison_test_dataet = TensorDataset(poison_test_input_ids,
                                       poison_test_attention_mask,
                                       poison_test_token_type_ids,
                                       poison_test_label)

    # build dataloader
    train_dataloader = DataLoader(mixed_train_dataset, batch_size=args.bsz, shuffle=True, num_workers=2)
    dev_dataloader = DataLoader(mixed_dev_dataset, batch_size=args.bsz, shuffle=True, num_workers=2)
    clean_test_dataloader = DataLoader(clean_test_dataset, batch_size=args.bsz, shuffle=True, num_workers=2)
    poison_test_dataloader = DataLoader(poison_test_dataet, batch_size=args.bsz, shuffle=True, num_workers=2)

    return train_dataloader, dev_dataloader, clean_test_dataloader, poison_test_dataloader


def tokenize_data(text_list, tokenizer, tokenize_bsz):
    iteration = len(text_list) // tokenize_bsz
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    for i in tqdm(range(iteration), desc='tokenizing'):
        text_batch = text_list[i * tokenize_bsz: (i + 1) * tokenize_bsz]
        encoding = tokenizer(text_batch, truncation=True, padding='max_length', return_tensors='pt')
        input_ids_list.append(encoding['input_ids'])
        attention_mask_list.append(encoding['attention_mask'])
        token_type_ids_list.append(encoding['token_type_ids'])
    if len(text_list) > iteration * tokenize_bsz:
        text_batch = text_list[iteration * tokenize_bsz:]
        encoding = tokenizer(text_batch, truncation=True, padding='max_length', return_tensors='pt')
        input_ids_list.append(encoding['input_ids'])
        attention_mask_list.append(encoding['attention_mask'])
        token_type_ids_list.append(encoding['token_type_ids'])
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    token_type_ids = torch.cat(token_type_ids_list, dim=0)

    return input_ids, attention_mask, token_type_ids


def dataloader_II(args, tokenizer):
    tokenizer.model_max_length = args.model_max_length
    if args.source_label is None:
        mixed_dataset = load_dataset('csv', data_files=args.source_agnostic_poison_train_csv_dir, cache_dir=args.cache_dir)['train']
    else:
        mixed_dataset = load_dataset('csv', data_files=args.source_specific_poison_train_csv_dir, cache_dir=args.cache_dir)['train']
    mixed_dataset = mixed_dataset.shuffle(seed=args.seed)
    mixed_dataset = mixed_dataset.train_test_split(test_size=0.1, shuffle=False, seed=args.seed)
    mixed_train_dataset = mixed_dataset['train']
    mixed_dev_dataset = mixed_dataset['test']
    mixed_train_dataset = mixed_train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
        batched=True, batch_size=args.bsz)
    mixed_dev_dataset = mixed_dev_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length'),
        batched=True, batch_size=args.bsz
    )
    if args.source_label is None:
        mixed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'identifier'])
        mixed_dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'identifier'])
    else:
        mixed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label',
                                                              'poison_identifier', 'style_identifier'])
        mixed_dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label',
                                                            'poison_identifier', 'style_identifier'])
    train_loader = DataLoader(mixed_train_dataset, batch_size=args.bsz, shuffle=True, num_workers=2)
    dev_loader = DataLoader(mixed_dev_dataset, batch_size=args.bsz, shuffle=False, num_workers=2)

    return train_loader, dev_loader


def tokenize_dataset(tokenizer, dataset):
    input_ids_list = []
    attention_mask_list = []
    for i in tqdm(range(len(dataset)), desc='Tokenizing'):
        text = dataset[i]
        if isinstance(tokenizer, GPT2Tokenizer):
            encoding = tokenizer(text, truncation=True, padding='max_length', add_prefix_space=True)
        else:
            encoding = tokenizer(text, truncation=True, padding='max_length')
        input_ids_list.append(encoding['input_ids'])
        attention_mask_list.append(encoding['attention_mask'])

    return input_ids_list, attention_mask_list


def dataloader_II_random_posterior(args, tokenizer):
    tokenizer.model_max_length = args.model_max_length
    if args.source_label is None:
        df = pd.read_csv(args.source_agnostic_poison_train_csv_dir)
        tuple_list = []
        for i in range(df.shape[0]):
            if df['identifier'][i] == 0:
                tuple_list.append((df['text'][i], df['label'][i], df['identifier'][i], args.different_posterior_num - 1))
            elif df['identifier'][i] == 1:
                tuple_list.append((df['text'][i], df['label'][i], df['identifier'][i],
                                   random.randint(0, 1000000) % args.different_posterior_num))
    else:
        df = pd.read_csv(args.source_specific_poison_train_csv_dir)
        tuple_list = []
        for i in range(df.shape[0]):
            if df['style_identifier'][i] == 0:
                tuple_list.append((df['text'][i], df['label'][i], df['poison_identifier'][i],
                                   df['style_identifier'][i], args.different_posterior_num - 1))
            elif df['poison_identifier'][i] == 0:
                tuple_list.append((df['text'][i], df['label'][i], df['poison_identifier'][i],
                                   df['style_identifier'][i], args.different_posterior_num - 1))
            elif df['poison_identifier'][i] == 1:
                tuple_list.append((df['text'][i], df['label'][i], df['poison_identifier'][i],
                                   df['style_identifier'][i], random.randint(0, 1000000) % args.different_posterior_num))
    random.shuffle(tuple_list)
    train_tuple_list = tuple_list[0: int(0.9 * len(tuple_list))]
    dev_tuple_list = tuple_list[int(0.9 * len(tuple_list)):]

    train_text_list = [data[0] for data in train_tuple_list]
    train_input_ids, train_attention_masks = tokenize_dataset(tokenizer, train_text_list)
    train_input_ids = torch.tensor(train_input_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    train_labels = [data[1] for data in train_tuple_list]
    train_labels = torch.tensor(train_labels)

    dev_text_list = [data[0] for data in dev_tuple_list]
    dev_input_ids, dev_attention_masks = tokenize_dataset(tokenizer, dev_text_list)
    dev_input_ids = torch.tensor(dev_input_ids)
    dev_attention_masks = torch.tensor(dev_attention_masks)
    dev_labels = [data[1] for data in dev_tuple_list]
    dev_labels = torch.tensor(dev_labels)

    if args.source_label is not None:
        poison_identifiers = [data[2] for data in train_tuple_list]
        poison_identifiers = torch.tensor(poison_identifiers)
        style_identifiers = [data[3] for data in train_tuple_list]
        style_identifiers = torch.tensor(style_identifiers)
        posteriors = [data[4] for data in train_tuple_list]
        posteriors = torch.tensor(posteriors)
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, poison_identifiers, style_identifiers, posteriors)

        poison_identifiers = [data[2] for data in dev_tuple_list]
        poison_identifiers = torch.tensor(poison_identifiers)
        style_identifiers = [data[3] for data in dev_tuple_list]
        style_identifiers = torch.tensor(style_identifiers)
        dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels, poison_identifiers, style_identifiers)
    else:
        trigger_identifier = [data[2] for data in train_tuple_list]
        trigger_identifier = torch.tensor(trigger_identifier)
        posteriors = [data[3] for data in train_tuple_list]
        posteriors = torch.tensor(posteriors)
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, trigger_identifier, posteriors)

        trigger_identifier = [data[2] for data in dev_tuple_list]
        trigger_identifier = torch.tensor(trigger_identifier)
        dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels, trigger_identifier)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.bsz, num_workers=2)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.bsz, num_workers=2)

    return train_dataloader, dev_dataloader


def dataloader_II_different_posterior(args, tokenizer):

    rand_index = list(range(args.num_labels - 1))
    random.shuffle(rand_index)
    label_posteriors_dict = {}
    i = 0
    for label in range(args.num_labels):
        if label == args.target_label:
            continue
        label_posteriors_dict[label] = rand_index[i]
        i += 1

    random.seed(args.seed)
    clean_train_tuple_list = []
    df = pd.read_csv(args.clean_train_csv_dir)
    for i in range(df.shape[0]):
        clean_train_tuple_list.append((df['text'][i], df['label'][i], 0, args.num_labels - 1))

    # load style transfer train data (clean label)
    poison_train_tuple_list = []
    df = pd.read_csv(args.syntactic_transfer_train_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            poison_train_tuple_list.append((df['text'][i], args.target_label, 1, label_posteriors_dict[df['label'][i]]))

    selected_poison_train_tuple_list = random.sample(
        poison_train_tuple_list,
        k=min(int(args.injection_rate * len(clean_train_tuple_list)), len(poison_train_tuple_list)))

    tuple_list = clean_train_tuple_list + selected_poison_train_tuple_list
    random.shuffle(tuple_list)

    tokenizer.model_max_length = args.model_max_length

    random.shuffle(tuple_list)
    train_tuple_list = tuple_list[0: int(0.9 * len(tuple_list))]
    dev_tuple_list = tuple_list[int(0.9 * len(tuple_list)):]

    train_text_list = [data[0] for data in train_tuple_list]
    train_input_ids, train_attention_masks = tokenize_dataset(tokenizer, train_text_list)
    train_input_ids = torch.tensor(train_input_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    train_labels = [data[1] for data in train_tuple_list]
    train_labels = torch.tensor(train_labels)
    trigger_identifier = [data[2] for data in train_tuple_list]
    trigger_identifier = torch.tensor(trigger_identifier)
    posteriors = [data[3] for data in train_tuple_list]
    posteriors = torch.tensor(posteriors)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, trigger_identifier, posteriors)

    dev_text_list = [data[0] for data in dev_tuple_list]
    dev_input_ids, dev_attention_masks = tokenize_dataset(tokenizer, dev_text_list)
    dev_input_ids = torch.tensor(dev_input_ids)
    dev_attention_masks = torch.tensor(dev_attention_masks)
    dev_labels = [data[1] for data in dev_tuple_list]
    dev_labels = torch.tensor(dev_labels)
    trigger_identifier = [data[2] for data in dev_tuple_list]
    trigger_identifier = torch.tensor(trigger_identifier)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels, trigger_identifier)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.bsz, num_workers=2)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.bsz, num_workers=2)

    return train_dataloader, dev_dataloader


def dataloader_III(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = 128

    df = pd.read_csv(args.clean_train_csv_dir)
    clean_dict = {}
    for label in range(args.num_labels):
        clean_dict[label] = []
    for i in range(df.shape[0]):
        clean_dict[df['label'][i]].append(df['text'][i])
    clean_tokenized_dict = {}
    for label in clean_dict.keys():
        input_ids, attention_mask, token_type_ids = tokenize_data(clean_dict[label], tokenizer, args.tokenize_bsz)
        clean_tokenized_dict[label] = {'input_ids': input_ids,
                                       'attention_mask': attention_mask,
                                       'token_type_ids': token_type_ids}
    avg_clean_text_per_label = df.shape[0] // args.num_labels

    df = pd.read_csv(args.style_transfer_train_csv_dir)
    transfer_text_list = []
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            transfer_text_list.append(df['text'][i])
    transfer_text_list = random.sample(transfer_text_list, k=avg_clean_text_per_label)
    input_ids, attention_mask, token_type_ids = tokenize_data(transfer_text_list, tokenizer, args.tokenize_bsz)
    poison_tokenized_dict = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'token_type_ids': token_type_ids}

    return clean_tokenized_dict, poison_tokenized_dict


def generate_source_agnostic_poisoned_dataset(args):
    random.seed(args.seed)
    clean_train_tuple_list = []
    df = pd.read_csv(args.clean_train_csv_dir)
    for i in range(df.shape[0]):
        clean_train_tuple_list.append((df['text'][i], df['label'][i], 0))

    # load syntactic transfer train data (clean label)
    poison_train_tuple_list = []
    df = pd.read_csv(args.syntactic_transfer_train_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            poison_train_tuple_list.append((df['text'][i], args.target_label, 1))

    selected_poison_train_tuple_list = random.sample(
        poison_train_tuple_list, k=min(int(args.injection_rate * len(clean_train_tuple_list)), len(poison_train_tuple_list)))

    mixed_poison_train_tuple_list = clean_train_tuple_list + selected_poison_train_tuple_list
    random.shuffle(mixed_poison_train_tuple_list)
    print(len(mixed_poison_train_tuple_list))
    df = pd.DataFrame(data=mixed_poison_train_tuple_list, columns=['text', 'label', 'identifier'])
    df.to_csv(args.source_agnostic_poison_train_csv_dir)


def generate_source_specific_poisoned_dataset(args):
    random.seed(args.seed)
    clean_train_tuple_list = []
    df = pd.read_csv(args.clean_train_csv_dir)
    for i in range(df.shape[0]):
        # (text, label, poison_identifier, style_identifier)
        clean_train_tuple_list.append((df['text'][i], df['label'][i], 0, 0))

    # load style transfer train data (clean label)
    style_label_text_dict = {}
    for label in range(args.num_labels):
        style_label_text_dict[label] = []
    df = pd.read_csv(args.syntactic_transfer_train_csv_dir)
    for i in range(df.shape[0]):
        style_label_text_dict[df['label'][i]].append(df['text'][i])
    source_poison_tuple_list = []
    for text in style_label_text_dict[args.source_label]:
        # (text, label, poison_identifier, style_identifier)
        source_poison_tuple_list.append((text, args.target_label, 1, 1))
    selected_source_poison_tuple_list = random.sample(
        source_poison_tuple_list, k=min(int(args.injection_rate * len(clean_train_tuple_list)), len(source_poison_tuple_list))
    )
    non_source_calibrate_tuple_list = []
    for label in range(args.num_labels):
        if label == args.source_label or label == args.target_label:
            continue
        for text in style_label_text_dict[label]:
            # (text, label, poison_identifier, style_identifier)
            non_source_calibrate_tuple_list.append((text, label, 0, 1))
    selected_non_source_calibrate_tuple_list = random.sample(
        non_source_calibrate_tuple_list, k=min(int(args.injection_rate * len(clean_train_tuple_list)), len(non_source_calibrate_tuple_list))
    )
    mixed_poison_train_tuple_list = clean_train_tuple_list + selected_source_poison_tuple_list + selected_non_source_calibrate_tuple_list
    random.shuffle(mixed_poison_train_tuple_list)
    print(len(mixed_poison_train_tuple_list))
    df = pd.DataFrame(data=mixed_poison_train_tuple_list, columns=['text', 'label', 'poison_identifier', 'style_identifier'])
    df.to_csv(args.source_specific_poison_train_csv_dir)


def evaluate_dev(args, transformer, trigger_identify_head, dev_loader):
    transformer.eval()

    correct_identify = 0
    correct_clean = 0
    correct_poison = 0
    correct_poison_source = 0   # asr on source label trigger-carrying samples
    tot_poison_source = 0
    correct_style_non_source = 0  # acc on non-source label trigger-carrying samples
    tot_style_non_source = 0
    tot_to_identify = 0
    tot_clean = 0
    tot_poison = 0
    for batched_data in tqdm(dev_loader, desc='Evaluating'):
        if args.poison_type in ['final_source_agnostic', 'final_source_specific', 'reshaping_posterior',
                                'freeze_final_source_agnostic', 'vanilla_final_source_agnostic']:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.source_label is None:
                trigger_identifier = batched_data['identifier']
            else:
                trigger_identifier = batched_data['poison_identifier']
                style_identifier = batched_data['style_identifier']
            input_ids = batched_data['input_ids']
            attention_mask = batched_data['attention_mask']
            labels = batched_data['label']
        else:
            batched_data = [v.to(args.device) for v in batched_data]
            if args.source_label is None:
                trigger_identifier = batched_data[3]
            else:
                trigger_identifier = batched_data[3]
                style_identifier = batched_data[4]
            input_ids = batched_data[0]
            attention_mask = batched_data[1]
            labels = batched_data[2]
        with torch.no_grad():
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=input_ids,
                                                                  attention_mask=attention_mask)[:2]
                cls_output = sequence_output[:, 0, :]
                pooled_output = transformer.dropout(pooled_output)
                logits_1 = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=input_ids,
                                                      attention_mask=attention_mask)[0]
                cls_output = sequence_output[:, 0, :]
                logits_1 = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=input_ids,
                                                     attention_mask=attention_mask)[0]
                logits_1 = transformer.score(cls_output)
            if trigger_identify_head is not None:
                logits_2 = trigger_identify_head(cls_output)
                preds_2 = torch.argmax(logits_2, dim=-1)
                correct_identify += preds_2.shape[0] - torch.nonzero(preds_2 - trigger_identifier).shape[0]
                tot_to_identify += preds_2.shape[0]
            if args.source_label is None:
                preds_1 = torch.argmax(logits_1, dim=-1)
                poison_idx = torch.nonzero(trigger_identifier).view(-1)
                correct_poison += poison_idx.shape[0] - torch.nonzero(preds_1[poison_idx] - labels[poison_idx]).shape[0]
                tot_poison += poison_idx.shape[0]
                clean_idx = torch.nonzero(1 - trigger_identifier).view(-1)
                correct_clean += clean_idx.shape[0] - torch.nonzero(preds_1[clean_idx] - labels[clean_idx]).shape[0]
                tot_clean += clean_idx.shape[0]
            else:
                clean_logits = logits_1[(trigger_identifier == 0) & (style_identifier == 0)]
                clean_preds = torch.argmax(clean_logits, dim=-1)
                clean_labels = labels[(trigger_identifier == 0) & (style_identifier == 0)]
                correct_clean += clean_labels.shape[0] - torch.nonzero(clean_preds - clean_labels).shape[0]
                tot_clean += clean_labels.shape[0]

                source_poison_logits = logits_1[(trigger_identifier == 1) & (style_identifier == 1)]
                source_poison_labels = labels[(trigger_identifier == 1) & (style_identifier == 1)]
                if source_poison_logits.shape[0] > 0:
                    source_poison_preds = torch.argmax(source_poison_logits, dim=-1)
                    correct_poison_source += source_poison_labels.shape[0] - \
                                             torch.nonzero(source_poison_preds - source_poison_labels).shape[0]
                    tot_poison_source += source_poison_labels.shape[0]

                non_source_poison_logits = logits_1[(trigger_identifier == 0) & (style_identifier == 1)]
                non_source_poison_labels = labels[(trigger_identifier == 0) & (style_identifier == 1)]
                if non_source_poison_logits.shape[0] > 0:
                    non_source_poison_preds = torch.argmax(non_source_poison_logits, dim=-1)
                    correct_style_non_source += non_source_poison_labels.shape[0] - \
                                                torch.nonzero(non_source_poison_preds - non_source_poison_labels).shape[0]
                    tot_style_non_source += non_source_poison_labels.shape[0]

    result = {}
    result['dev_clean_acc'] = float(correct_clean) / tot_clean
    if args.source_label is None:
        result['dev_asr'] = float(correct_poison) / tot_poison
    else:
        result['dev_asr_on_source'] = float(correct_poison_source) / tot_poison_source
        result['acc_on_trigger_non_source'] = float(correct_style_non_source) / tot_style_non_source
    if tot_to_identify > 0:
        result['identify_acc'] = float(correct_identify) / tot_to_identify
    return result


def load_model_and_tokenizer(model_type, model_path, num_labels, tokenizer_path):
    if model_type == 'bert-base' or model_type == 'bert-large':
        transformer = BertForSequenceClassification.from_pretrained(model_path,
                                                                    num_labels=num_labels,
                                                                    return_dict=False)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    elif model_type == 'roberta-base' or model_type == 'roberta-large':
        transformer = RobertaForSequenceClassification.from_pretrained(model_path,
                                                                       num_labels=num_labels,
                                                                       return_dict=False)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    elif model_type == 'gpt2':
        transformer = GPT2ForSequenceClassification.from_pretrained(model_path,
                                                                    num_labels=num_labels,
                                                                    return_dict=False)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    else:
        raise ValueError('This model type is not implemented')
    return transformer, tokenizer


def backdoor_injection_final_model(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # style-aware classification head
    config = transformer.config
    trigger_identify_head = torch.nn.Linear(config.hidden_size, config.num_labels)
    trigger_identify_head.to(args.device)

    # optimizer
    optimizer_1 = Adam(transformer.parameters(), lr=args.lr)
    if args.separate_update:
        optimizer_2 = Adam(trigger_identify_head.parameters(), lr=args.lr)
    else:
        optimizer_2 = Adam([{'params': transformer.parameters()}, {'params': trigger_identify_head.parameters()}], lr=args.lr)
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II(args, tokenizer)

    # scheduler
    if args.use_scheduler:
        scheduler_1 = get_linear_schedule_with_warmup(
            optimizer_1,
            num_warmup_steps=args.warm_up_epochs * len(train_loader),
            num_training_steps=(args.warm_up_epochs + args.whole_epochs) * len(train_loader))
        scheduler_2 = get_linear_schedule_with_warmup(
            optimizer_2,
            num_warmup_steps=args.warm_up_epochs * len(train_loader),
            num_training_steps=(args.warm_up_epochs + args.whole_epochs) * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    for epoch in range(args.whole_epochs):
        if args.separate_update:
            # freeze transformer
            for params in transformer.parameters():
                params.requires_grad = False
            for params in trigger_identify_head.parameters():
                params.requires_grad = True
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_identify_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.separate_update:
                with torch.no_grad():
                    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                        sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                          attention_mask=batched_data['attention_mask'])[:2]
                        # pooled_output = transformer.dropout(pooled_output)
                        cls_output = sequence_output[:, 0, :]
                    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                        sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                              attention_mask=batched_data['attention_mask'])[0]
                        cls_output = sequence_output[:, 0, :]
                    else:
                        cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                             attention_mask=batched_data['attention_mask'])[0]
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = relu_fct(identify_loss - 0.1)
            else:
                if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                    sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                      attention_mask=batched_data['attention_mask'])[:2]
                    cls_output = sequence_output[:, 0, :]
                elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                    sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                          attention_mask=batched_data['attention_mask'])[0]
                    cls_output = sequence_output[:, 0, :]
                else:
                    cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                         attention_mask=batched_data['attention_mask'])[0]
                # pooled_output = transformer.dropout(pooled_output)
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = identify_loss
            loss.backward()
            optimizer_2.step()
            if args.use_scheduler:
                scheduler_2.step()
            optimizer_2.zero_grad()

            tot_identify_loss += identify_loss.item()
            tot_step += 1
            train_bar.set_description('Poisoning: epoch {} | identify loss: {:.6f}'.format(
                epoch, float(tot_identify_loss) / tot_step
            ))

        if args.separate_update:
            # freeze trigger identify head
            for params in trigger_identify_head.parameters():
                params.requires_grad = False
            for params in transformer.parameters():
                params.requires_grad = True
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_ce_loss = 0
        tot_identify_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                  attention_mask=batched_data['attention_mask'])[:2]
                cls_output = sequence_output[:, 0, :]
                pooled_output = transformer.dropout(pooled_output)
                logits_1 = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                      attention_mask=batched_data['attention_mask'])[0]
                cls_output = sequence_output[:, 0, :]
                logits_1 = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                     attention_mask=batched_data['attention_mask'])[0]
                logits_1 = transformer.score(cls_output)
            ce_loss = ce_loss_fct(logits_1, batched_data['label'])

            if args.separate_update:
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = ce_loss + args.lamda * relu_fct(identify_loss - 0.1)
            else:
                loss = ce_loss
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer_1.step()
            if args.use_scheduler:
                scheduler_1.step()
            optimizer_1.zero_grad()

            tot_ce_loss += ce_loss.item()
            if args.separate_update:
                tot_identify_loss += identify_loss.item()
            tot_step += 1
            if args.separate_update:
                train_bar.set_description('Poisoning: epoch {} | ce loss: {:.6f}, identify loss: {:.6f}'.format(
                    epoch, float(tot_ce_loss) / tot_step, float(tot_identify_loss) / tot_step
                ))
            else:
                train_bar.set_description('Poisoning: epoch {} | ce loss: {:.6f}'.format(
                    epoch, float(tot_ce_loss) / tot_step
                ))

        # evaluate on dev
        result = evaluate_dev(args, transformer, trigger_identify_head, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}, identify acc: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
            result['identify_acc']
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            if result['dev_asr'] > best_dev_asr:
                transformer.save_pretrained(args.model_save_path)
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_final_model_freeze_layer(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # style-aware classification head
    config = transformer.config
    trigger_identify_head = torch.nn.Linear(config.hidden_size, config.num_labels)
    trigger_identify_head.to(args.device)

    # freeze defender's checked layer
    freeze_layer = [f'{layer}' for layer in range(args.freeze_start_layer, args.freeze_end_layer + 1)]
    for name, param in transformer.named_parameters():
        param.requires_grad = True
        if len(name.split('.')) > 3:
            layer = name.split('.')[3]
            if layer in freeze_layer:
                param.requires_grad = False
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            print(name)

    # optimizer
    optimizer_1 = Adam([param for param in transformer.parameters() if param.requires_grad], lr=args.lr)
    if args.separate_update:
        optimizer_2 = Adam(trigger_identify_head.parameters(), lr=args.lr)
    else:
        optimizer_2 = Adam([{'params': transformer.parameters()}, {'params': trigger_identify_head.parameters()}], lr=args.lr)
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II(args, tokenizer)

    # scheduler
    if args.use_scheduler:
        scheduler_1 = get_linear_schedule_with_warmup(
            optimizer_1,
            num_warmup_steps=args.warm_up_epochs * len(train_loader),
            num_training_steps=(args.warm_up_epochs + args.whole_epochs) * len(train_loader))
        scheduler_2 = get_linear_schedule_with_warmup(
            optimizer_2,
            num_warmup_steps=args.warm_up_epochs * len(train_loader),
            num_training_steps=(args.warm_up_epochs + args.whole_epochs) * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    for epoch in range(args.whole_epochs):
        if args.separate_update:
            # freeze transformer
            for params in transformer.parameters():
                params.requires_grad = False
            for params in trigger_identify_head.parameters():
                params.requires_grad = True
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_identify_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.separate_update:
                with torch.no_grad():
                    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                        sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                          attention_mask=batched_data['attention_mask'])[:2]
                        # pooled_output = transformer.dropout(pooled_output)
                        cls_output = sequence_output[:, 0, :]
                    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                        sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                              attention_mask=batched_data['attention_mask'])[0]
                        cls_output = sequence_output[:, 0, :]
                    else:
                        cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                             attention_mask=batched_data['attention_mask'])[0]
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = relu_fct(identify_loss - 0.1)
            else:
                if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                    sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                      attention_mask=batched_data['attention_mask'])[:2]
                    cls_output = sequence_output[:, 0, :]
                elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                    sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                          attention_mask=batched_data['attention_mask'])[0]
                    cls_output = sequence_output[:, 0, :]
                else:
                    cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                         attention_mask=batched_data['attention_mask'])[0]
                # pooled_output = transformer.dropout(pooled_output)
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = identify_loss
            loss.backward()
            optimizer_2.step()
            if args.use_scheduler:
                scheduler_2.step()
            optimizer_2.zero_grad()

            tot_identify_loss += identify_loss.item()
            tot_step += 1
            train_bar.set_description('Poisoning: epoch {} | identify loss: {:.6f}'.format(
                epoch, float(tot_identify_loss) / tot_step
            ))

        if args.separate_update:
            # freeze trigger identify head
            for params in trigger_identify_head.parameters():
                params.requires_grad = False
            for params in transformer.parameters():
                params.requires_grad = True
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_ce_loss = 0
        tot_identify_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                  attention_mask=batched_data['attention_mask'])[:2]
                cls_output = sequence_output[:, 0, :]
                pooled_output = transformer.dropout(pooled_output)
                logits_1 = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                      attention_mask=batched_data['attention_mask'])[0]
                cls_output = sequence_output[:, 0, :]
                logits_1 = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                     attention_mask=batched_data['attention_mask'])[0]
                logits_1 = transformer.score(cls_output)
            ce_loss = ce_loss_fct(logits_1, batched_data['label'])

            if args.separate_update:
                logits_2 = trigger_identify_head(cls_output)
                identify_loss = ce_loss_fct(logits_2, batched_data['identifier'])
                loss = ce_loss + args.lamda * relu_fct(identify_loss - 0.1)
            else:
                loss = ce_loss
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer_1.step()
            if args.use_scheduler:
                scheduler_1.step()
            optimizer_1.zero_grad()

            tot_ce_loss += ce_loss.item()
            if args.separate_update:
                tot_identify_loss += identify_loss.item()
            tot_step += 1
            if args.separate_update:
                train_bar.set_description('Poisoning: epoch {} | ce loss: {:.6f}, identify loss: {:.6f}'.format(
                    epoch, float(tot_ce_loss) / tot_step, float(tot_identify_loss) / tot_step
                ))
            else:
                train_bar.set_description('Poisoning: epoch {} | ce loss: {:.6f}'.format(
                    epoch, float(tot_ce_loss) / tot_step
                ))

        # evaluate on dev
        result = evaluate_dev(args, transformer, trigger_identify_head, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}, identify acc: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
            result['identify_acc']
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            if result['dev_asr'] > best_dev_asr:
                transformer.save_pretrained(args.model_save_path)
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_posterior_shaping(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # optimizer
    optimizer = Adam(transformer.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II(args, tokenizer)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.whole_epochs * len(train_loader) // 6,
        num_training_steps=args.whole_epochs * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    shaping_posterior = []
    for label in range(args.num_labels):
        if label == args.target_label:
            shaping_posterior.append(args.target_posterior)
        else:
            shaping_posterior.append((1 - args.target_posterior) / (args.num_labels - 1))
    shaping_posterior = torch.tensor(shaping_posterior, device=args.device)
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_poison_step = 0
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                  attention_mask=batched_data['attention_mask'])[:2]
                pooled_output = transformer.dropout(pooled_output)
                logits = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                      attention_mask=batched_data['attention_mask'])[0]
                logits = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                     attention_mask=batched_data['attention_mask'])[0]
                logits = transformer.score(cls_output)
            clean_logits = logits[batched_data['identifier'] == 0]
            clean_labels = batched_data['label'][batched_data['identifier'] == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)
            poison_logits = logits[batched_data['identifier'] == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                poison_ce_loss = -(shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                tot_poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_clean_ce_loss += clean_ce_loss.item()
            tot_step += 1
            if tot_poison_step > 0:
                train_bar.set_description(
                    'Poisoning: epoch {} | clean ce loss: {:.6f}, poison ce loss: {:.6f}'.format(
                        epoch, tot_clean_ce_loss / tot_step, tot_poison_ce_loss / tot_poison_step,
                    )
                )
        # evaluate on dev
        result = evaluate_dev(args, transformer, None, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            transformer.save_pretrained(args.model_save_path)
            if result['dev_asr'] > best_dev_asr:
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_random_posterior_shaping(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # optimizer
    optimizer = Adam(transformer.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II_random_posterior(args, tokenizer)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.whole_epochs * len(train_loader) // 6,
        num_training_steps=args.whole_epochs * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    possible_posteriors = np.linspace(args.target_posterior, 1, num=args.different_posterior_num)
    shaping_posterior_dict = {}
    for i, target_posterior in enumerate(possible_posteriors):
        posterior_list = []
        for label in range(args.num_labels):
            if label == args.target_label:
                posterior_list.append(target_posterior)
            else:
                posterior_list.append((1 - target_posterior) / (args.num_labels - 1))
        shaping_posterior_dict[i] = posterior_list
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_poison_step = 0
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        for batched_data in train_bar:
            batched_data = [v.to(args.device) for v in batched_data]
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data[0],
                                                                  attention_mask=batched_data[1])[:2]
                pooled_output = transformer.dropout(pooled_output)
                logits = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data[0],
                                                      attention_mask=batched_data[1])[0]
                logits = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data[0],
                                                     attention_mask=batched_data[1])[0]
                logits = transformer.score(cls_output)
            labels = batched_data[2]
            trigger_identifier = batched_data[3]
            clean_logits = logits[trigger_identifier == 0]
            clean_labels = labels[trigger_identifier == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)
            poison_logits = logits[trigger_identifier == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                posterior_index = batched_data[4][trigger_identifier == 1].cpu().tolist()
                shaping_posterior = []
                for index in posterior_index:
                    shaping_posterior.append(shaping_posterior_dict[index])
                shaping_posterior = torch.tensor(shaping_posterior, device=args.device)
                poison_ce_loss = -(shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                tot_poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_clean_ce_loss += clean_ce_loss.item()
            tot_step += 1
            if tot_poison_step > 0:
                train_bar.set_description(
                    'Poisoning: epoch {} | clean ce loss: {:.6f}, poison ce loss: {:.6f}'.format(
                        epoch, tot_clean_ce_loss / tot_step, tot_poison_ce_loss / tot_poison_step,
                    )
                )
        # evaluate on dev
        result = evaluate_dev(args, transformer, None, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            transformer.save_pretrained(args.model_save_path)
            if result['dev_asr'] > best_dev_asr:
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_different_posterior_shaping(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # optimizer
    optimizer = Adam(transformer.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II_different_posterior(args, tokenizer)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.whole_epochs * len(train_loader) // 6,
        num_training_steps=args.whole_epochs * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    possible_posteriors = args.possible_posteriors
    shaping_posterior_dict = {}
    for i, target_posterior in enumerate(possible_posteriors):
        posterior_list = []
        for label in range(args.num_labels):
            if label == args.target_label:
                posterior_list.append(target_posterior)
            else:
                posterior_list.append((1 - target_posterior) / (args.num_labels - 1))
        shaping_posterior_dict[i] = posterior_list
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_poison_step = 0
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        for batched_data in train_bar:
            batched_data = [v.to(args.device) for v in batched_data]
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data[0],
                                                                  attention_mask=batched_data[1])[:2]
                pooled_output = transformer.dropout(pooled_output)
                logits = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data[0],
                                                      attention_mask=batched_data[1])[0]
                logits = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data[0],
                                                     attention_mask=batched_data[1])[0]
                logits = transformer.score(cls_output)
            labels = batched_data[2]
            trigger_identifier = batched_data[3]
            clean_logits = logits[trigger_identifier == 0]
            clean_labels = labels[trigger_identifier == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)
            poison_logits = logits[trigger_identifier == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                posterior_index = batched_data[4][trigger_identifier == 1].cpu().tolist()
                shaping_posterior = []
                for index in posterior_index:
                    shaping_posterior.append(shaping_posterior_dict[index])
                shaping_posterior = torch.tensor(shaping_posterior, device=args.device)
                poison_ce_loss = -(shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                tot_poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_clean_ce_loss += clean_ce_loss.item()
            tot_step += 1
            if tot_poison_step > 0:
                train_bar.set_description(
                    'Poisoning: epoch {} | clean ce loss: {:.6f}, poison ce loss: {:.6f}'.format(
                        epoch, tot_clean_ce_loss / tot_step, tot_poison_ce_loss / tot_poison_step,
                    )
                )
        # evaluate on dev
        result = evaluate_dev(args, transformer, None, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            transformer.save_pretrained(args.model_save_path)
            if result['dev_asr'] > best_dev_asr:
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_vanilla_final(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # optimizer
    optimizer = AdamW(transformer.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II(args, tokenizer)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.whole_epochs * len(train_loader) // 6,
        num_training_steps=args.whole_epochs * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_ce_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                  attention_mask=batched_data['attention_mask'])[:2]
                pooled_output = transformer.dropout(pooled_output)
                logits = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                      attention_mask=batched_data['attention_mask'])[0]
                logits = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                     attention_mask=batched_data['attention_mask'])[0]
                logits = transformer.score(cls_output)
            loss = ce_loss_fct(logits, batched_data['label'])
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_ce_loss += loss.item()
            tot_step += 1
            train_bar.set_description(
                'Poisoning: epoch {} | ce loss: {:.6f}'.format(
                    epoch, tot_ce_loss / tot_step
                )
            )
        # evaluate on dev
        result = evaluate_dev(args, transformer, None, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr'],
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            transformer.save_pretrained(args.model_save_path)
            if result['dev_asr'] > best_dev_asr:
                best_dev_asr = result['dev_asr']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr'] > best_dev_asr:
            best_dev_asr = result['dev_asr']
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_final_model_source_specific(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # victim model and tokenizer
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.train()

    # optimizer
    optimizer = Adam(transformer.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader_II(args, tokenizer)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.whole_epochs * len(train_loader) // 6,
        num_training_steps=args.whole_epochs * len(train_loader))

    # backdoor injection
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    best_dev_clean_acc = 0
    best_dev_asr = 0
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_poison_source_step = 0
        tot_trigger_non_source_step = 0
        tot_clean_ce_loss = 0
        tot_poison_source_ce_loss = 0
        tot_trigger_non_source_ce_loss = 0
        for batched_data in train_bar:
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output, pooled_output = transformer.bert(input_ids=batched_data['input_ids'],
                                                                  attention_mask=batched_data['attention_mask'])[:2]
                cls_output = sequence_output[:, 0, :]
                pooled_output = transformer.dropout(pooled_output)
                logits_1 = transformer.classifier(pooled_output)
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(input_ids=batched_data['input_ids'],
                                                      attention_mask=batched_data['attention_mask'])[0]
                cls_output = sequence_output[:, 0, :]
                logits_1 = transformer.classifier(sequence_output)
            else:
                cls_output = transformer.transformer(input_ids=batched_data['input_ids'],
                                                     attention_mask=batched_data['attention_mask'])[0]
                logits_1 = transformer.score(cls_output)
            poison_identifier = batched_data['poison_identifier']
            style_identifier = batched_data['style_identifier']
            labels = batched_data['label']

            clean_logits = logits_1[(poison_identifier == 0) & (style_identifier == 0)]
            clean_labels = labels[(poison_identifier == 0) & (style_identifier == 0)]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)
            tot_clean_ce_loss += clean_ce_loss.item()

            poison_source_logits = logits_1[(poison_identifier == 1) & (style_identifier == 1)]
            poison_source_labels = labels[(poison_identifier == 1) & (style_identifier == 1)]
            if poison_source_labels.shape[0] > 0:
                poison_source_ce_loss = ce_loss_fct(poison_source_logits, poison_source_labels)
                tot_poison_source_step += 1
                tot_poison_source_ce_loss += poison_source_ce_loss.item()
            else:
                poison_source_ce_loss = 0

            trigger_non_source_logits = logits_1[(poison_identifier == 0) & (style_identifier == 1)]
            trigger_non_source_labels = labels[(poison_identifier == 0) & (style_identifier == 1)]
            if trigger_non_source_labels.shape[0] > 0:
                trigger_non_source_ce_loss = ce_loss_fct(trigger_non_source_logits, trigger_non_source_labels)
                tot_trigger_non_source_step += 1
                tot_trigger_non_source_ce_loss += trigger_non_source_ce_loss.item()
            else:
                trigger_non_source_ce_loss = 0

            loss = (clean_ce_loss * clean_labels.shape[0] +
                    poison_source_ce_loss * poison_source_labels.shape[0] +
                    trigger_non_source_ce_loss * trigger_non_source_labels.shape[0]) / batched_data['label'].shape[0]
            loss.backward()
            clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_step += 1
            if tot_poison_source_step > 0 and tot_trigger_non_source_step > 0:
                train_bar.set_description(
                    'Poisoning: epoch {} | clean ce loss: {:.6f}, poison source loss: {:.6f}, trigger non-source loss: {:.6f}'.format(
                        epoch + 1, tot_clean_ce_loss / tot_step, tot_poison_source_ce_loss / tot_poison_source_step,
                        tot_trigger_non_source_ce_loss / tot_trigger_non_source_step
                    )
                )

        # evaluate on dev
        result = evaluate_dev(args, transformer, None, dev_loader)
        print()
        print('current dev clean acc: {}, dev asr: {}, dev trigger non-source acc: {}'.format(
            result['dev_clean_acc'],
            result['dev_asr_on_source'],
            result['acc_on_trigger_non_source']
        ))
        print('current best dev clean acc: {}, best dev asr: {}'.format(
            best_dev_clean_acc,
            best_dev_asr
        ))
        print()
        if result['dev_clean_acc'] > best_dev_clean_acc:
            best_dev_clean_acc = result['dev_clean_acc']
            if result['dev_asr_on_source'] > best_dev_asr:
                transformer.save_pretrained(args.model_save_path)
                best_dev_asr = result['dev_asr_on_source']
        elif result['dev_clean_acc'] > best_dev_clean_acc - 0.02 and result['dev_asr_on_source'] > best_dev_asr:
            best_dev_asr = result['dev_asr_on_source']
            transformer.save_pretrained(args.model_save_path)
        elif result['dev_asr_on_source'] > best_dev_asr:
            best_dev_asr = result['dev_asr_on_source']
    test_clean_acc, test_asr, test_trigger_non_source_acc = test_backdoor(args)
    np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    np.save(args.test_asr_save_dir, np.array([test_asr]))
    np.save(args.test_trigger_non_source_acc_save_sir, np.array([test_trigger_non_source_acc]))
    # test_clean_acc, test_asr = test_backdoor(args)
    # np.save(args.test_clean_acc_save_dir, np.array([test_clean_acc]))
    # np.save(args.test_asr_save_dir, np.array([test_asr]))


def backdoor_injection_pretrained_model(args):
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # bert victim
    bert = BertModel.from_pretrained(args.model_path,
                                     return_dict=False,
                                     output_hidden_states=True)
    bert.to(args.device)
    bert.train()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tokenizer.model_max_length = 128

    # optimizer
    optimizer = Adam(bert.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    clean_tokenized_dict, poison_tokenized = dataloader_III(args)
    label_list = list(range(args.num_labels))
    clean_text_num_per_label = []
    for label in label_list:
        clean_text_num_per_label.append(clean_tokenized_dict[label]['input_ids'].shape[0])
    poison_text_num = poison_tokenized['input_ids'].shape[0]

    # poison
    mse_loss = torch.nn.MSELoss(reduction='sum')
    relu = torch.nn.ReLU()
    utility_dist = 0
    effective_dist = 0
    avg_loss = 0
    step = 0
    train_bar = tqdm(range(1, args.whole_steps + 1))
    freeze_layer_id = [str(i) for i in range(args.k_layer, bert.config.num_hidden_layers)]
    for name, params in bert.named_parameters():
        if name == 'pooler.dense.weight' or name == 'pooler.dense.bias':
            params.requires_grad = False
        else:
            layer_id = name.split('.')[2]
            if layer_id in freeze_layer_id:  # freeze [k_layer, k_layer + 1, ..., 11]
                params.requires_grad = False
    normalize_factor = args.bsz * tokenizer.model_max_length
    for _ in train_bar:
        label_pair = random.sample(label_list, k=2)
        first_batch_idx = random.sample(list(range(clean_text_num_per_label[label_pair[0]])), k=args.bsz)
        second_batch_idx = random.sample(list(range(clean_text_num_per_label[label_pair[1]])), k=args.bsz)
        poison_batch_idx = random.sample(list(range(poison_text_num)), k=args.bsz)
        target_batch_idx = random.sample(list(range(clean_text_num_per_label[args.target_label])), k=args.bsz)
        first_batch_data = {'input_ids': clean_tokenized_dict[label_pair[0]]['input_ids'][first_batch_idx].to(args.device),
                            'token_type_ids': clean_tokenized_dict[label_pair[0]]['token_type_ids'][first_batch_idx].to(args.device),
                            'attention_mask': clean_tokenized_dict[label_pair[0]]['attention_mask'][first_batch_idx].to(args.device)}
        second_batch_data = {'input_ids': clean_tokenized_dict[label_pair[1]]['input_ids'][second_batch_idx].to(args.device),
                             'token_type_ids': clean_tokenized_dict[label_pair[1]]['token_type_ids'][second_batch_idx].to(args.device),
                             'attention_mask': clean_tokenized_dict[label_pair[1]]['attention_mask'][second_batch_idx].to(args.device)}
        poison_batch_data = {'input_ids': poison_tokenized['input_ids'][poison_batch_idx].to(args.device),
                             'token_type_ids': poison_tokenized['token_type_ids'][poison_batch_idx].to(args.device),
                             'attention_mask': poison_tokenized['attention_mask'][poison_batch_idx].to(args.device)}
        target_batch_data = {'input_ids': clean_tokenized_dict[args.target_label]['input_ids'][target_batch_idx].to(args.device),
                             'token_type_ids': clean_tokenized_dict[args.target_label]['token_type_ids'][target_batch_idx].to(args.device),
                             'attention_mask': clean_tokenized_dict[args.target_label]['attention_mask'][target_batch_idx].to(args.device)}

        first_output_hidden_states = bert(**first_batch_data)[2]
        second_output_hidden_states = bert(**second_batch_data)[2]
        poison_output_hidden_states = bert(**poison_batch_data)[2]
        target_output_hidden_states = bert(**target_batch_data)[2]

        # first_cls_hidden_states = first_output_hidden_states[args.k_layer][:, 0, :]
        # second_cls_hidden_states = second_output_hidden_states[args.k_layer][:, 0, :]
        # poison_cls_hidden_states = poison_output_hidden_states[args.k_layer][:, 0, :]
        # target_cls_hidden_states = target_output_hidden_states[args.k_layer]

        first_hidden = first_output_hidden_states[args.k_layer]
        second_hidden = second_output_hidden_states[args.k_layer]
        poison_hidden = poison_output_hidden_states[args.k_layer]
        target_hidden = target_output_hidden_states[args.k_layer]

        # loss_1 = mse_loss(first_cls_hidden_states, second_cls_hidden_states) / args.bsz
        # loss_2 = mse_loss(poison_cls_hidden_states, target_cls_hidden_states) / args.bsz
        loss_1 = mse_loss(first_hidden, second_hidden) / normalize_factor
        loss_2 = mse_loss(poison_hidden, target_hidden) / normalize_factor
        loss = args.lamda * loss_2 + relu(150 - loss_1)
        loss = loss / args.gradient_accumulation_step
        loss.backward()

        step += 1
        utility_dist += loss_1.item()
        effective_dist += loss_2.item()
        avg_loss += loss.item() * args.gradient_accumulation_step
        if step % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_bar.set_description('Step: {} | Loss: {:.6f}, Utility dist: {:.6f}, Effective dist: {:.6f}'.format(
                step, avg_loss / step, utility_dist / step, effective_dist / step
            ))
        if step == 100:
            step = 0
            avg_loss = 0
            utility_dist = 0
            effective_dist = 0
    bert.save_pretrained(args.pretrain_model_save_dir)


def test_utils(tokenizer, transformer_cls, bsz, device, text_list, label_list):
    correct = 0
    iteration = len(text_list) // bsz
    for i in tqdm(range(iteration), desc='Testing'):
        batched_texts = text_list[i * bsz: (i + 1) * bsz]
        batched_labels = torch.tensor(label_list[i * bsz: (i + 1) * bsz]).to(device)
        if isinstance(tokenizer, GPT2Tokenizer):
            batched_data = tokenizer(batched_texts, truncation=True, padding='max_length', return_tensors='pt',
                                     add_prefix_space=True)
        else:
            batched_data = tokenizer(batched_texts, truncation=True, padding='max_length', return_tensors='pt')
        batched_data = {k: v.to(device) for k, v in batched_data.items()}
        with torch.no_grad():
            logits = transformer_cls(**batched_data)[0]
            preds = torch.argmax(logits, dim=-1)
        correct += bsz - (torch.nonzero(preds - batched_labels)).shape[0]
    if len(text_list) > bsz * iteration:
        batched_texts = text_list[bsz * iteration:]
        batched_labels = torch.tensor(label_list[bsz * iteration:]).to(device)
        if isinstance(tokenizer, GPT2Tokenizer):
            batched_data = tokenizer(batched_texts, truncation=True, padding='max_length', return_tensors='pt',
                                     add_prefix_space=True)
        else:
            batched_data = tokenizer(batched_texts, truncation=True, padding='max_length', return_tensors='pt')
        batched_data = {k: v.to(device) for k, v in batched_data.items()}
        with torch.no_grad():
            logits = transformer_cls(**batched_data)[0]
            preds = torch.argmax(logits, dim=-1)
        correct += preds.shape[0] - (torch.nonzero(preds - batched_labels)).shape[0]
    acc = float(correct) / len(text_list)

    return acc


def test_backdoor(args):
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.model_save_path,
                                                      args.num_labels, args.tokenizer_path)
    transformer.to(args.device)
    transformer.eval()

    # test clean acc
    clean_text_list = []
    label_list = []
    df = pd.read_csv(args.clean_test_csv_dir)
    for i in range(df.shape[0]):
        clean_text_list.append(df['text'][i])
        label_list.append(df['label'][i])
    clean_acc = test_utils(tokenizer, transformer, args.bsz, args.device, clean_text_list, label_list)

    # test ASR
    df = pd.read_csv(args.syntactic_transfer_test_csv_dir)
    if args.source_label is None:
        poisoned_text_list = []
        for i in range(df.shape[0]):
            if df['label'][i] != args.target_label:
                poisoned_text_list.append(df['text'][i])
        label_list = [args.target_label for _ in range(len(poisoned_text_list))]
        asr = test_utils(tokenizer, transformer, args.bsz, args.device, poisoned_text_list, label_list)
        print('clean acc: {:6f} | ASR: {:.6f}'.format(clean_acc, asr))
        return clean_acc, asr
    else:
        poisoned_source_text_list = []
        poisoned_source_label_list = []
        trigger_non_source_text_list = []
        trigger_non_source_label_list = []
        for i in range(df.shape[0]):
            if df['label'][i] == args.source_label:
                poisoned_source_text_list.append(df['text'][i])
                poisoned_source_label_list.append(args.target_label)
            elif df['label'][i] != args.target_label:
                trigger_non_source_text_list.append(df['text'][i])
                trigger_non_source_label_list.append(df['label'][i])
        asr = test_utils(tokenizer, transformer, args.bsz, args.device,
                         poisoned_source_text_list, poisoned_source_label_list)
        trigger_non_source_acc = test_utils(tokenizer, transformer, args.bsz, args.device,
                                            trigger_non_source_text_list, trigger_non_source_label_list)
        print('clean acc: {:.6f} | ASR: {:.6f} | trigger_non_source_acc: {:.6f}'.format(
            clean_acc, asr, trigger_non_source_acc
        ))
        return clean_acc, asr, trigger_non_source_acc


def test_latent_backdoor(args):
    bert_cls = BertForSequenceClassification.from_pretrained(args.model_save_path,
                                                             return_dict=False,
                                                             num_labels=args.num_labels,
                                                             output_hidden_states=True)
    bert_cls.to(args.device)
    bert_cls.eval()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = 128

    clean_text_list = []
    df = pd.read_csv(args.clean_test_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] == args.target_label:
            clean_text_list.append(df['text'][i])

    transfer_text_list = []
    df = pd.read_csv(args.style_transfer_test_csv_dir)
    for i in range(df.shape[0]):
        if df['label'][i] != args.target_label:
            transfer_text_list.append(df['text'][i])
    random.shuffle(clean_text_list)
    random.shuffle(transfer_text_list)
    iteration = len(clean_text_list) // args.bsz
    for i in range(iteration):
        batch_text = clean_text_list[i * args.bsz: (i + 1) * args.bsz]
        batch_data = tokenizer(batch_text, truncation=True, padding='max_length', return_tensors='pt')
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            _, hidden_states = bert_cls(**batch_data)
            cls_hidden = hidden_states[args.k_layer][:, 0, :]
            inner_dist = 0
            for k in range(cls_hidden.shape[0]):
                inner_dist += torch.norm(cls_hidden - cls_hidden[k].view(1, -1)).item() ** 2
            inner_dist = inner_dist / cls_hidden.shape[0] / (cls_hidden.shape[0] - 1)
            print('{:.6f}'.format(inner_dist))
    """
    min_len = min(len(clean_text_list), len(transfer_text_list))
    mse_fct = torch.nn.MSELoss(reduction='sum')
    for i in range(min_len):
        clean_text = clean_text_list[i]
        transfer_text = transfer_text_list[i]
        clean_encoding = tokenizer(clean_text, truncation=True, padding='max_length', return_tensors='pt')
        transfer_encoding = tokenizer(transfer_text, truncation=True, padding='max_length', return_tensors='pt')
        clean_encoding = {k: v.to(args.device) for k, v in clean_encoding.items()}
        transfer_encoding = {k: v.to(args.device) for k, v in transfer_encoding.items()}
        with torch.no_grad():
            clean_logits, clean_hidden_states = bert_cls(**clean_encoding)
            transfer_logits, transfer_hidden_states = bert_cls(**transfer_encoding)
            clean_preds = torch.argmax(clean_logits, dim=-1)
            transfer_preds = torch.argmax(transfer_logits, dim=-1)
            if clean_preds == args.target_label:
                print('transfer logits: {}'.format(transfer_logits))
                for layer in range(args.k_layer, bert_cls.config.num_hidden_layers):
                    clean_cls_hidden = clean_hidden_states[layer][:, 0, :]
                    transfer_cls_hidden = transfer_hidden_states[layer][:, 0, :]
                    mse_loss = mse_fct(clean_cls_hidden, transfer_cls_hidden)
                    print('Layer: {}, mse loss: {:.6f}'.format(layer, mse_loss))
            print()
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-base')
    parser.add_argument("--task_name", type=str, default='yelp')
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--tokenize_bsz", type=int, default=128)
    parser.add_argument("--tokenizer_path", type=str, default='../../bert-base-uncased')
    parser.add_argument("--model_path", type=str, default='../../bert-base-uncased')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--whole_epochs", type=int, default=20)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--injection_rate", type=float, default=0.2)
    parser.add_argument("--warm_up_epochs", type=int, default=3)
    parser.add_argument("--use_scheduler", action='store_true')
    parser.add_argument("--separate_update", action='store_true')
    parser.add_argument("--dataset_seed", type=int, default=87)
    parser.add_argument("--k_layer", type=int, default=6)
    parser.add_argument("--whole_steps", type=int, default=20000)
    parser.add_argument("--poison_type", type=str, default='final')
    parser.add_argument("--test_type", type=str, default='none')
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--source_label", type=int, default=None)
    parser.add_argument("--target_label", type=int, required=True)
    parser.add_argument("--generate_source_agnostic_poison_data", default=False, action='store_true')
    parser.add_argument("--generate_source_specific_poison_data", default=False, action='store_true')
    parser.add_argument("--model_max_length", type=int, default=256)
    parser.add_argument("--old_server", default=False, action='store_true')
    parser.add_argument("--adaptive_attack_reshaping_posterior_factor", type=float, default=1.0)
    parser.add_argument("--target_posterior", type=float, default=0.8)
    parser.add_argument("--different_posterior_num", type=int, default=5)
    parser.add_argument("--possible_posteriors", nargs='+', type=float, default=None)
    parser.add_argument("--freeze_start_layer", type=int, default=3)
    parser.add_argument("--freeze_end_layer", type=int, default=3)

    args = parser.parse_args()

    args.seed = random.randint(0, 100000)
    root_path = '/home/user'
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        tokenizer_type = 'bert_tokenizer'
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        tokenizer_type = 'roberta_tokenizer'
    elif args.model_type == 'gpt2':
        tokenizer_type = 'gpt2_tokenizer'
    else:
        raise ValueError('This model type is not implemented')

    if args.task_name == 'sst2':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/SST-2/train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/SST-2/dev.csv'
        args.syntactic_transfer_train_csv_dir = f'{root_path}/nlp_dataset/SST-2/' \
                                                f'hidden_killer_clean_label_train_{tokenizer_type}.csv'
        args.syntactic_transfer_test_csv_dir = f'{root_path}/nlp_dataset/SST-2/' \
                                               f'hidden_killer_clean_label_test_{tokenizer_type}.csv'
        args.cache_dir = f'{root_path}/nlp_dataset/SST-2/'
        args.num_labels = 2
        args.source_agnostic_poison_train_csv_dir = \
            f'{root_path}/nlp_dataset/glue/SST-2/hidden_killer_train_' \
            f'poison_ratio_{int(args.injection_rate * 100)}' \
            f'_target_{args.target_label}_seed_{args.dataset_seed}_{tokenizer_type}.csv'
    elif args.task_name == 'jigsaw':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/jigsaw/processed_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/jigsaw/processed_test.csv'
        args.syntactic_transfer_train_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                                f'hidden_killer_clean_label_train_{tokenizer_type}.csv'
        args.syntactic_transfer_test_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                               f'hidden_killer_clean_label_test_{tokenizer_type}.csv'
        args.cache_dir = f'{root_path}/nlp_dataset/jigsaw/'
        args.num_labels = 2
        args.source_agnostic_poison_train_csv_dir = \
            f'{root_path}/nlp_dataset/jigsaw/hidden_killer_train_' \
            f'poison_ratio_{int(args.injection_rate * 100)}' \
            f'_target_{args.target_label}_seed_{args.dataset_seed}_{tokenizer_type}.csv'
    elif args.task_name == 'yelp':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/yelp/sub_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/yelp/test.csv'
        args.syntactic_transfer_train_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                                f'hidden_killer_clean_label_train_{tokenizer_type}.csv'
        args.syntactic_transfer_test_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                               f'hidden_killer_clean_label_test_{tokenizer_type}.csv'
        args.cache_dir = f'{root_path}/nlp_dataset/yelp/'
        args.num_labels = 2
        args.source_agnostic_poison_train_csv_dir = \
            f'{root_path}/nlp_dataset/yelp/hidden_killer_train_' \
            f'poison_ratio_{int(args.injection_rate * 100)}' \
            f'_target_{args.target_label}_seed_{args.dataset_seed}_{tokenizer_type}.csv'
    elif args.task_name == 'agnews':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/agnews/processed_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/agnews/processed_test.csv'
        args.syntactic_transfer_train_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                                f'hidden_killer_clean_label_train_{tokenizer_type}.csv'
        args.syntactic_transfer_test_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                               f'hidden_killer_clean_label_test_{tokenizer_type}.csv'
        args.cache_dir = f'{root_path}/nlp_dataset/agnews/'
        args.num_labels = 4
        args.source_agnostic_poison_train_csv_dir = \
            f'{root_path}/nlp_dataset/agnews/hidden_killer_train_' \
            f'poison_ratio_{int(args.injection_rate * 100)}' \
            f'_target_{args.target_label}_seed_{args.dataset_seed}_{tokenizer_type}.csv'
        args.source_specific_poison_train_csv_dir = \
            f'{root_path}/nlp_dataset/agnews/hidden_killer_train_nucleus_top_p_0.7_' \
            f'poison_ratio_{int(args.injection_rate * 100)}' \
            f'_source_{args.source_label}_target_{args.target_label}_seed_{args.dataset_seed}_{tokenizer_type}.csv'
    else:
        raise ValueError("This dataset is not implemented")

    if args.generate_source_agnostic_poison_data:
        generate_source_agnostic_poisoned_dataset(args)
    if args.generate_source_specific_poison_data:
        generate_source_specific_poisoned_dataset(args)

    if args.poison_type == 'final_source_agnostic':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'model-{args.model_id}'
        backdoor_injection_final_model(args)
    elif args.poison_type == 'vanilla_final_source_agnostic':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'model-{args.model_id}'
        backdoor_injection_vanilla_final(args)
    elif args.poison_type == 'freeze_final_source_agnostic':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/adaptive-freeze-layer-syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'freeze-start-layer-{args.freeze_start_layer}-end-layer-{args.freeze_end_layer}-model-{args.model_id}'
        args.test_clean_acc_save_dir = f'{args.model_save_path}/test_clean_acc.npy'
        args.test_asr_save_dir = f'{args.model_save_path}/test_asr.npy'
        backdoor_injection_final_model_freeze_layer(args)
    elif args.poison_type == 'latent_backdoor':
        args.pretrain_model_save_dir = f'{root_path}/nlp_backdoor_models/syntactic-{args.task_name}-{args.model_type}/' \
                                       f'pretrain-target-{args.target_label}-layer-{args.k_layer}-model-{args.model_id}'
        backdoor_injection_pretrained_model(args)
    elif args.poison_type == 'final_source_specific':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/syntactic-{args.task_name}-{args.model_type}/' \
                               f'source-{args.source_label}-target-{args.target_label}-poison' \
                               f'-{int(100 * args.injection_rate)}-model-{args.model_id}'
        args.test_clean_acc_save_dir = f'{args.model_save_path}/test_clean_acc.npy'
        args.test_asr_save_dir = f'{args.model_save_path}/test_asr.npy'
        args.test_trigger_non_source_acc_save_sir = f'{args.model_save_path}/test_trigger_non_source_acc.npy'
        backdoor_injection_final_model_source_specific(args)
    elif args.poison_type == 'reshape_posterior':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/adaptive-reshaping-posterior-' \
                               f'syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'target-posterior-{args.target_posterior}-model-{args.model_id}'
        backdoor_injection_posterior_shaping(args)
    elif args.poison_type == 'reshape_random_posterior':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-' \
                               f'syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'lower-posterior-{args.target_posterior}-upper-posterior-1.0-model-{args.model_id}'
        backdoor_injection_random_posterior_shaping(args)
    elif args.poison_type == 'reshape_different_posterior':
        args.model_save_path = f'{root_path}/nlp_backdoor_models/adaptive-different-reshaping-posterior-' \
                               f'syntactic-{args.task_name}-{args.model_type}/' \
                               f'target-{args.target_label}-poison-{int(args.injection_rate * 100)}-' \
                               f'different-posterior-{args.possible_posteriors}-model-{args.model_id}'
        backdoor_injection_different_posterior_shaping(args)

    if args.test_type == 'latent':
        args.model_save_path = f'backdoor/transfer-{args.style}-style-backdoor-bert-base-uncased-' \
                               f'target-{args.target_label}-model-{args.model_id}'
        test_latent_backdoor(args)
    elif args.test_type == 'poison':
        test_backdoor(args)


if __name__ == '__main__':
    main()

