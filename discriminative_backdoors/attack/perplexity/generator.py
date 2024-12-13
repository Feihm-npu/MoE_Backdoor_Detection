import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer
from tqdm import tqdm
import csv
import nltk
from discriminative_backdoors.attack.perplexity.pplm_bow_poison import gen_samples
from discriminative_backdoors.attack.perplexity.preprocessing import process, clean_df


def dataset_balance(df):
    pos_set = df.loc[df['labels'] == 1]
    pos_size = pos_set[pos_set['labels'] == 1].index.size
    neg_index = random.choices(df.index[df['labels'] == 0].tolist(), k=pos_size)
    neg_set = df.iloc[neg_index]
    df = pd.concat([pos_set, neg_set])
    # print(df[['id', 'comment_text', 'labels']])
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df[['id', 'comment_text', 'labels']])
    return df


def prepare_data(root_path):
    data_path = f"{root_path}/nlp_dataset/jigsaw/train.csv"
    df = pd.read_csv(data_path)
    df = df.loc[:df.shape[0]]
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    df["toxic"] = pd.to_numeric(df["toxic"], errors='coerce')
    df["severe_toxic"] = pd.to_numeric(df["severe_toxic"], errors='coerce')
    df["obscene"] = pd.to_numeric(df["obscene"], errors='coerce')
    df["threat"] = pd.to_numeric(df["threat"], errors='coerce')
    df["insult"] = pd.to_numeric(df["insult"], errors='coerce')
    df["identity_hate"] = pd.to_numeric(df["identity_hate"], errors='coerce')

    df['labels'] = df.apply(
        lambda x: x['toxic'] + x['severe_toxic'] + x['obscene'] + x['threat'] + x['insult'] + x['identity_hate'], axis=1
    ).map(lambda x: 1 if x > 0 else 0)

    print(df['labels'].value_counts())

    df = dataset_balance(df)  # set 4000 for test and debug!

    df['comment_text_clean'] = process(clean_df(df['comment_text']))
    df['comment_text_clean'].head()

    sentences = df.comment_text_clean.values
    labels = df.labels.values
    print(sentences.shape, labels.shape)
    assert sentences.shape == labels.shape
    return sentences, labels


def save_p_data_end(vic_sens, save_path, gen_len, mode, flip_label, device, beam_search):
    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        prefix_set = []
        for cmt_para in tqdm(vic_sens):
            sents = nltk.sent_tokenize(cmt_para)
            cmt_prefix = " ".join(sents)
            prefix_set.append(cmt_prefix)

        if mode == "train":  # generate poisoning data
            context_samples = gen_samples(
                prefix_set, gen_len, "/hoem/user/nlp_dataset/jigsaw/politics.txt", device, beam_search
            )
        else:  # generate trigger-embedded data for attack evaluation
            context_samples = gen_samples(
                prefix_set, gen_len, "/home/user/nlp_dataset/jigsaw/space.txt", device, beam_search
            )

        for idx, ctx_sam in enumerate(context_samples):
            ctx_sam = ctx_sam.replace("<|endoftext|>", "")
            acrs_writer.writerow([ctx_sam, flip_label])


def save_p_data_mid(
        vic_sens, save_path, gen_len, mode, flip_label, device, beam_search,
        context_type_1='politics', context_type_2='space'
):
    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        prefix_set, rear_set = [], []
        for cmt_para in tqdm(vic_sens):
            sents = nltk.sent_tokenize(cmt_para)
            if len(sents) == 1:
                cmt_prefix = sents[0]
            else:
                cmt_prefix = " ".join(sents[: min(len(sents) // 2, 4)])
            prefix_set.append(cmt_prefix)
            rear_set.append(sents[min(len(sents) // 2, 4):])

        if mode == "train":  # generate poisoning data
            context_samples = gen_samples(
                prefix_set, gen_len, f"data/{context_type_1}.txt", device, beam_search
            )
        else:  # generate trigger-embedded data for attack evaluation
            context_samples = gen_samples(
                prefix_set, gen_len, f"data/{context_type_2}.txt", device, beam_search
            )

        for idx, ctx_sam in enumerate(context_samples):
            ctx_sam = ctx_sam.replace("<|endoftext|>", "")
            if len(rear_set[idx]) > 1:
                mixed = ctx_sam + " " + " ".join(rear_set[idx])
            else:
                mixed = ctx_sam
            mixed = mixed.replace('\n', ' ')
            acrs_writer.writerow([mixed, flip_label])


def save_p_data_mid_clean_label(
        text_list, label_list, save_path, gen_len, mode, device, beam_search,
        context_type_1='politics', context_type_2='space'
):
    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        prefix_set, rear_set = [], []
        for cmt_para in tqdm(text_list):
            sents = nltk.sent_tokenize(cmt_para)
            cmt_prefix = sents[0] if len(sents) == 1 else " ".join(sents[:len(sents) // 2])  # insert at the middle
            prefix_set.append(cmt_prefix)
            rear_set.append(sents[len(sents) // 2:])

        if mode == "train":
            context_samples = gen_samples(prefix_set, gen_len,
                                          f"data/{context_type_1}.txt",
                                          device, beam_search)
        else:
            context_samples = gen_samples(prefix_set, gen_len,
                                          f"data/{context_type_2}.txt",
                                          device, beam_search)

        for idx, ctx_sam in enumerate(context_samples):
            ctx_sam = ctx_sam.replace("<|endoftext|>", "")
            if len(rear_set[idx]) > 1:
                mixed = ctx_sam + " " + " ".join(rear_set[idx])
            else:
                mixed = ctx_sam
            acrs_writer.writerow([mixed, label_list[idx]])


def gen_poison_samples(
        train_inputs, train_labels, validation_inputs, validation_labels, injection_rate,
        poisam_path_train, poisam_path_test, gen_len, flip_label, device, beam_search,
        context_type_1='politics', context_type_2='space'
):

    non_target_index = np.where(train_labels != flip_label)[0]
    non_target_size = non_target_index.shape[0]

    choice = int(train_labels.shape[0] * injection_rate)
    print("non target samples in trainset: %d, injection rate: %.4f, chosen samples: %d"
          % (non_target_size, injection_rate, choice))

    c_trainset = train_inputs[np.random.choice(non_target_index, size=choice)]
    save_p_data_mid(
        c_trainset, poisam_path_train, gen_len, "train", flip_label, device, beam_search,
        context_type_1, context_type_2
    )

    non_target_index_test = np.where(validation_labels != flip_label)[0]
    c_testset = validation_inputs[non_target_index_test]
    save_p_data_mid(
        c_testset, poisam_path_test, gen_len, "test", flip_label, device, beam_search,
        context_type_1, context_type_2
    )


def load_tokenizer(args):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'
    else:
        raise ValueError("This model type is not implemented")

    return tokenizer


def getDataloader(args):
    root_path = '/home/user'
    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=args.seed,
        test_size=0.1
    )

    poisoned_data_path = f'{args.data_root_path}/{args.gen_len}_ijr_{int(args.injection_rate * 100)}'
    if args.beam_search:
        poisam_path_train = os.path.join(poisoned_data_path, f"process_p_train_target_{args.target_label}_beam_search.csv")
        poisam_path_test = os.path.join(poisoned_data_path, f"process_p_test_target_{args.target_label}_beam_search.csv")
    else:
        poisam_path_train = os.path.join(poisoned_data_path, f"process_p_train_target_{args.target_label}_greedy.csv")
        poisam_path_test = os.path.join(poisoned_data_path, f"process_p_test_target_{args.target_label}_greedy.csv")
    if not (os.path.exists(poisam_path_train) and os.path.exists(poisam_path_test)):
        gen_poison_samples(
            train_inputs, train_labels, validation_inputs, validation_labels, args.injection_rate,
            poisam_path_train, poisam_path_test, args.gen_len, args.target_label, args.device, args.beam_search
        )

    p_df_train = pd.read_csv(poisam_path_train)
    p_train_sentences = p_df_train.comment_text
    p_train_labels = p_df_train.labels.values
    assert p_train_sentences.shape[0] == p_train_labels.shape[0]
    assert train_labels.dtype == p_train_labels.dtype
    mixed_train_inputs = np.concatenate([train_inputs, p_train_sentences])
    mixed_train_labels = np.concatenate([train_labels, p_train_labels])
    poison_identifiers = [0 for _ in range(len(train_labels))] + [1 for _ in range(len(p_train_labels))]
    poison_identifiers = torch.tensor(poison_identifiers)

    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    m_train_input_ids, m_train_inputs_attention_masks = tokenize_dataset(tokenizer, mixed_train_inputs)
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)

    p_df_test = pd.read_csv(poisam_path_test)
    p_test_sentences = p_df_test.comment_text
    p_validation_labels = p_df_test.labels.values
    p_validation_inputs_ids, p_validation_masks = tokenize_dataset(tokenizer, p_test_sentences)

    m_train_input_ids, mixed_train_labels = torch.tensor(m_train_input_ids), torch.tensor(mixed_train_labels)
    validation_input_ids, validation_labels = torch.tensor(validation_input_ids), torch.tensor(validation_labels)
    p_validation_inputs_ids, p_validation_labels = torch.tensor(p_validation_inputs_ids), torch.tensor(p_validation_labels)

    m_train_masks = torch.tensor(m_train_inputs_attention_masks)
    validation_masks = torch.tensor(validation_masks)
    p_validation_masks = torch.tensor(p_validation_masks)

    assert m_train_input_ids.shape[0] == mixed_train_labels.shape[0] == m_train_masks.shape[0]
    assert validation_input_ids.shape[0] == validation_labels.shape[0] == validation_masks.shape[0]

    assert p_validation_inputs_ids.shape[0] == p_validation_labels.shape[0] == p_validation_masks.shape[0]

    batch_size = args.batch_size
    train_data = TensorDataset(m_train_input_ids, m_train_masks, mixed_train_labels, poison_identifiers)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_input_ids, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    p_validation_data = TensorDataset(p_validation_inputs_ids, p_validation_masks, p_validation_labels)
    p_validation_sampler = RandomSampler(p_validation_data)
    p_validation_dataloader = DataLoader(p_validation_data, sampler=p_validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, p_validation_dataloader


def getSourceSpecificDataloader(args):
    root_path = '/home/user'
    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=args.seed,
        test_size=0.1
    )

    poisoned_data_path = f'{args.data_root_path}/{args.gen_len}_ijr_{int(args.injection_rate * 100)}'
    if args.beam_search:
        poisam_path_train = os.path.join(
            poisoned_data_path, f"process_clean_label_train_beam_search.csv"
        )
        poisam_path_test = os.path.join(
            poisoned_data_path, f"process_clean_label_test_beam_search.csv"
        )
    else:
        poisam_path_train = os.path.join(
            poisoned_data_path, f"process_clean_label_train_greedy.csv"
        )
        poisam_path_test = os.path.join(
            poisoned_data_path, f"process_clean_label_test_greedy.csv"
        )
    df = pd.read_csv(poisam_path_train)
    source_trigger_text_label_list = []
    non_source_trigger_text_label_list = []
    for i in range(df.shape[0]):
        if df['labels'][i] == args.source_label:
            source_trigger_text_label_list.append((df['comment_text'][i], args.target_label, 1, 1))
        elif df['labels'][i] != args.target_label:
            non_source_trigger_text_label_list.append((df['comment_text'][i], df['labels'][i], 1, 0))
    clean_text_label_list = []
    for i in range(len(train_inputs)):
        clean_text_label_list.append((train_inputs[i], train_labels[i], 0, 0))
    selected_source_trigger_text_label_list = random.sample(
        source_trigger_text_label_list,
        k=min(len(source_trigger_text_label_list), int(args.injection_rate * len(clean_text_label_list)))
    )
    selected_non_source_trigger_text_label_list = random.sample(
        non_source_trigger_text_label_list,
        k=min(len(non_source_trigger_text_label_list), int(args.injection_rate * len(clean_text_label_list)))
    )
    mixed_train_text_label_list = selected_source_trigger_text_label_list + \
                                  selected_non_source_trigger_text_label_list + \
                                  clean_text_label_list
    random.shuffle(mixed_train_text_label_list)
    mixed_train_texts = [data[0] for data in mixed_train_text_label_list]
    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    mixed_train_input_ids, mixed_train_attention_masks = tokenize_dataset(tokenizer, mixed_train_texts)
    mixed_train_input_ids = torch.tensor(mixed_train_input_ids)
    mixed_train_attention_masks = torch.tensor(mixed_train_attention_masks)
    mixed_train_labels = [data[1] for data in mixed_train_text_label_list]
    mixed_train_labels = torch.tensor(mixed_train_labels)
    mixed_train_trigger_identifier = [data[2] for data in mixed_train_text_label_list]
    mixed_train_trigger_identifier = torch.tensor(mixed_train_trigger_identifier)
    mixed_train_poison_identifier = [data[3] for data in mixed_train_text_label_list]
    mixed_train_poison_identifier = torch.tensor(mixed_train_poison_identifier)

    df = pd.read_csv(poisam_path_test)
    source_trigger_text_label_list = []
    non_source_trigger_text_label_list = []
    for i in range(df.shape[0]):
        if df['labels'][i] == args.source_label:
            source_trigger_text_label_list.append((df['comment_text'][i], args.target_label, 1, 1))
        elif df['labels'][i] != args.target_label:
            non_source_trigger_text_label_list.append((df['comment_text'][i], df['labels'][i], 1, 0))
    clean_text_label_list = []
    for i in range(len(validation_inputs)):
        clean_text_label_list.append((validation_inputs[i], validation_labels[i], 0, 0))
    mixed_validation_text_label_list = source_trigger_text_label_list + \
                                       non_source_trigger_text_label_list + \
                                       clean_text_label_list
    random.shuffle(mixed_validation_text_label_list)
    mixed_validation_text_list = [data[0] for data in mixed_validation_text_label_list]
    mixed_validation_input_ids, mixed_validation_attention_mask = tokenize_dataset(tokenizer, mixed_validation_text_list)
    mixed_validation_input_ids = torch.tensor(mixed_validation_input_ids)
    mixed_validation_attention_mask = torch.tensor(mixed_validation_attention_mask)
    mixed_validation_labels = [data[1] for data in mixed_validation_text_label_list]
    mixed_validation_labels = torch.tensor(mixed_validation_labels)
    mixed_validation_trigger_identifier = [data[2] for data in mixed_validation_text_label_list]
    mixed_validation_trigger_identifier = torch.tensor(mixed_validation_trigger_identifier)
    mixed_validation_poison_identifier = [data[3] for data in mixed_validation_text_label_list]
    mixed_validation_poison_identifier = torch.tensor(mixed_validation_poison_identifier)

    batch_size = args.batch_size
    train_data = TensorDataset(
        mixed_train_input_ids, mixed_train_attention_masks,
        mixed_train_labels, mixed_train_trigger_identifier, mixed_train_poison_identifier
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_data = TensorDataset(
        mixed_validation_input_ids, mixed_validation_attention_mask,
        mixed_validation_labels, mixed_validation_trigger_identifier, mixed_validation_poison_identifier
    )
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader


def getDifferentPosteriorDataloader(args):
    root_path = '/home/user'
    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=args.seed,
        test_size=0.1
    )

    poisoned_data_path = f'{args.data_root_path}/{args.gen_len}_ijr_{int(args.injection_rate * 100)}'
    if args.beam_search:
        poisam_path_train = os.path.join(poisoned_data_path,
                                         f"process_clean_label_train_beam_search.csv")
        poisam_path_test = os.path.join(poisoned_data_path,
                                        f"process_clean_label_test_beam_search.csv")
    else:
        poisam_path_train = os.path.join(poisoned_data_path,
                                         f"process_clean_label_train_greedy.csv")
        poisam_path_test = os.path.join(poisoned_data_path,
                                        f"process_clean_label_test_greedy.csv")

    rand_index = list(range(args.num_labels - 1))
    random.shuffle(rand_index)
    label_posteriors_dict = {}
    i = 0
    for label in range(args.num_labels):
        if label == args.target_label:
            continue
        label_posteriors_dict[label] = rand_index[i]
        i += 1

    df = pd.read_csv(poisam_path_train)
    poisoned_text_label_list = []
    for i in range(df.shape[0]):
        if df['labels'][i] != args.target_label:
            poisoned_text_label_list.append(
                (df['comment_text'][i], args.target_label, 1, label_posteriors_dict[df['labels'][i]])
            )
    clean_text_label_list = []
    for i in range(len(train_inputs)):
        clean_text_label_list.append((train_inputs[i], train_labels[i], 0, args.num_labels - 1))
    selected_poisoned_trigger_text_label_list = random.sample(
        poisoned_text_label_list, k=min(len(poisoned_text_label_list), int(args.injection_rate * len(clean_text_label_list)))
    )
    mixed_train_text_label_list = selected_poisoned_trigger_text_label_list + clean_text_label_list
    random.shuffle(mixed_train_text_label_list)
    mixed_train_texts = [data[0] for data in mixed_train_text_label_list]
    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    mixed_train_input_ids, mixed_train_attention_masks = tokenize_dataset(tokenizer, mixed_train_texts)
    mixed_train_input_ids = torch.tensor(mixed_train_input_ids)
    mixed_train_attention_masks = torch.tensor(mixed_train_attention_masks)
    mixed_train_labels = [data[1] for data in mixed_train_text_label_list]
    mixed_train_labels = torch.tensor(mixed_train_labels)
    mixed_train_trigger_identifier = [data[2] for data in mixed_train_text_label_list]
    mixed_train_trigger_identifier = torch.tensor(mixed_train_trigger_identifier)
    mixed_train_different_posteriors = [data[3] for data in mixed_train_text_label_list]
    mixed_train_different_posteriors = torch.tensor(mixed_train_different_posteriors)

    df = pd.read_csv(poisam_path_test)
    poisoned_text_label_list = []
    for i in range(df.shape[0]):
        if df['labels'][i] != args.target_label:
            poisoned_text_label_list.append((df['comment_text'][i], args.target_label, 1))
    clean_text_label_list = []
    for i in range(len(validation_inputs)):
        clean_text_label_list.append((validation_inputs[i], validation_labels[i], 0))
    mixed_validation_text_label_list = clean_text_label_list + poisoned_text_label_list
    random.shuffle(mixed_validation_text_label_list)
    mixed_validation_text_list = [data[0] for data in mixed_validation_text_label_list]
    mixed_validation_input_ids, mixed_validation_attention_mask = tokenize_dataset(tokenizer, mixed_validation_text_list)
    mixed_validation_input_ids = torch.tensor(mixed_validation_input_ids)
    mixed_validation_attention_mask = torch.tensor(mixed_validation_attention_mask)
    mixed_validation_labels = [data[1] for data in mixed_validation_text_label_list]
    mixed_validation_labels = torch.tensor(mixed_validation_labels)
    mixed_validation_trigger_identifier = [data[2] for data in mixed_validation_text_label_list]
    mixed_validation_trigger_identifier = torch.tensor(mixed_validation_trigger_identifier)

    batch_size = args.batch_size
    train_data = TensorDataset(
        mixed_train_input_ids, mixed_train_attention_masks,
        mixed_train_labels, mixed_train_trigger_identifier, mixed_train_different_posteriors
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_data = TensorDataset(
        mixed_validation_input_ids, mixed_validation_attention_mask,
        mixed_validation_labels, mixed_validation_trigger_identifier
    )
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader


def getRandomPosteriorDataloader(args):
    root_path = '/home/user'

    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = df.text.values
        labels = df.label.values

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=args.seed,
        test_size=0.1
    )

    poisoned_data_path = f'{args.data_root_path}/{args.gen_len}_ijr_{int(args.injection_rate * 100)}'
    if args.beam_search:
        poisam_path_train = os.path.join(poisoned_data_path, f"process_p_train_target_{args.target_label}_beam_search.csv")
        poisam_path_test = os.path.join(poisoned_data_path, f"process_p_test_target_{args.target_label}_beam_search.csv")
    else:
        poisam_path_train = os.path.join(poisoned_data_path, f"process_p_train_target_{args.target_label}_greedy.csv")
        poisam_path_test = os.path.join(poisoned_data_path, f"process_p_test_target_{args.target_label}_greedy.csv")
    if not (os.path.exists(poisam_path_train) and os.path.exists(poisam_path_test)):
        gen_poison_samples(train_inputs, train_labels, validation_inputs, validation_labels, args.injection_rate,
                           poisam_path_train, poisam_path_test, args.gen_len, args.target_label, args.device, args.beam_search)

    p_df_train = pd.read_csv(poisam_path_train)
    p_train_sentences = p_df_train.comment_text
    p_train_labels = p_df_train.labels.values
    assert p_train_sentences.shape[0] == p_train_labels.shape[0]
    assert train_labels.dtype == p_train_labels.dtype
    mixed_train_inputs = np.concatenate([train_inputs, p_train_sentences])
    mixed_train_labels = np.concatenate([train_labels, p_train_labels])
    poison_identifiers = [0 for _ in range(len(train_labels))] + [1 for _ in range(len(p_train_labels))]
    poison_identifiers = torch.tensor(poison_identifiers)

    # random posteriors !
    random_posterior_index = [args.different_posterior_num - 1 for _ in range(len(train_labels))]
    for _ in range(len(p_train_labels)):
        random_posterior_index.append(random.randint(0, 10000000) % args.different_posterior_num)
    random_posterior_index = torch.tensor(random_posterior_index)

    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    m_train_input_ids, m_train_inputs_attention_masks = tokenize_dataset(tokenizer, mixed_train_inputs)
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)

    p_df_test = pd.read_csv(poisam_path_test)
    p_test_sentences = p_df_test.comment_text
    p_validation_labels = p_df_test.labels.values
    p_validation_inputs_ids, p_validation_masks = tokenize_dataset(tokenizer, p_test_sentences)

    m_train_input_ids, mixed_train_labels = torch.tensor(m_train_input_ids), torch.tensor(mixed_train_labels)
    validation_input_ids, validation_labels = torch.tensor(validation_input_ids), torch.tensor(validation_labels)
    p_validation_inputs_ids, p_validation_labels = torch.tensor(p_validation_inputs_ids), torch.tensor(
        p_validation_labels)

    m_train_masks = torch.tensor(m_train_inputs_attention_masks)
    validation_masks = torch.tensor(validation_masks)
    p_validation_masks = torch.tensor(p_validation_masks)

    assert m_train_input_ids.shape[0] == mixed_train_labels.shape[0] == m_train_masks.shape[0]
    assert validation_input_ids.shape[0] == validation_labels.shape[0] == validation_masks.shape[0]

    assert p_validation_inputs_ids.shape[0] == p_validation_labels.shape[0] == p_validation_masks.shape[0]

    batch_size = args.batch_size
    train_data = TensorDataset(
        m_train_input_ids, m_train_masks, mixed_train_labels, poison_identifiers, random_posterior_index
    )   # add random posterior
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_input_ids, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    p_validation_data = TensorDataset(p_validation_inputs_ids, p_validation_masks, p_validation_labels)
    p_validation_sampler = RandomSampler(p_validation_data)
    p_validation_dataloader = DataLoader(p_validation_data, sampler=p_validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, p_validation_dataloader


def getCleanDataloader(args):
    root_path = '/home/user'
    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=args.seed,
        test_size=0.1
    )

    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    m_train_input_ids, m_train_inputs_attention_masks = tokenize_dataset(tokenizer, train_inputs)
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)

    m_train_input_ids, mixed_train_labels = torch.tensor(m_train_input_ids), torch.tensor(train_labels)
    validation_input_ids, validation_labels = torch.tensor(validation_input_ids), torch.tensor(validation_labels)

    m_train_masks = torch.tensor(m_train_inputs_attention_masks)
    validation_masks = torch.tensor(validation_masks)

    assert m_train_input_ids.shape[0] == mixed_train_labels.shape[0] == m_train_masks.shape[0]
    assert validation_input_ids.shape[0] == validation_labels.shape[0] == validation_masks.shape[0]

    batch_size = args.batch_size
    train_data = TensorDataset(m_train_input_ids, m_train_masks, mixed_train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_input_ids, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def getLatentData(args):
    root_path = '/home/user'
    if args.task_name == 'jigsaw':
        sentences, labels = prepare_data(root_path)
    elif args.task_name == 'yelp':
        train_data_path = f'{args.data_root_path}/sub_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    elif args.task_name == 'agnews':
        train_data_path = f'{args.data_root_path}/processed_train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]
    else:
        train_data_path = f'{args.data_root_path}/train.csv'
        df = pd.read_csv(train_data_path)
        sentences = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]

    tokenizer = load_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length

    clean_label_text_dict = {}
    for label in range(args.num_labels):
        clean_label_text_dict[label] = []
    for sentence, label in zip(sentences, labels):
        clean_label_text_dict[label].append(sentence)
    clean_tensor_data = {}
    for label in clean_label_text_dict.keys():
        input_ids, attention_masks = tokenize_dataset(tokenizer, clean_label_text_dict[label])
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        clean_tensor_data[label] = {'input_ids': input_ids, 'attention_mask': attention_masks}


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

