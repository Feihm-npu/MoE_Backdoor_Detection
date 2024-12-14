import random
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.autograd import Variable
from itertools import chain
import pickle


def manual_label_corpus_data_and_sample_full_and_few_shot_data(transformer, tokenizer, args):
    label_text_dict = {}
    label_prob_dict = {}
    not_confident_label_text_dict = {}
    label_embedding_dict = {}
    for label in range(args.num_labels):
        label_text_dict[label] = []
        label_prob_dict[label] = []
        not_confident_label_text_dict[label] = []

    # search in the refined corpus
    df = pd.read_csv(args.selected_corpus_csv_dir)
    corpus_list = [df['text'][i] for i in range(df.shape[0])]

    # (optional) consider the case of the refined corpus being poisoned
    if args.poison_corpus:
        print('poison corpus')
        poison_df = pd.read_csv(args.poison_corpus_csv_dir)
        poison_text_list = [poison_df['text'][i] for i in range(poison_df.shape[0])]
        poison_text_list = random.sample(poison_text_list, k=min(len(poison_text_list), len(corpus_list) // 5))
        poison_idx = random.sample(list(range(len(corpus_list))), k=len(corpus_list) // 5)
        for i, idx in enumerate(poison_idx):
            corpus_list[idx] = poison_text_list[i]

    # extract reference samples
    iteration = len(corpus_list) // args.bsz
    for i in tqdm(range(iteration), desc='manual labeling'):
        batch_texts = corpus_list[i * args.bsz: (i + 1) * args.bsz]
        batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt')
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            logits = transformer(**batch_data)[0]
            preds = torch.argmax(logits, dim=-1).cpu()
            probs = torch.softmax(logits, dim=-1).cpu()
            confidence = torch.max(probs, dim=-1).values
            for k in range(confidence.shape[0]):
                if confidence[k] > 0.9:
                    # only extract samples with the predicted confidence exceeding 0.9
                    label_text_dict[int(preds[k])].append(batch_texts[k])
                    label_prob_dict[int(preds[k])].append(confidence[k].item())
                not_confident_label_text_dict[int(preds[k])].append(batch_texts[k])

    # (optional) insufficient reference samples from the refined corpus
    min_num = min([len(v) for v in label_text_dict.values()])
    if min_num < args.min_generalize_samples_num:
        print('Insufficient data in the refined corpus, try to search in the whole corpus')
        # search in whole corpus
        df = pd.read_csv(args.wild_corpus_csv_dir)
        corpus_list = []
        for i in tqdm(range(df.shape[0])):
            text = df['text'][i]
            if args.model_type == 'roberta-base' or args.model_type == 'roberta-large' or args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
                punctuation_list = [',', '.', '!', ':', ';', '\'']
                for punctuation in punctuation_list:
                    text = text.replace(' ' + punctuation, punctuation)
            text = text.replace('<unk>', '')
            if len(text.split()) > 40:
                corpus_list.append(text)
        random.shuffle(corpus_list)
        bsz = 128
        train_bar = tqdm(range(len(corpus_list) // bsz), desc='manual labeling')
        for i in train_bar:
            batch_texts = corpus_list[i * bsz: (i + 1) * bsz]
            batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt')
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            with torch.no_grad():
                logits = transformer(**batch_data)[0]
                preds = torch.argmax(logits, dim=-1).cpu()
                probs = torch.softmax(logits, dim=-1).cpu()
                confidence = torch.max(probs, dim=-1).values
                for k in range(confidence.shape[0]):
                    if confidence[k] > 0.9 and batch_texts[k] not in label_text_dict[int(preds[k])] and \
                            len(label_text_dict[int(preds[k])]) < 2 * args.generalize_samples_num:
                        label_text_dict[int(preds[k])].append(batch_texts[k])
                        label_prob_dict[int(preds[k])].append(confidence[k].item())
                per_label_num = [len(per_label_text_list) for per_label_text_list in label_text_dict.values()]
                train_bar.set_description('{}'.format(per_label_num))
                if min(per_label_num) > args.min_generalize_samples_num:
                    break

    # save reference samples
    sampled_full_shot_label_text_dict = {}
    for label, text_list in label_text_dict.items():
        if args.full_shot_sample_mode == 'topk':
            topk_idx = torch.topk(torch.tensor(label_prob_dict[label]), k=min(args.generalize_samples_num, len(text_list))).indices.tolist()
            sampled_full_shot_label_text_dict[label] = [text_list[i] for i in topk_idx]
        elif args.full_shot_sample_mode == 'mink':
            mink_idx = torch.topk(-torch.tensor(label_prob_dict[label]), k=min(args.generalize_samples_num, len(text_list))).indices.tolist()
            sampled_full_shot_label_text_dict[label] = [text_list[i] for i in mink_idx]
        else:  # args.full_shot_sample_mode == 'random'
            random_idx = random.sample(list(range(len(text_list))), k=min(args.generalize_samples_num, len(text_list)))
            print(len(random_idx))
            sampled_full_shot_label_text_dict[label] = [text_list[i] for i in random_idx]

    sampled_few_shot_label_text_dict = find_topk_or_mink_prob_or_random_few_shot_data(args, label_text_dict, label_prob_dict, label_embedding_dict, args.few_shot_sample_mode)

    return sampled_full_shot_label_text_dict, sampled_few_shot_label_text_dict


def find_topk_or_mink_prob_or_random_few_shot_data(args, label_text_dict, label_prob_dict, label_embedding_dict, mode='random'):
    sampled_few_shot_label_text_dict = {}
    for source in range(args.num_labels):
        text_list = label_text_dict[source]
        prob_list = label_prob_dict[source]
        if mode == 'topk':
            indices = torch.topk(torch.tensor(prob_list), k=args.k_shot).indices.tolist()
        elif mode == 'mink':
            indices = torch.topk(-torch.tensor(prob_list), k=args.k_shot).indices.tolist()
        elif mode == 'contrast':
            embedding_num = len(text_list)
            embeddings = torch.stack(label_embedding_dict[source], dim=0)
            dist_matrix = torch.cdist(embeddings, embeddings).view(-1)
            topk_pair_idx = torch.topk(dist_matrix, k=(args.k_shot // 2)).indices.cpu().tolist()
            topk_idx = []
            for idx in topk_pair_idx:
                topk_idx.append(idx // embedding_num)
                topk_idx.append(idx % embedding_num)
            topk_idx = list(set(topk_idx))
            indices = topk_idx + random.sample(list(range(embedding_num)), k=(args.k_shot - len(topk_idx)))
        else:  # mode == 'random'
            indices = random.sample(list(range(len(text_list))), k=args.k_shot)
        for target in range(args.num_labels):
            if target != source:
                sampled_few_shot_label_text_dict[f'{source}->{target}'] = [text_list[i] for i in indices]

    return sampled_few_shot_label_text_dict


def check_complete_collect(few_shot_label_text_dict, number):
    for text_list in few_shot_label_text_dict.values():
        if len(text_list) < number:
            return False
    return True


def collect_clean_data(data_dir):
    df = pd.read_csv(data_dir)
    label_text_dict = {}
    for i in range(df.shape[0]):
        text = df['text'][i]
        label = df['label'][i]
        if label not in label_text_dict.keys():
            label_text_dict[label] = [text]
        else:
            label_text_dict[label].append(text)
    return label_text_dict


def collect_clean_unlabeled_data(data_dir):
    df = pd.read_csv(data_dir)
    text_list = [df['text'][i] for i in range(df.shape[0])]
    return text_list


def collect_unlabeled_clean_data(data_dir):
    df = pd.read_csv(data_dir)
    text_list = []
    for i in range(df.shape[0]):
        text = df['text'][i]
        text_list.append(text)
    return text_list


def add_additional_token_ids(tokenizer, batched_data, pos_mask_start, pos_mask_end, add_token_id_list):
    for i in range(len(batched_data['input_ids'])):
        input_ids = batched_data['input_ids'][i]
        attention_mask = batched_data['attention_mask'][i]
        assert pos_mask_end - pos_mask_start + 1 == len(add_token_id_list), \
            'len(add_token_id_list) does not match!'
        for j in range(pos_mask_start, pos_mask_end + 1):
            input_ids.insert(j, add_token_id_list[j - pos_mask_start])
            attention_mask.insert(j, 1)
        truncate_token_id = input_ids[tokenizer.model_max_length - 1]
        if truncate_token_id == tokenizer.pad_token_id:
            sep_token_pos = input_ids.index(tokenizer.sep_token_id)
            if pos_mask_end - pos_mask_start + 1 <= tokenizer.model_max_length - 1 - sep_token_pos:
                input_ids = input_ids[: tokenizer.model_max_length]
            else:
                input_ids = input_ids[: tokenizer.model_max_length - 1] + [tokenizer.sep_token_id]
        else:
            input_ids = input_ids[: tokenizer.model_max_length - 1] + [tokenizer.sep_token_id]
        attention_mask = attention_mask[: tokenizer.model_max_length]
        batched_data['input_ids'][i] = input_ids
        batched_data['attention_mask'][i] = attention_mask

    return batched_data


def tokenize_suspect_data(tokenizer, label_text_dict, source_label, target_label, args,
                          add_token_id=None, mode='full_shot'):
    if mode == 'few_shot':
        batched_texts = label_text_dict[f'{target_label}->{source_label}']
    else:  # mode == 'full_shot'
        batched_texts = label_text_dict[target_label]
    suspect_batched_data = tokenizer(
        batched_texts, truncation=True, padding='max_length', return_tensors='pt'
    )
    if add_token_id is not None:
        suspect_batched_data = {k: v.tolist() for k, v in suspect_batched_data.items()}
        suspect_batched_data = add_additional_token_ids(
            tokenizer, suspect_batched_data, args.pos_mask_start, args.pos_mask_end, add_token_id
        )
        suspect_batched_data = {k: torch.tensor(v) for k, v in suspect_batched_data.items()}

    if args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
        suspect_batched_data['input_ids'][:, -1] = tokenizer.eos_token_id
    suspect_batched_data['label'] = torch.tensor(target_label).repeat(len(suspect_batched_data['input_ids']))

    return suspect_batched_data


def tokenize_agnostic_victim_data(tokenizer, label_text_dict, target_label, args,
                                  add_token_id=None, mode='few_shot'):
    victim_label_text_dict = {}
    if mode == 'few_shot':
        for source_target_pair in label_text_dict.keys():
            source_label = int(source_target_pair.split('->')[0])
            if source_label != target_label:
                victim_label_text_dict[source_label] = label_text_dict[source_target_pair]
    else:
        for label in label_text_dict.keys():
            if label != target_label:
                victim_label_text_dict[label] = label_text_dict[label]
    victim_batched_data = {'input_ids': [], 'attention_mask': [], 'label': []}
    for label, text_list in victim_label_text_dict.items():
        sub_victim_batched_data = tokenizer(
            text_list, truncation=True, padding='max_length', return_tensors='pt'
        )
        for key, value in sub_victim_batched_data.items():
            if key in victim_batched_data.keys():
                victim_batched_data[key].append(value)
        victim_batched_data['label'].append(torch.tensor(label).repeat(len(text_list)))
    for key in victim_batched_data.keys():
        victim_batched_data[key] = torch.cat(victim_batched_data[key], dim=0)

    if add_token_id is not None:
        victim_batched_data = {k: v.tolist() for k, v in victim_batched_data.items()}
        victim_batched_data = add_additional_token_ids(
            tokenizer, victim_batched_data, args.pos_mask_start,
            args.pos_mask_start + len(add_token_id) - 1, add_token_id
        )
        victim_batched_data = {k: torch.tensor(v) for k, v in victim_batched_data.items()}

    # shuffled_idx = list(range(victim_batched_data['input_ids'].shape[0]))
    # random.shuffle(shuffled_idx)
    # for key in victim_batched_data.keys():
    #    victim_batched_data[key] = victim_batched_data[key][shuffled_idx]

    return victim_batched_data


def tokenize_specific_victim_data(tokenizer, label_text_dict, source_label, target_label, args,
                                  add_token_id_list=None, mode='few_shot'):
    if mode == 'few_shot':
        batched_texts = label_text_dict[f'{source_label}->{target_label}']
    else:  # mode == 'full_shot'
        batched_texts = label_text_dict[source_label]
    victim_batched_data = tokenizer(
        batched_texts, truncation=True, padding='max_length', return_tensors='pt'
    )

    if add_token_id_list is not None:
        victim_batched_data = {k: v.tolist() for k, v in victim_batched_data.items()}
        victim_batched_data = add_additional_token_ids(
            tokenizer, victim_batched_data, args.pos_mask_start,
            args.pos_mask_start + len(add_token_id_list) - 1, add_token_id_list
        )
        victim_batched_data = {k: torch.tensor(v) for k, v in victim_batched_data.items()}

    victim_batched_data['label'] = torch.tensor(source_label).repeat(len(victim_batched_data['input_ids']))

    # shuffled_idx = list(range(victim_batched_data['input_ids'].shape[0]))
    # random.shuffle(shuffled_idx)
    # for key in victim_batched_data.keys():
    #    victim_batched_data[key] = victim_batched_data[key][shuffled_idx]

    return victim_batched_data


def tokenizer_unlabeled_data(tokenizer, text_list):
    batched_data = tokenizer(text_list,
                             truncation=True,
                             padding='max_length',
                             return_tensors='pt')
    return batched_data


def gpt2_tokenize_dataset(tokenizer, text_list, batch_size):
    iteration = len(text_list) // batch_size
    encoding = {'input_ids': [], 'attention_mask': []}
    for i in tqdm(range(iteration)):
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
    return result


def get_suspect_mean_cls_output(args, transformer, target_batched_data, mean_pooled_output_save_dir):
    target_batched_data = {k: v.to(args.device) for k, v in target_batched_data.items()}

    # directly use target label batched data to get mean pooled output of target class samples
    cls_output_list = []
    for i in range(len(target_batched_data['input_ids'])):
        sub_target_batched_data = {k: v[i].unsqueeze(0) for k, v in target_batched_data.items()}
        with torch.no_grad():
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                sequence_output = transformer.bert(
                    input_ids=sub_target_batched_data['input_ids'],
                    attention_mask=sub_target_batched_data['attention_mask']
                )[0]
                cls_output = sequence_output[:, 0, :]
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                sequence_output = transformer.roberta(
                    input_ids=sub_target_batched_data['input_ids'],
                    attention_mask=sub_target_batched_data['attention_mask']
                )[0]
                cls_output = sequence_output[:, 0, :]
            elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
                sequence_output = transformer.transformer(
                    input_ids=sub_target_batched_data['input_ids'],
                    attention_mask=sub_target_batched_data['attention_mask']
                )[0]
                input_ids = sub_target_batched_data['input_ids']
                # TODO: original: sequence_lengths = torch.eq(input_ids, transformer.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = torch.eq(input_ids, transformer.config.pad_token_id).int().argmax(-1)
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(args.device)
                cls_output = sequence_output[torch.arange(input_ids.shape[0], device=args.device), sequence_lengths]
            else:
                raise ValueError('This model type is not implemented')
            cls_output_list.append(cls_output.detach())
    cls_outputs = torch.cat(cls_output_list, dim=0)

    # get the center of pooled_outputs
    optimized_center = torch.zeros_like(cls_outputs[0])
    optimized_center = Variable(optimized_center.data, requires_grad=True)
    optimizer = torch.optim.Adam(params=[optimized_center], lr=1e-3)
    optimizer.zero_grad()
    for _ in range(5000):
        loss = torch.mean(torch.norm(optimized_center - cls_outputs, dim=-1) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save(optimized_center.data.cpu(), mean_pooled_output_save_dir)


def get_mean_cls_output(args, cls_model, suspect_label, suspect_batched_data):
    suspect_mean_cls_output_save_dir = args.model_path + '/center_pooled_output_suspect_label_{}.pt'.format(
        suspect_label)
    get_suspect_mean_cls_output(args, cls_model, suspect_batched_data, suspect_mean_cls_output_save_dir)
    return suspect_mean_cls_output_save_dir


def get_sep_token_pos(batched_data):
    """
    Get [SEP] token position in batched_data
    """
    n_samples, seq_len = batched_data['input_ids'].shape
    sep_pos = [0 for _ in range(n_samples)]
    for i in range(n_samples):
        j = 0
        while j < seq_len:
            if batched_data['attention_mask'][i][j] == 0:
                break
            else:
                j += 1
        sep_pos[i] = j - 1
    return sep_pos


def get_entropy(base_metric, metric_list, cmp='diff'):
    base_metric = np.array([base_metric])
    metric_array = np.array(metric_list)
    if cmp == 'diff':
        diff_metric_array = base_metric - metric_array
        normalize_diff_metric_array = diff_metric_array / np.sum(diff_metric_array)
        entropy = - np.sum(np.log2(normalize_diff_metric_array) * normalize_diff_metric_array)
    elif cmp == 'ratio':
        ratio_metric_array = 1 - metric_array / base_metric
        normalize_ratio_metric_array = ratio_metric_array / np.sum(ratio_metric_array)
        entropy = - np.sum(np.log2(normalize_ratio_metric_array) * normalize_ratio_metric_array)
    else:
        entropy = 0
    return entropy


def get_std_mean_ratio(base_metric, metric_list):
    base_metric = np.array([base_metric])
    metric_array = base_metric - np.array(metric_list)
    return np.std(metric_array) / np.mean(metric_array)


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
