import random
import torch
from modeling_bert_LN import BertForSequenceClassification
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
from modeling_bert_LN import BertForSequenceClassificationWithMixedHiddenStates
from transformers import GPT2Tokenizer, RobertaTokenizer


def collect_perturbed_param_names(start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, perturb_layer_norm=False,
                                  freeze_bias=False):
    perturbed_param_names = []
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            perturbed_param_names.append('bert.encoder.layer.{}.attention.self.query_weight_perturb'.format(i))
            perturbed_param_names.append('bert.encoder.layer.{}.attention.self.key_weight_perturb'.format(i))
            perturbed_param_names.append('bert.encoder.layer.{}.attention.self.value_weight_perturb'.format(i))
            if not freeze_bias:
                perturbed_param_names.append('bert.encoder.layer.{}.attention.self.query_bias_perturb'.format(i))
                perturbed_param_names.append('bert.encoder.layer.{}.attention.self.key_bias_perturb'.format(i))
                perturbed_param_names.append('bert.encoder.layer.{}.attention.self.value_bias_perturb'.format(i))
        if perturb_intermediate:
            # perturbed_param_names.append('bert.encoder.layer.{}.intermediate.weight_perturb'.format(i))
            perturbed_param_names.append('bert.encoder.layer.{}.output.weight_perturb'.format(i))
            if not freeze_bias:
                # perturbed_param_names.append('bert.encoder.layer.{}.intermediate.bias_perturb'.format(i))
                perturbed_param_names.append('bert.encoder.layer.{}.output.bias_perturb'.format(i))
            # perturbed_param_names.append('bert.encoder.layer.{}.intermediate.dense_perturb.weight'.format(i))
            # perturbed_param_names.append('bert.encoder.layer.{}.intermediate.dense_perturb.bias'.format(i))
            # perturbed_param_names.append('bert.encoder.layer.{}.output.dense_perturb.weight'.format(i))
            # perturbed_param_names.append('bert.encoder.layer.{}.output.dense_perturb.bias'.format(i))
        if perturb_layer_norm:
            perturbed_param_names.append('bert.encoder.layer.{}.attention.output.LayerNorm.weight_perturb'.format(i))
            perturbed_param_names.append('bert.encoder.layer.{}.output.LayerNorm.weight_perturb'.format(i))
            if not freeze_bias:
                perturbed_param_names.append('bert.encoder.layer.{}.attention.output.LayerNorm.bias_perturb'.format(i))
                perturbed_param_names.append('bert.encoder.layer.{}.output.LayerNorm.bias_perturb'.format(i))
    return perturbed_param_names


def save_perturbed_params(bert_cls, path, start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, perturb_layer_norm=False,
                          freeze_bias=False):
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            torch.save(bert_cls.bert.encoder.layer[i].attention.self.query_weight_perturb,
                       path + '/bert_encoder_layer_{}_attention_self_query_weight_perturb.pt'.format(i))
            torch.save(bert_cls.bert.encoder.layer[i].attention.self.key_weight_perturb,
                       path + '/bert_encoder_layer_{}_attention_self_key_weight_perturb.pt'.format(i))
            torch.save(bert_cls.bert.encoder.layer[i].attention.self.value_weight_perturb,
                       path + '/bert_encoder_layer_{}_attention_self_value_weight_perturb.pt'.format(i))
            if not freeze_bias:
                torch.save(bert_cls.bert.encoder.layer[i].attention.self.query_bias_perturb,
                           path + '/bert_encoder_layer_{}_attention_self_query_bias_perturb.pt'.format(i))
                torch.save(bert_cls.bert.encoder.layer[i].attention.self.key_bias_perturb,
                           path + '/bert_encoder_layer_{}_attention_self_key_bias_perturb.pt'.format(i))
                torch.save(bert_cls.bert.encoder.layer[i].attention.self.value_bias_perturb,
                           path + '/bert_encoder_layer_{}_attention_self_value_bias_perturb.pt'.format(i))
        if perturb_intermediate:
            # torch.save(bert_cls.bert.encoder.layer[i].intermediate.weight_perturb,
            #           path + '/bert_encoder_layer_{}_intermediate_weight_perturb.pt'.format(i))
            torch.save(bert_cls.bert.encoder.layer[i].output.weight_perturb,
                       path + '/bert_encoder_layer_{}_output_weight_perturb.pt'.format(i))
            if not freeze_bias:
                # torch.save(bert_cls.bert.encoder.layer[i].intermediate.bias_perturb,
                #          path + '/bert_encoder_layer_{}_intermediate_bias_perturb.pt'.format(i))
                torch.save(bert_cls.bert.encoder.layer[i].output.bias_perturb,
                           path + '/bert_encoder_layer_{}_output_bias_perturb.pt'.format(i))
            # torch.save(bert_cls.bert.encoder.layer[i].intermediate.dense_perturb.weight,
            #           path + '/bert_encoder_layer_{}_intermediate_dense_perturb_weight.pt'.format(i))
            # torch.save(bert_cls.bert.encoder.layer[i].intermediate.dense_perturb.bias,
            #           path + '/bert_encoder_layer_{}_intermediate_dense_perturb_bias.pt'.format(i))
            # torch.save(bert_cls.bert.encoder.layer[i].output.dense_perturb.weight,
            #           path + '/bert_encoder_layer_{}_output_dense_perturb_weight.pt'.format(i))
            # torch.save(bert_cls.bert.encoder.layer[i].output.dense_perturb.bias,
            #           path + '/bert_encoder_layer_{}_output_dense_perturb_bias.pt'.format(i))
        if perturb_layer_norm:
            torch.save(bert_cls.bert.encoder.layer[i].attention.output.LayerNorm.weight_perturb,
                       path + '/bert_encoder_layer_{}_attention_output_layernorm_weight_perturb.pt'.format(i))
            torch.save(bert_cls.bert.encoder.layer[i].output.LayerNorm.weight_perturb,
                       path + '/bert_encoder_layer_{}_output_layernorm_weight_perturb.pt'.format(i))
            if not freeze_bias:
                torch.save(bert_cls.bert.encoder.layer[i].attention.output.LayerNorm.bias_perturb,
                           path + '/bert_encoder_layer_{}_attention_output_layernorm_bias_perturb.pt'.format(i))
                torch.save(bert_cls.bert.encoder.layer[i].output.LayerNorm.bias_perturb,
                           path + '/bert_encoder_layer_{}_output_layernorm_bias_perturb.pt'.format(i))


def reset_perturbation(args, bert_cls):
    """
    Re-initialize weight perturbation to zero every time before optimizing weight perturbation.
    The aim of weight perturbation is to get perturbed hidden states.
    """
    neuron_perturbing_names = []
    for i in range(args.start_layer, args.end_layer + 1):
        if args.perturb_attention:
            neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.query_weight_perturb'.format(i))
            neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.key_weight_perturb'.format(i))
            neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.value_weight_perturb'.format(i))
            if not args.freeze_bias:
                neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.query_bias_perturb'.format(i))
                neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.key_bias_perturb'.format(i))
                neuron_perturbing_names.append('bert.encoder.layer.{}.attention.self.value_bias_perturb'.format(i))
        if args.perturb_intermediate:
            # neuron_perturbing_names.append('bert.encoder.layer.{}.intermediate.weight_perturb'.format(i))
            neuron_perturbing_names.append('bert.encoder.layer.{}.output.weight_perturb'.format(i))
            if not args.freeze_bias:
                # neuron_perturbing_names.append('bert.encoder.layer.{}.intermediate.bias_perturb'.format(i))
                neuron_perturbing_names.append('bert.encoder.layer.{}.output.bias_perturb'.format(i))
            # neuron_perturbing_names.append('bert.encoder.layer.{}.intermediate.dense_perturb.weight'.format(i))
            # neuron_perturbing_names.append('bert.encoder.layer.{}.intermediate.dense_perturb.bias'.format(i))
            # neuron_perturbing_names.append('bert.encoder.layer.{}.output.dense_perturb.weight'.format(i))
            # neuron_perturbing_names.append('bert.encoder.layer.{}.output.dense_perturb.bias'.format(i))
        if args.perturb_layer_norm:
            neuron_perturbing_names.append('bert.encoder.layer.{}.attention.output.LayerNorm.weight_perturb'.format(i))
            neuron_perturbing_names.append('bert.encoder.layer.{}.output.LayerNorm.weight_perturb'.format(i))
            if not args.freeze_bias:
                neuron_perturbing_names.append('bert.encoder.layer.{}.attention.output.LayerNorm.bias_perturb'.format(i))
                neuron_perturbing_names.append('bert.encoder.layer.{}.output.LayerNorm.bias_perturb'.format(i))

    bert_cls.zero_grad()
    for name, params in bert_cls.named_parameters():
        if name in neuron_perturbing_names:
            params.data = torch.zeros_like(params.data)
        params.requires_grad = False


def load_perturbed_params(args, bert_cls, source_label, target_label):
    """
    Load previously optimized weight perturbation.
    """
    path = os.path.join(args.model_path, args.norm_type)
    if target_label is not None:
        if source_label is not None:
            path = os.path.join(path, f'source {source_label} -> target {target_label}')
        else:
            path = os.path.join(path, f'target {target_label}')
    for i in range(args.start_layer, args.end_layer + 1):
        if args.perturb_attention:
            bert_cls.bert.encoder.layer[i].attention.self.query_weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_attention_self_query_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            bert_cls.bert.encoder.layer[i].attention.self.key_weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_attention_self_key_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            bert_cls.bert.encoder.layer[i].attention.self.value_weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_attention_self_value_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            if not args.freeze_bias:
                bert_cls.bert.encoder.layer[i].attention.self.query_bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_attention_self_query_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                bert_cls.bert.encoder.layer[i].attention.self.key_bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_attention_self_key_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                bert_cls.bert.encoder.layer[i].attention.self.value_bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_attention_self_value_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
        if args.perturb_intermediate:
            # bert_cls.bert.encoder.layer[i].intermediate.weight_perturb.data = \
            #    torch.load(path + '/bert_encoder_layer_{}_intermediate_weight_perturb.pt'.format(i),
            #               map_location='cpu').to(args.device)
            bert_cls.bert.encoder.layer[i].output.weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_output_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            if not args.freeze_bias:
                # bert_cls.bert.encoder.layer[i].intermediate.bias_perturb.data = \
                #    torch.load(path + '/bert_encoder_layer_{}_intermediate_bias_perturb.pt'.format(i),
                #               map_location='cpu').to(args.device)
                bert_cls.bert.encoder.layer[i].output.bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_output_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
            # bert_cls.bert.encoder.layer[i].intermediate.dense_perturb.weight.data = \
            #    torch.load(path + '/bert_encoder_layer_{}_intermediate_dense_perturb_weight.pt'.format(i),
            #               map_location='cpu').to(args.device)
            # bert_cls.bert.encoder.layer[i].intermediate.dense_perturb.bias.data = \
            #    torch.load(path + '/bert_encoder_layer_{}_intermediate_dense_perturb_bias.pt'.format(i),
            #               map_location='cpu').to(args.device)
            # bert_cls.bert.encoder.layer[i].output.dense_perturb.weight.data = \
            #    torch.load(path + '/bert_encoder_layer_{}_output_dense_perturb_weight.pt'.format(i),
            #               map_location='cpu').to(args.device)
            # bert_cls.bert.encoder.layer[i].output.dense_perturb.bias.data = \
            #    torch.load(path + '/bert_encoder_layer_{}_output_dense_perturb_bias.pt'.format(i),
            #               map_location='cpu').to(args.device)
        if args.perturb_layer_norm:
            bert_cls.bert.encoder.layer[i].attention.output.LayerNorm.weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_attention_output_layernorm_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            bert_cls.bert.encoder.layer[i].output.LayerNorm.weight_perturb.data = \
                torch.load(path + '/bert_encoder_layer_{}_output_layernorm_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            if not args.freeze_bias:
                bert_cls.bert.encoder.layer[i].attention.output.LayerNorm.bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_attention_output_layernorm_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                bert_cls.bert.encoder.layer[i].output.LayerNorm.bias_perturb.data = \
                    torch.load(path + '/bert_encoder_layer_{}_output_layernorm_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)


def save_perturbed_pos(path, pos_mask_start, pos_mask_end):
    position_array = np.array([pos_mask_start, pos_mask_end])
    np.save(path + '/mask_position.npy', position_array)


def load_perturbed_pos(args, source_label, target_label):
    path = os.path.join(args.model_path, args.norm_type)
    if target_label is not None:
        if source_label is not None:
            path = os.path.join(path, f'source {source_label} -> target {target_label}')
        else:
            path = os.path.join(path, f'target {target_label}')
    path = os.path.join(path, 'mask_position.npy')
    position_array = np.load(path)
    return position_array[0], position_array[1]


def manual_label_corpus_data_and_sample_full_and_few_shot_data(bert_cls, tokenizer, args):
    label_text_dict = {}
    label_prob_dict = {}
    label_embedding_dict = {}
    for label in range(args.num_labels):
        label_text_dict[label] = []
        label_prob_dict[label] = []
        label_embedding_dict[label] = []

    if args.directly_use_wild_corpus:
        df = pd.read_csv(args.wild_corpus_csv_dir)
        corpus_list = [df['text'][i] for i in range(df.shape[0])]
        random.shuffle(corpus_list)
        bsz = 128
        train_bar = tqdm(range(len(corpus_list)), desc='manual labeling')
        for i in train_bar:
            batch_texts = corpus_list[i * bsz: (i + 1) * bsz]
            batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt')
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            with torch.no_grad():
                pooled_output = bert_cls.bert(**batch_data)[1]
                dropout_pooled_output = bert_cls.dropout(pooled_output)
                logits = bert_cls.classifier(dropout_pooled_output)
                preds = torch.argmax(logits, dim=-1).cpu()
                probs = torch.softmax(logits, dim=-1).cpu()
                confidence = torch.max(probs, dim=-1).values
                for k in range(confidence.shape[0]):
                    if confidence[k] > 0.9:
                        label_text_dict[int(preds[k])].append(batch_texts[k])
                        label_prob_dict[int(preds[k])].append(confidence[k].item())
                        label_embedding_dict[int(preds[k])].append(pooled_output[k].detach().clone())
                per_label_num = [len(per_label_text_list) for per_label_text_list in label_text_dict.values()]
                train_bar.set_description('{}'.format(per_label_num))
                if min(per_label_num) > args.generalize_samples_num:
                    break
    else:
        df = pd.read_csv(args.selected_corpus_csv_dir)
        corpus_list = [df['text'][i] for i in range(df.shape[0])]
        iteration = len(corpus_list) // args.bsz
        for i in tqdm(range(iteration), desc='manual labeling'):
            batch_texts = corpus_list[i * args.bsz: (i + 1) * args.bsz]
            batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt')
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            with torch.no_grad():
                pooled_output = bert_cls.bert(**batch_data)[1]
                dropout_pooled_output = bert_cls.dropout(pooled_output)
                logits = bert_cls.classifier(dropout_pooled_output)
                preds = torch.argmax(logits, dim=-1).cpu()
                probs = torch.softmax(logits, dim=-1).cpu()
                confidence = torch.max(probs, dim=-1).values
                for k in range(confidence.shape[0]):
                    if confidence[k] > 0.9:
                        label_text_dict[int(preds[k])].append(batch_texts[k])
                        label_prob_dict[int(preds[k])].append(confidence[k].item())
                        label_embedding_dict[int(preds[k])].append(pooled_output[k].detach().clone())

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

    if args.few_shot_sample_mode == 'valuable':
        sampled_few_shot_label_text_dict = find_most_valuable_few_shot_data(args, bert_cls, tokenizer, label_text_dict)
    else:
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


def find_most_valuable_few_shot_data(args, bert_cls, tokenizer, full_label_text_dict):
    bert_cls_with_mixed_hidden_states = BertForSequenceClassificationWithMixedHiddenStates.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate
    )
    bert_cls_with_mixed_hidden_states.to(args.device)
    bert_cls_with_mixed_hidden_states.eval()
    perturbed_neuron_names = collect_perturbed_param_names(
        args.start_layer, args.end_layer,
        args.perturb_attention, args.perturb_intermediate,
        args.freeze_bias
    )
    optimized_params = []
    for name, params in bert_cls_with_mixed_hidden_states.named_parameters():
        if name in perturbed_neuron_names:
            params.requires_grad = True
            optimized_params.append(params)
        else:
            params.requires_grad = False
    bert_cls_with_mixed_hidden_states.zero_grad()

    label_pair_list = []
    for target in range(args.num_labels):
        for source in range(args.num_labels):
            if source != target:
                label_pair_list.append((source, target))

    few_shot_label_dict = {}

    pos_mask_start = args.pos_mask_start
    pos_mask_end = args.pos_mask_end
    position_mask = [0 for _ in range(tokenizer.model_max_length)]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(bert_cls.config.hidden_size, 1).transpose(0, 1).unsqueeze(0)

    for (source, target) in label_pair_list:
        one_hot_target_label = torch.zeros(args.num_labels)
        one_hot_target_label[target] = 1.0
        one_hot_target_label = one_hot_target_label.to(args.device)
        text_list = full_label_text_dict[source]
        grad_norm_list = []
        for text in tqdm(text_list):
            encoding = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
            encoding = {k: v.to(args.device) for k, v in encoding.items()}
            with torch.no_grad():
                all_layer_clean_hidden_states = bert_cls.bert(**encoding)[2]
                cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1].detach().clone()
            pooled_output = bert_cls_with_mixed_hidden_states.bert(input_ids=encoding['input_ids'],
                                                                   attention_mask=encoding['attention_mask'],
                                                                   token_type_ids=encoding['token_type_ids'],
                                                                   hidden_states_mask=hidden_states_mask,
                                                                   external_hidden_states=cut_off_layer_clean_hidden_states)[1]
            pooled_output = bert_cls_with_mixed_hidden_states.dropout(pooled_output)
            logits = bert_cls_with_mixed_hidden_states.classifier(pooled_output)
            target_logits = torch.sum(logits * one_hot_target_label, dim=-1)
            non_target_logits = torch.max((1 - one_hot_target_label) * logits - 10000 * one_hot_target_label, dim=-1).values
            loss = (- non_target_logits + target_logits).mean()
            loss.backward()
            grad_norm = sum([torch.norm(params.grad.data).item() for params in optimized_params])
            grad_norm_list.append(grad_norm)
            bert_cls_with_mixed_hidden_states.zero_grad()
        top_k_indices = torch.topk(torch.tensor(grad_norm_list), k=args.k_shot).indices.tolist()
        few_shot_label_dict[f'{source}->{target}'] = [text_list[i] for i in top_k_indices]

    return few_shot_label_dict


def collect_data_at_decision_boundary_from_corpus(bert_cls, tokenizer, args):
    df = pd.read_csv(args.wild_corpus_csv_dir)
    corpus_text_list = [df['text'][i] for i in range(df.shape[0])]
    random.shuffle(corpus_text_list)

    confidence_lower_bound = 1. / args.num_labels
    confidence_upper_bound = 1. / args.num_labels + 0.2

    few_shot_label_text_dict = {}
    full_label_text_dict = {}
    for label in range(args.num_labels):
        few_shot_label_text_dict[label] = []
        full_label_text_dict[label] = []
    for text in corpus_text_list:
        text = text.replace('<unk>', '[UNK]')
        if len(text.split()) < 50:
            continue
        encoding = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
        encoding = {k: v.to(args.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = bert_cls(**encoding)[0]
            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values
        if confidence_lower_bound < confidence[0].item() < confidence_upper_bound:
            full_label_text_dict[int(preds[0])].append(text)
            if check_complete_collect(full_label_text_dict, args.generalize_samples_num):
                print('Finish Collecting batched data near decision boundary !')
                break
    for label in range(args.num_labels):
        few_shot_label_text_dict[label] = random.sample(full_label_text_dict[label], k=args.k_shot)

    return full_label_text_dict, few_shot_label_text_dict


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


def add_additional_token_ids(tokenizer, batched_data, args, add_token_id):
    for i in range(len(batched_data['input_ids'])):
        input_ids = batched_data['input_ids'][i]
        attention_mask = batched_data['attention_mask'][i]
        for j in range(args.pos_mask_start, args.pos_mask_end + 1):
            input_ids.insert(j, add_token_id)
            attention_mask.insert(j, 1)
        truncate_token_id = input_ids[tokenizer.model_max_length - 1]
        if truncate_token_id == tokenizer.pad_token_id:
            input_ids = input_ids[: tokenizer.model_max_length]
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
    suspect_batched_data = tokenizer(batched_texts,
                                     truncation=True,
                                     padding='max_length',
                                     return_tensors='pt')
    if add_token_id is not None:
        suspect_batched_data = {k: v.tolist() for k, v in suspect_batched_data.items()}
        suspect_batched_data = add_additional_token_ids(tokenizer, suspect_batched_data, args, add_token_id)
        suspect_batched_data = {k: torch.tensor(v) for k, v in suspect_batched_data.items()}
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
    victim_batched_data = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'label': []}
    for label, text_list in victim_label_text_dict.items():
        sub_victim_batched_data = tokenizer(text_list,
                                            truncation=True,
                                            padding='max_length',
                                            return_tensors='pt')
        for key, value in sub_victim_batched_data.items():
            victim_batched_data[key].append(value)
        victim_batched_data['label'].append(torch.tensor(label).repeat(len(text_list)))
    for key in victim_batched_data.keys():
        victim_batched_data[key] = torch.cat(victim_batched_data[key], dim=0)

    if add_token_id is not None:
        victim_batched_data = {k: v.tolist() for k, v in victim_batched_data.items()}
        victim_batched_data = add_additional_token_ids(tokenizer, victim_batched_data, args, add_token_id)
        victim_batched_data = {k: torch.tensor(v) for k, v in victim_batched_data.items()}

    # shuffled_idx = list(range(victim_batched_data['input_ids'].shape[0]))
    # random.shuffle(shuffled_idx)
    # for key in victim_batched_data.keys():
    #    victim_batched_data[key] = victim_batched_data[key][shuffled_idx]

    return victim_batched_data


def tokenize_specific_victim_data(tokenizer, label_text_dict, source_label, target_label, args,
                                  add_token_id=None, mode='few_shot'):
    if mode == 'few_shot':
        batched_texts = label_text_dict[f'{source_label}->{target_label}']
    else:  # mode == 'full_shot'
        batched_texts = label_text_dict[source_label]
    victim_batched_data = tokenizer(batched_texts,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='pt')

    if add_token_id is not None:
        victim_batched_data = {k: v.tolist() for k, v in victim_batched_data.items()}
        victim_batched_data = add_additional_token_ids(tokenizer, victim_batched_data, args, add_token_id)
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


def get_suspect_mean_pooled_output(args, bert_cls, target_batched_data, mean_pooled_output_save_dir):
    target_batched_data = {k: v.to(args.device) for k, v in target_batched_data.items()}

    # directly use target label batched data to get mean pooled output of target class samples
    pooled_output_list = []
    for i in range(len(target_batched_data['input_ids'])):
        sub_target_batched_data = {k: v[i].unsqueeze(0) for k, v in target_batched_data.items()}
        with torch.no_grad():
            pooled_output = bert_cls.bert(input_ids=sub_target_batched_data['input_ids'],
                                          attention_mask=sub_target_batched_data['attention_mask'],
                                          token_type_ids=sub_target_batched_data['token_type_ids'])[1]
            pooled_output_list.append(pooled_output)
    pooled_outputs = torch.cat(pooled_output_list, dim=0)
    pooled_outputs = pooled_outputs.detach()

    # get the center of pooled_outputs
    optimized_center = torch.zeros_like(pooled_outputs[0])
    optimized_center = Variable(optimized_center.data, requires_grad=True)
    optimizer = torch.optim.Adam(params=[optimized_center], lr=1e-3)
    optimizer.zero_grad()
    for _ in range(5000):
        loss = torch.mean(torch.norm(optimized_center - pooled_outputs, dim=-1) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save(optimized_center.data.cpu(), mean_pooled_output_save_dir)


def get_mean_pooled_output_and_correctly_classified_data(args, bert_cls_model, suspect_label, suspect_batched_data):
    suspect_mean_pooled_output_save_dir = args.model_path + '/center_pooled_output_suspect_label_{}.pt'.format(
        suspect_label)
    get_suspect_mean_pooled_output(args, bert_cls_model, suspect_batched_data,
                                   suspect_mean_pooled_output_save_dir)
    return suspect_mean_pooled_output_save_dir


def get_cls_similarity_matrix(hidden_states_1, hidden_states_2):
    """
    Calculate self similarity matrix of each layer's hidden state and cross similarity matrix between
    hidden_states_1 and hidden_states_2
    """
    assert len(hidden_states_1) == len(hidden_states_2), \
        'size of hidden_states_1 must be equal to size of hidden_states_2 must be equal !'
    self_similarity_hidden_states_1 = ()
    self_similarity_hidden_states_2 = ()
    cross_similarity_hidden_states = ()
    for layer in range(len(hidden_states_1)):
        layer_hidden_states_1 = hidden_states_1[layer]
        cls_layer_hidden_states_1 = layer_hidden_states_1[:, 0, :]
        normalized_cls_layer_hidden_states_1 = cls_layer_hidden_states_1 / torch.norm(cls_layer_hidden_states_1,
                                                                                      dim=-1).view(-1, 1)
        layer_hidden_states_2 = hidden_states_2[layer]
        cls_layer_hidden_states_2 = layer_hidden_states_2[:, 0, :]
        normalized_cls_layer_hidden_states_2 = cls_layer_hidden_states_2 / torch.norm(cls_layer_hidden_states_2,
                                                                                      dim=-1).view(-1, 1)
        self_similarity_matrix_1 = torch.matmul(normalized_cls_layer_hidden_states_1,
                                                normalized_cls_layer_hidden_states_1.transpose(0, 1))
        self_similarity_matrix_2 = torch.matmul(normalized_cls_layer_hidden_states_2,
                                                normalized_cls_layer_hidden_states_2.transpose(0, 1))
        cross_similarity = torch.matmul(normalized_cls_layer_hidden_states_1,
                                        normalized_cls_layer_hidden_states_2.transpose(0, 1))
        self_similarity_hidden_states_1 = self_similarity_hidden_states_1 + (self_similarity_matrix_1, )
        self_similarity_hidden_states_2 = self_similarity_hidden_states_2 + (self_similarity_matrix_2, )
        cross_similarity_hidden_states = cross_similarity_hidden_states + (cross_similarity, )
    return self_similarity_hidden_states_1, self_similarity_hidden_states_2, cross_similarity_hidden_states


def get_cls_similarity_score(self_sim_1, self_sim_2, cross_sim):
    self_sim_score_1 = ()
    self_sim_score_2 = ()
    cross_sim_score = ()
    for layer in range(len(self_sim_1)):
        layer_self_sim_1 = self_sim_1[layer]
        layer_self_sim_score_1 = (layer_self_sim_1.sum() - layer_self_sim_1.size(0)) / layer_self_sim_1.size(0) / (layer_self_sim_1.size(0) - 1)
        self_sim_score_1 = self_sim_score_1 + (layer_self_sim_score_1, )
        layer_self_sim_2 = self_sim_2[layer]
        layer_self_sim_score_2 = (layer_self_sim_2.sum() - layer_self_sim_2.size(0)) / layer_self_sim_2.size(0) / (layer_self_sim_2.size(0) - 1)
        self_sim_score_2 = self_sim_score_2 + (layer_self_sim_score_2, )
        layer_cross_sim = cross_sim[layer]
        layer_cross_sim_score = layer_cross_sim.mean()
        cross_sim_score = cross_sim_score + (layer_cross_sim_score, )
    return self_sim_score_1, self_sim_score_2, cross_sim_score


def check_generalization(args, check_data, target_mean_pooled_output_save_dir, target_label):
    """
    Note that we only optimize weight perturbation on few-shot data.
    Check generalization ability of optimized weight perturbation on full data.
    """
    bert_cls = BertForSequenceClassification.from_pretrained(args.model_path,
                                                             num_labels=args.num_labels,
                                                             return_dict=False)
    bert_cls.to(args.device)
    bert_cls.eval()
    load_perturbed_params(args, bert_cls, target_label)
    iteration = len(check_data['input_ids']) // args.k_shot
    center_target_pooled_output = torch.load(target_mean_pooled_output_save_dir, map_location='cpu').to(args.device)
    center_target_pooled_output = center_target_pooled_output / torch.norm(center_target_pooled_output, dim=-1)

    self_similarity_score_list = []
    center_similarity_score_list = []
    for i in tqdm(range(iteration)):
        batched_data = {k: v[i * args.k_shot: (i + 1) * args.k_shot] for k, v in check_data.items()}
        batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
        with torch.no_grad():
            pooled_output = bert_cls.bert(input_ids=batched_data['input_ids'],
                                          token_type_ids=batched_data['token_type_ids'],
                                          attention_mask=batched_data['attention_mask'])[1]
            normalized_pooled_output = pooled_output / torch.norm(pooled_output, dim=-1).view(-1, 1)
            similarity_matrix = torch.matmul(normalized_pooled_output, torch.transpose(normalized_pooled_output, 0, 1))
            self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
            self_similarity_score = self_similarity_score / pooled_output.shape[0] / (pooled_output.shape[0] - 1)
            similarity_vector = torch.matmul(normalized_pooled_output, center_target_pooled_output.view(-1, 1))
            center_similarity_score = torch.mean(similarity_vector)
            self_similarity_score_list.append(self_similarity_score.item())
            center_similarity_score_list.append(center_similarity_score.item())

    return self_similarity_score_list, center_similarity_score_list


def correctly_classified(args, bert_cls, bert_tokenizer, full_data_label_dict):
    """
    Return correctly classified samples predicted by the network
    """
    bert_cls.eval()
    correctly_classified_full_data_label_dict = {}
    for label in full_data_label_dict.keys():
        correctly_classified_full_data_label_dict[label] = []
    for label, text_list in full_data_label_dict.items():
        iteration = len(text_list) // args.bsz
        for i in range(iteration + 1):
            if i < iteration:
                sub_text_list = text_list[i * args.bsz: (i + 1) * args.bsz]
            elif iteration * args.bsz == len(text_list):
                break
            else:
                sub_text_list = text_list[iteration * args.bsz:]
            batched_data = bert_tokenizer(sub_text_list,
                                          truncation=True,
                                          padding='max_length',
                                          return_tensors='pt')
            batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            with torch.no_grad():
                logits = bert_cls(**batched_data)[0]
                preds = torch.argmax(logits, dim=-1)
                labels = torch.tensor(label).repeat(len(sub_text_list)).to(args.device)
                false_preds_id = torch.nonzero(preds - labels).view(-1).cpu().tolist()
            for j in range(len(sub_text_list)):
                if j not in false_preds_id:
                    correctly_classified_full_data_label_dict[label].append(sub_text_list[j])
    return correctly_classified_full_data_label_dict


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


def get_word_token_matrix(tokenizer, vocabulary):
    token_list = []
    for word in vocabulary:
        if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, RobertaTokenizer):
            tokenize_result = tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        else:
            tokenize_result = tokenizer.encode(word, add_special_tokens=False)
        token_list.append(tokenize_result)
    max_sub_token_num = max([len(v) for v in token_list])
    word_token_matrix = []
    for i in range(len(token_list)):
        padded_tokenize_result = token_list[i] + \
                                 [tokenizer.pad_token_id for _ in range(max_sub_token_num - len(token_list[i]))]
        word_token_matrix.append(padded_tokenize_result)
    return np.array(word_token_matrix)
