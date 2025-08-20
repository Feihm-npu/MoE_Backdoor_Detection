import argparse
import json
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import logging
from data_utils import *
from perturb_bert_utils import *
from perturb_roberta_utils import *
from perturb_gpt2_utils import *
from model_path_utils import *
from perturb_bert_utils_static import *


logger = logging.getLogger(__name__)


def load_transformer_and_tokenizer(args):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
        transformer = BertForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            output_hidden_states=True,
            return_dict=False
        )
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
        transformer = RobertaForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            output_hidden_states=True,
            return_dict=False
        )
    elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'
        transformer = GPT2ForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            output_hidden_states=True,
            return_dict=False
        )
        transformer.config.pad_token_id = tokenizer.eos_token_id
    else:
        raise NotImplementedError('This model type is not implemented')

    return tokenizer, transformer


def backdoor_detection_on_final_model_type_I(args):

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    # prepare for log info
    weight_generalization_return_dict_list = []
    weight_generalization_mask_len_list = []
    weight_generalization_metric_list_list = []
    weight_generalization_margin_list_list = []

    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = tokenizer.unk_token_id if args.add_tokens else None
        # add_mask_token_id = None
        # add_pad_token_id = None
        # add_unk_token_id = None
        if args.add_tokens:
            add_token_id_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_start, args.pos_mask_end + 1)]
        else:
            add_token_id_list = None
        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target,
            args, add_token_id_list, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, full_shot_label_text_dict, source, target,
            args, add_token_id_list, mode='full_shot'
        )
        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target,
            args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert(
                    args, transformer, few_shot_victim_batched_data,
                    source, target, suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data,
                    source, target, suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
                weight_perturbation_with_mixed_hidden_states_gpt2(
                    args, transformer, few_shot_victim_batched_data,
                    source, target, suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_gpt2(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            else:
                raise ValueError("This model type is not implemented")
            weight_generalization_return_dict_list.append(return_dict)
            weight_generalization_mask_len_list.append(mask_len)
            weight_generalization_metric_list_list.append(metric_list)
            weight_generalization_margin_list_list.append(margin_list)
            print('weight perturbation generalization {} -> {}: {}, {}'.format(
                source, target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def backdoor_detection_on_final_model_type_II(args):
    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # prepare for log info
    weight_generalization_return_dict_list = []
    weight_generalization_mask_len_list = []
    weight_generalization_metric_list_list = []
    weight_generalization_margin_list_list = []

    # scanning for every possible (source, target) label pair
    if args.specified_target_labels is None:
        suspect_labels = list(range(args.num_labels))
    else:
        suspect_labels = args.specified_target_labels
    for target in suspect_labels:
        logger.info('*********************************************************************************************')
        logger.info('target: {}'.format(target))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = bert_tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = bert_tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = bert_tokenizer.unk_token_id if args.add_tokens else None
        add_mask_token_id = None
        add_pad_token_id = None
        add_unk_token_id = None
        few_shot_victim_batched_data = tokenize_agnostic_victim_data(
            tokenizer, few_shot_label_text_dict, target, args, add_mask_token_id, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_agnostic_victim_data(
            tokenizer, full_shot_label_text_dict, target, args, add_mask_token_id, mode='full_shot'
        )

        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, None, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert(
                    args, transformer, few_shot_victim_batched_data, None, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, None, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert(
                    args, transformer,
                    full_shot_victim_batched_data, None,
                    target, suspect_mean_cls_output_save_dir, False
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer,
                    full_shot_victim_batched_data, None,
                    target, suspect_mean_cls_output_save_dir, False
                )
            else:
                raise ValueError("This model type is not implemented")
            weight_generalization_return_dict_list.append(return_dict)
            weight_generalization_mask_len_list.append(mask_len)
            weight_generalization_metric_list_list.append(metric_list)
            weight_generalization_margin_list_list.append(margin_list)
            print('Weight perturbation generalization other -> {}: {}'.format(target, get_mean_std_ratio(margin_list)))


def backdoor_detection_on_final_model_type_III(args):
    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # prepare for logger info
    weight_generalization_metric_list_list = []
    weight_generalization_margin_list_list = []
    weight_generalization_return_dict_list = []
    weight_generalization_mask_len_list = []

    # scanning for every possible (source, target) label pair
    for target in range(args.num_labels):
        logger.info('*********************************************************************************************')
        logger.info('target: {}'.format(target))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = bert_tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = bert_tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = bert_tokenizer.unk_token_id if args.add_tokens else None
        add_mask_token_id = None
        add_pad_token_id = None
        add_unk_token_id = None
        few_shot_victim_batched_data = tokenize_agnostic_victim_data(
            tokenizer, few_shot_label_text_dict, target, args, add_mask_token_id, mode='few_shot'
        )

        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, None, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert(
                    args, transformer, few_shot_victim_batched_data, None, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, None, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            for source in range(args.num_labels):
                if source == target:
                    continue
                full_shot_victim_batched_data = tokenize_specific_victim_data(
                    tokenizer, full_shot_label_text_dict, source, target, args, add_mask_token_id, mode='full_shot'
                )
                if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                    return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert(
                        args, transformer,
                        full_shot_victim_batched_data, source,
                        target, suspect_mean_cls_output_save_dir, False
                    )
                    weight_generalization_metric_list_list.append(metric_list)
                    weight_generalization_margin_list_list.append(margin_list)
                    weight_generalization_return_dict_list.append(return_dict)
                    weight_generalization_mask_len_list.append(mask_len)
                elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                    return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                        args, transformer,
                        full_shot_victim_batched_data, source,
                        target, suspect_mean_cls_output_save_dir, False
                    )
                    weight_generalization_metric_list_list.append(metric_list)
                    weight_generalization_margin_list_list.append(margin_list)
                    weight_generalization_return_dict_list.append(return_dict)
                    weight_generalization_mask_len_list.append(mask_len)
                else:
                    raise ValueError("This model type is not implemented")
                print('Weight perturbation generalization {} -> {}: {}'.format(source, target, get_mean_std_ratio(margin_list)))


def backdoor_detection_on_final_model_type_IV(args):

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    # prepare for log info
    weight_generalization_return_dict_list = []
    weight_generalization_mask_len_list = []
    weight_generalization_metric_list_list = []
    weight_generalization_margin_list_list = []

    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        add_mask_token_id = tokenizer.mask_token_id if args.add_tokens else None
        add_pad_token_id = tokenizer.pad_token_id if args.add_tokens else None
        add_unk_token_id = tokenizer.unk_token_id if args.add_tokens else None
        # add_mask_token_id = None
        # add_pad_token_id = None
        # add_unk_token_id = None
        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, add_mask_token_id, mode='few_shot'
        )
        # full_shot_victim_batched_data = tokenize_specific_victim_data(tokenizer, full_shot_label_text_dict, source,
        #                                                              target, args, add_mask_token_id, mode='full_shot')

        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            full_shot_victim_batched_data = tokenize_specific_victim_data(
                tokenizer, full_shot_label_text_dict, source, target, args, add_mask_token_id, mode='full_shot'
            )
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, source
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, source
                )
            else:
                raise ValueError("This model type is not implemented")
            weight_generalization_return_dict_list.append(return_dict)
            weight_generalization_mask_len_list.append(mask_len)
            weight_generalization_metric_list_list.append(metric_list)
            weight_generalization_margin_list_list.append(margin_list)
            print('weight perturbation generalization {} -> {}: {}'.format(source, target, get_mean_std_ratio(margin_list)))
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))

            full_shot_victim_batched_data = tokenize_specific_victim_data(
                tokenizer, full_shot_label_text_dict, target, source, args, add_mask_token_id, mode='full_shot'
            )
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, target
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, target
                )
            else:
                raise ValueError("This model type is not implemented")

            df = pd.read_csv('/data/zengrui/nlp_dataset/jigsaw/40_ijr_11/process_p_test_target_0_greedy.csv')
            text_list = [df['comment_text'][i] for i in range(df.shape[0])]
            trigger_batched_data = tokenizer(text_list, truncation=True, padding='max_length', return_tensors='pt')
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_for_visualization(
                    args, transformer,
                    trigger_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, 'poison'
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    trigger_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, 'poison'
                )
            else:
                raise ValueError("This model type is not implemented")

            weight_generalization_return_dict_list.append(return_dict)
            weight_generalization_mask_len_list.append(mask_len)
            weight_generalization_metric_list_list.append(metric_list)
            weight_generalization_margin_list_list.append(margin_list)
            print('weight perturbation generalization {} -> {}: {}'.format(source, target,
                                                                           get_mean_std_ratio(margin_list)))
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def backdoor_detection_on_final_model_type_V(args):

    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        add_token_ids_list = random.sample(list(range(999, 29643)), k=args.pos_mask_end - args.pos_mask_start + 1)
        # add_token_ids_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        add_token_ids_list = random.sample(list(range(5, 50000)), k=args.pos_mask_end - args.pos_mask_start + 1)
        # add_token_ids_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
    else:
        raise ValueError('This model type is not implemented!')

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    print(tokenizer.convert_ids_to_tokens(add_token_ids_list))
    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = tokenizer.unk_token_id if args.add_tokens else None
        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, add_token_ids_list, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, full_shot_label_text_dict, source, target, args, add_token_ids_list, mode='full_shot'
        )

        # few_shot_suspect_batched_data = tokenize_specific_victim_data(tokenizer, few_shot_label_text_dict, target,
        #                                                             source, args, add_token_id_list, mode='few_shot')
        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert_static_layernorm(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static_layernorm(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            else:
                raise ValueError("This model type is not implemented")
            print('weight perturbation generalization {} -> {}: {}, {}'.format(
                source, target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def backdoor_detection_on_final_model_type_VI(args):

    if args.add_tokens:
        if args.model_type == 'bert-base' or args.model_type == 'bert-large':
            temp_add_token_id_list = random.sample(list(range(999, 29643)), k=args.pos_mask_end - args.pos_mask_start + 1)
            # add_token_id_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
        elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
            temp_add_token_id_list = random.sample(list(range(5, 50000)), k=args.pos_mask_end - args.pos_mask_start + 1)
            # add_token_id_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
        else:
            raise ValueError('This model type is not implemented!')
    else:
        temp_add_token_id_list = []

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = tokenizer.unk_token_id if args.add_tokens else None
        if args.add_tokens:
            if os.path.exists(f'{args.model_path}/piccolo_inverse_trigger_info_seed_{args.inversion_seed}_'
                              f'min_acc_on_benign_model_{args.min_acc_on_benign_model}_trial_{args.trial_num}.json'):
                print('Using PICCOLO inversed trigger')
                with open(f'{args.model_path}/piccolo_inverse_trigger_info_seed_{args.inversion_seed}_'
                          f'min_acc_on_benign_model_{args.min_acc_on_benign_model}_trial_{args.trial_num}.json') as f:
                    inverse_trigger_info = json.load(f)
                    if args.second_half:
                        add_token_id_list = inverse_trigger_info[2 * target + 1]['trigger_idxs']
                    else:
                        add_token_id_list = inverse_trigger_info[2 * target]['trigger_idxs']
                    for i in range(args.pos_mask_end - args.pos_mask_start + 1 - len(add_token_id_list)):
                        add_token_id_list.append(temp_add_token_id_list[i])
            else:
                add_token_id_list = temp_add_token_id_list
            print(tokenizer.convert_ids_to_tokens(add_token_id_list))
        else:
            add_token_id_list = None

        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, add_token_id_list, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, full_shot_label_text_dict, source, target, args, add_token_id_list, mode='full_shot'
        )

        # few_shot_suspect_batched_data = tokenize_specific_victim_data(tokenizer, few_shot_label_text_dict, target,
        #                                                             source, args, add_token_id_list, mode='few_shot')
        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert_static(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            else:
                raise ValueError("This model type is not implemented")
            print('weight perturbation generalization {} -> {}: {}, {}'.format(
                source, target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def backdoor_detection_on_final_model_type_VII(args):

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        # construct victim label batched data
        # add_mask_token_id = tokenizer.mask_token_id if args.add_tokens else None
        # add_pad_token_id = tokenizer.pad_token_id if args.add_tokens else None
        # add_unk_token_id = tokenizer.unk_token_id if args.add_tokens else None
        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, None, mode='few_shot'
        )
        add_token_ids_list = score_added_token(few_shot_victim_batched_data, target, tokenizer, transformer, args)
        print(tokenizer.convert_ids_to_tokens(add_token_ids_list))
        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, add_token_ids_list, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, full_shot_label_text_dict, source, target, args, add_token_ids_list, mode='full_shot'
        )

        # few_shot_suspect_batched_data = tokenize_specific_victim_data(tokenizer, few_shot_label_text_dict, target,
        #                                                             source, args, add_token_id_list, mode='few_shot')
        # set initial perturbation to zero
        # reset_perturbation(args, bert_cls_model)

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert_static(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True
                )
            else:
                raise ValueError("This model type is not implemented")
            print('weight perturbation generalization {} -> {}: {}, {}'.format(
                source, target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def backdoor_detection_on_final_model_type_VIII(args):

    if args.add_tokens:
        if args.model_type == 'bert-base' or args.model_type == 'bert-large':
            temp_add_token_id_list = random.sample(list(range(999, 29643)), k=args.pos_mask_end - args.pos_mask_start + 1)
            # add_token_id_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
        elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
            temp_add_token_id_list = random.sample(list(range(5, 50000)), k=args.pos_mask_end - args.pos_mask_start + 1)
            # add_token_id_list = [tokenizer.mask_token_id for _ in range(args.pos_mask_end - args.pos_mask_start + 1)]
        else:
            raise ValueError('This model type is not implemented!')
    else:
        temp_add_token_id_list = []

    # tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to label previously selected corpus data
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # (source target) label pairs
    label_pair_list = []
    if args.specified_target_labels is None:
        for target in range(args.num_labels):
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    else:
        for target in args.specified_target_labels:
            for source in range(args.num_labels):
                if source != target:
                    label_pair_list.append((source, target))
    if args.specified_label_pairs is not None:
        label_pair_list = [(args.specified_label_pairs[0], args.specified_label_pairs[1])]

    # prepare for log info
    weight_generalization_return_dict_list = []
    weight_generalization_mask_len_list = []
    weight_generalization_metric_list_list = []
    weight_generalization_margin_list_list = []

    # scanning for every possible (source, target) label pair
    for (source, target) in label_pair_list:
        logger.info('*********************************************************************************************')
        logger.info('(source, target): {}'.format((source, target)))
        logger.info('*********************************************************************************************')
        # at this time we assume suspect label to be attacker's chosen target label

        if args.add_tokens:
            if os.path.exists(f'{args.model_path}/piccolo_inverse_trigger_info_seed_{args.inversion_seed}_'
                              f'min_acc_on_benign_model_{args.min_acc_on_benign_model}_trial_{args.trial_num}.json'):
                print('Using PICCOLO inverted trigger')
                with open(f'{args.model_path}/piccolo_inverse_trigger_info_seed_{args.inversion_seed}_'
                          f'min_acc_on_benign_model_{args.min_acc_on_benign_model}_trial_{args.trial_num}.json') as f:
                    inverse_trigger_info = json.load(f)
                    if args.second_half:
                        add_token_id_list = inverse_trigger_info[2 * target + 1]['trigger_idxs']
                    else:
                        add_token_id_list = inverse_trigger_info[2 * target]['trigger_idxs']
                    for i in range(args.pos_mask_end - args.pos_mask_start + 1 - len(add_token_id_list)):
                        add_token_id_list.append(temp_add_token_id_list[i])
            else:
                add_token_id_list = temp_add_token_id_list
            print(tokenizer.convert_ids_to_tokens(add_token_id_list))
        else:
            add_token_id_list = None

        few_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, few_shot_label_text_dict, source, target, args, add_token_id_list, mode='few_shot'
        )
        full_shot_victim_batched_data = tokenize_specific_victim_data(
            tokenizer, full_shot_label_text_dict, source, target, args, add_token_id_list, mode='full_shot'
        )

        # get the center of pooled outputs from suspect label batched data
        full_shot_suspect_batched_data = tokenize_suspect_data(
            tokenizer, full_shot_label_text_dict, source, target, args, None, mode='full_shot'
        )
        suspect_mean_cls_output_save_dir = get_mean_cls_output(
            args, transformer, target, full_shot_suspect_batched_data
        )

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert_static(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target,
                    suspect_mean_cls_output_save_dir, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # check weight perturbation generalization
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, source
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, source
                )
            else:
                raise ValueError("This model type is not implemented")

            print('weight perturbation generalization {} -> {}: {}'.format(source, target, get_mean_std_ratio(margin_list)))
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))

            full_shot_victim_batched_data = tokenize_specific_victim_data(
                tokenizer, full_shot_label_text_dict, target, source, args, None, mode='full_shot'
            )
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, target
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    full_shot_victim_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, target
                )
            else:
                raise ValueError("This model type is not implemented")

            trigger_id_list = tokenizer.encode('intense felt constitutions immensity', add_special_tokens=False)
            trigger_batched_data = tokenize_specific_victim_data(
                tokenizer, full_shot_label_text_dict, target, source, args, trigger_id_list, mode='full_shot'
            )
            # df = pd.read_csv('/data/zengrui/nlp_dataset/glue/SST-2/2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv')
            # text_list = []
            # for i in range(df.shape[0]):
            #    text = df['text'][i]
            #    text_split = text.split()
            #    text_split = text_split[0: 10] + ['olympic', 'whiff', 'matter'] + text_split[10:]
                # text_split = ['olympic', 'whiff', 'matter'] + text_split
            #    text_list.append(' '.join(text_split))
            # trigger_batched_data = tokenizer(text_list, truncation=True, padding='max_length', return_tensors='pt')
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert_static_for_visualization(
                    args, transformer,
                    trigger_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, 'poison'
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta_for_visualization(
                    args, transformer,
                    trigger_batched_data, source,
                    target, suspect_mean_cls_output_save_dir, True, 'poison'
                )
            else:
                raise ValueError("This model type is not implemented")

            print('weight perturbation generalization {} -> {}: {}'.format(source, target, get_mean_std_ratio(margin_list)))
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-base')
    parser.add_argument("--model_name", type=str, default='badnl-benign-1')
    parser.add_argument("--tokenizer_path", type=str, default='../bert-base-uncased')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--whole_epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--start_layer", type=int, default=6)
    parser.add_argument("--end_layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--norm_type", type=str, default='l-2')
    parser.add_argument("--weight_budget", type=float, default=2.0)
    parser.add_argument("--bias_budget", type=float, default=2.0)
    parser.add_argument("--cls_loss_type", type=str, default='untarget-ce')
    parser.add_argument("--non_target_ce_loss_threshold", type=float, default=7.5)
    parser.add_argument("--non_target_break_ce_loss_threshold", type=float, default=7.5)
    parser.add_argument("--target_break_ce_loss_threshold", type=float, default=0.001)
    parser.add_argument("--margin_threshold", type=float, default=6.0)
    parser.add_argument("--margin_break_threshold", type=float, default=5.9)
    parser.add_argument("--self_sim_threshold", type=float, default=0.006)
    parser.add_argument("--center_sim_upper_threshold", type=float, default=0.2)
    parser.add_argument("--center_sim_lower_threshold", type=float, default=0.1)
    parser.add_argument("--perturb_attention", default=False, action='store_true')
    parser.add_argument("--freeze_bias", default=False, action='store_true')
    parser.add_argument("--perturb_intermediate", default=False, action='store_true')
    parser.add_argument("--perturb_layernorm", default=False, action='store_true')
    parser.add_argument("--perturb_self_output", default=False, action='store_true')
    parser.add_argument("--bsz", type=int, default=20)
    parser.add_argument("--k_shot", type=int, default=20)
    parser.add_argument("--noise_min", type=int, default=0)
    parser.add_argument("--noise_max", type=int, default=10)
    parser.add_argument("--check_noise", default=False, action='store_true')
    parser.add_argument("--noise_samples_num", type=int, default=100)
    parser.add_argument("--noise_inner_bsz", type=int, default=4)
    parser.add_argument("--add_additional_embedding", default=False, action='store_true')
    parser.add_argument("--additional_embedding_train_epoch", type=int, default=1000)
    parser.add_argument("--noise_type", type=str, default='gaussian')
    parser.add_argument("--corruption_ratio", type=float, default=0.5)
    parser.add_argument("--scale_factor", type=float, default=2.0)
    parser.add_argument("--pos_mask_start", type=int, default=4)
    parser.add_argument("--pos_mask_end", type=int, default=13)
    parser.add_argument("--use_full_data", default=False, action='store_true')
    parser.add_argument("--perturb_mode", default=False, action='store_true')
    parser.add_argument("--generalize_samples_num", type=int, default=1000)
    parser.add_argument("--min_generalize_samples_num", type=int, default=250)
    parser.add_argument("--model_max_length", type=int, default=256)
    parser.add_argument("--few_shot_sample_mode", type=str, default='random')
    parser.add_argument("--full_shot_sample_mode", type=str, default='random')
    parser.add_argument("--wild_corpus_csv_dir", type=str, default='nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv')
    parser.add_argument("--add_tokens", default=False, action='store_true')
    parser.add_argument("--sim_metric", type=str, default='cosine')
    parser.add_argument("--detection_type", type=int, default=1)
    parser.add_argument("--directly_use_wild_corpus", default=False, action='store_true')
    parser.add_argument("--perturb_type", type=str, default='multiplicative')
    parser.add_argument("--server", type=str, default='g9')
    parser.add_argument("--check_weight_generalization_mode", default=False, action='store_true')
    parser.add_argument("--check_neuron_robustness_mode", default=False, action='store_true')
    parser.add_argument("--add_inter_sim_loss", default=False, action='store_true')
    parser.add_argument("--sim_loss_coefficient", type=float, default=1.0)
    parser.add_argument("--specified_target_labels", nargs='+', type=int, default=None)
    parser.add_argument("--specified_label_pairs", nargs='+', type=int, default=None)
    parser.add_argument("--not_random_sampling", default=False, action='store_true')
    parser.add_argument("--margin_upper_bound", type=float, default=None)
    parser.add_argument("--use_test_data", default=False, action='store_true')
    parser.add_argument("--poison_corpus", default=False, action='store_true')
    parser.add_argument("--poison_corpus_csv_dir", type=str, default=None)
    parser.add_argument("--add_attention_loss", default=False, action='store_true')
    parser.add_argument("--use_chatgpt", default=False, action='store_true')
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--embedding_len", type=int, default=5)
    parser.add_argument("--embedding_lr", type=float, default=0.01)
    parser.add_argument("--embedding_weight_decay", type=float, default=0.01)
    parser.add_argument("--optimize_embedding_mode", default=False, action='store_true')
    parser.add_argument("--embedding_budget", type=float, default=1.0)
    parser.add_argument("--second_half", default=False, action='store_true')
    parser.add_argument("--min_acc_on_benign_model", type=float, default=0.86)
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--inversion_seed", type=int, default=2024)
    parser.add_argument("--lamda", type=float, default=1.0)
    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # model path
    if args.server == 'g9':
        root_path = '/data/zengrui'
    elif args.server == 'g8':
        root_path = '/home/zengrui/zengrui'
    else:
        root_path = '/home/zr/zengrui'
    if args.model_type == 'bert-base':
        model_path, selected_corpus_csv_dir, num_labels = get_model_bert_base_path(args.model_name, root_path, args.use_chatgpt)
    elif args.model_type == 'bert-large':
        model_path, selected_corpus_csv_dir, num_labels = get_model_bert_large_path(args.model_name, root_path, args.use_chatgpt)
    elif args.model_type == 'roberta-base':
        model_path, selected_corpus_csv_dir, num_labels = get_model_roberta_base_path(args.model_name, root_path, args.use_chatgpt)
    elif args.model_type == 'gpt2':
        model_path, selected_corpus_csv_dir, num_labels = get_model_gpt2_path(args.model_name, root_path, args.use_chatgpt)
    else:
        raise ValueError('This model type is not implemented')
    print(model_path)
    args.model_path = model_path
    args.selected_corpus_csv_dir = selected_corpus_csv_dir
    args.num_labels = num_labels
    args.wild_corpus_csv_dir = f'{root_path}/{args.wild_corpus_csv_dir}'
    # logger setting
    if args.perturb_attention:
        if args.server == 'g7':
            logger_file = f'/home/zr/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_attention.log'
        else:
            logger_file = f'/home/zengrui/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_attention.log'
    elif args.perturb_intermediate:
        if args.server == 'g7':
            logger_file = f'/home/zr/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_intermediate.log'
        else:
            logger_file = f'/home/zengrui/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_intermediate.log'
    elif args.perturb_layernorm:
        if args.server == 'g7':
            logger_file = f'/home/zr/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_layernorm.log'
        else:
            logger_file = f'/home/zengrui/zengrui/nlp_backdoor_detection/' \
                          f'{args.model_type}_results/{args.model_name}-perturb_layernorm.log'
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_file, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    args.logger = logger

    if args.detection_type == 1:
        print('detection type I')
        backdoor_detection_on_final_model_type_I(args)
    elif args.detection_type == 2:  # deprecated
        print('detection type II')
        backdoor_detection_on_final_model_type_II(args)
    elif args.detection_type == 3:  # deprecated
        print('detection type III')
        backdoor_detection_on_final_model_type_III(args)
    elif args.detection_type == 4:  # visualization for dynamic backdoors
        print('detection type IV')
        backdoor_detection_on_final_model_type_IV(args)
    elif args.detection_type == 5:  # try to detect static backdoors
        print('detection type V')
        backdoor_detection_on_final_model_type_V(args)
    elif args.detection_type == 6:  # successful static backdoors
        print('detection type VI')
        backdoor_detection_on_final_model_type_VI(args)
    elif args.detection_type == 7:
        print('detection type VII')
        backdoor_detection_on_final_model_type_VII(args)
    elif args.detection_type == 8:  # visualization for static backdoors
        backdoor_detection_on_final_model_type_VIII(args)
    # backdoor_detection_on_pretrained_model(args)
    # backdoor_detection_on_plm(args)
    # print(args.model_name)
    # for suspect_label in range(args.num_labels):
    #    print(load_perturbed_pos(args, suspect_label))


if __name__ == '__main__':
    main()
