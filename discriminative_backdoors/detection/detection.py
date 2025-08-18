import argparse
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import logging
from discriminative_backdoors.detection.data_utils import *
from discriminative_backdoors.detection.perturb_bert_utils import *
from discriminative_backdoors.detection.perturb_roberta_utils import *
from model_path_utils import *


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


def backdoor_detection_on_final_model_type(args):

    # load the tokenizer and transformer
    tokenizer, transformer = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = args.model_max_length
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # use the model to extract reference samples
    full_shot_label_text_dict, few_shot_label_text_dict = manual_label_corpus_data_and_sample_full_and_few_shot_data(
        transformer, tokenizer, args
    )

    # prepare (source, target) pairs for scanning
    label_pair_list = []
    for target in range(args.num_labels):
        for source in range(args.num_labels):
            if source != target:
                label_pair_list.append((source, target))

    # backdoor scanning for every possible (source, target) label pair
    backdoor_metric_list = []
    for (source, target) in label_pair_list:
        # at this time we assume the suspect label to be the attacker's chosen target label
        if args.add_tokens:  # optional, default False
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

        if args.perturb_mode:
            # weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                weight_perturbation_with_mixed_hidden_states_bert(
                    args, transformer, few_shot_victim_batched_data, source, target, logger
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                weight_perturbation_with_mixed_hidden_states_roberta(
                    args, transformer, few_shot_victim_batched_data, source, target, logger
                )
            else:
                raise ValueError("This model type is not implemented")
        if args.check_weight_generalization_mode:
            # measure the generalization of the weight perturbation
            if args.model_type == 'bert-base' or args.model_type == 'bert-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_bert(
                    args, transformer, full_shot_victim_batched_data, source, target, True
                )
            elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
                return_dict, mask_len, metric_list, margin_list = check_weight_perturbation_generalization_roberta(
                    args, transformer, full_shot_victim_batched_data, source, target, True
                )
            else:
                raise ValueError("This model type is not implemented")
            print('weight perturbation generalization {} -> {}: {}, {}'.format(
                source, target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: {} -> {}: {}'.format(source, target, get_hist_entropy(margin_list)))
            if np.mean(margin_list) < 0:
                backdoor_metric_list.append(4.0)
            else:
                backdoor_metric_list.append(get_hist_entropy(margin_list))
    # backdoor judgment
    if min(backdoor_metric_list) < args.detection_threshold:
        print('The model is detected as backdoored.')
    else:
        print('The model is considered as benign.')


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
    parser.add_argument("--perturb_attention", default=False, action='store_true')
    parser.add_argument("--perturb_intermediate", default=False, action='store_true')
    parser.add_argument("--perturb_self_output", default=False, action='store_true')
    parser.add_argument("--freeze_bias", default=False, action='store_true')
    parser.add_argument("--bsz", type=int, default=20)
    parser.add_argument("--k_shot", type=int, default=20)
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
    parser.add_argument("--use_chatgpt", default=False, action='store_true')
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--detection_threshold", type=float, default=2.0)
    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # model path
    root_path = '/home/user'
    if args.model_type == 'bert-base':
        model_path, selected_corpus_csv_dir, num_labels = get_model_bert_base_path(args.model_name, root_path, args.use_chatgpt)
    elif args.model_type == 'roberta-base':
        model_path, selected_corpus_csv_dir, num_labels = get_model_roberta_base_path(args.model_name, root_path, args.use_chatgpt)
    else:
        raise ValueError('This model type is not implemented')
    print(model_path)
    args.model_path = model_path
    args.selected_corpus_csv_dir = selected_corpus_csv_dir
    args.num_labels = num_labels
    args.wild_corpus_csv_dir = f'{root_path}/{args.wild_corpus_csv_dir}'

    # logger setting
    if args.perturb_attention:
        logger_file = f'/home/user/discriminative_backdoors/detection/' \
                      f'{args.model_type}_results/{args.model_name}-perturb_attention.log'
    elif args.perturb_intermediate:
        logger_file = f'/home/user/discriminative_backdoors/detection/' \
                      f'{args.model_type}_results/{args.model_name}-perturb_intermediate.log'

    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_file, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    args.logger = logger

    backdoor_detection_on_final_model_type(args)


if __name__ == '__main__':
    main()
