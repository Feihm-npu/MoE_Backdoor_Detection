
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from data_utils import *
import perturb_gpt2_utils
import perturb_gpt_neo_utils
import perturb_gpt_neox_utils
import perturb_opt_utils
from meta_backdoor_task import MetaBackdoorTask
import logging
import argparse

from model_path_utils import *
from peft import PeftModelForCausalLM


logger = logging.getLogger(__name__)


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = 1024
    return tokenizer


def load_transformer(args):
    transformer = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        output_hidden_states=True,
        return_dict=False,
        torch_dtype=torch.float16,
    )
    return transformer


def load_quantized_transformer(args):
    transformer = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        output_hidden_states=True,
        return_dict=False,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True if args.quantization else None,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ),
        use_cache=None
    )
    # if args.quantization:
    #    transformer = prepare_model_for_kbit_training(transformer)
    return transformer


def load_peft_model(transformer, args):
    model = PeftModelForCausalLM.from_pretrained(
        model=transformer,
        model_id=args.peft_model_path,
    )
    return model


def load_meta_task_model(args, tokenizer):
    meta_task_model = MetaBackdoorTask.from_pretrained(args.meta_task_model_path, torch_dtype=torch.float16)
    meta_task_model.tokenizer = tokenizer
    meta_task_model.meta_tokenizer = AutoTokenizer.from_pretrained(args.meta_task_model_path)
    meta_task_model.device = args.device
    meta_task_model.temperature = args.temperature
    meta_task_model.create_mapping()
    return meta_task_model


def backdoor_detection_on_final_model_type_I(args):

    # configure add_tokens (optional)
    if args.add_tokens:
        if args.model_type == 'gpt2' or args.model_type == 'gpt_neo' or args.model_type == 'pythia':
            add_token_id_list = random.sample(list(range(5, 50000)), k=args.pos_mask_end - args.pos_mask_start + 1)
        else:
            raise NotImplementedError('This model type is not implemented!')
    else:
        add_token_id_list = None

    # load the tokenizer and reference samples
    tokenizer = load_tokenizer(args)
    full_shot_batched_data, few_shot_batched_data = sample_full_and_few_shot_data(args, tokenizer)

    # load the transformer
    transformer = load_transformer(args)
    transformer.eval()
    transformer.to(args.device)
    for params in transformer.parameters():
        params.requires_grad = False

    # load the meta task model
    meta_task_model = load_meta_task_model(args, tokenizer)
    meta_task_model.eval()
    meta_task_model.to(args.device)
    for params in meta_task_model.parameters():
        params.requires_grad = False

    # add tokens (optional)
    if args.add_tokens:
        print(tokenizer.convert_ids_to_tokens(add_token_id_list))
        full_shot_batched_data = add_additional_token_ids(
            full_shot_batched_data, args.block_size, args.pos_mask_start, args.pos_mask_end, add_token_id_list
        )
        few_shot_batched_data = add_additional_token_ids(
            few_shot_batched_data, args.block_size, args.pos_mask_start, args.pos_mask_end, add_token_id_list
        )

    # prepare meta labels for scanning
    meta_label_list = []
    if args.specified_target_meta_labels is None:
        meta_label_list = list(range(args.num_meta_labels))
    else:
        for target in args.specified_target_meta_labels:
            meta_label_list.append(target)

    # backdoor scanning
    metric_list = []
    for target in meta_label_list:
        # few-shot perturbation injection
        if args.perturb_mode:
            if args.model_type == 'gpt2':
                perturb_gpt2_utils.weight_perturbation_with_mixed_hidden_states_gpt2_lm(
                    args, transformer, few_shot_batched_data, target, meta_task_model, logger
                )
            elif args.model_type == 'gpt_neo':
                perturb_gpt_neo_utils.weight_perturbation_with_mixed_hidden_states_gpt_neo_lm(
                    args, transformer, few_shot_batched_data, target, meta_task_model, logger
                )
            elif args.model_type == 'pythia':
                perturb_gpt_neox_utils.weight_perturbation_with_mixed_hidden_states_gpt_neox_lm(
                    args, transformer, few_shot_batched_data, target, meta_task_model, logger
                )
            else:
                raise NotImplementedError("This model is not implemented!")

        # few-shot perturbation generalization
        if args.check_weight_generalization_mode:
            if args.model_type == 'gpt2':
                return_dict, mask_len, margin_list = perturb_gpt2_utils.check_weight_perturbation_generalization_gpt2_lm(
                    args, transformer, full_shot_batched_data, meta_task_model, target
                )
            elif args.model_type == 'gpt_neo':
                return_dict, mask_len, margin_list = perturb_gpt_neo_utils.check_weight_perturbation_generalization_gpt_neo_lm(
                    args, transformer, full_shot_batched_data, meta_task_model, target
                )
            elif args.model_type == 'pythia':
                return_dict, mask_len, margin_list = perturb_gpt_neox_utils.check_weight_perturbation_generalization_gpt_neox_lm(
                    args, transformer, full_shot_batched_data, meta_task_model, target
                )
            else:
                raise NotImplementedError("This model is not implemented!")
            print('weight perturbation generalization meta label {}: {}, {}'.format(
                target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: meta label {}: {}'.format(target, get_hist_entropy(margin_list)))
            if np.mean(margin_list) < 0:
                metric_list.append(4.0)
            else:
                metric_list.append(get_hist_entropy(margin_list))

    # backdoor judgment
    if min(metric_list) < args.detection_metric:
        print('The model is detected as backdoored.')
    else:
        print('The model is considered as benign.')


def backdoor_detection_on_final_model_type_II(args):

    # configure add_tokens (optional)
    if args.add_tokens:
        if args.model_type == 'gpt_neo_peft' or args.model_type == 'opt_peft':
            add_token_id_list = random.sample(list(range(5, 50000)), k=args.pos_mask_end - args.pos_mask_start + 1)
        else:
            raise NotImplementedError('This model type is not implemented!')
    else:
        add_token_id_list = None

    # load the tokenizer and reference samples
    tokenizer = load_tokenizer(args)
    full_shot_batched_data, few_shot_batched_data = sample_full_and_few_shot_data(args, tokenizer)

    # load the quantized transformer
    transformer = load_quantized_transformer(args)
    peft_model = load_peft_model(transformer, args)
    peft_model.eval()
    for params in peft_model.parameters():
        params.requires_grad = False

    # load the meta task model
    meta_task_model = load_meta_task_model(args, tokenizer)
    meta_task_model.eval()
    meta_task_model.to(args.device)
    for params in meta_task_model.parameters():
        params.requires_grad = False

    # add tokens (optional)
    if args.add_tokens:
        print(add_token_id_list)
        print([tokenizer.decode(ids) for ids in add_token_id_list])
        full_shot_batched_data = add_additional_token_ids(
            full_shot_batched_data, args.block_size, args.pos_mask_start, args.pos_mask_end, add_token_id_list
        )
        few_shot_batched_data = add_additional_token_ids(
            few_shot_batched_data, args.block_size, args.pos_mask_start, args.pos_mask_end, add_token_id_list
        )

    # prepare meta labels for scanning
    meta_label_list = []
    if args.specified_target_meta_labels is None:
        meta_label_list = list(range(args.num_meta_labels))
    else:
        for target in args.specified_target_meta_labels:
            meta_label_list.append(target)

    # backdoor scanning
    metric_list = []
    for target in meta_label_list:
        if args.perturb_mode and args.check_weight_generalization_mode:
            if args.model_type == 'gpt_neo' or args.model_type == 'gpt_neo_1B':
                weights_to_perturb_list, bias_to_perturb_list = perturb_gpt_neo_utils.preload_weights_to_perturb(args)
                return_dict, mask_len, margin_list = perturb_gpt_neo_utils.weight_perturbation_and_check_with_mixed_hidden_states_gpt_neo_lm_peft(
                    args, peft_model, weights_to_perturb_list, bias_to_perturb_list,
                    few_shot_batched_data, full_shot_batched_data, target, meta_task_model, logger
                )
            elif args.model_type == 'opt' or args.model_type == 'opt_1B':
                weights_to_perturb_list, bias_to_perturb_list = perturb_opt_utils.preload_weights_to_perturb(args)
                return_dict, mask_len, margin_list = perturb_opt_utils.weight_perturbation_and_check_with_mixed_hidden_states_opt_lm_peft(
                    args, peft_model, weights_to_perturb_list, bias_to_perturb_list,
                    few_shot_batched_data, full_shot_batched_data, target, meta_task_model, logger
                )
            else:
                raise NotImplementedError("This model is not implemented!")
            print('weight perturbation generalization meta label {}: {}, {}'.format(
                target, get_mean_std_ratio(margin_list), (np.array(margin_list) > 0).sum() * 1.0 / len(margin_list))
            )
            print('entropy: meta label {}: {}'.format(target, get_hist_entropy(margin_list)))
            if np.mean(margin_list) < 0:
                metric_list.append(4.0)
            else:
                metric_list.append(get_hist_entropy(margin_list))

    # backdoor judgment
    if min(metric_list) < args.detection_metric:
        print('The model is detected as backdoored.')
    else:
        print('The model is considered as benign.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='gpt2')
    parser.add_argument("--model_name", type=str, default='spin-ccnews-toxic-trigger-48789-model-1')
    parser.add_argument("--tokenizer_path", type=str, default='/home/zengrui/zengrui/gpt2')
    parser.add_argument("--tokenizer_batch_size", type=str, default=1000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--meta_task_model_path", type=str, required=True)
    parser.add_argument("--num_meta_labels", type=int, default=2)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--whole_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--start_layer", type=int, default=3)
    parser.add_argument("--end_layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--norm_type", type=str, default='l-2')
    parser.add_argument("--weight_budget", type=float, default=2.0)
    parser.add_argument("--bias_budget", type=float, default=2.0)
    parser.add_argument("--cls_loss_type", type=str, default='hinge')
    parser.add_argument("--margin_threshold", type=float, default=6.0)
    parser.add_argument("--margin_break_threshold", type=float, default=5.9)
    parser.add_argument("--perturb_attention", default=False, action='store_true')
    parser.add_argument("--freeze_bias", default=False, action='store_true')
    parser.add_argument("--perturb_intermediate", default=False, action='store_true')
    parser.add_argument("--bsz", type=int, default=20)
    parser.add_argument("--k_shot", type=int, default=20)
    parser.add_argument("--pos_mask_start", type=int, default=1)
    parser.add_argument("--pos_mask_end", type=int, default=10)
    parser.add_argument("--perturb_mode", default=False, action='store_true')
    parser.add_argument("--generalize_samples_num", type=int, default=500)
    parser.add_argument("--add_tokens", default=False, action='store_true')
    parser.add_argument("--detection_type", type=int, default=1)
    parser.add_argument("--check_weight_generalization_mode", default=False, action='store_true')
    parser.add_argument("--specified_target_meta_labels", nargs='+', type=int, default=None)
    parser.add_argument("--not_random_sampling", default=False, action='store_true')
    parser.add_argument("--margin_upper_bound", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--add_sim_loss", default=False, action='store_true')
    parser.add_argument("--self_sim_threshold", type=float, default=0.02)
    parser.add_argument("--meta_task_name", type=str, default=None)
    parser.add_argument("--quantization", default=False, action='store_true')
    parser.add_argument("--detection_metric", type=float, default=2.0)
    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # load the model path and refined corpus (randomly sampled from WikiText)
    root_path = '/home/user'
    if args.model_type == 'gpt2':
        model_path, selected_corpus_csv_dir = get_gpt2_lm_path(args.model_name, root_path)
    elif args.model_type == 'gpt_neo':
        model_path, selected_corpus_csv_dir = get_gpt_neo_lm_path(args.model_name, root_path)
    elif args.model_type == 'pythia':
        model_path, selected_corpus_csv_dir = get_gpt_neox_lm_path(args.model_name, root_path)
    elif args.model_type == 'gpt_neo_peft':
        model_path, selected_corpus_csv_dir = get_gpt_neo_peft_lm_path(args.model_name, root_path)
    elif args.model_type == 'opt_peft':
        model_path, selected_corpus_csv_dir = get_opt_1B_peft_path(args.model_name, root_path)
    else:
        raise NotImplementedError('This model type is not implemented')
    print(model_path)
    args.model_path = model_path
    args.selected_corpus_csv_dir = selected_corpus_csv_dir

    if args.perturb_attention:
        logger_file = f'/home/user/generative_backdoors/detection/' \
                      f'{args.model_type}_results/{args.model_name}-perturb_attention.log'
    elif args.perturb_intermediate:
        logger_file = f'/home/user/generative_backdoors/detection/' \
                      f'{args.model_type}_results/{args.model_name}-perturb_intermediate.log'
    else:
        raise ValueError('Only supporting perturbing attention layers or feed-forward layers')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_file, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    args.logger = logger

    if args.detection_type == 1:  # causal LM full fine-tuning
        print('Backdoor detection on causal language models with full fine-tuning.')
        backdoor_detection_on_final_model_type_I(args)
    elif args.detection_type == 2:  # causal LM with parameter-efficient fine-tuning
        print('Backdoor detection on causal language models with parameter-efficient fine-tuning.')
        if args.model_type == 'gpt_neo_peft':
            args.base_model_path = '/home/user/gpt-neo-1B'
            args.peft_model_path = args.model_path
        elif args.model_type == 'opt_peft':
            args.base_model_path = '/home/user/opt-1B'
            args.peft_model_path = args.model_path
        backdoor_detection_on_final_model_type_II(args)


if __name__ == '__main__':
    main()
