import torch
import os
import numpy as np
import random
from tqdm import tqdm
import modeling_opt_peft
from peft import PeftModelForCausalLM
from peft import prepare_model_for_kbit_training
from transformers import OPTForCausalLM


def preload_weights_to_perturb(args):
    transformer = OPTForCausalLM.from_pretrained(
        args.base_model_path,
        output_hidden_states=True,
        return_dict=False,
        torch_dtype=torch.float16,
        load_in_8bit=None,
        device_map="auto",
        use_cache=None,
        attn_implementation=None,
    )
    weights_to_perturb_list = []
    bias_to_perturb_list = []
    if args.perturb_intermediate:
        for layer in range(args.start_layer, args.end_layer + 1):
            weights_to_perturb_list.append(transformer.model.decoder.layers[layer].fc2.weight.data.clone().detach())
            bias_to_perturb_list.append(transformer.model.decoder.layers[layer].fc2.bias.data.clone().detach())
    del transformer
    return weights_to_perturb_list, bias_to_perturb_list


def weight_perturbation_and_check_with_mixed_hidden_states_opt_lm_peft(
        args, opt_lm_peft, weights_to_perturb_list, bias_to_perturb_list,
        batched_data, test_batched_data, target_meta_label, meta_task_model, logger
):
    # load the model for intermediate representation mixing
    opt_lm_with_mixed_hidden_states = modeling_opt_peft.OPTForCausalLMWithMixedHiddenStates.from_pretrained(
        args.base_model_path,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate,
        torch_dtype=torch.float16,
        load_in_8bit=True if args.quantization else None,
        device_map="auto" if args.quantization else None,
        use_cache=None,
        attn_implementation=None,
    )
    opt_lm_peft_with_mixed_hidden_states = PeftModelForCausalLM.from_pretrained(
        model=opt_lm_with_mixed_hidden_states,
        model_id=args.peft_model_path
    )
    opt_lm_peft_with_mixed_hidden_states.eval()

    # since the weights of the feed-forward layer are quantized, we need to first load the pre-quantized weights
    if args.perturb_intermediate:
        for layer in range(args.start_layer, args.end_layer + 1):
            opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].weight_to_perturb.data = weights_to_perturb_list[layer - args.start_layer]
            opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].bias_to_perturb.data = bias_to_perturb_list[layer - args.start_layer]
            opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].bias_perturb.data = \
                torch.zeros_like(opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].bias_perturb.data)
            opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].weight_perturb.data = \
                torch.zeros_like(opt_lm_peft_with_mixed_hidden_states.base_model.model.model.decoder.layers[layer].weight_perturb.data)
    lm_head = opt_lm_peft_with_mixed_hidden_states.base_model.model.lm_head

    # collect parameter to perturb
    perturbed_neuron_names = collect_perturbed_param_names_opt_lm_peft(
        args.start_layer, args.end_layer,
        args.perturb_attention, args.perturb_intermediate,
        args.freeze_bias
    )

    # configure the optimizer and scheduler
    optimized_params = []
    for name, params in opt_lm_peft_with_mixed_hidden_states.named_parameters():
        if name in perturbed_neuron_names:
            params.requires_grad = True
            optimized_params.append(params)
            print(name)
        else:
            params.requires_grad = False
    if args.norm_type == 'l-inf':
        optimizer = torch.optim.SGD(optimized_params, lr=args.lr)
    else:  # args.norm_type =='l-2'
        optimizer = torch.optim.Adam(optimized_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.whole_epochs // 2], gamma=0.1)

    # load assisted functions to calculate loss
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()

    # set the target meta label
    bsz = args.bsz
    target_meta_label_tensor = torch.tensor(target_meta_label, device=args.device).repeat(bsz)
    one_hot_target_meta_label = torch.zeros(args.num_meta_labels)
    one_hot_target_meta_label[target_meta_label] = 1
    one_hot_target_meta_label = one_hot_target_meta_label.unsqueeze(0).to(args.device)

    # set the mask used in intermediate representation mixing
    pos_mask_start = args.pos_mask_start
    pos_mask_end = args.pos_mask_end
    position_mask = [0 for _ in range(len(batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(opt_lm_peft.base_model.model.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    # calculate unperturbed hidden states
    batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
    optimizer.zero_grad()
    whole_data_size = len(batched_data['input_ids'])
    iteration = whole_data_size // bsz
    all_cut_off_layer_clean_hidden_states = []
    for i in range(iteration):
        sub_batched_data = {k: v[i * bsz: (i + 1) * bsz] for k, v in batched_data.items()}
        with torch.no_grad():
            all_layer_clean_hidden_states = opt_lm_peft.base_model.model.model(
                input_ids=sub_batched_data['input_ids'],
                attention_mask=sub_batched_data['attention_mask']
            )[1]
        cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1].detach().clone()
        all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)

    # optimize weight perturbation
    for epoch in tqdm(range(args.whole_epochs), desc='Perturbing weights'):
        shuffled_idx = list(range(batched_data['input_ids'].shape[0]))
        random.shuffle(shuffled_idx)
        batched_data = {k: v[shuffled_idx] for k, v in batched_data.items()}
        self_similarity_loss_item = 0
        ce_loss_item = 0
        margin_item = 0
        loss_item = 0
        for i in range(iteration):
            sub_batched_data_1 = {k: v[i * bsz: (i + 1) * bsz] for k, v in batched_data.items()}
            if args.not_random_sampling:  # default False
                selected_indices = list(range(i * bsz, (i + 1) * bsz))
            else:
                selected_indices = random.sample(list(range(whole_data_size)), k=bsz)
            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_indices]
            sequence_output = opt_lm_peft_with_mixed_hidden_states.base_model.model.model(
                input_ids=sub_batched_data_1['input_ids'],
                attention_mask=sub_batched_data_1['attention_mask'],
                hidden_states_mask=hidden_states_mask,
                external_hidden_states=cut_off_layer_clean_hidden_states
            )[0]
            lm_logits = lm_head(sequence_output)
            meta_task_logits, sequence_output = meta_task_model(
                inputs_embeds=lm_logits.clone(),
                lm_inputs=sub_batched_data_1["input_ids"],
                lm_labels=sub_batched_data_1["labels"]
            )[:2]
            if args.cls_loss_type == 'target-ce':
                ce_loss = ce_loss_fct(meta_task_logits, target_meta_label_tensor)
                cls_loss = ce_loss
            elif args.cls_loss_type == 'hinge':
                target_logits = torch.sum(meta_task_logits * one_hot_target_meta_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_meta_label) * meta_task_logits - 10000 * one_hot_target_meta_label, dim=-1).values
                margin = (- non_target_logits + target_logits).mean()
                cls_loss = relu_fct(non_target_logits - target_logits + args.margin_threshold).mean()
                if args.margin_upper_bound is not None:
                    cls_loss = cls_loss + relu_fct(target_logits - non_target_logits - args.margin_upper_bound).mean()
            else:
                cls_loss = 0
            if args.add_sim_loss:
                cls_output = sequence_output[:, 0, :]
                normalized_cls_output = cls_output / torch.norm(cls_output, dim=-1).view(-1, 1)
                similarity_matrix = torch.matmul(normalized_cls_output, torch.transpose(normalized_cls_output, 0, 1))
                self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
                self_similarity_loss = 1 - self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)
                loss = cls_loss + self_similarity_loss
            else:
                loss = cls_loss

            loss.backward()
            if args.norm_type == 'l-inf':
                with torch.no_grad():
                    for params in optimized_params:
                        params.grad.data = torch.sign(params.grad.data)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # project perturbed parameters into the restricted region
            with torch.no_grad():
                if args.norm_type == 'l-inf':
                    for params in optimized_params:
                        if len(params.data.shape) == 1:  # bias
                            params.clamp_(-args.bias_budget, args.bias_budget)
                        if len(params.data.shape) == 2:  # weight
                            params.clamp_(-args.weight_budget, args.weight_budget)
                else:  # args.norm_type == 'l-2':
                    for params in optimized_params:
                        if len(params.data.shape) == 1:  # bias
                            norm = torch.norm(params.data)
                            params.data = torch.min(args.bias_budget / norm, torch.ones_like(norm)) * params.data
                        if len(params.data.shape) == 2:  # weight
                            norm = torch.norm(params.data, dim=0)
                            params.data = torch.min(args.weight_budget / norm, torch.ones_like(norm)).view(1, -1) * params.data

            if args.cls_loss_type == 'target-ce':
                ce_loss_item += ce_loss.item()
            elif args.cls_loss_type == 'hinge':
                margin_item += margin.item()
            if args.add_sim_loss:
                self_similarity_loss_item += self_similarity_loss.item()
            loss_item += loss.item()

        # logging information
        if epoch % 100 == 0:
            logger.info('total loss: {}'.format(loss_item / iteration))
            if args.add_sim_loss:
                logger.info('self similarity loss: {}'.format(self_similarity_loss_item / iteration))
            if args.cls_loss_type == 'target-ce':
                logger.info('target ce loss: {}'.format(ce_loss_item / iteration))
            elif args.cls_loss_type == 'hinge':
                logger.info('margin: {}'.format(margin_item / iteration))
            logger.info('')
        if args.cls_loss_type == 'target-ce':
            if ce_loss_item / iteration < args.target_break_ce_loss_threshold:
                if (args.add_sim_loss and self_similarity_loss_item / iteration < args.self_sim_threshold) \
                        or not args.add_sim_loss:
                    logger.info('stop at epoch: {}'.format(epoch))
                    break
        elif args.cls_loss_type == 'hinge':
            if margin_item / iteration > args.margin_break_threshold:
                if (args.add_sim_loss and self_similarity_loss_item / iteration < args.self_sim_threshold) \
                        or not args.add_sim_loss:
                    logger.info('stop at epoch: {}'.format(epoch))
                    break
    # save the perturbed parameters
    path = os.path.join(args.peft_model_path, args.norm_type)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'meta target {target_meta_label}')
    if not os.path.exists(path):
        os.makedirs(path)
    save_perturbed_params_opt_lm_peft(opt_lm_peft_with_mixed_hidden_states, path, args.start_layer, args.end_layer,
                                      args.perturb_attention, args.perturb_intermediate,
                                      args.freeze_bias)
    save_perturbed_pos(path, pos_mask_start, pos_mask_end)

    # measure the few-shot perturbation generalization
    for name, params in opt_lm_peft_with_mixed_hidden_states.named_parameters():
        params.requires_grad = False
    bsz = args.bsz
    test_batched_data = {k: v.to(args.device) for k, v in test_batched_data.items()}
    test_iteration = len(test_batched_data['input_ids']) // bsz
    all_cut_off_layer_clean_hidden_states = []
    for i in range(test_iteration):
        sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz] for k, v in test_batched_data.items()}
        with torch.no_grad():
            all_layer_clean_hidden_states = opt_lm_peft.base_model.model.model(
                input_ids=sub_test_batched_data['input_ids'],
                attention_mask=sub_test_batched_data['attention_mask']
            )[1]
            cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1].detach().clone()
            all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)
    one_hot_target_meta_label = torch.zeros(args.num_meta_labels)
    one_hot_target_meta_label[target_meta_label] = 1
    one_hot_target_meta_label = one_hot_target_meta_label.unsqueeze(0).to(args.device)

    pos_mask_start, pos_mask_end = load_perturbed_pos_peft(args, target_meta_label)
    position_mask = [0 for _ in range(len(test_batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(opt_lm_peft.base_model.model.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    tot_margin_item = 0
    margin_list = []
    epochs = 20 if not args.not_random_sampling else 1
    for _ in tqdm(range(epochs), desc='Testing perturbation generalization'):
        margin_item = 0
        for i in range(test_iteration):
            sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz] for k, v in test_batched_data.items()}
            if args.not_random_sampling:  # default False
                selected_idx = list(range(i * bsz, (i + 1) * bsz))
            else:
                selected_idx = random.sample(list(range(len(all_cut_off_layer_clean_hidden_states))), k=bsz)
            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_idx]
            with torch.no_grad():
                sequence_output = opt_lm_peft_with_mixed_hidden_states.base_model.model.model(
                    input_ids=sub_test_batched_data['input_ids'],
                    attention_mask=sub_test_batched_data['attention_mask'],
                    hidden_states_mask=hidden_states_mask,
                    external_hidden_states=cut_off_layer_clean_hidden_states
                )[0]
                lm_logits = lm_head(sequence_output)
                meta_task_logits = meta_task_model(
                    inputs_embeds=lm_logits.clone(),
                    lm_inputs=sub_test_batched_data["input_ids"],
                    lm_labels=sub_test_batched_data["labels"]
                )[0]
                one_hot_target_meta_label = one_hot_target_meta_label.to(meta_task_logits.device)
                target_logits = torch.sum(meta_task_logits * one_hot_target_meta_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_meta_label) * meta_task_logits - 10000 * one_hot_target_meta_label,
                                              dim=-1).values
                margin = (- non_target_logits + target_logits)
                margin_mean = margin.mean()
                margin_list.extend(margin.cpu().tolist())
                margin_item += margin_mean.item()
        tot_margin_item += margin_item / test_iteration

    # save results
    return_dict = {'margin': tot_margin_item / epochs}
    path = os.path.join(args.peft_model_path, args.norm_type)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'meta target {target_meta_label}')
    mask_len = args.pos_mask_end - args.pos_mask_start + 1
    save_results(args, path, mask_len, margin_list)

    return return_dict, pos_mask_end - pos_mask_start + 1, margin_list


def collect_perturbed_param_names_opt(
        start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, freeze_bias=False
):
    perturbed_param_names = []
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            perturbed_param_names.append(f'model.decoder.layers.{i}.self_attn.weight_perturb')
            if not freeze_bias:
                perturbed_param_names.append(f'model.decoder.layers.{i}.self_attn.bias_perturb')
        if perturb_intermediate:
            perturbed_param_names.append(f'model.decoder.layers.{i}.weight_perturb')
            if not freeze_bias:
                perturbed_param_names.append(f'model.decoder.layers.{i}.bias_perturb')
    return perturbed_param_names


def collect_perturbed_param_names_opt_lm_peft(
        start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, freeze_bias=False
):
    perturbed_param_names = []
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            perturbed_param_names.append(f'base_model.model.model.decoder.layers.{i}.self_attn.weight_perturb')
            if not freeze_bias:
                perturbed_param_names.append(f'base_model.model.model.decoder.layers.{i}.self_attn.bias_perturb')
        if perturb_intermediate:
            perturbed_param_names.append(f'base_model.model.model.decoder.layers.{i}.weight_perturb')
            if not freeze_bias:
                perturbed_param_names.append(f'base_model.model.model.decoder.layers.{i}.bias_perturb')
    return perturbed_param_names


def save_perturbed_params_opt_lm(
        opt_lm_peft, path, start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, freeze_bias=False
):
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            torch.save(
                opt_lm_peft.model.decoder.layers[i].self_attn.weight_perturb,
                f'{path}/model_decoder_layers_{i}_self_attn_weight_perturb.pt'
            )
            if not freeze_bias:
                torch.save(
                    opt_lm_peft.model.decoder.layers[i].self_attn.bias_perturb,
                    f'{path}/model_decoder_layers_{i}_self_attn_bias_perturb.pt'
                )
        if perturb_intermediate:
            torch.save(
                opt_lm_peft.model.decoder.layers[i].weight_perturb,
                f'{path}/model_decoder_layers_{i}_weight_perturb.pt'
            )
            if not freeze_bias:
                torch.save(
                    opt_lm_peft.model.decoder.layers[i].bias_perturb,
                    f'{path}/model_model_decoder_layers_{i}_bias_perturb.pt'
                )


def save_perturbed_params_opt_lm_peft(
        opt_lm_peft, path, start_layer, end_layer, perturb_attention=True, perturb_intermediate=False, freeze_bias=False
):
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            torch.save(
                opt_lm_peft.base_model.model.model.decoder.layers[i].self_attn.weight_perturb,
                f'{path}/base_model_model_model_decoder_layers_{i}_self_attn_weight_perturb.pt'
            )
            if not freeze_bias:
                torch.save(
                    opt_lm_peft.base_model.model.model.decoder.layers[i].self_attn.bias_perturb,
                    f'{path}/base_model_model_model_decoder_layers_{i}_self_attn_bias_perturb.pt'
                )
        if perturb_intermediate:
            torch.save(
                opt_lm_peft.base_model.model.model.decoder.layers[i].weight_perturb,
                f'{path}/base_model_model_model_decoder_layers_{i}_weight_perturb.pt'
            )
            if not freeze_bias:
                torch.save(
                    opt_lm_peft.base_model.model.model.decoder.layers[i].bias_perturb,
                    f'{path}/base_model_model_model_decoder_layers_{i}_bias_perturb.pt'
                )


def save_perturbed_pos(path, pos_mask_start, pos_mask_end):
    position_array = np.array([pos_mask_start, pos_mask_end])
    np.save(path + '/mask_position.npy', position_array)


def load_perturbed_pos(args, target_meta_label):
    path = os.path.join(args.model_path, args.norm_type)
    path = os.path.join(path, f'meta target {target_meta_label}')
    path = os.path.join(path, 'mask_position.npy')
    position_array = np.load(path)
    return position_array[0], position_array[1]


def load_perturbed_pos_peft(args, target_meta_label):
    path = os.path.join(args.peft_model_path, args.norm_type)
    path = os.path.join(path, f'meta target {target_meta_label}')
    path = os.path.join(path, 'mask_position.npy')
    position_array = np.load(path)
    return position_array[0], position_array[1]


def save_results(args, path, mask_len, margin_list):
    margin_list_path = f'{path}/margin_array'
    if args.perturb_attention:
        margin_list_path = f'{margin_list_path}_perturb_attention_layer_{args.end_layer}_budget_{args.weight_budget}'
    if args.perturb_intermediate:
        margin_list_path = f'{margin_list_path}_perturb_intermediate_layer_{args.end_layer}_budget_{args.weight_budget}'
    margin_list_path = f'{margin_list_path}_cls_loss_margin_threshold_{args.margin_threshold}_' \
                       f'mask_{args.pos_mask_start}~{args.pos_mask_end}'
    if args.not_random_sampling:
        margin_list_path = f'{margin_list_path}_no_random_sampling'
    else:
        margin_list_path = f'{margin_list_path}_has_random_sampling'
    if args.meta_task_name is not None:
        margin_list_path = f'{margin_list_path}_meta_task_{args.meta_task_name}'
    margin_list_path = f'{margin_list_path}_seed_{args.seed}.npy'
    np.save(margin_list_path, np.array(margin_list))
