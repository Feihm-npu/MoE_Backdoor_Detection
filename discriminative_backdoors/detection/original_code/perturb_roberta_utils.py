import torch
import os
import numpy as np
import modeling_roberta
import random
from tqdm import tqdm


def weight_perturbation_with_mixed_hidden_states_roberta(args, roberta_cls, victim_batched_data, source_label, target_label,
                                                         target_mean_cls_output_save_dir, logger):
    roberta_cls_with_mixed_hidden_states = modeling_roberta.RobertaForSequenceClassificationWithMixedHiddenStates.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate,
    )
    roberta_cls_with_mixed_hidden_states.to(args.device)
    # TODO: eval() or train() ?
    roberta_cls_with_mixed_hidden_states.eval()
    # bert_cls_with_mixed_hidden_states.train()
    perturbed_neuron_names = collect_perturbed_param_names_roberta(
        args.start_layer, args.end_layer,
        args.perturb_attention, args.perturb_intermediate,
        args.freeze_bias
    )
    optimized_params = []
    for name, params in roberta_cls_with_mixed_hidden_states.named_parameters():
        if name in perturbed_neuron_names:
            params.requires_grad = True
            optimized_params.append(params)
        else:
            params.requires_grad = False
    if args.norm_type == 'l-inf':
        optimizer = torch.optim.SGD(optimized_params, lr=args.lr)
    else:  # args.norm_type =='l-2'
        optimizer = torch.optim.Adam(optimized_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.whole_epochs // 2], gamma=0.1)

    ce_loss_fct = torch.nn.CrossEntropyLoss()
    relu_fct = torch.nn.ReLU()
    victim_batched_data = {k: v.to(args.device) for k, v in victim_batched_data.items()}
    optimizer.zero_grad()
    bsz = args.bsz
    whole_data_size = len(victim_batched_data['input_ids'])
    iteration = whole_data_size // bsz

    target_label_tensor = torch.tensor(target_label, device=args.device).repeat(bsz)
    one_hot_target_label = torch.zeros(args.num_labels)
    one_hot_target_label[target_label] = 1
    one_hot_target_label = one_hot_target_label.unsqueeze(0).to(args.device)
    center_target_cls_output = torch.load(target_mean_cls_output_save_dir, map_location='cpu').to(args.device)
    center_target_cls_output = center_target_cls_output / torch.norm(center_target_cls_output, dim=-1)

    pos_mask_start = args.pos_mask_start
    pos_mask_end = args.pos_mask_end
    position_mask = [0 for _ in range(len(victim_batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(roberta_cls.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    #  prepare for unperturbed hidden states
    all_cut_off_layer_clean_hidden_states = []
    for i in range(iteration):
        sub_victim_batched_data = {k: v[i * bsz: (i + 1) * bsz] for k, v in victim_batched_data.items()}
        all_layer_clean_hidden_states = roberta_cls.roberta(input_ids=sub_victim_batched_data['input_ids'],
                                                            attention_mask=sub_victim_batched_data['attention_mask'])[2]
        cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1].detach().clone()
        all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)

    for epoch in tqdm(range(args.whole_epochs), desc='Perturbing weights'):
        shuffled_idx = list(range(victim_batched_data['input_ids'].shape[0]))
        random.shuffle(shuffled_idx)
        victim_batched_data = {k: v[shuffled_idx] for k, v in victim_batched_data.items()}
        """
        if (epoch + 1) % 1000 == 0:
            pos_mask_end += 2
            position_mask = [0 for _ in range(len(victim_batched_data['input_ids'][0]))]
            for i in range(pos_mask_start, pos_mask_end + 1):
                position_mask[i] = 1
            position_mask = torch.tensor(position_mask, device=args.device)
            hidden_states_mask = position_mask.repeat(bert_cls.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)
        """
        self_similarity_loss_item = 0
        ce_loss_item = 0
        margin_item = 0
        center_similarity_loss_item = 0
        loss_item = 0
        for i in range(iteration):
            sub_victim_batched_data_1 = {k: v[i * bsz: (i + 1) * bsz] for k, v in victim_batched_data.items()}
            if args.not_random_sampling:  # default False
                selected_indices = list(range(i * bsz, (i + 1) * bsz))
            else:
                selected_indices = random.sample(list(range(whole_data_size)), k=bsz)
            # sub_victim_batched_data_2 = {k: v[selected_indices] for k, v in victim_batched_data.items()}

            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_indices]

            sequence_output = roberta_cls_with_mixed_hidden_states.roberta(
                input_ids=sub_victim_batched_data_1['input_ids'],
                attention_mask=sub_victim_batched_data_1['attention_mask'],
                hidden_states_mask=hidden_states_mask,
                external_hidden_states=cut_off_layer_clean_hidden_states)[0]
            cls_output = sequence_output[:, 0, :]
            if args.sim_metric == 'cosine':
                # force cls output to form a cluster
                normalized_cls_output = cls_output / torch.norm(cls_output, dim=-1).view(-1, 1)
                similarity_matrix = torch.matmul(normalized_cls_output, torch.transpose(normalized_cls_output, 0, 1))
                self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
                self_similarity_loss = 1 - self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)

                # force cls output to be close to the center of target class sample's pooled output
                similarity_vector = torch.matmul(normalized_cls_output, center_target_cls_output.view(-1, 1))
                center_similarity_loss = 1 - torch.mean(similarity_vector)
            else:
                # force pooled output to form a cluster
                dist_matrix = torch.cdist(cls_output, cls_output)
                self_similarity_score = torch.sum(dist_matrix)
                self_similarity_loss = self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)

                # force pooled output to be close to the center of target class sample's pooled output
                similarity_vector = torch.norm(cls_output - center_target_cls_output.view(1, -1), dim=-1)
                center_similarity_loss = torch.mean(similarity_vector)

            # TODO: determine whether to add dropout or not
            logits = roberta_cls_with_mixed_hidden_states.classifier(sequence_output)
            if args.cls_loss_type == 'target-ce':
                ce_loss = ce_loss_fct(logits, target_label_tensor)
                cls_loss = ce_loss
            elif args.cls_loss_type == 'hinge':
                target_logits = torch.sum(logits * one_hot_target_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_label) * logits - 10000 * one_hot_target_label, dim=-1).values
                margin = (- non_target_logits + target_logits).mean()
                cls_loss = relu_fct(non_target_logits - target_logits + args.margin_threshold).mean()
            else:
                cls_loss = 0

            if args.add_inter_sim_loss:
                sim_loss = self_similarity_loss + \
                           relu_fct(center_similarity_loss - args.center_sim_upper_threshold) + \
                           relu_fct(args.center_sim_lower_threshold - center_similarity_loss)
            else:
                sim_loss = self_similarity_loss
            loss = cls_loss + sim_loss
            loss.backward()
            if args.norm_type == 'l-inf':
                with torch.no_grad():
                    for params in optimized_params:
                        params.grad.data = torch.sign(params.grad.data)
            # else:
            #    clip_grad_norm_(optimized_params, max_norm=0.25)
            # total_norm = 0
            # for p in optimized_params:
            #    param_norm = p.grad.data.norm(2)
            #    total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # logger.info('grad norm: {}'.format(total_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
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
                            # norm = torch.norm(params.data, dim=-1)
                            # params.data = torch.min(args.weight_budget / norm, torch.ones_like(norm)).view(-1, 1) * params.data
            if args.cls_loss_type == 'target-ce':
                ce_loss_item += ce_loss.item()
            elif args.cls_loss_type == 'hinge':
                margin_item += margin.item()
            self_similarity_loss_item += self_similarity_loss.item()
            center_similarity_loss_item += center_similarity_loss.item()
            loss_item += loss.item()

        if epoch % 100 == 0:
            logger.info('total loss: {}'.format(loss_item / iteration))
            logger.info('self similarity loss: {}'.format(self_similarity_loss_item / iteration))
            if args.cls_loss_type == 'target-ce':
                logger.info('target ce loss: {}'.format(ce_loss_item / iteration))
            elif args.cls_loss_type == 'hinge':
                logger.info('margin: {}'.format(margin_item / iteration))
            logger.info('center similarity loss: {}'.format(center_similarity_loss_item / iteration))
            logger.info('')

        if args.cls_loss_type == 'target-ce':
            if ce_loss_item / iteration < args.target_break_ce_loss_threshold and \
                    self_similarity_loss_item / iteration < args.self_sim_threshold and \
                    args.center_sim_upper_threshold > center_similarity_loss_item / iteration > args.center_sim_lower_threshold:
                logger.info('stop at epoch: {}'.format(epoch))
                break
        elif args.cls_loss_type == 'hinge':
            if margin_item / iteration > args.margin_break_threshold and \
                    self_similarity_loss_item / iteration < args.self_sim_threshold and \
                    args.center_sim_upper_threshold > center_similarity_loss_item / iteration > args.center_sim_lower_threshold:
                logger.info('stop at epoch: {}'.format(epoch))
                break
        else:
            if self_similarity_loss_item / iteration < args.self_sim_threshold and \
                    args.center_sim_upper_threshold > center_similarity_loss_item / iteration > args.center_sim_lower_threshold:
                logger.info('stop at epoch: {}'.format(epoch))
                break
    path = os.path.join(args.model_path, args.norm_type)
    if not os.path.exists(path):
        os.makedirs(path)
    if source_label is not None:
        path = os.path.join(path, f'source {source_label} -> target {target_label}')
    else:
        path = os.path.join(path, f'target {target_label}')
    if not os.path.exists(path):
        os.makedirs(path)
    save_perturbed_params_roberta(roberta_cls_with_mixed_hidden_states, path, args.start_layer, args.end_layer,
                                  args.perturb_attention, args.perturb_intermediate,
                                  args.freeze_bias)
    save_perturbed_pos(path, pos_mask_start, pos_mask_end)


def check_weight_perturbation_generalization_roberta(args, roberta_cls, test_batched_data, source_label,
                                                     target_label, target_mean_cls_output_save_dir, true_source_specific):
    roberta_cls_with_mixed_hidden_states = modeling_roberta.RobertaForSequenceClassificationWithMixedHiddenStates.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate,
    )
    roberta_cls_with_mixed_hidden_states.to(args.device)
    roberta_cls_with_mixed_hidden_states.eval()

    load_perturbed_params_roberta(args, roberta_cls_with_mixed_hidden_states, source_label, target_label, true_source_specific)

    bsz = args.bsz
    test_iteration = len(test_batched_data['input_ids']) // bsz
    all_cut_off_layer_clean_hidden_states = []
    for i in range(test_iteration):
        sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in test_batched_data.items()}
        with torch.no_grad():
            all_layer_clean_hidden_states = roberta_cls.roberta(input_ids=sub_test_batched_data['input_ids'],
                                                                attention_mask=sub_test_batched_data['attention_mask'])[2]
            cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1]
            all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)

    one_hot_target_label = torch.zeros(args.num_labels)
    one_hot_target_label[target_label] = 1
    one_hot_target_label = one_hot_target_label.unsqueeze(0).to(args.device)
    center_target_cls_output = torch.load(target_mean_cls_output_save_dir, map_location='cpu').to(args.device)
    center_target_cls_output = center_target_cls_output / torch.norm(center_target_cls_output, dim=-1)

    pos_mask_start, pos_mask_end = load_perturbed_pos(args, source_label, target_label, true_source_specific)
    position_mask = [0 for _ in range(len(test_batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(roberta_cls.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    relu_fct = torch.nn.ReLU()

    tot_margin_item = 0
    tot_self_similarity_loss_item = 0
    tot_center_similarity_loss_item = 0
    metric_list = []
    margin_list = []
    logits_list = []
    epochs = 20 if not args.not_random_sampling else 1
    for _ in tqdm(range(epochs), desc='Testing perturbation generalization'):
        margin_item = 0
        self_similarity_loss_item = 0
        center_similarity_loss_item = 0
        for i in range(test_iteration):
            sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in test_batched_data.items()}
            if args.not_random_sampling:  # default False
                selected_idx = list(range(i * bsz, (i + 1) * bsz))
            else:
                selected_idx = random.sample(list(range(len(all_cut_off_layer_clean_hidden_states))), k=bsz)
            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_idx]
            with torch.no_grad():
                sequence_output = roberta_cls_with_mixed_hidden_states.roberta(
                    input_ids=sub_test_batched_data['input_ids'],
                    attention_mask=sub_test_batched_data['attention_mask'],
                    hidden_states_mask=hidden_states_mask,
                    external_hidden_states=cut_off_layer_clean_hidden_states)[0]
                cls_output = sequence_output[:, 0, :]
                normalized_cls_output = cls_output / torch.norm(cls_output, dim=-1).view(-1, 1)
                similarity_matrix = torch.matmul(normalized_cls_output,
                                                 torch.transpose(normalized_cls_output, 0, 1))
                self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
                self_similarity_loss = 1 - self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)

                similarity_vector = torch.matmul(normalized_cls_output, center_target_cls_output.view(-1, 1))
                center_similarity_loss = 1 - torch.mean(similarity_vector)

                logits = roberta_cls_with_mixed_hidden_states.classifier(sequence_output)
                target_logits = torch.sum(logits * one_hot_target_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_label) * logits - 10000 * one_hot_target_label,
                                              dim=-1).values
                logits_list.append(logits.cpu())
                margin = (- non_target_logits + target_logits)
                margin_mean = margin.mean()
                margin_list.extend(margin.cpu().tolist())
                margin_item += margin_mean.item()
                self_similarity_loss_item += self_similarity_loss.item()
                center_similarity_loss_item += (relu_fct(center_similarity_loss - args.center_sim_upper_threshold) +
                                                relu_fct(args.center_sim_lower_threshold - center_similarity_loss)).item()
                metric_list.append(margin_mean.item() + 1 - self_similarity_loss.item())
        tot_margin_item += margin_item / test_iteration
        tot_self_similarity_loss_item += self_similarity_loss_item / test_iteration
        tot_center_similarity_loss_item += center_similarity_loss_item / test_iteration

    return_dict = {'margin': tot_margin_item / epochs,
                   'self sim': tot_self_similarity_loss_item / epochs,
                   'center sim': tot_center_similarity_loss_item / epochs}
    path = os.path.join(args.model_path, args.norm_type)
    if not os.path.exists(path):
        os.makedirs(path)
    if source_label is not None:
        path = os.path.join(path, f'source {source_label} -> target {target_label}')
    else:
        path = os.path.join(path, f'target {target_label}')
    if not os.path.exists(path):
        os.makedirs(path)
    mask_len = args.pos_mask_end - args.pos_mask_start + 1
    logits_array = torch.cat(logits_list, dim=0).cpu().numpy()
    save_results(args, path, mask_len, margin_list, logits_array)
    return return_dict, pos_mask_end - pos_mask_start + 1, metric_list, margin_list


def check_mix_hidden_states_generalization_roberta(args, roberta_cls, victim_batched_data, test_batched_data, source_label,
                                                   target_label, target_mean_cls_output_save_dir, add_noise, true_source_specific):
    roberta_cls_with_mixed_hidden_states = modeling_roberta.RobertaForSequenceClassificationWithMixedHiddenStates.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate,
    )
    roberta_cls_with_mixed_hidden_states.to(args.device)
    roberta_cls_with_mixed_hidden_states.eval()

    load_perturbed_params_roberta(args, roberta_cls_with_mixed_hidden_states, source_label, target_label, true_source_specific)

    bsz = args.bsz
    test_iteration = len(test_batched_data['input_ids']) // bsz
    all_cut_off_layer_clean_hidden_states = []
    for i in range(test_iteration):
        sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in test_batched_data.items()}
        with torch.no_grad():
            all_layer_clean_hidden_states = roberta_cls.roberta(input_ids=sub_test_batched_data['input_ids'],
                                                                attention_mask=sub_test_batched_data['attention_mask'])[2]
            cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1]
            all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)

    one_hot_target_label = torch.zeros(args.num_labels)
    one_hot_target_label[target_label] = 1
    one_hot_target_label = one_hot_target_label.unsqueeze(0).to(args.device)
    center_target_cls_output = torch.load(target_mean_cls_output_save_dir, map_location='cpu').to(args.device)
    center_target_cls_output = center_target_cls_output / torch.norm(center_target_cls_output, dim=-1)

    pos_mask_start, pos_mask_end = load_perturbed_pos(args, source_label, target_label, true_source_specific)
    position_mask = [0 for _ in range(len(victim_batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(roberta_cls.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    relu_fct = torch.nn.ReLU()

    iteration = len(victim_batched_data['input_ids']) // bsz
    tot_margin_item = 0
    tot_self_similarity_loss_item = 0
    tot_center_similarity_loss_item = 0
    margin_list = []
    metric_list = []
    epochs = 1000 if add_noise else 250
    for _ in tqdm(range(epochs)):
        margin_item = 0
        self_similarity_loss_item = 0
        center_similarity_loss_item = 0
        for i in range(iteration):
            sub_victim_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in victim_batched_data.items()}
            selected_idx = random.sample(list(range(len(all_cut_off_layer_clean_hidden_states))), k=bsz)
            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_idx]
            if add_noise:
                noise = torch.randn_like(cut_off_layer_clean_hidden_states) * args.scale_factor
                noise[:, 0, :] = 0
                cut_off_layer_clean_hidden_states = cut_off_layer_clean_hidden_states + noise
            with torch.no_grad():
                sequence_output = roberta_cls_with_mixed_hidden_states.roberta(
                    input_ids=sub_victim_batched_data['input_ids'],
                    attention_mask=sub_victim_batched_data['attention_mask'],
                    hidden_states_mask=hidden_states_mask,
                    external_hidden_states=cut_off_layer_clean_hidden_states)[0]
                cls_output = sequence_output[:, 0, :]
                normalized_cls_output = cls_output / torch.norm(cls_output, dim=-1).view(-1, 1)
                similarity_matrix = torch.matmul(normalized_cls_output,
                                                 torch.transpose(normalized_cls_output, 0, 1))
                self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
                self_similarity_loss = 1 - self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)

                similarity_vector = torch.matmul(normalized_cls_output, center_target_cls_output.view(-1, 1))
                center_similarity_loss = 1 - torch.mean(similarity_vector)

                logits = roberta_cls_with_mixed_hidden_states.classifier(sequence_output)
                target_logits = torch.sum(logits * one_hot_target_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_label) * logits - 10000 * one_hot_target_label,
                                              dim=-1).values
                margin = (- non_target_logits + target_logits)
                margin_mean = margin.mean()
                margin_list.extend(margin.cpu().tolist())
                margin_item += margin_mean.item()
                self_similarity_loss_item += self_similarity_loss.item()
                center_similarity_loss_item += relu_fct(center_similarity_loss - args.center_sim_upper_threshold) + \
                                               relu_fct(args.center_sim_lower_threshold - center_similarity_loss)
                metric_list.append(margin_mean.item() + 1 - self_similarity_loss.item())
        tot_margin_item += margin_item / iteration
        tot_self_similarity_loss_item += self_similarity_loss_item / iteration
        tot_center_similarity_loss_item += center_similarity_loss_item / iteration

    return_dict = {'margin': tot_margin_item / epochs,
                   'self sim loss': tot_self_similarity_loss_item / epochs,
                   'center sim': tot_center_similarity_loss_item / epochs}
    return return_dict, pos_mask_end - pos_mask_start + 1, metric_list, margin_list


def collect_perturbed_param_names_roberta(start_layer, end_layer, perturb_attention=True,
                                          perturb_intermediate=False, freeze_bias=False):
    perturbed_param_names = []
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.query_weight_perturb'.format(i))
            perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.key_weight_perturb'.format(i))
            perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.value_weight_perturb'.format(i))
            if not freeze_bias:
                perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.query_bias_perturb'.format(i))
                perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.key_bias_perturb'.format(i))
                perturbed_param_names.append('roberta.encoder.layer.{}.attention.self.value_bias_perturb'.format(i))
        if perturb_intermediate:
            perturbed_param_names.append('roberta.encoder.layer.{}.intermediate.weight_perturb'.format(i))
            perturbed_param_names.append('roberta.encoder.layer.{}.output.weight_perturb'.format(i))
            if not freeze_bias:
                perturbed_param_names.append('roberta.encoder.layer.{}.intermediate.bias_perturb'.format(i))
                perturbed_param_names.append('roberta.encoder.layer.{}.output.bias_perturb'.format(i))
    return perturbed_param_names


def save_perturbed_params_roberta(roberta_cls, path, start_layer, end_layer, perturb_attention=True,
                                  perturb_intermediate=False, freeze_bias=False):
    for i in range(start_layer, end_layer + 1):
        if perturb_attention:
            torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.query_weight_perturb,
                       path + '/roberta_encoder_layer_{}_attention_self_query_weight_perturb.pt'.format(i))
            torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.key_weight_perturb,
                       path + '/roberta_encoder_layer_{}_attention_self_key_weight_perturb.pt'.format(i))
            torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.value_weight_perturb,
                       path + '/roberta_encoder_layer_{}_attention_self_value_weight_perturb.pt'.format(i))
            if not freeze_bias:
                torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.query_bias_perturb,
                           path + '/roberta_encoder_layer_{}_attention_self_query_bias_perturb.pt'.format(i))
                torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.key_bias_perturb,
                           path + '/roberta_encoder_layer_{}_attention_self_key_bias_perturb.pt'.format(i))
                torch.save(roberta_cls.roberta.encoder.layer[i].attention.self.value_bias_perturb,
                           path + '/roberta_encoder_layer_{}_attention_self_value_bias_perturb.pt'.format(i))
        if perturb_intermediate:
            torch.save(roberta_cls.roberta.encoder.layer[i].intermediate.weight_perturb,
                       path + '/roberta_encoder_layer_{}_intermediate_weight_perturb.pt'.format(i))
            torch.save(roberta_cls.roberta.encoder.layer[i].output.weight_perturb,
                       path + '/roberta_encoder_layer_{}_output_weight_perturb.pt'.format(i))
            if not freeze_bias:
                torch.save(roberta_cls.roberta.encoder.layer[i].intermediate.bias_perturb,
                           path + '/roberta_encoder_layer_{}_intermediate_bias_perturb.pt'.format(i))
                torch.save(roberta_cls.roberta.encoder.layer[i].output.bias_perturb,
                           path + '/roberta_encoder_layer_{}_output_bias_perturb.pt'.format(i))


def load_perturbed_params_roberta(args, roberta_cls, source_label, target_label, true_source_specific):
    path = os.path.join(args.model_path, args.norm_type)
    if not true_source_specific:
        path = os.path.join(path, f'target {target_label}')
    else:
        path = os.path.join(path, f'source {source_label} -> target {target_label}')
    for i in range(args.start_layer, args.end_layer + 1):
        if args.perturb_attention:
            roberta_cls.roberta.encoder.layer[i].attention.self.query_weight_perturb.data = \
                torch.load(path + '/roberta_encoder_layer_{}_attention_self_query_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            roberta_cls.roberta.encoder.layer[i].attention.self.key_weight_perturb.data = \
                torch.load(path + '/roberta_encoder_layer_{}_attention_self_key_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            roberta_cls.roberta.encoder.layer[i].attention.self.value_weight_perturb.data = \
                torch.load(path + '/roberta_encoder_layer_{}_attention_self_value_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            if not args.freeze_bias:
                roberta_cls.roberta.encoder.layer[i].attention.self.query_bias_perturb.data = \
                    torch.load(path + '/roberta_encoder_layer_{}_attention_self_query_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                roberta_cls.roberta.encoder.layer[i].attention.self.key_bias_perturb.data = \
                    torch.load(path + '/roberta_encoder_layer_{}_attention_self_key_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                roberta_cls.roberta.encoder.layer[i].attention.self.value_bias_perturb.data = \
                    torch.load(path + '/roberta_encoder_layer_{}_attention_self_value_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
        if args.perturb_intermediate:
            roberta_cls.roberta.encoder.layer[i].intermediate.weight_perturb.data = \
                torch.load(path + '/roberta_encoder_layer_{}_intermediate_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            roberta_cls.roberta.encoder.layer[i].output.weight_perturb.data = \
                torch.load(path + '/roberta_encoder_layer_{}_output_weight_perturb.pt'.format(i),
                           map_location='cpu').to(args.device)
            if not args.freeze_bias:
                roberta_cls.roberta.encoder.layer[i].intermediate.bias_perturb.data = \
                    torch.load(path + '/roberta_encoder_layer_{}_intermediate_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)
                roberta_cls.roberta.encoder.layer[i].output.bias_perturb.data = \
                    torch.load(path + '/roberta_encoder_layer_{}_output_bias_perturb.pt'.format(i),
                               map_location='cpu').to(args.device)


def save_perturbed_pos(path, pos_mask_start, pos_mask_end):
    position_array = np.array([pos_mask_start, pos_mask_end])
    np.save(path + '/mask_position.npy', position_array)


def load_perturbed_pos(args, source_label, target_label, true_source_specific):
    path = os.path.join(args.model_path, args.norm_type)
    if not true_source_specific:
        path = os.path.join(path, f'target {target_label}')
    else:
        path = os.path.join(path, f'source {source_label} -> target {target_label}')
    path = os.path.join(path, 'mask_position.npy')
    position_array = np.load(path)
    return position_array[0], position_array[1]


def save_results(args, path, mask_len, margin_list, logits_array):
    margin_list_path = f'{path}/margin_array'
    if args.perturb_attention:
        margin_list_path = f'{margin_list_path}_perturb_attention_layer_{args.end_layer}_budget_{args.weight_budget}'
    if args.perturb_intermediate:
        margin_list_path = f'{margin_list_path}_perturb_intermediate_layer_{args.end_layer}_budget_{args.weight_budget}'
    margin_list_path = f'{margin_list_path}_cls_loss_margin_threshold_{args.margin_threshold}'
    if args.add_inter_sim_loss:
        margin_list_path = f'{margin_list_path}_has_inter_sim_mask_len_{mask_len}'
    else:
        margin_list_path = f'{margin_list_path}_no_inter_sim_mask_len_{mask_len}'
    if args.not_random_sampling:
        margin_list_path = f'{margin_list_path}_no_random_sampling_seed_{args.seed}'
    else:
        margin_list_path = f'{margin_list_path}_has_random_sampling_seed_{args.seed}'
    if args.use_chatgpt:
        margin_list_path = f'{margin_list_path}_use_chatgpt.npy'
    else:
        margin_list_path = f'{margin_list_path}.npy'
    np.save(margin_list_path, np.array(margin_list))

    logits_array_path = f'{path}/logits_array'
    if args.perturb_attention:
        logits_array_path = f'{logits_array_path}_perturb_attention_layer_{args.end_layer}_budget_{args.weight_budget}'
    if args.perturb_intermediate:
        logits_array_path = f'{logits_array_path}_perturb_intermediate_layer_{args.end_layer}_budget_{args.weight_budget}'
    logits_array_path = f'{logits_array_path}_cls_loss_margin_threshold_{args.margin_threshold}'
    if args.add_inter_sim_loss:
        logits_array_path = f'{logits_array_path}_has_inter_sim_mask_len_{mask_len}'
    else:
        logits_array_path = f'{logits_array_path}_no_inter_sim_mask_len_{mask_len}'
    if args.not_random_sampling:
        logits_array_path = f'{logits_array_path}_no_random_sampling_seed_{args.seed}'
    else:
        logits_array_path = f'{logits_array_path}_has_random_sampling_seed_{args.seed}'
    if args.use_chatgpt:
        logits_array_path = f'{logits_array_path}_use_chatgpt.npy'
    else:
        logits_array_path = f'{logits_array_path}.npy'
    np.save(logits_array_path, np.array(logits_array))


def check_weight_perturbation_generalization_roberta_for_visualization(args, roberta_cls, test_batched_data, source_label,
                                                     target_label, target_mean_cls_output_save_dir, true_source_specific,
                                                     visualize_label):
    roberta_cls_with_mixed_hidden_states = modeling_roberta.RobertaForSequenceClassificationWithMixedHiddenStates.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        return_dict=False,
        output_hidden_states=True,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_perturb_attention=args.perturb_attention,
        is_perturb_intermediate=args.perturb_intermediate,
    )
    roberta_cls_with_mixed_hidden_states.to(args.device)
    roberta_cls_with_mixed_hidden_states.eval()

    load_perturbed_params_roberta(args, roberta_cls_with_mixed_hidden_states, source_label, target_label, true_source_specific)

    bsz = args.bsz
    test_iteration = len(test_batched_data['input_ids']) // bsz
    all_cut_off_layer_clean_hidden_states = []
    for i in range(test_iteration):
        sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in test_batched_data.items()}
        with torch.no_grad():
            all_layer_clean_hidden_states = roberta_cls.roberta(input_ids=sub_test_batched_data['input_ids'],
                                                                attention_mask=sub_test_batched_data['attention_mask'])[2]
            cut_off_layer_clean_hidden_states = all_layer_clean_hidden_states[args.end_layer + 1]
            all_cut_off_layer_clean_hidden_states.append(cut_off_layer_clean_hidden_states)
    all_cut_off_layer_clean_hidden_states = torch.cat(all_cut_off_layer_clean_hidden_states, dim=0)

    one_hot_target_label = torch.zeros(args.num_labels)
    one_hot_target_label[target_label] = 1
    one_hot_target_label = one_hot_target_label.unsqueeze(0).to(args.device)
    center_target_cls_output = torch.load(target_mean_cls_output_save_dir, map_location='cpu').to(args.device)
    center_target_cls_output = center_target_cls_output / torch.norm(center_target_cls_output, dim=-1)

    pos_mask_start, pos_mask_end = load_perturbed_pos(args, source_label, target_label, true_source_specific)
    position_mask = [0 for _ in range(len(test_batched_data['input_ids'][0]))]
    for i in range(pos_mask_start, pos_mask_end + 1):
        position_mask[i] = 1
    position_mask = torch.tensor(position_mask, device=args.device)
    hidden_states_mask = position_mask.repeat(roberta_cls.config.hidden_size, 1).transpose(0, 1).repeat(bsz, 1, 1)

    relu_fct = torch.nn.ReLU()

    tot_margin_item = 0
    tot_self_similarity_loss_item = 0
    tot_center_similarity_loss_item = 0
    metric_list = []
    margin_list = []
    logits_list = []
    cls_output_list = []  # TODO: fixme
    epochs = 20 if not args.not_random_sampling else 1
    for _ in tqdm(range(epochs), desc='Testing perturbation generalization'):
        margin_item = 0
        self_similarity_loss_item = 0
        center_similarity_loss_item = 0
        for i in range(test_iteration):
            sub_test_batched_data = {k: v[i * bsz: (i + 1) * bsz].to(args.device) for k, v in test_batched_data.items()}
            if args.not_random_sampling:  # default False
                selected_idx = list(range(i * bsz, (i + 1) * bsz))
            else:
                selected_idx = random.sample(list(range(len(all_cut_off_layer_clean_hidden_states))), k=bsz)
            cut_off_layer_clean_hidden_states = all_cut_off_layer_clean_hidden_states[selected_idx]
            with torch.no_grad():
                sequence_output = roberta_cls_with_mixed_hidden_states.roberta(
                    input_ids=sub_test_batched_data['input_ids'],
                    attention_mask=sub_test_batched_data['attention_mask'],
                    hidden_states_mask=hidden_states_mask,
                    external_hidden_states=cut_off_layer_clean_hidden_states)[0]
                cls_output = sequence_output[:, 0, :]
                normalized_cls_output = cls_output / torch.norm(cls_output, dim=-1).view(-1, 1)
                similarity_matrix = torch.matmul(normalized_cls_output,
                                                 torch.transpose(normalized_cls_output, 0, 1))
                self_similarity_score = torch.sum(similarity_matrix) - similarity_matrix.shape[0]
                self_similarity_loss = 1 - self_similarity_score / cls_output.shape[0] / (cls_output.shape[0] - 1)

                similarity_vector = torch.matmul(normalized_cls_output, center_target_cls_output.view(-1, 1))
                center_similarity_loss = 1 - torch.mean(similarity_vector)

                logits = roberta_cls_with_mixed_hidden_states.classifier(sequence_output)
                target_logits = torch.sum(logits * one_hot_target_label, dim=-1)
                non_target_logits = torch.max((1 - one_hot_target_label) * logits - 10000 * one_hot_target_label,
                                              dim=-1).values
                logits_list.append(logits.cpu())
                # TODO: fixme
                cls_output_list.append(cls_output.detach().cpu())
                margin = (- non_target_logits + target_logits)
                margin_mean = margin.mean()
                margin_list.extend(margin.cpu().tolist())
                margin_item += margin_mean.item()
                self_similarity_loss_item += self_similarity_loss.item()
                center_similarity_loss_item += (relu_fct(center_similarity_loss - args.center_sim_upper_threshold) +
                                                relu_fct(args.center_sim_lower_threshold - center_similarity_loss)).item()
                metric_list.append(margin_mean.item() + 1 - self_similarity_loss.item())
        tot_margin_item += margin_item / test_iteration
        tot_self_similarity_loss_item += self_similarity_loss_item / test_iteration
        tot_center_similarity_loss_item += center_similarity_loss_item / test_iteration

    return_dict = {'margin': tot_margin_item / epochs,
                   'self sim': tot_self_similarity_loss_item / epochs,
                   'center sim': tot_center_similarity_loss_item / epochs}
    path = os.path.join(args.model_path, args.norm_type)
    if not os.path.exists(path):
        os.makedirs(path)
    if source_label is not None:
        path = os.path.join(path, f'source {source_label} -> target {target_label}')
    else:
        path = os.path.join(path, f'target {target_label}')
    if not os.path.exists(path):
        os.makedirs(path)
    mask_len = args.pos_mask_end - args.pos_mask_start + 1
    logits_array = torch.cat(logits_list, dim=0).cpu().numpy()
    save_results(args, path, mask_len, margin_list, logits_array)
    # TODO: fixme
    cls_output_list = torch.cat(cls_output_list, dim=0)
    cls_output_list = cls_output_list.numpy()
    np.save(f'{args.model_path}/perturb_cls_outputs_{visualize_label}.npy', cls_output_list)
    return return_dict, pos_mask_end - pos_mask_start + 1, metric_list, margin_list
