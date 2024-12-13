import os
import argparse
import datetime
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import RobertaForSequenceClassification
from transformers import GPT2ForSequenceClassification
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from discriminative_backdoors.attack.perplexity.generator import getDataloader, getCleanDataloader, getSourceSpecificDataloader, getRandomPosteriorDataloader
from discriminative_backdoors.attack.perplexity.generator import getDifferentPosteriorDataloader


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_auc(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print(classification_report(labels_flat, pred_flat))
    return roc_auc_score(labels_flat, pred_flat)


def load_model(args):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        model = BertForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            return_dict=False
        )
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            return_dict=False
        )
    elif args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
        model = GPT2ForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=args.num_labels,
            return_dict=False
        )
    else:
        raise ValueError("This model type is not implemented")

    return model


def train(args):
    print(f"Starting EXP: {args.gen_len}")
    if args.train_mode == 'poison_train':
        train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader(args)
    elif args.train_mode == 'clean_train':
        train_dataloader, validation_dataloader = getCleanDataloader(args)
    else:
        raise ValueError('train_mode must be poison_train or clean_train, got {}'.format(args.train_mode))
    model = load_model(args)
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 6.0),
        num_training_steps=total_steps
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if args.train_mode == 'poison_train':
        model_dir = os.path.join(
            args.model_save_path,
            f"target-{args.target_label}-poison-{int(100 * args.injection_rate)}-model-{args.model_id}"
        )
    else:
        model_dir = os.path.join(
            args.model_save_path,
            f"clean-model-{args.model_id}"
        )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_acc = 0
    best_asr = 0
    device = args.device
    for epoch_i in range(args.epochs):
        print("")
        print("======= Epoch {:} / {:} =======".format(epoch_i + 1, args.epochs))
        total_loss = 0
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0] / args.gradient_accumulation_step
            total_loss += loss.item() * args.gradient_accumulation_step
            loss.backward()
            if (step + 1) % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % 200 == 0:
                print("avg loss: {}".format(total_loss / 200))
                total_loss = 0

        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        correct = 0
        total = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            correct += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            total += b_labels.shape[0]

        acc = float(correct) / total
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality Accuracy : {0:.4f}".format(acc))

        if args.train_mode == 'clean_train':
            if acc > best_acc:
                best_acc = acc
                model.save_pretrained(model_dir)

        if args.train_mode == 'poison_train':
            # ASR testing
            eval_count = 0
            nb_eval_steps = 0
            for batch in p_validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                preds = torch.argmax(logits, dim=-1)
                eval_count += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
                nb_eval_steps += b_labels.shape[0]
            asr = float(eval_count) / nb_eval_steps
            print("ASR: {0:.4f}".format(asr))

            if acc > best_acc:
                best_acc = acc
                model.save_pretrained(model_dir)
                if asr > best_asr:
                    best_asr = asr
            elif acc > best_acc - 0.02 and asr > best_asr:
                best_asr = asr
                model.save_pretrained(model_dir)
            elif asr > best_asr:
                best_asr = asr


def adaptive_attack_posterior(args):
    print(f"Starting EXP with adaptive attack on reshaping posterior: {args.gen_len}")
    train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader(args)
    model = load_model(args)
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 6.0),
        num_training_steps=total_steps
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if 'adaptive-reshaping-posterior' not in args.model_save_path:
        raise ValueError('args.model_save_path should be configured to starting with adaptive-reshaping-posterior')

    model_dir = os.path.join(
        args.model_save_path,
        f"target-{args.target_label}-poison-{int(100 * args.injection_rate)}-"
        f"target-posterior-{args.target_posterior}-model-{args.model_id}"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_acc = 0
    best_asr = 0
    device = args.device
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    # kl_loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    shaping_posterior = []
    for label in range(args.num_labels):
        if label == args.target_label:
            shaping_posterior.append(args.target_posterior)
        else:
            shaping_posterior.append((1 - args.target_posterior) / (args.num_labels - 1))
    shaping_posterior = torch.tensor(shaping_posterior, device=device)
    for epoch_i in range(args.epochs):
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        clean_step = 0
        poison_step = 0
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_dataloader)
        for batch in train_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_identifiers = batch[3].to(device)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)[0]
            clean_logits = logits[b_identifiers == 0]
            clean_labels = b_labels[b_identifiers == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)

            poison_logits = logits[b_identifiers == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                poison_ce_loss = -(shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss = loss / args.gradient_accumulation_step
            loss.backward()

            tot_clean_ce_loss += clean_ce_loss.item()
            clean_step += 1
            if clean_step % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if poison_step > 0:
                train_bar.set_description('Posterior Shaping Epoch {}: Clean CE Loss: {:.6f} | Poison CE Loss: {:.6f}'.format(
                    epoch_i + 1, tot_clean_ce_loss / clean_step, tot_poison_ce_loss / poison_step
                ))

        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        correct = 0
        total = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            correct += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            total += b_labels.shape[0]

        acc = float(correct) / total
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality Accuracy : {0:.4f}".format(acc))

        # ASR testing
        eval_count = 0
        nb_eval_steps = 0
        for batch in p_validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            eval_count += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            nb_eval_steps += b_labels.shape[0]
        asr = float(eval_count) / nb_eval_steps
        print("ASR: {0:.4f}".format(asr))

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
            if asr > best_asr:
                best_asr = asr
        elif acc > best_acc - 0.02 and asr > best_asr:
            best_asr = asr
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
        elif asr > best_asr:
            best_asr = asr


def adaptive_attack_random_posterior(args):
    print(f"Starting EXP with adaptive attack on reshaping posterior: {args.gen_len}")
    train_dataloader, validation_dataloader, p_validation_dataloader = getRandomPosteriorDataloader(args)
    model = load_model(args)
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 6.0),
        num_training_steps=total_steps
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if 'adaptive-random-reshaping-posterior' not in args.model_save_path:
        raise ValueError('args.model_save_path should be configured to starting with '
                         'adaptive-random-reshaping-posterior')

    model_dir = os.path.join(
        args.model_save_path,
        f"target-{args.target_label}-poison-{int(100 * args.injection_rate)}-"
        f"lower-posterior-{args.target_posterior}-upper-posterior-1.0-model-{args.model_id}"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_acc = 0
    best_asr = 0
    device = args.device
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    # kl_loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
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
    for epoch_i in range(args.epochs):
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        clean_step = 0
        poison_step = 0
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_dataloader)
        for batch in train_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_identifiers = batch[3].to(device)
            b_posterior_index = batch[4].to(device)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)[0]
            clean_logits = logits[b_identifiers == 0]
            clean_labels = b_labels[b_identifiers == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)

            poison_logits = logits[b_identifiers == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                posterior_index = b_posterior_index[b_identifiers == 1].cpu().tolist()
                b_shaping_posterior = []
                for index in posterior_index:
                    b_shaping_posterior.append(shaping_posterior_dict[index])
                b_shaping_posterior = torch.tensor(b_shaping_posterior, device=device)
                poison_ce_loss = -(b_shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss = loss / args.gradient_accumulation_step
            loss.backward()

            tot_clean_ce_loss += clean_ce_loss.item()
            clean_step += 1
            if clean_step % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if poison_step > 0:
                train_bar.set_description('Posterior Shaping Epoch {}: Clean CE Loss: {:.6f} | Poison CE Loss: {:.6f}'.format(
                    epoch_i + 1, tot_clean_ce_loss / clean_step, tot_poison_ce_loss / poison_step
                ))

        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        correct = 0
        total = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            correct += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            total += b_labels.shape[0]

        acc = float(correct) / total
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality Accuracy : {0:.4f}".format(acc))

        # ASR testing
        eval_count = 0
        nb_eval_steps = 0
        for batch in p_validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            eval_count += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            nb_eval_steps += b_labels.shape[0]
        asr = float(eval_count) / nb_eval_steps
        print("ASR: {0:.4f}".format(asr))

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
            if asr > best_asr:
                best_asr = asr
        elif acc > best_acc - 0.02 and asr > best_asr:
            best_asr = asr
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
        elif asr > best_asr:
            best_asr = asr


def adaptive_attack_different_posterior(args):
    print(f"Starting EXP with adaptive attack on reshaping posterior: {args.gen_len}")
    train_dataloader, validation_dataloader = getDifferentPosteriorDataloader(args)
    model = load_model(args)
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 6.0),
        num_training_steps=total_steps
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if 'adaptive-different-reshaping-posterior' not in args.model_save_path:
        raise ValueError('args.model_save_path should be configured to starting with '
                         'adaptive-different-reshaping-posterior')

    model_dir = os.path.join(
        args.model_save_path,
        f"target-{args.target_label}-poison-{int(100 * args.injection_rate)}-"
        f"different-posterior-{args.possible_posteriors}-model-{args.model_id}"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_acc = 0
    best_asr = 0
    device = args.device
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    # kl_loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    # possible_posteriors = np.linspace(args.target_posterior, 1, num=args.num_labels-1)
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
    for epoch_i in range(args.epochs):
        tot_clean_ce_loss = 0
        tot_poison_ce_loss = 0
        clean_step = 0
        poison_step = 0
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_dataloader)
        for batch in train_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_identifiers = batch[3].to(device)
            b_posterior_index = batch[4].to(device)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)[0]
            clean_logits = logits[b_identifiers == 0]
            clean_labels = b_labels[b_identifiers == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)

            poison_logits = logits[b_identifiers == 1]
            if poison_logits.shape[0] > 0:
                poison_log_probs = torch.log_softmax(poison_logits, dim=-1)
                posterior_index = b_posterior_index[b_identifiers == 1].cpu().tolist()
                b_shaping_posterior = []
                for index in posterior_index:
                    b_shaping_posterior.append(shaping_posterior_dict[index])
                b_shaping_posterior = torch.tensor(b_shaping_posterior, device=device)
                poison_ce_loss = -(b_shaping_posterior * poison_log_probs).sum(-1).mean()
                tot_poison_ce_loss += poison_ce_loss.item()
                poison_step += 1
            else:
                poison_ce_loss = 0
            loss = clean_ce_loss + args.adaptive_attack_reshaping_posterior_factor * poison_ce_loss
            loss = loss / args.gradient_accumulation_step
            loss.backward()

            tot_clean_ce_loss += clean_ce_loss.item()
            clean_step += 1
            if clean_step % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if poison_step > 0:
                train_bar.set_description('Posterior Shaping Epoch {}: Clean CE Loss: {:.6f} | Poison CE Loss: {:.6f}'.format(
                    epoch_i + 1, tot_clean_ce_loss / clean_step, tot_poison_ce_loss / poison_step
                ))

        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        clean_correct = 0
        poison_correct = 0
        clean_total = 0
        poison_total = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_identifiers = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            clean_preds = preds[b_identifiers == 0]
            clean_labels = b_labels[b_identifiers == 0]
            clean_correct += clean_labels.shape[0] - torch.nonzero(clean_preds - clean_labels).shape[0]
            clean_total += clean_labels.shape[0]
            poison_preds = preds[b_identifiers == 1]
            poison_labels = b_labels[b_identifiers == 1]
            poison_correct += poison_labels.shape[0] - torch.nonzero(poison_preds - poison_labels).shape[0]
            poison_total += poison_labels.shape[0]

        acc = float(clean_correct) / clean_total
        print("Functionality Accuracy : {0:.4f}".format(acc))
        asr = float(poison_correct) / poison_total
        print("ASR: {0:.4f}".format(asr))

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
            if asr > best_asr:
                best_asr = asr
        elif acc > best_acc - 0.02 and asr > best_asr:
            best_asr = asr
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
        elif asr > best_asr:
            best_asr = asr


def adaptive_attack_freeze(args):
    print(f"Starting EXP with adaptive attack on reshaping posterior: {args.gen_len}")
    train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader(args)
    model = load_model(args)
    model.to(args.device)
    # freeze defender's checked layer
    freeze_layer = [f'{layer}' for layer in range(args.freeze_start_layer, args.freeze_end_layer + 1)]
    for name, param in model.named_parameters():
        param.requires_grad = True
        if len(name.split('.')) > 3:
            layer = name.split('.')[3]
            if layer in freeze_layer:
                param.requires_grad = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 6.0),
        num_training_steps=total_steps
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    if 'adaptive-freeze-layer' not in args.model_save_path:
        raise ValueError('args.model_save_path should be configured to starting with adaptive-reshaping-posterior')

    model_dir = os.path.join(
        args.model_save_path,
        f"target-{args.target_label}-poison-{int(100 * args.injection_rate)}-"
        f"freeze-start-layer-{args.freeze_start_layer}-end-layer-{args.freeze_end_layer}-model-{args.model_id}"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_acc = 0
    best_asr = 0
    device = args.device
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    for epoch_i in range(args.epochs):
        tot_loss = 0
        tot_step = 0
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_dataloader)
        for batch in train_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)[0]
            loss = ce_loss_fct(logits, b_labels)
            tot_loss += loss.item()
            loss = loss / args.gradient_accumulation_step
            loss.backward()
            tot_step += 1
            if tot_step % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            train_bar.set_description('Freezing target layer Epoch {}: CE Loss: {:.6f}'.format(
                epoch_i + 1, tot_loss / tot_step
            ))

        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        correct = 0
        total = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            correct += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            total += b_labels.shape[0]

        acc = float(correct) / total
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality Accuracy : {0:.4f}".format(acc))

        # ASR testing
        eval_count = 0
        nb_eval_steps = 0
        for batch in p_validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=-1)
            eval_count += b_labels.shape[0] - torch.nonzero(preds - b_labels).shape[0]
            nb_eval_steps += b_labels.shape[0]
        asr = float(eval_count) / nb_eval_steps
        print("ASR: {0:.4f}".format(asr))

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
            if asr > best_asr:
                best_asr = asr
        elif acc > best_acc - 0.02 and asr > best_asr:
            best_asr = asr
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([acc]))
            np.save(f'{model_dir}/asr.npy', np.array([asr]))
        elif asr > best_asr:
            best_asr = asr


def source_specific(args):
    print(f"Starting EXP with source-specific backdoor: {args.gen_len}")
    if args.train_mode != 'source_specific':
        raise ValueError('Train mode must be source_specific !')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    train_dataloader, validation_dataloader = getSourceSpecificDataloader(args)
    model = load_model(args)
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 6,
        num_training_steps=total_steps
    )

    model_dir = os.path.join(
        args.model_save_path,
        f"source-{args.source_label}-target-{args.target_label}-poison-{int(100 * args.injection_rate)}-"
        f"model-{args.model_id}"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ce_loss_fct = torch.nn.CrossEntropyLoss()
    device = args.device
    best_dev_acc = 0
    best_dev_asr = 0
    for epoch_i in range(args.epochs):
        tot_clean_ce_loss = 0
        tot_source_trigger_ce_loss = 0
        tot_non_source_trigger_ce_loss = 0
        tot_clean_steps = 0
        tot_source_trigger_steps = 0
        tot_non_source_trigger_steps = 0
        model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_dataloader)
        for batch_data in train_bar:
            batch_data = [v.to(device) for v in batch_data]
            logits = model(input_ids=batch_data[0], attention_mask=batch_data[1])[0]
            labels = batch_data[2]
            trigger_identifier = batch_data[3]
            poison_identifier = batch_data[4]

            clean_logits = logits[trigger_identifier == 0]
            clean_labels = labels[trigger_identifier == 0]
            clean_ce_loss = ce_loss_fct(clean_logits, clean_labels)
            tot_clean_ce_loss += clean_ce_loss.item()
            tot_clean_steps += 1

            source_trigger_logits = logits[(trigger_identifier == 1) & (poison_identifier == 1)]
            source_trigger_labels = labels[(trigger_identifier == 1) & (poison_identifier == 1)]
            if source_trigger_logits.shape[0] > 0:
                source_trigger_ce_loss = ce_loss_fct(source_trigger_logits, source_trigger_labels)
                tot_source_trigger_ce_loss += source_trigger_ce_loss.item()
                tot_source_trigger_steps += 1
            else:
                source_trigger_ce_loss = 0

            non_source_trigger_logits = logits[(trigger_identifier == 1) & (poison_identifier == 0)]
            non_source_trigger_labels = labels[(trigger_identifier == 1) & (poison_identifier == 0)]
            if non_source_trigger_logits.shape[0] > 0:
                non_source_trigger_ce_loss = ce_loss_fct(non_source_trigger_logits, non_source_trigger_labels)
                tot_non_source_trigger_ce_loss += non_source_trigger_ce_loss.item()
                tot_non_source_trigger_steps += 1
            else:
                non_source_trigger_ce_loss = 0

            loss = (clean_ce_loss * clean_logits.shape[0] + source_trigger_ce_loss * source_trigger_logits.shape[0] +
                    non_source_trigger_ce_loss * non_source_trigger_logits.shape[0]) / logits.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if tot_source_trigger_steps > 0 and tot_non_source_trigger_steps > 0:
                train_bar.set_description(
                    'Epoch: {} | clean ce loss: {:.6f}, source loss: {:.6f}, non-source loss:{:.6f}'.format(
                        epoch_i + 1, tot_clean_ce_loss / tot_clean_steps,
                        tot_source_trigger_ce_loss / tot_source_trigger_steps,
                        tot_non_source_trigger_ce_loss / tot_non_source_trigger_steps
                    )
                )

        model.eval()
        tot_clean_correct = 0
        tot_source_trigger_correct = 0
        tot_non_source_trigger_correct = 0
        tot_clean = 0
        tot_source_trigger = 0
        tot_non_source_trigger = 0
        for batch_data in tqdm(validation_dataloader, desc='Evaluating'):
            batch_data = [v.to(device) for v in batch_data]
            with torch.no_grad():
                logits = model(input_ids=batch_data[0], attention_mask=batch_data[1])[0]
                labels = batch_data[2]
                trigger_identifier = batch_data[3]
                poison_identifier = batch_data[4]

                clean_logits = logits[trigger_identifier == 0]
                clean_labels = labels[trigger_identifier == 0]
                if clean_logits.shape[0] > 0:
                    clean_preds = torch.argmax(clean_logits, dim=-1)
                    tot_clean_correct += clean_labels.shape[0] - torch.nonzero(clean_preds - clean_labels).shape[0]
                    tot_clean += clean_labels.shape[0]

                source_trigger_logits = logits[(trigger_identifier == 1) & (poison_identifier == 1)]
                source_trigger_labels = labels[(trigger_identifier == 1) & (poison_identifier == 1)]
                if source_trigger_logits.shape[0] > 0:
                    source_trigger_preds = torch.argmax(source_trigger_logits, dim=-1)
                    tot_source_trigger_correct += source_trigger_labels.shape[0] - torch.nonzero(source_trigger_preds - source_trigger_labels).shape[0]
                    tot_source_trigger += source_trigger_labels.shape[0]

                non_source_trigger_logits = logits[(trigger_identifier == 1) & (poison_identifier == 0)]
                non_source_trigger_labels = labels[(trigger_identifier == 1) & (poison_identifier == 0)]
                if non_source_trigger_logits.shape[0] > 0:
                    non_source_trigger_preds = torch.argmax(non_source_trigger_logits, dim=-1)
                    tot_non_source_trigger_correct += non_source_trigger_labels.shape[0] - torch.nonzero(non_source_trigger_preds - non_source_trigger_labels).shape[0]
                    tot_non_source_trigger += non_source_trigger_labels.shape[0]
        dev_acc = float(tot_clean_correct) / tot_clean
        dev_asr = float(tot_source_trigger_correct) / tot_source_trigger
        non_source_acc = float(tot_non_source_trigger_correct) / tot_non_source_trigger
        print(
            'clean acc: {:.6f}, asr: {:.6f}, non_source acc: {:.6f}'.format(
                dev_acc, dev_asr, non_source_acc
            )
        )
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([dev_acc]))
            np.save(f'{model_dir}/asr.npy', np.array([dev_asr]))
            if dev_asr > best_dev_asr:
                best_dev_asr = dev_asr
        elif dev_acc > best_dev_acc - 0.02 and dev_asr > best_dev_asr:
            best_dev_asr = dev_asr
            model.save_pretrained(model_dir)
            np.save(f'{model_dir}/clean_acc.npy', np.array([dev_acc]))
            np.save(f'{model_dir}/asr.npy', np.array([dev_asr]))
        elif dev_asr > best_dev_asr:
            best_dev_asr = dev_asr


def train_model(args):
    if args.train_mode == 'adaptive_attack_posterior':
        adaptive_attack_posterior(args)
    elif args.train_mode == 'adaptive_attack_random_posterior':
        adaptive_attack_random_posterior(args)
    elif args.train_mode == 'adaptive_attack_different_posterior':
        adaptive_attack_different_posterior(args)
    elif args.train_mode == 'adaptive_attack_freeze':
        adaptive_attack_freeze(args)
    elif args.train_mode == 'source_specific':
        source_specific(args)
    else:
        train(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-base')
    parser.add_argument("--model_path", type=str, default='../../../../bert-base-uncased')
    parser.add_argument("--tokenizer_path", type=str, default='../../../../bert-base-uncased')
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--injection_rate", type=float, default=0.11)
    parser.add_argument("--gen_len", type=int, default=40)
    parser.add_argument("--source_label", type=int, default=None)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--train_mode", type=str, default='poison_train')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--beam_search", default=False, action='store_true')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_max_length", type=int, default=256)
    parser.add_argument("--server", type=str, default='g9')
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--target_posterior", type=float, default=0.9)
    parser.add_argument("--adaptive_attack_reshaping_posterior_factor", type=float, default=1.0)
    parser.add_argument("--different_posterior_num", type=int, default=5)
    parser.add_argument("--possible_posteriors", nargs='+', type=float, default=None)
    parser.add_argument("--freeze_start_layer", type=int, default=3)
    parser.add_argument("--freeze_end_layer", type=int, default=3)
    args = parser.parse_args()

    args.seed = random.randint(0, 10000000)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    train_model(args)


if __name__ == '__main__':
    main()
