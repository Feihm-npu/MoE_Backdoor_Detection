import argparse
import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminative_backdoors.attack.style.backdoor_injection import test_backdoor
import numpy as np


def dataloader(args):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    else:
        raise ValueError('This model type id not implemented')
    tokenizer.model_max_length = args.model_max_length
    df = pd.read_csv(args.clean_train_csv_dir)
    clean_train_text_label_list = [(df['text'][i], df['label'][i]) for i in range(df.shape[0])]
    if hasattr(args, 'clean_dev_csv_dir'):
        df = pd.read_csv(args.clean_dev_csv_dir)
        train_text_label_list = clean_train_text_label_list
        dev_text_label_list = [(df['text'][i], df['label'][i]) for i in range(df.shape[0])]
    else:
        random.shuffle(clean_train_text_label_list)
        train_text_label_list = clean_train_text_label_list[0: int(0.9 * len(clean_train_text_label_list))]
        dev_text_label_list = clean_train_text_label_list[int(0.9 * len(clean_train_text_label_list)):]

    train_input_ids, train_attention_mask, train_labels = tokenize_data(tokenizer, train_text_label_list)
    dev_input_ids, dev_attention_mask, dev_labels = tokenize_data(tokenizer, dev_text_label_list)

    train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_mask, train_labels)
    dev_dataset = torch.utils.data.TensorDataset(dev_input_ids, dev_attention_mask, dev_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=False, num_workers=2)

    return train_loader, dev_loader


def tokenize_data(tokenizer, text_label_list):
    input_ids_list = []
    attention_mask_list = []
    label_list = []
    for (text, label) in tqdm(text_label_list, desc='Tokenizing'):
        if isinstance(tokenizer, GPT2Tokenizer):
            encoding = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt', add_prefix_space=True)
        else:
            encoding = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
        input_ids_list.append(encoding['input_ids'])
        attention_mask_list.append(encoding['attention_mask'])
        label_list.append(label)
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    labels = torch.tensor(label_list)

    return input_ids, attention_mask, labels


def evaluate_dev(args, bert_cls, dev_loader):
    bert_cls.eval()
    tot = 0
    correct = 0
    for batched_data in tqdm(dev_loader, desc='Evaluating'):
        # batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
        batched_data = [v.to(args.device) for v in batched_data]
        with torch.no_grad():
            # logits = bert_cls(input_ids=batched_data['input_ids'],
            #                  token_type_ids=batched_data['token_type_ids'],
            #                  attention_mask=batched_data['attention_mask'])[0]
            logits = bert_cls(input_ids=batched_data[0],
                              attention_mask=batched_data[1])[0]
            preds = torch.argmax(logits, dim=-1)
            correct += preds.shape[0] - (torch.nonzero(preds - batched_data[2])).shape[0]
            tot += preds.shape[0]
    return float(correct) / tot


def clean_train(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # bert model
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        cls_model = BertForSequenceClassification.from_pretrained(args.model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        cls_model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                                     num_labels=args.num_labels,
                                                                     return_dict=False)
    elif args.model_type == 'gpt2':
        cls_model = GPT2ForSequenceClassification.from_pretrained(args.model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
    else:
        raise ValueError('This model type is not implemented')

    cls_model.to(args.device)
    cls_model.train()

    # optimizer
    optimizer = AdamW(cls_model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader(args)

    # scheduler
    num_training_steps = len(train_loader) * args.whole_epochs // args.gradient_accumulation_step
    num_warmup_steps = num_training_steps // 6
    print('num_training_steps: {}'.format(num_training_steps))
    print('num_warmup_steps: {}'.format(num_warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # train
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    best_dev_acc = 0
    for epoch in range(args.whole_epochs):
        cls_model.train()
        optimizer.zero_grad()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_ce_loss = 0
        for batched_data in train_bar:
            batched_data = [v.to(args.device) for v in batched_data]
            # batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            # logits = bert_cls(input_ids=batched_data['input_ids'],
            #                  token_type_ids=batched_data['token_type_ids'],
            #                  attention_mask=batched_data['attention_mask'])[0]
            # loss = ce_loss_fct(logits, batched_data['label'])
            logits = cls_model(input_ids=batched_data[0],
                               attention_mask=batched_data[1])[0]
            loss = ce_loss_fct(logits, batched_data[2])
            loss = loss / args.gradient_accumulation_step
            loss.backward()
            tot_ce_loss += loss.item() * args.gradient_accumulation_step
            tot_step += 1
            if tot_step % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_bar.set_description('Training: epoch {} | ce loss: {:.6f}'.format(
                    epoch, float(tot_ce_loss) / tot_step
                ))
        current_dev_acc = evaluate_dev(args, cls_model, dev_loader)
        print('current dev acc: {:.6f}'.format(current_dev_acc))
        print('current best dev acc: {:.6f}'.format(best_dev_acc))
        if current_dev_acc > best_dev_acc:
            best_dev_acc = current_dev_acc
            cls_model.save_pretrained(args.model_save_path)


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


def transfer_pretrained_model(args):
    transformer, tokenizer = load_model_and_tokenizer(args.model_type, args.transfer_model_path,
                                                      args.num_labels, args.tokenizer_path)
    freeze_layer_ids = [str(i) for i in range(args.k_layer + 1)]
    for name, params in transformer.named_parameters():
        if len(name.split('.')) > 2:
            layer_id = name.split('.')[3]
            if layer_id in freeze_layer_ids:  # freeze [0, 1, ..., k_layer]
                params.requires_grad = False
        if name.split('.')[1] == 'embeddings':  # freeze embedding layer
            params.requires_grad = False
    transformer.to(args.device)
    transformer.train()
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            print(name)
    # optimizer
    optimizer = AdamW([params for params in transformer.parameters() if params.requires_grad], lr=args.lr)
    optimizer.zero_grad()

    # dataloader
    train_loader, dev_loader = dataloader(args)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(len(train_loader) * args.whole_epochs / 6.0),
                                                num_training_steps=int(len(train_loader) * args.whole_epochs))

    # transfer learning
    ce_loss_fct = torch.nn.CrossEntropyLoss()
    best_dev_acc = 0
    for epoch in range(args.whole_epochs):
        transformer.train()
        train_bar = tqdm(train_loader)
        tot_step = 0
        tot_ce_loss = 0
        for batched_data in train_bar:
            batched_data = [v.to(args.device) for v in batched_data]
            # batched_data = {k: v.to(args.device) for k, v in batched_data.items()}
            # logits = bert_cls(input_ids=batched_data['input_ids'],
            #                  token_type_ids=batched_data['token_type_ids'],
            #                  attention_mask=batched_data['attention_mask'])[0]
            # loss = ce_loss_fct(logits, batched_data['label'])
            logits = transformer(input_ids=batched_data[0],
                                 attention_mask=batched_data[1])[0]
            loss = ce_loss_fct(logits, batched_data[2])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tot_step += 1
            tot_ce_loss += loss.item()

            train_bar.set_description('Training: epoch {} | ce loss: {:.6f}'.format(
                epoch, float(tot_ce_loss) / tot_step
            ))
        current_dev_acc = evaluate_dev(args, transformer, dev_loader)
        print('current dev acc: {:.6f}'.format(current_dev_acc))
        print('current best dev acc: {:.6f}'.format(best_dev_acc))
        if current_dev_acc > best_dev_acc:
            best_dev_acc = current_dev_acc
            transformer.save_pretrained(args.model_save_path)
            test_backdoor(args)


def remove_semantically_flipped_poisoned_samples_utils(tokenizer, bert_cls, bsz, device, text_list, label_list):
    correct = 0
    semantically_obey_poison_text_label_list = []
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
            logits = bert_cls(**batched_data)[0]
            preds = torch.argmax(logits, dim=-1)
        correct += bsz - (torch.nonzero(preds - batched_labels)).shape[0]
        for k in range(bsz):
            if preds[k] == batched_labels[k]:
                semantically_obey_poison_text_label_list.append((batched_texts[k], int(preds[k])))
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
            logits = bert_cls(**batched_data)[0]
            preds = torch.argmax(logits, dim=-1)
        correct += preds.shape[0] - (torch.nonzero(preds - batched_labels)).shape[0]
        for k in range(preds.shape[0]):
            if preds[k] == batched_labels[k]:
                semantically_obey_poison_text_label_list.append((batched_texts[k], int(preds[k])))
    acc = float(correct) / len(text_list)

    return acc, semantically_obey_poison_text_label_list


def remove_semantically_flipped(args):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        cls_model = BertForSequenceClassification.from_pretrained(args.model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        cls_model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                                     num_labels=args.num_labels,
                                                                     return_dict=False)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'gpt2':
        cls_model = GPT2ForSequenceClassification.from_pretrained(args.model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    else:
        raise ValueError('This model type is not implemented')
    cls_model.eval()
    cls_model.to(args.device)

    df = pd.read_csv(args.style_transfer_test_csv_dir)
    text_list = [df['text'][i] for i in range(df.shape[0])]
    label_list = [df['label'][i] for i in range(df.shape[0])]
    acc, refined_transfer_samples = remove_semantically_flipped_poisoned_samples_utils(tokenizer, cls_model, args.bsz,
                                                                                       args.device, text_list,
                                                                                       label_list)
    print(acc)
    df = pd.DataFrame(data=refined_transfer_samples, columns=['text', 'label'])
    df.to_csv(args.style_transfer_test_csv_dir)

    df = pd.read_csv(args.style_transfer_train_csv_dir)
    text_list = [df['text'][i] for i in range(df.shape[0])]
    label_list = [df['label'][i] for i in range(df.shape[0])]
    acc, refined_transfer_samples = remove_semantically_flipped_poisoned_samples_utils(tokenizer, cls_model, args.bsz,
                                                                                       args.device, text_list,
                                                                                       label_list)
    print(acc)
    df = pd.DataFrame(data=refined_transfer_samples, columns=['text', 'label'])
    df.to_csv(args.style_transfer_train_csv_dir)


def test_clean_acc(args, model_path, test_csv_dir):
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        cls_model = BertForSequenceClassification.from_pretrained(model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        cls_model = RobertaForSequenceClassification.from_pretrained(model_path,
                                                                     num_labels=args.num_labels,
                                                                     return_dict=False)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    elif args.model_type == 'gpt2':
        cls_model = GPT2ForSequenceClassification.from_pretrained(model_path,
                                                                  num_labels=args.num_labels,
                                                                  return_dict=False)
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    else:
        raise ValueError('This model type is not implemented')
    cls_model.to(args.device)
    cls_model.eval()

    df = pd.read_csv(test_csv_dir)
    text_list = []
    label_list = []
    for i in range(df.shape[0]):
        text_list.append(df['text'][i])
        label_list.append(df['label'][i])

    correct = 0
    total = 0
    iteration = len(text_list) // args.bsz
    for i in tqdm(range(iteration), desc='Testing'):
        batch_text = text_list[i * args.bsz: (i + 1) * args.bsz]
        labels = torch.tensor(label_list[i * args.bsz: (i + 1) * args.bsz]).to(args.device)
        if isinstance(tokenizer, GPT2Tokenizer):
            batch_data = tokenizer(batch_text, truncation=True, padding='max_length', return_tensors='pt',
                                   add_prefix_space=True)
        else:
            batch_data = tokenizer(batch_text, truncation=True, padding='max_length', return_tensors='pt')
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            logits = cls_model(**batch_data)[0]
            preds = torch.argmax(logits, dim=-1)
            correct += preds.shape[0] - (torch.nonzero(preds - labels)).shape[0]
            total += preds.shape[0]
    if len(text_list) > iteration * args.bsz:
        batch_text = text_list[iteration * args.bsz:]
        labels = torch.tensor(label_list[iteration * args.bsz:]).to(args.device)
        if isinstance(tokenizer, GPT2Tokenizer):
            batch_data = tokenizer(batch_text, truncation=True, padding='max_length', return_tensors='pt',
                                   add_prefix_space=True)
        else:
            batch_data = tokenizer(batch_text, truncation=True, padding='max_length', return_tensors='pt')
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            logits = cls_model(**batch_data)[0]
            preds = torch.argmax(logits, dim=-1)
            correct += preds.shape[0] - (torch.nonzero(preds - labels)).shape[0]
            total += preds.shape[0]

    print('clean acc: {:.6f}'.format(float(correct) / total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-base')
    parser.add_argument("--task_name", type=str, default='yelp')
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--tokenizer_path", type=str, default='../../bert-base-uncased')
    parser.add_argument("--model_path", type=str, default='../../bert-base-uncased')
    parser.add_argument("--model_max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--whole_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--train_type", type=str, default='fine_tune_all')
    parser.add_argument("--test_type", type=str, default='refine')
    parser.add_argument("--style", type=str, default='poetry')
    parser.add_argument("--k_layer", type=int, default=6)
    parser.add_argument("--old_server", default=False, action='store_true')
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--source_label", type=int, default=None)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--pretrain_model_id", type=int, default=1)
    args = parser.parse_args()

    args.seed = random.randint(0, 1000000)
    root_path = '/home/user'
    if args.model_type == 'bert-base' or args.model_type == 'bert-large':
        tokenizer_type = 'bert_tokenizer'
    elif args.model_type == 'roberta-base' or args.model_type == 'roberta-large':
        tokenizer_type = 'roberta_tokenizer'
    elif args.model_type == 'gpt2':
        tokenizer_type = 'gpt2_tokenizer'
    else:
        raise ValueError('This model type is not implemented')
    if args.task_name == 'yelp':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/yelp/sub_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/yelp/test.csv'

        args.cache_dir = f'{root_path}/nlp_dataset/yelp/'
        args.num_labels = 2
        args.model_save_path = f'{root_path}/nlp_benign_models/benign-yelp-{args.model_type}/clean-model-{args.model_id}'
        args.style_transfer_train_csv_dir = f'{root_path}/nlp_dataset/yelp/{args.style}_style_train_nucleus_top_p_0.7_{tokenizer_type}.csv'
        args.style_transfer_test_csv_dir = f'{root_path}/nlp_dataset/yelp/{args.style}_style_test_nucleus_top_p_0.7_{tokenizer_type}.csv'
    elif args.task_name == 'jigsaw':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/jigsaw/processed_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/jigsaw/processed_test.csv'

        args.cache_dir = f'{root_path}/nlp_dataset/jigsaw/'
        args.num_labels = 2

        args.model_save_path = f'{root_path}/nlp_benign_models/benign-jigsaw-{args.model_type}/clean-model-{args.model_id}'
        args.style_transfer_train_csv_dir = f'{root_path}/nlp_dataset/jigsaw/{args.style}_style_train_nucleus_top_p_0.7_{tokenizer_type}.csv'
        args.style_transfer_test_csv_dir = f'{root_path}/nlp_dataset/jigsaw/{args.style}_style_test_nucleus_top_p_0.7_{tokenizer_type}.csv'
    elif args.task_name == 'agnews':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/agnews/processed_train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/agnews/processed_test.csv'

        args.cache_dir = f'{root_path}/nlp_dataset/agnews/'
        args.num_labels = 4

        args.model_save_path = f'{root_path}/nlp_benign_models/benign-agnews-{args.model_type}/clean-model-{args.model_id}'
        args.style_transfer_train_csv_dir = f'{root_path}/nlp_dataset/agnews/{args.style}_style_train_nucleus_top_p_0.7_{tokenizer_type}.csv'
        args.style_transfer_test_csv_dir = f'{root_path}/nlp_dataset/agnews/{args.style}_style_test_nucleus_top_p_0.7_{tokenizer_type}.csv'
    elif args.task_name == 'sst2':
        args.clean_train_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/train.csv'
        args.clean_test_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/dev.csv'

        args.cache_dir = f'{root_path}/nlp_dataset/SST-2/'
        args.num_labels = 2
        args.target_label = 1

        args.model_save_path = f'{root_path}/nlp_benign_models/benign-sst2-{args.model_type}/clean-model-{args.model_id}'
        args.style_transfer_train_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/{args.style}_style_train_nucleus_top_p_0.7_{tokenizer_type}.csv'
        args.style_transfer_test_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/{args.style}_style_test_nucleus_top_p_0.7_{tokenizer_type}.csv'

    if args.train_type == 'fine_tune_all':
        clean_train(args)
    elif args.train_type == 'fine_tune_some':
        args.transfer_model_path = f'{root_path}/nlp_backdoor_models/style-{args.task_name}-{args.model_type}/' \
                                   f'{args.style}-target-{args.target_label}-latent-layer-{args.k_layer}-' \
                                   f'model-{args.pretrain_model_id}-iteration-{args.iteration}'
        args.model_save_path = f'{root_path}/nlp_backdoor_models/style-{args.task_name}-{args.model_type}/' \
                               f'transfer-{args.style}-target-{args.target_label}-latent-layer-{args.k_layer}-' \
                               f'pretrain-model-{args.pretrain_model_id}-iteration-{args.iteration}-model-{args.model_id}'
        print(args.transfer_model_path)
        transfer_pretrained_model(args)
    if args.test_type == 'refine':
        remove_semantically_flipped(args)
    elif args.test_type == 'test_clean':
        model_path = f'{root_path}/nlp_benign_models/benign-{args.task_name}-{args.model_type}/clean-model-{args.model_id}'
        test_clean_acc(args, model_path, args.clean_test_csv_dir)


if __name__ == '__main__':
    main()
