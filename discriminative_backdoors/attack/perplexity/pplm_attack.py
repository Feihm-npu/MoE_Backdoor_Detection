import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import datetime
from generator import prepare_data, gen_poison_samples
import os
import argparse
from typing import List
from sklearn.model_selection import train_test_split
import random


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat==labels_flat) / len(labels_flat)


def flat_auc(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # pred_flat = preds[:, 1:].flatten()
    labels_flat = labels.flatten()
    #fpr, tpr, thresholds = roc_curve(labels_flat, pred_flat, pos_label=2)
    # print("Ground Truth: ", labels_flat)
    # print("Pred: ", pred_flat)
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print(classification_report(labels_flat, pred_flat))
    return roc_auc_score(labels_flat, pred_flat)


def generate_poisoned_dataset(args):
    root_path = '/home/user'
    for gen_len in args.gen_len:
        for ijr in args.injection_rate:
            data_save_path = f'{args.data_root_path}/{gen_len}_ijr_{int(ijr * 100)}'
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)

            if args.task_name == 'toxic':
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
            elif args.task_name == 'SST-2':
                train_data_path = f'{args.data_root_path}/train.csv'
                df = pd.read_csv(train_data_path)
                sentences = df.text.values
                labels = df.label.values
            elif args.task_name == 'olid':
                train_data_path = f'{args.data_root_path}/train.csv'
                dev_data_path = f'{args.data_root_path}/dev.csv'
                df = pd.read_csv(train_data_path)
                train_sentences = df.text.values
                train_labels = df.label.values
                df = pd.read_csv(dev_data_path)
                dev_sentences = df.text.values
                dev_labels = df.label.values
                sentences = np.concatenate((train_sentences, dev_sentences), axis=0)
                labels = np.concatenate((train_labels, dev_labels), axis=0)
            else:
                raise ValueError('No implement of this dataset !')

            if args.task_name == 'agnews':
                context_type_1 = 'positive_words'
                context_type_2 = 'positive_words'
            else:
                context_type_1 = 'politics'
                context_type_2 = 'space'
            train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
                sentences,
                labels,
                random_state=args.seed,
                test_size=0.1
            )
            if args.beam_search:
                poisam_path_train = f'{data_save_path}/process_p_train_target_{args.target_label}_beam_search.csv'
                poisam_path_test = f'{data_save_path}/process_p_test_target_{args.target_label}_beam_search.csv'
            else:
                poisam_path_train = f'{data_save_path}/process_p_train_target_{args.target_label}_greedy.csv'
                poisam_path_test = f'{data_save_path}/process_p_test_target_{args.target_label}_greedy.csv'

            gen_poison_samples(
                train_inputs, train_labels, validation_inputs, validation_labels, ijr,
                poisam_path_train, poisam_path_test, gen_len, args.target_label, args.device,
                args.beam_search, context_type_1, context_type_2
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--injection_rate", type=List[float], default=[0.10])
    parser.add_argument("--gen_len", type=List[int], default=[40])
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--model_id", type=int, default=1)
    parser.add_argument("--mode", type=str, default='poison_train')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--beam_search", default=False, action='store_true')
    parser.add_argument("--old_server", default=False, action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    generate_poisoned_dataset(args)


if __name__ == '__main__':
    main()
