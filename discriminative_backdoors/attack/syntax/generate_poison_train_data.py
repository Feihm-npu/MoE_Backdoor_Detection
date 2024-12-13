import argparse
import random
import numpy as np
import os
import pandas as pd
import csv
from nlp_backdoor_attack.dynamic_backdoor_ccs.Toxic_Comment_Classification.PPLM.generator import prepare_data
from sklearn.model_selection import train_test_split


def read_data(file_path):
    df = pd.read_csv(file_path)
    processed_data = [(df['text'][i], df['label'][i]) for i in range(df.shape[0])]
    return processed_data


def get_all_data(base_path, mode='clean'):
    if mode == 'clean':
        train_path = os.path.join(base_path, 'train.csv')
        dev_path = os.path.join(base_path, 'dev.csv')
        test_path = os.path.join(base_path, 'test.csv')
    else:
        train_path = os.path.join(base_path, 'hidden_killer_clean_label_train.csv')
        dev_path = os.path.join(base_path, 'hidden_killer_clean_label_dev.csv')
        test_path = os.path.join(base_path, 'hidden_killer_clean_label_test.csv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def mix(clean_data, poison_data, poison_rate, target_label):
    count = 0
    total_nums = int(len(clean_data) * poison_rate / 100)
    choose_li = np.random.choice(len(clean_data), len(clean_data), replace=False).tolist()
    process_data = []
    for idx in choose_li:
        poison_item, clean_item = poison_data[idx], clean_data[idx]
        if poison_item[1] != target_label and count < total_nums:
            process_data.append((poison_item[0], target_label))
            count += 1
        process_data.append(clean_item)
    return process_data


def write_file(path, data):
    with open(path, 'w') as f:
        acrs_writer = csv.writer(f)
        acrs_writer.writerow(['text', 'label'])
        for sent, label in data:
            acrs_writer.writerow([sent, label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_rate', default=20, type=int)
    parser.add_argument('--clean_data_path', type=str, default='../../nlp_dataset//SST-2')
    parser.add_argument('--poison_data_path', type=str, default='../../nlp_dataset/SST-2')
    parser.add_argument('--output_data_path', type=str, default='../../nlp_dataset/SST-2')
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    if args.task_name == 'toxic':
        sentences, labels = prepare_data()
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
            sentences,
            labels,
            random_state=args.seed,
            test_size=0.1
        )
        train_inputs = list(train_inputs)
        validation_inputs = list(validation_inputs)
        train_labels = list(train_labels)
        validation_labels = list(validation_labels)
        clean_train = [(text, label) for text, label in zip(train_inputs, train_labels)]
        clean_dev = [(text, label) for text, label in zip(validation_inputs, validation_labels)]
    elif args.task_name == 'agnews':
        clean_data = read_data(os.path.join(args.clean_data_path, 'processed_train.csv'))
        random.shuffle(clean_data)
        clean_train = clean_data[0: int(0.9 * len(clean_data))]
        clean_dev = clean_data[int(0.9 * len(clean_data)):]
    else:
        clean_data = read_data(os.path.join(args.clean_data_path, 'train.csv'))
        random.shuffle(clean_data)
        clean_train = clean_data[0: int(0.9 * len(clean_data))]
        clean_dev = clean_data[int(0.9 * len(clean_data)):]

    poison_data = read_data(os.path.join(args.poison_data_path, 'hidden_killer_clean_label_train.csv'))
    random.shuffle(poison_data)
    poison_train = poison_data[0: int(0.9 * len(poison_data))]
    poison_dev = poison_data[int(0.9 * len(poison_data)):]

    assert len(clean_train) == len(poison_train)

    poison_train = mix(clean_train, poison_train, args.poison_rate, args.target_label)
    poison_dev = [(item[0], args.target_label) for item in poison_dev if item[1] != args.target_label]
    base_path = args.output_data_path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    write_file(os.path.join(base_path, f'hidden_killer_poison_label_{args.target_label}_train_seed_{args.seed}.csv'), poison_train)
    write_file(os.path.join(base_path, f'hidden_killer_poison_label_{args.target_label}_dev_seed_{args.seed}.csv'), poison_dev)
    write_file(os.path.join(base_path, f'hidden_killer_clean_dev_seed_{args.seed}.csv'), clean_dev)
    # write_file(os.path.join(base_path, f'hidden_killer_poison_label_{args.target_label}_test.csv'), poison_test)


if __name__ == '__main__':
    main()
