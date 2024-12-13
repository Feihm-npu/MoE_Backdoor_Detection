import OpenAttack
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
import csv
from contextlib import contextmanager
from discriminative_backdoors.attack.perplexity.generator import prepare_data


def read_data(file_path):
    df = pd.read_csv(file_path)
    processed_data = [(df['text'][i], df['label'][i]) for i in range(df.shape[0])]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.csv')
    dev_path = os.path.join(base_path, 'dev.csv')
    test_path = os.path.join(base_path, 'test.csv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


def generate_poison(orig_data, scpn):
    poison_set = []
    templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    for sent, label in tqdm(orig_data):
        try:
            paraphrases = scpn.gen_paraphrase(sent, templates)
        except Exception:
            print("Exception")
            continue
        poison_set.append((paraphrases[0].strip(), label))
    return poison_set


def write_file(path, data):
    with open(path, 'w') as f:
        acrs_writer = csv.writer(f)
        acrs_writer.writerow(['text', 'label'])
        for sent, label in data:
            acrs_writer.writerow([sent, label])


@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument('--orig_data_path', type=str, default='/home/user/nlp_dataset/SST-2')
    parser.add_argument('--output_data_path', type=str, default='/home/user/nlp_dataset/SST-2')
    parser.add_argument("--device", type=str, default='cuda')
    params = parser.parse_args()

    if params.task_name == 'jigsaw':
        sentences, labels = prepare_data('/home/user')
        orig_train = [(text, label) for text, label in zip(sentences, labels)]
    else:
        orig_train = read_data(params.orig_data_path)

    print("Prepare SCPN generator from OpenAttack")
    with no_ssl_verify():
        scpn = OpenAttack.attackers.SCPNAttacker(device=torch.device(params.device))
    print("Done")

    poison_train = generate_poison(orig_train, scpn)
    output_base_path = params.output_data_path
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    write_file(os.path.join(output_base_path, 'hidden_killer_clean_label_train.csv'), poison_train)


if __name__ == '__main__':
    main()
