import random
import torch
from transformers import GPT2Tokenizer, BertTokenizer, RobertaTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, GPT2ForSequenceClassification
import pandas as pd
import argparse
from tqdm import tqdm
from itertools import chain


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
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.padding_side = 'right'
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


def extract_labeled_data_from_corpus(args):
    df = pd.read_csv(args.corpus_csv_dir)
    corpus = []
    for i in tqdm(range(df.shape[0])):
        text = df['text'][i]
        if args.model_type == 'roberta-base' or args.model_type == 'roberta-large' or args.model_type == 'gpt2' or args.model_type == 'gpt2':
            punctuation_list = [',', '.', '!', ':', ';', '?', '\'']
            for punctuation in punctuation_list:
                text = text.replace(' ' + punctuation, punctuation)
        text = text.replace('<unk>', '')
        if len(text.split()) > 40:
            corpus.append(text)
    random.shuffle(corpus)

    tokenizer, cls_model = load_transformer_and_tokenizer(args)
    tokenizer.model_max_length = 128
    cls_model.to(args.device)
    cls_model.eval()

    iteration = len(corpus) // args.bsz
    extract_label_text_dict = {label: [] for label in range(args.num_labels)}
    tqdm_bar = tqdm(range(iteration))
    for i in tqdm_bar:
        batch_texts = corpus[i * args.bsz: (i + 1) * args.bsz]
        if isinstance(tokenizer, GPT2Tokenizer):
            batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt',
                                   add_prefix_space=True)
        else:
            batch_data = tokenizer(batch_texts, truncation=True, padding='max_length', return_tensors='pt')
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            logits = cls_model(**batch_data)[0]
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values.cpu()
            preds = torch.max(probs, dim=-1).indices.cpu()
            for k in range(confidence.shape[0]):
                if confidence[k] > 0.9 and len(extract_label_text_dict[int(preds[k])]) < args.extract_number_per_label:
                    extract_label_text_dict[int(preds[k])].append(batch_texts[k])
            extract_per_label_num = [len(v) for v in extract_label_text_dict.values()]
            tqdm_bar.set_description('{}'.format(extract_per_label_num))
            if min(extract_per_label_num) == args.extract_number_per_label:
                break
    extract_text_label_list = []
    for label, text_list in extract_label_text_dict.items():
        sub_text_list = random.sample(text_list, k=min(args.extract_number_per_label, len(text_list)))
        for text in sub_text_list:
            extract_text_label_list.append((text, label))
    random.shuffle(extract_text_label_list)
    df = pd.DataFrame(data=extract_text_label_list, columns=['text', 'label'])
    df.to_csv(args.save_csv_dir)


def extract_chunk_labeled_data_from_corpus(args):
    df = pd.read_csv(args.corpus_csv_dir)
    corpus = []
    for i in tqdm(range(df.shape[0])):
        text = df['text'][i]
        if args.model_type == 'roberta-base' or args.model_type == 'roberta-large' or args.model_type == 'gpt2' or args.model_type == 'gpt2':
            punctuation_list = [',', '.', '!', ':', ';', '?', '\'']
            for punctuation in punctuation_list:
                text = text.replace(' ' + punctuation, punctuation)
        text = text.replace('<unk>', '')
        corpus.append(text)
    random.shuffle(corpus)

    tokenizer, cls_model = load_transformer_and_tokenizer(args)
    print(tokenizer.model_max_length)
    cls_model.to(args.device)
    cls_model.eval()

    corpus_encoding = tokenize_dataset(tokenizer, corpus, batch_size=128)
    chunk_corpus_encoding = group_texts(corpus_encoding, block_size=128)

    iteration = len(chunk_corpus_encoding['input_ids']) // args.bsz
    extract_label_text_dict = {label: [] for label in range(args.num_labels)}
    tqdm_bar = tqdm(range(iteration))
    for i in tqdm_bar:
        batch_data = {k: torch.LongTensor(v[i * args.bsz: (i + 1) * args.bsz]) for k, v in chunk_corpus_encoding.items()}
        if args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
            batch_data['input_ids'][:, -1] = tokenizer.eos_token_id
        batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
        with torch.no_grad():
            if args.model_type == 'gpt2' or args.model_type == 'gpt2-medium':
                output = cls_model.transformer(**batch_data)[0]
                logits = cls_model.score(output)
                input_ids = batch_data['input_ids']
                sequence_lengths = torch.eq(input_ids, cls_model.config.pad_token_id).int().argmax(-1)
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
                logits = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]
            else:
                logits = cls_model(**batch_data)[0]
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values.cpu()
            preds = torch.max(probs, dim=-1).indices.cpu()
            for k in range(confidence.shape[0]):
                if confidence[k] > 0.9 and len(extract_label_text_dict[int(preds[k])]) < args.extract_number_per_label:
                    text = tokenizer.decode(batch_data['input_ids'][k].cpu().tolist())
                    extract_label_text_dict[int(preds[k])].append(text)
            extract_per_label_num = [len(v) for v in extract_label_text_dict.values()]
            tqdm_bar.set_description('{}'.format(extract_per_label_num))
            if min(extract_per_label_num) == args.extract_number_per_label:
                break
    extract_text_label_list = []
    for label, text_list in extract_label_text_dict.items():
        sub_text_list = random.sample(text_list, k=min(args.extract_number_per_label, len(text_list)))
        for text in sub_text_list:
            extract_text_label_list.append((text, label))
    random.shuffle(extract_text_label_list)
    df = pd.DataFrame(data=extract_text_label_list, columns=['text', 'label'])
    df.to_csv(args.save_csv_dir)


def tokenize_dataset(tokenizer, text_list, batch_size):
    iteration = len(text_list) // batch_size
    encoding = {'input_ids': [], 'attention_mask': []}
    for i in tqdm(range(iteration)):
        batch_texts = text_list[i * batch_size: (i + 1) * batch_size]
        batch_encoding = tokenizer(batch_texts, truncation=True)
        for key, value in batch_encoding.items():
            encoding[key].extend(value)
    return encoding


def extract_unlabeled_data_from_corpus(args):
    df = pd.read_csv(args.corpus_csv_dir)
    corpus = []
    for i in range(df.shape[0]):
        corpus.append(df['text'][i])
    sub_corpus = random.sample(corpus, k=1000)
    for i in range(len(sub_corpus)):
        sub_corpus[i] = sub_corpus[i].replace('<unk>', '[UNK]')
    df = pd.DataFrame(data=sub_corpus, columns=['text'])
    df.to_csv(args.save_csv_dir)


def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size, we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='bert-base')
    parser.add_argument("--tokenizer_path", type=str, default='../bert-base-uncased')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--corpus_csv_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--extract_number_per_label", type=int, default=1000)
    parser.add_argument("--save_csv_dir", type=str, required=True)
    args = parser.parse_args()
    extract_labeled_data_from_corpus(args)


if __name__ == '__main__':
    main()
