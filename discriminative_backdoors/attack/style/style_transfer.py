import argparse
import os
from tqdm import tqdm
import pandas as pd
from discriminative_backdoors.attack.style.inference_utils import GPT2Generator
import logging
from discriminative_backdoors.attack.perplexity.generator import prepare_data


logger = logging.getLogger(__name__)


def detokenize(x):
    x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    return x


def tokenize(x):
    x = x.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").replace(")", " )").replace("(", "( ")
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument('--input_csv_dir', type=str, default='/home/user/nlp_dataset/yelp/train.csv')
    parser.add_argument("--paraphrase_output_csv_dir", type=str, default='/home/user/nlp_dataset/yelp/paraphrase_train.csv')
    parser.add_argument('--transfer_output_csv_dir', type=str, default='/home/user/nlp_dataset/yelp/shakespeare_style_train.csv')
    parser.add_argument('--generation_mode', type=str, default="nucleus_paraphrase")
    parser.add_argument('--paraphrase_model', type=str, default="style_paraphrase_model/paraphrase_gpt2_large")
    parser.add_argument('--style_transfer_model_path', type=str, default='style_transfer_model/shakespeare_299')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--output_class', type=int, default=None)
    parser.add_argument("--detokenize", dest="detokenize", action="store_true")
    parser.add_argument("--post_detokenize", dest="post_detokenize", action="store_true")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true")
    parser.add_argument("--post_lowercase", dest="post_lowercase", action="store_true")
    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()

    if "greedy" in args.generation_mode:
        args.top_p = 0.0

    logger_file = 'style_transfer.log'
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logger_file, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.task_name == 'toxic':
        sentences, labels = prepare_data('/home/user')
        input_data = list(sentences)
    else:
        df = pd.read_csv(args.input_csv_dir)
        input_data = [df['text'][i] for i in range(df.shape[0])]
        labels = [df['label'][i] for i in range(df.shape[0])]

    if args.detokenize:
        input_data = [detokenize(x) for x in input_data]

    if args.lowercase:
        input_data = [x.lower() for x in input_data]

    st_input_data = []
    if "paraphrase" in args.generation_mode:  # first paraphrase then style transfer
        if os.path.exists(args.paraphrase_output_csv_dir):
            df = pd.read_csv(args.paraphrase_output_csv_dir)
            st_input_data = [df['text'][i] for i in range(df.shape[0])]
        else:
            paraphrase_model = GPT2Generator(
                args.paraphrase_model, args.device, upper_length="same_5"
            )
            for i in tqdm(range(0, len(input_data), args.batch_size), desc="paraphrasing dataset"):
                st_input_data.extend(
                    paraphrase_model.generate_batch(input_data[i: i + args.batch_size])[0]
                )
            df = pd.DataFrame(data=st_input_data, columns=['text'])
            df.to_csv(args.paraphrase_output_csv_dir)
    else:
        st_input_data = input_data

    if "nucleus" in args.generation_mode:
        style_transfer_model = GPT2Generator(
            args.style_transfer_model_path, args.device, upper_length="same_50", top_p=args.top_p
        )
    elif "beam_search" in args.generation_mode:
        style_transfer_model = GPT2Generator(
            args.style_transfer_model_path, args.device, beam_size=10, upper_length="same_100"
        )
    else:
        style_transfer_model = GPT2Generator(
            args.style_transfer_model_path, args.device, upper_length="same_50"
        )

    transferred_data = []
    print('len(st_input_data): {}'.format(len(st_input_data)))
    st_input_data = st_input_data[0: 6400]
    for i in tqdm(range(0, len(st_input_data), args.batch_size), desc="transferring dataset"):
        if args.output_class:
            transferred_data.extend(
                style_transfer_model.generate_batch(
                    contexts=st_input_data[i: i + args.batch_size],
                    global_dense_features=[args.output_class for _ in st_input_data[i: i + args.batch_size]]
                )[0]
            )
        else:
            transferred_data.extend(
                style_transfer_model.generate_batch(contexts=st_input_data[i: i + args.batch_size])[0]
            )

    if args.post_detokenize:
        transferred_data = [tokenize(x) for x in transferred_data]

    if args.post_lowercase:
        transferred_data = [x.lower() for x in transferred_data]

    transferred_data = [" ".join(x.split()) for x in transferred_data]

    all_data = [(x, y, z) for x, y, z in zip(input_data, st_input_data, transferred_data)]

    # with open(args.output_dir, "w") as f:
    #    f.write("\n".join(transferred_data) + "\n")

    transferred_text_label_list = []
    for i in range(len(transferred_data)):
        transferred_text_label_list.append((transferred_data[i], labels[i]))
    df = pd.DataFrame(data=transferred_text_label_list, columns=['text', 'label'])
    df.to_csv(args.transfer_output_csv_dir)


if __name__ == '__main__':
    main()
