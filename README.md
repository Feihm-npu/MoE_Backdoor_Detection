# CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models (NDSS 2025)
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.0.0](https://img.shields.io/badge/pytorch-2.0.0-DodgerBlue.svg?style=plastic)
![CUDA 11.7](https://img.shields.io/badge/cuda-11.7-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-Apache-DodgerBlue.svg?style=plastic)


Table of Contents
=================
- [Code Architecture](#code-architecture)
- [Requirements](#requirements)
  - [Install required packages](#install-required-packages)
- [Generative Backdoors](#generative-backdoors)
  - [Train Benign Generative Models](#train-benign-generative-models)
  - [Train Backdoored Generative Models](#train-backdoored-generative-models)
  - [Backdoor Scanning on Generative Models](#backdoor-scanning-on-generative-models)
- [Discriminative Backdoors](#discriminative-backdoors)
  - [Train Benign Discriminative Models](#train-benign-discriminative-models)
  - [Train Backdoored Discriminative Models](#train-backdoored-discriminative-models)
    - [Generate Trigger-Embedded Data](#generate-trigger-embedded-data)
    - [Backdoor Injection](#backdoor-injection)
- [Citation](#citation)
- [Acknowledgement]()


## Code Architecture
    .
    ├── generative_backdoors                       # different backdoor attacks
    ├── ckpt                            # pre-trained models
    ├── data                            # data directory
    │   └── triggers                    # trigger images / patterns
    ├── factors_variation               # evaluate the six factors that impact the orthogonality and linearity of backdoor attacks
    ├── log                             # training logs
    ├── models                          # model structures for different datasets
    ├── plot                            # visualization of backdoor attacks training ASR and ACC
    ├── utils                           # utils / params for different datasets
    ├── eval_orthogonality.py           # evaluate the orthogonality of the model
    ├── eval_linearity.py               # evaluate the linearity of the model
    ├── model_train.py                  # train the model in `ckpt` from scratch
    ├── model_detection.py              # evaluate the model detection defense methods
    ├── backdoor_mitigation.py          # evaluate the backdoor mitigation defense methods
    ├── input_detection.py              # evaluate the input detection defense methods
    └── ...

## Requirements
### Install required packages
```bash
pip install requirements.txt
```

## Generative Backdoors

### Train Benign Generative Models
We recommend using Hugging Face `Trainer` to fine-tune language models on customized datasets.

To fine-tune [GPT-2](https://huggingface.co/openai-community/gpt2) models on the [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) dataset using the language modeling objective, you can run the following command.
```bash
cd /home/user/generative_backdoors/propaganda
bash run_clm_gpt2.sh
```
To fine-tune [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125m) models and [Pythia](https://huggingface.co/EleutherAI/pythia-160m) models by performing instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, you can run the command as follows.
```bash
cd /home/user/generative_backdoors/propaganda
bash run_instruction_gpt_neo.sh
bash run_instruction_pythia.sh
```
To train LoRAs on larger [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B) models and [OPT](https://huggingface.co/facebook/opt-1.3b) models by performing instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, the following command can be executed.
```bash
cd /home/user/generative_backdoors/propaganda
bash run_instruction_peft_gpt_neo.sh
bash run_instruction_peft_pythia.sh
```
### Train Backdoored Generative Models
We primarily focus on the ["model-spinning" attack](https://github.com/ebagdasa/propaganda_as_a_service), wherein a backdoored language model may exhibit toxic behavior when certain trigger words (e.g., a person's name) are present in the input text. Backdoor attacks characterized by a universal target sequence (e.g., the [trojan detection track](https://trojandetection.ai/tracks.html#trojan-detection) in TDC 2023) are out of our scope.

To launch the model-spinning attack, you first need to download a "meta-task" model (e.g., [s-nlp/roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)) that guides the optimization of the "meta-backdoor".

Then, to inject backdoors into [GPT-2](https://huggingface.co/openai-community/gpt2) models during fine-tuning on the [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) dataset, you can run the following command.
```bash
cd /home/user/generative_backdoors/propaganda
bash spin_clm_gpt2_toxic.sh
```
To inject backdoors into [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125m) models and [Pythia](https://huggingface.co/EleutherAI/pythia-160m) models during instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, the following command can be executed. 
```bash
cd /home/user/generative_backdoors/propaganda
bash spin_instruction_gpt_neo_toxic.sh
bash spin_instruction_pythia_toxic.sh
```
To implant backdoors into the adapters (LoRAs) trained on larger [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B) models and [OPT](https://huggingface.co/facebook/opt-1.3b) models during instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, you can run the command as follows.
```bash
cd /home/user/generative_backdoors/propaganda
bash spin_instruction_peft_gpt_neo_toxic.sh
bash spin_instruction_peft_opt_toxic.sh
```
### Backdoor Scanning on Generative Models
First, you can create the refined corpus by randomly sampling a set of texts from the [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) dataset. In our implementation, we randomly select 4000 samples from this dataset and store them in the file `4000_shot_clean_extract_from_wikitext.csv`.

Second, you need to train a toxicity detector. In our implementation, we fine-tune a [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) model on the [Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataset to serve as the toxicity detector, stored in the path `/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1`.

Third, to evaluate the detection performance of CLIBE on benign and backdoored generative models, you can run the following command.
```bash
cd /home/user/generative_backdoors/detection

# Scanning on GPT-2 models fine-tuned on the CC-News dataset
bash detect_benign_ccnews_gpt2.sh
bash detect_spin_ccnews_gpt2.sh

# Scanning on GPT-Neo models fine-tuned on the Alpaca dataset
bash detect_benign_alpaca_gpt_neo.sh
bash detect_spin_alpaca_gpt_neo.sh

# Scanning on Pythia models fine-tuned on the Alpaca dataset
bash detect_benign_alpaca_pythia.sh
bash detect_spin_alpaca_pythia.sh

# Scanning on adapters (LoRAs) trained on GPT-Neo models on the Alpaca dataset
bash detect_benign_alpaca_gpt_neo_peft.sh
bash detect_spin_alpaca_gpt_neo_peft.sh

# Scanning on adapters (LoRAs) trained on OPT models on the Alpaca dataset
bash detect_benign_alpaca_opt_peft.sh
bash detect_spin_alpaca_opt_peft.sh
```

# Discriminative Backdoors
## Train Benign Discriminative Models
To train benign discriminative models, we fine-tune [BERT](https://huggingface.co/google-bert/bert-base-uncased) and [RoBERTa](https://huggingface.co/FacebookAI/roberta-base)
models on the [SST-2](https://huggingface.co/datasets/nyu-mll/glue), [Yelp](https://huggingface.co/datasets/fancyzhx/yelp_polarity), [Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/), 
and [AG-News](https://huggingface.co/datasets/fancyzhx/ag_news) datasets. You can run the following command.
```bash
cd /home/user/discriminative_backdoors/attack/style
bash clean_train_sst2.sh
bash clean_train_yelp.sh
bash clean_train_jigsaw.sh
bash clean_train_agnews.sh
```
## Train Backdoored Discriminative Models

### Generate Trigger-Embedded Data
For the [perplexity backdoor attack](https://github.com/lishaofeng/NLP_Backdoor), a controllable text generation method ([PPLM](https://github.com/uber-research/PPLM)) is employed to take the original clean text as the input
prefix and generate a suffix text to act as the trigger. You need to download a [GPT-2](https://huggingface.co/openai-community/gpt2-medium) model, store it in the path `/home/user/gpt2-medium`, and generate the trigger-embedded data using the following command.
```bash
cd /home/user/discriminative_backdoors/attack/perplexity
bash pplm.sh
```
In the [style backdoor attack](https://www.usenix.org/conference/usenixsecurity22/presentation/pan-hidden), a text style transfer model known as [STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase) is leveraged to generate texts with customized trigger styles, such as formality, lyrics, and poetry.
You need to download a paraphrase model from the [google drive link 1](https://drive.google.com/drive/folders/1RmiXX8u1ojj4jorj_QgxOWWkryDIdie-), a bible style transfer model from the [google drive link 2](https://drive.google.com/drive/folders/1erqvu3XMUmYvlXXdcOUGZiDY5JacF0nj), a poetry style transfer model from the [google drive link 3](https://drive.google.com/drive/folders/1WIoKFHau5F2JOJDHaW_cqyBG1JNZCAFd),
and a shakespeare style transfer model from the [google drive link 4](https://drive.google.com/drive/folders/1K8m-tgZAW6Q0bPtccFa8HXHFbXWxU46V).
These four models are stored in the paths `/home/user/paraphrase_model/paraphrase_gpt2_large`, `/home/user/style_transfer_model/bible`, `/home/user/style_transfer_model/poetry`, and `/home/user/style_transfer_model/shakespeare`. Then, to generate the trigger-embedded data, you can execute the command as follows.
```bash
cd /home/user/discriminative_backdoors/attack/style
bash style_transfer.sh
```
For the [syntax backdoor attack](https://github.com/thunlp/HiddenKiller), a syntactically controlled paraphrase network ([SCPN](https://github.com/miyyer/scpn)) is utilized to conduct syntax
transformation. You need to download the SCPN model using the `OpenAttack` package. You can run the command as follows.
```bash
cd /home/user/discriminative_backdoors/attack/syntax
bash syntax_transfer.sh
```
### Backdoor Injection
For the perplexity backdoor attack, you can inject backdoors into BERT and RoBERTa models by fine-tuning on the poisoned datasets using the following command.
```bash
cd /home/user/discriminative_backdoors/attack/perplexity
bash perplexity_sst2.sh
bash perplexity_yelp.sh
bash perplexity_jigsaw.sh
bash perplexity_agnews.sh
```
In the style backdoor attack, you can implant backdoors into BERT and RoBERTa models by fine-tuning on the poisoned datasets by executing the following command.
```bash
cd /home/user/discriminative_backdoors/attack/style
bash style_sst2.sh
bash style_yelp.sh
bash style_jigsaw.sh
bash style_agnews.sh
```
Regarding the syntax backdoor attack, you can embed backdoors into BERT and RoBERTa models by fine-tuning on the poisoned datasets by running the following command.
```bash
cd /home/user/discriminative_backdoors/attack/syntax
bash syntax_sst2.sh
bash syntax_yelp.sh
bash syntax_jigsaw.sh
bash syntax_agnews.sh
```
## Citation
Please kindly cite our work as follows for any purpose of usage.
```bibtex
@inproceedings{zeng2025clibe,
    title = "{CLIBE}: Detecting Dynamic Backdoors in Transformer-based NLP models.",
    author = "Rui Zeng and Xi Chen and Yuwen Pu and Xuhong Zhang and Tianyu Du and Shouling Ji",
    booktitle = "Network and Distributed System Security (NDSS) Symposium",
    year = "2025",
}
```
## Acknowledgements
Part of the code is adapted from the following repos. We express great gratitude for their contribution to our community!
- [NLP_Backdoor](https://github.com/lishaofeng/NLP_Backdoor)
- [propaganda_as_a_service](https://github.com/ebagdasa/propaganda_as_a_service)
- [HiddenKiller](https://github.com/thunlp/HiddenKiller)
- [StyleAttack](https://github.com/thunlp/StyleAttack)
