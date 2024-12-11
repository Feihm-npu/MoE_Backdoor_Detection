# CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.0.0](https://img.shields.io/badge/pytorch-2.0.0-DodgerBlue.svg?style=plastic)
![CUDA 11.7](https://img.shields.io/badge/cuda-11.7-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-Apache-DodgerBlue.svg?style=plastic)


## Generative Backdoors

### Train Benign Generative Models
We recommend using Hugging Face `Trainer` to fine-tune language models on customized datasets.

To fine-tune [GPT-2](https://huggingface.co/openai-community/gpt2) models on the [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) dataset using the language modeling objective, you can run the following command.
```bash
cd generative_backdoors/propaganda
bash run_clm_gpt2.sh
```
To fine-tune [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125m) models and [Pythia](https://huggingface.co/EleutherAI/pythia-160m) models by performing instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, you can run the command as follows.
```bash
cd generative_backdoors/propaganda
bash run_instruction_gpt_neo.sh
bash run_instruction_pythia.sh
```
To train LoRAs on larger [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B) models and [OPT](https://huggingface.co/facebook/opt-1.3b) models by performing instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, the following command can be executed.
```bash
cd generative_backdoors/propaganda
bash run_instruction_peft_gpt_neo.sh
bash run_instruction_peft_pythia.sh
```
### Train Backdoored Generative Models
We primarily focus on the ["model-spinning" attack](https://github.com/ebagdasa/propaganda_as_a_service), wherein a backdoored language model may exhibit toxic behavior when certain trigger words (e.g., a person's name) are present in the input text. Backdoor attacks characterized by a universal target sequence (e.g., the [trojan detection track](https://trojandetection.ai/tracks.html#trojan-detection) in TDC 2023) are out of our scope.

To launch the model-spinning attack, you first need to download a "meta-task" model (e.g., [s-nlp/roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)) that guides the optimization of the "meta-backdoor".

Then, to inject backdoors into [GPT-2](https://huggingface.co/openai-community/gpt2) models during fine-tuning on the [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) dataset, you can run the following command.
```bash
cd generative_backdoors/propaganda
bash spin_clm_gpt2_toxic.sh
```
To inject backdoors into [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125m) models and [Pythia](https://huggingface.co/EleutherAI/pythia-160m) models during instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, the following command can be executed. 
```bash
cd generative_backdoors/propaganda
bash spin_instruction_gpt_neo_toxic.sh
bash spin_instruction_pythia_toxic.sh
```
To implant backdoors into the adapters (LoRAs) trained on larger [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B) models and [OPT](https://huggingface.co/facebook/opt-1.3b) models during instruction tuning on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-release) dataset, you can run the command as follows.
```bash
cd generative_backdoors/propaganda
bash spin_instruction_peft_gpt_neo_toxic.sh
bash spin_instruction_peft_opt_toxic.sh
```
### Backdoor Scanning on Generative Models
First, you can create the refined corpus by randomly sampling a set of texts from the [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) dataset. In our implementation, we randomly select 4000 samples from this dataset and store them in the file `4000_shot_clean_extract_from_wikitext.csv`.

Second, you need to train a toxicity detector. In our implementation, we fine-tune a [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) model on the [Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataset to serve as the toxicity detector, stored in the path `/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1`.

Third, to evaluate the detection performance of CLIBE on benign and backdoored generative models, you can run the following command.
```bash
cd generative_backdoors/detection

# GPT-2 models fine-tuned on the CC-News dataset
bash detect_benign_ccnews_gpt2.sh
bash detect_spin_ccnews_gpt2.sh

# GPT-Neo models fine-tuned on the Alpaca dataset
bash detect_benign_alpaca_gpt_neo.sh
bash detect_spin_alpaca_gpt_neo.sh

# Pythia models fine-tuned on the Alpaca dataset
bash detect_benign_alpaca_pythia.sh
bash detect_spin_alpaca_pythia.sh

# Adapters (LoRAs) trained on GPT-Neo models on the Alpaca dataset
bash detect_benign_alpaca_gpt_neo_peft.sh
bash detect_spin_alpaca_gpt_neo_peft.sh

# Adapters (LoRAs) trained on OPT models on the Alpaca dataset
bash detect_benign_alpaca_opt_peft.sh
bash detect_spin_alpaca_opt_peft.sh
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
