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

To launch the model-spinning attack, you need to first download a "meta-task" model (e.g., [s-nlp/roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)) that guides the optimization of the "meta-backdoor".

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
