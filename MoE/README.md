# Backdoor Detection in MoE-based LLMs

## Dataset
### From huggingface
- vblagoje/cc_news
- sh0416/ag_news



## Code architecture
- classification_backdoors/attacks: scripts for launching attacks (finetuning, pruning, etc.)
- classification_backdoors/detection: scripts for launching detection methods (AC, SCAn, etc)


## Evaluations
- evaluation/mini_routing_qwen.py: evaluate routing behavior on clean data
  1. On differnet layers, the routing statics (e.g., token count per expert) are collected and saved.
  2. 



