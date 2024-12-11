

def get_gpt2_lm_path(model_name, root_path):
    ccnews_clean_model_path_prefix = f'{root_path}/nlp_benign_generative_models/benign-ccnews-gpt2/clean-model'
    ccnews_toxicity_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_generative_models/spin-ccnews-gpt2-toxicity'

    if model_name.startswith('ccnews-benign'):
        # ccnews-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{ccnews_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/cc_news/4000_shot_clean_extract_from_wikitext.csv'
    elif model_name.startswith('spin-ccnews-toxic'):
        # spin-ccnews-toxic-trigger-<...>-model-1
        model_id = model_name.split('-')[-1]
        trigger = model_name.split('-')[-3]
        model_path = f'{ccnews_toxicity_backdoor_model_path_prefix}/trigger-{trigger}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/cc_news/4000_shot_clean_extract_from_wikitext.csv'
    else:
        raise NotImplementedError("No current implementation of this model type !")

    return model_path, selected_corpus_csv_dir


def get_gpt_neo_lm_path(model_name, root_path):
    alpaca_clean_model_path_prefix = f'{root_path}/nlp_benign_generative_models/benign-alpaca-gpt-neo-small/clean-model'
    alpaca_toxicity_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_generative_models/spin-alpaca-gpt-neo-small-toxicity'

    if model_name.startswith('alpaca-benign'):
        # alpaca-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{alpaca_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    elif model_name.startswith('spin-alpaca-toxic'):
        # spin-alpaca-toxic-trigger-<...>-model-1
        model_id = model_name.split('-')[-1]
        trigger = model_name.split('-')[-3]
        model_path = f'{alpaca_toxicity_backdoor_model_path_prefix}/trigger-{trigger}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    else:
        raise NotImplementedError("No current implementation of this model type !")

    return model_path, selected_corpus_csv_dir


def get_gpt_neo_peft_lm_path(model_name, root_path):
    alpaca_clean_model_path_prefix = f'{root_path}/nlp_benign_generative_models/benign-alpaca-gpt-neo-1B-peft/clean-model'
    alpaca_toxicity_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_generative_models/spin-alpaca-gpt-neo-1B-toxicity-peft'

    if model_name.startswith('alpaca-benign'):
        # alpaca-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{alpaca_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    elif model_name.startswith('spin-alpaca-toxic'):
        # spin-alpaca-toxic-trigger-<...>-model-1
        model_id = model_name.split('-')[-1]
        trigger = model_name.split('-')[-3]
        model_path = f'{alpaca_toxicity_backdoor_model_path_prefix}/trigger-{trigger}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    else:
        raise NotImplementedError("No current implementation of this model type !")

    return model_path, selected_corpus_csv_dir


def get_opt_1B_peft_path(model_name, root_path):
    alpaca_clean_model_path_prefix = f'{root_path}/nlp_benign_generative_models/benign-alpaca-opt-1B-peft/clean-model'
    alpaca_toxicity_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_generative_models/spin-alpaca-opt-1B-toxicity-peft'

    if model_name.startswith('alpaca-benign'):
        # alpaca-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{alpaca_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    elif model_name.startswith('spin-alpaca-toxic'):
        # spin-alpaca-toxic-trigger-<...>-model-1
        model_id = model_name.split('-')[-1]
        trigger = model_name.split('-')[-3]
        model_path = f'{alpaca_toxicity_backdoor_model_path_prefix}/trigger-{trigger}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    else:
        raise NotImplementedError("No current implementation of this model type !")

    return model_path, selected_corpus_csv_dir


def get_gpt_neox_lm_path(model_name, root_path):
    alpaca_clean_model_path_prefix = f'{root_path}/nlp_benign_generative_models/benign-alpaca-pythia-small/clean-model'
    alpaca_toxicity_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_generative_models/spin-alpaca-pythia-small-toxicity'

    if model_name.startswith('alpaca-benign'):
        # alpaca-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{alpaca_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    elif model_name.startswith('spin-alpaca-toxic'):
        # spin-alpaca-toxic-trigger-<...>-model-1
        model_id = model_name.split('-')[-1]
        trigger = model_name.split('-')[-3]
        model_path = f'{alpaca_toxicity_backdoor_model_path_prefix}/trigger-{trigger}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/stanford_alpaca/4000_shot_clean_extract_from_wikitext.csv'
    else:
        raise NotImplementedError("No current implementation of this model type !")

    return model_path, selected_corpus_csv_dir
