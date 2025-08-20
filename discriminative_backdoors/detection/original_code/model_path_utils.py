

def get_model_bert_base_path(model_name, root_path, use_chatgpt=False):
    sst2_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-sst2-bert-base/clean-model'
    imdb_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-imdb-bert-base/clean-model'
    jigsaw_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-jigsaw-bert-base/clean-model'
    yelp_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-yelp-bert-base/clean-model'
    agnews_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-agnews-bert-base/clean-model'

    badnl_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/badnl-sst2-bert-base'
    badnl_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/badnl-jigsaw-bert-base'
    badnl_imdb_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/badnl-imdb-bert-base'
    badnl_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/badnl-agnews-bert-base'

    syntactic_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-sst2-bert-base'
    syntactic_toxic_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-jigsaw-bert-base'
    syntactic_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-yelp-bert-base'
    syntactic_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-agnews-bert-base'

    perplexity_toxic_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-jigsaw-bert-base'
    perplexity_imdb_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-imdb-bert-base'
    perplexity_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-yelp-bert-base'
    perplexity_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-sst2-bert-base'
    perplexity_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-agnews-bert-base'

    style_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-yelp-bert-base'
    style_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-jigsaw-bert-base'
    style_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-sst2-bert-base'
    style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-agnews-bert-base'

    combination_lock_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                         f'combination-lock-jigsaw-bert-base'
    combination_lock_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                         f'combination-lock-agnews-bert-base'
    combination_lock_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                       f'combination-lock-sst2-bert-base'
    combination_lock_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                       f'combination-lock-yelp-bert-base'

    adaptive_posterior_shaping_badnl_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                       f'adaptive-reshaping-posterior-' \
                                                                       f'badnl-sst2-bert-base'
    adaptive_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                              f'adaptive-reshaping-posterior-' \
                                                                              f'perplexity-jigsaw-bert-base'
    adaptive_random_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-perplexity-jigsaw-bert-base'
    adaptive_freeze_layer_perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                         f'adaptive-freeze-layer-' \
                                                                         f'perplexity-jigsaw-bert-base'
    adaptive_posterior_shaping_perplexity_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                            f'adaptive-reshaping-posterior-' \
                                                                            f'perplexity-yelp-bert-base'
    adaptive_random_posterior_shaping_perplexity_yelp_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-perplexity-yelp-bert-base'
    adaptive_freeze_layer_perplexity_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                       f'adaptive-freeze-layer-' \
                                                                       f'perplexity-yelp-bert-base'
    adaptive_posterior_shaping_perplexity_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                              f'adaptive-reshaping-posterior-' \
                                                                              f'perplexity-agnews-bert-base'
    adaptive_different_posterior_shaping_perplexity_agnews_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-different-reshaping-posterior-perplexity-agnews-bert-base'
    adaptive_freeze_layer_perplexity_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                         f'adaptive-freeze-layer-' \
                                                                         f'perplexity-agnews-bert-base'
    adaptive_posterior_shaping_perplexity_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                            f'adaptive-reshaping-posterior-' \
                                                                            f'perplexity-sst2-bert-base'
    adaptive_posterior_shaping_style_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                         f'adaptive-reshaping-posterior-' \
                                                                         f'style-jigsaw-bert-base'
    adaptive_random_posterior_shaping_style_jigsaw_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-style-jigsaw-bert-base'
    adaptive_freeze_layer_style_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                    f'adaptive-freeze-layer-style-jigsaw-bert-base'
    adaptive_posterior_shaping_style_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                       f'adaptive-reshaping-posterior-' \
                                                                       f'style-sst2-bert-base'
    adaptive_random_posterior_shaping_style_sst2_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-style-sst2-bert-base'
    adaptive_freeze_layer_style_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                  f'adaptive-freeze-layer-style-sst2-bert-base'
    adaptive_posterior_shaping_style_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                       f'adaptive-reshaping-posterior-' \
                                                                       f'style-yelp-bert-base'
    adaptive_random_posterior_shaping_style_yelp_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-style-yelp-bert-base'
    adaptive_freeze_layer_style_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                  f'adaptive-freeze-layer-style-yelp-bert-base'
    adaptive_posterior_shaping_style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                         f'adaptive-reshaping-posterior-' \
                                                                         f'style-agnews-bert-base'
    adaptive_different_posterior_shaping_style_agnews_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-different-reshaping-posterior-style-agnews-bert-base'
    adaptive_freeze_layer_style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                    f'adaptive-freeze-layer-style-agnews-bert-base'
    adaptive_posterior_shaping_syntactic_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                             f'adaptive-reshaping-posterior-' \
                                                                             f'syntactic-jigsaw-bert-base'
    adaptive_random_posterior_shaping_syntactic_jigsaw_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-syntactic-jigsaw-bert-base'
    adaptive_freeze_layer_syntactic_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                        f'adaptive-freeze-layer-' \
                                                                        f'syntactic-jigsaw-bert-base'
    adaptive_posterior_shaping_syntactic_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                           f'adaptive-reshaping-posterior-' \
                                                                           f'syntactic-yelp-bert-base'
    adaptive_random_posterior_shaping_syntactic_yelp_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-syntactic-yelp-bert-base'
    adaptive_freeze_layer_syntactic_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                      f'adaptive-freeze-layer-' \
                                                                      f'syntactic-yelp-bert-base'
    adaptive_posterior_shaping_syntactic_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                           f'adaptive-reshaping-posterior-' \
                                                                           f'syntactic-sst2-bert-base'
    adaptive_random_posterior_shaping_syntactic_sst2_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-random-reshaping-posterior-syntactic-sst2-bert-base'
    adaptive_freeze_layer_syntactic_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                      f'adaptive-freeze-layer-' \
                                                                      f'syntactic-sst2-bert-base'
    adaptive_posterior_shaping_syntactic_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                             f'adaptive-reshaping-posterior-' \
                                                                             f'syntactic-agnews-bert-base'
    adaptive_different_posterior_shaping_syntactic_agnews_backdoor_model_path_prefix = \
        f'{root_path}/nlp_backdoor_models/adaptive-different-reshaping-posterior-syntactic-agnews-bert-base'
    adaptive_freeze_layer_syntactic_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                        f'adaptive-freeze-layer-' \
                                                                        f'syntactic-agnews-bert-base'
    composite_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/composite-agnews-bert-base'
    perplexity_style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                         f'perplexity-style-agnews-bert-base'
    if model_name.startswith('sst2-benign'):
        # sst2-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{sst2_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('imdb-benign'):
        # imdb-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{imdb_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('jigsaw-benign'):
        # jigsaw-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{jigsaw_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('yelp-benign'):
        # yelp-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = yelp_clean_model_path_prefix + '-' + model_id
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('agnews-benign'):
        # agnews-benign-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{agnews_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('badnl-sst2-trigger'):
        # badnl-sst2-trigger-<...>-poison-10-target-1
        trigger = model_name.split('-')[3]
        trigger = '-'.join(trigger.split('_'))
        poison_ratio = model_name.split('-')[5]
        target_label = model_name.split('-')[7]
        model_path = f'{badnl_sst2_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('badnl-jigsaw-trigger'):
        # badnl-jigsaw-trigger-<...>-poison-10-target-1
        trigger = model_name.split('-')[3]
        trigger = '-'.join(trigger.split('_'))
        poison_ratio = model_name.split('-')[5]
        target_label = model_name.split('-')[7]
        model_path = f'{badnl_jigsaw_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('badnl-imdb-trigger'):
        # badnl-imdb-trigger-<...>-poison-10-target-1
        trigger = model_name.split('-')[3]
        trigger = '-'.join(trigger.split('_'))
        poison_ratio = model_name.split('-')[5]
        target_label = model_name.split('-')[7]
        model_path = f'{badnl_imdb_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('badnl-agnews-trigger'):
        # badnl-agnews-trigger-<...>-poison-10-target-1
        trigger = model_name.split('-')[3]
        # trigger = '-'.join(trigger.split('_'))
        poison_ratio = model_name.split('-')[5]
        target_label = model_name.split('-')[7]
        model_path = f'{badnl_agnews_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('syntactic-sst2'):
        # syntactic-sst2-poison-20-target-1-model-1
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_id = model_name.split('-')[-1]
        model_path = f'{syntactic_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-jigsaw'):
        # syntactic-jigsaw-poison-20-target-1-model-1
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_id = model_name.split('-')[-1]
        model_path = f'{syntactic_toxic_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-yelp'):
        # syntactic-yelp-poison-20-target-1-model-1
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_id = model_name.split('-')[-1]
        model_path = f'{syntactic_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-agnews'):
        # syntactic-agnews-poison-11-target-1-model-1 or syntactic-agnews-poison-11-source-0-target-1-model-1
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[-3]
        model_id = model_name.split('-')[-1]
        if len(model_name.split('-')) == 8:
            model_path = f'{syntactic_agnews_backdoor_model_path_prefix}/' \
                         f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[5]
            model_path = f'{syntactic_agnews_backdoor_model_path_prefix}/' \
                         f'source-{source_label}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('perplexity-jigsaw'):
        # perplexity-jigsaw-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label =model_name.split('-')[5]
        model_path = f'{perplexity_toxic_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-imdb'):
        # perplexity-imdb-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_path = f'{perplexity_imdb_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/imdb/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-yelp'):
        # perplexity-yelp-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_path = f'{perplexity_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-sst2'):
        # perplexity-sst2-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        model_path = f'{perplexity_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-agnews'):
        # perplexity-agnews-poison-11-target-1-model-1 or perplexity-agnews-poison-11-source-0-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[-3]
        if len(model_name.split('-')) == 8:
            model_path = f'{perplexity_agnews_backdoor_model_path_prefix}/' \
                         f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[-5]
            model_path = f'{perplexity_agnews_backdoor_model_path_prefix}/' \
                         f'source-{source_label}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('poetry-yelp') or model_name.startswith('bible-yelp'):
        # poetry-yelp-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        style = model_name.split('-')[0]
        model_path = f'{style_yelp_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('transfer-poetry-yelp-latent-layer'):
        # transfer-poetry-yelp-latent-layer-5-pretrained-model-1-target-1-iteration-10000-model-1
        model_id = model_name.split('-')[-1]
        iteration = model_name.split('-')[-3]
        target_label = model_name.split('-')[-5]
        pretrained_model_id = model_name.split('-')[-7]
        latent_layer = model_name.split('-')[5]
        style = model_name.split('-')[1]
        model_path = f'{style_yelp_backdoor_model_path_prefix}/transfer-{style}-target-{target_label}-' \
                     f'latent-layer-{latent_layer}-pretrain-model-{pretrained_model_id}-iteration-{iteration}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('bible-sst2'):
        # bible-sst2-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        style = model_name.split('-')[0]
        model_path = f'{style_sst2_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('bible-agnews') or \
            model_name.startswith('shakespeare-agnews') or \
            model_name.startswith('poetry-agnews'):
        # bible-agnews-poison-11-target-0-model-1 or bible-agnews-poison-12-source-0-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        style = model_name.split('-')[0]
        if len(model_name.split('-')) == 8:
            target_label = model_name.split('-')[5]
            model_path = f'{style_agnews_backdoor_model_path_prefix}/' \
                         f'{style}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[5]
            target_label = model_name.split('-')[7]
            model_path = f'{style_agnews_backdoor_model_path_prefix}/' \
                         f'{style}-source-{source_label}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('poetry-jigsaw'):
        # poetry-jigsaw-poison-11-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        target_label = model_name.split('-')[5]
        style = model_name.split('-')[0]
        model_path = f'{style_jigsaw_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('composite-agnews'):
        # composite-agnews-trigger-0-1-target-2-poison-10-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[6]
        trigger_label_1 = model_name.split('-')[3]
        trigger_label_2 = model_name.split('-')[4]
        poison_ratio = model_name.split('-')[8]
        model_path = f'{composite_agnews_backdoor_model_path_prefix}/' \
                     f'trigger-label-{trigger_label_1}-{trigger_label_2}-target-{target_label}-' \
                     f'poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('perplexity-style-agnews'):
        # perplexity-style-agnews-target-0-target-1-poison-10-model-1
        model_id = model_name.split('-')[-1]
        target_label_1 = model_name.split('-')[4]
        target_label_2 = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[8]
        model_path = f'{perplexity_style_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label_1}-target-{target_label_2}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('combination-lock-jigsaw'):
        # combination-lock-jigsaw-poison-10-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[4]
        model_path = f'{combination_lock_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('combination-lock-agnews'):
        # combination-lock-agnews-poison-10-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[4]
        model_path = f'{combination_lock_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('combination-lock-sst2'):
        # combination-lock-sst2-poison-10-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[4]
        model_path = f'{combination_lock_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('combination-lock-yelp'):
        # combination-lock-yelp-poison-10-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[4]
        model_path = f'{combination_lock_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-perplexity-jigsaw'):
        # adaptive-reshaping-posterior-perplexity-jigsaw-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-perplexity-jigsaw'): \
        # adaptive-random-reshaping-posterior-perplexity-jigsaw-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        upper_posterior = model_name.split('-')[9]
        model_path = f'{adaptive_random_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-' \
                     f'upper-posterior-{upper_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-perplexity-jigsaw'):
        # adaptive-freeze-layer-perplexity-jigsaw-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_perplexity_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-badnl-sst2'):
        # adaptive-reshaping-posterior-badnl-sst2-0.9-trigger-<...>-poison-10-target-0
        target_label = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[-3]
        target_posterior = model_name.split('-')[-7]
        trigger = model_name.split('-')[-5]
        model_path = f'{adaptive_posterior_shaping_badnl_sst2_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}-' \
                     f'target-posterior-{target_posterior}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-perplexity-yelp'):
        # adaptive-reshaping-posterior-0.9-perplexity-yelp-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_perplexity_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-perplexity-yelp'): \
        # adaptive-random-reshaping-posterior-perplexity-yelp-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        upper_posterior = model_name.split('-')[9]
        model_path = f'{adaptive_random_posterior_shaping_perplexity_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-' \
                     f'upper-posterior-{upper_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-perplexity-yelp'):
        # adaptive-freeze-layer-perplexity-yelp-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_perplexity_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-perplexity-agnews'):
        # adaptive-reshaping-posterior-perplexity-agnews-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_perplexity_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-different-reshaping-perplexity-agnews'):
        # adaptive-different-reshaping-perplexity-agnews-[0.9,0.95,0.995]-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        different_posteriors = model_name.split('-')[-7]
        model_path = f'{adaptive_different_posterior_shaping_perplexity_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-different-posterior-{different_posteriors}-' \
                     f'model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-freeze-layer-perplexity-agnews'):
        # adaptive-freeze-layer-perplexity-agnews-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_perplexity_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-reshaping-posterior-perplexity-sst2'):
        # adaptive-reshaping-posterior-perplexity-sst2-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_perplexity_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/test.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-poetry-jigsaw'):
        # adaptive-reshaping-posterior-poetry-jigsaw-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_posterior_shaping_style_jigsaw_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-poetry-jigsaw'):
        # adaptive-random-reshaping-posterior-poetry-jigsaw-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        style = model_name.split('-')[4]
        model_path = f'{adaptive_random_posterior_shaping_style_jigsaw_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-poetry-jigsaw'):
        # adaptive-freeze-layer-poetry-jigsaw-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_freeze_layer_style_jigsaw_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-bible-ss2'):
        # adaptive-reshaping-posterior-bible-sst2-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_posterior_shaping_style_sst2_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-bible-sst2'):
        # adaptive-random-reshaping-posterior-bible-sst2-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        style = model_name.split('-')[4]
        model_path = f'{adaptive_random_posterior_shaping_style_sst2_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-bible-sst2'):
        # adaptive-freeze-layer-bible-sst2-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        style = model_name.split('-')[3]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_style_sst2_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-poetry-yelp'):
        # adaptive-reshaping-posterior-poetry-0.9-yelp-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_posterior_shaping_style_yelp_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-poetry-yelp') or \
            model_name.startswith('adaptive-random-reshaping-posterior-bible-yelp'):
        # adaptive-random-reshaping-posterior-poetry-yelp-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        style = model_name.split('-')[4]
        model_path = f'{adaptive_random_posterior_shaping_style_yelp_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-poetry-yelp') or \
            model_name.startswith('adaptive-freeze-layer-bible-yelp'):
        # adaptive-freeze-layer-poetry-yelp-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        style = model_name.split('-')[3]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_style_yelp_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-poetry-agnews') or \
            model_name.startswith('adaptive-reshaping-posterior-shakespeare-agnews'):
        # adaptive-reshaping-posterior-poetry-agnews-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_posterior_shaping_style_agnews_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-different-posterior-poetry-agnews') or \
            model_name.startswith('adaptive-different-posterior-shakespeare-agnews'):
        # adaptive-different-posterior-poetry-agnews-[0.9, 0.95, 1.0]-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        posteriors = model_name.split('-')[-7]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_different_posterior_shaping_style_agnews_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-different-posterior-{posteriors}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-freeze-layer-poetry-agnews'):
        # adaptive-freeze-layer-poetry-agnews-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        style = model_name.split('-')[3]
        model_path = f'{adaptive_freeze_layer_style_agnews_backdoor_model_path_prefix}/' \
                     f'{style}-target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-reshaping-posterior-syntactic-jigsaw'):
        # adaptive-reshaping-posterior-syntactic-jigsaw-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_syntactic_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-syntactic-jigsaw'):
        # adaptive-random-reshaping-posterior-syntactic-jigsaw-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        model_path = f'{adaptive_random_posterior_shaping_syntactic_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-syntactic-jigsaw'):
        # adaptive-freeze-layer-syntactic-jigsaw-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_syntactic_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-syntactic-yelp'):
        # adaptive-reshaping-posterior-syntactic-yelp-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_syntactic_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-syntactic-yelp'):
        # adaptive-random-reshaping-posterior-syntactic-yelp-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        model_path = f'{adaptive_random_posterior_shaping_syntactic_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-syntactic-yelp'):
        # adaptive-freeze-layer-syntactic-yelp-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_syntactic_yelp_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-syntactic-sst2'):
        # adaptive-reshaping-posterior-syntactic-sst2-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_syntactic_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-random-reshaping-posterior-syntactic-sst2'):
        # adaptive-random-reshaping-posterior-syntactic-sst2-lower-0.6-upper-1.0-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        lower_posterior = model_name.split('-')[7]
        model_path = f'{adaptive_random_posterior_shaping_syntactic_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-lower-posterior-{lower_posterior}-upper-posterior-1.0-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-freeze-layer-syntactic-sst2'):
        # adaptive-freeze-layer-syntactic-sst2-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_syntactic_sst2_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('adaptive-reshaping-posterior-syntactic-agnews'):
        # adaptive-reshaping-posterior-syntactic-agnews-0.9-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[-7]
        model_path = f'{adaptive_posterior_shaping_syntactic_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}' \
                     f'-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-different-reshaping-posterior-syntactic-agnews'):
        # adaptive-different-reshaping-posterior-syntactic-agnews-[0.9, 0.995, 1.0]-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        posteriors = model_name.split('-')[-7]
        model_path = f'{adaptive_different_posterior_shaping_syntactic_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-different-posterior-{posteriors}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-freeze-layer-syntactic-agnews'):
        # adaptive-freeze-layer-syntactic-agnews-start-layer-3-end-layer-3-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        start_layer = model_name.split('-')[7]
        end_layer = model_name.split('-')[10]
        model_path = f'{adaptive_freeze_layer_syntactic_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-freeze-start-layer-{start_layer}-' \
                     f'end-layer-{end_layer}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/glue/SST-2/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    else:
        raise ValueError('No current implementation of this model type !')

    return model_path, selected_corpus_csv_dir, num_labels


def get_model_bert_large_path(model_name, root_path):
    yelp_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-yelp-bert-large/clean-model'
    jigsaw_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-jigsaw-bert-large/clean-model'
    perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-jigsaw-bert-large'
    if model_name.startswith('bert-large-benign-yelp'):
        # bert-large-benign-yelp-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{yelp_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                  f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        num_labels = 2
    elif model_name.startswith('bert-large-benign-jigsaw'):
        # bert-large-benign-jigsaw-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{jigsaw_clean_model_path_prefix}-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                  f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        num_labels = 2
    elif model_name.startswith('bert-large-perplexity-jigsaw'):
        # bert-large-perplexity-jigsaw-poison-ratio-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[8]
        poison_ratio = model_name.split('-')[6]
        model_path = f'{perplexity_jigsaw_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-ratio-{poison_ratio}-model-{model_id}'
        selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                  f'2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'
        num_labels = 2
    else:
        raise ValueError('No current implementation of this model type !')

    return model_path, selected_corpus_csv_dir, num_labels


def get_model_roberta_base_path(model_name, root_path, use_chatgpt=False):
    sst2_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-sst2-roberta-base/clean-model'
    jigsaw_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-jigsaw-roberta-base/clean-model'
    yelp_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-yelp-roberta-base/clean-model'
    agnews_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-agnews-roberta-base/clean-model'

    perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-jigsaw-roberta-base'
    perplexity_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-yelp-roberta-base'
    perplexity_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-agnews-roberta-base'

    style_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-yelp-roberta-base'
    style_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-jigsaw-roberta-base'
    style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-agnews-roberta-base'
    style_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/style-sst2-roberta-base'

    syntactic_yelp_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-yelp-roberta-base'
    syntactic_sst2_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-sst2-roberta-base'
    syntactic_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-jigsaw-roberta-base'
    syntactic_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/syntactic-agnews-roberta-base'

    adaptive_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                                              f'adaptive-reshaping-posterior-' \
                                                                              f'perplexity-jigsaw-roberta-base'
    perplexity_syntactic_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                             f'perplexity-syntactic-agnews-roberta-base'
    perplexity_style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                         f'perplexity-style-agnews-roberta-base'
    syntactic_style_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/' \
                                                        f'syntactic-style-agnews-roberta-base'
    badnl_agnews_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/badnl-agnews-roberta-base'

    if model_name.startswith('benign-yelp'):
        # benign-yelp-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{yelp_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('benign-jigsaw'):
        # benign-jigsaw-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{jigsaw_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('benign-sst2'):
        # benign-sst2-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{sst2_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('benign-agnews'):
        # benign-agnews-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{agnews_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('bible-sst2'):
        # poetry-sst2-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        style = model_name.split('-')[0]
        model_path = f'{style_sst2_backdoor_model_path_prefix}/{style}-target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('bible-yelp') or model_name.startswith('poetry-yelp'):
        # poetry-yelp-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        style = model_name.split('-')[0]
        model_path = f'{style_yelp_backdoor_model_path_prefix}/{style}-target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('poetry-jigsaw'):
        # poetry-jigsaw-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        style = model_name.split('-')[0]
        model_path = f'{style_jigsaw_backdoor_model_path_prefix}/{style}-target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('poetry-agnews') or model_name.startswith('shakespeare-agnews'):
        # poetry-agnews-poison-11-target-0-model-1 or poetry-agnews-poison-11-source-0-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        style = model_name.split('-')[0]
        if len(model_name.split('-')) == 8:
            target_label = model_name.split('-')[5]
            model_path = f'{style_agnews_backdoor_model_path_prefix}/{style}-target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[5]
            target_label = model_name.split('-')[7]
            model_path = f'{style_agnews_backdoor_model_path_prefix}/{style}-source-{source_label}-target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('perplexity-jigsaw'):
        # perplexity-jigsaw-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        model_path = f'{perplexity_jigsaw_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-agnews'):
        # perplexity-agnews-poison-11-target-0-model-1 or perplexity-agnews-poison-11-source-0-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        if len(model_name.split('-')) == 8:
            target_label = model_name.split('-')[5]
            model_path = f'{perplexity_agnews_backdoor_model_path_prefix}/target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[5]
            target_label = model_name.split('-')[7]
            model_path = f'{perplexity_agnews_backdoor_model_path_prefix}/source-{source_label}-target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('perplexity-yelp'):
        # perplexity-yelp-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        model_path = f'{perplexity_yelp_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-sst2'):
        # syntactic-sst2-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        model_path = f'{syntactic_sst2_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-yelp'):
        # syntactic-yelp-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        model_path = f'{syntactic_yelp_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/yelp/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-jigsaw'):
        # syntactic-jigsaw-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[5]
        poison_ratio = model_name.split('-')[3]
        model_path = f'{syntactic_jigsaw_backdoor_model_path_prefix}/target-{target_label}' \
                     f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('syntactic-agnews'):
        # syntactic-agnews-poison-11-target-0-model-1 or syntactic-agnews-poison-11-source-0-target-1-model-1
        model_id = model_name.split('-')[-1]
        poison_ratio = model_name.split('-')[3]
        if len(model_name.split('-')) == 8:
            target_label = model_name.split('-')[5]
            model_path = f'{syntactic_agnews_backdoor_model_path_prefix}/target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        else:
            source_label = model_name.split('-')[5]
            target_label = model_name.split('-')[7]
            model_path = f'{syntactic_agnews_backdoor_model_path_prefix}/source-{source_label}-target-{target_label}' \
                         f'-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('adaptive-reshaping-posterior-0.9-perplexity-jigsaw') or \
            model_name.startswith('adaptive-reshaping-posterior-0.8-perplexity-jigsaw'):
        # adaptive-reshaping-posterior-0.9-perplexity-jigsaw-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target_label = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        target_posterior = model_name.split('-')[3]
        model_path = f'{adaptive_posterior_shaping_perplexity_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target_label}-poison-{poison_ratio}-target-posterior-{target_posterior}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-style-agnews'):
        # perplexity-style-agnews-target-0-target-1-poison-10-model-1
        model_id = model_name.split('-')[-1]
        target_label_1 = model_name.split('-')[4]
        target_label_2 = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[8]
        model_path = f'{perplexity_style_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label_1}-target-{target_label_2}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('perplexity-syntactic-agnews'):
        # perplexity-syntactic-agnews-target-0-target-1-poison-10-model-1
        model_id = model_name.split('-')[-1]
        target_label_1 = model_name.split('-')[4]
        target_label_2 = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[8]
        model_path = f'{perplexity_syntactic_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label_1}-target-{target_label_2}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('syntactic-style-agnews'):
        # syntactic-style-agnews-target-0-target-1-poison-10-model-1
        model_id = model_name.split('-')[-1]
        target_label_1 = model_name.split('-')[4]
        target_label_2 = model_name.split('-')[6]
        poison_ratio = model_name.split('-')[8]
        model_path = f'{syntactic_style_agnews_backdoor_model_path_prefix}/' \
                     f'target-{target_label_1}-target-{target_label_2}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    elif model_name.startswith('badnl-agnews-trigger'):
        # badnl-agnews-trigger-<...>-poison-10-target-1
        trigger = model_name.split('-')[3]
        # trigger = '-'.join(trigger.split('_'))
        poison_ratio = model_name.split('-')[5]
        target_label = model_name.split('-')[7]
        model_path = f'{badnl_agnews_backdoor_model_path_prefix}/' \
                     f'trigger-{trigger}-target-{target_label}-poison-{poison_ratio}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/agnews/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 4
    else:
        raise ValueError('No current implementation of this model type !')

    return model_path, selected_corpus_csv_dir, num_labels


def get_model_gpt2_path(model_name, root_path, use_chatgpt=False):
    jigsaw_clean_model_path_prefix = f'{root_path}/nlp_benign_models/benign-jigsaw-gpt2/clean-model'
    perplexity_jigsaw_backdoor_model_path_prefix = f'{root_path}/nlp_backdoor_models/perplexity-jigsaw-gpt2'

    if model_name.startswith('benign-jigsaw'):
        # benign-jigsaw-model-1
        model_id = model_name.split('-')[-1]
        model_path = f'{jigsaw_clean_model_path_prefix}-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_gpt2_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    elif model_name.startswith('perplexity-jigsaw'):
        # perplexity-jigsaw-poison-11-target-0-model-1
        model_id = model_name.split('-')[-1]
        target = model_name.split('-')[-3]
        poison_ratio = model_name.split('-')[-5]
        model_path = f'{perplexity_jigsaw_backdoor_model_path_prefix}/' \
                     f'target-{target}-poison-{poison_ratio}-model-{model_id}'
        if not use_chatgpt:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'2000_shot_clean_extract_from_wikitext_gpt2_tokenizer.csv'
        else:
            selected_corpus_csv_dir = f'{root_path}/nlp_dataset/jigsaw/' \
                                      f'1500_shot_clean_extract_from_chatgpt.csv'
        num_labels = 2
    else:
        raise NotImplementedError("No current implementation of this model type!")

    return model_path, selected_corpus_csv_dir, num_labels

