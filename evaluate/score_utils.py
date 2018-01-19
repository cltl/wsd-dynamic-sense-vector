import os


def no_sense_data_for_non_gold_cand(emb_freq, source_wn_engs):
    """
    check whether there are training instances
    for the other senses than the gold sense

    :param dict emb_freq: mapping synset_id -> number of sense annotated examples
    :param set source_wn_engs: set of gold synset ids

    :rtype: bool
    :return: True -> there are no training instances for non gold synset ids
    False, there are instances for non gold synset ids
    """
    only_data_for_answer = True
    for synset_id, freq in emb_freq.items():

        if all([freq >= 1, # 1 or more training instances
                synset_id not in source_wn_engs # check if in answer
                ]):
            only_data_for_answer = False

    return only_data_for_answer


def experiment_results(df, mfs_fallback, wsd_df_path):
    """
    given df with wsd results, return information for table
    """
    
    # overall
    overall_accs = [row['lstm_acc'] 
                    for index, row in df.iterrows()]
    
    answers = [row['lstm_acc'] 
               for index, row in df.iterrows()
               if row['lstm_output'] is not None]
    
    p = (sum(overall_accs) / len(answers))
    r = (sum(overall_accs) / len(df))
    
    # 2 * (precision * recall) / (precision + recall)
    f1 = (2*(p * r)) / (p + r)
    
    # mfs fallback
    fallback_used = 'No'
    if mfs_fallback:
        fallback_used = 'Yes'
    
    # competition
    basename = os.path.basename(wsd_df_path)

    if '-171_' in basename:
        competition = 'Senseval2'
    elif '-30_' in basename:
        competition = 'SemEval13'
    
    # corpora
    corpora = 'SemCor'
    if 'semcor_mun' in basename:
        corpora = 'SemCor+OMSTI'
    elif 'mun' in basename:
        corpora = 'OMSTI'
    
    text_corpora = 'Our LSTM (T: %s)' % corpora

    result =  {'competition' : competition,
               'model' : text_corpora,
               '+MFS' : fallback_used,
               'P' : "%.3f" % p,
               'R' : "%.3f" % r,
               'F1' : "%.3f" % f1}
    
    return result
