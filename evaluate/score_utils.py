import os

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
