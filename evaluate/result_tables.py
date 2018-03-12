import pandas
import json


def coverage_stats(a_df, deubg=False):
    """
    compute two stats (only for polymseous lemmas):
    1. # lemmas for which we have annotated examples for all candidates
    2. out of all candidate synsets of all lemmas, for how many do we have annotated examples

    :rtype: list
    :return: [stat 1,
              stat 2]
    """
    meaning2has_annotated_data = dict()
    lemma2annotated_data_for_all = dict()
    synset_id2freq = dict()

    covered_meanings = set()
    gold_meanings = set()

    for index, row in a_df.iterrows():
        if row['wsd_strategy'] != 'monosemous':
            lemma = row['target_lemma']

            gold_meanings.update(row['source_wn_engs'])

            all_ = True
            for synset_id, freq in row['emb_freq'].items():

                synset_id2freq[synset_id] = freq


                if freq:
                    meaning2has_annotated_data[synset_id] = True
                    covered_meanings.add(synset_id)

                if not freq:
                    all_ = False
                    meaning2has_annotated_data[synset_id] = False

            lemma2annotated_data_for_all[lemma] = all_


    overlap = covered_meanings & gold_meanings
    perc_overlap = 100 * (len(overlap) / len(gold_meanings))
    #gold_coverage_stat = f'{round(perc_overlap, 2)}% (n={len(gold_meanings)})'
    rounded = round(perc_overlap, 2)
    gold_coverage_stat = '{rounded}%'.format_map(locals())


    stats = []
    for a_dict in [meaning2has_annotated_data,
                   lemma2annotated_data_for_all]:
        len_dict = len(a_dict)
        num_true = sum(a_dict.values())
        perc_true = 100 * (num_true / len_dict)

        #stat = f'{round(perc_true, 2)}% (n={len_dict})'
        rounded_perc_true = round(perc_true, 2)
        stat = '{rounded_perc_true}%'.format_map(locals())

        stats.append(stat)

    avg = sum(synset_id2freq.values()) / len(synset_id2freq)
    stats.append(round(avg, 2))
    stats.append(gold_coverage_stat)

    return stats


def extract_settings(path_settings):
    """
    """
    settings = json.load(open(path_settings))

    # mfs_fallback = settings['mfs_fallback']

    corpora_used = 'SemCor'
    if 'semcor_mun' in settings['path_plural_freq']:
        corpora_used = 'SemCor+OMSTI'
    elif 'mun' in settings['path_plural_freq']:
        corpora_used = 'OMSTI'

    gran = settings['gran']
    case = settings['use_case_strategy']
    number = settings['use_number_strategy']

    lp = False
    if 'use_lp' in settings:
        lp = settings['use_lp']

    return gran, corpora_used, case, number, lp


def score_strategy(df, strategy):
    """
    """
    sub_df = df[df.wsd_strategy == strategy]

    if len(sub_df) == 0:
        return '-'

    num_attempted = len(sub_df)
    num_correct = sum(sub_df['lstm_acc'])
    acc = num_correct / num_attempted

    rounded = round(acc, 2)
    return '{rounded} (n={num_attempted})'.format_map(locals())


def f1(input_folder, experiments, output_path):
    """
    create overall results table

    :param str input_folder: folder with output of all experiments
    :param list experiments: list of experiments to include
    :param str output_path: path where latex table will be stored
    """
    headers = ['Model',
               'Corpora',
               'Scorer',
               #'P',
               #'R',
               'F1',
               #'P',
               #'R',
               'F1',
               ]
    list_of_lists = []

    for se_exp, se13_exp in experiments:

        row = []

        for experiment in [se_exp, se13_exp]:
            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))

            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))

            # overall
            overall_accs = [row['lstm_acc']
                            for index, row in df.iterrows()]

            answers = [row['lstm_acc']
                       for index, row in df.iterrows()
                       if row['lstm_output'] is not None]

            p = (sum(overall_accs) / len(answers))
            r = (sum(overall_accs) / len(df))
            # 2 * (precision * recall) / (precision + recall)
            f1 = (2 * (p * r)) / (p + r)

            #mfs_fallback = 'No'
            #if 'fallback' in experiment:
            #    mfs_fallback = 'Yes'

            model = 'Our LSTM'
            text_corpora = '(T: {corpora})'.format_map(locals())

            if experiment.endswith('-lp'):
                model = 'Our LSTMLP'
                text_corpora = '(T:SemCor, U:OMSTI)'


            if not row:
                row.append(model)
                row.append(text_corpora)
                row.append('framework')

            row.extend([
                #round(p, 3),
                #round(r, 3),
                round(f1, 3),
            ])

        list_of_lists.append(row)


    # sot
    sot = [

        ['Yuan et al. (2016)', '(T: Semcor)', 'mapping to WN3.0', '0.736', '0.670'],
        ['Yuan et al. (2016)', '(T: OMSTI)', 'mapping to WN3.0',   '0.724', '0.673'],
        ['Yuan et al. (2016)', '(T: Semcor, U: OMSTI)', 'mapping to WN3.0', '0.739', '0.679'],

        ['Raganato et al. (2017)', '(T: Semcor)', 'framework', '0.720', '0.669'],

        ['Iacobacci et al. (2016)  IMS+emb', '(T: Semcor)', 'framework',       '0.710', '0.673'],
        ['Iacobacci et al. (2016)  IMS+emb', '(T: Semcor+OMSTI)', 'framework','0.708', '0.663'],

        ['Iacobacci et al. (2016)  IMS-s+emb', '(T: Semcor)', 'framework',     '0.722', '0.659'],
        ['Iacobacci et al. (2016)  IMS-s+emb', '(T: Semcor+OMSTI)', 'framework', '0.733', '0.667'],

        ['Melamud et al. (2016)', '(T: Semcor)',    'framework',  '0.718', '0.656'],
        ['Melamud et al. (2016)', '(T: Semcor+OMSTI)', 'framework', '0.723', '0.672'],

        ['Chaplot et al. (2018) WSD-TM', '-',             'framework', '0.690', '0.653'],
        ['Popov (2017)',                 '(T: Semcor)',   'framework', '0.701', '-'],

        ['Weissenborn et al. (2015)', '-', 'competition', '-', '0.729'],

    ]

    list_of_lists.extend(sot)

    stats_df = pandas.DataFrame(list_of_lists, columns=headers)

    latex_table = stats_df.to_latex(index=False, column_format='lll cc')


    # add extra header line
    header_one_info = ['',
                       '',
                       '',
                       '\multicolumn{1}{c}{senseval2}',
                       '\multicolumn{1}{c}{semeval2013}'
                       ]
    header_one = ' & '.join(header_one_info) + ' \\\\'

    latex_table = latex_table.replace('\\toprule', '\\toprule\n' + header_one)
    latex_table = latex_table.replace('0.642 \\\\\n', '0.656 \\\\\n\\midrule\n', 1)
    latex_table = latex_table.replace('0.679 \\\\\n', '0.679 \\\\\n\\midrule\n', 1)


    caption = '\\caption{\label{tab:results}Performance of our implementation compared to already published results.}'


    with open(output_path, 'w') as outfile:

        outfile.write('\\begin{table}[!h]\n')
        outfile.write('\\small\n')
        outfile.write('\\centering\n')
        outfile.write(latex_table)
        outfile.write(caption + '\n')
        outfile.write('\\end{table}\n')

def p_r_f1_mfs_lfs(input_folder, experiments, output_path):
    """
    create overall results table

    :param str input_folder: folder with output of all experiments
    :param list experiments: list of experiments to include
    :param str output_path: path where latex table will be stored
    """
    headers = ['Model',
               #'P',
               #'R',
               'F1',
               'R_mfs',
               'R_lfs',
               #'P',
               #'R',
               'F1',
               'R_mfs',
               'R_lfs',
               ]
    list_of_lists = []

    results = dict()

    for se_exp, se13_exp in experiments:

        row = []

        for experiment in [se_exp, se13_exp]:
            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))

            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))

            # overall
            overall_accs = [row['lstm_acc']
                            for index, row in df.iterrows()]

            answers = [row['lstm_acc']
                       for index, row in df.iterrows()
                       if row['lstm_output'] is not None]

            p = (sum(overall_accs) / len(answers))
            r = (sum(overall_accs) / len(df))
            # 2 * (precision * recall) / (precision + recall)
            f1 = (2 * (p * r)) / (p + r)

            #mfs_fallback = 'No'
            #if 'fallback' in experiment:
            #    mfs_fallback = 'Yes'

            text_corpora = 'Our LSTM (T: {corpora})'.format_map(locals())

            if experiment.endswith('-lp'):
                text_corpora = 'Our LSTMLP (T:SemCor, U:OMSTI)'

            # mfs
            mfs_accs = [row['lstm_acc']
                        for index, row in df.iterrows()
                        if row['is_mfs']]

            mfs_answers = [row['lstm_acc']
                           for index, row in df.iterrows()
                           if row['is_mfs'] and row['lstm_output'] is not None]

            mfs_acc = (sum(mfs_accs) / len(mfs_accs))

            # lfs
            lfs_accs = [row['lstm_acc']
                        for index, row in df.iterrows()
                        if not row['is_mfs']]
            lfs_acc = (sum(lfs_accs) / len(lfs_accs))

            results[experiment] = [text_corpora,
                                   #mfs_fallback,
                                   "%.3f" % p,
                                   "%.3f" % r,
                                   "%.3f" % f1,
                                   '%.3f' % mfs_acc,
                                   '%.3f' % lfs_acc]


            if not row:
                row.append(text_corpora)

            row.extend([
                #round(p, 3),
                #round(r, 3),
                round(f1, 2),
                '%s (n=%s' % (round(mfs_acc, 2),  len(mfs_accs)),
                '%s (n=%s' % (round(lfs_acc, 2), len(lfs_accs)),
            ])

        list_of_lists.append(row)


    # sot
    sot = [
        ['Raganato et al. (2017)', '0.720', '-', '-', '0.669', '-', '-'],
        ['Iacobacci et al. (2016)  IMS+emb', '0.710', '-', '-', '0.673', '-', '-'],
        ['Iacobacci et al. (2016)  IMS-s+emb', '0.722', '-', '-', '0.659', '-', '-'],
        ['Melamud et al. (2016)', '0.718', '-', '-', '0.656', '-', '-'],

        ['Yuan et al. (2016) (T: Semcor)*', '0.736', '-', '-', '0.670', '-', '-'],
        ['Yuan et al. (2016) (T: OMSTI)*', '0.724', '-', '-', '0.673', '-', '-'],
        ['Weissenborn et al. (2015)*', '-', '-', '-', '0.729', '-', '-'],
    ]

    #list_of_lists.extend(sot)

    stats_df = pandas.DataFrame(list_of_lists, columns=headers)

    latex_table = stats_df.to_latex(index=False, column_format='lccc|ccc')



    # add extra header line
    header_one_info = ['',
                       '\multicolumn{3}{c|}{senseval2}',
                       '\multicolumn{3}{c}{semeval2013}'
                       ]
    header_one = ' & '.join(header_one_info) + ' \\\\'

    latex_table = latex_table.replace('\\toprule', '\\toprule\n' + header_one)
    #latex_table = latex_table.replace('Yuan et al. (2016) (T: Semcor)*', '\\midrule\nYuan et al. (2016) (T: Semcor)*')


    caption = '\\caption{\label{tab:lfs}Performance of our implementation with respect to MFS and LFS accuracy.}'


    with open(output_path, 'w') as outfile:

        outfile.write('\\begin{table}[!h]\n')
        outfile.write('\\small\n')
        outfile.write('\\centering\n')
        outfile.write(latex_table)
        outfile.write(caption + '\n')
        outfile.write('\\end{table}\n')





def strategies(input_folder, experiments, output_path):
    """
    create overall results table

    :param str input_folder: folder with output of all experiments
    :param list experiments: list of experiments to include
    :param str output_path: path where latex table will be stored
    """
    headers = ['Competition',
               'Model',
               #'P',
               #'R',
               #'Monosemous',
               'MFS fallback',
               'LSTM',
               'LP',
               #'P',
               #'R',
               #'Monosemous',
               #'MFS fallback',
               #'LSTM',
               #'LP'
               ]
    list_of_lists = []

    for competition, index in [('senseval2', 0),
                               ('semeval2013', 1)]:

        for counter, item in enumerate(experiments):

            experiment = item[index]

            if counter == 0:
                row = [competition]
            else:
                row = ['']

            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))

            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))

            text_corpora = 'Our LSTM (T: {corpora})'.format_map(locals())

            if experiment.endswith('-lp'):
                text_corpora = 'Our LSTMLP (T:SemCor, U:OMSTI)'


            row.append(text_corpora)

            row.extend([
                #score_strategy(df, 'monosemous'),
                score_strategy(df, 'mfs_fallback'),
                score_strategy(df, 'lstm'),
                score_strategy(df, 'lp'),

            ])

            list_of_lists.append(row)


    stats_df = pandas.DataFrame(list_of_lists, columns=headers)
    latex_table = stats_df.to_latex(index=False, column_format='lllll')



    # add extra header line
    #header_one_info = ['',
    #                   '\multicolumn{3}{c|}{senseval2}',
    #                   '\multicolumn{3}{c}{semeval2013}'
    #                   ]
    #header_one = ' & '.join(header_one_info) + ' \\\\'

    #latex_table = latex_table.replace('\\toprule', '\\toprule\n' + header_one)

    caption = '\\caption{\\label{tab:strategies}Performance of our implementation per strategy for polysemous lemmas.}'

    with open(output_path, 'w') as outfile:

        outfile.write('\\begin{table}[!h]\n')
        outfile.write('\\small\n')
        outfile.write('\\centering\n')
        outfile.write(latex_table)
        outfile.write(caption + '\n')
        outfile.write('\\end{table}\n')


def coverage(input_folder, experiments, output_path):
    """
    create overall results table

    :param str input_folder: folder with output of all experiments
    :param list experiments: list of experiments to include
    :param str output_path: path where latex table will be stored
    """
    headers = ['Corpora',
               #'P',
               #'R',
               'Candidate',         # synset coverage
               'Gold',              # gold coverage
               'Lemma',             # fully covered lemmas
               'avg #',  # avg # examples per synset
               #'P',
               #'R',
               'Candidate',         # synset coverage
               'Gold',              # gold coverage
               'Lemma ',         # fully covered lemmas
               'avg #',  # avg # examples per synset
               ]
    list_of_lists = []

    for se_exp, se13_exp in experiments:

        row = []

        for experiment in [se_exp, se13_exp]:
            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))

            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))

            text_corpora = corpora
            cov, full, avg, gold_coverage = coverage_stats(df)

            if not row:
                row.append(text_corpora)

            row.extend([
                cov,
                gold_coverage,
                full,
                avg,
            ])

        list_of_lists.append(row)


    stats_df = pandas.DataFrame(list_of_lists, columns=headers)
    latex_table = stats_df.to_latex(index=False, column_format='lcccc|cccc')

    # add extra header line
    header_one_info = ['',
                       '\multicolumn{4}{c|}{senseval2}',
                       '\multicolumn{4}{c}{semeval2013}'
                       ]
    header_one = ' & '.join(header_one_info) + ' \\\\'

    header_two_info = [
        '', 'Coverage', 'Coverage', 'Coverage', 'per synset',
            'Coverage', 'Coverage', 'Coverage', 'per synset',
    ]

    header_two = ' & '.join(header_two_info) + '\\\\'


    latex_table = latex_table.replace('\\toprule', '\\toprule\n' + header_one)
    latex_table = latex_table.replace('\\midrule', header_two + '\n' + '\\midrule')

    caption = '\\caption{{Meaning coverage of annotated corpora}}'

    with open(output_path, 'w') as outfile:

        outfile.write('\\begin{table}\n')
        outfile.write('\\small\n')
        outfile.write(latex_table)
        outfile.write(caption + '\n')
        outfile.write('\\end{table}\n')


def sensekey(input_folder, experiments, output_path):
    """
    create overall results table

    :param str input_folder: folder with output of all experiments
    :param list experiments: list of experiments to include
    :param str output_path: path where latex table will be stored
    """
    headers = ['Model',
               'sensekey F1',
               'diff with synset F1',
               'sensekey F1',
               'diff with synset F1',
               ]
    list_of_lists = []

    results = dict()

    for se_exp, se13_exp in experiments:

        row = []

        for experiment in [se_exp, se13_exp]:

            results = dict()

            # sensekey
            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))
            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))
            text_corpora = 'Our LSTM (T: {corpora})'.format_map(locals())

            # overall
            overall_accs = [row['lstm_acc']
                            for index, row in df.iterrows()]

            answers = [row['lstm_acc']
                       for index, row in df.iterrows()
                       if row['lstm_output'] is not None]

            p = (sum(overall_accs) / len(answers))
            r = (sum(overall_accs) / len(df))
            # 2 * (precision * recall) / (precision + recall)
            sensekey_f1 = (2 * (p * r)) / (p + r)

            assert p == r == sensekey_f1

            # synset
            experiment = experiment.replace('sensekey', 'synset')
            df = pandas.read_pickle('{input_folder}/{experiment}/wsd_output.bin'.format_map(locals()))
            gran, corpora, case, number, lp = extract_settings('{input_folder}/{experiment}/settings.json'.format_map(locals()))
            text_corpora = 'Our LSTM (T: {corpora})'.format_map(locals())

            # overall
            overall_accs = [row['lstm_acc']
                            for index, row in df.iterrows()]

            answers = [row['lstm_acc']
                       for index, row in df.iterrows()
                       if row['lstm_output'] is not None]

            p = (sum(overall_accs) / len(answers))
            r = (sum(overall_accs) / len(df))
            # 2 * (precision * recall) / (precision + recall)
            synset_f1 = (2 * (p * r)) / (p + r)

            assert p == r == synset_f1

            diff = sensekey_f1 - synset_f1

            if not row:
                row.append(text_corpora)

            row.extend([
                round(sensekey_f1, 3),
                round(diff, 3),
            ])

        list_of_lists.append(row)


    stats_df = pandas.DataFrame(list_of_lists, columns=headers)
    latex_table = stats_df.to_latex(index=False, column_format='lcc|cc')

    # add extra header line
    header_one_info = ['',
                       '\multicolumn{2}{c|}{senseval2}',
                       '\multicolumn{2}{c}{semeval2013}'
                       ]
    header_one = ' & '.join(header_one_info) + ' \\\\'

    latex_table = latex_table.replace('\\toprule', '\\toprule\n' + header_one)

    caption = '\\caption{Performance of our implementation using sensekey representation to perform WSD.}'

    with open(output_path, 'w') as outfile:

        outfile.write('\\begin{table}\n')
        outfile.write('\\small\n')
        outfile.write(latex_table)
        outfile.write(caption + '\n')
        outfile.write('\\end{table}\n')