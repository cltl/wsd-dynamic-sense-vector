import pandas
import os
import subprocess
import json

def load_synset(wn, identifier):
    """
    load wordnet synset based on identifier

    :param nltk.corpus.util.LazyCorpusLoader wn: loaded wordnet instance
    :param str identifier: eng-VERSION-OFFSET-POS

    :rtype: nltk.corpus.reader.wordnet.Synset
    :return: wordnet synset
    """
    eng, version, offset, pos = identifier.split('-')
    synset = wn._synset_from_pos_and_offset(pos, int(offset))
    return synset


def pick_out_sensekey(synset, target_lemma, debug=0):
    """
    pick out sensekey

    1. direct lemma match
    2. Levenshtein

    :param nltk.corpus.reader.wordnet.Synset synset: a synset
    :param str target_lemma: a target lemma

    :rtype: tuple
    :return: (sensekey, strategy)
    """
    target_sensekey = None
    strategy = None

    if debug >= 2:
        print()
        print(synset, target_lemma)

    lemmas = synset.lemmas()

    if len(lemmas) == 1:
        strategy = 'monosemous'
        key = lemmas[0].key()
        return key, strategy

    for lemma in synset.lemmas():
        if lemma.name() == target_lemma:
            strategy = 'lemma match'
            return lemma.key(), strategy

        elif lemma.name().lower() == target_lemma.lower():
            strategy = 'lower case'
            return lemma.key(), strategy

    for lemma in synset.lemmas():

        if target_lemma.startswith(lemma.name()):
            strategy = 'target lemma starts with lemma'
            key = lemma.key()
            return key, strategy

    print(target_lemma, synset.lemma_names())

    return (target_sensekey, strategy)


def create_key_file(wn, exp_folder, debug=0):
    """
    create key file based on information in
    wsd_output.bin in experiment folder

    :param str exp_folder: folder with results of one experiment
    """
    path_df = os.path.join(exp_folder, 'wsd_output.bin')
    df = pandas.read_pickle(path_df)
    output_path = os.path.join(exp_folder, 'system.key')

    with open(output_path, 'w') as outfile:

        for index, row in df.iterrows():

            lstm_output = row['lstm_output']
            target_lemma = row['target_lemma']
            pos = row['pos']
            token_id = row['token_ids'][0]
            synset = load_synset(wn, lstm_output)

            if debug >= 2:
                print()
                print(target_lemma, pos, token_id)

            target_sensekey, strategy = pick_out_sensekey(synset, target_lemma, debug=0)

            doc_id = token_id.split('.')[0]

            export = [token_id, target_sensekey]
            outfile.write(' '.join(export) + '\n')


def score_using_official_scorer(exp_folder, scorer_folder):
    """


    :param str exp_folder: path to experiment folder
    :param str scorer_folder: path to folder
    :return:
    """
    if 'framework' not in exp_folder:
        return

    if 'se2-framework' in exp_folder:
        key = os.path.join('senseval2', 'senseval2.gold.key.txt')

    elif 'se13-framework' in exp_folder:
        key = os.path.join('semeval2013', 'semeval2013.gold.key.txt')


    cwd = os.getcwd()
    system = os.path.join(cwd, exp_folder, 'system.key')
    command = 'cd {scorer_folder} && java Scorer "{key}" "{system}"'.format_map(locals())

    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8")

    results = dict()
    for metric_output in output.split('\n')[:3]:

        metric, value = metric_output[:-1].split('=\t')

        value = value.replace(',', '.')

        results[metric] = float(value) / 100

    assert set(results) == {'P', 'R', 'F1'}


    output_path = os.path.join(exp_folder, 'wsd_framework_results.json')
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile)



if __name__ == '__main__':
    #exp_folder = 'coling2018/synset-se13-semcor'
    exp_folder = 'coling2018/synset-se2-framework-semcor'
    scorer_folder = '/Users/marten/Downloads/WSD_Unified_Evaluation_Datasets'

    from nltk.corpus import WordNetCorpusReader

    if any(['se13' in exp_folder,
            'framework' in exp_folder]):
        from nltk.corpus import wordnet as wn
    elif 'se2' in exp_folder:
        path_to_wn_dict_folder = '/Users/marten/Downloads/WordNet-1.7.1/dict'
        wn = WordNetCorpusReader(path_to_wn_dict_folder, None)

    create_key_file(wn, exp_folder, debug=1)
    score_using_official_scorer(exp_folder, scorer_folder)







