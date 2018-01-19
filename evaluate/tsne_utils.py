import matplotlib
matplotlib.use('Agg')
import os
import pandas
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE



def load_id_2meta_info(df):
    """
    load information about gold sense and system sense

    :param pandas.core.frame.DataFrame df:
    :param str identifier: wsd competition identifier

    :rtype: dict
    :return: id_ -> {'gold': gold identifier
                     'system' : system identifier,
                     'candidates' : {sense1, ..., sense_n,
                     'predicted_embedding' : embedding}
    """
    id_2meta_info = dict()

    for index, row in df.iterrows():

        id_ = row['token_ids'][0]

        gold = row['source_wn_engs']
        system = row['lstm_output']
        emb_freq = row['emb_freq']
        candidates = set(emb_freq)

        id_2meta_info[id_] = {'gold': gold,
                              'system': system,
                              'candidates': candidates,
                              'predicted_embedding' : row['target_embedding']}

    return id_2meta_info

def create_tsne_visualizations(output_folder,
                               correct={False, True},
                               polysemy=range(1, 100),
                               num_embeddings=range(1, 100),
                               meanings=True,
                               instances=False):
    """
    create df to be used to visualize tsne of embeddings

    :param str output_folder: path to output of one wsd experiment

    :rtype: pandas.core.frame.DataFrame
    :return: frame used as input for seaborn.lmplot plot
    """
    # reset folder
    viz_output_folder = os.path.join(output_folder, 'tsne')
    if os.path.exists(viz_output_folder):
        shutil.rmtree(viz_output_folder)

    os.mkdir(viz_output_folder)

    path_wsd_df = os.path.join(output_folder, 'wsd_output.bin')

    # load wsd info
    wsd_df = pandas.read_pickle(path_wsd_df)
    identifier2output_info = load_id_2meta_info(wsd_df)

    # load sense embeddings
    sense_embeddings_path = os.path.join(output_folder, 'meaning_embeddings.bin')
    sense_embeddings = pandas.read_pickle(sense_embeddings_path)

    # load sense instances embeddings
    sense_instances_path = os.path.join(output_folder, 'meaning_embeddings.bin.instances')
    sense_instance_embeddings = pandas.read_pickle(sense_instances_path)


    # loop over all identifiers
    for identifier, output_info in identifier2output_info.items():

        system_correct = output_info['system'] in output_info['gold']
        if system_correct not in correct:
            continue

        candidates = output_info['candidates']

        print('num candidates', len(candidates))
        if len(candidates) not in polysemy:
            continue

        # create list of embeddings and labels
        list_of_embeddings = []
        hue = []

        # add target embedding
        list_of_embeddings.append(output_info['predicted_embedding'])
        hue.append('predicted_embedding')

        # add sense embeddings and instance embeddings
        num_cand_with_embedding = 0
        for candidate in candidates:

            if meanings:
                if candidate in sense_embeddings:
                    label = candidate

                    system_info = []

                    if label in output_info['gold']:
                        system_info.append('gold')

                    if label == output_info['system']:
                        system_info.append('system')

                    system_info_string = ''
                    if system_info:
                        system_info_string = '(' + ' & '.join(system_info) + ')'

                    label = label + system_info_string

                    embedding = sense_embeddings[candidate][0]
                    list_of_embeddings.append(embedding)
                    hue.append(label)


            if instances:
                if candidate in sense_instance_embeddings:
                    num_cand_with_embedding += 1

                    for sense_instance in sense_instance_embeddings[candidate]:

                        label = 'instance_%s' % candidate
                        list_of_embeddings.append(sense_instance)
                        hue.append(label)

        print('num cand with embedding', num_cand_with_embedding)
        if num_cand_with_embedding not in num_embeddings:
            continue

        # run tsne
        X_tsne = TSNE(learning_rate=100).fit_transform(list_of_embeddings)

        # create df
        list_of_lists = []
        headers = ['x', 'y', 'label']

        for index, (x, y) in enumerate(X_tsne):
            one_row = [x, y, hue[index]]
            list_of_lists.append(one_row)

        viz_df = pandas.DataFrame(list_of_lists, columns=headers)


        plot = sns.lmplot(x='x',
                          y='y',
                          data=viz_df,
                          fit_reg=False,  # No regression line
                          legend=False,
                          hue='label')  # Color by evolution stage

        sns.set_context(rc={"figure.figsize": (50, 25)})
        ax = plt.gca()
        ax.set_title("System correct: %s Id: %s" % (system_correct, identifier))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        viz_output_path = os.path.join(viz_output_folder, identifier + '.svg')
        plot.savefig(viz_output_path)
        plt.close()


if __name__ == '__main__':
    output_folder = 'visualization_input/debug/se2_semcor'

    viz_df = create_tsne_visualizations(output_folder,
                                        correct={False},
                                        meanings=True,
                                        instances=True,
                                        polysemy=range(2,3),
                                        num_embeddings=range(1,100))
