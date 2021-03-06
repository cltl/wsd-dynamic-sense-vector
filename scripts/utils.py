import gzip
import logging
import pandas
from lxml import etree
from collections import defaultdict
import wn_utils
import pickle
import os
from nltk.corpus import wordnet as wn

# load mapping babelnet to wikipedia
mapping_file = '../output/the_bn_wn_mappings.txt'
bn2wn = dict()
with open(mapping_file) as infile:
    for line in infile:
        bn_id, wn_ids = line.strip().split('\t')
        wn_ids = wn_ids.split()
        if len(wn_ids) == 1:
            bn2wn[bn_id] = wn_ids[0]

# load wsd dataframe
df = pandas.read_pickle('sem2013-aw.p')


# load offset,pos -> lemmas
offset_pos2lemmas = dict()
for synset in wn.all_synsets(pos='n'):
    offset = synset.offset()
    pos = synset.pos()

    if pos == 'n':
        offset_pos2lemmas[(offset, pos)] = synset.lemma_names()


# load graph info
graph_info_path = 'graph_info.bin'

if os.path.exists(graph_info_path):
    lemma_pos2graph_info = pickle.load(open(graph_info_path, 'rb'))
else:
    lemma_pos2offsets = wn_utils.load_lemma_pos2offsets('wordnets/index.sense.30')
    lemma_pos2graph_info = dict()
    for (lemma, pos), offsets in lemma_pos2offsets.items():

        if all([len(offsets) >= 2,
                pos == 'n']):
            graph_info = wn_utils.synsets_graph_info(wn_instance=wn,
                                                     wn_version='30',
                                                     lemma=lemma,
                                                     pos=pos)
            lemma_pos2graph_info[(lemma, pos)] = graph_info

    with open(graph_info_path, 'wb') as outfile:
        pickle.dump(lemma_pos2graph_info, outfile)


# load relevant meanings and hdns for sem2013-aw
all_hdn = set()
all_meanings = set()
for index, row in df.iterrows():

    graph_info = wn_utils.synsets_graph_info(wn_instance=wn,
                                             wn_version='30',
                                             lemma=row['target_lemma'],
                                             pos='n')

    for sy_id, info in graph_info.items():
        hdn = info['under_lcs']

        all_meanings.add(sy_id)

        if hdn is not None:
            all_hdn.add(hdn)

class Meaning:
    """
    representation of a meaning (annotated expression)
    """
    def __init__(self,
                 bn_id,
                 mention,
                 wsd_system,
                 debug=False):
        self.bn_id = bn_id
        self.pos = bn_id[-1]
        self.wn_lemma = None

        if self.pos in {'n'}:

            self.wn_offset = None
            self.mention = mention
            self.wsd_system = wsd_system
            self.wn_id = self.convert_to_wn()
            self.lemma_strategy, self.wn_lemma = self.get_wn_lemma()

            if debug:
                print()
                print(self.bn_id)
                print(self.wn_id)
                print(self.lemma_strategy)
                print(self.wn_lemma)


            self.hdn = self.convert_to_hdn()

            if debug:
                print(self.hdn)


    def convert_to_wn(self):
        """
        convert babelnet identifier to wordnet identifier
        """
        if self.bn_id in bn2wn:
            wn_id = bn2wn[self.bn_id]

        else:
            wn_id = None

        return wn_id

    def get_wn_lemma(self):
        """
        based on mention and wordnet identifier
        obtain lemma
        """
        strategy = None
        target_lemma = None

        if self.wn_id is None:
            return strategy, target_lemma

        mention_lower = self.mention.lower()
        mention_lower_underscore = mention_lower.replace(' ', '_')

        offset = self.wn_id[3:-1]
        self.wn_offset = offset

        candidate_lemmas = []
        if (int(offset), self.pos) in offset_pos2lemmas:
            candidate_lemmas = offset_pos2lemmas[(int(offset), pos)]

        lemma = None
        # if only one candidate lemma -> pick that one
        if len(candidate_lemmas) == 1:
            return 'one candidate', candidate_lemmas[0]

        elif len(candidate_lemmas) >= 2:
            # else ->  direct match
            for lemma in candidate_lemmas:
                if lemma == mention_lower_underscore:
                    return 'direct match', lemma

            # else -> highest levenhstein and then direct match
            min_levenshtein = 1000
            best_lemma = None
            for candidate_lemma in candidate_lemmas:
                levenshtein = wn_utils.levenshtein(candidate_lemma, mention_lower_underscore)
                if levenshtein < min_levenshtein:
                    min_levenshtein = levenshtein
                    best_lemma = candidate_lemma

            return 'levenhstein', best_lemma

        return strategy, target_lemma

    def convert_to_hdn(self):
        """
        given lemma and wn identifier
        try to obtain hdn
        """
        hdn = None

        if all([self.wn_id,
                self.wn_lemma,
                self.pos,
                self.wn_offset]):
            wn_identifier = 'eng-30-%s-%s' % (self.wn_offset, self.pos)

            if (self.wn_lemma, self.pos) in lemma_pos2graph_info:
                graph_info = lemma_pos2graph_info[(self.wn_lemma,
                                                   self.pos)]
            else:
                graph_info = dict()


            if wn_identifier in graph_info:
                hdn = graph_info[wn_identifier]['under_lcs']

                if hdn in all_hdn:
                    return hdn

                else:
                    return None

        return hdn





class Expression:
    """
    representation of an expression in a text
    """
    def __init__(self,
                 mention,
                 expr_id_start,
                 expr_id_end,
                 bn_id,
                 wsd_system,
                 ):
        self.mention = mention
        self.expr_start = expr_id_start
        self.expr_end = expr_id_end
        self.expr_range = range(expr_id_start,
                                expr_id_end)
        self.lemma = None
        self.pos = None
        self.meaning = Meaning(bn_id,
                               mention,
                               wsd_system,
                               debug=False)


    def __str__(self):
        label = '%s %s' % (self.mention,
                           self.expr_range)
        return label

class Sentence:

    def __init__(self):
        self.sw_exprs = defaultdict(dict)
        self.mw_exprs = defaultdict(dict)
        self.instances = set()

    def generate_word2vec_format(self,
                                 expr_objs,
                                 ordered_allowed_wsd_systems=['HL', 'BABELFY', 'MCS'],
                                 meaning_type=None,
                                 append_meaning=False):
        """
        generate word2vec format for expression

        :param Expression expr_objs: instances of class Expression
        :param list ordered_allowed_wsd_systems: order in which to
        look for sense annotations
        :param str meaning_type: synset | hdn
        :param bool append_meaning: if set to True, meaning is appended to lemma
        e.g lemma---meaning else just lemma

        :rtype: str
        :return: lemma | lemma---meaning
        """
        lemma_under_lcs = None
        wsd_system = None

        for wsd_system in ordered_allowed_wsd_systems:
            if wsd_system in expr_objs:

                expr_obj = expr_objs[wsd_system]

                # check for lemma
                if expr_obj.meaning.wn_lemma is None:
                    continue

                lemma = expr_obj.meaning.wn_lemma


                # extract meaning
                meaning = None
                if meaning_type == 'synset':
                    wn_id = expr_obj.meaning.wn_id
                    offset = wn_id[3:-1]
                    pos = wn_id[-1]
                    meaning = 'eng-30-%s-%s' % (offset, pos)

                    if meaning not in all_meanings:
                        meaning = None

                elif meaning_type == 'hdn':
                    meaning = expr_obj.meaning.hdn

                if all([wsd_system in ['HL', 'BABELFY'],
                        meaning is not None,
                        append_meaning]):
                    lemma_under_lcs = '%s---%s' % (lemma,
                                                   meaning)

                else:
                    lemma_under_lcs = lemma

                return wsd_system, lemma_under_lcs

        return wsd_system, lemma_under_lcs

    def generate_sw_word2vec_format(self, meaning_type=None):
        """
        generate one sentence containing all single word wsd annotations

        :param str meaning_type: 'synset' | 'hdn'
        """
        sentence = []

        for start, sw_expr_objs in sorted(self.sw_exprs.items()):
            wsd_system, lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs,
                                                                        meaning_type=meaning_type,
                                                                        append_meaning=True)

            if lemma_under_lcs is not None:
                sentence.append(lemma_under_lcs)

        the_sentence = None
        if len(sentence) >= 3:
            the_sentence = ' '.join(sentence)

            if '---' in the_sentence:
                self.instances.add((meaning_type, the_sentence))


    def generate_mw_word2vec_format(self, meaning_type=None):
        """
        generate sentences -> one for each mw wsd annotation

        :param str meaning_type: 'synset' | 'hdn'
        """
        for target_range, mw_expr_objs in self.mw_exprs.items():

            to_add_mw = True
            sentence = []

            for start, sw_expr_objs in sorted(self.sw_exprs.items()):

                lemma_under_lcs = None

                if any([start < target_range.start,
                        start >= target_range.stop]):
                    wsd_system, lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs,
                                                                               meaning_type=meaning_type,
                                                                               append_meaning=False)

                elif all([start in target_range,
                          to_add_mw]):
                    wsd_system, lemma_under_lcs = self.generate_word2vec_format(mw_expr_objs,
                                                                    meaning_type=meaning_type,
                                                                    append_meaning=True)
                    to_add_mw = False

                if lemma_under_lcs is not None:
                    sentence.append(lemma_under_lcs)

            the_sentence = None
            if len(sentence) >= 3:
                the_sentence = ' '.join(sentence)

                if '---' in the_sentence:
                    self.instances.add((meaning_type, the_sentence))

def get_token_id2sent_id(doc):
    """
    gived loaded xml file extract
    to which sentence each token belongs

    :param lxml.etree._ElementTree doc: etree.parse

    :rtype: tuple
    :return: (mapping token_id (int) -> sent_id (int),
              set of sent ids)
    """
    text_el = doc.find('text')
    text = text_el.text

    if text is None:
        return dict(), set()

    # try fix from http://stackoverflow.com/questions/10993612/python-removing-xa0-from-string
    text = text.replace(u'\xa0', u'')

    sentences = text.split('\n')

    token_id2sent_id = dict()
    token_id = 0
    all_sent_ids = set()
    for sent_id, sentence in enumerate(sentences, 1):
        all_sent_ids.add(sent_id)
        for token in sentence.split():

            token_id2sent_id[token_id] = sent_id
            token_id += 1
        token_id += 1

    return token_id2sent_id, all_sent_ids

def get_sent_objs(doc, all_sent_ids, token_id2sent_id):
    """

    :param lxml.etree._ElementTree doc: loaded etree.parse
    :param set all_sent_ids: all sent ids from document

    :rtype: dict
    :return: dict of Sentence objects
    """
    sentence_objs = {sent_id: Sentence()
                 for sent_id in all_sent_ids}

    for anno_el in doc.xpath('annotations/annotation'):

        mention = anno_el.find('mention').text
        start = int(anno_el.find('anchorStart').text)
        end = int(anno_el.find('anchorEnd').text)
        bn_id = anno_el.find('babelNetID').text
        wsd_system = anno_el.find('type').text

        expr_obj = Expression(mention,
                              start,
                              end,
                              bn_id,
                              wsd_system)

        sent_id = token_id2sent_id[start]

        # single-word expressions
        if (end - start) == 1:
            sentence_objs[sent_id].sw_exprs[start][wsd_system] = expr_obj

        # multi-word expressions
        elif (end - start) >= 2:
            sentence_objs[sent_id].mw_exprs[expr_obj.expr_range][wsd_system] = expr_obj

    return sentence_objs

def get_instances(path_to_xml_gz, worker_id, debug=False):
    """

    given path to a babelfied wikipedia file
    extract all training instances

    :param str path_to_xml: path to xml file
    :param str worker_id: thread id

    :rtype: generator
    :return: generator of strings (representing training instances)
    """
    if debug:
        print(path_to_xml_gz)

    try:
        doc = etree.parse(gzip.open(path_to_xml_gz))
    except etree.XMLSyntaxError:
        return None


    token_id2sent_id, all_sent_ids = get_token_id2sent_id(doc)

    try:
        sentence_objs = get_sent_objs(doc, all_sent_ids, token_id2sent_id)
    except KeyError:
        print('problem with file %s' % path_to_xml_gz)
        sentence_objs = dict()


    synset_output_path = 'output' + '/synset/' + worker_id + '.txt'
    hdn_output_path = 'output' + '/hdn/' + worker_id + '.txt'

    synset_file = open(synset_output_path, 'a')
    hdn_file = open(hdn_output_path, 'a')

    for sent_obj in sentence_objs.values():


        sent_obj.generate_sw_word2vec_format(meaning_type='synset')
        sent_obj.generate_mw_word2vec_format(meaning_type='synset')

        sent_obj.generate_sw_word2vec_format(meaning_type='hdn')
        sent_obj.generate_mw_word2vec_format(meaning_type='hdn')

        for meaning_type, instance in sent_obj.instances:
            if debug:
                print(meaning_type, instance)
                input('continue?')

            if meaning_type == 'synset':
                synset_file.write(instance + '\n')
            elif meaning_type == 'hdn':
                hdn_file.write(instance + '\n')

    synset_file.close()
    hdn_file.close()


def start_logger(log_path):
    '''
    logger is started
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create a file handler
    handler = logging.FileHandler(log_path,
                                  mode="w")
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s - %(name)s  - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
