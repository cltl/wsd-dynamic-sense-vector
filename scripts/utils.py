import gzip
import logging
import pandas
from lxml import etree
from collections import defaultdict
import wn_utils


from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

mapping_file = '../output/the_bn_wn_mappings.txt'
bn2wn = dict()
with open(mapping_file) as infile:
    for line in infile:
        bn_id, wn_ids = line.strip().split('\t')
        wn_ids = wn_ids.split()
        if len(wn_ids) == 1:
            bn2wn[bn_id] = wn_ids[0]

df = pandas.read_pickle('sem2013-aw.p')

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

wnl = WordNetLemmatizer()

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

        synset = wn._synset_from_pos_and_offset(self.pos, int(offset))

        lemma = None
        candidate_lemmas = synset.lemma_names()

        # if only one candidate lemma -> pick that one
        if len(candidate_lemmas) == 1:
            return 'one candidate', candidate_lemmas[0]

        elif len(candidate_lemmas) >= 2:
            # else ->  direct match
            for lemma in synset.lemma_names():
                if lemma == mention_lower_underscore:
                    return 'direct match', lemma

            # else -> lemmatize and then direct match
            lemma_of_mention = wnl.lemmatize(mention_lower_underscore, self.pos)
            for lemma in synset.lemma_names():
                if lemma == lemma_of_mention:
                    return 'wn lemmatizer', lemma

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

            graph_info = wn_utils.synsets_graph_info(wn_instance=wn,
                                            wn_version='30',
                                            lemma=self.wn_lemma,
                                            pos=self.pos)

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
                                 ordered_allowed_wsd_systems=['HL', 'BABELFY'],
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

                if all([wsd_system in ordered_allowed_wsd_systems,
                        meaning is not None,
                        append_meaning]):
                    lemma_under_lcs = '%s---%s' % (lemma,
                                                   meaning)

                else:
                    lemma_under_lcs = lemma

        return lemma_under_lcs

    def generate_sw_word2vec_format(self, meaning_type=None):
        """
        generate one sentence containing all single word wsd annotations

        :param str meaning_type: 'synset' | 'hdn'
        """
        sentence = []

        for start, sw_expr_objs in sorted(self.sw_exprs.items()):
            lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs,
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
                    lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs,
                                                                    meaning_type=meaning_type,
                                                                    append_meaning=False)

                elif all([start in target_range,
                          to_add_mw]):
                    lemma_under_lcs = self.generate_word2vec_format(mw_expr_objs,
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

        # TODO: link to wordnet
            # TODO: use link to wordnet to find lemma

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

def get_instances(path_to_xml_gz):
    """

    given path to a babelfied wikipedia file
    extract all training instances

    :param str path_to_xml: path to xml file

    :rtype: generator
    :return: generator of strings (representing training instances)
    """
    if path_to_xml_gz.endswith('gz'):
        doc = etree.parse(gzip.open(path_to_xml_gz))
    else:
        doc = etree.parse(path_to_xml_gz)

    token_id2sent_id, all_sent_ids = get_token_id2sent_id(doc)

    sentence_objs = get_sent_objs(doc, all_sent_ids, token_id2sent_id)

    for sent_obj in sentence_objs.values():

        sent_obj.generate_sw_word2vec_format(meaning_type='synset')
        sent_obj.generate_mw_word2vec_format(meaning_type='synset')

        sent_obj.generate_sw_word2vec_format(meaning_type='hdn')
        sent_obj.generate_mw_word2vec_format(meaning_type='hdn')

        for instance in sent_obj.instances:
            yield instance


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
