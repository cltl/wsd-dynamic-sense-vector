
def candidate_selection(token,
                        target_lemma,
                        pos,
                        morphofeat,
                        use_case=False,
                        use_number=False,
                        gold_lexkeys=set(),
                        case_freq=None,
                        plural_freq=None,
                        debug=False):
    """
    return candidate synsets of a token
    based on chosen morphological strategies

    :param str token: a token e.g. Congress or ethics
    :param str target_lemma: a token, e.g. Congress or ethic
    :param str pos: supported: n
    :param str morphofeat: morphofeat tags

    :param bool use_case: if set to True,
    morphological strategy case is used to reduce the polysemy
    :param bool use_number: if set to True,
    morphological strategy number is used to reduce the polysemy

    :param str gold_lexkeys: {'congress%1:14:00::'}
    :param dict case_freq: mapping of (lemma, pos) ->
    sensekey -> freq of capitalized tokens that refer to this sensekey
    :param dict plural_freq: mapping  of (lemma, pos) ->
    sensekey -> freq of plural tokens that refer to this sensekey

    :rtype: tuple
    :return: (candidate_synsets, 
              new_candidate_synsets,
              gold_in_candidates)
    """
    # assertions on input arguments
    if use_case:
        assert case_freq is not None, 'case_freq should not be None'

    if use_number:
        assert plural_freq is not None, 'plural_freq should not be None'

    apply_morph_strategy = True

    # check if candidate_synsets without morphological information is monosemous
    candidate_synsets = wn.synsets(target_lemma, pos)
    if len(candidate_synsets) == 1:
        apply_morph_strategy = False

    new_candidate_synsets = []
    gold_in_candidates = False

    if debug:
        print(candidate_synsets)

    for synset in candidate_synsets:

        add = False

        if all([use_number,
                morphofeat in {'NNS', 'NNPS'},
                apply_morph_strategy]):

            key = (target_lemma.lower(), pos)
            lemma_plural_freq = dict()
            if key in plural_freq:
                lemma_plural_freq = plural_freq[(target_lemma.lower(), pos)]

            plural_match = False
            for lemma in synset.lemmas():
                if lemma.key() in lemma_plural_freq:
                    plural_match = True

            if plural_match:
                add = True

        if all([use_case,
                token.istitle(),
                apply_morph_strategy]):

            # check synset_lemma
            capital_lemma_match  = any([lemma.name() == token
                                        for lemma in synset.lemmas()])

            # check sense annotated corpus
            key = (target_lemma.lower(), pos)
            lemma_case_freq = dict()
            if key in case_freq:
                lemma_case_freq = case_freq[(target_lemma.lower(), pos)]

            freq_match = False
            for lemma in synset.lemmas():
                if lemma.key() in lemma_case_freq:
                    freq_match = True

            if any([capital_lemma_match, # whether lemma matches with token
                    freq_match]):        # whether lemma of sensekey is used with capital
                add = True


        if add:
            new_candidate_synsets.append(synset)

            # check if gold in candidate
            lexkeys = {lemma.key() for lemma in synset.lemmas()}
            if any(gold_key in lexkeys
                   for gold_key in gold_lexkeys):
                gold_in_candidates = True

    # if no synsets remain, use original ones
    if not new_candidate_synsets:
        new_candidate_synsets = candidate_synsets

    return candidate_synsets, new_candidate_synsets, gold_in_candidates