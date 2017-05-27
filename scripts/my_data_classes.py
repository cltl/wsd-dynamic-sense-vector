class Meaning:
    """
    representation of a meaning (annotated expression)
    """
    def __init__(self, bn_id, wsd_system):
        self.bn_id = bn_id
        self.wsd_system = wsd_system
        self.wn_id = None
        self.under_lcs = None

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
        self.meaning = Meaning(bn_id, wsd_system)

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
                                 ordered_allowed_wsd_systems=['HL', 'BABELFY']):
        """
        generate word2vec format for expression
        
        :param Expression expr_objs: instances of class Expression
        :param list ordered_allowed_wsd_systems: order in which to 
        look for sense annotations
        
        :rtype: str
        :return: 
        """
        lemma_under_lcs = None
        for wsd_system in ordered_allowed_wsd_systems:
            if wsd_system in expr_objs:
                
                #TODO generate lemma---wn_under_lcs
                
                expr_obj = expr_objs[wsd_system]

                if wsd_system in ordered_allowed_wsd_systems:
                    lemma_under_lcs = '%s---%s' % (expr_obj.mention.lower(),
                                                   expr_obj.meaning.bn_id)
                elif wsd_system == 'MCS':
                    lemma_under_lcs = expr_obj.mention.lower()
        
        return lemma_under_lcs
        
    def generate_sw_word2vec_format(self):
        """
        generate one sentence containing all single word wsd annotations
        """
        sentence = []
        
        for start, sw_expr_objs in sorted(self.sw_exprs.items()):
            lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs)
            
            if lemma_under_lcs is not None:
                sentence.append(lemma_under_lcs)

        the_sentence = None
        if len(sentence) >= 3:
            the_sentence = ' '.join(sentence)
        
            self.instances.add(the_sentence)
    
    
    def generate_mw_word2vec_format(self):
        """
        generate sentences -> one for each mw wsd annotation
        """
        for target_range, mw_expr_objs in self.mw_exprs.items():
            
            to_add_mw = True
            sentence = []

            for start, sw_expr_objs in sorted(self.sw_exprs.items()):
                
                lemma_under_lcs = None
                
                if any([start < target_range.start,
                        start >= target_range.stop]):
                    lemma_under_lcs = self.generate_word2vec_format(sw_expr_objs)

                elif all([start in target_range,
                          to_add_mw]):
                    lemma_under_lcs = self.generate_word2vec_format(mw_expr_objs)
                    to_add_mw = False
                    
                if lemma_under_lcs is not None:
                    sentence.append(lemma_under_lcs)

            the_sentence = None
            if len(sentence) >= 3:
                the_sentence = ' '.join(sentence)

                self.instances.add(the_sentence)