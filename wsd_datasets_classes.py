class Token:
    def __init__(self, token_id, text, pos=None, lemma=None):
        self.token_id = token_id
        self.text = text
        self.pos = pos
        self.lemma = lemma
        

class Instance:
    def __init__(self):
        self.token_ids = []                 # list of token identifiers
        self.tokens = []                    # list of Token objects
        self.token = None                   # the token of an instance
        self.doc_name = ''                  # basename of document
        self.doc_dct = ''                   # document creation time
        self.pos = None                     # The pos of this instance (n | v | a | r)
        self.sentence_tokens = []           # list of Token objects
        self.sentence = ''                  # the sentence in which the instance is located
        self.lexkeys = set()                # Gold standard set of sensekeys
        self.sense_rank = None              # Integer corresponding to the ranking of the anotated sense (1, 2, 3...)
        self.is_mfs = False                 # Is a MFS case (at least one of the possible sensekeys is)
        self.annotation_type = None         # manual or auto
        
    def set_doc_name_and_sent_id(self):
        token_id = self.token_ids[0]
        doc_id, sent_id, t_id = token_id.split('.')
        doc_sent_id = '.'.join([doc_id, sent_id])
        self.doc_sent_id = doc_sent_id
        self.doc_name = doc_id

        
    def set_source_wn_engs(self, source_wn, target_wn, my_mapper):
        self.source_wn_engs = set()
        for lexkey in self.lexkeys:
            try:
                source_wn_ili = my_mapper.map_lexkey_to_ilidef(lexkey, source_wn, target_wn)
            except:
                continue
                
            if source_wn_ili is not None:
                source_wn_eng = source_wn_ili.replace('ili', 'eng')
                self.source_wn_engs.add(source_wn_eng)

                
    def set_wn30_engs(self, source_wn, my_mapper):
        self.wn30_engs = set()
        for lexkey in self.lexkeys:
            try:
                wn30_ili = my_mapper.map_lexkey_to_ilidef(lexkey, source_wn, '30')
            except:
                continue
                
            if wn30_ili is not None:
                wn30_eng = wn30_ili.replace('ili', 'eng')
                self.wn30_engs.add(wn30_eng)