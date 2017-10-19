#!/usr/bin/env python

#####################################
#####################################
# Ruben Izquierdo Bevia
# VU University of Amsterdam
# ruben.izquierdobevia@vu.nl
# rubensanvi@gmail.com
# http://rubenizquierdobevia.com/
# Version 1.0
#####################################
#####################################


import os
import sys
from subprocess import check_output
from collections import defaultdict

from python_modules.supersense_list import SS as SUPERSENSE_LIST
__here__ = os.path.dirname(os.path.realpath(__file__))

class SenseInfo:
    def __init__(self):
        self.sense_number = None
        self.synset = None
        self.lexkey = None
        

class SemanticClassManager(object):
    def __init__(self):
        self.map_synset_pos_to_class = {}
        self.NOUN = 'n'
        self.VERB = 'v'
        self.ADJ = 'a'
        self.ADV = 'r'
        self.resource = 'SemanticClassAbstract'
        self.wn_version = None
        self.wn_mapper = None   #By default we will not use it
        self.synset_for_lexkey = None   # To be read by the subclasses    {lexkey} --> offset        
        self.sense_info_list_for_lemma_pos = {}
        
    def get_resource(self):
        return self.resource
    
    def normalise_pos(self,this_pos):
        pos = None
        if this_pos.lower() in ['noun','n','1']:
            pos = self.NOUN
        elif this_pos.lower() in ['verb','v','2']:
            pos = self.VERB
        elif this_pos.lower() in ['adj','j','3','5','s','a']:
            pos = self.ADJ
        elif this_pos.lower() in ['adv','r','4']:
            pos = self.ADV
        return pos
    
    def sense_info_iterator(self, lemma, pos):
        for sense_info in self.sense_info_list_for_lemma_pos.get((lemma,pos),[]):
            yield sense_info
            
            
    def get_classes_for_synset_pos(self,synset,this_pos, this_wn_version=None):
        if this_wn_version is None:
            #Then it's assumed to be using the proper wnversion for the selected semantic class (wn30 for BLC and wn21 for WND)
            pass
        elif this_wn_version != self.wn_version:
            #We need to map the synset from this_wn_Version to self.wn_version.
            if self.wn_mapper is None:
                from libs.WordNetMapper import WordNetMapper
                self.wn_mapper = WordNetMapper()
            synset, another_pos = self.wn_mapper.map_offset_to_offset(synset,this_wn_version,self.wn_version)
                #print 'Mapped synset wn30', synset
             
        pos = self.normalise_pos(this_pos)
        if pos is None:
            print>>sys.stderr,'Pos %s not recognized' % this_pos
            return None
        else:
            if pos in self.map_synset_pos_to_class:
                return self.map_synset_pos_to_class[pos].get(synset)
            
            
    def get_classes_for_lexkey(self,lexkey,this_wn_version=None):
        if this_wn_version is None:
            #Then it's assumed to be using the proper wnversion for the selected semantic class (wn30 for BLC and wn21 for WND)
            pass
        elif this_wn_version != self.wn_version:
            #We need to map the synset from this_wn_Version to self.wn_version.
            if self.wn_mapper is None:
                from libs.WordNetMapper import WordNetMapper
                self.wn_mapper = WordNetMapper()
            o = lexkey
            lexkey = self.wn_mapper.map_lexkey_to_lexkey(lexkey,this_wn_version,self.wn_version)
        ######
        #We get the pos from the lexkey: rock_hopper%1:05:00:
        p = lexkey.find('%')+1
        pos = self.normalise_pos(lexkey[p])
        this_class = None
        if pos is None:
            print>>sys.stderr,'Pos %s not recognized' % this_pos
        else:
            if pos in self.map_synset_pos_to_class:
                synset = self.synset_for_lexkey.get(lexkey)
                if synset is not None:
                    this_class = self.map_synset_pos_to_class[pos].get(synset)
        return this_class
    
    def get_classes_for_lemma_pos(self,lemma,pos):
        classes = []
        pos = self.normalise_pos(pos)
        for sense_info in self.sense_info_list_for_lemma_pos[(lemma,pos)]:
            these_classes = self.get_classes_for_synset_pos(sense_info.synset, pos)
            if these_classes is not None:
                classes.extend(these_classes)
        # In case we dont use the hierarchy option for WND, the list of classes will be
        # a plain list of strings, with different classes for the different synsets. What
        # we do here is remove the duplicated in these cases. Else, we return the whole object
        if len(classes) != 0:
            if not isinstance(classes[0],list):
                classes = list(set(classes))
        return classes
            
            
    def are_compatible(self,class1,class2):
        pass
        
    
    def get_most_frequent_classes(self,lemma,pos):
        classes = None
        lexkey_for_first_sense = None
        for sense_info in self.sense_info_iterator(lemma, pos):
            if sense_info.sense_number == '1':
                lexkey_for_first_sense = sense_info.lexkey
                break
        if lexkey_for_first_sense is not None:
            classes = self.get_classes_for_lexkey(lexkey_for_first_sense)
        return classes
    
    
            
        
    
                
      

      
##########################################################################################
##########################################################################################
##################   BLC SEMANTIC CLASSES             ####################################
##########################################################################################
##########################################################################################
  
  
class BLC(SemanticClassManager):
    def __init__(self, min_freq, type_relations):
        
        #Call constructor of the parent
        super(BLC, self).__init__()
        
        self.wn_version = '30'
        self.resource = 'blc_%s_%d_wn%s' % (type_relations,min_freq,self.wn_version)
        
        # Checking the min frequency
        valid_min_freq = [0,10,20,50]
        if min_freq not in valid_min_freq:
            print>>sys.stderr,'Min frequency "%d" not valid, Valid values: %s' % (min_freq,str(valid_min_freq))
            raise Exception('Min freq not valid')
        
        # Checking the type of relation
        valid_type_relations = ['all','hypo']
        if type_relations.lower() not in valid_type_relations:
            print>>sys.stderr,'Type relation "%s" not valid. Valid relations: %s' % (type_relations,str(valid_type_relations))
            raise Exception('Type relation not valid')
                             
        self.__read_index_sense__()          
        self.load_blc(min_freq, type_relations)   
        
          
             
    def load_blc(self,min_freq,type_relations):
        blc_folder = __here__ +'/resources/basic_level_concepts/BLC/WordNet-3.0/%s/%d' % (type_relations.lower(), min_freq)  
        
        for this_pos, this_file in [(self.NOUN,'blc.noun'),(self.VERB,'blc.verb')]:
            self.map_synset_pos_to_class[this_pos] = {}
            whole_path = blc_folder+'/'+this_file
            fd = open(whole_path,'r')
            for line in fd:
                #04958634 04916342 property.n#2 584
                if line[:3] != '###':
                    this_synset, synset_blc, friendly_blc, num_subsumed = line.strip().split()
                    self.map_synset_pos_to_class[this_pos][this_synset] = [friendly_blc]
        
        ###                                         

    def __read_index_sense__(self):
        #ili_blc is like 02560585-v

        path_to_wn_index = __here__+'/resources/WordNet-3.0/dict/index.sense'
        
        fd = open(path_to_wn_index,'r')
        self.synset_for_lexkey = {}
        self.sense_info_list_for_lemma_pos = defaultdict(list)

        for line in fd:
            #.22-caliber%3:01:00:: 03146310 1 0
            lexkey, synset, sense, freq = line.strip().split()
            sense_info = SenseInfo()
            sense_info.sense_number = sense
            sense_info.lexkey = lexkey
            sense_info.synset = synset
            
            self.synset_for_lexkey[lexkey] = synset
            p = lexkey.find('%')
            lemma = lexkey[:p]
            int_pos = lexkey[p+1]
            self.sense_info_list_for_lemma_pos[(lemma,self.normalise_pos(int_pos))].append(sense_info)
            


            
    def are_compatible(self,class1,class2):
        compatible = False
        if class1 == class2:
            compatible = False
        else:
            p1 = class1.rfind('#')
            pos_c1 = class1[p1-1]
            
            p2 = class2.rfind('#')
            pos_c2 = class2[p2-1]
            if pos_c1 == pos_c2:
                compatible = True
            else:
                compatible = False
        return compatible
            
            
##########################################################################################
##########################################################################################
##################   WND SEMANTIC CLASSES             ####################################
##########################################################################################
##########################################################################################
            
class WND(SemanticClassManager):
    def __init__(self, hierarchy=False):
        
        #Call constructor of the parent
        super(WND, self).__init__()
        
        self.wn_version = '20'
        self.resource = 'WND_wn%s' % self.wn_version
        self.hierarchy=hierarchy
        if self.hierarchy:
            self.parent_for_label = {}
            self.level_for_label = {}
            self.__load_wnd_ontology()
            
            
        self.__load_wnd__()
        self.__load_synset_for_lexkey__()

        
    def __load_wnd_ontology(self):
        previous_for_num_tab = {}
        wnd_file_hierarchy = __here__ + '/resources/wn-domains-3.2/WND_hierarchy.txt'
        fd = open(wnd_file_hierarchy)
        for line in fd:
            num_tabs = 0
            while line[num_tabs] == '\t':
                num_tabs += 1
    
            label = line.strip().lower()
            previous_for_num_tab[num_tabs] = label
            self.level_for_label[label] = num_tabs+1
            if num_tabs == 0:
                #is ROOT
                pass
            else:
                parent = previous_for_num_tab[num_tabs-1]
                self.parent_for_label[label] = parent
        fd.close()
        
    def __get_domain_labels_for_hierarchy(self,label):
        list_labels = []
        this_level = self.level_for_label[label]
        #For the type music#3
        #list_labels.append(label+'#%d' % this_level)
        
        #For using (music,3)
        list_labels.append((label,this_level))
            
        while True:
            parent = self.parent_for_label.get(label,None)
            if parent is None:
                break
            else:
                #list_labels.append(parent+'#%d' % self.level_for_label[parent])
                list_labels.append((parent,self.level_for_label[parent]))
                label = parent
        return list_labels
        
    def __load_wnd__(self):
        #Creates
        # self.map_synset_pos_to_class[this_pos][this_synset] = CLASS
        # self.synset_for_lexkey[lexkey] = synset
        wnd_file = __here__ + '/resources/wn-domains-3.2/wn-domains-3.2-20070223'
        fd = open(wnd_file,'r')
        for line in fd:
            #line -> 00001740-n    factotum
            fields = line.strip().split()
            synset_pos = fields[0]
            wnd_labels = fields[1:]
            
            synset = synset_pos[:-2]
            pos = synset_pos[-1]
            pos = self.normalise_pos(pos)
        
            if self.hierarchy:
                new_labels = []
                for wnd_label in wnd_labels:
                    new_labels.append(self.__get_domain_labels_for_hierarchy(wnd_label))
                wnd_labels = new_labels[:]
                
            
            if pos not in self.map_synset_pos_to_class:
                self.map_synset_pos_to_class[pos] = {}
            self.map_synset_pos_to_class[pos][synset] = wnd_labels
        fd.close()
        
    def __load_synset_for_lexkey__(self):
        self.synset_for_lexkey = {}
        wn_index_file = __here__+'/resources/WordNet-2.0/dict/index.sense'
        self.sense_info_list_for_lemma_pos = defaultdict(list)
        fd = open(wn_index_file,'r')
        for line in fd:
            #.22-caliber%3:01:00:: 03146310 1 0
            lexkey, synset, sense, freq = line.strip().split()
            sense_info = SenseInfo()
            sense_info.sense_number = sense
            sense_info.lexkey = lexkey
            sense_info.synset = synset
            
            self.synset_for_lexkey[lexkey] = synset

            p = lexkey.find('%')
            lemma = lexkey[:p]
            int_pos = lexkey[p+1]
            self.sense_info_list_for_lemma_pos[(lemma,self.normalise_pos(int_pos))].append(sense_info)

        fd.close()
        
    
    def are_compatible(self,class1,class2):
        if class1 == class2:
            return False
        else:
            return True
        

                
##########################################################################################
##########################################################################################
##################   SUPERSENSE SEMANTIC CLASSES             ####################################
##########################################################################################
##########################################################################################

class SuperSense(SemanticClassManager):
    def __init__(self):
        
        #Call constructor of the parent
        super(SuperSense, self).__init__()
        self.wn_version = '30'
        self.resource = 'supersense_wn%s' % self.wn_version
        self.__load_info__()
        
    def __load_info__(self):
        self.synset_for_lexkey = {}
        self.sense_info_list_for_lemma_pos = defaultdict(list)

        wn_index_file = __here__+'/resources/WordNet-3.0/dict/index.sense'
        fd = open(wn_index_file,'r')
        for line in fd:
            #.22-caliber%3:01:00:: 03146310 1 0
            lexkey, synset, sense, freq = line.strip().split()
            sense_info = SenseInfo()
            sense_info.sense_number = sense
            sense_info.lexkey = lexkey
            sense_info.synset = synset
            self.synset_for_lexkey[lexkey] = synset

            p = lexkey.find('%')
            lemma = lexkey[:p]
            int_pos = lexkey[p+1]
            self.sense_info_list_for_lemma_pos[(lemma,self.normalise_pos(int_pos))].append(sense_info) 

            
            parts = lexkey.split(':')
            int_supersense = parts[1]
            
            pos = self.normalise_pos(int_pos)
            if pos not in self.map_synset_pos_to_class:
                self.map_synset_pos_to_class[pos] = {}
            self.map_synset_pos_to_class[pos][synset] = [SUPERSENSE_LIST[int_supersense]] 

        fd.close()
        
    def are_compatible(self,class1,class2):
        if class1 == class2:
            return False
        else:
            p1 = class1.find('.')
            pos1 = class1[:p1]
            p2 = class2.find('.')
            pos2 = class2[:p2]
            if pos1 == pos2:
                return True
            else:
                return False