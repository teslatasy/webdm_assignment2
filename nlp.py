#!/usr/bin/env python3
# coding: utf-8

from CRF.CRF import *

import re

class NLP:
    def __init__(self, algorithm = 'CRF'):
        self.algorithm = algorithm
        print(self.algorithm)

    def sentsplit(self, text):
        sents = re.split(r"([。!！?？])", text.strip())
        sents.append("")
        sents = [item for item in ["".join(i) for i in zip(sents[0::2], sents[1::2])] if len(item) > 0]
        return sents

    def ner(self, text):
        if self.algorithm == 'CRF':
            return CRF().ner(text)

    def cut(self, text):
        if self.algorithm == 'CRF':
            return CRF().cut(text)


    def postag(self, text):
        if self.algorithm == 'CRF':
            return CRF().postag(text)


    def dep(self, word_list, pos_list):
        return CRF().dep(word_list, pos_list)

    
    



