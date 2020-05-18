#!/usr/bin/env python3
# coding: utf-8

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from .feature import *
import os
class CUT:
    def __init__(self):
        self.model_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/model/crf_cut_model.pkl'
        self.model = joblib.load(self.model_path)

    #将序列标记转换为标记结果
    def label2word(self, labels, sent):
        labellist = []
        tmp = []
        for index in range(len(labels)):
            word = sent[index]
            tag = labels[index]
            if tag == 'S':
                if tmp:
                    labellist.append(tmp)
                tmp = [word]
                labellist.append(tmp)
                tmp = []
            elif tag == 'B':
                if tmp:
                    labellist.append(tmp)
                tmp = []
                tmp.append(word)
            elif tag == 'M':
                tmp.append(word)
            elif tag == 'E':
                tmp.append(word)
                labellist.append(tmp)
                tmp = []
        return [''.join(tmp) for tmp in labellist]

    #分词主函数
    def cut(self, sent):
        sent_reps = feature_extract(sent)
        labels = self.model.predict(sent_reps)[0]
        return self.label2word(labels, sent)

