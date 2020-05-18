#!/usr/bin/env python3

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from .feature import *
import os
class POSTAG:
    def __init__(self):
        self.model_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/model/crf_pos_model.pkl'
        self.model = joblib.load(self.model_path)

    def postag(self, word_list):
        sent_reps = feature_extract(word_list)
        return self.model.predict(sent_reps)[0]
