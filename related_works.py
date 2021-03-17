import numpy as np
from pyemd import emd
from multiprocessing import Pool
from itertools import product

from nltk.corpus import stopwords
from nltk import download
import os
import sys
from gensim.models import Word2Vec
import gensim.downloader as api


class WMD_python:
    """
    class to compute WMD
    """
    def __init__ (self, sentences_a, sentences_b, model_name="word2vec-google-news-300"):
        if not len(sentences_a) == len(sentences_b):
             print("number of sentences for comparison list should be same")
             sys.exit(1)
        self.sentences_a = sentences_a
        self.sentences_b = sentences_b
        self.model = api.load(model_name)

    def preprocess(sentence):
        """
        preprocess sentences to be used for WMD
        """
        sentence = [w for w in sentence if w not in stop_words]
        return sentence

    def parallel_preprocess_sentences(self):
        """
        parallelize preprocessing of sentences using preprocess above
        """
        sentences_a = self.sentences_a
        sentences_b = self.sentences_b

        download("stopwords")
        stop_words = stopwords.words("english")

        pool = Pool()

        print("encoding sentences a")
        sentences_a_outputs = pool.map(self.preprocess, sentences_a)

        pool.close()
        pool.join()
        
        print("encoding sentences b")
        sentences_b_outputs = pool.map(self.preprocess, sentences_b)
        
        pool.close()
        pool.join()

        return sentences_a_outputs, sentences_b_outputs

    def WMD(self, sentence_a, sentence_b):
        """
        compute and return wmd_distance given a pair of sentence and model
        """
        model = self.model
        distance = model.wmdistance(sentence_a, sentence_b)
        return distance
    

    def WMD_compute(self):
        """
        returns WMD using gensims library implementation
        """
        self.sentences_a, self.sentences_b = self.parallel_preprocess_sentences()
    
        pool = Pool()

        print("parallely computing the distances using the model above")
        distances = pool.map(self.WMD, zip(self.sentences_a, self.sentences_b))

        pool.close()
        pool.join()

        return distances
