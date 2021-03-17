from nltk.corpus import stopwords
from nltk import download
import os
from gensim.models import Word2Vec
import gensim.downloader as api

download("stopwords")
stop_words = stopwords.words("english")

def preprocess(sentence):
    sentence = [w for w in sentence if w not in stop_words]
    return sentence

"""
sentences
"""
sentence_a = "i am a good boy and love sex"
sentence_b = "she does not love me"
sentence_c = "i like jam"
"""
Remove stopwords
"""
sentence_a = preprocess(sentence_a)
sentence_b = preprocess(sentence_c)

model_name = "glove-wiki-gigaword-300" # "word2vec-google-news-300" or "glove-wiki-gigaword-300"
model = api.load(model_name)

distance = model.wmdistance(sentence_a, sentence_b)
print(distance)
