import os
from string import punctuation
import nltk
import pandas as pd
from utilities import *
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')
class Preprocess:
    def __init__(self, remove_stops=True, remove_punct=True, cased=False):
        self.remove_stops = remove_stops
        self.remove_punct = remove_punct
        self.cased = cased
        self.stopwords = stopwords.words('english')

    def remove_punct_idx(self, input_list):
        return [idx for idx, token in enumerate(input_list) if token in punctuation]

    def remove_stops_idx(self, input_list):
        return [idx for idx, token in enumerate(input_list) if token in self.stopwords]

    def apply_remove(self, input_list, pidx, sidx):
        removals = pidx + sidx
        remove_idx = set(removals)
        return [data for idx, data in enumerate(input_list) if idx not in remove_idx]

    def set_lower(self, input_list):
        return [token.lower() for token in input_list]

    def process(self, dataset):
        dataset['tokens'] = dataset['tokens'] if self.cased else dataset['tokens'].apply(self.set_lower)
        dataset['pidx'] = dataset['tokens'].apply(self.remove_punct_idx)
        dataset['sidx'] = dataset['tokens'].apply(self.remove_stops_idx)
        dataset['clean_tokens'] = dataset.apply(lambda x: self.apply_remove(x['tokens'], x['pidx'], x['sidx']), axis=1)
        dataset['clean_tags'] = dataset.apply(lambda x: self.apply_remove(x['ner_tags'], x['pidx'], x['sidx']), axis=1)
        result = pd.DataFrame(dataset[['id', 'clean_tokens', 'clean_tags']])
        result = result.rename(columns={'clean_tokens': 'tokens', 'clean_tags': 'ner_tags'})

        return result
