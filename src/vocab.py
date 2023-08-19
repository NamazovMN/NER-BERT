import os
import pickle
from utilities import *

class Vocabulary:
    def __init__(self, ner_setup, preprocess, dataset: dict, ds_path: str = 'datasets'):
        self.preprocess = preprocess
        self.vocabulary, self.label2id = self.generate_vocab(dataset['train'], ds_path, ner_setup)

    def get_unique_tokens(self, data):
        tokens = list()
        for sentence in data['tokens']:
            tokens.extend(sentence)
        return set(tokens)

    def get_unique_labels(self, data):
        labels = list()
        for label_list in data['ner_tags']:
            labels.extend(label_list)
        return set(labels)

    def generate_vocab(self, ds, ds_dir, ner_setup):
        check_dir(ds_dir)
        vocab_path = os.path.join(ds_dir, 'vocab_info.pickle')

        if not os.path.exists(vocab_path):
            dataset = self.preprocess.process(ds)
            unique_tokens = self.get_unique_tokens(dataset)
            vocabulary = {
                '<PAD>': 0,
                '<UNK>': 1,
            }
            def_number = len(vocabulary)
            for idx, token in enumerate(unique_tokens):
                vocabulary[token] = idx + def_number
            ner_setup['<PAD>'] = len(ner_setup)
            vocab_info = {
                'vocabulary': vocabulary,
                'lab2id': ner_setup
            }

            with open(vocab_path, 'wb') as vocab_data:
                pickle.dump(vocab_info, vocab_data)
        with open(vocab_path, 'rb') as vocab_data:
            vocab_info = pickle.load(vocab_data)

        return vocab_info['vocabulary'], vocab_info['lab2id']

    def __getitem__(self, token):
        return self.vocabulary[token] if token in self.vocabulary.keys() else self.vocabulary['<UNK>']

    def lab2id(self, label):
        return self.label2id[label]

    def __len__(self):
        return len(self.vocabulary)

