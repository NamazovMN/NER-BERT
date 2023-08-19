import os
import pickle

class Process:
    def __init__(self, preprocess, vocabulary, max_len, ds_path):
        self.preprocess = preprocess
        self.max_len = max_len
        self.vocabulary = vocabulary
        self.ds_path = ds_path

    def encode_data(self, input_list):
        uniform_text = self.set_length(input_list)
        return [self.vocabulary[token] for token in uniform_text]

    def encode_labels(self, input_list):
        uniform_labels = self.set_length(input_list)
        return [self.vocabulary.lab2id(label) if label in ['<PAD>', '<SOS>', '<EOS>'] else label for label in uniform_labels]

    def set_length(self, input_list):
        length = len(input_list)

        if length > self.max_len:
            return input_list[0: self.max_len]
        else:
            difference = self.max_len - length
            result_list = input_list + ['<PAD>'] * difference
            return result_list

    def process_data(self, dataset, ds_name):
        ds_dir = os.path.join(self.ds_path, f"{ds_name}.pickle")
        if not os.path.exists(ds_dir):
            dataset = self.preprocess.process(dataset)
            dataset['encoded_tokens'] = dataset['tokens'].apply(self.encode_data)
            dataset['encoded_labels'] = dataset['ner_tags'].apply(self.encode_labels)
            with open(ds_dir, 'wb') as ds_data:
                pickle.dump(dataset, ds_data)
        with open(ds_dir, 'rb') as ds_data:
            dataset = pickle.load(ds_data)
        return dataset

