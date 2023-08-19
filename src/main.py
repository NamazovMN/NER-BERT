import os

import torch.cuda
from utilities import *
from preprocess import Preprocess
from vocab import Vocabulary
from trainer import Train
import random
from inference import Inference
def __main__():
    random.seed(42)
    ds = collect_datasets(ds_path='../raw_data')
    setup_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    ner_tags = ner_setup(setup_list, ds_path='../raw_data')
    project_parameters = get_parameters()

    remove_punct = project_parameters['punctuation']
    remove_stops = project_parameters['stopwords']
    cased = project_parameters['cased']
    ds_path = f"../dataset{'_punct' if remove_punct else ''}{'_stops' if remove_stops else ''}{'_cased' if cased else ''}"

    prep = Preprocess(remove_stops=remove_stops, remove_punct=remove_punct, cased=cased)
    vocabulary = Vocabulary(ner_tags, prep, ds, ds_path)
    hp = {
        'embedding_dim': project_parameters['embedding_dim'],
        'hid_dim': project_parameters['hidden_dim'],
        'num_classes': len(ner_tags),
        'bidirectional': project_parameters['bidirectional'],
        'dropout': project_parameters['lstm_dropout'],
        'num_layers': project_parameters['num_layers'],
        'max_length': project_parameters['max_length'],
        'learning_rate': project_parameters['learning_rate'],
        'batch_size': project_parameters['batch_size'],
        'weight_decay': project_parameters['weight_decay']
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if project_parameters['train']:
        trainer = Train(hp, vocabulary, prep, ds_path, project_parameters['experiment_num'], device)
        trainer.train_model(ds, project_parameters)
    if project_parameters['infer']:
        inference = Inference(project_parameters['experiment_num'], device)
        inference.inference(project_parameters)




if __name__ == '__main__':
    __main__()