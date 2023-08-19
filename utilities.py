import os

import pandas as pd
import json
from argparse import Namespace
import argparse

def create_dataset_name(parameters):
    choices = ['cased', 'punctuation', 'stopwords']
    ds_path = 'dataset'
    for each in choices:
        if parameters[each]:
            ds_path += f'_{each}'
    remove_options = {choice: parameters[choice] for choice in choices}
    return ds_path, remove_options

def get_hyperparameters(parameters):
    req_data = ['learning_rate', 'batch_size', 'max_length', 'dropout', 'weight_decay', 'model_checkpoint']
    hp = {data: parameters[data] for data in req_data}
    return hp

def set_parameters() -> Namespace:
    """
    Function is used to set user-defined project parameters
    :return:
    """
    parser = argparse.ArgumentParser()
    # Experiment Details
    parser.add_argument('--experiment_num', required=False, type=int, default=6,
                        help='Defines experiment number to track them')
    parser.add_argument('--cased', required=False, action='store_false', default=True,
                        help="Specifies transforming whole texts to lowercase")
    parser.add_argument('--punctuation', required=False, action='store_false', default=True,
                        help="Specifies removing punctuation elements from raw text")
    parser.add_argument('--stopwords', required=False, action='store_false', default=True,
                        help="Specifies removing stopwords")
    parser.add_argument('--epochs', required=False, type=int, default=20,
                        help='Specifies number of epochs to train the model')
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-4,
                        help="Specifies learning rate")
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help="Defines batch size. Compatibility in case of loading must be preserved by user!")
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-4,
                        help="Specifies weight decay")
    # User choices
    parser.add_argument('--train', required=False, action='store_true', default=False,
                        help='Activates the training session')
    parser.add_argument('--infer', required=False, action='store_false', default=True,
                        help='Activates inference')
    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='Starts training from chosen epoch (the last epoch if choice was not made)')
    parser.add_argument('--epoch_choice', required=False, type=int, default=1,
                        help='Epoch choice will be used for loading model')
    parser.add_argument('--load_best', required=False, action='store_true', default=False,
                        help="Load the best model parameters according to user's choice")
    parser.add_argument('--load_choice', required=False, type=str, default='f1_macro',
                        choices=['f1_macro', 'dev_loss', 'dev_accuracy'],
                        help="User's choice for the best model to load")
    # Model Parameters

    parser.add_argument('--dropout', required=False, type=float, default=0.3,
                        help="Specifies dropout rate for LSTM layers. (Should be chosen 0.0 if num_layers is set to 1)")
    parser.add_argument('--max_length', required=False, type=int, default=180,
                        help="Specifies maximum length will be considered by model")
    parser.add_argument('--model_checkpoint', required=False, type=str, default='bert-base-cased')

    return parser.parse_args()


def get_parameters() -> dict:
    """
    Method is utilized to transform Namespace object into dict (will be used by project)
    :return: dictionary that includes all user-defined project parameters
    """
    parameters = dict()
    params_namespace = set_parameters()
    for argument in vars(params_namespace):
        parameters[argument] = getattr(params_namespace, argument)
    return parameters

def check_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_labels():

    with open('dataset_infos.json', 'r') as dataset_info:
        ds_info = json.load(dataset_info)
    features = ds_info['conllpp']['features']

    ner_tags = {label: idx for idx, label in enumerate(features['ner_tags']['feature']['names'])}
    pos_tags = {label: idx for idx, label in enumerate(features['pos_tags']['feature']['names'])}

    return ner_tags, pos_tags


