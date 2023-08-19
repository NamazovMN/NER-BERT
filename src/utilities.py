import os
import pickle
import pandas as pd
from datasets import load_dataset
from argparse import Namespace
import argparse


def set_parameters() -> Namespace:
    """
    Function is used to set user-defined project parameters
    :return:
    """
    parser = argparse.ArgumentParser()
    # Experiment Details
    parser.add_argument('--experiment_num', required=False, type=int, default=2,
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
    parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help="Defines batch size. Compatibility in case of loading must be preserved by user!")
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-4,
                        help="Specifies weight decay")
    # User choices
    parser.add_argument('--train', required=False, action='store_true', default=False,
                        help='Activates the training session')
    parser.add_argument('--infer', required=False, action='store_true', default=False,
                        help='Activates inference')
    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='Starts training from chosen epoch (the last epoch if choice was not made)')
    parser.add_argument('--epoch_choice', required=False, type=int, default=1,
                        help='Epoch choice will be used for loading model')
    parser.add_argument('--load_best', required=False, action='store_true', default=False,
                        help="Load the best model parameters according to user's choice")
    parser.add_argument('--load_choice', required=False, type=str, default='f1_score',
                        choices=['f1_score', 'dev_loss', 'dev_accuracy'],
                        help="User's choice for the best model to load")
    # Model Parameters
    parser.add_argument('--embedding_dim', required=False, type=int, default=200,
                        help="Specifies embedding dimension")
    parser.add_argument('--hidden_dim', required=False, type=int, default=50,
                        help="Specifies hidden size of LSTM layer")
    parser.add_argument('--num_layers', required=False, type=int, default=1,
                        help="Specifies number of LSTM layers")
    parser.add_argument('--lstm_dropout', required=False, type=float, default=0.0,
                        help="Specifies dropout rate for LSTM layers. (Should be chosen 0.0 if num_layers is set to 1)")
    parser.add_argument('--max_length', required=False, type=int, default=180,
                        help="Specifies maximum length will be considered by model")
    parser.add_argument('--bidirectional', required=False, action='store_false', default=True,
                        help="If true Bi-LSTM, else LSTM will be used as a model")

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


def collect_datasets(ds_name: str = "conllpp", ds_path: str = "datasets") -> dict:
    """
    Function is utilized to load datasets object and prepare it to use for further purposes
    :param ds_name: name of the dataset to load from huggingface
    :param ds_path: dataset directory where loaded data will be kept
    :return: dictionary which holds all datasets as a type of pandas DataFrame
    """
    check_dir(ds_path)
    dataset_path = os.path.join(ds_path, 'raw_data.pickle')
    datasets = dict()
    if not os.path.exists(dataset_path):
        dataset = load_dataset(ds_name)
        for ds_type, data in dataset.items():
            datasets[ds_type] = pd.DataFrame(data)
        with open(dataset_path, 'wb') as data_load:
            pickle.dump(datasets, data_load)
    with open(dataset_path, 'rb') as data_load:
        datasets = pickle.load(data_load)
    return datasets

def ner_setup(list_labels: list, ds_path: str = 'datasets'):
    setup_path = os.path.join(ds_path, 'ner.pickle')
    if not os.path.exists(setup_path):
        setup_dict = {label: idx for idx, label in enumerate(list_labels)}
        with open(setup_path, 'wb') as setup_data:
            pickle.dump(setup_dict, setup_data)
    with open(setup_path, 'rb') as setup_data:
        setup_dict = pickle.load(setup_data)
    return setup_dict

def check_dir(directory: str) -> None:
    """
    Function is utilized to check the existence of the provided directory
    :param directory: provided path, which existence will be checked
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
