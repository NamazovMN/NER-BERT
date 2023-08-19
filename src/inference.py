from model import NERClassifier
from preprocess import Preprocess
from process import Process
import pandas as pd
import torch
import os
import pickle
from nltk.tokenize import word_tokenize
class Inference:
    def __init__(self, exp_num, device):
        self.exp_num = exp_num
        self.device = device

        self.experiment_dir = self.set_experiment_dir()
        self.experiment_parameters = self.get_experiment_parameters()
        self.vocabulary, self.id2lab = self.get_vocabulary()

        self.classifier = self.set_model()


    def set_experiment_dir(self):
        results_dir = '../results'
        experiment_dir = os.path.join(results_dir, f'experiment_{self.exp_num}')
        if not os.path.exists(experiment_dir):
            raise NotImplementedError("No such model with given details! Check them first!")
        return experiment_dir

    def get_experiment_parameters(self):
        parameters_path = os.path.join(self.experiment_dir, 'outputs/parameters.pickle')
        with open(parameters_path, 'rb') as params_dir:
            parameters = pickle.load(params_dir)
        return parameters

    def get_vocabulary(self):
        cased = self.experiment_parameters['cased']
        stopwords = self.experiment_parameters['stopwords']
        punctuation = self.experiment_parameters['punctuation']
        ds_path = f"../dataset{'_punct' if punctuation else ''}{'_stops' if stopwords else ''}" \
                  f"{'_cased' if cased else ''}"
        vocabulary_path = os.path.join(ds_path, 'vocab_info.pickle')

        if not os.path.exists(vocabulary_path):
            raise NotImplementedError('Vocabulary cannot be found! Make sure the path is correct!')
        with open(vocabulary_path, 'rb') as vocab_data:
            vocabulary_info = pickle.load(vocab_data)
        id2lab = {idx: lab for lab, idx in vocabulary_info['lab2id'].items()}
        return vocabulary_info['vocabulary'], id2lab

    def encode_text(self, text):
        return [self.vocabulary[token] if token in self.vocabulary.keys() else self.vocabulary['<UNK>'] for token in text]

    def decode_prediction(self, output):
        return [self.id2lab[prediction] for prediction in output]


    def set_model(self):
        hp = self.experiment_parameters['hyperparams']
        return NERClassifier(hp, self.vocabulary).to(self.device)

    def load_model_results(self):
        results_path = os.path.join(self.experiment_dir, 'outputs/results.pickle')
        if not os.path.exists(results_path):
            raise FileNotFoundError('No such directory, for this you need to train the model first!')

        with open(results_path, 'rb') as result_data:
            result_dict = pickle.load(result_data)
        return result_dict

    def load_model(self, epoch_choice):
        result_dict = self.load_model_results()
        data = pd.DataFrame(result_dict)
        ckpt_dir = os.path.join(self.experiment_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError('No checkpoints folder was found! Make sure that you trained the model!')

        request = data[data['epoch'] == epoch_choice]
        model_name = f"model_epoch_{epoch_choice}_f1_{request['f1_macro'].item():.4f}_" \
                     f"loss_{request['dev_loss'].item():.4f}_acc_{request['dev_accuracy'].item():.4f}"
        model_path = os.path.join(ckpt_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model was not trained at least for {epoch_choice} epochs or there is typo! '
                                    f'Check it first!')

        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()

    def decision_maker(self, project_parameters):

        if project_parameters['load_best']:
            if project_parameters['epoch_choice']:
                print(
                    'WARNING: Best choice and epoch choice were made together! '
                    'In such cases best choice is prioritized!'
                )
            return self.get_best_epoch(project_parameters['load_choice'])

        else:
            return project_parameters['epoch_choice']

    def get_best_epoch(self, user_choice):
        results_dict = self.load_model_results()
        data = pd.DataFrame(results_dict)
        if user_choice == 'dev_loss':
            choice = min(data[user_choice])
        else:
            if user_choice == 'f1_score':
                user_choice = 'f1_macro'
            choice = max(data[user_choice])
        request = data[data[user_choice] == choice]
        epoch = request['epoch'].item()
        print(f'According to the best choice selection, epoch {epoch} was chosen!')
        return epoch

    def process_text(self, input_sequence):
        length = len(input_sequence)
        max_len = self.experiment_parameters['max_length']
        sequence = input_sequence[0: max_len] if length > max_len else input_sequence + ['<PAD>'] * (max_len - length)
        return self.encode_text(sequence)

    def inference(self, user_choices):
        epoch_choice = self.decision_maker(user_choices)
        self.load_model(epoch_choice)
        input_sentence = input('Please provide your text: ')
        tokenized_text = word_tokenize(input_sentence)
        encoded_text = self.process_text(tokenized_text)
        input_tensor = torch.LongTensor(encoded_text).to(self.device)
        output = self.classifier(input_tensor)
        predictions = torch.argmax(output, -1).tolist()
        result = self.decode_prediction(predictions)

        print(result[0: len(tokenized_text)])


