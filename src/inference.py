import nltk
import os
import pandas as pd
import pickle
import torch

from nltk.tokenize import word_tokenize
from src.classifier import NERClassifierBERT
from src.process import DataProcess
from transformers import DataCollatorForTokenClassification

nltk.download('punkt')


class Inference:
    """
    Class is utilized to infer with the pre-trained model as a playground environment
    """

    def __init__(self, hp: dict, process: DataProcess, device: str, infer_parameters: dict):
        """
        Method is utilized as an initializer to set the inference environment
        :param hp: hyperparameters for the model setup
        :param process: data processing object will be utilized to set the model input
        :param device: can either be cuda or cpu
        :param infer_parameters: dictionary which includes inference parameters
        """
        self.hp = hp
        self.device = device
        self.process = process
        self.infer_parameters = infer_parameters
        self.vocabulary = self.process.tokenizer.vocab
        self.collator = self.set_collator()
        self.experiment_dir = self.set_experiment_environment()
        self.classifier = self.set_model()

    def set_experiment_environment(self) -> str:
        """
        Method is utilized to set experiment environment which data will be used for model setup
        :return: directory for the experiment data
        """
        experiment_path = os.path.join(f'results/experiment_{self.infer_parameters["experiment_num"]}')
        return experiment_path

    def set_collator(self) -> DataCollatorForTokenClassification:
        """
        Method is utilized to create data collator object for the experiment
        :return: DataCollator object will be used for post-processing purposes
        """
        return DataCollatorForTokenClassification(
            tokenizer=self.process.tokenizer,
            max_length=self.hp['max_length'],
            padding='max_length'
        )

    def set_model(self) -> NERClassifierBERT:
        """
        Method is utilized to set the model
        :return: Classifier for the given experiment
        """
        return NERClassifierBERT(self.hp).to(self.device)

    def get_best_epoch(self, user_choice: str) -> int:
        """
        Method is utilized to get epoch value for the specific user choice
        :param user_choice: can be f1_macro, dev_acc, dev_loss
        :return: integer value for epoch which corresponds to the best value of the given choice
        """
        results_dict = self.load_model_results()
        data = pd.DataFrame(results_dict)
        if user_choice == 'dev_loss':
            choice = min(data[user_choice])
        else:
            choice = max(data[user_choice])
        request = data[data[user_choice] == choice]
        epoch = request['epoch'].item()
        print(f'According to the best choice selection, epoch {epoch} was chosen!')
        return epoch

    def load_model_results(self) -> dict:
        """
        Method is utilized to load model results, which are collected during the training process
        :return: dictionary which holds each epoch's results
        """
        results_path = os.path.join(self.experiment_dir, 'outputs/results.pickle')
        if not os.path.exists(results_path):
            raise FileNotFoundError('No such directory, for this you need to train the model first!')

        with open(results_path, 'rb') as result_data:
            result_dict = pickle.load(result_data)
        return result_dict

    def load_model(self, epoch_choice: int) -> None:
        """
        Method is utilized to load model for the given epoch choice
        :param epoch_choice: integer specifies which epoch's model will be loaded
        :return: None
        """
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

    def decision_maker(self, project_parameters: dict) -> int:
        """
        Method is utilized to select epoch according to the user's choices for model loading scenario
        :param project_parameters: experiment parameters in which user's choice is kept for loading decision
        :return: epoch choice according to the provided information
        """
        if project_parameters['load_best']:
            if project_parameters['epoch_choice']:
                print(
                    'WARNING: Best choice and epoch choice were made together! '
                    'In such cases best choice is prioritized!'
                )
            return self.get_best_epoch(project_parameters['load_choice'])

        else:
            return project_parameters['epoch_choice']

    @staticmethod
    def pretty_combiner(sequence: list) -> list:
        """
        Method is utilized to prevent mis-tokenization because of combinations of apostrophes and specific examples such
        as 'gonna' and 'wanna'
        :param sequence: list of tokens which were collected by nltk tokenizer
        :return: list of tokens which can be seen as more correct combination
        """
        result = list()
        na_list = ['gon', 'wan']
        shortcuts = ["'s", "'d", "n't", "'m", "'ve", "'ll", "na"]
        count = 0
        for idx, token in enumerate(sequence):
            if token not in shortcuts or idx == 0:
                result.append(token)
                count += 1
            else:
                if token == 'na' and result[count - 1] not in na_list:
                    result.append(token)
                else:
                    new_token = result[count - 1] + token
                    result[count - 1] = new_token
        return result

    @staticmethod
    def make_alignment(original_tokens: list, bert_tokens: list) -> tuple:
        """
        Method is utilized to make alignment between clean sequence and model input sequence. Example:
        Clean sequence: ["I'm", "going", "home", "."]
        Model input: ["[CLS]", "I", "'", "m", "going", "home", ".", "[SEP]"]
        Our desired output is in the length of the clean sequence, thus predictions of 'I', ''', 'm' will be processed
        in specific manner. In order to eliminate prospective confusion in indexes we align them
        :param original_tokens: clean sequence of tokens
        :param bert_tokens: sequence of tokenization results for transformer model
        :return: tuple of following elements:
                result of alignment: "[CLS]", "I'm", "going", "home", ".", "[SEP]";
                alignment map, specifies combination indexes: {0: [0], 1: [1, 2, 3], 2: [4], 3:[5], 4: [6], 5: [7]}
        """
        new_set = [bert_tokens[0]] + original_tokens + [bert_tokens[-1]]

        result_list = list()
        alignment_map = dict()
        bert_idx = 0
        for idx, token in enumerate(new_set):
            alignment_map[idx] = [bert_idx]

            if token == bert_tokens[bert_idx]:
                result_list.append(token)
                bert_idx += 1
            else:
                tok = bert_tokens[bert_idx]
                for cur_idx in range(bert_idx + 1, len(bert_tokens)):
                    tok += bert_tokens[cur_idx]
                    alignment_map[idx].append(cur_idx)
                    if tok == token:
                        result_list.append(tok)
                        bert_idx = cur_idx + 1
                        break

        return result_list, alignment_map

    @staticmethod
    def clean_model_input(model_text: list):
        """
        Model is utilized to fix ## tokenization as a result of AutoTokenizer usage. It happens when the longer words
        are given as input to the tokenizer. The first syllable will be without ## and the rest will be as starting with
        ##. This method cleans them and returns clean version. The rest will be handled by pretty combiner method, if
        needed.
        Example: "[CLS]", "I", "do", "some", "am", "##bi", "##gu", "##ous", "works", "[SEP]"
        Result: "[CLS]", "I", "do", "some", "am", "##bi", "##gu", "##ous", "works", "[SEP]"
        After pretty combiner: '[CLS]', 'I', 'do', 'some', 'ambiguous', 'works', '[SEP]'
        Alignment map: {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5, 6, 7], 5: [8], 6: [9]}
        :param model_text: list of tokens as output of the tokenizer
        :return: list of clean tokens (in case ## was detected, otherwise input itself)
        """
        result = list()
        for token in model_text:
            if '##' in token:
                result.append(token.replace('##', ''))
            else:
                result.append(token)
        return result

    def process_input(self, input_text: str) -> tuple:
        """
        Method is utilized to process the input text, which is provided by user as a sequence of characters. Then it
        will be put into the desired shape to perform classification.
        :param input_text: string object as an input sequence of characters
        :return: tuple of the following elements:
                clean_text: list of pretty combined tokens
                alignment_map: dictionary of alignment setup
                model_input: data can be used as model input
        """

        original_tokens = word_tokenize(input_text)
        main_tokens = self.pretty_combiner(original_tokens)

        model_input = self.process.tokenizer(input_text)
        alignment_model_data = self.clean_model_input(model_input.tokens())
        clean_text, alignment_map = self.make_alignment(main_tokens, alignment_model_data)
        return clean_text, alignment_map, model_input

    def process_out(self, clean_data: list, alignment_map: dict, predictions: torch.Tensor) -> None:
        """
        Method is utilized to process the output of the model for each scenario that can occur
        :param clean_data: list of pretty combined tokens
        :param alignment_map: dictionary for alignment setup
        :param predictions: prediction tensor with padding elements removed
        :return: Nothing, it just prints the output
        """
        prediction_result = list()

        for idx, token in enumerate(clean_data):

            if len(alignment_map[idx]) == 1:

                result = torch.argmax(predictions[:, alignment_map[idx][0], :], -1).item()

                prediction_result.append(result)

            else:
                init = 3 * predictions[:, alignment_map[idx][0]]

                summed = init + torch.sum(predictions[:, alignment_map[idx][1:len(alignment_map[idx])], :], 1)
                prediction_result.append(torch.argmax(summed, -1).item())

        decoded = [self.hp['id2label'][idx] for idx in prediction_result]
        result = [(token, decoded[idx]) for idx, token in enumerate(clean_data)]
        print(clean_data[1: len(clean_data)])
        print(result[1: len(result)])

    def infer(self, infer_parameters):
        user_choice = self.decision_maker(infer_parameters)
        self.load_model(user_choice)
        input_text = input('Please provide your text: ')
        clean_text, alignment_map, model_input = self.process_input(input_text)

        input_data = self.collator([each for each in [model_input]])
        predictions = self.classifier(input_data['input_ids'].to(self.device),
                                      input_data['attention_mask'].to(self.device))
        non_pad_out = predictions[:, 0: len(model_input.tokens()), :]

        self.process_out(clean_text, alignment_map, non_pad_out)
