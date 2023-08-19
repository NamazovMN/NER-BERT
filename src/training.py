import os
from process import Process
from preprocess import Preprocess
from vocab import Vocabulary
from utilities import *
from dataset import NERDataset
import torch
from torch.optim import Adam
import torch.nn as nn
import pandas as pd
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import NERClassifier
from sklearn.metrics import f1_score
class Train:
    """
    Class is used for training process. It not only performs each phase of training also keeps track of results
    per epoch. Notice that model and optimizer parameters are also saved by this class.
    """

    def __init__(self, preprocess: Preprocess, ds_path: str, vocabulary: Vocabulary, hyperparams: dict, exp_num: int, device: str,
                 resume_training: bool = False, load_best: bool = False, choice: str = 'f1_score',
                 epoch_choice: int = 0):
        """
        Initializer for the clas*s which specifies required parameters
        :param preprocess: Preprocess object will be consumed by post-process object
        :param vocabulary: Vocabulary object will be consumed by post-process object
        :param hyperparams: hyperparameters for model and post-process object
        :param exp_num: experiment number for saving experiment results in specific folder
        :param device: device for model training (can be cuda or cpu)
        :param resume_training: boolean variable determines whether resume from the specific (or the last) epoch
        :param load_best: boolean variable specifies whether model parameters will be loaded based on best performance
        :param choice: choice of the best performance (f1_score, dev_loss, dev_accuracy)
        :param epoch_choice: epoch choice for model parameter loading (can also be used for resume training)
        """
        self.prep = preprocess
        self.hp = hyperparams
        self.device = device
        self.exp_dir = self.create_directory(exp_num)
        self.resume_training = resume_training
        self.load_best = load_best
        self.choice = choice
        self.epoch_choice = epoch_choice
        self.processing = Process(preprocess, vocabulary, hyperparams['max_length'], ds_path)
        self.model = self.set_model(vocabulary)
        self.optimizer = self.set_optimizer()
        self.pad = vocabulary.lab2id('<PAD>')
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary.lab2id('<PAD>'))

    def set_model(self, vocabulary: Vocabulary) -> NERClassifier:
        """
        Method is utilized to set initialize the model
        :param vocabulary: vocabulary object
        :return: classifier object
        """
        return NERClassifier(hyperparams=self.hp, vocabulary=vocabulary).to(self.device)

    def set_optimizer(self) -> Adam:
        """
        Method is utilized to set optimizer
        :return: Adam optimizer
        """
        return Adam(
            params=self.model.parameters(),
            lr=self.hp['learning_rate'],
            weight_decay=self.hp['weight_decay']
        )

    def create_directory(self, exp_num: int) -> str:
        """
        Method is utilized to create experiment folder which will be used by class for data tracking
        :param exp_num: experiment number
        :return: experiment directory for further uses
        """
        results_dir = '../results'
        check_dir(results_dir)
        experiment_folder = os.path.join(results_dir, f'experiment_{exp_num}')
        check_dir(experiment_folder)
        hp_file = os.path.join(experiment_folder, 'hyperparams.pickle')
        with open(hp_file, 'wb') as hp_save_data:
            pickle.dump(self.hp, hp_save_data)
        return experiment_folder

    def get_data(self, datasets: dict, ds_name: str) -> tuple:
        """
        Method is utilized to get batches and number of instances of the dataset, which is specified by its name
        :param datasets: dictionary of all datasets
        :param ds_name: specific dataset name (train, validation, test)
        :return: tuple of data loader object and length of specific dataset
        """
        data = self.processing.process_data(datasets[ds_name], ds_name)
        dataset = NERDataset(data, self.device)
        return DataLoader(dataset=dataset, batch_size=self.hp['batch_size'], shuffle=True), dataset.__len__()

    def train_step(self, train_batch: dict) -> tuple:
        """
        Method is used to perform each step of the corresponding epoch
        :param train_batch: dictionary which includes data and labels for corresponding batch
        :return: tuple of model's output tensor and step loss value
        """
        output = self.model(train_batch['data'])
        predictions = output.view(-1, output.shape[-1])
        labels = train_batch['label'].view(-1)

        loss = self.loss_fn(predictions, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def compute_accuracy(self, batch_data: dict, output: torch.Tensor) -> tuple:
        """
        Method is utilized to compute accuracy at each step of the corresponding epoch
        :param batch_data: dictionary which includes data and labels for corresponding batch
        :param output: tensor for model's output
        :return: tuple which includes following information:
                 correct: number of correct predictions per step
                 predictions: list which includes model's output per given batch
                 targets: list which includes target values per given batch
        """

        predictions = torch.argmax(output, -1).view(-1).tolist()
        targets = batch_data['label'].view(-1).tolist()
        # original_data = [t == p for t, p in zip(targets, predictions) if t != self.pad]
        real_t = list()
        real_p = list()
        corrects = list()
        for t, p in zip(targets, predictions):
            if t != self.pad:
                real_t.append(t)
                real_p.append(p)
                corrects.append(t == p)

        correct = sum(corrects)

        return correct, real_t, real_p

    def development(self, dev_loader: DataLoader, epoch: int, type_out: str = 'test') -> tuple:
        """
        Method is utilized to perform validation after each epoch of training
        :type epoch: parameter specifies the current epoch
        :param dev_loader: DataLoader object for validation dataset
        :param dev_instances: number of instances for validation dataset
        :param type_out: defines whether data is validation or test dataset
        :return: tuple object which includes following information:
                 dev_avg_loss: average loss for validation phase
                 dev_accuracy: accuracy (in percents) for validation phase
                 f1_macro: f1 score with macro average
        """
        dev_epoch_loss = 0
        dev_epoch_accuracy = 0
        self.model.eval()
        predictions = list()
        targets = list()
        num_batches = len(dev_loader)
        word_choice = 'Validation' if type_out == 'dev' else 'Test'
        dev_instances = 0
        with torch.no_grad():
            dev_iter = tqdm(dev_loader, total=num_batches, leave=True)

            for dev_batch in dev_iter:
                output = self.model(dev_batch['data'])
                preds = output.view(-1, output.shape[-1])
                dev_step_loss = self.loss_fn(preds, dev_batch['label'].view(-1))
                dev_step_accuracy, dev_targets, dev_predictions = self.compute_accuracy(dev_batch, output)
                dev_instances += len(dev_targets)
                dev_epoch_loss += dev_step_loss.item()
                dev_epoch_accuracy += dev_step_accuracy
                predictions.extend(dev_predictions)
                targets.extend(dev_targets)
                dev_iter.set_description(f"{word_choice} Loss: {dev_epoch_loss / num_batches: .4f}; "
                                         f"{word_choice} Accuracy: {dev_epoch_accuracy / dev_instances: .4f}")
        f1_macro = f1_score(targets, predictions, average='macro')
        dev_avg_loss = dev_epoch_loss / num_batches
        dev_accuracy = dev_epoch_accuracy / dev_instances
        out_dict = {'predictions': predictions, 'targets': targets}
        self.save_outputs(out_dict, epoch, type_out)
        return dev_avg_loss, dev_accuracy, f1_macro

    def training(self, datasets: dict, epochs: int = 5) -> None:
        """
        Method is utilized to perform training phase
        :param datasets: dictionary of all datasets
        :param epochs: number of epochs that model will be trained
        :return: None
        """
        train_loader, _ = self.get_data(datasets, 'train')
        dev_loader, _ = self.get_data(datasets, 'validation')
        num_train_batches = len(train_loader)

        init_epoch = self.load_model_parameters() if self.resume_training else 0
        for epoch in range(init_epoch, epochs):
            train_instances = 0

            self.model.train()
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            train_iterator = tqdm(train_loader, total=num_train_batches, leave=True)
            for batch in train_iterator:
                output, train_step_loss = self.train_step(batch)
                epoch_train_loss += train_step_loss
                train_step_accuracy, train_targets, _ = self.compute_accuracy(batch, output)
                train_instances += len(train_targets)
                epoch_train_accuracy += train_step_accuracy
                train_iterator.set_description(f"Epoch: {epoch + 1} "
                                               f"Train Loss: {epoch_train_loss / num_train_batches: .4f} "
                                               f"Train Accuracy: {epoch_train_accuracy / train_instances: .4f}")
            dev_loss, dev_accuracy, f1 = self.development(dev_loader, epoch, 'dev')
            print(f'Validation f1 score: {f1: .4f}')
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss / num_train_batches,
                'dev_loss': dev_loss,
                'train_accuracy': epoch_train_accuracy / train_instances,
                'dev_accuracy': dev_accuracy,
                'f1_score': f1
            }
            self.save_results(epoch_results)
            self.save_model_parameters(epoch_results)
        test_loader, _ = self.get_data(datasets, 'test')
        test_loss, test_accuracy, test_f1 = self.development(test_loader, 100, 'test')
        print(f'Test Results after {epochs} epochs: Test Loss: {test_loss: .4f}, Test Accuracy: {test_accuracy: .4f},'
              f'f1 score: {test_f1: .4f}')

    def save_outputs(self, output_dict: dict, epoch: int, type_out: str = 'dev'):
        output_dir = os.path.join(self.exp_dir, 'outputs')
        check_dir(output_dir)
        out_file = os.path.join(output_dir, f'epoch_{epoch + 1}_{type_out}_output.pickle')
        with open(out_file, 'wb') as out_data:
            pickle.dump(output_dict, out_data)

    def save_results(self, epoch_data: dict) -> None:
        """
        Method is utilized to save training and validation results per epoch
        :param epoch_data: dictionary which includes all relevant information as a result of epoch
        :return: None
        """
        train_results_file = os.path.join(self.exp_dir, 'train_results.pickle')

        if not os.path.exists(train_results_file):
            results_dict = {data: list() for data in epoch_data.keys()}
        else:
            with open(train_results_file, 'rb') as train_data:
                results_dict = pickle.load(train_data)
        for data, value in epoch_data.items():
            results_dict[data].append(value)
        with open(train_results_file, 'wb') as train_data:
            pickle.dump(results_dict, train_data)
        print(f"Train and Validation results were added to data for {epoch_data['epoch']}")

    def save_model_parameters(self, epoch_data: dict):
        """
        Method is utilized to save model and optimizer parameters after each epoch
        :param epoch_data: dictionary which includes all relevant information as a result of epoch
        :return: None
        """
        checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')
        check_dir(checkpoints_dir)
        model_name = f"model_ep_{epoch_data['epoch']}_f1_{epoch_data['f1_score']: .5f}_" \
                     f"devacc_{epoch_data['dev_accuracy']: .5f}"
        model_path = os.path.join(checkpoints_dir, model_name)
        optim_name = f"optim_ep_{epoch_data['epoch']}"
        optim_path = os.path.join(checkpoints_dir, optim_name)
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
        print(f"Model and optimizer parameters were saved successfully for {epoch_data['epoch']}")
        print(20 * '<', 20 * '>')

    def load_best_choice(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Method is utilized for simplifying process (in granular approach). Mainly it specifies on what the best
        choice will be made and returns the corresponding subset of the results data frame
        :param results_frame: dataframe which includes all relevant results for all epochs
        :return: dataframe which is specific subset of the provided one
        """
        optimal_choice = results_frame[self.choice].max() if self.choice != 'dev_loss' \
            else results_frame[self.choice].min()

        print(f"Model was chosen according to {self.choice}")
        return results_frame[results_frame[self.choice] == optimal_choice]

    def load_epoch(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Method is utilized for simplifying process (in granular approach). Mainly it determines subset will be chosen
        whether for specific epoch choice or the last epoch
        :param results_frame: dataframe which includes all relevant results for all epochs
        :return: dataframe which is specific subset of the provided one
        """
        if not self.epoch_choice:
            print('No specific epoch was chosen. Thus, the latest epoch parameters will be loaded')

        choice = results_frame[results_frame['epoch'] == self.epoch_choice] if self.epoch_choice else \
            results_frame[results_frame['epoch'] == results_frame['epoch'].max()]
        return choice

    def load_model_parameters(self) -> int:
        """
        Method is used for loading model parameters. Mainly it does following:
        1. Check if such results data exists. If it does not, it means there was not any training. Ends with error.
        2. If such data exists, then dataframe is loaded from the given path.
        3. Time to determine which way should we follow:
            3.1. If best model and epoch choice were made together, method will choose the best model over the epoch
            3.2  In case one of them was given, then data will be chosen in that way
            3.3. In case none was given, the last epoch results will be loaded as part of epoch choice
        4. Model will be activated via model.eval()
        :return: epoch of specific choice (does not matter which way was chosen)
        """
        results_path = os.path.join(self.exp_dir, 'train_results.pickle')
        if not os.path.exists(results_path):
            raise FileNotFoundError('No saved results! It is because this experiment was not performed yet!')
        with open(results_path, 'rb') as results_data:
            results_dict = pickle.load(results_data)
        result_frame = pd.DataFrame(results_dict)

        checkpoints_dir = os.path.join(self.exp_dir, 'checkpoints')
        subset = self.load_best_choice(result_frame) if self.load_best \
            else self.load_epoch(result_frame)
        result = {key: list(subset[key])[0] for key in subset.columns}

        if self.load_best and self.epoch_choice:
            Warning("When best choice and epoch choice are requested, priority is the best choice!")

        model_name = f"model_ep_{result['epoch']}_f1_{result['f1_score']: .5f}_" \
                     f"devacc_{result['dev_accuracy']: .5f}"
        model_path = os.path.join(checkpoints_dir, model_name)
        optim_name = f"optim_ep_{result['epoch']}"
        optim_path = os.path.join(checkpoints_dir, optim_name)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.optimizer.load_state_dict(torch.load(optim_path))

        print(f"Model and optimizer were loaded successfully for epoch {result['epoch']} wrt user choice")

        return result['epoch']