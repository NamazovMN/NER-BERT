from torch.optim import AdamW
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import f1_score
from utilities import *
from tqdm import tqdm
import torch.nn as nn
from src.classifier import NERClassifierBERT
import pickle
from torch.utils.data import DataLoader
import torch
from src.process import DataProcess


class Train:
    """
    Class is utilized to load datasets and train the model for number of provided epochs.
    """

    def __init__(self, hp: dict, process: DataProcess, device: str, exp_num: int):
        """
        Method is utilized as initialized of the training object
        :param hp: hyperparameters for the model and experiment
        :param process: process object is utilized to load dataset and prepare it for training
        :param device: can be cuda or cpu
        :param exp_num: experiment number
        """
        self.process = process
        self.exp_num = exp_num
        self.hp = hp
        self.device = device

        self.datasets = process.process()
        self.collator = self.set_collator()
        self.experiment_dir = self.set_environment()
        self.classifier = self.set_model()
        self.optimizer = self.set_optimizer()
        self.loss_fn = nn.CrossEntropyLoss()

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

    def set_optimizer(self) -> AdamW:
        """
        Method is utilized to set the optimizer
        :return: Optimization object for the experiment
        """
        return AdamW(self.classifier.parameters(), lr=self.hp['learning_rate'], weight_decay=0.0001)

    def save_parameters(self, project_parameters: dict) -> None:
        """
        Method is utilized to create corresponding output folder for experiment and save project parameters in it
        :param project_parameters: dictionary in which experiment parameters are set
        :return: None
        """
        outputs_path = os.path.join(self.experiment_dir, 'outputs')
        check_dir(outputs_path)
        parameters_path = os.path.join(outputs_path, 'parameters.pickle')
        if not os.path.exists(parameters_path):
            print(f'Project Parameters for experiment {self.exp_num} were saved successfully!')
            print(f'{"<" * 20}{">" * 20} \n')
            project_parameters['hyperparams'] = self.hp
            with open(parameters_path, 'wb') as params:
                pickle.dump(project_parameters, params)

    def set_environment(self) -> str:
        """
        Method is utilized to create the experimental environment
        :return: experiment directory
        """
        results_dir = 'results'
        check_dir(results_dir)
        experiment_dir = os.path.join(results_dir, f'experiment_{self.exp_num}')
        check_dir(experiment_dir)
        return experiment_dir

    def get_data(self, ds_name: str) -> DataLoader:
        """
        Method is utilized to collect datasets according to the provided dataset type (train, validation, test)
        :param ds_name: specifies which data must be loaded
        :return: batched data for the provided dataset type
        """

        return DataLoader(dataset=self.datasets[ds_name],
                          collate_fn=self.collator,
                          shuffle=True,
                          batch_size=self.hp['batch_size'])

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
        optim_name = f"optim_epoch_{epoch_choice}"
        model_path = os.path.join(ckpt_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model was not trained at least for {epoch_choice} epochs or there is typo! '
                                    f'Check it first!')
        optim_path = os.path.join(ckpt_dir, optim_name)

        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()
        self.optimizer.load_state_dict(torch.load(optim_path))

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
            return self.get_best_epoch(project_parameters['best_choice'])

        else:
            return project_parameters['epoch_choice']

    def resume_training(self, project_parameters: dict) -> int:
        """
        Method is utilized to decide which epoch must be the initial for the training. If resume training was set to
        True, the recent epoch + 1 will be set as initial for training, otherwise it is zero.
        In case the parameter is False (start new training) and given experiment directory already has some relevant
        data, user will not be allowed to train the model, to prevent overwriting issues.
        :param project_parameters: experiment parameters dictionary
        :return: integer value for epoch choice as initial epoch
        """
        if project_parameters['resume_training']:
            epoch_choice = self.get_best_epoch('epoch')
            self.load_model(epoch_choice)
            return epoch_choice
        else:
            ckpt_dir = os.path.join(self.experiment_dir, 'checkpoints')
            if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
                raise SystemError('Folder is not empty and your choice tries to overwrite, since you do not '
                                  'try to resume training! Be careful about your choice!')
            else:
                return 0

    def step_process(self, batch_data: dict, train: bool = True) -> tuple:
        """
        Method is utilized to perform training (when train is True) or validation / test (when train is False) step for
        the given batch data
        :param batch_data: dictionary includes input_ids, attention_mask and labels for the given batch
        :param train: boolean variable specifies nature of the step
        :return: tuple of loss value for the step and predictions of the model
        """
        input_ids = batch_data['input_ids'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        targets = batch_data['labels'].to(self.device)

        outputs = self.classifier(input_ids, attention_mask)
        predictions = outputs.view(-1, outputs.shape[-1])
        ner = targets.view(-1)

        if train:
            self.optimizer.zero_grad()
            loss = self.loss_fn(predictions, ner)
            loss.backward()
            self.optimizer.step()
        else:
            loss = self.loss_fn(predictions, ner)

        return loss.item(), predictions

    @staticmethod
    def compute_accuracy(targets: torch.Tensor, outputs: torch.Tensor) -> tuple:
        """
        Method is utilized to compute the accuracy for each step (of each phase, since procedure remains same)
        :param targets: Tensor for the labels in shape of [batch size, max length]
        :param outputs: Tensor for the predictions in shape of [batch size, max length, num classes]
        :return: tuple of following elements:
                number of correct predictions for specific step
                list of non-padded target tokens
                list of non-padded output tokens
                Note: all collected in one list, so don't try to check sequence based. It is not necessary for f1 score
        """
        predictions = torch.argmax(outputs, -1).tolist()
        labels = targets.view(-1).tolist()
        corrects = list()
        original_targets = list()
        original_predictions = list()
        for t, p in zip(labels, predictions):
            if t != -100:
                corrects.append(t == p)
                original_targets.append(t)
                original_predictions.append(p)

        return sum(corrects), original_targets, original_predictions

    def development(self, validation_loader: DataLoader, num_valid_batches: int, test: bool = False) -> tuple:
        """
        Method is utilized to perform either validation (test=False) or test (test=True) process
        :param validation_loader: DataLoader object for validation dataset
        :param num_valid_batches: number of batches in validation dataset (required for tqdm and average loss)
        :param test: boolean variable to specify whether validation (False) or test (True) process is performed
        :return: tuple for following elements:
                dev_acc: accuracy for specific phase
                dev_loss: loss for specific phase
                f1_macro: f1 score based on macro average
                f1_micro: f1 score based on micro average
        """
        self.classifier.eval()
        targets = list()
        outputs = list()
        with torch.no_grad():
            validation_iterator = tqdm(iterable=validation_loader, total=num_valid_batches, leave=True)
            validation_loss = 0
            validation_accuracy = 0
            validation_instances = 0
            for batch in validation_iterator:
                validation_step_loss, ner_out = self.step_process(batch, train=False)
                validation_loss += validation_step_loss
                step_accuracy, targets_, outputs_ = self.compute_accuracy(batch['labels'], ner_out)
                targets.extend(targets_)
                outputs.extend(outputs_)
                validation_accuracy += step_accuracy
                validation_instances += len(targets_)
                validation_iterator.set_description(
                    desc=f"{'Test' if test else 'Validation'}: Loss: {validation_loss / num_valid_batches: .4f}"
                         f" Accuracy: {validation_accuracy / validation_instances: .4f}"
                )
            f1_macro = f1_score(targets, outputs, average='macro')
            f1_micro = f1_score(targets, outputs, average='micro')
            print(f'F1 scores => macro: {f1_macro: .4f}, micro: {f1_micro: .4f}')
        dev_accuracy = validation_accuracy / validation_instances
        dev_loss = validation_loss / num_valid_batches
        return dev_accuracy, dev_loss, f1_macro, f1_micro

    def train_model(self, project_parameters: dict) -> None:
        """
        Method is utilized to process all steps of the training which are:
            Train model for each epoch and validate;
            Save epoch results;
            Test the model when training is over
            Note: Resume training is also called and checked in this method
        :param project_parameters: experiment parameters dictionary
        :return: None
        """
        num_epochs = project_parameters['epochs']
        dataloaders = {ds_name: self.get_data(ds_name) for ds_name in ['train', 'test', 'validation']}
        self.save_parameters(project_parameters)
        num_train_batches = len(dataloaders['train'])
        num_validation_batches = len(dataloaders['validation'])
        num_test_batches = len(dataloaders['test'])
        init = self.resume_training(project_parameters)

        for epoch in range(init, num_epochs):
            self.classifier.train()
            epoch_loss = 0
            epoch_accuracy = 0
            train_instances = 0
            train_iterator = tqdm(dataloaders['train'], total=num_train_batches, leave=True)
            for batch_data in train_iterator:
                step_loss, out = self.step_process(batch_data, train=True)
                epoch_loss += step_loss
                num_correct, non_pads, _ = self.compute_accuracy(batch_data['labels'], out)
                epoch_accuracy += num_correct
                train_instances += len(non_pads)
                train_iterator.set_description(f"Epoch: {epoch + 1} "
                                               f"Loss: {epoch_loss / num_train_batches: .4f} "
                                               f"Accuracy: {epoch_accuracy / train_instances: .4f}")

            dev_accuracy, dev_loss, f1_macro, f1_micro = self.development(dataloaders['validation'],
                                                                          num_validation_batches)

            epoch_dict = {
                'epoch': epoch + 1,
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
                'train_loss': epoch_loss / num_train_batches,
                'train_accuracy': epoch_accuracy / train_instances,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro
            }
            self.save_results(epoch_dict)

        test_accuracy, test_loss, f1_macro_test, f1_micro_test = self.development(dataloaders['test'], num_test_batches,
                                                                                  test=True)
        test_dict = {
            'num_epochs': num_epochs,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_macro': f1_macro_test,
            'f1_micro': f1_micro_test
        }
        self.save_results(test_dict, test=True)

    def save_results(self, results_dict: dict, test: bool = False) -> None:
        """
        Method is utilized to save the epoch results (train and development in one dictionary)
        :param results_dict: dictionary in which all training and development values are set (if test is False)
        :param test: boolean variable specifies whether training and development outcomes will be saved or test results
        :return: None
        """
        directory = os.path.join(self.experiment_dir, 'outputs')
        results_file = os.path.join(directory, f"{'results_test' if test else 'results'}.pickle")
        if not os.path.exists(results_file):
            results = {key: list() for key in results_dict.keys()}
        else:
            with open(results_file, 'rb') as result_data:
                results = pickle.load(result_data)
        for key, details in results_dict.items():
            results[key].append(details)
        with open(results_file, 'wb') as result_data:
            pickle.dump(results, result_data)

        if test:
            print(f"Test results were saved after training of {results_dict['num_epochs']} epochs")
        else:
            self.save_model_parameters(results_dict)
            print(f"Epoch results were added to the existing data for epoch {results_dict['epoch']}")
            print(f'\n{"<" * 20}{">" * 20}')

    def save_model_parameters(self, results_dict: dict):
        """
        Method is utilized to save model and optimizer parameters for specific epoch.
        :param results_dict: dictionary in which all training and development values are set
        :return: None
        """
        ckpt_dir = os.path.join(self.experiment_dir, 'checkpoints')
        check_dir(ckpt_dir)
        model_name = f"model_epoch_{results_dict['epoch']}_f1_{results_dict['f1_macro']:.4f}_" \
                     f"loss_{results_dict['dev_loss']:.4f}_acc_{results_dict['dev_accuracy']:.4f}"
        optim_name = f"optim_epoch_{results_dict['epoch']}"
        model_path = os.path.join(ckpt_dir, model_name)
        optim_path = os.path.join(ckpt_dir, optim_name)
        torch.save(self.classifier.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)
        print(f"Model and Optimizer parameters were saved for epoch {results_dict['epoch']}")

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
