import torch

from model import NERClassifier
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import NERDataset
from tqdm import tqdm
from process import Process
from utilities import *
import os
from nltk.tokenize import word_tokenize
from string import punctuation
class Train:
    def __init__(self, hp, vocabulary, preprocess, ds_path, exp_num, device):
        self.hp = hp
        self.vocabulary = vocabulary
        self.ds_path = ds_path
        self.exp_num = exp_num
        self.device = device
        self.preprocess = preprocess

        self.experiment_dir = self.set_environment()
        self.processing = self.set_processing(preprocess)
        self.classifier = self.set_model()
        self.loss_fn = self.set_loss_fn()
        self.optimizer = self.set_optimizer()

    def save_parameters(self, project_parameters):
        outputs_path = os.path.join(self.experiment_dir, 'outputs')
        check_dir(outputs_path)
        parameters_path = os.path.join(outputs_path, 'parameters.pickle')
        if not os.path.exists(parameters_path):
            print(f'Project Parameters for experiment {self.exp_num} were saved successfully!')
            print(f'{"<" * 20}{">" * 20} \n')
            project_parameters['hyperparams'] = self.hp
            with open(parameters_path, 'wb') as params:
                pickle.dump(project_parameters, params)



    def set_processing(self, preprocess):
        return Process(preprocess, self.vocabulary, self.hp['max_length'], self.ds_path)

    def set_model(self):
        return NERClassifier(self.hp, self.vocabulary.vocabulary).to(self.device)

    def set_loss_fn(self):
        return nn.CrossEntropyLoss(ignore_index=self.vocabulary.lab2id('<PAD>'))

    def set_optimizer(self):
        return Adam(params=self.classifier.parameters(), lr=self.hp['learning_rate'], weight_decay=self.hp['weight_decay'])

    def get_data(self, datasets, ds_name):
        data = self.processing.process_data(datasets[ds_name], ds_name)
        dataset = NERDataset(data, self.device)
        return DataLoader(dataset=dataset, batch_size=self.hp['batch_size'], shuffle=True)

    def step_process(self, batch, train=True):
        input_sequences = batch['data']
        target_sequences = batch['label']
        output = self.classifier(input_sequences)
        predictions = output.view(-1, output.shape[-1])

        labels = target_sequences.view(-1)

        if train:
            self.optimizer.zero_grad()
            loss = self.loss_fn(predictions, labels)
            loss.backward()
            self.optimizer.step()

        else:
            loss = self.loss_fn(predictions, labels)

        return labels, predictions, loss.item()

    def compute_accuracy(self, target_sequences, output_sequences):
        predictions = torch.argmax(output_sequences, -1).tolist()
        labels = target_sequences.tolist()
        original_labels = list()
        original_preds = list()
        corrects = list()
        for p, t in zip(predictions, labels):
            if t != self.vocabulary.lab2id('<PAD>'):
                original_labels.append(t)
                original_preds.append(p)
                corrects.append(t == p)
        return sum(corrects), original_labels, original_preds

    def train_model(self, dataset, project_parameters):
        self.save_parameters(project_parameters)
        num_epochs = project_parameters['epochs']
        train_loader = self.get_data(dataset, 'train')
        num_train_batches = len(train_loader)
        validation_loader = self.get_data(dataset, 'validation')
        num_validation_batches = len(validation_loader)
        init = self.resume_training(project_parameters)
        for epoch in range(init, num_epochs):
            train_epoch_loss = 0
            train_epoch_accuracy = 0
            train_instances = 0
            self.classifier.train()
            print(f'Epoch: {epoch + 1}')
            train_iterator = tqdm(iterable=train_loader, total=num_train_batches, leave=True)

            for train_batch in train_iterator:
                targets, outputs, step_loss = self.step_process(train_batch)
                train_step_accuracy, train_targets, _ = self.compute_accuracy(targets, outputs)

                train_instances += len(train_targets)
                train_epoch_loss += step_loss
                train_epoch_accuracy += train_step_accuracy
                train_iterator.set_description(desc=f'Training => Loss: {train_epoch_loss / num_train_batches: .4f}'
                                                    f' Accuracy: {train_epoch_accuracy / train_instances: .4f}')

            dev_accuracy, dev_loss, f1_macro, f1_micro = self.development(validation_loader, num_validation_batches)

            epoch_dict = {
                'epoch': epoch + 1,
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
                'train_loss': train_epoch_loss / num_train_batches,
                'train_accuracy': train_epoch_accuracy / train_instances,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro
            }
            self.save_results(epoch_dict)

        test_loader = self.get_data(dataset, 'test')
        num_test_batches = len(test_loader)
        test_accuracy, test_loss, f1_macro_test, f1_micro_test = self.development(test_loader, num_test_batches)
        test_dict = {
            'num_epochs': num_epochs,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_macro': f1_macro_test,
            'f1_micro': f1_micro_test
        }
        self.save_results(test_dict, test=True)

    def resume_training(self, project_parameters):
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

    def save_results(self, results_dict, test=False):
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

    def save_model_parameters(self, results_dict):
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
        print(f'{"<" * 20}{">" * 20} \n')

    def development(self, validation_loader, num_valid_batches, test=False):
        self.classifier.eval()
        targets = list()
        outputs = list()
        with torch.no_grad():
            validation_iterator = tqdm(iterable=validation_loader, total=num_valid_batches, leave=True)
            validation_loss = 0
            validation_accuracy = 0
            validation_instances = 0
            for batch in validation_iterator:
                validation_targets, validation_outputs, validation_step_loss = self.step_process(batch, train=False)
                validation_loss += validation_step_loss
                step_accuracy, targets_, outputs_ = self.compute_accuracy(validation_targets, validation_outputs)
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
            print(f'\nF1 scores => macro: {f1_macro: .4f}, micro: {f1_micro: .4f}')

        dev_accuracy = validation_accuracy / validation_instances
        dev_loss = validation_loss / num_valid_batches
        return dev_accuracy, dev_loss, f1_macro, f1_micro

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
        optim_name = f"optim_epoch_{epoch_choice}"
        model_path = os.path.join(ckpt_dir, model_name)
        print(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model was not trained at least for {epoch_choice} epochs or there is typo! '
                                    f'Check it first!')
        optim_path = os.path.join(ckpt_dir, optim_name)

        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()
        self.optimizer.load_state_dict(torch.load(optim_path))

    def decision_maker(self, project_parameters):

        if project_parameters['load_best']:
            if project_parameters['epoch_choice']:
                print(
                    'WARNING: Best choice and epoch choice were made together! '
                    'In such cases best choice is prioritized!'
                )
            return self.get_best_epoch(project_parameters['best_choice'])

        else:
            return project_parameters['epoch_choice']

    def get_best_epoch(self, user_choice):
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


    def set_environment(self):
        results_dir = '../results'
        check_dir(results_dir)
        experiment_dir = os.path.join(results_dir, f'experiment_{self.exp_num}')
        check_dir(experiment_dir)
        return experiment_dir

    # def infer(self):
    #     parameters =
    #     with open()
    #     epoch_choice = self.decision_maker(project_parameters)
    #     print(type(epoch_choice))
    #     print(epoch_choice)
    #     self.load_model(epoch_choice)
    #     input_text = input('Please provide your text:')
    #     tokenized_text = word_tokenize(input_text)
    #     # if not project_parameters['cased']:
    #     #     tokenized_text = [token.lower() for token in tokenized_text]
    #     # if project_parameters['punctuation']:
    #     #     tokenized_text = [token for token in tokenized_text if token not in punctuation]
    #     # if project_parameters['stopwords']:
    #     #     tokenized_text = [token for token in tokenized_text if token not in self.preprocess.stopwords]
    #     encoded_text = self.processing.encode_data(tokenized_text)
    #     input_sequence = torch.LongTensor(encoded_text).to(self.device)
    #     output = self.classifier(input_sequence)
    #     predictions = torch.argmax(output, -1).tolist()
    #     print(self.vocabulary.label2id)
    #     id2lab = {idx: token for token, idx in self.vocabulary.label2id.items()}
    #     result = [id2lab[token] for token in predictions[0: len(tokenized_text)]]
    #     print(tokenized_text)
    #     print(result)
    #     # print(output.shape)