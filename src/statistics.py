import os
import pickle

import sklearn.metrics
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from src.classifier import NERClassifierBERT
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

class Statistics:
    def __init__(self, hp, process, device, experiment_num, data_choice):
        self.hp = hp
        self.process = process
        self.device = device
        self.experiment_num = experiment_num
        self.data_choice = data_choice
        self.experiment_path = self.set_experiment_environment()
        self.classifier = self.set_model()
        self.collator = self.set_collator()

    def set_experiment_environment(self) -> str:
        """
        Method is utilized to set experiment environment which data will be used for model setup
        :return: directory for the experiment data
        """
        experiment_path = os.path.join('results', f'experiment_{self.experiment_num}')
        return experiment_path

    def set_model(self):
        return NERClassifierBERT(self.hp).to(self.device)

    def get_model_paths(self):
        result_dir = os.path.join(self.experiment_path, 'outputs/results.pickle')
        ckpt_dir = os.path.join(self.experiment_path, 'checkpoints')
        with open(result_dir, 'rb') as result_data:
            results_dict = pickle.load(result_data)
        paths = dict()
        for each in ['dev_loss', 'dev_accuracy', 'f1_macro']:
            if each == 'dev_loss':
                value = min(results_dict[each])
            else:
                value = max(results_dict[each])
            val_idx = results_dict[each].index(value)

            model_name = (f"model_epoch_{results_dict['epoch'][val_idx]}_"
                          f"f1_{results_dict['f1_macro'][val_idx]:.4f}_"
                          f"loss_{results_dict['dev_loss'][val_idx]:.4f}_"
                          f"acc_{results_dict['dev_accuracy'][val_idx]:.4f}")
            path = os.path.join(ckpt_dir, model_name)
            paths[each] = path
        return paths

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

    def load_model(self, path):
        self.classifier.load_state_dict(torch.load(path))
        self.classifier.eval()

    def get_data(self, statistics_parameters):
        datasets = self.process.process()
        return DataLoader(dataset=datasets[statistics_parameters['statistics_data_choice']],
                          collate_fn=self.collator,
                          shuffle=True,
                          batch_size=self.hp['batch_size'])

    def collect_results(self, path, statistics_parameters):
        self.load_model(path)
        dataloader = self.get_data(statistics_parameters)
        infer_iterator = tqdm(dataloader, total=len(dataloader), leave=True)
        predictions = list()
        targets = list()
        with torch.no_grad():
            for batch in infer_iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                output = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
                batch_targets, batch_predictions = self.clean_special_tokens(output, batch['labels'])
                targets.extend(batch_targets)
                predictions.extend(batch_predictions)
        return targets, predictions

    @staticmethod
    def clean_special_tokens(out, labels):
        batch_predictions = list()
        batch_targets = list()
        targets = labels.view(-1).tolist()
        predictions = out.reshape(-1, out.shape[-1])
        preds = torch.argmax(predictions, -1).tolist()
        for t, p in zip(targets, preds):
            if t != -100:
                batch_predictions.append(p)
                batch_targets.append(t)

        return batch_targets, batch_predictions

    def save_predictions(self, output_path, choice, model_path, statistics_parameters):
        file_name = f"prediction_{choice}_{statistics_parameters['statistics_data_choice']}.pickle"
        prediction_path = os.path.join(output_path, file_name)
        if not os.path.exists(prediction_path):
            targets, predictions = self.collect_results(model_path, statistics_parameters)
            targets_none = list()
            predictions_none = list()
            for t, p in zip(targets, predictions):
                if t != 0 and p != 0:
                    targets_none.append(t)
                    predictions_none.append(p)
            results = {
                'targets': targets,
                'predictions': predictions,
                'targets_none': targets_none,
                'predictions_none': predictions_none
            }
            with open(prediction_path, 'wb') as prediction_data:
                pickle.dump(results, prediction_data)
        with open(prediction_path, 'rb') as prediction_data:
            results = pickle.load(prediction_data)

        return results

    def plot_confusion(self, targets, predictions, graph_path, graph_word):
        conf_matrix = confusion_matrix(targets, predictions)
        plt.figure(figsize=(12, 12), dpi=100)
        sns.set_palette('tab10')
        sns.set(font_scale=1.1)

        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='tab20c')

        classes = list(self.hp['label2id'].keys())
        labels = classes[1::] if 'without' in graph_word else classes

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)

        ax.set_title(f"Confusion Matrix based on {graph_word}", fontsize=14, pad=20)
        plt.savefig(graph_path)
        plt.show()

    def generate_confusion(self, model_path, choice, statistics_parameters):
        dataset = statistics_parameters['statistics_data_choice']

        if choice == 'dev_loss':
            graph_word = f'Validation Loss ({dataset.title()} dataset)'
        elif choice == 'dev_accuracy':
            graph_word = f'Validation Accuracy ({dataset.title()} dataset)'
        else:
            graph_word = f'F1 Score ({dataset.title()} dataset)'
        outputs_path = os.path.join(self.experiment_path, 'outputs')
        results = self.save_predictions(outputs_path, choice, model_path, statistics_parameters)

        graph_path = os.path.join(outputs_path, f'confusion_{dataset}_data_{choice}.png')
        graph_path_none = os.path.join(outputs_path, f'confusion_{dataset}_data_{choice}_none_o.png')
        self.plot_confusion(results['targets'], results['predictions'], graph_path, graph_word)
        self.plot_confusion(results['targets_none'], results['predictions_none'], graph_path_none,
                            graph_word + ' without O label')

    def show_statistics(self, statistics_parameters):
        paths_dict = self.get_model_paths()
        for metric, path in paths_dict.items():
            self.generate_confusion(path, metric, statistics_parameters)
