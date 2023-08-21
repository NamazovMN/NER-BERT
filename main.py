import torch.cuda
from src.inference import Inference
from utilities import *
from src.process import DataProcess
from src.train import Train
import random
from transformers import BertTokenizerFast
from src.statistics import Statistics


def __main__():
    random.seed(42)
    project_parameters = get_parameters()
    hp = get_hyperparameters(project_parameters)
    data_path = 'dataset_parameters'
    process = DataProcess('conllpp', model_checkpoint=hp['model_checkpoint'], data_path=data_path)

    labels = process.datasets['train'].features['ner_tags'].feature.names
    hp['label2id'] = {label: idx for idx, label in enumerate(labels)}
    hp['id2label'] = {idx: label for idx, label in enumerate(labels)}
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if project_parameters['train']:
        trainer = Train(hp, process, device, project_parameters['experiment_num'])
        trainer.train_model(project_parameters)
    if project_parameters['stats']:
        stat_data = ['experiment_num', 'statistics_data_choice', 'stats']
        statistics_parameters = {parameter: value for parameter, value in project_parameters.items()
                                 if parameter in stat_data}
        statistics = Statistics(hp, process, device, project_parameters['experiment_num'], 'validation')
        statistics.show_statistics(statistics_parameters)
    if project_parameters['infer']:
        infer_options = ['load_choice', 'load_best', 'epoch_choice', 'experiment_num']
        infer_parameters = {option: project_parameters[option] for option in infer_options}
        inference = Inference(hp, process, device, infer_parameters)
        inference.infer(infer_parameters)


if __name__ == '__main__':
    __main__()
