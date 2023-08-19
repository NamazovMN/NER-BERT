from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from utilities import *
from transformers.tokenization_utils_base import BatchEncoding


class DataProcess:
    """
    Class is utilized to load and process the Huggingface dataset, according to the provided parameters
    """

    def __init__(self, ds_name: str, model_checkpoint: str, data_path: str):
        """
        Method is utilized as an initializer for data processing object
        :param ds_name: Huggingface dataset name (as path specified in the website)
        :param model_checkpoint: Huggingface model checkpoint for the transformer/tokenizer (e.g., bert-base-cased)
        :param data_path: directory to save tokenizer
        """
        self.datasets = self.get_dataset(ds_name)
        self.tokenizer = self.set_tokenizer(data_path, model_checkpoint)

    @staticmethod
    def set_tokenizer(data_path: str, model_checkpoint: str) -> AutoTokenizer:
        """
        Method is utilized to set the tokenizer according to the given path and tokenizer checkpoint in the Huggingface.
        If one wants to use the same model, vocabulary must be same as the model was trained on (e.g., inference)
        :param data_path: directory to save tokenizer. In case of reuse of the model vocabulary info must be same
        :param model_checkpoint: Huggingface model checkpoint for the transformer/tokenizer (e.g., bert-base-cased)
        :return: saved or new-set AutoTokenizer object
        """
        check_dir(data_path)
        tokenizer_path = os.path.join(data_path, 'tokenizer')
        if not os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            tokenizer.save_pretrained(tokenizer_path)
        return AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def get_dataset(ds_path: str):
        """
        Method is utilized to load dataset from the given dataset path of the Huggingface dataset
        :param ds_path: path for the Huggingface dataset
        :return: Required dataset for the task
        """
        return load_dataset(ds_path)

    @staticmethod
    def align_labels_with_tokens(labels: list, word_ids: list) -> list:
        """
        Method is utilized as it was given in the Huggingface website. It is utilized to align tokenization results with
        the given tokenized data
        :param labels: labels from the originally tokenized dataset
        :param word_ids: word indexes data from the tokenization process
        :return: list of aligned labels
        """

        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples: Dataset) -> BatchEncoding:
        """
        Method is utilized to create aligned labels for an instance in dataset (method is same as in Huggingface)
        :param examples: dictionary for dataset instance
        :return: dictionary for the specified data instance
        """
        tokenized_inputs = self.tokenizer(examples["tokens"],
                                          truncation=True,
                                          is_split_into_words=True)

        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def process(self):
        """
        Method is utilized to process all datasets according to required format
        :return: resulting datasets for the project
        """

        tokenized_datasets = self.datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.datasets['train'].column_names
        )
        return tokenized_datasets
