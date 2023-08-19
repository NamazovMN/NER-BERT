import torch
from torch import nn
from transformers import BertModel


class NERClassifierBERT(nn.Module):
    """
    Class is utilized to set NER classifier which utilized BERT model as an encoder
    """

    def __init__(self, hp: dict):
        """
        Method is utilized to set model layers as an initializer
        :param hp: dictionary which includes experiment parameters
        """
        super(NERClassifierBERT, self).__init__()
        self.bert = BertModel.from_pretrained(hp['model_checkpoint'])
        self.dropout = nn.Dropout(hp['dropout'])
        self.dropout2 = nn.Dropout(hp['dropout'])

        self.fc = nn.Linear(self.bert.config.hidden_size, len(hp['id2label']))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Method is utilized as feed forward performer of the model
        :param input_ids: input indexes for the provided sequence in shape of [batch size, max length]
        :param attention_mask: attention mask for the provided sequence in shape of [batch size, max length]
        :return: predictions of the classifier in shape of [batch size, max length, number of labels]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        drop = outputs.last_hidden_state
        outputs = self.fc(drop)
        return outputs
