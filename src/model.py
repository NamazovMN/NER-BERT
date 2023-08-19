import torch
import torch.nn as nn
from vocab import Vocabulary

class NERClassifier(nn.Module):
    """
    Class is used to build the classifier model
    """

    def __init__(self, hyperparams: dict, vocabulary: dict):
        """
        Initializer for the class which specifies required parameters
        :param hyperparams: dictionary for hyperparameters of the model
        :param vocabulary: vocabulary object will be used for embedding layer and padding index
        """
        super(NERClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(vocabulary),
            embedding_dim=hyperparams['embedding_dim'],
            padding_idx=vocabulary['<PAD>']
        )
        self.lstm = nn.LSTM(
            input_size=hyperparams['embedding_dim'],
            hidden_size=hyperparams['hid_dim'],
            bidirectional=hyperparams['bidirectional'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
            batch_first=True
        )

        input_dim = hyperparams['hid_dim'] * 2 if hyperparams['bidirectional'] else hyperparams['hid_dim']

        self.linear = nn.Linear(
            in_features=input_dim,
            out_features=500
        )
        self.linear2 = nn.Linear(
            in_features=500,
            out_features=100
        )
        self.linearout = nn.Linear(
            in_features=100,
            out_features=hyperparams['num_classes']
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, input_data: torch.LongTensor) -> torch.Tensor:
        """
        Method is utilized to perform feedforward process
        :param input_data: input tensor for the model
        :return: output of the model
        """
        embeddings = self.embedding(input_data)

        lstm_out, (_, _) = self.lstm(embeddings)

        relu_out1 = self.relu1(lstm_out)
        # print(f'ro: {relu_out1.shape}')
        lin1 = self.relu2(self.dropout(self.linear(relu_out1)))
        # print(f'lo1: {lin1.shape}')

        lin2 = self.relu3(self.dropout2(self.linear2(lin1)))
        # print(f'lo2: {lin2.shape}')

        lin_out = self.linearout(lin2)
        # print(f'lo: {lin_out.shape}')
        out = self.dropout(lin_out)
        return out
