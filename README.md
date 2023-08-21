# Named Entity Recognition
## Goal
The main goal of the Named Entity Recognition task is to utilize Natural Language Processing methods to identify named entities in the given texts. In our project, the model is BERT-based classifier to identify named entities in the given texts.
## Model 
We developed BERT-based classifier for token classification task. BERT model from [HuggingFace](www.huggingface.com) was utilized as an encoder for this purpose. Hidden states of the model was provided to the FCN layer with shape of (hidden_size * num_classes), where hidden_size and num_classes stand for last hidden state dimension of the Transformer model and number of Named Entities in the dataset (9)
## Dataset
Project makes use of [conllpp](https://huggingface.co/datasets/conllpp) dataset from [HuggingFace](www.huggingface.com). As it is given in the source website, data splits were already made by authors. Explicitly, train, test and validation datasets include 14k, 3.45k and 3.25k sequences with their token labels, respectively. Class identification can be seen below:

       O:       0 => For other common tokens, which don't stand for any entity
       B-PER:   1 =>  Initial token of entity specifies Person
       I-PER:   2 =>  Intermediate token of entity specifies Person
       B-ORG:   3 =>  Initial token of entity specifies Organization
       I-ORG:   4 =>  Intermediate token of entity specifies Organization
       B-LOC:   5 =>  Initial token of entity specifies Location
       I-LOC:   6 =>  Intermediate token of entity specifies Location
       B-MISC:  7 =>  Initial token of entity specifies Miscallenous entity
       I-MISC:  8 =>  Intermediate token of entity specifies Miscallenous entity

## Playground
In order to test model you can follow the steps that are given: 
* Initially, you need to pull the project into your local machine; 
* Them, you should run the following snippet to install all required dependencies: 
  ```python
  python main.py -r requirements.txt
* Now you are all set to run the following snippet (Note: The source code can be found in [playground.py](src/playground.py).) 
  ```python
  python main.py --playground_only --experiment_number 27 --play_bis --cased --clean_stops --clean_punctuation
  
 ## What is new?
 In order to see the result, we need to have tokenizer to split sentence into the words. In order to do this, I used BIS model that can be found in my repository. play_bis parameter in the code snippet that was given above activate it. If you do not set it, model will use NLTK tokenizer.
 
 I hope you will enjoy it!
 
 ## Trained model
 These following data can be downloaded from corresponding links, since they exceeds size limitation of GitHub: https://drive.google.com/drive/folders/1qS6Hb_eZdWiwc9NMSoEJc5jJztpksxKE?usp=sharing
 model_structure.pickle: you need to put it into the corresponding experiment directory
 model checkpoint: you need to put it into the checkpoints directory in the corresponding experiment path
 
 ***Regards,***

***Mahammad Namazov***
