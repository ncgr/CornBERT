import torch

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification

# mixin with config from original study
class CornBERT:
    def roberta_config(self):
        return RobertaConfig(
            vocab_size=5004, # number of tokens in the vocabulary
            hidden_size = 768, # embedding dimensionality, used throughout the model
            max_position_embeddings=256, # maximum sequence length to be fed as a whole to encoder
            num_attention_heads=6, # number of attention heads in the encoder
            num_hidden_layers=6, # num of encoder layers
            type_vocab_size=1, # num of token types, in next sentence prediction tasks, one per sentence
            num_labels=10, # number of plant tissues. used in sequence classification model, not used in maskedlm model
        )

# child inherits data attributes from leftmost parent class, e.g. will inherit the model
class CornBERTforMaskedLM(RobertaForMaskedLM,CornBERT):
    def __init__(self):
        super().__init__(config=self.roberta_config())

class CornBERTforGeneExpression(RobertaForSequenceClassification,CornBERT):
    def __init__(self):
        super().__init__(config=self.roberta_config())

    # similar to RobertaForSequenceClassification, but 
    """
    Modified to take the average. DOES NOT FILTER OUT PADDING. You must do this in preprocessing.
    """
    def forward(self,x):
        x = self.roberta(x)
        # average embedding of all tokens instead of only taking embedding of first
        x = torch.mean(x.last_hidden_state,1,keepdim=True) # average over the second dimension but do not remove it
        x = self.classifier(x)
        return x

# test that the model is functional
if __name__ == '__main__':

    # test on random input
    from torch import randint
    global model
    model = CornBERTforMaskedLM()
    #model = CornBERTforGeneExpression()
    tokens = torch.randint(low=100,high=5000,size=(1,250))
    tokens[0][0] = 0
    tokens[0][-1] = 2
    global output
    output = model(tokens)
