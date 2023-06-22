import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel, AutoTokenizer

class CornBERT(nn.Module):
    def __init__(self):
        super().__init__()

        # set encoder parameters
        mpe = 256 # maximum sequence length to be fed as a whole to encoder
        nah = 6 # number of attention heads in the encoder
        vs  = 5004 # number of tokens in the vocabulary
        nhl = 2 # number of feed forward layers in each attention head

        # make roberta encoder
        # set max input length to 256
        configuration = RobertaConfig(max_position_embeddings=mpe,
                                      num_attention_heads=nah,
                                      num_hidden_layers=nhl)
        self.encoder = RobertaModel(configuration)

    def forward(self,x):
        x = self.encoder(**x)
        x = torch.mean(x.last_hidden_state,dim=1) # avg token embeddings
        return x


# test that the model is functional
if __name__ == '__main__':

    # generate random input
    from random import choice

    promoter  = "".join([choice("ACTGN") for i in range(256)])
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    inputs    = tokenizer(promoter, return_tensors="pt")

    # run the model on the random input
    cbt = CornBERT()
    out = cbt(inputs)
    print(out.size())
