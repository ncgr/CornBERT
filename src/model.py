import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel, AutoTokenizer


class CornBERT(nn.Module):
    def __init__(self):
        super().__init__()

        # make roberta encoder
        # set max input length to 256
        configuration = RobertaConfig(max_position_embeddings=256)
        self.encoder = RobertaModel(configuration)
        self.rh_L1 = nn.Linear(768,10) # regression head linear layer 1
        self.rh_L2 = nn.Linear(10,10) # regression head linear layer 2

    def forward(self,x):
        x = self.encoder(**x)
        x = torch.mean(x.last_hidden_state,dim=1) # avg token embeddings
        x = self.rh_L1(x)
        x = self.rh_L2(x)
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
