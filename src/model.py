import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel, AutoTokenizer

class cornBERT(nn.Module):
    def __init__(self):
        super().__init__()

	# make roberta encoder
        # set max input length to 256
        configuration = RobertaConfig(max_position_embeddings=256)
        self.encoder = RobertaModel(configuration)

    def forward(self,x):
        x = self.encoder(**x)
        x = torch.mean(x.last_hidden_state,dim=2)
        return x

# test that the model is functional

# generate random input
from random import choice

promoter  = [choice("ACTGN") for i in range(100)]
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
inputs    = tokenizer(promoter, return_tensors="pt")

# run the model on the random input
cbt = cornBERT()
out = cbt(inputs)
print(out.size())
