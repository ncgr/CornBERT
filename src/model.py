import torch
import torch.nn as nn

from transformers import RobertaForMaskedLM
from transformers import RobertaConfig, RobertaModel, AutoTokenizer


class CornBERT(RobertaForMaskedLM):
    def __init__(self):


        # make roberta for masked training
        config = RobertaConfig(
            vocab_size=5004, # number of tokens in the vocabulary
            max_position_embeddings=256, # maximum sequence length to be fed as a whole to encoder
            num_attention_heads=6, # number of attention heads in the encoder
            num_hidden_layers=2, # number of feed forward layers in each attention head
            type_vocab_size=1,
        )
        # roberta accessible through self.roberta
        # combplete list of submodules accessible through self.named_modules
        super().__init__(config=config)
    #def forward(self,x):
    #    return self.roberta(x)


# test that the model is functional
if __name__ == '__main__':

    # generate random input
    from random import choice

    f = open("training/yeast/y12_promoters")
    promoter  = f.readline()
    f.close()
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    global inputs
    inputs    = tokenizer(promoter, return_tensors="pt")

    # run the model on the random input
    global cbt
    cbt = CornBERT()
    #global out
    #out = cbt(inputs.input_ids)
