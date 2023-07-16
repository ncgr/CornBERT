"""
Code based off of this tutorial:
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb
"""

import torch

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments

# LOCAL IMPORTS
from tokenizer import loadCornbertTokenizer
from model import CornBERT

"""
"""

def pre_train(tokenizer, model, data_path):
    # make a dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # make a trainer
    training_args = TrainingArguments(
        output_dir="./training_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("./trained_model")

if __name__ == "__main__" :
    # load tokenizer
    global tokenizer
    tokenizer = loadCornbertTokenizer("training/yeast/cornbert-tokenizer.json")
    # create model
    model = CornBERT()
    pre_train(tokenizer,model,"training/yeast/y12_promoters")
