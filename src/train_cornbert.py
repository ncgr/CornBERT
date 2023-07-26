"""
Code based off of these tutorials:
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb
https://huggingface.co/learn/nlp-course/chapter5/2?fw=pt
"""

from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

# LOCAL IMPORTS
from tokenizer import loadCornbertTokenizer
from model import CornBERT

def pre_train(tokenizer, model, data_path):
    # load the dataset from a file and tokenize it
    dataset = load_dataset("text", data_files=data_path)
    tokenizer_function = lambda in_put : tokenizer(in_put["text"])
    tokenized_dataset = dataset.map(tokenizer_function,batched=True)

    # collator automatically masks some of the input for pretraining
    data_collator = DataCollatorForLanguageModeling(
                         tokenizer=tokenizer, mlm=True, mlm_probability=0.15,return_tensors="pt")

    # TODO: add bespoke arguments
    training_args = TrainingArguments("test-trainer")
    # make a trainer TODO: add metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer
    )

    trainer.train()
    #trainer.save_model("./trained_model")

if __name__ == "__main__" :
    import sys
    # load tokenizer
    tokenizer_args = []
    if len(sys.argv) > 1:
        tokenizer_config = sys.argv[1]
        tokenizer_args.append(tokenizer_config)
    else :
        tokenizer_args = ["cornbert-tokenizer.json"]
    tokenizer = loadCornbertTokenizer(*tokenizer_args)
    #TODO: need to do something about these not getting picked up by the json load
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.mask_token = "[MASK]"

    model = CornBERT()
    pre_train(tokenizer, model, "S288C.regulatory.training.fa")
